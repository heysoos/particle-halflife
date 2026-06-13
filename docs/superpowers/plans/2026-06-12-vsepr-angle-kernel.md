# VSEPR Angle-Locking Kernel — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or
> subagent-driven-development) to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an angular force between a composite's bonds so geometry adopts real
molecular angles (degree-2 → defined angle, degree-3 → Y, degree-4 → cross) instead
of floppy straight chains. Two force laws — `vsepr` (emergent even spreading) and
`harmonic` (hash-θ₀ prescribed angle) — behind one shared kernel, selected by
`config.angle_mode`.

**Architecture:** Per-particle angle list recomputed from the edge graph each step
(no changes to fusion/fission/ring-closure). A `(N, P_max, 3)` triple array drives a
vmapped per-triple force law that scatters into the `(N, 2)` force buffer with
min-image displacement, mirroring `compute_edge_bond_forces`. Gated Python-side on
the static `angle_mode`/`bond_mode` so the off path is dead-code-eliminated.

**Tech Stack:** JAX (jit, vmap, scatter `mode='drop'`, `associative_scan`), existing
hash-chemistry helpers.

**Spec:** `docs/superpowers/specs/2026-06-12-vsepr-angle-kernel-design.md`

**Conventions (from CLAUDE.md):** WSL Pattern B — run tests with
`JAX_PLATFORMS=cpu .venv/bin/python -m pytest ...` (force CPU; the user's live sim may
hold the GPU). Never `git add -A`. Preserve existing comments. Commit only the named
files per task.

---

### Task 1: Config fields + runtime `k_angle`

**Files:**
- Modify: `halflife/config.py` (add four fields to `SimConfig`)
- Modify: `halflife/state.py:270-305` (`PhysicsParams` + `initialize_physics_params`)
- Test: `tests/test_angle_kernel.py` (new)

- [ ] **Step 1: Write failing test**

```python
# tests/test_angle_kernel.py
import dataclasses
import jax.numpy as jnp
from halflife.config import SimConfig
from halflife.state import initialize_physics_params


def test_angle_config_defaults():
    c = SimConfig()
    assert c.angle_mode == "off"          # default: existing behaviour unchanged
    assert c.k_angle == 10.0
    assert c.theta_min_deg == 90.0
    assert c.theta_max_deg == 180.0


def test_physics_params_seeds_k_angle():
    c = dataclasses.replace(SimConfig(), k_angle=7.5)
    p = initialize_physics_params(c)
    assert float(p.k_angle) == 7.5
```

- [ ] **Step 2: Run, verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_angle_kernel.py -q`
Expected: FAIL (`SimConfig` has no `angle_mode`).

- [ ] **Step 3: Add the config fields**

In `halflife/config.py`, near the bond-mode / `k_bond` fields (around line 191), add:

```python
    # ── Angle-locking (covalent bond geometry, edges mode) ────────────────────
    # Angular force between a composite's bonds so geometry isn't floppy.
    #   "off"      → no angle force (default; bit-identical to pre-feature)
    #   "vsepr"    → bond directions repel & spread evenly (2π/degree); emergent
    #               rest angle, no frustration at degree ≥ 3. Fixes floppy chains.
    #   "harmonic" → bonds pulled toward a hash-derived target angle θ0 per central
    #               species (cosine form). Robust 2-D route to prescribed bent
    #               low-valence shapes (water-analog). Intended for degree ≤ 2.
    angle_mode: str = "off"
    k_angle: float = 10.0          # angle stiffness; seeds runtime PhysicsParams.k_angle
    theta_min_deg: float = 90.0    # harmonic θ0 band floor (degrees)
    theta_max_deg: float = 180.0   # harmonic θ0 band ceiling (degrees)
```

- [ ] **Step 4: Add the runtime scalar**

In `halflife/state.py`, add to `PhysicsParams` (after `r_rest_scale`, line 285) — append a
field; constructed by keyword everywhere so order is safe, but confirm with
`grep -rn "PhysicsParams(" halflife tests` that nothing builds it positionally:

```python
    k_angle:                  jnp.ndarray  # () float32 — angle-locking stiffness (edges mode)
```

And seed it in `initialize_physics_params` (after the `k_bond=` line, 301):

```python
        k_angle=jnp.float32(config.k_angle),
```

- [ ] **Step 5: Run, verify pass**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_angle_kernel.py -q`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add halflife/config.py halflife/state.py tests/test_angle_kernel.py
git commit -m "feat(angles): add angle_mode/k_angle config + runtime PhysicsParams scalar"
```

---

### Task 2: Hash-derived rest angle θ₀

**Files:**
- Modify: `halflife/chemistry.py` (add `_hash_to_rest_angle` + `_species_rest_angles`
  next to `_hash_to_rest_length` / `_species_valences`, ~line 153-179)
- Test: `tests/test_angle_kernel.py`

- [ ] **Step 1: Write failing test**

```python
import math
import numpy as np
from halflife.chemistry import _hash_to_rest_angle, _species_rest_angles, _species_valences


def test_rest_angle_deterministic_and_in_band():
    c = dataclasses.replace(SimConfig(), num_species=6)
    lo, hi = math.radians(c.theta_min_deg), math.radians(c.theta_max_deg)
    angles = np.asarray(_species_rest_angles(c))
    assert angles.shape == (6,)
    assert np.all(angles >= lo - 1e-6) and np.all(angles <= hi + 1e-6)
    # deterministic per species index
    assert float(_hash_to_rest_angle(jnp.int32(3), c)) == float(angles[3])


def test_rest_angle_decorrelated_from_valence():
    # Different hash stream → θ0 ordering differs from valence ordering.
    c = dataclasses.replace(SimConfig(), num_species=12, max_valence=4)
    ang = np.asarray(_species_rest_angles(c))
    val = np.asarray(_species_valences(c))
    assert not np.array_equal(np.argsort(ang), np.argsort(val))
```

- [ ] **Step 2: Run, verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_angle_kernel.py -k rest_angle -q`
Expected: FAIL (`cannot import name '_hash_to_rest_angle'`).

- [ ] **Step 3: Implement**

In `halflife/chemistry.py`, after `_hash_to_rest_length` / `compute_r_rest_matrix`
(around line 154) and near `_species_valences`, add:

```python
def _hash_to_rest_angle(species: jnp.ndarray, config: SimConfig) -> jnp.ndarray:
    """
    Per-species target bond angle θ0 (radians) for angle_mode="harmonic".

    Same recipe as _hash_to_valence / _hash_to_rest_length: hash the species
    index, then re-mix with its OWN Fibonacci constant (0x85EBCA77, shift 15 —
    distinct from BE / valence / rest-length streams) so θ0 is decorrelated from
    the other per-species properties. Mapped into [theta_min_deg, theta_max_deg].

    Keyed on the CENTRAL species only (geometry is a property of the central
    atom), so a 2-bond atom of this species always prefers the same angle.

    Returns: scalar float32 in [theta_min, theta_max] radians.
    """
    h = _entity_hash_val(species, config).astype(jnp.uint32)
    h2 = (h * jnp.uint32(0x85EBCA77)) ^ (h >> jnp.uint32(15))
    frac = (h2 % jnp.uint32(1000)).astype(jnp.float32) / 999.0
    lo = jnp.float32(config.theta_min_deg * jnp.pi / 180.0)
    hi = jnp.float32(config.theta_max_deg * jnp.pi / 180.0)
    return lo + frac * (hi - lo)


def _species_rest_angles(config: SimConfig) -> jnp.ndarray:
    """Pre-compute the (num_species,) θ0 vector (radians). Fixed per config."""
    species_idx = jnp.arange(config.num_species, dtype=jnp.int32)
    return jax.vmap(lambda s: _hash_to_rest_angle(s, config))(species_idx)
```

- [ ] **Step 4: Run, verify pass**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_angle_kernel.py -k rest_angle -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add halflife/chemistry.py tests/test_angle_kernel.py
git commit -m "feat(angles): hash-derived per-species rest angle θ0 (harmonic mode)"
```

---

### Task 3: Per-particle neighbor list

**Files:**
- Modify: `halflife/step.py` (add `build_neighbor_list` near `compute_edge_bond_forces`)
- Test: `tests/test_angle_kernel.py`

**Helper for tests** — add this composite-builder at the top of the test file (it
hand-assembles a `WorldState` with explicit edges; reused by Tasks 3-7):

```python
import jax
from halflife.state import init_world_state  # adjust import if name differs


def _world_with_edges(edge_pairs, num_particles=8, positions=None):
    """Build a WorldState whose composite 0 contains the given undirected edges.
    edge_pairs: list of (i, j) particle ids. All referenced ids become members."""
    c = dataclasses.replace(SimConfig(), num_particles=num_particles,
                            max_composites=4, max_composite_size=8,
                            e_max=12, max_valence=4, bond_mode="edges")
    state = init_world_state(c, jax.random.PRNGKey(0))
    members = sorted({p for e in edge_pairs for p in e})
    comp = state.composites
    E = c.e_max
    edges = np.full((c.max_composites, E, 2), -1, np.int32)
    for k, (i, j) in enumerate(edge_pairs):
        edges[0, k] = (i, j)
    mem = np.full((c.max_composites, c.max_composite_size), -1, np.int32)
    mem[0, :len(members)] = members
    comp = comp._replace(
        alive=comp.alive.at[0].set(True),
        members=jnp.asarray(mem),
        member_count=comp.member_count.at[0].set(len(members)),
        edges=jnp.asarray(edges),
        edge_count=comp.edge_count.at[0].set(len(edge_pairs)),
    )
    parts = state.particles._replace(
        composite_id=state.particles.composite_id.at[jnp.asarray(members)].set(0),
    )
    if positions is not None:
        pos = np.asarray(state.particles.position)
        for pid, xy in positions.items():
            pos[pid] = xy
        parts = parts._replace(position=jnp.asarray(pos, jnp.float32))
    return state._replace(composites=comp, particles=parts), c
```

> NOTE before writing the test: open `halflife/state.py` and confirm the
> `WorldState` / `CompositeState` field names (`init_world_state`, `composite_id`,
> `edge_count`, etc.) match the helper above; adjust names if the codebase differs.
> This is the one place the plan touches state-construction internals.

- [ ] **Step 1: Write failing test**

```python
from halflife.step import build_neighbor_list


def _nbr_set(nbrs, pid):
    row = np.asarray(nbrs[pid])
    return set(int(x) for x in row if x >= 0)


def test_neighbor_list_chain():
    # chain 1-2-3: 2 is central (neighbors 1,3); ends have one neighbor
    state, c = _world_with_edges([(1, 2), (2, 3)])
    nbrs = build_neighbor_list(state.composites, c)
    assert _nbr_set(nbrs, 1) == {2}
    assert _nbr_set(nbrs, 2) == {1, 3}
    assert _nbr_set(nbrs, 3) == {2}
    assert _nbr_set(nbrs, 5) == set()      # free particle


def test_neighbor_list_branch_and_ring():
    # star: 0 bonded to 1,2,3 (degree 3)
    state, c = _world_with_edges([(0, 1), (0, 2), (0, 3)])
    nbrs = build_neighbor_list(state.composites, c)
    assert _nbr_set(nbrs, 0) == {1, 2, 3}
    # triangle ring 4-5-6
    state, c = _world_with_edges([(4, 5), (5, 6), (6, 4)])
    nbrs = build_neighbor_list(state.composites, c)
    assert _nbr_set(nbrs, 4) == {5, 6}
    assert _nbr_set(nbrs, 5) == {4, 6}
```

- [ ] **Step 2: Run, verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_angle_kernel.py -k neighbor -q`
Expected: FAIL (`cannot import name 'build_neighbor_list'`).

- [ ] **Step 3: Implement**

In `halflife/step.py`, after `compute_edge_bond_forces` (line 198), add:

```python
def build_neighbor_list(composites, config: SimConfig) -> jnp.ndarray:
    """
    Per-particle adjacency from the edge graph: (N, max_valence) int32, -1 padded.

    Each undirected edge contributes its partner to BOTH endpoints' rows.
    Vectorized via argsort + segmented rank (no scan). Column width is
    max_valence because degree ≤ valence ≤ max_valence always; any overflow is
    dropped (mode='drop'). Free particles (no edges) get an all -1 row.
    """
    N = config.num_particles
    C = config.max_composites
    E = config.e_max
    V = config.max_valence

    e_idx = jnp.arange(E, dtype=jnp.int32)
    valid = composites.alive[:, None] & (e_idx[None, :] < composites.edge_count[:, None])  # (C,E)
    a = composites.edges[:, :, 0]  # (C,E)
    b = composites.edges[:, :, 1]

    # Directed pairs in both orientations.
    src = jnp.concatenate([a.reshape(-1), b.reshape(-1)])   # (2CE,)
    dst = jnp.concatenate([b.reshape(-1), a.reshape(-1)])
    vmask = jnp.concatenate([valid.reshape(-1), valid.reshape(-1)])
    src = jnp.where(vmask, src, N)                          # invalid → OOB sentinel

    M = src.shape[0]
    order = jnp.argsort(src)                                # groups become contiguous
    src_s = src[order]
    dst_s = dst[order]
    pos = jnp.arange(M, dtype=jnp.int32)
    is_new = jnp.concatenate([jnp.array([True]), src_s[1:] != src_s[:-1]])
    # group_start[p] = index where p's src-group began → col = pos - group_start
    group_start = jax.lax.associative_scan(jnp.maximum, jnp.where(is_new, pos, 0))
    col = pos - group_start                                 # 0,1,2,... within group

    nbrs = jnp.full((N, V), -1, dtype=jnp.int32)
    # src_s==N (OOB row) and col>=V both dropped.
    nbrs = nbrs.at[src_s, col].set(dst_s, mode='drop')
    return nbrs
```

- [ ] **Step 4: Run, verify pass**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_angle_kernel.py -k neighbor -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add halflife/step.py tests/test_angle_kernel.py
git commit -m "feat(angles): per-particle neighbor list from edge graph"
```

---

### Task 4: Angle (triple) list

**Files:**
- Modify: `halflife/step.py` (add `build_angle_list` after `build_neighbor_list`)
- Test: `tests/test_angle_kernel.py`

- [ ] **Step 1: Write failing test**

```python
from halflife.step import build_angle_list


def _valid_triples(angles, pid):
    out = []
    for t in np.asarray(angles[:, :, :]).reshape(-1, 3) if False else []:
        pass
    return out


def test_angle_list_counts():
    # degree-2 center → 1 triple; degree-3 center → 3 triples (C(3,2))
    state, c = _world_with_edges([(1, 2), (2, 3)])
    nbrs = build_neighbor_list(state.composites, c)
    angles = build_angle_list(nbrs, c)                 # (N, P_max, 3)
    P = c.max_valence * (c.max_valence - 1) // 2
    assert angles.shape == (c.num_particles, P, 3)
    # particle 2 is the only valid center
    rows2 = np.asarray(angles[2])
    valid2 = rows2[(rows2[:, 0] >= 0)]
    assert valid2.shape[0] == 1                        # one triple
    assert set(valid2[0][[0, 2]]) == {1, 3}            # neighbors are 1,3
    assert valid2[0][1] == 2                           # center is 2

    state, c = _world_with_edges([(0, 1), (0, 2), (0, 3)])
    nbrs = build_neighbor_list(state.composites, c)
    angles = build_angle_list(nbrs, c)
    rows0 = np.asarray(angles[0])
    assert rows0[(rows0[:, 0] >= 0)].shape[0] == 3     # C(3,2) = 3 triples


def test_angle_list_free_particle_empty():
    state, c = _world_with_edges([(1, 2), (2, 3)])
    angles = build_angle_list(build_neighbor_list(state.composites, c), c)
    assert np.all(np.asarray(angles[6]) == -1)         # free particle, no triples
```

- [ ] **Step 2: Run, verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_angle_kernel.py -k angle_list -q`
Expected: FAIL (`cannot import name 'build_angle_list'`).

- [ ] **Step 3: Implement**

In `halflife/step.py`, after `build_neighbor_list`, add:

```python
import itertools  # at top of file with the other imports


def build_angle_list(nbrs: jnp.ndarray, config: SimConfig) -> jnp.ndarray:
    """
    Enumerate angle triples (i, j, k) — center j with neighbors i, k — from the
    per-particle neighbor list. (N, P_max, 3) int32, -1 padded.

    P_max = C(max_valence, 2); the column-pair list is generated in Python from
    the static max_valence, so the enumeration unrolls at trace time. A triple is
    invalid (all -1) if either neighbor column is -1.
    """
    N = config.num_particles
    V = config.max_valence
    pairs = list(itertools.combinations(range(V), 2))      # static, P_max entries
    cols_p = jnp.asarray([p for p, _ in pairs], dtype=jnp.int32)
    cols_q = jnp.asarray([q for _, q in pairs], dtype=jnp.int32)

    j_idx = jnp.arange(N, dtype=jnp.int32)
    i_ids = nbrs[:, cols_p]                                 # (N, P)
    k_ids = nbrs[:, cols_q]                                 # (N, P)
    j_ids = jnp.broadcast_to(j_idx[:, None], i_ids.shape)   # (N, P)

    angles = jnp.stack([i_ids, j_ids, k_ids], axis=-1)      # (N, P, 3)
    invalid = (i_ids < 0) | (k_ids < 0)
    return jnp.where(invalid[..., None], jnp.int32(-1), angles)
```

- [ ] **Step 4: Run, verify pass**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_angle_kernel.py -k angle_list -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add halflife/step.py tests/test_angle_kernel.py
git commit -m "feat(angles): per-particle angle-triple list"
```

---

### Task 5: `compute_angle_forces` — VSEPR mode

**Files:**
- Modify: `halflife/step.py` (add `compute_angle_forces`)
- Test: `tests/test_angle_kernel.py`

- [ ] **Step 1: Write failing test** (analytic + relaxation)

```python
from halflife.step import compute_angle_forces
from halflife.state import initialize_physics_params


def test_vsepr_two_bonds_open_and_conserve():
    # center 2 at origin; neighbor 1 at +x, neighbor 3 at +y → θ = 90°.
    # VSEPR should push 1 and 3 apart (toward 180°) and conserve momentum.
    state, c = _world_with_edges(
        [(1, 2), (2, 3)],
        positions={2: (5.0, 5.0), 1: (6.0, 5.0), 3: (5.0, 6.0)},
    )
    c = dataclasses.replace(c, angle_mode="vsepr", boundary_mode="open")
    phys = initialize_physics_params(c)
    F = np.asarray(compute_angle_forces(state, c, phys))
    # tangential opening: F on 1 has -y component, F on 3 has -x component
    assert F[1][1] < -1e-4
    assert F[3][0] < -1e-4
    # momentum conserved over the triple
    assert np.allclose(F[1] + F[2] + F[3], 0.0, atol=1e-4)


def test_vsepr_straight_is_equilibrium():
    # 1-2-3 collinear (180°) → ~zero angle force, no NaN
    state, c = _world_with_edges(
        [(1, 2), (2, 3)],
        positions={2: (5.0, 5.0), 1: (4.0, 5.0), 3: (6.0, 5.0)},
    )
    c = dataclasses.replace(c, angle_mode="vsepr", boundary_mode="open")
    F = np.asarray(compute_angle_forces(state, c, initialize_physics_params(c)))
    assert np.all(np.isfinite(F))
    assert np.allclose(F[1], 0.0, atol=1e-4)


def test_vsepr_relaxes_three_bonds_to_Y():
    # Integrate angle-only dynamics on a degree-3 star from a squished start;
    # the three pairwise angles should converge toward 120°.
    import jax
    state, c = _world_with_edges(
        [(0, 1), (0, 2), (0, 3)],
        positions={0: (5., 5.), 1: (6., 5.0), 2: (6., 5.3), 3: (4., 5.)},
    )
    c = dataclasses.replace(c, angle_mode="vsepr", boundary_mode="open")
    phys = initialize_physics_params(c)
    pos = np.asarray(state.particles.position)
    for _ in range(400):
        F = np.asarray(compute_angle_forces(
            state._replace(particles=state.particles._replace(
                position=jnp.asarray(pos))), c, phys))
        for pid in (1, 2, 3):
            pos[pid] += 0.02 * F[pid]                       # tiny overdamped step
            v = pos[pid] - pos[0]
            pos[pid] = pos[0] + v / (np.linalg.norm(v) + 1e-9)  # keep |bond|≈1
    def ang(p, q):
        u = pos[p] - pos[0]; w = pos[q] - pos[0]
        return np.degrees(np.arccos(np.clip(u@w/(np.linalg.norm(u)*np.linalg.norm(w)), -1, 1)))
    angs = sorted([ang(1, 2), ang(1, 3), ang(2, 3)])
    assert angs[0] > 90 and abs(np.mean(angs) - 120) < 15
```

- [ ] **Step 2: Run, verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_angle_kernel.py -k vsepr -q`
Expected: FAIL (`cannot import name 'compute_angle_forces'`).

- [ ] **Step 3: Implement** (both branches written now; harmonic is exercised in Task 6)

In `halflife/step.py`, after `build_angle_list`, add (import `_species_rest_angles`
from `halflife.chemistry` at the top of the file):

```python
def compute_angle_forces(state: WorldState, config: SimConfig,
                         physics: PhysicsParams) -> jnp.ndarray:
    """
    Per-triple angular forces over composite bonds. (N, 2) float32.

    Force law selected by static config.angle_mode:
      "vsepr"    — chord-Coulomb repulsion between bond directions; emergent even
                   spread (2π/degree), no frustration at degree ≥ 3.
      "harmonic" — pull cos θ toward cos θ0(central species) (smooth cosine form).

    Both laws are purely tangential (rotate bonds, never stretch them) and conserve
    linear & angular momentum per triple (F_j = -(F_i + F_k)). Min-image displacement
    matches compute_edge_bond_forces.
    """
    particles = state.particles
    composites = state.composites
    N = config.num_particles
    k_angle = physics.k_angle

    angles = build_angle_list(build_neighbor_list(composites, config), config)  # (N,P,3)
    i_id, j_id, k_id = angles[..., 0], angles[..., 1], angles[..., 2]            # (N,P)
    valid = i_id >= 0

    safe = lambda x: jnp.where(x >= 0, x, 0)
    pi = particles.position[safe(i_id)]   # (N,P,2)
    pj = particles.position[safe(j_id)]
    pk = particles.position[safe(k_id)]

    def min_image(d):
        if config.boundary_mode == "periodic":
            d = d - config.world_width  * jnp.round(d[..., 0:1] / config.world_width)  * jnp.array([1., 0.])
            d = d - config.world_height * jnp.round(d[..., 1:2] / config.world_height) * jnp.array([0., 1.])
        return d

    r_ji = min_image(pi - pj)
    r_jk = min_image(pk - pj)
    Lji = jnp.linalg.norm(r_ji, axis=-1) + 1e-10
    Ljk = jnp.linalg.norm(r_jk, axis=-1) + 1e-10
    ui = r_ji / Lji[..., None]
    uk = r_jk / Ljk[..., None]
    c = jnp.clip((ui * uk).sum(-1), -1.0, 1.0)   # cos θ, (N,P)

    if config.angle_mode == "vsepr":
        w = ui - uk
        d2 = (w * w).sum(-1) + 1e-6              # |û_i − û_k|² ; ε-softened core
        inv = d2 ** (-1.5)
        proj_i = w - ui * (ui * w).sum(-1, keepdims=True)        # tangential ⟂ û_i
        nw = -w
        proj_k = nw - uk * (uk * nw).sum(-1, keepdims=True)      # tangential ⟂ û_k
        g = (k_angle * inv)[..., None]
        f_i = g / Lji[..., None] * proj_i
        f_k = g / Ljk[..., None] * proj_k
    elif config.angle_mode == "harmonic":
        cos0 = jnp.cos(_species_rest_angles(config))            # (S,)
        c0 = cos0[particles.species[safe(j_id)]]                # (N,P)
        g = (k_angle * (c - c0))[..., None]
        f_i = -g / Lji[..., None] * (uk - c[..., None] * ui)
        f_k = -g / Ljk[..., None] * (ui - c[..., None] * uk)
    else:  # "off" — should not be reached (gated in simulation_step)
        return jnp.zeros((N, 2), dtype=jnp.float32)

    f_j = -(f_i + f_k)
    mask = valid[..., None].astype(jnp.float32)
    f_i, f_j, f_k = f_i * mask, f_j * mask, f_k * mask

    drop = lambda ids: jnp.where(valid, ids, N)
    forces = jnp.zeros((N, 2), dtype=jnp.float32)
    forces = forces.at[drop(i_id).reshape(-1)].add(f_i.reshape(-1, 2), mode='drop')
    forces = forces.at[drop(j_id).reshape(-1)].add(f_j.reshape(-1, 2), mode='drop')
    forces = forces.at[drop(k_id).reshape(-1)].add(f_k.reshape(-1, 2), mode='drop')
    return forces
```

- [ ] **Step 4: Run, verify pass**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_angle_kernel.py -k vsepr -q`
Expected: PASS (3 tests). If the Y-relaxation is flaky, raise iteration count or
lower the step size — it is an overdamped integrator, not a unit assertion.

- [ ] **Step 5: Commit**

```bash
git add halflife/step.py tests/test_angle_kernel.py
git commit -m "feat(angles): compute_angle_forces — VSEPR chord-Coulomb law"
```

---

### Task 6: Harmonic-θ₀ mode

**Files:**
- Test only: `tests/test_angle_kernel.py` (the `harmonic` branch was written in Task 5)

- [ ] **Step 1: Write failing test**

```python
def test_harmonic_drives_toward_theta0():
    # Put a degree-2 center at a 90° angle; harmonic should drive cos θ toward
    # cos θ0 of the center's species. Check the force sign matches (c - c0).
    state, c = _world_with_edges(
        [(1, 2), (2, 3)],
        positions={2: (5., 5.), 1: (6., 5.), 3: (5., 6.)},   # θ = 90°, c = 0
    )
    c = dataclasses.replace(c, angle_mode="harmonic", boundary_mode="open")
    phys = initialize_physics_params(c)
    sp = int(np.asarray(state.particles.species)[2])
    c0 = float(np.cos(np.asarray(_species_rest_angles(c))[sp]))
    F = np.asarray(compute_angle_forces(state, c, phys))
    assert np.all(np.isfinite(F))
    assert np.allclose(F[1] + F[2] + F[3], 0.0, atol=1e-4)   # momentum
    # If θ0 < 90° (c0 > 0) the bonds should close (θ decreasing); if θ0 > 90°
    # (c0 < 0) they should open. Either way the net effect is non-zero here.
    assert not np.allclose(F[1], 0.0, atol=1e-4)


def test_harmonic_smooth_at_180():
    state, c = _world_with_edges(
        [(1, 2), (2, 3)],
        positions={2: (5., 5.), 1: (4., 5.), 3: (6., 5.)},   # collinear, c = -1
    )
    c = dataclasses.replace(c, angle_mode="harmonic", boundary_mode="open")
    F = np.asarray(compute_angle_forces(state, c, initialize_physics_params(c)))
    assert np.all(np.isfinite(F))                            # no cusp / NaN at 180°


def test_harmonic_relaxes_degree2_to_theta0():
    state, c = _world_with_edges(
        [(1, 2), (2, 3)],
        positions={2: (5., 5.), 1: (6., 5.), 3: (5.2, 6.)},
    )
    c = dataclasses.replace(c, angle_mode="harmonic", boundary_mode="open")
    phys = initialize_physics_params(c)
    sp = int(np.asarray(state.particles.species)[2])
    theta0 = np.degrees(np.asarray(_species_rest_angles(c))[sp])
    pos = np.asarray(state.particles.position)
    for _ in range(600):
        F = np.asarray(compute_angle_forces(
            state._replace(particles=state.particles._replace(
                position=jnp.asarray(pos))), c, phys))
        for pid in (1, 3):
            pos[pid] += 0.02 * F[pid]
            v = pos[pid] - pos[2]
            pos[pid] = pos[2] + v / (np.linalg.norm(v) + 1e-9)
    u = pos[1] - pos[2]; w = pos[3] - pos[2]
    theta = np.degrees(np.arccos(np.clip(u@w/(np.linalg.norm(u)*np.linalg.norm(w)), -1, 1)))
    assert abs(theta - theta0) < 12
```

- [ ] **Step 2: Run, verify pass** (implementation already exists from Task 5)

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_angle_kernel.py -k harmonic -q`
Expected: PASS (3 tests). The relaxation test depends on the hashed θ0 for the
center species; if the species happens to hash θ0 ≈ 90° (already at rest), nudge the
start positions so the initial angle differs from θ0, or pin the species via the
helper so the test is deterministic.

- [ ] **Step 3: Commit**

```bash
git add tests/test_angle_kernel.py
git commit -m "test(angles): harmonic-θ0 force + relaxation coverage"
```

---

### Task 7: Wire into `simulation_step`, gated

**Files:**
- Modify: `halflife/step.py:267-273` (force summation)
- Test: `tests/test_angle_kernel.py`

- [ ] **Step 1: Write failing test**

```python
def test_off_mode_is_bit_identical():
    # angle_mode="off" must not change the integrated state vs a run with the
    # kernel absent. Compare one step's positions with off vs a tiny vsepr k.
    import jax
    from halflife.step import simulation_step
    base = dataclasses.replace(SimConfig(), num_particles=60, num_species=3,
                               max_composites=40, max_composite_size=8,
                               e_max=12, bond_mode="edges", angle_mode="off")
    from halflife.state import init_world_state
    from halflife.spatial import build_cell_list  # adjust to real step entry args
    s0 = init_world_state(base, jax.random.PRNGKey(1))
    # ... run one simulation_step in off mode, snapshot positions ...
    # then a second config identical but angle_mode="vsepr" and assert positions DIFFER.
    on = dataclasses.replace(base, angle_mode="vsepr")
    # (Fill in using the project's standard simulation_step invocation — see
    #  tests/test_step.py for the exact call signature / params construction.)
```

> The exact `simulation_step` call signature (params, neighbors, physics threading)
> is established in `tests/test_step.py` — copy that harness rather than guessing.
> The assertion that matters: **off ⇒ positions identical to a no-kernel run;
> vsepr ⇒ positions differ.**

- [ ] **Step 2: Run, verify it fails / is red**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_angle_kernel.py -k off_mode -q`

- [ ] **Step 3: Implement the gate**

In `halflife/step.py`, immediately after the bond-force block (after line 273,
before the `# ── Phase 4: Integration` comment), add:

```python
    # Angle-locking forces (VSEPR / harmonic) — geometry of bonded members.
    # Gated on edges mode (the angle list derives from the edge graph) and the
    # static angle_mode, so XLA traces only the live branch — zero cost when off.
    if config.angle_mode != "off" and config.bond_mode == "edges":
        forces = forces + compute_angle_forces(state, config, physics)
```

- [ ] **Step 4: Run, verify pass**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_angle_kernel.py -q`
Expected: all green.

- [ ] **Step 5: Regression — existing suites still pass**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_step.py tests/test_chemistry.py -q`
Expected: PASS (the `off` default means no behavioural change).

- [ ] **Step 6: Commit**

```bash
git add halflife/step.py tests/test_angle_kernel.py
git commit -m "feat(angles): wire compute_angle_forces into simulation_step (gated)"
```

---

### Task 8: Live app default + `k_angle` slider

**Files:**
- Modify: `halflife/main.py` (`build_config` — set `angle_mode="vsepr"`)
- Modify: `halflife/renderer.py:802-809` (add the slider to the edges branch)

- [ ] **Step 1: Live default**

In `halflife/main.py build_config`, alongside the existing `emit_events=True` live
default, set:

```python
    kwargs['angle_mode'] = 'vsepr'   # live app shows angle locking; headless/tests stay "off"
```

(Match the file's actual kwargs-assembly style — grep for `emit_events` in
`main.py` and mirror it.)

- [ ] **Step 2: Slider**

In `halflife/renderer.py`, in `_rebuild_physics_sliders` `edges` branch (line 803),
append to `bond_slots`:

```python
                ("k_angle",      "angle k",     _phys("k_angle"),      "{:.1f}", None),
```

- [ ] **Step 3: Manual smoke (user-run, GPU)**

```bash
.venv/bin/python -m halflife.main
```
Confirm: app launches, an "angle k" slider appears in edges mode, dragging it
visibly stiffens/relaxes bond geometry (chains straighten, branches hold a Y/cross).
No console NaN warnings.

- [ ] **Step 4: Commit**

```bash
git add halflife/main.py halflife/renderer.py
git commit -m "feat(angles): default live app to vsepr + k_angle slider"
```

---

### Task 9: Docs

**Files:**
- Modify: `CLAUDE.md` (Configuration section + a short "Angle locking" note)
- Modify: `README.md` (force-kernel section)

- [ ] **Step 1: CLAUDE.md** — under Configuration, document `angle_mode` /
  `k_angle` / `theta_min_deg` / `theta_max_deg`; add a paragraph mirroring the
  ring-closure / valence prose: VSEPR even-spreading vs harmonic-θ0; gated on
  `angle_mode != "off" AND bond_mode == "edges"`; per-particle angle list
  recomputed each step; live app defaults to `vsepr`, headless/tests `off`. Note the
  `harmonic` mode is intended for degree ≤ 2 (over-determined at degree ≥ 3) and that
  the angle potential is **not** in `energy.py` conservation tracking (v1).
  **Preserve all existing comments/prose** — additive only.

- [ ] **Step 2: README.md** — add the angle term to the force-kernel description
  with the VSEPR vs harmonic equations from the spec (§5.3).

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs(angles): document VSEPR/harmonic angle-locking kernel"
```

---

## Self-review notes

- **Spec coverage:** config/physics (T1), θ0 hash (T2), neighbor list (T3), angle
  list (T4), VSEPR force (T5), harmonic force (T6), gated wiring (T7), live
  default + slider (T8), docs (T9). All spec §5-6 components covered.
- **Type consistency:** `compute_angle_forces(state, config, physics)` signature is
  fixed across T5/T7; `build_neighbor_list(composites, config)` and
  `build_angle_list(nbrs, config)` consistent across T3/T4/T5; `_species_rest_angles`
  returns radians, cos'd in both the kernel and tests.
- **Known soft spots flagged inline:** (a) the `_world_with_edges` helper touches
  `state.py` internals — verify field names first; (b) the `simulation_step`
  invocation in T7 must be copied from `tests/test_step.py`, not guessed; (c)
  relaxation tests are overdamped integrators with tolerance bands, not exact
  assertions — adjust iterations/step if flaky; (d) harmonic relaxation depends on
  the hashed θ0 — pin or nudge so the start angle ≠ θ0.
- **Out of scope (per spec §8):** electrostatic polarity, runtime lone pairs,
  energy-conservation accounting, dihedrals.
