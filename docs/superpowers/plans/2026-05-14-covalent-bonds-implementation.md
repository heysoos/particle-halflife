# Sparse Covalent Bonds — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the COM-spring bond force with sparse covalent bonds — explicit edges between specific particle pairs with hash-derived per-species-pair rest lengths — and allow ring closure via relaxed intra-composite fusion. The current star-spring stays in the codebase, selectable via a `bond_mode` config flag for live A/B comparison.

**Architecture:** Three layers, built bottom-up. **(1) Data:** `CompositeState` gains `edges` and `edge_count` arrays; `InteractionParams` gains an `r_rest` matrix; `SimConfig` gains `bond_mode`, `k_bond`, `r_rest_min/max`, `allow_ring_closure`, `max_ring_closures_per_step`. **(2) Helpers:** pure functions for the edge-spring force kernel, the per-particle `degree` scatter, the composite-level free-bond cache, the hash-to-rest-length helper, and the path-spanning-tree edge builder. **(3) Step wiring:** dispatch the bond-force kernel by `bond_mode`; modify the fusion gate to use per-particle valence; append edges in the fusion scan body; add a new ring-closure phase (Phase 6b); rebuild edges in fission. Finally a runtime keyboard toggle in `main.py` and a renderer update to draw real edges instead of the existing forward-slot visual heuristic.

**Tech Stack:** JAX (jit, vmap, scan, lax), pytest, ModernGL+pygame for the live render.

**Spec:** [docs/superpowers/specs/2026-05-14-covalent-bonds-design.md](../specs/2026-05-14-covalent-bonds-design.md)

---

## File structure

| File | Responsibility | Touched in tasks |
|---|---|---|
| `halflife/config.py` | Add `bond_mode`, `k_bond`, `r_rest_min`, `r_rest_max`, `allow_ring_closure`, `max_ring_closures_per_step`, and derived `E_max` property | A |
| `halflife/state.py` | Add `edges`, `edge_count` to `CompositeState`; initialize empty; add `r_rest` to `InteractionParams` with hash-derived init | B, C |
| `halflife/chemistry.py` | Add `_hash_to_rest_length`, `compute_degree`, `compute_composite_free_bonds`; modify fusion gate to per-particle valence; append edges in `fusion_scan_body`; new `attempt_ring_closure`; rebuild edges in `apply_composite_decay`; helper `build_path_spanning_tree`; `initialize_edges_for_existing_composites` | D, H, I, J, K, L |
| `halflife/step.py` | New `compute_edge_bond_forces`; bond-mode dispatch in `simulation_step`; wire ring closure and `degree` updates | E, F, G |
| `halflife/main.py` | Add `M` key handling that cycles `bond_mode` and re-initializes edges when entering `"edges"` mode | M |
| `halflife/renderer.py` | Switch bond drawing from forward-slot heuristic to iteration over `composites.edges` | N |
| `tests/test_chemistry.py` | Add tests for `_hash_to_rest_length`, `compute_degree`, `compute_composite_free_bonds`, per-particle fusion gate, edge append on fusion, ring closure, fission edge rebuild, toggle init | D, G, H, I, J, K, L |
| `tests/test_step.py` | Add test for `compute_edge_bond_forces` on a known 2-particle composite | E |
| `tests/test_integration.py` (new) | End-to-end smoke test: 200 steps with `bond_mode="edges"` doesn't crash and produces composites | O |

---

## Conventions

- This repo runs Python natively in WSL. Activate the venv once per shell: `source .venv/bin/activate`.
- Tests need `PYTHONPATH=.` because there's no `pyproject.toml` or `setup.py` in the repo root.
- **Force CPU for tests** to avoid GPU contention with a possibly-live sim: `JAX_PLATFORMS=cpu PYTHONPATH=. pytest …`.
- Git has no global identity in WSL; every commit must include `-c user.email='heysoos@local' -c user.name='Heysoos'`.
- Never use `git add -A` or `git add .` — there are untracked files (`.idea/`, `__pycache__/`, `bash.exe.stackdump`, `init_prompt.txt`) that must not be committed.
- **Preserve all existing comments.** This project's `CLAUDE.md` explicitly forbids comment deletion. When editing functions, keep every comment that isn't describing code being removed. New code follows the default "minimal comments" rule.

Baseline: before starting, verify the test suite passes:

```bash
source .venv/bin/activate
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/ -x --tb=short
```

Expected: green baseline. If anything fails, fix it before starting; this plan assumes a green baseline.

---

## Task A: Add `SimConfig` knobs for the new bond model

**Files:**
- Modify: `halflife/config.py` (add fields, add `e_max` property)
- Test: `tests/test_chemistry.py` (smoke test that config initializes with new fields)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chemistry.py` (new section near the bottom, before `if __name__`):

```python
def test_config_has_bond_mode_fields():
    """SimConfig exposes the new bond-mode knobs with safe defaults."""
    config = SimConfig()
    assert config.bond_mode in ("edges", "star_spring", "off")
    assert config.bond_mode == "star_spring", "Default should preserve current behavior"
    assert config.k_bond > 0
    assert config.r_rest_min > 0
    assert config.r_rest_max > config.r_rest_min
    assert isinstance(config.allow_ring_closure, bool)
    assert config.max_ring_closures_per_step > 0
    # Derived: E_max = M * max_valence // 2
    assert config.e_max == (config.max_composite_size * config.max_valence) // 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/test_chemistry.py::test_config_has_bond_mode_fields -v
```

Expected: FAIL with `AttributeError: 'SimConfig' object has no attribute 'bond_mode'`.

- [ ] **Step 3: Add the fields and derived property to `halflife/config.py`**

In `halflife/config.py`, find the existing `use_bond_forces` / `spring_k` block (around lines 100-107) and add the new fields **after** it (do not delete or modify the existing ones — `star_spring` mode reads them):

```python
    # ── Sparse covalent bonds (new bond model) ───────────────────────────────
    # bond_mode selects which kernel runs in Phase 3b:
    #   "edges"       — sparse covalent bonds (new)
    #   "star_spring" — current COM-spring (legacy; reads spring_k, use_bond_forces)
    #   "off"         — no bond force; pure pairwise dynamics
    # Static field — changes trigger one-time JAX retrace per mode.
    bond_mode: str = "star_spring"

    # Harmonic stiffness for edge-mode bonds. Much larger than spring_k because
    # each edge is a local constraint, not an aggregate COM tie — so a bonded
    # pair at displacement 1 from rest length should still feel a force well
    # above the species-pair attraction (~1).
    k_bond: float = 20.0

    # Range for hash-derived per-species-pair rest lengths.
    # r_rest_min: comfortably outside repulsion_radius so bonded pairs don't
    #             sit inside the hard core.
    # r_rest_max: comfortably inside fusion_radius so bonds don't auto-cleave
    #             on small perturbations.
    r_rest_min: float = 1.2
    r_rest_max: float = 3.6

    # Ring closure: allow intra-composite fusion when both members still have
    # per-particle free bonds (degree[i] < v_{species[i]}).
    allow_ring_closure: bool = True
    max_ring_closures_per_step: int = 50
```

Then add the `e_max` property at the bottom of the `SimConfig` class alongside the other `@property` accessors:

```python
    @property
    def e_max(self) -> int:
        """Maximum edges per composite: enough for any all-bonds-used graph."""
        return (self.max_composite_size * self.max_valence) // 2
```

- [ ] **Step 4: Run test to verify it passes**

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/test_chemistry.py::test_config_has_bond_mode_fields -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add halflife/config.py tests/test_chemistry.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "feat(config): add bond_mode/k_bond/r_rest/ring-closure knobs

Adds the new sparse-bond config surface alongside the existing star-spring
knobs. bond_mode defaults to 'star_spring' so behavior is unchanged until
the new force kernel is wired in a later task."
```

---

## Task B: Add `r_rest` matrix to `InteractionParams` with hash-derived init

**Files:**
- Modify: `halflife/state.py` (add field, add init helper)
- Modify: `halflife/chemistry.py` (add `_hash_to_rest_length`)
- Test: `tests/test_chemistry.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chemistry.py`:

```python
def test_r_rest_matrix_shape_and_symmetry():
    """r_rest is (S, S), symmetric, and values fall in [r_rest_min, r_rest_max]."""
    config = SimConfig(num_species=5, bond_mode="edges")
    params = initialize_interaction_params(config, seed=42)
    assert params.r_rest.shape == (5, 5)
    # Symmetric: r_rest[i, j] == r_rest[j, i]
    diff = np.asarray(params.r_rest - params.r_rest.T)
    assert np.max(np.abs(diff)) < 1e-6, f"r_rest is not symmetric: max diff {np.max(np.abs(diff))}"
    # In range
    vals = np.asarray(params.r_rest)
    assert vals.min() >= config.r_rest_min - 1e-6
    assert vals.max() <= config.r_rest_max + 1e-6


def test_r_rest_is_deterministic_per_hash_modulus():
    """Same config + hash_modulus → same r_rest matrix (hash-determined)."""
    c1 = SimConfig(num_species=5)
    c2 = SimConfig(num_species=5)
    p1 = initialize_interaction_params(c1, seed=42)
    p2 = initialize_interaction_params(c2, seed=99)  # seed ignored by r_rest
    np.testing.assert_array_almost_equal(np.asarray(p1.r_rest), np.asarray(p2.r_rest))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/test_chemistry.py::test_r_rest_matrix_shape_and_symmetry tests/test_chemistry.py::test_r_rest_is_deterministic_per_hash_modulus -v
```

Expected: FAIL with `AttributeError: 'InteractionParams' object has no attribute 'r_rest'`.

- [ ] **Step 3: Add `_hash_to_rest_length` to `halflife/chemistry.py`**

In `halflife/chemistry.py`, after `_hash_to_valence` (around line 116), add:

```python
def _hash_to_rest_length(s_i: jnp.ndarray, s_j: jnp.ndarray,
                         config: SimConfig) -> jnp.ndarray:
    """
    Hash-derived bond rest length for species pair (s_i, s_j).

    Order-independent (uses the same commutative additive hash as composites)
    so r_rest[i, j] == r_rest[j, i] without an explicit symmetry pass.
    Re-mixed with a Fibonacci hash to decorrelate from binding energy and
    valence so universes with the same num_species but different hash_modulus
    get genuinely different bond chemistries.

    Returns:
        scalar float32 in [config.r_rest_min, config.r_rest_max]
    """
    h_i = _entity_hash_val(s_i, config).astype(jnp.uint32)
    h_j = _entity_hash_val(s_j, config).astype(jnp.uint32)
    h = (h_i + h_j) % jnp.uint32(config.hash_modulus)
    # Fibonacci re-mix to decorrelate from BE / valence streams.
    h_mix = (h * jnp.uint32(0x9E3779B1)) ^ (h >> jnp.uint32(11))
    frac = (h_mix % jnp.uint32(1000)).astype(jnp.float32) / 999.0
    return jnp.float32(config.r_rest_min) + frac * jnp.float32(
        config.r_rest_max - config.r_rest_min
    )
```

- [ ] **Step 4: Add `r_rest` field to `InteractionParams` and init in `initialize_interaction_params`**

In `halflife/state.py`, modify the `InteractionParams` NamedTuple (around line 154) — add `r_rest` as the last field:

```python
class InteractionParams(NamedTuple):
    """
    Species-dependent pairwise force parameters. Not part of WorldState.
    Passed as a regular JAX array argument (not static), so these can be
    changed without recompiling the simulation step.

    Per-pair radii are stored as FRACTIONS of config.interaction_radius:
      r_peak[i,j]   = interaction_radius * peak_fraction[i,j]
      r_cutoff[i,j] = interaction_radius * cutoff_fraction[i,j]
    Repulsion is global (physics.repulsion_radius, scalar) — steric exclusion
    is universal across species pairs. Ordering invariant enforced at init:
      0 < repulsion_fraction < peak_fraction[i,j] < cutoff_fraction[i,j] <= 1
    """
    # All matrices: (num_species, num_species) float32
    attraction:        jnp.ndarray  # signed strength in [-1, 1]
    peak_fraction:     jnp.ndarray  # peak-attraction radius / interaction_radius
    cutoff_fraction:   jnp.ndarray  # zero-force radius / interaction_radius
    r_rest:            jnp.ndarray  # hash-derived bond rest length per species pair
```

Then in `initialize_interaction_params` (around line 173), after the existing `cutoff_fraction` setup, add the `r_rest` matrix construction and include it in the returned tuple:

```python
    # ... existing code that builds attraction, peak_fraction, cutoff_fraction ...

    # Hash-derived per-species-pair bond rest length. Symmetric by construction
    # (uses the commutative species-pair hash). Independent of `seed` — this
    # is part of the "universe" determined by config.hash_modulus, like valence.
    from halflife.chemistry import _hash_to_rest_length  # local import: chemistry imports state
    species_idx = jnp.arange(S, dtype=jnp.int32)
    r_rest = jax.vmap(
        lambda i: jax.vmap(lambda j: _hash_to_rest_length(i, j, config))(species_idx)
    )(species_idx)  # (S, S)

    return InteractionParams(
        attraction=attraction,
        peak_fraction=peak_fraction,
        cutoff_fraction=cutoff_fraction,
        r_rest=r_rest,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/test_chemistry.py::test_r_rest_matrix_shape_and_symmetry tests/test_chemistry.py::test_r_rest_is_deterministic_per_hash_modulus -v
```

Expected: both PASS.

- [ ] **Step 6: Run the full chemistry test suite to ensure nothing regressed**

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/test_chemistry.py -x --tb=short
```

Expected: all pass (existing tests untouched).

- [ ] **Step 7: Commit**

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add halflife/state.py halflife/chemistry.py tests/test_chemistry.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "feat(chemistry): add r_rest matrix with hash-derived rest lengths

Adds r_rest as a hash-determined (S,S) symmetric matrix on InteractionParams,
populated at config time via _hash_to_rest_length. Different hash_modulus
values produce different bond chemistries. Not yet consumed by any kernel."
```

---

## Task C: Add `edges` and `edge_count` to `CompositeState`

**Files:**
- Modify: `halflife/state.py` (add fields to `CompositeState`, init in `initialize_world`)
- Test: `tests/test_chemistry.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chemistry.py`:

```python
def test_composite_state_has_edges_fields():
    """CompositeState exposes edges and edge_count, initialized empty."""
    config = SimConfig(num_species=3, num_particles=100)
    world = initialize_world(config, seed=0)
    C = config.max_composites
    E = config.e_max
    assert world.composites.edges.shape == (C, E, 2)
    assert world.composites.edges.dtype == jnp.int32
    assert world.composites.edge_count.shape == (C,)
    assert world.composites.edge_count.dtype == jnp.int32
    # Edges initialized to -1 (sentinel = unused slot)
    edges_np = np.asarray(world.composites.edges)
    assert (edges_np == -1).all()
    # Edge count initialized to 0
    counts_np = np.asarray(world.composites.edge_count)
    assert (counts_np == 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/test_chemistry.py::test_composite_state_has_edges_fields -v
```

Expected: FAIL with `AttributeError` on `composites.edges`.

- [ ] **Step 3: Add fields to `CompositeState` and init them**

In `halflife/state.py`, modify the `CompositeState` NamedTuple (around line 41) — keep all existing fields, add `edges` and `edge_count` at the end:

```python
class CompositeState(NamedTuple):
    """
    State of the composite pool. Leading dimension is MAX_COMPOSITES.
    A composite is a group of bonded particles.

    members[c, :member_count[c]] are the particle indices belonging to composite c.
    slots beyond member_count are padded with -1.

    edges[c, :edge_count[c]] are particle-id pairs (i, j) for each bond.
    slots beyond edge_count are padded with -1. Bond force is a harmonic
    spring per edge in edge-mode; ignored in star_spring / off mode.
    """
    members:        jnp.ndarray  # (C, M) int32   — particle indices per composite
    member_count:   jnp.ndarray  # (C,)   int32   — number of valid members
    alive:          jnp.ndarray  # (C,)   bool    — active composite mask
    binding_energy: jnp.ndarray  # (C,)   float32 — energy released on formation
    half_life:      jnp.ndarray  # (C,)   float32 — composite decay half-life
    age:            jnp.ndarray  # (C,)   float32 — time since formation
    species_hash:   jnp.ndarray  # (C,)   uint32  — hash of sorted member species
    free_bonds:     jnp.ndarray  # (C,)   int32   — composite-level free-bond cache (Σ v_s − 2·edge_count)
    edges:          jnp.ndarray  # (C, E_max, 2) int32 — bond particle-id pairs; -1 = unused
    edge_count:     jnp.ndarray  # (C,)   int32   — number of valid edges
```

Then in `initialize_world` (around line 128), update the `CompositeState` construction:

```python
    # ── Composites ───────────────────────────────────────────────────────────
    E = config.e_max
    composites = CompositeState(
        members=jnp.full((C, M), -1, dtype=jnp.int32),
        member_count=jnp.zeros(C, dtype=jnp.int32),
        alive=jnp.zeros(C, dtype=bool),
        binding_energy=jnp.zeros(C, dtype=jnp.float32),
        half_life=jnp.zeros(C, dtype=jnp.float32),
        age=jnp.zeros(C, dtype=jnp.float32),
        species_hash=jnp.zeros(C, dtype=jnp.uint32),
        free_bonds=jnp.zeros(C, dtype=jnp.int32),
        edges=jnp.full((C, E, 2), -1, dtype=jnp.int32),
        edge_count=jnp.zeros(C, dtype=jnp.int32),
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/test_chemistry.py::test_composite_state_has_edges_fields -v
```

Expected: PASS.

- [ ] **Step 5: Run the full test suite to catch any field-order regressions**

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/ -x --tb=short
```

Expected: all pass. Existing tests that construct `CompositeState._replace(...)` should still work because we only added fields at the end.

- [ ] **Step 6: Commit**

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add halflife/state.py tests/test_chemistry.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "feat(state): add edges and edge_count fields to CompositeState

Adds two new (C, E_max, 2) and (C,) arrays for explicit bond edges. Initialized
empty (-1 sentinel for unused slots). Not yet consumed by any kernel; star_spring
and off modes ignore them."
```

---

## Task D: Implement `compute_degree` and `compute_composite_free_bonds`

**Files:**
- Modify: `halflife/chemistry.py` (add two helpers)
- Test: `tests/test_chemistry.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chemistry.py`:

```python
def test_compute_degree_on_known_edges():
    """degree[i] equals the count of edges incident to particle i."""
    from halflife.chemistry import compute_degree
    config = SimConfig(num_species=3, num_particles=10, max_composites=4)
    world = initialize_world(config, seed=0)
    # Hand-build two composites:
    # composite 0: members [0, 1, 2], edges [(0,1), (1,2)]  → degrees: 0→1, 1→2, 2→1
    # composite 1: members [3, 4], edges [(3,4)]            → degrees: 3→1, 4→1
    C = config.max_composites
    E = config.e_max
    edges = np.full((C, E, 2), -1, dtype=np.int32)
    edge_count = np.zeros(C, dtype=np.int32)
    alive = np.zeros(C, dtype=bool)
    edges[0, 0] = (0, 1); edges[0, 1] = (1, 2); edge_count[0] = 2; alive[0] = True
    edges[1, 0] = (3, 4);                       edge_count[1] = 1; alive[1] = True
    composites = world.composites._replace(
        edges=jnp.asarray(edges),
        edge_count=jnp.asarray(edge_count),
        alive=jnp.asarray(alive),
    )

    degree = compute_degree(composites, config)
    deg = np.asarray(degree)
    assert deg[0] == 1
    assert deg[1] == 2
    assert deg[2] == 1
    assert deg[3] == 1
    assert deg[4] == 1
    assert (deg[5:] == 0).all()  # particles 5-9 are free, degree 0


def test_compute_composite_free_bonds_matches_per_particle():
    """composite_free_bonds[c] = Σ (v_s[species[m]] - degree[m]) over members."""
    from halflife.chemistry import compute_degree, compute_composite_free_bonds, _species_valences
    config = SimConfig(num_species=3, num_particles=10, max_composites=4, max_valence=4)
    world = initialize_world(config, seed=0)
    # Same setup as above
    C = config.max_composites
    E = config.e_max
    edges = np.full((C, E, 2), -1, dtype=np.int32)
    edge_count = np.zeros(C, dtype=np.int32)
    alive = np.zeros(C, dtype=bool)
    members = np.full((C, config.max_composite_size), -1, dtype=np.int32)
    member_count = np.zeros(C, dtype=np.int32)
    edges[0, 0] = (0, 1); edges[0, 1] = (1, 2); edge_count[0] = 2; alive[0] = True
    members[0, :3] = (0, 1, 2); member_count[0] = 3
    edges[1, 0] = (3, 4);                       edge_count[1] = 1; alive[1] = True
    members[1, :2] = (3, 4); member_count[1] = 2
    composites = world.composites._replace(
        edges=jnp.asarray(edges),
        edge_count=jnp.asarray(edge_count),
        alive=jnp.asarray(alive),
        members=jnp.asarray(members),
        member_count=jnp.asarray(member_count),
    )

    sv = _species_valences(config)  # (S,)
    degree = compute_degree(composites, config)
    cfb = compute_composite_free_bonds(world.particles, composites, degree, sv, config)
    cfb_np = np.asarray(cfb)
    sp = np.asarray(world.particles.species)
    sv_np = np.asarray(sv)
    # Expected per-composite free bonds
    expected_0 = (sv_np[sp[0]] - 1) + (sv_np[sp[1]] - 2) + (sv_np[sp[2]] - 1)
    expected_1 = (sv_np[sp[3]] - 1) + (sv_np[sp[4]] - 1)
    assert cfb_np[0] == expected_0
    assert cfb_np[1] == expected_1
    assert (cfb_np[2:] == 0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/test_chemistry.py::test_compute_degree_on_known_edges tests/test_chemistry.py::test_compute_composite_free_bonds_matches_per_particle -v
```

Expected: FAIL with `ImportError: cannot import name 'compute_degree' from 'halflife.chemistry'`.

- [ ] **Step 3: Implement `compute_degree` in `halflife/chemistry.py`**

Add after `_species_valences` (around line 117):

```python
def compute_degree(composites, config: SimConfig) -> jnp.ndarray:
    """
    Per-particle edge-incidence count, summed across all alive composites.

    For each valid edge (i, j) in every alive composite, increment degree[i]
    and degree[j] by 1. Returns (N,) int32. Used by the per-particle valence
    gate in fusion and by the ring-closure scan.

    Args:
        composites: CompositeState
        config:     SimConfig (static)

    Returns:
        (N,) int32 — degree[i] = number of edges incident to particle i
    """
    N = config.num_particles
    C = config.max_composites
    E = config.e_max

    # Mask edges by alive composite AND valid slot index (<= edge_count[c]).
    # Each edge slot contributes 2 scatter-adds (one per endpoint).
    e_idx = jnp.arange(E, dtype=jnp.int32)  # (E,)
    valid = composites.alive[:, None] & (e_idx[None, :] < composites.edge_count[:, None])  # (C, E)

    pid_a = composites.edges[:, :, 0]  # (C, E)
    pid_b = composites.edges[:, :, 1]  # (C, E)

    # Route invalid entries to index N (OOB, dropped via mode='drop').
    drop_a = jnp.where(valid, pid_a, N)
    drop_b = jnp.where(valid, pid_b, N)

    degree = jnp.zeros(N, dtype=jnp.int32)
    degree = degree.at[drop_a.reshape(-1)].add(1, mode='drop')
    degree = degree.at[drop_b.reshape(-1)].add(1, mode='drop')
    return degree
```

- [ ] **Step 4: Implement `compute_composite_free_bonds` in `halflife/chemistry.py`**

Add immediately after `compute_degree`:

```python
def compute_composite_free_bonds(particles, composites, degree: jnp.ndarray,
                                  species_valences: jnp.ndarray,
                                  config: SimConfig) -> jnp.ndarray:
    """
    Per-composite free-bond cache.

    composite_free_bonds[c] = Σ (v_{species[m]} − degree[m]) over m in members[c]
                            = Σ v_{species[m]} − 2 · edge_count[c]

    (Equivalent because each edge contributes 1 to two endpoint degrees.)

    Used as the cheap (C,) skip mask for the ring-closure scan: composites with
    free_bonds < 2 contribute zero work because they can't add another edge.

    Args:
        particles, composites: state
        degree:                (N,) int32 from compute_degree
        species_valences:      (S,) int32 from _species_valences
        config:                SimConfig (static)

    Returns:
        (C,) int32 — composite-level free bonds
    """
    M = config.max_composite_size
    C = config.max_composites
    m_idx = jnp.arange(M, dtype=jnp.int32)

    def per_composite(c):
        members = composites.members[c]  # (M,)
        n = composites.member_count[c]
        valid = composites.alive[c] & (members >= 0) & (m_idx < n)
        safe_m = jnp.where(valid, members, 0)
        per_particle_free = species_valences[particles.species[safe_m]] - degree[safe_m]
        return jnp.sum(jnp.where(valid, per_particle_free, 0))

    return jax.vmap(per_composite)(jnp.arange(C, dtype=jnp.int32))
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/test_chemistry.py::test_compute_degree_on_known_edges tests/test_chemistry.py::test_compute_composite_free_bonds_matches_per_particle -v
```

Expected: both PASS.

- [ ] **Step 6: Commit**

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add halflife/chemistry.py tests/test_chemistry.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "feat(chemistry): add compute_degree and compute_composite_free_bonds

Pure helpers used by the per-particle valence fusion gate and by the
ring-closure scan's skip mask. Not yet wired into the step."
```

---

## Task E: Implement `compute_edge_bond_forces`

**Files:**
- Modify: `halflife/step.py` (add new function, keep existing `compute_bond_forces`)
- Test: `tests/test_step.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_step.py`:

```python
def test_edge_bond_force_on_two_particle_composite():
    """
    Two particles bonded by a single edge feel equal-and-opposite forces
    pointing along the connecting axis, with magnitude k_bond * (r - r_rest).
    """
    from halflife.step import compute_edge_bond_forces
    from halflife.state import initialize_world, initialize_interaction_params, initialize_physics_params
    config = SimConfig(num_species=3, num_particles=4, max_composites=2,
                       boundary_mode="reflect", world_width=100.0, world_height=100.0,
                       k_bond=10.0, r_rest_min=2.0, r_rest_max=2.0)
    world = initialize_world(config, seed=0)
    params = initialize_interaction_params(config, seed=0)
    physics = initialize_physics_params(config)

    # Place particles 0 and 1 on the x-axis 5 units apart (r > r_rest=2 → attractive)
    pos = np.array([[10.0, 10.0],   # particle 0
                    [15.0, 10.0],   # particle 1
                    [50.0, 50.0],   # particle 2 (free, no bond)
                    [60.0, 60.0]],  # particle 3 (free, no bond)
                   dtype=np.float32)
    species = np.array([0, 0, 1, 1], dtype=np.int32)  # all species 0/1
    composite_id = np.array([0, 0, -1, -1], dtype=np.int32)
    members = np.full((2, config.max_composite_size), -1, dtype=np.int32)
    members[0, :2] = (0, 1)
    edges = np.full((2, config.e_max, 2), -1, dtype=np.int32)
    edges[0, 0] = (0, 1)
    edge_count = np.array([1, 0], dtype=np.int32)
    alive = np.array([True, False], dtype=bool)

    world = world._replace(
        particles=world.particles._replace(
            position=jnp.asarray(pos), species=jnp.asarray(species),
            composite_id=jnp.asarray(composite_id),
        ),
        composites=world.composites._replace(
            members=jnp.asarray(members), member_count=jnp.array([2, 0], dtype=jnp.int32),
            alive=jnp.asarray(alive), edges=jnp.asarray(edges),
            edge_count=jnp.asarray(edge_count),
        ),
    )

    forces = compute_edge_bond_forces(world, params, config, physics)
    f = np.asarray(forces)
    # Expected: r = 5, r_rest = 2 → stretched by 3, restoring force pulls inward
    # F on particle 0 = -k_bond * (r - r_rest) * (pos_0 - pos_1)/r
    #                 = -10 * (5 - 2) * (-1, 0)/1 ... wait, (pos_0 - pos_1)/r = (-5, 0)/5 = (-1, 0)
    # Hmm let me redo: harmonic spring force on i = -k * (r - r_rest) * r̂_ij where r̂_ij = (pos_i - pos_j)/r
    # r̂ for particle 0 = (10-15, 0)/5 = (-1, 0). Stretch 3, k=10 → force on 0 = -10 * 3 * (-1, 0) = (30, 0).
    # That points particle 0 toward particle 1 — correct (attractive when stretched).
    np.testing.assert_allclose(f[0], [30.0, 0.0], atol=1e-4)
    np.testing.assert_allclose(f[1], [-30.0, 0.0], atol=1e-4)
    np.testing.assert_allclose(f[2], [0.0, 0.0], atol=1e-4)
    np.testing.assert_allclose(f[3], [0.0, 0.0], atol=1e-4)
```

Note: the test uses `r_rest_min == r_rest_max == 2.0` so the hash-derived rest length collapses to exactly 2.0 regardless of species pair.

- [ ] **Step 2: Run test to verify it fails**

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/test_step.py::test_edge_bond_force_on_two_particle_composite -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `compute_edge_bond_forces` in `halflife/step.py`**

Add after `compute_bond_forces` (around line 115; do NOT modify or delete `compute_bond_forces` — it stays for star_spring mode):

```python
def compute_edge_bond_forces(state: WorldState, params: InteractionParams,
                              config: SimConfig, physics: PhysicsParams) -> jnp.ndarray:
    """
    Per-edge harmonic spring forces (sparse covalent bonds).

    For each valid edge (i, j) in every alive composite:
        F_on_i = -k_bond * (r - r_rest[s_i, s_j]) * (pos_i - pos_j) / r
        F_on_j = -F_on_i
    Forces are scatter-added into a (N, 2) buffer. Min-image displacement is
    used so bonds across periodic boundaries don't snap.

    Cost: O(C · E_max) vmap cells. Per-cell work is one min-image distance,
    one species-pair gather (r_rest), one harmonic spring evaluation.

    Returns: (N, 2) float32 — additional forces from bonds
    """
    particles = state.particles
    composites = state.composites
    N = config.num_particles
    C = config.max_composites
    E = config.e_max
    k = jnp.float32(config.k_bond)

    e_idx = jnp.arange(E, dtype=jnp.int32)

    def per_composite(c):
        is_alive = composites.alive[c]
        count    = composites.edge_count[c]
        valid_e  = is_alive & (e_idx < count)  # (E,)

        pid_a = composites.edges[c, :, 0]  # (E,)
        pid_b = composites.edges[c, :, 1]  # (E,)
        safe_a = jnp.where(pid_a >= 0, pid_a, 0)
        safe_b = jnp.where(pid_b >= 0, pid_b, 0)

        pa = particles.position[safe_a]  # (E, 2)
        pb = particles.position[safe_b]  # (E, 2)
        sa = particles.species[safe_a]   # (E,)
        sb = particles.species[safe_b]   # (E,)

        d = pa - pb
        if config.boundary_mode == "periodic":
            d = d - config.world_width  * jnp.round(d[..., 0:1] / config.world_width)  * jnp.array([1., 0.])
            d = d - config.world_height * jnp.round(d[..., 1:2] / config.world_height) * jnp.array([0., 1.])
        r = jnp.linalg.norm(d, axis=-1) + 1e-10  # (E,)
        d_hat = d / r[:, None]                    # (E, 2)
        r_rest = params.r_rest[sa, sb]            # (E,)

        # F_on_i = -k * (r - r_rest) * d_hat   ; d_hat = (pos_i - pos_j) / r
        # When r > r_rest (stretched), force pulls i toward j (along -d_hat in actual position space).
        f_on_a = -k * (r - r_rest)[:, None] * d_hat  # (E, 2)
        f_on_b = -f_on_a                              # Newton's third law

        # Mask out invalid edges
        mask = valid_e[:, None].astype(jnp.float32)
        f_on_a = f_on_a * mask
        f_on_b = f_on_b * mask

        # Route invalid pids to OOB index N → mode='drop' silently discards.
        drop_a = jnp.where(valid_e, pid_a, N)
        drop_b = jnp.where(valid_e, pid_b, N)
        return drop_a, drop_b, f_on_a, f_on_b

    pid_a_all, pid_b_all, f_a_all, f_b_all = jax.vmap(per_composite)(
        jnp.arange(C, dtype=jnp.int32)
    )  # (C, E), (C, E), (C, E, 2), (C, E, 2)

    flat_pid_a = pid_a_all.reshape(-1)
    flat_pid_b = pid_b_all.reshape(-1)
    flat_f_a   = f_a_all.reshape(-1, 2)
    flat_f_b   = f_b_all.reshape(-1, 2)

    forces = jnp.zeros((N, 2), dtype=jnp.float32)
    forces = forces.at[flat_pid_a].add(flat_f_a, mode='drop')
    forces = forces.at[flat_pid_b].add(flat_f_b, mode='drop')
    return forces
```

- [ ] **Step 4: Run test to verify it passes**

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/test_step.py::test_edge_bond_force_on_two_particle_composite -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add halflife/step.py tests/test_step.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "feat(step): add compute_edge_bond_forces (per-edge harmonic springs)

New pure helper for the edges-mode bond force. Existing compute_bond_forces
(COM star spring) is preserved for star_spring mode. Not yet wired into
simulation_step."
```

---

## Task F: Dispatch `bond_mode` in `simulation_step`

**Files:**
- Modify: `halflife/step.py:174-184` (the existing `use_bond_forces` block in `simulation_step`)
- Test: `tests/test_step.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_step.py`:

```python
def test_simulation_step_runs_with_bond_mode_edges():
    """Smoke: simulation_step accepts bond_mode='edges' and produces a valid state."""
    from halflife.step import simulation_step
    config = SimConfig(num_species=3, num_particles=50, max_composites=10,
                       bond_mode="edges")
    world = initialize_world(config, seed=0)
    params = initialize_interaction_params(config, seed=0)
    physics = initialize_physics_params(config)
    new_state = simulation_step(world, params, config, physics)
    assert new_state.particles.position.shape == (50, 2)
    # All particles still alive, no NaNs in position/velocity
    assert not jnp.isnan(new_state.particles.position).any()
    assert not jnp.isnan(new_state.particles.velocity).any()


def test_simulation_step_runs_with_bond_mode_off():
    """Smoke: bond_mode='off' produces a valid state with no bond forces."""
    from halflife.step import simulation_step
    config = SimConfig(num_species=3, num_particles=50, max_composites=10,
                       bond_mode="off")
    world = initialize_world(config, seed=0)
    params = initialize_interaction_params(config, seed=0)
    physics = initialize_physics_params(config)
    new_state = simulation_step(world, params, config, physics)
    assert not jnp.isnan(new_state.particles.position).any()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/test_step.py::test_simulation_step_runs_with_bond_mode_edges tests/test_step.py::test_simulation_step_runs_with_bond_mode_off -v
```

Expected: FAIL (likely because the dispatch isn't there, or because adding 'edges' triggers an unhandled path).

- [ ] **Step 3: Replace the bond-force block in `simulation_step`**

In `halflife/step.py`, find the existing block at lines ~178-181:

```python
    # Bond forces (optional — expensive; enable via config.use_bond_forces)
    if config.use_bond_forces:
        bond_forces = compute_bond_forces(state, config, physics)
        forces = forces + bond_forces
```

Replace it with:

```python
    # Bond forces — dispatched on static config.bond_mode so XLA traces only
    # the live branch. Existing use_bond_forces flag is honored for backward
    # compat in star_spring mode (bond_mode='star_spring' + use_bond_forces=False
    # is equivalent to bond_mode='off').
    if config.bond_mode == "edges":
        bond_forces = compute_edge_bond_forces(state, params, config, physics)
        forces = forces + bond_forces
    elif config.bond_mode == "star_spring" and config.use_bond_forces:
        bond_forces = compute_bond_forces(state, config, physics)
        forces = forces + bond_forces
    # bond_mode == "off" → no bond force added
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/test_step.py::test_simulation_step_runs_with_bond_mode_edges tests/test_step.py::test_simulation_step_runs_with_bond_mode_off -v
```

Expected: both PASS.

- [ ] **Step 5: Run the full test suite to ensure star_spring mode still works**

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/ -x --tb=short
```

Expected: all pass. Star_spring is the default, so existing tests are unaffected.

- [ ] **Step 6: Commit**

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add halflife/step.py tests/test_step.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "feat(step): dispatch bond force kernel by config.bond_mode

Replaces the use_bond_forces if-block with a three-way static dispatch on
bond_mode ∈ {'edges', 'star_spring', 'off'}. Star_spring path is unchanged."
```

---

## Task G: Wire `degree` into the simulation step (Phase 5b)

**Files:**
- Modify: `halflife/step.py` (compute degree once per step before Phase 6)

- [ ] **Step 1: Plan note**

This task is plumbing — there's no new logic, just placing the `degree` compute where downstream tasks (H, I, J) can consume it. We don't need a new test; existing tests pass through `simulation_step` and will fail if this plumbing is wrong.

- [ ] **Step 2: Modify `simulation_step` to compute `degree` after Phase 5**

In `halflife/step.py`, find the section after `apply_boundary` and before `attempt_fusion` (around lines 195-201). Insert the `degree` compute:

```python
    # ── Phase 5b: Per-particle degree cache ─────────────────────────────────
    # degree[i] counts edges incident to particle i across all alive composites.
    # Used by the per-particle valence gate in Phase 6 and by Phase 6b ring
    # closure. Cheap (O(C · E_max) scatter-add). Recomputed once per step from
    # the pre-fusion edge state; phases 6 and 6b update it incrementally via
    # their scan carries.
    from halflife.chemistry import compute_degree, compute_composite_free_bonds, _species_valences
    degree = compute_degree(state.composites, config)
    species_valences = _species_valences(config)
    composite_free_bonds = compute_composite_free_bonds(
        state.particles, state.composites, degree, species_valences, config
    )
```

(The imports are intentionally local — moving them to the top of the file would create a circular import because `chemistry.py` imports from `state.py` which is needed by `step.py`. The runtime cost of a local import is one-time at JIT trace.)

Then pass `degree`, `species_valences`, and `composite_free_bonds` to `attempt_fusion` — modify the call site:

```python
    # ── Phase 6: Fusion ───────────────────────────────────────────────────────
    state, degree = attempt_fusion(
        state, neighbors, params, config, physics,
        degree=degree, species_valences=species_valences,
    )
```

Note: `attempt_fusion`'s signature changes — it now takes and returns `degree`. The actual modification of `attempt_fusion` happens in Tasks H and I; this commit only sets up the call site. To keep Task G self-contained and the suite green, also modify `attempt_fusion` minimally to accept the new kwargs and pass them through unchanged:

In `halflife/chemistry.py`, change the signature of `attempt_fusion` (around line 550):

```python
def attempt_fusion(state: WorldState, neighbors: jnp.ndarray,
                   params: InteractionParams, config: SimConfig,
                   physics: PhysicsParams,
                   degree: jnp.ndarray = None,
                   species_valences: jnp.ndarray = None,
                   metrics=None) -> tuple:
    """
    [existing docstring preserved]
    """
    # If callers didn't pass degree (legacy path), compute it locally so the
    # function works standalone too. New step.py path always passes it.
    if degree is None:
        degree = compute_degree(state.composites, config)
    if species_valences is None:
        species_valences = _species_valences(config)
    # ... existing body unchanged ...
```

And change the return statement at the bottom of `attempt_fusion` (around line 911) from `return state._replace(...)` to `return state._replace(...), degree` — i.e., return both the new state and the (currently unchanged) degree.

The existing `all_entity_free_bonds` computation inside `attempt_fusion` can stay as-is for Task G; Task H will reorient it onto per-particle degree.

- [ ] **Step 3: Run the full suite**

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/ -x --tb=short
```

Expected: all pass. The new plumbing is a no-op (degree is computed but not yet consumed).

- [ ] **Step 4: Commit**

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add halflife/step.py halflife/chemistry.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "feat(step): compute per-particle degree cache in Phase 5b

Plumbs degree, species_valences, and composite_free_bonds through to
attempt_fusion. Not yet consumed; sets up Tasks H, I, J."
```

---

## Task H: Switch fusion gate to per-particle valence

**Files:**
- Modify: `halflife/chemistry.py` — `attempt_fusion` (gate uses per-particle `degree`)
- Test: `tests/test_chemistry.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chemistry.py`:

```python
def test_per_particle_fusion_gate_blocks_saturated_rep():
    """
    A composite rep that's already saturated (degree == v_s) cannot fuse with
    a free particle even if the composite as a whole has free bonds.
    """
    from halflife.chemistry import _species_valences
    config = SimConfig(num_species=3, num_particles=10, max_composites=4,
                       max_valence=2, boundary_mode="reflect",
                       world_width=20.0, world_height=20.0,
                       fusion_radius=2.0, fusion_threshold=0.0)
    world = initialize_world(config, seed=0)
    params = initialize_interaction_params(config, seed=0)
    physics = initialize_physics_params(config)

    # Hand-build a 3-member composite where the rep (particle 0) is bonded
    # to BOTH siblings (degree[0] = 2 = v_s for max_valence=2). Composite has
    # remaining slack on particles 1 and 2 but rep is saturated.
    sv = np.asarray(_species_valences(config))
    # Force species_valences to be all 2 by setting max_valence=2 ... actually
    # max_valence=2 means v_s ∈ [1, 2] — we need v_s[s_0] == 2 to make rep saturated.
    # Find a species with valence 2 and assign it to the rep.
    target_species = int(np.where(sv == 2)[0][0])

    pos = np.array([[5.0, 5.0],   # rep (will be doubly-bonded)
                    [4.0, 5.0],   # sibling A
                    [6.0, 5.0],   # sibling B
                    [5.5, 5.0],   # free particle within fusion_radius of rep
                    [50.0, 50.0]] + [[50.0+i, 50.0] for i in range(5)],
                   dtype=np.float32)[:10]
    species = np.full(10, target_species, dtype=np.int32)
    composite_id = np.array([0, 0, 0, -1, -1, -1, -1, -1, -1, -1], dtype=np.int32)
    members = np.full((4, config.max_composite_size), -1, dtype=np.int32)
    members[0, :3] = (0, 1, 2)
    edges = np.full((4, config.e_max, 2), -1, dtype=np.int32)
    edges[0, 0] = (0, 1); edges[0, 1] = (0, 2)  # rep 0 bonded to both
    edge_count = np.array([2, 0, 0, 0], dtype=np.int32)
    alive = np.array([True, False, False, False], dtype=bool)

    world = world._replace(
        particles=world.particles._replace(
            position=jnp.asarray(pos), species=jnp.asarray(species),
            composite_id=jnp.asarray(composite_id),
        ),
        composites=world.composites._replace(
            members=jnp.asarray(members), member_count=jnp.array([3,0,0,0]),
            alive=jnp.asarray(alive), edges=jnp.asarray(edges),
            edge_count=jnp.asarray(edge_count),
        ),
    )

    # Run one fusion attempt
    from halflife.spatial import build_cell_list, find_all_neighbors
    cell_list = build_cell_list(world.particles.position, config)
    neighbors = find_all_neighbors(world.particles.position, cell_list, config)
    new_state, _ = attempt_fusion(world, neighbors, params, config, physics)

    # Particle 3 should NOT have been absorbed into composite 0 (rep is saturated)
    assert np.asarray(new_state.particles.composite_id)[3] == -1, \
        "Saturated rep should not fuse with free particle"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/test_chemistry.py::test_per_particle_fusion_gate_blocks_saturated_rep -v
```

Expected: FAIL (current composite-level gate would allow fusion if composite has slack).

- [ ] **Step 3: Modify the fusion gate in `attempt_fusion`**

In `halflife/chemistry.py`, find the existing `all_entity_free_bonds` computation in `attempt_fusion` (around lines 597-610) and replace it with a per-particle version:

```python
    # ── Pre-cache per-particle free bonds ─────────────────────────────────────
    # Per-particle: free_bond[i] = v_{species[i]} − degree[i].
    # For free particles degree[i] = 0 so free_bond[i] = v_{species[i]}.
    # For composite members this is stricter than the previous composite-level
    # check: requires the specific rep doing the fusion to have unused valence.
    all_particle_free_bonds = species_valences[particles.species] - degree  # (N,) int32
```

Then in `check_neighbor` (around line 657), replace the existing valence gate with:

```python
            # Valence gate: each particle (NOT each composite) must have at
            # least one free bond. The new edge consumes one bond on each
            # endpoint, so both reps must individually have free_bond ≥ 1.
            if config.use_valence:
                has_free_bonds = (
                    (all_particle_free_bonds[i] >= 1)
                    & (all_particle_free_bonds[j] >= 1)
                )
            else:
                has_free_bonds = jnp.bool_(True)
```

**Do NOT delete `all_entity_free_bonds` yet** — it is still referenced by the `merged_free_bonds` calculation in `fusion_scan_body` (around line 783). Removing it now would break the build between Tasks H and I. Task I rewrites `merged_free_bonds` and removes the now-orphaned `all_entity_free_bonds` in the same commit, keeping the suite green.

- [ ] **Step 4: Run test to verify it passes**

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/test_chemistry.py::test_per_particle_fusion_gate_blocks_saturated_rep -v
```

Expected: PASS.

- [ ] **Step 5: Run the full chemistry suite to catch regressions**

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/test_chemistry.py -x --tb=short
```

Expected: all pass. The per-particle gate is stricter, but for free particles it's equivalent (their degree is 0).

- [ ] **Step 6: Commit**

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add halflife/chemistry.py tests/test_chemistry.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "feat(chemistry): switch fusion gate to per-particle valence

The fusion gate now checks that each specific rep has unused valence
(degree[i] < v_{species[i]}), not just that the composite as a whole has
free bonds. Necessary for clean ring-closure semantics in a later task."
```

---

## Task I: Append edges in `fusion_scan_body`

**Files:**
- Modify: `halflife/chemistry.py` — `fusion_scan_body` (build merged edge list + new edge)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chemistry.py`:

```python
def test_fusion_appends_edge_free_plus_free():
    """When two free particles fuse, the new composite has exactly one edge: (i, j)."""
    config = SimConfig(num_species=3, num_particles=10, max_composites=4,
                       boundary_mode="reflect", world_width=20.0, world_height=20.0,
                       fusion_radius=2.0, fusion_threshold=0.0)
    world = initialize_world(config, seed=0)
    params = initialize_interaction_params(config, seed=0)
    physics = initialize_physics_params(config)

    # Place particles 0 and 1 within fusion_radius of each other
    pos = np.array([[5.0, 5.0], [6.0, 5.0]] + [[50.0+i, 50.0] for i in range(8)],
                   dtype=np.float32)
    species = np.zeros(10, dtype=np.int32)
    world = world._replace(particles=world.particles._replace(
        position=jnp.asarray(pos), species=jnp.asarray(species)
    ))

    from halflife.spatial import build_cell_list, find_all_neighbors
    cell_list = build_cell_list(world.particles.position, config)
    neighbors = find_all_neighbors(world.particles.position, cell_list, config)
    new_state, _ = attempt_fusion(world, neighbors, params, config, physics)

    # One composite was created; it should hold exactly one edge (0, 1).
    alive = np.asarray(new_state.composites.alive)
    c = int(np.where(alive)[0][0])
    assert np.asarray(new_state.composites.edge_count)[c] == 1
    e = np.asarray(new_state.composites.edges[c, 0])
    assert sorted(e.tolist()) == [0, 1]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/test_chemistry.py::test_fusion_appends_edge_free_plus_free -v
```

Expected: FAIL (edge_count stays 0 because edges aren't written).

- [ ] **Step 3: Modify `fusion_scan_body` to merge edge lists**

In `halflife/chemistry.py`, locate `fusion_scan_body` (around line 729) and modify the per-iteration logic to also build a merged edge list and write it to the target composite.

After the existing `merged_members` compaction (around lines 800-810), insert the analogous edge build:

```python
        # ── Build merged edge list ─────────────────────────────────────────
        # i-side edges
        i_edges_comp = composites_state.edges[ci]                          # (E_max, 2)
        i_edges_free = jnp.full((E_max, 2), -1, dtype=jnp.int32)
        i_edges = jnp.where(i_is_free, i_edges_free, i_edges_comp)

        # j-side edges
        j_edges_comp = composites_state.edges[cj]
        j_edges_free = jnp.full((E_max, 2), -1, dtype=jnp.int32)
        j_edges = jnp.where(j_is_free, j_edges_free, j_edges_comp)

        # The new fusion edge: rep_i ↔ rep_j (= safe_i, safe_j since both are reps).
        # For free+free, the reps are i and j themselves. For free+comp, rep is the
        # free particle on one side and members[c, 0] on the other; we passed in
        # both already via the merged-members construction.
        new_edge = jnp.array([safe_i, safe_j], dtype=jnp.int32)[None, :]   # (1, 2)

        # Concatenate (2·E_max + 1, 2) buffer
        merged_edges_raw = jnp.concatenate([i_edges, j_edges, new_edge], axis=0)
        edge_valid = merged_edges_raw[:, 0] >= 0  # both -1 or both valid
        # Compact valid entries to front, drop overflow
        epos    = jnp.cumsum(edge_valid.astype(jnp.int32)) - 1
        eout    = jnp.where(edge_valid, epos, E_max)  # invalid → OOB
        merged_edges = jnp.full((E_max, 2), -1, dtype=jnp.int32).at[eout].set(
            merged_edges_raw, mode='drop'
        )
        merged_edge_count = jnp.sum(edge_valid.astype(jnp.int32))
```

Where `E_max = config.e_max` — add this near the top of `attempt_fusion` alongside `M = config.max_composite_size`:

```python
    E_max = config.e_max
```

Then in the target-composite write block (around lines 816-841), append the edges and edge_count updates:

```python
        new_comp_edges = composites_state.edges.at[safe_target].set(
            jnp.where(can_fuse, merged_edges, composites_state.edges[safe_target])
        )
        new_comp_edge_count = composites_state.edge_count.at[safe_target].set(
            jnp.where(can_fuse, merged_edge_count, composites_state.edge_count[safe_target])
        )
        # ... existing `composites_state._replace(...)` updated below
```

Update the `_replace(...)` call to include the new fields:

```python
        new_composites = composites_state._replace(
            members=new_members,
            alive=new_comp_alive,
            binding_energy=new_comp_be,
            half_life=new_comp_hl,
            member_count=new_comp_count_arr,
            species_hash=new_comp_hash,
            free_bonds=new_comp_free_bonds,
            edges=new_comp_edges,
            edge_count=new_comp_edge_count,
        )
```

Also update `merged_free_bonds`. The existing line was:

```python
        merged_free_bonds = (
            all_entity_free_bonds[safe_i] + all_entity_free_bonds[safe_j] - jnp.int32(2)
        )
```

Replace with per-particle accounting recomputed from the merged edges:

```python
        # New composite's free bonds: Σ v_s − 2 · edge_count.
        # Member species sum (same logic as compute_composite_free_bonds but
        # against the just-computed merged_members).
        merged_member_species = particles.species[
            jnp.where(merged_members >= 0, merged_members, 0)
        ]
        merged_member_valid = merged_members >= 0
        sum_v = jnp.sum(jnp.where(
            merged_member_valid, species_valences[merged_member_species], 0
        ))
        merged_free_bonds = sum_v - jnp.int32(2) * merged_edge_count
```

Now that nothing else references `all_entity_free_bonds`, delete its computation block (around lines 597-610 of `chemistry.py` — the entire `if config.use_valence: all_entity_free_bonds = jax.vmap(...)... else: all_entity_free_bonds = jnp.zeros(...)` section). `species_valences` is still needed and is now passed in via the function signature from Task G.

Finally, also update `degree` in the carry as new edges are added. Modify the scan carry to include `degree`:

```python
    # ── Initial carry now includes degree ──
    claimed_init       = jnp.zeros(N, dtype=bool)
    composite_id_init  = particles.composite_id
    comp_count_init    = jnp.sum(composites.alive.astype(jnp.int32))
    free_slot_ptr_init = jnp.int32(0)
    degree_init        = degree  # passed in from step.py via Task G

    (_, final_composite_id, final_composites, _, _, final_degree), _ = jax.lax.scan(
        fusion_scan_body,
        (claimed_init, composite_id_init, composites, comp_count_init, free_slot_ptr_init, degree_init),
        scan_indices,
    )
```

Update the scan carry signature in `fusion_scan_body`:

```python
    def fusion_scan_body(carry, i):
        claimed, new_composite_id, composites_state, comp_count, free_slot_ptr, degree_carry = carry
        # ... existing body ...

        # Increment degree for the two new edge endpoints when fusion fires
        delta = can_fuse.astype(jnp.int32)
        degree_carry = degree_carry.at[safe_i].add(delta)
        degree_carry = degree_carry.at[safe_j].add(delta)

        return (new_claimed, new_composite_id, new_composites, new_comp_count,
                new_free_slot_ptr, degree_carry), None
```

Finally, the returned tuple from `attempt_fusion`:

```python
    return state._replace(
        particles=new_particles,
        composites=final_composites,
        rng_key=key,
    ), final_degree
```

- [ ] **Step 4: Run test to verify it passes**

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/test_chemistry.py::test_fusion_appends_edge_free_plus_free -v
```

Expected: PASS.

- [ ] **Step 5: Run the full chemistry suite to ensure other cases still work**

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/test_chemistry.py -x --tb=short
```

Expected: all pass. The new logic in star_spring mode is harmless (the edges array sits there unused).

- [ ] **Step 6: Commit**

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add halflife/chemistry.py tests/test_chemistry.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "feat(chemistry): append edges in fusion_scan_body

Every fusion event now appends one new edge (rep_i, rep_j) to the target
composite and preserves edges from any absorbed composite. degree is
threaded through the scan carry so per-particle valence stays in sync."
```

---

## Task J: Implement ring-closure scan (Phase 6b)

**Files:**
- Modify: `halflife/chemistry.py` — add `attempt_ring_closure`
- Modify: `halflife/step.py` — call it after Phase 6
- Test: `tests/test_chemistry.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chemistry.py`:

```python
def test_ring_closure_adds_edge_between_same_composite_members():
    """
    Two members of the same composite within fusion_radius, both with per-
    particle free bonds, close a ring (gain one new edge).
    """
    from halflife.chemistry import attempt_ring_closure, _species_valences
    config = SimConfig(num_species=3, num_particles=10, max_composites=4,
                       max_valence=4, allow_ring_closure=True,
                       boundary_mode="reflect", world_width=20.0, world_height=20.0,
                       fusion_radius=2.0, fusion_threshold=0.0)
    world = initialize_world(config, seed=0)
    params = initialize_interaction_params(config, seed=0)
    physics = initialize_physics_params(config)

    # Build a chain composite 0—1—2—3 where 0 and 3 are within fusion_radius
    # (chain has folded back on itself).
    pos = np.array([[5.0, 5.0],   # 0
                    [6.5, 5.0],   # 1
                    [7.5, 6.0],   # 2
                    [6.0, 6.0]]   # 3 — within fusion_radius of 0
                   + [[50.0+i, 50.0] for i in range(6)], dtype=np.float32)
    species = np.zeros(10, dtype=np.int32)
    composite_id = np.array([0, 0, 0, 0, -1, -1, -1, -1, -1, -1], dtype=np.int32)
    members = np.full((4, config.max_composite_size), -1, dtype=np.int32)
    members[0, :4] = (0, 1, 2, 3)
    edges = np.full((4, config.e_max, 2), -1, dtype=np.int32)
    edges[0, 0] = (0, 1); edges[0, 1] = (1, 2); edges[0, 2] = (2, 3)
    edge_count = np.array([3, 0, 0, 0], dtype=np.int32)
    alive = np.array([True, False, False, False], dtype=bool)

    world = world._replace(
        particles=world.particles._replace(
            position=jnp.asarray(pos), species=jnp.asarray(species),
            composite_id=jnp.asarray(composite_id),
        ),
        composites=world.composites._replace(
            members=jnp.asarray(members), member_count=jnp.array([4,0,0,0]),
            alive=jnp.asarray(alive), edges=jnp.asarray(edges),
            edge_count=jnp.asarray(edge_count),
        ),
    )

    # Compute pre-state degree
    from halflife.chemistry import compute_degree
    degree = compute_degree(world.composites, config)
    sv = _species_valences(config)

    from halflife.spatial import build_cell_list, find_all_neighbors
    cell_list = build_cell_list(world.particles.position, config)
    neighbors = find_all_neighbors(world.particles.position, cell_list, config)

    new_state, _ = attempt_ring_closure(
        world, neighbors, params, config, physics,
        degree=degree, species_valences=sv,
    )

    # Composite 0 should now have 4 edges (the ring-closing one).
    assert np.asarray(new_state.composites.edge_count)[0] == 4
    # The new edge should connect 0 and 3.
    edges_after = np.asarray(new_state.composites.edges[0])
    found = any(sorted(edges_after[e].tolist()) == [0, 3] for e in range(4))
    assert found, f"Expected (0, 3) edge after ring closure, got {edges_after[:4]}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/test_chemistry.py::test_ring_closure_adds_edge_between_same_composite_members -v
```

Expected: FAIL with `ImportError: cannot import name 'attempt_ring_closure'`.

- [ ] **Step 3: Implement `attempt_ring_closure` in `halflife/chemistry.py`**

Add this function (placement: at end of file, after `attempt_fusion`):

```python
def attempt_ring_closure(state: WorldState, neighbors: jnp.ndarray,
                          params: InteractionParams, config: SimConfig,
                          physics: PhysicsParams,
                          degree: jnp.ndarray,
                          species_valences: jnp.ndarray) -> tuple:
    """
    Phase 6b: same-composite ring closure.

    For each pair of same-composite members within fusion_radius where both
    have per-particle free bonds (degree < v_s), add one new edge between them.
    Touches ONLY edges, edge_count, and degree — no member-list / composite_id
    changes.

    Gated by config.allow_ring_closure (static); when False, returns the state
    unchanged.

    Returns:
        (new_state, new_degree)
    """
    if not config.allow_ring_closure:
        return state, degree

    particles = state.particles
    composites = state.composites
    key, subkey = jax.random.split(state.rng_key)
    N = config.num_particles
    C = config.max_composites
    E_max = config.e_max
    fusion_r2 = config.fusion_radius ** 2

    # ── Skip mask: only particles in composites with free_bonds ≥ 2 can host ─
    # a new ring edge. Composite-level free bonds = Σ free_bond[m] over members.
    composite_free_bonds = compute_composite_free_bonds(
        particles, composites, degree, species_valences, config
    )
    # Per-particle: free_bond[i] = v_{species[i]} - degree[i]
    particle_free_bonds = species_valences[particles.species] - degree  # (N,) int32
    # Per-particle skip: must be in a composite with ≥2 free bonds AND have ≥1.
    safe_cid = jnp.clip(particles.composite_id, 0, C - 1)
    can_attempt = (particles.composite_id >= 0) & \
                  (composite_free_bonds[safe_cid] >= 2) & \
                  (particle_free_bonds >= 1)  # (N,)

    # ── Find best ring partner per particle ─────────────────────────────────
    def find_ring_partner(i):
        nbrs = neighbors[i]
        i_attempt = can_attempt[i]
        cid_i = particles.composite_id[i]
        pos_i = particles.position[i]

        def check(j):
            valid = (
                (j >= 0) & (j != i) &
                (j > i)  # consider each pair once
                & can_attempt[j]
                & (particles.composite_id[j] == cid_i)  # same composite
            )
            d = pos_i - particles.position[j]
            if config.boundary_mode == "periodic":
                d = d - config.world_width  * jnp.round(d[0] / config.world_width) * jnp.array([1., 0.])
                d = d - config.world_height * jnp.round(d[1] / config.world_height) * jnp.array([0., 1.])
            dist2 = jnp.dot(d, d)
            in_range = dist2 < fusion_r2
            ok = valid & in_range
            return jnp.where(ok, j, jnp.int32(-1)), jnp.where(ok, dist2, jnp.float32(jnp.inf))

        partners, dists = jax.vmap(check)(nbrs)
        best_idx = jnp.argmin(dists)
        return jnp.where(i_attempt, partners[best_idx], jnp.int32(-1))

    all_partners = jax.vmap(find_ring_partner)(jnp.arange(N, dtype=jnp.int32))  # (N,)

    # ── Conflict resolution: take up to max_ring_closures candidates ────────
    has_partner = all_partners >= 0
    perm = jax.random.permutation(subkey, N)
    shuffled = has_partner[perm]
    cum = jnp.cumsum(shuffled.astype(jnp.int32))
    cand = jnp.where(shuffled & (cum <= config.max_ring_closures_per_step), perm, N)
    scan_indices = jnp.sort(cand)[:config.max_ring_closures_per_step]

    def ring_body(carry, i):
        composites_state, degree_carry, claimed = carry
        valid_i = i < N
        safe_i = jnp.minimum(i, N - 1)
        j = jnp.where(valid_i, all_partners[safe_i], jnp.int32(-1))
        safe_j = jnp.where(j >= 0, j, 0)

        # Recheck per-particle valence using the live degree.
        free_i_now = species_valences[particles.species[safe_i]] - degree_carry[safe_i]
        free_j_now = species_valences[particles.species[safe_j]] - degree_carry[safe_j]

        # Dedup: is (safe_i, safe_j) already in edges[cid]?
        cid = jnp.clip(particles.composite_id[safe_i], 0, C - 1)
        c_edges = composites_state.edges[cid]  # (E_max, 2)
        already = jnp.any(
            ((c_edges[:, 0] == safe_i) & (c_edges[:, 1] == safe_j)) |
            ((c_edges[:, 0] == safe_j) & (c_edges[:, 1] == safe_i))
        )

        can_close = (
            (j >= 0)
            & ~claimed[safe_i] & ~claimed[safe_j]
            & (free_i_now >= 1) & (free_j_now >= 1)
            & ~already
        )

        # Append (safe_i, safe_j) to edges[cid] at slot edge_count[cid].
        slot = composites_state.edge_count[cid]
        safe_slot = jnp.where(can_close, slot, jnp.int32(E_max))  # OOB → drop
        new_edge = jnp.where(can_close, jnp.array([safe_i, safe_j], dtype=jnp.int32),
                              jnp.array([-1, -1], dtype=jnp.int32))
        c_edges_new = c_edges.at[safe_slot].set(new_edge, mode='drop')
        composites_state = composites_state._replace(
            edges=composites_state.edges.at[cid].set(c_edges_new),
            edge_count=composites_state.edge_count.at[cid].set(
                jnp.where(can_close, slot + 1, composites_state.edge_count[cid])
            ),
            free_bonds=composites_state.free_bonds.at[cid].set(
                jnp.where(can_close, composites_state.free_bonds[cid] - 2,
                           composites_state.free_bonds[cid])
            ),
        )

        # Update degree
        delta = can_close.astype(jnp.int32)
        degree_carry = degree_carry.at[safe_i].add(delta)
        degree_carry = degree_carry.at[safe_j].add(delta)

        # Mark claimed
        claimed = claimed.at[safe_i].set(claimed[safe_i] | can_close)
        claimed = claimed.at[safe_j].set(claimed[safe_j] | can_close)

        return (composites_state, degree_carry, claimed), None

    (final_composites, final_degree, _), _ = jax.lax.scan(
        ring_body,
        (composites, degree, jnp.zeros(N, dtype=bool)),
        scan_indices,
    )

    return state._replace(composites=final_composites, rng_key=key), final_degree
```

- [ ] **Step 4: Wire `attempt_ring_closure` into `simulation_step`**

In `halflife/step.py`, immediately after the `attempt_fusion` call, add:

```python
    # ── Phase 6b: Ring closure (intra-composite fusion) ───────────────────────
    from halflife.chemistry import attempt_ring_closure
    state, degree = attempt_ring_closure(
        state, neighbors, params, config, physics,
        degree=degree, species_valences=species_valences,
    )
```

- [ ] **Step 5: Run test to verify it passes**

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/test_chemistry.py::test_ring_closure_adds_edge_between_same_composite_members -v
```

Expected: PASS.

- [ ] **Step 6: Run the full suite — make sure existing dynamics still work**

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/ -x --tb=short
```

Expected: all pass. Ring closure is gated on per-particle free bonds, and free particles have degree 0, so it's a strict addition.

- [ ] **Step 7: Commit**

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add halflife/chemistry.py halflife/step.py tests/test_chemistry.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "feat(chemistry): add Phase 6b ring closure

Two members of the same composite within fusion_radius, both with per-
particle free bonds, can now close a ring. Touches only edges, edge_count,
free_bonds, and degree. Gated by config.allow_ring_closure."
```

---

## Task K: Rebuild edges in fission (`apply_composite_decay`)

**Files:**
- Modify: `halflife/chemistry.py` — `apply_composite_decay` (drop parent edges, build path-spanning tree per product)
- Test: `tests/test_chemistry.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chemistry.py`:

```python
def test_fission_rebuilds_spanning_tree_per_product():
    """
    A 4-member composite that fissions into two 2-member products should
    leave each product with exactly 1 edge (the spanning-tree edge).
    """
    config = SimConfig(num_species=3, num_particles=10, max_composites=4,
                       boundary_mode="reflect", world_width=20.0, world_height=20.0,
                       half_life_min=0.001, half_life_max=0.001,  # decay every step
                       fusion_cost=0.0)
    world = initialize_world(config, seed=0)
    params = initialize_interaction_params(config, seed=0)
    physics = initialize_physics_params(config)

    pos = np.array([[5.0, 5.0], [5.5, 5.0], [6.0, 5.0], [6.5, 5.0]]
                   + [[50.0+i, 50.0] for i in range(6)], dtype=np.float32)
    species = np.zeros(10, dtype=np.int32)
    composite_id = np.array([0, 0, 0, 0, -1, -1, -1, -1, -1, -1], dtype=np.int32)
    members = np.full((4, config.max_composite_size), -1, dtype=np.int32)
    members[0, :4] = (0, 1, 2, 3)
    edges = np.full((4, config.e_max, 2), -1, dtype=np.int32)
    edges[0, 0] = (0, 1); edges[0, 1] = (1, 2); edges[0, 2] = (2, 3)
    edge_count = np.array([3, 0, 0, 0], dtype=np.int32)
    alive = np.array([True, False, False, False], dtype=bool)
    half_life = np.array([0.001, 0.0, 0.0, 0.0], dtype=np.float32)

    world = world._replace(
        particles=world.particles._replace(
            position=jnp.asarray(pos), species=jnp.asarray(species),
            composite_id=jnp.asarray(composite_id),
        ),
        composites=world.composites._replace(
            members=jnp.asarray(members), member_count=jnp.array([4,0,0,0]),
            alive=jnp.asarray(alive), edges=jnp.asarray(edges),
            edge_count=jnp.asarray(edge_count),
            half_life=jnp.asarray(half_life),
        ),
    )

    new_state = apply_composite_decay(world, config, physics)

    # The original composite has fissioned. Both products should each have a
    # spanning tree (n-1 edges). For a 2-member product, that's exactly 1.
    alive_after = np.asarray(new_state.composites.alive)
    counts_after = np.asarray(new_state.composites.member_count)
    edge_counts_after = np.asarray(new_state.composites.edge_count)
    for c in np.where(alive_after)[0]:
        n = counts_after[c]
        assert edge_counts_after[c] == max(0, n - 1), \
            f"Composite {c} has {n} members but {edge_counts_after[c]} edges (expected {n-1})"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/test_chemistry.py::test_fission_rebuilds_spanning_tree_per_product -v
```

Expected: FAIL (current fission doesn't touch edges, so they stay at the parent's count).

- [ ] **Step 3: Modify `apply_composite_decay` to rebuild edges per product**

In `halflife/chemistry.py`, find `apply_composite_decay` (around line 231). After the existing `per_product` vmap that produces `p0_members`, `p0_count`, `p0_hash`, etc., add a helper that builds a path-spanning tree from a compacted member list:

```python
    def _path_edges_from_members(members_arr, count_arr):
        """
        Build a path-spanning tree through members in slot order: edges are
        (members[0], members[1]), (members[1], members[2]), ...
        Returns (edges: (E_max, 2), edge_count: int32).
        """
        E = config.e_max
        # Edge k = (members[k], members[k+1]) for k in [0, count-1)
        k_idx = jnp.arange(E, dtype=jnp.int32)
        valid_edge = k_idx < jnp.maximum(count_arr - jnp.int32(1), jnp.int32(0))
        # Safe gather
        safe_k = jnp.minimum(k_idx, jnp.int32(M - 2))
        a = members_arr[safe_k]
        b = members_arr[safe_k + 1]
        a_out = jnp.where(valid_edge, a, jnp.int32(-1))
        b_out = jnp.where(valid_edge, b, jnp.int32(-1))
        new_edges = jnp.stack([a_out, b_out], axis=-1)  # (E, 2)
        new_edge_count = jnp.maximum(count_arr - jnp.int32(1), jnp.int32(0))
        return new_edges, new_edge_count
```

Then after the existing `p0_hl_all`/`p1_hl_all` computations, build edge arrays for both products:

```python
    p0_edges, p0_edge_count_all = jax.vmap(_path_edges_from_members)(p0_members, p0_count)
    p1_edges, p1_edge_count_all = jax.vmap(_path_edges_from_members)(p1_members, p1_count)
```

Now extend the parent-slot writes (around lines 488-497) to include the new edge arrays. Find:

```python
    new_alive = jnp.where(fissions, p0_alive, composites.alive)
    new_members = jnp.where(fissions[:, None], p0_members, composites.members)
    new_member_count = jnp.where(fissions, p0_count, composites.member_count)
    new_species_hash = jnp.where(fissions, p0_hash, composites.species_hash)
    new_binding_energy = jnp.where(fissions, p0_be_all, composites.binding_energy)
    new_half_life = jnp.where(fissions, p0_hl_all, composites.half_life)
    new_free_bonds = jnp.where(fissions, p0_free_bonds, composites.free_bonds)
    new_age = jnp.where(fissions, jnp.float32(0.0), composites.age)
```

Add immediately after:

```python
    new_edges = jnp.where(fissions[:, None, None], p0_edges, composites.edges)
    new_edge_count = jnp.where(fissions, p0_edge_count_all, composites.edge_count)
```

And in the product-1 scatter block (around lines 516-523), add the analogous writes:

```python
    new_edges          = new_edges.at[drop_targets].set(p1_edges, mode='drop')
    new_edge_count     = new_edge_count.at[drop_targets].set(p1_edge_count_all, mode='drop')
```

Finally, update the `composites._replace(...)` call at the bottom of `apply_composite_decay` (around line 525) to include the new fields:

```python
    new_composites = composites._replace(
        members=new_members,
        member_count=new_member_count,
        alive=new_alive,
        binding_energy=new_binding_energy,
        half_life=new_half_life,
        age=new_age,
        species_hash=new_species_hash,
        free_bonds=new_free_bonds,
        edges=new_edges,
        edge_count=new_edge_count,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/test_chemistry.py::test_fission_rebuilds_spanning_tree_per_product -v
```

Expected: PASS.

- [ ] **Step 5: Run the full suite**

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/ -x --tb=short
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add halflife/chemistry.py tests/test_chemistry.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "feat(chemistry): rebuild spanning-tree edges on fission

Parent edges are dropped on fission; each product builds a fresh path-
spanning tree through its hash-sorted members. n-1 edges per product
maintains the spanning-tree valence accounting."
```

---

## Task L: Toggle-time edge initialization

**Files:**
- Modify: `halflife/chemistry.py` — add `initialize_edges_for_existing_composites`
- Test: `tests/test_chemistry.py`

- [ ] **Step 1: Write the failing test**

```python
def test_initialize_edges_builds_spanning_tree_for_alive_composites():
    """
    Called when toggling INTO edges mode: every alive composite gets a path-
    spanning tree through its members. Composites with edges already set
    retain them.
    """
    from halflife.chemistry import initialize_edges_for_existing_composites
    config = SimConfig(num_species=3, num_particles=10, max_composites=4)
    world = initialize_world(config, seed=0)
    # Build a 3-member composite with NO edges (as if star_spring mode)
    members = np.full((4, config.max_composite_size), -1, dtype=np.int32)
    members[0, :3] = (0, 1, 2)
    composite_id = np.array([0, 0, 0, -1, -1, -1, -1, -1, -1, -1], dtype=np.int32)
    alive = np.array([True, False, False, False], dtype=bool)
    world = world._replace(
        particles=world.particles._replace(composite_id=jnp.asarray(composite_id)),
        composites=world.composites._replace(
            members=jnp.asarray(members),
            member_count=jnp.array([3, 0, 0, 0], dtype=jnp.int32),
            alive=jnp.asarray(alive),
            edge_count=jnp.array([0, 0, 0, 0], dtype=jnp.int32),
        ),
    )
    new_composites = initialize_edges_for_existing_composites(world.composites, config)
    # Composite 0 should now have 2 edges spanning 3 members
    assert int(np.asarray(new_composites.edge_count[0])) == 2
    e0 = np.asarray(new_composites.edges[0, 0])
    e1 = np.asarray(new_composites.edges[0, 1])
    # Path through hash-sorted members; verify the edge set is a valid spanning tree
    pids = {0, 1, 2}
    used = set(e0.tolist()) | set(e1.tolist())
    assert used <= pids and len(used) == 3, f"Edges should span all members, got {e0}, {e1}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/test_chemistry.py::test_initialize_edges_builds_spanning_tree_for_alive_composites -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `initialize_edges_for_existing_composites` in `halflife/chemistry.py`**

```python
def initialize_edges_for_existing_composites(composites: CompositeState,
                                              config: SimConfig) -> CompositeState:
    """
    For every alive composite, replace its edge list with a fresh path-
    spanning tree through its members in slot order. Used when toggling into
    'edges' bond mode from a state that didn't have edges populated.

    Free particles (composite_id == -1) are unaffected.

    Returns: updated CompositeState (composites only).
    """
    C = config.max_composites
    M = config.max_composite_size
    E = config.e_max
    e_idx = jnp.arange(E, dtype=jnp.int32)

    def per_composite(c):
        is_alive = composites.alive[c]
        n = composites.member_count[c]
        members = composites.members[c]
        valid_edge = (e_idx < jnp.maximum(n - jnp.int32(1), jnp.int32(0)))
        safe_k = jnp.minimum(e_idx, jnp.int32(M - 2))
        a = members[safe_k]
        b = members[safe_k + 1]
        a_out = jnp.where(valid_edge & is_alive, a, jnp.int32(-1))
        b_out = jnp.where(valid_edge & is_alive, b, jnp.int32(-1))
        new_edges = jnp.stack([a_out, b_out], axis=-1)  # (E, 2)
        new_count = jnp.where(
            is_alive, jnp.maximum(n - jnp.int32(1), jnp.int32(0)), jnp.int32(0)
        )
        return new_edges, new_count

    new_edges, new_edge_count = jax.vmap(per_composite)(
        jnp.arange(C, dtype=jnp.int32)
    )
    return composites._replace(edges=new_edges, edge_count=new_edge_count)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/test_chemistry.py::test_initialize_edges_builds_spanning_tree_for_alive_composites -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add halflife/chemistry.py tests/test_chemistry.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "feat(chemistry): add initialize_edges_for_existing_composites

Helper for star_spring → edges mode transitions. Builds a path-spanning
tree per alive composite so existing composites don't dissolve when bond
mode flips."
```

---

## Task M: Runtime `M` key toggle in `main.py`

**Files:**
- Modify: `halflife/main.py` — add `M` key handling

- [ ] **Step 1: Plan note**

No new automated test — this is interactive UI plumbing. Manual smoke test is in Step 4.

- [ ] **Step 2: Add `M` key handler in `halflife/main.py`**

In the `KEYDOWN` handler (around line 186 of `main.py`), add a new branch after the `pygame.K_b` block:

```python
                elif event.key == pygame.K_m:
                    # Cycle bond_mode: edges → star_spring → off → edges
                    from halflife.chemistry import initialize_edges_for_existing_composites
                    cycle = {"edges": "star_spring", "star_spring": "off", "off": "edges"}
                    new_mode = cycle[config.bond_mode]
                    print(f"Bond mode: {config.bond_mode} → {new_mode}")
                    # If toggling INTO 'edges', seed a spanning tree per alive composite
                    # so existing composites don't dissolve when the edge force kicks in.
                    if new_mode == "edges":
                        pending_state = pending_state._replace(
                            composites=initialize_edges_for_existing_composites(
                                pending_state.composites, config
                            )
                        )
                    # Rebuild config (frozen dataclass) with new bond_mode. This
                    # triggers a JAX retrace on the next run_n call, but JAX caches
                    # per static-arg hash so subsequent toggles to the same mode
                    # reuse the cached compile.
                    import dataclasses
                    config = dataclasses.replace(config, bond_mode=new_mode)
                    run_n = make_run_n_steps(config)
```

Note: `config` is a `frozen` dataclass, so we use `dataclasses.replace` to make a new one. The outer-scope `config` is also rebound — this works because we're inside the function body of `run()` where `config` is a local variable, not the module-level one.

Also update the `print("Running. Controls: ...")` line near the top (around line 127) to mention M:

```python
    print("Running. Controls: Space=pause, +/-=speed, B=composite mode, M=bond mode, R=reset, Q=quit")
```

And the docstring at the top of `main.py` (around line 22) gets the new key:

```python
  B           — toggle composite visualization (bonds ↔ merged)
  M           — toggle bond mode (edges ↔ star_spring ↔ off)
```

- [ ] **Step 3: Sanity check that the module still imports**

```bash
JAX_PLATFORMS=cpu python -c "from halflife.main import run; print('ok')"
```

Expected: prints `ok` with no exception.

- [ ] **Step 4: Manual smoke test**

Run the simulator briefly and verify the M key cycles modes:

```bash
.venv/bin/python -m halflife.main
```

Press `M` twice and observe the console output:
- First press: `Bond mode: star_spring → off`
- Second press: `Bond mode: off → edges` (brief pause for first-time recompile expected)
- Third press: `Bond mode: edges → star_spring` (no recompile — already cached)

Close the window when done. Don't commit recordings/screenshots.

- [ ] **Step 5: Commit**

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add halflife/main.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "feat(main): add M key to cycle bond_mode at runtime

Cycles edges ↔ star_spring ↔ off. When entering edges mode, runs a one-shot
spanning-tree initialization over all alive composites so they don't dissolve.
Triggers a JAX retrace on first switch into each mode; subsequent toggles
are instant due to the JIT cache."
```

---

## Task N: Renderer — draw real edges instead of forward-slot heuristic

**Files:**
- Modify: `halflife/renderer.py` (~lines 888-947)

- [ ] **Step 1: Plan note**

No automated test — visual change. Manual verification in Step 3.

- [ ] **Step 2: Modify bond emission in `halflife/renderer.py`**

In `halflife/renderer.py`, locate the bond-drawing block in the `update` method (lines 888-947). The current code iterates per-composite-member and emits forward-slot lines. Replace the inner emission with iteration over `composites.edges`:

Find the existing block that builds `bond_verts`. Replace its inner logic with edge-driven emission. The key change:

```python
                # ── Edge-driven bond drawing (replaces forward-slot heuristic) ──
                # Read the sparse edge list directly from composites.edges.
                # Falls back to the legacy heuristic if bond_mode != 'edges'
                # AND edge_count is zero everywhere (i.e., star_spring mode
                # with no edges populated).
                edges_arr = np.asarray(state.composites.edges)        # (C, E, 2)
                edge_count = np.asarray(state.composites.edge_count)  # (C,)
                alive = np.asarray(state.composites.alive)            # (C,)
                positions = np.asarray(state.particles.position)      # (N, 2)
                species = np.asarray(state.particles.species)         # (N,)

                if self.config.bond_mode == "edges" or edge_count.sum() > 0:
                    # Collect all valid edges across alive composites
                    edge_pairs = []
                    for c in np.where(alive)[0]:
                        ec = int(edge_count[c])
                        if ec == 0:
                            continue
                        edge_pairs.append(edges_arr[c, :ec])
                    if edge_pairs:
                        all_edges = np.concatenate(edge_pairs, axis=0)  # (E_total, 2)
                        mem_a = all_edges[:, 0]
                        mem_b = all_edges[:, 1]
                        pos_a = positions[mem_a]
                        pos_b = positions[mem_b]
                        # Min-image wrap for periodic boundary so bonds across
                        # the wrap aren't drawn as long diagonal stripes.
                        if self.config.boundary_mode == "periodic":
                            d = pos_b - pos_a
                            d -= self.config.world_width  * np.round(d[:, 0:1] / self.config.world_width)  * np.array([1., 0.])
                            d -= self.config.world_height * np.round(d[:, 1:2] / self.config.world_height) * np.array([0., 1.])
                            pos_b = pos_a + d
                        n_pairs = len(all_edges)
                        alpha = 1.0
                        bond_verts = np.empty((n_pairs * 2, 6), dtype=np.float32)
                        bond_verts[0::2, :2] = pos_a
                        bond_verts[1::2, :2] = pos_b
                        bond_verts[0::2, 2:5] = self.species_colors[species[mem_a]]
                        bond_verts[1::2, 2:5] = self.species_colors[species[mem_b]]
                        bond_verts[0::2, 5] = alpha
                        bond_verts[1::2, 5] = alpha
                        n_bytes = min(bond_verts.nbytes, self._bond_buf_size)
                        self.bond_vbo.write(bond_verts.tobytes()[:n_bytes])
                        self._n_bond_vertices = n_bytes // (self._bond_vertex_size * 4)
                    else:
                        self._n_bond_vertices = 0
                else:
                    # Legacy forward-slot heuristic for star_spring mode with no edges.
                    # [original block here — preserve it verbatim]
                    pass  # ← copy the existing forward-slot code here
```

**Important:** Do NOT delete the existing forward-slot heuristic code. Keep it in the `else` branch so star_spring mode with no edges still gets visual bonds. Move the existing block into the `else:` arm.

- [ ] **Step 3: Manual smoke test**

```bash
.venv/bin/python -m halflife.main
```

- Default mode is star_spring — bonds should appear as before.
- Press `M` to cycle to `off` — bonds disappear.
- Press `M` again to enter `edges` mode — bonds appear, but now drawn from `composites.edges`. With the just-initialized spanning trees, the visual should look similar (path through members).
- Let the sim run a few seconds in edges mode. Form new composites and watch the bonds attach to specific particle pairs rather than every-neighbor-pair.

- [ ] **Step 4: Commit**

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add halflife/renderer.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "feat(renderer): draw bonds from sparse edges in edges mode

Bond lines are now sourced from composites.edges when bond_mode == 'edges'
(or when any composite has edges populated). The legacy forward-slot
heuristic is preserved as a fallback for star_spring mode."
```

---

## Task O: End-to-end integration test

**Files:**
- Create: `tests/test_covalent_bonds_integration.py`

- [ ] **Step 1: Write the integration test**

Create `tests/test_covalent_bonds_integration.py`:

```python
"""
End-to-end smoke test for sparse covalent bonds.

Runs ~200 simulation steps with bond_mode='edges' and verifies:
  - No NaNs in positions or velocities
  - At least some composites form
  - All alive composites have edge_count >= max(0, member_count - 1)
    (spanning-tree invariant)
  - All edges reference particles that are in the same composite
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
import numpy as np

from halflife.config import SimConfig
from halflife.state import (
    initialize_world, initialize_interaction_params, initialize_physics_params,
)
from halflife.step import make_run_n_steps


def test_edges_mode_runs_for_200_steps_without_crashing():
    """Sim runs in edges mode and produces a valid final state."""
    config = SimConfig(
        num_species=3, num_particles=200, max_composites=50,
        bond_mode="edges", allow_ring_closure=True,
        fusion_radius=2.0, fusion_threshold=0.3,
    )
    world = initialize_world(config, seed=42)
    params = initialize_interaction_params(config, seed=43)
    physics = initialize_physics_params(config)

    run_n = make_run_n_steps(config)
    final = run_n(world, params, physics, 200)
    jax.block_until_ready(final)

    pos = np.asarray(final.particles.position)
    vel = np.asarray(final.particles.velocity)
    assert not np.isnan(pos).any(), "NaN in position"
    assert not np.isnan(vel).any(), "NaN in velocity"
    # Position stays in [0, world_size] under periodic boundary
    assert (pos[:, 0] >= 0).all() and (pos[:, 0] <= config.world_width).all()
    assert (pos[:, 1] >= 0).all() and (pos[:, 1] <= config.world_height).all()


def test_edges_mode_spanning_tree_invariant():
    """After many steps, alive composites have edge_count >= n - 1."""
    config = SimConfig(
        num_species=3, num_particles=200, max_composites=50,
        bond_mode="edges", allow_ring_closure=True,
        fusion_radius=2.0, fusion_threshold=0.3,
    )
    world = initialize_world(config, seed=42)
    params = initialize_interaction_params(config, seed=43)
    physics = initialize_physics_params(config)
    run_n = make_run_n_steps(config)
    final = run_n(world, params, physics, 200)
    jax.block_until_ready(final)

    alive = np.asarray(final.composites.alive)
    counts = np.asarray(final.composites.member_count)
    e_counts = np.asarray(final.composites.edge_count)
    for c in np.where(alive)[0]:
        n = counts[c]
        if n < 2:
            continue  # size-1 composites shouldn't exist post-fusion; skip
        assert e_counts[c] >= n - 1, \
            f"Composite {c}: {n} members but {e_counts[c]} edges (< spanning tree)"


def test_edges_mode_edges_reference_same_composite_members():
    """Every edge's endpoints have composite_id == c (no stale edges)."""
    config = SimConfig(
        num_species=3, num_particles=200, max_composites=50,
        bond_mode="edges", allow_ring_closure=True,
        fusion_radius=2.0, fusion_threshold=0.3,
    )
    world = initialize_world(config, seed=42)
    params = initialize_interaction_params(config, seed=43)
    physics = initialize_physics_params(config)
    run_n = make_run_n_steps(config)
    final = run_n(world, params, physics, 200)
    jax.block_until_ready(final)

    alive = np.asarray(final.composites.alive)
    edges = np.asarray(final.composites.edges)
    e_counts = np.asarray(final.composites.edge_count)
    cids = np.asarray(final.particles.composite_id)
    for c in np.where(alive)[0]:
        for e in range(int(e_counts[c])):
            a, b = edges[c, e]
            assert cids[a] == c, f"Edge ({a},{b}) in c={c} but cids[{a}]={cids[a]}"
            assert cids[b] == c, f"Edge ({a},{b}) in c={c} but cids[{b}]={cids[b]}"
```

- [ ] **Step 2: Run the integration test**

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/test_covalent_bonds_integration.py -v --tb=short
```

Expected: all three tests PASS. The first run will take ~30 s due to JIT compilation.

- [ ] **Step 3: Run the entire test suite one more time to confirm clean state**

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/ -x --tb=short
```

Expected: all tests pass (existing + new).

- [ ] **Step 4: Commit**

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add tests/test_covalent_bonds_integration.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "test(integration): end-to-end sparse-bonds smoke test

Runs 200 steps in edges mode with ring closure enabled and verifies:
  - no NaNs
  - spanning-tree invariant holds (edge_count >= n-1 for each alive composite)
  - all edges reference particles whose composite_id matches the composite slot"
```

---

## Done

After Task O, sparse covalent bonds with ring closure are fully wired and selectable at runtime via the `M` key. The default mode remains `star_spring` until you confirm dynamics are stable, then flip the default in `halflife/config.py` in a separate commit.

**Next experiments (out of scope for this plan, in spec's "Out of scope" section):**
- Bond multiplicity (single/double/triple)
- Angle constraints for rigid double bonds
- Closest-pair fusion bonding instead of rep-to-rep
- Morse-style bonds that can break under stress
- Bond potential energy in `compute_total_energy`
