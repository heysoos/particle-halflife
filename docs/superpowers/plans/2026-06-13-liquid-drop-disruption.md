# Liquid-Drop Disruption Term — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or
> subagent-driven-development) to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the liquid-drop half-life a long-range "Coulomb" disruption term
(R_g monopole) so fissility grows with composite size — un-pinning half-lives
from `half_life_max` and making big/compact composites fission before they reach
the `max_composite_size` buffer.

**Architecture:** A new O(n) per-composite radius-of-gyration helper (periodic
min-image) feeds `E_coulomb = disruption_scale·n²/R_g` into the existing
`compute_liquid_drop_half_life` fissility `x`. Two scalars (`disruption_scale`,
`cohesion_hl_scale`) become runtime-tunable `PhysicsParams` + sliders. Energy-only
(no new forces); `disruption_scale=0` reproduces today's behavior.

**Tech Stack:** JAX (vmap-free vectorized gathers, min-image periodic wrap),
existing liquid-drop / hash-chemistry plumbing.

**Spec:** `docs/superpowers/specs/2026-06-13-liquid-drop-disruption-design.md`

**Conventions (from CLAUDE.md):** WSL Pattern B — run tests with
`JAX_PLATFORMS=cpu .venv/bin/python -m pytest ...` (force CPU; the user's live sim
may hold the GPU). Never `git add -A`. **Preserve existing comments** — additive
only. Commit only the named files per task.

---

### Task 1: Config fields + runtime PhysicsParams scalars

**Files:**
- Modify: `halflife/config.py` (add `disruption_scale`; change `cohesion_hl_scale` default)
- Modify: `halflife/state.py` (`PhysicsParams` + `initialize_physics_params`)
- Test: `tests/test_liquid_drop_disruption.py` (new)

- [ ] **Step 1: Write failing test**

```python
# tests/test_liquid_drop_disruption.py
"""Tests for the liquid-drop Coulomb-analog disruption term (R_g monopole).

See docs/superpowers/plans/2026-06-13-liquid-drop-disruption.md.
Run: JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_liquid_drop_disruption.py -q
"""
import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np

from halflife.config import SimConfig
from halflife.state import initialize_world, initialize_physics_params


def test_disruption_config_defaults():
    c = SimConfig()
    assert c.disruption_scale == 0.5
    assert c.cohesion_hl_scale == 5.0          # raised from 1.0 so t_coh de-saturates


def test_physics_params_seeds_disruption_scalars():
    c = dataclasses.replace(SimConfig(), disruption_scale=0.7, cohesion_hl_scale=4.0)
    p = initialize_physics_params(c)
    assert float(p.disruption_scale) == 0.7
    assert float(p.cohesion_hl_scale) == 4.0
```

- [ ] **Step 2: Run, verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_liquid_drop_disruption.py -q`
Expected: FAIL (`SimConfig` has no `disruption_scale`).

- [ ] **Step 3: Add the config fields**

In `halflife/config.py`, the stability block currently reads (around lines 97-101):

```python
    stability_mode: str = "liquid_drop"
    surface_energy_coeff: float = 0.5   # a_s — cohesion penalty × n^(2/3)
    cohesion_hl_scale: float = 1.0      # per-member cohesion needed for max stability
    fissility_exponent: float = 1.0     # sharpness of the collapse as x → 1
    composite_size_decay_scale: float = 0.05   # size penalty on composite half-life (legacy mode + creation-time placeholder values)
```

Change the `cohesion_hl_scale` default and add `disruption_scale` right after it:

```python
    stability_mode: str = "liquid_drop"
    surface_energy_coeff: float = 0.5   # a_s — cohesion penalty × n^(2/3)
    cohesion_hl_scale: float = 5.0      # per-member cohesion needed for max stability (≈⟨E_b⟩ so t_coh is a real gradient, not saturated at 1)
    # Long-range Coulomb-analog disruption: E_coulomb = disruption_scale·n²/R_g.
    # Grows super-linearly with size (R_g ∝ √n for a compact blob), so fissility
    # x climbs with n and big/compact composites fission at a tunable critical
    # size. 0 → legacy hard-core-only fissility (backward-compatible). Runtime-
    # tunable via PhysicsParams ("disrupt k" slider). Final default tuned with
    # the diagnostic (see spec §6).
    disruption_scale: float = 0.5
    fissility_exponent: float = 1.0     # sharpness of the collapse as x → 1
    composite_size_decay_scale: float = 0.05   # size penalty on composite half-life (legacy mode + creation-time placeholder values)
```

- [ ] **Step 4: Add the runtime scalars**

In `halflife/state.py`, add two fields to `PhysicsParams` after `k_angle`:

```python
    k_angle:                  jnp.ndarray  # () float32 — angle-locking stiffness (edges mode)
    disruption_scale:         jnp.ndarray  # () float32 — liquid-drop Coulomb-analog constant (0 = legacy hard-core-only fissility)
    cohesion_hl_scale:        jnp.ndarray  # () float32 — per-member cohesion for max stability (runtime; was config.cohesion_hl_scale)
```

And seed them in `initialize_physics_params` after the `k_angle=` line:

```python
        k_angle=jnp.float32(config.k_angle),
        disruption_scale=jnp.float32(config.disruption_scale),
        cohesion_hl_scale=jnp.float32(config.cohesion_hl_scale),
```

- [ ] **Step 5: Run, verify pass**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_liquid_drop_disruption.py -q`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add halflife/config.py halflife/state.py tests/test_liquid_drop_disruption.py
git commit -m "feat(stability): add disruption_scale + de-saturate cohesion_hl_scale (config + runtime)"
```

---

### Task 2: Radius-of-gyration helper (periodic min-image)

**Files:**
- Modify: `halflife/chemistry.py` (add `compute_radius_of_gyration` just before
  `compute_liquid_drop_half_life`, ~line 2103)
- Test: `tests/test_liquid_drop_disruption.py`

**Shared test helper** — add at the top of the test file (hand-builds a
`WorldState` with one composite at explicit member positions; reused by Tasks 2-3):

```python
def _composite_world(member_pos, edges, num_particles=20,
                     world=(200.0, 112.5), boundary="periodic", species_val=2):
    """WorldState with composite 0 = the given members at the given positions.

    member_pos: {pid: (x, y)} for the composite's members.
    edges:      list of (i, j) bond pairs among those members.
    Other particles are parked far away as free particles.
    """
    c = dataclasses.replace(
        SimConfig(), num_particles=num_particles, max_composites=4,
        max_composite_size=16, max_valence=4, num_species=3,
        bond_mode="edges", boundary_mode=boundary,
        world_width=world[0], world_height=world[1],
    )
    state = initialize_world(c, seed=0)
    members = sorted(member_pos)
    pos = np.array(state.particles.position)          # writable copy
    for pid, xy in member_pos.items():
        pos[pid] = xy
    species = np.full(num_particles, species_val, np.int32)   # high-E_b self-pair
    cid = np.full(num_particles, -1, np.int32)
    for pid in members:
        cid[pid] = 0
    mem = np.full((c.max_composites, c.max_composite_size), -1, np.int32)
    mem[0, :len(members)] = members
    E = c.e_max
    edge_arr = np.full((c.max_composites, E, 2), -1, np.int32)
    for k, (i, j) in enumerate(edges):
        edge_arr[0, k] = (i, j)
    comp = state.composites._replace(
        alive=state.composites.alive.at[0].set(True),
        members=jnp.asarray(mem),
        member_count=state.composites.member_count.at[0].set(len(members)),
        edges=jnp.asarray(edge_arr),
        edge_count=state.composites.edge_count.at[0].set(len(edges)),
    )
    parts = state.particles._replace(
        position=jnp.asarray(pos, jnp.float32),
        species=jnp.asarray(species),
        composite_id=jnp.asarray(cid),
    )
    return state._replace(composites=comp, particles=parts), c


def _grid_members(k, spacing=1.0, x0=20.0, y0=20.0):
    """k×k grid of members + grid edges (right/down neighbors). Returns
    (member_pos dict keyed by pid 0..k²-1, edges list)."""
    pos, edges = {}, []
    idx = lambda r, col: r * k + col
    for r in range(k):
        for col in range(k):
            pos[idx(r, col)] = (x0 + col * spacing, y0 + r * spacing)
            if col + 1 < k:
                edges.append((idx(r, col), idx(r, col + 1)))
            if r + 1 < k:
                edges.append((idx(r, col), idx(r + 1, col)))
    return pos, edges
```

- [ ] **Step 1: Write failing test**

```python
def test_rg_open_boundary_chain_vs_cluster():
    from halflife.chemistry import compute_radius_of_gyration
    # collinear chain at x=0,2,4 (y=5): centroid x=2, R_g = sqrt((4+0+4)/3)=sqrt(8/3)
    state, c = _composite_world(
        {0: (0.0, 5.0), 1: (2.0, 5.0), 2: (4.0, 5.0)},
        [(0, 1), (1, 2)], boundary="reflect")
    rg = np.asarray(compute_radius_of_gyration(state.particles, state.composites, c))
    assert np.isclose(rg[0], math.sqrt(8.0 / 3.0), rtol=1e-4)
    # tight cluster → much smaller R_g than the spread chain
    state2, c2 = _composite_world(
        {0: (0.0, 0.0), 1: (0.1, 0.0), 2: (0.0, 0.1)},
        [(0, 1), (1, 2)], boundary="reflect")
    rg2 = np.asarray(compute_radius_of_gyration(state2.particles, state2.composites, c2))
    assert rg2[0] < rg[0]


def test_rg_periodic_wrap_unwrapped():
    from halflife.chemistry import compute_radius_of_gyration
    # members straddle the x-wrap (world_width=200): true unwrapped x = 199,201,203.
    state, c = _composite_world(
        {0: (199.0, 5.0), 1: (1.0, 5.0), 2: (3.0, 5.0)},
        [(0, 1), (1, 2)], boundary="periodic")
    rg = np.asarray(compute_radius_of_gyration(state.particles, state.composites, c))
    # min-image unwrap → same R_g as the open chain (sqrt(8/3)), NOT the huge
    # value a naive (centroid≈67) computation would give.
    assert np.isclose(rg[0], math.sqrt(8.0 / 3.0), rtol=1e-4)


def test_rg_dead_composite_is_zero():
    from halflife.chemistry import compute_radius_of_gyration
    state, c = _composite_world(
        {0: (0.0, 5.0), 1: (2.0, 5.0)}, [(0, 1)], boundary="reflect")
    rg = np.asarray(compute_radius_of_gyration(state.particles, state.composites, c))
    assert rg[1] == 0.0 and rg[2] == 0.0 and rg[3] == 0.0   # composites 1-3 dead
```

- [ ] **Step 2: Run, verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_liquid_drop_disruption.py -k rg_ -q`
Expected: FAIL (`cannot import name 'compute_radius_of_gyration'`).

- [ ] **Step 3: Implement**

In `halflife/chemistry.py`, immediately before `compute_liquid_drop_half_life`
(the `# ── Liquid-Drop Stability ...` banner near line 2102), add:

```python
def compute_radius_of_gyration(particles, composites, config: SimConfig) -> jnp.ndarray:
    """
    (C,) RMS radius of each composite's members about their centroid.

    Computed with periodic min-image unwrapping relative to the first member, so
    a composite straddling the world-wrap edge gets a correct (small) R_g rather
    than an inflated naive-centroid value. Exact while a composite spans < half
    the world in each axis — always true here (composites ≪ world). Dead / empty
    composites return 0.

    O(C · MAX_COMPOSITE_SIZE) — same order as the e_coh edge gather; no O(n²).
    """
    C = config.max_composites
    S = config.max_composite_size
    s_idx = jnp.arange(S, dtype=jnp.int32)
    members = composites.members                                   # (C, S) int32, -1 pad
    valid = (composites.alive[:, None]
             & (s_idx[None, :] < composites.member_count[:, None])
             & (members >= 0))                                     # (C, S)

    safe_m = jnp.where(members >= 0, members, 0)
    pos = particles.position[safe_m]                               # (C, S, 2)
    ref = pos[:, 0:1, :]                                           # (C, 1, 2) first member

    d = pos - ref                                                  # (C, S, 2)
    if config.boundary_mode == "periodic":
        d = d - config.world_width  * jnp.round(d[..., 0:1] / config.world_width)  * jnp.array([1., 0.])
        d = d - config.world_height * jnp.round(d[..., 1:2] / config.world_height) * jnp.array([0., 1.])

    w = valid.astype(jnp.float32)                                  # (C, S)
    cnt = jnp.maximum(w.sum(axis=1), 1.0)                          # (C,)
    centroid = (d * w[..., None]).sum(axis=1) / cnt[:, None]       # (C, 2) offset from ref
    dev = d - centroid[:, None, :]                                 # (C, S, 2)
    rg2 = ((dev * dev).sum(axis=-1) * w).sum(axis=1) / cnt         # (C,)
    return jnp.sqrt(rg2)
```

- [ ] **Step 4: Run, verify pass**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_liquid_drop_disruption.py -k rg_ -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add halflife/chemistry.py tests/test_liquid_drop_disruption.py
git commit -m "feat(stability): per-composite radius of gyration (periodic min-image)"
```

---

### Task 3: Wire disruption into `compute_liquid_drop_half_life`

**Files:**
- Modify: `halflife/chemistry.py` (`compute_liquid_drop_half_life` body, ~lines 2122-2141)
- Test: `tests/test_liquid_drop_disruption.py`

- [ ] **Step 1: Write failing test**

```python
def _hl0(state, c, physics=None):
    from halflife.chemistry import compute_liquid_drop_half_life
    physics = physics or initialize_physics_params(c)
    rep_pe = jnp.zeros(c.num_particles, dtype=jnp.float32)   # spaced members → no hard core
    hl = compute_liquid_drop_half_life(state.particles, state.composites, rep_pe, c, physics)
    return float(hl[0])


def test_disruption_shortens_larger_blobs():
    # Same density, bigger blob → higher fissility → shorter half-life.
    small, c1 = _composite_world(*_grid_members(2), num_particles=20)
    big,   c2 = _composite_world(*_grid_members(4), num_particles=20)
    assert _hl0(big, c2) < _hl0(small, c1)


def test_disruption_shape_sensitivity_chain_vs_blob():
    # Equal member count (9), compact blob vs extended chain → blob shorter-lived.
    blob_pos, blob_edges = _grid_members(3)                       # 3×3 = 9
    chain_pos = {i: (20.0 + i, 20.0) for i in range(9)}           # 1×9 line
    chain_edges = [(i, i + 1) for i in range(8)]
    blob,  cb = _composite_world(blob_pos, blob_edges, num_particles=16)
    chain, cc = _composite_world(chain_pos, chain_edges, num_particles=16)
    assert _hl0(chain, cc) > _hl0(blob, cb)


def test_disruption_off_reproduces_legacy_pinning():
    # disruption_scale=0 and a well-bonded, spaced composite → pins at half_life_max
    # (the pre-feature behavior: x≈0, t_coh saturated for this composition).
    state, c = _composite_world(*_grid_members(2), num_particles=20)
    physics = initialize_physics_params(
        dataclasses.replace(c, disruption_scale=0.0, cohesion_hl_scale=1.0))
    assert np.isclose(_hl0(state, c, physics), c.half_life_max, rtol=1e-5)


def test_cohesion_scale_gradient():
    # Equal geometry, different mean bond energy (species 0 vs 2 self-pairs) →
    # different half-life once cohesion is de-saturated (cohesion_hl_scale=5).
    a, ca = _composite_world(*_grid_members(2), num_particles=20, species_val=0)
    b, cb = _composite_world(*_grid_members(2), num_particles=20, species_val=2)
    assert _hl0(a, ca) != _hl0(b, cb)
```

- [ ] **Step 2: Run, verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_liquid_drop_disruption.py -k disruption_or_cohesion -q`
(or run the whole file; the four new tests fail because the Coulomb term isn't wired yet —
`test_disruption_shortens_larger_blobs` / `shape_sensitivity` fail since half-lives are still
size-independent.)

- [ ] **Step 3: Implement**

In `halflife/chemistry.py`, the current tail of `compute_liquid_drop_half_life`
reads:

```python
    n = composites.member_count.astype(jnp.float32)
    e_coh = bond_sum - config.surface_energy_coeff * n ** (2.0 / 3.0)

    x = e_rep / (2.0 * jnp.maximum(e_coh, 1e-6))
    t_coh = jnp.clip(e_coh / (config.cohesion_hl_scale * jnp.maximum(n, 1.0)), 0.0, 1.0)
    stab = t_coh * jnp.clip(1.0 - x, 0.0, 1.0) ** config.fissility_exponent
    hl = config.half_life_min + (config.half_life_max - config.half_life_min) * stab
    return jnp.where(composites.alive, hl, composites.half_life)
```

Replace from the `x = ...` line through the `stab = ...` line (keep `n` / `e_coh`
and the `hl` / `return` lines, and **all existing comments above**) with:

```python
    # Disruption — long-range Coulomb-analog monopole self-energy. Uniform unit
    # "charge" per member: E_coulomb = disruption_scale·n²/R_g grows super-
    # linearly with size for a compact blob (R_g ∝ √n → E_coulomb/E_coh ∝ √n),
    # so big/compact composites cross the fissility threshold and fission. Shape-
    # aware: extended composites have large R_g → low disruption. Summed with the
    # short-range hard-core e_rep (the over-crammed term). disruption_scale = 0
    # collapses x back to the legacy hard-core-only fissility.
    rg = compute_radius_of_gyration(particles, composites, config)
    e_coulomb = physics.disruption_scale * n * n / (rg + 1e-6)
    e_dis = e_coulomb + e_rep

    x = e_dis / (2.0 * jnp.maximum(e_coh, 1e-6))
    t_coh = jnp.clip(e_coh / (physics.cohesion_hl_scale * jnp.maximum(n, 1.0)), 0.0, 1.0)
    stab = t_coh * jnp.clip(1.0 - x, 0.0, 1.0) ** config.fissility_exponent
```

(Note the two reads now come from `physics.disruption_scale` /
`physics.cohesion_hl_scale` — runtime-tunable, no recompile. `config.cohesion_hl_scale`
is no longer read here; it only seeds the physics value.)

- [ ] **Step 4: Run, verify pass**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_liquid_drop_disruption.py -q`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Regression — existing liquid-drop + chemistry suites**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_liquid_drop.py tests/test_chemistry.py -q`
Expected: PASS. The existing `test_liquid_drop.py` comparisons (crammed < relaxed;
zero-cohesion → hl_min; legacy untouched) are preserved: the Coulomb term and the
`cohesion_hl_scale` change both cancel in those same-composition comparisons, and
`test_chemistry.py`'s half-life bound is an upper bound (the fix only lowers
half-lives). If any assertion checks an *exact* half-life value that shifts,
update it to reflect the new formula (do not weaken a behavioral assertion).

- [ ] **Step 6: Commit**

```bash
git add halflife/chemistry.py tests/test_liquid_drop_disruption.py
git commit -m "feat(stability): Coulomb-analog disruption (R_g monopole) into liquid-drop half-life"
```

---

### Task 4: Renderer sliders + fit-floor adjustment

**Files:**
- Modify: `halflife/renderer.py` (`_rebuild_physics_sliders` — add two sliders;
  lower the fit floor so the edges panel still fits)

- [ ] **Step 1: Add the two stability sliders**

In `halflife/renderer.py`, in the general `slider_specs` list (the fusion-chemistry
group), after the `binding_energy_scale` row, add:

```python
            ("binding_energy_scale",     "bind energy", _phys("binding_energy_scale"), "{:.3f}", None),
            ("disruption_scale",         "disrupt k",   _phys("disruption_scale"),     "{:.2f}", None),
            ("cohesion_hl_scale",        "cohesion",    _phys("cohesion_hl_scale"),    "{:.2f}", None),
```

(No new `None` group break — two extra rows, not three, to stay within the panel
height. They are log-scale multiplier sliders, so the live range is 0.1×–10× the
default: `disrupt k` 0.05–5.0, `cohesion` 0.5–50.)

- [ ] **Step 2: Lower the fit floor so 14 sliders fit the edges panel**

In `_rebuild_physics_sliders`, the fit-to-window block sets `_MIN_ROW_H, _MIN_GAP`.
Two more rows in edges mode (14 sliders + 2 gaps ≈ 560px into ~420px available)
need a slightly tighter floor. Change:

```python
        _MIN_ROW_H, _MIN_GAP = 30, 8
```

to:

```python
        _MIN_ROW_H, _MIN_GAP = 28, 8   # 28px keeps the label clear of the handle
                                       # below it while fitting the edges panel
                                       # (14 sliders) into a 720px window.
```

- [ ] **Step 3: Compile check**

Run: `cd "/mnt/c/Users/Heysoos/Documents/Pycharm Projects/halflife-particle" && .venv/bin/python -m py_compile halflife/renderer.py`
Expected: no output (clean compile).

- [ ] **Step 4: Manual smoke (user-run, GPU)**

```bash
.venv/bin/python -m halflife.main
```
Confirm: app launches; in edges mode the Params panel shows "disrupt k" and
"cohesion" sliders and **all** sliders (down to "angle k") remain on-screen;
dragging "disrupt k" up visibly shrinks the largest composites (they fission
sooner); no console NaN warnings.

- [ ] **Step 5: Commit**

```bash
git add halflife/renderer.py
git commit -m "feat(stability): disrupt k + cohesion sliders; tighten panel fit floor"
```

---

### Task 5: Docs

**Files:**
- Modify: `CLAUDE.md` (Liquid-Drop Stability section + Configuration block)
- Modify: `README.md` (Decay and Fission / stability description)

- [ ] **Step 1: CLAUDE.md** — in the "Liquid-Drop Stability" section, document the
  new disruption term: `E_coulomb = disruption_scale·n²/R_g` (R_g = periodic
  min-image radius of gyration), summed with the hard-core `e_rep` into the
  fissility `x`; the `√n` size scaling and resulting tunable critical size; that
  `disruption_scale`/`cohesion_hl_scale` are now runtime `PhysicsParams` knobs
  (sliders), with `disruption_scale=0` = legacy hard-core-only. Note
  `cohesion_hl_scale` default moved `1.0 → 5.0` (de-saturates `t_coh`). In the
  Configuration code block, add `disruption_scale=0.5` and update
  `cohesion_hl_scale` to `5.0` with a short comment. **Preserve all existing
  prose/comments — additive only.**

- [ ] **Step 2: README.md** — in the composite stability / "Decay and Fission"
  description, add the disruption term and the `n²/R_g` Coulomb-analog with the
  one-line intuition (compact blobs fission past a critical size; extended shapes
  stay stable).

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs(stability): document Coulomb-analog disruption term"
```

---

### Task 6: Empirical tuning & validation (manual, user-run)

Not a code task — the `disruption_scale=0.5` default is a placeholder pending a
diagnostic pass (spec §6).

- [ ] **Step 1: Diagnostic run**

```bash
.venv/bin/python -m halflife.analysis --scenario current_experiment --steps 15000 --sample-every 250 --platform gpu
```

- [ ] **Step 2: Inspect** the size histogram (Tier 1/5) and half-life spread.
  Target: max composite size sits **well below 256**; half-lives span a range
  rather than pinning at `half_life_max`.

- [ ] **Step 3: Tune** `disruption_scale` (raise to shrink critical size, lower to
  grow it), via the live slider first, then set the chosen value as the
  `config.py` default. If changed, update this plan's default and the
  `current_experiment` preset note in `CLAUDE.md` / `cli.py` if relevant.

---

## Self-review notes

- **Spec coverage:** config/physics (T1), R_g helper incl. periodic wrap (T2),
  disruption wired into fissility + de-saturated cohesion (T3), sliders (T4),
  docs (T5), tuning procedure (T6). All spec §3-6 components covered.
- **Type consistency:** `compute_radius_of_gyration(particles, composites, config)`
  → `(C,)`, consumed in `compute_liquid_drop_half_life` (T2/T3). New scalars read
  from `physics.disruption_scale` / `physics.cohesion_hl_scale` (seeded T1, read
  T3). Slider field names match the `PhysicsParams` fields (T1/T4).
- **Backward-compat verified by reasoning:** `disruption_scale=0` zeroes
  `e_coulomb`, so `x` reverts to `e_rep/(2 e_coh)`; existing `test_liquid_drop.py`
  comparisons hold because both the Coulomb term and the cohesion-scale change
  cancel in same-composition comparisons. T3 Step 5 confirms empirically.
- **Known soft spots:** (a) the `_composite_world` helper touches `state.py`
  internals — field names (`members`, `member_count`, `composite_id`, `edges`,
  `edge_count`, `alive`) verified against the existing `_two_member_world` helper
  in `tests/test_liquid_drop.py`; (b) `e_max` is a computed `@property`, NOT passed
  to `dataclasses.replace`; (c) grid-blob tests depend on hashed bond energies of
  the chosen `species_val` — `species_val=2` (high-E_b self-pair, per the existing
  liquid-drop test) keeps cohesion positive so the fissility term can discriminate.
- **Out of scope (per spec §7):** force-coupled disruption, full O(n²) pairwise
  Coulomb, species-dependent charge, energy-conservation accounting.
```
