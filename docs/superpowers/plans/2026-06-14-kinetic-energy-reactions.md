# Kinetic-Energy-Coupled Reactions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make fusion and fission energy-conserving and kinetic-energy-coupled — endothermic fission must pay its binding deficit out of fragment-relative KE, and fusion is gated by a collision-energy activation barrier with the released energy stored as bond vibration.

**Architecture:** Reuses the existing fission Q-value (`q = ΣBE(products) − BE(parent)`) and the existing scission system. A shared two-body helper reduces any split to a reduced-mass / relative-velocity pair; the fission veto and the split kick are rewritten in terms of it; the fusion gate adds one hash-derived activation-energy term. All new behavior is behind config flags defaulting to today's behavior; `main.py` opts in.

**Tech Stack:** JAX (jit/vmap, static_argnums config), pytest (CPU for tests).

**Spec:** [docs/superpowers/specs/2026-06-14-kinetic-energy-reactions-design.md](../specs/2026-06-14-kinetic-energy-reactions-design.md)

---

## ⚠️ CHECKPOINT BEFORE PART B (Tasks 6–9): revisit the activation gate

> **User flagged this at planning time (2026-06-14):** *"I'm slightly suspicious about
> the new hash-derived activation gate. I will need to return to this idea when I want
> the plan to be implemented."*
>
> **Do NOT start Task 6 without an explicit go-ahead from the user.** Part A (Tasks 1–5,
> energy-conserving fission) is independently shippable and not affected by this concern —
> implement and land it first. When Part A is done, **stop and re-open the activation-gate
> design with the user** before touching fusion. Open questions to resolve with them:
> - Is a *hash-derived* per-pair `E_a` the right model, or should activation energy be
>   derived from something physical (e.g. tied to bond energy `E_b`, or a single global
>   barrier) rather than an independent random hash stream?
> - Does the born-hot-bond disposal interact acceptably with scission (churn rate)?
>
> Part A gives both exothermic and endothermic *fission*. Part B (fusion activation) is
> the part under suspicion. Keep them separable.

---

## File Structure

| File | Change |
|---|---|
| `halflife/energy.py` | Flip `compute_total_energy` to `KE − BE`; warning comment on `apply_soft_energy_conservation`. |
| `halflife/config.py` | New knobs: `endothermic_fission_mode`, `enable_fusion_activation`, `activation_energy_scale`. Deprecate `forbid_endothermic_fission`. |
| `halflife/chemistry.py` | New `compute_two_body_split` helper; `_apply_binary_splits` signed-kick rework; `apply_composite_decay` energy-gated veto; `_hash_to_activation_energy` + `compute_activation_energy_matrix`; activation gate in `check_neighbor`; born-hot velocity injection in both fusion-apply paths. |
| `halflife/main.py` | `build_config` opts in. |
| `tests/test_kinetic_reactions.py` | New focused test module for all the above. |

---

# PART A — Energy-conserving fission (shippable on its own)

## Task 1: Energy-ledger sign fix

**Files:**
- Modify: `halflife/energy.py:28-31` (`compute_total_energy`), `halflife/energy.py:34-60` (`apply_soft_energy_conservation`)
- Test: `tests/test_kinetic_reactions.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_kinetic_reactions.py`:

```python
"""Tests for kinetic-energy-coupled reactions (exo/endothermic)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp
import numpy as np

from halflife.config import SimConfig
from halflife.state import initialize_world


def test_total_energy_is_ke_minus_be():
    """Total energy uses the KE - BE convention (bound states lower energy)."""
    from halflife.energy import (compute_total_energy, compute_kinetic_energy,
                                 compute_binding_energy)
    config = SimConfig(num_species=3, num_particles=20, max_composites=4)
    world = initialize_world(config, seed=0)
    # Give one composite a nonzero binding energy.
    be = np.zeros(config.max_composites, dtype=np.float32)
    alive = np.zeros(config.max_composites, dtype=bool)
    be[0] = 5.0; alive[0] = True
    comps = world.composites._replace(
        binding_energy=jnp.asarray(be), alive=jnp.asarray(alive))
    world = world._replace(composites=comps)

    ke = float(compute_kinetic_energy(world.particles))
    bind = float(compute_binding_energy(world.composites))
    total = float(compute_total_energy(world))
    assert bind == 5.0
    assert abs(total - (ke - bind)) < 1e-4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_kinetic_reactions.py::test_total_energy_is_ke_minus_be -v`
Expected: FAIL (total currently equals `ke + bind`, so `total - (ke - bind) == 2*bind == 10`).

- [ ] **Step 3: Implement the sign flip**

In `halflife/energy.py`, change `compute_total_energy` (preserve the docstring, update its wording):

```python
def compute_total_energy(state: WorldState) -> jnp.ndarray:
    """Total energy = kinetic - binding (bound states sit lower in energy)."""
    return (compute_kinetic_energy(state.particles) -
            compute_binding_energy(state.composites))
```

Add a warning comment to `apply_soft_energy_conservation` just below its docstring (do
not change its math — it is commented out at the call site and re-enabling it is out of
scope):

```python
    # WARNING: the target_ke line below assumes the OLD total = KE + BE sign
    # convention. compute_total_energy now uses KE - BE, so this arithmetic is
    # inconsistent and MUST be revisited before re-enabling the thermostat
    # (currently commented out in step.py Phase 8). See the 2026-06-14
    # kinetic-energy-reactions spec.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_kinetic_reactions.py::test_total_energy_is_ke_minus_be -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add halflife/energy.py tests/test_kinetic_reactions.py
git commit -m "fix(energy): total energy = KE - BE (bound states lower energy)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Config knobs

**Files:**
- Modify: `halflife/config.py` (near `forbid_endothermic_fission`, line ~127; near `bond_energy_scale`, line ~243)
- Test: `tests/test_kinetic_reactions.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_kinetic_reactions.py`:

```python
def test_new_config_defaults():
    """New knobs exist and default to current behavior."""
    config = SimConfig()
    assert config.endothermic_fission_mode == "forbid"
    assert config.enable_fusion_activation is False
    assert config.activation_energy_scale == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_kinetic_reactions.py::test_new_config_defaults -v`
Expected: FAIL with `AttributeError: 'SimConfig' object has no attribute 'endothermic_fission_mode'`

- [ ] **Step 3: Add the config fields**

In `halflife/config.py`, immediately after the `forbid_endothermic_fission: bool = True`
line (keep that line and its comment), add:

```python
    # ── Energy-conserving reactions (2026-06-14 spec) ─────────────────────────
    # endothermic_fission_mode selects the fission veto:
    #   "forbid"       → today's hard wall: a decay roll whose best cut has Q<0
    #                    is suppressed entirely (forbid_endothermic_fission=True
    #                    behavior; that boolean is now deprecated, see below).
    #   "energy_gated" → endothermic fission is ALLOWED iff the two fragments'
    #                    relative kinetic energy can pay the deficit:
    #                    KE_rel + Q >= 0. The split then drains |Q| of KE.
    # forbid_endothermic_fission is DEPRECATED and superseded by this field;
    # it is retained only so old callers/presets don't break. Prefer the mode.
    endothermic_fission_mode: str = "forbid"
```

After the `bond_break_attempt_rate` / `max_scissions_per_step` block (line ~245), add:

```python
    # ── Fusion activation energy (2026-06-14 spec; PART B — under review) ──────
    # When True, a pair fuses only if its relative collision energy clears a
    # hash-derived per-species-pair activation barrier E_a in
    # [0, activation_energy_scale]; the released KE_rel + Q is stored as
    # vibration of the new bond (scission breaks it if it exceeds E_b).
    enable_fusion_activation: bool = False
    activation_energy_scale: float = 1.0  # ceiling for hash-derived E_a (tuning seed)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_kinetic_reactions.py::test_new_config_defaults -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add halflife/config.py tests/test_kinetic_reactions.py
git commit -m "feat(config): add endothermic_fission_mode + fusion activation knobs

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `compute_two_body_split` shared helper

The reduced-mass / relative-velocity decomposition used by both the fission veto and the
split kick. With `mass = ones`, fragment mass equals member count, but use
`particles.mass` so it stays correct if masses ever vary.

**Files:**
- Modify: `halflife/chemistry.py` (add helper near `compute_degree`, ~line 243)
- Test: `tests/test_kinetic_reactions.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_kinetic_reactions.py`:

```python
def test_two_body_split_decomposition():
    """compute_two_body_split returns correct reduced mass and relative velocity."""
    from halflife.chemistry import compute_two_body_split
    config = SimConfig(num_species=3, num_particles=10)
    world = initialize_world(config, seed=0)
    M = config.max_composite_size

    # Members 0,1 -> fragment 0 ; members 2,3 -> fragment 1.
    vel = np.zeros((config.num_particles, 2), dtype=np.float32)
    vel[0] = (1.0, 0.0); vel[1] = (1.0, 0.0)   # fragment 0 COM vel = (1,0)
    vel[2] = (-1.0, 0.0); vel[3] = (-1.0, 0.0) # fragment 1 COM vel = (-1,0)
    parts = world.particles._replace(velocity=jnp.asarray(vel))

    members = np.full(M, -1, dtype=np.int32); members[:4] = (0, 1, 2, 3)
    assignment = np.full(M, -1, dtype=np.int32); assignment[:4] = (0, 0, 1, 1)

    mu, v_rel, V0, V1, M0, M1 = compute_two_body_split(
        jnp.asarray(members), jnp.asarray(assignment), parts, config)

    assert abs(float(M0) - 2.0) < 1e-5   # two unit-mass members
    assert abs(float(M1) - 2.0) < 1e-5
    assert abs(float(mu) - 1.0) < 1e-5   # 2*2/(2+2) = 1
    assert np.allclose(np.asarray(V0), [1.0, 0.0], atol=1e-5)
    assert np.allclose(np.asarray(V1), [-1.0, 0.0], atol=1e-5)
    assert np.allclose(np.asarray(v_rel), [2.0, 0.0], atol=1e-5)
    # KE_rel = 0.5 * mu * |v_rel|^2 = 0.5 * 1 * 4 = 2.0
    ke_rel = 0.5 * float(mu) * float(np.sum(np.asarray(v_rel) ** 2))
    assert abs(ke_rel - 2.0) < 1e-5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_kinetic_reactions.py::test_two_body_split_decomposition -v`
Expected: FAIL with `ImportError: cannot import name 'compute_two_body_split'`

- [ ] **Step 3: Implement the helper**

In `halflife/chemistry.py`, add after `compute_degree` (around line 278):

```python
def compute_two_body_split(member_ids: jnp.ndarray, assignment: jnp.ndarray,
                           particles, config: SimConfig):
    """
    Two-body (reduced-mass) decomposition of a binary composite split.

    Splits the members into fragment p0 (assignment == 0) and p1 (== 1) and
    returns the masses, COM velocities, reduced mass and relative velocity of
    the two fragments — the quantities energy-conserving fission needs to gate
    and apply the split. Only the fragment-COM relative motion can do the
    reaction; each member's motion relative to its own fragment COM is internal
    vibration and is not represented here.

    Args:
        member_ids: (M,) int32 — composite member particle ids (-1 = padding)
        assignment: (M,) int32 — 0 → p0, 1 → p1, -1 → not a member
        particles:  ParticleState (provides mass, velocity)
        config:     SimConfig (static)

    Returns:
        (mu, v_rel, V0, V1, M0, M1)
          M0, M1 : scalar float32 fragment masses
          V0, V1 : (2,) float32 fragment COM velocities (0 if fragment empty)
          mu     : scalar float32 reduced mass M0*M1/(M0+M1) (0 if either empty)
          v_rel  : (2,) float32 = V0 - V1
    """
    safe = jnp.where(member_ids >= 0, member_ids, 0)
    mass = particles.mass[safe]                 # (M,)
    vel = particles.velocity[safe]              # (M, 2)
    in0 = (assignment == 0) & (member_ids >= 0)
    in1 = (assignment == 1) & (member_ids >= 0)
    M0 = jnp.sum(jnp.where(in0, mass, 0.0))
    M1 = jnp.sum(jnp.where(in1, mass, 0.0))
    V0 = jnp.sum(jnp.where(in0[:, None], mass[:, None] * vel, 0.0), axis=0) / (M0 + 1e-8)
    V1 = jnp.sum(jnp.where(in1[:, None], mass[:, None] * vel, 0.0), axis=0) / (M1 + 1e-8)
    mu = (M0 * M1) / (M0 + M1 + 1e-8)
    v_rel = V0 - V1
    return mu, v_rel, V0, V1, M0, M1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_kinetic_reactions.py::test_two_body_split_decomposition -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add halflife/chemistry.py tests/test_kinetic_reactions.py
git commit -m "feat(chemistry): compute_two_body_split reduced-mass split helper

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Signed-kick rework of `_apply_binary_splits`

Replace the non-negative additive kick (`kick_energy`, `per_member` velocity add) with a
momentum-conserving relative-velocity rescale to `KE_rel + q`. The argument
`kick_energy` becomes a **signed** per-row `q`. The bond-scission caller passes zeros, so
`q = 0` leaves fragment velocities unchanged (preserving scission's no-kick semantics).

**Files:**
- Modify: `halflife/chemistry.py:407-763` (`_apply_binary_splits`): `per_split` (add masses/velocities + δV) and `per_member` (apply δV).
- Test: `tests/test_kinetic_reactions.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_kinetic_reactions.py`. These call `_apply_binary_splits` directly on
a hand-built 4-member composite split into two dimers, checking momentum conservation and
`KE_after − KE_before == q`.

```python
def _two_dimer_composite(seed=0):
    """Build a world with one alive 4-member composite (members 0..3),
    two internal edges (0-1, 2-3) so a cut at the right edge gives dimers."""
    config = SimConfig(num_species=3, num_particles=10, max_composites=4,
                       boundary_mode="free")
    world = initialize_world(config, seed=seed)
    C, E, M = config.max_composites, config.e_max, config.max_composite_size
    members = np.full((C, M), -1, dtype=np.int32)
    member_count = np.zeros(C, dtype=np.int32)
    edges = np.full((C, E, 2), -1, dtype=np.int32)
    edge_count = np.zeros(C, dtype=np.int32)
    alive = np.zeros(C, dtype=bool)
    members[0, :4] = (0, 1, 2, 3); member_count[0] = 4
    edges[0, 0] = (0, 1); edges[0, 1] = (2, 3); edge_count[0] = 2
    alive[0] = True
    # Positions: fragment 0 near x=0, fragment 1 near x=5 (well separated).
    pos = np.asarray(world.particles.position).copy()
    pos[0] = (0.0, 0.0); pos[1] = (0.5, 0.0)
    pos[2] = (5.0, 0.0); pos[3] = (5.5, 0.0)
    vel = np.zeros((config.num_particles, 2), dtype=np.float32)
    comps = world.composites._replace(
        members=jnp.asarray(members), member_count=jnp.asarray(member_count),
        edges=jnp.asarray(edges), edge_count=jnp.asarray(edge_count),
        alive=jnp.asarray(alive))
    parts = world.particles._replace(position=jnp.asarray(pos),
                                     velocity=jnp.asarray(vel))
    return config, world._replace(particles=parts, composites=comps)


def test_split_conserves_momentum_and_releases_q():
    """Exothermic split: total momentum unchanged, KE rises by exactly q."""
    from halflife.chemistry import _apply_binary_splits
    from halflife.state import initialize_physics_params
    config, world = _two_dimer_composite()
    physics = initialize_physics_params(config)
    M = config.max_composite_size

    # One split, slot 0, assignment 0,0,1,1.
    split_slots = jnp.array([0], dtype=jnp.int32)
    fires = jnp.array([True])
    assignment = np.full((1, M), -1, dtype=np.int32)
    assignment[0, :4] = (0, 0, 1, 1)
    q = jnp.array([3.0], dtype=jnp.float32)   # exothermic release

    p0 = float(jnp.sum(world.particles.mass[:, None] * world.particles.velocity, axis=0)[0])
    ke0 = 0.5 * float(jnp.sum(world.particles.mass *
                              jnp.sum(world.particles.velocity ** 2, axis=-1)))

    new_p, new_c, _ = _apply_binary_splits(
        world.particles, world.composites, split_slots, fires,
        jnp.asarray(assignment), q, config, physics)

    p1 = float(jnp.sum(new_p.mass[:, None] * new_p.velocity, axis=0)[0])
    ke1 = 0.5 * float(jnp.sum(new_p.mass * jnp.sum(new_p.velocity ** 2, axis=-1)))
    assert abs(p1 - p0) < 1e-3            # momentum conserved (was 0)
    assert abs((ke1 - ke0) - 3.0) < 1e-3  # KE released == q


def test_split_zero_q_leaves_velocity_unchanged():
    """q = 0 (scission's no-kick path) must not change any velocity."""
    from halflife.chemistry import _apply_binary_splits
    from halflife.state import initialize_physics_params
    config, world = _two_dimer_composite()
    physics = initialize_physics_params(config)
    M = config.max_composite_size
    split_slots = jnp.array([0], dtype=jnp.int32)
    fires = jnp.array([True])
    assignment = np.full((1, M), -1, dtype=np.int32)
    assignment[0, :4] = (0, 0, 1, 1)
    q = jnp.array([0.0], dtype=jnp.float32)

    new_p, _, _ = _apply_binary_splits(
        world.particles, world.composites, split_slots, fires,
        jnp.asarray(assignment), q, config, physics)
    assert np.allclose(np.asarray(new_p.velocity),
                       np.asarray(world.particles.velocity), atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_kinetic_reactions.py::test_split_conserves_momentum_and_releases_q tests/test_kinetic_reactions.py::test_split_zero_q_leaves_velocity_unchanged -v`
Expected: FAIL — current `_apply_binary_splits` treats `kick_energy` as non-negative and
adds `sqrt(2·e/n)` along the COM-position axis ignoring pre-existing velocity, so `KE`
rises by the wrong amount and the `q=0` case still applies a kick.

- [ ] **Step 3: Rework `per_split` and `per_member`**

In `halflife/chemistry.py`, in `_apply_binary_splits`:

(a) Rename the parameter for clarity (signed Q, not a magnitude). Change the signature
`kick_energy` → `q_signed` and update the docstring line describing it to:
`q_signed: (K,) float32 — signed Q-value per row (ΣBE(products) − BE(parent))`.

(b) Extend `per_split(k)` to also compute the momentum-conserving velocity shifts. Replace
its `return com0, com1, ...` with the version below (keep the existing position-COM math
above it intact):

```python
    def per_split(k):
        c = safe_slots[k]
        n = composites.member_count[c]
        member_ids = composites.members[c]
        safe_ids = jnp.where(member_ids >= 0, member_ids, 0)
        valid = (member_ids >= 0) & (m_idx < n)
        ref = particles.position[safe_ids[0]]

        def disp_from_ref(idx):
            d = particles.position[safe_ids[idx]] - ref
            if config.boundary_mode == "periodic":
                d = d - config.world_width  * jnp.round(d[0] / config.world_width)  * jnp.array([1., 0.])
                d = d - config.world_height * jnp.round(d[1] / config.world_height) * jnp.array([0., 1.])
            return d

        rels = jax.vmap(disp_from_ref)(jnp.arange(M))  # (M, 2)
        a = assignment[k]
        in_p0 = valid & (a == 0)
        in_p1 = valid & (a == 1)
        n0 = jnp.sum(in_p0.astype(jnp.float32))
        n1 = jnp.sum(in_p1.astype(jnp.float32))
        com0 = ref + jnp.sum(rels * in_p0[:, None].astype(jnp.float32), axis=0) / (n0 + 1e-8)
        com1 = ref + jnp.sum(rels * in_p1[:, None].astype(jnp.float32), axis=0) / (n1 + 1e-8)

        # ── Energy-conserving relative-velocity update ──────────────────────
        # KE_rel_after = max(KE_rel_before + q, 0); rescale v_rel about the
        # conserved pair-COM velocity. Direction follows the existing relative
        # velocity when it is non-negligible, else the COM-position axis (which
        # recovers the old exothermic-from-rest kick).
        mu, v_rel, V0, V1, M0, M1 = compute_two_body_split(
            member_ids, a, particles, config)
        ke_rel = 0.5 * mu * jnp.sum(v_rel ** 2)
        target_ke = jnp.maximum(ke_rel + q_signed[k], 0.0)
        target_speed = jnp.sqrt(2.0 * target_ke / (mu + 1e-8))

        # COM-position axis (min-image) fallback direction.
        d = com0 - com1
        if config.boundary_mode == "periodic":
            d = d - config.world_width  * jnp.round(d[0] / config.world_width)  * jnp.array([1., 0.])
            d = d - config.world_height * jnp.round(d[1] / config.world_height) * jnp.array([0., 1.])
        d_hat_pos = d / (jnp.linalg.norm(d) + 1e-8)

        vmag = jnp.linalg.norm(v_rel)
        VREL_EPS = jnp.float32(1e-4)
        dir_hat = jnp.where(vmag > VREL_EPS, v_rel / (vmag + 1e-8), d_hat_pos)
        v_rel_after = target_speed * dir_hat
        dv = v_rel_after - v_rel

        both = (M0 > 0.0) & (M1 > 0.0)
        w0 = M1 / (M0 + M1 + 1e-8)
        w1 = -M0 / (M0 + M1 + 1e-8)
        dV0 = jnp.where(both, w0 * dv, jnp.zeros(2))
        dV1 = jnp.where(both, w1 * dv, jnp.zeros(2))

        return com0, com1, n0.astype(jnp.int32), n1.astype(jnp.int32), dV0, dV1

    all_com0, all_com1, all_n0, all_n1, all_dV0, all_dV1 = jax.vmap(per_split)(
        jnp.arange(K, dtype=jnp.int32))
```

(c) Rewrite the kick inside `per_member(k, m)`. Replace the block that computes
`d`, `d_hat`, `e_per`, `v0`, `v1`, and `kick` (the section from `# Direction along
COM-COM axis (min-image).` through the `kick = jnp.where(...)` assignment) with:

```python
        # Energy-conserving COM-velocity shift: every member of a fragment gets
        # its fragment's COM-velocity change (preserves internal vibration;
        # M0*dV0 + M1*dV1 == 0 so total momentum is conserved).
        kick = jnp.where(
            a == 0,
            all_dV0[k],
            jnp.where(a == 1, all_dV1[k], jnp.zeros(2)),
        )
```

Delete the now-unused `all_n0`/`all_n1` references inside `per_member` only if they were
solely used for the old `v0`/`v1` formulas; `n0`/`n1` are still returned for the
`forms_p0`/`forms_p1` size checks (`all_n0[k]`, `all_n1[k]`), so keep those reads.

(d) Update the call site in `apply_composite_decay` (Task 5) to pass signed `q`. The
bond-scission caller already passes a zeros array; rename nothing there.

- [ ] **Step 4: Run tests to verify they pass**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_kinetic_reactions.py::test_split_conserves_momentum_and_releases_q tests/test_kinetic_reactions.py::test_split_zero_q_leaves_velocity_unchanged -v`
Expected: PASS

- [ ] **Step 5: Regression — existing fission/scission tests still green**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_chemistry.py -k "fission or scission or decay or split" -v`
Expected: PASS. If a test asserts an *exact* post-fission velocity magnitude (the old
`sqrt(q/n)` kick), it was pinning non-conserving behavior — update it to assert momentum
conservation and `ΔKE == q` instead, and note the change in the commit.

- [ ] **Step 6: Commit**

```bash
git add halflife/chemistry.py tests/test_kinetic_reactions.py
git commit -m "feat(chemistry): momentum-conserving signed split kick (KE_after = KE_before + Q)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Energy-gated endothermic fission veto

Compute `KE_rel` for the chosen cut in `choose_cut`, and in `energy_gated` mode allow
`Q < 0` fissions when `KE_rel + Q ≥ 0`. Pass signed `q` (not `max(q,0)`) to the split.

**Files:**
- Modify: `halflife/chemistry.py:843-898` (`apply_composite_decay`: `choose_cut`, veto, call)
- Test: `tests/test_kinetic_reactions.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_kinetic_reactions.py`. Same 4-member composite; force a decay roll
by setting `half_life` tiny. In `energy_gated` mode, a hot composite (large relative
fragment velocity) whose best cut is endothermic should fission; a cold one should not.

```python
def _force_decay_config(mode):
    return SimConfig(num_species=3, num_particles=10, max_composites=4,
                     boundary_mode="free", endothermic_fission_mode=mode,
                     forbid_endothermic_fission=(mode == "forbid"))


def _run_decay(world, config):
    from halflife.chemistry import apply_composite_decay
    from halflife.state import initialize_physics_params
    import jax
    physics = initialize_physics_params(config)
    # Pin a tiny half-life so the decay roll fires with prob ~1.
    hl = np.asarray(world.composites.half_life).copy()
    hl[0] = 1e-6
    world = world._replace(composites=world.composites._replace(
        half_life=jnp.asarray(hl)))
    # Use a fixed key that rolls a fission for slot 0 (prob ~1 with tiny HL).
    world = world._replace(rng_key=jax.random.PRNGKey(0))
    out = apply_composite_decay(world, config, physics)
    return out  # WorldState (emit_events False)


def test_energy_gated_endothermic_needs_kinetic_energy():
    """In energy_gated mode an endothermic split fires only when fragments are hot."""
    # We don't know a priori whether slot-0's best cut is exo or endothermic for
    # these species, so test the GATE invariant directly: with mode=forbid an
    # endothermic best-cut never fires; with energy_gated + enough KE it can.
    config = _force_decay_config("energy_gated")
    _, hot = _two_dimer_composite()
    # Make the two dimers approach fast: fragment 0 -> +x, fragment 1 -> -x.
    vel = np.zeros((config.num_particles, 2), dtype=np.float32)
    vel[0] = (10.0, 0.0); vel[1] = (10.0, 0.0)
    vel[2] = (-10.0, 0.0); vel[3] = (-10.0, 0.0)  # KE_rel large
    hot = hot._replace(particles=hot.particles._replace(velocity=jnp.asarray(vel)))
    hot_out = _run_decay(hot, config)

    _, cold = _two_dimer_composite()  # zero velocity → KE_rel = 0
    cold_out = _run_decay(cold, config)

    # If slot-0's best cut is endothermic (Q<0): hot fissions (member 0 leaves
    # the composite), cold does not. If it's exothermic both fission — in which
    # case this asserts the weaker (always-true) invariant. Either way, hot must
    # fission whenever cold does.
    def slot0_dissolved(out):
        return int(np.asarray(out.composites.member_count)[0]) < 4
    assert slot0_dissolved(hot_out) or (not slot0_dissolved(cold_out))
```

> Implementation note: this test is intentionally robust to whichever sign slot-0's best
> cut has for the default hash universe. A sharper test can be added once the universe's
> Q sign for this member set is known — pick species via `--override` style construction
> so the best cut is provably endothermic, then assert hot-fires / cold-suppressed
> strictly. Leave a TODO referencing this.

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_kinetic_reactions.py::test_energy_gated_endothermic_needs_kinetic_energy -v`
Expected: FAIL — `apply_composite_decay` does not yet read `endothermic_fission_mode`
(the kwarg `forbid_endothermic_fission=False` currently lets Q<0 fire with zero kick,
ignoring KE), so the gate is wrong.

- [ ] **Step 3: Implement the energy gate**

In `halflife/chemistry.py`, in `choose_cut(k)` (within `apply_composite_decay`), after the
existing `q = score[v] - composites.binding_energy[c]` line and the `in_p1` / `a`
computation, compute `KE_rel` for the chosen assignment and return it:

```python
        in_p1 = descendant_mask(parent, v.astype(jnp.int32), M, iters) & valid_m
        a = jnp.where(valid_m,
                      jnp.where(in_p1, jnp.int32(1), jnp.int32(0)),
                      jnp.int32(-1))
        has_cut = jnp.any(cand_v)

        # Fragment-relative KE available to pay an endothermic deficit.
        mu, v_rel, _, _, _, _ = compute_two_body_split(members, a, particles, config)
        ke_rel = 0.5 * mu * jnp.sum(v_rel ** 2)
        return a, q, has_cut, ke_rel

    assignment, q_all, has_cut, ke_rel_all = jax.vmap(choose_cut)(
        jnp.arange(K_f, dtype=jnp.int32))
```

Then replace the veto block:

```python
    fires = fiss_valid & has_cut
    if config.endothermic_fission_mode == "energy_gated":
        # Endothermic allowed iff fragments can pay |Q| out of relative KE.
        fires = fires & (ke_rel_all + q_all >= 0.0)
    else:  # "forbid" — today's hard wall (also honors the deprecated boolean)
        fires = fires & (q_all >= 0.0)
    # Signed Q to the split (KE_after = KE_before + Q). In forbid mode every
    # firing row has q >= 0; in energy_gated mode q may be negative and the
    # split drains |Q| from the fragments' relative motion.
    new_particles, new_composites, events = _apply_binary_splits(
        particles, composites, fiss_idx, fires, assignment, q_all, config, physics)
```

(Delete the old `kick = jnp.maximum(q_all, 0.0)` line and the old `_apply_binary_splits`
call that passed `kick`.)

> Backward-compat: when `endothermic_fission_mode` is left at its `"forbid"` default
> (and the deprecated `forbid_endothermic_fission=True`), behavior matches today's veto.
> The split kick now also accounts for pre-existing fragment velocity (Task 4), so the
> exothermic kick is energy-consistent rather than `sqrt(q/n)`-from-rest.

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_kinetic_reactions.py::test_energy_gated_endothermic_needs_kinetic_energy -v`
Expected: PASS

- [ ] **Step 5: Full Part-A regression**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_chemistry.py tests/test_kinetic_reactions.py -v`
Expected: PASS (existing suite green with defaults; new tests green).

- [ ] **Step 6: Commit**

```bash
git add halflife/chemistry.py tests/test_kinetic_reactions.py
git commit -m "feat(chemistry): energy-gated endothermic fission (KE_rel + Q >= 0)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### ✅ END OF PART A — STOP HERE

Part A is complete and shippable: fission now conserves energy and supports both
exothermic and endothermic splits behind `endothermic_fission_mode`. **Do not proceed to
Part B without re-opening the activation-gate design with the user** (see the checkpoint
banner at the top). Consider landing Part A (PR / merge) before starting Part B.

---

# PART B — Fusion activation energy (⚠️ REVISIT DESIGN BEFORE IMPLEMENTING)

> Tasks 6–9 implement the hash-derived fusion activation gate and born-hot bonds. The
> user is suspicious of the hash-derived `E_a` model and wants to revisit it first. The
> tasks below assume the design as specced; adjust them to whatever model the user lands
> on (e.g. `E_a` tied to bond energy, or a single global barrier) before writing code.

## Task 6: Hash-derived activation energy

**Files:**
- Modify: `halflife/chemistry.py` (add near `_hash_to_bond_energy` / `compute_bond_energy_matrix`, ~line 176-205)
- Test: `tests/test_kinetic_reactions.py`

- [ ] **Step 1: Write the failing test**

```python
def test_activation_energy_matrix_decorrelated():
    """E_a matrix is symmetric, in [0, scale], and not rank-identical to E_b."""
    from halflife.chemistry import (compute_activation_energy_matrix,
                                    compute_bond_energy_matrix)
    config = SimConfig(num_species=6, activation_energy_scale=2.0)
    ea = np.asarray(compute_activation_energy_matrix(config))
    assert ea.shape == (6, 6)
    assert np.allclose(ea, ea.T, atol=1e-6)         # symmetric (commutative hash)
    assert (ea >= 0).all() and (ea <= 2.0 + 1e-5).all()
    eb = np.asarray(compute_bond_energy_matrix(config))
    iu = np.triu_indices(6)
    assert (np.argsort(ea[iu]) != np.argsort(eb[iu])).any()  # decorrelated stream
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_kinetic_reactions.py::test_activation_energy_matrix_decorrelated -v`
Expected: FAIL with `ImportError: cannot import name 'compute_activation_energy_matrix'`

- [ ] **Step 3: Implement** (own Fibonacci stream, distinct from BE/valence/rest-length/bond-energy)

In `halflife/chemistry.py`, after `compute_bond_energy_matrix` (~line 205):

```python
def _hash_to_activation_energy(s_i: jnp.ndarray, s_j: jnp.ndarray,
                               config: SimConfig) -> jnp.ndarray:
    """
    Hash-derived fusion activation energy for species pair (s_i, s_j).

    Order-independent (commutative additive pair hash) and re-mixed with a
    Fibonacci constant DISTINCT from the BE (2654435761,>>13), valence
    (0x9E3779B1,>>13), rest-length (0x9E3779B1,>>11) and bond-energy
    (0x85EBCA6B,>>9) streams, so E_a is decorrelated from all of them.

    Returns: scalar float32 in [0, config.activation_energy_scale]
    """
    h_i = _entity_hash_val(s_i, config).astype(jnp.uint32)
    h_j = _entity_hash_val(s_j, config).astype(jnp.uint32)
    h = (h_i + h_j) % jnp.uint32(config.hash_modulus)
    h_mix = (h * jnp.uint32(0xC2B2AE35)) ^ (h >> jnp.uint32(16))
    frac = (h_mix % jnp.uint32(1000)).astype(jnp.float32) / 999.0
    return frac * config.activation_energy_scale


@functools.partial(jax.jit, static_argnums=(0,))
def compute_activation_energy_matrix(config: SimConfig) -> jnp.ndarray:
    """(num_species, num_species) activation-energy matrix. Static per config."""
    species_idx = jnp.arange(config.num_species, dtype=jnp.int32)
    return jax.vmap(
        lambda i: jax.vmap(
            lambda j: _hash_to_activation_energy(i, j, config)
        )(species_idx)
    )(species_idx)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_kinetic_reactions.py::test_activation_energy_matrix_decorrelated -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add halflife/chemistry.py tests/test_kinetic_reactions.py
git commit -m "feat(chemistry): hash-derived fusion activation energy matrix

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Activation gate in `check_neighbor`

**Files:**
- Modify: `halflife/chemistry.py:1206-1308` (`attempt_fusion`: matrix build + `check_neighbor`)
- Test: `tests/test_kinetic_reactions.py`

- [ ] **Step 1: Write the failing test**

Place two free particles of the same species within `fusion_radius`, identical except
relative speed; with `enable_fusion_activation=True` a slow pair must not fuse and a fast
pair must (assuming the pair clears the BE threshold — pick species so it does, or set
`fusion_threshold` low).

```python
def _two_free_particles(rel_speed, config):
    """World with particles 0,1 (species 0) adjacent; particle 1 approaches 0
    at rel_speed along x; all others far away."""
    world = initialize_world(config, seed=0)
    pos = np.asarray(world.particles.position).copy()
    pos[:] = 1000.0  # park everyone far away
    pos[0] = (0.0, 0.0); pos[1] = (config.fusion_radius * 0.5, 0.0)
    vel = np.zeros((config.num_particles, 2), dtype=np.float32)
    vel[1] = (-rel_speed, 0.0)   # particle 1 moves toward 0
    sp = np.asarray(world.particles.species).copy(); sp[0] = 0; sp[1] = 0
    parts = world.particles._replace(position=jnp.asarray(pos),
                                     velocity=jnp.asarray(vel),
                                     species=jnp.asarray(sp))
    return world._replace(particles=parts)


def _did_fuse(world, config):
    from halflife.spatial import build_cell_list, find_all_neighbors
    from halflife.chemistry import attempt_fusion, _species_valences, compute_degree
    from halflife.state import initialize_physics_params, initialize_interaction_params
    physics = initialize_physics_params(config)
    params = initialize_interaction_params(config, seed=42)
    cl = build_cell_list(world.particles, config)
    neighbors = find_all_neighbors(world.particles, cl, config)
    deg = compute_degree(world.composites, config)
    sv = _species_valences(config)
    out = attempt_fusion(world, neighbors, params, config, physics,
                         degree=deg, species_valences=sv)
    new_state = out[0]
    return bool(np.asarray(new_state.particles.composite_id)[0] >= 0)


def test_fusion_activation_blocks_slow_pair():
    config = SimConfig(num_species=3, num_particles=20, fusion_threshold=0.0,
                       enable_fusion_activation=True, activation_energy_scale=5.0,
                       boundary_mode="free")
    slow = _two_free_particles(0.1, config)
    fast = _two_free_particles(100.0, config)
    assert _did_fuse(slow, config) is False
    assert _did_fuse(fast, config) is True
```

> See `tests/test_chemistry.py` fusion tests (~line 803-948) for the cell-list /
> neighbor build pattern mirrored above.

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_kinetic_reactions.py::test_fusion_activation_blocks_slow_pair -v`
Expected: FAIL — no activation gate yet, so the slow pair also fuses.

- [ ] **Step 3: Implement the gate**

In `halflife/chemistry.py` `attempt_fusion`, after `fusion_r2 = physics.fusion_radius ** 2`
(~line 1213), build the matrix once (statically gated so it is DCE'd when off):

```python
    if config.enable_fusion_activation:
        activation_matrix = compute_activation_energy_matrix(config)  # (S, S)
```

Inside `check_neighbor(j)`, after `be_eff = _hash_to_binding_energy(...)` and before
`can_fuse`, add the activation term:

```python
            # Activation gate: the pair must collide hard enough. KE_rel is the
            # reduced-mass relative kinetic energy of the two contacting
            # particles i and j (only the relative approach can react).
            if config.enable_fusion_activation:
                mi = particles.mass[i]
                mj = particles.mass[j]
                mu_ij = (mi * mj) / (mi + mj + 1e-8)
                v_rel_ij = particles.velocity[i] - particles.velocity[j]
                ke_rel_ij = 0.5 * mu_ij * jnp.dot(v_rel_ij, v_rel_ij)
                e_a = activation_matrix[particles.species[i], particles.species[j]]
                has_activation = ke_rel_ij >= e_a
            else:
                has_activation = jnp.bool_(True)
```

Then add `& has_activation` to the `can_fuse` conjunction:

```python
            can_fuse = (
                valid & in_range
                & (be_eff > physics.fusion_threshold)
                & has_free_bonds
                & ~would_overflow
                & has_activation
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_kinetic_reactions.py::test_fusion_activation_blocks_slow_pair -v`
Expected: PASS

- [ ] **Step 5: Regression (both fusion modes)**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_chemistry.py -k fusion -v`
Expected: PASS (gate is off by default, so DCE'd; existing fusion behavior unchanged).

- [ ] **Step 6: Commit**

```bash
git add halflife/chemistry.py tests/test_kinetic_reactions.py
git commit -m "feat(chemistry): fusion activation-energy gate (KE_rel >= E_a)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Born-hot bond vibration on fusion

On a successful fusion, give the two contact particles a relative velocity carrying
`KE_rel + Q` along their bond axis. Over-energetic bonds (`> E_b`) are then snapped by the
existing scission system next step.

**Files:**
- Modify: `halflife/chemistry.py` — `_fusion_apply_matching` (the contact pair `safe_i`,
  `safe_j` with merged `be_eff`) and `fusion_scan_body` (the `safe_i`, `safe_j`, `be_eff`),
  both write `new_velocity` for the contact pair.
- Test: `tests/test_kinetic_reactions.py`

- [ ] **Step 1: Write the failing test**

A high-speed fusion that clears activation should fuse and then, within a couple of steps,
scission back apart because the born-hot bond exceeds `E_b`.

```python
def test_born_hot_bond_scissions_when_overenergetic():
    """A very fast fusion sticks for an instant then snaps (born-hot > E_b)."""
    from halflife.step import simulation_step
    from halflife.state import initialize_physics_params, initialize_interaction_params
    config = SimConfig(num_species=3, num_particles=20, fusion_threshold=0.0,
                       enable_fusion_activation=True, activation_energy_scale=0.1,
                       enable_bond_scission=True, bond_energy_scale=1.0,
                       bond_mode="edges", boundary_mode="free")
    world = _two_free_particles(200.0, config)  # enormous KE_rel
    physics = initialize_physics_params(config)
    params = initialize_interaction_params(config, seed=42)
    # Step once: should fuse (clears tiny E_a) and inject a large vibration.
    s1 = simulation_step(world, params, config, physics)
    # Step a few more: scission should break the over-hot bond.
    s = s1
    for _ in range(5):
        s = simulation_step(s, params, config, physics)
    # Particle 0 ends free again (the violent collision did not stick).
    assert int(np.asarray(s.particles.composite_id)[0]) < 0
```

> `simulation_step(state, params, config, physics)` — argument order confirmed against
> `halflife/step.py:381`.

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_kinetic_reactions.py::test_born_hot_bond_scissions_when_overenergetic -v`
Expected: FAIL — without born-hot injection the fused pair keeps its small velocities and
the bond is not over-stretched, so it never scissions; particle 0 stays bound.

- [ ] **Step 3: Implement born-hot injection (both fusion-apply paths)**

The released energy is `E_release = KE_rel(i,j) + Q`, where `Q = BE(merged) − BE(i-entity)
− BE(j-entity)`. For free+free, the entity BEs are 0 so `Q = be_eff`. To keep this task
self-contained and avoid recomputing composite entity BEs, scope born-hot injection to the
**free+free** case (the dominant fusion and the only one that starts from two bare
particles); composite mergers keep today's velocity handling. Set the contact pair's
relative speed so `½ μ_ij |v_rel_new|² = KE_rel + be_eff`, along the bond axis
`pos_i − pos_j`, conserving the pair COM velocity.

Add a helper near `_apply_binary_splits` (reused by both paths):

```python
def _born_hot_contact_velocity(particles, i, j, e_release, config):
    """
    Return (new_vi, new_vj): the contact pair's velocities after storing
    e_release as relative (vibrational) kinetic energy along the bond axis,
    conserving their COM velocity. e_release should be >= 0 (clamped).
    """
    mi = particles.mass[i]; mj = particles.mass[j]
    msum = mi + mj + 1e-8
    vi = particles.velocity[i]; vj = particles.velocity[j]
    vcom = (mi * vi + mj * vj) / msum
    mu_ij = (mi * mj) / msum
    target_ke = jnp.maximum(e_release, 0.0)
    speed = jnp.sqrt(2.0 * target_ke / (mu_ij + 1e-8))
    d = particles.position[i] - particles.position[j]
    if config.boundary_mode == "periodic":
        d = d - config.world_width  * jnp.round(d[0] / config.world_width)  * jnp.array([1., 0.])
        d = d - config.world_height * jnp.round(d[1] / config.world_height) * jnp.array([0., 1.])
    d_hat = d / (jnp.linalg.norm(d) + 1e-8)
    v_rel_new = speed * d_hat
    new_vi = vcom + (mj / msum) * v_rel_new
    new_vj = vcom - (mi / msum) * v_rel_new
    return new_vi, new_vj
```

In `_fusion_apply_matching`, where the merged state's velocities are written, for each
accepted free+free pair compute `KE_rel` from the pre-fusion velocities of `safe_i`,
`safe_j`, set `e_release = KE_rel + be_eff[k]`, and scatter `new_vi/new_vj` into the
velocity array at `safe_i/safe_j` (gated on `can_fuse & i_is_free & j_is_free`). Mirror the
same in `fusion_scan_body` for its single `(safe_i, safe_j)` per iteration. Use
`.at[...].set(...)` with `mode='drop'` and the existing per-pair validity masks so
non-free or non-firing pairs are untouched.

> Implementation detail: both paths already build a `new_velocity` (matching path) or carry
> velocities in the scan; thread the born-hot writes through that same array. Keep the
> writes gated so composite mergers and non-fusing rows are byte-for-byte unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_kinetic_reactions.py::test_born_hot_bond_scissions_when_overenergetic -v`
Expected: PASS

- [ ] **Step 5: Regression**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_chemistry.py tests/test_kinetic_reactions.py -v`
Expected: PASS (born-hot gated to free+free + activation-on; defaults unchanged).

- [ ] **Step 6: Commit**

```bash
git add halflife/chemistry.py tests/test_kinetic_reactions.py
git commit -m "feat(chemistry): born-hot bond vibration on free+free fusion

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Live-app opt-in

**Files:**
- Modify: `halflife/main.py:83-105` (`build_config`)
- Manual verification (no unit test — it's app wiring)

- [ ] **Step 1: Enable the new mechanics in `build_config`**

In `halflife/main.py`, in `build_config`, alongside the existing `kwargs['emit_events']`
and `kwargs['angle_mode']` assignments, add:

```python
    kwargs['endothermic_fission_mode'] = 'energy_gated'
    kwargs['enable_fusion_activation'] = True
```

- [ ] **Step 2: Smoke-run the live app**

Run (native WSL):

```bash
source .venv/bin/activate && python -m halflife.main
```

Expected: app launches, simulates without NaNs/crashes; composites form, break, and the
sim stays stable for at least a minute. Watch for runaway fuse/scission churn (the
born-hot path) — if composites can't form at all, lower `activation_energy_scale`; if
nothing ever breaks, the energy gate / E_a may need tuning. Note observed behavior in the
commit message.

- [ ] **Step 3: Commit**

```bash
git add halflife/main.py
git commit -m "feat(main): live app opts into energy-gated fission + fusion activation

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Final verification

- [ ] **Full chemistry + new suite on CPU**

```bash
JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_chemistry.py tests/test_kinetic_reactions.py -v
```
Expected: all PASS.

- [ ] **Analysis pipeline unaffected** (fission events still emit correctly)

```bash
JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_analysis_events.py tests/test_analysis_pipeline.py -v
```
Expected: all PASS.

- [ ] **Update CLAUDE.md** — add a short "Kinetic-Energy-Coupled Reactions" subsection
  under the chemistry docs describing `endothermic_fission_mode`, the `KE_rel + Q ≥ 0`
  gate, the `KE − BE` energy convention, fusion activation, and born-hot bonds. Commit
  separately.

---

## Open items carried from the spec

- `activation_energy_scale` default (`1.0`) is a tuning seed — retune against observed
  live churn (Task 9 Step 2).
- Whether `forbid_endothermic_fission` survives as a deprecated alias or is removed
  (currently kept as a deprecated boolean that maps to `"forbid"`).
- `VREL_EPS = 1e-4` (Task 4) — the threshold below which the split kick falls back to the
  COM-position axis. Revisit if fragments with tiny but real relative motion misbehave.
- **Part B activation-gate model** — revisit with the user before Task 6 (see banner).
- Born-hot injection is scoped to free+free fusion (Task 8); extending to composite
  mergers needs the entity BEs threaded in — defer until/if needed.
