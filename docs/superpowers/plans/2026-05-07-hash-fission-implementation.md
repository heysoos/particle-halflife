# Hash-fission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strip polarity machinery and rewrite composite decay so the hash determines a binary partition of members into two product groups, committing the project to hash-chemistry maximalism.

**Architecture:** Six cleanup tasks remove polarity from the data model and force kernel without changing dynamics qualitatively. Then a pure helper `_hash_to_partition` is added (TDD), and `apply_composite_decay` is rewritten to call it: parent composite breaks into two products via hash-determined member assignment, particles conserved, species conserved. Dead code from earlier designs is deleted last.

**Tech Stack:** JAX (jit, vmap, scan, lax), pytest, ModernGL+pygame for the live render.

**Spec:** [docs/superpowers/specs/2026-05-07-hash-fission-design.md](../specs/2026-05-07-hash-fission-design.md)

---

## File structure

| File | Responsibility | Touched in tasks |
|---|---|---|
| `halflife/state.py` | Remove `polarity` from `InteractionParams`, `net_polarity` from `CompositeState`, `polarity_fusion_scale` & `polarity_stability_scale` from `PhysicsParams` and `initialize_physics_params` | A, C, D, E |
| `halflife/config.py` | Remove `polarity_fusion_scale`, `polarity_stability_scale` defaults; reduce `num_species` from 64 to 12 | E, G |
| `halflife/interactions.py` | Drop `attr_mod` parameters from `pairwise_force`, `compute_forces_for_particle`, `compute_all_forces`; force kernel becomes pure species-attraction | A |
| `halflife/step.py` | Remove `attr_mod` prep block before `compute_all_forces` call | A |
| `halflife/chemistry.py` | Drop polarity bonus on BE in `find_entity_partner`; drop neutrality boost & `net_polarity` write in `fusion_scan_body`; rewrite `apply_composite_decay` to use binary fission via `_hash_to_partition`; add `_hash_to_partition`; delete commented dead code | B, H, I, K |
| `halflife/profiler.py` | Drop `attr_mod`/`net_polarity` from forces measurement and `CompositeSnapshot` | A, D |
| `halflife/renderer.py` | Remove `pol fuse` and `pol stab` slider rows | F |
| `tests/test_chemistry.py` | Update `test_composite_half_life_valid` to not use `polarity_stability_scale`; add fission conservation test | B, I |
| `tests/test_performance.py` | Drop `attr_mod` arg from `compute_all_forces` calls | A |
| `tests/test_hash.py` | Add tests for `_hash_to_partition` | H |

---

## Conventions

- This repo runs Python natively in WSL. Activate the venv once per shell: `source .venv/bin/activate`.
- Tests need `PYTHONPATH=.` because there's no `pyproject.toml` or `setup.py` in the repo root.
- Git has no global identity in WSL; every commit must include `-c user.email='heysoos@local' -c user.name='Heysoos'`.
- Never use `git add -A` or `git add .` — there are untracked files (`.idea/`, `__pycache__/`, `bash.exe.stackdump`, `init_prompt.txt`) that must not be committed.

Baseline: before starting, verify the test suite passes:

```bash
source .venv/bin/activate
PYTHONPATH=. pytest tests/ -x --tb=short
```

Expected: 28 passed (or whatever the suite is at). If anything fails, fix it before starting; this plan assumes a green baseline.

---

## Task A: Drop `attr_mod` machinery from force kernel

**Files:**
- Modify: `halflife/interactions.py:31-198` (multiple call sites)
- Modify: `halflife/step.py:174-184`
- Modify: `halflife/profiler.py:295-340`
- Modify: `tests/test_performance.py:215-340`

- [ ] **Step 1: Run baseline test to confirm green**

```bash
source .venv/bin/activate
PYTHONPATH=. pytest tests/ -x --tb=short -q
```

Expected: all tests pass.

- [ ] **Step 2: Remove `attr_mod_i`/`attr_mod_j` parameters from `pairwise_force` in `halflife/interactions.py`**

Replace lines 77-124 of `halflife/interactions.py`:

```python
def pairwise_force(pos_i: jnp.ndarray, pos_j: jnp.ndarray,
                   species_i: jnp.ndarray, species_j: jnp.ndarray,
                   params: InteractionParams, config: SimConfig,
                   physics: PhysicsParams) -> jnp.ndarray:
    """
    Compute force on particle i due to particle j.

    Args:
        pos_i, pos_j:      (2,) float32 — positions
        species_i/j:       scalar int32  — species indices
        params:            InteractionParams
        config:            SimConfig (static)
        physics:           PhysicsParams (runtime-tunable)

    Returns:
        (2,) float32 — force vector on i (pointing away from j if repulsive)
    """
    # Displacement i ← j  (minimum image for periodic)
    d = pos_i - pos_j
    if config.boundary_mode == "periodic":
        d = d - config.world_width  * jnp.round(d[0] / config.world_width) * jnp.array([1., 0.])
        d = d - config.world_height * jnp.round(d[1] / config.world_height) * jnp.array([0., 1.])

    r = jnp.linalg.norm(d) + 1e-10
    d_hat = d / r

    # Look up species-pair parameters
    aij = params.attraction[species_i, species_j]
    r_a = params.peak_fraction[species_i, species_j] * config.interaction_radius
    r_c = params.cutoff_fraction[species_i, species_j] * config.interaction_radius

    f_mag = particle_life_force(r, aij * physics.attraction_scale,
                                physics.repulsion_radius, r_a,
                                r_c * physics.r_cutoff_scale,
                                physics.repulsion_strength)

    return -f_mag * d_hat
```

- [ ] **Step 3: Remove `attr_mod` from `compute_forces_for_particle` in `halflife/interactions.py`**

Replace lines 127-164:

```python
def compute_forces_for_particle(i: jnp.ndarray,
                                  positions: jnp.ndarray,
                                  species: jnp.ndarray,
                                  neighbors: jnp.ndarray,
                                  params: InteractionParams,
                                  config: SimConfig,
                                  physics: PhysicsParams) -> jnp.ndarray:
    """
    Net force on particle i from all its neighbors.
    """
    pos_i = positions[i]
    sp_i  = species[i]

    def force_from_neighbor(j):
        valid = (j >= 0)
        pos_j = jnp.where(valid, positions[j], pos_i)
        sp_j  = jnp.where(valid, species[j], sp_i)
        f = pairwise_force(pos_i, pos_j, sp_i, sp_j, params, config, physics)
        return jnp.where(valid, f, jnp.zeros(2))

    forces = jax.vmap(force_from_neighbor)(neighbors)
    return jnp.sum(forces, axis=0)
```

- [ ] **Step 4: Remove `attr_mod` from `compute_all_forces` in `halflife/interactions.py`**

Replace lines 167-198:

```python
def compute_all_forces(positions: jnp.ndarray,
                        species: jnp.ndarray,
                        neighbors: jnp.ndarray,
                        params: InteractionParams,
                        config: SimConfig,
                        physics: PhysicsParams) -> jnp.ndarray:
    """
    Compute net force for every particle simultaneously (outer vmap).
    """
    particle_indices = jnp.arange(config.num_particles, dtype=jnp.int32)

    def forces_for_i(i):
        return compute_forces_for_particle(
            i, positions, species, neighbors[i], params, config, physics
        )

    return jax.vmap(forces_for_i)(particle_indices)
```

- [ ] **Step 5: Remove `attr_mod` prep in `halflife/step.py:173-184`**

Replace those lines (the `# ── Phase 3: Force Computation ──` block) with:

```python
    # ── Phase 3: Force Computation ────────────────────────────────────────────
    forces = compute_all_forces(
        particles.position, particles.species, neighbors, params, config, physics
    )
```

- [ ] **Step 6: Remove `attr_mod` from `halflife/profiler.py`**

Open `halflife/profiler.py` and find the `attr_mod = jnp.where(is_comp, state.composites.net_polarity[safe_cid], 1.0)` line (around line 302). The block looks like:

```python
    is_comp = state.particles.composite_id >= 0
    safe_cid = jnp.clip(state.particles.composite_id, 0, config.max_composites - 1)
    attr_mod  = jnp.where(is_comp, state.composites.net_polarity[safe_cid], 1.0)
```

Delete those three lines entirely. Then update the two `compute_all_forces(...)` calls below them (around lines 315 and 338): remove the trailing `, attr_mod` argument from each call.

After edit, the calls should look like:

```python
    compute_all_forces(state.particles.position, state.particles.species,
                       nb_fixed, params, config, physics).block_until_ready()
```

and

```python
    compute_all_forces(state.particles.position, state.particles.species,
                       nb_fixed, params, config, physics)
```

- [ ] **Step 7: Remove `attr_mod` from `tests/test_performance.py`**

Find the two regions (around lines 215-227 and 295-340) that mirror the profiler. Delete the `attr_mod = jnp.ones(...)` and `attr_mod = jnp.where(is_comp, state.composites.net_polarity[safe_cid], 1.0)` lines and the trailing `, attr_mod` arg from `compute_all_forces` calls.

After edit, calls should look like:

```python
compute_all_forces(positions, species, nb, params, config, physics)
```

and similar for the JIT-warmup variants.

- [ ] **Step 8: Run tests to verify**

```bash
PYTHONPATH=. pytest tests/ -x --tb=short -q
```

Expected: all tests still pass. The dynamics now skip the polarity scaling entirely — composites generate full attraction forces.

- [ ] **Step 9: Smoke-run the sim for 5 seconds**

```bash
timeout 5 python -m halflife.main || true
```

Expected: window opens, composites form, no NaN/exception. Window closes when timeout fires.

- [ ] **Step 10: Commit**

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add \
  halflife/interactions.py halflife/step.py halflife/profiler.py tests/test_performance.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "refactor: drop attr_mod machinery from force kernel

Polarity-based attraction scaling removed in preparation for hash-fission
rebuild. Forces now use raw species-pair attraction with no per-particle
modifier; composites are no longer rendered inert by mean-polarity
neutralization."
```

---

## Task B: Drop polarity bonuses in fusion

**Files:**
- Modify: `halflife/chemistry.py:344-358` (polarity bonus in `find_entity_partner`)
- Modify: `halflife/chemistry.py:464-486` (neutrality boost & `net_polarity` calc in `fusion_scan_body`)
- Modify: `halflife/chemistry.py:540-551` (`net_polarity` write in `fusion_scan_body`)
- Modify: `tests/test_chemistry.py:142` (drop `polarity_stability_scale` from expected max half-life)

- [ ] **Step 1: Drop polarity bonus on `be_eff` in `find_entity_partner`**

In `halflife/chemistry.py`, find the block (around lines 344-363):

```python
            be = _hash_to_binding_energy(merged_h, config, physics)

            # Polarity bonus
            c_j = jnp.clip(particles.composite_id[j], 0, config.max_composites - 1)
            pi = jnp.where(
                particles.composite_id[i] < 0,
                params.polarity[particles.species[i]],
                composites.net_polarity[c_i]
            )
            pj = jnp.where(
                particles.composite_id[j] < 0,
                params.polarity[particles.species[j]],
                composites.net_polarity[c_j]
            )
            be_eff = be + physics.polarity_fusion_scale * (-pi * pj)

            # Size cap: don't grow beyond buffer
            would_overflow = (cnt_i + cnt_j) > M

            can_fuse = valid & in_range & (be_eff > physics.fusion_threshold) & ~would_overflow
```

Replace with:

```python
            be_eff = _hash_to_binding_energy(merged_h, config, physics)

            # Size cap: don't grow beyond buffer
            would_overflow = (cnt_i + cnt_j) > M

            can_fuse = valid & in_range & (be_eff > physics.fusion_threshold) & ~would_overflow
```

(Both the polarity lookup block and `be = ...; be_eff = be + ...` are gone; `be_eff` is now just the hash-derived binding energy.)

- [ ] **Step 2: Drop neutrality boost and `net_polarity` calc in `fusion_scan_body`**

In `halflife/chemistry.py`, find the block (around lines 464-486):

```python
        # Energy-based half-life: high binding energy → stable, low → unstable
        t = jnp.clip(
            (be - physics.fusion_threshold) / (1.0 - physics.fusion_threshold + 1e-8),
            0.0, 1.0
        )
        hl_base = config.half_life_min + (config.half_life_max - config.half_life_min) * t
        size_penalty = 1.0 + config.composite_size_decay_scale * jnp.maximum(
            0.0, mc.astype(jnp.float32) - 2.0
        )
        hl = hl_base / size_penalty

        # Mean polarity of merged entity
        pi = jnp.where(i_is_free,
                        params.polarity[particles.species[safe_i]],
                        composites_state.net_polarity[ci])
        pj = jnp.where(j_is_free,
                        params.polarity[particles.species[safe_j]],
                        composites_state.net_polarity[cj])
        cnt_i_scalar = jnp.where(i_is_free, jnp.int32(1), composites_state.member_count[ci])
        cnt_j_scalar = jnp.where(j_is_free, jnp.int32(1), composites_state.member_count[cj])
        net_pol = (pi * cnt_i_scalar.astype(jnp.float32) +
                   pj * cnt_j_scalar.astype(jnp.float32)) / (mc.astype(jnp.float32) + 1e-8)
        neutrality = 1.0 - jnp.abs(net_pol)
        hl_eff = hl * (1.0 + physics.polarity_stability_scale * neutrality)
```

Replace with:

```python
        # Energy-based half-life: high binding energy → stable, low → unstable
        t = jnp.clip(
            (be - physics.fusion_threshold) / (1.0 - physics.fusion_threshold + 1e-8),
            0.0, 1.0
        )
        hl_base = config.half_life_min + (config.half_life_max - config.half_life_min) * t
        size_penalty = 1.0 + config.composite_size_decay_scale * jnp.maximum(
            0.0, mc.astype(jnp.float32) - 2.0
        )
        hl_eff = hl_base / size_penalty
```

Note `be` no longer exists (we renamed to `be_eff` in Step 1). Update the `t = ...` line accordingly:

```python
        t = jnp.clip(
            (be_eff - physics.fusion_threshold) / (1.0 - physics.fusion_threshold + 1e-8),
            0.0, 1.0
        )
```

(And anywhere else `be` was a free variable inside `fusion_scan_body`, rename to `be_eff`. The variable was bound at line ~429 as `be = jnp.where(valid_i, all_be[safe_i], jnp.float32(0.0))` — rename that line and all its uses to `be_eff`.)

- [ ] **Step 3: Drop `net_polarity` write in `fusion_scan_body`**

Find the block (around lines 540-552):

```python
        new_comp_net_pol = composites_state.net_polarity.at[safe_target].set(
            jnp.where(can_fuse, net_pol, composites_state.net_polarity[safe_target])
        )

        new_composites = composites_state._replace(
            members=new_members,
            alive=new_comp_alive,
            binding_energy=new_comp_be,
            half_life=new_comp_hl,
            member_count=new_comp_count_arr,
            species_hash=new_comp_hash,
            net_polarity=new_comp_net_pol,
        )
```

Delete the `new_comp_net_pol = ...` lines and the `net_polarity=new_comp_net_pol` field from the `_replace`. Result:

```python
        new_composites = composites_state._replace(
            members=new_members,
            alive=new_comp_alive,
            binding_energy=new_comp_be,
            half_life=new_comp_hl,
            member_count=new_comp_count_arr,
            species_hash=new_comp_hash,
        )
```

- [ ] **Step 4: Drop `polarity_stability_scale` from expected max half-life in `tests/test_chemistry.py`**

In `tests/test_chemistry.py`, find around line 142:

```python
    max_expected_hl = (
        _config.half_life_max
        * (1 + _config.polarity_stability_scale)  # polarity bonus
        * 2.0  # generous margin
    )
```

Replace with:

```python
    max_expected_hl = _config.half_life_max * 2.0  # generous margin
```

- [ ] **Step 5: Run tests**

```bash
PYTHONPATH=. pytest tests/ -x --tb=short -q
```

Expected: all tests pass. Composites now have BE-only-driven half-life (no neutrality bonus).

- [ ] **Step 6: Smoke-run the sim**

```bash
timeout 5 python -m halflife.main || true
```

Expected: composites form, no exceptions.

- [ ] **Step 7: Commit**

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add \
  halflife/chemistry.py tests/test_chemistry.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "refactor: drop polarity bonuses in fusion BE and half-life

Composite binding energy is now purely the hash-derived value — no ionic
bonding bonus from species polarity. Half-life is BE-driven only — no
neutrality boost on neutral composites. Both feedback loops contributed
to the inert-blob steady state."
```

---

## Task C: Drop `polarity` field from `InteractionParams`

**Files:**
- Modify: `halflife/state.py:154-221`

- [ ] **Step 1: Remove the `polarity` field from `InteractionParams` NamedTuple**

In `halflife/state.py:154-171`, change the class to:

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
```

(Drop the `polarity:` line.)

- [ ] **Step 2: Remove polarity sampling and field from `initialize_interaction_params`**

In `halflife/state.py:174-221`, find:

```python
    key = jax.random.PRNGKey(seed)
    S = config.num_species
    key, k1, k2, k3, k4 = jax.random.split(key, 5)
```

Change `5` → `4` (we no longer need a 4th subkey for polarity):

```python
    key = jax.random.PRNGKey(seed)
    S = config.num_species
    key, k1, k2, k3 = jax.random.split(key, 4)
```

Find and delete:

```python
    # Per-species polarity charge: uniform in [-1, 1]
    polarity = jax.random.uniform(k4, (S,), minval=-1.0, maxval=1.0)
```

In the `return InteractionParams(...)`, delete the `polarity=polarity,` line. Result:

```python
    return InteractionParams(
        attraction=attraction,
        peak_fraction=peak_fraction,
        cutoff_fraction=cutoff_fraction,
    )
```

Also update the docstring lines mentioning polarity (around line 184): rewrite to drop the polarity sentence. Result:

```python
    """
    Initialize random interaction parameters.

    The attraction matrix is the primary knob for rich behaviour:
    - Asymmetric values (A attracts B but B repels A) produce chasing dynamics
    - Symmetric negative values produce clustering
    - Symmetric positive values produce avoidance

    peak_fraction and cutoff_fraction are sampled per-species-pair so each
    pair gets its own force-shape (peak position + range), not just amplitude.
    The fractions are sampled in [0.3, 0.95] then sorted per-pair to enforce
    peak < cutoff with at least a 0.1 gap.

    Args:
        config: SimConfig
        seed:   random seed for the interaction matrix
    Returns:
        InteractionParams
    """
```

- [ ] **Step 3: Run tests**

```bash
PYTHONPATH=. pytest tests/ -x --tb=short -q
```

Expected: all pass. After Tasks A and B, no caller reads `params.polarity` anymore.

- [ ] **Step 4: Commit**

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add halflife/state.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "refactor: drop polarity field from InteractionParams"
```

---

## Task D: Drop `net_polarity` from `CompositeState` and profiler

**Files:**
- Modify: `halflife/state.py:41-57` and `:127-138` (CompositeState definition + init)
- Modify: `halflife/profiler.py:50-65` (CompositeSnapshot dataclass)

- [ ] **Step 1: Remove `net_polarity` from `CompositeState` NamedTuple**

In `halflife/state.py:41-57`, delete the `net_polarity` line. Result:

```python
class CompositeState(NamedTuple):
    """
    State of the composite pool. Leading dimension is MAX_COMPOSITES.
    A composite is a group of bonded particles.

    members[c, :member_count[c]] are the particle indices belonging to composite c.
    slots beyond member_count are padded with -1.
    """
    members:        jnp.ndarray  # (C, M) int32   — particle indices per composite
    member_count:   jnp.ndarray  # (C,)   int32   — number of valid members
    alive:          jnp.ndarray  # (C,)   bool    — active composite mask
    binding_energy: jnp.ndarray  # (C,)   float32 — energy released on formation
    half_life:      jnp.ndarray  # (C,)   float32 — composite decay half-life
    age:            jnp.ndarray  # (C,)   float32 — time since formation
    species_hash:   jnp.ndarray  # (C,)   uint32  — hash of sorted member species
```

- [ ] **Step 2: Remove `net_polarity` from `initialize_world` composite init**

In `halflife/state.py:127-138`, the `composites = CompositeState(...)` block. Delete the `net_polarity=jnp.zeros(C, dtype=jnp.float32),` line. Result:

```python
    composites = CompositeState(
        members=jnp.full((C, M), -1, dtype=jnp.int32),
        member_count=jnp.zeros(C, dtype=jnp.int32),
        alive=jnp.zeros(C, dtype=bool),
        binding_energy=jnp.zeros(C, dtype=jnp.float32),
        half_life=jnp.zeros(C, dtype=jnp.float32),
        age=jnp.zeros(C, dtype=jnp.float32),
        species_hash=jnp.zeros(C, dtype=jnp.uint32),
    )
```

- [ ] **Step 3: Remove `net_polarity` field from `CompositeSnapshot` in `halflife/profiler.py`**

In `halflife/profiler.py`, find the `@dataclass class CompositeSnapshot` (around line 50). Remove the `net_polarity: float` field. Where the snapshot is constructed (search for `net_polarity=` in the file), delete that argument too.

If the snapshot is rendered to text/HUD output anywhere (search for `.net_polarity`), drop those references.

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=. pytest tests/ -x --tb=short -q
```

Expected: all pass.

- [ ] **Step 5: Smoke-run the sim**

```bash
timeout 5 python -m halflife.main || true
```

Expected: no exceptions.

- [ ] **Step 6: Commit**

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add \
  halflife/state.py halflife/profiler.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "refactor: drop net_polarity from CompositeState and profiler"
```

---

## Task E: Drop `polarity_*_scale` from `PhysicsParams` and `SimConfig`

**Files:**
- Modify: `halflife/state.py:226-259` (PhysicsParams + initialize_physics_params)
- Modify: `halflife/config.py:94-98`

- [ ] **Step 1: Remove `polarity_fusion_scale` and `polarity_stability_scale` from `PhysicsParams`**

In `halflife/state.py:226-243`, the `PhysicsParams` NamedTuple. Delete the two polarity scalar lines. Result:

```python
class PhysicsParams(NamedTuple):
    """
    Physics scalars that can be adjusted at runtime without recompiling.
    Passed as a regular JAX argument (not static), so slider changes take
    effect on the next simulation step without triggering recompilation.
    """
    damping:                  jnp.ndarray  # () float32 — velocity damping per step
    repulsion_strength:       jnp.ndarray  # () float32 — hard-core repulsion magnitude
    fusion_threshold:         jnp.ndarray  # () float32 — min binding energy to fuse [0,1]
    binding_energy_scale:     jnp.ndarray  # () float32 — energy released on fusion
    repulsion_radius:         jnp.ndarray  # () float32 — inner hard-core repulsion radius
    r_cutoff_scale:           jnp.ndarray  # () float32 — multiplier on per-species r_cutoff
    spring_k:                 jnp.ndarray  # () float32 — composite COM-spring stiffness
    attraction_scale:         jnp.ndarray  # () float32 — global attraction magnitude multiplier
    dt:                       jnp.ndarray  # () float32 — integration timestep
```

- [ ] **Step 2: Remove polarity scalars from `initialize_physics_params`**

In `halflife/state.py:245-259`, the `initialize_physics_params` body. Delete the two `polarity_*_scale=...` lines. Result:

```python
def initialize_physics_params(config: SimConfig) -> PhysicsParams:
    """Create PhysicsParams from SimConfig defaults."""
    return PhysicsParams(
        damping=jnp.float32(config.damping),
        repulsion_strength=jnp.float32(config.repulsion_strength),
        fusion_threshold=jnp.float32(config.fusion_threshold),
        binding_energy_scale=jnp.float32(config.binding_energy_scale),
        repulsion_radius=jnp.float32(config.repulsion_radius),
        r_cutoff_scale=jnp.float32(1.0),
        spring_k=jnp.float32(50.0),
        attraction_scale=jnp.float32(1.0),
        dt=jnp.float32(config.dt),
    )
```

- [ ] **Step 3: Remove polarity defaults from `SimConfig`**

In `halflife/config.py:94-98`, find:

```python
    # ── Polarity Chemistry ────────────────────────────────────────────────────
    # Each species has a signed polarity charge p[s] ∈ [-1, 1].
    # Opposite polarities fuse more readily; neutral composites live longer.
    polarity_fusion_scale:      float = 0.3   # bonus/penalty to binding energy
    polarity_stability_scale:   float = 0.5   # neutrality boost to composite half-life
```

Delete the entire block (header comment + both fields).

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=. pytest tests/ -x --tb=short -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add \
  halflife/state.py halflife/config.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "refactor: drop polarity_*_scale from PhysicsParams and SimConfig"
```

---

## Task F: Remove polarity sliders from UI

**Files:**
- Modify: `halflife/renderer.py:411-412`

- [ ] **Step 1: Delete the two polarity slider rows**

In `halflife/renderer.py`, find the `slider_specs` list (around line 400). Delete these two lines:

```python
            ("polarity_fusion_scale",    "pol fuse",    0.3,   "{:.3f}", None),
            ("polarity_stability_scale", "pol stab",    0.5,   "{:.3f}", None),
```

The list now skips from `bind energy` directly to the `None` group separator.

- [ ] **Step 2: Smoke-run the sim**

```bash
timeout 5 python -m halflife.main || true
```

Expected: window opens, slider panel renders without `pol fuse`/`pol stab` rows, no errors.

- [ ] **Step 3: Run tests**

```bash
PYTHONPATH=. pytest tests/ -x --tb=short -q
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add halflife/renderer.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "ui: remove polarity sliders from params panel"
```

---

## Task G: Reduce `num_species` default to 12

**Files:**
- Modify: `halflife/config.py:31`

- [ ] **Step 1: Change `num_species` default**

In `halflife/config.py`, find:

```python
    num_species: int = 64           # number of distinct particle types
```

Replace with:

```python
    num_species: int = 12           # number of distinct particle types
```

- [ ] **Step 2: Run tests**

```bash
PYTHONPATH=. pytest tests/ -x --tb=short -q
```

Expected: all pass. Tests that build `SimConfig()` directly will now use 12 species; this should not break anything since species count only affects shapes that scale with it.

- [ ] **Step 3: Smoke-run the sim**

```bash
timeout 8 python -m halflife.main || true
```

Expected: window opens with visibly fewer color classes; composites form normally.

- [ ] **Step 4: Commit**

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add halflife/config.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "config: reduce num_species 64→12 for hash-fission rebuild

Smaller species count makes the reaction graph small enough to study —
~150 reachable composite types instead of ~4000 — and lets PL force
kernel signal show through the local average instead of washing out."
```

---

## Task H: Implement `_hash_to_partition` (TDD)

**Files:**
- Create: nothing — implementation lives in `halflife/chemistry.py`
- Modify: `halflife/chemistry.py` (add new helper near top, after `_hash_to_binding_energy`)
- Test: `tests/test_hash.py`

The function determines a binary partition of composite members into two product groups:

- For each member slot `i ∈ [0, n_members)`, derive a sort key from the hash and slot index.
- Sort slots by their sort key.
- Use a hash-derived pivot `p ∈ [1, n_members - 1]` to split the sorted slots into product 0 (first `p`) and product 1 (rest).
- Assignment is recorded back into a `(max_composite_size,)` array indexed by the original member slot.
- Slots `i >= n_members` get `-1`.

Both products are guaranteed non-empty because `pivot ∈ [1, n_members - 1]`.

- [ ] **Step 1: Write the failing test in `tests/test_hash.py`**

Append to `tests/test_hash.py`:

```python
# ── Tests for _hash_to_partition ─────────────────────────────────────────────

def test_partition_deterministic():
    """Same (h, n_members) must produce the same assignment every call."""
    from halflife.chemistry import _hash_to_partition
    config = SimConfig()
    h = jnp.uint32(123_456_789)
    n = jnp.int32(5)
    a1 = _hash_to_partition(h, n, config)
    a2 = _hash_to_partition(h, n, config)
    assert jnp.all(a1 == a2), f"non-deterministic: {a1} vs {a2}"


def test_partition_assignments_in_valid_range():
    """For valid slots i < n_members, assignment[i] in {0, 1}; else -1."""
    from halflife.chemistry import _hash_to_partition
    config = SimConfig()
    M = config.max_composite_size
    for h_val in [1, 100, 10_000, 999_999, 2**31 - 1]:
        for n_val in [2, 3, 5, 8, 16]:
            h = jnp.uint32(h_val)
            n = jnp.int32(n_val)
            a = jnp.asarray(_hash_to_partition(h, n, config))
            assert a.shape == (M,)
            for i in range(M):
                if i < n_val:
                    assert a[i] in (0, 1), f"slot {i} of n={n_val}, h={h_val}: got {a[i]}"
                else:
                    assert a[i] == -1, f"padding slot {i}: got {a[i]}"


def test_partition_both_products_nonempty():
    """For any (h, n>=2), both products must have >=1 member assigned."""
    from halflife.chemistry import _hash_to_partition
    config = SimConfig()
    for h_val in [0, 1, 100, 99_999, 2_654_435_761]:
        for n_val in [2, 3, 4, 5, 8, 16, 32]:
            h = jnp.uint32(h_val)
            n = jnp.int32(n_val)
            a = jnp.asarray(_hash_to_partition(h, n, config))
            valid = a[:n_val]
            count_0 = int(jnp.sum(valid == 0))
            count_1 = int(jnp.sum(valid == 1))
            assert count_0 + count_1 == n_val, f"missing members for n={n_val}, h={h_val}"
            assert count_0 >= 1, f"product 0 empty for n={n_val}, h={h_val}"
            assert count_1 >= 1, f"product 1 empty for n={n_val}, h={h_val}"


def test_partition_distribution_varies_with_hash():
    """Different hash values must produce different partition shapes (sometimes)."""
    from halflife.chemistry import _hash_to_partition
    config = SimConfig()
    n = jnp.int32(8)
    pivots_seen = set()
    for h_val in range(20):
        a = jnp.asarray(_hash_to_partition(jnp.uint32(h_val * 1_000_003), n, config))
        valid = a[:8]
        pivot = int(jnp.sum(valid == 0))
        pivots_seen.add(pivot)
    assert len(pivots_seen) >= 4, f"too few pivot values across 20 hashes: {pivots_seen}"
```

- [ ] **Step 2: Run the new tests and confirm they fail**

```bash
PYTHONPATH=. pytest tests/test_hash.py::test_partition_deterministic tests/test_hash.py::test_partition_assignments_in_valid_range tests/test_hash.py::test_partition_both_products_nonempty tests/test_hash.py::test_partition_distribution_varies_with_hash -v
```

Expected: all four FAIL with `ImportError: cannot import name '_hash_to_partition' from 'halflife.chemistry'`.

- [ ] **Step 3: Implement `_hash_to_partition` in `halflife/chemistry.py`**

In `halflife/chemistry.py`, just below `_hash_to_binding_energy` (around line 95, before the dead `_hash_to_decay_products` comment block), add:

```python
def _hash_to_partition(h: jnp.ndarray, n_members: jnp.ndarray,
                       config: SimConfig) -> jnp.ndarray:
    """
    Determine a binary partition of composite members for fission.

    For each member slot i ∈ [0, n_members), compute a sort key from
    hash_mix(h, i) and rank slots by that key. The first `pivot` slots
    in sorted order go to product 0, the rest to product 1. Both products
    are guaranteed non-empty because pivot ∈ [1, n_members - 1].

    Args:
        h:          scalar uint32 — composite's species hash
        n_members:  scalar int32 — number of valid members (>= 2)
        config:     SimConfig (static)

    Returns:
        assignment: (max_composite_size,) int32 — values in {0, 1} for slots
                    i < n_members, else -1.
    """
    M = config.max_composite_size

    # Per-slot sort key: mix h with slot index using Fibonacci-style hash.
    # Pure JAX, JIT-safe, deterministic.
    slot_idx = jnp.arange(M, dtype=jnp.uint32)
    sort_keys = (h.astype(jnp.uint32) * jnp.uint32(2_654_435_761)
                 + slot_idx * jnp.uint32(1_000_003))
    # Mark padding slots (i >= n_members) with max key so they sort last.
    sort_keys = jnp.where(
        jnp.arange(M, dtype=jnp.int32) < n_members,
        sort_keys,
        jnp.uint32(0xFFFFFFFF),
    )

    # Argsort: order[k] = slot whose sort_key is k-th smallest.
    order = jnp.argsort(sort_keys)  # (M,)

    # Pivot in [1, n_members - 1] from a different region of the hash.
    # n_members >= 2 guarantees this range is non-empty.
    pivot = jnp.int32(1) + (
        ((h >> jnp.uint32(12)).astype(jnp.int32)) % jnp.maximum(n_members - 1, jnp.int32(1))
    )

    # In sorted order, assign first `pivot` to product 0, rest to product 1.
    sorted_assignment = jnp.where(
        jnp.arange(M, dtype=jnp.int32) < pivot,
        jnp.int32(0),
        jnp.int32(1),
    )
    # Mark padding slots (rank >= n_members in sorted order) with -1.
    sorted_assignment = jnp.where(
        jnp.arange(M, dtype=jnp.int32) < n_members,
        sorted_assignment,
        jnp.int32(-1),
    )

    # Scatter back to original slot order: assignment[order[k]] = sorted_assignment[k].
    assignment = jnp.full(M, -1, dtype=jnp.int32).at[order].set(sorted_assignment)
    return assignment
```

- [ ] **Step 4: Run the tests and confirm they pass**

```bash
PYTHONPATH=. pytest tests/test_hash.py::test_partition_deterministic tests/test_hash.py::test_partition_assignments_in_valid_range tests/test_hash.py::test_partition_both_products_nonempty tests/test_hash.py::test_partition_distribution_varies_with_hash -v
```

Expected: all four PASS.

- [ ] **Step 5: Run full test suite to confirm no regression**

```bash
PYTHONPATH=. pytest tests/ -x --tb=short -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add \
  halflife/chemistry.py tests/test_hash.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "feat(chemistry): add _hash_to_partition for binary fission

Pure JAX-jitted helper that determines, for a composite of n members
and species hash h, which members go to product 0 vs product 1 in a
binary fission. Uses Fibonacci-style hash mixing to derive per-slot
sort keys, argsorts them, and splits at a hash-derived pivot in
[1, n-1]. Both products are guaranteed non-empty.

This is the central primitive for the hash-fission rebuild — composite
decay paths are now fully deterministic from the species multiset
(Sayama-style)."
```

---

## Task I: Refactor `apply_composite_decay` for binary fission (TDD)

**Files:**
- Modify: `halflife/chemistry.py:143-256` (the entire `apply_composite_decay` function)
- Test: `tests/test_chemistry.py` (new fission test)

The new fission flow: for each composite that decays this step, partition its members into two product groups via `_hash_to_partition`. Each product becomes either a free particle (product size 1) or a new composite (product size ≥ 2). The parent composite's slot becomes product 0; product 1 takes a fresh free slot. Each product's COM gets a velocity kick along the COM-vs-COM axis (momentum-conserving, equal-and-opposite). Particles never change species.

- [ ] **Step 1: Write the failing fission conservation test**

Append to `tests/test_chemistry.py`:

```python
def test_fission_conserves_particles_and_species():
    """
    Run the sim with very short half-life so composites decay aggressively.
    Total particle count and per-species counts must be exactly preserved.
    """
    config = SimConfig(
        half_life_min=5.0,
        half_life_max=20.0,
        num_particles=500,
    )
    state = initialize_world(config, seed=0)
    params = initialize_interaction_params(config, seed=42)
    physics = initialize_physics_params(config)
    step_fn = jax.jit(simulation_step, static_argnums=(2,))
    state = step_fn(state, params, config, physics)  # warm-up

    initial_species = jnp.asarray(state.particles.species)
    initial_count = config.num_particles
    initial_per_species = jnp.bincount(initial_species, length=config.num_species)

    for s in range(800):
        state = step_fn(state, params, config, physics)

    final_species = jnp.asarray(state.particles.species)
    final_count = state.particles.position.shape[0]
    final_per_species = jnp.bincount(final_species, length=config.num_species)

    assert final_count == initial_count, (
        f"particle count not conserved: {initial_count} → {final_count}"
    )
    assert jnp.all(initial_species == final_species), (
        "particle species changed — fission must not transmute"
    )
    assert jnp.all(initial_per_species == final_per_species), (
        f"per-species counts changed:\n  initial={initial_per_species}\n  final={final_per_species}"
    )


def test_fission_produces_two_products():
    """
    Run with short half-life and check that some composites have produced
    fission products of size 1 (free particle) AND size 2+ (new composite),
    indicating binary partitioning is actually splitting members.
    """
    config = SimConfig(
        half_life_min=5.0,
        half_life_max=20.0,
        num_particles=500,
    )
    state = initialize_world(config, seed=0)
    params = initialize_interaction_params(config, seed=42)
    physics = initialize_physics_params(config)
    step_fn = jax.jit(simulation_step, static_argnums=(2,))

    state = step_fn(state, params, config, physics)
    sizes_seen = set()
    for s in range(800):
        state = step_fn(state, params, config, physics)
        alive = jnp.asarray(state.composites.alive)
        mc = jnp.asarray(state.composites.member_count)
        for size in mc[alive].tolist():
            sizes_seen.add(int(size))

    # We must observe size-2 composites at minimum (from fission of size-3+).
    # Size-3+ composites should also occur (from fusion or fission of size-5+).
    assert 2 in sizes_seen, f"never saw size-2 composites in 800 steps: {sorted(sizes_seen)}"
    assert max(sizes_seen) >= 3, f"never saw size-3+ composites: {sorted(sizes_seen)}"
```

- [ ] **Step 2: Run the new tests and confirm `test_fission_conserves_particles_and_species` may pass already (current decay also conserves), but `test_fission_produces_two_products` should pass too**

```bash
PYTHONPATH=. pytest tests/test_chemistry.py::test_fission_conserves_particles_and_species tests/test_chemistry.py::test_fission_produces_two_products -v
```

Expected: probably both PASS even with the existing "release everything as free" code, because fusion will reform composites of various sizes. **This is a guard test** — its job is to not regress when we change the fission code in Step 4.

- [ ] **Step 3: Add a more discriminating test that catches the rewrite**

Append to `tests/test_chemistry.py`:

```python
def test_fission_creates_intermediate_size_products():
    """
    With binary fission, a size-5 composite should split into products
    of sizes (1,4), (2,3), (3,2), or (4,1). This produces composites at
    sizes 2, 3, 4 that wouldn't easily form purely through fusion in the
    same time window.

    With the OLD `release everything as free` decay, a size-5 composite
    fully dissociates to 5 free particles, and intermediate sizes would
    only re-form through subsequent fusion (slow). With NEW binary fission,
    intermediate sizes appear immediately.

    Counts the number of size-3 composite-instances observed across many
    steps — under binary fission this should exceed a threshold that the
    old behaviour wouldn't reach as quickly.
    """
    config = SimConfig(
        half_life_min=10.0,    # short enough to see fission
        half_life_max=30.0,
        fusion_threshold=0.4,  # high enough to make spontaneous size-3 fusion rare
        num_particles=500,
    )
    state = initialize_world(config, seed=0)
    params = initialize_interaction_params(config, seed=42)
    physics = initialize_physics_params(config)
    step_fn = jax.jit(simulation_step, static_argnums=(2,))
    state = step_fn(state, params, config, physics)

    size_3_instances = 0
    for s in range(500):
        state = step_fn(state, params, config, physics)
        alive = jnp.asarray(state.composites.alive)
        mc = jnp.asarray(state.composites.member_count)
        size_3_instances += int(jnp.sum((alive) & (mc == 3)))

    # With binary fission, expect plenty of size-3 instances — at least 50
    # observations across 500 steps. Old "release all" decay would struggle
    # to maintain this many because all decays produce free particles only.
    assert size_3_instances >= 50, (
        f"too few size-3 composite-instances observed: {size_3_instances} "
        "(binary fission should produce these readily)"
    )
```

- [ ] **Step 4: Run the new test and observe it fails under current code**

```bash
PYTHONPATH=. pytest tests/test_chemistry.py::test_fission_creates_intermediate_size_products -v
```

Expected: **FAIL** with current "release everything as free" decay. (If by chance it passes, that means existing dynamics already produce enough size-3 composites at this config — adjust fusion_threshold higher and re-run until it fails.)

- [ ] **Step 5: Rewrite `apply_composite_decay` in `halflife/chemistry.py`**

Replace the entire body of `apply_composite_decay` (lines 145-256) with:

```python
def apply_composite_decay(state: WorldState, config: SimConfig,
                           physics: PhysicsParams) -> WorldState:
    """
    Apply binary fission decay to all alive composites.

    A decaying composite is partitioned into two products by _hash_to_partition.
    Product 0 reuses the parent's composite slot. Product 1 takes a fresh free
    slot from the composite pool. Products of size 1 become free particles;
    products of size ≥ 2 become new composites with hash-derived properties.

    Particle species are never modified — only their composite_id and velocity.

    Energy: parent.binding_energy * (1 - fission_cost) is split equally between
    the two products as kinetic energy, applied as a momentum-conserving kick
    along the COM-vs-COM axis (product 0 → +direction, product 1 → -direction).
    Each product's members all get the same kick (the product moves as a unit).

    Args:
        state:   WorldState
        config:  SimConfig (static)
        physics: PhysicsParams — provides dt for the per-step decay probability

    Returns:
        Updated WorldState
    """
    particles = state.particles
    composites = state.composites
    key, subkey = jax.random.split(state.rng_key)
    N = config.num_particles
    M = config.max_composite_size
    C = config.max_composites

    # ── Roll for which composites decay this step ───────────────────────────
    rand = jax.random.uniform(subkey, (C,))
    ln2 = jnp.log(2.0)
    decay_prob = 1.0 - jnp.exp(-physics.dt * ln2 / (composites.half_life + 1e-10))
    fissions = composites.alive & (rand < decay_prob)  # (C,) bool

    # Pre-allocate fresh composite slots for product 1 of each fissioning composite.
    # find_free_slots returns a (C,) array — we'll use the first
    # max_fissions_per_step entries as needed via cumsum.
    free_slots = find_free_slots(composites.alive, C)  # (C,) int32

    # Number of fissions occurring this step (Python-int via array op).
    # We assign each fissioning composite a "fission rank" via cumsum so it
    # picks free_slots[rank] as its product-1 target.
    fission_rank = jnp.cumsum(fissions.astype(jnp.int32)) - 1  # (C,) — -1 for non-fissioning

    # ── Per-composite: compute partition assignment ────────────────────────
    def per_composite(c):
        """Returns assignment[(M,)], com_0[(2,)], com_1[(2,)], target_slot_for_p1."""
        n = composites.member_count[c]
        h = composites.species_hash[c]
        assignment = _hash_to_partition(h, n, config)  # (M,) ∈ {-1, 0, 1}

        # Compute each product's COM (using min-image displacement from member 0).
        member_ids = composites.members[c]  # (M,)
        safe_ids = jnp.where(member_ids >= 0, member_ids, 0)
        valid = (member_ids >= 0) & (jnp.arange(M) < n)
        ref = particles.position[safe_ids[0]]

        def disp_from_ref(idx):
            d = particles.position[safe_ids[idx]] - ref
            if config.boundary_mode == "periodic":
                d = d - config.world_width  * jnp.round(d[0] / config.world_width)  * jnp.array([1., 0.])
                d = d - config.world_height * jnp.round(d[1] / config.world_height) * jnp.array([0., 1.])
            return d

        rels = jax.vmap(disp_from_ref)(jnp.arange(M))  # (M, 2)

        in_p0 = valid & (assignment == 0)
        in_p1 = valid & (assignment == 1)
        n0 = jnp.sum(in_p0.astype(jnp.float32))
        n1 = jnp.sum(in_p1.astype(jnp.float32))
        com0 = ref + jnp.sum(rels * in_p0[:, None].astype(jnp.float32), axis=0) / (n0 + 1e-8)
        com1 = ref + jnp.sum(rels * in_p1[:, None].astype(jnp.float32), axis=0) / (n1 + 1e-8)

        # Target slot for product 1 (looked up via fission_rank → free_slots).
        rank = fission_rank[c]
        target_p1 = free_slots[jnp.clip(rank, 0, C - 1)]

        return assignment, com0, com1, target_p1, n0.astype(jnp.int32), n1.astype(jnp.int32)

    all_assignment, all_com0, all_com1, all_target_p1, all_n0, all_n1 = jax.vmap(per_composite)(
        jnp.arange(C, dtype=jnp.int32)
    )
    # Shapes: (C, M), (C, 2), (C, 2), (C,), (C,), (C,)

    # ── Update each member particle's composite_id and velocity ────────────
    # For each member of a fissioning composite, route it to either:
    #   product 0 → keeps composite_id = c (parent slot reused)
    #   product 1 → new composite_id = all_target_p1[c]
    #   if its product has only 1 member: composite_id = -1 (free)
    #
    # The kick direction:
    #   product 0 members: along (com0 - com1) normalized
    #   product 1 members: along (com1 - com0) normalized = opposite
    # Magnitude per product: sqrt(2 * E_per_product / M_product) where
    #   E_per_product = parent.binding_energy * (1 - fission_cost) / 2

    def per_member(c, m):
        """Returns (pid, valid, new_cid, kick) for member slot m of composite c."""
        does_fission = fissions[c]
        n = composites.member_count[c]
        member_id = composites.members[c, m]
        valid = does_fission & (m < n) & (member_id >= 0)

        a = all_assignment[c, m]  # 0, 1, or -1
        com0 = all_com0[c]
        com1 = all_com1[c]
        n0 = all_n0[c]
        n1 = all_n1[c]
        target_p1 = all_target_p1[c]

        # Direction along COM-COM axis (min-image).
        d = com0 - com1
        if config.boundary_mode == "periodic":
            d = d - config.world_width  * jnp.round(d[0] / config.world_width)  * jnp.array([1., 0.])
            d = d - config.world_height * jnp.round(d[1] / config.world_height) * jnp.array([0., 1.])
        d_norm = jnp.linalg.norm(d) + 1e-8
        d_hat = d / d_norm

        # Energy split: half to each product.
        e_per = composites.binding_energy[c] * (1.0 - config.fission_cost) * 0.5

        # Mass per product (assumes mass=1 per particle, which is the convention here).
        mass_p0 = n0.astype(jnp.float32)
        mass_p1 = n1.astype(jnp.float32)
        v0 = jnp.sqrt(jnp.maximum(0.0, 2.0 * e_per / (mass_p0 + 1e-8)))
        v1 = jnp.sqrt(jnp.maximum(0.0, 2.0 * e_per / (mass_p1 + 1e-8)))

        # Kick: product 0 → +d_hat * v0, product 1 → -d_hat * v1.
        kick = jnp.where(
            a == 0,
            d_hat * v0,
            jnp.where(a == 1, -d_hat * v1, jnp.zeros(2)),
        )

        # New composite_id:
        #   a==0 and n0>=2 → c (reuse parent slot)
        #   a==0 and n0==1 → -1 (free)
        #   a==1 and n1>=2 → target_p1
        #   a==1 and n1==1 → -1 (free)
        new_cid = jnp.where(
            a == 0,
            jnp.where(n0 >= 2, c, jnp.int32(-1)),
            jnp.where(a == 1,
                      jnp.where(n1 >= 2, target_p1, jnp.int32(-1)),
                      particles.composite_id[jnp.where(member_id >= 0, member_id, 0)]),
        )

        return member_id, valid, new_cid, kick

    # Vmap over (C, M) grid.
    pid_grid, valid_grid, cid_grid, kick_grid = jax.vmap(
        lambda c: jax.vmap(lambda m: per_member(c, m))(jnp.arange(M, dtype=jnp.int32))
    )(jnp.arange(C, dtype=jnp.int32))
    # Shapes: (C, M), (C, M), (C, M), (C, M, 2)

    flat_pid   = pid_grid.reshape(-1)        # (C*M,)
    flat_valid = valid_grid.reshape(-1)
    flat_cid   = cid_grid.reshape(-1)
    flat_kick  = kick_grid.reshape(-1, 2)

    # Route invalid entries to OOB index N (dropped) — see notes in
    # apply_composite_decay's git history about the index-0 race.
    drop_pids = jnp.where(flat_valid, flat_pid, N)
    new_composite_id = particles.composite_id.at[drop_pids].set(flat_cid, mode='drop')

    # Velocity adds — duplicates accumulate, invalid entries add 0, so safe form is fine.
    safe_pids = jnp.where(flat_valid, flat_pid, 0)
    new_velocity = particles.velocity.at[safe_pids].add(
        jnp.where(flat_valid[:, None], flat_kick, 0.0)
    )

    # ── Update CompositeState: parent slot becomes product 0; target_p1 slot becomes product 1 ──
    # Compute per-product member arrays and species hashes.
    def per_product(c):
        """Returns members_p0, count_p0, hash_p0, members_p1, count_p1, hash_p1."""
        does_fission = fissions[c]
        assignment = all_assignment[c]
        member_ids = composites.members[c]
        n = composites.member_count[c]

        # Compact members of each product to front using cumsum (same trick as fusion).
        in_p0 = (assignment == 0) & (member_ids >= 0) & (jnp.arange(M) < n)
        in_p1 = (assignment == 1) & (member_ids >= 0) & (jnp.arange(M) < n)

        pos_p0 = jnp.cumsum(in_p0.astype(jnp.int32)) - 1
        out_pos_p0 = jnp.where(in_p0, pos_p0, M)
        members_p0 = jnp.full(M, -1, dtype=jnp.int32).at[out_pos_p0].set(member_ids, mode='drop')
        count_p0 = jnp.sum(in_p0.astype(jnp.int32))

        pos_p1 = jnp.cumsum(in_p1.astype(jnp.int32)) - 1
        out_pos_p1 = jnp.where(in_p1, pos_p1, M)
        members_p1 = jnp.full(M, -1, dtype=jnp.int32).at[out_pos_p1].set(member_ids, mode='drop')
        count_p1 = jnp.sum(in_p1.astype(jnp.int32))

        # Species hashes via commutative sum over each product's members.
        def hash_for_product(members_arr, count_arr):
            safe = jnp.where(members_arr >= 0, members_arr, 0)
            sp = particles.species[safe]
            valid_m = (members_arr >= 0) & (jnp.arange(M) < count_arr)
            hvals = jax.vmap(lambda s: _entity_hash_val(s, config))(sp)
            return jnp.sum(jnp.where(valid_m, hvals, 0)) % config.hash_modulus

        h_p0 = hash_for_product(members_p0, count_p0).astype(jnp.uint32)
        h_p1 = hash_for_product(members_p1, count_p1).astype(jnp.uint32)

        return members_p0, count_p0, h_p0, members_p1, count_p1, h_p1

    p0_members, p0_count, p0_hash, p1_members, p1_count, p1_hash = jax.vmap(per_product)(
        jnp.arange(C, dtype=jnp.int32)
    )

    # ── Write product 0 into parent slot c (in place) ──
    # If product 0 has size >= 2: keep composite alive, update members/count/hash, recompute BE & HL.
    # If product 0 has size 1: kill composite (its single member becomes free).
    p0_alive = fissions & (p0_count >= 2)

    # Half-life from BE + size penalty. Same formula as fusion_scan_body.
    # Take both args explicitly so both can be vmapped (closing over p0_count
    # would mis-broadcast against scalar `be` under vmap).
    def _hl_from_be_and_n(be, n):
        t = jnp.clip((be - physics.fusion_threshold) / (1.0 - physics.fusion_threshold + 1e-8), 0.0, 1.0)
        hl_base = config.half_life_min + (config.half_life_max - config.half_life_min) * t
        size_penalty = 1.0 + config.composite_size_decay_scale * jnp.maximum(
            0.0, n.astype(jnp.float32) - 2.0
        )
        return hl_base / size_penalty

    p0_be_all = jax.vmap(lambda h: _hash_to_binding_energy(h, config, physics))(p0_hash)
    p0_hl_all = jax.vmap(_hl_from_be_and_n)(p0_be_all, p0_count)

    new_alive = jnp.where(fissions, p0_alive, composites.alive)
    new_members = jnp.where(fissions[:, None], p0_members, composites.members)
    new_member_count = jnp.where(fissions, p0_count, composites.member_count)
    new_species_hash = jnp.where(fissions, p0_hash, composites.species_hash)
    new_binding_energy = jnp.where(fissions, p0_be_all, composites.binding_energy)
    new_half_life = jnp.where(fissions, p0_hl_all, composites.half_life)
    # Reset age on the parent slot (it's now a fresh product).
    new_age = jnp.where(fissions, jnp.float32(0.0), composites.age)

    # ── Write product 1 into all_target_p1[c] when fissions[c] AND p1_count[c] >= 2 ──
    p1_writes = fissions & (p1_count >= 2)

    # We need to scatter-write into composite slots indexed by all_target_p1.
    # Use `at[].set(..., mode='drop')` to silently ignore non-fissioning rows.
    # Also guard against negative indices: `find_free_slots` returns -1 when
    # there aren't enough free slots, and JAX's negative-index default would
    # wrap to [C-1] — clobbering the last composite. Route those to C (OOB)
    # so `mode='drop'` actually drops them.
    drop_targets = jnp.where(
        p1_writes & (all_target_p1 >= 0),
        all_target_p1,
        C,  # OOB → drop
    )

    p1_be_all = jax.vmap(lambda h: _hash_to_binding_energy(h, config, physics))(p1_hash)
    p1_hl_all = jax.vmap(_hl_from_be_and_n)(p1_be_all, p1_count)

    new_alive       = new_alive.at[drop_targets].set(p1_writes, mode='drop')
    new_members     = new_members.at[drop_targets].set(p1_members, mode='drop')
    new_member_count = new_member_count.at[drop_targets].set(p1_count, mode='drop')
    new_species_hash = new_species_hash.at[drop_targets].set(p1_hash, mode='drop')
    new_binding_energy = new_binding_energy.at[drop_targets].set(p1_be_all, mode='drop')
    new_half_life   = new_half_life.at[drop_targets].set(p1_hl_all, mode='drop')
    new_age         = new_age.at[drop_targets].set(jnp.float32(0.0), mode='drop')

    new_composites = composites._replace(
        members=new_members,
        member_count=new_member_count,
        alive=new_alive,
        binding_energy=new_binding_energy,
        half_life=new_half_life,
        age=new_age,
        species_hash=new_species_hash,
    )

    new_particles = particles._replace(
        composite_id=new_composite_id,
        velocity=new_velocity,
    )

    return state._replace(
        particles=new_particles,
        composites=new_composites,
        rng_key=key,
    )
```

Note: this references `_entity_hash_val` and `_hash_to_binding_energy` — both already exist in `chemistry.py`. The `find_free_slots` import is already at the top of the file.

- [ ] **Step 6: Run all three new tests**

```bash
PYTHONPATH=. pytest tests/test_chemistry.py::test_fission_conserves_particles_and_species tests/test_chemistry.py::test_fission_produces_two_products tests/test_chemistry.py::test_fission_creates_intermediate_size_products -v
```

Expected: all three PASS. Conservation holds; size-3 instances exceed 50.

- [ ] **Step 7: Run full test suite**

```bash
PYTHONPATH=. pytest tests/ -x --tb=short -q
```

Expected: all tests pass.

- [ ] **Step 8: Smoke-run the sim for 15 seconds**

```bash
timeout 15 python -m halflife.main || true
```

Expected: window opens; composites form, decay into pieces (you may see size variation in the histogram), no exceptions. Population should look noticeably more dynamic than before — composites breaking apart in interesting ways rather than just dissociating to free particles.

- [ ] **Step 9: Commit**

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add \
  halflife/chemistry.py tests/test_chemistry.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "feat(chemistry): hash-determined binary fission

Composite decay now partitions members into two product groups via
_hash_to_partition rather than dissociating all members to free state.
Each product becomes a free particle (size 1) or a new composite (size
>= 2), with hash-derived binding energy and half-life. Velocity kicks
along the COM-vs-COM axis are momentum-conserving; the parent slot is
reused for product 0 and a free slot is allocated for product 1.

Particles never change species — fission only rearranges bonding.

Tests added: particle/species conservation across 800 short-half-life
steps, observation of size-3 composite-instances (caught by binary
fission, not by old release-everything decay)."
```

---

## Task J: Visual verification

**Files:** none modified — observational only.

- [ ] **Step 1: Run sim, observe behavior, take notes**

```bash
python -m halflife.main
```

Watch for at least 60 seconds. Look for:

1. **Composite turnover.** Composites should be forming via fusion AND breaking via fission. The histogram in the stats panel should fluctuate (not a frozen pile of size-2s).
2. **Variety of sizes.** Sizes 2, 3, 4, 5+ should all be visible at various times.
3. **No NaN / no crashes.** Watch the FPS counter — should stay reasonable (>20).
4. **Free particle pool dynamics.** Free particle count should oscillate as fissions release singletons and fusion consumes them.
5. **No species drift.** If you can render species histograms (or just check via the stats), species counts should be constant.

If anything looks broken (frozen sim, all composites identical, exceptions in console, FPS in single digits): **stop and debug** before proceeding to Task K. Capture the screenshot and note specific symptoms.

If everything looks good, press `Q` to exit and proceed.

- [ ] **Step 2: Run final test suite to confirm nothing broke since Task I**

```bash
PYTHONPATH=. pytest tests/ -x --tb=short -q
```

Expected: all pass.

This task does not produce a commit — it's a checkpoint.

---

## Task K: Delete dead commented-out code

**Files:**
- Modify: `halflife/utils.py:18-58` (commented `hash_multiset`, `hash_scalar`)
- Modify: `halflife/chemistry.py:80-141` (commented `_hash_to_half_life`, `_hash_to_decay_products`)
- Modify: `tests/test_hash.py:130-160` (commented `test_half_life_distribution`)

- [ ] **Step 1: Delete `hash_multiset` comment block in `halflife/utils.py`**

Find the region in `halflife/utils.py` starting around line 18 with the `# hash_multiset() and hash_scalar() were the original sort-based polynomial...` banner and including the commented-out `def hash_multiset(...)` and `def hash_scalar(...)` blocks. Delete the whole region (banner + commented code).

- [ ] **Step 2: Delete `_hash_to_half_life` and `_hash_to_decay_products` comment blocks in `halflife/chemistry.py`**

Find the regions in `halflife/chemistry.py` starting around lines 80 and 108. Delete both blocks (banner comments + commented-out functions). The file should go from `_hash_to_binding_energy` directly to `_hash_to_partition` to `apply_composite_decay`.

- [ ] **Step 3: Delete the commented `test_half_life_distribution` block in `tests/test_hash.py`**

Find the region around line 130 with `# ── REMOVED 2026-05-05: covered dead code (_hash_to_half_life)` and the commented test below it. Delete the whole region.

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=. pytest tests/ -x --tb=short -q
```

Expected: all pass.

- [ ] **Step 5: Smoke-run sim**

```bash
timeout 5 python -m halflife.main || true
```

Expected: no exceptions.

- [ ] **Step 6: Commit**

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add \
  halflife/utils.py halflife/chemistry.py tests/test_hash.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "cleanup: delete commented-out dead code

hash_multiset, hash_scalar, _hash_to_half_life, _hash_to_decay_products,
and test_half_life_distribution were superseded by the hash-fission
rebuild. The commented placeholders served as a paper trail across the
audit + implementation; they're now redundant with the spec doc and
this commit history."
```

---

## Task L: Add reaction-network observability counter

**Files:**
- Test: `tests/test_chemistry.py`

A composite "type" is its sorted multiset of member species. With binary fission and 12 species, the reachable graph should produce a substantial set of distinct types over time — but selection should keep the *active* set much smaller than the theoretical max.

This isn't a unit test — it's an **observability instrument** that the project can use later to detect autocatalytic loops. We add it as a slow integration test that prints a report.

- [ ] **Step 1: Add the observability test**

Append to `tests/test_chemistry.py`:

```python
def test_observability_distinct_composite_types():
    """
    Observability instrument (slow): count distinct composite types observed
    over 1000 steps and report. A 'type' is the sorted multiset of member
    species (a tuple of sorted species ints).

    With binary fission and 12 species, a healthy reaction network should
    produce many distinct types but stabilize at far fewer than the
    combinatorial max — that's selection.

    This test does not assert dynamics; it asserts only that the counter
    *runs* and produces a nonzero result. Use the printed numbers to study
    the network.
    """
    config = SimConfig(
        num_particles=1000,
        half_life_min=20.0,
        half_life_max=80.0,
    )
    state = initialize_world(config, seed=0)
    params = initialize_interaction_params(config, seed=42)
    physics = initialize_physics_params(config)
    step_fn = jax.jit(simulation_step, static_argnums=(2,))
    state = step_fn(state, params, config, physics)

    types_ever_seen = set()
    types_alive_at_step = []

    for s in range(1000):
        state = step_fn(state, params, config, physics)
        if s % 50 == 0:
            alive = jnp.asarray(state.composites.alive)
            members = jnp.asarray(state.composites.members)
            mc = jnp.asarray(state.composites.member_count)
            species = jnp.asarray(state.particles.species)
            current = set()
            for c_idx in jnp.where(alive)[0].tolist():
                n = int(mc[c_idx])
                mids = members[c_idx, :n].tolist()
                spc = sorted(int(species[m]) for m in mids if m >= 0)
                key = tuple(spc)
                current.add(key)
                types_ever_seen.add(key)
            types_alive_at_step.append((s, len(current)))

    print(f"\nDistinct composite types ever seen: {len(types_ever_seen)}")
    print(f"Types alive at sampled steps:")
    for s, n in types_alive_at_step:
        print(f"  step {s:4d}: {n} distinct types alive")

    assert len(types_ever_seen) > 0, "no composite types ever observed"
```

- [ ] **Step 2: Run the test, observe the report**

```bash
PYTHONPATH=. pytest tests/test_chemistry.py::test_observability_distinct_composite_types -v -s
```

Expected: PASSes; prints something like:

```
Distinct composite types ever seen: <some number, likely 50-500>
Types alive at sampled steps:
  step    0: <N> distinct types alive
  step   50: <N>
  ...
```

The actual numbers are diagnostic — record them. Healthy dynamics: alive-count grows, plateaus, and is much smaller than ever-seen (selection pressure). Pathology: alive-count rapidly converges to 1 or 2 types (runaway), or alive-count tracks ever-seen exactly (no selection).

- [ ] **Step 3: Run full suite to confirm nothing regressed**

```bash
PYTHONPATH=. pytest tests/ -x --tb=short -q
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git -c user.email='heysoos@local' -c user.name='Heysoos' add tests/test_chemistry.py
git -c user.email='heysoos@local' -c user.name='Heysoos' commit -m "test: observability — count distinct composite types over time

Slow integration test that prints a report of how many distinct
composite types (sorted member species multisets) appear and persist
under the new fission dynamics. Doesn't assert on the numbers — the
output is diagnostic, used to study the reaction network and detect
autocatalytic-loop signatures (alive-count ≪ ever-seen-count)."
```

---

## Done

After all 12 tasks land:

- Polarity machinery is fully removed.
- `num_species` defaults to 12.
- `_hash_to_partition` exists and is unit-tested.
- `apply_composite_decay` does binary fission via `_hash_to_partition`.
- Particle counts and species are exactly conserved across simulation runs.
- Dead commented-out code is gone.
- An observability test exists for studying reaction-network dynamics.

Total commits: **11** (one per task A–F, plus G, H, I, K, L; J is a checkpoint with no commit).

Estimated time: ~3-4 hours for an attentive engineer; the longest steps are the `apply_composite_decay` rewrite (Task I) and the visual verification (Task J). Everything else is mechanical.
