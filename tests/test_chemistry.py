"""
tests/test_chemistry.py — Fusion, decay, and fission integration tests.

Primary goal: verify that composites actually form (test_fusion_occurs is
expected to FAIL with the current hash bug), and that decay/fission work.

Run standalone:  python tests/test_chemistry.py
Run under pytest: pytest tests/test_chemistry.py -v
"""

import sys
import os
import traceback
import functools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
import numpy as np

from halflife.config import SimConfig
from halflife.state import (
    initialize_world,
    initialize_interaction_params,
    initialize_physics_params,
)
from halflife.step import simulation_step

# JIT-compiled step (compiled once per config, cached)
_step_jit = jax.jit(simulation_step, static_argnums=(2,))

_config = SimConfig()
_params = initialize_interaction_params(_config, seed=42)
_physics = initialize_physics_params(_config)


@functools.lru_cache(maxsize=None)
def _get_base_state():
    """Return initial state (computed once per process)."""
    return initialize_world(_config, seed=0)


def _run_steps(n: int, config=_config, seed=0):
    """Run n steps from a fresh initial state, return final state."""
    state = initialize_world(config, seed=seed)
    params = initialize_interaction_params(config, seed=42)
    physics = initialize_physics_params(config)
    step_fn = jax.jit(simulation_step, static_argnums=(2,))
    # Warm-up JIT on first call
    state = step_fn(state, params, config, physics)
    state.particles.position.block_until_ready()
    for _ in range(n - 1):
        state = step_fn(state, params, config, physics)
    state.particles.position.block_until_ready()
    return state


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_fusion_occurs():
    """
    After 200 steps, at least one composite should have formed.
    EXPECTED TO FAIL with the current hash bug (BE≈0 → no fusions).
    On failure, prints the binding energy matrix for diagnosis.
    """
    state = _run_steps(200)
    n_composites = int(jnp.sum(state.composites.alive.astype(jnp.int32)))

    if n_composites == 0:
        # Print diagnostic: binding energies for all species pairs
        from halflife.chemistry import _entity_hash_val, _hash_to_binding_energy
        S = _config.num_species
        print(f"\nDiagnostic: binding energy matrix ({S}x{S})")
        matrix = np.zeros((S, S))
        for i in range(S):
            for j in range(S):
                hi = int(_entity_hash_val(jnp.int32(i), _config))
                hj = int(_entity_hash_val(jnp.int32(j), _config))
                merged = (hi + hj) % _config.hash_modulus
                matrix[i, j] = float(_hash_to_binding_energy(jnp.uint32(merged), _config, _physics))
        print(np.array2string(matrix, precision=3, suppress_small=True))
        print(f"All BEs are zero: {np.allclose(matrix, 0)}")
        print(f"fusion_threshold={_config.fusion_threshold}")

    assert n_composites > 0, (
        f"No composites formed after 200 steps. "
        f"Check binding energy matrix above — likely the hash bug."
    )


def test_fusion_count_grows():
    """
    Over 500 steps, the cumulative number of fusion events should exceed 10.
    Tracks composites.alive delta each step.
    EXPECTED TO FAIL with the current hash bug.
    """
    config = _config
    state = initialize_world(config, seed=1)
    params = initialize_interaction_params(config, seed=42)
    physics = initialize_physics_params(config)
    step_fn = jax.jit(simulation_step, static_argnums=(2,))

    # Warm up
    state = step_fn(state, params, config, physics)
    state.particles.position.block_until_ready()

    cumulative_fusions = 0
    prev_n_comp = int(jnp.sum(state.composites.alive.astype(jnp.int32)))

    for step in range(499):
        state = step_fn(state, params, config, physics)
        n_comp = int(jnp.sum(state.composites.alive.astype(jnp.int32)))
        delta = max(0, n_comp - prev_n_comp)
        cumulative_fusions += delta
        prev_n_comp = n_comp

    print(f"\nCumulative fusion events over 500 steps: {cumulative_fusions}")
    assert cumulative_fusions > 10, (
        f"Only {cumulative_fusions} fusion events in 500 steps (need > 10). "
        f"Hash bug likely."
    )


def test_composite_half_life_valid():
    """
    After fusions occur, all alive composites should have half_life in (0, max_hl].
    """
    # Use more steps to give time for fusions
    state = _run_steps(300)
    composites = state.composites
    alive_mask = composites.alive

    n_alive = int(jnp.sum(alive_mask.astype(jnp.int32)))
    if n_alive == 0:
        print("\nNo composites alive — skipping half-life range check (hash bug?)")
        return  # Can't test if no composites exist (hash bug prevents fusion)

    max_expected_hl = _config.half_life_max * 2.0  # generous margin

    alive_hls = jnp.where(alive_mask, composites.half_life, jnp.inf)
    min_hl = float(jnp.min(jnp.where(alive_mask, composites.half_life, jnp.inf)))
    max_hl = float(jnp.max(jnp.where(alive_mask, composites.half_life, -jnp.inf)))

    print(f"\nAlive composites: {n_alive}, half_life range: [{min_hl:.1f}, {max_hl:.1f}]")
    assert min_hl > 0, f"Composite with half_life <= 0: {min_hl}"
    assert max_hl <= max_expected_hl, (
        f"Composite half_life {max_hl:.1f} exceeds expected max {max_expected_hl:.1f}"
    )


def test_decay_occurs():
    """
    With very short half-lives, composites should eventually fission after forming.
    Requires fusions to work first — will trivially pass (no composites to decay)
    if the hash bug is present.
    """
    # Short half-life config: composites decay quickly once formed
    config = SimConfig(
        half_life_min=10.0,
        half_life_max=30.0,
        num_particles=500,
    )
    state = initialize_world(config, seed=2)
    params = initialize_interaction_params(config, seed=42)
    physics = initialize_physics_params(config)
    step_fn = jax.jit(simulation_step, static_argnums=(2,))

    # Warm up
    state = step_fn(state, params, config, physics)
    state.particles.position.block_until_ready()

    max_composites_seen = 0
    fission_observed = False

    for _ in range(500):
        state = step_fn(state, params, config, physics)
        n = int(jnp.sum(state.composites.alive.astype(jnp.int32)))
        if n > max_composites_seen:
            max_composites_seen = n
        if max_composites_seen > 0 and n < max_composites_seen:
            fission_observed = True
            break

    print(f"\nMax composites seen: {max_composites_seen}, fission observed: {fission_observed}")

    if max_composites_seen == 0:
        print("  (No fusions occurred — likely hash bug. Decay test inconclusive.)")
        return  # Can't test decay without fusion

    assert fission_observed, (
        f"Composites formed (max={max_composites_seen}) but no fission observed in 500 steps "
        f"with short half-lives (min={config.half_life_min})."
    )


def test_no_particle_loss():
    """
    Particle count is fixed (no creation/destruction) and must always equal
    config.num_particles regardless of fusion/fission activity.
    """
    config = _config
    state = initialize_world(config, seed=3)
    params = initialize_interaction_params(config, seed=42)
    physics = initialize_physics_params(config)
    step_fn = jax.jit(simulation_step, static_argnums=(2,))
    state = step_fn(state, params, config, physics)
    state.particles.position.block_until_ready()

    expected = config.num_particles
    for _ in range(299):
        state = step_fn(state, params, config, physics)
    actual = state.particles.position.shape[0]

    print(f"\nParticle count: expected={expected}, actual={actual} (fixed)")
    assert actual == expected, (
        f"Particle count changed: {actual} != {expected}"
    )


def test_composite_member_consistency():
    """
    After fusion: every alive composite should have member_count > 0, all member
    indices valid, and those particles' composite_id pointing back to this composite.
    """
    state = _run_steps(300)
    composites = state.composites
    particles = state.particles

    n_checked = 0
    errors = []
    for c in range(_config.max_composites):
        if not bool(composites.alive[c]):
            continue
        n_checked += 1
        mc = int(composites.member_count[c])

        if mc <= 0:
            errors.append(f"Composite {c}: member_count={mc} <= 0")
            continue

        for m in range(mc):
            pid = int(composites.members[c, m])
            if pid < 0 or pid >= _config.num_particles:
                errors.append(f"Composite {c} member[{m}]={pid} is invalid index")
                continue
            cid = int(particles.composite_id[pid])
            if cid != c:
                errors.append(
                    f"Composite {c} member[{m}]={pid} has composite_id={cid} (expected {c})"
                )

    print(f"\nChecked {n_checked} alive composites for member consistency.")
    if errors:
        for e in errors[:10]:
            print(f"  ERROR: {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

    assert not errors, f"{len(errors)} member consistency errors found (first: {errors[0]})"


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
    """
    config = SimConfig(
        half_life_min=10.0,
        half_life_max=30.0,
        fusion_threshold=0.4,
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

    assert size_3_instances >= 50, (
        f"too few size-3 composite-instances observed: {size_3_instances} "
        "(binary fission should produce these readily)"
    )


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


# ── Tests for valence / free-bond saturation gate ────────────────────────────

def test_valence_off_unchanged():
    """
    With use_valence=False, max_valence should not affect dynamics — both
    configs run the same number of steps with identical seed, no valence
    gate engaged.
    """
    base = SimConfig(num_particles=500, use_valence=False)
    state_a = _run_steps(200, config=base, seed=11)
    state_b = _run_steps(
        200,
        config=SimConfig(num_particles=500, use_valence=False, max_valence=2),
        seed=11,
    )
    n_a = int(jnp.sum(state_a.composites.alive.astype(jnp.int32)))
    n_b = int(jnp.sum(state_b.composites.alive.astype(jnp.int32)))
    assert n_a == n_b, (
        f"max_valence should not affect dynamics when use_valence=False: "
        f"got {n_a} vs {n_b}"
    )


def test_valence_on_conserves_particles_and_species():
    """
    With valence on (including fission shattering of structurally unsound
    products), particle count and per-species counts must still be exactly
    conserved.
    """
    config = SimConfig(
        num_particles=500,
        half_life_min=5.0,
        half_life_max=20.0,
        use_valence=True,
        max_valence=4,
    )
    state = initialize_world(config, seed=0)
    params = initialize_interaction_params(config, seed=42)
    physics = initialize_physics_params(config)
    step_fn = jax.jit(simulation_step, static_argnums=(2,))
    state = step_fn(state, params, config, physics)  # warm-up

    initial_species = jnp.asarray(state.particles.species)
    initial_per_species = jnp.bincount(initial_species, length=config.num_species)

    for _ in range(800):
        state = step_fn(state, params, config, physics)

    final_species = jnp.asarray(state.particles.species)
    final_per_species = jnp.bincount(final_species, length=config.num_species)

    assert state.particles.position.shape[0] == config.num_particles
    assert jnp.all(initial_species == final_species), (
        "particle species changed under valence — fission must not transmute"
    )
    assert jnp.all(initial_per_species == final_per_species), (
        f"per-species counts changed under valence:\n"
        f"  initial={initial_per_species}\n  final={final_per_species}"
    )


def test_valence_free_bonds_nonnegative():
    """
    Invariant: every alive composite must have free_bonds >= 0. Negative
    free_bonds is a structurally unsound molecule (more bonds required by
    the tree than the members offer), and the fission shatter path should
    prevent any such composite from being alive.
    """
    config = SimConfig(
        num_particles=500,
        half_life_min=50.0,
        half_life_max=200.0,
        use_valence=True,
        max_valence=4,
    )
    state = _run_steps(400, config=config, seed=7)

    alive = jnp.asarray(state.composites.alive)
    fb = jnp.asarray(state.composites.free_bonds)
    n_alive = int(alive.sum())
    if n_alive == 0:
        print("\nno composites alive — test inconclusive")
        return

    alive_fb = fb[alive]
    n_bad = int(jnp.sum(alive_fb < 0))
    print(f"\nChecked {n_alive} alive composites; free_bonds<0 count: {n_bad}")
    assert n_bad == 0, (
        f"{n_bad}/{n_alive} alive composites have free_bonds<0 "
        f"(structurally unsound — fission shatter path failed)"
    )


def test_valence_saturation_caps_size_at_max_valence_1():
    """
    With max_valence=1 every species has v_s=1. A 2-particle composite has
    free_bonds = 2 − 2 = 0, so it's immediately saturated and cannot grow.
    No composite of size >= 3 should ever form.

    This is the strongest possible saturation test: deterministic ceiling
    that depends only on the valence gate working.
    """
    config = SimConfig(
        num_particles=500,
        half_life_min=50.0,
        half_life_max=200.0,
        fusion_threshold=0.3,    # permissive so growth would happen if allowed
        use_valence=True,
        max_valence=1,
    )
    state = _run_steps(400, config=config, seed=3)
    alive = state.composites.alive
    mc = state.composites.member_count
    max_size = int(jnp.max(jnp.where(alive, mc, 0)))
    print(f"\nmax_valence=1 max composite size observed: {max_size}")
    assert max_size <= 2, (
        f"max_valence=1 should cap size at 2 (every species has v=1, so size-2 "
        f"is immediately saturated); got max_size={max_size}"
    )


def test_valence_limit_growth_vs_off():
    """
    With valence on at small max_valence, composites should not grow as large
    as without valence. This is the population-level signature of saturation.
    """
    base = dict(
        num_particles=500,
        half_life_min=50.0,
        half_life_max=200.0,
        fusion_threshold=0.3,    # permissive so growth happens
        max_composite_size=64,
    )

    state_off = _run_steps(400, config=SimConfig(**base, use_valence=False), seed=3)
    state_on  = _run_steps(400, config=SimConfig(**base, use_valence=True, max_valence=2), seed=3)

    max_off = int(jnp.max(jnp.where(state_off.composites.alive,
                                      state_off.composites.member_count, 0)))
    max_on  = int(jnp.max(jnp.where(state_on.composites.alive,
                                      state_on.composites.member_count, 0)))

    print(f"\nMax composite size: valence off={max_off}, valence on (v<=2)={max_on}")
    assert max_on < max_off, (
        f"valence did not reduce max composite size: off={max_off} vs on={max_on}"
    )


# ── Tests for sparse covalent bonds (bond_mode knobs) ─────────────────────────

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


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    passed = failed = 0
    for name, fn in [(n, v) for n, v in sorted(globals().items()) if n.startswith('test_')]:
        try:
            fn()
            print(f'  PASS  {name}')
            passed += 1
        except Exception as e:
            print(f'  FAIL  {name}: {e}')
            traceback.print_exc()
            failed += 1
    print(f'\n{passed} passed, {failed} failed')
    sys.exit(failed)


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
