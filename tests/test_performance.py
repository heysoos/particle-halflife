"""
tests/test_performance.py — Step-time benchmarks and per-phase profiling.

These benchmarks always pass (they just print diagnostics). Their purpose is
to surface where simulation time is spent so bottlenecks can be identified.

Run standalone:  python tests/test_performance.py
Run under pytest: pytest tests/test_performance.py -v -s

NOTE: Performance tests run on whatever device is available (GPU if present).
      Do NOT force CPU here — we want real numbers.
"""

import sys
import os
import traceback
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import numpy as np

from halflife.config import SimConfig
from halflife.state import (initialize_world, initialize_interaction_params,
                              initialize_physics_params)
from halflife.step import simulation_step, compute_bond_forces
from halflife.spatial import build_cell_list, find_all_neighbors
from halflife.interactions import compute_all_forces
from halflife.chemistry import attempt_fusion, apply_composite_decay
from halflife.energy import compute_total_energy, apply_soft_energy_conservation

_config  = SimConfig()
_params  = initialize_interaction_params(_config, seed=42)
_physics = initialize_physics_params(_config)


def _time_fn(fn, n_warmup=3, n_bench=50):
    """
    Time a zero-argument callable. Returns (mean_ms, std_ms).
    Runs n_warmup calls first (not timed), then n_bench timed calls.
    Each call blocks until JAX computation is complete.
    """
    for _ in range(n_warmup):
        fn()

    times_ms = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    return np.mean(times_ms), np.std(times_ms)


# ── Benchmarks ─────────────────────────────────────────────────────────────────

def print_config_summary():
    """Print key SimConfig parameters for context in test output."""
    c = _config
    print("\n" + "="*60)
    print("SimConfig summary:")
    print(f"  num_particles={c.num_particles}")
    print(f"  num_species={c.num_species}, max_composites={c.max_composites}")
    print(f"  max_composite_size={c.max_composite_size}, max_neighbors={c.max_neighbors}")
    print(f"  cell_capacity={c.cell_capacity}, num_cells={c.num_cells}")
    print(f"  max_fusions_per_step={c.max_fusions_per_step}")
    print(f"  interaction_radius={c.interaction_radius}, fusion_radius={c.fusion_radius}")
    print(f"  use_bond_forces={c.use_bond_forces}")
    print(f"  JAX devices: {jax.devices()}")
    print("="*60)


def benchmark_full_step():
    """
    Warm up JIT, then time 50 full simulation steps.
    Soft assert: ms/step < 50ms (hard failure only above 200ms).
    """
    config  = _config
    params  = _params
    physics = _physics
    state   = initialize_world(config, seed=0)
    step_fn = jax.jit(simulation_step, static_argnums=(2,))

    state_holder = [state]

    def one_step():
        state_holder[0] = step_fn(state_holder[0], params, config, physics)
        state_holder[0].particles.position.block_until_ready()

    mean_ms, std_ms = _time_fn(one_step, n_warmup=5, n_bench=50)
    steps_per_sec = 1000.0 / mean_ms if mean_ms > 0 else float('inf')

    print(f"\nbenchmark_full_step:")
    print(f"  mean={mean_ms:.2f}ms  std={std_ms:.2f}ms  "
          f"steps/sec={steps_per_sec:.1f}")

    assert mean_ms < 200.0, (
        f"Step time {mean_ms:.1f}ms exceeds hard limit of 200ms. "
        f"Something is severely wrong."
    )
    if mean_ms > 50.0:
        print(f"  SOFT WARNING: {mean_ms:.1f}ms > 50ms target. "
              f"Performance optimization needed.")


def benchmark_neighbor_finding():
    """Time build_cell_list + find_all_neighbors alone."""
    config = _config
    state  = initialize_world(config, seed=0)
    positions = state.particles.position

    build_jit     = jax.jit(build_cell_list,    static_argnums=(1,))
    neighbors_jit = jax.jit(find_all_neighbors, static_argnums=(2,))

    # Warm up
    cl = build_jit(positions, config)
    nb = neighbors_jit(positions, cl, config)
    nb.block_until_ready()

    def one_call():
        cl = build_jit(positions, config)
        nb = neighbors_jit(positions, cl, config)
        nb.block_until_ready()

    mean_ms, std_ms = _time_fn(one_call, n_warmup=3, n_bench=50)
    print(f"\nbenchmark_neighbor_finding:")
    print(f"  mean={mean_ms:.2f}ms  std={std_ms:.2f}ms")


def benchmark_fusion_only():
    """
    Time attempt_fusion alone on a state with particles packed closely together
    (many candidate pairs within fusion_radius).
    """
    config  = _config
    params  = _params
    physics = _physics

    state = initialize_world(config, seed=0)
    key = jax.random.PRNGKey(99)
    packed_pos = jax.random.uniform(
        key,
        shape=state.particles.position.shape,
        minval=jnp.array([90.0, 90.0]),
        maxval=jnp.array([110.0, 110.0]),
    )
    state = state._replace(
        particles=state.particles._replace(position=packed_pos)
    )

    build_jit     = jax.jit(build_cell_list,    static_argnums=(1,))
    neighbors_jit = jax.jit(find_all_neighbors, static_argnums=(2,))
    cl        = build_jit(state.particles.position, config)
    neighbors = neighbors_jit(state.particles.position, cl, config)
    neighbors.block_until_ready()

    fusion_jit = jax.jit(attempt_fusion, static_argnums=(3,))

    state_holder = [state]

    def one_call():
        state_holder[0] = fusion_jit(state, neighbors, params, config, physics)
        state_holder[0].particles.composite_id.block_until_ready()

    mean_ms, std_ms = _time_fn(one_call, n_warmup=3, n_bench=50)
    print(f"\nbenchmark_fusion_only:")
    print(f"  mean={mean_ms:.2f}ms  std={std_ms:.2f}ms")


def benchmark_bond_forces():
    """
    Time compute_bond_forces on a state with many alive composites.
    We run a few steps first to get some composites.
    """
    config  = _config
    params  = _params
    physics = _physics
    state   = initialize_world(config, seed=0)
    step_fn = jax.jit(simulation_step, static_argnums=(2,))

    for _ in range(20):
        state = step_fn(state, params, config, physics)
    state.particles.position.block_until_ready()

    n_composites = int(jnp.sum(state.composites.alive.astype(jnp.int32)))
    bond_jit = jax.jit(compute_bond_forces, static_argnums=(1,))

    forces = bond_jit(state, config, _physics)
    forces.block_until_ready()

    def one_call():
        forces = bond_jit(state, config, _physics)
        forces.block_until_ready()

    mean_ms, std_ms = _time_fn(one_call, n_warmup=3, n_bench=100)
    print(f"\nbenchmark_bond_forces (composites={n_composites}):")
    print(f"  mean={mean_ms:.2f}ms  std={std_ms:.2f}ms")


def benchmark_compute_forces():
    """Time compute_all_forces alone (forces phase only, neighbors pre-built)."""
    config  = _config
    params  = _params
    physics = _physics
    state   = initialize_world(config, seed=0)

    build_jit     = jax.jit(build_cell_list,    static_argnums=(1,))
    neighbors_jit = jax.jit(find_all_neighbors, static_argnums=(2,))
    forces_jit    = jax.jit(compute_all_forces, static_argnums=(4,))

    particles = state.particles
    cl  = build_jit(particles.position, config)
    nb  = neighbors_jit(particles.position, cl, config)
    nb.block_until_ready()

    attr_mod = jnp.ones(config.num_particles, dtype=jnp.float32)

    f = forces_jit(particles.position, particles.species,
                   nb, params, config, physics, attr_mod)
    f.block_until_ready()

    def one_call():
        f = forces_jit(particles.position, particles.species,
                       nb, params, config, physics, attr_mod)
        f.block_until_ready()

    mean_ms, std_ms = _time_fn(one_call, n_warmup=3, n_bench=50)
    print(f"\nbenchmark_compute_forces:")
    print(f"  mean={mean_ms:.2f}ms  std={std_ms:.2f}ms")


def benchmark_composite_decay():
    """Time apply_composite_decay alone on a state with composites."""
    config  = _config
    params  = _params
    physics = _physics
    state   = initialize_world(config, seed=0)
    step_fn = jax.jit(simulation_step, static_argnums=(2,))

    for _ in range(20):
        state = step_fn(state, params, config, physics)
    state.particles.position.block_until_ready()

    n_composites = int(jnp.sum(state.composites.alive.astype(jnp.int32)))
    decay_jit = jax.jit(apply_composite_decay, static_argnums=(1,))

    r = decay_jit(state, config)
    r.particles.composite_id.block_until_ready()

    def one_call():
        r = decay_jit(state, config)
        r.particles.composite_id.block_until_ready()

    mean_ms, std_ms = _time_fn(one_call, n_warmup=3, n_bench=50)
    print(f"\nbenchmark_composite_decay (composites={n_composites}):")
    print(f"  mean={mean_ms:.2f}ms  std={std_ms:.2f}ms")


def benchmark_per_phase_breakdown():
    """
    Time each simulation phase independently using a fixed realistic state.
    Prints a table with mean ms and % of phase-sum total.
    Also times the fused full step to show XLA fusion savings.
    """
    config  = _config
    params  = _params
    physics = _physics

    # Warm up simulation to get a realistic state (composites forming)
    state   = initialize_world(config, seed=0)
    step_fn = jax.jit(simulation_step, static_argnums=(2,))
    for _ in range(20):
        state = step_fn(state, params, config, physics)
    state.particles.position.block_until_ready()

    n_composites = int(jnp.sum(state.composites.alive.astype(jnp.int32)))
    print(f"\nbenchmark_per_phase_breakdown  "
          f"(particles={config.num_particles}, composites={n_composites}):")

    # JIT each phase
    build_jit   = jax.jit(build_cell_list,       static_argnums=(1,))
    nb_jit      = jax.jit(find_all_neighbors,    static_argnums=(2,))
    forces_jit  = jax.jit(compute_all_forces,    static_argnums=(4,))
    bond_jit    = jax.jit(compute_bond_forces,   static_argnums=(1,))
    fusion_jit  = jax.jit(attempt_fusion,        static_argnums=(3,))
    decay_jit   = jax.jit(apply_composite_decay, static_argnums=(1,))

    @jax.jit
    def energy_phase(s):
        e = compute_total_energy(s)
        return apply_soft_energy_conservation(s, e)

    # Pre-compute fixed intermediates (frozen state)
    particles = state.particles
    is_comp   = particles.composite_id >= 0
    safe_cid  = jnp.clip(particles.composite_id, 0, config.max_composites - 1)
    attr_mod  = jnp.where(is_comp, state.composites.net_polarity[safe_cid], 1.0)

    cl_fixed = build_jit(particles.position, config)
    cl_fixed.particle_ids.block_until_ready()
    nb_fixed = nb_jit(particles.position, cl_fixed, config)
    nb_fixed.block_until_ready()

    # Warm up all JITs
    for _ in range(3):
        build_jit(particles.position, config).particle_ids.block_until_ready()
        nb_jit(particles.position, cl_fixed, config).block_until_ready()
        forces_jit(particles.position, particles.species,
                   nb_fixed, params, config, physics, attr_mod).block_until_ready()
        if config.use_bond_forces:
            bond_jit(state, config, physics).block_until_ready()
        fusion_jit(state, nb_fixed, params, config, physics).particles.composite_id.block_until_ready()
        decay_jit(state, config).particles.composite_id.block_until_ready()
        energy_phase(state).particles.velocity.block_until_ready()
        step_fn(state, params, config, physics).particles.position.block_until_ready()

    phases = []

    def _t(name, fn):
        ms, std = _time_fn(fn, n_warmup=0, n_bench=50)
        phases.append((name, ms, std))

    _t("1. build_cell_list",
       lambda: build_jit(particles.position, config)
                        .particle_ids.block_until_ready())
    _t("2. find_all_neighbors",
       lambda: nb_jit(particles.position, cl_fixed, config)
                     .block_until_ready())
    _t("3. compute_all_forces",
       lambda: forces_jit(particles.position, particles.species,
                          nb_fixed, params, config, physics, attr_mod)
                         .block_until_ready())
    if config.use_bond_forces:
        _t("4. compute_bond_forces",
           lambda: bond_jit(state, config, physics).block_until_ready())
    _t("5. attempt_fusion",
       lambda: fusion_jit(state, nb_fixed, params, config, physics)
                         .particles.composite_id.block_until_ready())
    _t("6. apply_composite_decay",
       lambda: decay_jit(state, config).particles.composite_id.block_until_ready())
    _t("7. energy_conservation",
       lambda: energy_phase(state).particles.velocity.block_until_ready())

    full_ms, full_std = _time_fn(
        lambda: step_fn(state, params, config, physics).particles.position.block_until_ready(),
        n_warmup=0, n_bench=50
    )

    total_ms = sum(ms for _, ms, _ in phases)
    w = max(len(n) for n, _, _ in phases) + 2

    print(f"  {'Phase':<{w}} | {'mean ms':>8} | {'std ms':>7} | {'% total':>8}")
    print("  " + "-" * (w + 32))
    for name, ms, std in phases:
        pct = 100.0 * ms / total_ms if total_ms > 0 else 0.0
        print(f"  {name:<{w}} | {ms:>8.3f} | {std:>7.3f} | {pct:>7.1f}%")
    print("  " + "-" * (w + 32))
    print(f"  {'TOTAL (phases summed)':<{w}} | {total_ms:>8.3f} |         | {'100.0%':>8}")
    savings = total_ms - full_ms
    savings_pct = 100.0 * savings / total_ms if total_ms > 0 else 0.0
    print(f"  {'FULL STEP (jit fused)':<{w}} | {full_ms:>8.3f} | {full_std:>7.3f} |  (fused)")
    print(f"  {'XLA fusion savings':<{w}} | {savings:>8.3f} |         | {savings_pct:>7.1f}%")


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    print_config_summary()

    passed = failed = 0
    benchmarks = [
        ('benchmark_full_step',           benchmark_full_step),
        ('benchmark_neighbor_finding',    benchmark_neighbor_finding),
        ('benchmark_fusion_only',         benchmark_fusion_only),
        ('benchmark_bond_forces',         benchmark_bond_forces),
        ('benchmark_compute_forces',      benchmark_compute_forces),
        ('benchmark_composite_decay',     benchmark_composite_decay),
        ('benchmark_per_phase_breakdown', benchmark_per_phase_breakdown),
    ]

    for name, fn in benchmarks:
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


# ── pytest entry points ───────────────────────────────────────────────────────

def test_benchmark_full_step():
    print_config_summary()
    benchmark_full_step()


def test_benchmark_neighbor_finding():
    benchmark_neighbor_finding()


def test_benchmark_fusion_only():
    benchmark_fusion_only()


def test_benchmark_bond_forces():
    benchmark_bond_forces()


def test_benchmark_compute_forces():
    benchmark_compute_forces()


def test_benchmark_composite_decay():
    benchmark_composite_decay()


def test_benchmark_per_phase_breakdown():
    benchmark_per_phase_breakdown()
