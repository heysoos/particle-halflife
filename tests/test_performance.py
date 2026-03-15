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
from halflife.state import initialize_world, initialize_interaction_params
from halflife.step import simulation_step, compute_bond_forces
from halflife.spatial import build_cell_list, find_all_neighbors
from halflife.chemistry import attempt_fusion

_config = SimConfig()
_params = initialize_interaction_params(_config, seed=42)


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
    print(f"  max_particles={c.max_particles}, num_particles_init={c.num_particles_init}")
    print(f"  num_species={c.num_species}, max_composites={c.max_composites}")
    print(f"  max_composite_size={c.max_composite_size}, max_neighbors={c.max_neighbors}")
    print(f"  cell_capacity={c.cell_capacity}, num_cells={c.num_cells}")
    print(f"  max_fusions_per_step={c.max_fusions_per_step}, "
          f"max_decay_per_step={c.max_decay_per_step}")
    print(f"  interaction_radius={c.interaction_radius}, fusion_radius={c.fusion_radius}")
    print(f"  use_bond_forces={c.use_bond_forces}")
    print(f"  JAX devices: {jax.devices()}")
    print("="*60)


def benchmark_full_step():
    """
    Warm up JIT, then time 50 full simulation steps.
    Soft assert: ms/step < 50ms (hard failure only above 200ms).
    """
    config = _config
    params = _params
    state = initialize_world(config, seed=0)
    step_fn = jax.jit(simulation_step, static_argnums=(2,))

    # State is mutable across benchmark calls; use a list to allow mutation in closure
    state_holder = [state]

    def one_step():
        state_holder[0] = step_fn(state_holder[0], params, config)
        state_holder[0].particles.alive.block_until_ready()

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
    state = initialize_world(config, seed=0)
    positions = state.particles.position
    alive = state.particles.alive

    build_jit = jax.jit(build_cell_list, static_argnums=(2,))
    neighbors_jit = jax.jit(find_all_neighbors, static_argnums=(3,))

    # Warm up
    cl = build_jit(positions, alive, config)
    nb = neighbors_jit(positions, alive, cl, config)
    nb.block_until_ready()

    def one_call():
        cl = build_jit(positions, alive, config)
        nb = neighbors_jit(positions, alive, cl, config)
        nb.block_until_ready()

    mean_ms, std_ms = _time_fn(one_call, n_warmup=3, n_bench=50)
    print(f"\nbenchmark_neighbor_finding:")
    print(f"  mean={mean_ms:.2f}ms  std={std_ms:.2f}ms")


def benchmark_fusion_only():
    """
    Time attempt_fusion alone on a state with particles packed closely together
    (many candidate pairs within fusion_radius).
    """
    config = _config
    params = _params

    # Create state with particles clustered in the center to maximize fusion candidates
    state = initialize_world(config, seed=0)
    # Pack particles into a small region by modifying positions
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

    # Build neighbors once (reused across calls)
    build_jit = jax.jit(build_cell_list, static_argnums=(2,))
    neighbors_jit = jax.jit(find_all_neighbors, static_argnums=(3,))
    cl = build_jit(state.particles.position, state.particles.alive, config)
    neighbors = neighbors_jit(state.particles.position, state.particles.alive, cl, config)
    neighbors.block_until_ready()

    fusion_jit = jax.jit(attempt_fusion, static_argnums=(3,))

    state_holder = [state]

    def one_call():
        # Reset to packed state each time to avoid "all claimed" after first fusion
        state_holder[0] = fusion_jit(state, neighbors, params, config)
        state_holder[0].particles.alive.block_until_ready()

    mean_ms, std_ms = _time_fn(one_call, n_warmup=3, n_bench=50)
    print(f"\nbenchmark_fusion_only:")
    print(f"  mean={mean_ms:.2f}ms  std={std_ms:.2f}ms")


def benchmark_bond_forces():
    """
    Time compute_bond_forces on a state with many alive composites.
    We run a few steps first to get some composites (may be 0 if hash bug present).
    """
    config = _config
    params = _params
    state = initialize_world(config, seed=0)
    step_fn = jax.jit(simulation_step, static_argnums=(2,))

    # Run a few steps to try to get composites
    for _ in range(20):
        state = step_fn(state, params, config)
    state.particles.alive.block_until_ready()

    n_composites = int(jnp.sum(state.composites.alive.astype(jnp.int32)))
    bond_jit = jax.jit(compute_bond_forces, static_argnums=(1,))

    # Warm up
    forces = bond_jit(state, config)
    forces.block_until_ready()

    def one_call():
        forces = bond_jit(state, config)
        forces.block_until_ready()

    mean_ms, std_ms = _time_fn(one_call, n_warmup=3, n_bench=100)
    print(f"\nbenchmark_bond_forces (composites={n_composites}):")
    print(f"  mean={mean_ms:.2f}ms  std={std_ms:.2f}ms")


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    print_config_summary()

    passed = failed = 0
    benchmarks = [
        ('benchmark_full_step',       benchmark_full_step),
        ('benchmark_neighbor_finding', benchmark_neighbor_finding),
        ('benchmark_fusion_only',     benchmark_fusion_only),
        ('benchmark_bond_forces',     benchmark_bond_forces),
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
# pytest discovers test_ functions; these just delegate to the benchmarks above.

def test_benchmark_full_step():
    print_config_summary()
    benchmark_full_step()


def test_benchmark_neighbor_finding():
    benchmark_neighbor_finding()


def test_benchmark_fusion_only():
    benchmark_fusion_only()


def test_benchmark_bond_forces():
    benchmark_bond_forces()
