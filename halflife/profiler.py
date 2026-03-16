"""
halflife/profiler.py — Standalone per-phase profiling harness.

Produces a per-phase timing table showing where each simulation step's
wall-clock time goes, plus optional scale sweep and JAX profiler trace.

Usage:
    python -m halflife.profiler                         # default N
    python -m halflife.profiler --n 1000               # 1000 init particles
    python -m halflife.profiler --scale-sweep           # N={500,1000,2000,4000}
    python -m halflife.profiler --trace /tmp/hl-trace   # Perfetto trace
    python -m halflife.profiler --memory                # device memory profile
    python -m cProfile -m halflife.profiler             # Python dispatch overhead

Output example:
    Phase                      |  mean ms |  std ms |  % total
    ---------------------------|----------|---------|----------
    1. build_cell_list          |     0.12 |    0.02 |      0.9%
    2. find_all_neighbors       |     3.45 |    0.08 |     27.1%
    ...
    TOTAL (phases summed)       |    12.74 |         |    100.0%
    FULL STEP (jit fused)       |     5.82 |    0.09 |   (fused)
    XLA fusion savings          |     6.92 |         |     54.3%
"""

import argparse
import os
import time
from collections import OrderedDict

import jax
import jax.numpy as jnp
import numpy as np

from halflife.config import SimConfig
from halflife.state import (initialize_world, initialize_interaction_params,
                              initialize_physics_params)
from halflife.spatial import build_cell_list, find_all_neighbors
from halflife.interactions import compute_all_forces
from halflife.step import simulation_step, compute_bond_forces
from halflife.chemistry import attempt_fusion, apply_composite_decay
from halflife.energy import compute_total_energy, apply_soft_energy_conservation


# ── Timing Utility ────────────────────────────────────────────────────────────

def _time_fn(fn, n_warmup=3, n_bench=50, n_runs=1):
    """
    Time a zero-argument callable. Returns (mean_ms, std_ms).
    fn must call .block_until_ready() internally to sync the GPU.

    n_runs > 1: runs n_bench iterations n_runs separate times, then reports
    the mean and std of the per-run means. This gives inter-run stability
    (measurement uncertainty) rather than within-run variance.
    """
    for _ in range(n_warmup):
        fn()
    run_means = []
    for _ in range(n_runs):
        times = []
        for _ in range(n_bench):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000.0)
        run_means.append(float(np.mean(times)))
    return float(np.mean(run_means)), float(np.std(run_means))


# ── Config Factory ────────────────────────────────────────────────────────────

def make_config(num_particles):
    """
    Create a SimConfig scaled to num_particles.
    Keeps the same ratios as the default config:
      max_composites = num_particles // 2
    """
    return SimConfig(
        num_particles=num_particles,
        max_composites=max(64, num_particles // 2),
    )


# ── Per-Phase Profiler ────────────────────────────────────────────────────────

def profile_all_phases(config, physics, params, n_warmup=3, n_bench=50, n_runs=3):
    """
    Time each simulation phase independently using a fixed realistic state.

    Each phase is JIT-compiled separately and called with frozen inputs from
    a state that has been warmed up for 20 steps (composites present).
    Phases are timed with the same fixed input each iteration — not the
    evolving state — so timings are stable and reproducible.

    Note: phases summed > full fused step due to XLA kernel fusion savings.

    Returns:
        results:   OrderedDict[phase_name -> (mean_ms, std_ms)]
        full_ms:   float — mean ms for the fully fused simulation_step
        full_std:  float — std ms for the fully fused simulation_step
    """
    # Build realistic state with composites
    state   = initialize_world(config, seed=0)
    step_fn = jax.jit(simulation_step, static_argnums=(2,))

    print("  Warming up (20 steps to build composites)...", flush=True)
    for _ in range(20):
        state = step_fn(state, params, config, physics)
    state.particles.position.block_until_ready()

    n_alive = config.num_particles
    n_comp  = int(jnp.sum(state.composites.alive.astype(jnp.int32)))
    print(f"  State: {n_alive} alive particles, {n_comp} alive composites", flush=True)

    # JIT-compile all phase functions
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

    # Pre-compute fixed intermediates from frozen state
    particles = state.particles
    is_comp   = particles.composite_id >= 0
    safe_cid  = jnp.clip(particles.composite_id, 0, config.max_composites - 1)
    attr_mod  = jnp.where(is_comp, state.composites.net_polarity[safe_cid], 1.0)

    cl_fixed = build_jit(particles.position, config)
    cl_fixed.particle_ids.block_until_ready()
    nb_fixed = nb_jit(particles.position, cl_fixed, config)
    nb_fixed.block_until_ready()

    # Warm up all JITs before timed runs
    print("  JIT warm-up...", flush=True)
    for _ in range(n_warmup):
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

    print(f"  Timing phases ({n_runs} runs × {n_bench} iters each)...", flush=True)
    results = OrderedDict()

    results['1. build_cell_list'] = _time_fn(
        lambda: build_jit(particles.position, config)
                          .particle_ids.block_until_ready(),
        n_warmup=0, n_bench=n_bench, n_runs=n_runs,
    )
    results['2. find_all_neighbors'] = _time_fn(
        lambda: nb_jit(particles.position, cl_fixed, config)
                      .block_until_ready(),
        n_warmup=0, n_bench=n_bench, n_runs=n_runs,
    )
    results['3. compute_all_forces'] = _time_fn(
        lambda: forces_jit(particles.position, particles.species,
                           nb_fixed, params, config, physics, attr_mod)
                          .block_until_ready(),
        n_warmup=0, n_bench=n_bench, n_runs=n_runs,
    )
    if config.use_bond_forces:
        results['4. compute_bond_forces'] = _time_fn(
            lambda: bond_jit(state, config, physics).block_until_ready(),
            n_warmup=0, n_bench=n_bench, n_runs=n_runs,
        )
    results['5. attempt_fusion'] = _time_fn(
        lambda: fusion_jit(state, nb_fixed, params, config, physics)
                          .particles.composite_id.block_until_ready(),
        n_warmup=0, n_bench=n_bench, n_runs=n_runs,
    )
    results['6. apply_composite_decay'] = _time_fn(
        lambda: decay_jit(state, config).particles.composite_id.block_until_ready(),
        n_warmup=0, n_bench=n_bench, n_runs=n_runs,
    )
    results['7. energy_conservation'] = _time_fn(
        lambda: energy_phase(state).particles.velocity.block_until_ready(),
        n_warmup=0, n_bench=n_bench, n_runs=n_runs,
    )

    full_ms, full_std = _time_fn(
        lambda: step_fn(state, params, config, physics).particles.position.block_until_ready(),
        n_warmup=0, n_bench=n_bench, n_runs=n_runs,
    )

    return results, full_ms, full_std


# ── Table Printer ─────────────────────────────────────────────────────────────

def print_phase_table(results, full_ms, full_std, config):
    """Print formatted per-phase timing table."""
    total_ms = sum(ms for ms, _ in results.values())
    w = max(len(k) for k in results) + 2

    print(f"\nPhase breakdown  "
          f"(N={config.num_particles}, "
          f"max_composites={config.max_composites}, "
          f"bond_forces={config.use_bond_forces})")

    header = f"  {'Phase':<{w}} | {'mean ms':>8} | {'±std':>7} | {'% total':>8}"
    sep    = "  " + "-" * (w + 32)
    print(header)
    print(sep)
    for name, (ms, std) in results.items():
        pct = 100.0 * ms / total_ms if total_ms > 0 else 0.0
        print(f"  {name:<{w}} | {ms:>8.3f} | {std:>7.3f} | {pct:>7.1f}%")
    print(sep)
    print(f"  {'TOTAL (phases summed)':<{w}} | {total_ms:>8.3f} |         | {'100.0%':>8}")
    savings     = total_ms - full_ms
    savings_pct = 100.0 * savings / total_ms if total_ms > 0 else 0.0
    print(f"  {'FULL STEP (jit fused)':<{w}} | {full_ms:>8.3f} | {full_std:>7.3f} |   (fused)")
    print(f"  {'XLA fusion savings':<{w}} | {savings:>8.3f} |         | {savings_pct:>7.1f}%")


# ── Scale Sweep ───────────────────────────────────────────────────────────────

# Short keys for scale-sweep columns (subset of full phase names)
_SWEEP_COLS = [
    ('full_step',              'full_step'),
    ('2. find_all_neighbors',  'neighbors'),
    ('3. compute_all_forces',  'forces'),
    ('4. compute_bond_forces', 'bond'),
    ('5. attempt_fusion',      'fusion'),
    ('6. apply_composite_decay', 'decay'),
    ('7. energy_conservation', 'energy'),
]


def scale_sweep(Ns=(500, 1000, 2000, 4000), n_bench=30, n_runs=3):
    """
    Run profile_all_phases at each num_particles value and print a comparison table.
    Each N requires fresh JIT compilation — expect ~30-60s total.
    """
    rows = []
    for N in Ns:
        cfg   = make_config(N)
        prms  = initialize_interaction_params(cfg, seed=42)
        phys  = initialize_physics_params(cfg)
        print(f"\n--- Scale sweep: num_particles={N} ---", flush=True)
        results, full_ms, _ = profile_all_phases(cfg, phys, prms,
                                                  n_warmup=3, n_bench=n_bench,
                                                  n_runs=n_runs)
        row = {'N': N, 'full_step': full_ms}
        for long_key, short_key in _SWEEP_COLS[1:]:  # skip full_step (added above)
            row[short_key] = results.get(long_key, (0.0, 0.0))[0]
        rows.append(row)

    cols = [short for _, short in _SWEEP_COLS]
    col_w = 10
    header = f"  {'N':>6} | " + " | ".join(f"{c:>{col_w}}" for c in cols)
    sep    = "  " + "-" * (7 + (col_w + 3) * len(cols))
    print("\n\nScale Sweep Results (mean ms per phase):")
    print(header)
    print(sep)
    for row in rows:
        vals = " | ".join(f"{row.get(c, 0.0):>{col_w}.2f}" for c in cols)
        print(f"  {row['N']:>6} | {vals}")


# ── JAX Profiler Trace ────────────────────────────────────────────────────────

def run_trace(config, physics, params, output_dir="/tmp/halflife-trace"):
    """
    Capture a JAX/XLA profiler trace over 20 simulation steps.
    View the resulting .perfetto-trace file at ui.perfetto.dev.
    """
    os.makedirs(output_dir, exist_ok=True)
    state   = initialize_world(config, seed=0)
    step_fn = jax.jit(simulation_step, static_argnums=(2,))

    # Warm up outside the trace window
    print(f"\nWarming up for trace (5 steps)...", flush=True)
    for _ in range(5):
        state = step_fn(state, params, config, physics)
    state.particles.position.block_until_ready()

    print(f"Capturing trace to: {output_dir}  (20 steps)", flush=True)
    with jax.profiler.trace(output_dir, create_perfetto_link=True):
        for _ in range(20):
            state = step_fn(state, params, config, physics)
        state.particles.position.block_until_ready()

    print(f"\nTrace saved to: {output_dir}")
    print("Open ui.perfetto.dev → 'Open trace file' → select the .perfetto-trace file")


# ── Device Memory Profile ─────────────────────────────────────────────────────

def run_memory_profile(config, physics, params,
                       output_path="/tmp/halflife-memory.pb"):
    """
    Capture a device memory profile after 20 simulation steps.
    Saved as a pprof proto — view with:
        go install github.com/google/pprof@latest
        pprof -http=localhost:8080 /tmp/halflife-memory.pb
    """
    state   = initialize_world(config, seed=0)
    step_fn = jax.jit(simulation_step, static_argnums=(2,))
    for _ in range(20):
        state = step_fn(state, params, config, physics)
    state.particles.position.block_until_ready()

    profile_bytes = jax.profiler.device_memory_profile()
    with open(output_path, 'wb') as f:
        f.write(profile_bytes)

    print(f"\nMemory profile saved to: {output_path}")
    print(f"View with: pprof -http=localhost:8080 {output_path}")
    print("(Install pprof: go install github.com/google/pprof@latest)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Half-Life Particle Profiler")
    p.add_argument('--n', type=int, default=None,
                   help='num_particles override (default: SimConfig default = 2000)')
    p.add_argument('--scale-sweep', action='store_true',
                   help='Run at num_particles={500,1000,2000,4000} and print comparison')
    p.add_argument('--trace', metavar='DIR', nargs='?', const='/tmp/halflife-trace',
                   help='Emit JAX Perfetto trace to DIR (view at ui.perfetto.dev)')
    p.add_argument('--memory', action='store_true',
                   help='Save device memory profile to /tmp/halflife-memory.pb')
    p.add_argument('--n-bench', type=int, default=50,
                   help='Timed iterations per phase per run (default: 50)')
    p.add_argument('--n-runs', type=int, default=3,
                   help='Independent timing runs per phase; std is across run-means (default: 3)')
    args = p.parse_args()

    config  = make_config(args.n) if args.n is not None else SimConfig()
    params  = initialize_interaction_params(config, seed=42)
    physics = initialize_physics_params(config)

    print("=" * 62)
    print("Half-Life Particle Profiler")
    print("=" * 62)
    print(f"Device:           {jax.devices()[0]}")
    print(f"num_particles:    {config.num_particles}")
    print(f"max_composites:   {config.max_composites}")
    print(f"max_composite_size: {config.max_composite_size}")
    print(f"max_neighbors:    {config.max_neighbors}  "
          f"cell_capacity: {config.cell_capacity}")
    print(f"use_bond_forces:  {config.use_bond_forces}")

    # Always run the per-phase breakdown
    results, full_ms, full_std = profile_all_phases(
        config, physics, params, n_warmup=3, n_bench=args.n_bench, n_runs=args.n_runs
    )
    print_phase_table(results, full_ms, full_std, config)

    if args.scale_sweep:
        scale_sweep(n_bench=max(10, args.n_bench // 3), n_runs=args.n_runs)

    if args.trace is not None:
        run_trace(config, physics, params, output_dir=args.trace)

    if args.memory:
        run_memory_profile(config, physics, params)


if __name__ == '__main__':
    main()
