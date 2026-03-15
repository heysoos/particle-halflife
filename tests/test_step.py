"""
tests/test_step.py — Full simulation step integration tests.

Verifies the simulation step doesn't produce NaN/Inf and respects
physical invariants (positions in bounds, velocity clamped, energy bounded).

Run standalone:  python tests/test_step.py
Run under pytest: pytest tests/test_step.py -v
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
from halflife.state import initialize_world, initialize_interaction_params
from halflife.step import simulation_step
from halflife.energy import compute_total_energy

_config = SimConfig()
_params = initialize_interaction_params(_config, seed=42)
_step_jit = jax.jit(simulation_step, static_argnums=(2,))


@functools.lru_cache(maxsize=None)
def _run_cached(n_steps: int, seed: int = 0):
    """Run n_steps from seed, cache result (JIT warm-up included)."""
    config = _config
    params = _params
    state = initialize_world(config, seed=seed)
    state = _step_jit(state, params, config)
    state.particles.alive.block_until_ready()
    for _ in range(n_steps - 1):
        state = _step_jit(state, params, config)
    state.particles.alive.block_until_ready()
    return state


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_no_nan_in_positions():
    """After 100 steps, position, velocity, and energy must contain no NaN/Inf."""
    state = _run_cached(100)
    particles = state.particles
    alive = np.array(particles.alive)

    pos = np.array(particles.position)
    vel = np.array(particles.velocity)
    energy = np.array(particles.energy)

    nan_pos = np.sum(~np.isfinite(pos[alive]))
    nan_vel = np.sum(~np.isfinite(vel[alive]))
    nan_energy = np.sum(~np.isfinite(energy[alive]))

    print(f"\nNaN/Inf check after 100 steps:")
    print(f"  position: {nan_pos}, velocity: {nan_vel}, energy: {nan_energy}")

    assert nan_pos == 0, f"{nan_pos} NaN/Inf values in position after 100 steps"
    assert nan_vel == 0, f"{nan_vel} NaN/Inf values in velocity after 100 steps"
    assert nan_energy == 0, f"{nan_energy} NaN/Inf values in energy after 100 steps"


def test_positions_in_bounds():
    """
    With periodic boundaries, all alive particle positions must stay in
    [0, world_width) × [0, world_height).
    """
    state = _run_cached(100)
    pos = np.array(state.particles.position)
    alive = np.array(state.particles.alive)
    W = _config.world_width
    H = _config.world_height
    eps = 1e-3  # small tolerance for float rounding at exactly 0 / W

    alive_pos = pos[alive]
    out_x = np.sum((alive_pos[:, 0] < -eps) | (alive_pos[:, 0] >= W + eps))
    out_y = np.sum((alive_pos[:, 1] < -eps) | (alive_pos[:, 1] >= H + eps))

    print(f"\nPosition bounds check: out_x={out_x}, out_y={out_y} "
          f"(world={W}x{H}, {alive.sum()} alive)")
    assert out_x == 0, f"{out_x} particles have x outside [0, {W}]"
    assert out_y == 0, f"{out_y} particles have y outside [0, {H}]"


def test_velocity_clamped():
    """All alive particle velocities must have |v| <= max_velocity + epsilon."""
    state = _run_cached(100)
    vel = np.array(state.particles.velocity)
    alive = np.array(state.particles.alive)
    max_v = _config.max_velocity
    eps = 0.1  # small tolerance for in-step computation before clamp

    speeds = np.linalg.norm(vel[alive], axis=1)
    violations = np.sum(speeds > max_v + eps)
    max_speed = speeds.max() if len(speeds) > 0 else 0.0

    print(f"\nVelocity clamp check: max_speed={max_speed:.3f}, max_velocity={max_v}, "
          f"violations={violations}")
    assert violations == 0, (
        f"{violations} alive particles have speed > {max_v + eps:.2f}. "
        f"Max speed: {max_speed:.3f}"
    )


def test_energy_bounded():
    """
    Total energy must not diverge to infinity or collapse to near-zero over 200 steps.
    Damping (0.995/step) causes some energy loss; soft conservation partially compensates.
    We accept up to 90% loss (damping is expected) but reject divergence or total collapse.
    """
    config = _config
    params = _params
    state = initialize_world(config, seed=0)
    step_fn = _step_jit

    # Warm up JIT
    state = step_fn(state, params, config)
    state.particles.alive.block_until_ready()
    initial_energy = float(compute_total_energy(state))

    if initial_energy < 1e-6:
        print("\nInitial energy is near zero — skipping energy drift check.")
        return

    for _ in range(199):
        state = step_fn(state, params, config)

    final_energy = float(compute_total_energy(state))
    ratio = final_energy / (initial_energy + 1e-8)

    print(f"\nEnergy: initial={initial_energy:.3f}, final={final_energy:.3f}, "
          f"ratio={ratio:.3f} (1.0=conserved, <1=damped, >1=grew)")

    assert final_energy >= 0, f"Total energy went negative: {final_energy:.3f}"
    assert ratio < 10.0, (
        f"Energy grew {ratio:.1f}x over 200 steps. Runaway instability."
    )
    assert ratio > 0.05, (
        f"Energy collapsed to {ratio*100:.1f}% of initial. "
        f"Particles may all be dead or velocities zeroed out."
    )


def test_step_count_increments():
    """step_count must increase by exactly 1 per call."""
    config = _config
    params = _params
    state = initialize_world(config, seed=5)
    step_fn = _step_jit

    state = step_fn(state, params, config)
    state.step_count.block_until_ready()

    for expected in range(2, 12):
        prev = int(state.step_count)
        state = step_fn(state, params, config)
        curr = int(state.step_count)
        assert curr == prev + 1, (
            f"step_count did not increment by 1: {prev} → {curr}"
        )

    print(f"\nstep_count after 11 steps: {int(state.step_count)}")


def test_composite_id_valid():
    """
    After 100 steps:
    - Alive particles with composite_id >= 0 must point to alive composites.
    - Alive particles with composite_id < 0 are free (no constraint).
    """
    state = _run_cached(100)
    particles = state.particles
    composites = state.composites
    alive = np.array(particles.alive)
    cid = np.array(particles.composite_id)
    comp_alive = np.array(composites.alive)

    errors = []
    for i in np.where(alive)[0]:
        c = cid[i]
        if c >= 0:
            if c >= _config.max_composites:
                errors.append(f"Particle {i}: composite_id={c} >= max_composites={_config.max_composites}")
            elif not comp_alive[c]:
                errors.append(f"Particle {i}: composite_id={c} points to dead composite")

    print(f"\nComposite ID validity check: {len(errors)} errors among {alive.sum()} alive particles")
    if errors:
        for e in errors[:5]:
            print(f"  {e}")

    assert not errors, f"{len(errors)} composite_id validity errors (first: {errors[0]})"


def test_reset_deterministic():
    """
    Two independent runs from the same seed must produce identical states after 50 steps.
    """
    config = _config
    params_a = initialize_interaction_params(config, seed=42)
    params_b = initialize_interaction_params(config, seed=42)
    step_fn = _step_jit

    state_a = initialize_world(config, seed=7)
    state_b = initialize_world(config, seed=7)

    # Warm up with both (same operations)
    state_a = step_fn(state_a, params_a, config)
    state_b = step_fn(state_b, params_b, config)
    state_a.particles.alive.block_until_ready()
    state_b.particles.alive.block_until_ready()

    for _ in range(49):
        state_a = step_fn(state_a, params_a, config)
        state_b = step_fn(state_b, params_b, config)

    pos_a = np.array(state_a.particles.position)
    pos_b = np.array(state_b.particles.position)
    alive_a = np.array(state_a.particles.alive)
    alive_b = np.array(state_b.particles.alive)

    pos_max_diff = np.max(np.abs(pos_a - pos_b))
    alive_match = np.all(alive_a == alive_b)

    print(f"\nDeterminism check after 50 steps: max_pos_diff={pos_max_diff:.6f}, "
          f"alive_match={alive_match}")
    assert alive_match, "Alive masks differ between two identical runs"
    assert pos_max_diff < 1e-5, (
        f"Max position difference {pos_max_diff:.6f} > 1e-5. Run is not deterministic."
    )


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
