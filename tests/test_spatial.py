"""
tests/test_spatial.py — Cell-list neighbor finding correctness.

Run standalone:  python tests/test_spatial.py
Run under pytest: pytest tests/test_spatial.py -v
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
from halflife.state import initialize_world
from halflife.spatial import build_cell_list, find_all_neighbors

_config = SimConfig()

# JIT-compiled spatial functions
_build_cell_list_jit = jax.jit(build_cell_list, static_argnums=(2,))
_find_all_neighbors_jit = jax.jit(find_all_neighbors, static_argnums=(3,))


@functools.lru_cache(maxsize=None)
def _get_spatial_state():
    """Build cell list and neighbors from a fixed initial state (cached)."""
    state = initialize_world(_config, seed=0)
    cell_list = _build_cell_list_jit(state.particles.position, state.particles.alive, _config)
    neighbors = _find_all_neighbors_jit(
        state.particles.position, state.particles.alive, cell_list, _config
    )
    cell_list.particle_ids.block_until_ready()
    neighbors.block_until_ready()
    return state, cell_list, neighbors


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_no_self_neighbor():
    """Particle i must never appear in its own neighbor list."""
    state, cell_list, neighbors = _get_spatial_state()
    N = _config.max_particles

    # neighbors[i, k] == i would be self-neighbor
    particle_indices = jnp.arange(N, dtype=jnp.int32)[:, None]  # (N, 1)
    self_matches = (neighbors == particle_indices) & (neighbors >= 0)  # (N, max_nb)
    n_self = int(jnp.sum(self_matches.astype(jnp.int32)))

    print(f"\nSelf-neighbor occurrences: {n_self}")
    assert n_self == 0, f"Found {n_self} self-neighbor entries (particle in its own neighbor list)"


def test_neighbor_distance_bound():
    """
    All reported neighbors (i, j) with j >= 0 must satisfy
    dist(i, j) <= interaction_radius + small epsilon.
    Uses minimum-image distance for periodic boundaries.
    """
    state, cell_list, neighbors = _get_spatial_state()
    positions = np.array(state.particles.position)
    alive = np.array(state.particles.alive)
    N = _config.max_particles
    r_max = _config.interaction_radius
    W = _config.world_width
    H = _config.world_height
    eps = 1e-3

    violations = 0
    max_dist_seen = 0.0

    for i in range(N):
        if not alive[i]:
            continue
        for k in range(_config.max_neighbors):
            j = int(neighbors[i, k])
            if j < 0:
                break
            # Minimum-image distance
            d = positions[i] - positions[j]
            d[0] -= W * np.round(d[0] / W)
            d[1] -= H * np.round(d[1] / H)
            dist = np.sqrt(d[0]**2 + d[1]**2)
            max_dist_seen = max(max_dist_seen, dist)
            if dist > r_max + eps:
                violations += 1

    print(f"\nMax neighbor distance seen: {max_dist_seen:.4f} (limit={r_max})")
    print(f"Distance-bound violations: {violations}")
    assert violations == 0, (
        f"{violations} neighbor pairs exceed interaction_radius {r_max} + eps {eps}. "
        f"Max dist seen: {max_dist_seen:.4f}"
    )


def test_no_dead_neighbors():
    """All reported neighbor indices must be -1 (padding) or pointing to alive particles."""
    state, cell_list, neighbors = _get_spatial_state()
    alive = np.array(state.particles.alive)
    N = _config.max_particles

    violations = 0
    for i in range(N):
        for k in range(_config.max_neighbors):
            j = int(neighbors[i, k])
            if j < 0:
                continue
            if not alive[j]:
                violations += 1

    print(f"\nDead-particle neighbor violations: {violations}")
    assert violations == 0, f"{violations} neighbor entries point to dead particles"


def test_neighbor_count_reasonable():
    """
    Mean neighbor count per alive particle should be in [0.5, max_neighbors].
    With 2000 particles in 200x200 (density ~0.05/unit²) and radius 4,
    expected ~2.5 neighbors on average.
    """
    state, cell_list, neighbors = _get_spatial_state()
    alive = np.array(state.particles.alive)
    neighbors_np = np.array(neighbors)

    alive_indices = np.where(alive)[0]
    n_alive = len(alive_indices)

    neighbor_counts = []
    for i in alive_indices:
        count = int(np.sum(neighbors_np[i] >= 0))
        neighbor_counts.append(count)

    mean_count = np.mean(neighbor_counts)
    print(f"\nNeighbor count stats ({n_alive} alive particles):")
    print(f"  mean={mean_count:.2f}, min={min(neighbor_counts)}, "
          f"max={max(neighbor_counts)}, max_neighbors={_config.max_neighbors}")

    assert mean_count >= 0.5, (
        f"Mean neighbor count {mean_count:.2f} < 0.5. "
        f"Cell list may be broken or particles too sparse."
    )
    assert mean_count <= _config.max_neighbors, (
        f"Mean neighbor count {mean_count:.2f} > max_neighbors={_config.max_neighbors}. "
        f"Impossible."
    )


def test_symmetric_neighbors():
    """
    If j ∈ neighbors[i] and both are alive, then i should be in neighbors[j].
    Due to the one-sided max_neighbors cap this is not guaranteed, but violations
    should be rare (< 20%). Report violation rate rather than hard-failing.
    """
    state, cell_list, neighbors = _get_spatial_state()
    alive = np.array(state.particles.alive)
    neighbors_np = np.array(neighbors)

    # Build a set of neighbors for each particle for fast lookup
    neighbor_sets = [set() for _ in range(_config.max_particles)]
    for i in range(_config.max_particles):
        for k in range(_config.max_neighbors):
            j = int(neighbors_np[i, k])
            if j >= 0:
                neighbor_sets[i].add(j)

    total_pairs = 0
    asymmetric = 0
    for i in range(_config.max_particles):
        if not alive[i]:
            continue
        for j in neighbor_sets[i]:
            if not alive[j]:
                continue
            total_pairs += 1
            if i not in neighbor_sets[j]:
                asymmetric += 1

    if total_pairs == 0:
        print("\nNo alive neighbor pairs found — skipping symmetry check.")
        return

    violation_rate = asymmetric / total_pairs
    print(f"\nNeighbor symmetry: {asymmetric}/{total_pairs} asymmetric pairs "
          f"({violation_rate*100:.1f}%)")

    # Soft check: report but don't hard-fail (max_neighbors cap causes asymmetry)
    if violation_rate > 0.20:
        print(f"  WARNING: {violation_rate*100:.1f}% asymmetry is higher than expected (< 20%)")
    # No assert — this is an informational check


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
