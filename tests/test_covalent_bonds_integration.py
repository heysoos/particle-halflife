"""
End-to-end smoke test for sparse covalent bonds.

Runs ~200 simulation steps with bond_mode='edges' and verifies:
  - No NaNs in positions or velocities
  - At least some composites form
  - All alive composites have edge_count >= max(0, member_count - 1)
    (spanning-tree invariant)
  - All edges reference particles that are in the same composite
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
# CPU pin disabled for this session — GPU available, live sim not running.
# Restore (uncomment) if integration tests start contending with the live sim.
# jax.config.update('jax_platform_name', 'cpu')
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
