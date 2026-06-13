"""
tests/test_liquid_drop.py — fissility-based composite stability.

Phase C of the fission redesign: cohesion (Σ bond E_b − surface term) vs
disruption (internal hard-core repulsion PE, the Coulomb analog) sets a live
half-life that collapses as fissility x → 1.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax.numpy as jnp
import numpy as np

from halflife.config import SimConfig
from halflife.state import (initialize_world, initialize_interaction_params,
                            initialize_physics_params)


def _two_member_world(config, gap, species_val=0):
    """One 2-member composite with member distance `gap`; rest parked far away.

    species_val sets both members' species — pick a high-E_b self-pair (e.g.
    2 at num_species=3) when the test needs positive cohesion.
    """
    N = config.num_particles
    world = initialize_world(config, seed=0)
    pos = np.array([[5.0, 5.0], [5.0 + gap, 5.0]]
                   + [[30.0 + i, 30.0] for i in range(N - 2)], dtype=np.float32)
    composite_id = np.full(N, -1, dtype=np.int32)
    composite_id[:2] = 0
    members = np.full((config.max_composites, config.max_composite_size), -1, dtype=np.int32)
    members[0, :2] = (0, 1)
    edges = np.full((config.max_composites, config.e_max, 2), -1, dtype=np.int32)
    edges[0, 0] = (0, 1)
    member_count = np.zeros(config.max_composites, dtype=np.int32); member_count[0] = 2
    edge_count = np.zeros(config.max_composites, dtype=np.int32); edge_count[0] = 1
    alive = np.zeros(config.max_composites, dtype=bool); alive[0] = True
    species = np.full(N, species_val, dtype=np.int32)
    return world._replace(
        particles=world.particles._replace(
            position=jnp.asarray(pos),
            species=jnp.asarray(species),
            composite_id=jnp.asarray(composite_id),
        ),
        composites=world.composites._replace(
            members=jnp.asarray(members), member_count=jnp.asarray(member_count),
            alive=jnp.asarray(alive), edges=jnp.asarray(edges),
            edge_count=jnp.asarray(edge_count),
        ),
    )


def test_internal_repulsion_pe_closed_form():
    """Two same-composite members at r < repulsion_radius: per-composite PE
    (sum of per-particle halves) equals R·(rr − r)²/(2·rr)."""
    from halflife.interactions import compute_all_forces
    from halflife.spatial import build_cell_list, find_all_neighbors
    config = SimConfig(num_particles=10, max_composites=4, num_species=3,
                       boundary_mode="reflect", world_width=40.0, world_height=40.0)
    physics = initialize_physics_params(config)
    params = initialize_interaction_params(config, seed=0)
    gap = 0.4    # inside repulsion_radius = 0.8
    world = _two_member_world(config, gap)

    cell_list = build_cell_list(world.particles.position, config)
    neighbors = find_all_neighbors(world.particles.position, cell_list, config)
    forces, rep_pe = compute_all_forces(
        world.particles.position, world.particles.species,
        world.particles.composite_id, neighbors, params, config, physics)

    rr = config.repulsion_radius
    expected_pair_pe = config.repulsion_strength * (rr - gap) ** 2 / (2.0 * rr)
    # both members see the pair → per-particle sum is 2× the pair PE
    total = float(rep_pe[0] + rep_pe[1])
    assert np.isclose(total * 0.5, expected_pair_pe, rtol=1e-4), \
        f"got {total * 0.5}, expected {expected_pair_pe}"
    # free particles far away contribute nothing
    assert float(jnp.sum(rep_pe[2:])) == 0.0
    assert forces.shape == (10, 2)


def test_repulsion_pe_zero_for_free_and_distant():
    """Same-composite members OUTSIDE the hard core have zero repulsion PE."""
    from halflife.interactions import compute_all_forces
    from halflife.spatial import build_cell_list, find_all_neighbors
    config = SimConfig(num_particles=10, max_composites=4, num_species=3,
                       boundary_mode="reflect", world_width=40.0, world_height=40.0)
    physics = initialize_physics_params(config)
    params = initialize_interaction_params(config, seed=0)
    world = _two_member_world(config, gap=1.5)   # outside rr = 0.8

    cell_list = build_cell_list(world.particles.position, config)
    neighbors = find_all_neighbors(world.particles.position, cell_list, config)
    _, rep_pe = compute_all_forces(
        world.particles.position, world.particles.species,
        world.particles.composite_id, neighbors, params, config, physics)
    assert float(jnp.sum(rep_pe)) == 0.0


def test_crammed_composite_gets_shorter_half_life():
    """Same 2-member composite; the overlapping one (r < rr) must get a
    shorter live half-life than the relaxed one."""
    from halflife.chemistry import compute_liquid_drop_half_life
    from halflife.interactions import compute_all_forces
    from halflife.spatial import build_cell_list, find_all_neighbors
    config = SimConfig(num_particles=10, max_composites=4, num_species=3,
                       boundary_mode="reflect", world_width=40.0, world_height=40.0)
    physics = initialize_physics_params(config)
    params = initialize_interaction_params(config, seed=0)

    def live_hl(gap):
        # Species 2 self-pair has high E_b (6.1 >> surface term) → positive
        # cohesion, so t_coh > 0 and the fissility term can discriminate.
        world = _two_member_world(config, gap, species_val=2)
        cell_list = build_cell_list(world.particles.position, config)
        neighbors = find_all_neighbors(world.particles.position, cell_list, config)
        _, rep_pe = compute_all_forces(
            world.particles.position, world.particles.species,
            world.particles.composite_id, neighbors, params, config, physics)
        hl = compute_liquid_drop_half_life(
            world.particles, world.composites, rep_pe, config, physics)
        return float(hl[0])

    assert live_hl(0.2) < live_hl(1.5)


def test_zero_cohesion_collapses_to_min_half_life():
    """A composite whose surface term swamps its bond aggregate sits at
    half_life_min regardless of repulsion."""
    from halflife.chemistry import compute_liquid_drop_half_life
    config = SimConfig(num_particles=10, max_composites=4, num_species=3,
                       boundary_mode="reflect", world_width=40.0, world_height=40.0,
                       surface_energy_coeff=1e6)   # cohesion guaranteed negative
    physics = initialize_physics_params(config)
    world = _two_member_world(config, gap=1.5)
    rep_pe = jnp.zeros(config.num_particles, dtype=jnp.float32)
    hl = compute_liquid_drop_half_life(
        world.particles, world.composites, rep_pe, config, physics)
    assert np.isclose(float(hl[0]), config.half_life_min, rtol=1e-5)


def test_legacy_mode_leaves_half_life_alone():
    """stability_mode='legacy': one simulation step must not rewrite a
    non-decaying composite's stored half-life."""
    import jax
    from halflife.step import simulation_step
    # Scission off so the dimer survives the step (it would otherwise snap if
    # its hash-derived E_b were below the stretch strain at gap=1.5); this
    # test is about legacy stability_mode leaving half_life untouched.
    config = SimConfig(num_particles=10, max_composites=4, num_species=3,
                       boundary_mode="reflect", world_width=40.0, world_height=40.0,
                       stability_mode="legacy", enable_bond_scission=False)
    physics = initialize_physics_params(config)
    params = initialize_interaction_params(config, seed=0)
    world = _two_member_world(config, gap=1.5)
    sentinel = 12345.0
    hl = np.zeros(config.max_composites, dtype=np.float32)
    hl[0] = sentinel
    world = world._replace(composites=world.composites._replace(
        half_life=jnp.asarray(hl)))
    step_fn = jax.jit(simulation_step, static_argnums=(2,))
    state = step_fn(world, params, config, physics)
    # composite 0 survives (hl huge) and keeps its stored half-life
    assert bool(state.composites.alive[0])
    assert np.isclose(float(state.composites.half_life[0]), sentinel)
