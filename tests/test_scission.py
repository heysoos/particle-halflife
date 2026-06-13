"""
tests/test_scission.py — chemical (per-bond) breaking channel.

Kinetic break: stretch strain >= E_b snaps the bond deterministically.
Thermal break: Arrhenius roll below threshold (kT → 0 disables it).
A broken bridge splits the composite; a broken ring edge only removes the edge.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax.numpy as jnp
import numpy as np

from halflife.config import SimConfig
from halflife.state import (initialize_world, initialize_interaction_params,
                            initialize_physics_params)
from halflife.chemistry import apply_bond_scission

# kT tiny → thermal channel off; only deterministic kinetic breaks fire.
_BASE = dict(num_species=3, num_particles=10, max_composites=4,
             boundary_mode="reflect", world_width=40.0, world_height=40.0,
             half_life_min=1e9, half_life_max=1e9,   # no half-life decay noise
             bond_temperature=1e-7)


def _world_with_composite(config, pos, edge_pairs, species=None):
    n = max(max(p) for p in edge_pairs) + 1
    N = config.num_particles
    world = initialize_world(config, seed=0)
    if species is None:
        species = np.zeros(N, dtype=np.int32)
    composite_id = np.full(N, -1, dtype=np.int32)
    composite_id[:n] = 0
    members = np.full((config.max_composites, config.max_composite_size), -1, dtype=np.int32)
    members[0, :n] = np.arange(n)
    edges = np.full((config.max_composites, config.e_max, 2), -1, dtype=np.int32)
    for k, (a, b) in enumerate(edge_pairs):
        edges[0, k] = (a, b)
    member_count = np.zeros(config.max_composites, dtype=np.int32)
    member_count[0] = n
    edge_count = np.zeros(config.max_composites, dtype=np.int32)
    edge_count[0] = len(edge_pairs)
    alive = np.zeros(config.max_composites, dtype=bool)
    alive[0] = True
    hl = np.full(config.max_composites, 1e9, dtype=np.float32)
    return world._replace(
        particles=world.particles._replace(
            position=jnp.asarray(pos), species=jnp.asarray(species),
            composite_id=jnp.asarray(composite_id),
            velocity=jnp.zeros((N, 2), dtype=jnp.float32),
        ),
        composites=world.composites._replace(
            members=jnp.asarray(members), member_count=jnp.asarray(member_count),
            alive=jnp.asarray(alive), edges=jnp.asarray(edges),
            edge_count=jnp.asarray(edge_count), half_life=jnp.asarray(hl),
        ),
    )


def test_kinetic_break_snaps_overstretched_dimer():
    """Stretch a dimer's bond way past r_rest: strain >> bond_energy_scale →
    deterministic snap, both particles free, slot dead."""
    config = SimConfig(**_BASE)
    pos = np.array([[5.0, 5.0], [12.0, 5.0]]      # 7 units apart, r_rest <= 1.5
                   + [[30.0 + i, 30.0] for i in range(8)], dtype=np.float32)
    world = _world_with_composite(config, pos, [(0, 1)])
    params = initialize_interaction_params(config, seed=0)
    physics = initialize_physics_params(config)

    state = apply_bond_scission(world, params, config, physics)

    assert not bool(state.composites.alive[0])
    cid = np.asarray(state.particles.composite_id)
    assert cid[0] == -1 and cid[1] == -1


def test_no_break_at_rest_length():
    """A dimer sitting exactly at its rest length has zero strain; with kT≈0
    the thermal channel is dead too → nothing breaks over many calls."""
    config = SimConfig(**_BASE)
    params = initialize_interaction_params(config, seed=0)
    physics = initialize_physics_params(config)
    r01 = float(params.r_rest[0, 0])
    pos = np.array([[5.0, 5.0], [5.0 + r01, 5.0]]
                   + [[30.0 + i, 30.0] for i in range(8)], dtype=np.float32)
    world = _world_with_composite(config, pos, [(0, 1)])

    state = world
    for _ in range(50):
        state = apply_bond_scission(state, params, config, physics)
    assert bool(state.composites.alive[0])
    assert int(state.composites.edge_count[0]) == 1


def test_ring_edge_break_removes_edge_without_split():
    """Triangle with ONE overstretched edge: the edge goes, but the composite
    stays connected through the other two bonds — no split."""
    config = SimConfig(**_BASE)
    params = initialize_interaction_params(config, seed=0)
    physics = initialize_physics_params(config)
    # 0 and 1 far apart (their edge snaps); both near 2 (those bonds at ~rest).
    pos = np.array([[5.0, 5.0], [12.0, 5.0], [8.5, 5.5]]
                   + [[30.0 + i, 30.0] for i in range(7)], dtype=np.float32)
    # Species 1 has valence 3 (species 0 is valence 1), so the surviving
    # 3-member / 2-edge product has free_bonds = 3*3 − 2*2 = 5 >= 0 and stays
    # a composite. With species 0 it would be over-bonded and shatter.
    species = np.ones(10, dtype=np.int32)
    world = _world_with_composite(config, pos, [(0, 1), (1, 2), (2, 0)], species=species)

    state = apply_bond_scission(world, params, config, physics)

    assert bool(state.composites.alive[0])
    assert int(state.composites.member_count[0]) == 3
    assert int(state.composites.edge_count[0]) == 2
    edges_after = {tuple(sorted(e)) for e in
                   np.asarray(state.composites.edges[0][:2]).tolist()}
    assert edges_after == {(1, 2), (0, 2)}


def test_bridge_break_splits_chain_and_conserves_particles():
    """4-chain with the MIDDLE bond overstretched → two dimers; particle and
    species conservation; composite_id consistency."""
    config = SimConfig(**_BASE)
    params = initialize_interaction_params(config, seed=0)
    physics = initialize_physics_params(config)
    r00 = float(params.r_rest[0, 0])
    pos = np.array([[5.0, 5.0], [5.0 + r00, 5.0],          # bond 0-1 at rest
                    [12.0, 5.0], [12.0 + r00, 5.0]]        # bond 2-3 at rest
                   + [[30.0 + i, 30.0] for i in range(6)], dtype=np.float32)
    # middle bond (1,2) spans 12.0 − (5+r00) ≈ 6 units → snaps
    world = _world_with_composite(config, pos, [(0, 1), (1, 2), (2, 3)])
    species_before = np.asarray(world.particles.species).copy()

    state = apply_bond_scission(world, params, config, physics)

    cid = np.asarray(state.particles.composite_id)
    alive = np.asarray(state.composites.alive)
    counts = np.asarray(state.composites.member_count)
    # Two alive composites of size 2
    assert sorted(counts[alive].tolist()) == [2, 2]
    # {0,1} together, {2,3} together, in different composites
    assert cid[0] == cid[1] and cid[2] == cid[3] and cid[0] != cid[2]
    # conservation
    assert (np.asarray(state.particles.species) == species_before).all()
    for c in np.where(alive)[0]:
        mem = np.asarray(state.composites.members[c][:counts[c]])
        assert (np.sort(np.where(cid == c)[0]) == np.sort(mem)).all()
