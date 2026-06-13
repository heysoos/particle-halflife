"""Tests for the VSEPR / harmonic angle-locking kernel (halflife/step.py).

See docs/superpowers/plans/2026-06-12-vsepr-angle-kernel.md.
Run with: JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_angle_kernel.py -q
"""
import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np

from halflife.config import SimConfig
from halflife.state import initialize_world, initialize_physics_params


def _world_with_edges(edge_pairs, num_particles=8, positions=None):
    """Build a WorldState whose composite 0 contains the given undirected edges.

    edge_pairs: list of (i, j) particle ids. All referenced ids become members.
    positions:  optional {pid: (x, y)} overrides for member particle positions.

    Note: e_max is a computed @property of SimConfig (derived from
    max_composite_size * max_valence), so it is NOT passed to replace().
    """
    c = dataclasses.replace(SimConfig(), num_particles=num_particles,
                            max_composites=4, max_composite_size=8,
                            max_valence=4, bond_mode="edges")
    state = initialize_world(c, seed=0)
    members = sorted({p for e in edge_pairs for p in e})
    comp = state.composites
    E = c.e_max
    edges = np.full((c.max_composites, E, 2), -1, np.int32)
    for k, (i, j) in enumerate(edge_pairs):
        edges[0, k] = (i, j)
    mem = np.full((c.max_composites, c.max_composite_size), -1, np.int32)
    mem[0, :len(members)] = members
    comp = comp._replace(
        alive=comp.alive.at[0].set(True),
        members=jnp.asarray(mem),
        member_count=comp.member_count.at[0].set(len(members)),
        edges=jnp.asarray(edges),
        edge_count=comp.edge_count.at[0].set(len(edge_pairs)),
    )
    parts = state.particles._replace(
        composite_id=state.particles.composite_id.at[jnp.asarray(members)].set(0),
    )
    if positions is not None:
        pos = np.asarray(state.particles.position)
        for pid, xy in positions.items():
            pos[pid] = xy
        parts = parts._replace(position=jnp.asarray(pos, jnp.float32))
    return state._replace(composites=comp, particles=parts), c


# ── Task 1: config + runtime k_angle ─────────────────────────────────────────

def test_angle_config_defaults():
    c = SimConfig()
    assert c.angle_mode == "off"          # default: existing behaviour unchanged
    assert c.k_angle == 10.0
    assert c.theta_min_deg == 90.0
    assert c.theta_max_deg == 180.0


def test_physics_params_seeds_k_angle():
    c = dataclasses.replace(SimConfig(), k_angle=7.5)
    p = initialize_physics_params(c)
    assert float(p.k_angle) == 7.5


# ── Task 2: hash-derived rest angle θ0 ───────────────────────────────────────

def test_rest_angle_deterministic_and_in_band():
    from halflife.chemistry import _hash_to_rest_angle, _species_rest_angles
    c = dataclasses.replace(SimConfig(), num_species=6)
    lo, hi = math.radians(c.theta_min_deg), math.radians(c.theta_max_deg)
    angles = np.asarray(_species_rest_angles(c))
    assert angles.shape == (6,)
    assert np.all(angles >= lo - 1e-6) and np.all(angles <= hi + 1e-6)
    # deterministic per species index
    assert float(_hash_to_rest_angle(jnp.int32(3), c)) == float(angles[3])


def test_rest_angle_decorrelated_from_valence():
    from halflife.chemistry import _species_rest_angles, _species_valences
    # Different hash stream → θ0 ordering differs from valence ordering.
    c = dataclasses.replace(SimConfig(), num_species=12, max_valence=4)
    ang = np.asarray(_species_rest_angles(c))
    val = np.asarray(_species_valences(c))
    assert not np.array_equal(np.argsort(ang), np.argsort(val))


# ── Task 3: per-particle neighbor list ───────────────────────────────────────

def _nbr_set(nbrs, pid):
    row = np.asarray(nbrs[pid])
    return set(int(x) for x in row if x >= 0)


def test_neighbor_list_chain():
    from halflife.step import build_neighbor_list
    # chain 1-2-3: 2 is central (neighbors 1,3); ends have one neighbor
    state, c = _world_with_edges([(1, 2), (2, 3)])
    nbrs = build_neighbor_list(state.composites, c)
    assert _nbr_set(nbrs, 1) == {2}
    assert _nbr_set(nbrs, 2) == {1, 3}
    assert _nbr_set(nbrs, 3) == {2}
    assert _nbr_set(nbrs, 5) == set()      # free particle


def test_neighbor_list_branch_and_ring():
    from halflife.step import build_neighbor_list
    # star: 0 bonded to 1,2,3 (degree 3)
    state, c = _world_with_edges([(0, 1), (0, 2), (0, 3)])
    nbrs = build_neighbor_list(state.composites, c)
    assert _nbr_set(nbrs, 0) == {1, 2, 3}
    # triangle ring 4-5-6
    state, c = _world_with_edges([(4, 5), (5, 6), (6, 4)])
    nbrs = build_neighbor_list(state.composites, c)
    assert _nbr_set(nbrs, 4) == {5, 6}
    assert _nbr_set(nbrs, 5) == {4, 6}
