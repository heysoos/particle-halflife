"""Tests for the liquid-drop Coulomb-analog disruption term (R_g monopole).

See docs/superpowers/plans/2026-06-13-liquid-drop-disruption.md.
Run: JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_liquid_drop_disruption.py -q
"""
import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np

from halflife.config import SimConfig
from halflife.state import initialize_world, initialize_physics_params


def _composite_world(member_pos, edges, num_particles=20,
                     world=(200.0, 112.5), boundary="periodic", species_val=2):
    """WorldState with composite 0 = the given members at the given positions.

    member_pos: {pid: (x, y)} for the composite's members.
    edges:      list of (i, j) bond pairs among those members.
    Other particles are parked far away as free particles.
    """
    c = dataclasses.replace(
        SimConfig(), num_particles=num_particles, max_composites=4,
        max_composite_size=16, max_valence=4, num_species=3,
        bond_mode="edges", boundary_mode=boundary,
        world_width=world[0], world_height=world[1],
    )
    state = initialize_world(c, seed=0)
    members = sorted(member_pos)
    pos = np.array(state.particles.position)          # writable copy
    for pid, xy in member_pos.items():
        pos[pid] = xy
    species = np.full(num_particles, species_val, np.int32)   # high-E_b self-pair
    cid = np.full(num_particles, -1, np.int32)
    for pid in members:
        cid[pid] = 0
    mem = np.full((c.max_composites, c.max_composite_size), -1, np.int32)
    mem[0, :len(members)] = members
    E = c.e_max
    edge_arr = np.full((c.max_composites, E, 2), -1, np.int32)
    for k, (i, j) in enumerate(edges):
        edge_arr[0, k] = (i, j)
    comp = state.composites._replace(
        alive=state.composites.alive.at[0].set(True),
        members=jnp.asarray(mem),
        member_count=state.composites.member_count.at[0].set(len(members)),
        edges=jnp.asarray(edge_arr),
        edge_count=state.composites.edge_count.at[0].set(len(edges)),
    )
    parts = state.particles._replace(
        position=jnp.asarray(pos, jnp.float32),
        species=jnp.asarray(species),
        composite_id=jnp.asarray(cid),
    )
    return state._replace(composites=comp, particles=parts), c


def _grid_members(k, spacing=1.0, x0=20.0, y0=20.0):
    """k×k grid of members + grid edges (right/down neighbors). Returns
    (member_pos dict keyed by pid 0..k²-1, edges list)."""
    pos, edges = {}, []
    idx = lambda r, col: r * k + col
    for r in range(k):
        for col in range(k):
            pos[idx(r, col)] = (x0 + col * spacing, y0 + r * spacing)
            if col + 1 < k:
                edges.append((idx(r, col), idx(r, col + 1)))
            if r + 1 < k:
                edges.append((idx(r, col), idx(r + 1, col)))
    return pos, edges


# ── Task 1: config + runtime scalars ─────────────────────────────────────────

def test_disruption_config_defaults():
    c = SimConfig()
    assert c.disruption_scale == 0.5
    assert c.cohesion_hl_scale == 5.0          # raised from 1.0 so t_coh de-saturates


def test_physics_params_seeds_disruption_scalars():
    c = dataclasses.replace(SimConfig(), disruption_scale=0.7, cohesion_hl_scale=4.0)
    p = initialize_physics_params(c)
    assert np.isclose(float(p.disruption_scale), 0.7)   # float32 round-trip
    assert np.isclose(float(p.cohesion_hl_scale), 4.0)
