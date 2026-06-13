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
