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
        pos = np.array(state.particles.position)   # writable copy (JAX array is read-only)
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


# ── Task 4: angle (triple) list ──────────────────────────────────────────────

def test_angle_list_counts():
    from halflife.step import build_neighbor_list, build_angle_list
    # degree-2 center → 1 triple; degree-3 center → 3 triples (C(3,2))
    state, c = _world_with_edges([(1, 2), (2, 3)])
    nbrs = build_neighbor_list(state.composites, c)
    angles = build_angle_list(nbrs, c)                 # (N, P_max, 3)
    P = c.max_valence * (c.max_valence - 1) // 2
    assert angles.shape == (c.num_particles, P, 3)
    # particle 2 is the only valid center
    rows2 = np.asarray(angles[2])
    valid2 = rows2[(rows2[:, 0] >= 0)]
    assert valid2.shape[0] == 1                        # one triple
    assert set(valid2[0][[0, 2]]) == {1, 3}            # neighbors are 1,3
    assert valid2[0][1] == 2                           # center is 2

    state, c = _world_with_edges([(0, 1), (0, 2), (0, 3)])
    nbrs = build_neighbor_list(state.composites, c)
    angles = build_angle_list(nbrs, c)
    rows0 = np.asarray(angles[0])
    assert rows0[(rows0[:, 0] >= 0)].shape[0] == 3     # C(3,2) = 3 triples


def test_angle_list_free_particle_empty():
    from halflife.step import build_neighbor_list, build_angle_list
    state, c = _world_with_edges([(1, 2), (2, 3)])
    angles = build_angle_list(build_neighbor_list(state.composites, c), c)
    assert np.all(np.asarray(angles[6]) == -1)         # free particle, no triples


# ── Task 5: compute_angle_forces — VSEPR mode ────────────────────────────────

def test_vsepr_two_bonds_open_and_conserve():
    from halflife.step import compute_angle_forces
    # center 2 at origin; neighbor 1 at +x, neighbor 3 at +y → θ = 90°.
    # VSEPR should push 1 and 3 apart (toward 180°) and conserve momentum.
    state, c = _world_with_edges(
        [(1, 2), (2, 3)],
        positions={2: (5.0, 5.0), 1: (6.0, 5.0), 3: (5.0, 6.0)},
    )
    c = dataclasses.replace(c, angle_mode="vsepr", boundary_mode="open")
    phys = initialize_physics_params(c)
    F = np.asarray(compute_angle_forces(state, c, phys))
    # tangential opening: F on 1 has -y component, F on 3 has -x component
    assert F[1][1] < -1e-4
    assert F[3][0] < -1e-4
    # momentum conserved over the triple
    assert np.allclose(F[1] + F[2] + F[3], 0.0, atol=1e-4)


def test_vsepr_straight_is_equilibrium():
    from halflife.step import compute_angle_forces
    # 1-2-3 collinear (180°) → ~zero angle force, no NaN
    state, c = _world_with_edges(
        [(1, 2), (2, 3)],
        positions={2: (5.0, 5.0), 1: (4.0, 5.0), 3: (6.0, 5.0)},
    )
    c = dataclasses.replace(c, angle_mode="vsepr", boundary_mode="open")
    F = np.asarray(compute_angle_forces(state, c, initialize_physics_params(c)))
    assert np.all(np.isfinite(F))
    assert np.allclose(F[1], 0.0, atol=1e-4)


def test_vsepr_relaxes_three_bonds_to_Y():
    from halflife.step import compute_angle_forces
    # Integrate angle-only dynamics on a degree-3 star from a squished start;
    # the three pairwise angles should converge toward 120°.
    state, c = _world_with_edges(
        [(0, 1), (0, 2), (0, 3)],
        positions={0: (5., 5.), 1: (6., 5.0), 2: (6., 5.3), 3: (4., 5.)},
    )
    c = dataclasses.replace(c, angle_mode="vsepr", boundary_mode="open")
    phys = initialize_physics_params(c)
    pos = np.array(state.particles.position)   # writable copy
    for _ in range(400):
        F = np.asarray(compute_angle_forces(
            state._replace(particles=state.particles._replace(
                position=jnp.asarray(pos))), c, phys))
        for pid in (1, 2, 3):
            pos[pid] += 0.02 * F[pid]                       # tiny overdamped step
            v = pos[pid] - pos[0]
            pos[pid] = pos[0] + v / (np.linalg.norm(v) + 1e-9)  # keep |bond|≈1
    def ang(p, q):
        u = pos[p] - pos[0]; w = pos[q] - pos[0]
        return np.degrees(np.arccos(np.clip(u@w/(np.linalg.norm(u)*np.linalg.norm(w)), -1, 1)))
    angs = sorted([ang(1, 2), ang(1, 3), ang(2, 3)])
    assert angs[0] > 90 and abs(np.mean(angs) - 120) < 15


# ── Task 6: harmonic-θ0 mode ─────────────────────────────────────────────────

def test_harmonic_drives_toward_theta0():
    from halflife.step import compute_angle_forces
    from halflife.chemistry import _species_rest_angles
    # Put a degree-2 center at a 90° angle; harmonic should drive cos θ toward
    # cos θ0 of the center's species. Check the force sign matches (c - c0).
    state, c = _world_with_edges(
        [(1, 2), (2, 3)],
        positions={2: (5., 5.), 1: (6., 5.), 3: (5., 6.)},   # θ = 90°, c = 0
    )
    c = dataclasses.replace(c, angle_mode="harmonic", boundary_mode="open")
    phys = initialize_physics_params(c)
    F = np.asarray(compute_angle_forces(state, c, phys))
    assert np.all(np.isfinite(F))
    assert np.allclose(F[1] + F[2] + F[3], 0.0, atol=1e-4)   # momentum
    # θ0 of the center species is ≠ 90°, so the net effect here is non-zero.
    assert not np.allclose(F[1], 0.0, atol=1e-4)


def test_harmonic_smooth_at_180():
    from halflife.step import compute_angle_forces
    state, c = _world_with_edges(
        [(1, 2), (2, 3)],
        positions={2: (5., 5.), 1: (4., 5.), 3: (6., 5.)},   # collinear, c = -1
    )
    c = dataclasses.replace(c, angle_mode="harmonic", boundary_mode="open")
    F = np.asarray(compute_angle_forces(state, c, initialize_physics_params(c)))
    assert np.all(np.isfinite(F))                            # no cusp / NaN at 180°


def test_harmonic_relaxes_degree2_to_theta0():
    from halflife.step import compute_angle_forces
    from halflife.chemistry import _species_rest_angles
    state, c = _world_with_edges(
        [(1, 2), (2, 3)],
        positions={2: (5., 5.), 1: (6., 5.), 3: (5.2, 6.)},
    )
    c = dataclasses.replace(c, angle_mode="harmonic", boundary_mode="open")
    phys = initialize_physics_params(c)
    sp = int(np.asarray(state.particles.species)[2])
    theta0 = np.degrees(np.asarray(_species_rest_angles(c))[sp])
    pos = np.array(state.particles.position)   # writable copy
    for _ in range(600):
        F = np.asarray(compute_angle_forces(
            state._replace(particles=state.particles._replace(
                position=jnp.asarray(pos))), c, phys))
        for pid in (1, 3):
            pos[pid] += 0.02 * F[pid]
            v = pos[pid] - pos[2]
            pos[pid] = pos[2] + v / (np.linalg.norm(v) + 1e-9)
    u = pos[1] - pos[2]; w = pos[3] - pos[2]
    theta = np.degrees(np.arccos(np.clip(u@w/(np.linalg.norm(u)*np.linalg.norm(w)), -1, 1)))
    assert abs(theta - theta0) < 12
