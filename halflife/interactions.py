"""
Pairwise force computation (Particle Life-style).

Each particle species pair (i, j) has:
  - A signed attraction strength:  params.attraction[si, sj]
  - A peak-attraction fraction:    params.peak_fraction[si, sj]
  - A cutoff fraction:             params.cutoff_fraction[si, sj]
  - A global repulsion radius:     physics.repulsion_radius (same for all species)
  - A global span:                 config.interaction_radius (the unit fractions multiply)

Force kernel shape (vs distance r), with
    r_peak   = interaction_radius * peak_fraction[si, sj]
    r_cutoff = interaction_radius * cutoff_fraction[si, sj]:
    r < repulsion_radius:            strong repulsive core
    repulsion_radius <= r < r_peak:  asymmetric ramp UP to attraction[si, sj]
    r_peak <= r < r_cutoff:          asymmetric ramp DOWN to 0
    r >= r_cutoff:                   zero

Both ramps are linear; r_peak is no longer constrained to the midpoint, so
each species pair gets its own force-shape (where the well is + how wide).

Forces are computed GPU-parallel via double vmap:
  outer vmap: over all particles i
  inner vmap: over all neighbors of particle i
"""

import jax
import jax.numpy as jnp

from halflife.state import InteractionParams, PhysicsParams
from halflife.config import SimConfig


def particle_life_force(r: jnp.ndarray, attraction: jnp.ndarray,
                         r_repulse: jnp.ndarray, r_peak: jnp.ndarray,
                         r_cutoff: jnp.ndarray,
                         repulsion_strength: jnp.ndarray) -> jnp.ndarray:
    """
    Scalar force magnitude for the Particle Life kernel.

    Sign convention matches pairwise_force's `return -f_mag * d_hat`:
      f_mag > 0  →  force toward j  →  ATTRACTIVE
      f_mag < 0  →  force away from j  →  REPULSIVE

    Args:
        r:                  scalar — distance between particles
        attraction:         scalar — signed strength in [-1, 1]
        r_repulse:          scalar — repulsion cutoff (global, slider-tunable)
        r_peak:             scalar — peak-attraction radius (per species pair)
        r_cutoff:           scalar — zero-force cutoff (per species pair)
        repulsion_strength: scalar — hard-core repulsion magnitude scale

    Returns:
        scalar float32 — force magnitude (negative = repulsive, positive = attractive)
    """
    eps = 1e-8

    # Hard-core repulsion: NEGATIVE so that -f_mag * d_hat points away from j.
    # Ramps from -repulsion_strength at r=0 to 0 at r=r_repulse.
    f_repulse = jnp.where(
        r < r_repulse,
        -repulsion_strength * (1.0 - r / (r_repulse + eps)),
        0.0
    )

    # Attraction zone: asymmetric triangle — POSITIVE = attractive, NEGATIVE = repulsive.
    #   ramps up linearly from 0 at r_repulse to attraction at r_peak,
    #   then ramps down linearly from attraction at r_peak to 0 at r_cutoff.
    # The two halves can have different widths since r_peak is no longer
    # constrained to the midpoint of [r_repulse, r_cutoff].
    up_width   = jnp.maximum(r_peak - r_repulse, eps)
    down_width = jnp.maximum(r_cutoff - r_peak, eps)
    in_up   = (r >= r_repulse) & (r < r_peak)
    in_down = (r >= r_peak)    & (r < r_cutoff)
    f_up    = attraction * ((r - r_repulse) / up_width)
    f_down  = attraction * (1.0 - (r - r_peak) / down_width)
    f_attract = jnp.where(in_up, f_up, jnp.where(in_down, f_down, 0.0))

    return f_repulse + f_attract


def pairwise_force(pos_i: jnp.ndarray, pos_j: jnp.ndarray,
                   species_i: jnp.ndarray, species_j: jnp.ndarray,
                   params: InteractionParams, config: SimConfig,
                   physics: PhysicsParams) -> jnp.ndarray:
    """
    Compute force on particle i due to particle j.

    Args:
        pos_i, pos_j:      (2,) float32 — positions
        species_i/j:       scalar int32  — species indices
        params:            InteractionParams
        config:            SimConfig (static)
        physics:           PhysicsParams (runtime-tunable)

    Returns:
        (2,) float32 — force vector on i (pointing away from j if repulsive)
    """
    # Displacement i ← j  (minimum image for periodic)
    d = pos_i - pos_j
    if config.boundary_mode == "periodic":
        d = d - config.world_width  * jnp.round(d[0] / config.world_width) * jnp.array([1., 0.])
        d = d - config.world_height * jnp.round(d[1] / config.world_height) * jnp.array([0., 1.])

    r = jnp.linalg.norm(d) + 1e-10  # distance (avoid div-by-zero)
    d_hat = d / r                     # unit direction

    # Look up species-pair parameters; convert fractions to absolute distances.
    # interaction_radius is the static unit; per-pair fractions in (0, 1] scale it.
    aij = params.attraction[species_i, species_j]
    r_a = params.peak_fraction[species_i, species_j] * config.interaction_radius
    r_c = params.cutoff_fraction[species_i, species_j] * config.interaction_radius

    f_mag = particle_life_force(r, aij * physics.attraction_scale,
                                physics.repulsion_radius, r_a,
                                r_c * physics.r_cutoff_scale,
                                physics.repulsion_strength)

    # Positive f_mag = repulsive = in direction of d (i away from j)
    # Negative f_mag = attractive = in direction of -d (i toward j)
    return -f_mag * d_hat  # sign convention: negative = toward j = attractive


def compute_forces_for_particle(i: jnp.ndarray,
                                  positions: jnp.ndarray,
                                  species: jnp.ndarray,
                                  composite_id: jnp.ndarray,
                                  neighbors: jnp.ndarray,
                                  params: InteractionParams,
                                  config: SimConfig,
                                  physics: PhysicsParams) -> tuple:
    """
    Net force on particle i from all its neighbors, plus its same-composite
    hard-core repulsion potential energy (the liquid-drop "Coulomb" term).

    Args:
        i:            scalar int32 — particle index
        positions:    (N, 2)
        species:      (N,)
        composite_id: (N,) int32 — composite index, -1 = free
        neighbors:    (max_neighbors,) int32 — neighbor indices for particle i
        params:       InteractionParams
        config:       SimConfig (static)
        physics:      PhysicsParams (runtime-tunable)

    Returns:
        (force (2,) float32, rep_pe () float32) — rep_pe is particle i's summed
        hard-core PE against SAME-COMPOSITE neighbors. Each pair is counted
        from both endpoints, so per-composite totals must be halved.
    """
    pos_i = positions[i]
    sp_i  = species[i]
    cid_i = composite_id[i]

    def contrib_from_neighbor(j):
        valid = (j >= 0)
        safe_j = jnp.where(valid, j, 0)
        pos_j = jnp.where(valid, positions[safe_j], pos_i)  # safe fallback
        sp_j  = jnp.where(valid, species[safe_j], sp_i)
        f = pairwise_force(pos_i, pos_j, sp_i, sp_j, params, config, physics)

        # Same-composite hard-core PE (liquid-drop disruption term). The
        # min-image + norm below duplicates pairwise_force's internal math on
        # identical inputs — XLA CSEs it, so this is effectively free.
        # Closed form: U(r) = ∫ᵣ^rr R·(1 − s/rr) ds = R·(rr − r)²/(2·rr),
        # matching the hard-core ramp in particle_life_force.
        d = pos_i - pos_j
        if config.boundary_mode == "periodic":
            d = d - config.world_width  * jnp.round(d[0] / config.world_width) * jnp.array([1., 0.])
            d = d - config.world_height * jnp.round(d[1] / config.world_height) * jnp.array([0., 1.])
        r = jnp.linalg.norm(d) + 1e-10
        rr = physics.repulsion_radius
        same_comp = valid & (cid_i >= 0) & (composite_id[safe_j] == cid_i)
        u = jnp.where(same_comp & (r < rr),
                      physics.repulsion_strength * (rr - r) ** 2 / (2.0 * rr + 1e-10),
                      0.0)
        return jnp.where(valid, f, jnp.zeros(2)), u

    # vmap over the neighbor array
    forces, pes = jax.vmap(contrib_from_neighbor)(neighbors)  # (max_neighbors, 2), (max_neighbors,)
    return jnp.sum(forces, axis=0), jnp.sum(pes)


def compute_all_forces(positions: jnp.ndarray,
                        species: jnp.ndarray,
                        composite_id: jnp.ndarray,
                        neighbors: jnp.ndarray,
                        params: InteractionParams,
                        config: SimConfig,
                        physics: PhysicsParams) -> tuple:
    """
    Compute net force for every particle simultaneously (outer vmap), plus the
    per-particle same-composite hard-core repulsion PE (liquid-drop term).

    Args:
        positions:    (N, 2) float32
        species:      (N,)   int32
        composite_id: (N,)   int32
        neighbors:    (N, max_neighbors) int32
        params:       InteractionParams
        config:       SimConfig (static)
        physics:      PhysicsParams (runtime-tunable)

    Returns:
        (forces (N, 2) float32, rep_pe (N,) float32)
    """
    particle_indices = jnp.arange(config.num_particles, dtype=jnp.int32)

    def forces_for_i(i):
        return compute_forces_for_particle(
            i, positions, species, composite_id, neighbors[i], params, config, physics
        )

    return jax.vmap(forces_for_i)(particle_indices)   # (N, 2), (N,)
