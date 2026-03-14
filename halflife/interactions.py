"""
Pairwise force computation (Particle Life-style).

Each particle species pair (i, j) has:
  - A signed attraction strength:  params.attraction[si, sj]
  - A peak attraction radius:       params.r_attract[si, sj]
  - A cutoff radius:                params.r_cutoff[si, sj]
  - A global repulsion radius:      config.repulsion_radius (same for all species)

Force kernel shape (vs distance r):
    r < repulsion_radius:            strong repulsive core
    repulsion_radius <= r < r_attract: ramp toward peak attraction
    r_attract <= r < r_cutoff:         ramp down to zero
    r >= r_cutoff:                      zero

This is exactly the "Particle Life" kernel, with species-dependent
attraction sign and magnitude.

Forces are computed GPU-parallel via double vmap:
  outer vmap: over all particles i
  inner vmap: over all neighbors of particle i
"""

import jax
import jax.numpy as jnp

from halflife.state import InteractionParams
from halflife.config import SimConfig


def particle_life_force(r: jnp.ndarray, attraction: jnp.ndarray,
                         r_repulse: float, r_attract: jnp.ndarray,
                         r_cutoff: jnp.ndarray) -> jnp.ndarray:
    """
    Scalar force magnitude for the Particle Life kernel.

    Positive = repulsive (pushes apart), negative = attractive (pulls together).

    Args:
        r:          scalar — distance between particles
        attraction: scalar — signed strength in [-1, 1]
        r_repulse:  float  — repulsion cutoff (config-level constant)
        r_attract:  scalar — peak attraction radius
        r_cutoff:   scalar — zero-force cutoff

    Returns:
        scalar float32 — force magnitude (positive = repulsive)
    """
    eps = 1e-8

    # Hard-core repulsion: linear ramp from r_repulse to r=0
    # F_repulse = strength * (1 - r / r_repulse)
    f_repulse = jnp.where(
        r < r_repulse,
        1.0 - r / (r_repulse + eps),
        0.0
    )

    # Attraction zone: triangle kernel
    #   ramps up from 0 at r_repulse to peak at r_attract,
    #   then ramps down to 0 at r_cutoff
    half_width = (r_cutoff - r_repulse) * 0.5 + eps
    peak = r_repulse + half_width
    f_attract = jnp.where(
        (r >= r_repulse) & (r < r_cutoff),
        attraction * (1.0 - jnp.abs(r - peak) / half_width),
        0.0
    )

    return f_repulse + f_attract


def pairwise_force(pos_i: jnp.ndarray, pos_j: jnp.ndarray,
                   species_i: jnp.ndarray, species_j: jnp.ndarray,
                   params: InteractionParams, config: SimConfig) -> jnp.ndarray:
    """
    Compute force on particle i due to particle j.

    Args:
        pos_i, pos_j:      (2,) float32 — positions
        species_i/j:       scalar int32  — species indices
        params:            InteractionParams
        config:            SimConfig (static)

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

    # Look up species-pair parameters
    aij = params.attraction[species_i, species_j]
    r_a = params.r_attract[species_i, species_j]
    r_c = params.r_cutoff[species_i, species_j]

    f_mag = particle_life_force(r, aij, config.repulsion_radius, r_a, r_c)

    # Positive f_mag = repulsive = in direction of d (i away from j)
    # Negative f_mag = attractive = in direction of -d (i toward j)
    return -f_mag * d_hat  # sign convention: negative = toward j = attractive


def compute_forces_for_particle(i: jnp.ndarray,
                                  positions: jnp.ndarray,
                                  species: jnp.ndarray,
                                  alive: jnp.ndarray,
                                  neighbors: jnp.ndarray,
                                  params: InteractionParams,
                                  config: SimConfig) -> jnp.ndarray:
    """
    Net force on particle i from all its neighbors.

    Args:
        i:          scalar int32 — particle index
        positions:  (N, 2)
        species:    (N,)
        alive:      (N,)
        neighbors:  (max_neighbors,) int32 — neighbor indices for particle i
        params:     InteractionParams
        config:     SimConfig (static)

    Returns:
        (2,) float32 — total force on particle i
    """
    pos_i = positions[i]
    sp_i  = species[i]

    def force_from_neighbor(j):
        valid = (j >= 0) & alive[j]
        pos_j = jnp.where(valid, positions[j], pos_i)  # safe fallback
        sp_j  = jnp.where(valid, species[j], sp_i)
        f = pairwise_force(pos_i, pos_j, sp_i, sp_j, params, config)
        return jnp.where(valid, f, jnp.zeros(2))

    # vmap over the neighbor array
    forces = jax.vmap(force_from_neighbor)(neighbors)  # (max_neighbors, 2)
    return jnp.sum(forces, axis=0)


def compute_all_forces(positions: jnp.ndarray,
                        species: jnp.ndarray,
                        alive: jnp.ndarray,
                        neighbors: jnp.ndarray,
                        params: InteractionParams,
                        config: SimConfig) -> jnp.ndarray:
    """
    Compute net force for every particle simultaneously (outer vmap).

    Args:
        positions:  (N, 2) float32
        species:    (N,)   int32
        alive:      (N,)   bool
        neighbors:  (N, max_neighbors) int32
        params:     InteractionParams
        config:     SimConfig (static)

    Returns:
        (N, 2) float32 — force vectors per particle
    """
    particle_indices = jnp.arange(config.max_particles, dtype=jnp.int32)

    def forces_for_i(i):
        return compute_forces_for_particle(
            i, positions, species, alive, neighbors[i], params, config
        )

    forces = jax.vmap(forces_for_i)(particle_indices)   # (N, 2)
    # Dead particles have zero force
    return forces * alive[:, None].astype(jnp.float32)
