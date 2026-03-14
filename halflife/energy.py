"""
Energy accounting and soft conservation.

Total energy = sum of particle kinetic energies + sum of composite binding energies.
This should be approximately conserved across steps (excluding intentional inputs
like fusion energy release). A soft correction prevents long-term drift.
"""

import jax.numpy as jnp

from halflife.state import ParticleState, CompositeState, WorldState


def compute_kinetic_energy(particles: ParticleState) -> jnp.ndarray:
    """
    Sum of kinetic energies across all alive particles.
    KE_i = 0.5 * mass_i * |velocity_i|^2
    """
    ke = 0.5 * particles.mass * jnp.sum(particles.velocity ** 2, axis=-1)
    return jnp.sum(ke * particles.alive.astype(jnp.float32))


def compute_binding_energy(composites: CompositeState) -> jnp.ndarray:
    """Sum of binding energies across all alive composites."""
    return jnp.sum(composites.binding_energy * composites.alive.astype(jnp.float32))


def compute_total_energy(state: WorldState) -> jnp.ndarray:
    """Total energy = kinetic + binding."""
    return (compute_kinetic_energy(state.particles) +
            compute_binding_energy(state.composites))


def apply_soft_energy_conservation(state: WorldState,
                                    target_energy: jnp.ndarray,
                                    max_correction: float = 0.01) -> WorldState:
    """
    Apply a small uniform velocity scaling to keep total energy near target.

    If energy drifted above target, scale velocities down; if below, scale up.
    The correction is capped at max_correction (1%) per step to avoid instability.

    Args:
        state:          WorldState
        target_energy:  scalar float32 — desired total energy
        max_correction: float — maximum fractional velocity change per step

    Returns:
        WorldState with corrected velocities
    """
    current_ke = compute_kinetic_energy(state.particles)
    target_ke = target_energy - compute_binding_energy(state.composites)

    # Ratio of desired to actual KE; velocity scales as sqrt
    ratio = jnp.where(current_ke > 1e-6, target_ke / (current_ke + 1e-10), 1.0)
    scale = jnp.sqrt(jnp.clip(ratio, 1.0 - max_correction, 1.0 + max_correction))

    new_vel = state.particles.velocity * scale
    new_particles = state.particles._replace(velocity=new_vel)
    return state._replace(particles=new_particles)
