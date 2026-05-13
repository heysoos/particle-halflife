"""
JAX utility functions used across modules.

Includes:
  - find_free_slots() — locate available slots in a masked pool
  - boundary helpers  — periodic wrap and reflective bounce
  - color helpers     — species index → RGB color
"""

import jax
import jax.numpy as jnp
import numpy as np


# ── Free-Slot Finding ────────────────────────────────────────────────────────

def find_free_slots(alive: jnp.ndarray, n_needed: int) -> jnp.ndarray:
    """
    Find the first n_needed False slots in a boolean alive array.

    O(N log N) via argsort — avoids the O(N²) vmap-of-min approach.

    Returns:
        (n_needed,) int32 array of free slot indices.
        If fewer than n_needed slots are free, remaining entries are -1.

    Args:
        alive:    (N,) bool — True = occupied, False = free
        n_needed: int — number of free slots to find (compile-time constant)
    """
    N = alive.shape[0]
    # Free slots get their real index; occupied slots get sentinel N (sorts to end)
    candidates = jnp.where(~alive, jnp.arange(N, dtype=jnp.int32), N)
    sorted_candidates = jnp.sort(candidates)   # free slots first, sentinels last
    slots = sorted_candidates[:n_needed]
    return jnp.where(slots >= N, -1, slots)


# ── Boundary Conditions ──────────────────────────────────────────────────────

def apply_periodic_boundary(position: jnp.ndarray,
                             world_width: float, world_height: float) -> jnp.ndarray:
    """
    Wrap positions to stay within [0, world_width] x [0, world_height].
    Args:
        position: (..., 2) float32
    Returns:
        position wrapped to world bounds
    """
    x = position[..., 0] % world_width
    y = position[..., 1] % world_height
    return jnp.stack([x, y], axis=-1)


def apply_reflective_boundary(position: jnp.ndarray, velocity: jnp.ndarray,
                               world_width: float, world_height: float):
    """
    Reflect positions and velocities off world boundaries.
    Args:
        position: (..., 2) float32
        velocity: (..., 2) float32
    Returns:
        (position, velocity) with reflections applied
    """
    px, py = position[..., 0], position[..., 1]
    vx, vy = velocity[..., 0], velocity[..., 1]

    # X axis
    hit_left  = px < 0.0
    hit_right = px > world_width
    px = jnp.where(hit_left,  -px,              px)
    px = jnp.where(hit_right, 2 * world_width - px, px)
    vx = jnp.where(hit_left | hit_right, -vx, vx)

    # Y axis
    hit_bottom = py < 0.0
    hit_top    = py > world_height
    py = jnp.where(hit_bottom, -py,               py)
    py = jnp.where(hit_top,    2 * world_height - py, py)
    vy = jnp.where(hit_bottom | hit_top, -vy, vy)

    return (jnp.stack([px, py], axis=-1),
            jnp.stack([vx, vy], axis=-1))


def apply_boundary(position: jnp.ndarray, velocity: jnp.ndarray,
                   config) -> tuple:
    """
    Dispatch to periodic or reflective boundary based on config.boundary_mode.
    Called outside JIT (config is static).
    """
    if config.boundary_mode == "periodic":
        new_pos = apply_periodic_boundary(position, config.world_width, config.world_height)
        return new_pos, velocity
    else:
        return apply_reflective_boundary(position, velocity,
                                         config.world_width, config.world_height)


# ── Displacement (handles periodic wrapping for force computation) ────────────

# ── REMOVED 2026-05-05: never adopted ──────────────────────────────────────
# pairwise_displacement() was meant to be the single source of truth for
# minimum-image displacement. In practice every call site (force kernel,
# bond forces, neighbor finding, fusion) inlines its own min-image instead
# (audit nub: "min-image is duplicated in 4 places"). Helper has zero callers.
# Kept commented as a future refactor target — adopting it would dedupe the
# four inline copies. Safe to delete in a follow-up.
#
# def pairwise_displacement(pos_i: jnp.ndarray, pos_j: jnp.ndarray,
#                            config) -> jnp.ndarray:
#     """
#     Compute displacement vector from j to i, accounting for periodic boundaries.
#     Returns the minimum-image displacement.
#
#     Args:
#         pos_i: (2,) float32
#         pos_j: (2,) float32
#     Returns:
#         (2,) float32 displacement d = pos_i - pos_j (shortest path)
#     """
#     d = pos_i - pos_j
#     if config.boundary_mode == "periodic":
#         # Minimum image convention
#         d = d - config.world_width  * jnp.round(d[0] / config.world_width)  * jnp.array([1.0, 0.0])
#         d = d - config.world_height * jnp.round(d[1] / config.world_height) * jnp.array([0.0, 1.0])
#     return d


# ── Color Palette ────────────────────────────────────────────────────────────

def make_species_colors(num_species: int) -> np.ndarray:
    """
    Generate a visually distinct HSV color palette for num_species species.
    Returns: (num_species, 3) float32 RGB array in [0,1].
    """
    import colorsys
    colors = []
    for i in range(num_species):
        hue = i / num_species
        # Alternate saturation/value for visual separation
        sat = 0.85 if (i % 2 == 0) else 0.65
        val = 0.95 if (i % 3 != 2) else 0.75
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        colors.append([r, g, b])
    return np.array(colors, dtype=np.float32)


# ── Misc JAX Helpers ─────────────────────────────────────────────────────────

# ── REMOVED 2026-05-05: orphans, no callers ────────────────────────────────
# safe_normalize() — never called.
# count_alive() — orphan from the particle-alive era. ParticleState.alive was
# removed in commit b0c049f; the only remaining alive mask is on composites,
# and its callers inline `jnp.sum(alive.astype(jnp.int32))` rather than going
# through this helper. Kept commented for revival reference; safe to delete
# in a follow-up.
#
# def safe_normalize(v: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
#     """Normalize a vector, returning zero vector if near-zero magnitude."""
#     norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
#     return v / jnp.maximum(norm, eps)
#
#
# def count_alive(alive: jnp.ndarray) -> jnp.ndarray:
#     """Count number of True entries in alive mask."""
#     return jnp.sum(alive.astype(jnp.int32))
