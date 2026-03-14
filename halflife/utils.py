"""
JAX utility functions used across modules.

Includes:
  - hash_multiset()   — polynomial rolling hash on sorted int arrays (JIT-safe)
  - find_free_slots() — locate available slots in a masked pool
  - boundary helpers  — periodic wrap and reflective bounce
  - color helpers     — species index → RGB color
"""

import jax
import jax.numpy as jnp
import numpy as np


# ── Numeric Hashing ──────────────────────────────────────────────────────────

def hash_multiset(species_sorted: jnp.ndarray, count: jnp.ndarray,
                  prime_a: int, prime_b: int, modulus: int) -> jnp.ndarray:
    """
    Polynomial rolling hash over a sorted multiset of species IDs.

    Produces a deterministic uint32 hash from a sorted integer array.
    Sorting ensures multiset semantics: hash({A,B}) == hash({B,A}).

    Args:
        species_sorted: (max_size,) int32 — sorted species IDs, padded with -1
        count:          scalar int32 — number of valid entries
        prime_a, prime_b, modulus: hash constants from SimConfig

    Returns:
        scalar uint32 hash value
    """
    def body(carry, i):
        h = carry
        s = species_sorted[i]
        # Only incorporate valid entries (i < count)
        valid = (i < count) & (s >= 0)
        new_h = (h * prime_a + s + prime_b) % modulus
        h = jnp.where(valid, new_h, h)
        return h, None

    max_size = species_sorted.shape[0]
    h_init = jnp.array(0, dtype=jnp.int32)
    h_final, _ = jax.lax.scan(body, h_init, jnp.arange(max_size))
    return h_final.astype(jnp.uint32)


def hash_scalar(species: jnp.ndarray, prime_a: int, prime_b: int,
                modulus: int) -> jnp.ndarray:
    """Hash a single species ID (for single-particle decay)."""
    return jnp.array(
        (prime_a * (species + 1) + prime_b) % modulus,
        dtype=jnp.uint32
    )


# ── Free-Slot Finding ────────────────────────────────────────────────────────

def find_free_slots(alive: jnp.ndarray, n_needed: int) -> jnp.ndarray:
    """
    Find the first n_needed False slots in a boolean alive array.

    Returns:
        (n_needed,) int32 array of free slot indices.
        If fewer than n_needed slots are free, the remaining entries
        are filled with -1 (invalid).

    Args:
        alive:    (N,) bool — True = occupied, False = free
        n_needed: int — number of free slots to find
    """
    free_mask = ~alive
    # Assign ordinal positions to free slots, -1 to occupied slots
    # cumsum of free_mask gives how many free slots we've seen so far
    cumsum = jnp.cumsum(free_mask)  # (N,) int32: 1-indexed count of free slots seen
    # Slot index i gets ordinal k if cumsum[i]==k and free_mask[i]==True
    # We want the indices where ordinal is in [1, n_needed]
    slot_indices = jnp.arange(alive.shape[0], dtype=jnp.int32)
    # For each ordinal 1..n_needed, find the corresponding slot
    def get_slot(ordinal):
        # slot where cumsum == ordinal and free_mask == True
        match = jnp.where(free_mask & (cumsum == ordinal), slot_indices, alive.shape[0])
        return jnp.min(match)

    slots = jax.vmap(get_slot)(jnp.arange(1, n_needed + 1, dtype=jnp.int32))
    # Mark slots past the end of the array as -1 (invalid)
    slots = jnp.where(slots >= alive.shape[0], -1, slots)
    return slots


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

def pairwise_displacement(pos_i: jnp.ndarray, pos_j: jnp.ndarray,
                           config) -> jnp.ndarray:
    """
    Compute displacement vector from j to i, accounting for periodic boundaries.
    Returns the minimum-image displacement.

    Args:
        pos_i: (2,) float32
        pos_j: (2,) float32
    Returns:
        (2,) float32 displacement d = pos_i - pos_j (shortest path)
    """
    d = pos_i - pos_j
    if config.boundary_mode == "periodic":
        # Minimum image convention
        d = d - config.world_width  * jnp.round(d[0] / config.world_width)  * jnp.array([1.0, 0.0])
        d = d - config.world_height * jnp.round(d[1] / config.world_height) * jnp.array([0.0, 1.0])
    return d


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

def safe_normalize(v: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Normalize a vector, returning zero vector if near-zero magnitude."""
    norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return v / jnp.maximum(norm, eps)


def count_alive(alive: jnp.ndarray) -> jnp.ndarray:
    """Count number of True entries in alive mask."""
    return jnp.sum(alive.astype(jnp.int32))
