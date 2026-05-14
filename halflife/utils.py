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

def oklch_to_linear_srgb(L: float, C: float, h_deg: float) -> np.ndarray:
    """
    Convert OKLCh (L, C, hue°) → linear sRGB. Returns (3,) float32 in [0, 1]
    after clamping; out-of-gamut OKLCh inputs are clipped (acceptable for the
    moderate-chroma palette used here).

    Math from Björn Ottosson (2020) https://bottosson.github.io/posts/oklab/
    """
    import math
    h_rad = math.radians(h_deg)
    a = C * math.cos(h_rad)
    b = C * math.sin(h_rad)
    # OKLab → cone-response basis (cubic root applied in inverse)
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l3, m3, s3 = l_ ** 3, m_ ** 3, s_ ** 3
    # Cone responses → linear sRGB
    r =  4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3
    g = -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3
    b_ = -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3
    out = np.array([r, g, b_], dtype=np.float32)
    return np.clip(out, 0.0, 1.0)


def make_species_colors(num_species: int) -> np.ndarray:
    """
    Generate a perceptually-uniform palette for num_species species using OKLCh.
    Returns: (num_species, 3) float32 *linear-sRGB* in [0, 1].

    The renderer composites with ACES tonemap + sRGB OETF, so colors stay in
    linear space all the way through the scene FBO. Adjacent species get
    slightly different lightness and chroma so neighboring hues separate
    visually as well as chromatically.
    """
    L_base = 0.72   # target lightness — bright against dark background
    C_base = 0.13   # moderate chroma — pops without being neon
    HUE_OFFSET = 20.0  # offsets the wheel so species 0 isn't pure-red
    colors = np.zeros((num_species, 3), dtype=np.float32)
    for i in range(num_species):
        Li = L_base + (0.04 if (i % 2 == 0) else -0.04)
        Ci = C_base + (0.02 if (i % 3 != 2) else -0.02)
        h_deg = (i * 360.0 / num_species + HUE_OFFSET) % 360.0
        colors[i] = oklch_to_linear_srgb(Li, Ci, h_deg)
    return colors


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
