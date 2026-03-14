"""
Cell-list spatial indexing for O(N) neighbor queries.

Building a cell list:
  1. Assign each particle to a grid cell based on its position
  2. Sort particles by cell index (argsort)
  3. Pack into a fixed-size (num_cells, cell_capacity) lookup array

Finding neighbors:
  For particle i, scan the 3x3 neighborhood of cells around i's cell.
  Return up to max_neighbors particle indices, padded with -1.

All operations are fully JIT-compilable (no Python control flow inside JAX).
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple

from halflife.config import SimConfig


class CellList(NamedTuple):
    """
    Spatial cell-list index.

    particle_ids[cell_idx, k] = particle index of the k-th particle in cell cell_idx.
    cell_counts[cell_idx] = number of particles in that cell (up to cell_capacity).
    did_overflow = True if any cell exceeded cell_capacity (needs realloc).
    """
    particle_ids: jnp.ndarray  # (num_cells, cell_capacity) int32
    cell_counts:  jnp.ndarray  # (num_cells,) int32
    did_overflow: jnp.ndarray  # () bool


def _particle_to_cell_xy(position: jnp.ndarray, config: SimConfig) -> jnp.ndarray:
    """Map a position to (cell_x, cell_y) integer coordinates."""
    cx = jnp.floor(position[..., 0] / config.cell_size).astype(jnp.int32)
    cy = jnp.floor(position[..., 1] / config.cell_size).astype(jnp.int32)
    # Clamp to grid bounds (handles particles at or beyond world edge)
    cx = jnp.clip(cx, 0, config.num_cells_x - 1)
    cy = jnp.clip(cy, 0, config.num_cells_y - 1)
    return cx, cy


def _linearize_cell(cx: jnp.ndarray, cy: jnp.ndarray, config: SimConfig) -> jnp.ndarray:
    """Convert (cell_x, cell_y) to a linear cell index."""
    return cx * config.num_cells_y + cy


def build_cell_list(positions: jnp.ndarray, alive: jnp.ndarray,
                    config: SimConfig) -> CellList:
    """
    Build a cell list from particle positions.

    Args:
        positions: (N, 2) float32
        alive:     (N,)   bool
        config:    SimConfig (static)

    Returns:
        CellList with particle_ids (num_cells, cell_capacity) and counts
    """
    N = config.max_particles
    num_cells = config.num_cells
    cap = config.cell_capacity

    # Compute cell index for each particle (-1 for dead particles)
    cx, cy = _particle_to_cell_xy(positions, config)
    cell_idx = _linearize_cell(cx, cy, config)  # (N,) int32
    # Dead particles get a sentinel beyond the grid
    cell_idx = jnp.where(alive, cell_idx, num_cells)  # num_cells = invalid sentinel

    # Build cell list by iterating particles in sorted order.
    # We use a scatter approach: for each cell, collect particles.
    # JAX doesn't support dynamic indexing inside vmap easily, so we use
    # an argsort + prefix-sum approach.

    # Sort particle indices by their cell assignment
    sort_order = jnp.argsort(cell_idx, stable=True)  # (N,) — particles sorted by cell
    sorted_cells = cell_idx[sort_order]              # (N,) — corresponding cell ids

    # Count particles per cell
    # Use one-hot encoding summed over particles (memory-efficient via segment sums)
    cell_counts = jnp.zeros(num_cells + 1, dtype=jnp.int32)
    cell_counts = cell_counts.at[sorted_cells].add(1)
    cell_counts = cell_counts[:num_cells]  # drop the invalid-sentinel cell

    # Overflow detection
    did_overflow = jnp.any(cell_counts > cap)

    # Build (num_cells, cell_capacity) particle_ids array.
    # For each sorted particle, determine its position WITHIN its cell.
    # Use cumsum within each cell: the k-th occurrence in cell c goes to column k.
    # We compute a "local offset" = cumsum(sorted_cells == c) - 1 for each cell c.

    # Global cumsum then subtract cell-start offset:
    global_cumsum = jnp.cumsum(jnp.ones(N, dtype=jnp.int32))  # 1-indexed position in sorted array
    # Cell start index for each sorted particle:
    cell_starts = jnp.concatenate([jnp.array([0]), jnp.cumsum(cell_counts)[:-1]])
    local_offset = global_cumsum - 1 - cell_starts[
        jnp.clip(sorted_cells, 0, num_cells - 1)
    ]  # (N,) — position within cell [0, cell_count-1]

    # Clamp offset to capacity (handles overflow gracefully)
    local_offset_clamped = jnp.clip(local_offset, 0, cap - 1)

    # Scatter into particle_ids
    particle_ids = jnp.full((num_cells, cap), -1, dtype=jnp.int32)

    # Only write alive particles within valid cells
    valid = alive & (sorted_cells < num_cells) & (local_offset < cap)
    row = jnp.clip(sorted_cells, 0, num_cells - 1)
    col = local_offset_clamped

    # We need to do: particle_ids[row[i], col[i]] = sort_order[i] for valid[i]
    # Use flat index scatter
    flat_idx = row * cap + col
    flat_ids = jnp.full(num_cells * cap, -1, dtype=jnp.int32)
    flat_ids = flat_ids.at[
        jnp.where(valid, flat_idx, num_cells * cap - 1)  # safe out-of-bounds index
    ].set(jnp.where(valid, sort_order, -1), mode='drop')
    particle_ids = flat_ids.reshape(num_cells, cap)

    return CellList(
        particle_ids=particle_ids,
        cell_counts=cell_counts,
        did_overflow=did_overflow,
    )


def find_neighbors_for_particle(i: int, positions: jnp.ndarray,
                                  alive: jnp.ndarray, cell_list: CellList,
                                  config: SimConfig) -> jnp.ndarray:
    """
    Find all particles within interaction_radius of particle i.

    Scans the 3x3 neighborhood of cells around particle i's cell.
    Returns a fixed-size array of neighbor indices padded with -1.

    Args:
        i:          scalar int — index of the query particle
        positions:  (N, 2) float32
        alive:      (N,)   bool
        cell_list:  CellList
        config:     SimConfig (static)

    Returns:
        (max_neighbors,) int32 — neighbor indices, padded with -1
    """
    pos_i = positions[i]
    cx_i, cy_i = _particle_to_cell_xy(pos_i[None], config)
    cx_i, cy_i = cx_i[0], cy_i[0]

    cap = config.cell_capacity
    num_cells = config.num_cells
    r2 = config.interaction_radius ** 2

    # Collect candidates from 3x3 neighborhood (9 cells x cell_capacity particles)
    max_candidates = 9 * cap

    def get_candidates_from_offset(carry, offset):
        dx, dy = offset[0], offset[1]
        # Neighbor cell coordinates (with periodic wrap or clamp)
        ncx = (cx_i + dx) % config.num_cells_x
        ncy = (cy_i + dy) % config.num_cells_y
        cell_lin = _linearize_cell(ncx, ncy, config)
        # Get all particle ids from this cell
        pids = cell_list.particle_ids[cell_lin]  # (cap,)
        return carry, pids

    # 3x3 offsets: dx in [-1,0,1], dy in [-1,0,1]
    offsets = jnp.array([
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1],  [0, 0],  [0, 1],
        [1, -1],  [1, 0],  [1, 1],
    ], dtype=jnp.int32)  # (9, 2)

    _, candidate_blocks = jax.lax.scan(get_candidates_from_offset, None, offsets)
    # candidate_blocks: (9, cap) int32
    candidates = candidate_blocks.reshape(-1)  # (9*cap,) int32

    # Filter: alive, within radius, not self
    def check_candidate(pid):
        valid = (pid >= 0) & (pid != i) & alive[pid]
        pos_j = jnp.where(valid, positions[pid], pos_i)  # avoid OOB on dead
        # Minimum image displacement
        d = pos_i - pos_j
        if config.boundary_mode == "periodic":
            d = d.at[0].set(d[0] - config.world_width * jnp.round(d[0] / config.world_width))
            d = d.at[1].set(d[1] - config.world_height * jnp.round(d[1] / config.world_height))
        dist2 = jnp.dot(d, d)
        return jnp.where(valid & (dist2 < r2), pid, -1)

    filtered = jax.vmap(check_candidate)(candidates)  # (max_candidates,) — -1 for non-neighbors

    # Pack into fixed-size (max_neighbors,) output, keeping first max_neighbors valid ones
    is_valid = filtered >= 0  # (max_candidates,) bool
    cumcount = jnp.cumsum(is_valid.astype(jnp.int32))  # 1-indexed count of valid seen
    max_nb = config.max_neighbors
    indices = jnp.arange(max_candidates, dtype=jnp.int32)

    def pack_slot(k):
        # slot k gets the candidate where cumcount==k+1 and is_valid
        match = jnp.where(is_valid & (cumcount == k + 1), filtered, -1)
        return jnp.max(match)  # exactly one match, or -1 if fewer than k+1 neighbors

    neighbors = jax.vmap(pack_slot)(jnp.arange(max_nb, dtype=jnp.int32))
    return neighbors


def find_all_neighbors(positions: jnp.ndarray, alive: jnp.ndarray,
                        cell_list: CellList, config: SimConfig) -> jnp.ndarray:
    """
    Find neighbors for ALL particles simultaneously using vmap.

    Args:
        positions:  (N, 2) float32
        alive:      (N,)   bool
        cell_list:  CellList
        config:     SimConfig (static)

    Returns:
        (N, max_neighbors) int32 — neighbor indices per particle, padded with -1
    """
    find_fn = lambda i: find_neighbors_for_particle(i, positions, alive, cell_list, config)
    return jax.vmap(find_fn)(jnp.arange(config.max_particles, dtype=jnp.int32))
