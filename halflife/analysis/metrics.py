"""Per-step metric collection.

All functions here take state arrays and config and return plain dicts of
JAX scalars/arrays. Designed to be called inside lax.scan bodies — the
outputs stack naturally into per-step time series.
"""

import jax
import jax.numpy as jnp

from halflife.config import SimConfig
from halflife.chemistry import compute_degree, _species_valences


def size_metrics(composites, config: SimConfig) -> dict:
    """Tier 1 macroscopic metrics from CompositeState.

    Returns a dict whose values are all JAX-traceable so this can be called
    inside a lax.scan body. Outputs:
      max_size, mean_size, median_size, alive_count,
      free_particle_fraction, size_histogram (1..max_composite_size)
    """
    alive = composites.alive
    counts = composites.member_count

    alive_counts = jnp.where(alive, counts, 0)
    n_alive = jnp.sum(alive.astype(jnp.int32))
    total_in_composites = jnp.sum(alive_counts.astype(jnp.int32))

    max_size = jnp.max(jnp.where(alive, counts, 0))
    safe_n = jnp.maximum(n_alive, 1)
    mean_size = jnp.where(
        n_alive > 0,
        jnp.sum(alive_counts.astype(jnp.float32)) / safe_n.astype(jnp.float32),
        jnp.float32(0.0),
    )

    sorted_sizes = jnp.sort(alive_counts)
    median_idx = jnp.clip(n_alive // 2, 0, sorted_sizes.shape[0] - 1)
    median_size = jnp.where(
        n_alive > 0,
        sorted_sizes[median_idx].astype(jnp.float32),
        jnp.float32(0.0),
    )

    free_particle_fraction = jnp.where(
        config.num_particles > 0,
        1.0 - total_in_composites.astype(jnp.float32) / jnp.float32(config.num_particles),
        jnp.float32(1.0),
    )

    # Histogram over sizes 1..max_composite_size (bin 0 = "size 1", etc.)
    bins = jnp.arange(1, config.max_composite_size + 1, dtype=jnp.int32)
    size_histogram = jax.vmap(
        lambda b: jnp.sum(jnp.where(alive & (counts == b), 1, 0).astype(jnp.int32))
    )(bins)

    return {
        'max_size': max_size,
        'mean_size': mean_size,
        'median_size': median_size,
        'alive_count': n_alive,
        'free_particle_fraction': free_particle_fraction,
        'size_histogram': size_histogram,
    }


def valence_edge_metrics(particles, composites, config: SimConfig) -> dict:
    """Tier 2 valence/edge metrics.

    Returns:
      edge_count_total      — sum of edge_count over alive composites
      ring_count_total      — sum of (edge_count - (size - 1)) over alive composites
                              (extra edges beyond a spanning tree)
      degree_saturation_pct — fraction of particles with degree == v_species
      free_bonds_histogram  — bincount of free_bonds[alive] (length = max possible + 1)
      degree_histogram      — bincount of per-particle degree (length = max_valence + 1)
    """
    species_valences = _species_valences(config)
    degree = compute_degree(composites, config)

    # Degree saturation: degree[i] == v_{species[i]}
    particle_v = species_valences[particles.species]
    saturated = (degree == particle_v).astype(jnp.float32)
    degree_saturation_pct = jnp.mean(saturated)

    # Free-bonds histogram: free_bonds in alive composites can range
    # from 0 to (max_composite_size * max_valence) inclusive in theory.
    # Use a generous bin range.
    fb_max = config.max_composite_size * config.max_valence
    fb_bins = jnp.arange(fb_max + 1, dtype=jnp.int32)
    free_bonds_histogram = jax.vmap(
        lambda b: jnp.sum(jnp.where(composites.alive & (composites.free_bonds == b), 1, 0))
    )(fb_bins)

    # Per-particle degree histogram (0..max_valence)
    deg_bins = jnp.arange(config.max_valence + 1, dtype=jnp.int32)
    degree_histogram = jax.vmap(
        lambda b: jnp.sum((degree == b).astype(jnp.int32))
    )(deg_bins)

    # Edge / ring counts
    spanning_edges = jnp.where(
        composites.alive,
        jnp.maximum(composites.member_count - 1, 0),
        0,
    )
    edge_count_total = jnp.sum(
        jnp.where(composites.alive, composites.edge_count, 0).astype(jnp.int32)
    )
    ring_count_total = jnp.sum(
        jnp.where(composites.alive, composites.edge_count - spanning_edges, 0).astype(jnp.int32)
    )

    return {
        'edge_count_total': edge_count_total,
        'ring_count_total': ring_count_total,
        'degree_saturation_pct': degree_saturation_pct,
        'free_bonds_histogram': free_bonds_histogram,
        'degree_histogram': degree_histogram,
    }
