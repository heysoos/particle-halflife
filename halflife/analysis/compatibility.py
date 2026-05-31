"""Pure-chemistry fusion compatibility matrices.

Given the static chemistry params (SimConfig, PhysicsParams), compute for
every pair of composite types whether they could in principle fuse — that
is, whether the merged BE passes the fusion threshold AND whether each side
has at least one free bond at their structural maximum.

Pure: no simulation, no event log, no state arrays. Sub-second even for
hundreds of unique composite types.
"""

from typing import Sequence, List

import numpy as np
import jax
import jax.numpy as jnp

from halflife.config import SimConfig
from halflife.state import PhysicsParams
from halflife.chemistry import (
    _entity_hash_val,
    _hash_to_binding_energy,
    _species_valences,
)


def max_free_bonds(member_species: Sequence[int], config: SimConfig) -> int:
    """Structural upper bound on free_bonds for a fresh n-member composite.

    Formula: Σ v_{s_i} − 2 * (n − 1) — spanning-tree minimum: n−1 edges,
    each consuming one bond on each endpoint. Free particles (n=1) get
    just v_s.
    """
    v = np.asarray(_species_valences(config))
    n = len(member_species)
    sum_v = sum(int(v[s]) for s in member_species)
    return sum_v - 2 * max(0, n - 1)


def _hash_multiset(member_species: Sequence[int], config: SimConfig) -> int:
    """Commutative additive hash matching the on-device kernel."""
    h = 0
    for s in member_species:
        h = (h + int(_entity_hash_val(jnp.int32(s), config))) % config.hash_modulus
    return h


def species_pair_compat_matrix(config: SimConfig, physics: PhysicsParams):
    """Per-species-pair fusion compatibility (Matrix 4a).

    Returns three (S, S) numpy arrays:
      be          — merged binding energy for the pair (float32)
      passes_be   — whether be >= physics.fusion_threshold (bool)
      passes_val  — whether both species have v >= 1 (always True, but
                    included for API symmetry with the top-K matrix where
                    saturation can happen)
    """
    S = config.num_species

    # Per-species values via vmap.
    species_idx = jnp.arange(S, dtype=jnp.int32)
    hvals = jax.vmap(lambda s: _entity_hash_val(s, config))(species_idx)  # (S,) uint32

    # Pairwise merged hash via outer sum then mod.
    H = np.asarray(hvals).astype(np.int64)
    merged = (H[:, None] + H[None, :]) % config.hash_modulus  # (S, S) int64

    # Compute BE for each cell on the host (vectorize the JAX call).
    be = np.zeros((S, S), dtype=np.float32)
    for i in range(S):
        for j in range(S):
            be[i, j] = float(_hash_to_binding_energy(
                jnp.uint32(int(merged[i, j])), config, physics
            ))

    passes_be = be >= float(physics.fusion_threshold)
    # Free particles always pass valence (v >= 1 by construction).
    passes_val = np.ones((S, S), dtype=bool)

    return be, passes_be, passes_val


def observed_pair_compat_matrix(
    hashes: np.ndarray,           # (K,) uint32 — unique species_hashes of observed composites
    multisets: List,              # length K — each entry is a sorted tuple of species ints
    config: SimConfig,
    physics: PhysicsParams,
):
    """Top-K observed-composite fusion compatibility (Matrix 4b).

    Args:
      hashes:    unique species_hash values, one per observed composite type
      multisets: parallel list of per-type member multisets (sorted tuples of species)
      config:    SimConfig
      physics:   PhysicsParams

    Returns: (be, passes_be, passes_val) each shape (K, K).
    """
    K = len(hashes)
    H = hashes.astype(np.int64)
    merged = (H[:, None] + H[None, :]) % config.hash_modulus

    be = np.zeros((K, K), dtype=np.float32)
    for i in range(K):
        for j in range(K):
            be[i, j] = float(_hash_to_binding_energy(
                jnp.uint32(int(merged[i, j])), config, physics
            ))

    passes_be = be >= float(physics.fusion_threshold)

    # passes_val: max_free_bonds(multisets[i]) >= 1 AND ditto for j.
    mfb = np.array([max_free_bonds(m, config) for m in multisets])
    passes_val = (mfb[:, None] >= 1) & (mfb[None, :] >= 1)

    return be, passes_be, passes_val


def size_pair_compat_matrix(
    hashes: np.ndarray,           # (K,) uint32 — unique species_hashes of observed composites
    multisets: List,              # length K — sorted tuple of species per type
    config: SimConfig,
    physics: PhysicsParams,
):
    """Size-binned aggregation of the observed-composite compatibility matrix (Matrix 4b, size view).

    For every size pair (i, j) in [1..max_composite_size]², aggregate over all
    observed composite type pairs (a, b) where size(a)==i and size(b)==j:
      mean_be[i, j]   — mean merged BE over those pairs
      passes_be[i, j] — True iff at least one pair passes BE threshold AND
                        the cell has any observed data
      passes_val[i,j] — True iff at least half of the pairs pass valence

    Cells with no observed pairs render as NaN (greyed by the plot helper),
    matching the same visual language as 4a/4b. This answers: "for the
    composite sizes I'm actually producing, can they chemistry-wise combine?"
    """
    M = config.max_composite_size
    sizes = np.array([len(m) for m in multisets], dtype=np.int32)

    # Compute pairwise merged BE across all observed type pairs (re-uses
    # observed_pair_compat math; cheap because K is small in practice).
    be_pair, passes_be_pair, passes_val_pair = observed_pair_compat_matrix(
        hashes, multisets, config, physics,
    )

    # Pre-bucket type indices by size for O(1) cell lookup.
    by_size = {}
    for idx, s in enumerate(sizes):
        by_size.setdefault(int(s), []).append(idx)

    be         = np.full((M + 1, M + 1), np.nan, dtype=np.float32)
    passes_be  = np.zeros((M + 1, M + 1), dtype=bool)
    passes_val = np.ones((M + 1, M + 1), dtype=bool)
    for i in range(1, M + 1):
        a_idx = by_size.get(i)
        if not a_idx:
            continue
        for j in range(1, M + 1):
            b_idx = by_size.get(j)
            if not b_idx:
                continue
            cell_bes  = be_pair[np.ix_(a_idx, b_idx)]
            cell_pbe  = passes_be_pair[np.ix_(a_idx, b_idx)]
            cell_pval = passes_val_pair[np.ix_(a_idx, b_idx)]
            be[i, j]         = float(cell_bes.mean())
            passes_be[i, j]  = bool(cell_pbe.any())
            passes_val[i, j] = bool(cell_pval.mean() >= 0.5)

    return be, passes_be, passes_val
