"""Event log → composite transition matrices.

Three matrix shapes, all built from a flat (sentinels-filtered) ReactionEvent:
  - size-binned: (max_composite_size+1, max_composite_size+1) — mass-flow view
  - top-K:       (K+1, K+1) — K most-trafficked hashes + "other"
  - full:        (U, U) — every observed unique hash

Cell semantics: for every fusion event A+B→C, increment matrix[A, C] and
matrix[B, C]. For every fission event C→A+B, increment matrix[C, A] and
matrix[C, B]. Each event contributes 2 cells.
"""

from collections import Counter
from typing import Tuple, List

import numpy as np

from halflife.state import ReactionEvent
from halflife.analysis.events import KIND_FUSION, KIND_FISSION


def _iter_edges(events: ReactionEvent):
    """Yield (source_hash, source_size, product_hash, product_size) per cell.

    For fusion: yields (A, C) and (B, C). For fission: yields (C, A) and (C, B).
    """
    kind = np.asarray(events.kind)
    sh = np.asarray(events.source_hashes)
    ss = np.asarray(events.source_sizes)
    ph = np.asarray(events.product_hashes)
    ps = np.asarray(events.product_sizes)

    for i in range(kind.shape[0]):
        if kind[i] == KIND_FUSION:
            # A + B → C; yield (A, C) and (B, C). Product is in slot 0.
            for src_idx in (0, 1):
                yield (int(sh[i, src_idx]), int(ss[i, src_idx]),
                       int(ph[i, 0]),       int(ps[i, 0]))
        elif kind[i] == KIND_FISSION:
            # C → A + B; yield (C, A) and (C, B). Source is in slot 0.
            for prod_idx in (0, 1):
                yield (int(sh[i, 0]),         int(ss[i, 0]),
                       int(ph[i, prod_idx]),  int(ps[i, prod_idx]))


def size_bin_transition_matrix(events: ReactionEvent, max_composite_size: int) -> np.ndarray:
    """(max_composite_size+1, max_composite_size+1) matrix of size→size transitions.

    Index 0 = empty / sentinel — practically unused. Bins 1..max_composite_size
    are the live size classes.
    """
    M = max_composite_size + 1
    matrix = np.zeros((M, M), dtype=np.int64)
    for _sh, ss, _ph, ps in _iter_edges(events):
        if 0 <= ss < M and 0 <= ps < M:
            matrix[ss, ps] += 1
    return matrix


def top_k_transition_matrix(
    events: ReactionEvent, K: int = 30
) -> Tuple[np.ndarray, List[str]]:
    """(K+1, K+1) matrix on the K most-trafficked species hashes, sorted by size.

    Sort key: (size ascending, hash ascending). "Trafficked" = total incidence
    (row + col before truncation). The last row/col is "other" and collects
    all tail traffic.

    Returns (matrix, labels) where labels[i] is the human-readable hash for
    row/col i ("0x..." or "other" for the last).
    """
    edges = list(_iter_edges(events))
    if not edges:
        return np.zeros((1, 1), dtype=np.int64), ['other']

    # Map hash → size (first seen wins; should be deterministic).
    hash_size = {}
    incidence = Counter()
    for sh, ss, ph, ps in edges:
        hash_size.setdefault(sh, ss)
        hash_size.setdefault(ph, ps)
        incidence[sh] += 1
        incidence[ph] += 1

    # Pick top K by incidence; sort selected by (size ascending, hash ascending).
    top_hashes = [h for h, _ in incidence.most_common(K)]
    top_hashes.sort(key=lambda h: (hash_size[h], h))
    h_to_idx = {h: i for i, h in enumerate(top_hashes)}
    other_idx = K  # last row/col

    matrix = np.zeros((K + 1, K + 1), dtype=np.int64)
    for sh, _ss, ph, _ps in edges:
        i = h_to_idx.get(sh, other_idx)
        j = h_to_idx.get(ph, other_idx)
        matrix[i, j] += 1

    labels = [f"0x{h:08x}" for h in top_hashes] + ['other']
    # Trim if there were fewer than K unique hashes.
    actual_k = len(top_hashes)
    if actual_k < K:
        matrix = matrix[:actual_k + 1, :actual_k + 1]
        labels = labels[:actual_k + 1]
    return matrix, labels


def full_transition_matrix(
    events: ReactionEvent,
) -> Tuple[np.ndarray, List[str]]:
    """(U, U) matrix over every observed unique hash, sorted by size ascending."""
    edges = list(_iter_edges(events))
    if not edges:
        return np.zeros((0, 0), dtype=np.int64), []

    hash_size = {}
    for sh, ss, ph, ps in edges:
        hash_size.setdefault(sh, ss)
        hash_size.setdefault(ph, ps)

    sorted_hashes = sorted(hash_size, key=lambda h: (hash_size[h], h))
    h_to_idx = {h: i for i, h in enumerate(sorted_hashes)}
    U = len(sorted_hashes)
    matrix = np.zeros((U, U), dtype=np.int64)
    for sh, _ss, ph, _ps in edges:
        matrix[h_to_idx[sh], h_to_idx[ph]] += 1
    labels = [f"0x{h:08x}" for h in sorted_hashes]
    return matrix, labels
