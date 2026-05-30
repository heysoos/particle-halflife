"""Tests for the event-log → transition-matrix builder."""
import numpy as np
import pytest

from halflife.state import ReactionEvent
from halflife.analysis.transitions import (
    size_bin_transition_matrix,
    top_k_transition_matrix,
    full_transition_matrix,
)
from halflife.analysis.events import KIND_FUSION, KIND_FISSION


def _make_event(kind, src_hashes, src_sizes, prod_hashes, prod_sizes):
    """Build a 1-event ReactionEvent from a small spec."""
    return ReactionEvent(
        kind=np.array([kind], dtype=np.int32),
        source_slots=np.array([[0, 1 if kind == KIND_FUSION else -1]], dtype=np.int32),
        source_hashes=np.array([src_hashes], dtype=np.uint32),
        source_sizes=np.array([src_sizes], dtype=np.int32),
        product_slots=np.array([[10, 11 if kind == KIND_FISSION else -1]], dtype=np.int32),
        product_hashes=np.array([prod_hashes], dtype=np.uint32),
        product_sizes=np.array([prod_sizes], dtype=np.int32),
    )


def _concat_events(*evs):
    return ReactionEvent(
        kind=np.concatenate([e.kind for e in evs]),
        source_slots=np.concatenate([e.source_slots for e in evs]),
        source_hashes=np.concatenate([e.source_hashes for e in evs]),
        source_sizes=np.concatenate([e.source_sizes for e in evs]),
        product_slots=np.concatenate([e.product_slots for e in evs]),
        product_hashes=np.concatenate([e.product_hashes for e in evs]),
        product_sizes=np.concatenate([e.product_sizes for e in evs]),
    )


def test_size_bin_matrix_fusion_contributes_to_both_source_rows():
    """One fusion A(size 2) + B(size 3) → C(size 5) should add to cells (2,5) and (3,5)."""
    evt = _make_event(
        KIND_FUSION,
        src_hashes=[100, 200], src_sizes=[2, 3],
        prod_hashes=[300, 0], prod_sizes=[5, 0],
    )
    M = size_bin_transition_matrix(evt, max_composite_size=8)
    assert M[2, 5] == 1
    assert M[3, 5] == 1
    # No other cells should be set.
    M[2, 5] = 0
    M[3, 5] = 0
    assert M.sum() == 0


def test_size_bin_matrix_fission_contributes_to_both_product_cols():
    """One fission C(size 5) → A(size 2) + B(size 3) should add to cells (5,2) and (5,3)."""
    evt = _make_event(
        KIND_FISSION,
        src_hashes=[300, 0], src_sizes=[5, 0],
        prod_hashes=[100, 200], prod_sizes=[2, 3],
    )
    M = size_bin_transition_matrix(evt, max_composite_size=8)
    assert M[5, 2] == 1
    assert M[5, 3] == 1
    M[5, 2] = 0
    M[5, 3] = 0
    assert M.sum() == 0


def test_top_k_matrix_buckets_tail_into_other():
    """K=2 with 4 unique hashes: top 2 stay, remaining 2 collapse into 'other' row/col."""
    # Build several events so we have rankable hashes.
    evts = [
        _make_event(KIND_FUSION, [1, 2], [1, 1], [3, 0], [2, 0]),  # hashes 1,2 → 3
        _make_event(KIND_FUSION, [1, 2], [1, 1], [3, 0], [2, 0]),  # again — boosts 1,2,3
        _make_event(KIND_FUSION, [4, 5], [1, 1], [9, 0], [2, 0]),  # rare hashes 4,5,9
    ]
    batch = _concat_events(*evts)
    M, labels = top_k_transition_matrix(batch, K=2)
    # Shape: K+1 by K+1 with "other" appended.
    assert M.shape == (3, 3)
    assert labels[-1] == 'other'


def test_full_transition_matrix_uses_every_observed_hash():
    """U×U where U = unique hashes across all events."""
    evt = _make_event(KIND_FUSION, [1, 2], [1, 1], [3, 0], [2, 0])
    M, labels = full_transition_matrix(evt)
    assert len(labels) == 3  # hashes 1, 2, 3
    assert M.shape == (3, 3)
