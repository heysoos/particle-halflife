"""Unit tests for halflife/analysis/openendedness.py."""
import numpy as np
import pytest

from halflife.analysis import openendedness as oe


def test_slice_windows_equal_count():
    assert oe.slice_windows(15000, windows=5) == [
        (0, 3000), (3000, 6000), (6000, 9000), (9000, 12000), (12000, 15000)
    ]


def test_slice_windows_count_absorbs_remainder():
    # 10 / 3 → last window absorbs the extra step
    assert oe.slice_windows(10, windows=3) == [(0, 3), (3, 6), (6, 10)]


def test_slice_windows_fixed_width():
    assert oe.slice_windows(15000, window_width=4000) == [
        (0, 4000), (4000, 8000), (8000, 12000), (12000, 15000)
    ]


def test_slice_windows_default_is_five():
    assert len(oe.slice_windows(1000)) == 5


def test_slice_windows_single_window():
    assert oe.slice_windows(500, windows=1) == [(0, 500)]


def test_slice_windows_width_ge_nsteps():
    assert oe.slice_windows(500, window_width=9999) == [(0, 500)]


def test_slice_windows_mutually_exclusive():
    with pytest.raises(ValueError):
        oe.slice_windows(1000, windows=5, window_width=100)


def test_window_index_assignment():
    w = [(0, 3000), (3000, 6000), (6000, 9000)]
    assert oe._window_index(0, w) == 0
    assert oe._window_index(2999, w) == 0
    assert oe._window_index(3000, w) == 1
    assert oe._window_index(9000, w) == 2   # final end is inclusive
    assert oe._window_index(99999, w) is None


from dataclasses import dataclass


@dataclass
class _Snap:
    """Minimal stand-in for CompositeSnapshot (same field names)."""
    alive: np.ndarray
    member_count: np.ndarray
    species_hash: np.ndarray
    members: np.ndarray
    edges: np.ndarray
    edge_count: np.ndarray


def _snap_two_composites(edges_c0, edges_c1, species_by_pid):
    """Build a 2-composite snapshot. Composite 0 uses particle ids 0..2,
    composite 1 uses 3..5. edges_* are lists of (pid_a, pid_b)."""
    C, M, E = 2, 6, 6
    alive = np.array([True, True])
    members = np.full((C, M), -1, np.int32)
    members[0, :3] = [0, 1, 2]
    members[1, :3] = [3, 4, 5]
    member_count = np.array([3, 3], np.int32)
    edges = np.full((C, E, 2), -1, np.int32)
    edge_count = np.zeros(C, np.int32)
    for c, elist in enumerate((edges_c0, edges_c1)):
        for k, (a, b) in enumerate(elist):
            edges[c, k] = [a, b]
        edge_count[c] = len(elist)
    species_hash = np.array([111, 222], np.uint32)
    return _Snap(alive, member_count, species_hash, members, edges, edge_count), \
        np.asarray(species_by_pid, np.int32)


def test_composition_type_ids_are_species_hash_of_alive():
    snap, species = _snap_two_composites([(0, 1)], [(3, 4)], [0, 0, 0, 0, 0, 0])
    ids = oe.composition_type_ids(snap)
    assert ids.dtype == np.uint64
    assert sorted(ids.tolist()) == [111, 222]


def test_composition_ignores_dead_composites():
    snap, species = _snap_two_composites([(0, 1)], [(3, 4)], [0] * 6)
    snap.alive[1] = False
    assert oe.composition_type_ids(snap).tolist() == [111]


def test_structure_chain_differs_from_ring():
    # Both composites: 3 particles, all species 0. c0 = chain (2 edges),
    # c1 = ring/triangle (3 edges). Same atoms, different topology.
    snap, species = _snap_two_composites(
        [(0, 1), (1, 2)], [(3, 4), (4, 5), (5, 3)], [0, 0, 0, 0, 0, 0])
    ids = oe.structure_type_ids(snap, species)
    assert ids[0] != ids[1]


def test_structure_isomorphic_graphs_match():
    # c0 chain 0-1-2, c1 chain 3-4-5 with reversed edge order + same species.
    snap, species = _snap_two_composites(
        [(0, 1), (1, 2)], [(5, 4), (4, 3)], [0, 0, 0, 0, 0, 0])
    ids = oe.structure_type_ids(snap, species)
    assert ids[0] == ids[1]


def test_structure_species_labels_matter():
    # Same chain topology but different species labeling → different structure id.
    snap, species = _snap_two_composites(
        [(0, 1), (1, 2)], [(3, 4), (4, 5)], [0, 0, 0, 1, 0, 0])
    ids = oe.structure_type_ids(snap, species)
    assert ids[0] != ids[1]


def test_structure_edgeless_multimember_falls_back_to_composition():
    # edge_count 0 but 3 members (valence-off / star_spring): structure id is a
    # deterministic function of the species multiset, equal for identical multisets.
    snap, species = _snap_two_composites([], [], [2, 2, 2, 2, 2, 2])
    ids = oe.structure_type_ids(snap, species)
    assert ids[0] == ids[1]


def test_discovery_curve_is_cumulative_distinct():
    # 3 snapshots: {A}, {A,B}, {A,B,C} → cumulative 1, 2, 3.
    sets = [np.array([10], np.uint64),
            np.array([10, 20], np.uint64),
            np.array([10, 20, 30], np.uint64)]
    steps = [100, 200, 300]
    s, cum = oe.discovery_curve(sets, steps)
    assert s.tolist() == [100, 200, 300]
    assert cum.tolist() == [1, 2, 3]


def test_discovery_curve_no_double_count():
    sets = [np.array([10, 10], np.uint64), np.array([10], np.uint64)]
    _, cum = oe.discovery_curve(sets, [100, 200])
    assert cum.tolist() == [1, 1]


def test_novelty_rate_bins_first_appearance_by_window():
    # A,B first seen in window 0 (steps 100,200); C in window 1 (step 400).
    sets = [np.array([10], np.uint64), np.array([10, 20], np.uint64),
            np.array([10, 20, 30], np.uint64)]
    steps = [100, 200, 400]
    windows = [(0, 300), (300, 600)]
    counts = oe.novelty_rate(sets, steps, windows)
    assert counts.tolist() == [2, 1]


def test_total_composition_types_from_events_counts_distinct_products():
    from halflife.state import ReactionEvent
    ev = ReactionEvent(
        kind=np.array([1, 1, 2], np.int32),
        source_slots=np.zeros((3, 2), np.int32),
        source_hashes=np.zeros((3, 2), np.uint32),
        source_sizes=np.zeros((3, 2), np.int32),
        product_slots=np.zeros((3, 2), np.int32),
        product_hashes=np.array([[5, 0], [5, 0], [7, 9]], np.uint32),
        product_sizes=np.array([[3, 1], [3, 1], [2, 2]], np.int32),
    )
    # Distinct product hashes among size>=2 products: {5, 7, 9} → 3.
    assert oe.total_composition_types_from_events(ev) == 3
