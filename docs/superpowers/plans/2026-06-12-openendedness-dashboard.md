# Open-Endedness Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Tier 5 — Open-Endedness & Temporal Evolution" section to the composite diagnostic report that quantifies novelty accumulation over long runs across two type axes (composition and structure), with time-window comparison, then render it on a 15k-step run.

**Architecture:** One new pure-Python library module (`halflife/analysis/openendedness.py`) computes type identities, windowing, and five metrics from a cached `RunResult` — no JAX, no I/O. The report layer (`plots.py`, `report.py`) consumes its arrays; the CLI gains two windowing flags. Everything is host-side post-processing, so it works under `--from-cache` with no re-simulation.

**Tech Stack:** Python, numpy, matplotlib (Agg backend, base64-PNG embedding), pytest. Reuses the existing `halflife/analysis/` pipeline (`RunResult`, `CompositeSnapshot`, `ReactionEvent`).

**Spec:** [docs/superpowers/specs/2026-06-12-openendedness-dashboard-design.md](../specs/2026-06-12-openendedness-dashboard-design.md)

---

## File Structure

| File | Responsibility |
|---|---|
| `halflife/analysis/openendedness.py` (create) | Type-identity (composition + WL structure hash), `slice_windows`, 5 metric functions, headline scalar. Pure library. |
| `tests/test_analysis_openendedness.py` (create) | Unit tests for windowing, WL hash, and each metric on synthetic data. |
| `halflife/analysis/plots.py` (modify) | 5 Tier-5 plot builders returning base64 PNGs. |
| `halflife/analysis/report.py` (modify) | Tier 5 HTML section; `render_html` gains `windows`/`window_width` params and builds type-id arrays. |
| `halflife/analysis/cli.py` (modify) | `--windows` / `--window-width` flags, mutual-exclusion validation, thread into `render_html`. |
| `tests/test_analysis_pipeline.py` (modify) | Assert Tier 5 markup is present end-to-end. |
| `CLAUDE.md` (modify) | Document the new flags and Tier 5 in the analysis section. |

**Data representation convention (used by every metric):** a snapshot's type
membership is a **per-composite array with repeats** — one type id per alive
composite (`(n_alive,)`). `np.unique` dedups where a set is needed; counts give
abundance. Both axes (composition, structure) produce this same shape, so all
metrics are axis-agnostic and called once per axis.

---

## Task 1: Windowing helper

**Files:**
- Create: `halflife/analysis/openendedness.py`
- Test: `tests/test_analysis_openendedness.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analysis_openendedness.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_analysis_openendedness.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'halflife.analysis.openendedness'`

- [ ] **Step 3: Write minimal implementation**

```python
# halflife/analysis/openendedness.py
"""Open-endedness & temporal-evolution metrics for the diagnostic report.

Pure host-side post-processing over a cached RunResult — no JAX, no I/O. Two
type axes are reported: composition (the species multiset, via species_hash)
and structure (the bond-graph topology, via a Weisfeiler-Lehman hash). Every
metric consumes a list of per-snapshot "type id arrays" (one id per alive
composite, repeats allowed) so it is agnostic to which axis it is fed.

All time-resolved metrics are at SNAPSHOT cadence: the flat event stream in
RunResult loses per-step alignment, and structure ids need the edge arrays
that only snapshots carry. See the design spec for the trade-off.
"""

import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Windowing ───────────────────────────────────────────────────────────────

def slice_windows(n_steps: int,
                  windows: Optional[int] = None,
                  window_width: Optional[int] = None) -> List[Tuple[int, int]]:
    """Slice [0, n_steps) into (start, end) ranges.

    windows=N        → N equal ranges; the last absorbs any remainder.
    window_width=W   → ceil(n_steps/W) ranges of width W (last may be shorter).
    both None        → default 5 windows.
    both set         → ValueError (mutually exclusive).
    """
    if windows is not None and window_width is not None:
        raise ValueError("slice_windows: pass windows OR window_width, not both")
    if windows is None and window_width is None:
        windows = 5

    if windows is not None:
        windows = max(1, min(int(windows), n_steps))
        base = n_steps // windows
        bounds = []
        start = 0
        for i in range(windows):
            end = n_steps if i == windows - 1 else start + base
            bounds.append((start, end))
            start = end
        return bounds

    w = int(window_width)
    bounds = []
    start = 0
    while start < n_steps:
        bounds.append((start, min(start + w, n_steps)))
        start += w
    return bounds


def _window_index(step: int, windows: List[Tuple[int, int]]) -> Optional[int]:
    """Index of the window containing `step`; final window's end is inclusive."""
    for i, (start, end) in enumerate(windows):
        if start <= step < end:
            return i
    if windows and step == windows[-1][1]:
        return len(windows) - 1
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_analysis_openendedness.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add halflife/analysis/openendedness.py tests/test_analysis_openendedness.py
git commit -m "feat(analysis): window-slicing helper for open-endedness metrics

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 2: Type identity (composition + Weisfeiler–Lehman structure hash)

**Files:**
- Modify: `halflife/analysis/openendedness.py`
- Test: `tests/test_analysis_openendedness.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_analysis_openendedness.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_analysis_openendedness.py -k "composition or structure" -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'composition_type_ids'`

- [ ] **Step 3: Write minimal implementation**

```python
# Append to halflife/analysis/openendedness.py

# ── Type identity ───────────────────────────────────────────────────────────

def _stable_hash(obj) -> int:
    """Deterministic 64-bit hash of a hashable structure (process-stable, unlike
    Python's salted hash() for strings/bytes)."""
    return int.from_bytes(
        hashlib.blake2b(repr(obj).encode('utf-8'), digest_size=8).digest(),
        'little',
    )


def composition_type_ids(snapshot) -> np.ndarray:
    """Per-alive-composite composition type ids (= species_hash). uint64, repeats."""
    return snapshot.species_hash[snapshot.alive].astype(np.uint64)


def _wl_graph_hash(node_labels: List[int],
                   adjacency: List[List[int]],
                   rounds: int = 3) -> int:
    """Weisfeiler-Lehman hash of a labeled undirected graph.

    Each round replaces a node's label with a hash of (own label, sorted
    multiset of neighbor labels); the graph hash is a hash of the sorted final
    label multiset. Isomorphic labeled graphs hash equal; chain vs ring of the
    same atoms hash differently. WL can rarely collide on non-isomorphic regular
    graphs — accepted for this scale (see spec).
    """
    labels = [int(s) for s in node_labels]
    for _ in range(rounds):
        labels = [
            _stable_hash((labels[v], tuple(sorted(labels[u] for u in adjacency[v]))))
            for v in range(len(labels))
        ]
    return _stable_hash(tuple(sorted(labels)))


def structure_type_ids(snapshot, particles_species: np.ndarray,
                       rounds: int = 3) -> np.ndarray:
    """Per-alive-composite structure type ids (WL hash of the bond graph).

    particles_species: (num_particles,) int32 — species are constant over a run,
    so the caller reconstructs them once (re-init world with the run seed).
    A multi-member composite with edge_count == 0 (valence-off / star_spring,
    where edges are physics-inert) falls back to a hash of its species multiset.
    """
    alive_idx = np.nonzero(snapshot.alive)[0]
    out = np.empty(len(alive_idx), dtype=np.uint64)
    for k, c in enumerate(alive_idx):
        n = int(snapshot.member_count[c])
        members = snapshot.members[c, :n]
        members = members[members >= 0]
        local = {int(pid): i for i, pid in enumerate(members)}
        node_labels = [int(particles_species[pid]) for pid in members]
        adjacency: List[List[int]] = [[] for _ in range(len(members))]
        ec = int(snapshot.edge_count[c])
        for e in range(ec):
            a, b = int(snapshot.edges[c, e, 0]), int(snapshot.edges[c, e, 1])
            if a in local and b in local:
                adjacency[local[a]].append(local[b])
                adjacency[local[b]].append(local[a])
        if ec == 0 and len(members) > 1:
            h = _stable_hash(('composition', tuple(sorted(node_labels))))
        else:
            h = _wl_graph_hash(node_labels, adjacency, rounds)
        out[k] = np.uint64(h)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_analysis_openendedness.py -k "composition or structure" -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add halflife/analysis/openendedness.py tests/test_analysis_openendedness.py
git commit -m "feat(analysis): composition + Weisfeiler-Lehman structure type ids

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 3: Discovery curve, novelty rate, headline scalar

**Files:**
- Modify: `halflife/analysis/openendedness.py`
- Test: `tests/test_analysis_openendedness.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_analysis_openendedness.py

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_analysis_openendedness.py -k "discovery or novelty or total_composition" -v`
Expected: FAIL with `AttributeError: ... has no attribute 'discovery_curve'`

- [ ] **Step 3: Write minimal implementation**

```python
# Append to halflife/analysis/openendedness.py

# ── Time-resolved novelty ───────────────────────────────────────────────────

def discovery_curve(type_id_arrays: List[np.ndarray],
                    snapshot_steps: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Cumulative count of distinct type ids ever seen up to each snapshot."""
    seen = set()
    cum = []
    for ids in type_id_arrays:
        seen.update(ids.tolist())
        cum.append(len(seen))
    return np.asarray(snapshot_steps), np.asarray(cum, dtype=np.int64)


def _first_appearance(type_id_arrays: List[np.ndarray],
                      snapshot_steps: List[int]) -> Dict[int, int]:
    """Map each type id to the step of the first snapshot it appears in."""
    first: Dict[int, int] = {}
    for ids, step in zip(type_id_arrays, snapshot_steps):
        for t in ids.tolist():
            first.setdefault(t, step)
    return first


def novelty_rate(type_id_arrays: List[np.ndarray],
                 snapshot_steps: List[int],
                 windows: List[Tuple[int, int]]) -> np.ndarray:
    """Count of types whose first snapshot-appearance falls in each window."""
    first = _first_appearance(type_id_arrays, snapshot_steps)
    counts = np.zeros(len(windows), dtype=np.int64)
    for step in first.values():
        wi = _window_index(step, windows)
        if wi is not None:
            counts[wi] += 1
    return counts


def total_composition_types_from_events(events) -> int:
    """Distinct composite product hashes (size >= 2) across the complete event
    stream — the true total incl. types born and gone between snapshots."""
    if events.product_hashes.size == 0:
        return 0
    mask = events.product_sizes >= 2
    vals = events.product_hashes[mask]
    return int(np.unique(vals).size)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_analysis_openendedness.py -k "discovery or novelty or total_composition" -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add halflife/analysis/openendedness.py tests/test_analysis_openendedness.py
git commit -m "feat(analysis): discovery curve, novelty rate, event-total headline

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 4: Hill diversity

**Files:**
- Modify: `halflife/analysis/openendedness.py`
- Test: `tests/test_analysis_openendedness.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_analysis_openendedness.py

def test_hill_diversity_uniform_three_types():
    # 3 distinct types, one composite each → q0=q1=q2=3 (perfectly even).
    sets = [np.array([1, 2, 3], np.uint64)]
    h = oe.hill_diversity(sets)
    assert h['q0'][0] == 3.0
    assert abs(h['q1'][0] - 3.0) < 1e-9
    assert abs(h['q2'][0] - 3.0) < 1e-9


def test_hill_diversity_skewed():
    # Type 1 dominates (8) vs type 2 (rare, 2): richness=2 but effective < 2.
    sets = [np.array([1] * 8 + [2] * 2, np.uint64)]
    h = oe.hill_diversity(sets)
    assert h['q0'][0] == 2.0
    assert 1.0 < h['q1'][0] < 2.0
    assert h['q2'][0] < h['q1'][0]   # inverse-Simpson penalizes dominance more


def test_hill_diversity_empty_snapshot_is_zero():
    h = oe.hill_diversity([np.array([], np.uint64)])
    assert h['q0'][0] == 0.0 and h['q1'][0] == 0.0 and h['q2'][0] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_analysis_openendedness.py -k "hill" -v`
Expected: FAIL with `AttributeError: ... has no attribute 'hill_diversity'`

- [ ] **Step 3: Write minimal implementation**

```python
# Append to halflife/analysis/openendedness.py

# ── Diversity ───────────────────────────────────────────────────────────────

def hill_diversity(type_id_arrays: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """Per-snapshot Hill numbers of the currently-alive type distribution.

    Abundance of a type = number of alive composites of that type (each
    composite is one individual). q0 = richness, q1 = exp(Shannon),
    q2 = inverse-Simpson. Empty snapshots → 0.
    """
    n = len(type_id_arrays)
    q0 = np.zeros(n); q1 = np.zeros(n); q2 = np.zeros(n)
    for i, ids in enumerate(type_id_arrays):
        if ids.size == 0:
            continue
        _, counts = np.unique(ids, return_counts=True)
        p = counts / counts.sum()
        q0[i] = counts.size
        q1[i] = np.exp(-np.sum(p * np.log(p)))
        q2[i] = 1.0 / np.sum(p * p)
    return {'q0': q0, 'q1': q1, 'q2': q2}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_analysis_openendedness.py -k "hill" -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add halflife/analysis/openendedness.py tests/test_analysis_openendedness.py
git commit -m "feat(analysis): Hill-number diversity of alive composite types

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 5: Window turnover + per-window size histograms

**Files:**
- Modify: `halflife/analysis/openendedness.py`
- Test: `tests/test_analysis_openendedness.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_analysis_openendedness.py

def test_window_turnover_identical_windows_zero():
    # Two windows with identical type composition → 0 dissimilarity off-diagonal.
    sets = [np.array([1, 2], np.uint64), np.array([1, 2], np.uint64)]
    steps = [100, 400]
    windows = [(0, 300), (300, 600)]
    t = oe.window_turnover(sets, steps, windows)
    assert t['jaccard'].shape == (2, 2)
    assert t['jaccard'][0, 1] == 0.0
    assert t['bray_curtis'][0, 1] == 0.0


def test_window_turnover_disjoint_windows_one():
    # Completely different types between windows → Jaccard = 1.
    sets = [np.array([1, 1], np.uint64), np.array([9, 9], np.uint64)]
    steps = [100, 400]
    windows = [(0, 300), (300, 600)]
    t = oe.window_turnover(sets, steps, windows)
    assert t['jaccard'][0, 1] == 1.0
    assert t['bray_curtis'][0, 1] == 1.0


def test_window_turnover_diagonal_is_self_zero():
    sets = [np.array([1, 2], np.uint64), np.array([2, 3], np.uint64)]
    steps = [100, 400]
    windows = [(0, 300), (300, 600)]
    t = oe.window_turnover(sets, steps, windows)
    assert t['jaccard'][0, 0] == 0.0 and t['jaccard'][1, 1] == 0.0


def test_per_window_size_hist_means_rows():
    per_step = {'size_histogram': np.array([
        [0, 2, 0], [0, 4, 0],          # window 0 (steps 0-1) → mean [0,3,0]
        [0, 0, 6], [0, 0, 8],          # window 1 (steps 2-3) → mean [0,0,7]
    ], dtype=np.float32)}
    windows = [(0, 2), (2, 4)]
    out = oe.per_window_size_hist(per_step, windows)
    assert np.allclose(out[0], [0, 3, 0])
    assert np.allclose(out[1], [0, 0, 7])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_analysis_openendedness.py -k "turnover or per_window_size" -v`
Expected: FAIL with `AttributeError: ... has no attribute 'window_turnover'`

- [ ] **Step 3: Write minimal implementation**

```python
# Append to halflife/analysis/openendedness.py

# ── Turnover & size facets ──────────────────────────────────────────────────

def window_turnover(type_id_arrays: List[np.ndarray],
                    snapshot_steps: List[int],
                    windows: List[Tuple[int, int]]) -> Dict[str, np.ndarray]:
    """Pairwise window dissimilarity over aggregated type composition.

    jaccard:     1 - |A∩B| / |A∪B| on presence/absence type sets.
    bray_curtis: Σ|a_k - b_k| / Σ(a_k + b_k) on per-window summed abundance.
    Empty-vs-empty window pairs → NaN (rendered as a neutral cell).
    """
    nW = len(windows)
    win_ab: List[Dict[int, int]] = [dict() for _ in range(nW)]
    for ids, step in zip(type_id_arrays, snapshot_steps):
        wi = _window_index(step, windows)
        if wi is None or ids.size == 0:
            continue
        u, c = np.unique(ids, return_counts=True)
        d = win_ab[wi]
        for t, cnt in zip(u.tolist(), c.tolist()):
            d[t] = d.get(t, 0) + int(cnt)

    jac = np.full((nW, nW), np.nan)
    bc = np.full((nW, nW), np.nan)
    for i in range(nW):
        for j in range(nW):
            a, b = win_ab[i], win_ab[j]
            union = set(a) | set(b)
            if not union:
                continue
            inter = set(a) & set(b)
            jac[i, j] = 1.0 - len(inter) / len(union)
            num = sum(abs(a.get(t, 0) - b.get(t, 0)) for t in union)
            den = sum(a.get(t, 0) + b.get(t, 0) for t in union)
            bc[i, j] = (num / den) if den else np.nan
    return {'jaccard': jac, 'bray_curtis': bc}


def per_window_size_hist(per_step_metrics: Dict[str, np.ndarray],
                         windows: List[Tuple[int, int]]) -> List[np.ndarray]:
    """Mean composite-size histogram within each window (from per-step metrics)."""
    hist = per_step_metrics['size_histogram']   # (n_steps, S)
    out = []
    for (start, end) in windows:
        seg = hist[start:end]
        out.append(seg.mean(axis=0) if len(seg) else np.zeros(hist.shape[1]))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_analysis_openendedness.py -v`
Expected: PASS (all tests in the file — ~24)

- [ ] **Step 5: Commit**

```bash
git add halflife/analysis/openendedness.py tests/test_analysis_openendedness.py
git commit -m "feat(analysis): window turnover (Jaccard/Bray-Curtis) + size facets

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 6: Tier 5 plot builders

**Files:**
- Modify: `halflife/analysis/plots.py`
- Test: `tests/test_analysis_openendedness.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_analysis_openendedness.py

def test_tier5_plot_builders_return_base64():
    from halflife.analysis import plots
    steps = np.array([100, 200, 300])
    win_labels = ['W1\n0-150', 'W2\n150-300']

    a = plots.plot_discovery_curves(
        steps, np.array([1, 2, 3]), np.array([1, 1, 2]), total_comp_events=5)
    b = plots.plot_novelty_rate(win_labels, np.array([2, 1]), np.array([1, 1]))
    c = plots.plot_hill_diversity(
        steps,
        {'q0': np.array([1., 2., 3.]), 'q1': np.array([1., 2., 3.]), 'q2': np.array([1., 2., 3.])},
        {'q0': np.array([1., 1., 2.]), 'q1': np.array([1., 1., 2.]), 'q2': np.array([1., 1., 2.])},
    )
    d = plots.plot_turnover_grid(
        {'jaccard': np.zeros((2, 2)), 'bray_curtis': np.zeros((2, 2))},
        {'jaccard': np.zeros((2, 2)), 'bray_curtis': np.zeros((2, 2))},
        win_labels,
    )
    e = plots.plot_window_size_facets(
        [np.array([0, 3, 0]), np.array([0, 0, 7])], win_labels)

    for img in (a, b, c, d, e):
        assert isinstance(img, str) and len(img) > 100   # non-empty base64
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_analysis_openendedness.py -k "tier5_plot" -v`
Expected: FAIL with `AttributeError: module 'halflife.analysis.plots' has no attribute 'plot_discovery_curves'`

- [ ] **Step 3: Write minimal implementation**

```python
# Append to halflife/analysis/plots.py
# (uses _fig_to_base64, _style_matrix_axes, _C_PRIMARY, _C_SECONDARY,
#  _C_TERTIARY, _C_ACCENT already defined at the top of this module.)

# ── Tier 5: Open-endedness & temporal evolution ─────────────────────────────

def plot_discovery_curves(steps, comp_cum, struct_cum, total_comp_events: int) -> str:
    """Cumulative distinct-types vs step, composition + structure overlaid."""
    fig = Figure(figsize=(5.5, 2.6), constrained_layout=True)
    ax = fig.subplots()
    ax.plot(steps, comp_cum, color=_C_PRIMARY, lw=1.6, label='composition')
    ax.plot(steps, struct_cum, color=_C_ACCENT, lw=1.6, label='structure')
    ax.axhline(total_comp_events, color=_C_SECONDARY, lw=0.9, ls='--',
               label=f'all composition types (events): {total_comp_events}')
    ax.set_xlabel('step'); ax.set_ylabel('distinct types (cumulative)')
    ax.set_title('Type discovery curve')
    ax.grid(True); ax.legend(loc='upper left')
    return _fig_to_base64(fig)


def plot_novelty_rate(window_labels, comp_counts, struct_counts) -> str:
    """Grouped bars: new types first seen per window, both axes."""
    fig = Figure(figsize=(5.5, 2.6), constrained_layout=True)
    ax = fig.subplots()
    x = np.arange(len(window_labels))
    ax.bar(x - 0.2, comp_counts, width=0.4, color=_C_PRIMARY, label='composition')
    ax.bar(x + 0.2, struct_counts, width=0.4, color=_C_ACCENT, label='structure')
    ax.set_xticks(x); ax.set_xticklabels(window_labels, fontsize=6.5)
    ax.set_ylabel('new types'); ax.set_title('Novelty rate per window')
    ax.grid(True, axis='y'); ax.legend()
    return _fig_to_base64(fig)


def _plot_hill_panel(ax, steps, hill, title):
    ax.plot(steps, hill['q0'], color=_C_SECONDARY, lw=1.4, label='q=0 richness')
    ax.plot(steps, hill['q1'], color=_C_PRIMARY, lw=1.4, label='q=1 Shannon')
    ax.plot(steps, hill['q2'], color=_C_TERTIARY, lw=1.4, label='q=2 Simpson')
    ax.set_xlabel('step'); ax.set_ylabel('effective # types')
    ax.set_title(title); ax.grid(True); ax.legend(fontsize=6.5)


def plot_hill_diversity(steps, comp_hill, struct_hill) -> str:
    """Two side-by-side panels: alive-type diversity for each axis."""
    fig = Figure(figsize=(5.5, 2.6), constrained_layout=True)
    ax1, ax2 = fig.subplots(1, 2)
    _plot_hill_panel(ax1, steps, comp_hill, 'Composition diversity')
    _plot_hill_panel(ax2, steps, struct_hill, 'Structure diversity')
    return _fig_to_base64(fig)


def plot_turnover_grid(comp_turnover, struct_turnover, window_labels) -> str:
    """2×2 heatmaps: {Jaccard, Bray-Curtis} × {composition, structure}."""
    fig = Figure(figsize=(5.5, 5.2), constrained_layout=True)
    axes = fig.subplots(2, 2)
    panels = [
        (axes[0, 0], comp_turnover['jaccard'],     'Composition · Jaccard'),
        (axes[0, 1], struct_turnover['jaccard'],   'Structure · Jaccard'),
        (axes[1, 0], comp_turnover['bray_curtis'], 'Composition · Bray-Curtis'),
        (axes[1, 1], struct_turnover['bray_curtis'],'Structure · Bray-Curtis'),
    ]
    for ax, M, title in panels:
        im = ax.imshow(np.nan_to_num(M, nan=0.0), cmap='viridis', vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(window_labels)))
        ax.set_yticks(range(len(window_labels)))
        ax.set_xticklabels([l.split('\n')[0] for l in window_labels], fontsize=6.5)
        ax.set_yticklabels([l.split('\n')[0] for l in window_labels], fontsize=6.5)
        ax.set_title(title, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _fig_to_base64(fig)


def plot_window_size_facets(per_window_hist, window_labels) -> str:
    """Overlaid composite-size distributions, one line per window."""
    fig = Figure(figsize=(5.5, 2.6), constrained_layout=True)
    ax = fig.subplots()
    cmap = plt.get_cmap('plasma')
    n = len(per_window_hist)
    for i, (h, label) in enumerate(zip(per_window_hist, window_labels)):
        sizes = np.arange(1, len(h) + 1)
        ax.plot(sizes, h, lw=1.3, color=cmap(i / max(n - 1, 1)),
                label=label.split('\n')[0])
    ax.set_xlabel('composite size'); ax.set_ylabel('mean count')
    ax.set_yscale('symlog'); ax.set_title('Size distribution by window')
    ax.grid(True); ax.legend(fontsize=6.5)
    return _fig_to_base64(fig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_analysis_openendedness.py -k "tier5_plot" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add halflife/analysis/plots.py tests/test_analysis_openendedness.py
git commit -m "feat(analysis): Tier 5 plot builders (discovery/novelty/diversity/turnover/size)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 7: Tier 5 report section + render_html wiring

**Files:**
- Modify: `halflife/analysis/report.py` (template ~line 270, `render_html` ~line 343–517)
- Test: covered by Task 9's pipeline test

- [ ] **Step 1: Add the Tier 5 HTML block to the template**

In `halflife/analysis/report.py`, find the `<footer>` near the end of
`_HTML_TEMPLATE` (currently right after the Tier 4 `</div>`, ~line 272) and
insert this block immediately BEFORE `<footer>`. This matches the verified
Tier 1/2 markup exactly: `<div class="plot-grid">` wrapping bare `<img>` tags
(this codebase does NOT use `<figure>`/`<figcaption>`), captions go in the
`<h3>`/`<p class="note">` text and the existing matrix uses `class="matrix-wrap"`:

```html
<h2>Tier 5 — Open-endedness &amp; temporal evolution</h2>
<p class="note">Two type axes: <strong>composition</strong> (species multiset) and <strong>structure</strong> (bond-graph topology, Weisfeiler-Lehman hash). Time-resolved at snapshot cadence ({sample_every} steps); {n_windows} windows.</p>
<p class="note">A discovery curve that keeps climbing and sustained per-window novelty/turnover indicate ongoing open-endedness; a plateau means the chemistry has closed. Structure metrics are only meaningful in <code>bond_mode="edges"</code> runs.</p>

<h3>5a–c, 5e: discovery, novelty, diversity, size by window</h3>
<div class="plot-grid">
  <img src="data:image/png;base64,{img_oe_discovery}">
  <img src="data:image/png;base64,{img_oe_novelty}">
  <img src="data:image/png;base64,{img_oe_diversity}">
  <img src="data:image/png;base64,{img_oe_size_facets}">
</div>
<h3>5d: Window-to-window turnover</h3>
<div class="matrix-wrap"><img src="data:image/png;base64,{img_oe_turnover}"></div>
```

- [ ] **Step 2: Build the Tier 5 images and format args in `render_html`**

Change the `render_html` signature (line 343) to:

```python
def render_html(result: RunResult, top_k: int = 30,
                windows: int = None, window_width: int = None) -> str:
```

`particles_species` is already reconstructed at ~line 355. Immediately BEFORE
the final `return _HTML_TEMPLATE.format(` (line 478), add:

```python
    # ── Tier 5: open-endedness ───────────────────────────────────────────────
    from halflife.analysis import openendedness as oe
    win = oe.slice_windows(result.n_steps, windows=windows, window_width=window_width)
    win_labels = [f"W{i+1}\n{s}-{e}" for i, (s, e) in enumerate(win)]
    snap_steps = [s.step for s in result.snapshots]

    comp_sets = [oe.composition_type_ids(s) for s in result.snapshots]
    struct_sets = [oe.structure_type_ids(s, particles_species) for s in result.snapshots]

    d_steps, comp_cum = oe.discovery_curve(comp_sets, snap_steps)
    _, struct_cum = oe.discovery_curve(struct_sets, snap_steps)
    total_comp_ev = oe.total_composition_types_from_events(result.events)

    img_oe_discovery = plots.plot_discovery_curves(d_steps, comp_cum, struct_cum, total_comp_ev)
    img_oe_novelty = plots.plot_novelty_rate(
        win_labels,
        oe.novelty_rate(comp_sets, snap_steps, win),
        oe.novelty_rate(struct_sets, snap_steps, win),
    )
    img_oe_diversity = plots.plot_hill_diversity(
        d_steps, oe.hill_diversity(comp_sets), oe.hill_diversity(struct_sets))
    img_oe_turnover = plots.plot_turnover_grid(
        oe.window_turnover(comp_sets, snap_steps, win),
        oe.window_turnover(struct_sets, snap_steps, win),
        win_labels,
    )
    img_oe_size_facets = plots.plot_window_size_facets(
        oe.per_window_size_hist(result.per_step_metrics, win), win_labels)
```

Then add these keyword args to the `_HTML_TEMPLATE.format(...)` call (alongside
the existing `img_*` args):

```python
        n_windows=len(win),
        img_oe_discovery=img_oe_discovery,
        img_oe_novelty=img_oe_novelty,
        img_oe_diversity=img_oe_diversity,
        img_oe_turnover=img_oe_turnover,
        img_oe_size_facets=img_oe_size_facets,
```

- [ ] **Step 3: Confirm the template still has matching braces**

The Step-1 markup uses the verified `plot-grid` / `matrix-wrap` classes (checked
against report.py lines 209–262). Confirm no stray literal `{`/`}` were
introduced (they would break `str.format`):

Run: `grep -n 'plot-grid\|matrix-wrap\|img_oe_' halflife/analysis/report.py | head`
Expected: the new Tier 5 `plot-grid`, `matrix-wrap`, and five `{img_oe_*}`
placeholders are present.

- [ ] **Step 4: Smoke-render to confirm no KeyError/format error**

Run:
```bash
JAX_PLATFORMS=cpu .venv/bin/python -c "
import dataclasses
from halflife.config import SimConfig
from halflife.analysis.runner import run_diagnostic
from halflife.analysis.report import render_html
cfg = dataclasses.replace(SimConfig(), num_particles=80, num_species=3, max_composites=80, max_composite_size=8, max_fusions_per_step=30, emit_events=True)
r = run_diagnostic(cfg, n_steps=200, seed=0, sample_every=50)
html = render_html(r, windows=3)
assert 'Tier 5' in html and 'img_oe_discovery' not in html  # placeholder substituted
print('OK', len(html), 'bytes')
"
```
Expected: `OK <N> bytes` (no `KeyError`, no leftover `{img_oe_*}` placeholders).

- [ ] **Step 5: Commit**

```bash
git add halflife/analysis/report.py
git commit -m "feat(analysis): render Tier 5 open-endedness section in the report

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 8: CLI windowing flags

**Files:**
- Modify: `halflife/analysis/cli.py` (args ~line 109–133, render call ~line 179)

- [ ] **Step 1: Add the flags**

After the `--top-k` argument (line 117) add:

```python
    p.add_argument('--windows',      type=int, default=None,
                   help="Number of equal time windows for Tier 5 (default 5; "
                        "mutually exclusive with --window-width).")
    p.add_argument('--window-width', type=int, default=None,
                   help="Fixed window width in steps for Tier 5 (mutually "
                        "exclusive with --windows).")
```

- [ ] **Step 2: Validate mutual exclusion right after `args = p.parse_args(argv)` (line 133)**

```python
    if args.windows is not None and args.window_width is not None:
        raise SystemExit("--windows and --window-width are mutually exclusive")
```

- [ ] **Step 3: Thread into the render call (line 179)**

Replace:

```python
    html = render_html(result, top_k=args.top_k)
```

with:

```python
    html = render_html(result, top_k=args.top_k,
                       windows=args.windows, window_width=args.window_width)
```

- [ ] **Step 4: Verify the flags parse and re-window from cache**

Run (CPU, tiny, writes then re-renders from cache with a different windowing):
```bash
JAX_PLATFORMS=cpu .venv/bin/python -m halflife.analysis --scenario branching_world \
  --steps 200 --sample-every 50 --platform cpu --windows 4 \
  --out /tmp/oe_test.html && echo "--- re-window from cache ---" && \
JAX_PLATFORMS=cpu .venv/bin/python -m halflife.analysis --scenario branching_world \
  --steps 200 --sample-every 50 --platform cpu --from-cache --window-width 80 \
  --out /tmp/oe_test2.html && grep -c "Tier 5" /tmp/oe_test.html /tmp/oe_test2.html
```
Expected: both files report `1` Tier-5 occurrence; the second run prints
`loading cached run` (no re-simulation).

- [ ] **Step 5: Commit**

```bash
git add halflife/analysis/cli.py
git commit -m "feat(analysis): --windows / --window-width CLI flags for Tier 5

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 9: Pipeline test asserts Tier 5

**Files:**
- Modify: `tests/test_analysis_pipeline.py` (`test_full_pipeline_produces_html_with_all_sections`)

- [ ] **Step 1: Add Tier 5 assertions**

Find `test_full_pipeline_produces_html_with_all_sections` (it already builds
`html = render_html(result)` and asserts Tiers 1–4). Add after the existing
tier assertions:

```python
    # Tier 5 — open-endedness section present and fully substituted.
    assert 'Tier 5' in html
    assert 'Type discovery' in html or 'discovery' in html
    assert '{img_oe_' not in html        # no leftover format placeholders
```

- [ ] **Step 2: Run the pipeline test**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_analysis_pipeline.py -v`
Expected: PASS

- [ ] **Step 3: Run the full analysis suite to confirm no regressions**

Run:
```bash
JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_analysis_events.py \
  tests/test_analysis_metrics.py tests/test_analysis_transitions.py \
  tests/test_analysis_compatibility.py tests/test_analysis_pipeline.py \
  tests/test_analysis_openendedness.py -q
```
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_analysis_pipeline.py
git commit -m "test(analysis): assert Tier 5 markup in end-to-end report

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 10: 15k deliverable run + docs

**Files:**
- Modify: `CLAUDE.md` (analysis section: flags table + Tier 5 description)
- Output: `tests/reports/diag_current_experiment_<ts>.html`

- [ ] **Step 1: Check the GPU is free of the user's live sim**

Run: `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader`
If a live sim is running (high util / the user says so), append `--platform cpu`
to the next step instead of `gpu`, or ask the user to pause it. Do NOT contend.

- [ ] **Step 2: Run the 15k diagnostic**

Run:
```bash
.venv/bin/python -m halflife.analysis --scenario current_experiment \
  --steps 15000 --sample-every 100 --platform gpu --windows 5
```
Expected: prints run rate, caches the RunResult, writes
`tests/reports/diag_current_experiment_<ts>.html`. ~1–2 min on GPU.

- [ ] **Step 3: Read the windows back as a written comparison**

Extract the per-window numbers from the cached result (no re-sim) and summarize
for the user — discovery-curve slope across windows, novelty rate trend
(climbing/flat/decaying), diversity trend, and turnover magnitude:
```bash
JAX_PLATFORMS=cpu .venv/bin/python -c "
from halflife.analysis.runner import load_run_result
from halflife.analysis import openendedness as oe
from halflife.state import initialize_world
import numpy as np
r = load_run_result('tests/reports/cache/current_experiment_n15000_seed0_every100.pkl.gz')
sp = np.asarray(initialize_world(r.config, seed=r.seed).particles.species)
win = oe.slice_windows(r.n_steps, windows=5)
steps = [s.step for s in r.snapshots]
comp = [oe.composition_type_ids(s) for s in r.snapshots]
stru = [oe.structure_type_ids(s, sp) for s in r.snapshots]
print('windows:', win)
print('composition novelty/window:', oe.novelty_rate(comp, steps, win).tolist())
print('structure   novelty/window:', oe.novelty_rate(stru, steps, win).tolist())
print('comp cumulative:', oe.discovery_curve(comp, steps)[1][-1],
      '| struct cumulative:', oe.discovery_curve(stru, steps)[1][-1],
      '| total comp (events):', oe.total_composition_types_from_events(r.events))
"
```
Report the trends to the user in prose (this is the deliverable analysis).

- [ ] **Step 4: Update CLAUDE.md**

In the `## Composite Diagnostic Reports` section, (a) add the two flags to the
"Common flags" table:

```markdown
| `--windows N`         | 5 | Tier 5 time-window count (mutually exclusive with --window-width) |
| `--window-width W`    | none | Tier 5 fixed window width in steps |
```

and (b) add a short paragraph after the Tier 4 description:

```markdown
The **Tier 5** open-endedness section quantifies novelty accumulation over the
run on two type axes — composition (`species_hash`) and structure
(Weisfeiler-Lehman bond-graph hash). It shows the cumulative type-discovery
curve, per-window novelty rate, Hill-number diversity, window-to-window turnover
(Jaccard / Bray-Curtis), and per-window size facets. All of it is host-side
post-processing on the cached `RunResult`, so `--from-cache --windows N`
re-renders a different windowing instantly. Resolved at `sample_every` cadence;
structure metrics are only meaningful in `bond_mode="edges"` runs.
```

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude): document Tier 5 open-endedness section + windowing flags

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

(The generated HTML report under `tests/reports/` is a build artifact — do not
commit it unless the user asks.)

---

## Self-Review

**Spec coverage:**
- Composition + structure type ids → Task 2 ✓
- WL hash, no new dependency → Task 2 (`_wl_graph_hash`, blake2b) ✓
- `slice_windows` both flags, mutual exclusion → Task 1 + Task 8 ✓
- Discovery curve + headline event total → Task 3 ✓
- Novelty rate per window → Task 3 ✓
- Hill diversity (q0/q1/q2, composite-count abundance) → Task 4 ✓
- Window turnover (Jaccard + Bray-Curtis, NaN handling) → Task 5 ✓
- Per-window size facets reusing `size_histogram` → Task 5 ✓
- 5 plot builders → Task 6 ✓
- Tier 5 report section + `--from-cache` re-window → Task 7 + Task 8 ✓
- CLI flags → Task 8 ✓
- Tests (WL invariance, windowing edges, synthetic metrics) → Tasks 1–6 ✓
- 15k current_experiment deliverable + written comparison → Task 10 ✓
- Edgeless-multimember structure fallback → Task 2 ✓
- Error handling (empty snapshots → 0; mutual-exclusion error) → Tasks 1,4,8 ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases" — all steps carry real code or exact commands. Task 7's HTML markup was reconciled against the actual report.py template (verified `plot-grid` / `matrix-wrap` classes; no `<figure>`/`<figcaption>` in this codebase), so no guesses remain.

**Type consistency:** Metric functions consistently take `type_id_arrays: List[np.ndarray]` (per-composite, repeats) + `snapshot_steps: List[int]` + `windows: List[Tuple[int,int]]`. Plot builders' arg names match the `render_html` call in Task 7 (`plot_discovery_curves(steps, comp_cum, struct_cum, total_comp_events)`, `plot_turnover_grid(comp_turnover, struct_turnover, window_labels)`, etc.). `render_html(result, top_k, windows, window_width)` signature matches the CLI call in Task 8. `structure_type_ids(snapshot, particles_species)` matches its caller in Task 7.

No issues outstanding.
