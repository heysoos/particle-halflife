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


# ── Bond-graph degree & topology (structure axis, snapshot-resolved) ─────────
#
# The WL structure hash above answers "how many distinct arrangements?" but hides
# *what kind* of arrangement. These two views surface the shape directly, and in
# particular separate the count-vs-mass story: many tiny chains can dominate the
# composite COUNT while a few large cyclic networks hold most of the particle MASS.

def _snapshot_degree_topology(snapshot, num_particles: int):
    """One snapshot → (deg_counts[4], topo_count[3], topo_mass[3]).

    deg_counts: histogram of bonded-particle degree, bucketed 1 / 2 / 3 / 4+
                (degree = number of incident edges; only particles with ≥1 bond).
    topo_count: composite count in each {chain, tree-branch, cyclic} class.
    topo_mass:  particles (Σ member_count) in each class.

    A composite (n ≥ 2) is classified:
      cyclic       if edge_count ≥ n          (a cycle exists: ≥ n edges on n nodes)
      tree-branch  elif max member degree ≥ 3 (a tree with a Y-junction)
      chain        else                       (a path / dimer, all degrees ≤ 2)
    """
    deg = np.zeros(num_particles, dtype=np.int64)
    topo_c = np.zeros(3, dtype=np.int64)
    topo_m = np.zeros(3, dtype=np.int64)
    for c in np.nonzero(snapshot.alive)[0]:
        n = int(snapshot.member_count[c])
        if n < 2:
            continue
        ne = int(snapshot.edge_count[c])
        ep = snapshot.edges[c, :ne]
        a = ep[:, 0]; b = ep[:, 1]
        a = a[a >= 0]; b = b[b >= 0]
        np.add.at(deg, a, 1)
        np.add.at(deg, b, 1)
        local = np.concatenate([a, b])
        maxdeg = int(np.bincount(local).max()) if local.size else 0
        if ne >= n:
            k = 2          # cyclic
        elif maxdeg >= 3:
            k = 1          # tree-branch
        else:
            k = 0          # chain
        topo_c[k] += 1
        topo_m[k] += n
    bonded = deg[deg >= 1]
    deg_counts = np.array([
        int(np.sum(bonded == 1)), int(np.sum(bonded == 2)),
        int(np.sum(bonded == 3)), int(np.sum(bonded >= 4)),
    ], dtype=np.int64)
    return deg_counts, topo_c, topo_m


def degree_topology_windowed(snapshots, snapshot_steps: List[int],
                             windows: List[Tuple[int, int]],
                             num_particles: int) -> Dict[str, np.ndarray]:
    """Per-window bonded-particle degree distribution and topology split.

    Returns dict of fractions (rows sum to 1; empty windows are all-zero):
      deg_frac:   (nW, 4) fraction of bonded particles at degree 1 / 2 / 3 / 4+
      topo_count: (nW, 3) fraction of composites that are chain/tree-branch/cyclic
      topo_mass:  (nW, 3) fraction of bonded particles living in each class
    """
    nW = len(windows)
    deg = np.zeros((nW, 4), dtype=np.int64)
    tc = np.zeros((nW, 3), dtype=np.int64)
    tm = np.zeros((nW, 3), dtype=np.int64)
    for s, step in zip(snapshots, snapshot_steps):
        wi = _window_index(step, windows)
        if wi is None:
            continue
        d, c, m = _snapshot_degree_topology(s, num_particles)
        deg[wi] += d; tc[wi] += c; tm[wi] += m

    def _frac(M):
        tot = M.sum(axis=1, keepdims=True)
        return np.divide(M.astype(float), tot, out=np.zeros(M.shape, dtype=float),
                         where=tot > 0)

    return {'deg_frac': _frac(deg), 'topo_count': _frac(tc), 'topo_mass': _frac(tm)}
