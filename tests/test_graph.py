"""
tests/test_graph.py — unit tests for halflife/graph.py.

All graph functions operate on ONE composite's bond graph in local member-slot
space (nodes = slots [0, M), edges = (la, lb) local-slot pairs + validity
mask). Tiny fixed shapes, CPU-friendly, no WorldState needed.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax.numpy as jnp
import numpy as np

from halflife.graph import bfs_tree, subtree_sums, descendant_mask, reachable_mask

M = 8       # slots
E = 8       # edge rows
ITERS = 8   # >= any test graph's diameter


def _edges(pairs):
    """Build (la, lb, evalid) edge arrays from a list of local-slot pairs."""
    la = np.zeros(E, np.int32)
    lb = np.zeros(E, np.int32)
    ev = np.zeros(E, bool)
    for i, (a, b) in enumerate(pairs):
        la[i], lb[i], ev[i] = a, b, True
    return jnp.asarray(la), jnp.asarray(lb), jnp.asarray(ev)


def test_bfs_chain():
    """Chain 0-1-2-3: dist = hop count, parent = previous node, rest unreached."""
    la, lb, ev = _edges([(0, 1), (1, 2), (2, 3)])
    dist, parent = bfs_tree(la, lb, ev, M, ITERS)
    assert dist[:4].tolist() == [0, 1, 2, 3]
    assert parent[:4].tolist() == [-1, 0, 1, 2]
    INF = M + 1
    assert all(d == INF for d in dist[4:].tolist())
    assert all(p == -1 for p in parent[4:].tolist())


def test_bfs_cycle_parent_tiebreak_is_min_slot():
    """Square 0-1-2-3-0: slot 2 is reachable at dist 2 via 1 or 3 → parent 1."""
    la, lb, ev = _edges([(0, 1), (1, 2), (2, 3), (3, 0)])
    dist, parent = bfs_tree(la, lb, ev, M, ITERS)
    assert dist[:4].tolist() == [0, 1, 2, 1]
    assert int(parent[2]) == 1


def test_bfs_undirected_both_endpoint_orders():
    """Edge direction in the list must not matter: (1,0),(2,1) ≡ (0,1),(1,2)."""
    la, lb, ev = _edges([(1, 0), (2, 1)])
    dist, parent = bfs_tree(la, lb, ev, M, ITERS)
    assert dist[:3].tolist() == [0, 1, 2]
    assert parent[:3].tolist() == [-1, 0, 1]


def test_subtree_sums_chain():
    """Chain rooted at 0: subtree sums accumulate suffixes."""
    la, lb, ev = _edges([(0, 1), (1, 2), (2, 3)])
    _, parent = bfs_tree(la, lb, ev, M, ITERS)
    base_h = jnp.asarray([1, 10, 100, 1000, 0, 0, 0, 0], jnp.uint32)
    base_c = jnp.asarray([1, 1, 1, 1, 0, 0, 0, 0], jnp.int32)
    sub_h, sub_c = subtree_sums(parent, base_h, base_c, M, ITERS)
    assert sub_h[:4].tolist() == [1111, 1110, 1100, 1000]
    assert sub_c[:4].tolist() == [4, 3, 2, 1]


def test_subtree_sums_star():
    """Star rooted at 0 with leaves 1, 2, 3."""
    la, lb, ev = _edges([(0, 1), (0, 2), (0, 3)])
    _, parent = bfs_tree(la, lb, ev, M, ITERS)
    base_h = jnp.asarray([1, 10, 100, 1000, 0, 0, 0, 0], jnp.uint32)
    base_c = jnp.asarray([1, 1, 1, 1, 0, 0, 0, 0], jnp.int32)
    sub_h, sub_c = subtree_sums(parent, base_h, base_c, M, ITERS)
    assert int(sub_h[0]) == 1111
    assert sub_h[1:4].tolist() == [10, 100, 1000]
    assert sub_c[:4].tolist() == [4, 1, 1, 1]


def test_descendant_mask_chain():
    """Cutting chain 0-1-2-3 at slot 2 marks {2, 3}."""
    la, lb, ev = _edges([(0, 1), (1, 2), (2, 3)])
    _, parent = bfs_tree(la, lb, ev, M, ITERS)
    mask = descendant_mask(parent, jnp.int32(2), M, ITERS)
    assert mask[:4].tolist() == [False, False, True, True]


def test_reachable_mask_bridge_vs_cycle():
    """After removing a bridge, the far side is unreachable; a cycle stays whole."""
    # Chain 0-1-2 with edge (1,2) removed → from 1 reach {0,1} only.
    la, lb, ev = _edges([(0, 1)])
    reach = reachable_mask(la, lb, ev, jnp.int32(1), M, ITERS)
    assert reach[:3].tolist() == [True, True, False]
    # Triangle 0-1-2 with edge (0,1) removed → still connected via 2.
    la, lb, ev = _edges([(1, 2), (2, 0)])
    reach = reachable_mask(la, lb, ev, jnp.int32(0), M, ITERS)
    assert reach[:3].tolist() == [True, True, True]
