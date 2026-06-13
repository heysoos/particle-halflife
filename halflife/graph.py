"""
Bond-graph algorithms over composite-local edge lists.

Every function operates on ONE composite's bond graph expressed in local
member-slot space: nodes are slots [0, M), edges are (la, lb) pairs of local
slots with a validity mask. Callers translate global particle-id edges into
local slots (via a pid→slot scatter) and vmap these functions over a
compacted batch of composites.

JIT notes: fixed shapes everywhere; iteration via lax.fori_loop. A sweep
count of `iters` fully handles graphs whose BFS depth from slot 0 is at most
`iters`; deeper nodes stay "unreached" (dist = INF, parent = -1) and callers
must treat them as belonging to the root fragment. config.fission_label_iters
picks the cap — correctness needs iters >= graph diameter, cost is linear in
iters. This is the load-bearing perf knob flagged in
notes/2026-06-01-fission-bond-breaking-redesign.md.
"""

import jax
import jax.numpy as jnp


def bfs_tree(la: jnp.ndarray, lb: jnp.ndarray, evalid: jnp.ndarray,
             M: int, iters: int) -> tuple:
    """
    BFS from local slot 0 over an undirected edge list.

    Args:
        la, lb:  (E,) int32 — edge endpoints in local slot space
        evalid:  (E,) bool  — which edge rows are real
        M:       static int — number of slots
        iters:   static int — sweep count (>= graph diameter for full reach)

    Returns:
        dist:   (M,) int32 — hop distance from slot 0; INF (= M + 1) if unreached
        parent: (M,) int32 — BFS-tree parent slot; -1 for the root and unreached
    """
    INF = jnp.int32(M + 1)
    safe_a = jnp.where(evalid, la, 0)
    safe_b = jnp.where(evalid, lb, 0)
    drop_a = jnp.where(evalid, la, M)   # invalid rows → OOB index M, dropped
    drop_b = jnp.where(evalid, lb, M)

    dist0 = jnp.full(M, INF, dtype=jnp.int32).at[0].set(0)

    def sweep(_, dist):
        # Relax both directions of every edge (scatter-min keeps it parallel-safe).
        cand_b = jnp.where(evalid, dist[safe_a] + 1, INF)
        cand_a = jnp.where(evalid, dist[safe_b] + 1, INF)
        dist = dist.at[drop_b].min(cand_b, mode='drop')
        dist = dist.at[drop_a].min(cand_a, mode='drop')
        return dist

    dist = jax.lax.fori_loop(0, iters, sweep, dist0)

    # parent[child] = lowest-slot neighbor exactly one hop closer to the root.
    # min-scatter makes the tie-break deterministic (same graph → same tree).
    cand_for_b = jnp.where(evalid & (dist[safe_a] + 1 == dist[safe_b]), safe_a, INF)
    cand_for_a = jnp.where(evalid & (dist[safe_b] + 1 == dist[safe_a]), safe_b, INF)
    parent = jnp.full(M, INF, dtype=jnp.int32)
    parent = parent.at[drop_b].min(cand_for_b, mode='drop')
    parent = parent.at[drop_a].min(cand_for_a, mode='drop')
    parent = jnp.where((parent >= INF) | (dist >= INF), jnp.int32(-1), parent)
    return dist, parent


def subtree_sums(parent: jnp.ndarray, base_u32: jnp.ndarray,
                 base_i32: jnp.ndarray, M: int, iters: int) -> tuple:
    """
    For a rooted tree (parent pointers from bfs_tree), compute for every slot
    v the sum of base values over v's subtree (v included). Two channels:

      - uint32 (hash values): accumulates with mod-2^32 wraparound, matching
        the product-hash convention elsewhere (sum in uint32, THEN % modulus).
        This makes "complement hash = total − subtree" exact in uint32.
      - int32 (member counts).

    Fixed-point sweep: acc[v] = base[v] + Σ_{c: parent[c]==v} acc[c]
    converges once iters >= tree height; values are stable afterwards.
    Padding slots must carry base value 0 (callers mask before calling).

    Returns: (sub_u32, sub_i32) — both (M,)
    """
    drop_par = jnp.where(parent >= 0, parent, M)  # root/unreached → dropped

    def sweep(_, carry):
        acc_h, acc_c = carry
        nh = base_u32 + jnp.zeros(M, dtype=jnp.uint32).at[drop_par].add(acc_h, mode='drop')
        nc = base_i32 + jnp.zeros(M, dtype=jnp.int32).at[drop_par].add(acc_c, mode='drop')
        return nh, nc

    return jax.lax.fori_loop(0, iters, sweep, (base_u32, base_i32))


def descendant_mask(parent: jnp.ndarray, cut_slot: jnp.ndarray,
                    M: int, iters: int) -> jnp.ndarray:
    """
    Mark every slot in the subtree rooted at cut_slot (cut_slot included).
    This materializes the bipartition implied by cutting the tree edge
    (cut_slot, parent[cut_slot]). Unreached slots (parent == -1) are never
    marked — they stay with the root fragment.

    Returns: (M,) bool
    """
    safe_par = jnp.where(parent >= 0, parent, 0)
    has_par = parent >= 0
    seed = jnp.arange(M, dtype=jnp.int32) == cut_slot

    def sweep(_, mask):
        return seed | (has_par & mask[safe_par])

    return jax.lax.fori_loop(0, iters, sweep, seed)


def reachable_mask(la: jnp.ndarray, lb: jnp.ndarray, evalid: jnp.ndarray,
                   start_slot: jnp.ndarray, M: int, iters: int) -> jnp.ndarray:
    """
    Mark every slot reachable from start_slot over the (undirected) edge list.
    Bond scission uses this to test whether removing an edge disconnected the
    composite: pass the edge list WITHOUT the removed edge and start from one
    of its endpoints — if the other endpoint is still reached, the edge was
    part of a cycle and no split happens.

    Returns: (M,) bool
    """
    safe_a = jnp.where(evalid, la, 0)
    safe_b = jnp.where(evalid, lb, 0)
    reach0 = jnp.zeros(M, dtype=bool).at[start_slot].set(True)

    def sweep(_, reach):
        to_b = evalid & reach[safe_a]
        to_a = evalid & reach[safe_b]
        reach = reach.at[jnp.where(to_b, lb, M)].set(True, mode='drop')
        reach = reach.at[jnp.where(to_a, la, M)].set(True, mode='drop')
        return reach

    return jax.lax.fori_loop(0, iters, sweep, reach0)
