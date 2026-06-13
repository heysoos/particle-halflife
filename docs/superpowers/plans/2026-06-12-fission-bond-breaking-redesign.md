# Fission & Bond-Breaking Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace slot-order hash-partition fission with physical bond-cut fission (fixes the long-bond bug), add a per-bond chemical scission channel (kinetic + thermal breaking), and replace the BE→half-life formula with a liquid-drop fissility law.

**Architecture:** Three sequential phases on `halflife/chemistry.py` + `halflife/step.py`. Phase A extracts a shared binary-split applier (`_apply_binary_splits`) and rewrites `apply_composite_decay` to fracture along the best bond cut (BFS spanning tree + subtree hash sums + max-product-BE scoring, Q-value kick). Phase B adds `apply_bond_scission` reusing the same applier. Phase C computes internal hard-core repulsion PE in the force kernel and derives a live half-life from fissility `x = E_rep / (2·E_coh)`. New module `halflife/graph.py` holds the fixed-shape graph algorithms (BFS, subtree sums, descendant/reachable masks).

**Tech Stack:** JAX (jit/vmap/fori_loop, static shapes), pytest. Reference design: `notes/2026-06-01-fission-bond-breaking-redesign.md`.

**Design decisions locked here (resolving the note's open questions):**
- Endothermic whole-body fission: **forbidden by default** (`forbid_endothermic_fission: bool = True`, togglable; False → fires with zero kick).
- Thermal rate law: **Arrhenius** — `P = 1 − exp(−dt · ν₀ · exp(−(E_b − strain)/kT))`, knobs `bond_break_attempt_rate` (ν₀) and `bond_temperature` (kT). `kT=0` disables thermal cleanly. Only **stretch** strains a bond (compression never breaks).
- Component-labeling iteration cap: `fission_label_iters: int = 64` (covers graph diameter ≤ 64; deeper nodes degrade gracefully — they stay with the root fragment). Benchmarked in Phase A.
- Chemical channel scope: **at most one bond break per composite per step** (the most-overstretched breaking edge), budgeted by `max_scissions_per_step`. Binary splits only — reuses Phase A machinery; general multi-fragment labeling deferred.
- Fragment BE for size-1 products: hash BE of the singleton multiset (consistent energy landscape; `forbid_endothermic_fission` can therefore make some dimers half-life-stable — the chemical channel still breaks them kinetically).
- Morse potential: **not in scope** (note marks it optional; harmonic + dissociation threshold gives kinetic breaking already).

**Conventions for every task below (from CLAUDE.md):**
- Run Python via `.venv/bin/python` / `.venv/bin/pytest` from the project root (WSL-native, Pattern B — no `wsl bash -c` wrapper).
- Chemistry tests default to GPU; analysis suite runs with `JAX_PLATFORMS=cpu`. For parallel runs: `XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/pytest -n 4`.
- **Never `git add -A` or `git add .`** — name files explicitly.
- **Do not delete existing comments** unrelated to removed features. When moving code (e.g. into `_apply_binary_splits`), carry its comments along verbatim. Comments may only vanish with the exact feature they describe (`_hash_to_partition`, `_path_edges_from_members`, `fission_cost`).

---

## Task 0: Preliminaries — flush pending config change, baseline benchmark

`halflife/config.py` has an uncommitted user edit (`max_composite_size: 128 → 256`). Commit it separately first so later config commits stay clean.

**Files:**
- Modify: none (commit existing change)
- Create: `/tmp/bench_fission.py` (throwaway, NOT committed)

- [ ] **Step 1: Commit the user's pending config bump**

```bash
git add halflife/config.py && git commit -m "config: bump max_composite_size to 256 (current experiment)"
```

- [ ] **Step 2: Write the baseline benchmark script**

Create `/tmp/bench_fission.py` (diagnostic scripts live in /tmp per CLAUDE.md; needs the explicit sys.path insert):

```python
"""Throwaway: steps/sec at default config. Run before & after the fission rewrite."""
import sys, time
sys.path.insert(0, "/mnt/c/Users/Heysoos/Documents/Pycharm Projects/halflife-particle")

from halflife.utils import enable_persistent_compilation_cache
enable_persistent_compilation_cache()

from halflife.config import SimConfig
from halflife.state import initialize_world, initialize_interaction_params, initialize_physics_params
from halflife.step import make_run_n_steps

config = SimConfig()
state = initialize_world(config, seed=0)
params = initialize_interaction_params(config, seed=42)
physics = initialize_physics_params(config)
run_n = make_run_n_steps(config)

# Warm-up: compile + let chemistry reach steady state (composites exist → fission path hot)
state = run_n(state, params, physics, 500)
state.particles.position.block_until_ready()

t0 = time.perf_counter()
state = run_n(state, params, physics, 300)
state.particles.position.block_until_ready()
dt = time.perf_counter() - t0
print(f"300 steps in {dt:.2f}s  →  {300/dt:.1f} steps/s  ({dt/300*1000:.2f} ms/step)")
```

- [ ] **Step 3: Run it on GPU and record the number**

Run: `.venv/bin/python /tmp/bench_fission.py`
Expected: a steps/s figure (~30 steps/s per CLAUDE.md). **Write the number down** — it goes in the Task 4 commit message.

---

# Phase A — bond-cut fission (the bug fix)

## Task 1: `halflife/graph.py` — local-slot graph algorithms

**Files:**
- Create: `halflife/graph.py`
- Create: `tests/test_graph.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_graph.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_graph.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'halflife.graph'`

- [ ] **Step 3: Implement `halflife/graph.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_graph.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add halflife/graph.py tests/test_graph.py && git commit -m "feat(graph): local-slot bond-graph algorithms for bond-cut fission

BFS spanning tree, subtree sums (uint32 hash + int32 count channels),
descendant mask, reachability mask. Fixed-shape lax.fori_loop sweeps;
iteration cap = the fission_label_iters knob introduced next."
```

## Task 2: Failing tests for bond-cut fission

**Files:**
- Modify: `tests/test_chemistry.py` (replace `test_fission_rebuilds_spanning_tree_per_product` at line ~1012; add new tests after it)

- [ ] **Step 1: Replace the stale spanning-tree-rebuild test and add the new behavioral tests**

In `tests/test_chemistry.py`, **delete** the whole function `test_fission_rebuilds_spanning_tree_per_product` (lines ~1012–1060 — it asserts the old "fresh path through slot order" behavior that minted the long bonds) and put these in its place. Note the shared fixture builder first:

```python
def _make_single_composite_world(config, pos, species, edge_pairs,
                                  half_life=1e-4, binding_energy=0.0):
    """
    Build a world with one alive composite (slot 0) over the first
    len(edge_pairs)+1... — actually over every particle whose composite_id
    is 0 below. Members are particles [0, n); remaining particles are free
    and parked far away. edge_pairs are (pid, pid) bond tuples.
    """
    n = max(max(p) for p in edge_pairs) + 1
    N = config.num_particles
    world = initialize_world(config, seed=0)
    composite_id = np.full(N, -1, dtype=np.int32)
    composite_id[:n] = 0
    members = np.full((config.max_composites, config.max_composite_size), -1, dtype=np.int32)
    members[0, :n] = np.arange(n)
    edges = np.full((config.max_composites, config.e_max, 2), -1, dtype=np.int32)
    for k, (a, b) in enumerate(edge_pairs):
        edges[0, k] = (a, b)
    member_count = np.zeros(config.max_composites, dtype=np.int32)
    member_count[0] = n
    edge_count = np.zeros(config.max_composites, dtype=np.int32)
    edge_count[0] = len(edge_pairs)
    alive = np.zeros(config.max_composites, dtype=bool)
    alive[0] = True
    hl = np.zeros(config.max_composites, dtype=np.float32)
    hl[0] = half_life
    be = np.zeros(config.max_composites, dtype=np.float32)
    be[0] = binding_energy
    return world._replace(
        particles=world.particles._replace(
            position=jnp.asarray(pos), species=jnp.asarray(species),
            composite_id=jnp.asarray(composite_id),
        ),
        composites=world.composites._replace(
            members=jnp.asarray(members), member_count=jnp.asarray(member_count),
            alive=jnp.asarray(alive), edges=jnp.asarray(edges),
            edge_count=jnp.asarray(edge_count), half_life=jnp.asarray(hl),
            binding_energy=jnp.asarray(be),
        ),
    )


def test_fission_never_mints_new_bonds():
    """
    THE long-bond-bug regression test. Fission products may only keep edges
    that already existed in the parent — never invent new pairs. The member
    SLOT order is deliberately scrambled relative to the bond topology so the
    old slot-order path rebuild would mint non-edges.
    """
    config = SimConfig(num_species=3, num_particles=10, max_composites=4,
                       boundary_mode="reflect", world_width=20.0, world_height=20.0,
                       half_life_min=0.001, half_life_max=0.001)
    pos = np.array([[5.0, 5.0], [5.5, 5.0], [6.0, 5.0], [6.5, 5.0]]
                   + [[50.0 + i, 50.0] for i in range(6)], dtype=np.float32)
    species = np.zeros(10, dtype=np.int32)
    world = _make_single_composite_world(
        config, pos, species, [(0, 1), (1, 2), (2, 3)], half_life=1e-4)
    # Scramble slot order: members (0, 2, 1, 3) — bonds are still the chain.
    members = np.asarray(world.composites.members).copy()
    members[0, :4] = (0, 2, 1, 3)
    world = world._replace(composites=world.composites._replace(
        members=jnp.asarray(members)))
    physics = initialize_physics_params(config)

    original_edges = {(0, 1), (1, 2), (2, 3)}
    state = world
    for _ in range(20):  # decay fires probabilistically; 20 rolls at hl=1e-4 ≈ certain
        state = apply_composite_decay(state, config, physics)

    edges_after = np.asarray(state.composites.edges)
    counts_after = np.asarray(state.composites.edge_count)
    alive_after = np.asarray(state.composites.alive)
    for c in np.where(alive_after)[0]:
        for e in range(counts_after[c]):
            pair = tuple(sorted(edges_after[c, e].tolist()))
            assert pair in original_edges, \
                f"Fission minted new bond {pair} not in {original_edges}"


def test_fission_products_keep_internal_edges_and_stay_consistent():
    """After a 4-chain fissions, every alive product's edges reference only
    its own members, and edge_count >= n - 1 (connected fragments)."""
    config = SimConfig(num_species=3, num_particles=10, max_composites=4,
                       boundary_mode="reflect", world_width=20.0, world_height=20.0,
                       half_life_min=0.001, half_life_max=0.001)
    pos = np.array([[5.0, 5.0], [5.5, 5.0], [6.0, 5.0], [6.5, 5.0]]
                   + [[50.0 + i, 50.0] for i in range(6)], dtype=np.float32)
    species = np.zeros(10, dtype=np.int32)
    world = _make_single_composite_world(
        config, pos, species, [(0, 1), (1, 2), (2, 3)], half_life=1e-4)
    physics = initialize_physics_params(config)

    state = apply_composite_decay(world, config, physics)

    alive_after = np.asarray(state.composites.alive)
    members_after = np.asarray(state.composites.members)
    counts_after = np.asarray(state.composites.member_count)
    edges_after = np.asarray(state.composites.edges)
    ecounts_after = np.asarray(state.composites.edge_count)
    cid_after = np.asarray(state.particles.composite_id)
    fired = not alive_after[0] or counts_after[0] < 4
    if fired:  # one decay roll is probabilistic; only assert when it fired
        for c in np.where(alive_after)[0]:
            mem = set(members_after[c, :counts_after[c]].tolist())
            assert ecounts_after[c] >= counts_after[c] - 1
            for e in range(ecounts_after[c]):
                a, b = edges_after[c, e].tolist()
                assert a in mem and b in mem, \
                    f"Product {c} edge ({a},{b}) references non-members of {mem}"
            for pid in mem:
                assert cid_after[pid] == c


def test_endothermic_fission_suppressed_by_default():
    """With parent BE far above any possible product-BE sum, every cut has
    Q < 0 → forbid_endothermic_fission (default True) keeps it alive."""
    config = SimConfig(num_species=3, num_particles=10, max_composites=4,
                       boundary_mode="reflect", world_width=20.0, world_height=20.0,
                       half_life_min=0.001, half_life_max=0.001)
    pos = np.array([[5.0, 5.0], [5.5, 5.0], [6.0, 5.0], [6.5, 5.0]]
                   + [[50.0 + i, 50.0] for i in range(6)], dtype=np.float32)
    species = np.zeros(10, dtype=np.int32)
    # binding_energy=10: product BEs are <= 1.0 each (binding_energy_scale=1)
    # so Q = BE0 + BE1 - 10 < 0 for every cut.
    world = _make_single_composite_world(
        config, pos, species, [(0, 1), (1, 2), (2, 3)],
        half_life=1e-4, binding_energy=10.0)
    physics = initialize_physics_params(config)

    state = world
    for _ in range(20):
        state = apply_composite_decay(state, config, physics)
    assert bool(state.composites.alive[0]), "endothermic fission should be barred"
    assert int(state.composites.member_count[0]) == 4
    # Same setup with the barrier off → it does fission (kick-less).
    config_open = SimConfig(num_species=3, num_particles=10, max_composites=4,
                            boundary_mode="reflect", world_width=20.0, world_height=20.0,
                            half_life_min=0.001, half_life_max=0.001,
                            forbid_endothermic_fission=False)
    state = world
    for _ in range(20):
        state = apply_composite_decay(state, config_open, physics)
    assert int(state.composites.member_count[0]) < 4 or not bool(state.composites.alive[0])


def test_exothermic_fission_kicks_products_apart():
    """Parent BE 0 → Q = BE0 + BE1 >= 0; if any positive-Q cut fires, member
    velocities change (COM-axis kick)."""
    config = SimConfig(num_species=3, num_particles=10, max_composites=4,
                       boundary_mode="reflect", world_width=20.0, world_height=20.0,
                       half_life_min=0.001, half_life_max=0.001)
    pos = np.array([[5.0, 5.0], [5.5, 5.0], [6.0, 5.0], [6.5, 5.0]]
                   + [[50.0 + i, 50.0] for i in range(6)], dtype=np.float32)
    species = np.arange(10, dtype=np.int32) % 3   # mixed species → nonzero frag BEs
    world = _make_single_composite_world(
        config, pos, species, [(0, 1), (1, 2), (2, 3)],
        half_life=1e-4, binding_energy=0.0)
    world = world._replace(particles=world.particles._replace(
        velocity=jnp.zeros((10, 2), dtype=jnp.float32)))
    physics = initialize_physics_params(config)

    state = world
    for _ in range(20):
        state = apply_composite_decay(state, config, physics)
    fired = int(state.composites.member_count[0]) < 4 or not bool(state.composites.alive[0])
    assert fired, "composite with hl=1e-4 should have fissioned within 20 rolls"
    speeds = np.linalg.norm(np.asarray(state.particles.velocity[:4]), axis=1)
    assert speeds.max() > 0.0, "exothermic fission should impart a kick"
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_chemistry.py -k "mints or keep_internal or endothermic or kicks_products" -v`
Expected: FAIL — `test_endothermic_fission_suppressed_by_default` errors with `TypeError: ... unexpected keyword argument 'forbid_endothermic_fission'`; `test_fission_never_mints_new_bonds` fails its `pair in original_edges` assertion (old code mints path edges).

(No commit yet — tests + implementation commit together in Task 3.)

## Task 3: Rewrite `apply_composite_decay` — bond-cut fracture + shared splitter

**Files:**
- Modify: `halflife/config.py` (add 2 fields, remove `fission_cost`)
- Modify: `halflife/chemistry.py` (replace lines ~345–829: `_hash_to_partition`, `apply_composite_decay`; delete `_product_free_bonds`; add `_hl_from_be_and_size`, `_apply_binary_splits`)
- Modify: `tests/test_hash.py` (delete the 4 `_hash_to_partition` tests, lines ~158–215)
- Modify: `tests/test_chemistry.py` (drop `fission_cost=0.0` kwarg at line ~1020 — that line is inside the test deleted in Task 2; verify no other use remains)

- [ ] **Step 1: Config — add the new knobs, remove `fission_cost`**

In `halflife/config.py`, **delete** these two lines (the feature they describe is replaced by the Q-value kick):

```python
    # Cost multiplier for fission (energy required = binding_energy * fission_cost)
    fission_cost: float = 0.5
```

and add, after the `composite_size_decay_scale` block:

```python
    # ── Fission fracture (bond-cut, 2026-06-12) ──────────────────────────────
    # Fission no longer partitions members by hashing slot indices — it
    # fractures along the bond cut that maximizes total product binding
    # energy (the hash-BE landscape acting as the shell-structure analog).
    # Products keep the parent edges internal to them; crossing edges break.
    # The kick is the Q-value max(BE(p0) + BE(p1) − BE(parent), 0), replacing
    # the old binding_energy * (1 − fission_cost) release.
    #
    # Iteration cap for the graph sweeps (BFS / subtree sums / fragment
    # labeling) inside fission and bond scission. Correct for bond graphs of
    # diameter <= this value; members beyond the horizon stay with the root
    # fragment (graceful degradation for extreme chains). Cost is linear.
    fission_label_iters: int = 64
    # Barrier analog: when True, a decay roll whose best cut has Q < 0 is
    # suppressed entirely — hash-favored ("magic") composites become stable
    # against spontaneous fission and only break kinetically/thermally via
    # bond scission. When False, endothermic fission fires with zero kick.
    forbid_endothermic_fission: bool = True
```

- [ ] **Step 2: chemistry.py — delete `_hash_to_partition` and `_product_free_bonds`**

Delete the function `_hash_to_partition` (lines ~345–401) and the function `_product_free_bonds` (lines ~327–342) including their docstrings (both describe the replaced slot-order partition machinery). Add the import at the top of the file, after the other halflife imports:

```python
from halflife.graph import bfs_tree, subtree_sums, descendant_mask, reachable_mask
```

(`reachable_mask` is used by Phase B; importing now avoids touching the line twice.)

- [ ] **Step 3: chemistry.py — add `_hl_from_be_and_size` (module level, above `apply_composite_decay`)**

```python
def _hl_from_be_and_size(be: jnp.ndarray, n: jnp.ndarray,
                         config: SimConfig, physics: PhysicsParams) -> jnp.ndarray:
    """
    Half-life from BE + size penalty. Same formula as fusion_scan_body /
    _fusion_apply_matching use inline; promoted to module level so the
    binary-split applier can reuse it for fission products.
    """
    t = jnp.clip((be - physics.fusion_threshold) / (1.0 - physics.fusion_threshold + 1e-8), 0.0, 1.0)
    hl_base = config.half_life_min + (config.half_life_max - config.half_life_min) * t
    size_penalty = 1.0 + config.composite_size_decay_scale * jnp.maximum(
        0.0, n.astype(jnp.float32) - 2.0
    )
    return hl_base / size_penalty
```

- [ ] **Step 4: chemistry.py — replace `apply_composite_decay` (and its per-fission internals) with the bond-cut version + `_apply_binary_splits`**

Replace everything from the `# ── Composite Decay / Fission ──...` section header down to (and including) the current `return new_state` at the end of `apply_composite_decay` (line ~829) with the code below. Comments from the old body that still apply (scatter `mode='drop'` rationale, free-slot collision note, fission-budget note, event-emission contract) are carried over verbatim inside.

```python
# ── Binary Split Applier (shared by fission and bond scission) ────────────────

def _apply_binary_splits(particles, composites, split_slots, fires, assignment,
                         kick_energy, config: SimConfig, physics: PhysicsParams):
    """
    Apply a batch of binary composite splits — the shared back half of
    half-life fission (apply_composite_decay) and chemical bond scission
    (apply_bond_scission, Phase B).

    For each batch row k with fires[k] True, the composite in slot
    split_slots[k] divides into product 0 (assignment[k] == 0) and product 1
    (assignment[k] == 1):

      - Product 0 reuses the parent slot; product 1 claims a fresh free slot.
      - Each product keeps exactly the parent edges internal to its member
        set — crossing edges break, and NO edges are ever minted. Bond
        lengths can therefore only come from bonds that already existed:
        this is the long-bond-bug fix.
      - Products of size 1 become free particles. Under valence, a product
        whose edge-based free bonds (Σ v_s − 2·edge_count) go negative
        shatters into free particles. Splits only remove edges, so degrees
        never grow and this is unreachable from a valid parent — kept as a
        cheap invariant guard.
      - kick_energy[k] splits equally between the two products as a kick
        along the product COM-COM axis (same per-member application as the
        old fission code: each product moves as a unit).

    Args:
        particles, composites: pre-split state pieces (authoritative)
        split_slots: (K,) int32 — parent composite slot per row; C = padding
        fires:       (K,) bool  — whether row k actually splits
        assignment:  (K, M) int32 ∈ {-1, 0, 1} — fragment label per member slot
        kick_energy: (K,) float32 — total kinetic energy released by row k
        config, physics: statics / runtime scalars

    Returns:
        (new_particles, new_composites, events) — events has leading dim K
        with kind == KIND_FISSION where fires. Callers discard events when
        config.emit_events is False; XLA DCEs the build code.
    """
    N = config.num_particles
    M = config.max_composite_size
    C = config.max_composites
    E_max = config.e_max
    K = split_slots.shape[0]
    m_idx = jnp.arange(M, dtype=jnp.int32)

    safe_slots = jnp.minimum(split_slots, C - 1)

    # Pre-allocate fresh composite slots for product 1: the k-th split takes
    # free_slots[k]. No collision with parents — parents are alive,
    # find_free_slots only returns dead slots.
    target_p1 = find_free_slots(composites.alive, K)  # (K,) int32, -1 = exhausted

    # ── Fragment lookup by particle id ───────────────────────────────────────
    # Batch rows are member-disjoint (a particle lives in one composite), so a
    # single (N,) array serves the whole batch.
    member_grid = composites.members[safe_slots]                       # (K, M)
    count_grid = composites.member_count[safe_slots]                   # (K,)
    valid_grid = (member_grid >= 0) & (m_idx[None, :] < count_grid[:, None]) \
                 & fires[:, None]
    flat_pids = jnp.where(valid_grid, member_grid, N).reshape(-1)
    frag_of_pid = jnp.full(N, -1, dtype=jnp.int32).at[flat_pids].set(
        assignment.reshape(-1), mode='drop')

    # ── Per-split: fragment COMs (min-image from member 0) ───────────────────
    def per_split(k):
        c = safe_slots[k]
        n = composites.member_count[c]
        member_ids = composites.members[c]
        safe_ids = jnp.where(member_ids >= 0, member_ids, 0)
        valid = (member_ids >= 0) & (m_idx < n)
        ref = particles.position[safe_ids[0]]

        def disp_from_ref(idx):
            d = particles.position[safe_ids[idx]] - ref
            if config.boundary_mode == "periodic":
                d = d - config.world_width  * jnp.round(d[0] / config.world_width)  * jnp.array([1., 0.])
                d = d - config.world_height * jnp.round(d[1] / config.world_height) * jnp.array([0., 1.])
            return d

        rels = jax.vmap(disp_from_ref)(jnp.arange(M))  # (M, 2)
        a = assignment[k]
        in_p0 = valid & (a == 0)
        in_p1 = valid & (a == 1)
        n0 = jnp.sum(in_p0.astype(jnp.float32))
        n1 = jnp.sum(in_p1.astype(jnp.float32))
        com0 = ref + jnp.sum(rels * in_p0[:, None].astype(jnp.float32), axis=0) / (n0 + 1e-8)
        com1 = ref + jnp.sum(rels * in_p1[:, None].astype(jnp.float32), axis=0) / (n1 + 1e-8)
        return com0, com1, n0.astype(jnp.int32), n1.astype(jnp.int32)

    all_com0, all_com1, all_n0, all_n1 = jax.vmap(per_split)(
        jnp.arange(K, dtype=jnp.int32))

    # ── Compact each product's members & compute its hash ────────────────────
    def per_product(k):
        c = safe_slots[k]
        member_ids = composites.members[c]
        n = composites.member_count[c]
        a = assignment[k]

        in_p0 = (a == 0) & (member_ids >= 0) & (m_idx < n)
        in_p1 = (a == 1) & (member_ids >= 0) & (m_idx < n)

        # Compact members of each product to front using cumsum (same trick
        # as fusion). Invalid entries route to OOB index M and drop.
        def compact(mask):
            pos = jnp.cumsum(mask.astype(jnp.int32)) - 1
            out = jnp.where(mask, pos, M)
            mem = jnp.full(M, -1, dtype=jnp.int32).at[out].set(member_ids, mode='drop')
            return mem, jnp.sum(mask.astype(jnp.int32))

        members_p0, count_p0 = compact(in_p0)
        members_p1, count_p1 = compact(in_p1)

        # Species hashes via commutative sum over each product's members.
        def hash_for_product(members_arr, count_arr):
            safe = jnp.where(members_arr >= 0, members_arr, 0)
            sp = particles.species[safe]
            valid_m = (members_arr >= 0) & (m_idx < count_arr)
            hvals = jax.vmap(lambda s: _entity_hash_val(s, config))(sp)
            return (jnp.sum(jnp.where(valid_m, hvals, 0)) % config.hash_modulus).astype(jnp.uint32)

        return (members_p0, count_p0, hash_for_product(members_p0, count_p0),
                members_p1, count_p1, hash_for_product(members_p1, count_p1))

    p0_members, p0_count, p0_hash, p1_members, p1_count, p1_hash = jax.vmap(per_product)(
        jnp.arange(K, dtype=jnp.int32))

    # ── Per-product edges: keep parent edges internal to each fragment ───────
    def split_edges(k):
        c = safe_slots[k]
        ga = composites.edges[c, :, 0]
        gb = composites.edges[c, :, 1]
        evalid = (jnp.arange(E_max) < composites.edge_count[c]) & (ga >= 0)
        fa = frag_of_pid[jnp.where(ga >= 0, ga, 0)]
        fb = frag_of_pid[jnp.where(gb >= 0, gb, 0)]

        def compact(keep):
            pos = jnp.cumsum(keep.astype(jnp.int32)) - 1
            out = jnp.where(keep, pos, E_max)
            e = jnp.full((E_max, 2), -1, dtype=jnp.int32).at[out].set(
                composites.edges[c], mode='drop')
            return e, jnp.sum(keep.astype(jnp.int32))

        e0, n_e0 = compact(evalid & (fa == 0) & (fb == 0))
        e1, n_e1 = compact(evalid & (fa == 1) & (fb == 1))
        return e0, n_e0, e1, n_e1

    p0_edges, p0_edge_count_all, p1_edges, p1_edge_count_all = jax.vmap(split_edges)(
        jnp.arange(K, dtype=jnp.int32))

    # ── Per-product free bonds (edge-based) and structural validity ──────────
    species_valences_split = _species_valences(config)

    def product_free_bonds(members_arr, count_arr, e_cnt):
        safe = jnp.where(members_arr >= 0, members_arr, 0)
        vs = species_valences_split[particles.species[safe]]
        valid_m = (members_arr >= 0) & (m_idx < count_arr)
        return jnp.sum(jnp.where(valid_m, vs, 0)) - jnp.int32(2) * e_cnt

    p0_free_bonds = jax.vmap(product_free_bonds)(p0_members, p0_count, p0_edge_count_all)
    p1_free_bonds = jax.vmap(product_free_bonds)(p1_members, p1_count, p1_edge_count_all)

    if config.use_valence:
        p0_valid = p0_free_bonds >= 0
        p1_valid = p1_free_bonds >= 0
    else:
        p0_valid = jnp.ones(K, dtype=bool)
        p1_valid = jnp.ones(K, dtype=bool)

    # ── Update each member particle's composite_id and velocity ──────────────
    def per_member(k, m):
        c = safe_slots[k]
        n = composites.member_count[c]
        member_id = composites.members[c, m]
        valid = fires[k] & (m < n) & (member_id >= 0)

        a = assignment[k, m]
        com0 = all_com0[k]
        com1 = all_com1[k]
        n0 = all_n0[k]
        n1 = all_n1[k]

        # Direction along COM-COM axis (min-image).
        d = com0 - com1
        if config.boundary_mode == "periodic":
            d = d - config.world_width  * jnp.round(d[0] / config.world_width)  * jnp.array([1., 0.])
            d = d - config.world_height * jnp.round(d[1] / config.world_height) * jnp.array([0., 1.])
        d_hat = d / (jnp.linalg.norm(d) + 1e-8)

        # Energy split: half of the released kick energy to each product.
        e_per = kick_energy[k] * 0.5
        v0 = jnp.sqrt(jnp.maximum(0.0, 2.0 * e_per / (n0.astype(jnp.float32) + 1e-8)))
        v1 = jnp.sqrt(jnp.maximum(0.0, 2.0 * e_per / (n1.astype(jnp.float32) + 1e-8)))

        # Kick: product 0 → +d_hat * v0, product 1 → -d_hat * v1.
        # Note: the kick always fires (even for shattered products), because
        # the energy release happens regardless of whether the pieces then
        # bind into a sub-composite or fly apart as free particles.
        kick = jnp.where(
            a == 0,
            d_hat * v0,
            jnp.where(a == 1, -d_hat * v1, jnp.zeros(2)),
        )

        forms_p0 = (n0 >= 2) & p0_valid[k]
        forms_p1 = (n1 >= 2) & p1_valid[k]

        # New composite_id:
        #   a==0 and forms_p0 → c (reuse parent slot)
        #   a==0 and not forms_p0 → -1 (free particle — size-1 or unsound)
        #   a==1 and forms_p1 → target_p1[k]
        #   a==1 and not forms_p1 → -1 (free)
        #   a==-1 (padding) → preserve original (the scatter is gated on
        #                     `valid`, so this value is never written).
        safe_member = jnp.where(member_id >= 0, member_id, 0)
        new_cid = jnp.where(
            a == 0,
            jnp.where(forms_p0, c, jnp.int32(-1)),
            jnp.where(a == 1,
                      jnp.where(forms_p1, target_p1[k], jnp.int32(-1)),
                      particles.composite_id[safe_member]),
        )
        return member_id, valid, new_cid, kick

    pid_grid, valid_grid_m, cid_grid, kick_grid = jax.vmap(
        lambda k: jax.vmap(lambda m: per_member(k, m))(jnp.arange(M, dtype=jnp.int32))
    )(jnp.arange(K, dtype=jnp.int32))

    flat_pid   = pid_grid.reshape(-1)
    flat_valid = valid_grid_m.reshape(-1)
    flat_cid   = cid_grid.reshape(-1)
    flat_kick  = kick_grid.reshape(-1, 2)

    # Route invalid entries to OOB index N (dropped). Without mode='drop',
    # JAX scatters with duplicate indices have indeterminate behavior, so
    # M-1 invalid slots writing the read-back value to index 0 would race
    # against any real write to particle 0.
    drop_pids = jnp.where(flat_valid, flat_pid, N)
    new_composite_id = particles.composite_id.at[drop_pids].set(flat_cid, mode='drop')

    # Velocity adds — duplicates accumulate, invalid entries add 0, so safe form is fine.
    safe_pids = jnp.where(flat_valid, flat_pid, 0)
    new_velocity = particles.velocity.at[safe_pids].add(
        jnp.where(flat_valid[:, None], flat_kick, 0.0)
    )

    # ── Write product 0 into the parent slot (in place) ──
    p0_alive = fires & (p0_count >= 2) & p0_valid

    p0_be_all = jax.vmap(lambda h: _hash_to_binding_energy(h, config, physics))(p0_hash)
    p0_hl_all = jax.vmap(lambda be, n: _hl_from_be_and_size(be, n, config, physics))(
        p0_be_all, p0_count)

    # Scatter product-0 results into the parent slots (padding rows route to
    # OOB index C and drop).
    split_drop = jnp.where(fires, split_slots, C)
    new_alive = composites.alive.at[split_drop].set(p0_alive, mode='drop')
    new_members = composites.members.at[split_drop].set(p0_members, mode='drop')
    new_member_count = composites.member_count.at[split_drop].set(p0_count, mode='drop')
    new_species_hash = composites.species_hash.at[split_drop].set(p0_hash, mode='drop')
    new_binding_energy = composites.binding_energy.at[split_drop].set(p0_be_all, mode='drop')
    new_half_life = composites.half_life.at[split_drop].set(p0_hl_all, mode='drop')
    new_free_bonds = composites.free_bonds.at[split_drop].set(p0_free_bonds, mode='drop')
    # Reset age on the parent slot (it's now a fresh product).
    new_age = composites.age.at[split_drop].set(jnp.float32(0.0), mode='drop')
    new_edges = composites.edges.at[split_drop].set(p0_edges, mode='drop')
    new_edge_count = composites.edge_count.at[split_drop].set(p0_edge_count_all, mode='drop')

    # ── Write product 1 into target_p1[k] when it forms a composite ──
    p1_writes = fires & (p1_count >= 2) & p1_valid

    p1_be_all = jax.vmap(lambda h: _hash_to_binding_energy(h, config, physics))(p1_hash)
    p1_hl_all = jax.vmap(lambda be, n: _hl_from_be_and_size(be, n, config, physics))(
        p1_be_all, p1_count)

    # Guard against negative indices: find_free_slots returns -1 when there
    # aren't enough free slots, and JAX's negative-index default would wrap
    # to [C-1] — clobbering the last composite. Route those to C (OOB) so
    # mode='drop' actually drops them.
    drop_targets = jnp.where(
        p1_writes & (target_p1 >= 0),
        target_p1,
        C,  # OOB → drop
    )

    new_alive          = new_alive.at[drop_targets].set(p1_writes, mode='drop')
    new_members        = new_members.at[drop_targets].set(p1_members, mode='drop')
    new_member_count   = new_member_count.at[drop_targets].set(p1_count, mode='drop')
    new_species_hash   = new_species_hash.at[drop_targets].set(p1_hash, mode='drop')
    new_binding_energy = new_binding_energy.at[drop_targets].set(p1_be_all, mode='drop')
    new_half_life      = new_half_life.at[drop_targets].set(p1_hl_all, mode='drop')
    new_free_bonds     = new_free_bonds.at[drop_targets].set(p1_free_bonds, mode='drop')
    new_age            = new_age.at[drop_targets].set(jnp.float32(0.0), mode='drop')
    new_edges          = new_edges.at[drop_targets].set(p1_edges, mode='drop')
    new_edge_count     = new_edge_count.at[drop_targets].set(p1_edge_count_all, mode='drop')

    new_composites = composites._replace(
        members=new_members,
        member_count=new_member_count,
        alive=new_alive,
        binding_energy=new_binding_energy,
        half_life=new_half_life,
        age=new_age,
        species_hash=new_species_hash,
        free_bonds=new_free_bonds,
        edges=new_edges,
        edge_count=new_edge_count,
    )

    new_particles = particles._replace(
        composite_id=new_composite_id,
        velocity=new_velocity,
    )

    # ── Per-split event emission (kind=2 fission; callers discard when off) ──
    # source = the parent composite BEFORE the state update; products may be
    # size 1 (shattered free particle) or — for a cycle-edge scission with no
    # actual split — product 1 may be empty (size 0, hash 0).
    ev_kind = jnp.where(fires, jnp.int32(KIND_FISSION), jnp.int32(KIND_NONE))
    ev_src_slots = jnp.stack([
        jnp.where(fires, split_slots, jnp.int32(-1)),
        jnp.full((K,), -1, dtype=jnp.int32),
    ], axis=1)
    ev_src_hashes = jnp.stack([
        jnp.where(fires, composites.species_hash[safe_slots], jnp.uint32(0)),
        jnp.zeros((K,), dtype=jnp.uint32),
    ], axis=1)
    ev_src_sizes = jnp.stack([
        jnp.where(fires, composites.member_count[safe_slots], jnp.int32(0)),
        jnp.zeros((K,), dtype=jnp.int32),
    ], axis=1)
    ev_prod_slots = jnp.stack([
        jnp.where(fires, split_slots, jnp.int32(-1)),
        jnp.where(fires, target_p1, jnp.int32(-1)),
    ], axis=1)
    ev_prod_hashes = jnp.stack([
        jnp.where(fires, p0_hash, jnp.uint32(0)),
        jnp.where(fires & (p1_count > 0), p1_hash, jnp.uint32(0)),
    ], axis=1)
    ev_prod_sizes = jnp.stack([
        jnp.where(fires, p0_count, jnp.int32(0)),
        jnp.where(fires, p1_count, jnp.int32(0)),
    ], axis=1)
    events = ReactionEvent(
        kind=ev_kind,
        source_slots=ev_src_slots,
        source_hashes=ev_src_hashes,
        source_sizes=ev_src_sizes,
        product_slots=ev_prod_slots,
        product_hashes=ev_prod_hashes,
        product_sizes=ev_prod_sizes,
    )

    return new_particles, new_composites, events


# ── Composite Decay / Fission ─────────────────────────────────────────────────

def apply_composite_decay(state: WorldState, config: SimConfig,
                           physics: PhysicsParams):
    """
    Half-life ("nuclear") fission with bond-cut fracture (2026-06-12).

    A decaying composite no longer partitions by hashing slot indices — it
    fractures along a BOND CUT: among the edges of its BFS spanning tree, the
    cut that maximizes total product binding energy (the hash-BE landscape
    acting as the shell-structure / magic-number analog) defines the two
    fragments. The additive commutative hash lets every cut be scored from
    one subtree-sum pass: frag hashes are (subtree, total − subtree). Each
    fragment keeps the parent edges internal to it, so fission never mints
    new bonds — slot order is irrelevant and the long-bond bug is gone.

    Energy: the kick is the Q-value max(BE(p0) + BE(p1) − BE(parent), 0),
    split equally between products along the COM-COM axis (replaces the old
    binding_energy * (1 − fission_cost) release). With
    config.forbid_endothermic_fission (default True), a roll whose best cut
    has Q < 0 is suppressed — the composite survives (fission barrier).

    Particle species are never modified — only composite_id and velocity.

    Perf: per-fission work runs over a compacted batch of at most
    config.max_fissions_per_step fissioning composites, not all C slots.
    Fissions beyond the budget defer to the next step (the composite stays
    alive and re-rolls). Graph sweeps are capped at
    config.fission_label_iters. When emit_events is on, the fission
    ReactionEvent batch has leading dim min(max_fissions_per_step, C).

    Args:
        state:   WorldState
        config:  SimConfig (static)
        physics: PhysicsParams — provides dt for the per-step decay probability

    Returns:
        Updated WorldState (and the ReactionEvent batch when emit_events)
    """
    particles = state.particles
    composites = state.composites
    key, subkey = jax.random.split(state.rng_key)
    N = config.num_particles
    M = config.max_composite_size
    C = config.max_composites
    E_max = config.e_max
    iters = config.fission_label_iters
    m_idx = jnp.arange(M, dtype=jnp.int32)

    # ── Roll for which composites decay this step ───────────────────────────
    rand = jax.random.uniform(subkey, (C,))
    ln2 = jnp.log(2.0)
    decay_prob = 1.0 - jnp.exp(-physics.dt * ln2 / (composites.half_life + 1e-10))
    fissions = composites.alive & (rand < decay_prob)  # (C,) bool

    # ── Compact fissioning composites to a fixed batch (perf, 2026-06-12) ───
    # Gather the fissioning slots into a (K_f,) batch and run the heavy math
    # only there. Fissions beyond the budget are deferred: the composite
    # stays alive, unchanged, and simply re-rolls its decay next step.
    K_f = min(config.max_fissions_per_step, C)
    fission_rank = jnp.cumsum(fissions.astype(jnp.int32)) - 1
    selected = fissions & (fission_rank < K_f)
    cand = jnp.where(selected, jnp.arange(C, dtype=jnp.int32), C)
    fiss_idx = jnp.sort(cand)[:K_f]      # (K_f,) fissioning slot ids, C = padding
    fiss_valid = fiss_idx < C            # (K_f,)
    safe_fiss = jnp.minimum(fiss_idx, C - 1)

    # ── pid → local member slot, for the whole batch ─────────────────────────
    # Batch composites are member-disjoint, so one (N,) array serves all K_f.
    member_grid = composites.members[safe_fiss]                          # (K_f, M)
    count_grid = composites.member_count[safe_fiss]                      # (K_f,)
    valid_grid = (member_grid >= 0) & (m_idx[None, :] < count_grid[:, None]) \
                 & fiss_valid[:, None]
    flat = jnp.where(valid_grid, member_grid, N).reshape(-1)
    slot_of = jnp.zeros(N, dtype=jnp.int32).at[flat].set(
        jnp.tile(m_idx, K_f), mode='drop')

    # ── Per-fission: BFS tree → subtree sums → best bond cut ─────────────────
    def choose_cut(k):
        c = safe_fiss[k]
        n = composites.member_count[c]
        members = composites.members[c]
        valid_m = (members >= 0) & (m_idx < n)

        ga = composites.edges[c, :, 0]
        gb = composites.edges[c, :, 1]
        evalid = (jnp.arange(E_max) < composites.edge_count[c]) & (ga >= 0)
        la = slot_of[jnp.where(ga >= 0, ga, 0)]
        lb = slot_of[jnp.where(gb >= 0, gb, 0)]

        dist, parent = bfs_tree(la, lb, evalid, M, iters)

        # Per-slot hash values (masked to valid members) for the subtree pass.
        safe_members = jnp.where(members >= 0, members, 0)
        hvals = jax.vmap(lambda s: _entity_hash_val(s, config))(
            particles.species[safe_members]).astype(jnp.uint32)
        base_h = jnp.where(valid_m, hvals, jnp.uint32(0))
        base_c = valid_m.astype(jnp.int32)
        sub_h, sub_c = subtree_sums(parent, base_h, base_c, M, iters)

        # uint32 wraparound sum — same convention as the product hashes, so
        # the complement (total − subtree) is exact before the modulus.
        total_h = jnp.sum(base_h)

        # Candidate cut v ⇔ the spanning-tree edge (v, parent[v]).
        cand_v = valid_m & (parent >= 0)
        h1 = sub_h % jnp.uint32(config.hash_modulus)
        h0 = (total_h - sub_h) % jnp.uint32(config.hash_modulus)
        be1 = jax.vmap(lambda h: _hash_to_binding_energy(h, config, physics))(h1)
        be0 = jax.vmap(lambda h: _hash_to_binding_energy(h, config, physics))(h0)

        # Shell-effect analog: fracture along the cut that maximizes total
        # product binding energy. Deterministic (argmax ties → lowest slot).
        score = jnp.where(cand_v, be0 + be1, -jnp.inf)
        v = jnp.argmax(score)
        q = score[v] - composites.binding_energy[c]

        in_p1 = descendant_mask(parent, v.astype(jnp.int32), M, iters) & valid_m
        a = jnp.where(valid_m,
                      jnp.where(in_p1, jnp.int32(1), jnp.int32(0)),
                      jnp.int32(-1))
        has_cut = jnp.any(cand_v)  # size>=2 composites always have an edge; guard anyway
        return a, q, has_cut

    assignment, q_all, has_cut = jax.vmap(choose_cut)(jnp.arange(K_f, dtype=jnp.int32))

    fires = fiss_valid & has_cut
    if config.forbid_endothermic_fission:
        fires = fires & (q_all >= 0.0)
    kick = jnp.maximum(q_all, 0.0)

    new_particles, new_composites, events = _apply_binary_splits(
        particles, composites, fiss_idx, fires, assignment, kick, config, physics)

    new_state = state._replace(
        particles=new_particles,
        composites=new_composites,
        rng_key=key,
    )

    if config.emit_events:
        return new_state, events
    return new_state
```

- [ ] **Step 5: Update the docstrings that describe the old behavior**

In `halflife/chemistry.py`'s module docstring (lines ~22–25), replace:

```
Fission:
  Composite decay releases its member particles back to free state with
  their original species (no transmutation), and injects
  binding_energy * (1 - fission_cost) as radial kinetic energy.
```

with:

```
Fission:
  Composite decay fractures along the bond cut that maximizes total product
  binding energy (bond-cut fission). Fragments keep their internal bonds;
  species are conserved (no transmutation). The kick is the Q-value
  max(BE(p0) + BE(p1) − BE(parent), 0).
```

- [ ] **Step 6: Remove stale `_hash_to_partition` tests and the `fission_cost` kwarg**

- In `tests/test_hash.py`: delete the section starting at the comment `# ── Tests for _hash_to_partition ──...` (line ~158) through the last of its 4 tests (line ~215). The replacements live in `tests/test_graph.py` and the Task 2 chemistry tests.
- In `tests/test_chemistry.py`: search for `fission_cost` — the only use was in the test deleted in Task 2. Verify: `grep -rn fission_cost tests/ halflife/` → no hits (except possibly old comments in `test_composite_statistics.py`; if a `SimConfig(... fission_cost=...)` call remains anywhere, drop the kwarg).

- [ ] **Step 7: Run the chemistry + hash + graph suites**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/pytest tests/test_chemistry.py tests/test_hash.py tests/test_graph.py -x -q`
Expected: all pass — including Task 2's four new tests and the untouched conservation/valence tests (`test_fission_conserves_particles_and_species`, `test_valence_*`).

If `test_fission_produces_two_products` or `test_fission_creates_intermediate_size_products` fail because they relied on hash-partition specifics (e.g., particular product sizes): they were written against "binary fission produces two non-empty products" — bond-cut fission still satisfies this for exothermic cuts. If a failure traces to Q-suppression (composite legitimately never fissions because every cut is endothermic), set `binding_energy=0.0` on the fixture composite (Q ≥ 0 always) rather than weakening the assertion.

- [ ] **Step 8: Commit**

```bash
git add halflife/chemistry.py halflife/config.py tests/test_chemistry.py tests/test_hash.py && git commit -m "feat(fission): bond-cut fracture replaces slot-order hash partition

Fission now cuts the BFS-spanning-tree edge that maximizes total product
binding energy (subtree hash sums score all cuts in one pass); fragments
keep their internal bonds, so fission never mints edges — fixes the
long-bond bug (notes/2026-06-01). Kick = Q-value, replacing fission_cost;
endothermic fission barred by default (forbid_endothermic_fission).
Shared _apply_binary_splits applier extracted for the upcoming
chemical-scission channel."
```

## Task 4: Phase A verification — full suites, benchmark, docs

**Files:**
- Modify: `CLAUDE.md` (Hash Fission section + config snippet)
- Modify: `notes/2026-06-01-fission-bond-breaking-redesign.md` (status line)

- [ ] **Step 1: Full test suite**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/pytest tests/test_chemistry.py tests/test_step.py tests/test_hash.py tests/test_graph.py tests/test_covalent_bonds_integration.py -n 4 -q`
Expected: all pass. `test_edges_mode_spanning_tree_invariant` (edge_count ≥ n−1) must still hold — fragments are connected by construction.

- [ ] **Step 2: Analysis pipeline suite (kernel emission changed)**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_analysis_events.py tests/test_analysis_metrics.py tests/test_analysis_transitions.py tests/test_analysis_compatibility.py tests/test_analysis_pipeline.py -q`
Expected: all pass (event shapes unchanged in Phase A; only event *contents* differ).

- [ ] **Step 3: Post-change benchmark + long-bond check**

Run: `.venv/bin/python /tmp/bench_fission.py`
Record steps/s vs Task 0 baseline. Target per the note: **< 2× slower**; if the hit exceeds that, lower `fission_label_iters` (64 → 32) and re-measure before escalating.

Then verify the bug fix empirically — append to `/tmp/bench_fission.py` (or a new /tmp script) and run:

```python
# Long-bond check: max bond length across alive composites after the run.
import jax.numpy as jnp
comp = state.composites
e_idx = jnp.arange(config.e_max)
valid = comp.alive[:, None] & (e_idx[None, :] < comp.edge_count[:, None]) & (comp.edges[:, :, 0] >= 0)
pa = state.particles.position[jnp.where(comp.edges[:, :, 0] >= 0, comp.edges[:, :, 0], 0)]
pb = state.particles.position[jnp.where(comp.edges[:, :, 1] >= 0, comp.edges[:, :, 1], 0)]
d = pa - pb
d = d - config.world_width  * jnp.round(d[..., 0:1] / config.world_width)  * jnp.array([1., 0.])
d = d - config.world_height * jnp.round(d[..., 1:2] / config.world_height) * jnp.array([0., 1.])
r = jnp.linalg.norm(d, axis=-1)
print("max bond length:", float(jnp.max(jnp.where(valid, r, 0.0))))
```

Expected: max bond length **single digits** (was up to ~34 with the bug; the note measured mean ~1.5 with a 27+ tail at 800 steps).

- [ ] **Step 4: Update CLAUDE.md**

Replace the `## Hash Fission (binary partition)` section body with a description of bond-cut fission (cut scoring via subtree hash sums, Q-value kick, `forbid_endothermic_fission`, `fission_label_iters`, products keep internal edges, fission_cost removed). Update the `## Configuration` snippet if it mentions removed fields. Keep the section heading style.

- [ ] **Step 5: Mark the note as in-progress**

At the top of `notes/2026-06-01-fission-bond-breaking-redesign.md`, under the `**Topic:**` line, add (do not delete anything):

```markdown
**Status (2026-06-12):** Implementation underway — see
`docs/superpowers/plans/2026-06-12-fission-bond-breaking-redesign.md`.
Commit 1 (bond-cut fission) landed; benchmark: <baseline> → <after> steps/s.
```

(fill in the two numbers measured above)

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md notes/2026-06-01-fission-bond-breaking-redesign.md && git commit -m "docs: bond-cut fission section + redesign note status"
```

---

# Phase B — per-bond chemical scission

## Task 5: Hash-derived bond dissociation energy

**Files:**
- Modify: `halflife/config.py` (5 new fields)
- Modify: `halflife/chemistry.py` (add `_hash_to_bond_energy`, `compute_bond_energy_matrix` — place them right after `compute_r_rest_matrix`)
- Modify: `tests/test_chemistry.py` (add 2 tests after `test_r_rest_is_deterministic_per_hash_modulus`)

- [ ] **Step 1: Write the failing tests**

```python
def test_bond_energy_matrix_shape_symmetry_range():
    """Hash-derived dissociation energies: symmetric, in [0, bond_energy_scale]."""
    from halflife.chemistry import compute_bond_energy_matrix
    config = SimConfig(num_species=5)
    eb = np.asarray(compute_bond_energy_matrix(config))
    assert eb.shape == (5, 5)
    assert np.allclose(eb, eb.T)
    assert (eb >= 0.0).all() and (eb <= config.bond_energy_scale).all()


def test_bond_energy_decorrelated_from_rest_length_and_be():
    """E_b must not be a monotone copy of r_rest or pair BE across pairs
    (different Fibonacci remix stream). Sanity: rank orderings differ."""
    from halflife.chemistry import (compute_bond_energy_matrix,
                                    compute_r_rest_matrix,
                                    _entity_hash_val, _hash_to_binding_energy)
    config = SimConfig(num_species=6)
    physics = initialize_physics_params(config)
    eb = np.asarray(compute_bond_energy_matrix(config))
    rr = np.asarray(compute_r_rest_matrix(config, config.fusion_radius,
                                          config.repulsion_radius))
    iu = np.triu_indices(6)
    # identical rank order across all 21 pairs would mean correlated streams
    assert (np.argsort(eb[iu]) != np.argsort(rr[iu])).any()
```

- [ ] **Step 2: Run to verify failure**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_chemistry.py -k bond_energy -v`
Expected: FAIL — `ImportError: cannot import name 'compute_bond_energy_matrix'`

- [ ] **Step 3: Config fields**

Add to `halflife/config.py` after the `allow_ring_closure` / `max_ring_closures_per_step` block:

```python
    # ── Chemical bond scission (per-bond breaking channel, 2026-06-12) ───────
    # Every edge carries a hash-derived dissociation energy E_b (per species
    # pair, decorrelated from BE / valence / rest length). Two break modes:
    #   kinetic: stretch strain energy 0.5·k_bond·max(r − r_rest, 0)² >= E_b
    #            → the bond snaps deterministically (the harmonic well is no
    #            longer bottomless).
    #   thermal: below threshold, P = 1 − exp(−dt·ν0·exp(−(E_b − strain)/kT))
    #            (Arrhenius). kT = 0 disables thermal breaking entirely.
    # Compression never breaks a bond — only stretch counts.
    # At most one bond per composite breaks per step (the most-overstretched
    # breaking edge). A broken bridge splits the composite into its two
    # connected halves (no kick); a broken ring edge just removes the edge.
    # Requires bond_mode == "edges".
    enable_bond_scission: bool = True
    bond_energy_scale: float = 2.0        # E_b = hash_frac × this
    bond_temperature: float = 0.1         # kT for the Arrhenius thermal channel
    bond_break_attempt_rate: float = 0.1  # ν0 — attempt frequency per sim-time
    max_scissions_per_step: int = 32      # budget; excess breaks defer a step
```

- [ ] **Step 4: Implement the hash + matrix in `halflife/chemistry.py`**

```python
def _hash_to_bond_energy(s_i: jnp.ndarray, s_j: jnp.ndarray,
                         config: SimConfig) -> jnp.ndarray:
    """
    Hash-derived bond dissociation energy for species pair (s_i, s_j).

    Order-independent (commutative additive pair hash) and re-mixed with a
    Fibonacci-style constant DIFFERENT from the BE (2654435761, >>13),
    valence (0x9E3779B1, >>13) and rest-length (0x9E3779B1, >>11) streams so
    the four per-pair properties are mutually decorrelated.

    Returns: scalar float32 in [0, config.bond_energy_scale]
    """
    h_i = _entity_hash_val(s_i, config).astype(jnp.uint32)
    h_j = _entity_hash_val(s_j, config).astype(jnp.uint32)
    h = (h_i + h_j) % jnp.uint32(config.hash_modulus)
    h_mix = (h * jnp.uint32(0x85EBCA6B)) ^ (h >> jnp.uint32(9))
    frac = (h_mix % jnp.uint32(1000)).astype(jnp.float32) / 999.0
    return frac * config.bond_energy_scale


@functools.partial(jax.jit, static_argnums=(0,))
def compute_bond_energy_matrix(config: SimConfig) -> jnp.ndarray:
    """(num_species, num_species) dissociation-energy matrix. Static per
    config (like valence) — part of the universe, not of the run seed."""
    species_idx = jnp.arange(config.num_species, dtype=jnp.int32)
    return jax.vmap(
        lambda i: jax.vmap(
            lambda j: _hash_to_bond_energy(i, j, config)
        )(species_idx)
    )(species_idx)
```

- [ ] **Step 5: Run to verify pass, then commit**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_chemistry.py -k bond_energy -v`
Expected: 2 passed

```bash
git add halflife/config.py halflife/chemistry.py tests/test_chemistry.py && git commit -m "feat(scission): hash-derived per-pair bond dissociation energy + knobs"
```

## Task 6: Failing tests for `apply_bond_scission`

**Files:**
- Create: `tests/test_scission.py`

- [ ] **Step 1: Write the tests**

```python
"""
tests/test_scission.py — chemical (per-bond) breaking channel.

Kinetic break: stretch strain >= E_b snaps the bond deterministically.
Thermal break: Arrhenius roll below threshold (kT → 0 disables it).
A broken bridge splits the composite; a broken ring edge only removes the edge.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax.numpy as jnp
import numpy as np

from halflife.config import SimConfig
from halflife.state import (initialize_world, initialize_interaction_params,
                            initialize_physics_params)
from halflife.chemistry import apply_bond_scission

# kT tiny → thermal channel off; only deterministic kinetic breaks fire.
_BASE = dict(num_species=3, num_particles=10, max_composites=4,
             boundary_mode="reflect", world_width=40.0, world_height=40.0,
             half_life_min=1e9, half_life_max=1e9,   # no half-life decay noise
             bond_temperature=1e-7)


def _world_with_composite(config, pos, edge_pairs, species=None):
    n = max(max(p) for p in edge_pairs) + 1
    N = config.num_particles
    world = initialize_world(config, seed=0)
    if species is None:
        species = np.zeros(N, dtype=np.int32)
    composite_id = np.full(N, -1, dtype=np.int32)
    composite_id[:n] = 0
    members = np.full((config.max_composites, config.max_composite_size), -1, dtype=np.int32)
    members[0, :n] = np.arange(n)
    edges = np.full((config.max_composites, config.e_max, 2), -1, dtype=np.int32)
    for k, (a, b) in enumerate(edge_pairs):
        edges[0, k] = (a, b)
    member_count = np.zeros(config.max_composites, dtype=np.int32)
    member_count[0] = n
    edge_count = np.zeros(config.max_composites, dtype=np.int32)
    edge_count[0] = len(edge_pairs)
    alive = np.zeros(config.max_composites, dtype=bool)
    alive[0] = True
    hl = np.full(config.max_composites, 1e9, dtype=np.float32)
    return world._replace(
        particles=world.particles._replace(
            position=jnp.asarray(pos), species=jnp.asarray(species),
            composite_id=jnp.asarray(composite_id),
            velocity=jnp.zeros((N, 2), dtype=jnp.float32),
        ),
        composites=world.composites._replace(
            members=jnp.asarray(members), member_count=jnp.asarray(member_count),
            alive=jnp.asarray(alive), edges=jnp.asarray(edges),
            edge_count=jnp.asarray(edge_count), half_life=jnp.asarray(hl),
        ),
    )


def test_kinetic_break_snaps_overstretched_dimer():
    """Stretch a dimer's bond way past r_rest: strain >> bond_energy_scale →
    deterministic snap, both particles free, slot dead."""
    config = SimConfig(**_BASE)
    pos = np.array([[5.0, 5.0], [12.0, 5.0]]      # 7 units apart, r_rest <= 1.5
                   + [[30.0 + i, 30.0] for i in range(8)], dtype=np.float32)
    world = _world_with_composite(config, pos, [(0, 1)])
    params = initialize_interaction_params(config, seed=0)
    physics = initialize_physics_params(config)

    state = apply_bond_scission(world, params, config, physics)

    assert not bool(state.composites.alive[0])
    cid = np.asarray(state.particles.composite_id)
    assert cid[0] == -1 and cid[1] == -1


def test_no_break_at_rest_length():
    """A dimer sitting exactly at its rest length has zero strain; with kT≈0
    the thermal channel is dead too → nothing breaks over many calls."""
    config = SimConfig(**_BASE)
    params = initialize_interaction_params(config, seed=0)
    physics = initialize_physics_params(config)
    r01 = float(params.r_rest[0, 0])
    pos = np.array([[5.0, 5.0], [5.0 + r01, 5.0]]
                   + [[30.0 + i, 30.0] for i in range(8)], dtype=np.float32)
    world = _world_with_composite(config, pos, [(0, 1)])

    state = world
    for _ in range(50):
        state = apply_bond_scission(state, params, config, physics)
    assert bool(state.composites.alive[0])
    assert int(state.composites.edge_count[0]) == 1


def test_ring_edge_break_removes_edge_without_split():
    """Triangle with ONE overstretched edge: the edge goes, but the composite
    stays connected through the other two bonds — no split."""
    config = SimConfig(**_BASE)
    params = initialize_interaction_params(config, seed=0)
    physics = initialize_physics_params(config)
    # 0 and 1 far apart (their edge snaps); both near 2 (those bonds at ~rest).
    pos = np.array([[5.0, 5.0], [12.0, 5.0], [8.5, 5.5]]
                   + [[30.0 + i, 30.0] for i in range(7)], dtype=np.float32)
    world = _world_with_composite(config, pos, [(0, 1), (1, 2), (2, 0)])

    state = apply_bond_scission(world, params, config, physics)

    assert bool(state.composites.alive[0])
    assert int(state.composites.member_count[0]) == 3
    assert int(state.composites.edge_count[0]) == 2
    edges_after = {tuple(sorted(e)) for e in
                   np.asarray(state.composites.edges[0][:2]).tolist()}
    assert edges_after == {(1, 2), (0, 2)}


def test_bridge_break_splits_chain_and_conserves_particles():
    """4-chain with the MIDDLE bond overstretched → two dimers; particle and
    species conservation; composite_id consistency."""
    config = SimConfig(**_BASE)
    params = initialize_interaction_params(config, seed=0)
    physics = initialize_physics_params(config)
    r00 = float(params.r_rest[0, 0])
    pos = np.array([[5.0, 5.0], [5.0 + r00, 5.0],          # bond 0-1 at rest
                    [12.0, 5.0], [12.0 + r00, 5.0]]        # bond 2-3 at rest
                   + [[30.0 + i, 30.0] for i in range(6)], dtype=np.float32)
    # middle bond (1,2) spans 12.0 − (5+r00) ≈ 6 units → snaps
    world = _world_with_composite(config, pos, [(0, 1), (1, 2), (2, 3)])
    species_before = np.asarray(world.particles.species).copy()

    state = apply_bond_scission(world, params, config, physics)

    cid = np.asarray(state.particles.composite_id)
    alive = np.asarray(state.composites.alive)
    counts = np.asarray(state.composites.member_count)
    # Two alive composites of size 2
    assert sorted(counts[alive].tolist()) == [2, 2]
    # {0,1} together, {2,3} together, in different composites
    assert cid[0] == cid[1] and cid[2] == cid[3] and cid[0] != cid[2]
    # conservation
    assert (np.asarray(state.particles.species) == species_before).all()
    for c in np.where(alive)[0]:
        mem = np.asarray(state.composites.members[c][:counts[c]])
        assert (np.sort(np.where(cid == c)[0]) == np.sort(mem)).all()
```

- [ ] **Step 2: Run to verify failure**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_scission.py -v`
Expected: FAIL — `ImportError: cannot import name 'apply_bond_scission'`

## Task 7: Implement `apply_bond_scission` + wire into the step

**Files:**
- Modify: `halflife/chemistry.py` (new function after `attempt_ring_closure`)
- Modify: `halflife/step.py` (Phase 6c + event concat)
- Modify: `halflife/analysis/transitions.py` (skip size-0 products)
- Modify: `tests/test_analysis_events.py` (expected event-batch width)
- Modify: `halflife/analysis/runner.py` (shape comment, line ~115)

- [ ] **Step 1: Implement `apply_bond_scission` in chemistry.py**

```python
# ── Chemical Bond Scission (per-bond breaking channel) ────────────────────────

def apply_bond_scission(state: WorldState, params: InteractionParams,
                        config: SimConfig, physics: PhysicsParams):
    """
    Chemical (per-bond) breaking — the kinetic/thermal counterpart to
    half-life fission. Makes the harmonic well finite: every edge carries a
    hash-derived dissociation energy E_b (_hash_to_bond_energy), and each step

      kinetic: stretch strain 0.5·k_bond·max(r − r_rest, 0)² >= E_b snaps the
               bond deterministically;
      thermal: below threshold, the bond snaps with Arrhenius probability
               P = 1 − exp(−dt · ν0 · exp(−(E_b − strain)/kT)).

    Compression never breaks a bond. At most ONE bond per composite breaks
    per step (the most-overstretched breaking edge), and at most
    config.max_scissions_per_step composites break per step (excess defers a
    step, like the fusion/fission budgets). If the broken bond was a bridge,
    the composite splits into its two connected halves via
    _apply_binary_splits with zero kick (the spring's stored energy simply
    stops acting — the pairwise forces take over); if it was a ring edge,
    only the edge is removed (members and composite_id untouched, though the
    slot's hash-derived properties and age are refreshed by the applier).

    Requires bond_mode == "edges" (step.py gates statically; the early
    return below covers standalone use).

    Returns:
        Updated WorldState (and a ReactionEvent batch of leading dim
        min(max_scissions_per_step, C) when config.emit_events).
    """
    if not (config.enable_bond_scission and config.bond_mode == "edges"):
        return state

    particles = state.particles
    composites = state.composites
    key, subkey = jax.random.split(state.rng_key)
    N = config.num_particles
    M = config.max_composite_size
    C = config.max_composites
    E_max = config.e_max
    iters = config.fission_label_iters
    m_idx = jnp.arange(M, dtype=jnp.int32)
    e_idx = jnp.arange(E_max, dtype=jnp.int32)

    # ── Per-edge strain vs dissociation energy, over the (C, E) grid ────────
    ga = composites.edges[:, :, 0]   # (C, E)
    gb = composites.edges[:, :, 1]
    evalid = composites.alive[:, None] & (e_idx[None, :] < composites.edge_count[:, None]) & (ga >= 0)
    safe_a = jnp.where(ga >= 0, ga, 0)
    safe_b = jnp.where(gb >= 0, gb, 0)
    pa = particles.position[safe_a]  # (C, E, 2)
    pb = particles.position[safe_b]
    d = pa - pb
    if config.boundary_mode == "periodic":
        d = d - config.world_width  * jnp.round(d[..., 0:1] / config.world_width)  * jnp.array([1., 0.])
        d = d - config.world_height * jnp.round(d[..., 1:2] / config.world_height) * jnp.array([0., 1.])
    r = jnp.linalg.norm(d, axis=-1)  # (C, E)
    sa = particles.species[safe_a]
    sb = particles.species[safe_b]
    r_rest = params.r_rest[sa, sb] * physics.r_rest_scale
    # Only stretch strains a bond; compression never breaks it.
    stretch = jnp.maximum(r - r_rest, 0.0)
    strain = 0.5 * physics.k_bond * stretch ** 2

    bond_e = compute_bond_energy_matrix(config)[sa, sb]  # (C, E)

    kT = jnp.maximum(jnp.float32(config.bond_temperature), 1e-8)
    barrier = jnp.maximum(bond_e - strain, 0.0)
    rate = config.bond_break_attempt_rate * jnp.exp(-barrier / kT)
    p_thermal = 1.0 - jnp.exp(-physics.dt * rate)
    u = jax.random.uniform(subkey, (C, E_max))
    breaks = evalid & ((strain >= bond_e) | (u < p_thermal))

    # ── One break per composite: the most-overstretched breaking edge ───────
    over = jnp.where(breaks, strain - bond_e, -jnp.inf)
    chosen_e = jnp.argmax(over, axis=1).astype(jnp.int32)  # (C,)
    has_break = jnp.any(breaks, axis=1)                    # (C,)

    # ── Budget-compact to a (K_s,) batch (same trick as fission) ────────────
    K_s = min(config.max_scissions_per_step, C)
    rank = jnp.cumsum(has_break.astype(jnp.int32)) - 1
    sel = has_break & (rank < K_s)
    cand = jnp.where(sel, jnp.arange(C, dtype=jnp.int32), C)
    sciss_idx = jnp.sort(cand)[:K_s]
    sciss_valid = sciss_idx < C
    safe_sc = jnp.minimum(sciss_idx, C - 1)
    cut_e = chosen_e[safe_sc]                              # (K_s,)

    # ── Remove the chosen edge from each selected composite (compact) ───────
    def drop_edge(k):
        c = safe_sc[k]
        keep = (e_idx < composites.edge_count[c]) & (composites.edges[c, :, 0] >= 0) \
               & (e_idx != cut_e[k])
        pos = jnp.cumsum(keep.astype(jnp.int32)) - 1
        out = jnp.where(keep, pos, E_max)
        new_e = jnp.full((E_max, 2), -1, dtype=jnp.int32).at[out].set(
            composites.edges[c], mode='drop')
        return new_e, jnp.sum(keep.astype(jnp.int32))

    new_edges_k, new_ecnt_k = jax.vmap(drop_edge)(jnp.arange(K_s, dtype=jnp.int32))
    drop_slots = jnp.where(sciss_valid, sciss_idx, C)
    composites_cut = composites._replace(
        edges=composites.edges.at[drop_slots].set(new_edges_k, mode='drop'),
        edge_count=composites.edge_count.at[drop_slots].set(new_ecnt_k, mode='drop'),
    )

    # ── pid → local slot for the batch (member-disjoint rows) ───────────────
    member_grid = composites.members[safe_sc]
    count_grid = composites.member_count[safe_sc]
    valid_grid = (member_grid >= 0) & (m_idx[None, :] < count_grid[:, None]) \
                 & sciss_valid[:, None]
    flat = jnp.where(valid_grid, member_grid, N).reshape(-1)
    slot_of = jnp.zeros(N, dtype=jnp.int32).at[flat].set(
        jnp.tile(m_idx, K_s), mode='drop')

    # ── Bipartition by reachability over the remaining edges ────────────────
    # Fragment 0 = everything still reachable from the removed edge's "a"
    # endpoint; fragment 1 = the rest. If the removed edge was a ring edge,
    # everything stays reachable → fragment 1 is empty → the applier writes
    # product 0 (the whole composite, minus the edge) back to the parent slot.
    def label_split(k):
        c = safe_sc[k]
        n = composites.member_count[c]
        members = composites.members[c]
        valid_m = (members >= 0) & (m_idx < n)

        rga = composites_cut.edges[c, :, 0]
        rgb = composites_cut.edges[c, :, 1]
        revalid = (e_idx < composites_cut.edge_count[c]) & (rga >= 0)
        la = slot_of[jnp.where(rga >= 0, rga, 0)]
        lb = slot_of[jnp.where(rgb >= 0, rgb, 0)]

        # local slot of the removed edge's first endpoint (from ORIGINAL edges)
        cut_a_pid = composites.edges[c, cut_e[k], 0]
        start = slot_of[jnp.where(cut_a_pid >= 0, cut_a_pid, 0)]

        reach = reachable_mask(la, lb, revalid, start, M, iters)
        a = jnp.where(valid_m,
                      jnp.where(reach, jnp.int32(0), jnp.int32(1)),
                      jnp.int32(-1))
        return a

    assignment = jax.vmap(label_split)(jnp.arange(K_s, dtype=jnp.int32))

    # No kick: the snapped spring just stops acting; pairwise forces take over.
    kick = jnp.zeros(K_s, dtype=jnp.float32)

    new_particles, new_composites, events = _apply_binary_splits(
        particles, composites_cut, sciss_idx, sciss_valid, assignment, kick,
        config, physics)

    new_state = state._replace(
        particles=new_particles,
        composites=new_composites,
        rng_key=key,
    )

    if config.emit_events:
        return new_state, events
    return new_state
```

- [ ] **Step 2: Run the scission tests**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_scission.py -v`
Expected: 4 passed

- [ ] **Step 3: Wire into `halflife/step.py`**

Add after Phase 6b (ring closure) and before Phase 7 (decay):

```python
    # ── Phase 6c: Chemical bond scission (per-bond breaking channel) ──────────
    # Kinetic (strain >= E_b) + thermal (Arrhenius) bond breaking. Statically
    # gated: in star_spring/off modes the edge array is physics-inert, and the
    # enable flag lets experiments A/B the channel.
    from halflife.chemistry import apply_bond_scission
    if config.bond_mode == "edges" and config.enable_bond_scission:
        if config.emit_events:
            state, scission_events = apply_bond_scission(state, params, config, physics)
        else:
            state = apply_bond_scission(state, params, config, physics)
            scission_events = None
    else:
        scission_events = None
```

Then extend the event-log assembly at the bottom. Update the comment block (it documents `E = min(max_fusions, N) + min(max_fissions, C)`) to add `+ min(max_scissions, C) when bond scission is enabled`, and change the concatenation to:

```python
    if config.emit_events:
        from halflife.state import ReactionEvent
        batches = [fusion_events, fission_events]
        if scission_events is not None:
            batches.append(scission_events)
        events = ReactionEvent(
            kind=jnp.concatenate([b.kind for b in batches]),
            source_slots=jnp.concatenate([b.source_slots for b in batches], axis=0),
            source_hashes=jnp.concatenate([b.source_hashes for b in batches], axis=0),
            source_sizes=jnp.concatenate([b.source_sizes for b in batches], axis=0),
            product_slots=jnp.concatenate([b.product_slots for b in batches], axis=0),
            product_hashes=jnp.concatenate([b.product_hashes for b in batches], axis=0),
            product_sizes=jnp.concatenate([b.product_sizes for b in batches], axis=0),
        )
        return final_state, events
    return final_state
```

- [ ] **Step 4: Analysis ingest — skip empty products**

A ring-edge scission emits a fission event whose product 1 has size 0. In `halflife/analysis/transitions.py`, `_iter_edges`, change the fission branch to:

```python
        elif kind[i] == KIND_FISSION:
            # C → A + B; yield (C, A) and (C, B). Source is in slot 0.
            # Ring-edge scissions emit size-0 product-1 entries (the bond
            # broke but nothing split off) — skip those cells.
            for prod_idx in (0, 1):
                if ps[i, prod_idx] == 0:
                    continue
                yield (int(sh[i, 0]),         int(ss[i, 0]),
                       int(ph[i, prod_idx]),  int(ps[i, prod_idx]))
```

- [ ] **Step 5: Fix the event-batch width assertion**

In `tests/test_analysis_events.py` (line ~189), the per-step width test:

```python
    expected_e = (min(config.max_fusions_per_step, config.num_particles)
                  + min(config.max_fissions_per_step, config.max_composites)
                  + (min(config.max_scissions_per_step, config.max_composites)
                     if (config.bond_mode == "edges" and config.enable_bond_scission)
                     else 0))
```

Also update the docstring at line ~177 and the comment in `halflife/analysis/runner.py` line ~115 to mention the scission batch.

- [ ] **Step 6: Run the affected suites**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/pytest tests/test_scission.py tests/test_chemistry.py tests/test_step.py -n 4 -q`
Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_analysis_events.py tests/test_analysis_metrics.py tests/test_analysis_transitions.py tests/test_analysis_compatibility.py tests/test_analysis_pipeline.py -q`
Expected: all pass.

Watch for: integration tests that count composites may see slightly different steady states (a new breaking channel exists). If `test_fusion_count_grows` or composite-population assertions fail *marginally*, the physical defaults (`bond_energy_scale=2.0`, `kT=0.1`, `ν0=0.1`) may be too aggressive — first check the numbers (mean E_b = 1.0 vs typical strain ≈ 0.5·20·0.1² = 0.1 → thermal rate ν0·e^(−9) ≈ 0: breaks should be rare at rest). A mass die-off signals a bug (e.g. strain computed with compression), not a tuning issue.

- [ ] **Step 7: Commit**

```bash
git add halflife/chemistry.py halflife/step.py halflife/analysis/transitions.py halflife/analysis/runner.py tests/test_scission.py tests/test_analysis_events.py && git commit -m "feat(scission): kinetic + thermal per-bond breaking channel

Bonds now have finite depth: stretch strain >= hash-derived E_b snaps a
bond deterministically; below threshold an Arrhenius roll
(bond_temperature, bond_break_attempt_rate) breaks it thermally. One bond
per composite per step, max_scissions_per_step composites per step. Bridge
breaks split via _apply_binary_splits (zero kick); ring breaks just drop
the edge. Scission events ride the fission event kind."
```

## Task 8: Phase B docs

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Document the channel in CLAUDE.md**

Add a `## Chemical Bond Scission` section after the fission section: the two break modes with formulas, the one-break-per-composite-per-step rule, the budget, the ring-vs-bridge distinction, the config knobs, and that scission events appear as fission events in the analysis pipeline. Update the `## Configuration` snippet with the new knobs.

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md && git commit -m "docs(claude): chemical bond-scission channel"
```

---

# Phase C — liquid-drop nuclear stability

## Task 9: Internal repulsion PE from the force kernel

**Files:**
- Modify: `halflife/interactions.py` (`compute_forces_for_particle`, `compute_all_forces`)
- Modify: `halflife/step.py` (call site)
- Modify: `halflife/profiler.py` + `tests/test_performance.py` (call sites — grep first)
- Modify: `tests/test_chemistry.py` or new `tests/test_liquid_drop.py` (PE unit test)

- [ ] **Step 1: Write the failing test**

Create `tests/test_liquid_drop.py`:

```python
"""
tests/test_liquid_drop.py — fissility-based composite stability.

Phase C of the fission redesign: cohesion (Σ bond E_b − surface term) vs
disruption (internal hard-core repulsion PE, the Coulomb analog) sets a live
half-life that collapses as fissility x → 1.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax.numpy as jnp
import numpy as np

from halflife.config import SimConfig
from halflife.state import (initialize_world, initialize_interaction_params,
                            initialize_physics_params)


def _two_member_world(config, gap):
    """One 2-member composite with member distance `gap`; rest parked far away."""
    N = config.num_particles
    world = initialize_world(config, seed=0)
    pos = np.array([[5.0, 5.0], [5.0 + gap, 5.0]]
                   + [[30.0 + i, 30.0] for i in range(N - 2)], dtype=np.float32)
    composite_id = np.full(N, -1, dtype=np.int32)
    composite_id[:2] = 0
    members = np.full((config.max_composites, config.max_composite_size), -1, dtype=np.int32)
    members[0, :2] = (0, 1)
    edges = np.full((config.max_composites, config.e_max, 2), -1, dtype=np.int32)
    edges[0, 0] = (0, 1)
    member_count = np.zeros(config.max_composites, dtype=np.int32); member_count[0] = 2
    edge_count = np.zeros(config.max_composites, dtype=np.int32); edge_count[0] = 1
    alive = np.zeros(config.max_composites, dtype=bool); alive[0] = True
    return world._replace(
        particles=world.particles._replace(
            position=jnp.asarray(pos),
            species=jnp.zeros(N, dtype=jnp.int32),
            composite_id=jnp.asarray(composite_id),
        ),
        composites=world.composites._replace(
            members=jnp.asarray(members), member_count=jnp.asarray(member_count),
            alive=jnp.asarray(alive), edges=jnp.asarray(edges),
            edge_count=jnp.asarray(edge_count),
        ),
    )


def test_internal_repulsion_pe_closed_form():
    """Two same-composite members at r < repulsion_radius: per-composite PE
    (sum of per-particle halves) equals R·(rr − r)²/(2·rr)."""
    from halflife.interactions import compute_all_forces
    from halflife.spatial import build_cell_list, find_all_neighbors
    config = SimConfig(num_particles=10, max_composites=4, num_species=3,
                       boundary_mode="reflect", world_width=40.0, world_height=40.0)
    physics = initialize_physics_params(config)
    params = initialize_interaction_params(config, seed=0)
    gap = 0.4    # inside repulsion_radius = 0.8
    world = _two_member_world(config, gap)

    cell_list = build_cell_list(world.particles.position, config)
    neighbors = find_all_neighbors(world.particles.position, cell_list, config)
    forces, rep_pe = compute_all_forces(
        world.particles.position, world.particles.species,
        world.particles.composite_id, neighbors, params, config, physics)

    rr = config.repulsion_radius
    expected_pair_pe = config.repulsion_strength * (rr - gap) ** 2 / (2.0 * rr)
    # both members see the pair → per-particle sum is 2× the pair PE
    total = float(rep_pe[0] + rep_pe[1])
    assert np.isclose(total * 0.5, expected_pair_pe, rtol=1e-4), \
        f"got {total * 0.5}, expected {expected_pair_pe}"
    # free particles far away contribute nothing
    assert float(jnp.sum(rep_pe[2:])) == 0.0
    assert forces.shape == (10, 2)


def test_repulsion_pe_zero_for_free_and_distant():
    """Same-composite members OUTSIDE the hard core have zero repulsion PE."""
    from halflife.interactions import compute_all_forces
    from halflife.spatial import build_cell_list, find_all_neighbors
    config = SimConfig(num_particles=10, max_composites=4, num_species=3,
                       boundary_mode="reflect", world_width=40.0, world_height=40.0)
    physics = initialize_physics_params(config)
    params = initialize_interaction_params(config, seed=0)
    world = _two_member_world(config, gap=1.5)   # outside rr = 0.8

    cell_list = build_cell_list(world.particles.position, config)
    neighbors = find_all_neighbors(world.particles.position, cell_list, config)
    _, rep_pe = compute_all_forces(
        world.particles.position, world.particles.species,
        world.particles.composite_id, neighbors, params, config, physics)
    assert float(jnp.sum(rep_pe)) == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_liquid_drop.py -v`
Expected: FAIL — `compute_all_forces` takes no `composite_id` / returns no tuple.

- [ ] **Step 3: Extend the force kernel**

In `halflife/interactions.py`, change `compute_forces_for_particle` so the inner neighbor function also returns the same-composite hard-core PE, and thread `composite_id` through. The closed form comes from integrating the hard-core ramp: `U(r) = ∫ᵣ^rr R·(1 − s/rr) ds = R·(rr − r)²/(2·rr)`.

```python
def compute_forces_for_particle(i: jnp.ndarray,
                                  positions: jnp.ndarray,
                                  species: jnp.ndarray,
                                  composite_id: jnp.ndarray,
                                  neighbors: jnp.ndarray,
                                  params: InteractionParams,
                                  config: SimConfig,
                                  physics: PhysicsParams) -> tuple:
```

(keep the existing docstring; add to Args: `composite_id: (N,)`, and to Returns: `(force (2,), rep_pe ())` — `rep_pe` is this particle's summed hard-core PE against SAME-COMPOSITE neighbors, the liquid-drop "Coulomb" disruption term. Each pair is counted from both sides; per-composite totals must be halved.)

Inner function becomes:

```python
    pos_i = positions[i]
    sp_i  = species[i]
    cid_i = composite_id[i]

    def contrib_from_neighbor(j):
        valid = (j >= 0)
        safe_j = jnp.where(valid, j, 0)
        pos_j = jnp.where(valid, positions[safe_j], pos_i)  # safe fallback
        sp_j  = jnp.where(valid, species[safe_j], sp_i)
        f = pairwise_force(pos_i, pos_j, sp_i, sp_j, params, config, physics)

        # Same-composite hard-core PE (liquid-drop disruption term). The
        # min-image + norm below duplicates pairwise_force's internal math on
        # identical inputs — XLA CSEs it, so this is effectively free.
        d = pos_i - pos_j
        if config.boundary_mode == "periodic":
            d = d - config.world_width  * jnp.round(d[0] / config.world_width) * jnp.array([1., 0.])
            d = d - config.world_height * jnp.round(d[1] / config.world_height) * jnp.array([0., 1.])
        r = jnp.linalg.norm(d) + 1e-10
        rr = physics.repulsion_radius
        same_comp = valid & (cid_i >= 0) & (composite_id[safe_j] == cid_i)
        u = jnp.where(same_comp & (r < rr),
                      physics.repulsion_strength * (rr - r) ** 2 / (2.0 * rr + 1e-10),
                      0.0)
        return jnp.where(valid, f, jnp.zeros(2)), u

    forces, pes = jax.vmap(contrib_from_neighbor)(neighbors)
    return jnp.sum(forces, axis=0), jnp.sum(pes)
```

`compute_all_forces` gains the `composite_id` parameter (after `species`), passes it through, and returns `(forces (N,2), rep_pe (N,))` from the vmap. Update its docstring Returns accordingly.

- [ ] **Step 4: Update all call sites**

Run: `grep -rn "compute_all_forces(" halflife/ tests/ --include="*.py" | grep -v __pycache__`

At each site (known: `halflife/step.py` Phase 3, `halflife/profiler.py`, `tests/test_performance.py`), add the `composite_id` argument and unpack the tuple. In `step.py`:

```python
    forces, rep_pe = compute_all_forces(
        particles.position, particles.species, particles.composite_id,
        neighbors, params, config, physics
    )
```

(`rep_pe` is consumed in Task 10; until then it's unused — XLA DCEs it, and Python doesn't mind.)

- [ ] **Step 5: Run to verify pass + no regressions**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/pytest tests/test_liquid_drop.py tests/test_step.py tests/test_chemistry.py -n 4 -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add halflife/interactions.py halflife/step.py halflife/profiler.py tests/test_performance.py tests/test_liquid_drop.py && git commit -m "feat(forces): accumulate same-composite hard-core PE (liquid-drop Coulomb term)

Closed-form U(r) = R(rr−r)²/(2rr) summed per particle against
same-composite neighbors, piggybacking on the force pass (CSE'd min-image,
DCE'd when unused). Feeds the fissility law next."
```

## Task 10: Fissility half-life law

**Files:**
- Modify: `halflife/config.py` (4 new fields; docstring tweak on `composite_size_decay_scale`)
- Modify: `halflife/chemistry.py` (add `compute_liquid_drop_half_life`)
- Modify: `halflife/step.py` (Phase 6d before decay)
- Modify: `tests/test_liquid_drop.py` (3 more tests)

- [ ] **Step 1: Write the failing tests (append to tests/test_liquid_drop.py)**

```python
def test_crammed_composite_gets_shorter_half_life():
    """Same 2-member composite; the overlapping one (r < rr) must get a
    shorter live half-life than the relaxed one."""
    from halflife.chemistry import compute_liquid_drop_half_life
    from halflife.interactions import compute_all_forces
    from halflife.spatial import build_cell_list, find_all_neighbors
    config = SimConfig(num_particles=10, max_composites=4, num_species=3,
                       boundary_mode="reflect", world_width=40.0, world_height=40.0)
    physics = initialize_physics_params(config)
    params = initialize_interaction_params(config, seed=0)

    def live_hl(gap):
        world = _two_member_world(config, gap)
        cell_list = build_cell_list(world.particles.position, config)
        neighbors = find_all_neighbors(world.particles.position, cell_list, config)
        _, rep_pe = compute_all_forces(
            world.particles.position, world.particles.species,
            world.particles.composite_id, neighbors, params, config, physics)
        hl = compute_liquid_drop_half_life(
            world.particles, world.composites, rep_pe, config, physics)
        return float(hl[0])

    assert live_hl(0.2) < live_hl(1.5)


def test_zero_cohesion_collapses_to_min_half_life():
    """A composite whose surface term swamps its bond aggregate sits at
    half_life_min regardless of repulsion."""
    from halflife.chemistry import compute_liquid_drop_half_life
    config = SimConfig(num_particles=10, max_composites=4, num_species=3,
                       boundary_mode="reflect", world_width=40.0, world_height=40.0,
                       surface_energy_coeff=1e6)   # cohesion guaranteed negative
    physics = initialize_physics_params(config)
    world = _two_member_world(config, gap=1.5)
    rep_pe = jnp.zeros(config.num_particles, dtype=jnp.float32)
    hl = compute_liquid_drop_half_life(
        world.particles, world.composites, rep_pe, config, physics)
    assert np.isclose(float(hl[0]), config.half_life_min, rtol=1e-5)


def test_legacy_mode_leaves_half_life_alone():
    """stability_mode='legacy': one simulation step must not rewrite a
    non-decaying composite's stored half-life."""
    import jax
    from halflife.step import simulation_step
    config = SimConfig(num_particles=10, max_composites=4, num_species=3,
                       boundary_mode="reflect", world_width=40.0, world_height=40.0,
                       stability_mode="legacy")
    physics = initialize_physics_params(config)
    params = initialize_interaction_params(config, seed=0)
    world = _two_member_world(config, gap=1.5)
    sentinel = 12345.0
    hl = np.zeros(config.max_composites, dtype=np.float32)
    hl[0] = sentinel
    world = world._replace(composites=world.composites._replace(
        half_life=jnp.asarray(hl)))
    step_fn = jax.jit(simulation_step, static_argnums=(2,))
    state = step_fn(world, params, config, physics)
    # composite 0 survives (hl huge) and keeps its stored half-life
    assert bool(state.composites.alive[0])
    assert np.isclose(float(state.composites.half_life[0]), sentinel)
```

- [ ] **Step 2: Run to verify failure**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_liquid_drop.py -k "crammed or zero_cohesion or legacy" -v`
Expected: FAIL — no `compute_liquid_drop_half_life`, no `stability_mode` field.

- [ ] **Step 3: Config fields**

In `halflife/config.py`, replace the `# ── Composite Stability ──...` section comment + field with:

```python
    # ── Composite Stability ───────────────────────────────────────────────────
    # stability_mode picks how half-life is determined:
    #   "liquid_drop" (default) — live fissility law, recomputed every step:
    #       E_coh = Σ bond E_b − surface_energy_coeff · n^(2/3)
    #       x     = E_rep / (2 · E_coh)          (E_rep = internal hard-core PE)
    #       hl    = hl_min + (hl_max − hl_min) · t_coh · clip(1 − x, 0, 1)^fissility_exponent
    #       with t_coh = clip(E_coh / (cohesion_hl_scale · n), 0, 1).
    #     Big/crammed/weakly-bonded composites fission fast; the BE→hl values
    #     written at fusion/fission time become initial placeholders only.
    #   "legacy" — the original hash-BE → half-life formula with the size
    #     penalty below, fixed at creation time.
    # Static field — changing it retraces once.
    stability_mode: str = "liquid_drop"
    surface_energy_coeff: float = 0.5   # a_s — cohesion penalty × n^(2/3)
    cohesion_hl_scale: float = 1.0      # per-member cohesion needed for max stability
    fissility_exponent: float = 1.0     # sharpness of the collapse as x → 1
    composite_size_decay_scale: float = 0.05   # size penalty on composite half-life (legacy mode + creation-time placeholder values)
```

(the existing `composite_size_decay_scale` line moves into this block with the docstring suffix updated; do not change its value.)

- [ ] **Step 4: Implement `compute_liquid_drop_half_life` in chemistry.py**

```python
# ── Liquid-Drop Stability (live fissility half-life) ──────────────────────────

def compute_liquid_drop_half_life(particles, composites, rep_pe: jnp.ndarray,
                                  config: SimConfig,
                                  physics: PhysicsParams) -> jnp.ndarray:
    """
    (C,) live half-life from the liquid-drop competition: cohesion (aggregate
    bond dissociation energy − surface term) vs disruption (internal hard-core
    repulsion PE, computed by the force pass and passed in as per-particle
    rep_pe). Fissility x = E_rep / (2·E_coh); half-life collapses as x → 1
    and scales with normalized cohesion below that. Replaces the creation-time
    BE→half-life value (which remains as a placeholder for display until the
    first step touches it).

    Dead slots keep their stored half_life (the alive mask gates decay anyway).
    """
    C = config.max_composites
    E_max = config.e_max
    e_idx = jnp.arange(E_max, dtype=jnp.int32)

    # E_rep per composite: per-particle PE counted from both pair endpoints → ×0.5
    safe_cid = jnp.where(particles.composite_id >= 0, particles.composite_id, C)
    e_rep = jnp.zeros(C, dtype=jnp.float32).at[safe_cid].add(rep_pe, mode='drop') * 0.5

    # E_coh: Σ E_b over valid edges − surface term
    ga = composites.edges[:, :, 0]
    gb = composites.edges[:, :, 1]
    evalid = composites.alive[:, None] & (e_idx[None, :] < composites.edge_count[:, None]) & (ga >= 0)
    sa = particles.species[jnp.where(ga >= 0, ga, 0)]
    sb = particles.species[jnp.where(gb >= 0, gb, 0)]
    eb = compute_bond_energy_matrix(config)[sa, sb]          # (C, E)
    bond_sum = jnp.sum(jnp.where(evalid, eb, 0.0), axis=1)   # (C,)
    n = composites.member_count.astype(jnp.float32)
    e_coh = bond_sum - config.surface_energy_coeff * n ** (2.0 / 3.0)

    x = e_rep / (2.0 * jnp.maximum(e_coh, 1e-6))
    t_coh = jnp.clip(e_coh / (config.cohesion_hl_scale * jnp.maximum(n, 1.0)), 0.0, 1.0)
    stab = t_coh * jnp.clip(1.0 - x, 0.0, 1.0) ** config.fissility_exponent
    hl = config.half_life_min + (config.half_life_max - config.half_life_min) * stab
    return jnp.where(composites.alive, hl, composites.half_life)
```

- [ ] **Step 5: Wire into `halflife/step.py`** — insert between Phase 6c (scission) and Phase 7 (decay):

```python
    # ── Phase 6d: live liquid-drop half-life (before the decay roll) ──────────
    # Fissility law over the CURRENT bond graph + the internal repulsion PE
    # from this step's force pass (pre-integration positions — one-step lag,
    # negligible at dt=0.06). In legacy mode the creation-time half_life
    # stands and rep_pe is DCE'd out of the compiled step.
    if config.stability_mode == "liquid_drop":
        from halflife.chemistry import compute_liquid_drop_half_life
        new_hl = compute_liquid_drop_half_life(
            state.particles, state.composites, rep_pe, config, physics)
        state = state._replace(
            composites=state.composites._replace(half_life=new_hl))
    ```

- [ ] **Step 6: Run the new tests + the full suite**

Run: `JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_liquid_drop.py -v`
Expected: 5 passed.

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/pytest tests/test_chemistry.py tests/test_step.py tests/test_scission.py tests/test_graph.py tests/test_covalent_bonds_integration.py -n 4 -q`

Watch for: tests that assert composite populations after N steps (`test_fusion_occurs`, `test_decay_occurs`, covalent integration) now run under the fissility law. `test_composite_half_life_valid` checks half-lives lie in [half_life_min, half_life_max] — the law respects those bounds by construction. If a population test fails, diagnose with one diagnostic run (`.venv/bin/python -m halflife.analysis --scenario current_experiment --steps 1000 --platform gpu`) before touching defaults; the failure mode to expect is everything sitting at hl_min because `surface_energy_coeff` is too big for small composites (a 2-mer has bond_sum ≈ 1.0 vs surface 0.5·2^(2/3) ≈ 0.79 → e_coh ≈ 0.2 — alive but fragile; lower `surface_energy_coeff` to 0.25 if dimers die too fast).

- [ ] **Step 7: Commit**

```bash
git add halflife/config.py halflife/chemistry.py halflife/step.py tests/test_liquid_drop.py && git commit -m "feat(stability): liquid-drop fissility half-life replaces fixed BE formula

Live per-step law: cohesion = Σ bond E_b − a_s·n^(2/3) vs disruption =
internal hard-core repulsion PE; x = E_rep/(2E_coh); hl collapses as x→1
and scales with normalized cohesion. composite_size_decay_scale is now
legacy-mode only. stability_mode='legacy' preserves the old behavior."
```

## Task 11: Final verification, benchmark, docs

**Files:**
- Modify: `CLAUDE.md`, `notes/2026-06-01-fission-bond-breaking-redesign.md`

- [ ] **Step 1: Full test sweep**

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/pytest tests/ -n 4 -q --ignore=tests/test_composite_statistics.py --ignore=tests/test_performance.py
JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_analysis_events.py tests/test_analysis_metrics.py tests/test_analysis_transitions.py tests/test_analysis_compatibility.py tests/test_analysis_pipeline.py tests/test_analysis_openendedness.py -q
```

Expected: all pass.

- [ ] **Step 2: Final benchmark + eyeball run**

Run `/tmp/bench_fission.py` once more; record final steps/s vs the Task 0 baseline (target < 2× slower; the dominant new costs are the fori_loop sweeps and the (C, E) scission grid). Then a short diagnostic to confirm the world still *does chemistry*:

```bash
.venv/bin/python -m halflife.analysis --scenario current_experiment --steps 2000 --sample-every 100 --platform gpu
```

Open the HTML: composites should form, grow, and break; the Tier-3 fission row should show bond-cut products; no long-bond population (cross-check the max-bond-length probe from Task 4).

- [ ] **Step 3: Docs**

- CLAUDE.md: add the liquid-drop section (formula, knobs, legacy switch), note `compute_all_forces` now returns `(forces, rep_pe)` and takes `composite_id`.
- Note file: update the status line — all three commits landed, final benchmark numbers.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md notes/2026-06-01-fission-bond-breaking-redesign.md && git commit -m "docs: liquid-drop stability + redesign completion status"
```

---

## Self-Review Notes (resolved during planning)

- **Spec coverage:** note decisions 1–5 → Tasks 3 (cut + Q + barrier), 7 (chemical channel), 10 (liquid drop + folding `composite_size_decay_scale` into legacy). Morse force explicitly descoped (note marks it optional).
- **Free-slot headroom (note nub):** splits stay binary and budget-capped (`max_fissions_per_step`, `max_scissions_per_step`), so worst-case new slots per step = 64 + 32 ≪ 3000.
- **r_rest band (note nub):** confirmed current code spans `[repulsion_radius, fusion_radius]` via `params.r_rest` × `physics.r_rest_scale`; scission strain uses exactly that product (matching `compute_edge_bond_forces`).
- **Type consistency:** `_apply_binary_splits(particles, composites, split_slots, fires, assignment, kick_energy, config, physics)` is called with that exact signature from both `apply_composite_decay` (Task 3) and `apply_bond_scission` (Task 7). `compute_all_forces(positions, species, composite_id, neighbors, params, config, physics) -> (forces, rep_pe)` is used identically in Tasks 9 and 10 and both liquid-drop tests.
- **uint32 hash convention:** subtree sums accumulate raw uint32 (wraps mod 2^32) and apply `% hash_modulus` at the end — byte-identical to `hash_for_product`'s `sum % modulus`, so cut-scored fragment hashes equal the hashes later written to the product slots.
