# Sparse covalent bonds — design specification

**Date:** 2026-05-14
**Status:** Design review
**Related:** [2026-05-07-hash-fission-design.md](2026-05-07-hash-fission-design.md) (this builds on the binary-fission model)

## Goal

Replace the current COM-spring bond force with **sparse covalent bonds** — explicit edges between specific particle pairs, each acting as a harmonic spring with a species-pair rest length. Composites become trees (or trees + occasional rings) of bonded particles rather than star-shaped clusters tied to a center of mass.

This addresses two specific symptoms of the current simulator:

1. **All composites look the same — close-packed disks.** The COM-spring has no rest length, no topology, and no per-pair specificity, so every composite collapses to a disk whose only variable is size.
2. **Composites don't interact spatially with each other.** They behave as rigid point-mass blobs, bouncing off via hard-core repulsion instead of soft-deforming on contact.

Both symptoms come from the same root: the COM-spring imposes an aggregate constraint (every member → one point) instead of a topological one (specific pairs → specific distances). Materializing the n−1 (or more, for rings) edges per composite as real bonds, with per-pair rest lengths, gives:

- **Shape diversity** — chains, branches, rings, mixed-length structures all emerge from fusion history.
- **Soft inter-composite collisions** — local bonds stretch under contact, the rest of the composite drags along, composites deform around each other.
- **Coherent valence physics** — the spanning-tree accounting the existing `free_bonds` field implies becomes an actual physical structure rather than an abstract invariant.

## Design overview

1. **Edges are first-class state.** `CompositeState` gains an `edges: (C, E_max, 2) int32` field and an `edge_count: (C,) int32` field. Edges store particle-id pairs; `-1` marks unused slots. `E_max = (max_composite_size · max_valence) // 2`.
2. **Per-particle valence becomes canonical.** A new derived quantity `degree[i] ∈ [0, v_{species[i]}]` counts incident edges per particle. The composite-level `free_bonds` becomes a derived sum and the existing field can be repurposed or removed.
3. **Bond force is a per-edge harmonic spring.** `F = −k_bond · (|r_ij| − r_rest[s_i, s_j]) · r̂_ij`, scatter-added into the (N, 2) force buffer. Rest lengths come from a hash-derived `(S, S)` symmetric matrix.
4. **Fusion records edges.** Every fusion event appends one new edge `(rep_i, rep_j)` to the target composite. Cross-composite fusion also preserves the absorbed composite's existing edges verbatim (particle IDs are stable).
5. **Ring closure is allowed.** A new scan phase, run after the existing fusion scan, finds pairs of same-composite members with per-particle free bonds ≥ 1 that have drifted within `fusion_radius`. Each ring closure adds one edge; no member-list changes.
6. **Fission rebuilds spanning trees.** Parent edges are dropped; each fission product builds a fresh path-spanning tree through its members in hash-sorted order. Rings re-form later via ring closure when geometry permits.
7. **A/B-able via config flag.** `bond_mode: str ∈ {"edges", "star_spring", "off"}` lets the existing star-spring stay available for comparison runs.

## Data structures

### `CompositeState` — new fields

```
edges:      (C, E_max, 2) int32   — particle-id pairs, -1 = unused slot
edge_count: (C,)         int32   — number of valid edges (≤ E_max)
```

The existing `free_bonds` field can stay as a derived/cached value or be removed; the canonical accounting becomes per-particle `degree`.

**Sizing.** `E_max = (max_composite_size · max_valence) // 2`. Defaults: M=128, max_valence=4 → E_max = 256. Memory: `C · E_max · 2 · 4 B = 3000 · 256 · 8 = ~6 MB`. Trivial.

**Invariant.** For any alive composite c:
- `edge_count[c] ∈ [max(0, member_count[c] − 1), E_max]` (≥ spanning tree, ≤ buffer cap).
- All valid `edges[c, e]` pairs reference particles with `composite_id[i] == c`.
- The graph induced by `edges[c]` is connected (single connected component over `members[c]`).

### `InteractionParams` — new field

```
r_rest: (S, S) float32   — hash-derived bond rest length per species pair
```

Symmetric: `r_rest[i, j] == r_rest[j, i]`. Initialized once at config time from the species-pair index via the same per-species hash machinery as `_hash_to_valence`:

```python
def _hash_to_rest_length(s_i, s_j, config):
    h = (_entity_hash_val(s_i, config) + _entity_hash_val(s_j, config)) % modulus
    h_mix = (h * 2_654_435_761) ^ (h >> 13)
    frac = (h_mix % 1000) / 999.0
    return r_rest_min + frac * (r_rest_max - r_rest_min)
```

Range: `r_rest ∈ [repulsion_radius · 1.5, fusion_radius · 0.9]` — comfortably outside the hard-core repulsion zone, comfortably inside the kernel's attractive region. With defaults that's roughly `[1.2, 3.6]`.

### Derived state computed each step

```
degree: (N,) int32   — count of edges incident to each particle
```

Computed by scatter-adding 1 at `edges[c, e, 0]` and 1 at `edges[c, e, 1]` for all valid edges, across all alive composites. O(C · E_max) scatter-adds per step. Used by the fusion gate and the ring-closure gate.

```
free_bond[i] = species_valences[species[i]] − degree[i]
```

For free particles, `degree[i] = 0` so `free_bond[i] = v_{s_i}`.

### `SimConfig` — new knobs

```python
# Bond model selection
bond_mode: str = "edges"            # "edges" | "star_spring" | "off"
k_bond: float = 20.0                # harmonic stiffness for edge bonds (replaces spring_k role)
r_rest_min: float = 1.2             # bounds for hash-derived rest lengths
r_rest_max: float = 3.6

# Ring closure
allow_ring_closure: bool = True
max_ring_closures_per_step: int = 50
```

The existing `spring_k`, `use_bond_forces`, etc. remain untouched so `bond_mode = "star_spring"` reproduces current behavior exactly.

## Bond force kernel

For each composite c, for each valid edge e = (i, j) in `edges[c]`:

```
d        = position[i] − position[j]    # min-image for periodic boundary
r        = |d|
r̂        = d / r
r_target = r_rest[species[i], species[j]]
F_ij     = −k_bond · (r − r_target) · r̂   # restoring force toward rest length
```

`F_ij` is applied to particle i; `−F_ij` to particle j. Scatter-added into the global (N, 2) force buffer before integration.

**Topology of the compute.** Same vmap pattern as the existing `compute_bond_forces`:

```python
all_pids, all_valid, all_forces = jax.vmap(bond_force_for_composite)(
    jnp.arange(C, dtype=jnp.int32)
)  # (C, 2·E_max), (C, 2·E_max), (C, 2·E_max, 2)
```

Each composite emits up to `2·E_max` (pid, valid, force) triples — two per edge (one per endpoint). Flatten and scatter into the (N, 2) force buffer.

**Cost.** O(C · E_max) per step ≈ 770K vmap-cells (current star-spring is ~384K). Per-element work is simpler (no COM reduction). Net: roughly same wall time as current, probably slightly cheaper.

## Where each piece plugs into the step

Updated phase list in `simulation_step`:

```
Phase 1: Build cell list                          [unchanged]
Phase 2: Find all neighbors                       [unchanged]
Phase 3: Pairwise forces (Particle Life)          [unchanged]
Phase 3b: Bond forces (edge harmonic springs)     [REPLACES current compute_bond_forces]
Phase 4: Integration                              [unchanged]
Phase 5: Boundary                                 [unchanged]
Phase 5b: Compute per-particle `degree` cache     [NEW — scatter-add over all edges]
Phase 6: Fusion (inter-entity)                    [modified — appends edges, uses per-particle degree]
Phase 6b: Ring closure (intra-composite fusion)   [NEW — uses per-particle degree updated by Phase 6]
Phase 7: Decay/fission                            [modified — rebuilds edges]
Phase 8: Energy accounting                        [unchanged]
Phase 9: Update ages and counters                 [unchanged]
```

`degree` is initialized in Phase 5b from the pre-fusion edge state, then incrementally updated inside the Phase 6 and Phase 6b scan bodies as edges are added (threaded through the scan carry). This avoids recomputing it twice per step.

## Fusion changes (Phase 6)

The existing rep-based scan is unchanged. Three things change in the scan body and the candidate gate.

### Gate change — per-particle valence

The current gate checks composite-level `free_bonds(c) ≥ 1`. Replace with per-particle:

```python
free_bond_i = species_valences[species[i]] - degree[i]
free_bond_j = species_valences[species[j]] - degree[j]
has_free_bonds = (free_bond_i >= 1) & (free_bond_j >= 1)
```

For free particles `degree[i] = 0`, so `free_bond_i = v_{s_i}` — matches existing behavior. For composite reps, this is stricter than the existing composite-level check: it requires the specific rep doing the fusion to have unused valence, not just the composite in aggregate.

### Edge append in `fusion_scan_body`

After determining the target slot, append one new edge `(rep_i, rep_j)` and copy any edges from an absorbed composite. Per the four merge cases:

| Case | New edges in target |
|---|---|
| free + free | `[(i, j)]` (new composite slot) |
| free + composite c_j | `edges[c_j] ⊕ [(i, j)]` (grow c_j; i is the free particle, j is the rep of c_j) |
| composite c_i + free | `edges[c_i] ⊕ [(i, j)]` (grow c_i; i is the rep of c_i, j is the free particle) |
| composite c_i + composite c_j | `edges[target] ⊕ edges[absorbed] ⊕ [(rep_i, rep_j)]`, where `target = min(c_i, c_j)` and `absorbed = max(c_i, c_j)` |

Implementation: concatenate the source edge buffer(s) + new edge into a `(2·E_max + 1, 2)` scratch, mask invalid (-1) slots, cumsum-compact to a (E_max, 2) buffer, overflow dropped via `mode='drop'`. Same compaction pattern the existing fusion uses for member lists.

The new edge connects `rep_i` and `rep_j` — the two particles the fusion gate verified are within `fusion_radius`. This is geometrically natural (they're physically close) but topologically arbitrary (rep is the lowest-index member, which may be buried in the middle of its composite). The harmonic spring self-corrects within a few steps. A future phase-2 enhancement would scan members on each side for the *closest pair* instead.

### Cost

Edge bookkeeping inside the scan body adds O(E_max) element-ops per scan iteration → ~50K extra ops per step. ~1 ms additional.

## Ring closure (Phase 6b)

New scan phase. Per-particle (not per-rep), runs after Phase 6.

### Candidate scan

For each particle i in parallel, vmap over its `max_neighbors` neighbors j. A candidate passes the gate iff:

```
j > i                                             (each unordered pair considered once)
composite_id[i] >= 0
composite_id[i] == composite_id[j]                (same composite — inverse of inter-fusion gate)
free_bond[i] >= 1 and free_bond[j] >= 1            (per-particle valence — uses live degree[])
dist(i, j) < fusion_radius
```

Each particle picks its **closest** qualifying neighbor (rather than highest binding energy — ring closure is geometric, not chemical). Returns at most one candidate partner per particle.

**Skip optimization (baked into v1).** Before the inner vmap, mask out any particle whose composite has `free_bonds(c) < 2`:

```python
comp_id = particles.composite_id
comp_free = composite_free_bonds[jnp.clip(comp_id, 0, C-1)]
can_attempt = (comp_id >= 0) & (comp_free >= 2)
```

Saturated composites (the majority in steady state) contribute zero work. `composite_free_bonds[c] = Σ free_bond[m]` over members m of composite c, equivalent to `Σ v_{species[m]} − 2 · edge_count[c]`. Computed once per step from `degree`, stored as a `(C,)` cache so the skip mask is a single gather.

### Conflict-resolved scan

Same pattern as the existing fusion scan body. Iterate up to `max_ring_closures_per_step` candidates in random shuffled order, with a `claimed` array tracking which particles have been used. The scan body for each candidate (i, j):

1. Recheck per-particle valence using the live `degree` (incremented as edges are added during the scan).
2. Recheck dedup against `edges[composite_id[i]]` — skip if `(i, j)` already exists.
3. Append `(i, j)` to `edges[composite_id[i]]`. Increment `edge_count[c]`. Increment `degree[i]` and `degree[j]`. Mark both as claimed.

No member list changes. No `composite_id` changes. No `alive` changes. The ring closure is *purely* an edge-and-counter update.

### Cost

- Candidate scan: O(N · max_neighbors) ≈ 1.3M cells per step. Structurally identical to existing `find_entity_partner` — same order as Phase 6.
- Skip optimization eliminates ~80–90 % of inner work for saturated composites.
- Scan body: O(max_ring_closures · E_max) ≈ 12K. Trivial.

Total Phase 6b: roughly +1 fusion-scan-equivalent worth of step time, minus the saturation skip → estimated net **+15–30 % step time** for this phase alone.

## Fission changes (Phase 7)

Existing logic for member partition by hash is unchanged. Two changes for edges.

### Drop parent edges

Both products discard the parent composite's edges entirely. No edge-inheritance, no edge-partition logic. This is the simplest rule and matches the "fission rearranges chemistry" interpretation — rings break, bonds reshuffle.

### Build fresh spanning trees per product

For each product with `count >= 2` members, build a path through hash-sorted members:

```python
# product members already compacted to front of (M,) buffer
# (existing fission code does this via cumsum)
edges_product[e] = (members_product[e], members_product[e + 1])  for e in [0, count − 1)
edge_count_product = count − 1
```

The members are sorted by `_hash_to_partition`'s sort_keys (Fibonacci-mixed slot index), so the path order is hash-deterministic.

This satisfies the invariant (connected, spanning, n−1 edges) and is O(count) per product to construct.

### Cost

O(C · M) scatter to build product edge lists. ~770K ops per step worst case, but only the fissioning composites contribute non-trivial work. ~0.5 ms additional.

## Renderer changes

The current renderer draws bonds visually via a forward-slot heuristic ([halflife/renderer.py:888-947](../../../halflife/renderer.py#L888-L947)) — for each composite member at slot i, it draws lines to the next K members in slot order. This is unrelated to any actual bond data and would not reflect the new sparse-edge topology.

Replace with edge-driven rendering: iterate over `edges[c, e]` for each alive composite, emit one GL_LINES segment per valid edge between the endpoint particles. Min-image wrap as in the current code.

Bond count cap, buffer sizing, color logic, alpha-by-trail handling all carry over essentially unchanged — the only difference is the *source* of (pid_a, pid_b) pairs.

## A/B comparison and rollout

`bond_mode` is added as a static `SimConfig` field with three values:

- `"edges"` — new sparse covalent bonds (default for v1 once stable).
- `"star_spring"` — current COM-spring behavior. `compute_bond_forces` stays in the codebase guarded by `bond_mode == "star_spring"`.
- `"off"` — no bond force at all. Useful as a baseline to see what the species-pair pairwise kernel can do on its own.

The flag is `static_argnums`, so XLA traces only the live branch — zero runtime cost from the unused modes. JAX caches compiled functions per static-arg hash, so toggling between modes recompiles once per mode per session, then is instant on subsequent toggles.

### Runtime toggle (keyboard shortcut)

Add `M` to the keybindings in [main.py](../../../halflife/main.py), cycling `bond_mode` through `"edges" → "star_spring" → "off" → "edges"`. The main loop holds `bond_mode` as Python state, rebuilds (or reuses cached) `run_n_steps`, and continues stepping with the existing `WorldState`. Simulation state (positions, velocities, members, edges) is preserved across toggles.

### Toggle-time edge initialization

Switching `star_spring → edges` with no pre-existing edges leaves every alive composite edgeless, so the harmonic-spring force on it is zero and composites dissolve under thermal noise within a handful of steps. To avoid this, the toggle handler runs a one-shot **edge-initialization pass** when entering `"edges"` mode:

For each alive composite c, build a path-spanning tree through `members[c]` in hash-sorted order — identical construction to the fission rebuild. The hash-sort gives a deterministic edge order; the path is `[(members[c, 0], members[c, 1]), (members[c, 1], members[c, 2]), ..., (members[c, n−1], members[c, n]))]` with `n = member_count[c] − 1`. Set `edge_count[c] = n − 1`. Recompute `degree` from the new edges.

Same pass runs at simulation startup if the initial `bond_mode` is `"edges"` and composites already exist (e.g., loaded from a saved state). Free particles are unaffected.

The reverse direction (`edges → star_spring` or `edges → off`) needs no special handling — the abandoned edge data just sits idle.

### Default and rollout

For initial development, `bond_mode = "star_spring"` stays the default. Once edges-mode produces stable simulations, flip the default to `"edges"` in a separate commit.

## Cost summary

| Item | Estimate | Notes |
|---|---|---|
| Bond force kernel | ~same as current | O(C · E_max) is ~2× the current loop *iterations*, but per-element work is simpler (no COM reduction). Net wall-clock roughly even, possibly slightly cheaper. |
| Per-step `degree` compute | < 1 ms | O(C · E_max) scatter-add |
| Edge bookkeeping in fusion scan | ~1 ms | O(max_fusions · E_max) ≈ 50K ops |
| Ring closure scan | +15–30 % step time | One additional fusion-scan-equivalent, mitigated by saturation skip |
| Fission edge rebuild | < 1 ms | O(C · M) scatter, mostly masked |
| **Total estimated delta** | **+30–60 % step time** | Dominated by ring closure scan |

If perf becomes the bottleneck, available levers:

- Drop `max_valence` from 4 → 3 (halves `E_max`).
- Cap `max_ring_closures_per_step` more aggressively.
- Skip dedup in scan body (allow accidental duplicate edges — equivalent to early multiplicity > 1).

## Out of scope (deferred to Phase 2)

- **Bond multiplicity (single / double / triple).** Edge struct could later become `(i, j, m)`. Force scales by m, rest length shrinks ~0.85× per order. Bond order at fusion time would be hash-determined.
- **Angle constraints.** Real double bonds are rigid against rotation. In 2D, this is an angle penalty between adjacent edges sharing a particle. Adds a second force term and ~50 LOC.
- **Closest-pair fusion bonding.** Instead of bonding the lowest-index reps of two fusing composites, bond the geometrically-closest member pair. Adds O(M_a · M_b) per fusion event.
- **Bond breaking under stress.** Currently bonds are unbreakable until fission. A Morse-style potential with a finite breaking force would let "overstretched" bonds snap, fragmenting composites mid-simulation. Interesting but adds bond-lifecycle bookkeeping.
- **Energy bookkeeping for bond PE.** Bond potential energy (`0.5 · k_bond · (r − r_rest)²` per edge) isn't currently tracked. Whether to include it in `compute_total_energy` is a separate question once the basic mechanics work.

## Open questions

1. **Should `free_bonds` field on `CompositeState` be removed entirely**, since the canonical accounting is now per-particle `degree`? Removing it is cleaner; keeping it as a cached `Σ free_bond[i]` over members lets ring-closure skip masks operate on a single (C,) array without a gather. Recommendation: keep it, recompute as a sum each step. ~Free.

2. **Is the "fission resets all rings" rule too aggressive?** Alternative: partition parent edges by which side each endpoint went to, drop only the crossing edges, then optionally re-link forest-shaped products with extra edges. More fission-history-preserving but more code. Recommendation: ship v1 with the simple "rebuild trees" rule, revisit if it visibly destroys interesting structures.

3. **Should `r_rest` be hash-derived or directly tunable per species pair from a config matrix?** Hash-derived keeps with the project's "hash determines chemistry" theme and gives different universes different bond chemistries. Directly tunable is more friendly to live-knob experimentation. Recommendation: hash-derived for v1, with an option to override via config later.

4. **Should the new edge on inter-composite fusion connect the lowest-index reps, or the geometrically-closest member pair?** v1 uses rep-to-rep (matches existing fusion gate, simple). If rendering shows visibly ugly initial fusion bonds (rep buried mid-composite), switch to closest-pair. Recommendation: rep-to-rep for v1.
