# Performance Optimization Plan — 2026-06-12

Goal: maximize long-run simulation throughput (steps/sec) so open-endedness experiments
can run far longer in wall-clock time. All numbers measured on the RTX 3080 Laptop GPU
(8 GB), jax 0.6.2 + CUDA 12, default `SimConfig` (N=5000, S=12, C=3000, M=128,
E_max=256, max_neighbors=256), warmed state (~320 alive composites), `bond_mode="edges"`.
Benchmark scripts were throwaways in `/tmp/` (`bench_baseline.py`, `bench_ablation.py`,
`bench_event_rates.py`, `bench_neighbors.py`, `bench_fusion_split.py`,
`bench_capacity.py`); no base code was modified.

## Where the time goes (measured)

Baseline: **17.9 ms/step amortized** inside a `lax.scan` of 100 steps (≈ 56 steps/s).

Ablation (scan-100, phases disabled one at a time):

| Component | ms/step | share |
|---|---|---|
| `attempt_fusion` (Phase 6) | **11.3** | 63% |
| `apply_composite_decay` (Phase 7) | 2.1 | 12% |
| `attempt_ring_closure` (Phase 6b) | 1.9 | 11% |
| Physics (cell list + neighbors + forces + edge bonds + integrate + energy) | 3.7 | 21% |

The chemistry conflict-resolution scans dominate. Their cost is almost purely the
number of *sequential scan iterations* (~45 µs/iteration of tiny kernels — the GPU is
launch-bound, not compute-bound):

- `attempt_fusion` cap sweep (standalone): cap=200 → 13.0 ms, cap=64 → 8.5 ms,
  cap=16 → 6.1 ms, cap=1 → 4.1 ms (≈ pre-scan floor: partner finding + dedup).
- Actual event rates (measured with `emit_events=True`, 800 steps): steady state
  **4.7 fusions/step** (p99 = 15, max = 17) and **3.1 fissions/step**. Only the first
  ~50 condensation steps burst higher (max 186). The 200-iteration budget is ~97%
  no-op iterations in steady state.

## Tier 1 — config-only quick wins (no code changes)

**1. Lower the chemistry scan budgets.**
`max_fusions_per_step: 200 → 64` and `max_ring_closures_per_step: 50 → 16`:

- Measured end-to-end: **17.9 → 8.3 ms/step (2.2×)** (with the Tier-2 neighbor fix
  included; caps alone account for ~7.7 ms of that).
- Caps 32/8 give 6.4 ms/step (2.8×) if a slightly slower initial condensation is
  acceptable.
- Dynamics impact: caps only bind when more candidates exist than budget; excess
  candidates simply retry next step (the partner search reruns every step). In steady
  state p99 = 15 fusions/step, so cap=64 is conservative. During the initial transient
  the condensation spreads over a few more steps — same end state.
- Note: the comment on `max_fusions_per_step` ("= num_particles") is stale; it has been
  200 for a while.

## Tier 2 — drop-in code optimizations (identical results, faster)

**2. Vectorize neighbor packing in `find_neighbors_for_particle`** (`spatial.py`).
The current `pack_slot` does, per particle, a full max-reduction over all 576
candidates *for each of the 256 output slots* — O(max_neighbors × 9·cell_capacity)
per particle ≈ 737M comparisons/step. Replace with one row-wise `argsort` of validity
keys + `take_along_axis` (order-preserving compaction). Also replaces the
`lax.scan` over the 9 cell offsets with a batched gather.

- Measured: 5.6 → **1.8 ms** standalone; −1.7 ms/step fused.
- Verified **bit-identical neighbor sets** on a warmed state (0/5000 mismatches).

**3. Fix the stale benchmark harness.** `tests/test_performance.py` still calls
`attempt_fusion` with its pre-degree single-return signature, so the fusion benchmarks
crash. Worth fixing so regressions are visible; consider adding the scan-100 amortized
step time as the headline number (single-call timings have ±2-6 ms dispatch noise on
this machine, scan-amortized numbers are stable).

## Tier 3 — structural redesigns (the big wins)

**4. Parallel fusion: replace the sequential conflict-resolution scan with
mutual-best matching + batched apply.**
The scan exists to serialize conflicting candidates (`claimed` bookkeeping). But the
result is just a greedy matching on entities; the same guarantee can be computed in
parallel:

- After the existing per-entity dedup (Step 2.5 already does a scatter-max
  handshake!), keep a pair only if the two entities chose *each other* (mutual-best),
  or run 2 rounds of handshake to approach maximal matching.
- Compact winners to a fixed budget K (cumsum+sort, as today) and apply all fusions
  in one `vmap` over K: merged member lists, merged edge lists, property writes,
  `composite_id` scatter. No two winners share an entity, so all writes are disjoint.
- Prototype of the batch-apply (K=32, full member/edge merge + scatters):
  **0.5 ms** vs ~9 ms for the 200-iteration scan body. Fusion phase floor becomes
  pre-scan (~2 ms fused) + 0.5 ms.
- Dynamics: mutual-best matches a subset of what sequential greedy matches in one
  step; losers retry next step — same flavor of delay the cap already introduces.
  Per-step fusion *counts* should be validated with the existing
  `halflife.analysis` diagnostic (Tier 3 transition matrices, baseline vs new).

**5. Same treatment for ring closure.** Identical structure (scan exists only for
`claimed`/degree bookkeeping); mutual-best pairs + batched edge append. 1.9 → ~0.3 ms.

**6. Compact fission work in `apply_composite_decay`.** The heavy per-composite work
(`_hash_to_partition` argsort over M=128, COM computation, the (C × M) per-member
grid, per-product spanning trees) runs for **all 3000 composite slots** every step,
but only ~3 fissions/step actually fire (≤ 14 observed, even in transient). Compact
fissioning slot indices to a fixed budget K (e.g. 64, same find-free-slots sort
machinery), run the heavy math on (K, M) arrays, scatter results back. Est. 2.1 →
~0.5 ms. Identical dynamics up to an overflow guard that defers excess fissions one
step (never observed beyond 14).

Projected end state, Tiers 1+2+3: physics ~2.0 ms + fusion ~2.5 ms + ring ~0.3 ms +
decay ~0.5 ms ≈ **5-6 ms/step ≈ 170-200 steps/s (3-3.5×)**; with caps=32/8 closer to
4.5 ms. Tier 3 also makes per-iteration cost independent of `max_fusions_per_step`,
removing the throughput-vs-burst-capacity tradeoff entirely.

## Correctness findings surfaced by benchmarking (decide before/alongside perf work)

These affect the science, not just speed — flagging rather than fixing.

**A. Cell-list overflow is real and growing.** `did_overflow=True` from early on;
max cell occupancy reaches **372 vs cell_capacity=64** by step 2000 (clusters keep
densifying). Particles beyond slot 64 of a cell are *silently invisible* to all
neighbor queries (forces, fusion, ring closure) — and the survivors are biased by
cell-insertion order. `max_neighbors=256` also truncates (observed 225 *with*
overflow already dropping candidates). Options:
- `cell_capacity=192`: honest neighborhoods, measured cost +2.5 ms/step on the
  Tier-1 configuration (8.3 → 10.8 ms).
- Keep truncation but make it unbiased (e.g. distance-sorted or randomized packing) —
  cheaper, still lossy.
- Treat as physics: this regime has ~370 particles inside one 8×8 cell; if that
  density is undesirable, stronger hard-core repulsion would fix both physics and
  perf at once.

**B. int32 overflow in entity-hash recomputation.** `_compute_entity_hash` (and
`hash_for_product` in decay) sums up to 128 per-member hash values of ~1e8 in int32 —
overflows for composites larger than ~21 members, so the recomputed hash disagrees
with the incrementally-maintained `composites.species_hash` (measured: 204 particles
mismatched in a warmed state). Fusion BE decisions for large-composite mergers are
therefore computed from a differently-wrapped hash than the one stored on the
composite. Deterministic, but inconsistent chemistry. Fix options: reuse the stored
`species_hash` (verified equal counts; also removes an N×M gather), or accumulate in
uint64 / running-mod. Note: fixing changes large-composite chemistry.

**C. Composite-size buffer saturation.** p99 alive composite size = 128 =
`max_composite_size` — growth is being clipped by the JAX buffer, not by chemistry.
If open-endedness wants M=256, today that roughly doubles chemistry scan cost; after
Tier 3 the cost of larger M is far smaller (heavy work no longer × max_fusions
iterations). Do Tier 3 first if raising M.

## Non-issues (measured, no action needed)

- Renderer state transfer: 7.9 MB/frame via one `jax.device_get` — 0.05 ms. Fine.
- `max_neighbors=256`: justified (p99 actual neighbors = 208, max 225).
- `e_max=256`: max observed edge_count = 158. Fine.
- Force kernel / edge-bond forces: ~1.5 ms each standalone, well-vectorized.
- Entity-hash *cost* (vs correctness): 0.17 ms, immaterial.

## Implementation status (2026-06-12, same day)

Tiers 1 + 2 implemented and verified: **17.92 → 8.26 ms/step amortized (2.17×)**.
- Caps 64/16 in config.py. Dynamics check (matched seeds, emit_events, 800 steps):
  steady-state fusion/fission rates, bonded-particle counts, and max sizes
  statistically identical; transient clips at 64 but total condensation
  throughput unchanged; 2/800 steps saturate the new cap.
- Vectorized `find_all_neighbors`: exact output equality vs old implementation
  on uniform/clustered-overflow/reflect/boundary-pinned/warmed states.
- `test_performance.py` repaired (fusion tuple signature) and per-phase
  breakdown updated to mirror the live step (edges bonds, degree, ring closure).
- Test suites: spatial/step/hash 27 passed (CPU), chemistry+covalent 31 passed
  (GPU), analysis pipeline 22 passed (CPU).
- Pre-existing failure found (NOT from these changes; fails on clean main):
  `test_chemistry.py::test_valence_off_unchanged` — `max_valence` now affects
  dynamics even with `use_valence=False`, via `e_max = M·max_valence/2` (edge
  buffer size) and `attempt_ring_closure`, which applies valence gates
  unconditionally. Needs a semantics decision (should ring closure respect
  `use_valence`?).

## Tier 3 implementation status (2026-06-12, later same day)

All three Tier-3 redesigns implemented and verified. Final ladder at the
current experiment config (num_species=3, warmed 500-step state, scan-100):

| Pipeline | ms/step |
|---|---|
| Legacy chemistry (scan fusion+ring, full-C decay, caps 200/50) | 16.23 |
| + caps 64/16 + compacted decay (scan conflict resolution) | 6.26 |
| + matching fusion & ring closure (full new pipeline, default) | **2.56** |

≈ **390 steps/s**, ~6.3× over the legacy chemistry pipeline and ~7× over the
session-start baseline. Matching-mode cost is independent of
`max_fusions_per_step` (4.89 vs 4.95 ms at caps 64 vs 200, measured before
decay compaction landed).

- `fusion_mode: "matching" | "scan"` (static config, default "matching")
  gates BOTH fusion and ring closure. "scan" restores the full legacy
  conflict-resolution path for A/B comparison.
- Fission compaction: heavy per-fission work (partition argsort, COMs,
  per-member kick grid) runs over `min(max_fissions_per_step, C)` = 64
  compacted slots instead of all 3000. Excess fissions defer one step
  (never observed: p99=25, max 27 at num_species=3). Fission ReactionEvent
  batches are now budget-sized — two shape assertions in
  test_analysis_events.py updated to the new contract.
- Verification: full-pipeline invariant checker (membership consistency,
  species conservation, edge validity, free-bond cache, no duplicate
  membership) passes at every checkpoint over 800 steps; scan-vs-matching
  dynamics statistically equivalent (steady fusions 16.05 vs 15.37/step,
  fissions 14.91 vs 14.26, same population trajectories). Full test sweep:
  81 passed after fixing one test that hard-coded the scan path's arbitrary
  tie-break (ring-closure geometry made mutual-nearest-unambiguous; verified
  to pass under both modes).
- use_valence fix landed: ring closure now also gated on `use_valence`
  (mechanic is defined by free-bond accounting), `e_max` floored at M-1 so
  spanning trees always fit at max_valence=1. test_valence_off_unchanged
  passes.
- New finding (pre-existing, both modes): fission's path-spanning-tree
  rebuild ignores per-particle valence — a v=1 member placed mid-path gets
  degree 2 (observed: 19 over-valence composites at step 100 under scan,
  13 under matching). Product-level Σv−2(n−1) ≥ 0 can't catch it. Fix would
  be ordering v=1 members at path endpoints (feasible iff ≤2 such members,
  else shatter) — science decision pending.
- Tooling: persistent XLA compilation cache (~/.cache/halflife-jax) wired
  into tests/conftest.py, main.py, and the analysis CLI; pytest-xdist with
  `XLA_PYTHON_CLIENT_PREALLOCATE=false -n 4` cuts the chemistry suite
  278s → 129s (see CLAUDE.md "Fast test runs").

## Follow-up findings (2026-06-12, evening): GPU utilization + event bursts

### GPU utilization at N=10k

- During pure sim compute (scan-100) the GPU is **100% utilized** (nvidia-smi
  sampled). The ~35% seen in the live app is duty cycle: with
  `steps_per_frame=1` the GPU computes one step then idles through render/
  HUD/frame-clock. Raising steps per frame (`+` key, up to 64) runs a
  GPU-resident scan per frame and pushes utilization toward 100%.
- **Power state dominates absolute numbers on this laptop.** Mid-benchmark
  reading: 1365/2100 MHz at 17.8 W (power-capped). The same 5k config
  measured 2.56 ms/step in the morning and ~12 ms in the capped state —
  a ~5× swing. Relative comparisons within one session remain valid.
- New 10k hotspot found and fixed: the v2 argsort row-sort in neighbor
  packing cost 9.7 ms/step at 10k. Replaced with a prefix-sum + flat-scatter
  pack (v3) — **verified bit-identical output** against the ORIGINAL
  implementation on all five regression states; 2.3× faster than v2 at the
  neighbor build in same-process A/B (4.59 → 2.00 ms at 5k), full step at
  10k: 27.1 → 15.1 ms/step (same power state).

### Event-sprite "bursts" (user report) — diagnosis

It is a **rendering issue**, two-layered; the physics is not bursty:

1. Sim-side slot-flip rates have Fano factor ≈ 1.0 (Poisson, ~14 new +
   14 dead per step at 5k steady state) — no synchronization in the physics.
2. The renderer admission queue saturates and relaxes periodically
   (renderer.py:1283-1323): demand ≈ 28 sprites/frame × 50-frame lifetime
   ≈ 1400 ≫ the 200-sprite pool. Simulated with measured event series:
   admissions are ZERO for 76% of frames, then spike >20 when a whole
   cohort of sprites expires together — period ≈ 43 frames ≈ the sprite
   lifetime. That saturation-relaxation oscillation IS the visible burst.
3. Additionally the detector's semantics broke when binary hash-fission
   landed (it diffs comp_alive between frames; sprites predate the
   redesign): typical fission reuses the parent slot (no flip → INVISIBLE),
   fission product 1 claims a fresh slot (→ drawn GOLD as "fusion"), and
   comp+comp fusion kills the absorbed slot (→ drawn CYAN as "fission").
   Roughly half the gold rings are actually fissions and most real fissions
   never render. Related to the pending fission redesign discussion
   (2026-06-01 notes) — any multi-fragment scission would scramble this
   further.

Proposed fixes (not yet implemented):
- Quick: replace the saturating pool with a ring buffer (overwrite oldest;
  no zero-admission regime) and/or shorten sprite lifetime to match pool
  capacity (≈ pool / demand ≈ 7 frames at current rates).
- Right: source sprites from the kernel's real ReactionEvent batches
  (emit_events=True in the live app). Transfer is tiny
  (~128 events × 7 int fields/step), kinds/timing become exact, positions
  derivable from product slots, and it stays correct under the future
  fission redesign.

## Suggested order

1. Tier 1 caps (one-line config; 2.2× immediately) + fix `test_performance.py`.
2. Tier 2 neighbor packing (verified identical output).
3. Decide on findings A/B/C (science calls).
4. Tier 3: parallel fusion → decay compaction → ring closure, each validated with the
   `halflife.analysis` diagnostic against a baseline cache before merging.
