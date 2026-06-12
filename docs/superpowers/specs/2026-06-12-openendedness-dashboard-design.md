# Open-Endedness & Temporal-Evolution Dashboard — Design

**Date:** 2026-06-12
**Status:** Design, pending review
**Author:** Heysoos + Claude (brainstorming session)

## Problem

The recent performance work (spatial/chemistry kernels, 17.9 → ~2 ms/step at
num_species=3) makes much longer runs cheap. The open-endedness research goal —
*does the simulation keep producing novelty and change over extended time, or
does it freeze?* — now needs measurement, not just longer wall-clock. The
existing composite diagnostic report (`halflife/analysis/`, Tiers 1–4) describes
the **state** of the chemistry at sampled points but has no metric for **novelty
accumulation over time** and no **time-windowed comparison** to see how the
system changes across a run.

This adds a new report section — **Tier 5: Open-Endedness & Temporal Evolution**
— plus time-window slicing, and runs it on a 15k-step simulation as the first
concrete deliverable.

## Goals

- Quantify whether novelty is still accumulating late in a run (vs. plateauing).
- Compare composite statistics across **time windows** of one run, so the user
  can read "how things change over time" from a single report.
- Measure novelty along **two independent axes**:
  - **Composition-type** — the species multiset (existing `species_hash`).
  - **Structure-type** — the bond topology (chain vs. ring vs. branched of the
    same atoms), via a canonical graph hash.
- Pure **host-side post-processing** on the cached `RunResult` — no JIT kernel
  changes, no new emitted data, works with `--from-cache` (re-windowing a cached
  run re-renders instantly).
- Deliver a rendered 15k-step report and a written window-by-window comparison.

## Non-goals (v1)

- **No Bedau–Packard evolutionary activity** (the neutral-shadow-run statistic).
  Considered and deferred — it needs a second baseline simulation; revisit in v2
  if the v1 curves warrant it.
- **No per-step novelty timing from the event stream.** The flat `events` array
  in `RunResult` is sentinel-filtered and loses step alignment, so all
  time-resolved curves are resolved at **snapshot cadence** (`sample_every`).
  150 snapshots over 15k steps is ample. A runner change to step-tag events is a
  possible future refinement, explicitly out of scope here.
- **No new kernel instrumentation.** Everything reads data the pipeline already
  captures (`species_hash`, `members`/`edges`/`edge_count` snapshots,
  `size_histogram`, and the complete event stream for one headline scalar).
- **No live/interactive dashboard.** Static HTML, same as the existing report.

---

## Architecture

```
halflife/analysis/
├── openendedness.py   ← NEW: type-identity + 5 metric functions + windowing
├── runner.py          (unchanged — RunResult already carries everything needed)
├── metrics.py         (unchanged — reuse size_histogram)
├── plots.py           ← add Tier 5 plot builders
├── report.py          ← add Tier 5 HTML section
├── cli.py             ← add --windows / --window-width flags
└── ...
```

`openendedness.py` is the one new unit. It is a pure library: every function
takes a `RunResult` (or arrays pulled from one) and returns numpy arrays /
dicts. No JAX, no I/O, no plotting — so it is independently testable and the
report layer just consumes its outputs.

### Data sources (what each metric reads from `RunResult`)

| Need | Source | Cadence |
|---|---|---|
| Composition-type id | `snapshot.species_hash[alive]` | snapshot |
| Structure-type id | WL hash over `snapshot.members/edges/edge_count[alive]` | snapshot |
| Currently-alive type sets | per snapshot | snapshot |
| Per-window size hist | `per_step_metrics['size_histogram']` | per-step |
| Headline total composition-types | `events.product_hashes` (complete stream) | run-total scalar |

---

## Components

### 1. Type identity

```
composition_type_ids(snapshot) -> np.ndarray[uint32]   # = species_hash[alive]
structure_type_ids(snapshot, config) -> np.ndarray[uint64]
```

**Structure-type** uses a self-contained **Weisfeiler–Lehman (WL) hash** over
each alive composite's labeled bond graph:

- Nodes = member particles, initial label = species index.
- Edges = the composite's `edges[:edge_count]` pairs.
- Run *k* WL refinement rounds (k = 3): each round replaces a node's label with
  a hash of `(its label, sorted multiset of neighbor labels)`. The graph hash is
  a hash of the sorted final node-label multiset.
- Isomorphic graphs (same topology + same species labeling) → same hash;
  a 6-chain and a 6-ring of identical atoms → different hashes.

Rationale: no networkx dependency (not installed), ~25 lines, cheap for the
small graphs here, collision-rare. WL is a well-known graph-isomorphism
approximation; exact canonical labeling is unnecessary at this scale. **Known
limitation:** WL can (rarely) collide on non-isomorphic regular graphs; accepted
for v1 and noted in the report.

A free particle (composite of one member, no edges) is a degenerate structure
type keyed by its species — consistent across both axes.

### 2. Windowing

```
slice_windows(n_steps, windows=None, window_width=None) -> list[(start, end)]
```

- `windows=N` → N equal ranges (last absorbs remainder).
- `window_width=W` → ceil(n_steps / W) ranges of width W.
- Both `None` → default 5 windows. Both set → error (mutually exclusive).
- Pure function of `n_steps`; snapshots are assigned to a window by their
  `snapshot.step`. Re-windowing needs no re-simulation.

### 3. Metric functions

All return plain arrays/dicts; all operate per type-axis (composition,
structure) so the report can show them side by side.

1. `discovery_curve(snapshots, axis) -> (steps, cumulative_distinct)`
   Cumulative count of distinct type ids ever seen up to each snapshot.
   Plus `total_composition_types_from_events(events) -> int` headline scalar
   (true total incl. ephemeral types born/died between snapshots).
2. `novelty_rate(snapshots, windows, axis) -> per_window_new_counts`
   Types whose **first** snapshot-appearance falls in each window.
3. `hill_diversity(snapshots, axis) -> {q0, q1, q2}` per snapshot
   Effective number of *currently-alive* types: richness (q=0),
   exp(Shannon) (q=1), inverse-Simpson (q=2). **Abundance of a type = the
   number of alive composites of that type** (each composite is one
   "individual"); q=1/q=2 weight by that abundance distribution.
4. `window_turnover(snapshots, windows, axis) -> {jaccard, bray_curtis}`
   Square dissimilarity matrices between windows' aggregated type sets
   (Jaccard on presence/absence; Bray–Curtis on per-window summed abundance,
   same abundance = composite-count definition as Hill diversity).
5. `per_window_size_hist(per_step_metrics, windows) -> per_window_hist`
   Mean `size_histogram` within each window.

### 4. Report integration

New `Tier 5` section appended in `report.py`, built from `plots.py` builders,
using the existing 2-column CSS grid and matplotlib→base64-PNG pattern:

- 5a Discovery curves (composition + structure overlaid) + headline total.
- 5b Novelty rate per window (grouped bars, two axes).
- 5c Hill-diversity time series (q0/q1/q2), composition and structure panels.
- 5d Turnover heatmaps (Jaccard + Bray–Curtis), two axes.
- 5e Per-window size-distribution facets (overlaid lines).

### 5. CLI

`cli.py` gains `--windows N` and `--window-width W` (mutually exclusive),
threaded into `render_html`. Because windowing is post-processing, they apply
under `--from-cache` with no re-simulation.

---

## Data flow

```
run_diagnostic (existing)            openendedness.py (new)        report.py
─────────────────────────            ──────────────────────        ─────────
RunResult.snapshots ───────┬───────► structure_type_ids ──┐
RunResult.snapshots ───────┼───────► composition_type_ids ─┼──► discovery_curve ──┐
                           │                                │──► hill_diversity ───┼─► Tier 5
slice_windows(n_steps) ────┴───────► (window ranges) ──────┼──► novelty_rate ─────┤   plots
RunResult.per_step_metrics ────────► per_window_size_hist ──┘──► window_turnover ──┘
RunResult.events.product_hashes ───► total_composition_types (headline scalar)
```

## Error handling

- Empty run / no alive composites at a snapshot → diversity = 0, type set = ∅;
  curves render flat without dividing by zero.
- `windows` and `window_width` both set → explicit `ValueError` at CLI parse.
- Window with zero snapshots (very small `windows` × large `sample_every`
  mismatch) → empty type set, turnover row/col = NaN rendered as a neutral
  cell; logged once.
- WL hash on a composite whose `edge_count=0` but `member_count>1` (valence-off
  or star_spring runs where edges are inert) → falls back to the **composition**
  multiset as the structure id, and the report notes structure-axis metrics are
  only meaningful in `bond_mode="edges"`.

## Testing

Extend the analysis pytest suite (CPU, fast):

- **WL hash**: isomorphic relabelings → equal; chain vs. ring of identical atoms
  → unequal; permutation-invariance of member order; single-member degenerate.
- **Windowing**: even/uneven division, `windows=1`, `window_width >= n_steps`,
  mutual-exclusion error.
- **Metrics on a synthetic RunResult**: hand-built snapshots where the distinct
  type count, diversity, and turnover are known by construction; assert exact
  values and array shapes.

## Deliverable run

```
.venv/bin/python -m halflife.analysis --scenario current_experiment \
  --steps 15000 --sample-every 100 --platform gpu
```

Produces `tests/reports/diag_current_experiment_<ts>.html` with Tier 5, plus a
written window-by-window comparison of the composite statistics read back to the
user. (Run only when the GPU is free of the user's live sim; force CPU otherwise.)

## Open questions

None blocking. WL collision behavior and the snapshot-cadence limitation are
accepted v1 trade-offs, documented above and surfaced in the report notes.
