# Composite Diagnostic Pipeline — Design

**Date:** 2026-05-30
**Status:** Design, pending review
**Author:** Heysoos + Claude (brainstorming session)

## Problem

After the recent valence + sparse-bond work, the sim no longer trivially produces
large composites the way it used to. We need a tool that runs a single long
simulation and produces a human-readable diagnostic report — enough detail to
answer "*why* aren't composites growing?" without having to re-run the live app
and squint at sprites.

## Goals

- One CLI command → one HTML file → human reads it → decides what to tweak.
- Visibility into **valence saturation** as a candidate bottleneck (since that's
  the most likely cause of the regression).
- Visibility into the **chemical network** — which composite types react with
  which, in both directions (fusion, fission) — via heatmap matrices sorted by
  atomic weight (member count).
- **Zero runtime cost** in the live app: the diagnostic instrumentation must be
  gated by a static-arg flag so the live kernel is bit-for-bit unchanged.

## Non-goals (v1)

- **No kernel counters** for failed-fusion reasons (BE-too-low, valence-saturated,
  slot-full). State-only metrics first; if those don't explain the regression we
  add this in a phase 2.
- **No parameter sweeps.** That's what [`tests/test_composite_statistics.py`](../../../tests/test_composite_statistics.py)
  is for; not duplicating.
- **No side-by-side scenario comparison view** inside one report. Compare by
  running the tool twice and opening two HTMLs.
- **No live update / animation.** Static HTML output only.

---

## Architecture

### Module layout

```
halflife/analysis/
├── __init__.py
├── runner.py        — headless sim runner; lax.scan + periodic full-snapshot host-copy
├── events.py        — ReactionEvents NamedTuple, event-array helpers
├── transitions.py   — event log → composite-transition matrices (size-binned, top-K, full)
├── compatibility.py — pure-chemistry fusion compatibility matrices (Tier 4)
├── metrics.py       — pure per-step metric functions (size, free_bonds, degree, edges)
├── plots.py         — matplotlib helpers; each returns base64 PNG
├── report.py        — HTML template assembly
└── cli.py           — argparse + orchestration; `python -m halflife.analysis <args>`
```

### Data flow

```
CLI args (scenario, steps, seed, sample-every, out-path)
   ↓
Runner:
   Build SimConfig from named scenario (with emit_events=True).
   lax.scan over num_steps:
     - run simulation_step → (new_state, events_for_this_step)
     - compute per-step metrics inside scan:
         size_hist, fb_hist, degree_hist, max_size, alive_count,
         edge_count_total, ring_count_total
     - accumulate events into scan output (fixed-size padded arrays)
   Every K steps: host-copy a full snapshot of {alive, member_count, species_hash,
     members, edges} for distribution drill-downs.
   ↓
Transitions:
   Filter event stream where kind != 0.
   Per-fusion event: matrix[source_hash_a, product_hash] += 1
                     matrix[source_hash_b, product_hash] += 1
   Per-fission event: matrix[source_hash, product_hash_a] += 1
                      matrix[source_hash, product_hash_b] += 1
   Produce three matrices:
     - size_bin_matrix: (max_composite_size × max_composite_size)
     - top_k_matrix:    (K × K), K most-trafficked hashes, tail bucketed
     - full_matrix:     (U × U), every observed unique hash, sorted by size
   ↓
Compatibility (post-process, pure chemistry — no simulation):
   For every (composite type i, composite type j) pair:
     merged_hash = (H_i + H_j) % modulus
     be_merged   = _hash_to_binding_energy(merged_hash, physics)
     passes_be   = be_merged >= fusion_threshold * binding_energy_scale
     passes_val  = max_free_bonds(M_i) >= 1 AND max_free_bonds(M_j) >= 1
   Produce two matrices:
     - species_pair_compat: (S × S) — all species pairs, always-on
     - observed_pair_compat: (K × K) — same top-K hashes as top_k_matrix above
   ↓
Report:
   header (scenario name, config dump, seed, duration, git SHA)
   tier 1 — macroscopic time series (max size, alive count, free-particle fraction)
   tier 2 — valence / edge structure (free_bonds histogram, degree saturation, rings)
   tier 3 — chemical network matrices (three of them, empirical)
   tier 4 — fusion compatibility matrices (two of them, theoretical)
   footer (run timing, sim version)
```

---

## Event log (load-bearing)

This is the core mechanism that makes the chemical-network matrices accurate.
Without it, attribution is inferred from snapshot diffs and breaks down when
multiple events collide on the same step.

### New SimConfig flag

```python
emit_events: bool = False
```

Default `False`. SimConfig is `static_argnums` in all JIT'd functions, so this
flag is a **compile-time switch** — when `False`, the entire emission code path
is dead-code-eliminated before XLA compilation. Live app: unchanged.

### ReactionEvents structure

```python
class ReactionEvents(NamedTuple):
    kind:           jnp.ndarray  # (E_max,) int32     — 0=none, 1=fusion, 2=fission
    source_slots:   jnp.ndarray  # (E_max, 2) int32   — fusion: both filled; fission: (slot, -1)
    source_hashes:  jnp.ndarray  # (E_max, 2) uint32
    source_sizes:   jnp.ndarray  # (E_max, 2) int32
    product_slots:  jnp.ndarray  # (E_max, 2) int32   — fusion: (slot, -1); fission: both filled
    product_hashes: jnp.ndarray  # (E_max, 2) uint32
    product_sizes:  jnp.ndarray  # (E_max, 2) int32
```

`E_max` per step is sized per-kernel:
- Fusion event slots: `config.max_fusions_per_step` (currently 200)
- Fission event slots: `config.max_composites` (no per-step cap in the kernel
  today; every alive composite can decay each step in principle, though in
  practice it's a handful). May be tightened in implementation if profiling
  shows the worst-case allocation is wasteful.

Sentinel rows (`kind == 0`) are filtered post-scan.

### Kernel changes (minimal, output-only)

`attempt_fusion` and `apply_composite_decay` already know everything we need at
the moment they perform the reaction — they're just throwing it away.

```python
def attempt_fusion(state, params, config, physics):
    ...  # existing fusion logic, unchanged
    new_state = state._replace(...)
    if config.emit_events:
        events = _build_fusion_events(scan_outputs)
        return new_state, events
    return new_state, None
```

Same pattern in `apply_composite_decay`. `simulation_step` concatenates the two
event streams when `emit_events=True`, returns a single padded array per step.

**Verified zero cost when off:** the `if config.emit_events:` branch is evaluated
at trace time. The non-emission branch traces a kernel identical to today's.

### Shatter events (fission products that fail the valence check)

A fissioning composite that produces a structurally-unsound product (free_bonds < 0)
shatters into free particles. These are recorded as fission events with
`product_size == 1` and `product_hash == 0` for the shattered side. Lets us
count "how often does fission lose mass to shattering" as a metric.

---

## Metrics inventory (v1)

### Tier 1 — macroscopic (per step, cheap, always emitted)

| Metric | Shape | Source |
|---|---|---|
| `max_size` | scalar | `composites.member_count[alive].max()` |
| `mean_size` | scalar | `composites.member_count[alive].mean()` |
| `median_size` | scalar | sorted alive sizes, middle |
| `alive_count` | scalar | `composites.alive.sum()` |
| `free_particle_fraction` | scalar | `1 - composites.member_count[alive].sum() / N` |
| `size_histogram` | `(max_composite_size,)` | bincount of alive member_counts |

### Tier 2 — valence / edge structure (per step, cheap)

| Metric | Shape | Source |
|---|---|---|
| `free_bonds_histogram` | `(max_valence × max_composite_size,)` | bincount of `composites.free_bonds[alive]` |
| `degree_histogram` | `(max_valence + 1,)` | bincount of `compute_degree(composites)` per particle |
| `degree_saturation_pct` | scalar | `mean(degree[i] == v_{species[i]})` over all particles |
| `edge_count_total` | scalar | `composites.edge_count[alive].sum()` |
| `ring_count_total` | scalar | `sum(edge_count[c] - (member_count[c] - 1) for c alive)` |

`degree_saturation_pct` is the headline metric for "is valence the bottleneck?":
if it's >80%, valence is the gate.

### Tier 3 — chemical network (event-driven, post-processed)

Built from the event log. Three matrices, all sorted by atomic weight (member
count) ascending, all log-color scaled, all using the same red colormap so the
eye can compare.

**Matrix 1: size-binned (always renders, small, readable)**

- Shape: `(max_composite_size × max_composite_size)`
- Rows: source size class
- Cols: product size class
- Cell value: count of events where a source of that size produced a product of
  that size
- Upper-triangular density = fusion-dominated; lower = fission-dominated
- Always in the report

**Matrix 2: top-K composite-type (configurable, K defaults to 30)**

- Shape: `(K+1 × K+1)` — K most-trafficked species hashes + "other" row/col
- Rows/cols: composite types identified by `species_hash`
- Sort: by member count ascending, then by hash for stable ordering
- "Trafficked" = total events incident (sum of row + col before truncation)
- Tail bucketed into "other" row/col so totals match

**Matrix 3: full unique-composite (exploratory, may be cut after one report)**

- Shape: `(U × U)` where U = unique species hashes ever observed
- Same sort as Matrix 2
- Rendered as a single PNG with log-color scale
- Embedded in HTML with `overflow:scroll` so a huge matrix is browsable
- Explicitly flagged as "decide whether to keep after first use"

### Tier 4 — fusion compatibility (chemistry, not dynamics)

Pure post-processing of the multiset chemistry. No simulation involvement —
runs in milliseconds after the run finishes. Answers "what *could* happen"
versus Tier 3's "what *did* happen." Diffing the two visually is the core
diagnostic move for the regression question.

For every pair (i, j) of composite types under consideration:

```python
merged_hash = (H_i + H_j) % config.hash_modulus
be_merged   = _hash_to_binding_energy(merged_hash, physics)
passes_be   = be_merged >= physics.fusion_threshold * physics.binding_energy_scale
passes_val  = max_free_bonds(M_i) >= 1 AND max_free_bonds(M_j) >= 1
```

where `max_free_bonds(M) = Σ v_{s_i in M} − 2·(n − 1)` is the structural
upper bound on a fresh n-member composite of multiset M.

**Matrix 4a: species-pair compatibility (always-on)**

- Shape: `(S × S)`
- Rows/cols: free particles, indexed by species
- Cell color: merged BE for the pair (continuous, viridis-style colormap)
- Greyed out: cells where `passes_be == False`
- Hatched overlay: cells where `passes_val == False` (free particles of v=1
  paired with each other, etc.)
- Always rendered — small, fast, useful as a config-level "what does this
  universe even support?" chart even before any sim has run

**Matrix 4b: observed-composite compatibility (top-K)**

- Shape: `(K × K)` — same K-most-trafficked composite types as Matrix 2 in
  Tier 3, same sort order (size ascending, hash for tiebreak)
- Same cell encoding as 4a
- Designed for direct visual diff against Matrix 2: cells that are bright in
  4b but cold in Matrix 2 are pairs that *could* react but never did →
  either kinetic (they never met) or valence-saturated in practice (the
  *typical* free_bonds was below the structural max because of pre-existing
  edges)

The diff between 4a/4b and Tier 3 is the diagnostic gold:

| Tier 3 (empirical) | Tier 4 (compatibility) | Diagnosis |
|---|---|---|
| Few large→larger transitions | Lots of high-BE cells in top-right | Chemistry is fine — kinetics/diffusion bottleneck |
| Few large→larger transitions | Few high-BE cells in top-right | Chemistry is the bottleneck — most products fail BE |
| Few large→larger transitions | High-BE pairs hatched (valence-blocked) | **Valence is killing growth** (the regression hypothesis) |

---

## CLI surface

```bash
# Built-in scenario presets:
python -m halflife.analysis --scenario baseline --steps 10000 --seed 0
python -m halflife.analysis --scenario valence_off --steps 10000 --seed 0
python -m halflife.analysis --scenario polymer_world --steps 10000 --seed 0

# Override individual knobs:
python -m halflife.analysis --scenario baseline \
    --override "num_species=5,half_life_max=50" \
    --steps 10000 --seed 0

# Output:
# tests/reports/diag_<scenario>_<timestamp>.html (default)
# or --out path/to/custom.html
```

### Presets (initial set)

| Name | Differences from default `SimConfig` |
|---|---|
| `baseline` | (default config, no overrides) |
| `current_experiment` | `num_species=3, half_life_max=100` (user's current uncommitted experiment) |
| `valence_off` | `use_valence=False` |
| `polymer_world` | `max_valence=2, num_species=2` |
| `branching_world` | `max_valence=3, num_species=3` |
| `old_star_spring` | `bond_mode="star_spring"` (pre-edges-default behavior) |

### Flags

| Flag | Default | Purpose |
|---|---|---|
| `--scenario <name>` | `baseline` | Named preset |
| `--steps N` | `10000` | Total sim steps |
| `--seed S` | `0` | RNG seed |
| `--sample-every K` | `100` | Full snapshot every Kth step (compact metrics every step) |
| `--top-k N` | `30` | Number of unique composite types in Matrix 2 |
| `--full-matrix / --no-full-matrix` | `--full-matrix` | Whether to render Matrix 3 |
| `--override "k=v,k=v"` | none | Per-run SimConfig overrides |
| `--out PATH` | `tests/reports/diag_<scenario>_<ts>.html` | Output path |
| `--platform cpu\|gpu` | (auto) | Force JAX platform |

---

## HTML report layout

Single self-contained file. All plots as base64 PNGs (matplotlib + Agg backend),
no external assets. Layout top-to-bottom:

1. **Header** — scenario name, config dump (key params highlighted), seed, total
   steps, sample interval, run duration, git SHA from `git rev-parse HEAD`.
2. **At-a-glance stat cards** (4-up grid) — peak max size, final max size, mean
   alive count, fusion/fission event rates, degree saturation %.
3. **Tier 1: Macroscopic time series** — 2×2 grid: max size over time, alive
   count over time, size-distribution heatmap (size × time), free-particle
   fraction over time.
4. **Tier 2: Valence / edge structure** — 2×2 grid: free_bonds distribution
   heatmap (free_bonds × time), degree histogram heatmap (degree × time), degree
   saturation % over time, ring count over time.
5. **Tier 3: Chemical network (empirical)** — three matrices stacked vertically
   (size-binned, top-K, full-unique). Each has its own caption. Matrix 3
   wrapped in a `<div style="overflow:scroll;max-width:100%;max-height:800px">`.
6. **Tier 4: Fusion compatibility (theoretical)** — two matrices stacked: 4a
   species-pair, 4b top-K observed-composite. Matrix 4b uses the same K and
   sort order as Tier 3 / Matrix 2 so the eye can diff them by scrolling
   back and forth.
7. **Footer** — JAX device used, JIT compile time, total wall time, sim version.

---

## Memory budget

| Stream | Per-step size | 10k-step total |
|---|---|---|
| Compact per-step metrics | ~512 bytes | ~5 MB |
| Event log (E_max=200, ~50 bytes/event) | ~10 KB | ~100 MB |
| Full snapshots (K=100, C=1000) | ~650 KB per snap | ~65 MB |
| **Total** | — | **~170 MB** |

Comfortable on CPU for small-scale testing (will use C=200 or so), comfortable
on GPU for large runs. If a run is so big this is tight, the user can bump
`--sample-every` (cuts snapshot memory) or the analysis itself can stream
snapshots to disk rather than keep them in RAM (deferred to phase 2).

---

## Open questions / decisions made along the way

- **Matrix 3 retention** — explicit ask: "I want to see it to decide whether to
  keep it." Will revisit after first report.
- **Fusion-failure breakdown** — deferred to phase 2 (kernel counters). State-only
  metrics first.
- **Comparison view** — deliberately not built. User compares by running the
  tool twice and reading two HTMLs.

## What success looks like

Run the tool on `current_experiment` and `valence_off` separately. Open both
HTMLs side-by-side. If the regression is valence-driven, the answer should be
obvious from one of three places:

- **Degree saturation % over time (Tier 2)** — should be visibly higher in
  `current_experiment`.
- **Size-binned matrix (Tier 3 / Matrix 1)** — should show fewer transitions
  into high-size cells in `current_experiment`.
- **Tier 3 / Matrix 2 vs Tier 4 / Matrix 4b** — the same K composite types
  appear in both, with the same sort. Cells that are bright-and-compatible
  in 4b but cold in Matrix 2 are pairs that *could* fuse but never did. If
  most cold cells are hatched (valence-blocked) in 4b, that's the
  smoking-gun pattern.

If those plots don't make the answer obvious, the design has failed and we
need kernel counters in phase 2.
