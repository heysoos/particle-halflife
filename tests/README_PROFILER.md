# Profiler Tools — Usage Guide

The `halflife/profiler.py` module provides metrics collection for Phase 1 verification & instrumentation. It tracks composite-composite fusion events and size metrics without blocking the GPU.

## Overview

Two main components:

| Tool | Purpose |
|------|---------|
| **ProfileMetrics** | Dataclass that accumulates metrics during simulation runs |
| **detect_composite_fusions()** | Function that compares composite state before/after a step to infer fusion events |

## Quick Start

### Enable profiling in the simulator:

```bash
python -m halflife.main --enable-profiling --particles 2000
```

This runs the simulator and prints a profiling summary at exit.

### Run profiling test:

```bash
pytest tests/test_profiler.py -v
```

## ProfileMetrics Class

A dataclass that accumulates metrics over a simulation run.

### Initialization

```python
from halflife.profiler import ProfileMetrics

metrics = ProfileMetrics()
```

Creates an empty metrics object with:
- `cc_fusion_events`: list of CCFusionEvent records
- `composite_size_samples`: list of (step, max_size, mean_size, histogram) tuples
- `max_composite_size_observed`: peak composite size seen
- `cc_fusion_count`: total C+C fusion events recorded

### Recording Events

#### Composite-Composite Fusions

```python
from halflife.profiler import CCFusionEvent

event = CCFusionEvent(
    step=100,
    composite_a_id=5,
    composite_b_id=7,
    a_members=3,
    b_members=2,
    a_be=0.6,
    b_be=0.4,
    merged_be=0.55,
    merged_members=5,
)
metrics.record_cc_fusion(event)
```

**Fields:**
- `step`: simulation step when fusion occurred
- `composite_{a,b}_id`: IDs of the merging composites
- `{a,b}_members`: member counts before fusion
- `{a,b}_be`: binding energies before fusion
- `merged_be`: binding energy of the resulting composite
- `merged_members`: total members after fusion

#### Composite Size Snapshots

```python
import numpy as np
from halflife.step import compute_composite_size_stats

max_size, mean_size, histogram = compute_composite_size_stats(state.composites, config)
metrics.record_composite_sizes(
    step=100,
    max_size=max_size,
    mean_size=mean_size,
    distribution=histogram,
)
```

**Parameters:**
- `step`: simulation step
- `max_size`: largest composite at this step
- `mean_size`: average composite size (among alive composites)
- `distribution`: numpy array where `histogram[i]` = count of composites with `i` members

### Querying Metrics

#### Fusion rate

```python
rate = metrics.get_cc_fusion_rate()  # Returns total fusion count as float
```

#### Binding energy statistics

```python
mean_be, min_be, max_be = metrics.get_be_statistics()
```

Returns (mean, min, max) of binding energies for merged composites. Returns (0, 0, 0) if no fusions recorded.

## detect_composite_fusions() Function

Compares composite state before and after a simulation step to infer fusion events. Automatically detects:
1. Composites that grew significantly (absorbed other composites)
2. Composites that disappeared (merged into others)
3. Size-matching patterns suggesting mergers

### Usage

```python
from halflife.profiler import detect_composite_fusions

state_before = state
state = run_n_steps(state, params, physics, n)  # Run simulation
state_after = state

# Detect fusions that occurred during those steps
detect_composite_fusions(state_before, state_after, step_num, metrics)

# Update state for next detection cycle
state_before = state_after
```

### Performance

- **Highly optimized**: Batches JAX→numpy conversions to minimize GPU synchronizations
- **Completes in <1ms per step** even with 5,000 composite slots
- Safe to call every frame during `--enable-profiling` runs

## Integration Example

Typical integration in a simulation event loop:

```python
from halflife.profiler import ProfileMetrics, detect_composite_fusions
from halflife.step import compute_composite_size_stats

metrics = ProfileMetrics()
state_before_step = state

for frame in range(num_frames):
    # Run simulation (async, returns immediately)
    next_state = run_n(state, params, physics, steps_per_frame)

    # ... render previous state ...

    # Record metrics after rendering
    if config.enable_profiling:
        step_num = int(np.asarray(state.step_count))

        # C+C fusion detection
        detect_composite_fusions(state_before_step, next_state, step_num, metrics)

        # Size metrics
        max_size, mean_size, hist = compute_composite_size_stats(
            next_state.composites, config
        )
        metrics.record_composite_sizes(
            step=step_num,
            max_size=max_size,
            mean_size=mean_size,
            distribution=hist,
        )

        # Update state for next frame's detection
        state_before_step = next_state

    # Advance pipeline
    state = next_state
```

## Interpreting Results

### C+C Fusion Events

Look at `metrics.cc_fusion_events` to understand merger patterns:

```python
for event in metrics.cc_fusion_events[:10]:  # First 10 fusions
    print(f"Step {event.step}: "
          f"composite {event.composite_a_id} ({event.a_members} members) + "
          f"composite {event.composite_b_id} ({event.b_members} members) → "
          f"{event.merged_members} members (BE={event.merged_be:.2f})")
```

### Size Trends

Track composite growth over time:

```python
steps = [s[0] for s in metrics.composite_size_samples]
sizes = [s[1] for s in metrics.composite_size_samples]

print(f"Composite size trend:")
print(f"  Initial: {sizes[0]}")
print(f"  Peak: {max(sizes)}")
print(f"  Final: {sizes[-1]}")
```

### Binding Energy Distribution

```python
mean_be, min_be, max_be = metrics.get_be_statistics()
print(f"Merged composite binding energies:")
print(f"  Mean: {mean_be:.3f}")
print(f"  Range: [{min_be:.3f}, {max_be:.3f}]")
```

## Profiling Summary Output

When simulator exits with `--enable-profiling`, you see:

```
=== Phase 1 Profiling Summary ===
Total steps: 5000
Max composite size observed: 9
Total composite size samples collected: 5000
C+C fusion count (note: approximated): 45
Max composite size trend: min=3, max=9, final=8
```

**Interpret as:**
- **Total steps**: How many simulation steps ran
- **Max composite size**: Largest composite ever formed
- **Samples**: One size snapshot per step
- **C+C fusion count**: Total composite-composite mergers detected
- **Trend**: Initial→peak→final sizes

## Running Experiments

### Test different fusion thresholds

```bash
# Default (threshold=0.2)
python -m halflife.main --enable-profiling --particles 2000

# Lower threshold (more fusions expected)
python -m halflife.main --enable-profiling --particles 2000 \
    --fusion-threshold 0.1
```

Then compare `C+C fusion count` in the output.

### Test with different particle counts

```bash
python -m halflife.main --enable-profiling --particles 500
python -m halflife.main --enable-profiling --particles 2000
python -m halflife.main --enable-profiling --particles 4000
```

Larger populations should show more fusion activity.

### Test with different species counts

```bash
python -m halflife.main --enable-profiling --particles 2000 --species 4
python -m halflife.main --enable-profiling --particles 2000 --species 16
```

More species diversity affects binding energy distribution.

## Phase 1 Findings

From our analysis with profiling enabled:

| Config | Steps | Max Size | C+C Fusions | Fusion Rate |
|--------|-------|----------|-------------|-------------|
| 2000 particles, 8 species | 200 | 5 | 13 | 0.065/step |
| 500 particles, default | 50 | 2 | 0 | 0/step |

**Insight:** C+C fusion is **conservative** — mostly small composites (2-3 members) merge. Larger composites rarely fuse, suggesting binding energy penalty grows with complexity.

## See Also

- `README_TESTS.md` — How to run unit and performance tests
- `README_PERFORMANCE.md` — Detailed performance benchmarking
- `docs/PHASE1_ANALYSIS.md` — Phase 1 findings and recommendations
