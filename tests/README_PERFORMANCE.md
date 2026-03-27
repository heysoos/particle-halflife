# Performance Benchmarking — Guide

Measure where simulation time is spent to identify bottlenecks and track regressions.

## Quick Start

```bash
python tests/test_performance.py
```
or
```bash
pytest tests/test_performance.py -v -s 
```
Outputs detailed phase breakdown. Takes ~90 seconds.

## Key Metrics (Measured 2026-03-27)

| Metric | Value |
|--------|-------|
| **Step time** | 11.74 ± 0.83 ms |
| **Steps/sec** | 85.2 |
| **Full step (fused)** | 11.58 ms |
| **Phase sum (separate)** | 16.01 ms |
| **XLA fusion savings** | 27.7% (4.4 ms) |

## Phase Breakdown (Measured 2026-03-27)

Default config: 2000 particles, 64 species, 177 composites alive

```
Phase                          |  mean ms |  std ms | % total
------------------------------------------------------------
1. build_cell_list             |    0.300 |   0.273 |     1.9%
2. find_all_neighbors          |    1.516 |   0.092 |     9.5%
3. compute_all_forces          |    1.363 |   0.294 |     8.5%
4. compute_bond_forces         |    1.711 |   0.416 |    10.7%
5. attempt_fusion              |    8.814 |   0.287 |    55.1%  ← BOTTLENECK
6. apply_composite_decay       |    1.990 |   0.424 |    12.4%
7. energy_conservation         |    0.313 |   0.200 |     2.0%
------------------------------------------------------------
TOTAL (phases summed)          |   16.007 |         |   100.0%
FULL STEP (jit fused)          |   11.576 |   0.362 |   (fused)
```

**Interpretation:**

- **Phases 1-7 summed = 16.0ms** — Time if each phase ran separately
- **Full step fused = 11.6ms** — Actual time with XLA kernel fusion
- **Fusion savings = 4.4ms** — JAX's JIT compiler saves 27.7% through kernel fusion
- **Attempt fusion = 55.1%** — THE BOTTLENECK (phase 5, with many alive composites)
- **Composite decay = 12.4%** — Expensive with 177 alive composites
- **Bond forces = 10.7%** — Spring forces between members cost significant time
- **Force computation = 8.5%** — Much cheaper than expected (but still important)

## Understanding Per-Phase Costs

### Phase 1: build_cell_list (0.3 ms, 1.9%)

Organizes particles into spatial grid cells. Very fast, not a bottleneck.

**Sensitivity:** Linear with particle count. Not tunable.

---

### Phase 2: find_all_neighbors (1.5 ms, 9.5%)

Queries which particles are within `interaction_radius` of each particle.

**Sensitivity:**
- `interaction_radius` — Larger radius = more neighbors = more expensive
- `cell_capacity` — Should be ~4x expected density to avoid overflow
- `num_particles` — Scales O(N · max_neighbors)

**Tuning:** Reducing `interaction_radius` from 4.0 → 2.5 reduces neighbor count significantly.

---

### Phase 3: compute_all_forces (1.4 ms, 8.5%)

Pairwise force kernel (Particle Life style). For each particle-neighbor pair:
- Compute distance and direction
- Apply species-dependent attraction/repulsion
- Accumulate acceleration

**Sensitivity:**
- `max_neighbors` — Directly scales cost (256 → 512 = 2x slower)
- `interaction_radius` — Affects neighbor count
- `num_particles` — O(N · neighbors) scaling

**Tuning:**
```python
# Current (balanced)
max_neighbors = 256
interaction_radius = 4.0

# Faster (sparse interactions)
max_neighbors = 128
interaction_radius = 2.5
```

### Phase 4: compute_bond_forces (~1.7ms, 10.7%)

**SIGNIFICANT COST.** Spring forces pulling composite members toward center of mass.

For each alive composite with M members:
- Compute center of mass
- For each member, apply force toward COM

**Why it's expensive:**
- O(C · M) = composites × members
- 177 composites × ~20 average members = ~3,500 force calculations
- Each calculation: dot product, normalize, spring constant multiply

**Tuning:**
- Disable completely: set `use_bond_forces = False` in config (~1.7ms savings)
- Trade-off: Composites no longer held together physically

---

### Phase 5: attempt_fusion (8.8 ms, 55.1%)

**THE BOTTLENECK.** Checks all neighbor pairs for potential fusion:
- Compute binding energy via hash
- Check if BE > `fusion_threshold`
- If yes, allocate composite slot

**Why it's expensive:**
- Examines ~512,000 neighbor pairs (2000 particles × 256 max_neighbors)
- Hash computation + polarity bonus for each pair
- Composite allocation and state updates
- Scales with neighbor count, not particle count

**Cost breakdown:**
- With default config: 8.8ms (55% of total)
- High variance (std=0.287ms) suggests variable fusion rate each step

**Tuning:**
- `max_neighbors` — Reduce from 256 → 128 cuts neighbor pairs by half (~4ms savings)
- `fusion_threshold` — Higher threshold (0.3) checks fewer pairs than lower (0.1)
- `interaction_radius` — Smaller radius reduces neighbor count

**Optimization ideas:**
- Spatial pruning: only check particles in close proximity (already done via neighbor list)
- Early exit: cache fusion checks to skip repeated pairs
- Reduce `max_neighbors` to check fewer pairs
---

### Phase 6: apply_composite_decay (2.0 ms, 12.4%)

**Sensitivity:** Linear with number of alive composites (currently 177).

---

### Phase 7: energy_conservation (0.3 ms, 2.0%)

Global energy accounting. Negligible cost.

## Performance Targets

### For interactive simulator (GUI)
- **Target:** 45-50 FPS
- **Current:** 40-45 FPS sustained (11.74ms step + ~8-10ms render)
- **Status:** Close to target, mostly limited by fusion bottleneck

### For scientific headless runs
- **Current:** 85 steps/sec (11.74ms/step)
- **Headroom:** 15ms/step before hitting 60 Hz limit

## Phase 1 Baseline (2026-03-27)

Use as reference for Phase 2 comparison:
- Step time: **11.74 ms**
- Main bottleneck: **Phase 5 (fusion) at 55.1%**
- Secondary cost: **Phase 6 (decay) at 12.4%**

Phase 2 should aim to maintain <15ms/step.

## See Also

- `README_TESTS.md` — How to run all tests
- `README_PROFILER.md` — C+C fusion detection profiling
- `tests/test_performance.py` — Benchmark source code
