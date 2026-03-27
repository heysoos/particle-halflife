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

**Much cheaper than expected** due to GPU vectorization.

---

### Phase 4: compute_bond_forces (1.7 ms, 10.7%)

Spring forces pulling composite members toward center of mass.

**Why expensive:** O(C × M) = composites × members. With 177 composites and ~20 avg members = ~3500 force calculations per step.

**Tuning:**
- Disable completely: set `use_bond_forces = False` in config (~1.7ms savings)
- Trade-off: Composites no longer held together physically

---

### Phase 5: attempt_fusion (8.8 ms, 55.1%)

**THE BOTTLENECK.** Checks all neighbor pairs for potential fusion:
- Compute binding energy via hash
- Check if BE > `fusion_threshold`
- If yes, allocate composite slot

**Why expensive:** Examines ~512,000 neighbor pairs (2000 particles × 256 max_neighbors). Hash computation + state updates are the main cost.

**Tuning:**
- `max_neighbors` — Reduce from 256 → 128 cuts neighbor pairs by half (~4ms savings)
- `fusion_threshold` — Higher threshold (0.3) checks fewer pairs than lower (0.1)
- `interaction_radius` — Smaller radius reduces neighbor count

---

### Phase 6: apply_composite_decay (2.0 ms, 12.4%)

Probabilistic decay of composites. For each alive composite, roll dice on whether it decays.

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
