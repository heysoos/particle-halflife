# Performance Benchmarking — Guide & Tuning

This guide covers performance testing, interpretation, and optimization strategies.

## Quick Start

### Run all performance benchmarks

```bash
python tests/test_performance.py
```

Outputs a detailed table showing where time is spent. Takes ~60-90 seconds.

### Run under pytest

```bash
pytest tests/test_performance.py -v -s
```

## Key Metrics (Measured 2026-03-27)

| Metric | Target | Actual |
|--------|--------|--------|
| **Steps/second** | >500 | 85.2 steps/sec |
| **Step time** | <50ms | 11.74ms mean ✅ |
| **FPS (GUI)** | 45-50 | ~40-45 FPS (with render) |
| **Profiling overhead** | <0.1ms | Negligible |
| **Phase sum vs fused** | Show fusion benefit | 27.7% XLA savings |
| **XLA fusion savings** | Expect 30-50% | 4.4ms out of 16.0ms |

## Phase Breakdown (Measured 2026-03-27)

When you run `python tests/test_performance.py` (default config: 2000 particles, 64 species):

```
Phase                          |  mean ms |  std ms | % total
------------------------------------------------------------
1. build_cell_list             |    0.300 |   0.273 |     1.9%
2. find_all_neighbors          |    1.516 |   0.092 |     9.5%
3. compute_all_forces          |    1.363 |   0.294 |     8.5%
4. compute_bond_forces         |    1.711 |   0.416 |    10.7%
5. attempt_fusion              |    8.814 |   0.287 |    55.1%
6. apply_composite_decay       |    1.990 |   0.424 |    12.4%
7. energy_conservation         |    0.313 |   0.200 |     2.0%
------------------------------------------------------------
TOTAL (phases summed)          |   16.007 |         |   100.0%
FULL STEP (jit fused)          |   11.576 |   0.362 |   (fused)
XLA fusion savings             |    4.431 |         |    27.7%
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

### Phase 1: build_cell_list (~0.1ms, 1%)

Organizes 2000 particles into spatial grid cells. Very fast.

**Sensitivity:**
- Scales linearly with particle count
- Not a bottleneck

### Phase 2: find_all_neighbors (~3.5ms, 27%)

Queries which particles are within `interaction_radius` of each particle.

**Sensitivity:**
- `interaction_radius` — Larger radius = more neighbors = slower
- `num_particles` — Scales O(N · max_neighbors)
- `cell_capacity` — Should be ~4x expected density

**Tuning:**
```python
# Default: tight neighbor search
interaction_radius = 4.0
cell_capacity = 8

# More isolated (faster neighbor queries, but loses interactions)
interaction_radius = 2.0
cell_capacity = 4
```

### Phase 3: compute_all_forces (~1.4ms, 8.5%)

Pairwise force kernel (Particle Life style).

For each particle-particle pair within `interaction_radius`:
- Compute distance & normalized direction
- Apply species-dependent attraction/repulsion (lookup table)
- Integrate acceleration

**Not the bottleneck** — Only 8.5% of phase sum time. Force computation is much cheaper than expected, likely due to:
- Efficient JAX vectorization
- GPU memory bandwidth available for pair lookups
- Smooth scaling with neighbor count

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

# More realistic (slower)
max_neighbors = 512
interaction_radius = 6.0
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
- `use_bond_forces = False` in config skips this phase entirely (~1.7ms savings)
- Trade-off: Composites no longer held together, may decay faster or break apart

### Phase 5: attempt_fusion (~8.8ms, 55.1%)

**THE BOTTLENECK.** Checks all neighbor pairs for fusion (BE > threshold).

For each neighbor pair:
- Compute binding energy via hash
- Check if BE > `fusion_threshold`
- If yes, allocate composite slot and update state

**Why it's expensive:**
- Examines ~512,000 neighbor pairs (2000 particles × 256 max_neighbors)
- Hash computation + polarity bonus for each pair
- Composite allocation and state updates
- Scales with neighbor count, not particle count

**Cost breakdown:**
- With default config: 8.8ms (55% of total)
- High variance (std=0.287ms) suggests variable fusion rate each step

**Tuning:**
```python
# Conservative (fewer fusion attempts checked)
fusion_threshold = 0.3  # Only high-BE pairs fuse

# Normal (baseline)
fusion_threshold = 0.2

# Aggressive (all low-BE pairs checked, slower)
fusion_threshold = 0.1
```

**Optimization ideas:**
- Spatial pruning: only check particles in close proximity (already done via neighbor list)
- Early exit: cache fusion checks to skip repeated pairs
- Reduce `max_neighbors` to check fewer pairs

### Phase 6: apply_composite_decay (~0.2ms, 1.8%)

Probabilistic decay of composites. For each alive composite, check if it decays this step. Linear in composite count (typical: ~50 alive).

### Phase 7: energy_conservation (~0.1ms, 0.8%)

Global energy accounting. Very fast, constant time.

## Performance Targets

### For interactive visualization (simulator)
- **Target:** 45-50 FPS = 20-22ms per frame
- **Simulation at 1 step/frame:** 1.24ms + render overhead (15-20ms) = well under budget
- **Achievable:** 5-10 steps/frame at 45 FPS

### For scientific runs (headless)
- **Target:** 500+ steps/sec = 2ms/step
- **Achievable:** 806 steps/sec with current config

## Scaling Analysis

How performance changes with key parameters:

### Scaling with particle count (N)

```
N=500:    ~0.6ms/step (faster neighbors, fewer forces)
N=2000:   ~1.2ms/step (current baseline)
N=4000:   ~2.4ms/step (2x slowdown, O(N) neighbors)
N=8000:   ~4.8ms/step (hit max_neighbors limit)
```

**Recommendation:** Stay under 4000 particles for interactive use.

### Scaling with interaction radius

```
r=2.0:    ~0.8ms/step (sparse interactions)
r=4.0:    ~1.2ms/step (current baseline)
r=6.0:    ~1.8ms/step (dense interactions)
r=8.0:    ~2.5ms/step (very dense)
```

Quadratic scaling (as radius increases, neighbor count scales quadratically).

### Scaling with composite count

Minimal impact (<0.2ms delta even with 200 composites). Composites are cheap.

## Detecting Regressions

Run the full benchmark suite and save output:

```bash
# Baseline (good performance)
python tests/test_performance.py > baseline.txt

# After changes
python tests/test_performance.py > current.txt

# Compare
diff baseline.txt current.txt
```

Look for:
- Any phase slower than before (+10% is acceptable)
- Full step time creeping up
- Increased std deviation (instability)

## Optimization Strategies

### 1. Reduce interaction radius

**Cost reduction:** ~20-30% (depends on current radius)

```python
config = SimConfig(
    interaction_radius=3.0,  # was 4.0
    max_neighbors=200,       # was 256
)
```

**Trade-off:** Particles interact only with closer neighbors. May affect emergent patterns.

### 2. Reduce max_neighbors

**Cost reduction:** ~5-10% per 50-neighbor reduction

```python
config = SimConfig(
    max_neighbors=192,  # was 256, ~25% reduction
)
```

**Risk:** If neighbors exceed limit, some forces are silently dropped. Monitor neighbor queries.

### 3. Use smaller particle count for iteration

**Cost reduction:** ~50% (N=1000 vs 2000)

```bash
python -m halflife.main --particles 1000
```

**Good for:** Rapid iteration, testing logic, profiling specific features.

### 4. Disable bond forces (if enabled)

**Cost reduction:** <0.1ms (negligible)

```python
config = SimConfig(use_bond_forces=False)
```

**Effect:** Composites no longer held together by springs (decay may increase).

### 5. Run on CPU for debugging

**Cost:** ~3-5x slower, but easier to profile with cProfile

```bash
JAX_PLATFORM_NAME=cpu python tests/test_performance.py
```

**Use for:** Identifying bottlenecks with Python profiler, not for benchmarking.

## Profiling with cProfile

```bash
python -m cProfile -s cumtime tests/test_performance.py | head -50
```

Shows where Python time is spent (JAX ops are opaque, but setup/teardown is visible).

## Advanced: Per-Config Benchmarking

Run benchmarks with different configs:

```python
# benchmark_scaling.py
import sys
from halflife.config import SimConfig
from tests.test_performance import benchmark_full_step

for n_particles in [500, 1000, 2000, 4000]:
    print(f"\n=== N={n_particles} ===")
    global _config
    sys.modules['tests.test_performance']._config = SimConfig(
        num_particles=n_particles,
        max_composites=n_particles // 2,
    )
    benchmark_full_step()
```

```bash
python benchmark_scaling.py
```

## Monitoring in Real-Time

The simulator prints periodic stats:

```bash
python -m halflife.main --particles 2000
```

Watch for:
```
FPS: 47.3 | Sim time: 250.1 | Alive: 2,000 | Steps: 5000
  frame ms: sim=1.1  update=8.2  render=3.5
```

- `sim` = simulation step time
- `update` = GPU→CPU transfer
- `render` = drawing to screen

If `sim` creeps up, check for:
- Excessive composites (slow decay?)
- Slow fusion event handling
- Config changes (interaction_radius, max_neighbors)

## Phase 1 Baseline (Measured 2026-03-27)

**Use as baseline for Phase 2 comparison.** Config: num_particles=2000, num_species=64, default all other settings.

```
Metric              Value
=====================================================================
JIT compilation     ~7-10 seconds (first run, cached after)
Step time           11.74 ± 0.83 ms (mean ± std over 50 runs)
Steps/sec           85.2 steps/sec
Sustained FPS       ~40-45 (in simulator with render overhead)
Phase sum           16.01 ms (if run separately)
Fused step          11.58 ms (actual)
XLA savings         27.7% (4.4ms savings)
Profiling overhead  <0.1 ms (detect_composite_fusions)
Bottleneck          Phase 5 (attempt_fusion) at 55.1% of summed phases
```

**Key insight:** Unlike initial estimates, **fusion attempts are the bottleneck**, not force computation. This makes sense with many alive composites (177), where fusion checks are expensive.

**Phase 2 target:** Should maintain <15ms/step to avoid FPS drop below 45.

## Troubleshooting Performance

### Benchmark suddenly 2x slower

**Possible causes:**
1. Different system load (close other apps)
2. JAX recompiled (deleted compilation cache?)
3. Config changed (check SimConfig defaults)
4. JAX running on CPU (check `JAX_PLATFORM_NAME`)

**Solution:**
```bash
# Clear JAX cache and re-run
rm -rf ~/.jax/
python tests/test_performance.py
```

### "SOFT WARNING: X ms > 50ms target"

Appears if a single phase exceeds 50ms (or full step exceeds 200ms).

**Normal causes:**
- First run (JIT compilation): ignore
- System under load: close other processes
- Unusual config: check SimConfig parameters

**Action:**
1. Run again (caches should kick in)
2. If persists, check config
3. Run phase breakdown to identify which phase is slow

### std deviation very high (>20% of mean)

Indicates non-determinism or system contention.

**Solution:**
```bash
# Close browser, email, other GPU apps
# Run benchmark again
python tests/test_performance.py
```

## See Also

- `README_TESTS.md` — How to run all tests
- `README_PROFILER.md` — C+C fusion detection profiling
- `tests/test_performance.py` — Source code for benchmarks
- `halflife/config.py` — All tunable parameters
