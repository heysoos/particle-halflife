# Tests — Running and Understanding

This project uses **pytest** for unit and integration tests. Tests cover hash chemistry, spatial indexing, step logic, simulation performance, and profiler functionality.

## Quick Start

### Run all tests

```bash
pytest tests/ -v
```

### Run a specific test file

```bash
pytest tests/test_hash.py -v
pytest tests/test_spatial.py -v
pytest tests/test_chemistry.py -v
pytest tests/test_profiler.py -v
```

### Run a specific test function

```bash
pytest tests/test_chemistry.py::test_fusion_threshold_filtering -v
```

## Test Files Overview

| File | Focus | Count |
|------|-------|-------|
| **test_hash.py** | Hash chemistry engine, species multiset hashing | 4 tests |
| **test_spatial.py** | Cell list & neighbor finding performance | 4 tests |
| **test_step.py** | Integration testing, full simulation steps | 3 tests |
| **test_chemistry.py** | Fusion, decay, binding energy logic | 8 tests |
| **test_profiler.py** | Metrics collection & C+C fusion tracking | 7 tests |
| **test_performance.py** | Phase-by-phase benchmarks, step timing | 7 benchmarks |

## Detailed Test Descriptions

### test_hash.py — Hash Chemistry

Tests the deterministic hash-based reaction rules.

```bash
pytest tests/test_hash.py -v
```

**Key tests:**
- `test_hash_function_deterministic` — Same species multiset always hashes to same value
- `test_hash_function_order_independent` — Order doesn't matter: [A,B] == [B,A]
- `test_hash_derived_properties` — Properties derived from hash (BE, half-life) are consistent
- `test_hash_collision_statistics` — Hash function distributes well across species combinations

**Purpose:** Verify that the hash-based chemistry is **deterministic** and **stable** — same input always produces same output.

### test_spatial.py — Spatial Indexing

Tests cell-list based neighbor finding.

```bash
pytest tests/test_spatial.py -v
```

**Key tests:**
- `test_cell_list_construction` — Particles assigned to correct cells
- `test_neighbor_finding_correctness` — All neighbors within `interaction_radius` are found
- `test_periodic_boundary_wrapping` — Periodic boundaries handled correctly (neighbors across edges)
- `test_spatial_performance` — Neighbor finding completes in <5ms for 2000 particles

**Purpose:** Ensure spatial indexing correctly finds neighbors without false positives/negatives.

### test_step.py — Simulation Steps

Tests full simulation step integration.

```bash
pytest tests/test_step.py -v
```

**Key tests:**
- `test_step_runs_without_error` — Basic step execution
- `test_step_conserves_particle_count` — Particle count stays fixed
- `test_step_updates_state` — State changes (positions, velocities) as expected

**Purpose:** Smoke test that simulation loop runs and produces sensible state changes.

### test_chemistry.py — Fusion & Decay

Tests composite formation, growth, and decay.

```bash
pytest tests/test_chemistry.py -v
```

**Key tests:**
- `test_fusion_threshold_filtering` — Only fusions with BE > threshold occur
- `test_fusion_creates_composite` — Free particles merge into composite
- `test_fusion_preserves_members` — Merged composite contains all members
- `test_decay_probabilistic` — Composites decay with expected frequency
- `test_polarity_affects_fusion` — Opposite-polarity pairs get fusion bonus
- `test_composite_half_life_derived_from_hash` — Half-life derived consistently from species hash
- `test_multiple_fusions_per_step` — Multiple fusions can occur in one step
- `test_fission_products_match_hash` — Fission products determined by species hash

**Purpose:** Verify fusion/decay mechanics work as designed.

### test_profiler.py — Metrics Collection

Tests profiling metrics accumulation and C+C fusion detection.

```bash
pytest tests/test_profiler.py -v
```

**Key tests:**
- `test_profile_metrics_init` — Empty ProfileMetrics initializes correctly
- `test_record_cc_fusion_event` — Fusion events recorded with correct fields
- `test_be_statistics` — Binding energy mean/min/max computed correctly
- `test_record_composite_sizes` — Size samples accumulated and max tracked
- `test_get_cc_fusion_rate` — Fusion count returned as float
- `test_be_statistics_single_event` — Edge case (1 event: mean=min=max)
- `test_be_statistics_empty` — Edge case (no events: returns 0,0,0)

**Purpose:** Verify metrics infrastructure works correctly before integration tests.

### test_performance.py — Benchmarks

Tests step timing and phase-by-phase breakdown. Can run standalone or under pytest.

```bash
# Run standalone (prints diagnostic table)
python tests/test_performance.py

# Run under pytest
pytest tests/test_performance.py -v -s
```

**Benchmarks:**
1. `benchmark_full_step` — Total time for one simulation step
   - Goal: <50ms, hard limit: 200ms

2. `benchmark_neighbor_finding` — Cell list + neighbor queries alone
   - Typical: ~3-4ms

3. `benchmark_fusion_only` — Fusion phase with densely packed particles
   - Typical: <1ms

4. `benchmark_bond_forces` — Spring forces between composite members
   - Typical: <0.1ms

5. `benchmark_compute_forces` — Pairwise force kernel (main simulation cost)
   - Typical: ~8-10ms

6. `benchmark_composite_decay` — Probabilistic composite decay
   - Typical: <0.5ms

7. `benchmark_per_phase_breakdown` — Detailed table showing all 7 phases
   - Shows where time is spent and XLA fusion savings

**Purpose:** Monitor performance across phases and catch regressions.

## Understanding Test Output

### Example pytest output

```
tests/test_hash.py::test_hash_function_deterministic PASSED        [ 12%]
tests/test_hash.py::test_hash_function_order_independent PASSED    [ 25%]
tests/test_spatial.py::test_cell_list_construction PASSED          [ 37%]
tests/test_spatial.py::test_neighbor_finding_correctness PASSED    [ 50%]
tests/test_chemistry.py::test_fusion_threshold_filtering PASSED    [ 62%]
tests/test_profiler.py::test_profile_metrics_init PASSED           [ 75%]
tests/test_step.py::test_step_runs_without_error PASSED            [ 87%]
======================== 20 passed in 3.21s ========================
```

All tests pass ✓

### Example performance output

```
=== BENCHMARKS ===
benchmark_full_step:
  mean=1.24ms  std=0.18ms  steps/sec=806.5
  PASS

benchmark_per_phase_breakdown  (particles=2000, composites=42):
  Phase                          |  mean ms |  std ms | % total
  ------------------------------------------------
  1. build_cell_list             |    0.127 |   0.018 |      1.0%
  2. find_all_neighbors          |    3.450 |   0.087 |     27.1%
  3. compute_all_forces          |    8.234 |   0.156 |     64.7%
  4. compute_bond_forces         |    0.015 |   0.003 |      0.1%
  5. attempt_fusion              |    0.582 |   0.041 |      4.6%
  6. apply_composite_decay       |    0.234 |   0.019 |      1.8%
  7. energy_conservation         |    0.102 |   0.008 |      0.8%
  ------------------------------------------------
  TOTAL (phases summed)          |   12.744 |         |    100.0%
  FULL STEP (jit fused)          |    1.235 |    0.09 |   (fused)
  XLA fusion savings             |   11.509 |         |     90.3%
```

Key insight: JAX's XLA fusion is saving 90% of the time by fusing all phases into one kernel. Force computation (phase 3) is the bottleneck at 64.7% of the summed phases.

## Running Tests in Different Contexts

### Unit tests only (fast)

```bash
pytest tests/ -k "not performance" -v
```

Excludes performance benchmarks. Runs in ~5 seconds.

### Performance tests only

```bash
pytest tests/test_performance.py -v -s
```

Runs all 7 benchmarks. Takes ~60-90 seconds depending on system.

### Specific phase benchmark

```bash
pytest tests/test_performance.py::test_benchmark_per_phase_breakdown -v -s
```

Just the detailed breakdown table.

### With warnings and output capture disabled

```bash
pytest tests/ -v -s --tb=short
```

`-v` = verbose, `-s` = show print output, `--tb=short` = condensed tracebacks

## Debugging a Test

### Run with full traceback

```bash
pytest tests/test_chemistry.py::test_fusion_creates_composite -vv --tb=long
```

### Run with print statements visible

```bash
pytest tests/test_chemistry.py::test_fusion_creates_composite -v -s
```

### Run with pytest debugger on failure

```bash
pytest tests/test_chemistry.py::test_fusion_creates_composite --pdb
```

Drops to Python debugger on test failure.

## Test Dependencies & Setup

All tests:
1. Import JAX (may trigger GPU initialization)
2. Import halflife modules
3. May initialize PRNG keys and random state
4. Can take 10-30s to run due to JAX warmup

### First run

First time you run tests, JAX will compile kernels. Subsequent runs reuse the compiled cache.

```bash
# First run (slow, includes JIT compilation)
pytest tests/ -v  # ~30-60s

# Subsequent runs (fast, uses compiled cache)
pytest tests/ -v  # ~5-10s
```

## Common Issues

### ModuleNotFoundError: No module named 'halflife'

**Solution:** Make sure you're in the project root:
```bash
cd /path/to/halflife-particle
pytest tests/
```

### JAX GPU memory error

**Solution:** JAX allocated too much GPU memory. Run fewer tests or on CPU:
```bash
JAX_PLATFORM_NAME=cpu pytest tests/test_performance.py
```

### Performance test times vary wildly

**Solution:** Other processes are contending for GPU. Close other applications and re-run.

### ImportError: No module named pytest

**Solution:** Install pytest:
```bash
pip install pytest
```

## Writing New Tests

### Test template

```python
# tests/test_myfeature.py

import pytest
from halflife.config import SimConfig
from halflife.state import initialize_world, initialize_interaction_params
from halflife.mymodule import my_function

def test_my_feature_basic():
    """Brief description of what is being tested."""
    config = SimConfig(num_particles=100)
    state = initialize_world(config, seed=42)
    params = initialize_interaction_params(config, seed=43)

    result = my_function(state, params, config)

    assert result is not None
    assert len(result) == expected_length

def test_my_feature_edge_case():
    """Test an edge case or boundary condition."""
    config = SimConfig(num_particles=1)
    state = initialize_world(config, seed=0)

    result = my_function(state, None, config)

    assert result.particles.position.shape == (1, 2)
```

### Run your new test

```bash
pytest tests/test_myfeature.py -v
```

## See Also

- `README_PERFORMANCE.md` — Detailed performance analysis and tuning
- `README_PROFILER.md` — Phase 1 profiling tools and metrics
- `tests/test_*.py` — Individual test files (well-commented)
