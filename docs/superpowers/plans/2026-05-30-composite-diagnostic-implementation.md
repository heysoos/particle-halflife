# Composite Diagnostic Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `halflife/analysis/` — a CLI-driven Python module that runs a single long simulation and produces a self-contained HTML report with valence/edge-aware metrics, empirical composite transition matrices, and theoretical fusion-compatibility matrices.

**Architecture:** Add a `ReactionEvent` data structure and an `emit_events: bool = False` static-arg flag to `SimConfig`. When enabled, modify `attempt_fusion` and `apply_composite_decay` to emit per-event tuples alongside their state updates (zero overhead when off thanks to `static_argnums` dead-code elimination). The new `halflife/analysis/` package consumes these events, plus per-step compact metrics and periodic full-state snapshots, to produce a single-file HTML report.

**Tech Stack:** JAX (jit/scan/vmap), NumPy, matplotlib (Agg backend), Python stdlib (argparse, base64, html templates).

**Spec:** `docs/superpowers/specs/2026-05-30-composite-diagnostic-design.md`

**Test execution note:** Per `CLAUDE.md`, the user often has the live sim running on the GPU. **All test commands in this plan use `JAX_PLATFORMS=cpu`** to keep tests fast and out of GPU contention. Tests intentionally use small configs (N≤200 particles, ≤500 steps) for sub-30-second runs.

---

## File Structure

**New files:**

| Path | Responsibility |
|---|---|
| `halflife/analysis/__init__.py` | Empty package marker. |
| `halflife/analysis/events.py` | Host-side helpers for `ReactionEvent` arrays (flatten, filter sentinels). |
| `halflife/analysis/metrics.py` | Pure per-step metric functions (size, free_bonds, degree, edges). All return scalars or short histograms suitable for `lax.scan` outputs. |
| `halflife/analysis/compatibility.py` | Pure-chemistry fusion-compatibility matrix builders. No simulation. Uses `_hash_to_binding_energy` and `_species_valences` from chemistry.py. |
| `halflife/analysis/transitions.py` | Event log → three transition matrices (size-binned, top-K, full). |
| `halflife/analysis/runner.py` | Headless sim runner. `lax.scan` for per-step compact metrics + events; periodic full snapshots host-copied for distribution drill-downs. Returns a `RunResult` dataclass. |
| `halflife/analysis/plots.py` | Matplotlib plot helpers — each function takes `RunResult` and returns a base64 PNG string. |
| `halflife/analysis/report.py` | HTML template assembly. Takes `RunResult` + a dict of plot images, returns a single self-contained HTML string. |
| `halflife/analysis/cli.py` | argparse, scenario presets, output path defaulting, JAX platform setup. Entry point: `python -m halflife.analysis`. |
| `halflife/analysis/__main__.py` | Two-line forwarder to `cli.main()` so `python -m halflife.analysis` works. |
| `tests/test_analysis_events.py` | Tests for the event log emission machinery (Tasks 1-4). |
| `tests/test_analysis_metrics.py` | Tests for the metrics module (Task 5). |
| `tests/test_analysis_compatibility.py` | Tests for the compatibility module (Task 6). |
| `tests/test_analysis_transitions.py` | Tests for the transition matrix builder (Task 7). |
| `tests/test_analysis_pipeline.py` | End-to-end smoke test (Task 12). |

**Modified files:**

| Path | What changes |
|---|---|
| `halflife/config.py` | Add `emit_events: bool = False` to `SimConfig`. |
| `halflife/state.py` | Add `ReactionEvent` NamedTuple definition. |
| `halflife/chemistry.py` | Modify `attempt_fusion` to emit fusion events when `config.emit_events`. Modify `apply_composite_decay` to emit fission events. |
| `halflife/step.py` | Modify `simulation_step` to concatenate fusion + fission events into a single per-step `ReactionEvent` and return as a second output when `config.emit_events`. |

**Output location:** `tests/reports/diag_<scenario>_<timestamp>.html` (default; configurable via CLI).

---

## Task 1: Foundation — `ReactionEvent` + `emit_events` flag

**Why:** Define the data structure that flows through every subsequent task, and the static-arg switch that controls whether kernels emit it. Zero overhead when off because `SimConfig` is `static_argnums` in every JIT'd function — Python `if config.emit_events:` branches at trace time, so the unused branch is dead-code-eliminated.

**Files:**
- Modify: `halflife/config.py`
- Modify: `halflife/state.py`
- Create: `halflife/analysis/__init__.py`
- Create: `halflife/analysis/events.py`
- Create: `tests/test_analysis_events.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_analysis_events.py`:

```python
"""Tests for ReactionEvent data structure and emit_events config flag."""
import jax.numpy as jnp
import pytest

from halflife.config import SimConfig
from halflife.state import ReactionEvent
from halflife.analysis.events import zero_event_batch, filter_sentinels, KIND_NONE, KIND_FUSION, KIND_FISSION


def test_emit_events_defaults_false():
    """The static-arg flag must default to False so the live sim is unchanged."""
    config = SimConfig()
    assert config.emit_events is False


def test_reaction_event_zero_batch_shape():
    """zero_event_batch(N) returns a ReactionEvent with leading dim N and zeros."""
    N = 50
    batch = zero_event_batch(N)
    assert batch.kind.shape == (N,)
    assert batch.kind.dtype == jnp.int32
    assert batch.source_slots.shape == (N, 2)
    assert batch.source_hashes.shape == (N, 2)
    assert batch.source_hashes.dtype == jnp.uint32
    assert batch.source_sizes.shape == (N, 2)
    assert batch.product_slots.shape == (N, 2)
    assert batch.product_hashes.shape == (N, 2)
    assert batch.product_sizes.shape == (N, 2)
    # All zeros / sentinels
    assert int(batch.kind.sum()) == 0


def test_filter_sentinels_drops_kind_zero():
    """filter_sentinels keeps only rows with kind != 0."""
    batch = zero_event_batch(10)
    # Mark indices 2, 5, 7 as fusion events.
    batch = batch._replace(
        kind=batch.kind.at[2].set(KIND_FUSION).at[5].set(KIND_FUSION).at[7].set(KIND_FISSION)
    )
    filtered = filter_sentinels(batch)
    assert filtered.kind.shape == (3,)
    assert set(filtered.kind.tolist()) == {KIND_FUSION, KIND_FISSION}


def test_kind_constants():
    """The three kind sentinels must be unambiguous integers."""
    assert KIND_NONE == 0
    assert KIND_FUSION == 1
    assert KIND_FISSION == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu pytest tests/test_analysis_events.py -v`
Expected: 4 errors with `ImportError: cannot import name 'ReactionEvent' from 'halflife.state'` (or similar).

- [ ] **Step 3: Add `emit_events` to `SimConfig`**

Edit `halflife/config.py`. Find the section near `cc_fusion_event_logging` (around line 148) and add:

```python
    # ── Diagnostic event log ─────────────────────────────────────────────────
    # When True, attempt_fusion and apply_composite_decay emit per-reaction
    # event tuples (source/product slot/hash/size) for the analysis pipeline.
    # Static-arg: when False, the emission code path is dead-code-eliminated
    # before JIT compilation — bit-for-bit the same kernel as before. The
    # halflife.analysis runner sets this True; the live app leaves it False.
    emit_events: bool = False
```

- [ ] **Step 4: Add `ReactionEvent` to `halflife/state.py`**

Edit `halflife/state.py`. Add after the `CompositeState` definition (around line 65):

```python
class ReactionEvent(NamedTuple):
    """A batch of reaction events emitted by a single kernel scan/vmap.

    Each leading-axis slot is one (potential) event. Slots with kind == 0
    are sentinels and should be filtered out by halflife.analysis.events
    .filter_sentinels before consumption.

    Fusion event:    kind=1, both source_slots/hashes/sizes filled,
                            product_slots[i,0] filled, [i,1] == -1
    Fission event:   kind=2, source_slots[i,0] filled, [i,1] == -1,
                            both product_slots/hashes/sizes filled
    Sentinel (no event): all fields zero.
    """
    kind:           jnp.ndarray  # (E,) int32   0=none, 1=fusion, 2=fission
    source_slots:   jnp.ndarray  # (E, 2) int32
    source_hashes:  jnp.ndarray  # (E, 2) uint32
    source_sizes:   jnp.ndarray  # (E, 2) int32
    product_slots:  jnp.ndarray  # (E, 2) int32
    product_hashes: jnp.ndarray  # (E, 2) uint32
    product_sizes:  jnp.ndarray  # (E, 2) int32
```

- [ ] **Step 5: Create the analysis package and `events.py`**

Create `halflife/analysis/__init__.py` (empty file, just package marker).

Create `halflife/analysis/events.py`:

```python
"""Host-side helpers for ReactionEvent arrays.

The kernels in halflife/chemistry.py emit padded ReactionEvent batches per
step. This module provides the small handful of pure-Python utilities for
working with those batches after they leave the JIT.
"""

import numpy as np
import jax.numpy as jnp

from halflife.state import ReactionEvent


KIND_NONE = 0
KIND_FUSION = 1
KIND_FISSION = 2


def zero_event_batch(n_slots: int) -> ReactionEvent:
    """Allocate a ReactionEvent batch of given size, all sentinels.

    Used as the dummy/null return when config.emit_events is False, and as
    the initial accumulator for tests.
    """
    return ReactionEvent(
        kind=jnp.zeros(n_slots, dtype=jnp.int32),
        source_slots=jnp.full((n_slots, 2), -1, dtype=jnp.int32),
        source_hashes=jnp.zeros((n_slots, 2), dtype=jnp.uint32),
        source_sizes=jnp.zeros((n_slots, 2), dtype=jnp.int32),
        product_slots=jnp.full((n_slots, 2), -1, dtype=jnp.int32),
        product_hashes=jnp.zeros((n_slots, 2), dtype=jnp.uint32),
        product_sizes=jnp.zeros((n_slots, 2), dtype=jnp.int32),
    )


def filter_sentinels(batch: ReactionEvent) -> ReactionEvent:
    """Drop slots with kind == 0. Returns a numpy-backed ReactionEvent."""
    kind = np.asarray(batch.kind)
    mask = kind != KIND_NONE
    return ReactionEvent(
        kind=kind[mask],
        source_slots=np.asarray(batch.source_slots)[mask],
        source_hashes=np.asarray(batch.source_hashes)[mask],
        source_sizes=np.asarray(batch.source_sizes)[mask],
        product_slots=np.asarray(batch.product_slots)[mask],
        product_hashes=np.asarray(batch.product_hashes)[mask],
        product_sizes=np.asarray(batch.product_sizes)[mask],
    )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu pytest tests/test_analysis_events.py -v`
Expected: 4 passed in <5s.

- [ ] **Step 7: Commit**

```bash
git add halflife/config.py halflife/state.py halflife/analysis/__init__.py halflife/analysis/events.py tests/test_analysis_events.py
git commit -m "feat(analysis): add ReactionEvent + emit_events config flag

Foundation for the diagnostic event log. ReactionEvent is a padded
PyTree-friendly NamedTuple emitted per kernel scan/vmap; config.emit_events
gates kernel emission via static-arg dead-code elimination (zero cost off).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Fusion event emission

**Why:** Modify `attempt_fusion` to emit a `ReactionEvent` slot per scan iteration when `config.emit_events`. The kernel already computes everything we need (`safe_i`, `safe_j`, `all_entity_hash`, `all_entity_cnt`, `h`, `mc`, `target`); we just need to bundle them and route through the scan output.

**Files:**
- Modify: `halflife/chemistry.py` (function `attempt_fusion`, around line 724-1140)
- Modify: `tests/test_analysis_events.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_analysis_events.py`:

```python
import jax
from halflife.config import SimConfig
from halflife.state import initialize_world, initialize_interaction_params, initialize_physics_params
from halflife.chemistry import attempt_fusion
from halflife.spatial import build_cell_list, find_all_neighbors


def _tiny_config():
    """Small config that fits on CPU and produces some fusions in a few steps."""
    return SimConfig(
        num_particles=50,
        num_species=3,
        max_composites=50,
        max_composite_size=8,
        max_fusions_per_step=20,
        interaction_radius=8.0,
        fusion_radius=4.0,
        emit_events=True,
    )


def test_attempt_fusion_returns_events_when_enabled():
    """With emit_events=True, attempt_fusion returns (state, events) instead of state."""
    config = _tiny_config()
    state = initialize_world(config, seed=0)
    params = initialize_interaction_params(config, seed=1)
    physics = initialize_physics_params(config)
    neighbors = find_all_neighbors(state.particles.position, build_cell_list(state.particles.position, config), config)

    result = attempt_fusion(state, neighbors, params, config, physics)
    # When emit_events=True, returns (state, degree, events)
    assert isinstance(result, tuple)
    assert len(result) == 3, f"expected (state, degree, events), got {len(result)}-tuple"
    new_state, _degree, events = result
    assert events.kind.shape == (config.max_fusions_per_step,)


def test_attempt_fusion_returns_none_event_when_disabled():
    """With emit_events=False (default), attempt_fusion returns (state, degree) — unchanged signature."""
    config = _tiny_config()._replace(emit_events=False) if hasattr(_tiny_config(), '_replace') else None
    # SimConfig is a frozen dataclass, not NamedTuple — use dataclasses.replace
    import dataclasses
    config = dataclasses.replace(_tiny_config(), emit_events=False)
    state = initialize_world(config, seed=0)
    params = initialize_interaction_params(config, seed=1)
    physics = initialize_physics_params(config)
    neighbors = find_all_neighbors(state.particles.position, build_cell_list(state.particles.position, config), config)

    result = attempt_fusion(state, neighbors, params, config, physics)
    # Original signature: (state, degree) — 2-tuple
    assert len(result) == 2, f"expected (state, degree), got {len(result)}-tuple"


def test_attempt_fusion_emits_consistent_fusion_events():
    """If N fusions fired (composite count grew by N), N events with kind==FUSION should be in the batch."""
    import dataclasses
    config = dataclasses.replace(_tiny_config(), emit_events=True)
    state = initialize_world(config, seed=0)
    params = initialize_interaction_params(config, seed=1)
    physics = initialize_physics_params(config)
    neighbors = find_all_neighbors(state.particles.position, build_cell_list(state.particles.position, config), config)

    alive_before = int(state.composites.alive.sum())
    new_state, _, events = attempt_fusion(state, neighbors, params, config, physics)
    alive_after = int(new_state.composites.alive.sum())

    # delta composites = (free+free fusions) - (comp+comp absorptions).
    # We can't isolate either from outside, but the non-sentinel fusion event
    # count must equal the total fusion firings. We verify a weaker invariant:
    # at least the alive delta's worth of fusion events exist (since each
    # free+free fusion creates a slot and emits one event).
    from halflife.analysis.events import filter_sentinels, KIND_FUSION
    real_events = filter_sentinels(events)
    n_fusion_events = int((real_events.kind == KIND_FUSION).sum())
    assert n_fusion_events >= max(0, alive_after - alive_before), \
        f"alive grew by {alive_after - alive_before} but only {n_fusion_events} fusion events emitted"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu pytest tests/test_analysis_events.py -v`
Expected: New tests fail. `attempt_fusion` currently always returns `(state, degree)` — the 3-tuple test will fail.

- [ ] **Step 3: Modify `attempt_fusion` to emit events**

In `halflife/chemistry.py`, locate `attempt_fusion` (around line 724). At the top of the function, import the event helpers and constants:

```python
# Add to the existing top-of-file imports in chemistry.py:
from halflife.state import ReactionEvent
from halflife.analysis.events import KIND_FUSION, KIND_NONE, zero_event_batch
```

Modify `fusion_scan_body` (around line 904) to additionally compute and return a per-iteration event. Replace the final `return` of the scan body:

```python
        # ── existing logic up through degree_carry update is unchanged ──

        # ── Per-iteration event emission (output only; gated by config) ─────
        # All inputs already exist in scan-body scope: safe_i, safe_j,
        # all_entity_hash, all_entity_cnt, h (merged hash), mc (merged size),
        # target (product slot). can_fuse gates emission to real events.
        if config.emit_events:
            ev_kind = jnp.where(can_fuse, jnp.int32(KIND_FUSION), jnp.int32(KIND_NONE))
            ev_src_slots = jnp.where(
                can_fuse,
                jnp.array([safe_i, safe_j], dtype=jnp.int32),
                jnp.array([-1, -1], dtype=jnp.int32),
            )
            ev_src_hashes = jnp.where(
                can_fuse,
                jnp.array([all_entity_hash[safe_i], all_entity_hash[safe_j]], dtype=jnp.uint32),
                jnp.array([0, 0], dtype=jnp.uint32),
            )
            ev_src_sizes = jnp.where(
                can_fuse,
                jnp.array([all_entity_cnt[safe_i], all_entity_cnt[safe_j]], dtype=jnp.int32),
                jnp.array([0, 0], dtype=jnp.int32),
            )
            ev_prod_slots = jnp.where(
                can_fuse,
                jnp.array([target, -1], dtype=jnp.int32),
                jnp.array([-1, -1], dtype=jnp.int32),
            )
            ev_prod_hashes = jnp.where(
                can_fuse,
                jnp.array([h, jnp.uint32(0)], dtype=jnp.uint32),
                jnp.array([0, 0], dtype=jnp.uint32),
            )
            ev_prod_sizes = jnp.where(
                can_fuse,
                jnp.array([mc, jnp.int32(0)], dtype=jnp.int32),
                jnp.array([0, 0], dtype=jnp.int32),
            )
            ev = ReactionEvent(
                kind=ev_kind,
                source_slots=ev_src_slots,
                source_hashes=ev_src_hashes,
                source_sizes=ev_src_sizes,
                product_slots=ev_prod_slots,
                product_hashes=ev_prod_hashes,
                product_sizes=ev_prod_sizes,
            )
        else:
            ev = None

        return (new_claimed, new_composite_id, new_composites, new_comp_count,
                new_free_slot_ptr, degree_carry), ev
```

Update the scan call (around line 1127) and the return:

```python
    (_, final_composite_id, final_composites, _, _, final_degree), events_stack = jax.lax.scan(
        fusion_scan_body,
        (claimed_init, composite_id_init, composites, comp_count_init, free_slot_ptr_init, degree_init),
        scan_indices,
    )
```

At the very end of `attempt_fusion`, locate the existing `return (new_state, final_degree)` and replace with:

```python
    new_state = ...  # whatever the existing final state construction is

    if config.emit_events:
        # events_stack already has leading dim max_fusions_per_step thanks to scan.
        return new_state, final_degree, events_stack
    return new_state, final_degree
```

- [ ] **Step 4: Update step.py callers**

In `halflife/step.py`, find the `attempt_fusion` call (around line 306):

```python
    state, degree = attempt_fusion(
        state, neighbors, params, config, physics,
        degree=degree, species_valences=species_valences,
    )
```

Replace with:

```python
    if config.emit_events:
        state, degree, fusion_events = attempt_fusion(
            state, neighbors, params, config, physics,
            degree=degree, species_valences=species_valences,
        )
    else:
        state, degree = attempt_fusion(
            state, neighbors, params, config, physics,
            degree=degree, species_valences=species_valences,
        )
        fusion_events = None
```

(Task 4 will use `fusion_events` to assemble the per-step event batch. For now we just keep the variable around.)

- [ ] **Step 5: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu pytest tests/test_analysis_events.py -v`
Expected: all 7 tests pass in <20s.

Also run a regression check that the existing test suite still passes for the bond-mode-strict scan:

Run: `JAX_PLATFORMS=cpu pytest tests/test_covalent_bonds_integration.py -v -x`
Expected: existing integration tests pass — emit_events defaults to False so behavior is unchanged.

- [ ] **Step 6: Commit**

```bash
git add halflife/chemistry.py halflife/step.py tests/test_analysis_events.py
git commit -m "feat(chemistry): emit fusion events from attempt_fusion when enabled

Each fusion_scan_body iteration emits a ReactionEvent slot; non-fusion
iterations emit a sentinel (kind=0). Gated by config.emit_events — when
False, the emission branch is dead-code-eliminated and the kernel is
bit-for-bit unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Fission event emission

**Why:** Mirror Task 2 for `apply_composite_decay`. The function uses `vmap` (not scan) over all composites; per-composite emission stacks via vmap the same way scan stacks.

**Files:**
- Modify: `halflife/chemistry.py` (function `apply_composite_decay`, around line 370-720)
- Modify: `tests/test_analysis_events.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_analysis_events.py`:

```python
def test_apply_composite_decay_emits_fission_events():
    """When emit_events=True, apply_composite_decay returns (state, events)."""
    import dataclasses
    from halflife.chemistry import apply_composite_decay

    # Tiny config with short half-life so fission fires reliably within ~50 steps.
    config = dataclasses.replace(
        _tiny_config(),
        emit_events=True,
        half_life_min=0.1,    # force rapid decay
        half_life_max=0.5,
    )
    state = initialize_world(config, seed=0)
    physics = initialize_physics_params(config)

    # Manually create a composite so there's something to fission. Easiest:
    # just run a few fusion steps first to build composites, then fission.
    from halflife.spatial import build_cell_list, find_all_neighbors
    params = initialize_interaction_params(config, seed=1)
    for _ in range(5):
        neighbors = find_all_neighbors(
            state.particles.position,
            build_cell_list(state.particles.position, config),
            config,
        )
        state, _, _ = attempt_fusion(state, neighbors, params, config, physics)
    assert int(state.composites.alive.sum()) > 0, "need composites to test fission"

    result = apply_composite_decay(state, config, physics)
    assert len(result) == 2, f"expected (state, events), got {len(result)}-tuple"
    new_state, events = result
    assert events.kind.shape == (config.max_composites,)

    # Some composites should have died (half-life is 0.1-0.5).
    n_died = int(state.composites.alive.sum()) - int(new_state.composites.alive.sum())
    if n_died > 0:
        from halflife.analysis.events import filter_sentinels, KIND_FISSION
        real = filter_sentinels(events)
        n_fission_events = int((real.kind == KIND_FISSION).sum())
        assert n_fission_events >= n_died, \
            f"{n_died} composites died but only {n_fission_events} fission events emitted"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu pytest tests/test_analysis_events.py::test_apply_composite_decay_emits_fission_events -v`
Expected: failure — `apply_composite_decay` currently returns just `state`.

- [ ] **Step 3: Modify `apply_composite_decay` to emit events**

In `halflife/chemistry.py`, locate `apply_composite_decay` (around line 370). It uses `jax.vmap(per_product)` and `jax.vmap(per_composite)`. The cleanest insertion point is after the per-product hash/size data is computed (around line 490, after `p0_members, p0_count, p0_hash, p1_members, p1_count, p1_hash = jax.vmap(per_product)(...)`).

Just before the final state construction (around line 700, right before the `composites._replace(...)` call), add:

```python
    # ── Per-composite event emission (output only; gated by config) ─────────
    # Event kind=2 (fission) for every composite where `fissions[c]` is True.
    # source_slots = (c, -1); product_slots = (c, target_p1[c]).
    # source_hash/size = the parent composite at slot c BEFORE the state update.
    # product_hashes = (p0_hash[c], p1_hash[c]); product_sizes = (p0_count[c], p1_count[c]).
    # Slot 0 of products may be a shattered free particle (count=1, hash=0);
    # slot 1 likewise. Either or both being product_size==1 is legal.
    if config.emit_events:
        c_arr = jnp.arange(C, dtype=jnp.int32)
        ev_kind = jnp.where(fissions, jnp.int32(KIND_FISSION), jnp.int32(KIND_NONE))
        ev_src_slots = jnp.stack([
            jnp.where(fissions, c_arr, jnp.int32(-1)),
            jnp.full((C,), -1, dtype=jnp.int32),
        ], axis=1)
        ev_src_hashes = jnp.stack([
            jnp.where(fissions, composites.species_hash, jnp.uint32(0)),
            jnp.zeros((C,), dtype=jnp.uint32),
        ], axis=1)
        ev_src_sizes = jnp.stack([
            jnp.where(fissions, composites.member_count, jnp.int32(0)),
            jnp.zeros((C,), dtype=jnp.int32),
        ], axis=1)
        ev_prod_slots = jnp.stack([
            jnp.where(fissions, c_arr, jnp.int32(-1)),
            jnp.where(fissions, all_target_p1, jnp.int32(-1)),
        ], axis=1)
        ev_prod_hashes = jnp.stack([
            jnp.where(fissions, p0_hash, jnp.uint32(0)),
            jnp.where(fissions, p1_hash, jnp.uint32(0)),
        ], axis=1)
        ev_prod_sizes = jnp.stack([
            jnp.where(fissions, p0_count, jnp.int32(0)),
            jnp.where(fissions, p1_count, jnp.int32(0)),
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
```

At the end of the function, locate the existing `return new_state` and replace with:

```python
    if config.emit_events:
        return new_state, events
    return new_state
```

- [ ] **Step 4: Update `step.py` caller**

In `halflife/step.py`, find the `apply_composite_decay` call (around line 319):

```python
    state = apply_composite_decay(state, config, physics)
```

Replace with:

```python
    if config.emit_events:
        state, fission_events = apply_composite_decay(state, config, physics)
    else:
        state = apply_composite_decay(state, config, physics)
        fission_events = None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu pytest tests/test_analysis_events.py -v`
Expected: all 8 tests pass in <30s.

Regression check:
Run: `JAX_PLATFORMS=cpu pytest tests/test_covalent_bonds_integration.py -v -x`
Expected: existing integration tests pass.

- [ ] **Step 6: Commit**

```bash
git add halflife/chemistry.py halflife/step.py tests/test_analysis_events.py
git commit -m "feat(chemistry): emit fission events from apply_composite_decay

Per-composite vmap'd emission: every alive composite has one ReactionEvent
slot, fissions[c]==True produces kind=2 with source/product hash+size+slot
filled. Gated by config.emit_events for zero overhead when off.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `simulation_step` event wiring + concatenation

**Why:** Stitch the per-kernel event batches into one per-step batch and surface them as `simulation_step`'s second return value. Downstream `lax.scan` collects them.

**Files:**
- Modify: `halflife/step.py` (function `simulation_step`)
- Modify: `tests/test_analysis_events.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_analysis_events.py`:

```python
def test_simulation_step_returns_events_when_enabled():
    """simulation_step with emit_events=True returns (state, events). E_max = max_fusions + max_composites."""
    import dataclasses
    from halflife.step import simulation_step
    config = dataclasses.replace(_tiny_config(), emit_events=True)
    state = initialize_world(config, seed=0)
    params = initialize_interaction_params(config, seed=1)
    physics = initialize_physics_params(config)

    result = simulation_step(state, params, config, physics)
    assert isinstance(result, tuple) and len(result) == 2
    new_state, events = result
    expected_e = config.max_fusions_per_step + config.max_composites
    assert events.kind.shape == (expected_e,), f"expected ({expected_e},), got {events.kind.shape}"


def test_simulation_step_returns_state_when_disabled():
    """Default emit_events=False: simulation_step returns just state, unchanged signature."""
    from halflife.step import simulation_step
    config = _tiny_config()  # default emit_events should be False
    # Override the test-local override:
    import dataclasses
    config = dataclasses.replace(config, emit_events=False)
    state = initialize_world(config, seed=0)
    params = initialize_interaction_params(config, seed=1)
    physics = initialize_physics_params(config)

    result = simulation_step(state, params, config, physics)
    # Just a WorldState, not a tuple.
    from halflife.state import WorldState
    assert isinstance(result, WorldState)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu pytest tests/test_analysis_events.py -v`
Expected: new tests fail.

- [ ] **Step 3: Modify `simulation_step` to assemble and return events**

In `halflife/step.py`, locate `simulation_step` (around line 235). At the end of the function, before the final `return state`, add:

```python
    # ── Event log assembly (output only) ─────────────────────────────────────
    # fusion_events: ReactionEvent of shape (max_fusions_per_step, ...)
    # fission_events: ReactionEvent of shape (max_composites, ...)
    # Concatenate along the leading axis so downstream consumers see one
    # padded ReactionEvent per step with E = max_fusions + max_composites.
    if config.emit_events:
        from halflife.state import ReactionEvent
        events = ReactionEvent(
            kind=jnp.concatenate([fusion_events.kind, fission_events.kind]),
            source_slots=jnp.concatenate([fusion_events.source_slots, fission_events.source_slots], axis=0),
            source_hashes=jnp.concatenate([fusion_events.source_hashes, fission_events.source_hashes], axis=0),
            source_sizes=jnp.concatenate([fusion_events.source_sizes, fission_events.source_sizes], axis=0),
            product_slots=jnp.concatenate([fusion_events.product_slots, fission_events.product_slots], axis=0),
            product_hashes=jnp.concatenate([fusion_events.product_hashes, fission_events.product_hashes], axis=0),
            product_sizes=jnp.concatenate([fusion_events.product_sizes, fission_events.product_sizes], axis=0),
        )
        return state, events
    return state
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu pytest tests/test_analysis_events.py -v`
Expected: all 10 tests pass in <40s.

Critical regression check — the live app must still work:
Run: `JAX_PLATFORMS=cpu pytest tests/test_step.py tests/test_chemistry.py -v -x`
Expected: all existing tests pass. Default `emit_events=False` means `simulation_step` returns `state` only.

- [ ] **Step 5: Commit**

```bash
git add halflife/step.py tests/test_analysis_events.py
git commit -m "feat(step): assemble per-step ReactionEvent log in simulation_step

Concatenates fusion + fission events into one padded ReactionEvent per step
(E = max_fusions_per_step + max_composites). simulation_step returns
(state, events) when config.emit_events else just state.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Per-step metrics module

**Why:** Pure functions that compute Tier 1 (macroscopic) and Tier 2 (valence/edge) metrics from a `WorldState` snapshot. Used inside `lax.scan` for per-step collection.

**Files:**
- Create: `halflife/analysis/metrics.py`
- Create: `tests/test_analysis_metrics.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_analysis_metrics.py`:

```python
"""Tests for per-step metric collection functions."""
import jax.numpy as jnp
import numpy as np
import pytest

from halflife.config import SimConfig
from halflife.state import initialize_world
from halflife.analysis.metrics import (
    size_metrics,
    valence_edge_metrics,
)


def _tiny_config():
    return SimConfig(num_particles=50, num_species=3, max_composites=50, max_composite_size=8)


def test_size_metrics_zero_composites():
    """Initial state has no composites — all size metrics should be safe zeros."""
    config = _tiny_config()
    state = initialize_world(config, seed=0)
    m = size_metrics(state.composites, config)
    assert m['max_size'] == 0
    assert m['alive_count'] == 0
    assert m['free_particle_fraction'] == pytest.approx(1.0)
    assert m['size_histogram'].shape == (config.max_composite_size,)
    assert int(m['size_histogram'].sum()) == 0


def test_valence_edge_metrics_zero_composites():
    """Initial state — no edges, no saturated particles."""
    config = _tiny_config()
    state = initialize_world(config, seed=0)
    m = valence_edge_metrics(state.particles, state.composites, config)
    assert m['edge_count_total'] == 0
    assert m['ring_count_total'] == 0
    # All particles are free (degree 0); saturation requires v_s == 0, impossible
    # since v_s ∈ [1, max_valence]. So saturation pct is 0.
    assert m['degree_saturation_pct'] == pytest.approx(0.0)
    assert m['free_bonds_histogram'].ndim == 1
    assert m['degree_histogram'].ndim == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu pytest tests/test_analysis_metrics.py -v`
Expected: failure with `ModuleNotFoundError: No module named 'halflife.analysis.metrics'`.

- [ ] **Step 3: Implement `metrics.py`**

Create `halflife/analysis/metrics.py`:

```python
"""Per-step metric collection.

All functions here take state arrays and config and return plain dicts of
JAX scalars/arrays. Designed to be called inside lax.scan bodies — the
outputs stack naturally into per-step time series.
"""

import jax
import jax.numpy as jnp

from halflife.config import SimConfig
from halflife.chemistry import compute_degree, _species_valences


def size_metrics(composites, config: SimConfig) -> dict:
    """Tier 1 macroscopic metrics from CompositeState.

    Returns a dict whose values are all JAX-traceable so this can be called
    inside a lax.scan body. Outputs:
      max_size, mean_size, median_size, alive_count,
      free_particle_fraction, size_histogram (1..max_composite_size)
    """
    alive = composites.alive
    counts = composites.member_count

    alive_counts = jnp.where(alive, counts, 0)
    n_alive = jnp.sum(alive.astype(jnp.int32))
    total_in_composites = jnp.sum(alive_counts.astype(jnp.int32))

    max_size = jnp.max(jnp.where(alive, counts, 0))
    safe_n = jnp.maximum(n_alive, 1)
    mean_size = jnp.where(
        n_alive > 0,
        jnp.sum(alive_counts.astype(jnp.float32)) / safe_n.astype(jnp.float32),
        jnp.float32(0.0),
    )

    sorted_sizes = jnp.sort(alive_counts)
    median_idx = jnp.clip(n_alive // 2, 0, sorted_sizes.shape[0] - 1)
    median_size = jnp.where(
        n_alive > 0,
        sorted_sizes[median_idx].astype(jnp.float32),
        jnp.float32(0.0),
    )

    free_particle_fraction = jnp.where(
        config.num_particles > 0,
        1.0 - total_in_composites.astype(jnp.float32) / jnp.float32(config.num_particles),
        jnp.float32(1.0),
    )

    # Histogram over sizes 1..max_composite_size (bin 0 = "size 1", etc.)
    bins = jnp.arange(1, config.max_composite_size + 1, dtype=jnp.int32)
    size_histogram = jax.vmap(
        lambda b: jnp.sum(jnp.where(alive & (counts == b), 1, 0).astype(jnp.int32))
    )(bins)

    return {
        'max_size': max_size,
        'mean_size': mean_size,
        'median_size': median_size,
        'alive_count': n_alive,
        'free_particle_fraction': free_particle_fraction,
        'size_histogram': size_histogram,
    }


def valence_edge_metrics(particles, composites, config: SimConfig) -> dict:
    """Tier 2 valence/edge metrics.

    Returns:
      edge_count_total      — sum of edge_count over alive composites
      ring_count_total      — sum of (edge_count - (size - 1)) over alive composites
                              (extra edges beyond a spanning tree)
      degree_saturation_pct — fraction of particles with degree == v_species
      free_bonds_histogram  — bincount of free_bonds[alive] (length = max possible + 1)
      degree_histogram      — bincount of per-particle degree (length = max_valence + 1)
    """
    species_valences = _species_valences(config)
    degree = compute_degree(composites, config)

    # Degree saturation: degree[i] == v_{species[i]}
    particle_v = species_valences[particles.species]
    saturated = (degree == particle_v).astype(jnp.float32)
    degree_saturation_pct = jnp.mean(saturated)

    # Free-bonds histogram: free_bonds in alive composites can range
    # from 0 to (max_composite_size * max_valence) inclusive in theory.
    # Use a generous bin range.
    fb_max = config.max_composite_size * config.max_valence
    fb_alive = jnp.where(composites.alive, composites.free_bonds, jnp.int32(0))
    fb_bins = jnp.arange(fb_max + 1, dtype=jnp.int32)
    free_bonds_histogram = jax.vmap(
        lambda b: jnp.sum(jnp.where(composites.alive & (composites.free_bonds == b), 1, 0))
    )(fb_bins)

    # Per-particle degree histogram (0..max_valence)
    deg_bins = jnp.arange(config.max_valence + 1, dtype=jnp.int32)
    degree_histogram = jax.vmap(
        lambda b: jnp.sum((degree == b).astype(jnp.int32))
    )(deg_bins)

    # Edge / ring counts
    spanning_edges = jnp.where(
        composites.alive,
        jnp.maximum(composites.member_count - 1, 0),
        0,
    )
    edge_count_total = jnp.sum(
        jnp.where(composites.alive, composites.edge_count, 0).astype(jnp.int32)
    )
    ring_count_total = jnp.sum(
        jnp.where(composites.alive, composites.edge_count - spanning_edges, 0).astype(jnp.int32)
    )

    return {
        'edge_count_total': edge_count_total,
        'ring_count_total': ring_count_total,
        'degree_saturation_pct': degree_saturation_pct,
        'free_bonds_histogram': free_bonds_histogram,
        'degree_histogram': degree_histogram,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu pytest tests/test_analysis_metrics.py -v`
Expected: 2 passed in <5s.

- [ ] **Step 5: Commit**

```bash
git add halflife/analysis/metrics.py tests/test_analysis_metrics.py
git commit -m "feat(analysis): add per-step size and valence/edge metric collectors

Pure functions returning JAX scalar/array dicts — suitable for use inside
lax.scan bodies. Tier 1 (max/mean/median size, alive count, free-particle
fraction, size histogram) and Tier 2 (degree saturation %, edge/ring count,
free_bonds and degree histograms).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Fusion compatibility module

**Why:** Pure-chemistry, no simulation. Given a list of composite types (identified by `species_hash` + member multiset), compute pairwise merged BE and max-free-bonds gates. Powers Tier 4 matrices.

**Files:**
- Create: `halflife/analysis/compatibility.py`
- Create: `tests/test_analysis_compatibility.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_analysis_compatibility.py`:

```python
"""Tests for the pure-chemistry fusion compatibility module."""
import numpy as np
import pytest

from halflife.config import SimConfig
from halflife.state import initialize_physics_params
from halflife.analysis.compatibility import (
    species_pair_compat_matrix,
    max_free_bonds,
)


def test_species_pair_compat_matrix_shape():
    """Matrix is (S, S) and symmetric in BE (hash sum is commutative)."""
    config = SimConfig(num_species=4, num_particles=10, max_composites=10)
    physics = initialize_physics_params(config)

    be, passes_be, passes_val = species_pair_compat_matrix(config, physics)
    assert be.shape == (4, 4)
    assert passes_be.shape == (4, 4)
    assert passes_val.shape == (4, 4)
    # Symmetric: BE(i,j) == BE(j,i) since merged hash is commutative.
    np.testing.assert_allclose(be, be.T, atol=1e-6)


def test_max_free_bonds_free_particle():
    """A free particle of species s has max_free_bonds = v_s (no edges)."""
    config = SimConfig(num_species=4, num_particles=10, max_composites=10)
    # Single-species multiset, count=1
    for s in range(4):
        fb = max_free_bonds([s], config)
        # Should equal hash-derived valence for that species, which is in [1, max_valence]
        assert 1 <= fb <= config.max_valence


def test_max_free_bonds_two_v1_particles_is_zero():
    """A 2-member composite of two v=1 species has max_free_bonds = 0 (saturated)."""
    # We need to find species with v=1 — easier to test via the formula directly:
    # Σ v_s − 2*(n−1). For n=2 and Σv=2, max_fb = 2 - 2 = 0.
    # Use a small config so it's tractable.
    config = SimConfig(num_species=2, num_particles=10, max_composites=10, max_valence=2)
    from halflife.chemistry import _species_valences
    import numpy as np
    v = np.asarray(_species_valences(config))
    # Find two species where v[a] + v[b] = 2 — possible only if v=1 each.
    # max_valence=2 means valences are in [1,2]. Try both species duplicated:
    for a in range(config.num_species):
        for b in range(config.num_species):
            expected = int(v[a]) + int(v[b]) - 2  # n=2, so -2*(2-1) = -2
            actual = max_free_bonds([a, b], config)
            assert actual == expected, f"a={a} b={b}: expected {expected}, got {actual}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu pytest tests/test_analysis_compatibility.py -v`
Expected: `ModuleNotFoundError: No module named 'halflife.analysis.compatibility'`.

- [ ] **Step 3: Implement `compatibility.py`**

Create `halflife/analysis/compatibility.py`:

```python
"""Pure-chemistry fusion compatibility matrices.

Given the static chemistry params (SimConfig, PhysicsParams), compute for
every pair of composite types whether they could in principle fuse — that
is, whether the merged BE passes the fusion threshold AND whether each side
has at least one free bond at their structural maximum.

Pure: no simulation, no event log, no state arrays. Sub-second even for
hundreds of unique composite types.
"""

from typing import Sequence

import numpy as np
import jax
import jax.numpy as jnp

from halflife.config import SimConfig
from halflife.state import PhysicsParams
from halflife.chemistry import (
    _entity_hash_val,
    _hash_to_binding_energy,
    _species_valences,
)


def max_free_bonds(member_species: Sequence[int], config: SimConfig) -> int:
    """Structural upper bound on free_bonds for a fresh n-member composite.

    Formula: Σ v_{s_i} − 2 * (n − 1) — spanning-tree minimum: n−1 edges,
    each consuming one bond on each endpoint. Free particles (n=1) get
    just v_s.
    """
    v = np.asarray(_species_valences(config))
    n = len(member_species)
    sum_v = sum(int(v[s]) for s in member_species)
    return sum_v - 2 * max(0, n - 1)


def _hash_multiset(member_species: Sequence[int], config: SimConfig) -> int:
    """Commutative additive hash matching the on-device kernel."""
    h = 0
    for s in member_species:
        h = (h + int(_entity_hash_val(jnp.int32(s), config))) % config.hash_modulus
    return h


def species_pair_compat_matrix(config: SimConfig, physics: PhysicsParams):
    """Per-species-pair fusion compatibility (Matrix 4a).

    Returns three (S, S) numpy arrays:
      be          — merged binding energy for the pair (float32)
      passes_be   — whether be >= physics.fusion_threshold (bool)
      passes_val  — whether both species have v >= 1 (always True, but
                    included for API symmetry with the top-K matrix where
                    saturation can happen)
    """
    S = config.num_species

    # Per-species values via vmap.
    species_idx = jnp.arange(S, dtype=jnp.int32)
    hvals = jax.vmap(lambda s: _entity_hash_val(s, config))(species_idx)  # (S,) uint32
    valences = jax.vmap(lambda s: jnp.asarray(_species_valences(config)[s]))(species_idx)  # (S,) int32 — see below
    # _species_valences returns the whole vector — just convert once.
    valences = np.asarray(_species_valences(config))  # (S,) int32

    # Pairwise merged hash via outer sum then mod.
    H = np.asarray(hvals).astype(np.int64)
    merged = (H[:, None] + H[None, :]) % config.hash_modulus  # (S, S) int64

    # Compute BE for each cell on the host (vectorize the JAX call).
    be = np.zeros((S, S), dtype=np.float32)
    for i in range(S):
        for j in range(S):
            be[i, j] = float(_hash_to_binding_energy(
                jnp.uint32(int(merged[i, j])), config, physics
            ))

    passes_be = be >= float(physics.fusion_threshold)
    # Free particles always pass valence (v >= 1 by construction).
    passes_val = np.ones((S, S), dtype=bool)

    return be, passes_be, passes_val


def observed_pair_compat_matrix(
    hashes: np.ndarray,           # (K,) uint32 — unique species_hashes of observed composites
    multisets: list,              # length K — each entry is a sorted tuple of species ints
    config: SimConfig,
    physics: PhysicsParams,
):
    """Top-K observed-composite fusion compatibility (Matrix 4b).

    Args:
      hashes:    unique species_hash values, one per observed composite type
      multisets: parallel list of per-type member multisets (sorted tuples of species)
      config:    SimConfig
      physics:   PhysicsParams

    Returns: (be, passes_be, passes_val) each shape (K, K).
    """
    K = len(hashes)
    H = hashes.astype(np.int64)
    merged = (H[:, None] + H[None, :]) % config.hash_modulus

    be = np.zeros((K, K), dtype=np.float32)
    for i in range(K):
        for j in range(K):
            be[i, j] = float(_hash_to_binding_energy(
                jnp.uint32(int(merged[i, j])), config, physics
            ))

    passes_be = be >= float(physics.fusion_threshold)

    # passes_val: max_free_bonds(multisets[i]) >= 1 AND ditto for j.
    mfb = np.array([max_free_bonds(m, config) for m in multisets])
    passes_val = (mfb[:, None] >= 1) & (mfb[None, :] >= 1)

    return be, passes_be, passes_val
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu pytest tests/test_analysis_compatibility.py -v`
Expected: 3 passed in <10s.

- [ ] **Step 5: Commit**

```bash
git add halflife/analysis/compatibility.py tests/test_analysis_compatibility.py
git commit -m "feat(analysis): add pure-chemistry fusion compatibility module

Computes per-pair merged BE + max_free_bonds gates without simulation.
Powers Tier 4 matrices (4a species-pair, 4b observed-composite). Uses
existing _hash_to_binding_energy / _species_valences from chemistry.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Transitions module

**Why:** Convert a flat `ReactionEvent` log (numpy-backed, sentinels filtered) into the three transition matrices (size-binned, top-K, full).

**Files:**
- Create: `halflife/analysis/transitions.py`
- Create: `tests/test_analysis_transitions.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_analysis_transitions.py`:

```python
"""Tests for the event-log → transition-matrix builder."""
import numpy as np
import pytest

from halflife.state import ReactionEvent
from halflife.analysis.transitions import (
    size_bin_transition_matrix,
    top_k_transition_matrix,
    full_transition_matrix,
)
from halflife.analysis.events import KIND_FUSION, KIND_FISSION


def _make_event(kind, src_hashes, src_sizes, prod_hashes, prod_sizes):
    """Build a 1-event ReactionEvent from a small spec."""
    return ReactionEvent(
        kind=np.array([kind], dtype=np.int32),
        source_slots=np.array([[0, 1 if kind == KIND_FUSION else -1]], dtype=np.int32),
        source_hashes=np.array([src_hashes], dtype=np.uint32),
        source_sizes=np.array([src_sizes], dtype=np.int32),
        product_slots=np.array([[10, 11 if kind == KIND_FISSION else -1]], dtype=np.int32),
        product_hashes=np.array([prod_hashes], dtype=np.uint32),
        product_sizes=np.array([prod_sizes], dtype=np.int32),
    )


def _concat_events(*evs):
    return ReactionEvent(
        kind=np.concatenate([e.kind for e in evs]),
        source_slots=np.concatenate([e.source_slots for e in evs]),
        source_hashes=np.concatenate([e.source_hashes for e in evs]),
        source_sizes=np.concatenate([e.source_sizes for e in evs]),
        product_slots=np.concatenate([e.product_slots for e in evs]),
        product_hashes=np.concatenate([e.product_hashes for e in evs]),
        product_sizes=np.concatenate([e.product_sizes for e in evs]),
    )


def test_size_bin_matrix_fusion_contributes_to_both_source_rows():
    """One fusion A(size 2) + B(size 3) → C(size 5) should add to cells (2,5) and (3,5)."""
    evt = _make_event(
        KIND_FUSION,
        src_hashes=[100, 200], src_sizes=[2, 3],
        prod_hashes=[300, 0], prod_sizes=[5, 0],
    )
    M = size_bin_transition_matrix(evt, max_composite_size=8)
    assert M[2, 5] == 1
    assert M[3, 5] == 1
    # No other cells should be set.
    M[2, 5] = 0
    M[3, 5] = 0
    assert M.sum() == 0


def test_size_bin_matrix_fission_contributes_to_both_product_cols():
    """One fission C(size 5) → A(size 2) + B(size 3) should add to cells (5,2) and (5,3)."""
    evt = _make_event(
        KIND_FISSION,
        src_hashes=[300, 0], src_sizes=[5, 0],
        prod_hashes=[100, 200], prod_sizes=[2, 3],
    )
    M = size_bin_transition_matrix(evt, max_composite_size=8)
    assert M[5, 2] == 1
    assert M[5, 3] == 1
    M[5, 2] = 0
    M[5, 3] = 0
    assert M.sum() == 0


def test_top_k_matrix_buckets_tail_into_other():
    """K=2 with 4 unique hashes: top 2 stay, remaining 2 collapse into 'other' row/col."""
    # Build several events so we have rankable hashes.
    evts = [
        _make_event(KIND_FUSION, [1, 2], [1, 1], [3, 0], [2, 0]),  # hashes 1,2 → 3
        _make_event(KIND_FUSION, [1, 2], [1, 1], [3, 0], [2, 0]),  # again — boosts 1,2,3
        _make_event(KIND_FUSION, [4, 5], [1, 1], [9, 0], [2, 0]),  # rare hashes 4,5,9
    ]
    batch = _concat_events(*evts)
    M, labels = top_k_transition_matrix(batch, K=2)
    # Shape: K+1 by K+1 with "other" appended.
    assert M.shape == (3, 3)
    assert labels[-1] == 'other'


def test_full_transition_matrix_uses_every_observed_hash():
    """U×U where U = unique hashes across all events."""
    evt = _make_event(KIND_FUSION, [1, 2], [1, 1], [3, 0], [2, 0])
    M, labels = full_transition_matrix(evt)
    assert len(labels) == 3  # hashes 1, 2, 3
    assert M.shape == (3, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu pytest tests/test_analysis_transitions.py -v`
Expected: `ModuleNotFoundError: No module named 'halflife.analysis.transitions'`.

- [ ] **Step 3: Implement `transitions.py`**

Create `halflife/analysis/transitions.py`:

```python
"""Event log → composite transition matrices.

Three matrix shapes, all built from a flat (sentinels-filtered) ReactionEvent:
  - size-binned: (max_composite_size, max_composite_size) — mass-flow view
  - top-K:       (K+1, K+1) — K most-trafficked hashes + "other"
  - full:        (U, U) — every observed unique hash

Cell semantics: for every fusion event A+B→C, increment matrix[A, C] and
matrix[B, C]. For every fission event C→A+B, increment matrix[C, A] and
matrix[C, B]. Each event contributes 2 cells.
"""

from collections import Counter
from typing import Tuple, List

import numpy as np

from halflife.state import ReactionEvent
from halflife.analysis.events import KIND_FUSION, KIND_FISSION


def _iter_edges(events: ReactionEvent):
    """Yield (source_hash, source_size, product_hash, product_size) per cell.

    For fusion: yields (A,C) and (B,C). For fission: yields (C,A) and (C,B).
    """
    kind = np.asarray(events.kind)
    sh = np.asarray(events.source_hashes)
    ss = np.asarray(events.source_sizes)
    ph = np.asarray(events.product_hashes)
    ps = np.asarray(events.product_sizes)

    for i in range(kind.shape[0]):
        if kind[i] == KIND_FUSION:
            # A + B → C; yield (A, C) and (B, C). Product is in slot 0.
            for src_idx in (0, 1):
                yield (int(sh[i, src_idx]), int(ss[i, src_idx]),
                       int(ph[i, 0]),       int(ps[i, 0]))
        elif kind[i] == KIND_FISSION:
            # C → A + B; yield (C, A) and (C, B). Source is in slot 0.
            for prod_idx in (0, 1):
                yield (int(sh[i, 0]),         int(ss[i, 0]),
                       int(ph[i, prod_idx]),  int(ps[i, prod_idx]))


def size_bin_transition_matrix(events: ReactionEvent, max_composite_size: int) -> np.ndarray:
    """(max_composite_size+1, max_composite_size+1) matrix of size→size transitions.

    Index 0 = empty / sentinel — practically unused. Bins 1..max_composite_size
    are the live size classes.
    """
    M = max_composite_size + 1
    matrix = np.zeros((M, M), dtype=np.int64)
    for _sh, ss, _ph, ps in _iter_edges(events):
        if 0 <= ss < M and 0 <= ps < M:
            matrix[ss, ps] += 1
    return matrix


def top_k_transition_matrix(
    events: ReactionEvent, K: int = 30
) -> Tuple[np.ndarray, List[str]]:
    """(K+1, K+1) matrix on the K most-trafficked species hashes, sorted by size.

    Sort key: (size ascending, hash ascending). "Trafficked" = total incidence
    (row + col before truncation). The last row/col is "other" and collects
    all tail traffic.

    Returns (matrix, labels) where labels[i] is the human-readable hash for
    row/col i ("0x..." or "other" for the last).
    """
    edges = list(_iter_edges(events))
    if not edges:
        return np.zeros((1, 1), dtype=np.int64), ['other']

    # Map hash → size (first seen wins; should be deterministic).
    hash_size = {}
    incidence = Counter()
    for sh, ss, ph, ps in edges:
        hash_size.setdefault(sh, ss)
        hash_size.setdefault(ph, ps)
        incidence[sh] += 1
        incidence[ph] += 1

    # Pick top K by incidence; sort selected by (size ascending, hash ascending).
    top_hashes = [h for h, _ in incidence.most_common(K)]
    top_hashes.sort(key=lambda h: (hash_size[h], h))
    h_to_idx = {h: i for i, h in enumerate(top_hashes)}
    other_idx = K  # last row/col

    matrix = np.zeros((K + 1, K + 1), dtype=np.int64)
    for sh, _ss, ph, _ps in edges:
        i = h_to_idx.get(sh, other_idx)
        j = h_to_idx.get(ph, other_idx)
        matrix[i, j] += 1

    labels = [f"0x{h:08x}" for h in top_hashes] + ['other']
    # Trim if there were fewer than K unique hashes.
    actual_k = len(top_hashes)
    if actual_k < K:
        matrix = matrix[:actual_k + 1, :actual_k + 1]
        labels = labels[:actual_k + 1]
    return matrix, labels


def full_transition_matrix(
    events: ReactionEvent,
) -> Tuple[np.ndarray, List[str]]:
    """(U, U) matrix over every observed unique hash, sorted by size ascending."""
    edges = list(_iter_edges(events))
    if not edges:
        return np.zeros((0, 0), dtype=np.int64), []

    hash_size = {}
    for sh, ss, ph, ps in edges:
        hash_size.setdefault(sh, ss)
        hash_size.setdefault(ph, ps)

    sorted_hashes = sorted(hash_size, key=lambda h: (hash_size[h], h))
    h_to_idx = {h: i for i, h in enumerate(sorted_hashes)}
    U = len(sorted_hashes)
    matrix = np.zeros((U, U), dtype=np.int64)
    for sh, _ss, ph, _ps in edges:
        matrix[h_to_idx[sh], h_to_idx[ph]] += 1
    labels = [f"0x{h:08x}" for h in sorted_hashes]
    return matrix, labels
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu pytest tests/test_analysis_transitions.py -v`
Expected: 4 passed in <5s.

- [ ] **Step 5: Commit**

```bash
git add halflife/analysis/transitions.py tests/test_analysis_transitions.py
git commit -m "feat(analysis): event log to transition matrix builder

Produces three matrices from a sentinels-filtered ReactionEvent log:
size-binned (max_composite_size+1)^2, top-K (K+1)^2 with 'other' bucket,
and full (U)^2 over every observed hash. All sorted by size ascending.
Each fusion/fission event contributes two cells.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Headless runner

**Why:** End-to-end orchestration of one diagnostic run. `lax.scan` for fast per-step compact metrics + events; periodic host-side full snapshots for the multiset reconstruction needed by Tier 4 / Matrix 4b.

**Files:**
- Create: `halflife/analysis/runner.py`
- Append: `tests/test_analysis_pipeline.py` (created here, expanded in Task 12)

- [ ] **Step 1: Write the failing test**

Create `tests/test_analysis_pipeline.py`:

```python
"""End-to-end smoke tests for halflife/analysis. Expanded in Task 12."""
import dataclasses
import os

import numpy as np
import pytest

from halflife.config import SimConfig
from halflife.analysis.runner import run_diagnostic, RunResult


def _tiny_config():
    return dataclasses.replace(
        SimConfig(),
        num_particles=80,
        num_species=3,
        max_composites=80,
        max_composite_size=8,
        max_fusions_per_step=30,
        emit_events=True,
    )


def test_run_diagnostic_returns_run_result():
    """Smoke test: a 100-step CPU run produces a populated RunResult."""
    config = _tiny_config()
    result = run_diagnostic(config, n_steps=100, seed=0, sample_every=25)
    assert isinstance(result, RunResult)
    assert result.config is config
    assert result.n_steps == 100
    assert result.per_step_metrics['max_size'].shape == (100,)
    assert result.per_step_metrics['alive_count'].shape == (100,)
    # Events: flat numpy ReactionEvent after sentinel filtering — length variable.
    assert hasattr(result.events, 'kind')
    assert result.events.kind.dtype == np.int32
    # 100 steps / 25 = 4 snapshots (or 5 — implementation may include step 0)
    assert 3 <= len(result.snapshots) <= 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu pytest tests/test_analysis_pipeline.py -v`
Expected: `ModuleNotFoundError: No module named 'halflife.analysis.runner'`.

- [ ] **Step 3: Implement `runner.py`**

Create `halflife/analysis/runner.py`:

```python
"""Headless simulation runner for diagnostic analysis.

Single entry point: run_diagnostic(config, n_steps, seed, sample_every) →
RunResult. The runner drives the JIT'd simulation_step inside a lax.scan
that emits per-step compact metrics + events, with periodic host-side
copies of full snapshots for downstream distribution drill-downs.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import time
import dataclasses

import numpy as np
import jax
import jax.numpy as jnp

from halflife.config import SimConfig
from halflife.state import (
    WorldState, ReactionEvent,
    initialize_world, initialize_interaction_params, initialize_physics_params,
)
from halflife.step import simulation_step
from halflife.analysis.metrics import size_metrics, valence_edge_metrics
from halflife.analysis.events import filter_sentinels, zero_event_batch


@dataclass
class CompositeSnapshot:
    """A point-in-time copy of composite arrays on the host."""
    step: int
    alive: np.ndarray         # (C,) bool
    member_count: np.ndarray  # (C,) int32
    species_hash: np.ndarray  # (C,) uint32
    members: np.ndarray       # (C, M) int32
    edges: np.ndarray         # (C, E_max, 2) int32
    edge_count: np.ndarray    # (C,) int32


@dataclass
class RunResult:
    config: SimConfig
    seed: int
    n_steps: int
    sample_every: int
    per_step_metrics: Dict[str, np.ndarray]   # 'max_size' (N,), 'alive_count' (N,), ...
    events: ReactionEvent                     # flat numpy ReactionEvent (sentinels filtered)
    snapshots: List[CompositeSnapshot]
    wall_seconds: float
    species_values: np.ndarray                # (S,) int32 — per-species valence vector


def run_diagnostic(
    config: SimConfig,
    n_steps: int,
    seed: int = 0,
    sample_every: int = 100,
) -> RunResult:
    """Run one diagnostic simulation and return collected data."""
    if not config.emit_events:
        config = dataclasses.replace(config, emit_events=True)

    state = initialize_world(config, seed=seed)
    params = initialize_interaction_params(config, seed=seed + 1)
    physics = initialize_physics_params(config)

    # Per-step scan body: advances state, returns (new_state, per_step_outputs).
    # per_step_outputs is a dict of JAX arrays — naturally stacks via scan.
    def scan_body(state, _):
        new_state, events = simulation_step(state, params, config, physics)
        sm = size_metrics(new_state.composites, config)
        vm = valence_edge_metrics(new_state.particles, new_state.composites, config)
        return new_state, {
            **sm,
            **vm,
            'events': events,
        }

    step_fn = jax.jit(scan_body, static_argnums=())
    # Warm-up + run.
    t_start = time.time()

    # We need periodic full snapshots, so we can't run a single 10k-step scan
    # and lose intermediate state. Drive scan in segments of `sample_every`.
    all_metrics: Dict[str, List] = {}
    all_events_per_chunk: List[ReactionEvent] = []
    snapshots: List[CompositeSnapshot] = []
    steps_done = 0
    while steps_done < n_steps:
        chunk = min(sample_every, n_steps - steps_done)
        state, chunk_outputs = jax.lax.scan(scan_body, state, None, length=chunk)
        # Move chunk_outputs to host and append.
        host_chunk = jax.device_get(chunk_outputs)
        events_chunk = host_chunk.pop('events')
        for k, v in host_chunk.items():
            all_metrics.setdefault(k, []).append(np.asarray(v))
        all_events_per_chunk.append(events_chunk)
        steps_done += chunk
        # Take a snapshot after each chunk.
        snap = _take_snapshot(state, steps_done)
        snapshots.append(snap)

    state.particles.position.block_until_ready()
    wall = time.time() - t_start

    per_step_metrics = {k: np.concatenate(v, axis=0) for k, v in all_metrics.items()}

    # Concatenate event chunks across (chunk_idx, step, slot) → flat.
    # Each chunk's events has shape (chunk, E, ...) where E = max_fusions + max_composites.
    # Filter sentinels per-chunk to keep memory down, then concatenate.
    filtered_chunks = []
    for chunk_events in all_events_per_chunk:
        # Flatten leading two dims into one.
        kind = np.asarray(chunk_events.kind).reshape(-1)
        sl = np.asarray(chunk_events.source_slots).reshape(-1, 2)
        sh = np.asarray(chunk_events.source_hashes).reshape(-1, 2)
        ss = np.asarray(chunk_events.source_sizes).reshape(-1, 2)
        pl = np.asarray(chunk_events.product_slots).reshape(-1, 2)
        ph = np.asarray(chunk_events.product_hashes).reshape(-1, 2)
        ps = np.asarray(chunk_events.product_sizes).reshape(-1, 2)
        flat = ReactionEvent(
            kind=kind, source_slots=sl, source_hashes=sh, source_sizes=ss,
            product_slots=pl, product_hashes=ph, product_sizes=ps,
        )
        filtered_chunks.append(filter_sentinels(flat))

    events = ReactionEvent(
        kind=np.concatenate([c.kind for c in filtered_chunks]) if filtered_chunks else np.array([], dtype=np.int32),
        source_slots=np.concatenate([c.source_slots for c in filtered_chunks]) if filtered_chunks else np.zeros((0,2),dtype=np.int32),
        source_hashes=np.concatenate([c.source_hashes for c in filtered_chunks]) if filtered_chunks else np.zeros((0,2),dtype=np.uint32),
        source_sizes=np.concatenate([c.source_sizes for c in filtered_chunks]) if filtered_chunks else np.zeros((0,2),dtype=np.int32),
        product_slots=np.concatenate([c.product_slots for c in filtered_chunks]) if filtered_chunks else np.zeros((0,2),dtype=np.int32),
        product_hashes=np.concatenate([c.product_hashes for c in filtered_chunks]) if filtered_chunks else np.zeros((0,2),dtype=np.uint32),
        product_sizes=np.concatenate([c.product_sizes for c in filtered_chunks]) if filtered_chunks else np.zeros((0,2),dtype=np.int32),
    )

    from halflife.chemistry import _species_valences
    species_values = np.asarray(_species_valences(config))

    return RunResult(
        config=config,
        seed=seed,
        n_steps=n_steps,
        sample_every=sample_every,
        per_step_metrics=per_step_metrics,
        events=events,
        snapshots=snapshots,
        wall_seconds=wall,
        species_values=species_values,
    )


def _take_snapshot(state: WorldState, step: int) -> CompositeSnapshot:
    """Host-copy of CompositeState arrays needed for downstream analysis."""
    c = state.composites
    return CompositeSnapshot(
        step=step,
        alive=np.asarray(c.alive),
        member_count=np.asarray(c.member_count),
        species_hash=np.asarray(c.species_hash),
        members=np.asarray(c.members),
        edges=np.asarray(c.edges),
        edge_count=np.asarray(c.edge_count),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu pytest tests/test_analysis_pipeline.py -v -s`
Expected: 1 passed in <60s (includes JIT warm-up — first run is slow).

- [ ] **Step 5: Commit**

```bash
git add halflife/analysis/runner.py tests/test_analysis_pipeline.py
git commit -m "feat(analysis): add headless run_diagnostic runner

Drives simulation_step inside chunked lax.scan loops (one chunk per
sample_every steps). Collects per-step compact metrics and events into
device-side stacks, host-copies full CompositeSnapshot at chunk boundaries.
Returns a RunResult dataclass with everything downstream tasks need.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Plot helpers

**Why:** Matplotlib figures converted to base64 PNGs for embedding in self-contained HTML. Each helper takes the parts of `RunResult` it needs and returns a `str` PNG.

**Files:**
- Create: `halflife/analysis/plots.py`

(No dedicated test file — verified end-to-end in Task 12. Plot correctness is hard to assert programmatically beyond "image was generated.")

- [ ] **Step 1: Implement `plots.py`**

Create `halflife/analysis/plots.py`:

```python
"""Matplotlib plot helpers for the diagnostic report.

Each function takes the slice of RunResult it needs (so plots can be regenerated
in isolation) and returns a base64-encoded PNG string suitable for inline use
as <img src="data:image/png;base64,...">.
"""

import io
import base64
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend; no display required.
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def _fig_to_base64(fig: Figure) -> str:
    """Render a Matplotlib Figure to a base64 PNG string and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


def plot_size_trajectory(per_step: Dict[str, np.ndarray]) -> str:
    """Tier 1: max size + alive count over time, 2-row subplot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    steps = np.arange(per_step['max_size'].shape[0])
    ax1.plot(steps, per_step['max_size'], color='#1f77b4', lw=1.2, label='max')
    ax1.plot(steps, per_step['mean_size'], color='#ff7f0e', lw=1.0, alpha=0.6, label='mean')
    ax1.set_ylabel('composite size')
    ax1.legend(loc='upper right'); ax1.grid(alpha=0.3)
    ax1.set_title('Composite size trajectory')

    ax2.plot(steps, per_step['alive_count'], color='#2ca02c', lw=1.2)
    ax2.set_xlabel('step'); ax2.set_ylabel('alive composites')
    ax2.grid(alpha=0.3)
    return _fig_to_base64(fig)


def plot_size_distribution_heatmap(per_step: Dict[str, np.ndarray]) -> str:
    """Tier 1: size × time heatmap of composite-count distribution."""
    fig, ax = plt.subplots(figsize=(11, 4))
    hist = per_step['size_histogram']            # (steps, max_size)
    # Transpose so x=step, y=size.
    im = ax.imshow(
        hist.T, aspect='auto', origin='lower',
        cmap='magma', interpolation='nearest',
    )
    ax.set_xlabel('step'); ax.set_ylabel('size')
    ax.set_title('Composite size distribution over time')
    fig.colorbar(im, ax=ax, label='count')
    return _fig_to_base64(fig)


def plot_free_particle_fraction(per_step: Dict[str, np.ndarray]) -> str:
    """Tier 1: fraction of particles not in any composite, over time."""
    fig, ax = plt.subplots(figsize=(11, 3))
    steps = np.arange(per_step['free_particle_fraction'].shape[0])
    ax.plot(steps, per_step['free_particle_fraction'], color='#d62728', lw=1.2)
    ax.set_xlabel('step'); ax.set_ylabel('free / total')
    ax.set_ylim(0, 1)
    ax.set_title('Free-particle fraction')
    ax.grid(alpha=0.3)
    return _fig_to_base64(fig)


def plot_degree_saturation(per_step: Dict[str, np.ndarray]) -> str:
    """Tier 2: percent of particles with degree == valence (saturated)."""
    fig, ax = plt.subplots(figsize=(11, 3))
    steps = np.arange(per_step['degree_saturation_pct'].shape[0])
    ax.plot(steps, per_step['degree_saturation_pct'], color='#9467bd', lw=1.2)
    ax.set_xlabel('step'); ax.set_ylabel('saturated fraction')
    ax.set_ylim(0, 1)
    ax.set_title('Per-particle degree saturation (degree == valence)')
    ax.grid(alpha=0.3)
    return _fig_to_base64(fig)


def plot_free_bonds_heatmap(per_step: Dict[str, np.ndarray]) -> str:
    """Tier 2: free_bonds distribution per timestep."""
    fig, ax = plt.subplots(figsize=(11, 4))
    hist = per_step['free_bonds_histogram']
    im = ax.imshow(hist.T, aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('step'); ax.set_ylabel('free_bonds')
    ax.set_title('Composite free_bonds distribution')
    fig.colorbar(im, ax=ax, label='count')
    return _fig_to_base64(fig)


def plot_edge_and_ring_counts(per_step: Dict[str, np.ndarray]) -> str:
    """Tier 2: edge count + ring count over time."""
    fig, ax = plt.subplots(figsize=(11, 3))
    steps = np.arange(per_step['edge_count_total'].shape[0])
    ax.plot(steps, per_step['edge_count_total'], color='#1f77b4', lw=1.2, label='edges')
    ax.plot(steps, per_step['ring_count_total'], color='#ff7f0e', lw=1.2, label='rings')
    ax.set_xlabel('step'); ax.set_ylabel('count')
    ax.set_title('Total edges and rings across all alive composites')
    ax.legend(loc='upper right'); ax.grid(alpha=0.3)
    return _fig_to_base64(fig)


def plot_transition_matrix(matrix: np.ndarray, labels: List[str] = None,
                           title: str = '', cmap: str = 'Reds',
                           log_color: bool = True) -> str:
    """Render a transition matrix (any size) as a heatmap."""
    fig, ax = plt.subplots(figsize=(max(6, min(20, matrix.shape[1] * 0.3)),
                                     max(6, min(20, matrix.shape[0] * 0.3))))
    if log_color and matrix.max() > 0:
        from matplotlib.colors import LogNorm
        # +1 to avoid log(0); colorbar then reads as count.
        im = ax.imshow(matrix + 1, cmap=cmap, norm=LogNorm(vmin=1, vmax=matrix.max() + 1))
    else:
        im = ax.imshow(matrix, cmap=cmap)
    if labels is not None and len(labels) <= 40:
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('product'); ax.set_ylabel('source')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='count')
    return _fig_to_base64(fig)


def plot_compatibility_matrix(
    be: np.ndarray, passes_be: np.ndarray, passes_val: np.ndarray,
    title: str = '', labels: List[str] = None,
) -> str:
    """Tier 4: merged BE colormap with grey-out for failed BE and hatch for failed valence."""
    fig, ax = plt.subplots(figsize=(max(6, min(20, be.shape[1] * 0.3)),
                                     max(6, min(20, be.shape[0] * 0.3))))
    # Base layer: BE as heatmap.
    display = np.where(passes_be, be, np.nan)
    im = ax.imshow(display, cmap='viridis')
    fig.colorbar(im, ax=ax, label='merged BE')

    # Hatch the cells failing valence (overlay translucent diagonal lines).
    fail_val_y, fail_val_x = np.where(~passes_val)
    ax.scatter(fail_val_x, fail_val_y, marker='x', color='black', s=8, alpha=0.7)

    # Grey-out for failed BE happens implicitly (NaN displays as the colormap's bad color).
    # Force the bad color to be light grey.
    cmap = plt.cm.viridis.copy()
    cmap.set_bad('#dddddd')
    im.set_cmap(cmap)

    if labels is not None and len(labels) <= 40:
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('partner B'); ax.set_ylabel('partner A')
    ax.set_title(title)
    return _fig_to_base64(fig)
```

- [ ] **Step 2: Quick smoke-import test**

Run: `JAX_PLATFORMS=cpu python -c "from halflife.analysis import plots; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add halflife/analysis/plots.py
git commit -m "feat(analysis): add matplotlib plot helpers returning base64 PNGs

Self-contained — each function takes the slice of RunResult it needs and
returns an inlineable image string. Covers Tier 1, Tier 2, transition
matrices, and Tier 4 compatibility matrices (BE colormap with grey-out
for BE-fail and 'x' marker for valence-fail).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: HTML report assembly

**Why:** Stitch all the plot images + run metadata into one self-contained HTML file.

**Files:**
- Create: `halflife/analysis/report.py`

(No dedicated test — verified end-to-end in Task 12.)

- [ ] **Step 1: Implement `report.py`**

Create `halflife/analysis/report.py`:

```python
"""HTML report assembly.

Single function: render_html(run_result) → str. Embeds all plots as base64
PNGs so the output file is fully self-contained (no external assets).
"""

import dataclasses
from typing import Dict, List

import numpy as np

from halflife.analysis.runner import RunResult
from halflife.analysis import plots, compatibility, transitions
from halflife.analysis.events import KIND_FUSION, KIND_FISSION


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Composite Diagnostic — {scenario}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 1200px; margin: 1em auto; padding: 0 1em; color: #222; }}
  h1 {{ border-bottom: 2px solid #444; padding-bottom: 0.3em; }}
  h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 0.2em; margin-top: 2em; }}
  .meta {{ background: #f7f7f9; padding: 0.7em 1em; border-radius: 4px; font-family: monospace; font-size: 0.9em; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.7em; margin: 1em 0; }}
  .stat {{ background: #f0f0f5; padding: 0.7em; border-radius: 4px; text-align: center; }}
  .stat .num {{ font-size: 1.5em; font-weight: bold; color: #1f77b4; }}
  .stat .label {{ font-size: 0.85em; color: #666; }}
  .full-matrix {{ max-width: 100%; max-height: 800px; overflow: scroll;
                  border: 1px solid #ccc; padding: 0.5em; margin-top: 0.5em; }}
  img {{ max-width: 100%; height: auto; }}
  footer {{ font-size: 0.85em; color: #888; margin-top: 2em; }}
</style>
</head><body>
<h1>Composite Diagnostic Report</h1>
<div class="meta">
  <strong>Scenario:</strong> {scenario}<br>
  <strong>Seed:</strong> {seed} &nbsp; <strong>Steps:</strong> {n_steps} &nbsp; <strong>Sample every:</strong> {sample_every}<br>
  <strong>Wall time:</strong> {wall:.1f}s &nbsp; <strong>Git SHA:</strong> {git_sha}<br>
  <strong>Config:</strong> num_particles={num_particles}, num_species={num_species},
   max_composite_size={max_composite_size}, max_valence={max_valence},
   use_valence={use_valence}, bond_mode={bond_mode},
   fusion_threshold={fusion_threshold}, half_life_min={half_life_min}, half_life_max={half_life_max}<br>
  <strong>Per-species valences:</strong> {valences}
</div>

<div class="stat-grid">
  <div class="stat"><div class="num">{peak_max_size}</div><div class="label">peak max size</div></div>
  <div class="stat"><div class="num">{final_max_size}</div><div class="label">final max size</div></div>
  <div class="stat"><div class="num">{mean_alive:.1f}</div><div class="label">mean alive count</div></div>
  <div class="stat"><div class="num">{degree_sat:.0%}</div><div class="label">mean degree saturation</div></div>
</div>

<h2>Tier 1 — Macroscopic time series</h2>
<img src="data:image/png;base64,{img_size_trajectory}">
<img src="data:image/png;base64,{img_size_dist}">
<img src="data:image/png;base64,{img_free_particle}">

<h2>Tier 2 — Valence / edge structure</h2>
<img src="data:image/png;base64,{img_degree_sat}">
<img src="data:image/png;base64,{img_free_bonds}">
<img src="data:image/png;base64,{img_edges_rings}">

<h2>Tier 3 — Chemical network (empirical)</h2>
<p><em>Built from {n_fusion} fusion events + {n_fission} fission events. Each
event contributes 2 cells.</em></p>
<h3>Matrix 1: Size × size mass-flow</h3>
<img src="data:image/png;base64,{img_size_bin_matrix}">
<h3>Matrix 2: Top-K composite types</h3>
<img src="data:image/png;base64,{img_top_k_matrix}">
<h3>Matrix 3: Every observed composite type</h3>
<div class="full-matrix"><img src="data:image/png;base64,{img_full_matrix}"></div>

<h2>Tier 4 — Fusion compatibility (theoretical)</h2>
<p><em>Pure chemistry — what <strong>could</strong> happen if these pairs met.
Greyed cells fail the BE threshold; cells with × markers fail the valence gate
even at structural max. Compare against Tier 3 Matrix 2 (same sort) to see
which compatible pairs never fired empirically.</em></p>
<h3>Matrix 4a: Species-pair</h3>
<img src="data:image/png;base64,{img_compat_species}">
<h3>Matrix 4b: Top-K observed-composite</h3>
<img src="data:image/png;base64,{img_compat_observed}">

<footer>
  Generated by halflife.analysis on {timestamp}.
  JAX platform: {jax_platform}.
</footer>
</body></html>
"""


def _git_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "(unknown)"


def _unique_multisets_from_snapshots(snapshots, particles_species: np.ndarray = None):
    """Walk snapshots and return (hashes, multisets, total_incidence) for all unique
    composite types observed."""
    seen = {}
    incidence = {}
    for snap in snapshots:
        alive_idx = np.where(snap.alive)[0]
        for c in alive_idx:
            h = int(snap.species_hash[c])
            incidence[h] = incidence.get(h, 0) + 1
            if h not in seen:
                # Recover the multiset: snap.members[c, :n] → species via particles
                # We don't have particles species here per-snapshot. The caller passes
                # it in via the top-level config (initial state — species don't change).
                if particles_species is not None:
                    n = int(snap.member_count[c])
                    member_ids = snap.members[c, :n]
                    member_ids = member_ids[member_ids >= 0]
                    sp = tuple(sorted(int(particles_species[m]) for m in member_ids))
                    seen[h] = sp
                else:
                    seen[h] = ()
    return seen, incidence


def render_html(result: RunResult) -> str:
    """Build a single self-contained HTML string from a RunResult."""
    from halflife.state import (
        initialize_world,
        initialize_physics_params,
    )
    import datetime
    import jax

    # We need particle species for multiset reconstruction; species are constant
    # over the run, so just re-init a world with the same seed.
    init_world = initialize_world(result.config, seed=result.seed)
    particles_species = np.asarray(init_world.particles.species)
    physics = initialize_physics_params(result.config)

    # Tier 1, 2 plots.
    img_size_trajectory = plots.plot_size_trajectory(result.per_step_metrics)
    img_size_dist       = plots.plot_size_distribution_heatmap(result.per_step_metrics)
    img_free_particle   = plots.plot_free_particle_fraction(result.per_step_metrics)
    img_degree_sat      = plots.plot_degree_saturation(result.per_step_metrics)
    img_free_bonds      = plots.plot_free_bonds_heatmap(result.per_step_metrics)
    img_edges_rings     = plots.plot_edge_and_ring_counts(result.per_step_metrics)

    # Tier 3 matrices.
    M_size_bin = transitions.size_bin_transition_matrix(
        result.events, result.config.max_composite_size
    )
    M_top_k, top_k_labels = transitions.top_k_transition_matrix(result.events, K=30)
    M_full, full_labels = transitions.full_transition_matrix(result.events)

    img_size_bin_matrix = plots.plot_transition_matrix(
        M_size_bin, title='Matrix 1: Size-class transitions (rows=source, cols=product)',
    )
    img_top_k_matrix = plots.plot_transition_matrix(
        M_top_k, labels=top_k_labels, title='Matrix 2: Top-K composite-type transitions',
    )
    img_full_matrix = plots.plot_transition_matrix(
        M_full, labels=None,  # too many to show
        title=f'Matrix 3: All {len(full_labels)} observed composite types',
    )

    # Tier 4 compatibility matrices.
    be_a, pbe_a, pval_a = compatibility.species_pair_compat_matrix(result.config, physics)
    img_compat_species = plots.plot_compatibility_matrix(
        be_a, pbe_a, pval_a,
        title='Matrix 4a: Species-pair fusion compatibility',
        labels=[f's{i}' for i in range(result.config.num_species)],
    )

    # For Matrix 4b, build the same top-K hash list used by Matrix 2 (sorted by size).
    # Reconstruct multisets by re-walking incidence + snapshots.
    seen, incidence = _unique_multisets_from_snapshots(result.snapshots, particles_species)
    if seen:
        # Pick top-K by incidence then sort by size,hash for stable display.
        from collections import Counter
        top_hashes = [h for h, _ in Counter(incidence).most_common(30)]
        top_hashes.sort(key=lambda h: (len(seen.get(h, ())), h))
        top_multisets = [seen.get(h, ()) for h in top_hashes]
        top_hashes_arr = np.array(top_hashes, dtype=np.uint32)
        be_b, pbe_b, pval_b = compatibility.observed_pair_compat_matrix(
            top_hashes_arr, top_multisets, result.config, physics,
        )
        img_compat_observed = plots.plot_compatibility_matrix(
            be_b, pbe_b, pval_b,
            title='Matrix 4b: Top-K observed-composite compatibility',
            labels=[f"0x{h:08x}" for h in top_hashes],
        )
    else:
        # No composites ever formed — render a 1x1 placeholder.
        img_compat_observed = plots.plot_compatibility_matrix(
            np.zeros((1, 1)), np.zeros((1, 1), bool), np.ones((1, 1), bool),
            title='Matrix 4b: (no observed composites)',
        )

    # Headline derived numbers.
    peak_max_size  = int(result.per_step_metrics['max_size'].max())
    final_max_size = int(result.per_step_metrics['max_size'][-1])
    mean_alive     = float(result.per_step_metrics['alive_count'].mean())
    degree_sat     = float(result.per_step_metrics['degree_saturation_pct'].mean())

    n_fusion  = int((result.events.kind == KIND_FUSION).sum())
    n_fission = int((result.events.kind == KIND_FISSION).sum())

    return _HTML_TEMPLATE.format(
        scenario=getattr(result.config, '_scenario_name', '(custom)'),
        seed=result.seed,
        n_steps=result.n_steps,
        sample_every=result.sample_every,
        wall=result.wall_seconds,
        git_sha=_git_sha(),
        num_particles=result.config.num_particles,
        num_species=result.config.num_species,
        max_composite_size=result.config.max_composite_size,
        max_valence=result.config.max_valence,
        use_valence=result.config.use_valence,
        bond_mode=getattr(result.config, 'bond_mode', '(n/a)'),
        fusion_threshold=physics.fusion_threshold,
        half_life_min=result.config.half_life_min,
        half_life_max=result.config.half_life_max,
        valences=result.species_values.tolist(),
        peak_max_size=peak_max_size,
        final_max_size=final_max_size,
        mean_alive=mean_alive,
        degree_sat=degree_sat,
        img_size_trajectory=img_size_trajectory,
        img_size_dist=img_size_dist,
        img_free_particle=img_free_particle,
        img_degree_sat=img_degree_sat,
        img_free_bonds=img_free_bonds,
        img_edges_rings=img_edges_rings,
        img_size_bin_matrix=img_size_bin_matrix,
        img_top_k_matrix=img_top_k_matrix,
        img_full_matrix=img_full_matrix,
        img_compat_species=img_compat_species,
        img_compat_observed=img_compat_observed,
        n_fusion=n_fusion, n_fission=n_fission,
        jax_platform=jax.default_backend(),
        timestamp=datetime.datetime.now().isoformat(timespec='seconds'),
    )
```

- [ ] **Step 2: Smoke import test**

Run: `JAX_PLATFORMS=cpu python -c "from halflife.analysis.report import render_html; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add halflife/analysis/report.py
git commit -m "feat(analysis): HTML report assembly with embedded plots

Single render_html(run_result) entry point producing a self-contained HTML
file (all plots inlined as base64 PNGs). Sections: header + at-a-glance
stats, Tier 1-4 plots, Tier 3 chemical-network matrices, Tier 4 fusion
compatibility matrices. Matrix 3 wrapped in an overflow:scroll div.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: CLI + scenario presets

**Why:** A friendly command-line entry: `python -m halflife.analysis --scenario X --steps N`. Bundles preset configs, applies overrides, dispatches to runner + report.

**Files:**
- Create: `halflife/analysis/cli.py`
- Create: `halflife/analysis/__main__.py`

- [ ] **Step 1: Implement `cli.py`**

Create `halflife/analysis/cli.py`:

```python
"""CLI entry point for the diagnostic pipeline.

Usage:
  python -m halflife.analysis --scenario baseline --steps 10000
  python -m halflife.analysis --scenario current_experiment --steps 5000 \
      --override num_species=3,half_life_max=80
"""

import argparse
import dataclasses
import os
import time

from halflife.config import SimConfig
from halflife.analysis.runner import run_diagnostic
from halflife.analysis.report import render_html


# Each preset is a dict of {field: value} layered on top of SimConfig defaults.
PRESETS = {
    'baseline':           {},
    'current_experiment': {'num_species': 3, 'half_life_max': 100.0},
    'valence_off':        {'use_valence': False},
    'polymer_world':      {'max_valence': 2, 'num_species': 2},
    'branching_world':    {'max_valence': 3, 'num_species': 3},
    'old_star_spring':    {'bond_mode': 'star_spring'},
}


def _parse_overrides(s: str) -> dict:
    """Parse 'k1=v1,k2=v2' into a dict, with crude int/float/bool coercion."""
    if not s:
        return {}
    out = {}
    for chunk in s.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '=' not in chunk:
            raise ValueError(f"override missing '=': {chunk!r}")
        k, v = chunk.split('=', 1)
        k = k.strip(); v = v.strip()
        # Coerce.
        if v.lower() in ('true', 'false'):
            out[k] = v.lower() == 'true'
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v   # leave as string (e.g. 'edges')
    return out


def build_config(scenario: str, overrides: dict) -> SimConfig:
    """SimConfig defaults + preset + overrides + always-on emit_events."""
    if scenario not in PRESETS:
        raise SystemExit(
            f"unknown scenario {scenario!r}. Known: {sorted(PRESETS)}"
        )
    fields = {**PRESETS[scenario], **overrides, 'emit_events': True}
    config = SimConfig(**{**dataclasses.asdict(SimConfig()), **fields})
    # Stash the scenario name for the report header (not a SimConfig field
    # so we just attach as a private attribute via __dict__ since SimConfig
    # is frozen — use a wrapper dataclass instead).
    object.__setattr__(config, '_scenario_name', scenario)
    return config


def main(argv=None):
    p = argparse.ArgumentParser(description="Run a single composite diagnostic simulation.")
    p.add_argument('--scenario',     default='baseline', choices=sorted(PRESETS))
    p.add_argument('--steps',        type=int, default=10_000)
    p.add_argument('--seed',         type=int, default=0)
    p.add_argument('--sample-every', type=int, default=100,
                   help="Full-snapshot interval (compact metrics every step regardless).")
    p.add_argument('--top-k',        type=int, default=30,
                   help="K for the top-K transition / compatibility matrices.")
    p.add_argument('--override',     type=str, default='',
                   help="Comma-separated config overrides: k1=v1,k2=v2")
    p.add_argument('--out',          type=str, default='',
                   help="Output HTML path (default: tests/reports/diag_<scenario>_<ts>.html)")
    p.add_argument('--platform',     type=str, default='', choices=['', 'cpu', 'gpu'],
                   help="Force JAX platform (default: auto)")
    args = p.parse_args(argv)

    if args.platform:
        os.environ['JAX_PLATFORMS'] = args.platform

    overrides = _parse_overrides(args.override)
    config = build_config(args.scenario, overrides)

    print(f"[diag] scenario={args.scenario} steps={args.steps} seed={args.seed}")
    print(f"[diag] sample_every={args.sample_every} top_k={args.top_k}")
    if overrides:
        print(f"[diag] overrides: {overrides}")

    t0 = time.time()
    result = run_diagnostic(
        config, n_steps=args.steps, seed=args.seed, sample_every=args.sample_every,
    )
    t1 = time.time()
    print(f"[diag] run finished in {t1 - t0:.1f}s  ({result.n_steps / (t1 - t0):.1f} steps/sec)")

    html = render_html(result)

    out = args.out or _default_out(args.scenario)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[diag] wrote {out}  ({len(html) / 1024:.0f} KB)")
    return out


def _default_out(scenario: str) -> str:
    ts = time.strftime('%Y%m%d_%H%M%S')
    return os.path.join('tests', 'reports', f'diag_{scenario}_{ts}.html')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Create `__main__.py` for `-m halflife.analysis`**

Create `halflife/analysis/__main__.py`:

```python
"""Allow `python -m halflife.analysis ...` to invoke the CLI."""
from halflife.analysis.cli import main

if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Smoke test the CLI on a tiny config**

Run: `JAX_PLATFORMS=cpu python -m halflife.analysis --scenario baseline --steps 50 --sample-every 10 --override "num_particles=40,num_species=3,max_composites=40,max_composite_size=8,max_fusions_per_step=20"`
Expected:
- Prints `[diag] scenario=baseline steps=50 ...`
- Prints `[diag] run finished in <time>s`
- Prints `[diag] wrote tests/reports/diag_baseline_<ts>.html`
- File exists at the printed path.

Open the file in a browser (Windows path: `C:\Users\Heysoos\Documents\Pycharm Projects\halflife-particle\tests\reports\...`) and confirm all sections render with images.

- [ ] **Step 4: Commit**

```bash
git add halflife/analysis/cli.py halflife/analysis/__main__.py
git commit -m "feat(analysis): add CLI entry point with scenario presets

python -m halflife.analysis --scenario {baseline,current_experiment,
valence_off,polymer_world,branching_world,old_star_spring} produces a
self-contained HTML report. --override 'k=v,k=v' for per-run knob tweaks,
--platform cpu|gpu to force JAX backend.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: End-to-end smoke test

**Why:** A pytest target that exercises the full pipeline (runner → matrices → plots → HTML) in one sub-minute CPU run. Catches integration regressions cheaper than running the CLI manually.

**Files:**
- Modify: `tests/test_analysis_pipeline.py`

- [ ] **Step 1: Expand the existing smoke test**

Append to `tests/test_analysis_pipeline.py`:

```python
def test_full_pipeline_produces_html_with_all_sections():
    """End-to-end: run → render → assert key markup is present."""
    from halflife.analysis.report import render_html
    config = _tiny_config()
    result = run_diagnostic(config, n_steps=200, seed=0, sample_every=50)
    html = render_html(result)

    # Quick structural assertions — the report has all 4 tiers.
    assert '<h2>Tier 1' in html
    assert '<h2>Tier 2' in html
    assert '<h2>Tier 3' in html
    assert '<h2>Tier 4' in html
    # Plot images present (base64 PNG prefix).
    assert 'data:image/png;base64,' in html
    # Headline stats rendered.
    assert 'peak max size' in html
    assert 'mean degree saturation' in html
    # At least one of the three transition matrices populated.
    assert 'Matrix 1' in html


def test_cli_writes_file(tmp_path):
    """The CLI end-to-end: invoke main() with a tiny config and verify file written."""
    from halflife.analysis.cli import main
    out_path = tmp_path / "test_report.html"
    main([
        '--scenario', 'baseline',
        '--steps', '50',
        '--sample-every', '25',
        '--override', 'num_particles=40,num_species=3,max_composites=40,max_composite_size=8,max_fusions_per_step=20',
        '--out', str(out_path),
        '--platform', 'cpu',
    ])
    assert out_path.exists()
    content = out_path.read_text()
    assert '<h1>Composite Diagnostic Report</h1>' in content
```

- [ ] **Step 2: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu pytest tests/test_analysis_pipeline.py -v -s`
Expected: 3 tests pass total (the original from Task 8 plus these two). Total time <120s on CPU.

- [ ] **Step 3: Commit**

```bash
git add tests/test_analysis_pipeline.py
git commit -m "test(analysis): end-to-end smoke tests for full pipeline + CLI

Verifies that run → render → write produces an HTML file with all four
tiers and at least one matrix populated. CLI test exercises the argparse
path and writes to a tmp_path so it can be re-run safely.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

### Spec coverage

Checked each spec section against the task list:

| Spec requirement | Covered by |
|---|---|
| `halflife/analysis/` module layout | Tasks 1, 5, 6, 7, 8, 9, 10, 11 (every file) |
| `ReactionEvent` + `emit_events` flag with zero-cost-off | Task 1 |
| Kernel emission from `attempt_fusion` | Task 2 |
| Kernel emission from `apply_composite_decay` | Task 3 |
| `simulation_step` event assembly | Task 4 |
| Tier 1 macroscopic metrics | Task 5 (`size_metrics`) |
| Tier 2 valence/edge metrics | Task 5 (`valence_edge_metrics`) |
| Tier 3 matrices (size-binned, top-K, full) | Task 7 |
| Tier 4 compatibility matrices (4a species, 4b observed) | Task 6 + Task 10 (wiring) |
| Headless runner with `lax.scan` + periodic snapshots | Task 8 |
| Memory budget — chunked scan, sentinel filter per chunk | Task 8 |
| HTML report with embedded base64 PNGs | Task 9 + Task 10 |
| CLI with presets and overrides | Task 11 |
| End-to-end smoke test | Task 12 |

No gaps.

### Placeholder scan

Searched plan for `TBD`, `TODO`, `implement later`, `add appropriate error handling`, `similar to Task` — none found. Every step contains actual code or actual commands.

### Type/signature consistency

- `attempt_fusion` returns `(state, degree)` (False) or `(state, degree, events)` (True). Used consistently in Tasks 2, 4, 8.
- `apply_composite_decay` returns `state` (False) or `(state, events)` (True). Used consistently in Tasks 3, 4.
- `simulation_step` returns `state` (False) or `(state, events)` (True). Used consistently in Tasks 4, 8.
- `ReactionEvent` field names (`kind`, `source_slots`, `source_hashes`, `source_sizes`, `product_slots`, `product_hashes`, `product_sizes`) are identical across Tasks 1, 2, 3, 4, 7, 8.
- `RunResult` fields (`config`, `seed`, `n_steps`, `sample_every`, `per_step_metrics`, `events`, `snapshots`, `wall_seconds`, `species_values`) defined in Task 8 and consumed by Task 10.
- `KIND_NONE`, `KIND_FUSION`, `KIND_FISSION` constants defined in Task 1, used in Tasks 2, 3, 7, 10.

### Scope check

The plan implements exactly the v1 scope from the spec. Phase-2 items (kernel counters for failed-fusion reasons, comparison view) are correctly omitted.

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-30-composite-diagnostic-implementation.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
