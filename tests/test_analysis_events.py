"""Tests for ReactionEvent data structure and emit_events config flag."""
import jax.numpy as jnp

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


# ── Task 2: fusion event emission ────────────────────────────────────────────

import dataclasses
import jax
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


def _build_neighbors(state, config):
    """Helper: build the cell list + neighbor lookup that attempt_fusion needs."""
    cell_list = build_cell_list(state.particles.position, config)
    return find_all_neighbors(state.particles.position, cell_list, config)


def test_attempt_fusion_returns_events_when_enabled():
    """With emit_events=True, attempt_fusion returns (state, degree, events)."""
    config = _tiny_config()
    state = initialize_world(config, seed=0)
    params = initialize_interaction_params(config, seed=1)
    physics = initialize_physics_params(config)
    neighbors = _build_neighbors(state, config)

    result = attempt_fusion(state, neighbors, params, config, physics)
    assert isinstance(result, tuple)
    assert len(result) == 3, f"expected (state, degree, events), got {len(result)}-tuple"
    _new_state, _degree, events = result
    assert events.kind.shape == (config.max_fusions_per_step,)


def test_attempt_fusion_returns_state_degree_when_disabled():
    """With emit_events=False (default), attempt_fusion keeps its original (state, degree) signature."""
    config = dataclasses.replace(_tiny_config(), emit_events=False)
    state = initialize_world(config, seed=0)
    params = initialize_interaction_params(config, seed=1)
    physics = initialize_physics_params(config)
    neighbors = _build_neighbors(state, config)

    result = attempt_fusion(state, neighbors, params, config, physics)
    assert len(result) == 2, f"expected (state, degree), got {len(result)}-tuple"


def test_attempt_fusion_emits_consistent_fusion_events():
    """If composites grew by N from free+free fusions, at least N fusion events should be in the batch."""
    config = _tiny_config()
    state = initialize_world(config, seed=0)
    params = initialize_interaction_params(config, seed=1)
    physics = initialize_physics_params(config)
    neighbors = _build_neighbors(state, config)

    alive_before = int(state.composites.alive.sum())
    new_state, _, events = attempt_fusion(state, neighbors, params, config, physics)
    alive_after = int(new_state.composites.alive.sum())

    real_events = filter_sentinels(events)
    n_fusion_events = int((real_events.kind == KIND_FUSION).sum())
    # delta composites = (free+free fusions) - (comp+comp absorptions).
    # We can't isolate either from outside, but: every free+free fusion creates
    # a new slot AND emits one event, so the alive-delta is a lower bound on
    # the fusion-event count.
    assert n_fusion_events >= max(0, alive_after - alive_before), \
        f"alive grew by {alive_after - alive_before} but only {n_fusion_events} fusion events emitted"


# ── Task 3: fission event emission ───────────────────────────────────────────


def test_apply_composite_decay_emits_fission_events():
    """When emit_events=True, apply_composite_decay returns (state, events)."""
    import dataclasses
    from halflife.chemistry import apply_composite_decay

    # Tiny config with short half-life so fission fires reliably within ~5 fusion steps + decay.
    config = dataclasses.replace(
        _tiny_config(),
        emit_events=True,
        half_life_min=0.1,    # force rapid decay
        half_life_max=0.5,
    )
    state = initialize_world(config, seed=0)
    physics = initialize_physics_params(config)

    # Build some composites first by running a few fusion steps.
    params = initialize_interaction_params(config, seed=1)
    for _ in range(5):
        neighbors = _build_neighbors(state, config)
        state, _, _ = attempt_fusion(state, neighbors, params, config, physics)
    assert int(state.composites.alive.sum()) > 0, "need composites to test fission"

    result = apply_composite_decay(state, config, physics)
    assert len(result) == 2, f"expected (state, events), got {len(result)}-tuple"
    new_state, events = result
    assert events.kind.shape == (config.max_composites,)

    # Some composites should have died (half-life is 0.1-0.5 vs dt=physics.dt).
    n_died = int(state.composites.alive.sum()) - int(new_state.composites.alive.sum())
    if n_died > 0:
        from halflife.analysis.events import filter_sentinels, KIND_FISSION
        real = filter_sentinels(events)
        n_fission_events = int((real.kind == KIND_FISSION).sum())
        assert n_fission_events >= n_died, \
            f"{n_died} composites died but only {n_fission_events} fission events emitted"


# ── Task 4: simulation_step event wiring ─────────────────────────────────────


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
    import dataclasses
    config = dataclasses.replace(_tiny_config(), emit_events=False)
    state = initialize_world(config, seed=0)
    params = initialize_interaction_params(config, seed=1)
    physics = initialize_physics_params(config)

    result = simulation_step(state, params, config, physics)
    # Just a WorldState, not a tuple.
    from halflife.state import WorldState
    assert isinstance(result, WorldState)
