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
