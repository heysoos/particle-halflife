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
