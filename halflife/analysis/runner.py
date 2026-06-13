"""Headless simulation runner for diagnostic analysis.

Single entry point: run_diagnostic(config, n_steps, seed, sample_every) →
RunResult. The runner drives the JIT'd simulation_step inside a lax.scan
that emits per-step compact metrics + events, with periodic host-side
copies of full snapshots for downstream distribution drill-downs.

Caching: save_run_result / load_run_result persist a RunResult to disk
(gzipped pickle). Used by the CLI's --from-cache flag so report-only
iteration doesn't require re-running the simulation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import gzip
import os
import pickle
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
from halflife.analysis.events import filter_sentinels


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

    # Drive scan in segments of `sample_every`. (One big scan would lose
    # intermediate state for snapshots.)
    all_metrics: Dict[str, List] = {}
    all_events_per_chunk: List[ReactionEvent] = []
    snapshots: List[CompositeSnapshot] = []
    steps_done = 0
    t_start = time.time()
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
    # Each chunk's events has shape (chunk, E, ...) where
    # E = min(max_fusions, N) + min(max_fissions, C)
    #     [+ min(max_scissions, C) when bond scission is on] (budget-sized
    # batches; bond-scission events ride the fission kind).
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


# ── Cache I/O ──────────────────────────────────────────────────────────────
# Persisting a RunResult lets the user iterate on report presentation code
# without re-running the (~minutes-long) GPU simulation. Cache files are
# gzipped pickles — the snapshot arrays compress well (lots of -1 sentinels).

def save_run_result(result: RunResult, path: str) -> None:
    """Save a RunResult to a gzipped pickle file. Creates parent dirs."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with gzip.open(path, 'wb') as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_run_result(path: str) -> RunResult:
    """Load a RunResult from a gzipped pickle file."""
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)
