"""End-to-end smoke tests for halflife/analysis. Expanded in Task 12."""
import dataclasses

import numpy as np

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
