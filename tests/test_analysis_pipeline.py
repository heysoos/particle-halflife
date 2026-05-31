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
    # Headline stats rendered (label text from the stat grid).
    assert 'peak max size' in html
    assert 'mean degree sat' in html
    # Tier 3 toggle UI present.
    assert 'Size \xd7 size' in html
    assert 'Composite \xd7 composite' in html
    # Tier 4 sub-section present.
    assert 'Matrix 4a' in html
    assert 'Matrix 4b' in html


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
