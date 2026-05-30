"""Tests for per-step metric collection functions."""
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
