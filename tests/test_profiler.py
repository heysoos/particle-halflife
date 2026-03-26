import pytest
from halflife.profiler import ProfileMetrics, CCFusionEvent
import numpy as np


def test_profile_metrics_init():
    """ProfileMetrics initializes with empty event lists."""
    metrics = ProfileMetrics()
    assert metrics.cc_fusion_events == []
    assert metrics.composite_size_samples == []
    assert metrics.max_composite_size_observed == 0
    assert metrics.cc_fusion_count == 0


def test_record_cc_fusion_event():
    """ProfileMetrics records C+C fusion events correctly."""
    metrics = ProfileMetrics()

    event = CCFusionEvent(
        step=10,
        composite_a_id=5,
        composite_b_id=7,
        a_members=3,
        b_members=2,
        a_be=0.6,
        b_be=0.4,
        merged_be=0.55,
        merged_members=5,
    )

    metrics.record_cc_fusion(event)

    assert metrics.cc_fusion_count == 1
    assert len(metrics.cc_fusion_events) == 1
    assert metrics.cc_fusion_events[0].composite_a_id == 5
    assert metrics.cc_fusion_events[0].merged_members == 5


def test_be_statistics():
    """ProfileMetrics computes BE statistics correctly."""
    metrics = ProfileMetrics()

    metrics.record_cc_fusion(CCFusionEvent(10, 0, 1, 2, 2, 0.6, 0.4, 0.55, 4))
    metrics.record_cc_fusion(CCFusionEvent(11, 2, 3, 3, 3, 0.5, 0.5, 0.65, 6))

    mean, min_be, max_be = metrics.get_be_statistics()

    assert mean == pytest.approx(0.6)
    assert min_be == pytest.approx(0.55)
    assert max_be == pytest.approx(0.65)


def test_record_composite_sizes():
    """ProfileMetrics records composite size samples and updates max."""
    metrics = ProfileMetrics()
    distribution1 = np.array([1, 2])
    metrics.record_composite_sizes(0, 5, 2.5, distribution1)
    
    assert len(metrics.composite_size_samples) == 1
    assert metrics.composite_size_samples[0][0] == 0  # step
    assert metrics.composite_size_samples[0][1] == 5  # max_size
    assert metrics.composite_size_samples[0][2] == 2.5  # mean_size
    assert metrics.max_composite_size_observed == 5
    
    distribution2 = np.array([2, 3])
    metrics.record_composite_sizes(1, 8, 3.5, distribution2)
    assert metrics.max_composite_size_observed == 8


def test_get_cc_fusion_rate():
    """ProfileMetrics returns fusion count as float."""
    metrics = ProfileMetrics()
    assert metrics.get_cc_fusion_rate() == 0.0
    
    metrics.record_cc_fusion(CCFusionEvent(0, 0, 1, 2, 2, 0.5, 0.5, 0.6, 4))
    assert metrics.get_cc_fusion_rate() == 1.0
    
    metrics.record_cc_fusion(CCFusionEvent(1, 2, 3, 2, 2, 0.5, 0.5, 0.7, 4))
    assert metrics.get_cc_fusion_rate() == 2.0


def test_be_statistics_single_event():
    """ProfileMetrics handles single event (mean = min = max)."""
    metrics = ProfileMetrics()
    metrics.record_cc_fusion(CCFusionEvent(0, 0, 1, 2, 2, 0.5, 0.5, 0.75, 4))
    mean, min_be, max_be = metrics.get_be_statistics()
    assert mean == pytest.approx(0.75)
    assert min_be == pytest.approx(0.75)
    assert max_be == pytest.approx(0.75)
