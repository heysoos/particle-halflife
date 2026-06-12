"""Unit tests for halflife/analysis/openendedness.py."""
import numpy as np
import pytest

from halflife.analysis import openendedness as oe


def test_slice_windows_equal_count():
    assert oe.slice_windows(15000, windows=5) == [
        (0, 3000), (3000, 6000), (6000, 9000), (9000, 12000), (12000, 15000)
    ]


def test_slice_windows_count_absorbs_remainder():
    # 10 / 3 → last window absorbs the extra step
    assert oe.slice_windows(10, windows=3) == [(0, 3), (3, 6), (6, 10)]


def test_slice_windows_fixed_width():
    assert oe.slice_windows(15000, window_width=4000) == [
        (0, 4000), (4000, 8000), (8000, 12000), (12000, 15000)
    ]


def test_slice_windows_default_is_five():
    assert len(oe.slice_windows(1000)) == 5


def test_slice_windows_single_window():
    assert oe.slice_windows(500, windows=1) == [(0, 500)]


def test_slice_windows_width_ge_nsteps():
    assert oe.slice_windows(500, window_width=9999) == [(0, 500)]


def test_slice_windows_mutually_exclusive():
    with pytest.raises(ValueError):
        oe.slice_windows(1000, windows=5, window_width=100)


def test_window_index_assignment():
    w = [(0, 3000), (3000, 6000), (6000, 9000)]
    assert oe._window_index(0, w) == 0
    assert oe._window_index(2999, w) == 0
    assert oe._window_index(3000, w) == 1
    assert oe._window_index(9000, w) == 2   # final end is inclusive
    assert oe._window_index(99999, w) is None
