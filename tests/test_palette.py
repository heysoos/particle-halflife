"""
tests/test_palette.py — OKLCh species palette sanity checks.

The renderer expects make_species_colors(N) to return a (N, 3) float32 array of
linear-sRGB values in [0, 1] with every row distinct. oklch_to_linear_srgb is
the underlying conversion; we roundtrip a known neutral reference to catch
matrix typos.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from halflife.utils import make_species_colors, oklch_to_linear_srgb


def test_palette_shape_and_dtype():
    colors = make_species_colors(12)
    assert colors.shape == (12, 3)
    assert colors.dtype == np.float32


def test_palette_in_unit_range():
    colors = make_species_colors(32)
    assert (colors >= 0.0).all()
    assert (colors <= 1.0).all()


def test_palette_rows_distinct():
    # With moderate chroma and 24 hues spaced 15° apart, every species must be
    # visibly different — no two rows should be identical.
    colors = make_species_colors(24)
    seen = {tuple(row) for row in colors}
    assert len(seen) == 24


def test_oklch_neutral_is_grayscale():
    # OKLCh with C=0 (no chroma) at any hue should round-trip to a neutral
    # gray in linear sRGB: all three channels equal.
    rgb = oklch_to_linear_srgb(0.5, 0.0, 0.0)
    assert rgb.shape == (3,)
    assert abs(float(rgb[0]) - float(rgb[1])) < 1e-4
    assert abs(float(rgb[1]) - float(rgb[2])) < 1e-4
    # And mid-lightness should land near linear-sRGB ≈ 0.18 (mid gray in linear)
    assert 0.10 < float(rgb[0]) < 0.30


def test_oklch_known_red():
    # OKLCh(0.628, 0.258, 29.234°) is the reference value for sRGB pure red.
    # We allow loose tolerance because our impl clips to gamut and uses float32.
    rgb = oklch_to_linear_srgb(0.628, 0.258, 29.234)
    # Red channel should dominate clearly.
    assert float(rgb[0]) > float(rgb[1])
    assert float(rgb[0]) > float(rgb[2])
    assert float(rgb[0]) > 0.5
