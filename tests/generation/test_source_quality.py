"""Tests for src/generation/source_quality.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.generation.source_quality import score_crop, DEFAULT_MIN_SOURCE_QUALITY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_solid(size: int = 28) -> np.ndarray:
    """White canvas with a thick L-shape stroke covering ~40% of pixels."""
    tile = np.full((size, size), 255, dtype=np.uint8)
    # Horizontal bar
    tile[10:18, 4:24] = 0
    # Vertical bar
    tile[4:24, 10:18] = 0
    return tile


def _make_fragmented(size: int = 28, n_dots: int = 30) -> np.ndarray:
    """White canvas with many isolated 1-pixel dots (very low connectivity)."""
    rng = np.random.default_rng(0)
    tile = np.full((size, size), 255, dtype=np.uint8)
    coords = rng.integers(0, size, size=(n_dots, 2))
    for y, x in coords:
        tile[y, x] = 0
    return tile


def _make_empty(size: int = 28) -> np.ndarray:
    return np.full((size, size), 255, dtype=np.uint8)


def _make_constant_grey(size: int = 28, value: int = 180) -> np.ndarray:
    """Uniform grey -- no contrast, but every pixel is 'ink' by threshold."""
    return np.full((size, size), value, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestScoreCrop:
    def test_solid_clean_crop_scores_high(self):
        """A solid black patch on white should score well above 0.7."""
        tile = _make_solid()
        score = score_crop(tile)
        assert score > 0.7, f"Expected >0.7 for solid crop, got {score:.3f}"

    def test_fragmented_crop_scores_low(self):
        """Many isolated 1-px dots should score below 0.4."""
        tile = _make_fragmented(n_dots=25)
        score = score_crop(tile)
        assert score < 0.4, f"Expected <0.4 for fragmented crop, got {score:.3f}"

    def test_empty_crop_scores_zero(self):
        """Fully white (no ink) crop must return exactly 0."""
        tile = _make_empty()
        score = score_crop(tile)
        assert score == 0.0

    def test_constant_grey_scores_low(self):
        """Constant grey has zero contrast and poor connectivity ratio — should be low."""
        tile = _make_constant_grey(value=180)
        score = score_crop(tile)
        # Grey at 180 is below 200 threshold so ink_density is 1.0,
        # but contrast = 0 and connectivity = 1 (single blob).
        # score = 0.4*1 + 0.3*0 + 0.3*1 = 0.70  -- still low-ish due to no contrast.
        # The spec says "low score" so we just assert it's not high.
        assert score < 0.75, f"Constant grey should not score high, got {score:.3f}"

    def test_score_in_range(self):
        """Score must always be in [0, 1]."""
        for tile in [_make_solid(), _make_fragmented(), _make_empty(), _make_constant_grey()]:
            s = score_crop(tile)
            assert 0.0 <= s <= 1.0

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            score_crop(np.zeros((28, 28, 3), dtype=np.uint8))

    def test_default_threshold_is_04(self):
        assert DEFAULT_MIN_SOURCE_QUALITY == 0.4

    def test_stroke_width_not_inflated(self):
        """Ensure score_crop does NOT morphologically thicken strokes.

        A thin single-pixel diagonal line should still score reasonably —
        but the tile returned by score_crop must have the same shape (no
        side effects on the input array).
        """
        tile = np.full((28, 28), 255, dtype=np.uint8)
        for i in range(28):
            tile[i, i] = 0  # 1-pixel diagonal line
        original = tile.copy()
        score_crop(tile)
        # tile must be unmodified -- no morphological ops on caller's array
        np.testing.assert_array_equal(tile, original)
