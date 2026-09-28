"""Tests for per-glyph stroke-width normalisation (normalise_glyph_stroke_width)."""
from __future__ import annotations

import numpy as np
import pytest

from src.data_pipeline.preprocessing import normalise_glyph_stroke_width, estimate_stroke_width
from src.core.ink import INK_BINARIZE_THRESHOLD


def _make_tile(size: int = 28, stroke_px: int = 3) -> np.ndarray:
    """Return a white-background tile with a vertical black bar of exact given width."""
    tile = np.full((size, size), 255, dtype=np.uint8)
    cx = size // 2
    start = cx - stroke_px // 2
    tile[:, start: start + stroke_px] = 0
    return tile


def _measure_stroke(tile: np.ndarray) -> float:
    # estimate_stroke_width requires a boolean mask for correct numpy fancy-indexing
    ink_mask = (tile < INK_BINARIZE_THRESHOLD)
    return estimate_stroke_width(ink_mask)


class TestNormaliseGlyphStrokeWidth:
    def test_thick_strokes_reduced_to_target(self):
        """A tile with thick strokes (4px bar, measures ~3px) normalised to 2px should reach 2px.

        The distance transform on a 4px bar yields a median half-width of 1.0, so the
        measured stroke width is 2.0 * 1.0 * median ~= 3.0px. One erode iteration (delta=-1,
        ceil(1/2)=1) reduces it to exactly 2px as measured by estimate_stroke_width.
        """
        tile = _make_tile(size=28, stroke_px=4)
        result = normalise_glyph_stroke_width(tile, target_px=2.0)
        measured = _measure_stroke(result)
        assert measured <= 2.5, f"Expected <= 2.5px after normalisation, got {measured:.2f}px"

    def test_thin_strokes_increased_to_target(self):
        """A tile with thin strokes (1px) normalised to 2px should measure >= 1.5px."""
        tile = _make_tile(size=28, stroke_px=1)
        result = normalise_glyph_stroke_width(tile, target_px=2.0)
        measured = _measure_stroke(result)
        assert measured >= 1.5, f"Expected >= 1.5px after normalisation, got {measured:.2f}px"

    def test_empty_tile_returned_unchanged(self):
        """A fully white tile (no ink) must be returned unchanged."""
        tile = np.full((28, 28), 255, dtype=np.uint8)
        result = normalise_glyph_stroke_width(tile, target_px=2.0)
        np.testing.assert_array_equal(result, tile)

    def test_output_dtype_and_shape_preserved(self):
        """Output must be uint8 and same shape as input."""
        tile = _make_tile(size=28, stroke_px=3)
        result = normalise_glyph_stroke_width(tile, target_px=2.0)
        assert result.dtype == np.uint8
        assert result.shape == tile.shape

    def test_already_at_target_returned_unchanged(self):
        """A tile already at ~2px stroke width should be returned with minimal change."""
        tile = _make_tile(size=28, stroke_px=2)
        before = _measure_stroke(tile)
        result = normalise_glyph_stroke_width(tile, target_px=2.0)
        after = _measure_stroke(result)
        assert abs(after - before) <= 1.0, (
            f"Already-at-target tile changed too much: {before:.2f} -> {after:.2f}px"
        )


class TestBarColSpan:
    """Tests for _bar_col_span segment-aware column span logic."""

    def _make_token(self, label: str, row: int, col: int, segment_kind=None):
        from src.generation.layouts import LayoutToken
        return LayoutToken(
            flattened_label=label,
            glyph_key=label,
            yolo_class_name="result_bar" if label == "result_bar" else "digit_main",
            row=row,
            col=col,
            segment_kind=segment_kind,
        )

    def test_subtraction_bar_spans_row_above(self):
        """subtraction_bar should span only the main_* tokens on the row directly above it."""
        from src.generation.synth_pool import _bar_col_span
        tokens = [
            # dividend row (row=0): cols 0-3
            self._make_token("main_7", 0, 0),
            self._make_token("main_3", 0, 1),
            self._make_token("main_9", 0, 2),
            self._make_token("main_7", 0, 3),
            # partial product row above bar (row=1): cols 0-1
            self._make_token("main_6", 1, 0),
            self._make_token("main_5", 1, 1),
            # subtraction_bar at row=2
            self._make_token("result_bar", 2, 0, segment_kind="subtraction_bar"),
        ]
        bar_tok = tokens[-1]
        c0, c1 = _bar_col_span(bar_tok, tokens)
        # Should match the partial product row (row=1): cols 0-1, NOT the full 0-3 span
        assert c0 == 0
        assert c1 == 1

    def test_final_result_bar_spans_all_main(self):
        """final_result_bar (segment_kind=None) should span all main_* tokens."""
        from src.generation.synth_pool import _bar_col_span
        tokens = [
            self._make_token("main_4", 0, 0),
            self._make_token("main_5", 0, 1),
            self._make_token("main_6", 0, 2),
            self._make_token("result_bar", 1, 0, segment_kind=None),
        ]
        bar_tok = tokens[-1]
        c0, c1 = _bar_col_span(bar_tok, tokens)
        assert c0 == 0
        assert c1 == 2

    def test_intermediate_result_bar_spans_row_above(self):
        """intermediate_result_bar should span the partial product row directly above."""
        from src.generation.synth_pool import _bar_col_span
        tokens = [
            # top operand row (row=0): cols 0-2
            self._make_token("main_1", 0, 0),
            self._make_token("main_2", 0, 1),
            self._make_token("main_3", 0, 2),
            # partial product directly above bar (row=1): cols 1-2 only
            self._make_token("main_3", 1, 1),
            self._make_token("main_6", 1, 2),
            # intermediate_result_bar at row=2
            self._make_token("result_bar", 2, 0, segment_kind="intermediate_result_bar"),
        ]
        bar_tok = tokens[-1]
        c0, c1 = _bar_col_span(bar_tok, tokens)
        assert c0 == 1
        assert c1 == 2

    def test_subtraction_bar_fallback_when_row_above_empty(self):
        """When row above has no main_* tokens, subtraction_bar falls back to full span."""
        from src.generation.synth_pool import _bar_col_span
        tokens = [
            self._make_token("main_9", 0, 0),
            self._make_token("main_9", 0, 3),
            # row 1 is empty (no main_* tokens)
            self._make_token("result_bar", 2, 0, segment_kind="subtraction_bar"),
        ]
        bar_tok = tokens[-1]
        c0, c1 = _bar_col_span(bar_tok, tokens)
        # Falls back to full main_* span
        assert c0 == 0
        assert c1 == 3
