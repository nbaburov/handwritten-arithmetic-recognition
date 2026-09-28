"""Tests for the pixel<->grid coordinate mapper.

Covers the four mapper responsibilities the plan's WS2 done-definition names:

* round-trip within one cell tolerance (``tokens_to_grid`` then
  ``grid_rect_to_pixels`` returns a pixel rect containing the original cell
  center);
* y-inversion correct (recognizer row 0 = top = largest grading engine ``y``);
* ``reconcile_with_guide`` flags an off-cell token and agrees on an aligned one;
* ``verify_calibration`` flags an injected systematic offset and passes on an
  aligned set built from the captured event fixture's echoed positions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.grading.grid_mapping import (
    GridCalibration,
    build_calibration,
    grid_rect_to_pixels,
    reconcile_with_guide,
    tokens_to_grid,
    verify_calibration,
)
from src.grading.types import GridRect, GridToken, SessionCreated

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


# ---------------------------------------------------------------------------
# Calibration construction
# ---------------------------------------------------------------------------


def _square_calibration(
    *,
    grid_width: int = 4,
    grid_height: int = 3,
    canvas_px: float = 512.0,
    engine_x_left: int = 0,
    engine_y_top: int | None = None,
) -> GridCalibration:
    """A simple square-pixel calibration with no pixel origin offset."""
    if engine_y_top is None:
        engine_y_top = grid_height - 1
    return build_calibration(
        {"width": grid_width, "height": grid_height},
        {
            "canvas_px": canvas_px,
            "engine_x_left": engine_x_left,
            "engine_y_top": engine_y_top,
        },
    )


def _cell_center_token(
    calibration: GridCalibration, row: int, col: int, label: str = "main_5"
) -> dict[str, object]:
    """An assembled-token dict whose bbox is centered on guide cell (row, col)."""
    left = calibration.origin_x_px + col * calibration.cell_w_px
    top = calibration.origin_y_px + row * calibration.cell_h_px
    # A small bbox centered in the cell.
    cx = left + calibration.cell_w_px / 2.0
    cy = top + calibration.cell_h_px / 2.0
    half = min(calibration.cell_w_px, calibration.cell_h_px) / 4.0
    return {
        "label": label,
        "role": "main",
        "detail": "5",
        "row": row,
        "col": col,
        "grid_col": col,
        "within_row_ord": col,
        "within_col_ord": row,
        "bbox": [cx - half, cy - half, cx + half, cy + half],
        "confidence": 0.9,
        "spatial_conflict": False,
        "carry_conflict": False,
    }


def test_build_calibration_reads_grid_dims_from_view_model() -> None:
    cal = build_calibration({"width": 13, "height": 10}, {"canvas_px": 512.0})
    assert cal.grid_width == 13
    assert cal.grid_height == 10
    assert cal.cell_w_px == pytest.approx(512.0 / 13)
    assert cal.cell_h_px == pytest.approx(512.0 / 10)


def test_build_calibration_reads_real_create_view_model() -> None:
    """The captured create fixture parses into a 13x10 grid with a bottom-origin y.

    AUTOSHOW tokens (decorative UI helpers at y=1..3) are excluded from the
    origin derivation so they cannot skew engine_x_left toward their lower x=5.
    Only non-AUTOSHOW (FIXED) scaffold tokens are used; those start at x=9.
    """
    data = json.loads((FIXTURES / "create.res.json").read_text(encoding="utf-8"))
    session = SessionCreated.from_api(data)
    cal = build_calibration(session.view_model, {"canvas_px": 512.0})
    assert cal.grid_width == 13
    assert cal.grid_height == 10
    # Non-AUTOSHOW scaffold tokens: FIXED at x=9..12, y=4..6. Top row = y=6.
    assert cal.engine_y_top == 6
    # Leftmost non-AUTOSHOW token is the result bar at x=9, not AUTOSHOW at x=5.
    assert cal.engine_x_left == 9
    assert cal.invert_y is True


# ---------------------------------------------------------------------------
# y-inversion
# ---------------------------------------------------------------------------


def test_y_inversion_top_row_has_largest_grid_y() -> None:
    cal = _square_calibration(grid_width=4, grid_height=3, engine_x_left=0, engine_y_top=2)
    # Recognizer row 0 is the top row -> largest grading engine y.
    assert cal.grid_coords_for(row=0, col=0) == (0, 2)
    assert cal.grid_coords_for(row=1, col=0) == (0, 1)
    assert cal.grid_coords_for(row=2, col=0) == (0, 0)
    # Column maps straight through with the x offset.
    assert cal.grid_coords_for(row=0, col=3) == (3, 2)


def test_row_col_for_is_inverse_of_grid_coords_for() -> None:
    cal = _square_calibration(grid_width=5, grid_height=4, engine_x_left=2, engine_y_top=7)
    for row in range(4):
        for col in range(5):
            x, y = cal.grid_coords_for(row, col)
            assert cal.row_col_for(x, y) == (row, col)


def test_y_inversion_pixels_top_row_sits_higher_on_canvas() -> None:
    cal = _square_calibration(grid_width=4, grid_height=3, engine_x_left=0, engine_y_top=2)
    # Top row (y=2) should map to a smaller pixel top than the bottom row (y=0).
    top_cell = cal.cell_to_pixels(0, 2)
    bottom_cell = cal.cell_to_pixels(0, 0)
    assert top_cell[1] < bottom_cell[1]
    assert top_cell[1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# tokens_to_grid
# ---------------------------------------------------------------------------


def test_tokens_to_grid_uses_bbox_pixel_position_as_source_of_truth() -> None:
    """tokens_to_grid derives engine (x,y) from the ink's bbox center, not row/col.

    The bbox center is projected through pixel_cell_at -> grid_coords_for. A
    token whose bbox center sits at guide cell (0, 1) maps to engine (1, 2) regardless
    of what the GNN's row/col cluster indices say. For a token constructed with
    _cell_center_token the bbox IS centered on the named guide cell, so the result
    equals what the old row/col path would have given, but the mechanism is different.
    """
    cal = _square_calibration(grid_width=4, grid_height=3, engine_x_left=0, engine_y_top=2)
    tokens = [_cell_center_token(cal, row=0, col=1, label="main_7")]
    grid = tokens_to_grid(tokens, cal)
    assert len(grid) == 1
    assert grid[0].c == "7"
    assert grid[0].x == 1   # guide col 1 -> engine x = engine_x_left + 1 = 1
    assert grid[0].y == 2   # guide row 0 (top) -> engine y = engine_y_top - 0 = 2
    assert grid[0].tags == []


def test_tokens_to_grid_char_via_palette_display_glyph() -> None:
    cal = _square_calibration()
    minus = {
        "label": "op_minus",
        "row": 0,
        "col": 0,
        "grid_col": 0,
        "bbox": [0, 0, 10, 10],
    }
    grid = tokens_to_grid([minus], cal)
    assert grid[0].c == "−"  # display_glyph('op_minus') -> minus sign


def test_tokens_to_grid_skips_empty_glyph_tokens() -> None:
    """div_bracket renders as pure geometry (empty glyph) and is not sent."""
    cal = _square_calibration()
    bracket = {"label": "div_bracket", "row": 0, "col": 0, "grid_col": 0, "bbox": [0, 0, 8, 8]}
    assert tokens_to_grid([bracket], cal) == []


def test_tokens_to_grid_uses_bbox_center_not_grid_col() -> None:
    """tokens_to_grid ignores row/grid_col and uses the bbox center pixel position.

    A token with bbox centered at guide cell (0, 4) maps to engine x=4 regardless
    of the value of col or grid_col on the token dict. This guards against the
    bug where a partial drawing (child draws only answer digits in columns 2-3)
    causes the GNN to assign cluster_ids 0,1 to those columns, which would
    map them to engine x=0,1 instead of the correct x=2,3 from the pixel position.
    """
    cal = _square_calibration(grid_width=6, grid_height=2, engine_x_left=0, engine_y_top=1)
    # Build a token whose bbox center sits at guide cell (0, 4).
    correct_pos_token = _cell_center_token(cal, row=0, col=4, label="main_3")
    # Deliberately set misleading row/col/grid_col (what GNN might emit for a
    # partial scene where this column is the first one seen).
    correct_pos_token = dict(correct_pos_token)
    correct_pos_token["col"] = 0
    correct_pos_token["grid_col"] = 0
    correct_pos_token["row"] = 0
    grid = tokens_to_grid([correct_pos_token], cal)
    # Must map to col 4 from the bbox position, not col 0 from grid_col.
    assert grid[0].x == 4


# ---------------------------------------------------------------------------
# round-trip within one cell
# ---------------------------------------------------------------------------


def _single_cell_rect(x: int, y: int) -> GridRect:
    """A grading engine single-cell rect (top exclusive, right exclusive)."""
    return GridRect(left=x, top=y + 1, bottom=y, right=x + 1)


@pytest.mark.parametrize("row,col", [(0, 0), (0, 3), (1, 2), (2, 1), (2, 3)])
def test_round_trip_pixel_rect_contains_cell_center(row: int, col: int) -> None:
    """tokens_to_grid -> grid_rect_to_pixels returns a rect containing the center."""
    cal = _square_calibration(grid_width=4, grid_height=3, engine_x_left=0, engine_y_top=2)
    token = _cell_center_token(cal, row=row, col=col)
    cx = (token["bbox"][0] + token["bbox"][2]) / 2.0
    cy = (token["bbox"][1] + token["bbox"][3]) / 2.0

    grid = tokens_to_grid([token], cal)
    rect = grid_rect_to_pixels(_single_cell_rect(grid[0].x, grid[0].y), cal)

    left, top, right, bottom = rect
    assert left <= cx <= right
    assert top <= cy <= bottom
    # Within one cell tolerance: the rect is exactly one cell wide and tall.
    assert (right - left) == pytest.approx(cal.cell_w_px)
    assert (bottom - top) == pytest.approx(cal.cell_h_px)


def test_round_trip_with_pixel_origin_offset() -> None:
    """A non-zero pixel origin still round-trips within one cell."""
    cal = build_calibration(
        {"width": 4, "height": 3},
        {
            "canvas_w_px": 400.0,
            "canvas_h_px": 300.0,
            "origin_x_px": 40.0,
            "origin_y_px": 30.0,
            "engine_x_left": 0,
            "engine_y_top": 2,
        },
    )
    token = _cell_center_token(cal, row=1, col=2)
    cx = (token["bbox"][0] + token["bbox"][2]) / 2.0
    cy = (token["bbox"][1] + token["bbox"][3]) / 2.0
    grid = tokens_to_grid([token], cal)
    left, top, right, bottom = grid_rect_to_pixels(_single_cell_rect(grid[0].x, grid[0].y), cal)
    assert left <= cx <= right
    assert top <= cy <= bottom


def test_grid_rect_to_pixels_multi_cell_span() -> None:
    """A 2-wide hint rect (grading engine right exclusive) spans exactly two cells."""
    cal = _square_calibration(grid_width=5, grid_height=2, engine_x_left=0, engine_y_top=1)
    # Grading engine echoes a 2-wide number at left=2 as right=4 (cols 2 and 3).
    rect = grid_rect_to_pixels(GridRect(left=2, top=2, bottom=1, right=4), cal)
    left, top, right, bottom = rect
    assert (right - left) == pytest.approx(2 * cal.cell_w_px)
    assert left == pytest.approx(2 * cal.cell_w_px)


# ---------------------------------------------------------------------------
# reconcile_with_guide
# ---------------------------------------------------------------------------


def test_reconcile_agrees_on_aligned_token() -> None:
    cal = _square_calibration(grid_width=4, grid_height=3, engine_x_left=0, engine_y_top=2)
    token = _cell_center_token(cal, row=1, col=2)
    [reconciled] = reconcile_with_guide([token], cal)
    assert reconciled.disagreement is False
    assert reconciled.recognizer_row == 1
    assert reconciled.recognizer_col == 2
    assert reconciled.guide_row == 1
    assert reconciled.guide_col == 2


def test_reconcile_flags_off_cell_token() -> None:
    """Ink that physically sits in a different guide cell than the model's row/col."""
    cal = _square_calibration(grid_width=4, grid_height=3, engine_x_left=0, engine_y_top=2)
    token = _cell_center_token(cal, row=0, col=0)
    # Move the ink's bbox into the physical guide cell (row 2, col 3) while the
    # recognizer still claims (row 0, col 0).
    far = _cell_center_token(cal, row=2, col=3)
    token["bbox"] = far["bbox"]
    [reconciled] = reconcile_with_guide([token], cal)
    assert reconciled.disagreement is True
    assert reconciled.recognizer_row == 0
    assert reconciled.recognizer_col == 0
    assert reconciled.guide_row == 2
    assert reconciled.guide_col == 3
    # The model's grid coords are unchanged (never overwritten by the guide).
    assert (reconciled.grid_x, reconciled.grid_y) == cal.grid_coords_for(0, 0)


def test_reconcile_never_overwrites_model_grid_coords() -> None:
    """reconcile_with_guide keeps the model's row/col as grid_x/grid_y.

    Even when the bbox sits in a different guide cell, the reconciled record's
    grid_x/grid_y comes from the model's row/col (the GNN answer), not the
    guide cell. This is the soft disagreement path: the model is never moved.

    Note: tokens_to_grid now uses bbox position (not model row/col) to derive
    the sent engine cell, so the sent token and the reconciled grid_x/grid_y can
    legitimately differ when ink is off-cell. That divergence is exactly what
    reconcile_with_guide flags as disagreement=True.
    """
    cal = _square_calibration(grid_width=4, grid_height=3, engine_x_left=0, engine_y_top=2)
    token = _cell_center_token(cal, row=0, col=0)
    # Move bbox to guide cell (2, 3) while model still claims (row=0, col=0).
    token["bbox"] = _cell_center_token(cal, row=2, col=3)["bbox"]
    [reconciled] = reconcile_with_guide([token], cal)
    # reconcile preserves model coords in grid_x/grid_y, never overwrites them.
    assert (reconciled.grid_x, reconciled.grid_y) == cal.grid_coords_for(0, 0)
    # Disagreement is flagged because guide cell (2,3) != model (0,0).
    assert reconciled.disagreement is True
    # tokens_to_grid uses bbox position -> sends the guide cell (2,3).
    grid = tokens_to_grid([token], cal)
    assert (grid[0].x, grid[0].y) == cal.grid_coords_for(2, 3)


# ---------------------------------------------------------------------------
# verify_calibration
# ---------------------------------------------------------------------------


def _aligned_set_from_event_fixture() -> tuple[list[GridToken], list[dict]]:
    """Build a sent-token list and echoed elements that agree, from the capture.

    The captured event request sent tokens at grid coords; the captured event
    response echoes each token id inside an element at the same grid position.
    Pairing them yields a zero-offset (aligned) set.
    """
    res = json.loads((FIXTURES / "event.res.json").read_text(encoding="utf-8"))
    elements = res[0]["result"]["elements"]
    # Build sent tokens directly from the echoed positions so they are aligned.
    sent: list[GridToken] = []
    for element in elements:
        pos = element.get("position")
        for token_id in element.get("symbolIdList", []):
            sent.append(GridToken(id=str(token_id), c="0", x=int(pos["x"]), y=int(pos["y"])))
    return sent, elements


def test_verify_calibration_passes_on_aligned_set_from_fixture() -> None:
    sent, elements = _aligned_set_from_event_fixture()
    report = verify_calibration(sent, elements)
    assert report.aligned is True
    assert report.systematic_offset is None
    assert report.matched == len(sent)
    assert report.unmatched == 0


def test_verify_calibration_flags_injected_systematic_offset() -> None:
    sent, elements = _aligned_set_from_event_fixture()
    # Inject a constant +2 x, -1 y offset on every sent token.
    shifted = [GridToken(id=t.id, c=t.c, x=t.x + 2, y=t.y - 1, tags=t.tags) for t in sent]
    report = verify_calibration(shifted, elements)
    assert report.aligned is False
    assert report.systematic_offset == (2, -1)
    assert report.matched == len(sent)


def test_verify_calibration_inconsistent_offsets_are_not_systematic() -> None:
    sent, elements = _aligned_set_from_event_fixture()
    # Shift only the first token: offsets are now mixed.
    shifted = list(sent)
    first = shifted[0]
    shifted[0] = GridToken(id=first.id, c=first.c, x=first.x + 5, y=first.y, tags=first.tags)
    report = verify_calibration(shifted, elements)
    assert report.aligned is False
    assert report.systematic_offset is None


def test_verify_calibration_counts_unmatched_tokens() -> None:
    sent, elements = _aligned_set_from_event_fixture()
    sent.append(GridToken(id="UNSEEN", c="9", x=0, y=0))
    report = verify_calibration(sent, elements)
    # The unmatched token does not break alignment of the matched set.
    assert report.aligned is True
    assert report.unmatched == 1


def test_verify_calibration_accepts_parsed_elements() -> None:
    """verify_calibration works with parsed EvalElement objects, not just dicts."""
    from src.grading.types import EvalResult

    sent, raw_elements = _aligned_set_from_event_fixture()
    res = json.loads((FIXTURES / "event.res.json").read_text(encoding="utf-8"))
    parsed = EvalResult.from_api(res)
    report = verify_calibration(sent, parsed.elements)
    assert report.aligned is True
    assert report.matched == len(sent)


def test_verify_calibration_empty_when_no_matches() -> None:
    sent = [GridToken(id="A", c="1", x=0, y=0)]
    report = verify_calibration(sent, [])
    assert report.aligned is False
    assert report.matched == 0
    assert report.unmatched == 1
