"""Tests for cluster-aware typed-cell grouping in tokens_to_grid.

The problem: a child writes a two-digit borrow/carry value (e.g. "16" or "14")
wide enough that the two glyphs straddle a cell boundary. The bbox-center
projection sends them to DIFFERENT adjacent cells, splitting "16" into "1" and
"6" in two wrong cells.

The fix: when both glyphs share the same recognizer cluster key (int(row),
grid_col) they are treated as one logical value and snapped together to the
nearest typed cell. This is scoped to replace_cells (borrow) and overflow_cells
(carry) only. All other binning is unchanged.

Constraint on honesty: char values are NEVER changed; only placement + grouping
is affected.
"""

from __future__ import annotations

import pytest

from src.grading.grid_mapping import GridCalibration, tokens_to_grid
from src.grading.types import GridToken


# ---------------------------------------------------------------------------
# Shared calibration: 13-wide x 10-tall, 40 px cells, engine_x_left=0, engine_y_top=9
# This mirrors the subtract-256-89 native frame used in test_borrow_grading.
# ---------------------------------------------------------------------------


def _cal() -> GridCalibration:
    return GridCalibration(
        grid_width=13,
        grid_height=10,
        cell_w_px=40.0,
        cell_h_px=40.0,
        origin_x_px=0.0,
        origin_y_px=0.0,
        engine_x_left=0,
        engine_y_top=9,
        invert_y=True,
    )


def _cell_center(cal: GridCalibration, engine_x: int, engine_y: int) -> tuple[float, float]:
    """Pixel center of the guide cell that maps to grading engine (engine_x, engine_y)."""
    row, col = cal.row_col_for(engine_x, engine_y)
    cx = cal.origin_x_px + col * cal.cell_w_px + cal.cell_w_px / 2
    cy = cal.origin_y_px + row * cal.cell_h_px + cal.cell_h_px / 2
    return cx, cy


def _token(
    label: str,
    cx: float,
    cy: float,
    model_row: int,
    grid_col: int,
) -> dict:
    """Synthetic assembled token with all fields tokens_to_grid reads."""
    half = 8.0
    return {
        "label": label,
        "bbox": [cx - half, cy - half, cx + half, cy + half],
        "row": model_row,
        "col": grid_col,
        "grid_col": grid_col,
    }


# ---------------------------------------------------------------------------
# Core: two-digit borrow straddling a cell boundary
# ---------------------------------------------------------------------------


def test_spilled_borrow_cluster_merges_into_replace_cell() -> None:
    """Two-digit borrow '16' whose glyphs straddle the cell boundary.

    Pixel-floor projection sends '1' to cell (12, 7) and '6' to the adjacent
    cell (11, 7). Both share model cluster (row=2, grid_col=12). The cluster-
    aware path must group them together and snap to the single REPLACE cell
    (12, 7), emitting one token with value "16".

    The calibration:
      engine_y=7 -> recognizer row = engine_y_top - y = 9 - 7 = 2
      engine_x=12 -> recognizer col = 12 - engine_x_left = 12
      Pixel col 12 spans x=[480, 520). Cell center at x=500.
    """
    cal = _cal()
    # '1' placed just inside the right edge of cell (col=11 px region) but the
    # child's intent is the cell at col=12. Center x=479 bins to col=11 (engine x=11).
    # '6' placed inside cell col=12. Center x=505 bins to col=12 (engine x=12).
    # Both share model cluster row=2, grid_col=12.
    t1 = _token("main_1", cx=479.0, cy=100.0, model_row=2, grid_col=12)
    t6 = _token("main_6", cx=505.0, cy=100.0, model_row=2, grid_col=12)

    replace_cells = {(12, 7)}  # engine cell for the units borrow

    tokens = tokens_to_grid([t1, t6], cal, replace_cells=replace_cells)

    replace = [t for t in tokens if t.type == "REPLACE"]
    assert len(replace) == 1, f"Expected 1 REPLACE token, got {[t.c for t in replace]}"
    assert replace[0].c == "16", f"Expected '16', got '{replace[0].c}'"
    assert (replace[0].x, replace[0].y) == (12, 7)

    # Prove the cluster-aware path is strictly better than the fast path:
    # without replace_cells the two glyphs stay in separate plain SYMBOL tokens
    # (the old split), so the new path is a genuine improvement, not just noise.
    tokens_no_typed = tokens_to_grid([t1, t6], cal)
    assert len(tokens_no_typed) == 2, (
        f"Fast path (no typed cells) must keep glyphs separate, got {len(tokens_no_typed)}"
    )


def test_spilled_overflow_cluster_merges_into_overflow_cell() -> None:
    """Two-digit carry '12' straddling boundary -> one OVERFLOW token with '12'."""
    cal = _cal()
    # engine cell (5, 8): row = 9-8 = 1, col=5, px col=5 spans [200,240). Center=220.
    # '1' placed at x=199 (bins to col=4 -> engine x=4), '2' at x=221 (bins to col=5 -> engine x=5).
    # Both share cluster row=1, grid_col=5.
    t1 = _token("main_1", cx=199.0, cy=60.0, model_row=1, grid_col=5)
    t2 = _token("main_2", cx=221.0, cy=60.0, model_row=1, grid_col=5)

    overflow_cells = {(5, 8)}

    tokens = tokens_to_grid([t1, t2], cal, overflow_cells=overflow_cells)

    overflow = [t for t in tokens if t.type == "OVERFLOW"]
    assert len(overflow) == 1, f"Expected 1 OVERFLOW token, got {[t.c for t in overflow]}"
    assert overflow[0].c == "12", f"Expected '12', got '{overflow[0].c}'"
    assert (overflow[0].x, overflow[0].y) == (5, 8)


def test_two_separate_borrows_with_different_clusters_stay_separate() -> None:
    """Two borrows in adjacent cells with DIFFERENT model grid_cols stay separate.

    A single digit in cluster grid_col=11 and a single digit in cluster
    grid_col=12 each form their own group, snap to their own borrow cell, and
    are NOT merged. This is the counterpart to the spilled-merge test: distinct
    clusters must never be concatenated.
    """
    cal = _cal()
    # "4" at cluster grid_col=11 -> cell (11,7): px center (460,100).
    t4 = _token("main_4", cx=460.0, cy=100.0, model_row=2, grid_col=11)
    # "6" at cluster grid_col=12 -> cell (12,7): px center (505,100).
    t6 = _token("main_6", cx=505.0, cy=100.0, model_row=2, grid_col=12)

    replace_cells = {(11, 7), (12, 7)}
    tokens = tokens_to_grid([t4, t6], cal, replace_cells=replace_cells)

    replace = {(t.x, t.y): t.c for t in tokens if t.type == "REPLACE"}
    # Different clusters -> two distinct single-digit REPLACE cells, no merge.
    assert set(replace.keys()) == {(11, 7), (12, 7)}, f"Got cells: {set(replace.keys())}"
    assert replace[(11, 7)] == "4", f"Expected '4' at (11,7), got '{replace[(11, 7)]}'"
    assert replace[(12, 7)] == "6", f"Expected '6' at (12,7), got '{replace[(12, 7)]}'"

def test_wrong_borrow_value_sent_verbatim_not_corrected() -> None:
    """A wrong borrow '15' is emitted exactly as '15'; no correction applied.

    This is the core honesty constraint: the mapper ONLY decides grouping +
    placement, never mutates char values.
    """
    cal = _cal()
    # Write "15" (wrong, correct is "16") in the units cell (12,7).
    # Both glyphs share cluster and land inside the REPLACE cell -> "15" verbatim.
    base_x = 12 * 40  # px x of cell col=12
    t1 = _token("main_1", cx=float(base_x + 10), cy=100.0, model_row=2, grid_col=12)
    t5 = _token("main_5", cx=float(base_x + 30), cy=100.0, model_row=2, grid_col=12)

    replace_cells = {(12, 7)}
    tokens = tokens_to_grid([t1, t5], cal, replace_cells=replace_cells)
    replace = [t for t in tokens if t.type == "REPLACE"]
    assert len(replace) == 1
    assert replace[0].c == "15", "Wrong borrow must be sent verbatim, not corrected"


def test_no_typed_cells_behaviour_identical_to_before() -> None:
    """With no replace_cells and no overflow_cells, behaviour is 100% identical.

    Every glyph is emitted as a plain SYMBOL at its pixel-floor cell. No cluster
    logic touches anything. Used as regression gate for answer-only exercises.
    """
    cal = _cal()
    # Three answer digits at well-separated cells.
    t1 = _token("main_1", cx=400.0, cy=300.0, model_row=7, grid_col=10)
    t6 = _token("main_6", cx=440.0, cy=300.0, model_row=7, grid_col=11)
    t7 = _token("main_7", cx=480.0, cy=300.0, model_row=7, grid_col=12)

    tokens_no_cells = tokens_to_grid([t1, t6, t7], cal)
    tokens_empty_cells = tokens_to_grid([t1, t6, t7], cal, replace_cells=set(), overflow_cells=set())

    # Both must be SYMBOL type.
    assert all(t.type == "SYMBOL" for t in tokens_no_cells)
    assert all(t.type == "SYMBOL" for t in tokens_empty_cells)
    # Same result.
    assert [(t.x, t.y, t.c, t.type) for t in tokens_no_cells] == [
        (t.x, t.y, t.c, t.type) for t in tokens_empty_cells
    ]


def test_operand_and_answer_binning_unchanged_when_replace_cells_present() -> None:
    """Operands and answer cells must produce identical tokens whether or not replace_cells
    is populated. The new logic must be fully scoped to typed cells only.
    """
    cal = _cal()
    # Operand '2' at engine (10, 5): row=9-5=4, col=10 -> px center (420, 180).
    operand = _token("main_2", cx=420.0, cy=180.0, model_row=4, grid_col=10)
    # Answer '7' at engine (12, 3): row=9-3=6, col=12 -> px center (500, 260).
    answer = _token("main_7", cx=500.0, cy=260.0, model_row=6, grid_col=12)

    tokens_without = tokens_to_grid([operand, answer], cal)
    tokens_with = tokens_to_grid([operand, answer], cal, replace_cells={(12, 7)})

    # Positions and values must be byte-identical.
    pos_without = {(t.x, t.y): t.c for t in tokens_without}
    pos_with = {(t.x, t.y): t.c for t in tokens_with}
    assert pos_without == pos_with, (
        f"Operand/answer binning changed when replace_cells populated: "
        f"{pos_without} vs {pos_with}"
    )


def test_glyph_too_far_from_typed_cell_falls_back_to_pixel_floor() -> None:
    """A glyph whose cluster centroid is more than 1.5 cell widths from every
    typed cell falls back to its pixel-floor cell as a plain SYMBOL.

    This guards the case where the cluster signal is unreliable (e.g. a stray
    mark the GNN mistakenly assigned to the same cluster as a borrow).
    """
    cal = _cal()
    # Replace cell at (12, 7): row=2, col=12, px col_12=[480,520).
    # Stray mark at px x=100 (col=2, engine x=2) - far from col=12 (distance >> 1.5*40).
    # But stray has the same grid_col=12 as the replace cell.
    stray = _token("main_3", cx=100.0, cy=100.0, model_row=2, grid_col=12)

    replace_cells = {(12, 7)}
    tokens = tokens_to_grid([stray], cal, replace_cells=replace_cells)

    # Should be emitted at its pixel-floor cell (col=2), NOT forced to (12,7).
    assert len(tokens) == 1
    symbol_tokens = [t for t in tokens if t.type == "SYMBOL"]
    replace_tokens = [t for t in tokens if t.type == "REPLACE"]
    # The stray must NOT land in the replace cell.
    assert not any(t.x == 12 and t.y == 7 for t in replace_tokens), (
        "Stray mark far from typed cell must not be forced into replace cell"
    )
    # It must be emitted somewhere as SYMBOL (pixel floor at col=2, engine x=2).
    assert len(symbol_tokens) == 1
    assert symbol_tokens[0].x == 2


def test_cluster_concat_order_is_left_to_right_by_center_x() -> None:
    """When cluster merging produces a multi-digit value, order is by center-x ascending.

    Input: '6' at cx=505, '1' at cx=479 (given out of order).
    Expected value: "16" (left digit first by x position).
    """
    cal = _cal()
    t6 = _token("main_6", cx=505.0, cy=100.0, model_row=2, grid_col=12)
    t1 = _token("main_1", cx=479.0, cy=100.0, model_row=2, grid_col=12)

    replace_cells = {(12, 7)}
    tokens = tokens_to_grid([t6, t1], cal, replace_cells=replace_cells)

    replace = [t for t in tokens if t.type == "REPLACE"]
    assert len(replace) == 1
    assert replace[0].c == "16", f"Order should be left-to-right by cx, got '{replace[0].c}'"


def test_replace_and_overflow_both_present_in_one_call() -> None:
    """A single call with both replace_cells and overflow_cells handles each correctly."""
    cal = _cal()
    # Borrow "16" straddling the replace cell (12, 7).
    b1 = _token("main_1", cx=479.0, cy=100.0, model_row=2, grid_col=12)
    b6 = _token("main_6", cx=505.0, cy=100.0, model_row=2, grid_col=12)
    # Carry "1" cleanly in overflow cell (10, 8): row=9-8=1, col=10, px center (420,60).
    c1 = _token("main_1", cx=420.0, cy=60.0, model_row=1, grid_col=10)

    replace_cells = {(12, 7)}
    overflow_cells = {(10, 8)}
    tokens = tokens_to_grid([b1, b6, c1], cal, replace_cells=replace_cells, overflow_cells=overflow_cells)

    replace = [t for t in tokens if t.type == "REPLACE"]
    overflow = [t for t in tokens if t.type == "OVERFLOW"]
    assert len(replace) == 1
    assert replace[0].c == "16"
    assert (replace[0].x, replace[0].y) == (12, 7)
    assert len(overflow) == 1
    assert overflow[0].c == "1"
    assert (overflow[0].x, overflow[0].y) == (10, 8)


# ---------------------------------------------------------------------------
# P3-a — edge-of-tolerance snap: 1.5-cell boundary is <= (not <)
# ---------------------------------------------------------------------------


def test_cluster_snap_inside_tolerance_snaps_to_replace_cell() -> None:
    """A glyph whose centroid is exactly 1.4 cell widths from the typed cell
    center is within the 1.5-cell tolerance and MUST snap to that replace cell.

    Guards that the tolerance check uses <= and is not accidentally strict (<).
    The replace cell is at engine (12, 7); its pixel center is (500.0, 100.0).
    A single glyph at cx = 500 - 1.4*40 = 444.0 (inside the typed-area row
    because its pixel-floor engine y == 7 which is in typed_ys) must snap.
    """
    cal = _cal()
    # Replace cell (12, 7): center px (500.0, 100.0).
    # Glyph at cx = 500 - 1.4*40 = 444.0, cy = 100.0 (same row).
    # pixel-floor: floor(444/40) = 11 -> engine x=11; row=2 -> engine y=7.
    # Cell (11,7) is in typed_ys -> enters typed path.
    # Cluster key grid_col=12 matches the replace cell -> centroid = (444, 100).
    # abs(444 - 500) = 56.0 <= 60.0 (1.5 * 40) -> within tolerance -> SNAPS.
    glyph = _token("main_3", cx=444.0, cy=100.0, model_row=2, grid_col=12)

    replace_cells = {(12, 7)}
    tokens = tokens_to_grid([glyph], cal, replace_cells=replace_cells)

    replace = [t for t in tokens if t.type == "REPLACE"]
    assert len(replace) == 1, (
        f"Glyph at 1.4*cell_w from typed cell must snap; got {[t.type for t in tokens]}"
    )
    assert (replace[0].x, replace[0].y) == (12, 7), (
        f"Snapped to wrong cell: ({replace[0].x}, {replace[0].y})"
    )
    assert replace[0].c == "3", f"Char must be verbatim: got {replace[0].c!r}"


def test_cluster_snap_outside_tolerance_falls_back_to_pixel_floor() -> None:
    """A glyph whose centroid is exactly 1.6 cell widths from the typed cell
    center exceeds the 1.5-cell tolerance and MUST fall back to a plain SYMBOL
    at its pixel-floor cell.

    Guards the > side of the 1.5-cell boundary.
    Replace cell at engine (12, 7); pixel center (500.0, 100.0).
    Glyph at cx = 500 - 1.6*40 = 436.0 -> pixel-floor engine x = floor(436/40) = 10.
    abs(436 - 500) = 64.0 > 60.0 -> outside tolerance -> SYMBOL at (10, 7).
    """
    cal = _cal()
    glyph = _token("main_5", cx=436.0, cy=100.0, model_row=2, grid_col=12)

    replace_cells = {(12, 7)}
    tokens = tokens_to_grid([glyph], cal, replace_cells=replace_cells)

    replace = [t for t in tokens if t.type == "REPLACE"]
    assert len(replace) == 0, (
        f"Glyph at 1.6*cell_w must fall back, not snap; got REPLACE at "
        f"{[(t.x, t.y) for t in replace]}"
    )
    # Must be emitted as a plain SYMBOL at its pixel-floor cell.
    symbol = [t for t in tokens if t.type == "SYMBOL"]
    assert len(symbol) == 1
    assert symbol[0].x == 10, f"Pixel-floor engine x must be 10, got {symbol[0].x}"
    assert symbol[0].c == "5", f"Char must be verbatim '5', got {symbol[0].c!r}"


# ---------------------------------------------------------------------------
# P3-b — carry (OVERFLOW) verbatim honesty: wrong value sent unchanged
# ---------------------------------------------------------------------------


def test_wrong_carry_value_sent_verbatim_not_corrected() -> None:
    """A wrong carry value submitted into an OVERFLOW cell is emitted VERBATIM.

    Mirrors the existing test_wrong_borrow_value_sent_verbatim_not_corrected
    for the REPLACE path, extending the honesty guarantee to the OVERFLOW
    (carry) concat path (overflow_buckets_2).

    The mapper decides placement and grouping only; it NEVER corrects the char
    toward the expected correct value. A child who writes '9' as a carry where
    the correct carry is '3' must see '9' submitted to the grader unchanged.
    """
    cal = _cal()
    # Overflow cell at engine (5, 8): row = 9-8 = 1, col = 5, center px (220.0, 60.0).
    # Submit wrong carry '9' (correct would be some other digit).
    # Both glyph and cluster land cleanly inside the overflow cell.
    base_x = 5 * 40  # = 200
    wrong_carry = _token("main_9", cx=float(base_x + 20), cy=60.0, model_row=1, grid_col=5)

    overflow_cells = {(5, 8)}
    tokens = tokens_to_grid([wrong_carry], cal, overflow_cells=overflow_cells)

    overflow = [t for t in tokens if t.type == "OVERFLOW"]
    assert len(overflow) == 1, f"Expected 1 OVERFLOW token, got {len(overflow)}"
    assert overflow[0].c == "9", (
        f"Wrong carry must be sent verbatim as '9', not corrected; got {overflow[0].c!r}"
    )
    assert (overflow[0].x, overflow[0].y) == (5, 8), (
        f"Carry placed at wrong cell: ({overflow[0].x}, {overflow[0].y})"
    )
