"""Calibration integration tests: coordinate round-trips and full-correct grading.

For each bank exercise, this module verifies that:

1. A calibration built from ``BankExercise.evaluate_engine_origin`` correctly maps
   recognizer (row, col) to the grading engine evaluate-frame (x, y) for every
   expected token, including carry rows (y=7 for product).

2. A full-correct submission (all expected tokens, assembled via the calibration
   from recognizer row/col) passes through ``MockGradingClient.evaluate`` with
   ``progress == 1.0`` and every element status ``OK`` (no MISSING, no ERROR).

3. AUTOSHOW tokens in the create view-model do not corrupt the calibration origin
   derived from the fabricated scaffold (they are filtered by ``_default_engine_origin``).

These are the regression guard for the coordinate mapping: if the calibration
origin shifts or the grid_height shrinks, step cells fall outside the
addressable range and grading returns MISSING.

Note: subtract-256-89 was re-calibrated 2026-06-19 to the live engine evaluate
frame (grid 13x10, y=0..9, no borrow cells).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.grading import exercises as ex
from src.grading.exercises import BankExercise, list_exercises
from src.grading.grid_mapping import GridCalibration, build_calibration, tokens_to_grid
from src.grading.mock_client import MockGradingClient
from src.grading.types import GridToken
from src.core.run_config import GradingConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_BANK_IDS = [s.exercise_id for s in list_exercises()]


def _derive_bank(exercise_id: str) -> BankExercise:
    """Derive BankExercise from captured fixtures via MockGradingClient."""
    client = MockGradingClient(config=GradingConfig(), project_root=PROJECT_ROOT)
    spec = ex.get_exercise(exercise_id)
    created = client.create_session(spec)
    return client._sessions[created.session_id].exercise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _calibration_from_evaluate_origin(entry: BankExercise) -> GridCalibration:
    """Build a 512x512 calibration pinned to the exercise's evaluate-frame origin.

    ``evaluate_engine_origin = (engine_x_left, engine_y_top)`` is the grading engine grid
    coordinate of the top-left recognizer cell in the evaluate frame, chosen so
    that every expected token (operands, carries, borrows, quotient, answer) has
    a non-negative recognizer row within ``grid_height``.
    """
    engine_x_left, engine_y_top = entry.evaluate_engine_origin
    return build_calibration(
        {"width": entry.grid_width, "height": entry.grid_height},
        {
            "canvas_px": 512.0,
            "engine_x_left": engine_x_left,
            "engine_y_top": engine_y_top,
        },
    )


def _assembled_tokens_for_exercise(
    entry: BankExercise, calibration: GridCalibration
) -> list[dict]:
    """Build assembled-token dicts for every expected cell, using recognizer row/col.

    Each dict carries the recognizer (row, col) derived from the expected token's
    evaluate-frame (x, y) via ``calibration.row_col_for``, a synthetic bbox
    centered in the corresponding guide cell, and the char directly (not via
    display_glyph, which would convert ASCII operators to Unicode). We use
    GridToken directly in the submission to bypass the label/glyph mismatch so
    the calibration math is isolated from the glyph rendering concern.
    """
    tokens: list[dict] = []
    for i, token in enumerate(entry.expected):
        row, col = calibration.row_col_for(token.x, token.y)
        # Synthesise a bbox centered in the guide cell so reconcile_with_guide
        # would agree if it were called.
        cx = calibration.origin_x_px + (col + 0.5) * calibration.cell_w_px
        cy = calibration.origin_y_px + (row + 0.5) * calibration.cell_h_px
        half = min(calibration.cell_w_px, calibration.cell_h_px) / 4.0
        tokens.append(
            {
                "id": f"T{i}",
                "label": f"_direct_{token.c}",  # not processed via display_glyph in this path
                "row": row,
                "col": col,
                "grid_col": col,
                "bbox": [cx - half, cy - half, cx + half, cy + half],
            }
        )
    return tokens


def _grid_tokens_for_exercise(
    entry: BankExercise, calibration: GridCalibration
) -> list[GridToken]:
    """Build GridTokens directly from expected cells via calibration round-trip.

    Computes recognizer (row, col) from the evaluate-frame (x, y) using the
    calibration, then maps back to (x, y) via ``calibration.grid_coords_for``
    and confirms identity. Returns GridTokens with the expected char (``c``)
    so the mock grader can match them against the expected solution cells.
    """
    grid_tokens: list[GridToken] = []
    for i, token in enumerate(entry.expected):
        row, col = calibration.row_col_for(token.x, token.y)
        x, y = calibration.grid_coords_for(row, col)
        assert (x, y) == (token.x, token.y), (
            f"calibration round-trip failed for {entry.spec.exercise_id}: "
            f"expected ({token.x},{token.y}), got ({x},{y}) via row={row},col={col}"
        )
        tags = ["FIXED"] if token.given else []
        if token.role == "borrow":
            cell_type = "REPLACE"
        elif token.role == "carry":
            cell_type = "OVERFLOW"
        else:
            cell_type = "SYMBOL"
        grid_tokens.append(GridToken(id=f"T{i}", c=token.c, x=x, y=y, tags=tags, type=cell_type))
    return grid_tokens


# ---------------------------------------------------------------------------
# A) evaluate_engine_origin must reach all expected tokens
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exercise_id", _BANK_IDS)
def test_evaluate_origin_covers_all_expected_tokens(exercise_id: str) -> None:
    """Every expected token has a non-negative recognizer row within grid_height.

    This guards against the original bug where grid_height=10 with engine_y_top
    derived from the create view-model (y=6) capped the grader at y=6, leaving
    carries/borrows (y=11) and the division quotient (y=11) unreachable.
    """
    entry = _derive_bank(exercise_id)
    calibration = _calibration_from_evaluate_origin(entry)
    for token in entry.expected:
        row, col = calibration.row_col_for(token.x, token.y)
        assert row >= 0, (
            f"{exercise_id}: token role={token.role} at y={token.y} maps to "
            f"row={row} (negative, outside grid)"
        )
        assert row < entry.grid_height, (
            f"{exercise_id}: token role={token.role} at y={token.y} maps to "
            f"row={row} >= grid_height={entry.grid_height}"
        )
        assert col >= 0, (
            f"{exercise_id}: token role={token.role} at x={token.x} maps to "
            f"col={col} (negative, outside grid)"
        )


@pytest.mark.parametrize("exercise_id", _BANK_IDS)
def test_evaluate_origin_carry_borrow_or_quotient_row_reachable(exercise_id: str) -> None:
    """Top-row tokens (carry, answer) map to a non-negative recognizer row.

    These are the highest-y tokens in each exercise.  Before the coordinate fix,
    they fell at row < 0 and the grader never saw them.

    Note: subtract-256-89 has no borrow cells post 2026-06-19 fix (engine rejects
    them with Placeholder ERRORs).  The topmost tokens for subtraction are the
    top operand (256 at y=6) and the answer row (167 at y=3).
    """
    entry = _derive_bank(exercise_id)
    calibration = _calibration_from_evaluate_origin(entry)
    top_y_tokens = [
        t for t in entry.expected
        if t.role in ("carry", "borrow", "answer") and t.y == entry.answer_y
        or t.y >= max(tt.y for tt in entry.expected)
    ]
    for token in top_y_tokens:
        row, col = calibration.row_col_for(token.x, token.y)
        assert row >= 0, (
            f"{exercise_id}: top token (role={token.role}, y={token.y}) maps to "
            f"row={row} — must be reachable"
        )


# ---------------------------------------------------------------------------
# B) Full-correct submission -> progress == 1.0, all elements OK
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exercise_id", _BANK_IDS)
def test_full_correct_submission_via_calibration_earns_full_progress(
    exercise_id: str,
) -> None:
    """A correct submission assembled via the evaluate calibration grades at 1.0.

    This is the main regression guard for the coordinate risk: if the calibration
    origin is wrong, the recognizer's (row, col) will map to the wrong (x, y) and
    the grader will mark every cell MISSING even when the child drew it correctly.
    """
    entry = _derive_bank(exercise_id)
    calibration = _calibration_from_evaluate_origin(entry)

    # Build GridTokens from expected cells via calibration round-trip.
    grid_tokens = _grid_tokens_for_exercise(entry, calibration)

    client = MockGradingClient(config=GradingConfig(), project_root=PROJECT_ROOT)
    spec = ex.get_exercise(exercise_id)
    created = client.create_session(spec)
    result = client.evaluate(created.session_id, created.ref_id, grid_tokens)

    assert result.progress == pytest.approx(1.0), (
        f"{exercise_id}: expected progress=1.0, got {result.progress}. "
        f"Non-OK elements: {[(e.status, e.position) for e in result.elements if e.status != 'OK']}"
    )
    non_ok = [e for e in result.elements if e.status != "OK"]
    assert not non_ok, (
        f"{exercise_id}: {len(non_ok)} non-OK elements: "
        f"{[(e.status, e.position) for e in non_ok]}"
    )
    assert result.hint is None, (
        f"{exercise_id}: hint present despite full-correct submission: {result.hint}"
    )


# ---------------------------------------------------------------------------
# C) Carry/borrow rows specifically reachable for sum and subtract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exercise_id,carry_or_borrow_role",
    [
        ("product-38-29", "carry"),
        # subtract-256-89 no longer has borrow cells: engine returns Placeholder
        # ERRORs for borrow tokens in this exercise config (probed 2026-06-19).
    ],
)
def test_carry_borrow_rows_grade_correctly(
    exercise_id: str, carry_or_borrow_role: str
) -> None:
    """Carry cells (at y=7 for product) grade as OK when submitted correctly."""
    entry = _derive_bank(exercise_id)
    calibration = _calibration_from_evaluate_origin(entry)
    grid_tokens = _grid_tokens_for_exercise(entry, calibration)

    client = MockGradingClient(config=GradingConfig(), project_root=PROJECT_ROOT)
    spec = ex.get_exercise(exercise_id)
    created = client.create_session(spec)
    result = client.evaluate(created.session_id, created.ref_id, grid_tokens)

    carry_borrow_expected = [
        t for t in entry.expected if t.role == carry_or_borrow_role
    ]
    assert carry_borrow_expected, f"no {carry_or_borrow_role} tokens in {exercise_id}"

    # Grouped elements have their position at the leftmost cell. Verify that
    # every carry/borrow token's cell is covered by an OK element (either its
    # cell is the group's position, or its x is within [position.x, position.x + width)).
    ok_elements = [e for e in result.elements if e.status == "OK"]

    def _cell_covered(x: int, y: int) -> bool:
        for e in ok_elements:
            ex_x, ex_y = e.position
            if ex_y == y and ex_x <= x < ex_x + e.width:
                return True
        return False

    for token in carry_borrow_expected:
        assert _cell_covered(token.x, token.y), (
            f"{exercise_id}: {carry_or_borrow_role} at ({token.x},{token.y}) "
            f"not covered by any OK element. All elements: "
            f"{[(e.status, e.position, e.width) for e in result.elements]}"
        )


# ---------------------------------------------------------------------------
# D) Answer row correctness + wrong-digit grades mistake (subtract-256-89)
# ---------------------------------------------------------------------------


def test_answer_row_grades_correctly() -> None:
    """The subtract-256-89 answer '167' at y=3 (native frame) grades OK with full correct submission."""
    exercise_id = "subtract-256-89"
    entry = _derive_bank(exercise_id)
    calibration = _calibration_from_evaluate_origin(entry)
    grid_tokens = _grid_tokens_for_exercise(entry, calibration)

    client = MockGradingClient(config=GradingConfig(), project_root=PROJECT_ROOT)
    spec = ex.get_exercise(exercise_id)
    created = client.create_session(spec)
    result = client.evaluate(created.session_id, created.ref_id, grid_tokens)

    assert result.progress == pytest.approx(1.0), (
        f"subtract progress={result.progress}; "
        f"non-OK: {[(e.status, e.position) for e in result.elements if e.status != 'OK']}"
    )
    answer_expected = [t for t in entry.expected if t.role == "answer"]
    ok_elements = [e for e in result.elements if e.status == "OK"]

    def _covered(x: int, y: int) -> bool:
        for e in ok_elements:
            ex_x, ex_y = e.position
            if ex_y == y and ex_x <= x < ex_x + e.width:
                return True
        return False

    for token in answer_expected:
        assert _covered(token.x, token.y), (
            f"answer at ({token.x},{token.y}) not covered by OK element: "
            f"{[(e.status, e.position, e.width) for e in result.elements]}"
        )


def test_wrong_answer_digit_grades_mistake() -> None:
    """Wrong units digit in subtract-256-89 answer (167->166) grades progress<1.0 with ERROR."""
    exercise_id = "subtract-256-89"
    entry = _derive_bank(exercise_id)
    calibration = _calibration_from_evaluate_origin(entry)

    # Corrupt the units answer digit: '7' at (12,3) -> '6'.
    # x=12 is the units column in native frame; y=3 is the answer row.
    tokens: list[GridToken] = []
    for i, token in enumerate(entry.expected):
        char = token.c
        if token.role == "answer" and token.x == 12:
            char = "6"  # wrong units digit (correct is '7')
        tags = ["FIXED"] if token.given else []
        if token.role == "borrow":
            tok_type = "REPLACE"
        elif token.role == "carry":
            tok_type = "OVERFLOW"
        else:
            tok_type = "SYMBOL"
        tokens.append(GridToken(id=f"T{i}", c=char, x=token.x, y=token.y, tags=tags, type=tok_type))

    client = MockGradingClient(config=GradingConfig(), project_root=PROJECT_ROOT)
    spec = ex.get_exercise(exercise_id)
    created = client.create_session(spec)
    result = client.evaluate(created.session_id, created.ref_id, tokens)

    assert result.progress < 1.0, (
        f"Wrong answer digit must not grade progress=1.0; got {result.progress:.3f}"
    )
    errors = [e for e in result.elements if e.status == "ERROR"]
    assert errors, (
        "Wrong answer digit must produce at least one ERROR element; "
        f"elements: {[(e.status, e.position) for e in result.elements]}"
    )


# ---------------------------------------------------------------------------
# E) AUTOSHOW filtering in create view-model calibration (regression guard)
# ---------------------------------------------------------------------------


def test_autoshow_tokens_excluded_from_create_calibration_origin() -> None:
    """AUTOSHOW tokens in the create view-model do not skew engine_x_left.

    The captured create fixture has AUTOSHOW tokens at x=5,9 and y=1..3.
    Before the fix, these were included in the min(xs) calculation, setting
    engine_x_left=5 instead of the correct 9 (the leftmost FIXED scaffold token).
    The fixed ``_default_engine_origin`` excludes AUTOSHOW, so engine_x_left=9.
    """
    import json

    FIXTURES = Path(__file__).resolve().parent / "fixtures"
    from src.grading.types import SessionCreated

    data = json.loads((FIXTURES / "create.res.json").read_text(encoding="utf-8"))
    session = SessionCreated.from_api(data)
    cal = build_calibration(session.view_model, {"canvas_px": 512.0})

    # Non-AUTOSHOW FIXED tokens start at x=9 (result bar); AUTOSHOW is at x=5.
    assert cal.engine_x_left == 9, (
        f"AUTOSHOW filtering failed: engine_x_left={cal.engine_x_left}, expected 9. "
        "AUTOSHOW tokens at x=5 must be excluded."
    )
    assert cal.engine_y_top == 6  # max y of FIXED scaffold tokens (unchanged)


# ---------------------------------------------------------------------------
# F) Multi-digit grouping: partial submission reports multi-digit MISSING elements
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exercise_id,expected_missing_width",
    [
        ("subtract-256-89", 2),  # "89" operand groups into one MISSING width=2
    ],
)
def test_partial_submission_emits_multi_digit_missing_element(
    exercise_id: str, expected_missing_width: int
) -> None:
    """With only GIVEN tokens submitted, multi-digit cells group into width>1 MISSING.

    The captured ``event.res`` shows "89" as a single NumberElement with w=2
    rather than two separate w=1 elements. The mock must replicate this so the
    frontend can span the popup across the full multi-digit operand.
    """
    entry = _derive_bank(exercise_id)
    calibration = _calibration_from_evaluate_origin(entry)

    # Submit only the GIVEN scaffold tokens.
    given_tokens = [
        GridToken(id=f"G{i}", c=t.c, x=t.x, y=t.y, tags=["FIXED"])
        for i, t in enumerate(entry.expected)
        if t.given
    ]

    client = MockGradingClient(config=GradingConfig(), project_root=PROJECT_ROOT)
    spec = ex.get_exercise(exercise_id)
    created = client.create_session(spec)
    result = client.evaluate(created.session_id, created.ref_id, given_tokens)

    # At least one MISSING element with width >= 2 must be present.
    missing_elements = [e for e in result.elements if e.status == "MISSING"]
    assert missing_elements, f"{exercise_id}: no MISSING elements in partial submission"
    widths = [e.width for e in missing_elements]
    assert max(widths) >= expected_missing_width, (
        f"{exercise_id}: expected at least one MISSING element with width>={expected_missing_width},"
        f" got widths={widths}"
    )


# ---------------------------------------------------------------------------
# G) Hint spans full multi-digit operand and names the complete string
# ---------------------------------------------------------------------------


def test_hint_spans_full_operand_for_subtract_256_89() -> None:
    """With only GIVEN tokens submitted, the hint for "89" spans the full 2-cell rect.

    Captures the exact fixture-observed behavior: hint targetPositions rect is
    {left:14, top:10, bottom:9, right:16} (right is exclusive, spanning x=14,15)
    and messageArgs.missing == ["89"] (the complete operand string, not ["8","9"]).
    """
    exercise_id = "subtract-256-89"
    entry = _derive_bank(exercise_id)

    # Submit given tokens (256, -, 89, result_bar) to trigger hint.
    given_tokens = [
        GridToken(id=f"G{i}", c=t.c, x=t.x, y=t.y, tags=["FIXED"])
        for i, t in enumerate(entry.expected)
        if t.given
    ]

    client = MockGradingClient(config=GradingConfig(), project_root=PROJECT_ROOT)
    spec = ex.get_exercise(exercise_id)
    created = client.create_session(spec)
    result = client.evaluate(created.session_id, created.ref_id, given_tokens)

    assert result.hint is not None, "expected a hint with all given tokens submitted"

    # Hint must name the full answer string (e.g. "167") not just one digit.
    missing_arg = result.hint.message_args.get("missing", [])
    assert len(missing_arg) == 1, f"missing arg should be a single-element list: {missing_arg}"
    assert len(missing_arg[0]) >= 2, (  # multi-digit: answer "167" has 3 chars
        f"hint must name a multi-digit value, got '{missing_arg[0]}'"
    )

    # The hint target must span at least one full cell.
    assert result.hint.target_positions, "hint must carry target positions"
    rect = result.hint.target_positions[0]
    # For a multi-digit answer the rect width (right - left) equals the digit count.
    span = rect.right - rect.left
    answer_len = len(missing_arg[0])
    assert span == answer_len, (
        f"hint rect span {span} != answer digit count {answer_len}: {rect}"
    )
