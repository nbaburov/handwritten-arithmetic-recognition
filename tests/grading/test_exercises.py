"""Tests for the fixed exercise bank.

Covers bank integrity (two CMS exercises, unique ids, validated operands),
that every entry carries a complete and self-consistent expected solution,
and that out-of-spec operands are rejected at construction. The dead
inline-spec path and the four non-engine exercises have been removed; these
tests guard only what remains.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.grading import exercises as ex
from src.grading.derive import derive_exercise
from src.grading.exercises import (
    BankExercise,
    ExpectedToken,
    build_exercise_spec,
    expected_tokens_as_grid,
    get_exercise,
    list_cms_exercises,
    list_exercises,
)
from src.grading.mock_client import MockGradingClient
from src.grading.types import ExerciseSpec, SessionCreated
from src.core.run_config import GradingConfig

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES_DIR = _PROJECT_ROOT / "tests" / "grading" / "fixtures" / "derive"


def _derive_bank(exercise_id: str) -> BankExercise:
    """Derive a BankExercise from captured fixtures (no API key, no server)."""
    client = MockGradingClient(config=GradingConfig(), project_root=_PROJECT_ROOT)
    spec = ex.get_exercise(exercise_id)
    created = client.create_session(spec)
    return client._sessions[created.session_id].exercise

# The bank now contains exactly two CMS-backed exercises.
_EXPECTED_IDS = {"subtract-256-89", "product-38-29"}


# --- bank integrity ---------------------------------------------------------


def test_bank_has_exactly_two_exercises() -> None:
    specs = list_exercises()
    assert len(specs) == 2
    assert {s.exercise_id for s in specs} == _EXPECTED_IDS


def test_all_bank_exercises_have_cms_id() -> None:
    """Every bank entry must carry a CMS UUID (no inline-spec exercises)."""
    for spec in list_exercises():
        assert spec.cms_exercise_id, (
            f"exercise {spec.exercise_id!r} has no cms_exercise_id; "
            "every bank entry must be CMS-backed"
        )


def test_list_cms_exercises_returns_both() -> None:
    cms = list_cms_exercises()
    assert {s.exercise_id for s in cms} == _EXPECTED_IDS


def test_bank_ids_are_unique() -> None:
    ids = [s.exercise_id for s in list_exercises()]
    assert len(ids) == len(set(ids))


def test_list_exercises_returns_exercise_specs() -> None:
    for spec in list_exercises():
        assert isinstance(spec, ExerciseSpec)
        assert spec.exercise_id
        assert spec.prompt
        assert spec.template_args


def test_get_exercise_returns_matching_spec() -> None:
    spec = get_exercise("subtract-256-89")
    assert spec.template_name == "SUBTRACT_SHORT"
    assert spec.template_args == ["256", "89"]


def test_get_exercise_product_returns_correct_spec() -> None:
    spec = get_exercise("product-38-29")
    assert spec.template_name == "PRODUCT_SHORT"
    assert spec.template_args == ["38", "29"]
    assert spec.cms_exercise_id == "29612ec7-5d46-4816-8443-0ef16ae61cef"


def test_get_exercise_unknown_id_raises_keyerror_does_not_exist() -> None:
    with pytest.raises(KeyError):
        get_exercise("does-not-exist")


def test_get_exercise_unknown_id_raises_keyerror_nope() -> None:
    with pytest.raises(KeyError):
        get_exercise("nope")


# --- expected solution consistency -----------------------------------------


@pytest.mark.parametrize("exercise_id", [s.exercise_id for s in list_exercises()])
def test_every_exercise_carries_a_complete_expected_set(exercise_id: str) -> None:
    bank = _derive_bank(exercise_id)
    assert isinstance(bank, BankExercise)
    assert bank.expected, "exercise must carry an expected solution"
    # Expected cells are unique by grid cell (no two tokens collide).
    cells = [t.cell() for t in bank.expected]
    assert len(cells) == len(set(cells)), "expected cells must be unique"
    # There is at least one child-filled cell (something to grade).
    assert bank.expected_to_fill(), "exercise must have child-filled cells"
    # And at least one given scaffold cell (the printed problem).
    assert bank.given_tokens(), "exercise must have given scaffold cells"


@pytest.mark.parametrize(
    "exercise_id,answer,answer_y",
    [
        ("subtract-256-89", "167", 3),   # answer row y=3 in native solution frame
        ("product-38-29", "1102", 0),
    ],
)
def test_answer_row_spells_the_correct_result(
    exercise_id: str, answer: str, answer_y: int
) -> None:
    bank = _derive_bank(exercise_id)
    assert bank.answer_y == answer_y
    answer_cells = sorted(
        (t for t in bank.expected if t.role == "answer"), key=lambda t: t.x
    )
    spelled = "".join(t.c for t in answer_cells)
    assert spelled == answer
    assert all(t.y == answer_y for t in answer_cells)


def test_subtraction_has_borrow_cells() -> None:
    """256 - 89 encodes the 4 optional borrow/replacement cells.

    Derived from native solution frame (verified 2026-06-21 via derive_exercise):
    Cells in native frame: (10,7)=1, (11,7)=4, (12,7)=16, (11,8)=14.
    They are optional (answer-only still scores 1.0 in mock grading).
    """
    bank = _derive_bank("subtract-256-89")
    borrows = {(t.x, t.y): t.c for t in bank.expected if t.role == "borrow"}
    assert borrows == {(10, 7): "1", (11, 7): "4", (12, 7): "16", (11, 8): "14"}
    # All borrows are child-fill (not pre-printed scaffold).
    assert all(not t.given for t in bank.expected if t.role == "borrow")


def test_product_carries_required_carry_row() -> None:
    """38 x 29 has overflow/carry digits at y=7; the bank encodes them."""
    bank = _derive_bank("product-38-29")
    carries = [t for t in bank.expected if t.role == "carry"]
    assert len(carries) == 2
    assert {t.c for t in carries} == {"3", "7"}
    assert all(t.y == 7 for t in carries)


def test_product_has_two_partial_products() -> None:
    """38 x 29: partial product rows 342 (y=3) and 760 (y=2)."""
    bank = _derive_bank("product-38-29")
    partials_by_y: dict[int, str] = {}
    for t in sorted(bank.expected, key=lambda t: t.x):
        if t.role == "partial":
            partials_by_y.setdefault(t.y, "")
            partials_by_y[t.y] += t.c
    # y=3: "342", y=2: "760"
    assert partials_by_y.get(3) == "342", f"partial y=3: {partials_by_y}"
    assert partials_by_y.get(2) == "760", f"partial y=2: {partials_by_y}"


def test_product_given_includes_operands_operator_and_bar() -> None:
    """The product scaffold displays 38, x, 29, and the result bar as pre-printed.

    The result bar at y=4 x=1..3 is engine FIXED.  The operands 38 (y=6 x=2..3),
    operator '*' (y=5 x=1), and bottom operand 29 (y=5 x=2..3) are marked
    given=True so the demo scaffold shows the full problem.  All are submitted
    as FIXED tokens so engine credits them without the child drawing them.
    """
    bank = _derive_bank("product-38-29")
    given = bank.given_tokens()
    # Bar cells at y=4 x=1..3 (native frame, 3 cells)
    bar_cells = [(t.x, t.y, t.c) for t in given if t.c == "_"]
    assert sorted(bar_cells) == [(1, 4, "_"), (2, 4, "_"), (3, 4, "_")], (
        f"bar cells: {bar_cells}"
    )
    # Operand 38 at y=6
    top_op = sorted((t.x, t.c) for t in given if t.y == 6)
    assert top_op == [(2, "3"), (3, "8")], f"top operand: {top_op}"
    # Operator '*' at y=5 x=1 — SOLUTION token (not create-FIXED), so role='operand' in derive.
    # Verify it exists as a given cell with c='*' at the expected position.
    op_cells = [(t.x, t.y, t.c) for t in given if t.c == "*"]
    assert op_cells == [(1, 5, "*")], f"operator cell: {op_cells}"
    # Bottom operand 29 at y=5 x=2..3
    bot_op = sorted((t.x, t.c) for t in given if t.y == 5 and t.c in "0123456789")
    assert bot_op == [(2, "2"), (3, "9")], f"bottom operand: {bot_op}"
    # Total given count: 3 bar + 2 top + 1 op + 2 bot = 8
    assert len(given) == 8, f"expected 8 given tokens, got {len(given)}: {given}"


def test_expected_tokens_as_grid_round_trips_to_a_correct_submission() -> None:
    """The full expected set rendered to GridTokens is a correct submission."""
    bank = _derive_bank("subtract-256-89")
    tokens = expected_tokens_as_grid("subtract-256-89", bank)
    by_cell = {(t.x, t.y): t.c for t in tokens}
    for exp in bank.expected:
        assert by_cell[(exp.x, exp.y)] == exp.c


def test_subtract_grid_dimensions_match_engine_native_frame() -> None:
    """Grid width=13 height=10 matches the native solution frame from the derive fixture.

    The native frame (from session/solution) uses x=9..12 for columns, y=3..8
    for rows. width=13 covers all columns; height=10 covers all rows including
    the borrow rows at y=7..8.
    """
    bank = _derive_bank("subtract-256-89")
    assert bank.grid_width == 13
    assert bank.grid_height == 10


def test_subtract_evaluate_engine_origin() -> None:
    """evaluate_engine_origin=(9,8): leftmost col x=9, topmost used row y=8 (borrow row)."""
    bank = _derive_bank("subtract-256-89")
    assert bank.evaluate_engine_origin == (9, 8)


def test_subtract_given_includes_both_operands_operator_and_bar() -> None:
    """The subtraction scaffold displays 256, '-', 89, and the result bar as given.

    The child only fills the answer row (167) and optional borrows.
    Coordinates are native solution-frame values from the derive fixture.
    """
    bank = _derive_bank("subtract-256-89")
    given = bank.given_tokens()
    # Top operand 256 at y=6
    top256 = sorted((t.x, t.c) for t in given if t.role == "operand" and t.y == 6)
    assert top256 == [(10, "2"), (11, "5"), (12, "6")], f"top operand: {top256}"
    # Operator '-' at y=5 x=9
    ops = [(t.x, t.y, t.c) for t in given if t.role == "operator"]
    assert ops == [(9, 5, "-")], f"operator: {ops}"
    # Bottom operand 89 at y=5 x=11..12
    bot89 = sorted((t.x, t.c) for t in given if t.role == "operand" and t.y == 5)
    assert bot89 == [(11, "8"), (12, "9")], f"bottom operand 89: {bot89}"
    # Result bar '_' at y=4 x=9..12
    bar = sorted((t.x, t.y) for t in given if t.role == "result_bar")
    assert bar == [(9, 4), (10, 4), (11, 4), (12, 4)], f"bar: {bar}"
    # Child fill: 3 answer cells + 4 optional borrow cells.
    fill = bank.expected_to_fill()
    assert {t.role for t in fill} == {"answer", "borrow"}, f"child fill roles: {fill}"
    assert len([t for t in fill if t.role == "answer"]) == 3
    assert len([t for t in fill if t.role == "borrow"]) == 4


def test_product_grid_dimensions_match_engine_native_frame() -> None:
    """Grid width=4 height=8 matches the native solution frame from the derive fixture."""
    bank = _derive_bank("product-38-29")
    assert bank.grid_width == 4
    assert bank.grid_height == 8


def test_product_evaluate_engine_origin() -> None:
    """evaluate_engine_origin=(0,7): leftmost col x=0, topmost row y=7 (carry row)."""
    bank = _derive_bank("product-38-29")
    assert bank.evaluate_engine_origin == (0, 7)


# --- out-of-spec rejection --------------------------------------------------


def test_product_rejects_two_decimal_operands() -> None:
    with pytest.raises(ValueError):
        build_exercise_spec(
            exercise_id="bad-product",
            title="bad",
            prompt="bad",
            template_name="PRODUCT_SHORT",
            template_args=["1.5", "2.5"],
            attributes={},
        )


def test_product_allows_one_decimal_operand() -> None:
    spec = build_exercise_spec(
        exercise_id="ok-product",
        title="ok",
        prompt="ok",
        template_name="PRODUCT_SHORT",
        template_args=["1.5", "2"],
        attributes={},
    )
    assert spec.template_args == ["1.5", "2"]


def test_division_rejects_any_decimal_operand() -> None:
    with pytest.raises(ValueError):
        build_exercise_spec(
            exercise_id="bad-division",
            title="bad",
            prompt="bad",
            template_name="DIVISION_SHORT",
            template_args=["8.4", "4"],
            attributes={},
        )


def test_expected_token_cell_helper() -> None:
    token = ExpectedToken(x=3, y=4, c="7", role="answer")
    assert token.cell() == (3, 4)
    assert token.given is False


# --- cms_exercise_id guard --------------------------------------------------


def test_build_exercise_spec_without_cms_id_creates_spec() -> None:
    """build_exercise_spec without a cms_exercise_id still creates the spec.

    The bank always sets one, but the function itself does not enforce it;
    the HTTP client enforces it at create_session time.
    """
    spec = build_exercise_spec(
        exercise_id="test-no-cms",
        title="test",
        prompt="test",
        template_name="SUM_SHORT",
        template_args=["1", "2"],
        attributes={},
    )
    assert spec.cms_exercise_id == ""
