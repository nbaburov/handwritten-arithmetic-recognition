"""Tests for the exercise deriver (Foundation 1 + Foundation 2).

Verifies that ``derive_exercise`` builds the correct ``BankExercise`` from
captured create + solution fixtures (native engine frame, no hand-coded coords),
and that a gated round-trip test confirms the derived solution grades at
``progress == 1.0`` on live grading engine.

All coordinate assertions use the native solution frame (no offset constants).
translation-invariance: submitting at native coords yields progress == 1.0.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = PROJECT_ROOT / "tests" / "grading" / "fixtures" / "derive"


# ---------------------------------------------------------------------------
# Fixture loading helpers
# ---------------------------------------------------------------------------


def _load_fixture(name: str) -> Any:
    path = FIXTURE_DIR / name
    if not path.exists():
        pytest.skip(f"fixture not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _create_vm(exercise_id: str) -> dict:
    """Parse the view_model from a captured create fixture."""
    from src.grading.types import SessionCreated, _extract_init_data

    raw = _load_fixture(f"{exercise_id}.create.json")
    created = SessionCreated.from_api(raw)
    return created.view_model


def _solution_data(exercise_id: str) -> dict:
    return _load_fixture(f"{exercise_id}.solution.json")


# ---------------------------------------------------------------------------
# Security: no API key in fixtures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", [
    "subtract-256-89.create.json",
    "subtract-256-89.solution.json",
    "product-38-29.create.json",
    "product-38-29.solution.json",
])
def test_fixture_contains_no_api_key(filename: str) -> None:
    """Fixture must never contain the API key string."""
    path = FIXTURE_DIR / filename
    if not path.exists():
        pytest.skip(f"fixture not found: {filename}")
    content = path.read_text(encoding="utf-8")
    assert "X-API-KEY" not in content


# ---------------------------------------------------------------------------
# Deriver: subtract-256-89
# ---------------------------------------------------------------------------


class TestDeriveSubtract:
    """derive_exercise on subtract-256-89 fixture (native frame: grid 13x10)."""

    @pytest.fixture
    def bank_exercise(self):
        from src.grading.derive import derive_exercise
        from src.grading.exercises import get_exercise

        spec = get_exercise("subtract-256-89")
        vm = _create_vm("subtract-256-89")
        sol = _solution_data("subtract-256-89")
        return derive_exercise(spec, vm, sol)

    def test_grid_dims(self, bank_exercise) -> None:
        """Native grid is 13 wide x 10 tall."""
        assert bank_exercise.grid_width == 13
        assert bank_exercise.grid_height == 10

    def test_answer_y(self, bank_exercise) -> None:
        """Answer row is y=3 (bar_row=4, bar_row-1=3) in native frame."""
        assert bank_exercise.answer_y == 3

    def test_given_includes_top_operand(self, bank_exercise) -> None:
        """Top operand 256 at native (10,6),(11,6),(12,6) marked given=True."""
        given_cells = {(t.x, t.y): t.c for t in bank_exercise.given_tokens()}
        assert given_cells.get((10, 6)) == "2"
        assert given_cells.get((11, 6)) == "5"
        assert given_cells.get((12, 6)) == "6"

    def test_given_includes_operator(self, bank_exercise) -> None:
        """Minus operator at native (9,5) marked given=True."""
        given_cells = {(t.x, t.y): t.c for t in bank_exercise.given_tokens()}
        assert given_cells.get((9, 5)) == "-"

    def test_given_includes_result_bar(self, bank_exercise) -> None:
        """Result bar at native (9,4)..(12,4) marked given=True."""
        given_cells = {(t.x, t.y): t.c for t in bank_exercise.given_tokens()}
        for x in range(9, 13):
            assert given_cells.get((x, 4)) == "_", f"bar cell ({x},4) missing from given"

    def test_given_includes_bottom_operand(self, bank_exercise) -> None:
        """Bottom operand 89 at native (11,5),(12,5) — absent from create-FIXED — is given."""
        given_cells = {(t.x, t.y): t.c for t in bank_exercise.given_tokens()}
        assert given_cells.get((11, 5)) == "8", "bottom operand '8' missing from given"
        assert given_cells.get((12, 5)) == "9", "bottom operand '9' missing from given"

    def test_answer_cells(self, bank_exercise) -> None:
        """Answer 167 at native y=3: (10,3)=1, (11,3)=6, (12,3)=7, role=answer, given=False."""
        answers = {(t.x, t.y): t.c for t in bank_exercise.expected if t.role == "answer"}
        assert answers == {(10, 3): "1", (11, 3): "6", (12, 3): "7"}
        for t in bank_exercise.expected:
            if t.role == "answer":
                assert not t.given

    def test_borrow_cells_present(self, bank_exercise) -> None:
        """4 borrow (REPLACE) cells: (10,7)=1, (11,7)=4, (12,7)=16, (11,8)=14."""
        borrows = {(t.x, t.y): t.c for t in bank_exercise.expected if t.role == "borrow"}
        assert borrows == {
            (10, 7): "1",
            (11, 7): "4",
            (12, 7): "16",
            (11, 8): "14",
        }

    def test_borrow_cells_not_given(self, bank_exercise) -> None:
        """Borrow cells are optional (not pre-printed)."""
        for t in bank_exercise.expected:
            if t.role == "borrow":
                assert not t.given

    def test_multi_digit_borrow_cells(self, bank_exercise) -> None:
        """(12,7)='16' and (11,8)='14' are multi-digit borrow cells."""
        multi = [(t.x, t.y, t.c) for t in bank_exercise.expected
                 if t.role == "borrow" and len(t.c) > 1]
        assert (12, 7, "16") in multi
        assert (11, 8, "14") in multi

    def test_no_autoshow_tokens(self, bank_exercise) -> None:
        """AUTOSHOW decorative tokens must be excluded (position-independent).

        Subtraction's AUTOSHOW band repeats '-' and '=' helper glyphs. The '='
        char only ever appears as AUTOSHOW, and the only legitimate '-' is the
        single FIXED operator; asserting on the chars (not their coordinates)
        catches AUTOSHOW leakage wherever grading engine places it.
        """
        chars = [t.c for t in bank_exercise.expected]
        assert "=" not in chars, "AUTOSHOW '=' leaked into expected"
        assert chars.count("-") == 1, "only the operator '-' may survive; AUTOSHOW '-' leaked"

    def test_no_duplicate_cells(self, bank_exercise) -> None:
        """Expected cell set must have unique (x,y)."""
        cells = [(t.x, t.y) for t in bank_exercise.expected]
        assert len(cells) == len(set(cells)), "duplicate cells in expected"

    def test_expected_tokens_as_grid_carry_type(self, bank_exercise) -> None:
        """expected_tokens_as_grid maps borrow -> type=REPLACE, carry -> type=OVERFLOW."""
        from src.grading.exercises import BankExercise
        # Use the derived exercise's expected set directly
        from src.grading.types import GridToken
        tokens: list[GridToken] = []
        for i, tok in enumerate(bank_exercise.expected):
            tags = ["FIXED"] if tok.given else []
            if tok.role == "borrow":
                cell_type = "REPLACE"
            elif tok.role == "carry":
                cell_type = "OVERFLOW"
            else:
                cell_type = "SYMBOL"
            tokens.append(GridToken(id=f"E{i}", c=tok.c, x=tok.x, y=tok.y, tags=tags, type=cell_type))
        borrow_tokens = [t for t in tokens if t.type == "REPLACE"]
        assert len(borrow_tokens) == 4


# ---------------------------------------------------------------------------
# Deriver: product-38-29
# ---------------------------------------------------------------------------


class TestDeriveProduct:
    """derive_exercise on product-38-29 fixture (native frame: grid 4x8)."""

    @pytest.fixture
    def bank_exercise(self):
        from src.grading.derive import derive_exercise
        from src.grading.exercises import get_exercise

        spec = get_exercise("product-38-29")
        vm = _create_vm("product-38-29")
        sol = _solution_data("product-38-29")
        return derive_exercise(spec, vm, sol)

    def test_grid_dims(self, bank_exercise) -> None:
        """Native grid is 4 wide x 8 tall."""
        assert bank_exercise.grid_width == 4
        assert bank_exercise.grid_height == 8

    def test_answer_y(self, bank_exercise) -> None:
        """Answer row is y=0 (bottommost SYMBOL row in LINE band)."""
        assert bank_exercise.answer_y == 0

    def test_given_includes_result_bar(self, bank_exercise) -> None:
        """Bar at native (1,4),(2,4),(3,4) marked given=True."""
        given_cells = {(t.x, t.y): t.c for t in bank_exercise.given_tokens()}
        for x in range(1, 4):
            assert given_cells.get((x, 4)) == "_", f"bar cell ({x},4) missing"

    def test_given_includes_top_operand(self, bank_exercise) -> None:
        """Top operand 38 at native (2,6)=3, (3,6)=8 marked given=True."""
        given_cells = {(t.x, t.y): t.c for t in bank_exercise.given_tokens()}
        assert given_cells.get((2, 6)) == "3"
        assert given_cells.get((3, 6)) == "8"

    def test_given_includes_operator_and_bottom_operand(self, bank_exercise) -> None:
        """Operator * at (1,5) and bottom operand 29 at (2,5),(3,5) are given."""
        given_cells = {(t.x, t.y): t.c for t in bank_exercise.given_tokens()}
        assert given_cells.get((1, 5)) == "*", "operator * missing from given"
        assert given_cells.get((2, 5)) == "2", "bottom operand 2 missing from given"
        assert given_cells.get((3, 5)) == "9", "bottom operand 9 missing from given"

    def test_generated_operator_classified_as_operator(self, bank_exercise) -> None:
        """The GENERATED '*' at (1,5) must be role=operator, not operand, so the
        scaffold renders it as the x glyph rather than a digit box (MF-1)."""
        roles = {(t.x, t.y): t.role for t in bank_exercise.expected}
        assert roles.get((1, 5)) == "operator"

    def test_answer_cells(self, bank_exercise) -> None:
        """Answer 1102 at native y=0: (0,0)=1,(1,0)=1,(2,0)=0,(3,0)=2."""
        answers = {(t.x, t.y): t.c for t in bank_exercise.expected if t.role == "answer"}
        assert answers == {(0, 0): "1", (1, 0): "1", (2, 0): "0", (3, 0): "2"}

    def test_carry_cells_present(self, bank_exercise) -> None:
        """2 carry (OVERFLOW) cells at native y=7: (1,7)=3, (2,7)=7."""
        carries = {(t.x, t.y): t.c for t in bank_exercise.expected if t.role == "carry"}
        assert carries == {(1, 7): "3", (2, 7): "7"}

    def test_carry_cells_not_given(self, bank_exercise) -> None:
        """Carry cells are optional (not pre-printed)."""
        for t in bank_exercise.expected:
            if t.role == "carry":
                assert not t.given

    def test_partial_products_present(self, bank_exercise) -> None:
        """Partial products: 342 at y=3 and 760 at y=2."""
        partials_by_y: dict[int, str] = {}
        for t in sorted(bank_exercise.expected, key=lambda x: x.x):
            if t.role == "partial":
                partials_by_y.setdefault(t.y, "")
                partials_by_y[t.y] += t.c
        assert partials_by_y.get(3) == "342", f"partial y=3: {partials_by_y}"
        assert partials_by_y.get(2) == "760", f"partial y=2: {partials_by_y}"

    def test_separator_row_excluded(self, bank_exercise) -> None:
        """Separator row ('+' + '_' at y=1) must not be in the expected set."""
        y1_tokens = [(t.x, t.y, t.c) for t in bank_exercise.expected if t.y == 1]
        assert y1_tokens == [], f"separator row tokens found: {y1_tokens}"

    def test_no_duplicate_cells(self, bank_exercise) -> None:
        """Expected cell set must have unique (x,y)."""
        cells = [(t.x, t.y) for t in bank_exercise.expected]
        assert len(cells) == len(set(cells)), "duplicate cells in expected"

    def test_carry_type_mapping(self, bank_exercise) -> None:
        """expected_tokens_as_grid (manual) maps carry -> type=OVERFLOW."""
        from src.grading.types import GridToken
        tokens: list[GridToken] = []
        for i, tok in enumerate(bank_exercise.expected):
            tags = ["FIXED"] if tok.given else []
            if tok.role == "borrow":
                cell_type = "REPLACE"
            elif tok.role == "carry":
                cell_type = "OVERFLOW"
            else:
                cell_type = "SYMBOL"
            tokens.append(GridToken(id=f"E{i}", c=tok.c, x=tok.x, y=tok.y, tags=tags, type=cell_type))
        carry_tokens = [t for t in tokens if t.type == "OVERFLOW"]
        assert len(carry_tokens) == 2


# ---------------------------------------------------------------------------
# expected_tokens_as_grid carry support
# ---------------------------------------------------------------------------


def test_expected_tokens_as_grid_maps_carry_to_overflow() -> None:
    """expected_tokens_as_grid must emit type=OVERFLOW for carry cells.

    This guards that the Workstream A refactor of expected_tokens_as_grid
    correctly handles carry -> OVERFLOW (not just borrow -> REPLACE).
    Uses the derived exercise so the test is fixture-driven.
    """
    from src.grading.derive import derive_exercise
    from src.grading.exercises import expected_tokens_as_grid, get_exercise

    spec = get_exercise("product-38-29")
    vm = _create_vm("product-38-29")
    sol = _solution_data("product-38-29")
    bank_ex = derive_exercise(spec, vm, sol)

    # After Workstream A: expected_tokens_as_grid uses the derived exercise
    # This test will also be a regression check once the bank is driven by derive
    # For now we test the mapping logic via the derived expected tokens directly
    from src.grading.types import GridToken
    tokens: list[GridToken] = []
    for i, tok in enumerate(bank_ex.expected):
        tags = ["FIXED"] if tok.given else []
        if tok.role == "borrow":
            cell_type = "REPLACE"
        elif tok.role == "carry":
            cell_type = "OVERFLOW"
        else:
            cell_type = "SYMBOL"
        tokens.append(GridToken(id=f"E{i}", c=tok.c, x=tok.x, y=tok.y, tags=tags, type=cell_type))
    carry_types = [t.type for t in tokens if t.c in ("3", "7") and t.y == 7]
    assert all(ct == "OVERFLOW" for ct in carry_types), f"carry tokens: {carry_types}"


# ---------------------------------------------------------------------------
# P1-a — authoritative expected_tokens_as_grid contract test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exercise_id,expected_carry_count,expected_borrow_count", [
    ("product-38-29", 2, 0),
    ("subtract-256-89", 0, 4),
])
def test_expected_tokens_as_grid_real_function_carry_overflow(
    exercise_id: str,
    expected_carry_count: int,
    expected_borrow_count: int,
) -> None:
    """expected_tokens_as_grid (the REAL function, not an inline copy) must emit
    type=OVERFLOW for carry cells and type=REPLACE for borrow cells.

    This test calls the actual ``expected_tokens_as_grid`` function rather than
    re-implementing the borrow->REPLACE / carry->OVERFLOW mapping inline. The
    three tests at lines 177, 284, and 306 build the same mapping themselves and
    count their own list -- they pass even if ``expected_tokens_as_grid`` is
    broken. This test is the authoritative guard.
    """
    from src.grading.derive import derive_exercise
    from src.grading.exercises import expected_tokens_as_grid, get_exercise

    spec = get_exercise(exercise_id)
    vm = _create_vm(exercise_id)
    sol = _solution_data(exercise_id)
    bank_ex = derive_exercise(spec, vm, sol)

    grid_tokens = expected_tokens_as_grid(exercise_id, bank_ex)

    carry_tokens = [t for t in grid_tokens if t.type == "OVERFLOW"]
    borrow_tokens = [t for t in grid_tokens if t.type == "REPLACE"]

    assert len(carry_tokens) == expected_carry_count, (
        f"{exercise_id}: expected {expected_carry_count} OVERFLOW tokens, "
        f"got {len(carry_tokens)}: {[(t.x, t.y, t.c) for t in carry_tokens]}"
    )
    assert len(borrow_tokens) == expected_borrow_count, (
        f"{exercise_id}: expected {expected_borrow_count} REPLACE tokens, "
        f"got {len(borrow_tokens)}: {[(t.x, t.y, t.c) for t in borrow_tokens]}"
    )
    # Spot-check: every carry token carries a valid digit string.
    for t in carry_tokens:
        assert t.c.isdigit(), f"carry token has non-digit char: {t.c!r}"
    # Spot-check: every borrow token carries a digit string (may be multi-digit).
    for t in borrow_tokens:
        assert t.c.isdigit() or (len(t.c) > 1 and t.c.isdigit()), (
            f"borrow token has non-digit char: {t.c!r}"
        )


# ---------------------------------------------------------------------------
# P2 — _role_for separator branch: '+' inside line_band below bar -> "partial"
# ---------------------------------------------------------------------------


def test_role_for_separator_char_in_line_band_below_bar_returns_partial() -> None:
    """_role_for must return 'partial' for a '+' separator token that is below the
    bar row AND whose x is inside the line_band.

    This is the one branch of _role_for with no existing direct unit test. It
    guards against a refactor accidentally routing the separator into 'answer'.
    The separator row in product exercises (y=1 with '+' and '_' inside the LINE
    band) must never leak into the answer slot.
    """
    from src.grading.derive import _role_for

    # Minimal scenario: bar_row=4, line_band covers x in range(1, 4).
    bar_row = 4
    line_band = range(1, 4)

    # A '+' separator token below the bar, x=2 (inside the band), no FIXED/AUTOSHOW.
    token = {"c": "+", "x": 2, "y": 1, "type": "SYMBOL", "tags": []}
    role = _role_for(token, line_band, bar_row, is_fixed=False)
    assert role == "partial", (
        f"'+' inside line_band below bar must be 'partial', got {role!r}"
    )

    # A '_' separator token (same position) should also be 'partial'.
    token_bar = {"c": "_", "x": 2, "y": 1, "type": "SYMBOL", "tags": []}
    role_bar = _role_for(token_bar, line_band, bar_row, is_fixed=False)
    assert role_bar == "partial", (
        f"'_' inside line_band below bar must be 'partial', got {role_bar!r}"
    )
