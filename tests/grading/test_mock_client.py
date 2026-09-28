"""Tests for the in-process MockGradingClient grader.

Covers create (verbatim replay for both CMS fixtures), evaluate (correct /
partial / wrong-digit), info (marks earned only at full progress), and solution.
Shapes are pinned against the contract types so the mock can never drift from
the real engine.

The two surviving bank exercises are:
  - subtract-256-89: coordinate-corrected to engine evaluate frame (2026-06-19);
    no borrow cells (engine rejects them with Placeholder ERRORs); answer-only
    submission reaches progress == 1.0.
  - product-38-29: operands/operator now given=True so the scaffold shows the
    problem; carries + partial products are step cells.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from src.grading import exercises as ex
from src.grading.client import GradingClient
from src.grading.exercises import BankExercise
from src.grading.mock_client import MockGradingClient
from src.grading.types import (
    EvalResult,
    GridToken,
    ScoringInfo,
    SessionCreated,
)
from src.core.run_config import GradingConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ID = "subtract-256-89"
_BANK_IDS = [s.exercise_id for s in ex.list_exercises()]


@pytest.fixture
def client() -> MockGradingClient:
    return MockGradingClient(config=GradingConfig(), project_root=PROJECT_ROOT)


def _derive_bank(client: MockGradingClient, exercise_id: str) -> tuple[SessionCreated, BankExercise]:
    """Create a session and return (SessionCreated, derived BankExercise).

    MockGradingClient.create_session calls derive_exercise internally and
    stashes the result. We access it via the internal session store.
    """
    spec = ex.get_exercise(exercise_id)
    created = client.create_session(spec)
    mock_session = client._sessions[created.session_id]
    return created, mock_session.exercise


def _correct_tokens(bank: BankExercise) -> list[GridToken]:
    """Every expected cell as a submittable token (a correct submission)."""
    return ex.expected_tokens_as_grid(bank.exercise_id, bank)


def _given_only_tokens(bank: BankExercise) -> list[GridToken]:
    """Only the pre-given scaffold cells, so all child-filled cells are MISSING."""
    return [t for t in _correct_tokens(bank) if "FIXED" in t.tags]


def _child_fill_tokens(bank: BankExercise) -> list[GridToken]:
    """Only the child-filled cells (no scaffold)."""
    return [t for t in _correct_tokens(bank) if "FIXED" not in t.tags]


# --- protocol conformance ---------------------------------------------------


def test_mock_satisfies_client_protocol(client: MockGradingClient) -> None:
    assert isinstance(client, GradingClient)


# --- create -----------------------------------------------------------------


def test_create_fixture_replays_captured_view_model(client: MockGradingClient) -> None:
    """The 256-89 exercise replays the captured create session verbatim."""
    spec = ex.get_exercise(FIXTURE_ID)
    created = client.create_session(spec)
    assert isinstance(created, SessionCreated)
    # The captured derive fixture's session id and interaction ref id, byte-aligned.
    assert created.session_id == "01670e42-760e-487b-b2c3-a4e6b7630d1f"
    assert created.ref_id == "drYEz"
    assert created.marks_total == 1
    arith = created.view_model["view"]["elements"][0]["interactions"][0]["ans"]
    assert arith["type"] == "ARITHMETIC"
    assert arith["width"] == 13 and arith["height"] == 10
    assert arith["tokens"], "captured scaffold tokens must be present"


def test_create_product_replays_captured_view_model(client: MockGradingClient) -> None:
    """The product-38-29 exercise replays the captured create session verbatim."""
    spec = ex.get_exercise("product-38-29")
    created = client.create_session(spec)
    assert isinstance(created, SessionCreated)
    assert created.session_id  # real UUID from the captured fixture
    assert created.ref_id == "K0HwQ"  # ref from the captured fixture
    assert created.marks_total == 1
    arith = created.view_model["view"]["elements"][0]["interactions"][0]["ans"]
    assert arith["type"] == "ARITHMETIC"
    assert arith["width"] == 4 and arith["height"] == 8
    # Only 3 scaffold tokens: the result bar at y=4, x=1..3
    tokens = arith["tokens"]
    assert len(tokens) == 3
    assert all(t["c"] == "_" and t["y"] == 4 for t in tokens)


def test_create_unknown_exercise_raises(client: MockGradingClient) -> None:
    """Unknown exercise (no derive fixture) raises GradingError in mock mode."""
    from src.grading.client import GradingError
    from src.grading.types import ExerciseSpec

    bogus = ExerciseSpec(
        exercise_id="not-in-bank",
        title="x",
        prompt="x",
        template_name="SUM_SHORT",
        template_args=["1", "2"],
    )
    with pytest.raises(GradingError):
        client.create_session(bogus)


# --- evaluate: correct submission per exercise ------------------------------


@pytest.mark.parametrize("exercise_id", _BANK_IDS)
def test_correct_submission_all_ok_full_progress(
    client: MockGradingClient, exercise_id: str
) -> None:
    created, bank = _derive_bank(client, exercise_id)
    result = client.evaluate(created.session_id, created.ref_id, _correct_tokens(bank))
    assert isinstance(result, EvalResult)
    statuses = {e.status for e in result.elements}
    assert statuses == {"OK"}, f"all elements OK for a correct submission, got {statuses}"
    assert result.progress == pytest.approx(1.0)
    assert result.feedback == []
    assert result.hint is None, "no hint once everything is correct"


@pytest.mark.parametrize("exercise_id", _BANK_IDS)
def test_correct_submission_earns_marks(
    client: MockGradingClient, exercise_id: str
) -> None:
    created, bank = _derive_bank(client, exercise_id)
    client.evaluate(created.session_id, created.ref_id, _correct_tokens(bank))
    info = client.info(created.session_id)
    assert isinstance(info, ScoringInfo)
    assert info.finished is True
    assert info.marks_earned == info.marks_total == 1


# --- evaluate: partial submission per exercise ------------------------------


@pytest.mark.parametrize("exercise_id", _BANK_IDS)
def test_partial_submission_reports_missing(
    client: MockGradingClient, exercise_id: str
) -> None:
    created, bank = _derive_bank(client, exercise_id)
    # Submit only the given scaffold: every child-filled cell is MISSING.
    result = client.evaluate(created.session_id, created.ref_id, _given_only_tokens(bank))
    missing = [e for e in result.elements if e.status == "MISSING"]
    assert missing, "a partial submission must report MISSING elements"
    # MISSING elements carry no echoed symbol ids (matches the captured shape).
    assert all(e.symbol_id_list == [] for e in missing)
    assert result.progress < 1.0
    # And a positioned hint for the next missing cell.
    assert result.hint is not None
    assert result.hint.target_positions, "hint must carry a target rect"


@pytest.mark.parametrize("exercise_id", _BANK_IDS)
def test_partial_submission_earns_no_marks(
    client: MockGradingClient, exercise_id: str
) -> None:
    created, bank = _derive_bank(client, exercise_id)
    client.evaluate(created.session_id, created.ref_id, _given_only_tokens(bank))
    info = client.info(created.session_id)
    assert info.finished is False
    assert info.marks_earned == 0


def test_partial_progress_increases_toward_full(client: MockGradingClient) -> None:
    """Filling more elements raises progress monotonically toward 1.0.

    Uses product-38-29 which has 12 child-fill cells across multiple
    independently-graded elements (carries + partials + answer), so a half
    submission yields 0 < p < 1.  subtract-256-89 has a single answer element;
    partial digit submission stays at 0 until the full element is filled.
    """
    _MULTI_ELEMENT_ID = "product-38-29"
    created, bank = _derive_bank(client, _MULTI_ELEMENT_ID)
    fills = _child_fill_tokens(bank)
    given = _given_only_tokens(bank)
    half = given + fills[: len(fills) // 2]
    p_half = client.evaluate(created.session_id, created.ref_id, half).progress
    p_full = client.evaluate(
        created.session_id, created.ref_id, given + fills
    ).progress
    assert 0.0 < p_half < p_full == pytest.approx(1.0)


# --- evaluate: wrong-digit submission per exercise --------------------------


@pytest.mark.parametrize("exercise_id", _BANK_IDS)
def test_wrong_digit_submission_emits_error_feedback(
    client: MockGradingClient, exercise_id: str
) -> None:
    created, bank = _derive_bank(client, exercise_id)
    tokens = _correct_tokens(bank)
    # Corrupt one child-filled (answer) cell to the wrong character.
    answer_cells = {(t.x, t.y) for t in bank.expected if t.role == "answer"}
    corrupted: list[GridToken] = []
    flipped = False
    for token in tokens:
        if not flipped and (token.x, token.y) in answer_cells:
            wrong = "0" if token.c != "0" else "9"
            corrupted.append(replace(token, c=wrong))
            flipped = True
        else:
            corrupted.append(token)
    assert flipped, "test must corrupt one answer cell"
    result = client.evaluate(created.session_id, created.ref_id, corrupted)
    # An ERROR element at the corrupted cell.
    errors = [e for e in result.elements if e.status == "ERROR"]
    assert errors, "a wrong digit must produce an ERROR element"
    # And an EvalFeedback entry for the wrong cell.
    assert result.feedback, "a wrong digit must produce a feedback entry"
    assert any(f.message_type == "IncorrectValue" for f in result.feedback)
    assert result.progress < 1.0


def test_extra_token_at_unexpected_cell_is_feedback_not_overwrite(
    client: MockGradingClient,
) -> None:
    """A token at a cell no symbol is expected is reported, never silently kept."""
    created, bank = _derive_bank(client, FIXTURE_ID)
    tokens = _correct_tokens(bank)
    extra = GridToken(id="X1", c="9", x=0, y=0, tags=[])
    result = client.evaluate(created.session_id, created.ref_id, tokens + [extra])
    extras = [f for f in result.feedback if f.message_type == "UnexpectedSymbol"]
    assert extras, "an extra token must be reported as feedback"
    assert extras[0].symbol_id_list == ["X1"]


# --- hint placement ---------------------------------------------------------


def test_hint_targets_cell_nearest_answer_line(client: MockGradingClient) -> None:
    """With several missing cells the hint points at one nearest the answer row."""
    created, bank = _derive_bank(client, FIXTURE_ID)
    result = client.evaluate(created.session_id, created.ref_id, _given_only_tokens(bank))
    assert result.hint is not None
    rect = result.hint.target_positions[0]
    # The hint's bottom (grid y) is an answer-row cell, the nearest to answer_y.
    answer_rows = {t.y for t in bank.expected if t.role == "answer"}
    assert rect.bottom in answer_rows


def test_hint_uses_captured_message_vocabulary(client: MockGradingClient) -> None:
    """The subtraction hint uses the captured HorizontalSum messageType."""
    created, bank = _derive_bank(client, FIXTURE_ID)
    result = client.evaluate(created.session_id, created.ref_id, _given_only_tokens(bank))
    assert result.hint is not None
    assert result.hint.message_type == "HorizontalSum"


# --- solution + info edge cases ---------------------------------------------


def test_solution_returns_full_token_tree(client: MockGradingClient) -> None:
    """solution() returns the captured solution fixture for the session."""
    created, bank = _derive_bank(client, FIXTURE_ID)
    solution = client.solution(created.session_id)
    # The mock returns the raw captured solution fixture; it must have 'success'.
    assert solution.get("success") is True


def test_info_before_evaluate_is_unfinished(client: MockGradingClient) -> None:
    spec = ex.get_exercise(FIXTURE_ID)
    created = client.create_session(spec)
    info = client.info(created.session_id)
    assert info.finished is False
    assert info.marks_earned == 0
    assert info.marks_total == 1


def test_evaluate_unknown_session_raises(client: MockGradingClient) -> None:
    """Unknown session id raises GradingError (not bare KeyError)."""
    from src.grading.client import GradingError

    with pytest.raises(GradingError):
        client.evaluate("no-such-session", "ref", [])


def test_result_round_trips_through_contract(client: MockGradingClient) -> None:
    """The emitted EvalResult re-parses byte-aligned, proving the captured shape."""
    created, bank = _derive_bank(client, FIXTURE_ID)
    result = client.evaluate(created.session_id, created.ref_id, _given_only_tokens(bank))
    again = EvalResult.from_api(result.to_api())
    assert again == result


# --- completion-state feedback gate (subtract-256-89 canonical case) --------
# These tests guard the specific symptom: a correct 256-89 solution must reach
# progress==1.0 with no hint and no MISSING element for the result bar or any
# given/scaffold cell. Partial submissions must still produce a hint and a
# non-zero missing count (no over-suppression).

_SUBTRACT_ID = "subtract-256-89"


def _partial_answer_only_tokens(bank: BankExercise) -> list[GridToken]:
    """Only the child-fill answer cells (given scaffold excluded)."""
    return [
        t for t in ex.expected_tokens_as_grid(bank.exercise_id, bank)
        if "FIXED" not in t.tags and bank.expected[
            next(i for i, e in enumerate(bank.expected) if (e.x, e.y) == (t.x, t.y))
        ].role == "answer"
    ]


def test_correct_subtract_no_hint_no_missing_for_bar(client: MockGradingClient) -> None:
    """A correct 256-89 submission reaches progress 1.0 with no hint and no
    MISSING element for the result bar or any given/scaffold cell."""
    created, bank = _derive_bank(client, _SUBTRACT_ID)
    result = client.evaluate(created.session_id, created.ref_id, _correct_tokens(bank))

    assert result.progress == pytest.approx(1.0), "correct solution must reach 1.0"
    assert result.hint is None, "no hint when everything is correct"

    bar_cells = {(t.x, t.y) for t in bank.expected if t.role == "result_bar"}
    missing_positions = {(e.position[0], e.position[1]) for e in result.elements if e.status == "MISSING"}
    assert not bar_cells.intersection(missing_positions), (
        f"result_bar cells must never be MISSING at completion, got: "
        f"{bar_cells.intersection(missing_positions)}"
    )

    given_cells = {(t.x, t.y) for t in bank.expected if t.given}
    assert not given_cells.intersection(missing_positions), (
        f"given scaffold cells must never be MISSING at completion, got: "
        f"{given_cells.intersection(missing_positions)}"
    )


def test_correct_subtract_earns_marks_finished(client: MockGradingClient) -> None:
    """A correct 256-89 submission sets marks.finished True and earns 1 mark."""
    created, bank = _derive_bank(client, _SUBTRACT_ID)
    client.evaluate(created.session_id, created.ref_id, _correct_tokens(bank))
    info = client.info(created.session_id)
    assert info.finished is True
    assert info.marks_earned == info.marks_total == 1


def test_correct_subtract_drift_variant_no_hint(client: MockGradingClient) -> None:
    """A correct 256-89 submission with y-drift (all rows shifted +1) still reaches
    progress 1.0 and emits no hint (structure-relative matching absorbs drift)."""
    from dataclasses import replace as dc_replace
    created, bank = _derive_bank(client, _SUBTRACT_ID)
    tokens = _correct_tokens(bank)
    # Drift the answer/operand rows only. Borrows sit in fixed guide boxes and
    # are graded by exact cell, so they stay put (drifting one onto another
    # borrow cell would be a genuinely misplaced borrow, not handwriting drift).
    drifted = [
        dc_replace(t, y=t.y + 1) if (t.c not in ("", "_") and t.type != "REPLACE") else t
        for t in tokens
    ]
    result = client.evaluate(created.session_id, created.ref_id, drifted)
    assert result.progress == pytest.approx(1.0), "drifted correct tokens must still grade 1.0"
    assert result.hint is None, "no hint when correct (even with y-drift)"


def test_partial_subtract_still_emits_hint_and_missing_count(client: MockGradingClient) -> None:
    """Submitting only the given scaffold for 256-89 (no answer) must
    still produce a hint and report MISSING child-fill cells (no over-suppression)."""
    created, bank = _derive_bank(client, _SUBTRACT_ID)
    partial = _given_only_tokens(bank)
    result = client.evaluate(created.session_id, created.ref_id, partial)
    assert result.progress < 1.0, "partial submission must not reach 1.0"
    assert result.hint is not None, "partial submission must still produce a hint"
    missing = [e for e in result.elements if e.status == "MISSING"]
    assert len(missing) >= 1, "partial submission must report MISSING child-fill elements"
    bar_cells = {(t.x, t.y) for t in bank.expected if t.role == "result_bar"}
    missing_positions = {(e.position[0], e.position[1]) for e in missing}
    assert not bar_cells.intersection(missing_positions), (
        f"result_bar cells must not be MISSING even on a partial submission: "
        f"{bar_cells.intersection(missing_positions)}"
    )
    assert result.hint.target_positions, "hint must have a target rect"
    hint_rect = result.hint.target_positions[0]
    bar_ys = {t.y for t in bank.expected if t.role == "result_bar"}
    assert hint_rect.bottom not in bar_ys, (
        f"hint must not target the result bar row (y in {bar_ys}), got y={hint_rect.bottom}"
    )


# --- product-38-29 specific gates -------------------------------------------


def test_product_correct_submission_grades_full_progress(client: MockGradingClient) -> None:
    """A complete correct 38x29 submission reaches progress 1.0 with no errors."""
    created, bank = _derive_bank(client, "product-38-29")
    result = client.evaluate(created.session_id, created.ref_id, _correct_tokens(bank))
    assert result.progress == pytest.approx(1.0)
    assert result.feedback == []
    assert result.hint is None


def test_product_wrong_answer_digit_yields_error(client: MockGradingClient) -> None:
    """A wrong digit in the product answer (1102 -> 1103) must produce ERROR feedback."""
    created, bank = _derive_bank(client, "product-38-29")
    tokens = _correct_tokens(bank)
    # Corrupt the units answer digit: '2' at (3,0) -> '3'
    corrupted = [
        replace(t, c="3") if (t.x == 3 and t.y == 0 and t.c == "2") else t
        for t in tokens
    ]
    result = client.evaluate(created.session_id, created.ref_id, corrupted)
    errors = [e for e in result.elements if e.status == "ERROR"]
    assert errors, "wrong answer digit must produce an ERROR element"
    assert result.progress < 1.0


def test_product_missing_partial_product_yields_missing_step(client: MockGradingClient) -> None:
    """Omitting one partial-product row while answer is correct yields 'missing_step'."""
    created, bank = _derive_bank(client, "product-38-29")
    # Remove partial product at y=3 (342 row).
    tokens_no_partial = [
        t for t in _correct_tokens(bank)
        if t.y != 3
    ]
    result = client.evaluate(created.session_id, created.ref_id, tokens_no_partial)
    missing = [e for e in result.elements if e.status == "MISSING"]
    assert missing, "missing partial product must report MISSING elements"
    # Answer cells at y=0 should still be gradeable.
    assert result.progress < 1.0


# ---------------------------------------------------------------------------
# compute_feedback_summary: verdict + focus classification
# ---------------------------------------------------------------------------
# Tested against subtract-256-89.
# Post 2026-06-19 coordinate fix: no borrow cells in the bank.  engine returns
# Placeholder ERRORs for borrow tokens, so they are excluded.  The only
# child-fill cells are the answer row (167 at y=3, x=10..12).

from src.grading.mock_client import FeedbackSummary, compute_feedback_summary  # noqa: E402


def _eval_correct_subtract(client: MockGradingClient) -> tuple:
    """Return (bank, result) for a fully-correct 256-89 submission."""
    created, bank = _derive_bank(client, _SUBTRACT_ID)
    result = client.evaluate(created.session_id, created.ref_id, _correct_tokens(bank))
    return bank, result


def _eval_wrong_answer_digit_subtract(client: MockGradingClient) -> tuple:
    """Return (bank, result) for a submission where the units answer digit
    is wrong (7 -> 9).  Units answer is at x=12, y=3 in the native solution frame."""
    from dataclasses import replace as dc_replace
    created, bank = _derive_bank(client, _SUBTRACT_ID)
    tokens = _correct_tokens(bank)
    corrupted = [
        dc_replace(t, c="9") if (t.x == 12 and t.y == 3) else t
        for t in tokens
    ]
    result = client.evaluate(created.session_id, created.ref_id, corrupted)
    return bank, result


def _eval_incomplete_no_answer_subtract(client: MockGradingClient) -> tuple:
    """Return (bank, result) for given/scaffold only (no answer drawn)."""
    created, bank = _derive_bank(client, _SUBTRACT_ID)
    result = client.evaluate(created.session_id, created.ref_id, _given_only_tokens(bank))
    return bank, result


class TestComputeFeedbackSummary:
    """Unit tests for compute_feedback_summary: one verdict per scenario."""

    def test_correct_submission_yields_correct_outcome(self, client: MockGradingClient) -> None:
        """A fully-correct submission must yield outcome 'correct' with empty focus."""
        entry, result = _eval_correct_subtract(client)
        summary = compute_feedback_summary(entry, result)
        assert isinstance(summary, FeedbackSummary)
        assert summary.outcome == "correct"
        assert summary.focus_items == (), (
            f"correct outcome must have no focus items, got {summary.focus_items}"
        )

    def test_correct_headline_is_positive(self, client: MockGradingClient) -> None:
        """The correct headline must be a non-empty, child-friendly string."""
        entry, result = _eval_correct_subtract(client)
        summary = compute_feedback_summary(entry, result)
        assert summary.headline, "headline must be a non-empty string"
        assert "–" not in summary.headline and "—" not in summary.headline

    def test_wrong_answer_digit_yields_mistake(self, client: MockGradingClient) -> None:
        """A wrong answer digit (7 written as 9) -> outcome 'mistake'."""
        entry, result = _eval_wrong_answer_digit_subtract(client)
        summary = compute_feedback_summary(entry, result)
        assert summary.outcome == "mistake", (
            f"expected 'mistake' for wrong digit, got {summary.outcome!r}"
        )

    def test_wrong_digit_focus_has_symbol_id(self, client: MockGradingClient) -> None:
        """Mistake focus items must carry a symbol_id (the drawn glyph's token id)."""
        entry, result = _eval_wrong_answer_digit_subtract(client)
        summary = compute_feedback_summary(entry, result)
        assert summary.focus_items, "mistake must have at least one focus item"
        for item in summary.focus_items:
            assert item.kind == "error"
            assert item.symbol_id is not None, (
                "error focus item must carry a symbol_id"
            )
            assert item.grid_cell is None, (
                "error focus item must NOT carry a grid_cell"
            )

    def test_incomplete_submission_yields_incomplete(self, client: MockGradingClient) -> None:
        """Submitting only scaffold tokens (no answer) -> 'incomplete'."""
        entry, result = _eval_incomplete_no_answer_subtract(client)
        summary = compute_feedback_summary(entry, result)
        assert summary.outcome == "incomplete", (
            f"expected 'incomplete' when answer cells missing, got {summary.outcome!r}"
        )

    def test_incomplete_focus_is_answer_cells(self, client: MockGradingClient) -> None:
        """Incomplete focus items must be of kind 'missing_answer'."""
        entry, result = _eval_incomplete_no_answer_subtract(client)
        summary = compute_feedback_summary(entry, result)
        assert summary.focus_items, "incomplete must have at least one focus item"
        for item in summary.focus_items:
            assert item.kind == "missing_answer", (
                f"incomplete focus item kind must be 'missing_answer', got {item.kind!r}"
            )
            assert item.symbol_id is None
            assert item.grid_cell is not None

    def test_mistake_takes_priority_over_incomplete(self, client: MockGradingClient) -> None:
        """When a wrong digit AND missing answer cells exist simultaneously, outcome is 'mistake'.

        Submits only the units answer digit (x=15, y=7) but corrupted to '9'.
        The hundreds and tens answer cells (x=13, x=14) are omitted so the
        submission is both wrong AND incomplete.  Mistake takes priority.
        """
        from dataclasses import replace as dc_replace
        created, bank = _derive_bank(client, _SUBTRACT_ID)
        tokens = _correct_tokens(bank)
        # Keep FIXED scaffold and the units answer (corrupted); drop tens+hundreds.
        # Answer row is y=3; units is x=12 in the native solution frame.
        tokens_bad = [
            dc_replace(t, c="9") if (t.x == 12 and t.y == 3) else t
            for t in tokens
            if not (t.y == 3 and t.x != 12)
        ]
        result = client.evaluate(created.session_id, created.ref_id, tokens_bad)
        summary = compute_feedback_summary(bank, result)
        assert summary.outcome == "mistake", (
            f"mistake must take priority over incomplete, got {summary.outcome!r}"
        )


# ---------------------------------------------------------------------------
# Step-leniency gates: borrow/carry misread must never flip verdict to mistake
# ---------------------------------------------------------------------------


class TestStepLeniency:
    """Presence-only grading for step cells (carry, partial).

    The subtract-256-89 exercise no longer has borrow cells (removed 2026-06-19:
    engine returns Placeholder ERRORs for borrow tokens in this exercise config).
    Borrow step-leniency tests have been replaced by a test confirming a correct
    answer-only submission grades as 'correct' in the mock.
    """

    # --- subtract-256-89 (answer-only is sufficient) -----------------------

    def test_subtract_correct_answer_only_is_correct(
        self, client: MockGradingClient
    ) -> None:
        """Correct 256-89 answer digits (no borrows in bank) yield 'correct'."""
        created, bank = _derive_bank(client, "subtract-256-89")
        tokens = _correct_tokens(bank)
        result = client.evaluate(created.session_id, created.ref_id, tokens)
        summary = compute_feedback_summary(bank, result)
        assert result.progress == pytest.approx(1.0)
        assert summary.outcome == "correct"

    def test_subtract_wrong_answer_digit_is_mistake(
        self, client: MockGradingClient
    ) -> None:
        """Wrong answer units digit (167->168) yields 'mistake'.

        Units answer is at x=12, y=3 in the native solution frame.
        """
        from dataclasses import replace as dc_replace
        created, bank = _derive_bank(client, "subtract-256-89")
        tokens = _correct_tokens(bank)
        tokens_wrong = [
            dc_replace(t, c="8") if (t.x == 12 and t.y == 3) else t
            for t in tokens
        ]
        result = client.evaluate(created.session_id, created.ref_id, tokens_wrong)
        summary = compute_feedback_summary(bank, result)
        assert summary.outcome == "mistake"

    # --- product-38-29 (carry + partial product marks) ----------------------

    def test_product_correct_answer_misread_carry_is_correct(
        self, client: MockGradingClient
    ) -> None:
        """Correct 38x29 answer with carry '3' (at y=7) misread as '5' yields 'correct'."""
        from dataclasses import replace as dc_replace
        created, bank = _derive_bank(client, "product-38-29")
        tokens = _correct_tokens(bank)
        tokens_misread = [
            dc_replace(t, c="5") if t.y == 7 else t
            for t in tokens
        ]
        result = client.evaluate(created.session_id, created.ref_id, tokens_misread)
        summary = compute_feedback_summary(bank, result)
        assert result.progress == pytest.approx(1.0), (
            f"Correct 38x29 answer with misread carry must grade progress=1.0, "
            f"got {result.progress:.3f}"
        )
        assert summary.outcome == "correct"

    def test_product_correct_answer_no_carry_is_missing_step(
        self, client: MockGradingClient
    ) -> None:
        """Correct 38x29 answer with NO carry row yields 'missing_step'."""
        created, bank = _derive_bank(client, "product-38-29")
        tokens_no_carry = [t for t in _correct_tokens(bank) if t.y != 7]
        result = client.evaluate(created.session_id, created.ref_id, tokens_no_carry)
        summary = compute_feedback_summary(bank, result)
        assert summary.outcome == "missing_step", (
            f"Correct 38x29 answer with no carry must yield 'missing_step', "
            f"got {summary.outcome!r}"
        )
        assert result.progress < 1.0

    def test_product_wrong_answer_with_correct_carry_is_mistake(
        self, client: MockGradingClient
    ) -> None:
        """Wrong answer digit in 38x29 (1102->1103) with correct carries yields 'mistake'."""
        from dataclasses import replace as dc_replace
        created, bank = _derive_bank(client, "product-38-29")
        tokens = _correct_tokens(bank)
        # Answer '2' at (3,0) -> '3'
        tokens_wrong = [
            dc_replace(t, c="3") if (t.x == 3 and t.y == 0 and t.c == "2") else t
            for t in tokens
        ]
        result = client.evaluate(created.session_id, created.ref_id, tokens_wrong)
        summary = compute_feedback_summary(bank, result)
        assert summary.outcome == "mistake"


# ---------------------------------------------------------------------------
# EvalResult.from_api: null-result robustness
# ---------------------------------------------------------------------------


def test_eval_result_from_api_handles_null_result() -> None:
    """EvalResult.from_api([{result: null}]) must not raise and returns empty result."""
    data = [{"result": None, "updateSession": {}}]
    result = EvalResult.from_api(data)
    assert isinstance(result, EvalResult)
    assert result.elements == []
    assert result.progress == 0.0
    assert result.feedback == []
    assert result.hint is None


def test_eval_result_from_api_handles_empty_list() -> None:
    """EvalResult.from_api([]) must not raise."""
    result = EvalResult.from_api([])
    assert isinstance(result, EvalResult)
    assert result.elements == []


def test_structure_relative_match_empty_expected_returns_triple() -> None:
    """Guard: an empty expected set returns a 3-tuple (matched, extra, global_dy).

    Regression guard for the path where every expected cell is filtered out
    (e.g. an all-borrow expected set). The callers unpack three values, so a
    2-tuple return here would crash with ValueError. The current bank always has
    non-borrow cells, so only this direct test exercises the branch.
    """
    from src.grading.mock_client import _structure_relative_match

    tok = GridToken(id="A", c="7", x=15, y=7)
    result = _structure_relative_match([tok], [])
    assert len(result) == 3, "empty-expected path must return a 3-tuple"
    matched, extra, global_dy = result
    assert matched == {}
    assert [t.id for t in extra] == ["A"]
    assert global_dy == 0
