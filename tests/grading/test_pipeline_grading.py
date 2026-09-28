"""Pipeline grading regression: rendered 256-89 solution grades at progress == 1.0.

This test is the guard the calibration integration tests did NOT provide: it
runs the FULL two-stage pipeline (YOLO -> GNN -> assemble -> tokens_to_grid ->
mock evaluate) on a PIL-rendered image of a correct 256-89 worked solution, and
asserts that every child-supplied cell (borrows + answer) grades OK.

Why this test matters
---------------------
The existing ``test_calibration_integration.py`` tests build GridTokens directly
from the bank's expected (x, y) coordinates, bypassing the recognizer entirely.
They confirm the calibration math is correct but do NOT catch bugs in the
pixel->grid mapping that only manifest when real ink is mapped through
``tokens_to_grid``. This test runs the real recognizer so placement errors
(e.g. partial drawings where the GNN's scene-relative col_cluster_id is offset
from the guide grid's column indices) are caught here even if the calibration
math is fine.

The image is rendered with PIL at the exact pixel cells the calibration assigns
to the exercise's expected token positions. A legible, correctly-placed digit
must be recognized correctly (this is a calibration test not an accuracy test).
The test is skipped when model weights are absent (CI without weights).

Bugs caught by this test
------------------------
* tokens_to_grid using GNN row/grid_col (scene-relative) instead of bbox
  pixel position: causes answer digits to map to wrong engine cells when the child
  only draws a subset of columns, producing UnexpectedSymbol + MISSING and
  progress == 0.
* result_bar glyph sent as char to one bar cell instead of being skipped:
  causes ERROR for the bar cell and MISSING for the other three, producing
  IncorrectValue feedback and visible error popups on correct work.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from src.grading import exercises as ex
from src.grading.derive import derive_exercise
from src.grading.exercises import BankExercise
from src.grading.grid_mapping import build_calibration, tokens_to_grid
from src.grading.mock_client import MockGradingClient
from src.grading.types import SessionCreated
from src.core.run_config import GradingConfig
from src.data_pipeline.preprocessing import preprocess_for_pipeline
from src.demo.app import _centred_calibration_geometry, _CANVAS_PX
from src.inference.run import InferenceSession

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXERCISE_ID = "subtract-256-89"

_FIXTURES_DIR = PROJECT_ROOT / "tests" / "grading" / "fixtures" / "derive"


def _derive_bank_for(client: MockGradingClient, exercise_id: str):
    """Create a session and return (SessionCreated, BankExercise) via derive."""
    spec = ex.get_exercise(exercise_id)
    created = client.create_session(spec)
    bank = client._sessions[created.session_id].exercise
    return created, bank


def _make_mock_client() -> MockGradingClient:
    return MockGradingClient(config=GradingConfig(), project_root=PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_calibration_for_bank(bank: BankExercise):
    """Build the SAME calibration select_route builds for a derived BankExercise.

    Calls ``_centred_calibration_geometry`` exactly as ``select_route`` does so
    this helper guards the real centered server path, not the old uncentered
    default origin.
    """
    engine_x_left, engine_y_top = bank.evaluate_engine_origin
    centred_geom = _centred_calibration_geometry(bank, _CANVAS_PX)
    return build_calibration(
        {"width": bank.grid_width, "height": bank.grid_height},
        {
            **centred_geom,
            "grid_width": bank.grid_width,
            "grid_height": bank.grid_height,
            "engine_x_left": engine_x_left,
            "engine_y_top": engine_y_top,
        },
    )


def render_256_89_solution(calibration) -> np.ndarray:
    """Render a complete correct 256-89 worked solution on a 512x512 white canvas.

    Draws each glyph at the center of the guide cell corresponding to that
    token's expected grading engine (x, y) position. The result bar is drawn as a
    horizontal line spanning the bar cells. Uses a legible fixed-width font so
    YOLO + GNN can recognize every character.

    Returns the image as a (512, 512) uint8 numpy array (white background, black
    ink), ready for ``preprocess_for_pipeline``.
    """
    img = Image.new("L", (512, 512), color=255)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", size=28)
    except Exception:
        font = ImageFont.load_default()

    def cell_center(engine_x: int, engine_y: int):
        left, top, right, bottom = calibration.cell_to_pixels(engine_x, engine_y)
        return (left + right) / 2.0, (top + bottom) / 2.0

    def draw_char(char: str, engine_x: int, engine_y: int):
        cx, cy = cell_center(engine_x, engine_y)
        draw.text((cx, cy), char, fill=0, font=font, anchor="mm")

    def draw_bar(engine_x_start: int, engine_x_end: int, engine_y: int):
        left_px, top_px, _, bottom_px = calibration.cell_to_pixels(engine_x_start, engine_y)
        _, _, right_px, _ = calibration.cell_to_pixels(engine_x_end, engine_y)
        cy = (top_px + bottom_px) / 2.0
        draw.line([(left_px + 4, cy), (right_px - 4, cy)], fill=0, width=3)

    # 256 - 89 = 167.  Native solution frame (grid 13x10).
    # y=0 at bottom, y increases upward.  Bar cells included (drawn as line).
    draw_char("2", 10, 6)   # top operand hundreds
    draw_char("5", 11, 6)   # top operand tens
    draw_char("6", 12, 6)   # top operand units
    draw_char("-",  9, 5)   # operator
    draw_char("8", 11, 5)   # bottom operand tens
    draw_char("9", 12, 5)   # bottom operand units
    draw_bar(9, 12, 4)      # result bar x=9..12
    draw_char("1", 10, 3)   # answer hundreds
    draw_char("6", 11, 3)   # answer tens
    draw_char("7", 12, 3)   # answer units

    return np.array(img)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def inference_session():
    """Load the active model weights once for the whole module."""
    session = InferenceSession()
    session.auto_load(PROJECT_ROOT)
    return session


# ---------------------------------------------------------------------------
# Regression gate
# ---------------------------------------------------------------------------


def test_correct_256_89_solution_grades_full_progress(inference_session: InferenceSession):
    """A clean rendered 256-89 solution grades at progress == 1.0 with no errors.

    This is the primary regression guard for the pixel->grid mapping:
    * tokens_to_grid must use bbox pixel position (not GNN row/grid_col) so
      that even a partial scene (only answer+borrows drawn) maps tokens to the
      correct engine cells.
    * result_bar must be skipped in tokens_to_grid (like div_bracket) so its
      multi-dash display glyph does not produce a char mismatch ERROR against
      the bank's "_" expected char.
    """
    if not inference_session.is_loaded():
        pytest.skip("no model weights available; skipping pipeline grading test")

    client = _make_mock_client()
    created, bank = _derive_bank_for(client, EXERCISE_ID)
    calibration = _build_calibration_for_bank(bank)
    raw_image = render_256_89_solution(calibration)
    gray = preprocess_for_pipeline(raw_image)

    payload = inference_session.predict(gray)
    assembled_tokens = payload.get("tokens", [])

    assert assembled_tokens, (
        f"recognizer returned no tokens for the rendered 256-89 image; "
        f"equation_kind={payload.get('equation_kind')}"
    )

    grid_tokens = tokens_to_grid(assembled_tokens, calibration)
    assert grid_tokens, "tokens_to_grid returned empty list -- all tokens skipped"

    result = client.evaluate(created.session_id, created.ref_id, grid_tokens)

    assert result.progress == pytest.approx(1.0), (
        f"Expected progress=1.0 for correct 256-89 solution, got {result.progress:.3f}.\n"
        f"Non-OK elements: "
        f"{[(e.status, e.position, e.attribute_map) for e in result.elements if e.status != 'OK']}\n"
        f"Feedback: {[(fb.message_type, fb.message_args) for fb in result.feedback]}\n"
        f"Recognized tokens: "
        f"{[(t.get('label'), t.get('bbox'), t.get('row'), t.get('grid_col')) for t in assembled_tokens]}"
    )

    unexpected = [
        fb for fb in result.feedback if fb.message_type == "UnexpectedSymbol"
    ]
    assert not unexpected, (
        f"UnexpectedSymbol feedback on a correct solution: {unexpected}. "
        "Likely cause: tokens_to_grid mapping a token to an engine cell outside "
        "the expected set."
    )

    bar_errors = [
        fb for fb in result.feedback
        if fb.message_type == "IncorrectValue"
        and fb.message_args.get("expected") == "_"
    ]
    assert not bar_errors, (
        f"result_bar char mismatch: {bar_errors}. "
        "result_bar must be skipped in tokens_to_grid (like div_bracket), "
        "not sent as a dash string causing IncorrectValue against expected '_'."
    )


# ---------------------------------------------------------------------------
# WI-1 gate: translation-invariant grading
# ---------------------------------------------------------------------------


def _shift_tokens(tokens: list, dx: int, dy: int) -> list:
    """Return a copy of ``tokens`` with every GridToken shifted by (dx, dy)."""
    from src.grading.types import GridToken
    shifted = []
    for tok in tokens:
        shifted.append(GridToken(
            id=tok.id,
            c=tok.c,
            x=tok.x + dx,
            y=tok.y + dy,
            tags=list(tok.tags),
            type=tok.type,
        ))
    return shifted


def test_shifted_256_89_solution_grades_full_progress_mock_only():
    """WI-1 gate: translation-invariant grading.

    Builds the full expected token set for 256-89, shifts every non-bar
    token by (+2 cols, +1 row), replaces the bar with a SHORT bar spanning
    only 2 cells (not the expected 4), and asserts that the mock still
    grades progress == 1.0 with no ERROR or UnexpectedSymbol feedback.

    This test does NOT run the ML recognizer; it constructs GridToken objects
    directly from the bank so the token-to-cell mapping is exact. The
    translation-invariant search in MockGradingClient.evaluate must find
    (dx=+2, dy=+1) and align the shifted tokens back to the expected frame.
    """
    from src.grading.types import GridToken

    SHIFT_X, SHIFT_Y = 2, 1

    client = _make_mock_client()
    created, bank = _derive_bank_for(client, EXERCISE_ID)

    base_tokens = ex.expected_tokens_as_grid(EXERCISE_ID, bank)
    # Replace result_bar tokens with a SHORT bar spanning only 2 cells at
    # the shifted position. The bar cells in the bank are at x=9..12, y=4 (native).
    # A short bar covers only x=11..12 shifted -> x=13..14, y=5.
    bar_x_start = min(t.x for t in bank.expected if t.role == "result_bar")
    bar_x_end = max(t.x for t in bank.expected if t.role == "result_bar")
    bar_y = next(t.y for t in bank.expected if t.role == "result_bar")
    # Short bar: last 2 cells of the bar, shifted
    short_bar = [
        GridToken(id=f"SB{i}", c="_", x=bar_x_end - 1 + SHIFT_X + i, y=bar_y + SHIFT_Y, tags=["FIXED"])
        for i in range(2)
    ]
    non_bar_tokens = [t for t in base_tokens if t.c != "_"]
    shifted_non_bar = _shift_tokens(non_bar_tokens, SHIFT_X, SHIFT_Y)
    submission = shifted_non_bar + short_bar

    result = client.evaluate(created.session_id, created.ref_id, submission)

    assert result.progress == pytest.approx(1.0), (
        f"Shifted (+{SHIFT_X}, +{SHIFT_Y}) correct 256-89 must grade progress=1.0, "
        f"got {result.progress:.3f}.\n"
        f"Non-OK elements: "
        f"{[(e.status, e.position, e.attribute_map) for e in result.elements if e.status != 'OK']}\n"
        f"Feedback: {[(fb.message_type, fb.message_args) for fb in result.feedback]}"
    )
    error_feedback = [
        fb for fb in result.feedback
        if fb.message_type in ("ERROR", "IncorrectValue", "UnexpectedSymbol")
    ]
    assert not error_feedback, (
        f"Shifted correct solution must produce no error feedback; got: {error_feedback}"
    )


def test_unshifted_256_89_full_bar_still_grades_full_progress():
    """WI-1 regression: the original unshifted full-bar test still grades 1.0.

    The translation-invariant search must not break correct unshifted
    submissions (dx=dy=0 should be selected when the submission is already
    aligned to the expected frame).
    """
    client = _make_mock_client()
    created, bank = _derive_bank_for(client, EXERCISE_ID)
    tokens = ex.expected_tokens_as_grid(EXERCISE_ID, bank)
    result = client.evaluate(created.session_id, created.ref_id, tokens)
    assert result.progress == pytest.approx(1.0), (
        f"Unshifted correct 256-89 must still grade 1.0, got {result.progress:.3f}."
    )
    assert result.feedback == [], f"Unshifted correct submission must have no feedback"


# ---------------------------------------------------------------------------
# WI-1 DRIFT gate: per-row x-drift, correct answer still grades 1.0
# ---------------------------------------------------------------------------


def test_per_row_drift_correct_256_89_grades_full_progress():
    """DRIFT GATE (WI-1): per-row horizontal drift still grades progress == 1.0.

    Simulates the root-cause scenario: operand + borrow block at one x-position,
    answer row shifted ~1 grid cell further right (per-row horizontal drift).
    No single (dx, dy) translation aligns both; the structure-relative grader
    must match each row independently by right-aligned place-value slot.

    Exercise 256 - 89 = 167.  Native solution frame (grid 13x10):
      operand "256" at (10..12, 6)                   -- given problem
      operator "-"  at (9, 5)                        -- given problem
      operand "89"  at (11..12, 5)                   -- given problem
      result bar "_" at (9..12, 4)                   -- given, included as FIXED
      answer "167"  at (10..12, 3)                   -- child must fill (no borrows)

    This test submits a correct solution where the answer row is shifted one
    cell to the right (answer at x=11..13 instead of 10..12).
    """
    from src.grading.types import GridToken

    client = _make_mock_client()
    created, bank = _derive_bank_for(client, EXERCISE_ID)

    base_tokens = ex.expected_tokens_as_grid(EXERCISE_ID, bank)

    ANSWER_Y = bank.answer_y  # 3 for subtract-256-89 in native solution frame
    DRIFT_X = 1

    drifted: list[GridToken] = []
    for tok in base_tokens:
        if tok.y == ANSWER_Y and tok.c != "_":
            drifted.append(GridToken(
                id=tok.id, c=tok.c,
                x=tok.x + DRIFT_X, y=tok.y,
                tags=list(tok.tags),
            ))
        else:
            drifted.append(tok)

    result = client.evaluate(created.session_id, created.ref_id, drifted)

    assert result.progress == pytest.approx(1.0), (
        f"DRIFT GATE: correct 256-89 with answer row drifted {DRIFT_X} cell(s) right "
        f"must grade progress=1.0, got {result.progress:.3f}.\n"
        f"Non-OK elements: "
        f"{[(e.status, e.position, e.attribute_map) for e in result.elements if e.status != 'OK']}\n"
        f"Feedback: {[(fb.message_type, fb.message_args) for fb in result.feedback]}"
    )
    error_feedback = [
        fb for fb in result.feedback
        if fb.message_type in ("ERROR", "IncorrectValue", "UnexpectedSymbol")
    ]
    assert not error_feedback, (
        f"DRIFT GATE: correct solution with per-row drift must produce no error feedback; "
        f"got: {error_feedback}"
    )


def test_wrong_answer_still_produces_error_with_drift():
    """DRIFT GATE -- wrong digits must still flag ERROR even with per-row drift.

    Structure-relative grading must not over-tolerate: a child who writes
    answer 166 instead of the correct 167 must receive an ERROR on the wrong
    digit (the units "6" != expected "7"), even when the answer row drifts
    one cell to the right.
    """
    from src.grading.types import GridToken

    client = _make_mock_client()
    created, bank = _derive_bank_for(client, EXERCISE_ID)

    ANSWER_Y = bank.answer_y  # 3 for subtract-256-89 in native solution frame
    DRIFT_X = 1

    base_tokens = ex.expected_tokens_as_grid(EXERCISE_ID, bank)

    wrong: list[GridToken] = []
    for tok in base_tokens:
        if tok.y == ANSWER_Y and tok.c != "_":
            # Corrupt units digit: "7" -> "6"
            new_c = "6" if tok.c == "7" else tok.c
            wrong.append(GridToken(
                id=tok.id, c=new_c,
                x=tok.x + DRIFT_X, y=tok.y,
                tags=list(tok.tags),
            ))
        else:
            wrong.append(tok)

    result = client.evaluate(created.session_id, created.ref_id, wrong)

    assert result.progress < 1.0, (
        f"Wrong answer 166 (instead of 167) must not grade progress=1.0; "
        f"got {result.progress:.3f}"
    )
    errors = [e for e in result.elements if e.status == "ERROR"]
    assert errors, (
        "Wrong digit in answer row must produce at least one ERROR element"
    )
    incorrect = [fb for fb in result.feedback if fb.message_type == "IncorrectValue"]
    assert incorrect, (
        "Wrong digit must produce IncorrectValue feedback"
    )
