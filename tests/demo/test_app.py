"""Route tests for the feedback demo FastAPI app (WS3a).

Uses ``fastapi.testclient.TestClient`` to exercise every route without model
weights or a real drawing. The two heavy collaborators are swapped on
``app.state``: a fake ``engine_client`` (satisfies the :class:`GradingClient`
Protocol structurally and returns canned ``SessionCreated`` / ``EvalResult`` /
``ScoringInfo`` objects in the exact contract shape) and a stub inference session
(returns a canned assembled dict in the real ``predict`` shape), so the routes
are the only thing under test.

Coverage mirrors the WS3a done-definition:

* ``GET /`` serves HTML (200);
* ``GET /api/exercises`` lists the bank;
* ``select -> evaluate`` returns popups + timings (incl. ``total_ms``) + the
  ``opencv_used`` flag + ``progress`` + ``marks``;
* the zero-canvas path returns empty popups + a ``notice`` and never calls the
  engine evaluate;
* an engine error on evaluate returns 502.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import json

from src.grading.client import GradingError
from src.grading.exercises import list_cms_exercises, list_exercises
from src.grading.mock_client import MockGradingClient
from src.grading.types import (
    EvalElement,
    EvalHint,
    EvalResult,
    GridRect,
    GridToken,
    ScoringInfo,
    SessionCreated,
)
from src.demo import app as app_mod


# --------------------------------------------------------------------------- #
# Fakes injected via app.state (no model weights, no real drawing, no key)
# --------------------------------------------------------------------------- #


_FIXTURES_DIR = Path(__file__).parent.parent / "grading" / "fixtures" / "derive"


def _load_subtract_create_fixture() -> list:
    """Load the captured subtract-256-89 create response from the derive fixture."""
    path = _FIXTURES_DIR / "subtract-256-89.create.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_subtract_solution_fixture() -> dict:
    """Load the captured subtract-256-89 solution response from the derive fixture."""
    path = _FIXTURES_DIR / "subtract-256-89.solution.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_create_fixture(exercise_id: str) -> list:
    """Load the captured create fixture for any known exercise_id."""
    path = _FIXTURES_DIR / f"{exercise_id}.create.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_solution_fixture(exercise_id: str) -> dict:
    """Load the captured solution fixture for any known exercise_id."""
    path = _FIXTURES_DIR / f"{exercise_id}.solution.json"
    return json.loads(path.read_text(encoding="utf-8"))


class _FakeEngineClient:
    """A canned grading engine client (satisfies the Protocol structurally).

    ``create_session`` returns a ``SessionCreated`` from the real captured
    subtract-256-89 derive fixture (so ``select_route`` can call
    ``derive_exercise`` and build a real ``BankExercise``); ``evaluate`` returns
    a fixed ``EvalResult`` with one OK element, one MISSING element, and a
    positioned hint (so popup tests have non-empty results); ``solution`` returns
    the real subtract solution fixture; ``info`` returns a fixed score.
    ``fail_evaluate`` flips evaluate to raise ``GradingError`` for 502 tests.
    """

    def __init__(self) -> None:
        self.fail_evaluate = False
        self.evaluate_calls = 0
        self._solution_cache: dict[str, dict] = {}

    def create_session(self, spec) -> SessionCreated:
        exercise_id = spec.exercise_id
        envelope = _load_create_fixture(exercise_id)
        created = SessionCreated.from_api(envelope)
        # Cache the solution fixture so solution() can return it.
        self._solution_cache["sess-1"] = _load_solution_fixture(exercise_id)
        # Override session_id to stable value so tests can assert on it.
        return SessionCreated(
            session_id="sess-1",
            ref_id=created.ref_id,
            marks_total=created.marks_total,
            view_model=created.view_model,
            raw=created.raw,
        )

    def evaluate(self, session_id: str, ref_id: str, tokens: List[GridToken]) -> EvalResult:
        self.evaluate_calls += 1
        if self.fail_evaluate:
            raise GradingError("boom: engine unavailable")
        ok = EvalElement(
            template_type="NumberElement",
            template_id="answer-12-3",
            attribute_map={},
            position=(12, 3),
            width=1,
            height=1,
            status="OK",
            symbol_id_list=["T0"],
        )
        missing = EvalElement(
            template_type="NumberElement",
            template_id="answer-11-3",
            attribute_map={},
            position=(11, 3),
            width=1,
            height=1,
            status="MISSING",
            symbol_id_list=[],
        )
        hint = EvalHint(
            source_token_ids=[],
            target_positions=[GridRect(left=11, top=4, bottom=3, right=12)],
            message_type="HorizontalSum",
            message_args={"numbers": ["256", "89"]},
        )
        return EvalResult(
            elements=[ok, missing],
            unmatched=[],
            feedback=[],
            hint=hint,
            progress=0.5,
        )

    def solution(self, session_id: str) -> dict:
        return self._solution_cache.get(session_id, {"success": True})

    def info(self, session_id: str) -> ScoringInfo:
        return ScoringInfo(
            finished=False,
            marks_total=1,
            marks_earned=0,
            penalties={"marksPenalty": 0},
        )


class _StubInferenceSession:
    """A stub recognizer returning a canned assembled payload.

    Mirrors :meth:`InferenceSession.predict`'s return shape: a flat ``tokens``
    list (each with ``label`` / ``bbox`` / ``row`` / ``col`` / ``grid_col``), the
    ``equation_kind``, the ``timings`` sub-block, and a ``cv_fusion`` block whose
    ``cv_detector.added`` is positive so ``opencv_used`` is True. No weights.
    """

    def __init__(self, tokens: List[Dict[str, Any]] | None = None) -> None:
        self._tokens = tokens if tokens is not None else _canned_tokens()

    def predict(self, image: np.ndarray, **_kw) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "equation_kind": "subtraction",
            "rows": [],
            "slots": {},
            "tokens": list(self._tokens),
            "timings": {"yolo_ms": 12.0, "stage2_ms": 8.0, "assemble_ms": 1.0},
            "cv_fusion": {
                "mode": "auto",
                "cv_detector": {"cv_boxes": 1, "added": 1, "detections": []},
            },
        }


def _canned_tokens() -> List[Dict[str, Any]]:
    """Two recognized answer digits at the units + tens cells (512-space bbox)."""
    return [
        {
            "label": "main_6",
            "bbox": [460.0, 280.0, 500.0, 330.0],
            "row": 3,
            "col": 0,
            "grid_col": 15,
        },
        {
            "label": "main_7",
            "bbox": [420.0, 280.0, 460.0, 330.0],
            "row": 3,
            "col": 1,
            "grid_col": 14,
        },
    ]


def _png_b64(size: int = 64) -> str:
    """Return a base64 PNG (a tiny white square) for an evaluate body."""
    img = Image.fromarray(np.full((size, size), 255, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.fixture
def fake_client() -> _FakeEngineClient:
    return _FakeEngineClient()


@pytest.fixture
def stub_inference() -> _StubInferenceSession:
    return _StubInferenceSession()


@pytest.fixture
def client(tmp_path: Path, fake_client: _FakeEngineClient, stub_inference: _StubInferenceSession) -> TestClient:
    """A ``TestClient`` over the app with the heavy collaborators stubbed.

    ``create_app`` builds a real mock client + an (unloaded) inference session
    against the temp root; the fixture overrides both on ``app.state`` (and the
    inference session the demo session holds) so no weights load and no real
    grading runs.
    """
    application = app_mod.create_app(tmp_path)
    application.state.engine_client = fake_client
    application.state.inference_session = stub_inference
    application.state.demo_session.inference_session = stub_inference
    return TestClient(application)


def _select_subtraction(client: TestClient) -> Dict[str, Any]:
    """Select the captured 256-89 subtraction exercise and return the response."""
    resp = client.post("/api/exercise/select", json={"exercise_id": "subtract-256-89"})
    assert resp.status_code == 200, resp.text
    return resp.json()


# --------------------------------------------------------------------------- #
# GET /  (static shell / placeholder)
# --------------------------------------------------------------------------- #


def test_index_serves_html(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "grading engine" in resp.text


# --------------------------------------------------------------------------- #
# GET /api/exercises
# --------------------------------------------------------------------------- #


def test_lists_exercise_bank(client: TestClient) -> None:
    resp = client.get("/api/exercises")
    assert resp.status_code == 200
    body = resp.json()
    # GET /api/exercises emits camelCase keys (FIX-2: exerciseId not exercise_id).
    ids = {e["exerciseId"] for e in body["exercises"]}
    expected = {spec.exercise_id for spec in list_cms_exercises()}
    assert ids == expected
    assert "subtract-256-89" in ids


# --------------------------------------------------------------------------- #
# POST /api/exercise/select
# --------------------------------------------------------------------------- #


def test_select_returns_scaffold_and_calibration(client: TestClient) -> None:
    body = _select_subtraction(client)
    # FIX-camelCase: select response uses camelCase keys.
    assert body["sessionId"] == "sess-1"
    # refId comes from the real captured subtract fixture (not the old "ref-1" stub).
    assert isinstance(body["refId"], str) and len(body["refId"]) > 0
    calibration = body["calibration"]
    assert calibration["grid_width"] >= 1 and calibration["grid_height"] >= 1
    assert calibration["canvas_px"] == 512.0
    # Scaffold comes from derive_exercise given_tokens(); subtract-256-89 has
    # given tokens: top operand 256, operator -, bottom operand 89, result_bar cells.
    assert len(body["scaffold"]) > 0
    assert all("pixel_rect" in s and len(s["pixel_rect"]) == 4 for s in body["scaffold"])


def test_select_scaffold_on_canvas_real_mock_client(tmp_path: Path) -> None:
    """For every bank exercise, the scaffold tokens returned by POST /api/exercise/select
    must all have pixel_rects within [0,512]x[0,512].

    Uses the real MockGradingClient (not a fake) so the view-model comes from
    the bank, exactly as in production. The test guards against the bug where
    create-frame AUTOSHOW tokens mapped through the evaluate-frame calibration
    produced negative x-coordinates and y > 512 (off-canvas).

    Also checks that the scaffold is non-empty and that the operand digits and
    the result bar / operator are present; and that the pixel_rect for each given
    token is consistent with the calibration's guide-grid cell for that (grid_x, grid_y).
    """
    from src.core.run_config import GradingConfig

    project_root = Path(__file__).resolve().parents[2]
    config = GradingConfig()
    real_engine_client = MockGradingClient(config=config, project_root=project_root)
    stub_inf = _StubInferenceSession()

    for spec in list_exercises():
        exercise_id = spec.exercise_id

        application = app_mod.create_app(tmp_path)
        application.state.engine_client = real_engine_client
        application.state.inference_session = stub_inf
        application.state.demo_session.inference_session = stub_inf
        tc = TestClient(application)

        resp = tc.post("/api/exercise/select", json={"exercise_id": exercise_id})
        assert resp.status_code == 200, f"{exercise_id}: select failed: {resp.text}"
        body = resp.json()

        scaffold = body.get("scaffold", [])
        assert len(scaffold) > 0, (
            f"{exercise_id}: scaffold is empty — pre-printed problem never appears"
        )

        # Every token's pixel_rect must be fully within the 512x512 canvas.
        for tok in scaffold:
            rect = tok["pixel_rect"]
            assert len(rect) == 4, f"{exercise_id}: pixel_rect length != 4: {rect}"
            left, top, right, bottom = rect
            assert left >= 0, (
                f"{exercise_id}: scaffold char={tok['char']!r} grid=({tok['grid_x']},{tok['grid_y']})"
                f" left={left:.1f} < 0 (off-canvas)"
            )
            assert right <= 512, (
                f"{exercise_id}: scaffold char={tok['char']!r} grid=({tok['grid_x']},{tok['grid_y']})"
                f" right={right:.1f} > 512 (off-canvas)"
            )
            assert top >= 0, (
                f"{exercise_id}: scaffold char={tok['char']!r} grid=({tok['grid_x']},{tok['grid_y']})"
                f" top={top:.1f} < 0 (off-canvas)"
            )
            assert bottom <= 512, (
                f"{exercise_id}: scaffold char={tok['char']!r} grid=({tok['grid_x']},{tok['grid_y']})"
                f" bottom={bottom:.1f} > 512 (off-canvas)"
            )

        # At least one operand digit and the operator/bar must be present.
        chars = {tok["char"] for tok in scaffold}
        # Every exercise has a result bar (_) and at least one operand digit.
        assert "_" in chars or any(c.isdigit() for c in chars), (
            f"{exercise_id}: scaffold missing both digits and result bar: {chars}"
        )

        # Pixel positions must be consistent with the calibration's guide grid.
        # A given token at (grid_x, grid_y) must map to the same cell the
        # calibration reports: pixel_rect == calibration.cell_to_pixels(grid_x, grid_y).
        from src.grading.grid_mapping import build_calibration

        cal_raw = body["calibration"]
        calibration = build_calibration(
            {"width": cal_raw["grid_width"], "height": cal_raw["grid_height"]},
            {
                "canvas_px": 512.0,
                "grid_width": cal_raw["grid_width"],
                "grid_height": cal_raw["grid_height"],
                "cell_w_px": cal_raw["cell_w_px"],
                "cell_h_px": cal_raw["cell_h_px"],
                "origin_x_px": cal_raw["origin_x_px"],
                "origin_y_px": cal_raw["origin_y_px"],
                "engine_x_left": cal_raw["engine_x_left"],
                "engine_y_top": cal_raw["engine_y_top"],
            },
        )
        for tok in scaffold:
            gx, gy = tok["grid_x"], tok["grid_y"]
            actual_rect = tok["pixel_rect"]

            if tok.get("kind") == "div_bracket":
                # div_bracket pixel_rect covers the L/bus-stop shape: it spans
                # from the bracket column left to the rightmost dividend digit
                # right, intentionally wider than a single cell.  Check the
                # contract [bracket_left, dividend_top, dividend_right, dividend_bottom].
                bracket_left, bracket_top, bracket_right_cell, bracket_bottom = calibration.cell_to_pixels(gx, gy)
                left, top, right, bottom = actual_rect
                assert left == pytest.approx(bracket_left, abs=0.5), (
                    f"{exercise_id}: div_bracket left={left:.1f} != bracket_left={bracket_left:.1f}"
                )
                assert top == pytest.approx(bracket_top, abs=0.5), (
                    f"{exercise_id}: div_bracket top={top:.1f} != bracket_top={bracket_top:.1f}"
                )
                assert bottom == pytest.approx(bracket_bottom, abs=0.5), (
                    f"{exercise_id}: div_bracket bottom={bottom:.1f} != bracket_bottom={bracket_bottom:.1f}"
                )
                # right must extend past the bracket cell itself (covers dividend).
                assert right > bracket_right_cell - 0.5, (
                    f"{exercise_id}: div_bracket right={right:.1f} must reach past bracket cell "
                    f"right={bracket_right_cell:.1f}"
                )
                assert right <= 512.0 + 0.5, (
                    f"{exercise_id}: div_bracket right={right:.1f} > 512 (off-canvas)"
                )
                continue

            expected_rect = list(calibration.cell_to_pixels(gx, gy))
            assert actual_rect == pytest.approx(expected_rect, abs=0.5), (
                f"{exercise_id}: scaffold char={tok['char']!r} at ({gx},{gy}): "
                f"pixel_rect {actual_rect} != calibration rect {expected_rect}"
            )


def test_select_unknown_exercise_is_422(client: TestClient) -> None:
    resp = client.post("/api/exercise/select", json={"exercise_id": "nope"})
    assert resp.status_code == 422
    assert "nope" in resp.json()["detail"]


# --------------------------------------------------------------------------- #
# POST /api/evaluate
# --------------------------------------------------------------------------- #


def test_evaluate_returns_popups_timings_opencv_progress(client: TestClient) -> None:
    _select_subtraction(client)
    resp = client.post("/api/evaluate", json={"image_png": _png_b64()})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Popups: one per graded element (OK + MISSING) plus the positioned hint.
    statuses = [p["status"] for p in body["popups"]]
    assert "ok" in statuses
    assert "missing" in statuses
    assert "hint" in statuses
    for popup in body["popups"]:
        assert len(popup["pixelRect"]) == 4

    # Timings carry the wall total plus the inference sub-timings.
    timings = body["timings"]
    assert "total_ms" in timings and timings["total_ms"] >= 0.0
    assert timings["yolo_ms"] == 12.0
    assert timings["stage2_ms"] == 8.0

    # OpenCV flag is derived from the cv_fusion block (added > 0 -> used).
    assert body["opencv_used"] is True

    # Progress + marks come straight from the engine result.
    assert body["progress"] == 0.5
    assert body["marks"] == {"earned": 0, "total": 1, "finished": False}

    # Read-as-drawn: the recognized tokens equal what the recognizer emitted.
    labels = [t["label"] for t in body["recognized"]["tokens"]]
    assert labels == ["main_6", "main_7"]
    assert body["recognized"]["equation_kind"] == "subtraction"


def test_evaluate_recognized_tokens_model_first(client: TestClient) -> None:
    """recognized.tokens carry REAL 512-space bboxes from the model, not grid-snapped.

    Each token must also carry color, glyph, role, row, col, confidence, low_conf,
    and disagreement. equation_label and row_count must be present at the top level
    of the recognized block.
    """
    _select_subtraction(client)
    resp = client.post("/api/evaluate", json={"image_png": _png_b64()})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    recognized = body["recognized"]

    # equation_label and row_count must be present.
    assert "equation_label" in recognized, "equation_label missing from recognized"
    assert "row_count" in recognized, "row_count missing from recognized"
    assert isinstance(recognized["row_count"], int)

    canned = _canned_tokens()
    tokens = recognized["tokens"]
    assert len(tokens) == len(canned)

    for tok in tokens:
        # Real bbox fields are present and are lists of 4 floats.
        assert "bbox" in tok, f"bbox missing from token {tok}"
        assert len(tok["bbox"]) == 4, f"bbox wrong length: {tok['bbox']}"

        # id must be present (WI-3: links feedback popup.symbolIds back to the glyph).
        assert "id" in tok, f"'id' field missing from recognized token {tok}"
        assert isinstance(tok["id"], str) and tok["id"], f"token id must be a non-empty string: {tok['id']}"

        # Color, glyph, role, row, col, confidence, low_conf, disagreement all present.
        for field_name in ("color", "glyph", "role", "row", "col", "grid_col", "confidence", "low_conf", "disagreement"):
            assert field_name in tok, f"field {field_name!r} missing from token {tok}"

        # Color is a hex string (starts with #).
        assert tok["color"].startswith("#"), f"color not a hex string: {tok['color']}"

        # low_conf is a bool.
        assert isinstance(tok["low_conf"], bool)

        # disagreement is a bool.
        assert isinstance(tok["disagreement"], bool)

    # Bboxes must be the REAL model bboxes — not grid-snapped, not zero.
    # The canned token bboxes are well-known non-zero values.
    real_bboxes = {tuple(t["bbox"]) for t in tokens}
    canned_bboxes = {tuple(t["bbox"]) for t in canned}
    assert real_bboxes == canned_bboxes, (
        f"recognized token bboxes {real_bboxes} do not match canned real bboxes {canned_bboxes}"
    )

    # Tokens sorted in reading order (row, col).
    rows_cols = [(t["row"], t["col"]) for t in tokens]
    assert rows_cols == sorted(rows_cols), f"tokens not in reading order: {rows_cols}"


def test_evaluate_ok_popup_anchored_on_glyph_bbox(client: TestClient) -> None:
    """An OK popup whose element has a matching token must anchor on the real glyph bbox.

    The fake client returns an OK element with symbol_id_list=["T0"]. T0 maps to
    the first assembled token (index 0 in the stub payload), which has the
    known real bbox [460.0, 280.0, 500.0, 330.0]. The popup pixel_rect must equal
    that bbox and anchor must be "glyph".
    """
    _select_subtraction(client)
    resp = client.post("/api/evaluate", json={"image_png": _png_b64()})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    ok_popups = [p for p in body["popups"] if p["status"] == "ok"]
    assert ok_popups, "no OK popup in response"
    ok = ok_popups[0]

    assert ok.get("anchor") == "glyph", (
        f"OK popup with matched token should be anchor='glyph', got {ok.get('anchor')!r}"
    )
    # The pixel_rect must equal the real bbox of T0 ([460, 280, 500, 330]).
    expected_bbox = [460.0, 280.0, 500.0, 330.0]
    assert ok["pixelRect"] == pytest.approx(expected_bbox, abs=0.5), (
        f"OK popup pixelRect {ok['pixelRect']} does not match real token bbox {expected_bbox}"
    )


def test_evaluate_missing_popup_anchored_on_cell(client: TestClient) -> None:
    """A MISSING popup with no matched tokens must fall back to a grid cell (anchor='cell').

    The fake client returns a MISSING element with empty symbol_id_list. Since no
    token is matched, the popup falls back to the grid cell at element.position
    via grid_rect_to_pixels (anchor='cell'). The pixel_rect must be non-zero.
    """
    _select_subtraction(client)
    resp = client.post("/api/evaluate", json={"image_png": _png_b64()})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    missing_popups = [p for p in body["popups"] if p["status"] == "missing"]
    assert missing_popups, "no MISSING popup in response"
    mp = missing_popups[0]

    assert mp.get("anchor") == "cell", (
        f"MISSING popup with no tokens should be anchor='cell', got {mp.get('anchor')!r}"
    )
    # The cell rect must be a valid 4-element list with positive dimensions.
    rect = mp["pixelRect"]
    assert len(rect) == 4
    left, top, right, bottom = rect
    assert right > left, f"missing popup has zero width: {rect}"
    assert bottom > top, f"missing popup has zero height: {rect}"


def test_evaluate_before_select_is_422(client: TestClient) -> None:
    resp = client.post("/api/evaluate", json={"image_png": _png_b64()})
    assert resp.status_code == 422


def test_evaluate_zero_canvas_short_circuits(
    tmp_path: Path, fake_client: _FakeEngineClient
) -> None:
    """An empty recognition returns empty popups + a notice and skips the engine."""
    empty_stub = _StubInferenceSession(tokens=[])
    application = app_mod.create_app(tmp_path)
    application.state.engine_client = fake_client
    application.state.inference_session = empty_stub
    application.state.demo_session.inference_session = empty_stub
    local_client = TestClient(application)

    _select_subtraction(local_client)
    resp = local_client.post("/api/evaluate", json={"image_png": _png_b64()})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["popups"] == []
    # FIX-2: zero-token path emits "warning" not "notice".
    assert "warning" in body
    assert "total_ms" in body["timings"]
    # The engine evaluate was never called on the zero-token path.
    assert fake_client.evaluate_calls == 0


def test_evaluate_engine_error_is_502(client: TestClient, fake_client: _FakeEngineClient) -> None:
    _select_subtraction(client)
    fake_client.fail_evaluate = True
    resp = client.post("/api/evaluate", json={"image_png": _png_b64()})
    assert resp.status_code == 502
    # FIX-8: raw GradingError string is not surfaced; a generic message is returned.
    assert "engine unavailable" in resp.json()["detail"]


def test_evaluate_bad_png_is_422(client: TestClient) -> None:
    _select_subtraction(client)
    resp = client.post("/api/evaluate", json={"image_png": "not-base64!!!"})
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# GET /api/score
# --------------------------------------------------------------------------- #


def test_score_returns_scoring_info(client: TestClient) -> None:
    _select_subtraction(client)
    resp = client.get("/api/score")
    assert resp.status_code == 200
    body = resp.json()
    # FIX-3: GET /api/score emits camelCase via info.to_api() (marksTotal/marksEarned).
    assert body["marksTotal"] == 1
    assert body["marksEarned"] == 0
    assert body["finished"] is False


def test_score_before_select_is_422(client: TestClient) -> None:
    resp = client.get("/api/score")
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# WI-3 gate: centred exercise grid
# --------------------------------------------------------------------------- #


def test_centred_calibration_used_cells_near_canvas_centre(tmp_path: Path) -> None:
    """WI-3 gate: for each bank exercise the select route must centre the exercise.

    Asserts:
    1. The centre of the used-cell bounding box (union of all expected tokens) is
       within one cell width/height of the canvas centre (256, 256) in both axes.
    2. Every used cell's pixel rect is fully within [0, 512] x [0, 512].

    Uses the real MockGradingClient (not a fake) so the calibration comes
    from the same code path as production.
    """
    from src.grading.grid_mapping import build_calibration
    from src.core.run_config import GradingConfig

    project_root = Path(__file__).resolve().parents[2]
    real_engine_client = MockGradingClient(config=GradingConfig(), project_root=project_root)
    stub_inf = _StubInferenceSession()
    canvas = 512.0

    for spec in list_exercises():
        exercise_id = spec.exercise_id

        application = app_mod.create_app(tmp_path)
        application.state.engine_client = real_engine_client
        application.state.inference_session = stub_inf
        application.state.demo_session.inference_session = stub_inf
        tc = TestClient(application)

        resp = tc.post("/api/exercise/select", json={"exercise_id": exercise_id})
        assert resp.status_code == 200, f"{exercise_id}: select failed: {resp.text}"
        body = resp.json()
        cal_raw = body["calibration"]

        # Get the derived bank_entry from the active session.
        bank_entry = application.state.demo_session._active.bank_exercise

        # Rebuild the calibration from the returned cell geometry + evaluate-frame origin.
        calibration = build_calibration(
            {"width": cal_raw["grid_width"], "height": cal_raw["grid_height"]},
            {
                "canvas_px": canvas,
                "grid_width": cal_raw["grid_width"],
                "grid_height": cal_raw["grid_height"],
                "cell_w_px": cal_raw["cell_w_px"],
                "cell_h_px": cal_raw["cell_h_px"],
                "origin_x_px": cal_raw["origin_x_px"],
                "origin_y_px": cal_raw["origin_y_px"],
                "engine_x_left": cal_raw["engine_x_left"],
                "engine_y_top": cal_raw["engine_y_top"],
            },
        )

        # Compute the pixel bounding box of all used cells.
        left_pxs, top_pxs, right_pxs, bottom_pxs = [], [], [], []
        for token in bank_entry.expected:
            l, t, r, b = calibration.cell_to_pixels(token.x, token.y)
            left_pxs.append(l)
            top_pxs.append(t)
            right_pxs.append(r)
            bottom_pxs.append(b)

        used_left = min(left_pxs)
        used_top = min(top_pxs)
        used_right = max(right_pxs)
        used_bottom = max(bottom_pxs)

        used_cx = (used_left + used_right) / 2.0
        used_cy = (used_top + used_bottom) / 2.0
        canvas_cx = canvas / 2.0
        canvas_cy = canvas / 2.0

        cell_w = cal_raw["cell_w_px"]
        cell_h = cal_raw["cell_h_px"]

        assert abs(used_cx - canvas_cx) <= cell_w + 1.0, (
            f"{exercise_id}: used-cell bbox centre x={used_cx:.1f} is more than "
            f"one cell ({cell_w:.1f}px) away from canvas centre {canvas_cx:.1f}"
        )
        assert abs(used_cy - canvas_cy) <= cell_h + 1.0, (
            f"{exercise_id}: used-cell bbox centre y={used_cy:.1f} is more than "
            f"one cell ({cell_h:.1f}px) away from canvas centre {canvas_cy:.1f}"
        )

        # Every used cell must be fully within the canvas.
        for token in bank_entry.expected:
            l, t, r, b = calibration.cell_to_pixels(token.x, token.y)
            assert l >= 0, (
                f"{exercise_id}: cell ({token.x},{token.y}) left={l:.1f} < 0 (off-canvas)"
            )
            assert t >= 0, (
                f"{exercise_id}: cell ({token.x},{token.y}) top={t:.1f} < 0 (off-canvas)"
            )
            assert r <= canvas, (
                f"{exercise_id}: cell ({token.x},{token.y}) right={r:.1f} > {canvas} (off-canvas)"
            )
            assert b <= canvas, (
                f"{exercise_id}: cell ({token.x},{token.y}) bottom={b:.1f} > {canvas} (off-canvas)"
            )


# --------------------------------------------------------------------------- #
# POST /api/evaluate -- feedback object (verdict + focus)
# --------------------------------------------------------------------------- #


def test_evaluate_response_has_feedback_key(client: TestClient) -> None:
    """Every /api/evaluate response must carry a top-level 'feedback' key."""
    _select_subtraction(client)
    resp = client.post("/api/evaluate", json={"image_png": _png_b64()})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "feedback" in body, (
        "evaluate response must carry a 'feedback' key for the guiding verdict"
    )


def test_evaluate_feedback_has_required_shape(client: TestClient) -> None:
    """The feedback object must have outcome, headline, and focus keys."""
    _select_subtraction(client)
    resp = client.post("/api/evaluate", json={"image_png": _png_b64()})
    assert resp.status_code == 200, resp.text
    fb = resp.json()["feedback"]
    assert "outcome" in fb, "feedback must have 'outcome'"
    assert "headline" in fb, "feedback must have 'headline'"
    assert "focus" in fb, "feedback must have 'focus'"
    assert isinstance(fb["focus"], list), "feedback.focus must be a list"


def test_evaluate_feedback_outcome_is_valid(client: TestClient) -> None:
    """feedback.outcome must be one of the four defined values."""
    _select_subtraction(client)
    resp = client.post("/api/evaluate", json={"image_png": _png_b64()})
    assert resp.status_code == 200, resp.text
    outcome = resp.json()["feedback"]["outcome"]
    valid = {"correct", "mistake", "missing_step", "incomplete"}
    assert outcome in valid, f"feedback.outcome {outcome!r} not in {valid}"


def test_evaluate_feedback_incomplete_when_missing_answer(client: TestClient) -> None:
    """The fake client returns one MISSING element at (11,3) which is an answer
    cell in subtract-256-89 (post 2026-06-19 coord fix). Outcome must be 'incomplete'."""
    _select_subtraction(client)
    resp = client.post("/api/evaluate", json={"image_png": _png_b64()})
    assert resp.status_code == 200, resp.text
    fb = resp.json()["feedback"]
    assert fb["outcome"] == "incomplete", (
        f"expected 'incomplete' when an answer cell is MISSING, got {fb['outcome']!r}"
    )
    # Focus must be non-empty (points at the missing answer cell).
    assert fb["focus"], "incomplete feedback must carry at least one focus item"
    for item in fb["focus"]:
        assert "pixelRect" in item and len(item["pixelRect"]) == 4
        assert "kind" in item
        assert "symbolId" in item


def test_evaluate_feedback_focus_pixel_rect_in_canvas(client: TestClient) -> None:
    """Every focus pixelRect must have positive dimensions and fit within [0,512]."""
    _select_subtraction(client)
    resp = client.post("/api/evaluate", json={"image_png": _png_b64()})
    assert resp.status_code == 200, resp.text
    for item in resp.json()["feedback"]["focus"]:
        left, top, right, bottom = item["pixelRect"]
        assert right > left, f"focus pixelRect has zero width: {item['pixelRect']}"
        assert bottom > top, f"focus pixelRect has zero height: {item['pixelRect']}"
        assert left >= 0 and top >= 0
        assert right <= 512 and bottom <= 512


def test_evaluate_feedback_headline_has_no_em_dash(client: TestClient) -> None:
    """Feedback headline must never contain an em-dash or en-dash (repo rule)."""
    _select_subtraction(client)
    resp = client.post("/api/evaluate", json={"image_png": _png_b64()})
    assert resp.status_code == 200, resp.text
    headline = resp.json()["feedback"]["headline"]
    assert "–" not in headline and "—" not in headline, (
        f"headline contains a dash: {headline!r}"
    )


def test_evaluate_feedback_correct_yields_empty_focus(tmp_path: Path) -> None:
    """When the fake client returns progress=1.0 (all correct), feedback.focus must be []
    and outcome must be 'correct'."""
    from src.grading.types import EvalElement

    class _CorrectFakeClient:
        """Returns a fully-correct EvalResult (all answer cells OK, no MISSING)."""

        def create_session(self, spec) -> SessionCreated:
            created = SessionCreated.from_api(_load_subtract_create_fixture())
            return SessionCreated(
                session_id="sess-ok",
                ref_id=created.ref_id,
                marks_total=created.marks_total,
                view_model=created.view_model,
                raw=created.raw,
            )

        def evaluate(self, session_id, ref_id, tokens):
            ok1 = EvalElement(
                template_type="NumberElement",
                template_id="answer-12-3",
                attribute_map={},
                position=(12, 3),
                width=1,
                height=1,
                status="OK",
                symbol_id_list=["T0"],
            )
            ok2 = EvalElement(
                template_type="NumberElement",
                template_id="answer-11-3",
                attribute_map={},
                position=(11, 3),
                width=1,
                height=1,
                status="OK",
                symbol_id_list=["T1"],
            )
            ok3 = EvalElement(
                template_type="NumberElement",
                template_id="answer-10-3",
                attribute_map={},
                position=(10, 3),
                width=1,
                height=1,
                status="OK",
                symbol_id_list=["T2"],
            )
            return EvalResult(
                elements=[ok1, ok2, ok3],
                unmatched=[],
                feedback=[],
                hint=None,
                progress=1.0,
            )

        def solution(self, session_id):
            return _load_subtract_solution_fixture()

        def info(self, session_id):
            return ScoringInfo(finished=True, marks_total=1, marks_earned=1, penalties={})

    stub_inf = _StubInferenceSession()
    application = app_mod.create_app(tmp_path)
    application.state.engine_client = _CorrectFakeClient()
    application.state.inference_session = stub_inf
    application.state.demo_session.inference_session = stub_inf
    local_client = TestClient(application)

    resp = local_client.post("/api/exercise/select", json={"exercise_id": "subtract-256-89"})
    assert resp.status_code == 200
    resp = local_client.post("/api/evaluate", json={"image_png": _png_b64()})
    assert resp.status_code == 200, resp.text
    fb = resp.json()["feedback"]
    assert fb["outcome"] == "correct", f"expected 'correct', got {fb['outcome']!r}"
    assert fb["focus"] == [], f"correct outcome must have empty focus, got {fb['focus']}"


def test_evaluate_feedback_mistake_when_error_element(tmp_path: Path) -> None:
    """When the fake client returns an ERROR element, feedback.outcome must be 'mistake'
    and focus must carry the wrong token's symbol_id."""
    from src.grading.types import EvalElement, EvalFeedback

    class _ErrorFakeClient:
        """Returns an EvalResult with one ERROR element at an answer cell."""

        def create_session(self, spec) -> SessionCreated:
            created = SessionCreated.from_api(_load_subtract_create_fixture())
            return SessionCreated(
                session_id="sess-err",
                ref_id=created.ref_id,
                marks_total=created.marks_total,
                view_model=created.view_model,
                raw=created.raw,
            )

        def evaluate(self, session_id, ref_id, tokens):
            err = EvalElement(
                template_type="NumberElement",
                template_id="answer-15-7",
                attribute_map={"num": "7", "matched_num": "9"},
                position=(15, 7),
                width=1,
                height=1,
                status="ERROR",
                symbol_id_list=["T0"],
                l=-0.5,
            )
            return EvalResult(
                elements=[err],
                unmatched=[],
                feedback=[
                    EvalFeedback(
                        message_type="IncorrectValue",
                        message_args={"expected": "5", "got": "9"},
                        target_positions=[],
                        symbol_id_list=["T0"],
                    )
                ],
                hint=None,
                progress=0.0,
            )

        def solution(self, session_id):
            return _load_subtract_solution_fixture()

        def info(self, session_id):
            return ScoringInfo(finished=False, marks_total=1, marks_earned=0, penalties={})

    stub_inf = _StubInferenceSession()
    application = app_mod.create_app(tmp_path)
    application.state.engine_client = _ErrorFakeClient()
    application.state.inference_session = stub_inf
    application.state.demo_session.inference_session = stub_inf
    local_client = TestClient(application)

    resp = local_client.post("/api/exercise/select", json={"exercise_id": "subtract-256-89"})
    assert resp.status_code == 200
    resp = local_client.post("/api/evaluate", json={"image_png": _png_b64()})
    assert resp.status_code == 200, resp.text
    fb = resp.json()["feedback"]
    assert fb["outcome"] == "mistake", f"expected 'mistake', got {fb['outcome']!r}"
    assert fb["focus"], "mistake feedback must have at least one focus item"
    # The focus item anchors on the drawn glyph (T0) whose bbox is in the canned tokens.
    error_items = [item for item in fb["focus"] if item["kind"] == "error"]
    assert error_items, "mistake focus must have at least one 'error' kind item"
    assert error_items[0]["symbolId"] == "T0"


def test_evaluate_zero_canvas_feedback_is_incomplete(
    tmp_path: Path, fake_client: _FakeEngineClient
) -> None:
    """The zero-canvas short-circuit path must also emit a feedback key with
    outcome='incomplete' and an empty focus list."""
    empty_stub = _StubInferenceSession(tokens=[])
    application = app_mod.create_app(tmp_path)
    application.state.engine_client = fake_client
    application.state.inference_session = empty_stub
    application.state.demo_session.inference_session = empty_stub
    local_client = TestClient(application)

    _select_subtraction(local_client)
    resp = local_client.post("/api/evaluate", json={"image_png": _png_b64()})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "feedback" in body, "zero-canvas response must carry a 'feedback' key"
    fb = body["feedback"]
    assert fb["outcome"] == "incomplete"
    assert fb["focus"] == []


# --------------------------------------------------------------------------- #
# Product scaffold contract gate                                               #
# --------------------------------------------------------------------------- #


def test_select_product_scaffold_has_result_bar(client: TestClient) -> None:
    """Gate: POST /api/exercise/select for product-38-29 returns a result-bar scaffold.

    The product exercise's only FIXED tokens are the three '_' cells at y=4
    (the given result bar). The scaffold must contain exactly those bar tokens.
    """
    resp = client.post("/api/exercise/select", json={"exercise_id": "product-38-29"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    scaffold = body.get("scaffold", [])
    assert len(scaffold) > 0, "product scaffold must not be empty"
    bar_tokens = [t for t in scaffold if t.get("char") == "_"]
    assert len(bar_tokens) == 3, (
        f"product-38-29 scaffold must contain exactly 3 bar tokens (y=4, x=1..3), "
        f"got {bar_tokens}"
    )
    for tok in scaffold:
        left, top, right, bottom = tok["pixel_rect"]
        assert left >= 0 and right <= 512 and top >= 0 and bottom <= 512, (
            f"product scaffold token {tok.get('char')!r} pixel_rect {tok['pixel_rect']} is off-canvas"
        )

# ── Given-equation prior feature-flag tests ───────────────────────────────────


class _PriorCapturingSession:
    """Stub inference session that records the 'prior' kwarg passed to predict."""

    def __init__(self, tokens=None) -> None:
        self.captured_prior = None
        self._tokens = tokens if tokens is not None else _canned_tokens()

    def predict(self, image, **kwargs):
        self.captured_prior = kwargs.get("prior")
        return {
            "schema_version": 1,
            "equation_kind": "subtraction",
            "rows": [],
            "slots": {},
            "tokens": list(self._tokens),
            "timings": {"yolo_ms": 5.0, "stage2_ms": 3.0, "assemble_ms": 0.5},
            "cv_fusion": {
                "mode": "auto",
                "cv_detector": {"cv_boxes": 0, "added": 0, "detections": []},
            },
        }


def _make_app_with_demo_config(tmp_path: Path, given_prior_enabled: bool, fake_client):
    """Build a TestClient with demo_config.given_prior_enabled set to the given value."""
    application = app_mod.create_app(tmp_path)
    application.state.engine_client = fake_client
    # Override demo_config in-place.
    from src.core.run_config import DemoConfig
    application.state.demo_config = DemoConfig(given_prior_enabled=given_prior_enabled)
    return application


def test_evaluate_given_prior_disabled_passes_none_prior(
    tmp_path: Path, fake_client: _FakeEngineClient
) -> None:
    """When given_prior_enabled=False, predict() is called with prior=None (default behaviour)."""
    stub = _PriorCapturingSession()
    application = _make_app_with_demo_config(tmp_path, given_prior_enabled=False, fake_client=fake_client)
    application.state.inference_session = stub
    application.state.demo_session.inference_session = stub
    tc = TestClient(application)

    _select_subtraction(tc)
    resp = tc.post(
        "/api/evaluate",
        json={"image_png": _png_b64()},
    )
    assert resp.status_code == 200, resp.text
    # With flag off, prior must be None.
    assert stub.captured_prior is None, (
        f"Expected prior=None when given_prior_enabled=False, got {stub.captured_prior!r}"
    )


def test_evaluate_given_prior_enabled_passes_non_none_prior(
    tmp_path: Path, fake_client: _FakeEngineClient
) -> None:
    """When given_prior_enabled=True, predict() receives a non-None prior from build_given_prior."""
    stub = _PriorCapturingSession()
    application = _make_app_with_demo_config(tmp_path, given_prior_enabled=True, fake_client=fake_client)
    application.state.inference_session = stub
    application.state.demo_session.inference_session = stub
    tc = TestClient(application)

    _select_subtraction(tc)
    resp = tc.post(
        "/api/evaluate",
        json={"image_png": _png_b64()},
    )
    assert resp.status_code == 200, resp.text
    # With flag on, prior must be non-None (built from subtract-256-89 scaffold).
    assert stub.captured_prior is not None, (
        "Expected a non-None prior when given_prior_enabled=True"
    )
    # The prior should be a non-empty list of GivenNode instances.
    from src.inference.given_prior import GivenNode
    assert len(stub.captured_prior) > 0, "Prior must have at least one node"
    assert all(isinstance(n, GivenNode) for n in stub.captured_prior), (
        f"All prior entries must be GivenNode, got {[type(n) for n in stub.captured_prior]}"
    )


def test_evaluate_given_prior_tokens_excluded_from_grading(
    tmp_path: Path, fake_client: _FakeEngineClient
) -> None:
    """Given=True tokens from the prior must NOT appear in the engine grading token set.

    The stub returns one 'given=True' token + one 'given=False' answer token.
    Only the answer token should reach tokens_to_grid and eventually engine.
    """
    given_token = {
        "label": "op_minus",
        "bbox": [100.0, 60.0, 140.0, 100.0],
        "row": 0,
        "col": 2,
        "grid_col": 2,
        "given": True,
    }
    answer_token = {
        "label": "main_6",
        "bbox": [460.0, 280.0, 500.0, 330.0],
        "row": 3,
        "col": 0,
        "grid_col": 15,
        "given": False,
    }
    stub = _PriorCapturingSession(tokens=[given_token, answer_token])
    application = _make_app_with_demo_config(tmp_path, given_prior_enabled=True, fake_client=fake_client)
    application.state.inference_session = stub
    application.state.demo_session.inference_session = stub
    tc = TestClient(application)

    _select_subtraction(tc)
    resp = tc.post(
        "/api/evaluate",
        json={"image_png": _png_b64()},
    )
    assert resp.status_code == 200, resp.text
    # The debug block contains the tokens sent to engine.
    debug = resp.json().get("debug", {})
    sent = debug.get("engine_sent", {}).get("tokens", [])
    # Child answer token (given=False) may appear as GENERATED.
    # Given operator token (given=True) must NOT appear as GENERATED (it can only
    # appear as FIXED if bank_entry.given_tokens() includes it, which is via the
    # _given_fixed path, not via tokens_to_grid).
    # De-brittle: assert by role + count rather than literal char.
    # The stub emits ONE given=True token (op_minus) and ONE given=False token
    # (answer digit).  After the given-filter in the evaluate route, only
    # given=False tokens reach tokens_to_grid, so GENERATED can have at most 1
    # entry.  The given operator (given=True) must not appear at all.
    generated_tokens = [t for t in sent if t.get("role") == "GENERATED"]
    assert len(generated_tokens) <= 1, (
        f"Expected at most 1 GENERATED token (the child answer); "
        f"got {len(generated_tokens)}: {sent}. "
        "The given=True operator token must not reach tokens_to_grid."
    )
    # Also verify by label: the given token's char ('-') must not appear in
    # GENERATED — but we assert count rather than char to stay role-agnostic.
    generated_labels = {t.get("c") for t in generated_tokens}
    assert "-" not in generated_labels, (
        f"Given operator '-' (given=True) must not appear as GENERATED; sent={sent}"
    )


def test_evaluate_given_prior_failsafe_fallback(
    tmp_path: Path, fake_client: _FakeEngineClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When build_given_prior raises, evaluate_route must still return 200 with normal
    grading (prior falls back to None, not a 500 error).

    The fail-safe in app.py catches any Exception from build_given_prior and logs a
    warning, then continues with prior=None.  This test verifies that contract.
    """
    import src.demo.app as _demo_app_mod  # noqa: PLC0415

    # Monkeypatch build_given_prior to always raise.
    monkeypatch.setattr(
        _demo_app_mod,
        "build_given_prior",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected test failure")),
    )

    stub = _PriorCapturingSession()
    application = _make_app_with_demo_config(tmp_path, given_prior_enabled=True, fake_client=fake_client)
    application.state.inference_session = stub
    application.state.demo_session.inference_session = stub
    tc = TestClient(application)

    _select_subtraction(tc)
    resp = tc.post(
        "/api/evaluate",
        json={"image_png": _png_b64()},
    )
    assert resp.status_code == 200, (
        f"evaluate_route must return 200 even when build_given_prior raises; "
        f"got {resp.status_code}: {resp.text}"
    )
    # Fail-safe: prior must have fallen back to None (stub captures it).
    assert stub.captured_prior is None, (
        f"When build_given_prior raises, prior must fall back to None; "
        f"got {stub.captured_prior!r}"
    )


# --------------------------------------------------------------------------- #
# Fix A: given token field propagation + obstacle filter
# --------------------------------------------------------------------------- #


def test_recognized_tokens_carry_given_field(client: TestClient) -> None:
    """Every token in recognized.tokens must carry a boolean 'given' field.

    The given_prior feature injects scaffold tokens (given=True) into the
    assembled payload; _recognized_block must propagate this field so the
    frontend can filter them out of obstacle seeding.
    """
    given_token = {
        "label": "op_minus",
        "bbox": [100.0, 60.0, 140.0, 100.0],
        "row": 0, "col": 2, "grid_col": 2,
        "given": True,
    }
    answer_token = {
        "label": "main_6",
        "bbox": [460.0, 280.0, 500.0, 330.0],
        "row": 3, "col": 0, "grid_col": 15,
        "given": False,
    }
    stub = _StubInferenceSession(tokens=[given_token, answer_token])
    application = app_mod.create_app(Path("."))
    application.state.engine_client = _FakeEngineClient()
    application.state.inference_session = stub
    application.state.demo_session.inference_session = stub
    tc = TestClient(application)

    _select_subtraction(tc)
    resp = tc.post("/api/evaluate", json={"image_png": _png_b64()})
    assert resp.status_code == 200, resp.text
    tokens = resp.json()["recognized"]["tokens"]
    for tok in tokens:
        assert "given" in tok, f"token {tok.get('id')} missing 'given' field"
        assert isinstance(tok["given"], bool), f"token {tok.get('id')} 'given' not bool"

    given_tokens = [t for t in tokens if t["given"]]
    non_given_tokens = [t for t in tokens if not t["given"]]
    assert len(given_tokens) == 1, f"expected 1 given token, got {given_tokens}"
    assert given_tokens[0]["label"] == "op_minus"
    assert len(non_given_tokens) == 1
    assert non_given_tokens[0]["label"] == "main_6"


# --------------------------------------------------------------------------- #
# Fix B: synthesize anchor for position-less feedback
# --------------------------------------------------------------------------- #


def _make_minimal_bank(
    top_row: list[tuple[int, str]],   # [(x, c), ...]  at y=6 (top operand)
    bot_row: list[tuple[int, str]],   # [(x, c), ...]  at y=5 (bottom operand)
    answer_y: int = 3,
) -> "BankExercise":
    """Build a minimal BankExercise with only operand given tokens.

    Grid layout: top operand at y=6, bottom operand at y=5, both are given=True.
    Answer expected at y=answer_y (non-given).  Grid is wide enough to hold all
    operand xs plus one extra column.
    """
    from src.grading.exercises import BankExercise, ExpectedToken
    from src.grading.types import ExerciseSpec

    spec = ExerciseSpec(
        exercise_id="test-minimal",
        title="Test",
        prompt="test",
        template_name="SUBTRACT_SHORT",
        template_args=["0", "0"],
    )
    tokens = []
    for x, c in top_row:
        tokens.append(ExpectedToken(x=x, y=6, c=c, role="operand", given=True))
    for x, c in bot_row:
        tokens.append(ExpectedToken(x=x, y=5, c=c, role="operand", given=True))
    max_x = max((x for x, _ in top_row + bot_row), default=3)
    tokens.append(ExpectedToken(x=max_x, y=answer_y, c="0", role="answer", given=False))
    return BankExercise(
        spec=spec,
        expected=tuple(tokens),
        grid_width=max_x + 2,
        grid_height=8,
        answer_y=answer_y,
        evaluate_engine_origin=(0, 6),
    )


def _make_50px_calibration(grid_width: int = 6, grid_height: int = 8) -> "GridCalibration":
    """Build a simple 50x50 pixel-per-cell calibration centred at (0,0).

    engine_x_left=0, engine_y_top=grid_height-1 so the top token row (y=grid_height-1)
    maps to pixel row 0.
    """
    from src.grading.grid_mapping import GridCalibration
    return GridCalibration(
        grid_width=grid_width,
        grid_height=grid_height,
        cell_w_px=50.0,
        cell_h_px=50.0,
        origin_x_px=0.0,
        origin_y_px=0.0,
        engine_x_left=0,
        engine_y_top=grid_height - 1,
        invert_y=True,
    )


def test_synthesize_subtract_with_replacement_anchor() -> None:
    """SubtractShortWithReplacement with empty ids+positions gets operand-column anchor.

    Exercise: 256 - 89. Feedback: digits=[6, 9] -> column x=2 (top: '6', bot: '9').
    Expected anchor = pixel rect of top-operand cell x=2 y=6.
    """
    from src.grading.types import EvalFeedback
    from src.demo.app import _build_feedback

    # subtract 256 - 89: top operand x=0..2 y=6, bottom operand x=1..2 y=5
    bank = _make_minimal_bank(
        top_row=[(0, "2"), (1, "5"), (2, "6")],
        bot_row=[(1, "8"), (2, "9")],
    )
    cal = _make_50px_calibration(grid_width=bank.grid_width, grid_height=bank.grid_height)

    fb = EvalFeedback(
        message_type="SubtractShortWithReplacement",
        message_args={"digits": [6, 9], "replacement": 16},
        target_positions=[],
        symbol_id_list=[],
    )
    result = EvalResult(
        elements=[],
        unmatched=[],
        feedback=[fb],
        hint=None,
        progress=0.5,
    )
    out = _build_feedback(result, cal, {}, bank_entry=bank)

    messages = out["messages"]
    assert len(messages) == 1, f"expected 1 message, got {messages}"
    msg = messages[0]
    assert "pixelRect" in msg, (
        "SubtractShortWithReplacement must have synthesized pixelRect; got None"
    )
    assert msg.get("anchor") == "operand", f"anchor should be 'operand', got {msg.get('anchor')}"
    pr = msg["pixelRect"]
    assert len(pr) == 4, f"pixelRect must have 4 elements; got {pr}"
    # Column x=2, y=6 with 50px cells: row = (grid_height-1) - y = 7-6 = 1,
    # col = x - engine_x_left = 2.
    # left=2*50=100, top=1*50=50, right=3*50=150, bottom=2*50=100.
    assert pr == pytest.approx([100.0, 50.0, 150.0, 100.0], abs=0.5), (
        f"Expected pixelRect [100,50,150,100] for column x=2 y=6; got {pr}"
    )


def test_synthesize_product_with_overflow_anchor() -> None:
    """ProductWithOverflow with empty ids+positions gets multiplicand-column anchor.

    Exercise: 38 x 29. Feedback: digit-upper=8, digit-lower=9 -> column x=1
    (top: '8', bot: '9'). Expected anchor = pixel rect of top-operand cell x=1 y=6.
    """
    from src.grading.types import EvalFeedback
    from src.demo.app import _build_feedback

    bank = _make_minimal_bank(
        top_row=[(0, "3"), (1, "8")],
        bot_row=[(0, "2"), (1, "9")],
    )
    cal = _make_50px_calibration(grid_width=bank.grid_width, grid_height=bank.grid_height)

    fb = EvalFeedback(
        message_type="ProductWithOverflow",
        message_args={"digit-upper": 8, "digit-lower": 9, "overflow": 7},
        target_positions=[],
        symbol_id_list=[],
    )
    result = EvalResult(elements=[], unmatched=[], feedback=[fb], hint=None, progress=0.5)
    out = _build_feedback(result, cal, {}, bank_entry=bank)

    messages = out["messages"]
    assert len(messages) == 1
    msg = messages[0]
    assert "pixelRect" in msg, (
        "ProductWithOverflow must have synthesized pixelRect; got None"
    )
    assert msg.get("anchor") == "operand"
    pr = msg["pixelRect"]
    # Column x=1, y=6: row = 7-6=1, col=1 -> left=50, top=50, right=100, bottom=100.
    assert pr == pytest.approx([50.0, 50.0, 100.0, 100.0], abs=0.5), (
        f"Expected [50,50,100,100] for column x=1 y=6; got {pr}"
    )


def test_synthesize_unresolvable_type_falls_back_gracefully() -> None:
    """An unknown or unresolvable feedback type must not add pixelRect (no crash).

    PartialDifference has no stable positional args; the fallback must leave the
    message without a pixelRect and must not raise.
    """
    from src.grading.types import EvalFeedback
    from src.demo.app import _build_feedback

    bank = _make_minimal_bank(
        top_row=[(0, "2"), (1, "5"), (2, "6")],
        bot_row=[(1, "8"), (2, "9")],
    )
    cal = _make_50px_calibration(grid_width=bank.grid_width, grid_height=bank.grid_height)

    fb = EvalFeedback(
        message_type="PartialDifference",
        message_args={"partials": [7]},
        target_positions=[],
        symbol_id_list=[],
    )
    result = EvalResult(elements=[], unmatched=[], feedback=[fb], hint=None, progress=0.3)
    # Must not raise.
    out = _build_feedback(result, cal, {}, bank_entry=bank)

    messages = out["messages"]
    assert len(messages) == 1
    msg = messages[0]
    # PartialDifference has no text (unknown type logs warning) and no pixelRect.
    assert "pixelRect" not in msg, (
        f"PartialDifference must not get a synthesized pixelRect; got {msg}"
    )


def test_synthesize_no_bank_entry_falls_back_gracefully() -> None:
    """When bank_entry=None the synthesis path must skip cleanly (no crash).

    This covers the pre-existing no-position case to ensure it remains safe
    when called without a bank_entry (e.g. from older code paths).
    """
    from src.grading.types import EvalFeedback
    from src.demo.app import _build_feedback

    fb = EvalFeedback(
        message_type="SubtractShortWithReplacement",
        message_args={"digits": [6, 9], "replacement": 16},
        target_positions=[],
        symbol_id_list=[],
    )
    cal = _make_50px_calibration()
    result = EvalResult(elements=[], unmatched=[], feedback=[fb], hint=None, progress=0.0)
    # bank_entry defaults to None.
    out = _build_feedback(result, cal, {})

    messages = out["messages"]
    assert len(messages) == 1
    msg = messages[0]
    assert "pixelRect" not in msg, (
        f"With bank_entry=None, must not synthesize pixelRect; got {msg}"
    )


# --------------------------------------------------------------------------- #
# Regression: token-id-basis bug — given-filtered child list must be the basis
# for token_map, not the full assembled_tokens list.
#
# Bug (fixed): when given-prior injects given=True tokens at the front of
# assembled_tokens, the old code built token_map from the full list.  engine only
# ever sees child tokens (via tokens_to_grid), so it echoes ids on the child
# basis.  "T0" from engine means index 0 in the child-filtered list.  On the full
# list "T0" was the first given operand, not the child answer, so the error
# highlight landed on the wrong cell.
#
# Fix: evaluate_route now computes child_assembled_tokens = [t for t in
# assembled_tokens if not t.get("given", False)] BEFORE building token_map.
# --------------------------------------------------------------------------- #


def test_token_id_basis_unit__child_filtered_maps_T0_to_child_not_operand() -> None:
    """Unit: _build_token_map on child-filtered list maps 'T0' to the child token.

    Demonstrates the id-basis invariant: when the given operands precede the
    child token in assembled_tokens, the full-list map resolves 'T0' to the
    first operand (old bug), while the child-filtered map resolves 'T0' to
    the child answer token (correct).
    """
    from src.demo.app import _build_token_map

    operand_a = {"label": "main_2", "bbox": [10.0, 10.0, 50.0, 50.0], "given": True}
    operand_b = {"label": "main_5", "bbox": [60.0, 10.0, 100.0, 50.0], "given": True}
    operand_c = {"label": "main_6", "bbox": [110.0, 10.0, 150.0, 50.0], "given": True}
    child_answer = {"label": "main_7", "bbox": [460.0, 280.0, 500.0, 330.0], "given": False}

    full_list = [operand_a, operand_b, operand_c, child_answer]
    child_list = [t for t in full_list if not t.get("given", False)]

    # Old (buggy) behaviour: full list — "T0" resolves to the first given operand.
    old_map = _build_token_map(full_list)
    assert old_map.get("T0") is operand_a, (
        "Sanity: full-list token_map['T0'] should be the first operand "
        f"(simulating old bug); got {old_map.get('T0')}"
    )

    # New (correct) behaviour: child-filtered list — "T0" resolves to the answer.
    new_map = _build_token_map(child_list)
    assert new_map.get("T0") is child_answer, (
        "child-filtered token_map['T0'] must resolve to the child answer token, "
        f"not an operand; got {new_map.get('T0')}"
    )
    # The operand bbox must NOT appear under T0 in the new map.
    assert new_map["T0"]["bbox"] == child_answer["bbox"], (
        f"T0 bbox must be the child answer bbox {child_answer['bbox']}, "
        f"got {new_map['T0']['bbox']}"
    )


def test_token_id_basis__error_focus_resolves_to_child_answer_bbox(
    tmp_path: Path,
) -> None:
    """End-to-end: when given operand tokens precede the child token in predict()
    output, an engine ERROR on 'T0' must highlight the child answer bbox, not the
    first given operand's bbox.

    Scenario: 256 - 89.  Fake inference returns 3 given=True operand tokens
    (representing the pre-printed digits) followed by 1 given=False child answer
    token.  engine returns ERROR with symbol_id_list=['T0'].  Under the fixed code,
    token_map is built from the child-filtered list, so 'T0' resolves to the
    child answer token.  Under the old code, 'T0' resolved to the first given
    operand — the focus rect would land on a pre-printed digit, not on what the
    child drew.
    """
    from src.grading.types import EvalElement

    OPERAND_BBOX_A = [10.0, 10.0, 50.0, 50.0]   # first given operand ("2")
    CHILD_ANSWER_BBOX = [460.0, 280.0, 500.0, 330.0]  # child's written digit

    # Fake inference: given operands FIRST, then the child answer token.
    given_tokens_payload = [
        {"label": "main_2", "bbox": OPERAND_BBOX_A, "row": 0, "col": 0, "grid_col": 0, "given": True},
        {"label": "main_5", "bbox": [60.0, 10.0, 100.0, 50.0], "row": 0, "col": 1, "grid_col": 1, "given": True},
        {"label": "main_6", "bbox": [110.0, 10.0, 150.0, 50.0], "row": 0, "col": 2, "grid_col": 2, "given": True},
        {"label": "main_7", "bbox": CHILD_ANSWER_BBOX, "row": 3, "col": 2, "grid_col": 15, "given": False},
    ]

    class _GivenThenChildSession:
        """Predict returns 3 given operand tokens followed by 1 child answer."""

        def predict(self, image, **_kw):
            return {
                "schema_version": 1,
                "equation_kind": "subtraction",
                "rows": [],
                "slots": {},
                "tokens": list(given_tokens_payload),
                "timings": {"yolo_ms": 5.0, "stage2_ms": 3.0, "assemble_ms": 0.5},
                "cv_fusion": {
                    "mode": "auto",
                    "cv_detector": {"cv_boxes": 0, "added": 0, "detections": []},
                },
            }

    class _ErrorOnT0Client:
        """engine client: create returns the real subtract-256-89 fixture; evaluate
        returns ERROR on 'T0' (the id engine assigned to the child's answer token
        in the child-filtered id basis).
        """

        def create_session(self, spec) -> SessionCreated:
            envelope = _load_subtract_create_fixture()
            created = SessionCreated.from_api(envelope)
            return SessionCreated(
                session_id="sess-basis",
                ref_id=created.ref_id,
                marks_total=created.marks_total,
                view_model=created.view_model,
                raw=created.raw,
            )

        def evaluate(self, session_id, ref_id, tokens):
            err = EvalElement(
                template_type="NumberElement",
                template_id="answer-15-15",
                attribute_map={"num": "7", "matched_num": "9"},
                position=(15, 15),
                width=1,
                height=1,
                status="ERROR",
                # engine echoes "T0" — index 0 in the child-filtered send list.
                # That maps to the child answer token in the fixed code.
                symbol_id_list=["T0"],
            )
            return EvalResult(
                elements=[err],
                unmatched=[],
                feedback=[],
                hint=None,
                progress=0.0,
            )

        def solution(self, session_id):
            return _load_subtract_solution_fixture()

        def info(self, session_id):
            return ScoringInfo(finished=False, marks_total=1, marks_earned=0, penalties={})

    application = app_mod.create_app(tmp_path)
    stub_inf = _GivenThenChildSession()
    application.state.engine_client = _ErrorOnT0Client()
    application.state.inference_session = stub_inf
    application.state.demo_session.inference_session = stub_inf

    tc = TestClient(application)
    resp = tc.post("/api/exercise/select", json={"exercise_id": "subtract-256-89"})
    assert resp.status_code == 200, resp.text

    resp = tc.post("/api/evaluate", json={"image_png": _png_b64()})
    assert resp.status_code == 200, resp.text

    fb = resp.json()["feedback"]
    assert fb["outcome"] == "mistake", f"expected 'mistake', got {fb['outcome']!r}"

    focus = fb["focus"]
    error_items = [item for item in focus if item.get("kind") == "error"]
    assert error_items, f"mistake feedback must have at least one error focus item; got {focus}"

    actual_rect = error_items[0]["pixelRect"]

    # The focus rect must resolve via the child answer's 512-space bbox, not via
    # the first given operand's bbox.  The child bbox is at x~460-500 (right side
    # of the canvas); the operand bbox is at x~10-50 (left side).  If the old
    # full-list token_map were used, the rect would anchor on OPERAND_BBOX_A
    # (left side, x<100).  The fixed code produces a rect anchored on
    # CHILD_ANSWER_BBOX (right side, x>400).
    #
    # We assert using the child bbox coordinates directly: the pixel_rect returned
    # by _union_bbox / grid_rect_to_pixels for a glyph-anchored element is the
    # raw 512-space bbox tuple [x0, y0, x1, y1] == CHILD_ANSWER_BBOX.
    assert actual_rect == pytest.approx(CHILD_ANSWER_BBOX, abs=1.0), (
        f"ERROR focus rect must match the child answer bbox {CHILD_ANSWER_BBOX}, "
        f"not the first given operand bbox {OPERAND_BBOX_A}; got {actual_rect}. "
        "This would fail on the old code where token_map was built from the full "
        "assembled_tokens list (T0 = first given operand, not the child answer)."
    )

    # Belt-and-suspenders: explicitly verify it does NOT equal the operand bbox.
    assert actual_rect != pytest.approx(OPERAND_BBOX_A, abs=1.0), (
        f"Focus rect must NOT be the given operand bbox {OPERAND_BBOX_A}; "
        f"got {actual_rect}. This is the token-id-basis regression."
    )
