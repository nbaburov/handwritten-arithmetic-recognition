"""FastAPI application + CLI entry for the feedback demo.

Routes: GET /, GET /api/exercises, POST /api/exercise/select,
POST /api/evaluate, GET /api/score. No grading or pixel-mapping logic lives
here: the grading engine client grades, :mod:`src.grading.grid_mapping` maps
pixels<->grid and reconciles the writing guide, :mod:`src.inference.run`
recognizes.

Error contracts:
    * ``GradingError`` -> 502, so the frontend can show "engine unavailable"
      while still displaying the recognized result.
    * Zero recognized tokens (empty canvas) short-circuits before the engine: 200
      with empty popups and a ``warning``.
    * ``ValueError`` -> 422 via an exception handler, including evaluating before
      an exercise is selected.

Injectable boundaries: ``app.state.engine_client``, ``app.state.demo_session``, and
``app.state.inference_session`` are overridable so tests swap fakes without model
weights or real drawings.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from ..grading.client import GradingClient, GradingError, make_client
from ..grading.derive import derive_exercise
from ..grading.exercises import (
    BankExercise,
    get_exercise,
    list_cms_exercises,
    list_exercises,
)
from ..grading.given_prior_builder import build_given_prior
# compute_feedback_summary removed: verdict now comes directly from engine progress + elements.
from ..grading.grid_mapping import (
    GridCalibration,
    build_calibration,
    grid_rect_to_pixels,
    interaction_answer,
    reconcile_with_guide,
    token_id_for,
    tokens_to_grid,
)
from ..grading.types import (
    EvalElement,
    EvalFeedback,
    EvalHint,
    EvalResult,
    EvaluateResponse,
    ExerciseSpec,
    GridRect,
    GridToken,
    PopupSpec,
    ScoringInfo,
)
from ..core.config import resolve_project_root
from ..core.run_config import GradingConfig, DemoConfig, load_grading_config, load_demo_config
from ..data_pipeline.preprocessing import preprocess_for_pipeline
from ..inference.annotation import annotation_colour
from ..core.ontology import parse_flattened_label
from ..inference.palette import (
    LOW_CONF_THRESHOLD,
    display_glyph,
    equation_kind_display,
)
from ..inference.run import InferenceSession
from .session import DemoSession

logger = logging.getLogger(__name__)

# Fixed pipeline canvas (see preprocessing.py); calibration defaults to 512x512.
_CANVAS_PX: float = 512.0

# Resolved relative to this module so the app serves the vendored Konva UI
# regardless of the working directory.
_STATIC_DIR: Path = Path(__file__).resolve().parent / "static"

# Served by GET / when static/index.html does not exist; keeps the route live
# during backend-only development.
_PLACEHOLDER_HTML: str = (
    "<!doctype html><html><head><meta charset='utf-8'>"
    "<title>Feedback demo</title></head><body>"
    "<h1>Feedback demo</h1>"
    "<p>The canvas frontend (static/index.html) is not built yet. "
    "The JSON API is live: GET /api/exercises, POST /api/exercise/select, "
    "POST /api/evaluate, GET /api/score.</p>"
    "</body></html>"
)

# Backend fallback message catalog: grading engine messageType -> child-friendly text.
# The frontend owns the richer rendering catalog (messages.js); this copy keeps
# ---------------------------------------------------------------------------
# engine message renderers: every real messageType engine sends for these exercises,
# mapped to a child-friendly sentence (ages 6-12) that uses the engine's actual args.
#
# Vocabulary confirmed by live probe (2026-06-19) against the two CMS exercises:
#   subtract-256-89  and  product-38-29
#
# Real types observed:
#   SubtractShortWithReplacement  feedback  {digits:[a,b], replacement:N}
#     -> wrong units/tens needing borrow; engine tells us the two source digits
#        and the replacement value after borrowing.
#   BorrowedAlreadyFeedback       feedback  {unit:N}
#     -> wrong hundreds after a borrow was already applied; engine tells us
#        which column unit was borrowed from.
#   ProductWithOverflow            hint      {overflow?:N, digit-upper:N, digit-lower:N}
#     -> next-step hint for a multiplication column that produces a carry;
#        engine supplies the two factor digits and the overflow amount.
#
# Unknown types: log a warning and fall back to the status generic.
# ---------------------------------------------------------------------------

_EVAL_LOG_RELPATH = ("data", "demo", "sessions.jsonl")


def _append_eval_log(root: Path, record: Dict[str, Any]) -> None:
    """Append one evaluation record as a JSON line to ``data/demo/sessions.jsonl``.

    Best-effort debug trail of every Check: what the recognizer read, the exact
    grid tokens sent to grading engine (with ``type``), the engine's raw response (progress,
    elements, feedback, hint), and the rendered verdict. Logging never raises
    into the request path; a failure is warned and swallowed.
    """
    try:
        log_path = root.joinpath(*_EVAL_LOG_RELPATH)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception as exc:  # never break a request because logging failed
        logger.warning("eval log append failed: %s", exc)


def _render_message(message_type: str, message_args: Dict[str, Any], status: str) -> str:
    """Render a child-friendly sentence from the engine's messageType + args.

    Uses the engine's own fields to build the text; falls back to a neutral status
    phrase only for a genuinely-unknown messageType (logged so gaps are visible).
    Never invents specifics that engine did not supply.
    """
    a = message_args or {}

    if message_type == "SubtractShortWithReplacement":
        # engine: digits=[top, bottom], replacement=N (top after borrowing)
        digits = a.get("digits", [])
        replacement = a.get("replacement")
        if len(digits) >= 2 and replacement is not None:
            top, bottom = digits[0], digits[1]
            return (
                f"Borrow first: {top} becomes {replacement}, then take away {bottom}."
            )
        return "Borrow from the next column, then subtract."

    if message_type == "BorrowedAlreadyFeedback":
        # engine: unit=N (which column position was borrowed from, 1=tens, 2=hundreds)
        unit = a.get("unit")
        col_names = {1: "tens", 2: "hundreds", 3: "thousands"}
        col = col_names.get(unit, "this") if unit is not None else "this"
        return f"This column was already borrowed from. Remember that {col} is now one less."

    if message_type == "BorrowIncorrectFeedback":
        # engine sends no args; the borrow value written above the column is wrong.
        return "Check this borrow. Write the column's new value after you borrow ten."

    if message_type == "PartialDifference":
        # engine: the working so far only reaches a partial difference (often fired
        # when the borrow/replacement cells do not complete the method). args:
        # partials=[value]. Keep the child working through the columns.
        return "Not finished yet. Work through each column, borrowing where you need to."

    if message_type == "ProductWithOverflow":
        # engine: digit-upper=N, digit-lower=N, overflow=N (may be absent in hint)
        du = a.get("digit-upper")
        dl = a.get("digit-lower")
        overflow = a.get("overflow")
        if du is not None and dl is not None:
            if overflow is not None:
                return (
                    f"Multiply {du} by {dl}: the answer carries {overflow} to the next column."
                )
            return f"Multiply {du} by {dl} and remember to carry the overflow."
        return "Multiply these digits and carry any overflow."

    # Unknown / absent messageType: emit NOTHING. Feedback text comes only from
    # Grading engine; the app never invents a nudge to fill a gap. Unknown types are
    # logged (so a real gap is visible) and the empty string is dropped by the
    # caller, leaving just the engine's verdict + cell highlights.
    if message_type:
        logger.warning(
            "engine messageType not in catalog: %r args=%r status=%r",
            message_type, a, status,
        )
    return ""


# ---------------------------------------------------------------------------
# Request bodies (Pydantic): the frontend posts exactly these shapes
# ---------------------------------------------------------------------------

class SelectRequest(BaseModel):
    exercise_id: str = Field(..., description="A bank exercise id.")


class EvaluateRequest(BaseModel):
    """``image_png`` is a base64 PNG (bare or data-URL prefix accepted)."""

    image_png: str = Field(..., description="base64 PNG of the drawing.")


# ---------------------------------------------------------------------------
# Decoding + serialisation helpers
# ---------------------------------------------------------------------------

def _decode_png(image_png: str, where: str) -> np.ndarray:
    """Decode a base64 PNG (bare string or data-URL) to a numpy array.

    Raises ``ValueError`` naming ``where`` on a malformed payload; the route's
    exception handler converts it to 422.
    """
    _MAX_B64_BYTES = 2_000_000
    if not isinstance(image_png, str) or not image_png:
        raise ValueError(f"{where}: image_png must be a non-empty base64 string")
    payload = image_png
    if payload.startswith("data:"):
        comma = payload.find(",")
        if comma == -1:
            raise ValueError(f"{where}: malformed data-URL (no comma separator)")
        payload = payload[comma + 1:]
    if len(payload) > _MAX_B64_BYTES:
        raise ValueError(
            f"{where}: image_png payload too large ({len(payload)} bytes; limit {_MAX_B64_BYTES})"
        )
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{where}: image_png is not valid base64 ({exc})") from exc
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as exc:  # PIL raises a grab-bag; normalise to ValueError.
        raise ValueError(f"{where}: image_png is not a decodable image ({exc})") from exc
    return np.asarray(img)


def _spec_to_dict(spec: ExerciseSpec) -> Dict[str, Any]:
    """Serialise to the camelCase shape the picker renders; adds title/prompt."""
    d = spec.to_api()
    d["title"] = spec.title
    d["prompt"] = spec.prompt
    return d


def _calibration_to_dict(calibration: GridCalibration) -> Dict[str, Any]:
    """Serialise for the canvas renderer. Guide grid, scaffold, and mapper all
    consume this one object so they share a single coordinate frame.
    """
    return {
        "grid_width": calibration.grid_width,
        "grid_height": calibration.grid_height,
        "cell_w_px": calibration.cell_w_px,
        "cell_h_px": calibration.cell_h_px,
        "origin_x_px": calibration.origin_x_px,
        "origin_y_px": calibration.origin_y_px,
        "canvas_px": _CANVAS_PX,
        "engine_x_left": calibration.engine_x_left,
        "engine_y_top": calibration.engine_y_top,
    }


def _centred_calibration_geometry(
    bank_entry: BankExercise,
    canvas_px: float = _CANVAS_PX,
    margin_px: float = 40.0,
) -> Dict[str, Any]:
    """Return a ``canvas_geometry`` dict that sizes and centres the exercise.

    Scales the cell pitch so the used-cell bounding box (all expected tokens,
    given + to-fill) fills the canvas with ``margin_px`` padding on each side.
    Independent pitches per axis let tall exercises (many rows) use a smaller
    cell height without compressing the column width, and vice-versa.

    The engine (x, y) <-> pixel round-trip is still correct: ``cell_to_pixels`` and
    ``pixel_cell_at`` both use ``origin_x_px / cell_w_px`` and
    ``origin_y_px / cell_h_px``, so the same math applies with the new pitch.
    The engine integer grid coordinates are unchanged; only the pixel size of each
    cell grows.

    Centering algorithm:
    1. Find used-cell bounding box from all expected engine token extents.
    2. Scale pitch so the used block fills canvas minus 2*margin_px per axis.
    3. Shift origin so the used block is centred on (canvas_px/2, canvas_px/2).
    """
    all_tokens = list(bank_entry.expected)
    if not all_tokens:
        return {"canvas_px": canvas_px}

    engine_x_left, engine_y_top = bank_entry.evaluate_engine_origin

    # engine y is bottom-origin: higher y = topmost row.
    min_gx = min(t.x for t in all_tokens)
    max_gx = max(t.x for t in all_tokens)
    min_gy = min(t.y for t in all_tokens)  # bottommost row (e.g. answer)
    max_gy = max(t.y for t in all_tokens)  # topmost row (e.g. carry)

    used_cols = max_gx - min_gx + 1
    used_rows = max_gy - min_gy + 1

    # Scale pitch so used region fills canvas minus margins.
    available = canvas_px - 2.0 * margin_px
    cell_w_px = available / used_cols
    cell_h_px = available / used_rows

    # Top-left used cell in recognizer coords (row=0 at top, invert_y=True).
    # engine_y_top is the topmost (highest engine y) token; max_gy is that same value.
    top_used_row = engine_y_top - max_gy   # 0 when the topmost token is the origin row
    left_used_col = min_gx - engine_x_left

    # Pixel position of the used block's top-left corner at origin=(0,0).
    used_block_left_at_zero = left_used_col * cell_w_px
    used_block_top_at_zero = top_used_row * cell_h_px
    used_w_px = cell_w_px * used_cols
    used_h_px = cell_h_px * used_rows
    used_block_cx_at_zero = used_block_left_at_zero + used_w_px / 2.0
    used_block_cy_at_zero = used_block_top_at_zero + used_h_px / 2.0

    # Shift origin so the used block centre lands at canvas centre.
    canvas_cx = canvas_px / 2.0
    canvas_cy = canvas_px / 2.0
    origin_x = canvas_cx - used_block_cx_at_zero
    origin_y = canvas_cy - used_block_cy_at_zero

    return {
        "canvas_px": canvas_px,
        "cell_w_px": cell_w_px,
        "cell_h_px": cell_h_px,
        "origin_x_px": origin_x,
        "origin_y_px": origin_y,
    }


_OPERATOR_DISPLAY: Dict[str, str] = {
    "*": "×",
    "/": "÷",
    "+": "+",
    "-": "-",
}


def _scaffold_tokens(bank_entry: BankExercise, calibration: GridCalibration) -> List[Dict[str, Any]]:
    """Render the scaffold (pre-printed problem) as pixel-placed glyphs.

    Returns two categories:
    - GIVEN tokens (solid, pre-printed problem): the operands, operator, bar, and
      any structural elements. Rendered at full opacity by the frontend.
    - FILL placeholder tokens (dashed, empty outline): every cell the child must
      supply (answer, carries, partial products). Rendered as faint dashed boxes
      so the child sees exactly where to write.

    ``div_bracket`` emits ``kind="div_bracket"`` with a ``pixel_rect`` covering
    the full bus-stop L-shape: [bracket_left, dividend_top, dividend_right,
    dividend_bottom]. The frontend draws the vertical + horizontal lines from
    these four values.

    The ``*`` operator character is mapped to ``×`` for display (the bank stores
    the raw engine token ``*``; the display glyph is the cross symbol).
    """
    # Index by row so the div_bracket handler can find the rightmost dividend digit.
    given_by_row: Dict[int, List] = {}
    for tok in bank_entry.given_tokens():
        given_by_row.setdefault(tok.y, []).append(tok)

    out: List[Dict[str, Any]] = []

    # --- GIVEN tokens (solid, pre-printed) ---
    for token in bank_entry.given_tokens():
        x, y = token.x, token.y

        if token.role == "div_bracket":
            bracket_left, dividend_top, _bracket_right, dividend_bottom = calibration.cell_to_pixels(x, y)
            # Find the rightmost given digit in the same row to anchor the horizontal bar.
            siblings = given_by_row.get(y, [])
            digit_siblings = [t for t in siblings if t.role != "div_bracket" and t.c not in ("", "_")]
            if digit_siblings:
                max_digit_x = max(t.x for t in digit_siblings)
                _dl, _dt, dividend_right, _db = calibration.cell_to_pixels(max_digit_x, y)
            else:
                dividend_right = _bracket_right  # degenerate fallback
            out.append({
                "kind": "div_bracket",
                "char": "",
                "grid_x": x,
                "grid_y": y,
                "pixel_rect": [bracket_left, dividend_top, dividend_right, dividend_bottom],
                "tags": ["GIVEN"],
            })
            continue

        left, top, right, bottom = calibration.cell_to_pixels(x, y)
        if token.role == "result_bar":
            kind = "bar"
        elif token.role == "operator":
            kind = "operator"
        else:
            kind = "digit"
        # Map raw engine operator chars to display glyphs (e.g. * -> ×).
        char = _OPERATOR_DISPLAY.get(token.c, token.c) if token.role == "operator" else token.c
        out.append({
            "kind": kind,
            "char": char,
            "grid_x": x,
            "grid_y": y,
            "pixel_rect": [left, top, right, bottom],
            "tags": ["GIVEN"],
        })

    # --- FILL placeholder tokens (dashed outline, "write here") ---
    # ``role`` is carried through so the frontend can style optional working
    # cells (borrow / carry) differently from the answer line.
    for token in bank_entry.expected_to_fill():
        x, y = token.x, token.y
        left, top, right, bottom = calibration.cell_to_pixels(x, y)
        out.append({
            "kind": "placeholder",
            "char": "",
            "grid_x": x,
            "grid_y": y,
            "role": token.role,
            "pixel_rect": [left, top, right, bottom],
            "tags": ["FILL"],
        })

    return out


# ---------------------------------------------------------------------------
# Recognition -> response assembly (no grading or mapping logic of its own)
# ---------------------------------------------------------------------------

def _build_token_map(
    assembled_tokens: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Build a token_id -> assembled token map for popup bbox resolution.

    Uses the same id scheme as :func:`grid_mapping.token_id_for` (``T{index}`` when
    the token has no id), so map keys match what grading engine echoes back in
    ``element.symbol_id_list``.
    """
    return {token_id_for(tok, i): tok for i, tok in enumerate(assembled_tokens)}


def _union_bbox(
    token_ids: List[str],
    token_map: Dict[str, Dict[str, Any]],
) -> Optional[tuple]:
    """Return the ``(x0, y0, x1, y1)`` union of 512-space bboxes for the given ids.

    Returns ``None`` when no id resolves (caller falls back to grid-cell placement).
    """
    x0s, y0s, x1s, y1s = [], [], [], []
    for tid in token_ids:
        tok = token_map.get(tid)
        if tok is None:
            continue
        bbox = tok.get("bbox") or []
        if len(bbox) < 4:
            continue
        x0s.append(float(bbox[0]))
        y0s.append(float(bbox[1]))
        x1s.append(float(bbox[2]))
        y1s.append(float(bbox[3]))
    if not x0s:
        return None
    return (min(x0s), min(y0s), max(x1s), max(y1s))


def _panel_glyph(label: str) -> str:
    """Display glyph for the RECOGNIZED panel badge.

    ``carry_N`` / ``borrow_N`` labels encode a digit with a role prefix;
    ``display_glyph`` has no entry for these and would return the raw label string.
    Extract the bare digit so the badge shows "1" rather than "carry_1".
    All other labels (main_N, op_*, result_bar, div_bracket) pass through
    ``display_glyph`` unchanged.
    """
    role, detail = parse_flattened_label(label)
    if role in ("carry", "borrow") and detail is not None:
        return detail
    return display_glyph(label)


def _recognized_block(
    payload: Dict[str, Any],
    disagreement_ids: Optional[set] = None,
) -> Dict[str, Any]:
    """Build the model-first ``recognized`` block from an assembled payload.

    Emits the recognizer output as-drawn (no correction or grid-snapping):
    ``equation_kind``, ``equation_label``, ``row_count``, and a ``tokens`` list
    sorted in reading order (row, col, y0, x0). Each token carries its REAL
    512-space bbox, ``color``, ``glyph``, ``role``, ``row``, ``col``,
    ``confidence``, ``low_conf``, and the soft ``disagreement`` flag.

    ``disagreement_ids`` comes from :func:`reconcile_with_guide` (threaded in
    after reconciliation so this function stays pure).
    """
    if disagreement_ids is None:
        disagreement_ids = set()

    equation_kind = str(payload.get("equation_kind", "unknown"))
    raw_tokens = payload.get("tokens", [])

    # Prefer the operator token for the display label: the GNN eq_type head can
    # misclassify (e.g. "subtract" on a scene with op_plus), so the detected
    # operator is more reliable. Display-only; equation_kind is never changed.
    _OP_LABEL_MAP: Dict[str, str] = {
        "op_plus":   "Addition",
        "op_minus":  "Subtraction",
        "op_times":  "Multiplication",
        "op_divide": "Division",
    }
    _operator_labels = [
        _OP_LABEL_MAP[tok.get("label", "")]
        for tok in raw_tokens
        if tok.get("label", "") in _OP_LABEL_MAP
    ]
    if _operator_labels:
        equation_label = _operator_labels[0]
    else:
        equation_label = equation_kind_display(equation_kind)

    def _sort_key(tok: Dict[str, Any]) -> tuple:
        bbox = tok.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        return (int(tok.get("row", 0)), int(tok.get("col", 0)), float(bbox[1]), float(bbox[0]))

    ordered = sorted(raw_tokens, key=_sort_key)

    tokens_out: List[Dict[str, Any]] = []
    rows_seen: set = set()
    for index, token in enumerate(ordered, 1):
        label = str(token.get("label", ""))
        bbox = token.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        conf = float(token.get("confidence", 0.0))
        row = int(token.get("row", 0))
        col = int(token.get("col", 0))
        # grid_col is the scene-global column cluster id (not the within-row ordinal).
        # Older payloads that omit it fall back to col.
        grid_col = int(token.get("grid_col", col))
        rows_seen.add(row)
        tid = token_id_for(token, index - 1)  # index-1 to match _build_token_map (0-based)
        tokens_out.append({
            "id": tid,  # WI-3: stable id matching grid_mapping.token_id_for / symbol_id_list
            "index": index,
            "label": label,
            "glyph": _panel_glyph(label),
            "role": str(token.get("role", "")),
            "bbox": [float(v) for v in bbox],
            "row": row,
            "col": col,
            "grid_col": grid_col,
            "confidence": conf,
            "color": annotation_colour(label, conf),
            "low_conf": conf < LOW_CONF_THRESHOLD,
            "disagreement": tid in disagreement_ids,
            "given": bool(token.get("given", False)),
        })

    glyphs = "".join(t["glyph"] for t in tokens_out)
    summary = f"{equation_label}: {glyphs}" if glyphs else equation_label

    return {
        "equation_kind": equation_kind,
        "equation_label": equation_label,
        "row_count": len(rows_seen),
        "tokens": tokens_out,
        "summary": summary,
    }


def _opencv_used(payload: Dict[str, Any]) -> bool:
    """True only when CV-fusion actually ran AND added at least one detection.

    A skipped or zero-addition branch (YOLO already complete) reports False
    so the badge is honest about whether the fallback did any work.
    """
    cv_info = payload.get("cv_fusion")
    if not isinstance(cv_info, dict):
        return False
    if str(cv_info.get("mode", "")).lower() == "off":
        return False
    if cv_info.get("skipped"):
        return False
    cv_detector = cv_info.get("cv_detector", {})
    second_yolo = cv_info.get("second_yolo", {})
    added = 0
    if isinstance(cv_detector, dict):
        added += int(cv_detector.get("added", 0))
    if isinstance(second_yolo, dict):
        added += int(second_yolo.get("added", 0))
    # Legacy flat layout: top-level ``added`` count (pre-nested schema).
    added += int(cv_info.get("added", 0)) if isinstance(cv_info.get("added"), int) else 0
    return added > 0


def _timings_block(
    payload: Dict[str, Any], total_ms: float, engine_ms: Optional[float] = None
) -> Dict[str, float]:
    """Merge recognizer sub-timings with the route's wall-clock total.

    ``total_ms`` covers recognition + grading + mapping; the sub-timings are the
    pipeline breakdown from the recognizer payload. ``engine_ms`` is the engine API
    round-trip when available.
    """
    sub = payload.get("timings", {})
    result: Dict[str, float] = {
        "total_ms": round(float(total_ms), 2),
        "yolo_ms": float(sub.get("yolo_ms", 0.0)),
        "stage2_ms": float(sub.get("stage2_ms", 0.0)),
        "assemble_ms": float(sub.get("assemble_ms", 0.0)),
    }
    if engine_ms is not None:
        result["engine_ms"] = round(float(engine_ms), 2)
    return result


def _element_popup(
    element: EvalElement,
    calibration: GridCalibration,
    token_map: Dict[str, Dict[str, Any]],
) -> Optional[PopupSpec]:
    """Build a status popup for one graded element (OK / MISSING / ERROR).

    Anchors on the union of matched token bboxes (model-first, ``anchor="glyph"``)
    when ``symbol_id_list`` is non-empty; falls back to the grid cell rect
    (``anchor="cell"``) for MISSING elements where nothing was drawn. Returns
    ``None`` for unrecognised status (tolerant degradation).
    """
    status = element.status.lower()
    if status not in ("ok", "missing", "error"):
        return None
    message = _render_message("", {}, status)

    symbol_ids = list(element.symbol_id_list)
    union = _union_bbox(symbol_ids, token_map) if symbol_ids else None
    if union is not None:
        return PopupSpec(
            pixel_rect=union,
            status=status,
            message=message,
            kind="marker",
            symbol_ids=symbol_ids,
            anchor="glyph",
        )

    # MISSING or token not in map: fall back to grid cell.
    x, y = element.position
    rect = GridRect(left=x, top=y + 1, bottom=y, right=x + 1)
    pixel_rect = grid_rect_to_pixels(rect, calibration)
    return PopupSpec(
        pixel_rect=pixel_rect,
        status=status,
        message=message,
        kind="marker",
        symbol_ids=symbol_ids,
        anchor="cell",
    )


def _feedback_popup(
    feedback: EvalFeedback,
    calibration: GridCalibration,
    token_map: Dict[str, Dict[str, Any]],
) -> Optional[PopupSpec]:
    """Build an error callout for one wrong/extra-cell feedback entry.

    Prefers matched token bboxes; falls back to ``targetPositions[0]`` grid cell.
    """
    symbol_ids = list(feedback.symbol_id_list)
    union = _union_bbox(symbol_ids, token_map) if symbol_ids else None
    message = _render_message(feedback.message_type, feedback.message_args, "error")

    if union is not None:
        return PopupSpec(
            pixel_rect=union,
            status="error",
            message=message,
            kind="callout",
            symbol_ids=symbol_ids,
            anchor="glyph",
        )

    if not feedback.target_positions:
        return None
    pixel_rect = grid_rect_to_pixels(feedback.target_positions[0], calibration)
    return PopupSpec(
        pixel_rect=pixel_rect,
        status="error",
        message=message,
        kind="callout",
        symbol_ids=symbol_ids,
        anchor="cell",
    )


def _hint_popup(
    hint: EvalHint,
    calibration: GridCalibration,
    token_map: Dict[str, Dict[str, Any]],
) -> Optional[PopupSpec]:
    """Build the positioned next-step hint callout.

    Prefers source token bboxes; falls back to ``targetPositions[0]`` grid cell.
    """
    symbol_ids = list(hint.source_token_ids)
    union = _union_bbox(symbol_ids, token_map) if symbol_ids else None
    message = _render_message(hint.message_type, hint.message_args, "hint")

    if union is not None:
        return PopupSpec(
            pixel_rect=union,
            status="hint",
            message=message,
            kind="callout",
            symbol_ids=symbol_ids,
            anchor="glyph",
        )

    if not hint.target_positions:
        return None
    pixel_rect = grid_rect_to_pixels(hint.target_positions[0], calibration)
    return PopupSpec(
        pixel_rect=pixel_rect,
        status="hint",
        message=message,
        kind="callout",
        symbol_ids=symbol_ids,
        anchor="cell",
    )


def _build_popups(
    result: EvalResult,
    calibration: GridCalibration,
    token_map: Dict[str, Dict[str, Any]],
) -> List[PopupSpec]:
    """Project a graded :class:`EvalResult` into positioned :class:`PopupSpec`.

    One status marker per element, one callout per feedback entry, plus the hint.
    Model-first anchoring (real bboxes) when token ids resolve; grid-cell fallback
    for undrawn cells. Unrenderable entries are dropped tolerantly.
    """
    popups: List[PopupSpec] = []
    for element in result.elements:
        popup = _element_popup(element, calibration, token_map)
        if popup is not None:
            popups.append(popup)
    for feedback in result.feedback:
        popup = _feedback_popup(feedback, calibration, token_map)
        if popup is not None:
            popups.append(popup)
    if result.hint is not None:
        popup = _hint_popup(result.hint, calibration, token_map)
        if popup is not None:
            popups.append(popup)
    return popups


# ---------------------------------------------------------------------------
# Feedback summary projection (verdict + focus pixel rects)
# ---------------------------------------------------------------------------


def _synthesize_fb_anchor(
    fb: "EvalFeedback",
    bank_entry: Optional["BankExercise"],
    calibration: "GridCalibration",
) -> Optional[tuple]:
    """Derive a pixel-rect anchor for a position-less feedback message.

    Called only when ``fb.symbol_id_list`` is empty AND ``fb.target_positions``
    is empty.  Uses the feedback ``message_type`` and ``message_args`` to
    identify a target operand cell from ``bank_entry.given_tokens()``, then maps
    it to 512-space pixels via ``grid_rect_to_pixels``.

    Returns a 4-tuple ``(left, top, right, bottom)`` on success, or ``None``
    when the type is unrecognised or the cell cannot be resolved (caller falls
    back to the existing no-anchor behaviour, no crash).

    Resolver table
    --------------
    SubtractShortWithReplacement
        args ``digits=[top_digit, bottom_digit]``.  Finds the column where
        the top operand digit == top_digit AND the operand directly below
        (one row down) == bottom_digit.  Anchors to the top-row operand cell
        in that column (the minuend digit the child needs to borrow from).

    BorrowedAlreadyFeedback
        args ``unit=N`` (1=tens, 2=hundreds, 3=thousands, counting from right).
        Finds the top operand row and picks the Nth digit from the right.
        Anchors to that cell.

    ProductWithOverflow
        args ``digit-upper=N, digit-lower=N``.  Finds the column where the
        top operand (multiplicand) == digit-upper AND the bottom operand
        (multiplier) == digit-lower.  Anchors to the top-row cell.

    PartialDifference
        No stable positional args available; returns None (graceful fallback).

    All other types: returns None (graceful fallback).
    """
    if bank_entry is None:
        return None

    msg_type = fb.message_type
    a = fb.message_args or {}

    try:
        given = bank_entry.given_tokens()

        # Index operand-role given tokens by (x, y) and by y-level.
        operands = [t for t in given if t.role == "operand"]
        if not operands:
            return None

        # Determine the two operand y-levels (top = highest grid y, bottom = lower).
        y_levels = sorted({t.y for t in operands}, reverse=True)  # desc: higher y = upper row

        def _pixels_for_cell(x: int, y: int) -> tuple:
            rect = GridRect(left=x, top=y + 1, bottom=y, right=x + 1)
            return grid_rect_to_pixels(rect, calibration)

        if msg_type == "SubtractShortWithReplacement":
            digits = a.get("digits", [])
            if len(digits) < 2:
                return None
            top_digit, bottom_digit = str(digits[0]), str(digits[1])
            # Build per-column operand maps for the top and bottom operand rows.
            top_y = y_levels[0] if len(y_levels) >= 1 else None
            bot_y = y_levels[1] if len(y_levels) >= 2 else None
            if top_y is None or bot_y is None:
                return None
            top_by_x = {t.x: t for t in operands if t.y == top_y}
            bot_by_x = {t.x: t for t in operands if t.y == bot_y}
            # Find column where top digit matches and bottom digit matches.
            for x, top_tok in top_by_x.items():
                bot_tok = bot_by_x.get(x)
                if top_tok.c == top_digit and bot_tok is not None and bot_tok.c == bottom_digit:
                    return _pixels_for_cell(x, top_y)
            return None

        if msg_type == "BorrowedAlreadyFeedback":
            unit = a.get("unit")
            if unit is None:
                return None
            top_y = y_levels[0] if y_levels else None
            if top_y is None:
                return None
            top_ops = sorted([t for t in operands if t.y == top_y], key=lambda t: t.x, reverse=True)
            # unit=1 -> rightmost (units column), unit=2 -> tens, etc. (1-based from right)
            idx = int(unit) - 1
            if idx < 0 or idx >= len(top_ops):
                return None
            tok = top_ops[idx]
            return _pixels_for_cell(tok.x, top_y)

        if msg_type == "ProductWithOverflow":
            du = a.get("digit-upper")
            dl = a.get("digit-lower")
            if du is None or dl is None:
                return None
            du_str, dl_str = str(du), str(dl)
            top_y = y_levels[0] if len(y_levels) >= 1 else None
            bot_y = y_levels[1] if len(y_levels) >= 2 else None
            if top_y is None or bot_y is None:
                return None
            top_by_x = {t.x: t for t in operands if t.y == top_y}
            bot_by_x = {t.x: t for t in operands if t.y == bot_y}
            for x, top_tok in top_by_x.items():
                bot_tok = bot_by_x.get(x)
                if top_tok.c == du_str and bot_tok is not None and bot_tok.c == dl_str:
                    return _pixels_for_cell(x, top_y)
            return None

    except Exception:  # noqa: BLE001  — never raise into request path
        logger.warning(
            "_synthesize_fb_anchor failed for type=%r args=%r",
            msg_type, a, exc_info=True,
        )

    return None


def _build_feedback(
    result: "EvalResult",
    calibration: "GridCalibration",
    token_map: Dict[str, Dict[str, Any]],
    bank_entry: Optional["BankExercise"] = None,
) -> Dict[str, Any]:
    """Project an engine :class:`EvalResult` directly to ``{outcome, headline, messages, focus, hint}``.

    Verdict is the engine's own ``progress``: solved iff ``>= 1.0``. Focus markers and
    messages come from the engine's ``elements`` (ERROR/MISSING) and ``feedback`` entries.
    The hint is passed through as-is. No app-side heuristics or invented copy.

    ``focus`` items:
      - kind="error"   - child drew something wrong; pixelRect from the matched
                         token's 512-space bbox (glyph anchor).
      - kind="missing" - nothing drawn in a required cell; pixelRect from the
                         calibrated grid-cell rect (cell anchor).

    Unresolvable ERROR items (token not in map) are dropped; they have no
    bbox to draw on canvas.
    """
    solved = result.progress >= 1.0

    # Derive outcome from the engine's elements and progress.
    error_elements = [e for e in result.elements if e.status.upper() == "ERROR"]
    missing_elements = [e for e in result.elements if e.status.upper() == "MISSING"]

    # Headline is a minimal verdict LABEL for the engine's own progress / element status,
    # not a guidance nudge. The app never tells the child what to do; any "how to
    # fix it" text comes only from the engine's feedback messages + hint.
    if solved:
        outcome = "correct"
        headline = "Correct"
    elif error_elements:
        outcome = "mistake"
        headline = "Not quite"
    else:
        outcome = "incomplete"
        headline = "Not finished"

    # Build focus items for ERROR cells (glyph-anchored).
    focus_out: List[Dict[str, Any]] = []
    for element in error_elements:
        sym_ids = element.symbol_id_list
        union = _union_bbox(sym_ids, token_map) if sym_ids else None
        if union is None:
            # No token in map for this ERROR: fall back to grid cell.
            x, y = element.position
            rect = GridRect(left=x, top=y + 1, bottom=y, right=x + 1)
            pixel_rect = grid_rect_to_pixels(rect, calibration)
            focus_out.append({
                "pixelRect": list(pixel_rect),
                "symbolId": sym_ids[0] if sym_ids else None,
                "kind": "error",
                "anchor": "cell",
            })
        else:
            focus_out.append({
                "pixelRect": list(union),
                "symbolId": sym_ids[0] if sym_ids else None,
                "kind": "error",
                "anchor": "glyph",
            })

    # Build focus items for MISSING cells (grid-cell-anchored).
    for element in missing_elements:
        x, y = element.position
        rect = GridRect(left=x, top=y + 1, bottom=y, right=x + 1)
        pixel_rect = grid_rect_to_pixels(rect, calibration)
        focus_out.append({
            "pixelRect": list(pixel_rect),
            "symbolId": None,
            "kind": "missing",
            "anchor": "cell",
        })

    # Prefer a highlighted cell (ERROR first, then MISSING) as the anchor for
    # position-less feedback, so the bubble stays visually connected to the box
    # it explains instead of floating over an unrelated operand.
    _error_rects = [f["pixelRect"] for f in focus_out if f["kind"] == "error"]
    _missing_rects = [f["pixelRect"] for f in focus_out if f["kind"] == "missing"]
    primary_focus_rect = (
        _error_rects[0] if _error_rects
        else (_missing_rects[0] if _missing_rects else None)
    )

    # Build feedback messages from the engine's feedback list (rendered with real args).
    messages_out: List[Dict[str, Any]] = []
    for fb in result.feedback:
        text = _render_message(fb.message_type, fb.message_args, "error")
        # Resolve pixel rect: prefer token bboxes, then first targetPosition,
        # then synthesize from operand positions when both are absent.
        sym_ids = fb.symbol_id_list
        union = _union_bbox(sym_ids, token_map) if sym_ids else None
        if union is not None:
            pixel_rect_fb = list(union)
            anchor = "glyph"
        elif fb.target_positions:
            pixel_rect_fb = list(grid_rect_to_pixels(fb.target_positions[0], calibration))
            anchor = "cell"
        elif primary_focus_rect is not None:
            # No engine position: anchor to the highlighted error/missing cell (the red
            # box) so the bubble is visually connected to what it explains.
            pixel_rect_fb = list(primary_focus_rect)
            anchor = "error"
        else:
            # No highlighted cell either: attempt operand-column synthesis.
            synthesized = _synthesize_fb_anchor(fb, bank_entry, calibration)
            if synthesized is not None:
                pixel_rect_fb = list(synthesized)
                anchor = "operand"
            else:
                pixel_rect_fb = None
                anchor = None
        entry: Dict[str, Any] = {
            "messageType": fb.message_type,
            "text": text,
            "status": "error",
        }
        if pixel_rect_fb is not None:
            entry["pixelRect"] = pixel_rect_fb
            entry["anchor"] = anchor
        messages_out.append(entry)

    # Pass the engine's hint through as a rendered message.
    hint_out: Optional[Dict[str, Any]] = None
    if result.hint is not None:
        h = result.hint
        text = _render_message(h.message_type, h.message_args, "hint")
        sym_ids = h.source_token_ids
        union = _union_bbox(sym_ids, token_map) if sym_ids else None
        if union is not None:
            hint_pixel_rect = list(union)
            hint_anchor = "glyph"
        elif h.target_positions:
            hint_pixel_rect = list(grid_rect_to_pixels(h.target_positions[0], calibration))
            hint_anchor = "cell"
        else:
            hint_pixel_rect = None
            hint_anchor = None
        hint_out = {
            "messageType": h.message_type,
            "text": text,
            "status": "hint",
        }
        if hint_pixel_rect is not None:
            hint_out["pixelRect"] = hint_pixel_rect
            hint_out["anchor"] = hint_anchor

    return {
        "outcome": outcome,
        "headline": headline,
        "messages": messages_out,
        "hint": hint_out,
        "focus": focus_out,
    }


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

class _NoCacheStaticFiles(StaticFiles):
    """Serve static assets with ``Cache-Control: no-cache``.

    The browser revalidates every request (ETag-based, so unchanged files still
    get a fast 304), which means frontend edits show up on a normal refresh
    instead of needing a hard reload. This is a dev-demo convenience; it does
    not disable conditional caching, only forces revalidation.
    """

    def file_response(self, *args: Any, **kwargs: Any):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def create_app(project_root: Path) -> FastAPI:
    """Build the demo FastAPI app bound to ``project_root``.

    The three collaborators on ``app.state`` are overridable for tests:
    ``engine_client`` (mock by default), ``inference_session`` (auto-loaded), and
    ``demo_session`` (wraps the inference session). Tests swap fakes before issuing requests.
    """
    app = FastAPI(title="grading-demo", docs_url=None, redoc_url=None)
    root = Path(project_root)
    app.state.project_root = root

    config: GradingConfig = load_grading_config(root)
    app.state.grading_config = config
    app.state.engine_client = make_client(config, root)
    demo_config: DemoConfig = load_demo_config(root)
    app.state.demo_config = demo_config

    inference_session = InferenceSession()
    inference_session.auto_load(root)
    app.state.inference_session = inference_session
    app.state.demo_session = DemoSession(inference_session=inference_session)
    # Per-UUID layout cache: BankExercise is deterministic per exercise spec, so
    # derive_exercise + the solution() call run once per exercise per process.
    # The session must stay fresh (grading needs a live session id), so
    # create_session is always called; only derive_exercise is cached.
    app.state.derived_cache: dict[str, BankExercise] = {}

    # Mount only when the directory exists so a backend-only checkout still starts.
    if _STATIC_DIR.is_dir():
        app.mount("/static", _NoCacheStaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.exception_handler(ValueError)
    async def _value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
        """Map ValueError to 422. Routes needing a different status raise HTTPException directly."""
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        """Serve the static app shell, or the placeholder if index.html is absent."""
        index_html = _STATIC_DIR / "index.html"
        if index_html.is_file():
            return HTMLResponse(index_html.read_text(encoding="utf-8"))
        return HTMLResponse(_PLACEHOLDER_HTML)

    @app.get("/api/exercises")
    async def exercises_route() -> Dict[str, Any]:
        """List the exercises the demo grades against.

        Honors ``config.toml [grading] exercise_ids`` so adding a UUID there
        adds it to the picker; falls back to the validated set when unset.
        """
        configured = app.state.grading_config.exercise_ids or None
        return {"exercises": [_spec_to_dict(spec) for spec in list_cms_exercises(configured)]}

    @app.post("/api/exercise/select")
    async def select_route(body: SelectRequest) -> Dict[str, Any]:
        """Create a grading engine session for one exercise; return scaffold + calibration.

        Unknown exercise id raises ValueError (-> 422); engine failure -> 502.
        """
        configured = app.state.grading_config.exercise_ids or None
        allowed = {s.exercise_id for s in list_cms_exercises(configured)}
        if body.exercise_id not in allowed:
            raise ValueError(
                f"Unknown or unconfigured exercise: {body.exercise_id!r}"
            )
        try:
            spec = get_exercise(body.exercise_id)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc

        client: GradingClient = app.state.engine_client
        try:
            session_created = client.create_session(spec)
        except GradingError as exc:
            logger.warning("engine create failed: %s", exc)
            raise HTTPException(status_code=502, detail="engine unavailable") from exc

        # Derive the exercise layout from create view-model + solution response.
        # This replaces the static hand-written bank: derive_exercise reads the
        # engine grid dimensions, area-type map, and solution tokens to classify every
        # cell into roles (operand, answer, borrow, carry, partial) in the native
        # solution frame (translation-invariant; submitted verbatim to evaluate).
        #
        # NTH-2: BankExercise is deterministic per exercise spec; cache it so
        # solution() + derive_exercise run only once per exercise per process.
        # create_session is always called above to obtain a fresh live session_id.
        cache: dict[str, BankExercise] = app.state.derived_cache
        cache_key = spec.exercise_id
        if cache_key not in cache:
            try:
                solution_data = client.solution(session_created.session_id)
                cache[cache_key] = derive_exercise(
                    spec, session_created.view_model, solution_data
                )
            except GradingError as exc:
                logger.warning("engine solution/derive failed: %s", exc)
                raise HTTPException(status_code=502, detail="engine unavailable") from exc
            logger.debug("derived_cache: stored layout for %r", cache_key)
        else:
            logger.debug("derived_cache: reused layout for %r", cache_key)
        bank_entry = cache[cache_key]
        engine_x_left, engine_y_top = bank_entry.evaluate_engine_origin

        # Centred geometry: shift origin so the exercise block is visually centred
        # on the 512x512 canvas; pitch is never changed.
        centred_geom = _centred_calibration_geometry(bank_entry, _CANVAS_PX)
        calibration = build_calibration(
            session_created.view_model,
            {
                **centred_geom,
                "grid_width": bank_entry.grid_width,
                "grid_height": bank_entry.grid_height,
                "engine_x_left": engine_x_left,
                "engine_y_top": engine_y_top,
            },
        )
        demo_session: DemoSession = app.state.demo_session
        demo_session.select(spec, session_created, calibration, bank_entry)

        return {
            "exercise": _spec_to_dict(bank_entry.spec),
            "sessionId": session_created.session_id,
            "refId": session_created.ref_id,
            "marksTotal": session_created.marks_total,
            "calibration": _calibration_to_dict(calibration),
            "scaffold": _scaffold_tokens(bank_entry, calibration),
        }

    @app.post("/api/evaluate")
    async def evaluate_route(body: EvaluateRequest) -> Dict[str, Any]:
        """Recognize the drawing, grade it, and return positioned popups.

        Pipeline: decode + preprocess, recognize, map tokens to grid cells
        (``tokens_to_grid``), reconcile against the writing guide
        (``reconcile_with_guide``; flags disagreement, never moves model output),
        grade, then project elements/feedback/hint to 512-space pixel rects.
        Zero recognized tokens short-circuit before the engine (200 + warning).
        Engine failure -> 502. Arithmetic is never corrected.
        """
        _spec, session_created, calibration, bank_entry = _require_active(app)

        t_start = time.perf_counter()
        gray = preprocess_for_pipeline(_decode_png(body.image_png, "evaluate"))
        inference_session: InferenceSession = app.state.inference_session

        # Given-equation prior: when enabled, build the scaffold prior from the
        # active exercise and pass it to predict so the GNN sees given tiles.
        # Failure is non-fatal: fall back to prior=None (grading proceeds as today).
        # Given nodes are filtered OUT of the grading token set below -- grading
        # always uses bank_entry.given_tokens() as FIXED exactly as before.
        _prior = None
        demo_config: DemoConfig = app.state.demo_config
        if demo_config.given_prior_enabled:
            try:
                _prior = build_given_prior(bank_entry, calibration)
            except Exception as _exc:  # noqa: BLE001
                logger.warning("given_prior build failed, falling back to prior=None: %s", _exc)

        payload = inference_session.predict(gray, prior=_prior)

        opencv_used = _opencv_used(payload)
        assembled_tokens = payload.get("tokens", [])

        # Popup/focus builders resolve engine-echoed ids (element.symbol_id_list) in
        # token_map. engine only ever sees the given-filtered child tokens (via
        # tokens_to_grid below), so the id basis here MUST be that same filtered
        # list -- otherwise "T0" denotes a different glyph in each space and the
        # error highlight lands on the wrong cell (e.g. an operand instead of the
        # answer). child_assembled_tokens is reused for tokens_to_grid below.
        child_assembled_tokens = [t for t in assembled_tokens if not t.get("given", False)]
        token_map = _build_token_map(child_assembled_tokens)

        # reconcile_with_guide flags when model row/col disagrees with the guide
        # cell the ink sits in (soft flag; never mutates the model output).
        reconciled = reconcile_with_guide(assembled_tokens, calibration)
        disagreement_ids = {r.token_id for r in reconciled if r.disagreement}

        recognized = _recognized_block(payload, disagreement_ids)

        # Disagreement detail for the frontend's soft warning overlay.
        recognized["disagreements"] = [
            {
                "token_id": r.token_id,
                "char": r.char,
                "recognizer_row": r.recognizer_row,
                "recognizer_col": r.recognizer_col,
                "guide_row": r.guide_row,
                "guide_col": r.guide_col,
            }
            for r in reconciled
            if r.disagreement
        ]

        # Borrow cells must be sent to engine as type=REPLACE; carry cells as type=OVERFLOW.
        # Pass them so tokens_to_grid tags + concatenates ink in those cells. Optional:
        # answer-only drawings have no ink here and still score 1.0.
        _replace_cells = {
            (t.x, t.y) for t in bank_entry.expected_to_fill() if t.role == "borrow"
        }
        _overflow_cells = {
            (t.x, t.y) for t in bank_entry.expected_to_fill() if t.role == "carry"
        }
        # Filter given=True tokens out before grid-mapping: these are injected
        # Grading uses bank_entry.given_tokens() as FIXED (below); given prior nodes
        # never reach tokens_to_grid. child_assembled_tokens is built above so the
        # engine send-list and token_map share one id basis.
        child_tokens = tokens_to_grid(
            child_assembled_tokens, calibration,
            replace_cells=_replace_cells,
            overflow_cells=_overflow_cells,
        )

        # Empty canvas: short-circuit before the engine with 200 + warning.
        # Checked against child-drawn tokens only; given-token injection below
        # must not suppress this guard.
        if not child_tokens:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            response = EvaluateResponse(
                recognized=recognized,
                popups=[],
                timings=_timings_block(payload, total_ms),
                opencv_used=opencv_used,
                progress=0.0,
                marks={"earned": 0, "total": session_created.marks_total, "finished": False},
            )
            out = response.to_api()
            out["warning"] = "Nothing was drawn. Write your answer on the canvas, then press Check."
            out["feedback"] = {
                "outcome": "incomplete",
                "headline": "Write your answer on the canvas first.",
                "focus": [],
            }
            _append_eval_log(app.state.project_root, {
                "ts": time.time(),
                "exercise_id": _spec.exercise_id,
                "event": "empty_canvas",
                "recognized": {"token_count": len(assembled_tokens)},
                "warning": out["warning"],
            })
            return out

        # The given (pre-printed problem) is ALWAYS sent as FIXED so engine grades
        # it as scaffold, never as the child's answer. A child often re-draws
        # the whole sum (operands, operator, bar) on top of the guide; that ink
        # must not be graded, or the operands land as GENERATED tokens and engine
        # errors on them and never reaches the answer. So: send every given cell
        # as FIXED, and drop any child ink that falls on a given cell. Only ink
        # on fill cells (the answer / working) is graded.
        _given_cells = {(t.x, t.y) for t in bank_entry.given_tokens() if t.c != ""}
        _given_fixed = [
            GridToken(id=f"G{_gi}", c=_gt.c, x=_gt.x, y=_gt.y, tags=["FIXED"])
            for _gi, _gt in enumerate(bank_entry.given_tokens())
            if _gt.c != ""
        ]
        _child_fill = [t for t in child_tokens if (t.x, t.y) not in _given_cells]
        grid_tokens: list[GridToken] = _given_fixed + _child_fill

        client: GradingClient = app.state.engine_client
        t_engine_start = time.perf_counter()
        try:
            result: EvalResult = client.evaluate(
                session_created.session_id, session_created.ref_id, grid_tokens
            )
        except GradingError as exc:
            logger.warning("engine evaluate failed: %s", exc)
            raise HTTPException(status_code=502, detail="engine unavailable") from exc
        engine_ms = (time.perf_counter() - t_engine_start) * 1000.0

        popups = _build_popups(result, calibration, token_map)
        total_ms = (time.perf_counter() - t_start) * 1000.0

        finished = result.progress >= 1.0
        marks_earned = session_created.marks_total if finished else 0
        response = EvaluateResponse(
            recognized=recognized,
            popups=popups,
            timings=_timings_block(payload, total_ms, engine_ms=engine_ms),
            opencv_used=opencv_used,
            progress=result.progress,
            marks={
                "earned": marks_earned,
                "total": session_created.marks_total,
                "finished": finished,
            },
        )
        out = response.to_api()

        out["feedback"] = _build_feedback(result, calibration, token_map, bank_entry=bank_entry)
        out["debug"] = {
            "engine_sent": {
                "session_id": session_created.session_id,
                "tokens": [
                    {
                        "id": t.id,
                        "c": t.c,
                        "x": t.x,
                        "y": t.y,
                        "type": t.type,
                        "role": "FIXED" if "FIXED" in (t.tags or []) else "GENERATED",
                    }
                    for t in grid_tokens
                ],
            },
            "engine_received": {
                "progress": result.progress,
                "elements": [
                    {
                        "status": e.status,
                        "pos": list(e.position),
                        "ids": e.symbol_id_list,
                    }
                    for e in result.elements
                ],
                "feedback": [
                    {"type": f.message_type, "args": f.message_args}
                    for f in result.feedback
                ],
                "hint": {
                    "type": result.hint.message_type,
                    "pos": [p.to_api() for p in result.hint.target_positions],
                }
                if result.hint
                else None,
            },
        }
        # Persist the full exchange for offline debugging while the user plays.
        _append_eval_log(app.state.project_root, {
            "ts": time.time(),
            "exercise_id": _spec.exercise_id,
            "event": "evaluate",
            "recognized": {
                "equation_kind": recognized.get("equation_kind"),
                "row_count": recognized.get("row_count"),
                "tokens": [
                    {
                        "glyph": t.get("glyph"),
                        "label": t.get("label"),
                        "row": t.get("row"),
                        "grid_col": t.get("grid_col", t.get("col")),
                        "bbox": t.get("bbox"),
                        "confidence": t.get("confidence"),
                    }
                    for t in recognized.get("tokens", [])
                ],
            },
            "sent": out["debug"]["engine_sent"]["tokens"],
            "received": out["debug"]["engine_received"],
            "feedback": {
                "outcome": out["feedback"].get("outcome"),
                "headline": out["feedback"].get("headline"),
                "messages": [m.get("text") for m in out["feedback"].get("messages", [])],
                "hint": (out["feedback"].get("hint") or {}).get("text"),
            },
            "timings": out.get("timings"),
        })
        return out

    @app.get("/api/score")
    async def score_route() -> Dict[str, Any]:
        """Return the active session's score. Requires a selected exercise (422 otherwise)."""
        _spec, session_created, _calibration, _bank_entry = _require_active(app)
        client: GradingClient = app.state.engine_client
        try:
            info: ScoringInfo = client.info(session_created.session_id)
        except GradingError as exc:
            logger.warning("engine info failed: %s", exc)
            raise HTTPException(status_code=502, detail="engine unavailable") from exc
        return info.to_api()

    return app


def _require_active(app: FastAPI):
    """Return the active demo quad (spec, session_created, calibration, bank_exercise)."""
    demo_session: DemoSession = app.state.demo_session
    return demo_session.require_active()


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def parse_args(argv: "list[str] | None" = None):
    """Parse ``demo`` subcommand args (--host, --port, --project-root)."""
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="src demo",
        description="feedback demo: draw an exercise, get guided popups.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address for the local server (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="TCP port for the local server (default: 8000).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root containing data/ and src/ (default: auto-resolved).",
    )
    args = parser.parse_args(argv)
    if args.project_root is None:
        args.project_root = resolve_project_root(Path(__file__))
    return args


def main() -> None:
    """Launch uvicorn for ``python -m src demo``."""
    import uvicorn  # noqa: PLC0415

    args = parse_args()
    app = create_app(args.project_root)
    logger.info(
        "demo: serving http://%s:%d (project_root=%s)",
        args.host,
        args.port,
        args.project_root,
    )
    # workers=1 is mandatory: DemoSession lives in app.state memory. Multiple
    # workers each hold independent state and would serve mismatched calibrations.
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
        h11_max_incomplete_event_size=2_000_000,
    )


if __name__ == "__main__":
    main()
