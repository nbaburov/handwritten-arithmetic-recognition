"""FastAPI application + CLI entry for the set-maker tool (WS-G).

This module owns the six HTTP routes that drive the annotator and the uvicorn
launch wired into the ``setmaker`` subcommand (registered in ``src/cli.py``). It
owns no business logic: every route delegates to a single-responsibility module
in this package (``worklist`` plans, ``targets`` generates, ``detect`` finds
boxes, ``matcher`` fuses, ``exporters`` writes, ``state`` persists) and returns a
JSON shape the Konva frontend (WS-H) consumes verbatim.

Routes
------
    GET  /              -> the static app shell (``static/index.html`` when present,
                           a minimal placeholder until WS-H lands).
    POST /api/mode      -> set the mode ("eval" | "train"), build the worklist via
                           ``worklist.py``, persist it via ``state.py``, return the
                           initial progress.
    GET  /api/next      -> the worklist item at the cursor plus its generated target
                           (reference string + per-symbol grid GT) via ``targets.py``.
    POST /api/detect    -> decode the drawing, preprocess it, detect boxes via
                           ``detect.py``, fuse them with the cursor's target via
                           ``matcher.py``, return the prefilled ``AnnotationDraft``
                           list.
    POST /api/save      -> validate + write the finished scene via ``exporters.py``,
                           advance the cursor via ``state.py``, return the new
                           progress.
    GET  /api/progress  -> per-case + overall counts for the active mode via
                           ``state.py``.

Error contracts (the plan ``Error handling`` section)
-----------------------------------------------------
    * Generator pool missing / not built -> ``/api/next`` 503 with a clear message
      ("run ``validate`` to build the symbol pool first").
    * Export validation failure (bad bbox, unknown label, ...) -> ``/api/save`` 422
      naming the offending field; nothing is written.
    * Train mode with no usable eval history -> the worklist falls back to
      provisional weights and ``/api/mode`` returns a non-fatal ``warning`` string;
      it never crashes.
    * Detection returning zero boxes -> ``/api/detect`` 200 with an empty draft list
      plus a ``notice``; never a 500.

Injectable boundaries (testability)
-----------------------------------
The two heavy collaborators (the symbol-pool target generator and the YOLO
detector) are resolved through overridable hooks on ``app.state``
(``generate_target_fn`` / ``detect_fn``), defaulting to the real ``targets`` and
``detect`` functions. A test can swap in a synthetic target or a fixed detection
list and exercise every route without a symbol pool, model weights, or a real
drawing, keeping the routes themselves the only thing under test.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from ..core.config import resolve_project_root
from ..data_pipeline.preprocessing import preprocess_for_pipeline
from ..eval.harness import _ASSEMBLER_TO_LABEL as _SHORT_TO_LONG_EQUATION_KIND
from ..eval.labels import EQUATION_KINDS, ERROR_KINDS
from . import detect as detect_mod
from . import exporters, state, targets, worklist
from .types import AnnotationDraft, Mode, SessionState, TargetScene

logger = logging.getLogger(__name__)

# Structure-only / no-equation scenes carry ``equation_type == "ood_unknown"`` in
# the target; the eval sidecar has no such kind, so they are recorded as
# ``bare_digits`` (their value is the spatial-structure GT, per the plan). This is
# the single place the OOD->eval-kind mapping is defined.
_OOD_EVAL_EQUATION_KIND: str = "bare_digits"

# Default location of the static frontend shell (WS-H). Resolved relative to this
# module so the app serves the vendored Konva UI regardless of the cwd.
_STATIC_DIR: Path = Path(__file__).resolve().parent / "static"

# Placeholder shell served by ``GET /`` until the WS-H frontend exists, so the
# route is live and inspectable during backend development.
_PLACEHOLDER_HTML: str = (
    "<!doctype html><html><head><meta charset='utf-8'>"
    "<title>set-maker</title></head><body>"
    "<h1>set-maker</h1>"
    "<p>The annotator frontend (static/index.html) is not built yet. "
    "The JSON API is live: POST /api/mode, GET /api/next, POST /api/detect, "
    "POST /api/save, GET /api/progress.</p>"
    "</body></html>"
)


# ---------------------------------------------------------------------------
# Request bodies (Pydantic) — the frontend posts exactly these shapes
# ---------------------------------------------------------------------------

class ModeRequest(BaseModel):
    """Body for ``POST /api/mode``.

    ``mode`` selects the worklist to build. ``round_budget`` is the number of
    scenes to plan for a train round; it is required in train mode and ignored in
    eval mode (eval is deficit-driven, not budgeted).
    """

    mode: str = Field(..., description='"eval" or "train".')
    round_budget: Optional[int] = Field(
        default=None,
        description="Scenes to plan this train round (train mode only).",
    )


class DraftModel(BaseModel):
    """One editable box exchanged with the browser (mirrors ``AnnotationDraft``).

    ``bbox_px`` is ``[x0, y0, x1, y1]`` in 512 px space. Eval mode leaves
    ``row_index`` / ``col_index`` / ``equation_idx`` at their server-sent values
    (the eval exporter ignores them); train mode sends the human-confirmed grid.
    """

    bbox_px: List[float]
    fine_label: str
    row_index: int = -1
    col_index: int = -1
    equation_idx: int = 0
    confidence: float = 0.0
    source: str = "manual"
    flagged: bool = False


class DetectRequest(BaseModel):
    """Body for ``POST /api/detect``.

    ``image_png`` is the canvas drawing as a base64 PNG, with or without a
    ``data:image/png;base64,`` prefix. The detector runs on the preprocessed
    512x512 image and the result is fused with the cursor's target.
    """

    image_png: str = Field(..., description="base64 PNG of the drawing.")


class SaveRequest(BaseModel):
    """Body for ``POST /api/save``.

    ``image_png`` is the finished drawing (base64 PNG). ``drafts`` are the
    human-confirmed boxes. ``stem`` is the output filename stem; when omitted the
    server allocates the next free ``re-<case>-NNN`` / ``rt-<case>-NNN``.
    ``equation_kind`` (eval) / ``equation_type`` (train) and ``scene_case`` /
    ``case`` default to the cursor target's values when omitted. ``error_kind``
    (eval only) tags how the human drew the scene wrong on purpose (an
    ``ERROR_KINDS`` member); it is omitted from the sidecar when ``None``.
    """

    image_png: str = Field(..., description="base64 PNG of the finished drawing.")
    drafts: List[DraftModel] = Field(default_factory=list)
    stem: Optional[str] = None
    equation_kind: Optional[str] = None
    equation_type: Optional[str] = None
    scene_case: Optional[str] = None
    case: Optional[str] = None
    completion_stage: Optional[str] = None
    error_kind: Optional[str] = None


# ---------------------------------------------------------------------------
# Serialisation helpers (frozen dataclass <-> JSON dict)
# ---------------------------------------------------------------------------

def _draft_to_dict(draft: AnnotationDraft) -> Dict[str, Any]:
    """Serialise an ``AnnotationDraft`` to a JSON object (list-typed bbox)."""
    return {
        "bbox_px": [float(v) for v in draft.bbox_px],
        "fine_label": draft.fine_label,
        "row_index": int(draft.row_index),
        "col_index": int(draft.col_index),
        "equation_idx": int(draft.equation_idx),
        "confidence": float(draft.confidence),
        "source": draft.source,
        "flagged": bool(draft.flagged),
    }


def _target_to_dict(target: TargetScene) -> Dict[str, Any]:
    """Serialise a ``TargetScene`` to a JSON object (symbols as list of dicts)."""
    return {
        "case": target.case,
        "equation_type": target.equation_type,
        "completion_stage": target.completion_stage,
        "seed": target.seed,
        "reference": target.reference,
        "symbols": [
            {
                "fine_label": s.fine_label,
                "glyph_key": s.glyph_key,
                "yolo_class": s.yolo_class,
                "row_index": s.row_index,
                "col_index": s.col_index,
                "equation_idx": s.equation_idx,
                "bbox": list(s.bbox),
            }
            for s in target.symbols
        ],
    }


def _draft_from_model(model: DraftModel) -> AnnotationDraft:
    """Build a frozen ``AnnotationDraft`` from the posted ``DraftModel``.

    The bbox is coerced to the four-tuple the dataclass expects; field validation
    proper (bounds, label vocabulary) is deferred to the exporter so a single
    place owns the fail-loud-with-path rules.
    """
    bbox = tuple(float(v) for v in model.bbox_px)
    return AnnotationDraft(
        bbox_px=bbox,  # type: ignore[arg-type]
        fine_label=model.fine_label,
        row_index=int(model.row_index),
        col_index=int(model.col_index),
        equation_idx=int(model.equation_idx),
        confidence=float(model.confidence),
        source=model.source,  # type: ignore[arg-type]
        flagged=bool(model.flagged),
    )


def _decode_png(image_png: str, where: str) -> np.ndarray:
    """Decode a base64 PNG (optionally a data-URL) to a numpy image array.

    Accepts a bare base64 string or a ``data:image/...;base64,<payload>`` URL.
    Raises ``ValueError`` naming ``where`` on a malformed payload so the route can
    return 422 rather than 500.
    """
    if not isinstance(image_png, str) or not image_png:
        raise ValueError(f"{where}: image_png must be a non-empty base64 string")
    payload = image_png
    if payload.startswith("data:"):
        comma = payload.find(",")
        if comma == -1:
            raise ValueError(f"{where}: malformed data-URL (no comma separator)")
        payload = payload[comma + 1 :]
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


# ---------------------------------------------------------------------------
# Cursor / target resolution
# ---------------------------------------------------------------------------

def _current_state(app: FastAPI, mode: Mode) -> SessionState:
    """Load the persisted session state for ``mode`` (project root from app)."""
    root: Path = app.state.project_root
    return state.load(mode, root)


def _cursor_item(session: SessionState):
    """Return the worklist item at the cursor, or ``None`` when the list is done."""
    if 0 <= session.cursor < len(session.worklist):
        return session.worklist[session.cursor]
    return None


def _generate_for_item(app: FastAPI, item) -> TargetScene:
    """Generate the target for a worklist item via the injectable hook.

    Maps the symbol-pool failure (``ValueError`` from ``targets`` / generation,
    raised when the pool is missing or unbuilt) onto an ``HTTPException(503)`` so
    ``/api/next`` returns the plan's "run validate first" contract rather than a
    500.
    """
    root: Path = app.state.project_root
    gen: Callable[..., TargetScene] = app.state.generate_target_fn
    try:
        return gen(item.case, item.seed, item.completion_stage, project_root=root)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "symbol pool unavailable: run `validate` to build the symbol pool "
                f"first ({exc})"
            ),
        ) from exc


# ---------------------------------------------------------------------------
# Save helpers (defaults from the cursor target, stem allocation)
# ---------------------------------------------------------------------------

def _next_stem(prefix: str, case: str, out_dir: Path) -> str:
    """Allocate the next free zero-padded ``<prefix>-<case>-NNN`` stem.

    Scans existing files in ``out_dir`` for the ``<prefix>-<case>-`` pattern and
    returns the smallest unused three-digit index, so stems never collide on
    resume (mirrors the exporter's collision intent; the exporter still sanitises
    the result).
    """
    used: set[int] = set()
    if out_dir.is_dir():
        token = f"{prefix}-{case}-"
        for child in out_dir.iterdir():
            name = child.name
            if name.startswith(token):
                tail = name[len(token) :].split(".", 1)[0]
                if tail.isdigit():
                    used.add(int(tail))
    idx = 0
    while idx in used:
        idx += 1
    return f"{prefix}-{case}-{idx:03d}"


def _normalise_eval_equation_kind(equation_type: str) -> str:
    """Map a target ``equation_type`` to a valid eval ``equation_kind``.

    The generator emits the short ``EquationKind`` value ("add" / "subtract" /
    "multiply" / "divide" / "ood_unknown"), but the eval sidecar vocabulary is the
    long ``EQUATION_KINDS`` set ("addition" / ...). The short->long map is the
    single source of truth in ``src/eval/harness.py`` (reused, not redefined). The
    structure-only OOD sentinel has no equation, so it maps to ``bare_digits`` (the
    only valid eval kind for a no-equation scene). An already-long value passes
    through unchanged so a client override or a long-form target still works.
    """
    if equation_type in EQUATION_KINDS:
        return equation_type
    mapped = _SHORT_TO_LONG_EQUATION_KIND.get(equation_type)
    if mapped is not None and mapped in EQUATION_KINDS:
        return mapped
    return _OOD_EVAL_EQUATION_KIND


def _normalise_train_equation_type(equation_type: str) -> str:
    """Map a target ``equation_type`` to a valid train GT ``equation_type``.

    Same short->long normalisation as the eval side (one SSoT map), but the train
    GT keeps the OOD sentinel as ``ood_unknown`` (the value the synthetic
    generator mirrors) rather than collapsing it to ``bare_digits``. The result is
    a member of the exporter's ``TRAIN_EQUATION_TYPES`` set, which the exporter
    re-validates before any write.
    """
    if equation_type in exporters.TRAIN_EQUATION_TYPES:
        return equation_type
    mapped = _SHORT_TO_LONG_EQUATION_KIND.get(equation_type)
    if mapped is not None and mapped in exporters.TRAIN_EQUATION_TYPES:
        return mapped
    return exporters.OOD_EQUATION_TYPE


def _derive_equation_type_for_item(item) -> str:
    """Cheaply derive a worklist item's short ``equation_type`` WITHOUT the pool.

    Save-time fallback for a client that did not echo the scene kind (the GUI
    always does, but a direct API caller may not). It samples only the layout for
    the item's ``(case, seed)`` via ``sample_layout_for_case`` -- which builds
    layout tokens, not glyph crops -- so it never loads the symbol pool and so can
    never raise the 503 the full target generator can. Returns the short
    ``EquationKind`` value; callers normalise it to the long eval/train forms.
    """
    import random  # noqa: PLC0415

    from ..generation.layouts import sample_layout_for_case  # noqa: PLC0415
    from ..generation.layouts_types import SceneCase  # noqa: PLC0415

    kind, _tokens = sample_layout_for_case(SceneCase(item.case), random.Random(item.seed))
    return kind.value


def _eval_equation_kind(target: TargetScene, override: Optional[str]) -> str:
    """Resolve the eval ``equation_kind`` from the client override or the target.

    The client may echo a confirmed kind (already a valid ``EQUATION_KINDS``
    member); otherwise the target's ``equation_type`` is normalised via
    :func:`_normalise_eval_equation_kind`. The exporter re-validates the result.
    """
    if override is not None:
        return override
    return _normalise_eval_equation_kind(target.equation_type)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(project_root: Path) -> FastAPI:
    """Build the set-maker FastAPI app bound to ``project_root``.

    The project root is stored on ``app.state`` and passed to every module
    boundary (none of them discover paths on their own). The two heavy
    collaborators are exposed as overridable hooks for tests:
    ``app.state.generate_target_fn`` (defaults to ``targets.generate_target``) and
    ``app.state.detect_fn`` (defaults to ``detect.detect_boxes``).
    """
    app = FastAPI(title="set-maker", docs_url=None, redoc_url=None)
    app.state.project_root = Path(project_root)
    app.state.generate_target_fn = targets.generate_target
    app.state.detect_fn = detect_mod.detect_boxes

    # Serve the WS-H frontend assets (app.css / app.js / vendor/konva.min.js)
    # referenced by index.html under ``/static``. Mounted only when the directory
    # exists so a backend-only checkout (no built frontend) still starts cleanly;
    # the ``GET /`` placeholder shell needs no static assets.
    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.exception_handler(ValueError)
    async def _value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
        """Map an uncaught ``ValueError`` to 422 with the offending message.

        The modules raise ``ValueError`` (path/field included) on bad input; this
        keeps a stray one from surfacing as an opaque 500. Routes that need a
        different status (503 for the missing pool) raise ``HTTPException``
        directly, which FastAPI handles before this fallback.
        """
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        """Serve the static app shell, or a placeholder until WS-H lands."""
        index_html = _STATIC_DIR / "index.html"
        if index_html.is_file():
            return HTMLResponse(index_html.read_text(encoding="utf-8"))
        return HTMLResponse(_PLACEHOLDER_HTML)

    @app.post("/api/mode")
    async def set_mode(body: ModeRequest) -> Dict[str, Any]:
        """Set the mode, build + persist the worklist, return initial progress.

        Eval builds a deficit-driven worklist; train builds a gap-driven one and,
        when no usable eval history exists, falls back to provisional weights and
        returns a non-fatal ``warning`` (the planner logs the same reason).
        """
        root: Path = app.state.project_root
        mode = body.mode
        if mode not in ("eval", "train"):
            raise HTTPException(
                status_code=422,
                detail=f"mode must be 'eval' or 'train', got {mode!r}",
            )

        warning: Optional[str] = None
        # Load served seeds so each fresh session skips seeds already delivered
        # to the annotator in prior sessions (FIX 2 seed persistence).
        _served = state.load_served_seeds(mode, root)  # type: ignore[arg-type]
        if mode == "eval":
            items = worklist.build_eval_worklist(project_root=root, served_seeds=_served)
            round_budget = None
        else:
            round_budget = body.round_budget
            if round_budget is None or round_budget <= 0:
                raise HTTPException(
                    status_code=422,
                    detail="train mode requires a positive round_budget",
                )
            history = root / "reports" / "eval" / "history.jsonl"
            if not history.is_file():
                warning = (
                    "no eval history found (reports/eval/history.jsonl); the train "
                    "worklist uses provisional quota weights. Run `eval` to route "
                    "training to measured model gaps."
                )
            items = worklist.build_train_worklist(round_budget, project_root=root, served_seeds=_served)

        session = SessionState(
            mode=mode,  # type: ignore[arg-type]
            worklist=list(items),
            cursor=0,
            round_budget=round_budget,
        )
        state.save(session, root)

        result: Dict[str, Any] = {
            "mode": mode,
            "planned": len(items),
            "progress": state.progress(session),
        }
        if warning is not None:
            result["warning"] = warning
        return result

    @app.get("/api/next")
    async def next_item(mode: str) -> Dict[str, Any]:
        """Return the cursor worklist item plus its generated target.

        ``mode`` is a query parameter ("eval" | "train"). When the worklist is
        exhausted, returns ``{"done": true, "item": null, "target": null}``.
        Raises 503 when the symbol pool is unavailable (run ``validate`` first).
        """
        if mode not in ("eval", "train"):
            raise HTTPException(
                status_code=422,
                detail=f"mode must be 'eval' or 'train', got {mode!r}",
            )
        session = _current_state(app, mode)  # type: ignore[arg-type]
        item = _cursor_item(session)
        if item is None:
            # Explicit completion signal: the worklist is exhausted for this mode.
            # ``complete`` is the canonical flag; ``done`` is kept for backward
            # compatibility with existing frontend checks.
            return {
                "done": True,
                "complete": True,
                "mode": mode,
                "item": None,
                "target": None,
                "cursor": session.cursor,
                "total": len(session.worklist),
                "progress": state.progress(session),
            }
        target = _generate_for_item(app, item)
        # Record the seed as served on every /api/next call (not only on save) so
        # three consecutive fresh sessions each see a distinct equation (FIX 2).
        root: Path = app.state.project_root
        state.mark_seed_served(mode, item.seed, root)  # type: ignore[arg-type]
        return {
            "done": False,
            "complete": False,
            "mode": mode,
            "cursor": session.cursor,
            "total": len(session.worklist),
            "item": {
                "case": item.case,
                "seed": item.seed,
                "completion_stage": item.completion_stage,
                "status": item.status,
                "directive": item.directive,
            },
            "target": _target_to_dict(target),
            # Resolved scene kinds the client echoes back on /api/save so the
            # save route never regenerates the target (a 503 risk + extra pool
            # work). Normalised here to the long forms each exporter accepts.
            "equation_kind": _normalise_eval_equation_kind(target.equation_type),
            "equation_type": _normalise_train_equation_type(target.equation_type),
        }

    @app.post("/api/detect")
    async def detect_route(body: DetectRequest, mode: str) -> Dict[str, Any]:
        """Detect boxes on the drawing and fuse them with the cursor target.

        ``mode`` is a query parameter. Decodes + preprocesses the drawing, runs the
        injectable detector, and matches against the cursor target's symbols. Zero
        detections yield an empty draft list plus a ``notice`` (never a 500). When
        the worklist is exhausted there is no target, so the route returns the
        matcher's unmatched-detection drafts (all flagged) with a notice.
        """
        if mode not in ("eval", "train"):
            raise HTTPException(
                status_code=422,
                detail=f"mode must be 'eval' or 'train', got {mode!r}",
            )
        from .matcher import match  # noqa: PLC0415  (keep scipy import lazy)

        gray = preprocess_for_pipeline(_decode_png(body.image_png, "detect"))
        detect_fn: Callable[..., List] = app.state.detect_fn
        detections = detect_fn(gray, project_root=app.state.project_root)

        session = _current_state(app, mode)  # type: ignore[arg-type]
        item = _cursor_item(session)
        if item is None:
            target = TargetScene(
                case="",
                equation_type="ood_unknown",
                completion_stage="full",
                seed=-1,
                reference="",
                symbols=[],
            )
        else:
            target = _generate_for_item(app, item)

        drafts = match(target, list(detections))
        result: Dict[str, Any] = {
            "drafts": [_draft_to_dict(d) for d in drafts],
            "detection_count": len(detections),
            "target_symbol_count": len(target.symbols),
        }
        if not detections:
            result["notice"] = "no boxes detected; draw, then detect, or add boxes manually."
        return result

    @app.post("/api/save")
    async def save_route(body: SaveRequest, mode: str) -> Dict[str, Any]:
        """Validate + write the finished scene, advance the cursor, return progress.

        ``mode`` is a query parameter. Eval writes a sidecar (bbox + fine label);
        train writes the synthetic GT + YOLO ``.txt``. Scene-level fields
        (``equation_kind`` / ``equation_type`` / ``scene_case`` / ``case`` /
        ``completion_stage``) default to the cursor target's values when the client
        omits them. Export validation failure surfaces as 422 (via the exporter's
        ``ValueError``); nothing is written on failure.
        """
        if mode not in ("eval", "train"):
            raise HTTPException(
                status_code=422,
                detail=f"mode must be 'eval' or 'train', got {mode!r}",
            )
        root: Path = app.state.project_root
        session = _current_state(app, mode)  # type: ignore[arg-type]
        item = _cursor_item(session)

        drafts = [_draft_from_model(m) for m in body.drafts]
        # Preprocess to 512x512 grayscale before passing to the exporter. The
        # exporter now enforces that any numpy array it receives is already in
        # 512px space (matching the bbox coordinate frame), so the route owns
        # the preprocessing step, mirroring how the detect route does it.
        png = preprocess_for_pipeline(_decode_png(body.image_png, "save"))

        # Resolve scene-level defaults from the cursor item (its case) and, only
        # when the client did not supply them, from a freshly generated target.
        case = body.case or body.scene_case or (item.case if item is not None else None)
        completion_stage = body.completion_stage or (
            item.completion_stage if item is not None else "full"
        )

        # Error-directive enforcement: when the current worklist item is tagged
        # directive="error", the human must have chosen a valid error_kind before
        # saving (the frontend blocks Save until the dropdown is non-empty, but we
        # enforce the contract server-side too for direct API callers). This fires
        # for both eval AND train error items — the mode gate was removed so train
        # error saves without a tag are rejected the same way eval ones are.
        item_directive = item.directive if item is not None else "normal"
        if item_directive == "error":
            if not body.error_kind or body.error_kind not in ERROR_KINDS:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "error_kind is required for error-directive items; "
                        f"must be one of {sorted(ERROR_KINDS)}, got {body.error_kind!r}."
                    ),
                )

        if mode == "eval":
            # Prefer the kind the client echoed from /api/next (no regen, no 503);
            # only when omitted derive it cheaply from the item layout (no pool).
            equation_kind = body.equation_kind
            if equation_kind is None and item is not None:
                equation_kind = _normalise_eval_equation_kind(
                    _derive_equation_type_for_item(item)
                )
            stem = body.stem or _next_stem("re", case or "scene", exporters.eval_dir(root))
            # error_kind is the deliberate as-drawn error tag chosen by the human;
            # the exporter validates it against ERROR_KINDS and omits it when None.
            written = exporters.export_eval(
                stem,
                png,
                drafts,
                {
                    "equation_kind": equation_kind,
                    "scene_case": case,
                    "error_kind": body.error_kind,
                },
                root,
            )
        else:
            # Prefer the type the client echoed from /api/next (already the long
            # TRAIN_EQUATION_TYPES form); only when omitted derive it cheaply from
            # the item layout (no pool) and normalise it. No target regen either
            # way, so the save path can never raise the pool-missing 503.
            equation_type = body.equation_type
            if equation_type is None and item is not None:
                equation_type = _normalise_train_equation_type(
                    _derive_equation_type_for_item(item)
                )
            stem = body.stem or _next_stem("rt", case or "scene", exporters.train_dir(root))
            written = exporters.export_train(
                stem,
                png,
                drafts,
                {
                    "equation_type": equation_type,
                    "completion_stage": completion_stage,
                    "case": case,
                    "split": "train",
                    "error_kind": body.error_kind,
                },
                root,
            )

        advanced = state.advance(session)
        state.save(advanced, root)

        # Derive the actually-written stem from the exporter's returned paths
        # rather than from the caller-supplied ``stem`` variable. The exporter
        # may have collision-bumped the stem (e.g. ``re-addition-001`` ->
        # ``re-addition-001-001``) when the client re-submitted a duplicate name,
        # so echoing the pre-bump ``stem`` would return a stale value that does
        # not match any file on disk.
        written_stem = written["png"].stem  # strips the ".png" suffix only

        return {
            "saved": True,
            "stem": written_stem,
            "written": {k: str(v) for k, v in written.items()},
            "progress": state.progress(advanced),
        }

    @app.get("/api/progress")
    async def progress_route(mode: str) -> Dict[str, Any]:
        """Return per-case + overall progress for ``mode`` (query parameter)."""
        if mode not in ("eval", "train"):
            raise HTTPException(
                status_code=422,
                detail=f"mode must be 'eval' or 'train', got {mode!r}",
            )
        session = _current_state(app, mode)  # type: ignore[arg-type]
        return state.progress(session)

    return app


# ---------------------------------------------------------------------------
# CLI entry (argument parsing kept stable; now launches uvicorn)
# ---------------------------------------------------------------------------

def parse_args(argv: "list[str] | None" = None):
    """Parse the ``setmaker`` subcommand arguments.

    Resolves ``--project-root`` to the directory containing ``data/`` and ``src/``
    when not supplied. Argument names are the stable contract documented on the
    subcommand: ``--host`` / ``--port`` / ``--project-root`` / ``--mode``.
    """
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="src setmaker",
        description="Local set-maker GUI for building real eval + train scenes.",
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
    parser.add_argument(
        "--mode",
        choices=("eval", "train"),
        default=None,
        help="Optional initial worklist mode (eval or train).",
    )
    args = parser.parse_args(argv)
    if args.project_root is None:
        args.project_root = resolve_project_root(Path(__file__))
    return args


def main() -> None:
    """Entry point for ``python -m src setmaker``: launch uvicorn on localhost.

    Builds the FastAPI app bound to the resolved project root and serves it with
    uvicorn on the requested host/port. ``--mode`` is accepted for parity with the
    argument contract; the active mode is set by the first ``POST /api/mode`` call
    from the UI, so it is logged here but not auto-applied.
    """
    import uvicorn  # noqa: PLC0415

    args = parse_args()
    app = create_app(args.project_root)
    if args.mode is not None:
        logger.info(
            "setmaker: initial --mode=%s noted; the UI sets the active mode via "
            "POST /api/mode.",
            args.mode,
        )
    logger.info("setmaker: serving http://%s:%d (project_root=%s)", args.host, args.port, args.project_root)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
