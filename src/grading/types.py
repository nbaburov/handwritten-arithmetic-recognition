"""Contract dataclasses for the grading engine.

These frozen dataclasses mirror the captured real grading engine JSON exactly (see
``docs/research/engine-api-capture/`` and the golden copies under
``tests/grading/fixtures/``). Every type carries ``from_api`` /
``to_api`` (de)serialization helpers that parse the captured shapes and emit
them back byte-aligned, so the mock and the live engine client share one shape and
the contract tests can prove it by round-tripping each fixture.

The API uses camelCase keys (``sessionId``, ``symbolIdList``, ``messageType``);
the Python attributes use snake_case. The helpers are the only translation
boundary, so the rest of the package speaks Python only.

This module depends on nothing else in the package; it owns the contract and the
JSON translation, not HTTP, grading, or pixel mapping.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# The single grading engine interaction id is embedded as a key in the create
# response (``interactions: {refId: {...}}``) and the view-model is carried as a
# JSON blob inside an HTML ``<script type="grading/init-data">`` tag.
_INIT_DATA_RE = re.compile(
    r'<script type="grading/init-data">(?P<body>.*?)</script>',
    re.DOTALL,
)


# Semi-public by design: imported by ``src.grading.derive`` and accessed in
# tests. The leading underscore signals "not part of the public HTTP contract"
# but the function is deliberately reachable from sibling modules that build on
# the same HTML-blob parsing logic as ``SessionCreated.from_api``.
def _extract_init_data(html: str) -> dict[str, Any]:
    """Pull the parsed init-data JSON out of a create/solution ``html`` blob.

    Returns an empty dict when no init-data script is present.
    """
    match = _INIT_DATA_RE.search(html)
    if match is None:
        return {}
    return json.loads(match.group("body"))


@dataclass(frozen=True)
class GridToken:
    """One symbol on the grading engine W x H integer grid.

    Exactly the ``input[]`` / ``tokens[]`` element shape: a stable ``id``, the
    character ``c`` it carries, integer grid coords ``x`` (column) / ``y`` (row),
    and ``tags`` such as ``GENERATED`` / ``AUTOSHOW`` / ``FIXED``. The recognizer
    feeds these to ``evaluate``; grading engine echoes them back in element
    ``symbolIdList`` entries.

    ``type`` is the cell kind. It defaults to ``SYMBOL`` (a plain digit/operator).
    Borrow / replacement cells in column subtraction must be ``REPLACE`` or
    grading engine grades them as wrong (verified live: a borrow sent as ``SYMBOL``
    returns ``BorrowIncorrectFeedback`` even with the right value). Only a
    non-``SYMBOL`` type is emitted on the wire so plain tokens stay byte-aligned
    with the captured request fixtures.
    """

    id: str
    c: str
    x: int
    y: int
    tags: list[str] = field(default_factory=list)
    type: str = "SYMBOL"

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "GridToken":
        return cls(
            id=str(data["id"]),
            c=str(data["c"]),
            x=int(data["x"]),
            y=int(data["y"]),
            tags=[str(t) for t in data.get("tags", [])],
            type=str(data.get("type", "SYMBOL")),
        )

    def to_api(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "c": self.c, "x": self.x, "y": self.y, "tags": list(self.tags)}
        if self.type != "SYMBOL":
            out["type"] = self.type
        return out


@dataclass(frozen=True)
class ExerciseSpec:
    """A single bank exercise rendered to an inline grading engine ``exerciseSpec``.

    ``template_name`` is one of the bank templates (SUM_SHORT, SUBTRACT_SHORT,
    PRODUCT_SHORT, DIVISION_SHORT); ``template_args`` are the operands; the
    ``attributes`` and ``audience`` map onto the create call's ``attributes`` and
    audience selector. ``cms_exercise_id``, when set, makes ``create_session``
    start from that grading engine CMS exercise UUID instead of the inline spec
    (``the live engine`` only grades CMS exercises). This is our own bank
    shape, not a captured response, so ``from_api`` / ``to_api`` round-trip.
    """

    exercise_id: str
    title: str
    prompt: str
    template_name: str
    template_args: list[str]
    attributes: dict[str, str] = field(default_factory=dict)
    audience: str = "uk_KS2"
    cms_exercise_id: str = ""

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "ExerciseSpec":
        return cls(
            exercise_id=str(data["exerciseId"]),
            title=str(data["title"]),
            prompt=str(data["prompt"]),
            template_name=str(data["templateName"]),
            template_args=[str(a) for a in data.get("templateArgs", [])],
            attributes={str(k): str(v) for k, v in data.get("attributes", {}).items()},
            audience=str(data.get("audience", "uk_KS2")),
            cms_exercise_id=str(data.get("cmsExerciseId", "")),
        )

    def to_api(self) -> dict[str, Any]:
        return {
            "exerciseId": self.exercise_id,
            "title": self.title,
            "prompt": self.prompt,
            "templateName": self.template_name,
            "templateArgs": list(self.template_args),
            "attributes": dict(self.attributes),
            "audience": self.audience,
            "cmsExerciseId": self.cms_exercise_id,
        }


@dataclass(frozen=True)
class SessionCreated:
    """Parsed result of ``session/create``.

    ``ref_id`` is the single ARITHMETIC interaction id (the key of the
    ``interactions`` map); ``view_model`` is the parsed init-data carrying the
    grid ``width`` / ``height``, the area map, and the scaffold ``tokens[]``.
    ``raw`` preserves the original session envelope so ``to_api`` reproduces the
    captured create response exactly (including the ``html`` blob).
    """

    session_id: str
    ref_id: str
    marks_total: int
    view_model: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: list[Any] | dict[str, Any]) -> "SessionCreated":
        # The create route returns a list of envelopes; the demo uses a single session per create.
        envelope = data[0] if isinstance(data, list) else data
        session = envelope["sessions"][0]
        interactions: dict[str, Any] = session.get("interactions", {})
        ref_id = next(iter(interactions), "")
        view_model = _extract_init_data(str(session.get("html", "")))
        return cls(
            session_id=str(session["sessionId"]),
            ref_id=str(ref_id),
            marks_total=int(session.get("marksTotal", 0)),
            view_model=view_model,
            raw=session,
        )

    def to_api(self) -> list[dict[str, Any]]:
        if self.raw:
            return [{"success": True, "sessions": [dict(self.raw)]}]
        session = {
            "sessionId": self.session_id,
            "marksTotal": self.marks_total,
            "interactions": {self.ref_id: {}},
        }
        return [{"success": True, "sessions": [session]}]


@dataclass(frozen=True)
class GridRect:
    """A rectangular grid region in grading engine ``targetPositions`` shape.

    Coords are integer grid cells; ``top`` >= ``bottom`` because engine grid y is
    bottom-origin and increases upward.
    """

    left: int
    top: int
    bottom: int
    right: int

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "GridRect":
        return cls(
            left=int(data["left"]),
            top=int(data["top"]),
            bottom=int(data["bottom"]),
            right=int(data["right"]),
        )

    def to_api(self) -> dict[str, Any]:
        return {"left": self.left, "top": self.top, "bottom": self.bottom, "right": self.right}


@dataclass(frozen=True)
class EvalElement:
    """One graded cell/area in an ``EvalResult.elements`` list.

    ``position`` is the (x, y) grid origin; ``status`` is ``OK`` / ``MISSING`` /
    etc.; ``symbol_id_list`` echoes the submitted token ids that matched (absent
    for a MISSING element). ``l`` is the grading engine's likelihood score.
    """

    template_type: str
    template_id: str
    attribute_map: dict[str, Any]
    position: tuple[int, int]
    width: int
    height: int
    status: str
    symbol_id_list: list[str] = field(default_factory=list)
    l: float = 0.0

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "EvalElement":
        pos = data.get("position", {})
        return cls(
            template_type=str(data["templateType"]),
            template_id=str(data["templateId"]),
            attribute_map=dict(data.get("attributeMap", {})),
            position=(int(pos["x"]), int(pos["y"])),
            width=int(data["width"]),
            height=int(data["height"]),
            status=str(data["status"]),
            symbol_id_list=[str(s) for s in data.get("symbolIdList", [])],
            l=float(data.get("l", 0.0)),
        )

    def to_api(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "templateType": self.template_type,
            "templateId": self.template_id,
            "attributeMap": dict(self.attribute_map),
            "position": {"x": self.position[0], "y": self.position[1]},
            "width": self.width,
            "height": self.height,
            "status": self.status,
        }
        # engine omits ``symbolIdList`` on MISSING elements; preserve that asymmetry
        # so the round-trip is byte-aligned to the captured fixture.
        if self.symbol_id_list:
            out["symbolIdList"] = list(self.symbol_id_list)
        out["l"] = self.l
        return out


@dataclass(frozen=True)
class EvalHint:
    """The single positioned next-step hint in an ``EvalResult``.

    ``target_positions`` is the cell(s) the hint points at; ``message_type`` (for
    example ``HorizontalSum``) plus ``message_args`` feed the frontend catalog so
    mock and real hints render identically.
    """

    source_token_ids: list[str] = field(default_factory=list)
    target_positions: list[GridRect] = field(default_factory=list)
    message_type: str = ""
    message_args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "EvalHint":
        return cls(
            source_token_ids=[str(s) for s in data.get("sourceTokenIds", [])],
            target_positions=[GridRect.from_api(p) for p in data.get("targetPositions", [])],
            message_type=str(data.get("messageType", "")),
            message_args=dict(data.get("messageArgs", {})),
        )

    def to_api(self) -> dict[str, Any]:
        return {
            "sourceTokenIds": list(self.source_token_ids),
            "targetPositions": [p.to_api() for p in self.target_positions],
            "messageType": self.message_type,
            "messageArgs": dict(self.message_args),
        }


@dataclass(frozen=True)
class EvalFeedback:
    """A wrong/extra-cell feedback entry in ``EvalResult.feedback``.

    Shares ``messageType`` / ``messageArgs`` with the hint and carries the
    offending ``targetPositions`` / ``symbolIdList``. Unknown extra keys are
    preserved in ``extra`` so a richer real wrong-answer response round-trips
    without loss.
    """

    message_type: str = ""
    message_args: dict[str, Any] = field(default_factory=dict)
    target_positions: list[GridRect] = field(default_factory=list)
    symbol_id_list: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    _KNOWN = ("messageType", "messageArgs", "targetPositions", "symbolIdList")

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "EvalFeedback":
        extra = {k: v for k, v in data.items() if k not in cls._KNOWN}
        return cls(
            message_type=str(data.get("messageType", "")),
            message_args=dict(data.get("messageArgs", {})),
            target_positions=[GridRect.from_api(p) for p in data.get("targetPositions", [])],
            symbol_id_list=[str(s) for s in data.get("symbolIdList", [])],
            extra=extra,
        )

    def to_api(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "messageType": self.message_type,
            "messageArgs": dict(self.message_args),
            "targetPositions": [p.to_api() for p in self.target_positions],
            "symbolIdList": list(self.symbol_id_list),
        }
        out.update(self.extra)
        return out


@dataclass(frozen=True)
class EvalResult:
    """Parsed ``session/event`` evaluate result.

    Mirrors the captured ``[{result:{...}, updateSession}]`` envelope's
    ``result`` object: graded ``elements``, ``unmatched`` tokens, wrong/extra
    ``feedback``, the single positioned ``hint`` (or ``None``), and monotonic
    ``progress`` in [0, 1]. ``l`` is the grading engine's overall likelihood.
    """

    elements: list[EvalElement] = field(default_factory=list)
    unmatched: list[Any] = field(default_factory=list)
    feedback: list[EvalFeedback] = field(default_factory=list)
    hint: EvalHint | None = None
    progress: float = 0.0
    l: float = 0.0

    @classmethod
    def from_api(cls, data: list[Any] | dict[str, Any]) -> "EvalResult":
        # The event route returns a list of update envelopes; the demo reads the first.
        # result is null when an event produces no grade (e.g. inline specs on prod).
        if isinstance(data, list):
            envelope = data[0] if data else {}
        else:
            envelope = data or {}
        result = envelope.get("result", envelope) or {}
        raw_hint = result.get("hint")
        return cls(
            elements=[EvalElement.from_api(e) for e in result.get("elements", [])],
            unmatched=list(result.get("unmatched", [])),
            feedback=[EvalFeedback.from_api(f) for f in result.get("feedback", [])],
            hint=EvalHint.from_api(raw_hint) if raw_hint else None,
            progress=float(result.get("progress", 0.0)),
            l=float(result.get("l", 0.0)),
        )

    def to_api(self) -> list[dict[str, Any]]:
        result: dict[str, Any] = {
            "elements": [e.to_api() for e in self.elements],
            "unmatched": list(self.unmatched),
            "feedback": [f.to_api() for f in self.feedback],
            "hint": self.hint.to_api() if self.hint is not None else None,
            "progress": self.progress,
            "l": self.l,
        }
        return [{"result": result, "updateSession": True}]


@dataclass(frozen=True)
class ScoringInfo:
    """Parsed ``session/info`` scoring block.

    ``penalties`` is the raw penalty dict (``marksPenalty`` / ``hintsRequested``
    / ``mathErrors``); the field order in the captured fixture is not stable, so
    the round-trip compares it as a dict.
    """

    finished: bool
    marks_total: int
    marks_earned: int
    penalties: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "ScoringInfo":
        scoring = data.get("scoring", data)
        return cls(
            finished=bool(scoring["finished"]),
            marks_total=int(scoring["marksTotal"]),
            marks_earned=int(scoring["marksEarned"]),
            penalties=dict(scoring.get("penalties", {})),
        )

    def to_api(self) -> dict[str, Any]:
        return {
            "finished": self.finished,
            "marksTotal": self.marks_total,
            "marksEarned": self.marks_earned,
            "penalties": dict(self.penalties),
        }


# --- src/demo response contracts -------------------------------------------------
# These are our own API shapes between the FastAPI backend and the Konva
# frontend, not captured grading engine responses. They live here so the whole
# contract surface is in one module. ``pixel_rect`` is in 512-space.


@dataclass(frozen=True)
class PopupSpec:
    """One on-canvas popup the frontend renders over a drawn cell.

    ``status`` is one of {ok, missing, error, hint}; ``message`` is the
    human-rendered text (from ``messageType`` + ``messageArgs`` via the catalog);
    ``pixel_rect`` is ``(left, top, right, bottom)`` in 512-space; ``kind`` lets
    the frontend pick the bubble vs marker treatment.

    ``anchor`` indicates how ``pixel_rect`` was derived:
    * ``"glyph"``: union of the matched tokens' real 512-space bboxes (model-driven;
      preferred when the element has at least one matched token).
    * ``"cell"``: a grid cell projected via ``grid_rect_to_pixels`` (fallback for
      MISSING elements and hints whose ``symbol_ids`` is empty).
    """

    pixel_rect: tuple[float, float, float, float]
    status: str
    message: str
    kind: str
    symbol_ids: list[str] = field(default_factory=list)
    anchor: str = "cell"

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "PopupSpec":
        rect = data["pixelRect"]
        return cls(
            pixel_rect=(float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])),
            status=str(data["status"]),
            message=str(data["message"]),
            kind=str(data["kind"]),
            symbol_ids=[str(s) for s in data.get("symbolIds", [])],
            anchor=str(data.get("anchor", "cell")),
        )

    def to_api(self) -> dict[str, Any]:
        return {
            "pixelRect": [self.pixel_rect[0], self.pixel_rect[1], self.pixel_rect[2], self.pixel_rect[3]],
            "status": self.status,
            "message": self.message,
            "kind": self.kind,
            "symbolIds": list(self.symbol_ids),
            "anchor": self.anchor,
        }


@dataclass(frozen=True)
class EvaluateResponse:
    """The full ``POST /api/evaluate`` response body.

    ``recognized`` carries the read-as-drawn equation kind, the per-token list
    (label / glyph / bbox), and a short summary; ``popups`` are positioned over
    the canvas; ``timings`` are the inference sub-timings in ms; ``opencv_used``
    flags the CV-fusion fallback; ``marks`` is the running score.
    """

    recognized: dict[str, Any]
    popups: list[PopupSpec]
    timings: dict[str, float]
    opencv_used: bool
    progress: float
    marks: dict[str, Any]

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "EvaluateResponse":
        return cls(
            recognized=dict(data["recognized"]),
            popups=[PopupSpec.from_api(p) for p in data.get("popups", [])],
            timings={str(k): float(v) for k, v in data.get("timings", {}).items()},
            opencv_used=bool(data["opencv_used"]),
            progress=float(data["progress"]),
            marks=dict(data["marks"]),
        )

    def to_api(self) -> dict[str, Any]:
        return {
            "recognized": dict(self.recognized),
            "popups": [p.to_api() for p in self.popups],
            "timings": dict(self.timings),
            "opencv_used": self.opencv_used,
            "progress": self.progress,
            "marks": dict(self.marks),
        }
