"""Data contracts for the set-maker tool.

These are the typed, immutable units exchanged between the set-maker modules
(``targets``, ``detect``, ``matcher``, ``worklist``, ``exporters``, ``state``)
and across the FastAPI boundary. They are deliberately frozen so a draft or a
worklist entry cannot be mutated in place; advancing state is done by building a
new instance via :func:`dataclasses.replace`.

Field shapes mirror the existing synthetic ground-truth element shape produced by
``src/generation/synth_pool.py`` (see ``gt_tokens`` dicts) so the same downstream
matching and export logic can read a generated target and a hand-drawn draft
without redefining the symbol contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

# Directive on a worklist item: "normal" means draw the scene as given; "error"
# means draw the same target equation but WITH a deliberate mistake and tag it
# with an error_kind on save.
Directive = Literal["normal", "error"]

# Bounding box in 512x512 preprocessed-canvas pixel space: [x0, y0, x1, y1].
BBox = Tuple[float, float, float, float]

# Where a draft field came from. "matched" = target label copied onto a detection;
# "detected" = a detection with no matching target (no label yet); "manual" = a
# target with no matching detection, surfaced at the target bbox for the human.
DraftSource = Literal["matched", "detected", "manual"]

# Worklist mode selector. Eval builds the held-out real test set; train builds
# real fine-tune data routed by model gaps.
Mode = Literal["eval", "train"]

# Per-item lifecycle status.
ItemStatus = Literal["pending", "done"]


@dataclass(frozen=True)
class TargetSymbol:
    """One symbol in a generated target scene.

    Exactly the synthetic ground-truth ``symbols[]`` element shape (reused, not
    redefined): the seven keys written per token in ``synth_pool.py``. ``bbox`` is
    ``[x0, y0, x1, y1]`` in 512 px space.
    """

    fine_label: str
    glyph_key: str
    yolo_class: str
    row_index: int
    col_index: int
    equation_idx: int
    bbox: BBox


@dataclass(frozen=True)
class TargetScene:
    """A generated target equation for the annotator to redraw.

    ``reference`` is a clean typeset equation string (plus a light layout hint),
    not a synthetic handwriting render: the human draws their own hand and the
    target only supplies the symbol content and grid structure for matching.
    ``completion_stage`` is pinned by the caller (never sampled randomly here) so
    partial-exercise quotas are satisfiable. A scene holds a single equation, so
    every contained symbol carries ``equation_idx == 0``.
    """

    case: str
    equation_type: str
    completion_stage: str
    seed: int
    reference: str
    symbols: List[TargetSymbol]


@dataclass(frozen=True)
class AnnotationDraft:
    """The editable per-box unit exchanged with the browser.

    Eval mode reads only ``bbox_px`` and ``fine_label`` (row/col/equation_idx are
    inferred geometrically at eval time); train mode reads the full draft. Same
    type for both modes; the frontend shows or hides fields by mode and each
    exporter consumes only what it needs.

    ``source`` records provenance ("matched"|"detected"|"manual"). ``flagged`` is
    true when the box is low-confidence, unmatched, or count-mismatched, so the
    human reviews it before saving.
    """

    bbox_px: BBox
    fine_label: str
    row_index: int
    col_index: int
    equation_idx: int
    confidence: float
    source: DraftSource
    flagged: bool


@dataclass(frozen=True)
class WorklistItem:
    """One planned scene to draw: a (case, seed) with a pinned completion stage.

    ``completion_stage`` is pinned per item (not random) so the partial-scene
    quota is satisfiable. ``status`` advances from "pending" to "done" by building
    a new item via :func:`dataclasses.replace`. ``directive`` is "normal" for
    standard scenes and "error" for deliberately-wrong scenes where the human draws
    the equation with a mistake and must supply an ``error_kind`` tag on save.
    """

    case: str
    seed: int
    completion_stage: str
    status: ItemStatus = "pending"
    directive: Directive = "normal"


@dataclass(frozen=True)
class SessionState:
    """Persisted worklist + cursor for one mode (JSON on disk).

    Saved to ``data/setmaker/state/<mode>.json`` on every save so resume re-reads
    the stored worklist and cursor. ``round_budget`` is the train-mode round size
    (``None`` for eval). Timestamps are ISO-8601 UTC strings.
    """

    mode: Mode
    worklist: List[WorklistItem] = field(default_factory=list)
    cursor: int = 0
    round_budget: Optional[int] = None
    created_utc: str = ""
    last_saved_utc: str = ""
