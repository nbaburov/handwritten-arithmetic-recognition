"""Target <-> detection alignment for the set-maker (WS-D).

Owns one job: fuse a known :class:`~src.setmaker.types.TargetScene` (the symbol
content and grid structure produced a priori by ``targets.py``) with the
geometry-only :class:`~src.parsing.detection.Detection` list produced by
``detect.py``, emitting the prefilled :class:`~src.setmaker.types.AnnotationDraft`
list the browser edits.

Why a global optimal assignment, not greedy matching: on dense scenes (long
division, multi-row multiplication) detected boxes sit close together, and a
greedy nearest-target rule swaps adjacent symbols (the digit ``3`` claims the
``5`` slot because it is marginally closer, then ``5`` takes whatever is left).
Those swaps are silent label noise that the human rubber-stamps. The Hungarian
algorithm (``scipy.optimize.linear_sum_assignment``) instead minimises the total
centroid distance across the whole scene, so it is robust to local ties (plan
risk R1).

Matching logic:

* Build a cost matrix of Euclidean centroid distances between every detection
  and every target symbol, solve it optimally one-to-one, then *reject* any
  chosen pair whose distance exceeds :data:`DISTANCE_CEILING_PX` (Hungarian
  always returns a full assignment on the square sub-problem, so over-ceiling
  pairs are pruned after the solve).
* A surviving pair becomes a ``"matched"`` draft: ``fine_label``, ``row_index``,
  ``col_index`` and ``equation_idx`` are copied from the *target* (the detection
  supplies geometry only), ``bbox_px`` is the detection box, ``confidence`` is the
  detection confidence, and ``flagged`` is true when the match is shaky, i.e. the
  pair distance is above the softer :data:`DISTANCE_FLAG_PX` review bound OR the
  detection confidence is below :data:`CONFIDENCE_FLOOR`.
* A detection with no surviving target becomes a ``"detected"`` draft: it carries
  no label yet (:data:`UNLABELLED` ``fine_label`` and :data:`UNASSIGNED_INDEX``
  row/col/equation_idx), keeps its own box and confidence, and is always flagged
  so the human assigns a label.
* A target with no surviving detection is **silently dropped** -- no ``"manual"``
  ghost box is emitted. The annotator works only with boxes backed by actual
  detections; phantom boxes at expected target positions create confusion when the
  human has drawn only a partial scene or the detector missed symbols. The human
  can add missing boxes manually via the ``n`` key or the Add box button.

This module does NOT own writing files or any UI concern; it is a pure function
over the two contracts.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..parsing.detection import Detection
from .types import AnnotationDraft, TargetScene, TargetSymbol

__all__ = [
    "DISTANCE_CEILING_PX",
    "DISTANCE_FLAG_PX",
    "CONFIDENCE_FLOOR",
    "UNLABELLED",
    "UNASSIGNED_INDEX",
    "match",
]

# Hard rejection radius. A detection and a target whose centroids are farther
# apart than this (in 512 px space) are never the same symbol, so even if the
# Hungarian solve pairs them (it must return a full assignment on the square
# sub-problem) the pair is pruned and both fall through to the unmatched paths.
# 96 px is three quarters of a 128 px ``base_cell_px``: comfortably tolerant of
# a hand-drawn glyph sitting off the typeset target centre, but well short of the
# next grid cell so a neighbour can never be claimed.
DISTANCE_CEILING_PX: float = 96.0

# Soft review radius. A surviving (kept) match whose distance is above this but
# below the ceiling is plausibly correct yet far enough off-centre to deserve a
# human glance, so it is flagged without being rejected. Half the hard ceiling.
DISTANCE_FLAG_PX: float = 48.0

# Detection-confidence review floor. A kept match backed by a low-confidence
# detection is flagged for review (the label copied from the target is probably
# right, but the box geometry the detector found may be poor). Mirrors the
# set-maker prefill conf floor: real handwriting is systematically
# under-confident, so this only flags the genuinely faint boxes.
CONFIDENCE_FLOOR: float = 0.25

# An unmatched detection has no label yet; the human assigns one before save.
UNLABELLED: str = ""

# Sentinel row/col/equation_idx for a draft with no grid placement yet (an
# unmatched detection). Negative so it can never collide with a real 0-based
# index and is obvious in the UI / exporter as "needs a value".
UNASSIGNED_INDEX: int = -1


def _centroid(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Return the ``(cx, cy)`` centre of an ``[x0, y0, x1, y1]`` box."""
    x0, y0, x1, y1 = box
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _distance(
    detection: Detection, symbol: TargetSymbol
) -> float:
    """Euclidean centroid distance between a detection and a target symbol.

    ``Detection`` exposes ``cx``/``cy`` properties; the target symbol bbox is the
    typeset position the human is redrawing. Distance in 512 px space drives both
    the assignment cost and the flag/ceiling thresholds.
    """
    sx, sy = _centroid(symbol.bbox)
    return math.hypot(detection.cx - sx, detection.cy - sy)


def _cost_matrix(
    detections: Sequence[Detection], symbols: Sequence[TargetSymbol]
) -> np.ndarray:
    """Build the ``(n_detections, n_symbols)`` centroid-distance cost matrix."""
    # Both sequences must be non-empty here: the degenerate fast paths in
    # ``match`` handle the empty-detections and empty-symbols cases before
    # this function is ever called. An assertion rather than a ValueError
    # makes the invariant explicit and surfaces any caller regression loudly
    # (a zero-dimension matrix would silently give an empty assignment).
    assert len(detections) > 0 and len(symbols) > 0, (
        f"_cost_matrix requires non-empty detections and symbols; "
        f"got {len(detections)} detections and {len(symbols)} symbols"
    )
    cost = np.empty((len(detections), len(symbols)), dtype=np.float64)
    for i, det in enumerate(detections):
        for j, sym in enumerate(symbols):
            cost[i, j] = _distance(det, sym)
    return cost


def _matched_draft(detection: Detection, symbol: TargetSymbol, distance: float) -> AnnotationDraft:
    """Build a ``"matched"`` draft: target labels onto the detection geometry.

    The label fields (``fine_label``/``row_index``/``col_index``/``equation_idx``)
    come from the target because the target is the a-priori ground truth; the box
    and confidence come from the detection because it carries the real pixel
    geometry of what the human drew. Flagged when the pair is far apart (above the
    soft review bound) or the detection is low-confidence.
    """
    flagged = bool(distance > DISTANCE_FLAG_PX or detection.confidence < CONFIDENCE_FLOOR)
    return AnnotationDraft(
        bbox_px=(detection.x0, detection.y0, detection.x1, detection.y1),
        fine_label=symbol.fine_label,
        row_index=symbol.row_index,
        col_index=symbol.col_index,
        equation_idx=symbol.equation_idx,
        confidence=float(detection.confidence),
        source="matched",
        flagged=flagged,
    )


def _detected_draft(detection: Detection) -> AnnotationDraft:
    """Build a ``"detected"`` draft: a box with no target, hence no label yet.

    Always flagged: the human must assign a label and grid position before this
    box is trustworthy. Keeps the detection box and confidence.
    """
    return AnnotationDraft(
        bbox_px=(detection.x0, detection.y0, detection.x1, detection.y1),
        fine_label=UNLABELLED,
        row_index=UNASSIGNED_INDEX,
        col_index=UNASSIGNED_INDEX,
        equation_idx=UNASSIGNED_INDEX,
        confidence=float(detection.confidence),
        source="detected",
        flagged=True,
    )


def _manual_draft(symbol: TargetSymbol) -> AnnotationDraft:
    """Build a ``"manual"`` draft: a target the detector never found.

    Surfaced at the target's own bbox with the target labels so the human only
    has to reposition/confirm it (the symbol content and grid placement are known
    from the target). Confidence is 0.0 because no detection backs it. Always
    flagged.
    """
    return AnnotationDraft(
        bbox_px=symbol.bbox,
        fine_label=symbol.fine_label,
        row_index=symbol.row_index,
        col_index=symbol.col_index,
        equation_idx=symbol.equation_idx,
        confidence=0.0,
        source="manual",
        flagged=True,
    )


def match(target: TargetScene, detections: List[Detection]) -> List[AnnotationDraft]:
    """Fuse a known target with detected geometry into prefilled drafts.

    Solves the global optimal one-to-one assignment between ``detections`` and
    ``target.symbols`` over a centroid-distance cost matrix
    (``scipy.optimize.linear_sum_assignment``), prunes any chosen pair beyond
    :data:`DISTANCE_CEILING_PX`, then emits one :class:`AnnotationDraft` per
    detection only:

    * kept pair  -> ``"matched"`` (target labels on detection geometry);
    * leftover detection -> ``"detected"`` (flagged, no label);
    * leftover target    -> **no draft emitted** (ghost boxes suppressed).

    Unmatched targets are silently dropped. The annotator canvas shows only boxes
    backed by real detections, preventing phantom boxes when the human has drawn
    fewer symbols than the full target. Missing symbols can be added manually via
    the ``n`` key or the Add box button.

    Ordering is stable and deterministic: matched/detected drafts follow the
    input ``detections`` order. Degenerate inputs are handled without a solve:
    no detections yields ``[]`` regardless of the target; no target symbols
    yields a detected draft per detection; both empty yields ``[]``.

    Args:
        target: The generated target scene whose symbols supply labels + grid.
        detections: Geometry-only detections in 512 px space (may be empty).

    Returns:
        A ``List[AnnotationDraft]``; one per detection (matched or detected).
        Never ``None``.
    """
    symbols = target.symbols

    # Degenerate fast paths (also keep linear_sum_assignment off empty matrices).
    if not detections:
        return []
    if not symbols:
        return [_detected_draft(det) for det in detections]

    cost = _cost_matrix(detections, symbols)
    det_rows, sym_cols = linear_sum_assignment(cost)

    # Partition the optimal assignment into kept matches (within the ceiling) and
    # rejected pairs (over the ceiling). ``linear_sum_assignment`` returns at most
    # min(n_det, n_sym) pairs, so the leftovers below cover the size mismatch and
    # the over-ceiling rejections together.
    matched_det_to_sym: dict = {}
    for det_i, sym_j in zip(det_rows.tolist(), sym_cols.tolist()):
        if cost[det_i, sym_j] <= DISTANCE_CEILING_PX:
            matched_det_to_sym[det_i] = sym_j

    drafts: List[AnnotationDraft] = []
    # Walk detections in input order so matched + detected drafts keep a
    # stable, caller-predictable order. Unmatched targets produce no draft.
    for det_i, det in enumerate(detections):
        sym_j = matched_det_to_sym.get(det_i)
        if sym_j is None:
            drafts.append(_detected_draft(det))
        else:
            drafts.append(_matched_draft(det, symbols[sym_j], cost[det_i, sym_j]))

    return drafts
