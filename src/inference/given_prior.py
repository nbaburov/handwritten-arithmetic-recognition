"""Shared dataclasses for the given-equation prior feature (Foundation 1 / Workstream A).

These types carry a deterministic given-equation node (cell bbox, labels, font-rendered
tile) and the result of merging given nodes with child YOLO detections.

Consumers import from this module; the grading-specific builder lives separately
in src/grading/given_prior_builder.py and depends on these types.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    pass

from ..parsing.detection import Detection
from ..parsing.assemble import NodePrediction


@dataclass(frozen=True)
class GivenNode:
    """A single given/pre-printed equation token in 512-space integer coordinates.

    All four coordinate fields are integers derived deterministically from the
    grading engine grid calibration so they can serve as stable dict keys for the
    known_fine label map.

    Attributes:
        x0: Left pixel coordinate in the 512x512 canvas (inclusive).
        y0: Top pixel coordinate in the 512x512 canvas (inclusive).
        x1: Right pixel coordinate in the 512x512 canvas (exclusive).
        y1: Bottom pixel coordinate in the 512x512 canvas (exclusive).
        coarse_label: YOLO ontology class (e.g. "digit_main", "operator").
        fine_label: GNN fine label (e.g. "main_3", "op_plus") -- known truth.
        tile: (28, 28) uint8 grayscale font-rendered crop for compositing.
    """

    x0: int
    y0: int
    x1: int
    y1: int
    coarse_label: str
    fine_label: str
    tile: np.ndarray  # shape (28, 28), dtype uint8, grayscale


@dataclass(frozen=True)
class MergeResult:
    """Result of merging given-equation nodes with child YOLO detections.

    Attributes:
        detections: Merged detection list -- child detections (given-flagged where
            the child drew over a given cell) plus synthetic Detection(given=True)
            entries for unmatched given nodes. Length is capped so that
            child + given <= max_nodes (operators dropped last; child never evicted).
        compose_nodes: The GivenNode entries whose tiles must be painted onto the
            scene image before build_graph crops them. Only unmatched given nodes
            (those without a child detection at the same cell) appear here.
        known_fine: Maps each given-node bbox tuple (x0, y0, x1, y1) to its known
            fine label. Used by pin_given_labels to override fine_label on every
            NodePrediction whose bbox key matches.
    """

    detections: list[Detection]
    compose_nodes: list[GivenNode]
    known_fine: dict[tuple[int, int, int, int], str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Coarse labels that are considered "operator" for overflow-cap drop ordering.
# Operators are kept last when evicting given nodes under budget pressure.
_OPERATOR_COARSE: frozenset[str] = frozenset({"operator"})


def _bbox_key(x0: float, y0: float, x1: float, y1: float) -> tuple[int, int, int, int]:
    """Stable integer bbox key used throughout merge/pin so lookups always match."""
    return (int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))


def merge_given(
    yolo: list[Detection],
    prior: Sequence[GivenNode],
    max_nodes: int,
) -> MergeResult:
    """Merge child YOLO detections with given-equation nodes.

    Centre-in-cell dedupe: a child detection whose centre (cx, cy) falls inside a
    given node's bbox rect marks that cell as "drawn over" -- the child geometry is
    kept (re-emitted with ``given=True``), the synthetic for that cell is dropped,
    and the known fine label is recorded keyed by the CHILD detection's integer bbox.

    Unmatched given nodes become synthetic Detection(given=True, confidence=1.0)
    entries; they are also added to compose_nodes so their tiles get painted before
    build_graph crops them.

    Overflow cap: if ``len(child) + len(given_survivors) > max_nodes``, given nodes
    are dropped (non-operator nodes first, operator kept last) until within budget.
    Child detections are NEVER dropped.

    Returns MergeResult with:
      detections  -- child (given-flagged where matched) + synthetic survivors
      compose_nodes -- only unmatched given nodes whose tiles must be painted
      known_fine  -- bbox key -> fine label for EVERY given node (matched or not)
    """
    # Build known_fine for ALL given nodes upfront (matched or not).
    known_fine: dict[tuple[int, int, int, int], str] = {}
    for gn in prior:
        known_fine[_bbox_key(gn.x0, gn.y0, gn.x1, gn.y1)] = gn.fine_label

    # Centre-in-cell dedupe: match each child detection against given nodes.
    # Each given node can absorb at most one child (first hit wins).
    matched_given_indices: set[int] = set()
    child_detections: list[Detection] = []

    for det in yolo:
        cx = (det.x0 + det.x1) / 2.0
        cy = (det.y0 + det.y1) / 2.0
        hit_idx: int | None = None
        for i, gn in enumerate(prior):
            if i in matched_given_indices:
                continue
            if gn.x0 <= cx <= gn.x1 and gn.y0 <= cy <= gn.y1:
                hit_idx = i
                break
        if hit_idx is not None:
            # Child drew over this given cell -- re-emit child with given=True.
            # The known_fine key for this cell uses the CHILD's bbox.
            child_key = _bbox_key(det.x0, det.y0, det.x1, det.y1)
            known_fine[child_key] = prior[hit_idx].fine_label
            matched_given_indices.add(hit_idx)
            child_detections.append(dataclasses.replace(det, given=True))
        else:
            child_detections.append(det)

    # Collect unmatched given nodes as synthetic detections.
    unmatched_given: list[GivenNode] = [
        gn for i, gn in enumerate(prior) if i not in matched_given_indices
    ]

    # Overflow cap: drop given nodes (non-operator first, then operator) until within budget.
    budget_for_given = max_nodes - len(child_detections)
    if budget_for_given < 0:
        # Should not happen (caller never provides more child than max_nodes), but be safe.
        budget_for_given = 0

    if len(unmatched_given) > budget_for_given:
        # Partition: operators survive longest.
        non_ops = [gn for gn in unmatched_given if gn.coarse_label not in _OPERATOR_COARSE]
        ops = [gn for gn in unmatched_given if gn.coarse_label in _OPERATOR_COARSE]
        # Fill budget with operators first (kept), then non-operators.
        op_budget = min(len(ops), budget_for_given)
        non_op_budget = budget_for_given - op_budget
        unmatched_given = ops[:op_budget] + non_ops[:non_op_budget]

    # Build synthetic detections for surviving unmatched given nodes.
    synthetic_detections: list[Detection] = []
    compose_nodes: list[GivenNode] = []
    for gn in unmatched_given:
        synthetic_detections.append(
            Detection(
                label=gn.coarse_label,
                confidence=1.0,
                x0=float(gn.x0),
                y0=float(gn.y0),
                x1=float(gn.x1),
                y1=float(gn.y1),
                given=True,
            )
        )
        compose_nodes.append(gn)

    detections = child_detections + synthetic_detections
    return MergeResult(
        detections=detections,
        compose_nodes=compose_nodes,
        known_fine=known_fine,
    )


def compose_into_scene(
    gray: np.ndarray,
    compose_nodes: Sequence[GivenNode],
) -> np.ndarray:
    """Paint given-node tiles onto a copy of the scene image.

    Uses darken-only compositing (np.minimum) so existing child ink is preserved;
    white areas (255) stay white. Each 28x28 tile is resized to the node's bbox.
    Degenerate bboxes (w or h <= 0) are skipped silently.

    Args:
        gray: (512, 512) uint8 grayscale scene image. Not mutated.
        compose_nodes: GivenNode entries whose tiles to paint.

    Returns:
        New (512, 512) uint8 array with tiles composited in.
    """
    assert gray.shape == (512, 512), (
        f"compose_into_scene expects a (512,512) image, got {gray.shape}"
    )
    out = gray.copy()
    H, W = out.shape  # both 512

    for gn in compose_nodes:
        x0, y0, x1, y1 = gn.x0, gn.y0, gn.x1, gn.y1
        w = x1 - x0
        h = y1 - y0
        if w <= 0 or h <= 0:
            continue

        # Resize 28x28 tile to the node's bbox dimensions.
        tile_img = Image.fromarray(gn.tile, mode="L")
        tile_resized = np.array(tile_img.resize((w, h), Image.BILINEAR), dtype=np.uint8)

        # Clip to canvas bounds.
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(W, x1), min(H, y1)
        tw, th = cx1 - cx0, cy1 - cy0
        if tw <= 0 or th <= 0:
            continue

        out[cy0:cy1, cx0:cx1] = np.minimum(
            out[cy0:cy1, cx0:cx1],
            tile_resized[:th, :tw],
        )

    return out


def pin_given_labels(
    preds: list[NodePrediction],
    known_fine: dict[tuple[int, int, int, int], str],
) -> None:
    """Override fine_label on every given NodePrediction using the known-fine map.

    Mutates preds in place (NodePrediction is frozen; uses object.__setattr__).
    Non-given preds are untouched. A given pred whose bbox key is absent from
    known_fine is silently left unchanged.

    Args:
        preds: NodePrediction list from the GNN (may contain both given and child nodes).
        known_fine: Maps integer bbox tuple (x0, y0, x1, y1) to known fine label.
    """
    for pred in preds:
        if not pred.given:
            continue
        key = _bbox_key(pred.x0, pred.y0, pred.x1, pred.y1)
        fine = known_fine.get(key)
        if fine is None:
            continue
        object.__setattr__(pred, "fine_label", fine)
