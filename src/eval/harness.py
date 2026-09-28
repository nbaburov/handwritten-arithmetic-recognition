"""Harness core for the iter6 evaluation framework.

Public API
----------
SampleMetric : dataclass (frozen)
    Per-sample metric bundle produced by compute_per_sample_metrics.

AggregateMetric : dataclass (frozen)
    Aggregate metric bundle produced by compute_aggregate.

RunRecord : dataclass (frozen)
    Full record for one evaluation run.

_ASSEMBLER_TO_LABEL : Dict[str, str]
    Maps assembler short equation-kind names to canonical label long names.

compute_per_sample_metrics(pred, label, wall_ms) -> SampleMetric
    Compute per-sample metrics given a prediction dict and a SampleLabel.

compute_aggregate(samples) -> AggregateMetric
    Fold a list of SampleMetric into an AggregateMetric.

wilson_interval(successes, n, z) -> tuple[float, float]
    Wilson score confidence interval for a binomial proportion.

aggregate_by(samples, key) -> dict[str, AggregateMetric]
    Aggregate per-sample metrics grouped by "equation_kind" or "scene_case".

run_eval(project_root, config_id) -> RunRecord
    Run the full evaluation pipeline over data/eval/bank/*.png.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from src.core.config import DataPrepConfig
from src.data_pipeline.preprocessing import preprocess_for_pipeline
from src.eval.labels import EQUATION_KINDS, SampleLabel, load_label_sidecars
from src.core.artifact_paths import find_yolo_best_pt, resolve_gnn_best_pt
from src.inference.run import InferenceSession
from src.parsing.assemble import OOD_EQUATION_KIND
from src.parsing.match_tokens import greedy_match_scene

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module constant: assembler short name → label long name
# This is the ONLY place the mapping lives — edit here, nowhere else.
# ---------------------------------------------------------------------------

_ASSEMBLER_TO_LABEL: Dict[str, str] = {
    "add": "addition",
    "subtract": "subtraction",
    "multiply": "multiplication",
    "divide": "division",
    # W10: bare_digits is a new equation_kind value; it maps to itself
    # because the assembler emits it as the full label string directly.
    "bare_digits": "bare_digits",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SampleMetric:
    """Per-sample metric bundle.

    Parameters
    ----------
    name:
        Stem of the sample file (without extension).
    equation_kind_pred:
        Raw predicted equation_kind string from the assembler output.
    equation_kind_gt:
        Ground-truth equation_kind string from the label.
    equation_kind_ok:
        True if the predicted equation kind maps to the ground-truth kind.
    ood_triggered:
        True if the prediction reported equation_kind == "unknown" with an
        ood_reason field present.
    ood_honest:
        True if ood_triggered AND the ground-truth equation kind is in
        EQUATION_KINDS (i.e., the OOD gate fired unnecessarily).
    ood_reason:
        The ood_reason string from the prediction dict, or None when
        ood_triggered is False.
    n_pred:
        Number of predicted tokens (symbols) in the output, or None if the
        output has no tokens field.
    n_gt:
        Number of ground-truth symbols, or None when label.symbols is None.
    matched:
        Number of IoU-matched symbol pairs (greedy), or None when label.symbols
        is None.
    label_correct:
        Number of matched pairs where the flattened label also agreed, or None
        when label.symbols is None.
    recall:
        matched / n_gt, or None when label.symbols is None or n_gt == 0.
    mean_iou_matched:
        Mean IoU over matched pairs.
        NOTE: greedy_match_scene does not return per-pair IoU scores in its
        current API (only matched/label_ok/gt_count/pred_count/pairs as
        (gi, pi, ok) triples without the score). Consequently this field is
        always None until the match_tokens API is extended. Documented here
        for forward compatibility.
    row_acc:
        Fraction of IoU-matched pairs where the predicted row cluster ID equals
        the ground-truth row cluster ID (derived by gap-based clustering on
        label bbox_px y-centroids). None when label.symbols is None.
    col_acc:
        Fraction of IoU-matched pairs where the predicted col cluster ID equals
        the ground-truth col cluster ID (derived by gap-based clustering on
        label bbox_px x-centroids). None when label.symbols is None.
    wall_ms:
        Wall-clock time in milliseconds for session.predict() only (excludes
        preprocessing and image loading).
    scene_case:
        Optional scene sub-case identifier from the matched ``SampleLabel``
        (e.g. ``"addition_dense_carries"``), or ``None`` when the label has
        no ``scene_case`` field.  Used for per-case diagnostic aggregation.
    """

    name: str
    equation_kind_pred: str
    equation_kind_gt: str
    equation_kind_ok: bool
    ood_triggered: bool
    ood_honest: bool
    ood_reason: Optional[str]
    n_pred: Optional[int]
    n_gt: Optional[int]
    matched: Optional[int]
    label_correct: Optional[int]
    recall: Optional[float]
    mean_iou_matched: Optional[float]
    row_acc: Optional[float]
    col_acc: Optional[float]
    wall_ms: float
    scene_case: Optional[str] = None
    # iter11 (schema_version=3): per IoU-matched pair, the [gt_fine_label, pred_fine_label] strings.
    # Feeds the per-glyph confusion table (e.g. 1<->7). None when the label has no symbols.
    label_pairs: Optional[List[List[str]]] = None


@dataclass(frozen=True)
class AggregateMetric:
    """Aggregate metric bundle over a list of SampleMetric.

    Parameters
    ----------
    n_samples:
        Total number of samples.
    equation_kind_acc:
        Fraction of samples where equation_kind_ok is True.
    ood_rate:
        Fraction of samples where ood_triggered is True.
    ood_honest_rate:
        Fraction of samples where ood_honest is True.
    mean_label_acc:
        Mean of (label_correct / matched) over samples where matched > 0.
        None if no such samples exist.
    mean_iou_matched:
        Mean of per-sample mean_iou_matched over samples where it is not None.
        None if all samples have None.
    mean_recall:
        Mean recall over samples where n_gt is not None (i.e. label has symbols).
        None if no such samples exist.
    row_acc_macro:
        Macro-average of per-sample row_acc over samples where it is not None.
        None if no such samples exist.
    col_acc_macro:
        Macro-average of per-sample col_acc over samples where it is not None.
        None if no such samples exist.
    n_samples_with_spatial:
        Count of samples that contributed to row_acc_macro and col_acc_macro
        (i.e. samples where label.symbols is not None and at least one match exists).
    total_wall_ms:
        Total (sum) wall_ms across all samples.
    """

    n_samples: int
    equation_kind_acc: float
    ood_rate: float
    ood_honest_rate: float
    mean_label_acc: Optional[float]
    mean_iou_matched: Optional[float]
    mean_recall: Optional[float]
    row_acc_macro: Optional[float]
    col_acc_macro: Optional[float]
    n_samples_with_spatial: int
    total_wall_ms: float


@dataclass(frozen=True)
class RunRecord:
    """Full record for one evaluation run.

    Parameters
    ----------
    run_id:
        YYYYMMDDTHHMMSS UTC identifier for this run.
    config_id:
        Logical configuration identifier (e.g. "baseline").
    config_snapshot:
        Snapshot of inference configuration used for this run.
    torch_version:
        torch.__version__ string.
    ultralytics_version:
        ultralytics.__version__ string.
    python_version:
        sys.version string.
    yolo_sha:
        SHA256[:8] of the YOLO weights file, or None if no YOLO weights.
    gnn_sha:
        SHA256[:8] of the GNN weights file.
    aggregate:
        Aggregate metric bundle.
    samples:
        Per-sample metric bundles.
    per_equation_kind:
        Per equation-kind aggregate metrics (schema_version=2+).
        Maps equation_kind string to AggregateMetric.  Empty dict for
        records reconstructed from schema_version=1 files.
    per_scene_case:
        Per scene-case aggregate metrics (schema_version=2+).
        Maps scene_case string to AggregateMetric.  Empty dict when no
        sample has a scene_case, or when reconstructed from v1 files.
    """

    run_id: str
    config_id: str
    config_snapshot: Dict[str, Any]
    torch_version: str
    ultralytics_version: str
    python_version: str
    yolo_sha: Optional[str]
    gnn_sha: str
    aggregate: AggregateMetric
    samples: Tuple[SampleMetric, ...]
    per_equation_kind: Dict[str, AggregateMetric] = dataclasses.field(default_factory=dict)
    per_scene_case: Dict[str, AggregateMetric] = dataclasses.field(default_factory=dict)
    # iter11 (schema_version=3): per-fine-label confusion. confusion[gt][pred] = matched-pair count;
    # per_fine_label[gt] = {"correct", "total", "recall"}. Surfaces digit confusions like 1<->7.
    confusion: Dict[str, Dict[str, int]] = dataclasses.field(default_factory=dict)
    per_fine_label: Dict[str, Dict[str, float]] = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# Spatial clustering helper
# ---------------------------------------------------------------------------

def _gap_cluster(values: List[float]) -> List[int]:
    """Assign 0-based cluster IDs to *values* using gap-based sorting.

    Sorts values, finds the largest gap(s) as cluster boundaries, and returns
    cluster IDs in the original order. Uses a single largest-gap split
    iteratively until no gap exceeds 10% of the full range. This mirrors the
    logic the GNN uses internally for row/col assignment.

    Parameters
    ----------
    values:
        List of floats (e.g. y-centroids for rows, x-centroids for cols).

    Returns
    -------
    List[int]
        Cluster IDs in the same order as *values*. All zeros if len(values) < 2.
    """
    if len(values) < 2:
        return [0] * len(values)

    indexed = sorted(enumerate(values), key=lambda iv: iv[1])
    sorted_vals = [v for _, v in indexed]
    val_range = sorted_vals[-1] - sorted_vals[0]
    if val_range == 0:
        return [0] * len(values)

    # Find split points: gaps exceeding threshold fraction of range
    threshold = val_range * 0.10
    splits: List[int] = []  # indices i where a new cluster starts at i+1
    for i in range(len(sorted_vals) - 1):
        if sorted_vals[i + 1] - sorted_vals[i] > threshold:
            splits.append(i)

    # Assign cluster IDs
    cluster_map: List[int] = [0] * len(sorted_vals)
    cluster_id = 0
    for i, (orig_idx, _) in enumerate(indexed):
        if splits and i > splits[0]:
            cluster_id += 1
            splits.pop(0)
        cluster_map[orig_idx] = cluster_id

    return cluster_map


# ---------------------------------------------------------------------------
# Per-sample metric computation
# ---------------------------------------------------------------------------

def compute_per_sample_metrics(
    pred: Dict[str, Any],
    label: SampleLabel,
    wall_ms: float,
) -> SampleMetric:
    """Compute per-sample metrics for one prediction/label pair.

    Parameters
    ----------
    pred:
        Prediction dict as returned by InferenceSession.predict().
        Expected keys: ``equation_kind``, optionally ``ood_reason``,
        ``tokens``.
    label:
        Ground-truth SampleLabel for this sample.
    wall_ms:
        Wall-clock time in milliseconds for the predict() call only.

    Returns
    -------
    SampleMetric
        Frozen per-sample metric bundle.

    Notes
    -----
    ``mean_iou_matched`` is always None because ``greedy_match_scene`` does
    not expose per-pair IoU scores in its current API.  Extend
    ``src/parsing/match_tokens.py`` to return scores in ``pairs`` entries to
    enable this field.
    """
    pred_kind = pred.get("equation_kind", "")

    # equation_kind_ok: map short name → long name, compare to label
    equation_kind_ok: bool = _ASSEMBLER_TO_LABEL.get(pred_kind) == label.equation_kind

    # OOD detection
    ood_reason: Optional[str] = pred.get("ood_reason") or None
    ood_triggered: bool = (pred_kind == OOD_EQUATION_KIND) and (ood_reason is not None)
    ood_honest: bool = ood_triggered and (label.equation_kind in EQUATION_KINDS)

    # Token counts
    tokens = pred.get("tokens", [])
    n_pred: Optional[int] = len(tokens) if tokens is not None else 0

    # Spatial metrics — only when ground truth symbols are present
    if label.symbols is None:
        return SampleMetric(
            name=label.sample,
            equation_kind_pred=pred_kind,
            equation_kind_gt=label.equation_kind,
            equation_kind_ok=equation_kind_ok,
            ood_triggered=ood_triggered,
            ood_honest=ood_honest,
            ood_reason=ood_reason,
            n_pred=n_pred,
            n_gt=None,
            matched=None,
            label_correct=None,
            recall=None,
            mean_iou_matched=None,
            row_acc=None,
            col_acc=None,
            wall_ms=wall_ms,
            scene_case=label.scene_case,
        )

    n_gt: int = len(label.symbols)

    if n_gt == 0 or not tokens:
        # No GT symbols or no predictions — recall is 0 (not None, GT present)
        return SampleMetric(
            name=label.sample,
            equation_kind_pred=pred_kind,
            equation_kind_gt=label.equation_kind,
            equation_kind_ok=equation_kind_ok,
            ood_triggered=ood_triggered,
            ood_honest=ood_honest,
            ood_reason=ood_reason,
            n_pred=n_pred,
            n_gt=n_gt,
            matched=0,
            label_correct=0,
            recall=0.0 if n_gt > 0 else None,
            mean_iou_matched=None,
            row_acc=None,
            col_acc=None,
            wall_ms=wall_ms,
            scene_case=label.scene_case,
        )

    # Build gt_tokens and pred_tokens in the shape greedy_match_scene expects
    gt_tokens = [
        {"flattened": sym.flattened, "bbox_px": list(sym.bbox_px)}
        for sym in label.symbols
    ]
    pred_tokens_raw = [
        {
            "flattened": t.get("label", t.get("flattened", "")),
            "bbox_px": t.get("bbox", t.get("bbox_px", [0.0, 0.0, 1.0, 1.0])),
        }
        for t in tokens
        if isinstance(t, dict)
    ]

    result = greedy_match_scene(gt_tokens, pred_tokens_raw)
    matched: int = result["matched"]
    label_correct: int = result["label_ok"]
    recall: float = matched / n_gt if n_gt > 0 else 0.0

    # --- Row / col accuracy ---
    # GT row and col are derived by gap-based clustering of bbox_px centroids,
    # since label sidecars do not carry explicit row/col fields.
    # Predicted row/col come from detections[] in the prediction dict.
    row_acc: Optional[float] = None
    col_acc: Optional[float] = None

    pairs = result.get("pairs", [])  # list of (gt_idx, pred_idx, label_ok)
    detections: List[Dict[str, Any]] = pred.get("detections", [])

    if pairs and detections:
        # Assign GT cluster IDs from bbox centroids
        gt_y_centers = [
            (sym.bbox_px[1] + sym.bbox_px[3]) / 2.0 for sym in label.symbols
        ]
        gt_x_centers = [
            (sym.bbox_px[0] + sym.bbox_px[2]) / 2.0 for sym in label.symbols
        ]
        gt_row_ids = _gap_cluster(gt_y_centers)
        gt_col_ids = _gap_cluster(gt_x_centers)

        # Build a bbox-keyed lookup from detections to predicted row/col.
        # greedy_match_scene matches pred_tokens_raw by index; detections[] may
        # differ in length/order from pred_tokens_raw (detections includes all
        # predictions before gate filtering). We match by bbox proximity.
        det_bboxes = [d["bbox"] for d in detections]

        def _closest_det_idx(bbox_px: List[float]) -> int:
            """Return index of detection whose bbox centre is closest to bbox_px centre."""
            qx = (bbox_px[0] + bbox_px[2]) / 2.0
            qy = (bbox_px[1] + bbox_px[3]) / 2.0
            best_i, best_d = 0, float("inf")
            for i, db in enumerate(det_bboxes):
                dx = (db[0] + db[2]) / 2.0 - qx
                dy = (db[1] + db[3]) / 2.0 - qy
                d2 = dx * dx + dy * dy
                if d2 < best_d:
                    best_d, best_i = d2, i
            return best_i

        row_correct = 0
        col_correct = 0
        n_pairs = len(pairs)

        for gt_idx, pred_idx, _ in pairs:
            # Predicted bbox from pred_tokens_raw (what greedy_match_scene used)
            pred_bbox = pred_tokens_raw[pred_idx]["bbox_px"]
            det_idx = _closest_det_idx(pred_bbox)
            pred_row = int(detections[det_idx]["row"])
            pred_col = int(detections[det_idx]["col"])
            gt_row = gt_row_ids[gt_idx]
            gt_col = gt_col_ids[gt_idx]
            if pred_row == gt_row:
                row_correct += 1
            if pred_col == gt_col:
                col_correct += 1

        row_acc = row_correct / n_pairs
        col_acc = col_correct / n_pairs

    # iter11: capture matched (gt_label, pred_label) pairs for the confusion table.
    label_pairs: List[List[str]] = [
        [gt_tokens[gi]["flattened"], pred_tokens_raw[pi]["flattened"]]
        for gi, pi, _ in pairs
    ]

    return SampleMetric(
        name=label.sample,
        equation_kind_pred=pred_kind,
        equation_kind_gt=label.equation_kind,
        equation_kind_ok=equation_kind_ok,
        ood_triggered=ood_triggered,
        ood_honest=ood_honest,
        ood_reason=ood_reason,
        n_pred=n_pred,
        n_gt=n_gt,
        matched=matched,
        label_correct=label_correct,
        recall=recall,
        mean_iou_matched=None,  # greedy_match_scene does not return per-pair IoU
        row_acc=row_acc,
        col_acc=col_acc,
        wall_ms=wall_ms,
        scene_case=label.scene_case,
        label_pairs=label_pairs,
    )


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def compute_aggregate(samples: List[SampleMetric]) -> AggregateMetric:
    """Fold a list of SampleMetric into an AggregateMetric.

    Parameters
    ----------
    samples:
        Non-empty list of SampleMetric instances.

    Returns
    -------
    AggregateMetric

    Raises
    ------
    ValueError
        If ``samples`` is empty.
    """
    if not samples:
        raise ValueError("compute_aggregate requires at least one sample.")

    n = len(samples)
    equation_kind_acc = sum(s.equation_kind_ok for s in samples) / n
    ood_rate = sum(s.ood_triggered for s in samples) / n
    ood_honest_rate = sum(s.ood_honest for s in samples) / n
    total_wall_ms = sum(s.wall_ms for s in samples)

    # mean_label_acc: mean of (label_correct / matched) where matched > 0
    label_acc_vals = [
        s.label_correct / s.matched
        for s in samples
        if s.matched is not None and s.matched > 0 and s.label_correct is not None
    ]
    mean_label_acc: Optional[float] = (
        sum(label_acc_vals) / len(label_acc_vals) if label_acc_vals else None
    )

    # mean_iou_matched: mean over samples with non-None mean_iou_matched
    iou_vals = [s.mean_iou_matched for s in samples if s.mean_iou_matched is not None]
    mean_iou_matched: Optional[float] = (
        sum(iou_vals) / len(iou_vals) if iou_vals else None
    )

    # mean_recall: mean over samples with non-None recall
    recall_vals = [s.recall for s in samples if s.recall is not None]
    mean_recall: Optional[float] = (
        sum(recall_vals) / len(recall_vals) if recall_vals else None
    )

    # row_acc_macro / col_acc_macro: macro-average over samples with spatial data
    row_acc_vals = [s.row_acc for s in samples if s.row_acc is not None]
    col_acc_vals = [s.col_acc for s in samples if s.col_acc is not None]
    row_acc_macro: Optional[float] = (
        sum(row_acc_vals) / len(row_acc_vals) if row_acc_vals else None
    )
    col_acc_macro: Optional[float] = (
        sum(col_acc_vals) / len(col_acc_vals) if col_acc_vals else None
    )
    n_samples_with_spatial: int = len(row_acc_vals)

    return AggregateMetric(
        n_samples=n,
        equation_kind_acc=equation_kind_acc,
        ood_rate=ood_rate,
        ood_honest_rate=ood_honest_rate,
        mean_label_acc=mean_label_acc,
        mean_iou_matched=mean_iou_matched,
        mean_recall=mean_recall,
        row_acc_macro=row_acc_macro,
        col_acc_macro=col_acc_macro,
        n_samples_with_spatial=n_samples_with_spatial,
        total_wall_ms=total_wall_ms,
    )


# ---------------------------------------------------------------------------
# Wilson score interval
# ---------------------------------------------------------------------------

def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Compute the Wilson score confidence interval for a binomial proportion.

    Parameters
    ----------
    successes:
        Number of successes.
    n:
        Total number of trials.
    z:
        Z-score for the desired confidence level (default 1.96 = 95%).

    Returns
    -------
    Tuple[float, float]
        (lower, upper) bounds of the confidence interval, both in [0, 1].
        Returns (0.0, 0.0) when n == 0.
    """
    if n == 0:
        return (0.0, 0.0)
    import math
    p_hat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p_hat + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ---------------------------------------------------------------------------
# Per-group aggregation
# ---------------------------------------------------------------------------

def aggregate_by(
    samples: List[SampleMetric],
    key: str,
) -> Dict[str, AggregateMetric]:
    """Aggregate per-sample metrics grouped by a string key.

    Parameters
    ----------
    samples:
        List of SampleMetric instances.
    key:
        Which field to group by — must be ``"equation_kind"`` or
        ``"scene_case"``.  Samples where the key value is None are skipped.

    Returns
    -------
    Dict[str, AggregateMetric]
        Mapping from group value to its AggregateMetric.

    Raises
    ------
    ValueError
        If ``key`` is not one of the supported values.
    """
    if key not in ("equation_kind", "scene_case"):
        raise ValueError(
            f"aggregate_by: key must be 'equation_kind' or 'scene_case', got {key!r}"
        )

    groups: Dict[str, List[SampleMetric]] = {}
    for s in samples:
        if key == "equation_kind":
            group_val: Optional[str] = s.equation_kind_gt
        else:
            group_val = s.scene_case
        if group_val is None:
            continue
        if group_val not in groups:
            groups[group_val] = []
        groups[group_val].append(s)

    return {group: compute_aggregate(group_samples) for group, group_samples in groups.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from ..core.hashing import sha256_file as _sha256_file  # noqa: E402


# ---------------------------------------------------------------------------
# run_eval
# ---------------------------------------------------------------------------

def compute_label_confusion(
    samples: List[SampleMetric],
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, float]]]:
    """Aggregate per-fine-label confusion from matched (gt, pred) pairs across samples.

    Returns ``(confusion, per_fine_label)`` where ``confusion[gt][pred]`` is the matched-pair count
    and ``per_fine_label[gt] = {"correct", "total", "recall"}``. Surfaces digit confusions (e.g. 1<->7).
    """
    confusion: Dict[str, Dict[str, int]] = {}
    per: Dict[str, Dict[str, float]] = {}
    for s in samples:
        for pair in (s.label_pairs or []):
            gt, pred = pair[0], pair[1]
            confusion.setdefault(gt, {}).setdefault(pred, 0)
            confusion[gt][pred] += 1
            d = per.setdefault(gt, {"correct": 0.0, "total": 0.0, "recall": 0.0})
            d["total"] += 1.0
            if gt == pred:
                d["correct"] += 1.0
    for gt, d in per.items():
        d["recall"] = (d["correct"] / d["total"]) if d["total"] else 0.0
    return confusion, per


def run_eval(
    project_root: Path,
    config_id: str = "baseline",
    samples_dir: Optional[Path] = None,
    yolo_run_id: Optional[str] = None,
    gnn_run_id: Optional[str] = None,
) -> RunRecord:
    """Run the full evaluation over data/eval/bank/*.png.

    Parameters
    ----------
    project_root:
        Root of the handwritten-arithmetic-recognition project.
    config_id:
        Logical configuration identifier stored in the RunRecord.

    Returns
    -------
    RunRecord

    Raises
    ------
    ValueError
        If model weights cannot be found (wraps FileNotFoundError with a
        descriptive message naming the missing artifact path).
    """
    import torch  # noqa: PLC0415
    import ultralytics  # type: ignore  # noqa: PLC0415


    # Resolve config and artifact paths
    config = DataPrepConfig.from_project_root(project_root)

    # Resolve weights. By default use the active pointers; when an explicit run id is given, load that
    # run's best.pt directly (compare a fresh run vs baseline without touching active.json).
    if gnn_run_id:
        gnn_path = config.artifacts_gnn_dir / "runs" / gnn_run_id / "best.pt"
        if not gnn_path.is_file():
            raise ValueError(f"GNN run {gnn_run_id!r} has no best.pt at {gnn_path}.")
    else:
        try:
            gnn_path = resolve_gnn_best_pt(config)
        except FileNotFoundError as exc:
            raise ValueError(
                f"GNN weights not found. Train with: python -m src train --stage gnn. "
                f"Details: {exc}"
            ) from exc

    if yolo_run_id:
        yolo_path = config.artifacts_yolo_dir / "runs" / yolo_run_id / "best.pt"
        if not yolo_path.is_file():
            raise ValueError(f"YOLO run {yolo_run_id!r} has no best.pt at {yolo_path}.")
    else:
        yolo_path = find_yolo_best_pt(config)

    gnn_sha = _sha256_file(gnn_path)[:8]
    yolo_sha: Optional[str] = _sha256_file(yolo_path)[:8] if yolo_path else None

    # Build InferenceSession. Explicit run ids -> load those weights; else auto_load from active.json.
    session = InferenceSession()
    try:
        if gnn_run_id or yolo_run_id:
            session.load(gnn_weights=gnn_path, yolo_weights=yolo_path)
        else:
            session.auto_load(project_root)
    except Exception as exc:
        raise ValueError(
            f"Failed to load models from {project_root}. "
            f"Run train commands first. Details: {exc}"
        ) from exc

    if not session.is_loaded():
        raise ValueError(
            f"InferenceSession is not loaded after auto_load(). "
            f"Missing weights under {project_root / 'artifacts'}."
        )

    # Build config snapshot
    config_snapshot: Dict[str, Any] = {
        "config_id": config_id,
        "yolo_conf": 0.05,
        "tta": True,
        "yolo_inference_config": dataclasses.asdict(session._yolo_inference_config),
    }

    # Discover samples and labels
    if samples_dir is None:
        samples_dir = project_root / "data" / "eval" / "bank"
    labels: Dict[str, SampleLabel] = load_label_sidecars(samples_dir)

    png_files = sorted(samples_dir.glob("*.png"))
    sample_metrics: List[SampleMetric] = []

    warmup_done = False

    for png_path in png_files:
        stem = png_path.stem
        if stem not in labels:
            logger.warning("Skipping %s: no matching label sidecar found.", png_path.name)
            continue

        label = labels[stem]
        img = Image.open(png_path)
        gray = preprocess_for_pipeline(img)

        # Warmup: predict the first sample twice, discard first wall_ms
        if not warmup_done:
            _ = session.predict(gray, yolo_conf=0.05, tta=True)
            warmup_done = True

        t0 = time.perf_counter()
        pred = session.predict(gray, yolo_conf=0.05, tta=True)
        wall_ms = (time.perf_counter() - t0) * 1000.0

        metric = compute_per_sample_metrics(pred, label, wall_ms)
        sample_metrics.append(metric)

    if not sample_metrics:
        raise ValueError(
            f"No samples with labels found in {samples_dir}. "
            "Ensure *.png files have matching *.label.json sidecars."
        )

    aggregate = compute_aggregate(sample_metrics)
    per_equation_kind = aggregate_by(sample_metrics, "equation_kind")
    per_scene_case = aggregate_by(sample_metrics, "scene_case")
    confusion, per_fine_label = compute_label_confusion(sample_metrics)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    return RunRecord(
        run_id=run_id,
        config_id=config_id,
        config_snapshot=config_snapshot,
        torch_version=torch.__version__,
        ultralytics_version=ultralytics.__version__,
        python_version=sys.version,
        yolo_sha=yolo_sha,
        gnn_sha=gnn_sha,
        aggregate=aggregate,
        samples=tuple(sample_metrics),
        per_equation_kind=per_equation_kind,
        per_scene_case=per_scene_case,
        confusion=confusion,
        per_fine_label=per_fine_label,
    )


# ---------------------------------------------------------------------------
# F3: JSONL persistence + Markdown auto-summary
# ---------------------------------------------------------------------------

SCHEMA_VERSION: int = 3  # v3 (iter11): adds per-fine-label confusion + per_fine_label; samples carry label_pairs.
EVAL_HISTORY_FILENAME: str = "history.jsonl"
EVAL_LATEST_SUMMARY_FILENAME: str = "latest.md"

_RECORD_TYPE_KEY: str = "record_type"
_SCHEMA_RECORD_TYPE: str = "schema"


def _fmt_float(v: Optional[float], decimals: int = 4) -> str:
    """Format a float to `decimals` decimal places, or 'n/a' if None."""
    if v is None:
        return "n/a"
    return f"{v:.{decimals}f}"


def write_jsonl_record(record: RunRecord, history_path: Path) -> None:
    """Append *record* to a JSONL history file, creating schema header if needed.

    Parameters
    ----------
    record:
        The RunRecord to persist.
    history_path:
        Path to the ``eval_history.jsonl`` file (need not exist yet).

    Raises
    ------
    ValueError
        If the file already exists and its schema_version differs from
        ``SCHEMA_VERSION``.
    """
    field_names = [f.name for f in dataclasses.fields(RunRecord)]

    if not history_path.exists():
        schema_record = {
            _RECORD_TYPE_KEY: _SCHEMA_RECORD_TYPE,
            "schema_version": SCHEMA_VERSION,
            "fields": field_names,
            "created": datetime.now(timezone.utc).isoformat(),
        }
        history_path.write_text(
            json.dumps(schema_record, sort_keys=False) + "\n",
            encoding="utf-8",
        )
    else:
        with history_path.open(encoding="utf-8") as fh:
            first_line = fh.readline()
        existing_schema = json.loads(first_line)
        existing_version = existing_schema.get("schema_version")
        if existing_version != SCHEMA_VERSION:
            raise ValueError(
                f"JSONL schema version mismatch: file has schema_version="
                f"{existing_version}, but current SCHEMA_VERSION={SCHEMA_VERSION}."
            )

    run_dict = dataclasses.asdict(record)
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(run_dict, sort_keys=False) + "\n")


def _reconstruct_run_record(d: Dict[str, Any]) -> RunRecord:
    """Reconstruct a RunRecord from a plain dict (as parsed from JSON).

    Converts nested dicts back to ``AggregateMetric`` and ``SampleMetric``
    instances, and the samples list back to a tuple.

    Fields added after schema_version=1 (row_acc_macro, col_acc_macro,
    n_samples_with_spatial, row_acc, col_acc, scene_case) default to
    None/0/empty when absent so that old JSONL records remain loadable.
    schema_version=2 adds per_equation_kind, per_scene_case dicts.
    """
    agg_d = dict(d["aggregate"])
    agg_d.setdefault("row_acc_macro", None)
    agg_d.setdefault("col_acc_macro", None)
    agg_d.setdefault("n_samples_with_spatial", 0)
    aggregate = AggregateMetric(**agg_d)

    def _reconstruct_agg(a: Dict[str, Any]) -> AggregateMetric:
        ad = dict(a)
        ad.setdefault("row_acc_macro", None)
        ad.setdefault("col_acc_macro", None)
        ad.setdefault("n_samples_with_spatial", 0)
        return AggregateMetric(**ad)

    def _reconstruct_sample(s: Dict[str, Any]) -> SampleMetric:
        sd = dict(s)
        sd.setdefault("row_acc", None)
        sd.setdefault("col_acc", None)
        sd.setdefault("scene_case", None)
        sd.setdefault("label_pairs", None)
        return SampleMetric(**sd)

    samples = tuple(_reconstruct_sample(s) for s in d["samples"])

    per_equation_kind: Dict[str, AggregateMetric] = {
        k: _reconstruct_agg(v)
        for k, v in d.get("per_equation_kind", {}).items()
    }
    per_scene_case: Dict[str, AggregateMetric] = {
        k: _reconstruct_agg(v)
        for k, v in d.get("per_scene_case", {}).items()
    }

    return RunRecord(
        run_id=d["run_id"],
        config_id=d["config_id"],
        config_snapshot=d["config_snapshot"],
        torch_version=d["torch_version"],
        ultralytics_version=d["ultralytics_version"],
        python_version=d["python_version"],
        yolo_sha=d["yolo_sha"],
        gnn_sha=d["gnn_sha"],
        aggregate=aggregate,
        samples=samples,
        per_equation_kind=per_equation_kind,
        per_scene_case=per_scene_case,
        confusion=d.get("confusion", {}),
        per_fine_label=d.get("per_fine_label", {}),
    )


def load_baseline_record(
    history_path: Path, config_id: str
) -> Optional[RunRecord]:
    """Load the most recent RunRecord matching *config_id* from a JSONL file.

    Parameters
    ----------
    history_path:
        Path to the JSONL file.  Returns ``None`` if the file does not exist.
    config_id:
        The config_id to filter on.

    Returns
    -------
    RunRecord or None
        The last matching record reconstructed from JSON, or None if no match.
    """
    if not history_path.exists():
        return None

    matching: Optional[RunRecord] = None
    with history_path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if i == 0 and d.get(_RECORD_TYPE_KEY) == _SCHEMA_RECORD_TYPE:
                continue
            if d.get("config_id") == config_id:
                matching = _reconstruct_run_record(d)
    return matching


def render_markdown_summary(
    current: RunRecord,
    baseline: Optional[RunRecord],
    output_path: Path,
) -> None:
    """Write a Markdown summary comparing *current* to *baseline*.

    The write is atomic: content is written to a ``.tmp`` sibling and then
    renamed over *output_path* via ``os.replace``.

    Parameters
    ----------
    current:
        The RunRecord for the current run.
    baseline:
        Optional baseline RunRecord to diff against.
    output_path:
        Destination ``.md`` file path.
    """
    lines: List[str] = []

    # --- Title block ---
    lines.append("# Eval Summary")
    lines.append("")
    lines.append(f"Run: {current.run_id}")
    lines.append(f"Config: {current.config_id}")
    lines.append("")

    # --- Aggregate diff table ---
    # row_acc_macro and col_acc_macro appear first: client priority is row/col
    # placement accuracy above equation_kind classification.
    agg_fields = [
        ("n_samples", "n_samples"),
        ("row_acc_macro", "row_acc_macro [PRIMARY]"),
        ("col_acc_macro", "col_acc_macro [PRIMARY]"),
        ("n_samples_with_spatial", "n_spatial_samples"),
        ("equation_kind_acc", "eq_kind_acc"),
        ("ood_rate", "ood_rate"),
        ("ood_honest_rate", "ood_honest_rate"),
        ("mean_label_acc", "mean_label_acc"),
        ("mean_iou_matched", "mean_iou_matched"),
        ("mean_recall", "mean_recall"),
        ("total_wall_ms", "total_wall_ms (ms)"),
    ]

    if baseline is not None:
        lines.append("## Aggregate Metrics")
        lines.append("")
        lines.append("| Metric | Baseline | Current | Delta |")
        lines.append("|--------|----------|---------|-------|")
        for field_name, display_name in agg_fields:
            cur_val = getattr(current.aggregate, field_name)
            bas_val = getattr(baseline.aggregate, field_name)
            if field_name in ("n_samples", "n_samples_with_spatial"):
                cur_s = str(cur_val)
                bas_s = str(bas_val)
                delta_s = str(cur_val - bas_val) if (cur_val is not None and bas_val is not None) else "n/a"
            elif field_name == "total_wall_ms":
                cur_s = _fmt_float(cur_val, 1)
                bas_s = _fmt_float(bas_val, 1)
                if cur_val is not None and bas_val is not None:
                    delta_s = _fmt_float(cur_val - bas_val, 1)
                else:
                    delta_s = "n/a"
            else:
                cur_s = _fmt_float(cur_val)
                bas_s = _fmt_float(bas_val)
                if cur_val is not None and bas_val is not None:
                    delta_s = _fmt_float(cur_val - bas_val)
                else:
                    delta_s = "n/a"
            lines.append(f"| {display_name} | {bas_s} | {cur_s} | {delta_s} |")
    else:
        lines.append("## Aggregate Metrics")
        lines.append("")
        lines.append("| Metric | Current |")
        lines.append("|--------|---------|")
        for field_name, display_name in agg_fields:
            cur_val = getattr(current.aggregate, field_name)
            if field_name in ("n_samples", "n_samples_with_spatial"):
                cur_s = str(cur_val)
            elif field_name == "total_wall_ms":
                cur_s = _fmt_float(cur_val, 1)
            else:
                cur_s = _fmt_float(cur_val)
            lines.append(f"| {display_name} | {cur_s} |")

    lines.append("")

    # --- Per-fine-label recall + top confusions (iter11) ---
    # Worst recall first, with the labels each gt is most often confused FOR (e.g. main_1 -> main_7).
    if current.per_fine_label:
        lines.append("## Per Fine-Label (recall + top confusion)")
        lines.append("")
        lines.append("| fine_label | n | recall | top confusions (pred:count) |")
        lines.append("|---|---|---|---|")
        for gt in sorted(
            current.per_fine_label,
            key=lambda k: current.per_fine_label[k].get("recall", 0.0),
        ):
            d = current.per_fine_label[gt]
            n = int(d.get("total", 0))
            rec = d.get("recall", 0.0)
            row = current.confusion.get(gt, {})
            confs = sorted(
                ((p, c) for p, c in row.items() if p != gt), key=lambda x: -x[1]
            )[:3]
            conf_s = ", ".join(f"{p}:{c}" for p, c in confs) if confs else "-"
            lines.append(f"| {gt} | {n} | {_fmt_float(rec)} | {conf_s} |")
        lines.append("")

    # --- Per equation-kind breakdown table (with Wilson CIs) ---
    if current.per_equation_kind:
        lines.append("## Per Equation-Kind")
        lines.append("")
        lines.append(
            "| equation_kind | n | eq_kind_acc | CI_lo | CI_hi"
            " | row_acc_macro | CI_lo | CI_hi"
            " | col_acc_macro | CI_lo | CI_hi |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for kind in sorted(current.per_equation_kind):
            ag = current.per_equation_kind[kind]
            n = ag.n_samples
            eq_lo, eq_hi = wilson_interval(round(ag.equation_kind_acc * n), n)
            row_lo, row_hi = (
                wilson_interval(round((ag.row_acc_macro or 0.0) * n), n)
                if ag.row_acc_macro is not None else (0.0, 0.0)
            )
            col_lo, col_hi = (
                wilson_interval(round((ag.col_acc_macro or 0.0) * n), n)
                if ag.col_acc_macro is not None else (0.0, 0.0)
            )
            row_s = _fmt_float(ag.row_acc_macro)
            col_s = _fmt_float(ag.col_acc_macro)
            lines.append(
                f"| {kind} | {n}"
                f" | {_fmt_float(ag.equation_kind_acc)} | {_fmt_float(eq_lo)} | {_fmt_float(eq_hi)}"
                f" | {row_s} | {_fmt_float(row_lo)} | {_fmt_float(row_hi)}"
                f" | {col_s} | {_fmt_float(col_lo)} | {_fmt_float(col_hi)} |"
            )
        lines.append("")

    # --- Per scene-case breakdown table (diagnostic, no CI) ---
    # Omit entirely when every sample has scene_case=None (old banks).
    if current.per_scene_case:
        lines.append("## Per Scene-Case (diagnostic, no CI claim)")
        lines.append("")
        lines.append(
            "| scene_case | n | eq_kind_acc | row_acc_macro | col_acc_macro |"
        )
        lines.append("|---|---|---|---|---|")
        for case in sorted(current.per_scene_case):
            ag = current.per_scene_case[case]
            lines.append(
                f"| {case} | {ag.n_samples}"
                f" | {_fmt_float(ag.equation_kind_acc)}"
                f" | {_fmt_float(ag.row_acc_macro)}"
                f" | {_fmt_float(ag.col_acc_macro)} |"
            )
        lines.append("")

    # --- Per-sample table ---
    lines.append("## Per-Sample Results")
    lines.append("")
    lines.append("| Sample | GT | Pred | OK | row_acc | col_acc | OOD Reason | Wall (ms) |")
    lines.append("|--------|----|------|----|---------|---------|------------|-----------|")
    for sm in current.samples:
        ok_s = "yes" if sm.equation_kind_ok else "no"
        ood_s = sm.ood_reason if sm.ood_reason is not None else ""
        row_s = _fmt_float(sm.row_acc)
        col_s = _fmt_float(sm.col_acc)
        wall_s = _fmt_float(sm.wall_ms, 1)
        lines.append(
            f"| {sm.name} | {sm.equation_kind_gt} | {sm.equation_kind_pred}"
            f" | {ok_s} | {row_s} | {col_s} | {ood_s} | {wall_s} |"
        )

    lines.append("")

    # --- Footer ---
    generated = datetime.now(timezone.utc).isoformat()
    lines.append("---")
    lines.append(f"Generated: {generated}")
    lines.append(f"Python: {current.python_version}")
    lines.append(f"PyTorch: {current.torch_version}")
    lines.append(f"Ultralytics: {current.ultralytics_version}")
    lines.append(f"YOLO SHA: {current.yolo_sha}")
    lines.append(f"GNN SHA: {current.gnn_sha}")

    content = "\n".join(lines) + "\n"

    tmp_path = output_path.with_suffix(".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(str(tmp_path), str(output_path))
