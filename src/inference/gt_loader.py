from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


@dataclass
class GtSymbol:
    fine_label: str
    row_index: int
    col_index: int
    bbox: list[float]
    yolo_class: str


def _iou(a: list, b: list) -> float:
    """Compute IoU between two [x0, y0, x1, y1] boxes."""
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class MatchResult:
    gt: Optional["GtSymbol"]
    predicted_label: Optional[str]
    predicted_bbox: Optional[list[float]]
    predicted_row: Optional[int]
    predicted_col: Optional[int]
    iou: float
    is_correct: bool
    is_false_positive: bool
    is_missed: bool


@dataclass
class SceneScores:
    token_f1: float
    row_accuracy: float
    col_accuracy: float
    exact_match: bool
    n_correct: int
    n_total: int


def load_scene(
    image_path: Path,
) -> tuple[np.ndarray, Optional[list[GtSymbol]]]:
    """
    Load a scene image and its sibling gt.json (if present).

    Returns (grayscale_uint8_array, list_of_GtSymbol_or_None).
    gt.json expected alongside the image named 'gt.json'.
    """
    image_path = Path(image_path)
    gray = np.array(Image.open(image_path).convert("L"), dtype=np.uint8)

    gt_path = image_path.parent / "gt.json"
    if not gt_path.exists():
        return gray, None

    data = json.loads(gt_path.read_text(encoding="utf-8"))
    symbols = [
        GtSymbol(
            fine_label=sym["fine_label"],
            row_index=int(sym["row_index"]),
            col_index=int(sym["col_index"]),
            bbox=[float(v) for v in sym["bbox"]],
            yolo_class=sym.get("yolo_class", "digit_main"),
        )
        for sym in data.get("symbols", [])
    ]
    return gray, symbols


def match_predictions(
    tokens: list[dict],
    gt_symbols: list[GtSymbol],
    iou_threshold: float = 0.4,
) -> list[MatchResult]:
    """
    Greedy IoU matching between predicted tokens and GT symbols.

    tokens: flattened payload tokens with keys label, confidence, bbox, row, col.
    """
    results: list[MatchResult] = []
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()

    # Build all pairs above threshold, sort by IoU descending
    pairs: list[tuple[float, int, int]] = []
    for pi, tok in enumerate(tokens):
        for gi, gt in enumerate(gt_symbols):
            score = _iou(tok["bbox"], gt.bbox)
            if score >= iou_threshold:
                pairs.append((score, pi, gi))
    pairs.sort(reverse=True)

    # Greedy match
    for iou_score, pi, gi in pairs:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        tok = tokens[pi]
        gt = gt_symbols[gi]
        results.append(MatchResult(
            gt=gt,
            predicted_label=tok["label"],
            predicted_bbox=tok["bbox"],
            predicted_row=tok.get("row"),
            predicted_col=tok.get("col"),
            iou=iou_score,
            is_correct=tok["label"] == gt.fine_label,
            is_false_positive=False,
            is_missed=False,
        ))

    # Missed GT
    for gi, gt in enumerate(gt_symbols):
        if gi not in matched_gt:
            results.append(MatchResult(
                gt=gt,
                predicted_label=None,
                predicted_bbox=None,
                predicted_row=None,
                predicted_col=None,
                iou=0.0,
                is_correct=False,
                is_false_positive=False,
                is_missed=True,
            ))

    # False positives
    for pi, tok in enumerate(tokens):
        if pi not in matched_pred:
            results.append(MatchResult(
                gt=None,
                predicted_label=tok["label"],
                predicted_bbox=tok["bbox"],
                predicted_row=tok.get("row"),
                predicted_col=tok.get("col"),
                iou=0.0,
                is_correct=False,
                is_false_positive=True,
                is_missed=False,
            ))

    return results


def compute_scene_scores(
    tokens: list[dict],
    gt_symbols: list[GtSymbol],
) -> SceneScores:
    """Compute token F1, row/col accuracy, and exact match over matched pairs.

    FP definition: counts both unmatched predictions and IoU-matched pairs where
    the label is wrong. This is a strict token-level F1 (not standard COCO-style
    per-class FP/FN).
    """
    if not tokens and not gt_symbols:
        return SceneScores(token_f1=1.0, row_accuracy=1.0, col_accuracy=1.0,
                           exact_match=True, n_correct=0, n_total=0)

    matches = match_predictions(tokens, gt_symbols)

    tp = sum(1 for m in matches if m.is_correct)
    fp = sum(1 for m in matches if m.is_false_positive or (not m.is_missed and not m.is_correct))
    fn = sum(1 for m in matches if m.is_missed)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    token_f1 = (2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0.0)

    matched = [m for m in matches if not m.is_false_positive and not m.is_missed]
    row_ok = sum(1 for m in matched if m.predicted_row == m.gt.row_index)
    col_ok = sum(1 for m in matched if m.predicted_col == m.gt.col_index)
    n_matched = len(matched)

    row_accuracy = row_ok / n_matched if n_matched > 0 else 0.0
    col_accuracy = col_ok / n_matched if n_matched > 0 else 0.0
    exact_match = (tp == len(gt_symbols) and fp == 0 and fn == 0)

    return SceneScores(
        token_f1=token_f1,
        row_accuracy=row_accuracy,
        col_accuracy=col_accuracy,
        exact_match=exact_match,
        n_correct=tp,
        n_total=len(gt_symbols),
    )
