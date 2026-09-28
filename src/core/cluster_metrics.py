"""Permutation-invariant cluster accuracy helpers shared by training and evaluation."""

from __future__ import annotations

from typing import Sequence

from sklearn.metrics import adjusted_rand_score


def permutation_invariant_cluster_accuracy(
    pred_ids: Sequence[int],
    gt_ids: Sequence[int],
) -> float:
    """Compare two cluster-id sequences with permutation invariance.

    Uses sklearn's adjusted_rand_score, which corrects for chance grouping and
    returns 1.0 for any relabelling of the same partition, 0.0 for random,
    and can be slightly negative for worse-than-random. We clip to [0.0, 1.0]
    so it can stand in for an accuracy-like metric in run.json.

    Returns 0.0 when either input is empty or has fewer than 2 elements
    (ARI is undefined for single-point clusters).

    Args:
        pred_ids: Predicted cluster assignment per node (0-indexed, scan order).
        gt_ids: Ground-truth cluster assignment per node (any integer scheme).

    Returns:
        Clipped ARI in [0.0, 1.0].

    Raises:
        ValueError: If pred_ids and gt_ids have different lengths.
    """
    if len(pred_ids) != len(gt_ids):
        raise ValueError(
            f"pred_ids and gt_ids must have the same length, "
            f"got {len(pred_ids)} vs {len(gt_ids)}"
        )
    if len(pred_ids) < 2:
        return 0.0
    return max(0.0, float(adjusted_rand_score(gt_ids, pred_ids)))


def compute_scene_metrics(
    preds: "list[tuple[str, int, int]]",
    gts: "list[tuple[str, int, int]]",
    eq_pred: str,
    eq_gt: str,
) -> "dict[str, float]":
    """Compare predictions vs ground truth for a single scene.

    Args:
        preds: list of (fine_label, row_idx, col_idx) tuples
        gts:   list of (fine_label, row_idx, col_idx) tuples
        eq_pred: predicted equation type string
        eq_gt:   ground truth equation type string

    Returns:
        Dict with keys: fine_label_accuracy, row_accuracy, col_accuracy,
                        equation_type_accuracy, exact_scene_match
    """
    n = len(preds)
    if n == 0 or n != len(gts):
        return {k: 0.0 for k in
                ("fine_label_accuracy", "row_accuracy", "col_accuracy",
                 "equation_type_accuracy", "exact_scene_match")}
    fine_ok = sum(p[0] == g[0] for p, g in zip(preds, gts))
    row_ok = sum(p[1] == g[1] for p, g in zip(preds, gts))
    col_ok = sum(p[2] == g[2] for p, g in zip(preds, gts))
    return {
        "fine_label_accuracy":    fine_ok / n,
        "row_accuracy":           row_ok / n,
        "col_accuracy":           col_ok / n,
        "equation_type_accuracy": 1.0 if eq_pred == eq_gt else 0.0,
        "exact_scene_match":      1.0 if all(p == g for p, g in zip(preds, gts)) else 0.0,
    }
