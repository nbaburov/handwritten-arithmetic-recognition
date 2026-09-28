from __future__ import annotations

from typing import Dict, List, Tuple


def bbox_iou(a: List[float], b: List[float]) -> float:
    x0, y0, x1, y1 = a
    x2, y2, x3, y3 = b
    ix0 = max(x0, x2)
    iy0 = max(y0, y2)
    ix1 = min(x1, x3)
    iy1 = min(y1, y3)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_b = max(0.0, x3 - x2) * max(0.0, y3 - y2)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def greedy_match_scene(
    gt_tokens: List[dict],
    pred_tokens: List[dict],
    iou_thresh: float = 0.35,
) -> Dict[str, object]:
    pairs: List[Tuple[float, int, int]] = []
    for gi, g in enumerate(gt_tokens):
        for pi, p in enumerate(pred_tokens):
            score = bbox_iou(g["bbox_px"], p["bbox_px"])
            if score >= iou_thresh:
                pairs.append((score, gi, pi))
    pairs.sort(reverse=True)
    used_g: set[int] = set()
    used_p: set[int] = set()
    matched = 0
    label_ok = 0
    matched_pairs: List[Tuple[int, int, bool]] = []
    for score, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matched += 1
        ok = gt_tokens[gi]["flattened"] == pred_tokens[pi]["flattened"]
        if ok:
            label_ok += 1
        matched_pairs.append((gi, pi, ok))
    return {
        "matched": matched,
        "label_ok": label_ok,
        "gt_count": len(gt_tokens),
        "pred_count": len(pred_tokens),
        "pairs": matched_pairs,
    }
