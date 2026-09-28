"""Classical-OpenCV complementary detector fused into the inference path.

YOLO is trained on *finished* scenes and misses handwritten symbols in
*unfinished* ones. This module runs a connected-component detector over the
regions YOLO did **not** claim, reclassifies the extras with a single batched
YOLO pass, and merges them into YOLO's detections. It augments YOLO; it never
replaces it. The GNN, output schema, and training code are untouched.

Per-image budget contributed here: 0 or 1 batched YOLO call (the crop-classify
pass), made only when the CV detector finds unclaimed boxes.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from ..core.ontology import yolo_class_id_to_name
from ..core.run_config import CvFusionConfig
from ..parsing.detection import Detection
from ..parsing.match_tokens import bbox_iou

logger = logging.getLogger(__name__)

BBox = List[int]


# ── Classical CV symbol detector with proximity merging ───────────────────────

def detect_symbols_cv(
    gray: np.ndarray,
    min_area: int = 30,
    min_side: int = 4,
    pad: int = 4,
    merge_overlap_ratio: float = 0.5,
    proximity_merge_factor: float = 0.25,
) -> Tuple[List[BBox], List[float]]:
    """Find handwritten symbols via connected-component analysis.

    Adapted to take the pipeline's (H, W) uint8 grayscale array directly. No
    horizontal-line erasure and no dilation: line erasure was deleting minus
    signs and fraction bars, and dilation smudged tightly-written numbers into
    one component. Returns xyxy boxes (absolute pixels) and unit confidences.
    """
    H, W = gray.shape[:2]
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # NOTE: horizontal-line subtraction intentionally omitted (erased minus signs
    # and fraction bars). Dilation intentionally omitted (smudged "50000" into one
    # blob). For extremely thin/broken pen strokes, a light 2x2 dilate could be
    # reintroduced here.

    n_labels, _labels, stats, _cents = cv2.connectedComponentsWithStats(binary, connectivity=8)

    raw: List[BBox] = []
    for i in range(1, n_labels):
        x, y, w, h, area = stats[i]
        # Filter on pixel area + the LARGER extent (not min(w, h)) so a legitimately
        # thin stroke — a "1", a "+" arm — survives, while speckle is still dropped
        # BEFORE the proximity merge (keeps dense digits from bridging into one blob).
        if area < min_area or max(w, h) < min_side:
            continue
        # L-3: drop components whose aspect ratio exceeds 10:1 — these are almost
        # always horizontal result bars or division brackets, which YOLO detects
        # directly as structural tokens. Letting them through produces phantom GNN
        # nodes that conflict with the bar/bracket YOLO already found. This
        # assumption holds as long as bars and brackets are YOLO's responsibility;
        # if that changes, revisit this filter.
        if max(w, h) / max(min(w, h), 1) > 10:
            continue
        raw.append([int(x), int(y), int(x + w), int(y + h)])

    def _diag(b: BBox) -> float:
        return ((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5

    def _gap(a: BBox, b: BBox) -> float:
        dx = max(0, max(a[0] - b[2], b[0] - a[2]))
        dy = max(0, max(a[1] - b[3], b[1] - a[3]))
        return (dx ** 2 + dy ** 2) ** 0.5

    def _should_merge(a: BBox, b: BBox) -> bool:
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
        if smaller > 0 and inter / smaller > merge_overlap_ratio:
            return True
        g = _gap(a, b); sd = min(_diag(a), _diag(b))
        # Only merge if extremely close (proximity_merge_factor of the diagonal).
        return sd > 0 and g / sd < proximity_merge_factor

    # M-3 fix: cap iterations to len(raw)+1 to guard against degenerate geometry
    # that would otherwise loop forever (each iteration merges at least one pair,
    # so the true upper bound is len(raw)-1; the +1 gives one extra safety margin).
    max_iters = len(raw) + 1
    changed = True
    iters = 0
    while changed:
        if iters >= max_iters:
            logger.warning(
                "detect_symbols_cv: merge loop hit cap (%d iters, %d boxes remaining); stopping",
                iters, len(raw),
            )
            break
        iters += 1
        changed = False
        merged: List[BBox] = []
        used = [False] * len(raw)
        for i in range(len(raw)):
            if used[i]:
                continue
            x1, y1, x2, y2 = raw[i]
            for j in range(i + 1, len(raw)):
                if used[j]:
                    continue
                if _should_merge([x1, y1, x2, y2], raw[j]):
                    x1 = min(x1, raw[j][0]); y1 = min(y1, raw[j][1])
                    x2 = max(x2, raw[j][2]); y2 = max(y2, raw[j][3])
                    used[j] = True; changed = True
            merged.append([x1, y1, x2, y2]); used[i] = True
        raw = merged

    padded: List[BBox] = [
        [max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)]
        for x1, y1, x2, y2 in raw
    ]
    return padded, [1.0] * len(padded)


def flatten_to_white_paper(gray: np.ndarray, bright_thresh: int = 200) -> np.ndarray:
    """Flatten a dark photo surround (black border / background) to white while
    keeping the white paper and its dark text.

    Built for phone photos where a white sheet sits on a dark surface: after
    longest-edge resize + white-pad the result is `white pad → black frame →
    white paper → dark text`, which is out-of-distribution for YOLO and the GNN
    (trained on pure-white scenes). This finds the largest *interior* bright
    region (the paper sheet — the padding ring touches the image border and is
    skipped) and sets everything outside its bounding box to white.
    """
    h, w = gray.shape[:2]
    bright = (gray >= bright_thresh).astype(np.uint8)
    n, _labels, stats, _c = cv2.connectedComponentsWithStats(bright, connectivity=8)
    if n <= 1:
        return gray
    # The paper sheet is the largest bright region that does NOT touch the image
    # border (the border-touching bright ring is the white padding / clean-scene
    # background, which we must not key off). A digit's inner hole is also interior
    # but tiny, so the size guard below stops it from ever triggering a wipe.
    best = None
    best_area = 0
    for i in range(1, n):
        x, y, ww, hh = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                        int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        if x == 0 or y == 0 or x + ww >= w or y + hh >= h:
            continue  # white padding ring / background — not the paper
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area; best = (x, y, ww, hh)
    # No sizeable interior bright region ⇒ not a paper-on-dark-surround photo ⇒
    # leave the image untouched (this is what makes clean scenes a safe no-op).
    if best is None or best[2] * best[3] < 0.15 * w * h:
        return gray
    x, y, ww, hh = best
    out = np.full_like(gray, 255)
    out[y:y + hh, x:x + ww] = gray[y:y + hh, x:x + ww]
    return out


# ── Fusion helpers ────────────────────────────────────────────────────────────

def _mask_yolo_regions(
    gray: np.ndarray,
    yolo_dets: List[Detection],
    dilate_px: int = 5,
) -> np.ndarray:
    """Return a copy of ``gray`` with every (dilated) YOLO bbox whited out so the
    CV detector ignores regions YOLO already claimed."""
    h, w = gray.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for d in yolo_dets:
        x0 = max(0, int(math.floor(d.x0)))
        y0 = max(0, int(math.floor(d.y0)))
        x1 = min(w, int(math.ceil(d.x1)))
        y1 = min(h, int(math.ceil(d.y1)))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=1)
    masked = gray.copy()
    masked[mask > 0] = 255  # white = background
    return masked


def _tight_mask_yolo_ink(
    gray: np.ndarray,
    yolo_dets: List[Detection],
    dilate_px: int = 4,
    symbol_component_frac: float = 0.5,
) -> np.ndarray:
    """White out ONLY YOLO's actual detected symbol inside each bbox, leaving
    smaller *separate* ink (e.g. a carry digit that fell inside a loose bbox).

    For each YOLO bbox: Otsu-threshold the crop, find connected components, and
    mask only the components that are a large fraction (``symbol_component_frac``)
    of the largest one — i.e. the symbol YOLO actually detected. A carry sitting
    inside a loose bbox is a SMALLER, separate component, so it is NOT masked and
    survives for the CV detector to find. (Masking the whole Otsu crop would have
    erased the carry too, since it is just more dark ink inside the box.)"""
    h, w = gray.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for d in yolo_dets:
        x0 = max(0, int(math.floor(d.x0)))
        y0 = max(0, int(math.floor(d.y0)))
        x1 = min(w, int(math.ceil(d.x1)))
        y1 = min(h, int(math.ceil(d.y1)))
        if x1 <= x0 or y1 <= y0:
            continue
        crop = gray[y0:y1, x0:x1]
        _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if n <= 1:
            continue
        largest = int(stats[1:, cv2.CC_STAT_AREA].max())
        keep_symbol = np.zeros(binary.shape, dtype=bool)
        for ci in range(1, n):
            if stats[ci, cv2.CC_STAT_AREA] >= symbol_component_frac * largest:
                keep_symbol |= (labels == ci)
        region = mask[y0:y1, x0:x1]
        region[keep_symbol] = 255
        mask[y0:y1, x0:x1] = region
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=1)
    masked = gray.copy()
    masked[mask > 0] = 255
    return masked


def run_yolo_second_pass(
    gray: np.ndarray,
    yolo_dets: List[Detection],
    yolo_model: Any,
    *,
    conf: float,
    iou: float,
    agnostic_nms: bool,
    max_det: int,
    dilate_px: int,
) -> Tuple[List[Detection], Dict[str, Any]]:
    """Mask YOLO's ink pixels in the original image, run YOLO again on the
    masked image, and return (extra_detections, info)."""
    from .run import _run_yolo

    masked = _tight_mask_yolo_ink(gray, yolo_dets, dilate_px=dilate_px)
    detections = _run_yolo(
        masked, yolo_model, conf,
        iou=iou, agnostic_nms=agnostic_nms, max_det=max_det,
    )
    info: Dict[str, Any] = {
        "second_pass_conf": conf,
        "second_pass_dets": len(detections),
        "detections": [_detection_to_dict(d) for d in detections],
    }
    return detections, info


def _letterbox_crop(gray: np.ndarray, box: BBox, size: int = 512) -> np.ndarray:
    """Crop ``box`` and letterbox it to ``size``x``size`` (aspect-preserving resize
    with white padding), returned as a 3-channel RGB array for YOLO.

    Risk note: a single small symbol blown up to fill ``size`` is larger than the
    ~30-100px symbols YOLO saw at train time; if crop reclassification underfires,
    revisit this (e.g. native-scale paste) — exposed as ``letterbox_size`` config.
    """
    x0 = max(0, int(box[0])); y0 = max(0, int(box[1]))
    x1 = min(gray.shape[1], int(box[2])); y1 = min(gray.shape[0], int(box[3]))
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        crop = np.full((1, 1), 255, dtype=np.uint8)
    ch, cw = crop.shape[:2]
    scale = size / max(ch, cw)
    nh = max(1, int(round(ch * scale)))
    nw = max(1, int(round(cw * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size), 255, dtype=np.uint8)
    oy = (size - nh) // 2
    ox = (size - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return np.stack([canvas, canvas, canvas], axis=-1)


def _classify_crops_with_yolo(
    gray: np.ndarray,
    cv_boxes: List[BBox],
    yolo_model: Any,
    *,
    classify_conf: float,
    letterbox_size: int,
    classify_with_yolo: bool = True,
    keep_unclassified: bool = True,
    fallback_label: str = "digit_main",
    unclassified_conf: float = 0.20,
    keep_min_ink_frac: float = 0.10,
    min_area: int = 30,
) -> Tuple[List[Detection], int]:
    """Reclassify CV boxes with a SINGLE batched YOLO call. Each box keeps its own
    full-image coordinates; only the class label + confidence come from YOLO.

    A box YOLO can't classify (nothing returned, or below ``classify_conf``) is the
    common case for symbols YOLO already missed full-image (e.g. an isolated "+").
    Rather than drop it, when ``keep_unclassified`` is set it is kept with
    ``fallback_label`` @ ``unclassified_conf`` so the GNN — the actual fine-label
    classifier — labels it. When ``classify_with_yolo`` is False the YOLO call is
    skipped entirely (0 extra YOLO calls) and every box goes to the GNN as fallback.

    H-1 floor: before a box is kept as fallback, it must pass an ink-density gate:
    the fraction of dark pixels (below Otsu threshold) within the box must be >=
    ``keep_min_ink_frac`` (default 0.10) AND the box area must be >= ``min_area``.
    This prevents speckle from becoming phantom GNN nodes. Real carries fill well
    over 10% of their tight bounding box so they survive the floor. Set
    ``keep_min_ink_frac=0.0`` to disable the floor entirely.

    Returns ``(detections, n_fallback)``.
    """
    if not cv_boxes:
        return [], 0

    def _ink_fraction(box: BBox) -> float:
        """Compute fraction of dark pixels inside box on the 512 gray image.
        Uses a fixed threshold of 128 (pixels below 128 are considered dark ink)."""
        x0, y0, x1, y1 = box
        x0i, y0i, x1i, y1i = int(x0), int(y0), int(x1), int(y1)
        h_img, w_img = gray.shape[:2]
        x0i = max(0, x0i); y0i = max(0, y0i)
        x1i = min(w_img, x1i); y1i = min(h_img, y1i)
        if x1i <= x0i or y1i <= y0i:
            return 0.0
        roi = gray[y0i:y1i, x0i:x1i]
        total = roi.size
        if total == 0:
            return 0.0
        # Pixels below 128 are considered dark (ink) — conservative threshold
        # to avoid Otsu failure on near-empty ROIs.
        dark = int(np.sum(roi < 128))
        return dark / total

    def _passes_floor(box: BBox) -> bool:
        """Return True if the box has enough ink to be a real symbol, not speckle."""
        if keep_min_ink_frac <= 0.0:
            return True  # floor disabled
        x0, y0, x1, y1 = box
        area = (x1 - x0) * (y1 - y0)
        if area < min_area:
            return False
        return _ink_fraction(box) >= keep_min_ink_frac

    def _fallback(box: BBox) -> Detection:
        x0, y0, x1, y1 = box
        return Detection(label=fallback_label, confidence=unclassified_conf,
                         x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1))

    if not classify_with_yolo:
        if not keep_unclassified:
            return [], 0
        kept = [_fallback(b) for b in cv_boxes if _passes_floor(b)]
        return kept, len(kept)

    batch = [_letterbox_crop(gray, box, size=letterbox_size) for box in cv_boxes]
    results = yolo_model.predict(batch, conf=classify_conf, verbose=False) or []
    dets: List[Detection] = []
    n_fallback = 0
    for idx, box in enumerate(cv_boxes):
        res = results[idx] if idx < len(results) else None
        boxes = getattr(res, "boxes", None) if res is not None else None
        if boxes is not None and len(boxes) > 0:
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            best = int(confs.argmax())
            best_conf = float(confs[best])
            if best_conf >= classify_conf:
                label = yolo_class_id_to_name(int(cls_ids[best])) or fallback_label
                x0, y0, x1, y1 = box
                dets.append(Detection(label=label, confidence=best_conf,
                                      x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1)))
                continue
        if keep_unclassified and _passes_floor(box):
            dets.append(_fallback(box))
            n_fallback += 1
    return dets, n_fallback


def _merge_detections_nms(
    yolo_dets: List[Detection],
    cv_dets: List[Detection],
    iou_thresh: float = 0.3,
    use_ioa: bool = True,
) -> List[Detection]:
    """Class-agnostic greedy NMS over the YOLO+CV union. YOLO detections are
    considered first (each list sorted by confidence), so an overlapping CV
    detection is suppressed — i.e. ties prefer the original YOLO detection.

    ``use_ioa``: when True, also suppress via Intersection-over-smaller-Area —
    drops CV slivers fully inside a YOLO box. Set False when a small CV box that
    sits *inside* a loose YOLO box is a LEGITIMATE separate symbol (e.g. a carry
    inside a loose digit box); then only true IoU overlap suppresses."""
    ordered = (
        sorted(yolo_dets, key=lambda d: d.confidence, reverse=True)
        + sorted(cv_dets, key=lambda d: d.confidence, reverse=True)
    )

    def _overlap_exceeds(b1: List[float], b2: List[float]) -> bool:
        # Check standard IoU
        if bbox_iou(b1, b2) > iou_thresh:
            return True
        if not use_ioa:
            return False
        # Also check IoA (Intersection over Area of smaller box)
        # to suppress CV slivers inside YOLO boxes or vice versa.
        x0, y0, x1, y1 = b1
        x2, y2, x3, y3 = b2
        ix0 = max(x0, x2); iy0 = max(y0, y2)
        ix1 = min(x1, x3); iy1 = min(y1, y3)
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        area1 = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        area2 = max(0.0, x3 - x2) * max(0.0, y3 - y2)
        smaller = min(area1, area2)
        if smaller > 0 and (inter / smaller) > iou_thresh:
            return True
        return False

    kept: List[Detection] = []
    for det in ordered:
        box = [det.x0, det.y0, det.x1, det.y1]
        if all(not _overlap_exceeds(box, [k.x0, k.y0, k.x1, k.y1]) for k in kept):
            kept.append(det)
    return kept


def _detection_to_dict(d: Detection) -> Dict[str, Any]:
    return {"label": d.label, "confidence": d.confidence,
            "bbox": [d.x0, d.y0, d.x1, d.y1]}


def _dump_cv_debug(
    gray: np.ndarray,
    source: np.ndarray,
    yolo_dets: List[Detection],
    cv_boxes: List[BBox],
    cv_added: List[Detection],
) -> None:
    """Best-effort debug dump to /tmp/cv_debug when AK_CV_DEBUG=1: writes the input,
    the masked leftover the CV detector actually sees, and an overlay of YOLO boxes
    (red), all CV candidate boxes (blue), and CV detections kept after merge (green)."""
    import os
    if os.environ.get("AK_CV_DEBUG") != "1":
        return
    try:
        out = "/tmp/cv_debug"
        os.makedirs(out, exist_ok=True)
        cv2.imwrite(f"{out}/00_input_gray.png", gray)
        cv2.imwrite(f"{out}/01_masked_leftover.png", source)
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for d in yolo_dets:  # red = YOLO
            cv2.rectangle(overlay, (int(d.x0), int(d.y0)), (int(d.x1), int(d.y1)), (0, 0, 255), 1)
        for b in cv_boxes:   # blue = every CV candidate box
            cv2.rectangle(overlay, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 1)
        for d in cv_added:   # green = CV kept after merge
            cv2.rectangle(overlay, (int(d.x0), int(d.y0)), (int(d.x1), int(d.y1)), (0, 200, 0), 2)
        cv2.imwrite(f"{out}/02_overlay.png", overlay)
        logger.info("cv_debug dumped to %s (yolo=%d cv_candidates=%d cv_kept=%d)",
                    out, len(yolo_dets), len(cv_boxes), len(cv_added))
    except Exception as exc:  # never break inference for a debug dump
        logger.warning("cv_debug dump failed: %s", exc)


def run_cv_fusion(
    gray: np.ndarray,
    yolo_dets: List[Detection],
    yolo_model: Any,
    cfg: CvFusionConfig,
) -> Tuple[List[Detection], Dict[str, Any]]:
    """Run CV detector and optional second YOLO pass, merge with first-pass YOLO.

    A single NMS pass over (yolo_dets, cv_dets + second_dets) resolves all
    duplicates in one step: YOLO detections are sorted first so ties always
    prefer the first-pass YOLO detection over any CV/second-pass candidate.

    Returns ``(merged, info)`` where ``merged`` is all detections for the GNN
    and ``info`` surfaces both paths' results for GUI inspection.
    """
    info: Dict[str, Any] = {"mode": cfg.mode}

    # Path 1: Classical CV detector
    # When masking, use TIGHT-pixel masking (only YOLO's actual ink, not the loose
    # rectangle) so a carry that falls inside a loose YOLO bbox but outside the
    # symbol's own ink survives into the leftover for the CV detector to find.
    if cfg.mask_yolo:
        if cfg.cv_tight_mask:
            source = _tight_mask_yolo_ink(gray, yolo_dets, dilate_px=cfg.tight_mask_dilate_px)
        else:
            source = _mask_yolo_regions(gray, yolo_dets, dilate_px=cfg.mask_dilate_px)
    else:
        source = gray
    cv_boxes, _ = detect_symbols_cv(
        source,
        min_area=cfg.min_area,
        min_side=cfg.min_side,
        pad=cfg.pad,
        merge_overlap_ratio=cfg.merge_overlap_ratio,
        proximity_merge_factor=cfg.proximity_merge_factor,
    )
    cv_dets, n_fallback_cv = _classify_crops_with_yolo(
        gray, cv_boxes, yolo_model,
        classify_conf=cfg.classify_conf,
        letterbox_size=cfg.letterbox_size,
        classify_with_yolo=cfg.classify_with_yolo,
        keep_unclassified=cfg.keep_unclassified,
        fallback_label=cfg.fallback_label,
        unclassified_conf=cfg.unclassified_conf,
        keep_min_ink_frac=cfg.keep_min_ink_frac,
        min_area=cfg.min_area,
    )

    # Path 2: Second YOLO pass (if enabled)
    second_dets: List[Detection] = []
    if cfg.use_second_yolo_pass:
        second_dets, _ = run_yolo_second_pass(
            gray, yolo_dets, yolo_model,
            conf=cfg.second_pass_conf,
            iou=cfg.nms_iou,
            agnostic_nms=False,  # iter5: agnostic_nms=True caused cross-class overlap regressions
            max_det=80,
            dilate_px=cfg.tight_mask_dilate_px,
        )

    # When component-aware tight masking removed YOLO's actual ink, a surviving CV
    # box inside a loose YOLO bbox is a genuine miss (a carry), NOT a duplicate
    # sliver — so disable IoA suppression and let only true IoU overlap dedupe.
    merge_use_ioa = not cfg.cv_tight_mask

    # Single NMS pass: YOLO first so ties always prefer first-pass YOLO detections.
    # Both CV and second-YOLO extras are passed together; any cross-path duplicates
    # are suppressed in the same pass without a separate third NMS call.
    merged = _merge_detections_nms(
        yolo_dets,
        cv_dets + second_dets,
        iou_thresh=cfg.nms_iou,
        use_ioa=merge_use_ioa,
    )

    # Derive per-path "added" lists from the single merged result using value
    # equality (Detection is a frozen dataclass). A detection is "added by CV" if
    # it appears in merged, was produced by the CV path, and is not in yolo_dets.
    # Same logic for second-YOLO. This avoids brittle object-identity scans.
    yolo_set = set(yolo_dets)
    cv_set = set(cv_dets)
    second_set = set(second_dets)
    cv_added = [d for d in merged if d not in yolo_set and d in cv_set]
    second_added = [d for d in merged if d not in yolo_set and d in second_set]

    info.update({
        "cv_detector": {
            "cv_boxes": len(cv_boxes),
            "cv_fallback": n_fallback_cv,
            "added": len(cv_added),
            "detections": [_detection_to_dict(d) for d in cv_added],
        },
    })

    if cfg.use_second_yolo_pass:
        info.update({
            "second_yolo": {
                "second_pass_conf": cfg.second_pass_conf,
                "second_pass_dets": len(second_dets),
                "added": len(second_added),
                "detections": [_detection_to_dict(d) for d in second_added],
            },
        })

    logger.debug(
        "cv_fusion: yolo=%d cv_dets=%d (+%d) second_yolo=%d (+%d) total=%d",
        len(yolo_dets), len(cv_dets), len(cv_added), len(second_dets), len(second_added), len(merged),
    )
    _dump_cv_debug(gray, source, yolo_dets, cv_boxes, cv_added + second_added)

    return merged, info
