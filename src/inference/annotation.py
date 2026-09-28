from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.inference.gt_loader import GtSymbol

from src.inference.palette import (
    COARSE_DISPLAY_NAMES,
    GT_CORRECT,
    GT_EXTRA,
    GT_MISSED,
    GT_WRONG,
    LOW_CONF_COLOUR as _LOW_CONF_COLOUR,
    LOW_CONF_THRESHOLD as _LOW_CONF_THRESHOLD,
    ROLE_COLOURS as _ROLE_COLOURS,
    ROW_PALETTE as _ROW_PALETTE,
    display_glyph,
    display_name,
)


def _box_label(symbol: str, conf: float, idx: int, low: bool) -> str:
    """Clean, unique box/legend label: '#1 · 2 · 1.00'. The #index keeps every box
    unique so hover isolates one box; symbol is the glyph/name, not the raw class."""
    warn = " ⚠" if low else ""
    return f"#{idx} · {symbol} · {conf:.2f}{warn}"

# Map fine label prefix → coarse YOLO class
_FINE_TO_COARSE: dict[str, str] = {
    **{f"main_{i}": "digit_main" for i in range(10)},
    **{f"carry_{i}": "digit_carry" for i in range(10)},
    **{f"borrow_{i}": "digit_borrow" for i in range(10)},
    "op_plus": "operator", "op_minus": "operator",
    "op_times": "operator", "op_divide": "operator",
    "result_bar": "result_bar",
    "div_bracket": "divide_bracket",
}


def annotation_colour(fine_label: str, confidence: float) -> str:
    """Return hex colour for a bounding box given its fine label and confidence."""
    if confidence < _LOW_CONF_THRESHOLD:
        return _LOW_CONF_COLOUR
    coarse = _FINE_TO_COARSE.get(fine_label, "digit_main")
    return _ROLE_COLOURS.get(coarse, _ROLE_COLOURS["digit_main"])


def coarse_class(fine_label: str) -> str:
    """Map a fine label to its YOLO coarse class name."""
    return _FINE_TO_COARSE.get(fine_label, "digit_main")


def build_annotations(
    tokens: list[dict],
) -> tuple[list[tuple[tuple[int, int, int, int], str]], dict[str, str]]:
    """
    Convert flattened payload tokens to gr.AnnotatedImage format.

    tokens: list of dicts with keys label, confidence, bbox ([x0,y0,x1,y1]).

    Returns:
        annotations: list of ((x0,y0,x1,y1), label_string) tuples
        color_map:   dict mapping each label_string → hex colour
    """
    annotations: list[tuple[tuple[int, int, int, int], str]] = []
    color_map: dict[str, str] = {}

    # Reading order (row, col, then top-left) so the legend lists boxes top-to-bottom,
    # left-to-right and the #index follows the same order.
    ordered = sorted(
        tokens,
        key=lambda t: (t.get("row", 0), t.get("col", 0), t["bbox"][1], t["bbox"][0]),
    )
    for i, tok in enumerate(ordered, 1):
        label: str = tok["label"]
        conf: float = tok["confidence"]
        x0, y0, x1, y1 = tok["bbox"]
        bbox = (int(x0), int(y0), int(x1), int(y1))

        symbol = display_glyph(label) or display_name(label)
        label_str = _box_label(symbol, conf, i, conf < _LOW_CONF_THRESHOLD)
        colour = annotation_colour(label, conf)

        annotations.append((bbox, label_str))
        color_map[label_str] = colour

    return annotations, color_map


def build_yolo_annotations(
    tokens: list[dict],
) -> tuple[list[tuple[tuple[int, int, int, int], str]], dict[str, str]]:
    """Build coarse YOLO-class annotations — shows what Stage 1 detected."""
    annotations: list[tuple[tuple[int, int, int, int], str]] = []
    color_map: dict[str, str] = {}

    # Reading order by top-left (no row/col at the coarse stage).
    ordered = sorted(tokens, key=lambda t: (t["bbox"][1], t["bbox"][0]))
    for i, tok in enumerate(ordered, 1):
        conf: float = tok["confidence"]
        x0, y0, x1, y1 = tok["bbox"]
        bbox = (int(x0), int(y0), int(x1), int(y1))
        cls = coarse_class(tok["label"])
        name = COARSE_DISPLAY_NAMES.get(cls, cls)
        label_str = _box_label(name, conf, i, conf < _LOW_CONF_THRESHOLD)
        colour = _ROLE_COLOURS.get(cls, _ROLE_COLOURS["digit_main"])
        annotations.append((bbox, label_str))
        color_map[label_str] = colour

    return annotations, color_map


def build_cv_annotations(
    detections: list[dict],
) -> tuple[list[tuple[tuple[int, int, int, int], str]], dict[str, str]]:
    """Build annotations for the OpenCV-fusion extras (boxes YOLO missed).

    Each entry has keys: label (already a *coarse* YOLO class), confidence, bbox.
    Mirrors build_yolo_annotations but skips fine→coarse mapping since the CV
    branch tags detections with the coarse class assigned by the crop-classify pass.
    """
    annotations: list[tuple[tuple[int, int, int, int], str]] = []
    color_map: dict[str, str] = {}

    ordered = sorted(detections, key=lambda d: (d["bbox"][1], d["bbox"][0]))
    for i, det in enumerate(ordered, 1):
        conf = float(det.get("confidence", det.get("conf", 1.0)))
        x0, y0, x1, y1 = det["bbox"]
        bbox = (int(x0), int(y0), int(x1), int(y1))
        cls = str(det.get("label", "digit_main"))
        name = COARSE_DISPLAY_NAMES.get(cls, cls)
        label_str = _box_label(name, conf, i, conf < _LOW_CONF_THRESHOLD)
        colour = _ROLE_COLOURS.get(cls, _ROLE_COLOURS["digit_main"])
        annotations.append((bbox, label_str))
        color_map[label_str] = colour

    return annotations, color_map


def build_gt_annotations(
    tokens: list[dict],
    gt_symbols: list["GtSymbol"],
    iou_threshold: float = 0.4,
) -> tuple[list[tuple[tuple[int, int, int, int], str]], dict[str, str]]:
    """
    Build colour-coded annotations for GT comparison image.

    Greedy IoU matching (threshold=0.4):
    - Correct match (IoU >= threshold, fine_label matches): green, label "{label} ✓"
    - Wrong match (IoU >= threshold, fine_label differs): red, label "{predicted} ✗ → {expected}"
    - Missed GT (no predicted with IoU >= threshold): amber, label "missing: {expected}"
    - False positive (no GT with IoU >= threshold): orange, label "extra: {predicted}"
    """
    from src.inference.gt_loader import _iou  # lazy import to avoid circular dependency at module load
    annotations: list[tuple[tuple[int, int, int, int], str]] = []
    color_map: dict[str, str] = {}
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()

    # Build IoU matrix and greedily match
    pairs = []
    for pi, tok in enumerate(tokens):
        for gi, gt in enumerate(gt_symbols):
            score = _iou(tok["bbox"], gt.bbox)
            if score >= iou_threshold:
                pairs.append((score, pi, gi))
    pairs.sort(reverse=True)

    for _, pi, gi in pairs:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        tok = tokens[pi]
        gt = gt_symbols[gi]
        bbox = tuple(int(v) for v in tok["bbox"])
        if tok["label"] == gt.fine_label:
            label_str = f"{tok['label']} ✓"
            colour = GT_CORRECT
        else:
            label_str = f"{tok['label']} ✗ → {gt.fine_label}"
            colour = GT_WRONG
        annotations.append((bbox, label_str))  # type: ignore[arg-type]
        color_map[label_str] = colour

    # Missed GT
    for gi, gt in enumerate(gt_symbols):
        if gi not in matched_gt:
            bbox = tuple(int(v) for v in gt.bbox)
            label_str = f"missing: {gt.fine_label}"
            annotations.append((bbox, label_str))  # type: ignore[arg-type]
            color_map[label_str] = GT_MISSED

    # False positives
    for pi, tok in enumerate(tokens):
        if pi not in matched_pred:
            bbox = tuple(int(v) for v in tok["bbox"])
            label_str = f"extra: {tok['label']}"
            annotations.append((bbox, label_str))  # type: ignore[arg-type]
            color_map[label_str] = GT_EXTRA

    return annotations, color_map


def build_detection_annotations(
    detections: list[dict],
) -> tuple[list[tuple[tuple[int, int, int, int], str]], dict[str, str]]:
    """Build row/col-labelled annotations from payload detections[] entries.

    Each detection entry has keys: label, bbox, row, col, conf.
    Label string format: R<row>C<col> · <label> · <conf>
    Color is cycled by row cluster ID.

    This function is used when:
    - "Show row/col always" toggle is enabled (even on OOD scenes), or
    - equation_kind == "bare_digits" (rows populated from detections).
    """
    annotations: list[tuple[tuple[int, int, int, int], str]] = []
    color_map: dict[str, str] = {}

    ordered = sorted(detections, key=lambda d: (int(d.get("row", 0)), int(d.get("col", 0))))
    for i, det in enumerate(ordered, 1):
        row = int(det.get("row", 0))
        col = int(det.get("col", 0))
        label = str(det.get("label", "?"))
        conf = float(det.get("conf", 0.0))
        x0, y0, x1, y1 = det["bbox"]
        bbox = (int(x0), int(y0), int(x1), int(y1))
        colour = _ROW_PALETTE[row % len(_ROW_PALETTE)]
        symbol = display_glyph(label) or display_name(label)
        # Clean unique label with grid position: '#1 · 2 · R0C2 · 0.95'.
        warn = " ⚠" if conf < _LOW_CONF_THRESHOLD else ""
        label_str = f"#{i} · {symbol} · R{row}C{col} · {conf:.2f}{warn}"
        annotations.append((bbox, label_str))
        color_map[label_str] = colour

    return annotations, color_map
