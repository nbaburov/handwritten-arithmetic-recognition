from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Tuple

YOLO_CLASS_NAMES: Tuple[str, ...] = (
    "digit_main",
    "digit_carry",
    "digit_borrow",
    "operator",
    "result_bar",
    "divide_bracket",
)

YOLO_NAME_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(YOLO_CLASS_NAMES)}


def yolo_class_id_to_name(cls_id: int) -> Optional[str]:
    if 0 <= cls_id < len(YOLO_CLASS_NAMES):
        return YOLO_CLASS_NAMES[cls_id]
    return None

LEGACY_GLY_TO_OP_FLAT: Dict[str, str] = {
    "plus": "op_plus",
    "minus": "op_minus",
    "times": "op_times",
    "divide": "op_divide",
}

STRUCTURE_FLAT_LABELS: FrozenSet[str] = frozenset({"result_bar"})


def stage2_labels_ordered() -> List[str]:
    out: List[str] = []
    for role in ("main", "carry", "borrow"):
        for d in "0123456789":
            out.append(f"{role}_{d}")
    for g in ("op_plus", "op_minus", "op_times", "op_divide"):
        out.append(g)
    out.append("result_bar")
    out.append("div_bracket")      # fine label for long division anchor (distinct from YOLO class "divide_bracket")
    return out


def gnn_fine_labels_ordered() -> List[str]:
    """GNN fine-label vocabulary (16 classes) used post-A2 ontology refactor.

    Collapses the 30 main/carry/borrow trios into 10 role-agnostic digits.
    Role information is supplied at inference time by the YOLO coarse class
    (digit_main / digit_carry / digit_borrow), and recombined in the assembler
    via `full_label_from_gnn_and_yolo`.
    """
    out: List[str] = []
    for d in "0123456789":
        out.append(d)
    for g in ("op_plus", "op_minus", "op_times", "op_divide"):
        out.append(g)
    out.append("result_bar")
    out.append("div_bracket")
    return out


def gnn_label_from_full(full: str) -> Optional[str]:
    """Map a 36-class fine label string (e.g. "main_5", "carry_3") to its
    16-class GNN label ("5", "3", ...). Returns None if input is unknown.
    """
    role, detail = parse_flattened_label(full)
    if role in {"main", "carry", "borrow"} and detail is not None:
        return detail
    if role == "operator" and detail is not None:
        return f"op_{detail}"
    if role == "structure":
        return full
    if full == "div_bracket":
        return "div_bracket"
    return None


def full_label_from_gnn_and_yolo(gnn_label: str, yolo_coarse: str) -> str:
    """Combine a 16-class GNN prediction with a YOLO coarse class to produce
    the original 36-class fine label string.

    Examples:
      ("5", "digit_main")    -> "main_5"
      ("5", "digit_carry")   -> "carry_5"
      ("op_plus", "operator") -> "op_plus"
      ("result_bar", _)      -> "result_bar"
      ("div_bracket", "divide_bracket") -> "div_bracket"
    """
    if len(gnn_label) == 1 and gnn_label.isdigit():
        if yolo_coarse == "digit_carry":
            return f"carry_{gnn_label}"
        if yolo_coarse == "digit_borrow":
            return f"borrow_{gnn_label}"
        return f"main_{gnn_label}"
    if gnn_label.startswith("op_"):
        return gnn_label
    if gnn_label == "result_bar":
        return "result_bar"
    if gnn_label == "div_bracket":
        return "div_bracket"
    return gnn_label


def parse_flattened_label(flat: str) -> Tuple[str, Optional[str]]:
    if flat in STRUCTURE_FLAT_LABELS:
        return "structure", flat
    if len(flat) == 1 and flat.isdigit():
        return "main", flat
    if flat in LEGACY_GLY_TO_OP_FLAT:
        return "operator", flat
    if flat.startswith("op_"):
        return "operator", flat[3:]
    parts = flat.split("_", 1)
    if len(parts) != 2:
        return "unknown", None
    role, digit = parts[0], parts[1]
    if role in {"main", "carry", "borrow"} and len(digit) == 1 and digit.isdigit():
        return role, digit
    return "unknown", None


def flattened_to_yolo_class_name(flat: str) -> str:
    role, _ = parse_flattened_label(flat)
    if role == "main":
        return "digit_main"
    if role == "carry":
        return "digit_carry"
    if role == "borrow":
        return "digit_borrow"
    if role == "operator":
        return "operator"
    if role == "structure":
        if flat == "result_bar":
            return "result_bar"
    raise ValueError(f"Unknown flattened label for YOLO mapping: {flat}")


def glyph_key_for_pool(flat: str) -> str:
    role, detail = parse_flattened_label(flat)
    if role == "structure":
        return flat
    if role == "operator":
        mapping = {"plus": "plus", "minus": "minus", "times": "times", "divide": "divide"}
        return mapping.get(detail or "", detail or "")
    if role in {"main", "carry", "borrow"}:
        return detail or ""
    raise ValueError(f"Cannot resolve glyph pool key for: {flat}")


def default_required_stage2_labels() -> FrozenSet[str]:
    return frozenset(stage2_labels_ordered())


def digit_pool_folder_labels() -> FrozenSet[str]:
    """Return the folder names that must exist under data/raw/pool_emnist_28/ for the synthesis pipeline to work.

    Excludes carry/borrow (synthetic role assignment, no physical folders) and result_bar
    (drawn programmatically via strokes.py — no real crop folder needed).
    Only main digits and operators require physical crop folders (14 total).
    """
    main_digits: FrozenSet[str] = frozenset(f"main_{d}" for d in "0123456789")
    operators: FrozenSet[str] = frozenset(LEGACY_GLY_TO_OP_FLAT.values())
    return main_digits | operators


def safe_flattened_to_yolo_class_name(lab: str) -> str:
    try:
        return flattened_to_yolo_class_name(lab)
    except ValueError:
        if len(lab) == 1 and lab.isdigit():
            return "digit_main"
        if lab in LEGACY_GLY_TO_OP_FLAT:
            return "operator"
        if lab in STRUCTURE_FLAT_LABELS:
            return lab
        return "digit_main"


# Folders under data/raw/pool_emnist_28/ with this label name are ignored (legacy long-division asset).
EXCLUDED_DATASET_LABELS: FrozenSet[str] = frozenset()
