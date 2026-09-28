from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# Synthetic render: carry/borrow annotations smaller than main-line digits.
CARRY_BORROW_TOKEN_SCALE: float = 0.62
MAIN_TOKEN_SCALE: float = 1.0
OPERATOR_TOKEN_SCALE: float = 0.92

# Operands and results stay within 1-3 decimal digits; sums/products capped at 999.
MAX_SYNTH_VALUE: int = 999


class EquationKind(str, Enum):
    add = "add"
    subtract = "subtract"
    multiply = "multiply"
    divide = "divide"
    # Sentinel for OOD scenes -- never used as a GNN eq_type target (masked to -1 in train_gnn.py).
    ood_unknown = "ood_unknown"


class SceneCase(str, Enum):
    addition = "addition"
    subtraction = "subtraction"
    multiplication_simple = "multiplication-simple"
    multiplication_multi = "multiplication-multi"
    division_short = "division-short"
    division_long = "division-long"
    division_simple = "division-simple"
    # iter7 Workstream B: new layout variants
    addition_op_right = "addition_op_right"
    addition_no_bar = "addition_no_bar"
    subtraction_op_right = "subtraction_op_right"
    subtraction_no_bar = "subtraction_no_bar"
    multiplication_simple_op_right = "multiplication_simple_op_right"
    multiplication_simple_no_bar = "multiplication_simple_no_bar"
    # MC-3: 3-digit minus 3-digit with 2-3 forced borrows
    subtraction_heavy_borrow = "subtraction_heavy_borrow"
    # MC-4: addition with all columns carrying vs zero-carry baseline
    addition_dense_carries = "addition_dense_carries"
    addition_no_carries = "addition_no_carries"
    # iter11 (2026-06-22): crowded double-digit-in-one-cell variants (borrow "16" / carry "1X").
    subtraction_crowded_borrow = "subtraction_crowded_borrow"
    addition_crowded_carry = "addition_crowded_carry"
    # iter10 W10-ROBUSTNESS: isolation and bare-number cases
    bare_digits = "bare_digits"
    standalone_bar = "standalone_bar"
    standalone_bracket = "standalone_bracket"
    # Multi-row, multi-column block of bare main digits (no operator/bar/bracket).
    # Covers the row/col-clustering gap that single-row bare_digits cannot exercise.
    bare_digit_grid = "bare_digit_grid"
    # iter7 Workstream D: hard-negative OOD scenes


@dataclass(frozen=True)
class LayoutToken:
    row: int
    col: int
    flattened_label: str
    glyph_key: str
    yolo_class_name: str
    token_scale: float = MAIN_TOKEN_SCALE
    # iter11 (2026-06-22): sub-cell horizontal offset as a fraction of the column pitch, used to pack
    # two digits into ONE cell (same row/col) for crowded borrow/carry, e.g. "16". 0.0 = centred.
    # Placement adds cell_subpos * x_step to the glyph centre; row/col GT is unchanged (still the cell).
    cell_subpos: float = 0.0
    # Debug metadata only -- not a GNN supervision signal.
    segment_kind: Optional[str] = field(default=None, compare=False, hash=False)
