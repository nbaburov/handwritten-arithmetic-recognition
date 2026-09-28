from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple, assert_never

# ---------------------------------------------------------------------------
# Re-exports from sub-modules: callers importing from layouts get the same
# public surface as before the split.
# ---------------------------------------------------------------------------
from .layouts_types import (  # noqa: F401
    CARRY_BORROW_TOKEN_SCALE,
    MAIN_TOKEN_SCALE,
    OPERATOR_TOKEN_SCALE,
    MAX_SYNTH_VALUE,
    EquationKind,
    LayoutToken,
    SceneCase,
)
from .layouts_core import (  # noqa: F401
    DivStep,
    _div_steps,
    _mul_single_pass_carries,
    _sample_division_layout,
    layout_addition,
    layout_addition_no_bar,
    layout_addition_crowded_carry,
    layout_addition_dense_carries,
    layout_addition_no_carries,
    layout_bare_digits,
    layout_bare_digit_grid,
    layout_long_division,
    layout_multiplication,
    layout_multiplication_simple_no_bar,
    layout_multiplication_simple_op_right,
    layout_short_division,
    layout_simple_division,
    layout_standalone_bar,
    layout_standalone_bracket,
    layout_subtraction,
    layout_subtraction_no_bar,
    layout_subtraction_crowded_borrow,
    layout_subtraction_heavy_borrow,
    _layout_bounds,
)

# ---------------------------------------------------------------------------
# Operand samplers
# ---------------------------------------------------------------------------


def _random_operand_1_to_3_digits(rng: random.Random) -> int:
    digits = rng.randint(1, 3)
    if digits == 1:
        return rng.randint(1, 9)
    if digits == 2:
        return rng.randint(10, 99)
    return rng.randint(100, MAX_SYNTH_VALUE)


def _random_operand_1_to_4_digits(rng: random.Random) -> int:
    """1-4 decimal digits; 4-digit values at 15% probability."""
    if rng.random() < 0.15:
        return rng.randint(1000, 9999)
    return _random_operand_1_to_3_digits(rng)


# ---------------------------------------------------------------------------
# layout_bounds (public API -- delegates to layouts_core._layout_bounds)
# ---------------------------------------------------------------------------


def layout_bounds(tokens: List[LayoutToken]) -> Tuple[int, int, int, int]:
    return _layout_bounds(tokens)


# ---------------------------------------------------------------------------
# Generic sampler (uniform over EquationKind)
# ---------------------------------------------------------------------------


def sample_layout(rng: random.Random) -> Tuple[EquationKind, List[LayoutToken]]:
    # Exclude ood_unknown -- it is a sentinel for OOD scenes, not a valid arithmetic layout kind.
    _valid_kinds = [k for k in EquationKind if k != EquationKind.ood_unknown]
    kind_choice = rng.choice(_valid_kinds)
    if kind_choice == EquationKind.add:
        a = _random_operand_1_to_3_digits(rng)
        b = _random_operand_1_to_3_digits(rng)
        return layout_addition(a, b, rng)

    if kind_choice == EquationKind.subtract:
        a = _random_operand_1_to_3_digits(rng)
        b = _random_operand_1_to_3_digits(rng)
        if a < b:
            a, b = b, a
        if a < 1:
            a = 1
        if b < 1:
            b = 1
        return layout_subtraction(a, b)

    if kind_choice == EquationKind.multiply:
        a = _random_operand_1_to_3_digits(rng)
        b = _random_operand_1_to_3_digits(rng)
        return layout_multiplication(a, b, rng)

    if kind_choice == EquationKind.divide:
        return _sample_division_layout(rng)

    assert_never(kind_choice)


# ---------------------------------------------------------------------------
# Result override helper
# ---------------------------------------------------------------------------


def _apply_result_override(
    tokens: List[LayoutToken],
    override: "Optional[Sequence[int]]",
) -> List[LayoutToken]:
    """Replace result-row digit labels with override digits (units-only if multi-digit).

    override[i] replaces the i-th digit of the result row, right-to-left.
    Only replaces ``digit_main`` tokens on the result row (highest row index with main digits).
    """
    if override is None:
        return tokens
    # Find result row: highest row with digit_main tokens that are NOT carry/borrow rows
    main_rows = sorted({t.row for t in tokens if t.yolo_class_name == "digit_main"})
    if not main_rows:
        return tokens
    result_row = main_rows[-1]
    result_toks = sorted(
        [t for t in tokens if t.row == result_row and t.yolo_class_name == "digit_main"],
        key=lambda t: t.col
    )
    if not result_toks:
        return tokens
    # Build replacement map: apply override right-to-left, only up to len(override) digits
    import dataclasses
    override_list = list(override)
    # right-to-left: last token gets override[0], etc.
    replacements = {}
    for idx, tok in enumerate(reversed(result_toks)):
        if idx < len(override_list):
            new_label = f"main_{override_list[idx]}"
            new_key = str(override_list[idx])
            replacements[id(tok)] = dataclasses.replace(tok, flattened_label=new_label, glyph_key=new_key)
    return [replacements.get(id(t), t) for t in tokens]


# ---------------------------------------------------------------------------
# Dispatcher: sample_layout_for_case
# ---------------------------------------------------------------------------


def sample_layout_for_case(
    case: SceneCase,
    rng: random.Random,
    result_override: "Optional[Sequence[int]]" = None,
) -> Tuple[EquationKind, List[LayoutToken]]:
    """Dispatch to the correct layout function for the given SceneCase.

    Exhaustive over all SceneCase values -- assert_never guards against future
    enum additions that are missing a dispatch branch.
    result_override: if given, replace result-row digits after layout generation.
    """
    def _ret(eq: EquationKind, toks: List[LayoutToken]) -> Tuple[EquationKind, List[LayoutToken]]:
        return eq, _apply_result_override(toks, result_override)
    if case == SceneCase.addition:
        a = _random_operand_1_to_3_digits(rng)
        b = _random_operand_1_to_3_digits(rng)
        return _ret(*layout_addition(a, b, rng))

    if case == SceneCase.subtraction:
        a = _random_operand_1_to_3_digits(rng)
        b = _random_operand_1_to_3_digits(rng)
        if a < b:
            a, b = b, a
        a = max(a, 1)
        b = max(b, 1)
        return _ret(*layout_subtraction(a, b))

    if case == SceneCase.multiplication_simple:
        a = _random_operand_1_to_3_digits(rng)
        b = rng.randint(1, 9)
        return _ret(*layout_multiplication(a, b, rng))

    if case == SceneCase.multiplication_multi:
        a = _random_operand_1_to_3_digits(rng)
        b = rng.randint(10, 999)
        return _ret(*layout_multiplication(a, b, rng))

    if case == SceneCase.division_short:
        divisor = rng.randint(1, 9)
        dividend = rng.randint(2, 99)
        return _ret(*layout_short_division(dividend, divisor, rng))

    if case == SceneCase.division_long:
        return _ret(*_sample_division_layout(rng))

    if case == SceneCase.division_simple:
        b = rng.randint(1, 9)
        q = rng.randint(1, 99)
        a = b * q
        return _ret(*layout_simple_division(a, b, rng))

    # --- Workstream B: new layout variants ---
    if case == SceneCase.addition_op_right:
        a = _random_operand_1_to_3_digits(rng)
        b = _random_operand_1_to_3_digits(rng)
        return _ret(*layout_addition(a, b, rng, op_right=True))

    if case == SceneCase.addition_no_bar:
        a = _random_operand_1_to_3_digits(rng)
        b = _random_operand_1_to_3_digits(rng)
        return _ret(*layout_addition_no_bar(a, b, rng))

    if case == SceneCase.subtraction_op_right:
        a = _random_operand_1_to_3_digits(rng)
        b = _random_operand_1_to_3_digits(rng)
        if a < b:
            a, b = b, a
        a = max(a, 1)
        b = max(b, 1)
        return _ret(*layout_subtraction(a, b, op_right=True))

    if case == SceneCase.subtraction_no_bar:
        a = _random_operand_1_to_3_digits(rng)
        b = _random_operand_1_to_3_digits(rng)
        if a < b:
            a, b = b, a
        a = max(a, 1)
        b = max(b, 1)
        return _ret(*layout_subtraction_no_bar(a, b))

    if case == SceneCase.multiplication_simple_op_right:
        a = _random_operand_1_to_3_digits(rng)
        b = rng.randint(1, 9)
        return _ret(*layout_multiplication_simple_op_right(a, b, rng))

    if case == SceneCase.multiplication_simple_no_bar:
        a = _random_operand_1_to_3_digits(rng)
        b = rng.randint(1, 9)
        return _ret(*layout_multiplication_simple_no_bar(a, b, rng))

    # MC-3: subtraction with 2-3 forced borrows
    if case == SceneCase.subtraction_heavy_borrow:
        return _ret(*layout_subtraction_heavy_borrow(rng))

    # MC-4a: addition with 2-3 forced column carries
    if case == SceneCase.addition_dense_carries:
        return _ret(*layout_addition_dense_carries(rng))

    # MC-4b: addition with zero carries (clear negative-carry example)
    if case == SceneCase.addition_no_carries:
        return _ret(*layout_addition_no_carries(rng))

    # iter10 W10-ROBUSTNESS: isolation and bare-number cases
    if case == SceneCase.bare_digits:
        return _ret(*layout_bare_digits(rng))

    if case == SceneCase.bare_digit_grid:
        return _ret(*layout_bare_digit_grid(rng))

    if case == SceneCase.standalone_bar:
        return _ret(*layout_standalone_bar(rng))

    if case == SceneCase.standalone_bracket:
        return _ret(*layout_standalone_bracket(rng))

    if case == SceneCase.subtraction_crowded_borrow:
        return _ret(*layout_subtraction_crowded_borrow(rng))

    if case == SceneCase.addition_crowded_carry:
        return _ret(*layout_addition_crowded_carry(rng))

    assert_never(case)


# ---------------------------------------------------------------------------
# valid_stages_for_kind
# ---------------------------------------------------------------------------


def valid_stages_for_kind(kind: EquationKind) -> List[str]:
    """Return CompletionStage value strings valid for this equation kind."""
    if kind in (EquationKind.add, EquationKind.subtract):
        return [
            "full",
            "no_result",
            "no_bar_no_result",
            "partial_carries_or_borrows",
            "partial_result_digits",
            "missing_next_to_operator",
            "no_carries",
        ]
    if kind == EquationKind.multiply:
        return [
            "full",
            "no_result",
            "no_sum_bar_no_result",
            "first_pp_only",
            "pp_rows_no_sum_bar",
            "partial_result_digits",
            "missing_pp_carry_row",
            "no_carries",
            "first_pp_only",
        ]
    # divide
    return [
        "full",
        "no_final_remainder",
        "missing_quotient_digit_k",
        "steps_k_of_K",
        "bracket_only",
        "first_step_only",
    ]
