"""Completion-stage sampler for synthetic arithmetic scenes.

Five-bucket scheme:
  full    (0.65) — all tokens rendered
  done_80 (0.15) — last 20% of result-row digits dropped
  done_60 (0.10) — last 40% of result-row digits dropped
  done_40 (0.06) — entire result row + bar dropped
  done_20 (0.04) — operands only (most incomplete)

Internal token-removal helpers are equation-kind-aware and unchanged from
prior iterations so ground-truth row/col assignments are not affected.
"""
from __future__ import annotations

import random
from enum import Enum
from typing import List, Tuple

from src.generation.layouts import EquationKind, LayoutToken


class CompletionStage(str, Enum):
    full = "full"
    no_result = "no_result"
    no_bar_no_result = "no_bar_no_result"
    partial_carries_or_borrows = "partial_carries_or_borrows"
    partial_result_digits = "partial_result_digits"
    missing_next_to_operator = "missing_next_to_operator"
    no_carries = "no_carries"
    first_pp_only = "first_pp_only"
    pp_rows_no_sum_bar = "pp_rows_no_sum_bar"
    no_sum_bar_no_result = "no_sum_bar_no_result"
    missing_pp_carry_row = "missing_pp_carry_row"
    bracket_only = "bracket_only"
    first_step_only = "first_step_only"
    steps_k_of_K = "steps_k_of_K"
    missing_quotient_digit_k = "missing_quotient_digit_k"
    no_final_remainder = "no_final_remainder"
    # New 5-bucket aliases
    done_80 = "done_80"
    done_60 = "done_60"
    done_40 = "done_40"
    done_20 = "done_20"


# Production default weights — match config.toml [generation.completion].
# These are the only authoritative fallback values; do not re-declare inline.
_DEFAULT_WEIGHTS: dict[str, float] = {
    "full":    0.65,
    "done_80": 0.15,
    "done_60": 0.07,
    "done_40": 0.06,
    "done_20": 0.07,
}


def _max_row(tokens: List[LayoutToken]) -> int:
    return max(t.row for t in tokens)


def _is_carry_or_borrow(tok: LayoutToken) -> bool:
    return tok.yolo_class_name in {"digit_carry", "digit_borrow"}


# ---------------------------------------------------------------------------
# Token-removal helpers (unchanged — ground-truth row/col must not regress)
# ---------------------------------------------------------------------------

def _stage_add_sub_no_result(tokens: List[LayoutToken]) -> List[LayoutToken]:
    result_row = _max_row(tokens)
    return [t for t in tokens if t.row != result_row]


def _stage_add_sub_no_bar_no_result(tokens: List[LayoutToken]) -> List[LayoutToken]:
    result_row = _max_row(tokens)
    bar_row = result_row - 1
    return [t for t in tokens if t.row < bar_row]


def _stage_partial_carries_or_borrows(
    tokens: List[LayoutToken], rng: random.Random
) -> List[LayoutToken]:
    carry_borrow = [t for t in tokens if _is_carry_or_borrow(t)]
    if not carry_borrow:
        return tokens
    drop_fraction = rng.uniform(0.50, 0.80)
    n_drop = max(1, int(len(carry_borrow) * drop_fraction))
    to_drop = set(id(t) for t in rng.sample(carry_borrow, n_drop))
    return [t for t in tokens if id(t) not in to_drop]


def _stage_partial_result_digits(
    tokens: List[LayoutToken], rng: random.Random, drop_fraction: float = 0.5
) -> List[LayoutToken]:
    result_row = _max_row(tokens)
    result_digits = sorted(
        [t for t in tokens if t.row == result_row and t.yolo_class_name == "digit_main"],
        key=lambda t: t.col,
    )
    if len(result_digits) <= 1:
        return tokens
    n_remove = max(1, round(len(result_digits) * drop_fraction))
    n_remove = min(n_remove, len(result_digits) - 1)
    to_remove = set(id(t) for t in result_digits[-n_remove:])
    return [t for t in tokens if id(t) not in to_remove]


def _stage_missing_next_to_operator(tokens: List[LayoutToken]) -> List[LayoutToken]:
    row_b = 2
    operand_b_digits = sorted(
        [t for t in tokens if t.row == row_b and t.yolo_class_name == "digit_main"],
        key=lambda t: t.col,
    )
    if not operand_b_digits:
        return tokens
    rightmost = operand_b_digits[-1]
    return [t for t in tokens if id(t) != id(rightmost)]


def _stage_no_carries(tokens: List[LayoutToken]) -> List[LayoutToken]:
    return [t for t in tokens if not _is_carry_or_borrow(t)]


def _is_nm_multiply(tokens: List[LayoutToken]) -> bool:
    return any(t.row >= 5 for t in tokens)


def _stage_nm_no_sum_bar_no_result(tokens: List[LayoutToken]) -> List[LayoutToken]:
    result_row = _max_row(tokens)
    bar2_row = result_row - 1
    return [t for t in tokens if t.row < bar2_row]


def _stage_nm_first_pp_only(tokens: List[LayoutToken]) -> List[LayoutToken]:
    return [t for t in tokens if t.row <= 4]


def _nm_add_carry_row(tokens: List[LayoutToken]) -> int:
    return _max_row(tokens) - 2


def _stage_nm_missing_pp_carry_row(tokens: List[LayoutToken]) -> List[LayoutToken]:
    add_carry_row = _nm_add_carry_row(tokens)
    return [t for t in tokens if t.row != add_carry_row]


def _count_div_steps(tokens: List[LayoutToken]) -> int:
    bar_rows = {t.row for t in tokens if t.flattened_label == "result_bar"}
    return len(bar_rows)


def _step_rows_up_to(k: int) -> set:
    rows: set = {0, 1}
    for step_idx in range(k):
        pp_row = 2 + step_idx * 3
        bar_row = 3 + step_idx * 3
        rem_row = 4 + step_idx * 3
        rows.update({pp_row, bar_row, rem_row})
    return rows


def _stage_div_bracket_only(tokens: List[LayoutToken]) -> List[LayoutToken]:
    # MC-5 alignment: for division_short, row 1 contains divisor digits + div_bracket +
    # dividend digits (see layout_short_division). Keeping row==1 therefore gives
    # "bracket + dividend" which is exactly the MC-5 target scene. No new SceneCase is
    # needed; existing done_20 on division_short already covers this adequately.
    return [t for t in tokens if t.row == 1]


def _stage_div_first_step_only(tokens: List[LayoutToken]) -> List[LayoutToken]:
    keep_rows = {0, 1, 2, 3, 4}
    return [t for t in tokens if t.row in keep_rows]


def _stage_div_steps_k_of_K(
    tokens: List[LayoutToken], rng: random.Random
) -> List[LayoutToken]:
    n_steps = _count_div_steps(tokens)
    if n_steps <= 1:
        return tokens
    k = rng.randint(1, n_steps - 1)
    keep_rows = _step_rows_up_to(k)
    return [t for t in tokens if t.row in keep_rows]


def _stage_div_missing_quotient_digit_k(
    tokens: List[LayoutToken], rng: random.Random
) -> List[LayoutToken]:
    quotient_toks = [t for t in tokens if t.row == 0 and t.yolo_class_name == "digit_main"]
    if not quotient_toks:
        return tokens
    to_remove = rng.choice(quotient_toks)
    return [t for t in tokens if id(t) != id(to_remove)]


def _stage_div_no_final_remainder(tokens: List[LayoutToken]) -> List[LayoutToken]:
    final_row = _max_row(tokens)
    return [t for t in tokens if t.row != final_row]


# ---------------------------------------------------------------------------
# 5-bucket samplers — map new bucket names to kind-specific token removals
# ---------------------------------------------------------------------------

def _sample_add_sub_stage(
    tokens: List[LayoutToken], rng: random.Random,
    weights: dict,
) -> Tuple[CompletionStage, List[LayoutToken]]:
    wf   = weights.get("full",    _DEFAULT_WEIGHTS["full"])
    w80  = weights.get("done_80", _DEFAULT_WEIGHTS["done_80"])
    w60  = weights.get("done_60", _DEFAULT_WEIGHTS["done_60"])
    w40  = weights.get("done_40", _DEFAULT_WEIGHTS["done_40"])
    r = rng.random()
    if r < wf:
        return CompletionStage.full, list(tokens)
    r -= wf
    if r < w80:
        # done_80: drop last ~20% of result digits
        return CompletionStage.done_80, _stage_partial_result_digits(tokens, rng, drop_fraction=0.20)
    r -= w80
    if r < w60:
        # done_60: drop last ~40% of result digits
        return CompletionStage.done_60, _stage_partial_result_digits(tokens, rng, drop_fraction=0.40)
    r -= w60
    if r < w40:
        # done_40: drop entire result row + bar
        return CompletionStage.done_40, _stage_add_sub_no_bar_no_result(tokens)
    # done_20: operands only (no carries, no bar, no result)
    return CompletionStage.done_20, _stage_no_carries(_stage_add_sub_no_bar_no_result(tokens))


def _sample_mul_stage(
    tokens: List[LayoutToken], rng: random.Random,
    weights: dict,
) -> Tuple[CompletionStage, List[LayoutToken]]:
    is_nm = _is_nm_multiply(tokens)
    wf   = weights.get("full",    _DEFAULT_WEIGHTS["full"])
    w80  = weights.get("done_80", _DEFAULT_WEIGHTS["done_80"])
    w60  = weights.get("done_60", _DEFAULT_WEIGHTS["done_60"])
    w40  = weights.get("done_40", _DEFAULT_WEIGHTS["done_40"])
    r = rng.random()
    if r < wf:
        return CompletionStage.full, list(tokens)
    r -= wf
    if r < w80:
        return CompletionStage.done_80, _stage_partial_result_digits(tokens, rng, drop_fraction=0.20)
    r -= w80
    if r < w60:
        return CompletionStage.done_60, _stage_partial_result_digits(tokens, rng, drop_fraction=0.40)
    r -= w60
    if r < w40:
        if is_nm:
            return CompletionStage.done_40, _stage_nm_no_sum_bar_no_result(tokens)
        return CompletionStage.done_40, _stage_add_sub_no_bar_no_result(tokens)
    if is_nm:
        return CompletionStage.done_20, _stage_nm_first_pp_only(tokens)
    return CompletionStage.done_20, _stage_no_carries(_stage_add_sub_no_bar_no_result(tokens))


def _sample_no_bar_stage(
    tokens: List[LayoutToken], rng: random.Random,
    weights: dict,
) -> Tuple[CompletionStage, List[LayoutToken]]:
    """Stage sampler for no-bar scene cases (addition_no_bar, subtraction_no_bar,
    multiplication_simple_no_bar).

    These scenes have no result_bar token by construction, so done_80 (which normally
    drops 20% of result digits) is the finest meaningful partial: it mirrors done_80 for
    with-bar scenes. done_40 drops the entire result row; done_20 additionally strips
    carries/borrows so only operands remain.

    GAP-1 fix: previously this function was absent and the sampler short-circuited all
    no-bar scenes to CompletionStage.full, ignoring bucket weights entirely.
    """
    wf   = weights.get("full",    _DEFAULT_WEIGHTS["full"])
    w60  = weights.get("done_60", _DEFAULT_WEIGHTS["done_60"])
    w40  = weights.get("done_40", _DEFAULT_WEIGHTS["done_40"])
    # Normalise over the three valid buckets (done_80 is excluded; no bar means
    # the 80%-complete milestone is indistinguishable from full).
    total = wf + w60 + w40 + (1.0 - wf - w60 - w40)  # done_20 gets remainder
    w_done_20 = max(0.0, 1.0 - wf - w60 - w40)
    r = rng.random()
    if r < wf:
        return CompletionStage.full, list(tokens)
    r -= wf
    if r < w60:
        # done_60: drop last 40% of result-row digits
        return CompletionStage.done_60, _stage_partial_result_digits(tokens, rng, drop_fraction=0.40)
    r -= w60
    if r < w40:
        # done_40: drop entire result row (bar already absent)
        return CompletionStage.done_40, _stage_add_sub_no_result(tokens)
    # done_20: operands only (strip result row + carries/borrows)
    return CompletionStage.done_20, _stage_no_carries(_stage_add_sub_no_result(tokens))


def _sample_div_stage(
    tokens: List[LayoutToken], rng: random.Random,
    weights: dict,
) -> Tuple[CompletionStage, List[LayoutToken]]:
    wf   = weights.get("full",    _DEFAULT_WEIGHTS["full"])
    w80  = weights.get("done_80", _DEFAULT_WEIGHTS["done_80"])
    w60  = weights.get("done_60", _DEFAULT_WEIGHTS["done_60"])
    w40  = weights.get("done_40", _DEFAULT_WEIGHTS["done_40"])
    r = rng.random()
    if r < wf:
        return CompletionStage.full, list(tokens)
    r -= wf
    if r < w80:
        return CompletionStage.done_80, _stage_div_no_final_remainder(tokens)
    r -= w80
    if r < w60:
        return CompletionStage.done_60, _stage_div_steps_k_of_K(tokens, rng)
    r -= w60
    if r < w40:
        return CompletionStage.done_40, _stage_div_first_step_only(tokens)
    return CompletionStage.done_20, _stage_div_bracket_only(tokens)


def sample_completion_stage(
    kind: EquationKind, tokens: list[LayoutToken], rng: random.Random,
    weights: dict | None = None,
) -> Tuple[CompletionStage, list[LayoutToken]]:
    """Sample a completion stage for the given equation kind.

    Preserved for backward compatibility. New code should prefer
    sample_completion_stage_for_case.
    """
    if not tokens:
        return CompletionStage.full, list(tokens)

    w = weights if weights is not None else dict(_DEFAULT_WEIGHTS)

    if kind in (EquationKind.add, EquationKind.subtract):
        return _sample_add_sub_stage(tokens, rng, w)
    if kind == EquationKind.multiply:
        return _sample_mul_stage(tokens, rng, w)
    if kind == EquationKind.divide:
        return _sample_div_stage(tokens, rng, w)

    return CompletionStage.full, list(tokens)


def valid_stages_for_case(case: "SceneCase") -> List[str]:
    """Return CompletionStage value strings valid for this SceneCase."""
    from src.generation.layouts import SceneCase  # local import avoids circular dep

    if case in (
        SceneCase.addition,
        SceneCase.subtraction,
        SceneCase.addition_op_right,
        SceneCase.subtraction_op_right,
        SceneCase.subtraction_heavy_borrow,
        SceneCase.addition_dense_carries,
        SceneCase.addition_no_carries,
        SceneCase.subtraction_crowded_borrow,
        SceneCase.addition_crowded_carry,
    ):
        return ["full", "done_80", "done_60", "done_40", "done_20"]
    if case in (SceneCase.multiplication_simple, SceneCase.multiplication_multi):
        return ["full", "done_80", "done_60", "done_40", "done_20"]
    if case == SceneCase.division_short:
        return ["full", "done_80", "done_60", "done_40", "done_20"]
    if case == SceneCase.division_long:
        return ["full", "done_80", "done_60", "done_40", "done_20"]
    if case in (SceneCase.addition_no_bar, SceneCase.subtraction_no_bar, SceneCase.multiplication_simple_no_bar):
        return ["full", "done_60", "done_40", "done_20"]
    if case in (SceneCase.multiplication_simple_op_right,):
        return ["full", "done_80", "done_60", "done_40", "done_20"]
    if case == SceneCase.division_simple:
        return ["full", "done_80", "done_40"]
    # iter10 W10-ROBUSTNESS: isolation and bare-number cases — full stage only.
    # No progression semantics for bare numbers or isolated structural tokens.
    if case in (
        SceneCase.bare_digits,
        SceneCase.bare_digit_grid,
        SceneCase.standalone_bar,
        SceneCase.standalone_bracket,
    ):
        return ["full"]
    return ["full"]


def sample_completion_stage_for_case(
    case: "SceneCase",
    tokens: List[LayoutToken],
    rng: random.Random,
    weights: dict | None = None,
) -> Tuple[CompletionStage, List[LayoutToken]]:
    """Sample a completion stage valid for the given SceneCase."""
    from src.generation.layouts import SceneCase  # local import avoids circular dep

    if not tokens:
        return CompletionStage.full, list(tokens)

    w = weights if weights is not None else dict(_DEFAULT_WEIGHTS)

    if case in (
        SceneCase.addition,
        SceneCase.subtraction,
        SceneCase.addition_op_right,
        SceneCase.subtraction_op_right,
        SceneCase.subtraction_heavy_borrow,
        SceneCase.addition_dense_carries,
        SceneCase.addition_no_carries,
        SceneCase.subtraction_crowded_borrow,
        SceneCase.addition_crowded_carry,
    ):
        return _sample_add_sub_stage(tokens, rng, w)
    if case in (SceneCase.multiplication_simple, SceneCase.multiplication_multi):
        return _sample_mul_stage(tokens, rng, w)
    if case in (SceneCase.addition_no_bar, SceneCase.subtraction_no_bar, SceneCase.multiplication_simple_no_bar):
        # GAP-1 fix: no-bar scenes must apply progressive token removal for partial stages.
        # done_80 is not applicable (bar already absent; no structural token to show/hide).
        # done_60: drop last 40% of result-row digits.
        # done_40: drop entire result row (operands + operator only, bar already absent).
        # done_20: drop result row + carries/borrows (operands only, most incomplete).
        return _sample_no_bar_stage(tokens, rng, w)
    if case == SceneCase.multiplication_simple_op_right:
        return _sample_mul_stage(tokens, rng, w)
    if case in (SceneCase.division_short, SceneCase.division_long):
        return _sample_div_stage(tokens, rng, w)
    # iter10 W10-ROBUSTNESS: isolation and bare-number cases — full stage only.
    if case in (
        SceneCase.bare_digits,
        SceneCase.bare_digit_grid,
        SceneCase.standalone_bar,
        SceneCase.standalone_bracket,
    ):
        return CompletionStage.full, list(tokens)
    # division_simple
    wf = w.get("full",    _DEFAULT_WEIGHTS["full"])
    w80 = w.get("done_80", _DEFAULT_WEIGHTS["done_80"])
    r = rng.random()
    if r < wf:
        return CompletionStage.full, list(tokens)
    r -= wf
    if r < w80:
        return CompletionStage.done_80, _stage_add_sub_no_result(tokens)
    return CompletionStage.done_40, _stage_add_sub_no_bar_no_result(tokens)
