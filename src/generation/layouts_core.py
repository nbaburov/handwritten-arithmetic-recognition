from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from typing import List, Tuple

from .layouts_types import (
    CARRY_BORROW_TOKEN_SCALE,
    OPERATOR_TOKEN_SCALE,
    EquationKind,
    LayoutToken,
)

# ---------------------------------------------------------------------------
# Private arithmetic helpers
# ---------------------------------------------------------------------------


def _digit_to_cols(n: int, width: int) -> List[int]:
    s = str(abs(int(n))).zfill(width)
    return [int(ch) for ch in s]


def _add_carries(a: int, b: int, width: int) -> List[int]:
    a_d = _digit_to_cols(a, width)
    b_d = _digit_to_cols(b, width)
    carries = [0] * width
    c = 0
    for i in range(width - 1, -1, -1):
        s = a_d[i] + b_d[i] + c
        if i > 0:
            carries[i - 1] = 1 if s >= 10 else 0
        c = s // 10
    return carries


def _sub_borrows(minuend: int, subtrahend: int, width: int) -> Tuple[List[int], List[int]]:
    m_d = _digit_to_cols(minuend, width)
    s_d = _digit_to_cols(subtrahend, width)
    borrows = [0] * width
    work = list(m_d)
    for i in range(width - 1, -1, -1):
        if work[i] < s_d[i] and i > 0:
            borrows[i] = 1
            work[i] += 10
            j = i - 1
            while j >= 0 and work[j] == 0:
                work[j] = 9
                j -= 1
            if j >= 0:
                work[j] -= 1
    return borrows, work


def _sub_borrows_style_b(minuend: int, subtrahend: int, width: int) -> Tuple[List[int], List[int]]:
    """
    Returns (annotated_vals, borrower_cols).
    annotated_vals[j] = the NEW digit value written as a borrow annotation at column j:
      - For a zero column in the borrow chain: 9 (receives 10, gives 1, net = 9).
      - For the actual lender column: original_digit - 1.
    annotated_vals[j] = -1 if column j was not modified by borrowing.
    borrower_cols = list of column indices i where work[i] < s_d[i] (the columns that received +10).
    Column 0 is leftmost (most significant).
    """
    m_d = _digit_to_cols(minuend, width)
    s_d = _digit_to_cols(subtrahend, width)
    annotated_vals = [-1] * width
    borrower_cols: List[int] = []
    work = list(m_d)
    for i in range(width - 1, -1, -1):
        if work[i] < s_d[i] and i > 0:
            borrower_cols.append(i)
            work[i] += 10
            j = i - 1
            while j >= 0 and work[j] == 0:
                annotated_vals[j] = 9
                work[j] = 9
                j -= 1
            if j >= 0:
                annotated_vals[j] = work[j] - 1
                work[j] -= 1
    return annotated_vals, borrower_cols


def _mul_single_pass_carries(
    a_digits: List[int],
    b_digit: int,
    right_col: int,
    carry_row: int,
) -> List[LayoutToken]:
    """
    Returns carry tokens produced when multiplying a_digits by a single digit b_digit.
    Carry from position j (0-indexed from left) lands at col right_col-(len-1-j)-1.
    Only emits tokens for non-leftmost carries (leftmost carry flows into the result digit).
    carry_row: row_index to assign to all carry tokens.
    """
    tokens: List[LayoutToken] = []
    n = len(a_digits)
    c = 0
    for j in range(n - 1, -1, -1):
        col_j = right_col - (n - 1 - j)
        prod = a_digits[j] * b_digit + c
        new_c = prod // 10
        if new_c > 0 and j > 0:
            tokens.append(
                LayoutToken(
                    row=carry_row,
                    col=col_j - 1,
                    flattened_label=f"carry_{new_c}",
                    glyph_key=str(new_c),
                    yolo_class_name="digit_carry",
                    token_scale=CARRY_BORROW_TOKEN_SCALE,
                )
            )
        c = new_c
    return tokens


# ---------------------------------------------------------------------------
# DivStep dataclass (public type, placed here with the division logic)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DivStep:
    """One step in a long division: what quotient digit goes above, the PP, remainder, and which col."""
    quotient_digit: int
    product: int
    remainder: int
    right_col: int   # col_index of the rightmost digit of this step's partial dividend


def _div_steps(dividend: int, divisor: int) -> List[DivStep]:
    """
    Compute the per-step breakdown of dividend / divisor.
    Returns one DivStep per quotient digit.
    right_col is 0-indexed into the dividend string (col_index for the layout).
    """
    dividend_str = str(dividend)
    steps: List[DivStep] = []
    current = 0
    p = 0

    # Build first partial dividend: smallest prefix >= divisor
    while p < len(dividend_str) and current < divisor:
        current = current * 10 + int(dividend_str[p])
        p += 1

    while True:
        q_digit = current // divisor
        product = q_digit * divisor
        remainder = current - product
        right_col = p - 1   # index of the last digit consumed

        steps.append(DivStep(
            quotient_digit=q_digit,
            product=product,
            remainder=remainder,
            right_col=right_col,
        ))

        if p >= len(dividend_str):
            break

        current = remainder * 10 + int(dividend_str[p])
        p += 1

    return steps


# ---------------------------------------------------------------------------
# Core layout functions (one per arithmetic case)
# ---------------------------------------------------------------------------


def layout_long_division(
    dividend: int,
    divisor: int,
    rng: random.Random,
) -> Tuple[EquationKind, List[LayoutToken]]:
    """
    Emit tokens for long division: dividend / divisor.
    Col convention: leftmost dividend digit = col 0. Bracket at col -1.
    Divisor right-justified starting at col -(len(divisor_str)+1).
    Quotient digits at row 0, col = step.right_col.
    Per step k: PP at row 2+k*3, bar at row 3+k*3, remainder+bringdown at row 4+k*3.
    """
    steps = _div_steps(dividend, divisor)
    tokens: List[LayoutToken] = []
    divisor_str = str(divisor)
    dividend_str = str(dividend)
    v = len(divisor_str)
    q = len(steps)

    # row 0: quotient digits
    for step in steps:
        tokens.append(LayoutToken(
            row=0, col=step.right_col,
            flattened_label=f"main_{step.quotient_digit}",
            glyph_key=str(step.quotient_digit),
            yolo_class_name="digit_main",
        ))

    # row 1: divisor + bracket + dividend
    # divisor right-justified: ones digit at col -(1+1) = -2; for 2-digit: tens at -3, ones at -2
    divisor_start_col = -(v + 1)
    for j, ch in enumerate(divisor_str):
        tokens.append(LayoutToken(
            row=1, col=divisor_start_col + j,
            flattened_label=f"main_{ch}",
            glyph_key=ch,
            yolo_class_name="digit_main",
        ))

    tokens.append(LayoutToken(
        row=1, col=-1,
        flattened_label="div_bracket",
        glyph_key="div_bracket",
        yolo_class_name="divide_bracket",
    ))

    for j, ch in enumerate(dividend_str):
        tokens.append(LayoutToken(
            row=1, col=j,
            flattened_label=f"main_{ch}",
            glyph_key=ch,
            yolo_class_name="digit_main",
        ))

    # per-step rows
    for k, step in enumerate(steps):
        pp_row = 2 + k * 3
        bar_row = 3 + k * 3
        rem_row = 4 + k * 3

        # PP digits: right-justified to step.right_col
        # For quotient_digit=0, product=0: str(0)=="0" -> emits single "main_0"
        product_str = str(step.product)
        pp_left = step.right_col - len(product_str) + 1
        for j, ch in enumerate(product_str):
            tokens.append(LayoutToken(
                row=pp_row, col=pp_left + j,
                flattened_label=f"main_{ch}",
                glyph_key=ch,
                yolo_class_name="digit_main",
            ))

        # subtraction bar
        tokens.append(LayoutToken(
            row=bar_row, col=pp_left - 1,
            flattened_label="result_bar",
            glyph_key="result_bar",
            yolo_class_name="result_bar",
            segment_kind="subtraction_bar",
        ))

        # remainder row
        if k < q - 1:
            # remainder + brought-down next dividend digit
            next_step = steps[k + 1]
            rem_str = str(step.remainder) if step.remainder > 0 else ""
            brought_down_col = next_step.right_col
            # remainder digits sit left of the brought-down digit
            for j, ch in enumerate(rem_str):
                col = brought_down_col - len(rem_str) + j
                tokens.append(LayoutToken(
                    row=rem_row, col=col,
                    flattened_label=f"main_{ch}",
                    glyph_key=ch,
                    yolo_class_name="digit_main",
                ))
            # brought-down digit from dividend
            brought_digit = dividend_str[next_step.right_col]
            tokens.append(LayoutToken(
                row=rem_row, col=brought_down_col,
                flattened_label=f"main_{brought_digit}",
                glyph_key=brought_digit,
                yolo_class_name="digit_main",
            ))
        else:
            # final remainder
            final_rem = step.remainder
            final_str = str(final_rem) if final_rem > 0 else "0"
            rem_left = step.right_col - len(final_str) + 1
            for j, ch in enumerate(final_str):
                tokens.append(LayoutToken(
                    row=rem_row, col=rem_left + j,
                    flattened_label=f"main_{ch}",
                    glyph_key=ch,
                    yolo_class_name="digit_main",
                ))

    return EquationKind.divide, tokens


def _sample_division_layout(rng: random.Random) -> Tuple[EquationKind, List[LayoutToken]]:
    """
    Sample a valid (dividend, divisor) pair:
      - divisor 1-3 digits (1-999)
      - quotient 1-4 digits
      - dividend <= 999,999
    Canvas-fit guard: layouts with max_row > 15 overflow 512px and are retried.
    (15 rows x ~28px/row = 420px, which fits within 512.)
    """
    from .layouts_types import MAX_SYNTH_VALUE  # noqa: F401 -- available at module level but imported for clarity
    MAX_ROW_LIMIT: int = 15
    for _ in range(128):
        divisor = rng.randint(1, 999)
        max_quotient = min(9999, 999999 // divisor)
        if max_quotient < 1:
            continue
        quotient = rng.randint(1, max_quotient)
        remainder = rng.randint(0, divisor - 1)
        dividend = divisor * quotient + remainder
        if 1 <= dividend <= 999999:
            kind, tokens = layout_long_division(dividend, divisor, rng)
            if _layout_bounds(tokens)[1] <= MAX_ROW_LIMIT:
                return kind, tokens
            # Layout too tall -- retry with a fresh sample
            continue
    # Fallback: trivial case guaranteed to fit
    return layout_long_division(8, 4, rng)


def layout_addition(a: int, b: int, rng: random.Random, *, op_right: bool = False) -> Tuple[EquationKind, List[LayoutToken]]:
    r = a + b
    width = max(len(str(a)), len(str(b)), len(str(r)))
    carries = _add_carries(a, b, width)
    tokens: List[LayoutToken] = []
    carry_row = 0
    row_a = 1
    row_b = 2
    row_bar = 3
    row_res = 4
    col_offset = 2

    for i, bit in enumerate(carries):
        if not bit:
            continue
        # Optional two-digit "10" style carry (decade) in two small slots, when column allows.
        two_digit_ten = (
            rng.random() < 0.32
            and i > 0
            and carries[i - 1] == 0
            and (i == 1 or carries[i - 2] == 0)
        )
        sc = CARRY_BORROW_TOKEN_SCALE
        if two_digit_ten:
            tokens.append(
                LayoutToken(
                    row=carry_row,
                    col=col_offset + i - 1,
                    flattened_label="carry_1",
                    glyph_key="1",
                    yolo_class_name="digit_carry",
                    token_scale=sc,
                )
            )
            tokens.append(
                LayoutToken(
                    row=carry_row,
                    col=col_offset + i,
                    flattened_label="carry_0",
                    glyph_key="0",
                    yolo_class_name="digit_carry",
                    token_scale=sc,
                )
            )
        else:
            tokens.append(
                LayoutToken(
                    row=carry_row,
                    col=col_offset + i,
                    flattened_label=f"carry_{bit}",
                    glyph_key=str(bit),
                    yolo_class_name="digit_carry",
                    token_scale=sc,
                )
            )

    a_lead = width - len(str(a))   # leading zero positions to skip for operand a
    b_lead = width - len(str(b))   # leading zero positions to skip for operand b
    for i, d in enumerate(_digit_to_cols(a, width)):
        if i < a_lead:
            continue   # suppress leading zero padding -- students write "52" not "052"
        tokens.append(
            LayoutToken(
                row=row_a,
                col=col_offset + i,
                flattened_label=f"main_{d}",
                glyph_key=str(d),
                yolo_class_name="digit_main",
            )
        )

    for i, d in enumerate(_digit_to_cols(b, width)):
        if i < b_lead:
            continue   # suppress leading zero padding -- students write "7" not "07"
        tokens.append(
            LayoutToken(
                row=row_b,
                col=col_offset + i,
                flattened_label=f"main_{d}",
                glyph_key=str(d),
                yolo_class_name="digit_main",
            )
        )
    b_left_add = col_offset + (width - len(str(b)))
    r_lead = width - len(str(r))  # leading positions before result digits
    min_lead = min(a_lead, b_lead, r_lead)
    op_col = (col_offset + width) if op_right else (col_offset + min_lead - 1)
    tokens.append(
        LayoutToken(
            row=row_b,
            col=op_col,
            flattened_label="op_plus",
            glyph_key="plus",
            yolo_class_name="operator",
            token_scale=OPERATOR_TOKEN_SCALE,
        )
    )

    tokens.append(
        LayoutToken(
            row=row_bar,
            col=col_offset - 1,
            flattened_label="result_bar",
            glyph_key="result_bar",
            yolo_class_name="result_bar",
            segment_kind="final_result_bar",
        )
    )

    for i, d in enumerate(_digit_to_cols(r, width)):
        tokens.append(
            LayoutToken(
                row=row_res,
                col=col_offset + i,
                flattened_label=f"main_{d}",
                glyph_key=str(d),
                yolo_class_name="digit_main",
            )
        )
    return EquationKind.add, tokens


def layout_subtraction(a: int, b: int, *, op_right: bool = False) -> Tuple[EquationKind, List[LayoutToken]]:
    if a < b:
        a, b = b, a
    r = a - b
    width = max(len(str(a)), len(str(b)), len(str(r)))
    annotated_vals, borrower_cols = _sub_borrows_style_b(a, b, width)
    tokens: List[LayoutToken] = []
    borrow_row = 0
    row_a = 1
    row_b = 2
    row_bar = 3
    row_res = 4
    col_offset = 2
    sc = CARRY_BORROW_TOKEN_SCALE

    for i, new_val in enumerate(annotated_vals):
        if new_val >= 0:
            tokens.append(
                LayoutToken(
                    row=borrow_row,
                    col=col_offset + i,
                    flattened_label=f"borrow_{new_val}",
                    glyph_key=str(new_val),
                    yolo_class_name="digit_borrow",
                    token_scale=sc,
                )
            )

    for col_i in borrower_cols:
        tokens.append(
            LayoutToken(
                row=borrow_row,
                col=col_offset + col_i,
                flattened_label="borrow_1",
                glyph_key="1",
                yolo_class_name="digit_borrow",
                token_scale=sc,
            )
        )

    a_lead = width - len(str(a))   # a is always >= b so a has no leading zeros (a_lead=0)
    b_lead = width - len(str(b))   # b may be narrower than a
    for i, d in enumerate(_digit_to_cols(a, width)):
        if i < a_lead:
            continue
        tokens.append(
            LayoutToken(
                row=row_a,
                col=col_offset + i,
                flattened_label=f"main_{d}",
                glyph_key=str(d),
                yolo_class_name="digit_main",
            )
        )

    b_left_sub = col_offset + b_lead
    r_lead_sub = width - len(str(r))  # leading positions before result digits
    min_lead_sub = min(a_lead, b_lead, r_lead_sub)
    op_col_sub = (col_offset + width) if op_right else (col_offset + min_lead_sub - 1)
    tokens.append(
        LayoutToken(
            row=row_b,
            col=op_col_sub,
            flattened_label="op_minus",
            glyph_key="minus",
            yolo_class_name="operator",
            token_scale=OPERATOR_TOKEN_SCALE,
        )
    )
    for i, d in enumerate(_digit_to_cols(b, width)):
        if i < b_lead:
            continue   # suppress leading zeros -- students write "7" not "07"
        tokens.append(
            LayoutToken(
                row=row_b,
                col=col_offset + i,
                flattened_label=f"main_{d}",
                glyph_key=str(d),
                yolo_class_name="digit_main",
            )
        )

    tokens.append(
        LayoutToken(
            row=row_bar,
            col=col_offset - 1,
            flattened_label="result_bar",
            glyph_key="result_bar",
            yolo_class_name="result_bar",
            segment_kind="final_result_bar",
        )
    )

    for i, d in enumerate(_digit_to_cols(r, width)):
        tokens.append(
            LayoutToken(
                row=row_res,
                col=col_offset + i,
                flattened_label=f"main_{d}",
                glyph_key=str(d),
                yolo_class_name="digit_main",
            )
        )
    return EquationKind.subtract, tokens


def _pp_row(
    a: int,
    b_digit: int,
    pp_right_col: int,
    pp_row: int,
    carry_row: int,
    trailing_zeros: int = 0,
) -> List[LayoutToken]:
    """
    Emit tokens for one partial product pass: A x b_digit.
    pp_right_col: col_index of the rightmost non-zero digit of this PP (before trailing zeros).
    trailing_zeros: number of explicit zero tokens to append (Style B).
    The full row is right-justified to pp_right_col + trailing_zeros.
    carry_row: row_index for carry tokens (immediately above pp_row).
    Returns all tokens (carry + digit).
    """
    product = a * b_digit
    product_str = str(product)
    a_digits = _digit_to_cols(a, len(str(a)))
    tokens: List[LayoutToken] = []

    # carry tokens (based on the raw product's right column, before trailing zeros)
    tokens.extend(_mul_single_pass_carries(a_digits, b_digit, pp_right_col, carry_row))

    # PP digit tokens including trailing zeros, right-justified to pp_right_col + trailing_zeros
    full_str = product_str + "0" * trailing_zeros
    actual_right = pp_right_col + trailing_zeros
    left_col = actual_right - len(full_str) + 1
    for j, ch in enumerate(full_str):
        tokens.append(
            LayoutToken(
                row=pp_row,
                col=left_col + j,
                flattened_label=f"main_{ch}",
                glyph_key=ch,
                yolo_class_name="digit_main",
            )
        )
    return tokens


def layout_multiplication(a: int, b: int, rng: random.Random) -> Tuple[EquationKind, List[LayoutToken]]:
    result = a * b
    a_str, b_str, r_str = str(a), str(b), str(result)
    scene_width = max(len(a_str), len(b_str), len(r_str))
    col_offset = 2
    right_col = col_offset + scene_width - 1
    b_digits_list = [int(ch) for ch in b_str]  # left to right
    n_b = len(b_digits_list)

    tokens: List[LayoutToken] = []

    if n_b == 1:
        # N x 1: carry row=0, multiplicand row=1, multiplier row=2, bar row=3, result row=4
        a_digits = _digit_to_cols(a, len(a_str))
        tokens.extend(_mul_single_pass_carries(a_digits, b_digits_list[0], right_col, carry_row=0))

        a_left = right_col - len(a_str) + 1
        for j, ch in enumerate(a_str):
            tokens.append(LayoutToken(
                row=1, col=a_left + j,
                flattened_label=f"main_{ch}", glyph_key=ch, yolo_class_name="digit_main",
            ))

        b_left = right_col - len(b_str) + 1
        r_left_n1 = right_col - len(r_str) + 1
        tokens.append(LayoutToken(
            row=2, col=min(a_left, b_left, r_left_n1) - 1,
            flattened_label="op_times", glyph_key="times",
            yolo_class_name="operator", token_scale=OPERATOR_TOKEN_SCALE,
        ))
        for j, ch in enumerate(b_str):
            tokens.append(LayoutToken(
                row=2, col=b_left + j,
                flattened_label=f"main_{ch}", glyph_key=ch, yolo_class_name="digit_main",
            ))

        tokens.append(LayoutToken(
            row=3, col=col_offset - 1,
            flattened_label="result_bar", glyph_key="result_bar", yolo_class_name="result_bar",
            segment_kind="final_result_bar",
        ))

        r_left = right_col - len(r_str) + 1
        for j, ch in enumerate(r_str):
            tokens.append(LayoutToken(
                row=4, col=r_left + j,
                flattened_label=f"main_{ch}", glyph_key=ch, yolo_class_name="digit_main",
            ))

    else:
        # N x M where M>=2:
        # Row layout: A=0, B=1, bar1=2
        # Per PP k (0-indexed, ones first): carry at 3+k*2, digits at 4+k*2
        # add_carry_row = 3 + n_b*2
        # bar2 = 3 + n_b*2 + 1
        # result = 3 + n_b*2 + 2

        a_left = right_col - len(a_str) + 1
        for j, ch in enumerate(a_str):
            tokens.append(LayoutToken(
                row=0, col=a_left + j,
                flattened_label=f"main_{ch}", glyph_key=ch, yolo_class_name="digit_main",
            ))

        b_left = right_col - len(b_str) + 1
        r_left_nm = right_col - len(r_str) + 1
        tokens.append(LayoutToken(
            row=1, col=min(a_left, b_left, r_left_nm) - 1,
            flattened_label="op_times", glyph_key="times",
            yolo_class_name="operator", token_scale=OPERATOR_TOKEN_SCALE,
        ))
        for j, ch in enumerate(b_str):
            tokens.append(LayoutToken(
                row=1, col=b_left + j,
                flattened_label=f"main_{ch}", glyph_key=ch, yolo_class_name="digit_main",
            ))

        tokens.append(LayoutToken(
            row=2, col=col_offset - 1,
            flattened_label="result_bar", glyph_key="result_bar", yolo_class_name="result_bar",
            segment_kind="intermediate_result_bar",  # G: annotation hygiene fix
        ))

        # Partial product rows (ones digit of B first)
        # Style B: PP k gets k trailing explicit zeros; k>=1 rows get an op_plus token.
        b_digits_reversed = list(reversed(b_digits_list))
        for k, b_dig in enumerate(b_digits_reversed):
            pp_right = right_col - k   # rightmost non-zero digit column
            carry_row_k = 3 + k * 2
            pp_row_k = 4 + k * 2
            tokens.extend(_pp_row(a, b_dig, pp_right, pp_row_k, carry_row_k, trailing_zeros=k))

            if k >= 1:
                # + operator sits one column left of the full PP string's leftmost digit
                pp_val = a * b_dig
                full_len = len(str(pp_val)) + k  # product digits + trailing zeros
                pp_left = right_col - full_len + 1  # actual_right = right_col
                tokens.append(LayoutToken(
                    row=pp_row_k,
                    col=pp_left - 1,
                    flattened_label="op_plus",
                    glyph_key="plus",
                    yolo_class_name="operator",
                    token_scale=OPERATOR_TOKEN_SCALE,
                ))

        # Addition carry row: carries when summing all PPs to get result
        add_carry_row = 3 + n_b * 2
        # Place each PP's digits at absolute col positions
        col_sums: dict = {}
        for k, b_dig in enumerate(b_digits_reversed):
            pp_val = a * b_dig
            pp_right = right_col - k
            pp_str = str(pp_val)
            for j, ch in enumerate(pp_str):
                col = pp_right - len(pp_str) + 1 + j
                col_sums[col] = col_sums.get(col, 0) + int(ch)
        carry_c = 0
        for col in range(right_col, col_offset - 2, -1):
            total = col_sums.get(col, 0) + carry_c
            carry_c = total // 10
            if carry_c > 0 and col > col_offset:
                # Stacked-carry variant: when carry >= 2, children sometimes write two
                # stacked "1"s instead of a single digit. Emit at 10% probability.
                if carry_c >= 2 and rng.random() < 0.10:
                    for stacked_row in (add_carry_row, add_carry_row - 1):
                        tokens.append(LayoutToken(
                            row=stacked_row,
                            col=col - 1,
                            flattened_label="carry_1",
                            glyph_key="1",
                            yolo_class_name="digit_carry",
                            token_scale=CARRY_BORROW_TOKEN_SCALE,
                        ))
                else:
                    tokens.append(LayoutToken(
                        row=add_carry_row,
                        col=col - 1,
                        flattened_label=f"carry_{carry_c}",
                        glyph_key=str(carry_c),
                        yolo_class_name="digit_carry",
                        token_scale=CARRY_BORROW_TOKEN_SCALE,
                    ))

        bar2_row = 3 + n_b * 2 + 1
        tokens.append(LayoutToken(
            row=bar2_row, col=col_offset - 1,
            flattened_label="result_bar", glyph_key="result_bar", yolo_class_name="result_bar",
            segment_kind="final_result_bar",
        ))

        result_row = 3 + n_b * 2 + 2
        r_left = right_col - len(r_str) + 1
        for j, ch in enumerate(r_str):
            tokens.append(LayoutToken(
                row=result_row, col=r_left + j,
                flattened_label=f"main_{ch}", glyph_key=ch, yolo_class_name="digit_main",
            ))

        # A: collapse empty PP carry rows (and the addition-carry row).
        # For PP k (0-indexed), the nominal carry row is 3+k*2.  If no token
        # actually occupies that row, it is empty and would leave a visible blank
        # gap.  Also include the addition-carry row (3+n_b*2) which is empty when
        # no carry propagates during the final addition step.
        occupied_rows: set = {t.row for t in tokens}
        # Candidate carry rows: PP carries + the addition-carry row.
        candidate_carry_rows = sorted(
            {3 + k * 2 for k in range(n_b)} | {3 + n_b * 2}
        )
        empty_carry_rows = sorted(r for r in candidate_carry_rows if r not in occupied_rows)
        if empty_carry_rows:
            # Build a shift map: for each row > empty_carry_row, subtract the
            # number of empty carry rows that precede it.
            def _remap_row(r: int) -> int:
                shift = sum(1 for ecr in empty_carry_rows if ecr < r)
                return r - shift

            tokens = [dataclasses.replace(t, row=_remap_row(t.row)) for t in tokens]

    return EquationKind.multiply, tokens


def layout_short_division(
    dividend: int,
    divisor: int,
    rng: random.Random,
) -> Tuple[EquationKind, List[LayoutToken]]:
    """
    Short division bracket layout.
    Row 0: quotient digits right-justified to the last dividend column.
    Row 1: divisor digits at cols -(len(divisor_str)+1)..-2, div_bracket at col -1, dividend at cols 0..n.
    Row 2: remainder digits right-justified (only emitted when remainder != 0).
    No step rows.
    Caller is responsible for ensuring divisor is 1-9 and dividend is 2-99.
    """
    quotient = dividend // divisor
    remainder = dividend % divisor
    dividend_str = str(dividend)
    divisor_str = str(divisor)
    n = len(dividend_str)
    tokens: List[LayoutToken] = []

    # row 0: quotient right-justified to last dividend col (col n-1)
    quotient_str = str(quotient)
    q_left = (n - 1) - len(quotient_str) + 1
    for j, ch in enumerate(quotient_str):
        tokens.append(LayoutToken(
            row=0, col=q_left + j,
            flattened_label=f"main_{ch}",
            glyph_key=ch,
            yolo_class_name="digit_main",
        ))

    # row 1: divisor + bracket + dividend
    divisor_start_col = -(len(divisor_str) + 1)
    for j, ch in enumerate(divisor_str):
        tokens.append(LayoutToken(
            row=1, col=divisor_start_col + j,
            flattened_label=f"main_{ch}",
            glyph_key=ch,
            yolo_class_name="digit_main",
        ))
    tokens.append(LayoutToken(
        row=1, col=-1,
        flattened_label="div_bracket",
        glyph_key="div_bracket",
        yolo_class_name="divide_bracket",
    ))
    for j, ch in enumerate(dividend_str):
        tokens.append(LayoutToken(
            row=1, col=j,
            flattened_label=f"main_{ch}",
            glyph_key=ch,
            yolo_class_name="digit_main",
        ))

    # row 2: remainder (only when remainder != 0)
    if remainder != 0:
        rem_str = str(remainder)
        rem_left = (n - 1) - len(rem_str) + 1
        for j, ch in enumerate(rem_str):
            tokens.append(LayoutToken(
                row=2, col=rem_left + j,
                flattened_label=f"main_{ch}",
                glyph_key=ch,
                yolo_class_name="digit_main",
            ))

    return EquationKind.divide, tokens


def layout_simple_division(
    a: int,
    b: int,
    rng: random.Random,
) -> Tuple[EquationKind, List[LayoutToken]]:
    """
    Simple vertical-stack division layout (like addition/subtraction format).
    Exact division only: a = b * quotient (no remainder).
    Row 0: dividend digits (main_*).
    Row 1: op_divide token (col col_offset-1) + divisor digits.
    Row 2: result_bar.
    Row 3: quotient digits.
    No carry/borrow tokens.
    """
    if a % b != 0:
        raise ValueError(f"layout_simple_division requires exact division: {a} % {b} = {a % b}")
    quotient = a // b
    a_str = str(a)
    b_str = str(b)
    q_str = str(quotient)
    width = max(len(a_str), len(b_str), len(q_str))
    col_offset = 2
    tokens: List[LayoutToken] = []

    a_lead = width - len(a_str)
    for i, d in enumerate(_digit_to_cols(a, width)):
        if i < a_lead:
            continue
        tokens.append(LayoutToken(
            row=0, col=col_offset + i,
            flattened_label=f"main_{d}",
            glyph_key=str(d),
            yolo_class_name="digit_main",
        ))

    tokens.append(LayoutToken(
        row=1, col=col_offset - 1,
        flattened_label="op_divide",
        glyph_key="divide",
        yolo_class_name="operator",
        token_scale=OPERATOR_TOKEN_SCALE,
    ))
    b_lead = width - len(b_str)
    for i, d in enumerate(_digit_to_cols(b, width)):
        if i < b_lead:
            continue
        tokens.append(LayoutToken(
            row=1, col=col_offset + i,
            flattened_label=f"main_{d}",
            glyph_key=str(d),
            yolo_class_name="digit_main",
        ))

    tokens.append(LayoutToken(
        row=2, col=col_offset - 1,
        flattened_label="result_bar",
        glyph_key="result_bar",
        yolo_class_name="result_bar",
        segment_kind="final_result_bar",
    ))

    q_lead = width - len(q_str)
    for i, d in enumerate(_digit_to_cols(quotient, width)):
        if i < q_lead:
            continue
        tokens.append(LayoutToken(
            row=3, col=col_offset + i,
            flattened_label=f"main_{d}",
            glyph_key=str(d),
            yolo_class_name="digit_main",
        ))

    return EquationKind.divide, tokens


# ---------------------------------------------------------------------------
# Workstream B: layout variant functions (no-bar / op-right)
# ---------------------------------------------------------------------------


def layout_addition_no_bar(a: int, b: int, rng: random.Random) -> Tuple[EquationKind, List[LayoutToken]]:
    """
    Same as layout_addition but result_bar token removed entirely; result digits also removed.
    Models the child habit of writing operands + operator without drawing the result bar.
    The assembler will correctly fire OOD_REASON_NO_STRUCTURAL_TOKEN for this scene.
    """
    _, tokens = layout_addition(a, b, rng)
    tokens = [t for t in tokens if t.flattened_label != "result_bar"]
    max_row = max(t.row for t in tokens) if tokens else 0
    # Remove result row (row_res = row after result_bar which we removed)
    # result bar was on row 3, result digits on row 4; remove row 4
    tokens = [t for t in tokens if t.row != max_row]
    return EquationKind.add, tokens


def layout_subtraction_no_bar(a: int, b: int) -> Tuple[EquationKind, List[LayoutToken]]:
    """
    Same as layout_subtraction but result_bar and result digits removed.
    Models the child writing operands + minus without drawing the result bar.
    """
    _, tokens = layout_subtraction(a, b)
    tokens = [t for t in tokens if t.flattened_label != "result_bar"]
    if tokens:
        max_row = max(t.row for t in tokens)
        tokens = [t for t in tokens if t.row != max_row]
    return EquationKind.subtract, tokens


def layout_multiplication_simple_op_right(a: int, b: int, rng: random.Random) -> Tuple[EquationKind, List[LayoutToken]]:
    """
    N x 1 multiplication with the operator placed to the right of the multiplier row.
    Mirrors addition_op_right pattern for the multiply case.
    Only valid for single-digit b (multiplication_simple).
    """
    b = b % 10 if b >= 10 else b
    if b == 0:
        b = 1
    _, tokens = layout_multiplication(a, b, rng)
    # Find the op_times token and move it to the right
    scene_width = max(len(str(a)), len(str(b)), len(str(a * b)))
    col_offset = 2
    new_tokens = []
    for t in tokens:
        if t.flattened_label == "op_times" and t.row == 2:
            # Move from col_offset-1 to col_offset+scene_width
            import dataclasses
            t = dataclasses.replace(t, col=col_offset + scene_width)
        new_tokens.append(t)
    return EquationKind.multiply, new_tokens


def layout_multiplication_simple_no_bar(a: int, b: int, rng: random.Random) -> Tuple[EquationKind, List[LayoutToken]]:
    """
    N x 1 multiplication with result_bar and result row removed.
    Models child writing operands + operator without drawing the result bar.
    Only valid for single-digit b (multiplication_simple).
    """
    b = b % 10 if b >= 10 else b
    if b == 0:
        b = 1
    _, tokens = layout_multiplication(a, b, rng)
    tokens = [t for t in tokens if t.flattened_label != "result_bar"]
    if tokens:
        max_row = max(t.row for t in tokens)
        tokens = [t for t in tokens if t.row != max_row]
    return EquationKind.multiply, tokens


def layout_subtraction_heavy_borrow(rng: random.Random) -> Tuple[EquationKind, List[LayoutToken]]:
    """3-digit minus 3-digit subtraction with at least 2 forced borrows.

    Constructs minuend a and subtrahend b so that a >= b as whole numbers while
    at least 2 individual columns have b_i > a_i (triggering borrows before propagation).
    The construction works from least-significant to most-significant digit to keep
    the whole-number constraint without a post-hoc swap that would destroy the borrow
    structure.
    """
    def _force_borrow_operands(rng: random.Random, n_forced: int) -> Tuple[int, int]:
        # Build from most-significant digit first.
        # Strategy: for the hundreds digit, ensure a[0] > b[0] so a > b as 3-digit numbers
        # regardless of what happens in tens/units columns.  Then freely force borrows in
        # the remaining columns.
        a_digits = [0, 0, 0]
        b_digits = [0, 0, 0]

        # Hundreds (index 0): a[0] must be strictly > b[0] to guarantee a >= b overall.
        a_digits[0] = rng.randint(2, 9)
        b_digits[0] = rng.randint(1, a_digits[0] - 1)

        # Determine which of the remaining two columns (tens, units = indices 1, 2) get borrows.
        n_remaining = min(n_forced, 2)  # can force at most 2 remaining columns
        forced_cols = set(rng.sample([1, 2], n_remaining))

        for i in [1, 2]:
            a_digits[i] = rng.randint(0, 8)  # keep room for b > a at this col
            if i in forced_cols:
                lo = a_digits[i] + 1
                b_digits[i] = rng.randint(lo, 9)
            else:
                b_digits[i] = rng.randint(0, a_digits[i])

        a = a_digits[0] * 100 + a_digits[1] * 10 + a_digits[2]
        b = b_digits[0] * 100 + b_digits[1] * 10 + b_digits[2]
        # Belt-and-suspenders: guarantee a >= b (should always hold by construction above).
        if a < b:
            a, b = b, a
        a = max(100, a)
        b = max(1, b)
        return a, b

    n_forced = 3 if rng.random() < 0.5 else 2
    a, b = _force_borrow_operands(rng, n_forced)
    return layout_subtraction(a, b)


def layout_addition_dense_carries(rng: random.Random) -> Tuple[EquationKind, List[LayoutToken]]:
    """3-digit plus 3-digit with 2-3 columns generating carries.

    Operand pairs are sampled so that column-wise sums produce carries in at least
    2 columns. Forced columns use a_i >= 1 so there is always a valid b_i in [10-a_i, 9].
    """
    def _force_carry_operands(rng: random.Random, n_forced: int) -> Tuple[int, int]:
        # Sample a digits ensuring forced columns have a_i >= 1 (so 10-a_i <= 9).
        a_digits = [rng.randint(1, 9), rng.randint(1, 9), rng.randint(1, 9)]
        b_digits = [0, 0, 0]
        forced_cols = set(rng.sample(range(3), n_forced))
        for i in range(3):
            if i in forced_cols:
                # Column sum a_i + b_i >= 10 → b_i in [10 - a_i, 9].
                lo = 10 - a_digits[i]  # a_i >= 1 guarantees lo <= 9
                b_digits[i] = rng.randint(lo, 9)
            else:
                # Column sum < 10 → b_i in [0, 9 - a_i].
                hi = 9 - a_digits[i]
                b_digits[i] = rng.randint(0, max(0, hi))
        a = a_digits[0] * 100 + a_digits[1] * 10 + a_digits[2]
        b = b_digits[0] * 100 + b_digits[1] * 10 + b_digits[2]
        a = max(100, a)
        b = max(1, b)
        return a, b

    n_forced = 3 if rng.random() < 0.5 else 2
    a, b = _force_carry_operands(rng, n_forced)
    return layout_addition(a, b, rng)


# ---------------------------------------------------------------------------
# iter11 crowded double-digit-in-one-cell variants (2026-06-22)
# Two digit glyphs packed in ONE grid cell (same row/col) via cell_subpos, modelling a kid writing
# a borrow value "16" or a crowded carry. row/col GT is unchanged (both glyphs keep the cell). The
# recognizer learns to READ the tight pair; geometric inference will split them into two columns and
# grid_mapping's 1.5-snap re-merges borrow/carry downstream (main rows are unaffected here).
# ---------------------------------------------------------------------------

_DOUBLE_DIGIT_SUBPOS = 0.26  # left/right offset within the cell as a fraction of column pitch


def _pack_double_digit_borrows(
    tokens: List[LayoutToken], rng: random.Random
) -> List[LayoutToken]:
    """Co-locate each borrow "1" with the minuend digit it modifies, e.g. 6 -> "16", in one cell."""
    main_by_cell = {
        (t.row, t.col): t for t in tokens if t.yolo_class_name == "digit_main"
    }
    main_rows = [t.row for t in tokens if t.yolo_class_name == "digit_main"]
    minuend_row = min(main_rows) if main_rows else 0
    sc = CARRY_BORROW_TOKEN_SCALE
    out: List[LayoutToken] = []
    for t in tokens:
        if t.yolo_class_name == "digit_borrow":
            out.append(dataclasses.replace(t, token_scale=sc, cell_subpos=-_DOUBLE_DIGIT_SUBPOS))
            md = main_by_cell.get((minuend_row, t.col))
            digit = md.glyph_key if md is not None else str(rng.randint(0, 9))
            out.append(
                LayoutToken(
                    row=t.row,
                    col=t.col,
                    flattened_label=f"borrow_{digit}",
                    glyph_key=digit,
                    yolo_class_name="digit_borrow",
                    token_scale=sc,
                    cell_subpos=_DOUBLE_DIGIT_SUBPOS,
                )
            )
        else:
            out.append(t)
    return out


def _pack_double_digit_carries(
    tokens: List[LayoutToken], rng: random.Random
) -> List[LayoutToken]:
    """Render each carry as a two-digit value "1X" packed in one cell (X drawn for visual crowding)."""
    sc = CARRY_BORROW_TOKEN_SCALE
    out: List[LayoutToken] = []
    for t in tokens:
        if t.yolo_class_name == "digit_carry":
            out.append(dataclasses.replace(t, token_scale=sc, cell_subpos=-_DOUBLE_DIGIT_SUBPOS))
            digit = str(rng.randint(0, 9))
            out.append(
                LayoutToken(
                    row=t.row,
                    col=t.col,
                    flattened_label=f"carry_{digit}",
                    glyph_key=digit,
                    yolo_class_name="digit_carry",
                    token_scale=sc,
                    cell_subpos=_DOUBLE_DIGIT_SUBPOS,
                )
            )
        else:
            out.append(t)
    return out


def layout_subtraction_crowded_borrow(
    rng: random.Random,
) -> Tuple[EquationKind, List[LayoutToken]]:
    """Subtraction with forced borrows rendered as two digits in one cell (e.g. "16")."""
    kind, tokens = layout_subtraction_heavy_borrow(rng)
    return kind, _pack_double_digit_borrows(tokens, rng)


def layout_addition_crowded_carry(
    rng: random.Random,
) -> Tuple[EquationKind, List[LayoutToken]]:
    """Addition with dense carries rendered as two digits in one cell (crowded carry)."""
    kind, tokens = layout_addition_dense_carries(rng)
    return kind, _pack_double_digit_carries(tokens, rng)


def layout_bare_digits(rng: random.Random) -> Tuple[EquationKind, List[LayoutToken]]:
    """1-3 main-line digit tokens in a single row, no operator, no bar, no bracket.

    Models the 'just 24' robustness case: a child writes a number without
    starting an equation. Returns EquationKind.ood_unknown so the GNN eq_type
    head supervision is masked to -1 during training.
    """
    n_digits = rng.randint(1, 3)
    tokens: List[LayoutToken] = []
    for col in range(n_digits):
        d = rng.randint(0, 9)
        tokens.append(LayoutToken(
            row=0,
            col=col,
            flattened_label=f"main_{d}",
            glyph_key=str(d),
            yolo_class_name="digit_main",
            token_scale=1.0,
        ))
    return EquationKind.ood_unknown, tokens


def layout_bare_digit_grid(rng: random.Random) -> Tuple[EquationKind, List[LayoutToken]]:
    """A multi-row, multi-column block of bare main-line digits.

    n_rows in {2, 3} by n_cols in {2, 3} of main_<d> tokens placed at grid
    (row, col) positions with random digits 0-9. No operator, no result_bar, no
    div_bracket. Unlike single-row layout_bare_digits, this exercises the
    row/col-clustering heads across multiple distinct rows and columns. Returns
    EquationKind.ood_unknown so the GNN eq_type head supervision is masked to -1
    during training (a bare grid is not an equation).
    """
    n_rows = rng.randint(2, 3)
    n_cols = rng.randint(2, 3)
    tokens: List[LayoutToken] = []
    for row in range(n_rows):
        for col in range(n_cols):
            d = rng.randint(0, 9)
            tokens.append(LayoutToken(
                row=row,
                col=col,
                flattened_label=f"main_{d}",
                glyph_key=str(d),
                yolo_class_name="digit_main",
                token_scale=1.0,
            ))
    return EquationKind.ood_unknown, tokens


def layout_standalone_bar(rng: random.Random) -> Tuple[EquationKind, List[LayoutToken]]:
    """Single result_bar token on canvas. No digits, no operator, no bracket.

    Trains YOLO to detect result_bar in isolation. The assembler routes this
    scene to OOD (OOD_REASON_NO_DIGITS) since there are no digit detections.
    Returns EquationKind.ood_unknown so GNN eq_type supervision is masked.
    """
    tokens: List[LayoutToken] = [
        LayoutToken(
            row=0,
            col=0,
            flattened_label="result_bar",
            glyph_key="result_bar",
            yolo_class_name="result_bar",
            token_scale=1.0,
            segment_kind="final_result_bar",
        )
    ]
    return EquationKind.ood_unknown, tokens


def layout_standalone_bracket(rng: random.Random) -> Tuple[EquationKind, List[LayoutToken]]:
    """Single div_bracket token on canvas. No digits, no operator, no bar.

    Trains YOLO to detect div_bracket in isolation. The assembler routes this
    scene to OOD (OOD_REASON_NO_DIGITS) since there are no digit detections.
    Returns EquationKind.ood_unknown so GNN eq_type supervision is masked.
    """
    tokens: List[LayoutToken] = [
        LayoutToken(
            row=0,
            col=0,
            flattened_label="div_bracket",
            glyph_key="div_bracket",
            yolo_class_name="divide_bracket",
            token_scale=1.0,
        )
    ]
    return EquationKind.ood_unknown, tokens


def layout_addition_no_carries(rng: random.Random) -> Tuple[EquationKind, List[LayoutToken]]:
    """3-digit plus 3-digit where no column sum reaches 10 (zero carries).

    This is the explicit zero-carry counterpart to addition_dense_carries and gives the
    model clear negative examples for the carry annotation role.
    """
    def _no_carry_operands(rng: random.Random) -> Tuple[int, int]:
        a_digits = [rng.randint(1, 4), rng.randint(0, 4), rng.randint(0, 4)]
        b_digits = [rng.randint(0, 9 - a_digits[i]) for i in range(3)]
        a = a_digits[0] * 100 + a_digits[1] * 10 + a_digits[2]
        b = b_digits[0] * 100 + b_digits[1] * 10 + b_digits[2]
        a = max(100, a)
        b = max(1, b)
        return a, b

    a, b = _no_carry_operands(rng)
    return layout_addition(a, b, rng)


# ---------------------------------------------------------------------------
# layout_bounds (needed internally by _sample_division_layout)
# ---------------------------------------------------------------------------


def _layout_bounds(tokens: List[LayoutToken]) -> Tuple[int, int, int, int]:
    """Internal alias used by _sample_division_layout to avoid circular import."""
    if not tokens:
        return 0, 0, 0, 0
    rs = [t.row for t in tokens]
    cs = [t.col for t in tokens]
    return min(rs), max(rs), min(cs), max(cs)
