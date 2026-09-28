from __future__ import annotations

import unittest
from pathlib import Path

from src.core.ontology import (
    EXCLUDED_DATASET_LABELS,
    YOLO_CLASS_NAMES,
    YOLO_NAME_TO_ID,
    stage2_labels_ordered,
)


class TestOntologyUpdate(unittest.TestCase):
    def test_yolo_has_divide_bracket_as_6th(self) -> None:
        self.assertIn("divide_bracket", YOLO_CLASS_NAMES)
        self.assertEqual(len(YOLO_CLASS_NAMES), 6)
        self.assertEqual(YOLO_CLASS_NAMES[5], "divide_bracket")

    def test_divide_bracket_not_excluded(self) -> None:
        self.assertNotIn("divide_bracket", EXCLUDED_DATASET_LABELS)

    def test_stage2_labels_has_36_entries(self) -> None:
        labels = stage2_labels_ordered()
        self.assertEqual(len(labels), 36)

    def test_stage2_labels_last_is_divide_bracket(self) -> None:
        labels = stage2_labels_ordered()
        self.assertEqual(labels[-1], "div_bracket")

    def test_yolo_name_to_id_updated(self) -> None:
        self.assertEqual(YOLO_NAME_TO_ID["divide_bracket"], 5)


from src.generation.layouts import LayoutToken


class TestGtJsonSchema(unittest.TestCase):
    """Verify the gt.json dict keys produced for each token type."""

    def _make_symbol_token(self) -> LayoutToken:
        return LayoutToken(
            row=1, col=3,
            flattened_label="main_5",
            glyph_key="5",
            yolo_class_name="digit_main",
        )

    def _make_bar_token(self) -> LayoutToken:
        return LayoutToken(
            row=3, col=1,
            flattened_label="result_bar",
            glyph_key="result_bar",
            yolo_class_name="result_bar",
        )

    def _token_to_gt_dict(self, tok: LayoutToken, bbox: list) -> dict:
        """Mirror of the dict construction in synth_yolo._draw_equation_block."""
        return {
            "fine_label": tok.flattened_label,
            "glyph_key": tok.glyph_key,
            "yolo_class": tok.yolo_class_name,
            "row_index": tok.row,
            "col_index": tok.col,
            "bbox": bbox,
            "equation_idx": 0,
        }

    def test_symbol_gt_dict_keys(self) -> None:
        tok = self._make_symbol_token()
        d = self._token_to_gt_dict(tok, [10, 20, 30, 40])
        for required_key in ("fine_label", "row_index", "col_index", "bbox", "yolo_class"):
            self.assertIn(required_key, d, f"missing key: {required_key}")
        for old_key in ("flattened", "grid_row", "grid_col", "bbox_px"):
            self.assertNotIn(old_key, d, f"old key still present: {old_key}")

    def test_gt_payload_top_level_keys(self) -> None:
        payload = {
            "equation_type": "add",
            "split": "train",
            "symbols": [],
        }
        self.assertIn("equation_type", payload)
        self.assertNotIn("equation_kind", payload)
        self.assertIn("symbols", payload)
        self.assertNotIn("tokens", payload)


import random
from src.generation.layouts import layout_addition


class TestAdditionFourDigitResult(unittest.TestCase):
    def test_999_plus_999_generates_without_error(self) -> None:
        rng = random.Random(42)
        kind, tokens = layout_addition(999, 999, rng)
        result_digits = [t for t in tokens if t.row == 4]
        self.assertEqual(len(result_digits), 4)  # 1998 has 4 digits

    def test_result_digits_are_1998(self) -> None:
        rng = random.Random(42)
        _, tokens = layout_addition(999, 999, rng)
        result_row = sorted(
            [t for t in tokens if t.row == 4],
            key=lambda t: t.col,
        )
        self.assertEqual(
            "".join(t.flattened_label.split("_")[1] for t in result_row),
            "1998",
        )


from src.generation.layouts import layout_subtraction


class TestSubtractionBorrowStyleB(unittest.TestCase):
    def _borrow_tokens(self, a: int, b: int):
        _, tokens = layout_subtraction(a, b)
        return [t for t in tokens if t.yolo_class_name == "digit_borrow"]

    def test_no_borrow_emits_no_borrow_tokens(self) -> None:
        borrows = self._borrow_tokens(456, 123)
        self.assertEqual(borrows, [])

    def test_single_borrow_emits_reduced_digit(self) -> None:
        # 431 - 215: ones col: 1 < 5, borrow from tens (3 → 2, writes "2" annotation)
        # Also emits borrow_1 at the borrower column (ones col).
        borrows = self._borrow_tokens(431, 215)
        labels = {t.flattened_label for t in borrows}
        self.assertIn("borrow_2", labels)   # tens column: 3 − 1 = 2
        self.assertIn("borrow_1", labels)   # ones column: received the borrow
        self.assertEqual(len(borrows), 2)

    def test_cascade_through_zero_emits_borrow_9(self) -> None:
        # 300 - 1: borrow chain through tens (0 → 9, writes "9") and hundreds (3 → 2, writes "2")
        # Also emits borrow_1 at the borrower column (ones col).
        borrows = self._borrow_tokens(300, 1)
        labels = {t.flattened_label for t in borrows}
        self.assertIn("borrow_9", labels)   # tens column: zero becomes 9 after +10 −1
        self.assertIn("borrow_2", labels)   # hundreds column: 3 − 1 = 2
        self.assertIn("borrow_1", labels)   # ones column: received the borrow
        self.assertEqual(len(borrows), 3)

    def test_borrow_token_is_at_row_0(self) -> None:
        borrows = self._borrow_tokens(431, 215)
        for tok in borrows:
            self.assertEqual(tok.row, 0)


from src.generation.layouts import _mul_single_pass_carries


class TestMulSinglePassCarries(unittest.TestCase):
    def test_no_carry_when_product_lt_10(self) -> None:
        # 2 × 3 = 6, no carry
        tokens = _mul_single_pass_carries(a_digits=[2], b_digit=3, right_col=4, carry_row=0)
        self.assertEqual(tokens, [])

    def test_single_carry_37x4(self) -> None:
        # 37 × 4: ones 7×4=28 carry 2 to tens col; tens 3×4+2=14 no further carry (j=0)
        # a_digits=[3,7], right_col=5, carry_row=0
        tokens = _mul_single_pass_carries(a_digits=[3, 7], b_digit=4, right_col=5, carry_row=0)
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].flattened_label, "carry_2")
        self.assertEqual(tokens[0].col, 4)   # right_col - 1
        self.assertEqual(tokens[0].row, 0)

    def test_cascading_carries_379x4(self) -> None:
        # 379 × 4: 9×4=36 c=3; 7×4+3=31 c=3; 3×4+3=15 j=0 no emit
        # a_digits=[3,7,9], right_col=5, carry_row=0
        tokens = _mul_single_pass_carries(a_digits=[3, 7, 9], b_digit=4, right_col=5, carry_row=0)
        self.assertEqual(len(tokens), 2)
        cols = sorted(t.col for t in tokens)
        self.assertEqual(cols, [3, 4])
        labels = {t.col: t.flattened_label for t in tokens}
        self.assertEqual(labels[4], "carry_3")  # from ones
        self.assertEqual(labels[3], "carry_3")  # from tens


from src.generation.layouts import layout_multiplication, EquationKind


class TestLayoutMultiplication(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = random.Random(42)

    def _tokens_at_row(self, tokens, row):
        return sorted([t for t in tokens if t.row == row], key=lambda t: t.col)

    # --- N×1 cases ---

    def test_1x1_has_4_fixed_row_slots(self) -> None:
        _, tokens = layout_multiplication(7, 8, self.rng)
        rows = {t.row for t in tokens}
        # carry row (0) may be absent; rows 1,2,3,4 always present
        self.assertIn(1, rows)
        self.assertIn(2, rows)
        self.assertIn(3, rows)
        self.assertIn(4, rows)

    def test_1x1_result_correct(self) -> None:
        _, tokens = layout_multiplication(7, 8, self.rng)
        result_toks = self._tokens_at_row(tokens, 4)
        result_str = "".join(t.flattened_label.split("_")[1] for t in result_toks)
        self.assertEqual(result_str, "56")

    def test_n1_carry_appears_above_multiplicand_row(self) -> None:
        # 37 × 4 = 148: carry 2 in tens col (row 0)
        _, tokens = layout_multiplication(37, 4, self.rng)
        carry_toks = [t for t in tokens if t.yolo_class_name == "digit_carry"]
        self.assertTrue(len(carry_toks) > 0)
        for ct in carry_toks:
            self.assertEqual(ct.row, 0)

    def test_n1_multiplicand_at_row_1(self) -> None:
        _, tokens = layout_multiplication(23, 5, self.rng)
        row1 = self._tokens_at_row(tokens, 1)
        main_toks = [t for t in row1 if t.yolo_class_name == "digit_main"]
        self.assertTrue(len(main_toks) >= 1)

    def test_n1_result_row_is_4(self) -> None:
        _, tokens = layout_multiplication(99, 9, self.rng)   # 99×9=891
        result_toks = self._tokens_at_row(tokens, 4)
        result_str = "".join(t.flattened_label.split("_")[1] for t in result_toks)
        self.assertEqual(result_str, "891")

    # --- N×2 cases ---

    def test_n2_has_two_pp_rows(self) -> None:
        # 23 × 45 = 1035; PP1 at row 4, PP2 at row 6
        _, tokens = layout_multiplication(23, 45, self.rng)
        self.assertTrue(any(t.row == 4 for t in tokens))  # PP1
        self.assertTrue(any(t.row == 6 for t in tokens))  # PP2

    def test_n2_pp1_is_ones_digit_product(self) -> None:
        # 23 × 45: PP1 = 23×5 = 115
        _, tokens = layout_multiplication(23, 45, self.rng)
        pp1_toks = self._tokens_at_row(tokens, 4)
        pp1_str = "".join(t.flattened_label.split("_")[1] for t in pp1_toks if t.yolo_class_name == "digit_main")
        self.assertEqual(pp1_str, "115")

    def test_n2_pp2_is_tens_digit_product_with_trailing_zero(self) -> None:
        # 23 × 45: PP2 = 23×4 = 92 with 1 trailing zero → displayed as "920" (Style B)
        _, tokens = layout_multiplication(23, 45, self.rng)
        pp2_toks = self._tokens_at_row(tokens, 6)
        pp2_str = "".join(t.flattened_label.split("_")[1] for t in pp2_toks if t.yolo_class_name == "digit_main")
        self.assertEqual(pp2_str, "920")

    def test_n2_pp2_shares_rightmost_col_with_pp1(self) -> None:
        # Style B: both PP rows right-justify to the same column; trailing zeros fill the right side.
        # PP2 (k=1) has 1 trailing zero: its rightmost digit (a zero) is at the same col as PP1's rightmost.
        _, tokens = layout_multiplication(23, 45, self.rng)
        pp1_main = [t for t in tokens if t.row == 4 and t.yolo_class_name == "digit_main"]
        pp2_main = [t for t in tokens if t.row == 6 and t.yolo_class_name == "digit_main"]
        self.assertEqual(max(t.col for t in pp2_main), max(t.col for t in pp1_main))

    def test_n2_result_correct(self) -> None:
        # 23 × 45 = 1035, at row 9
        _, tokens = layout_multiplication(23, 45, self.rng)
        result_toks = self._tokens_at_row(tokens, 9)
        result_str = "".join(t.flattened_label.split("_")[1] for t in result_toks if t.yolo_class_name == "digit_main")
        self.assertEqual(result_str, "1035")

    def test_n2_two_result_bars(self) -> None:
        _, tokens = layout_multiplication(23, 45, self.rng)
        bar_toks = [t for t in tokens if t.flattened_label == "result_bar"]
        self.assertEqual(len(bar_toks), 2)
        self.assertIn(2, {t.row for t in bar_toks})  # first bar
        self.assertIn(8, {t.row for t in bar_toks})  # second bar

    # --- N×3 cases ---

    def test_n3_has_three_pp_rows(self) -> None:
        # 12 × 123 = 1476; three partial-product rows exist above the final result bar.
        # After empty-carry-row compaction the exact row numbers may vary; verify by
        # counting distinct rows occupied by digit_main tokens between the two bars.
        _, tokens = layout_multiplication(12, 123, self.rng)
        bar_rows = sorted({t.row for t in tokens if t.flattened_label == "result_bar"})
        self.assertGreaterEqual(len(bar_rows), 2, "Expected at least two result_bar rows")
        bar1_row, bar2_row = bar_rows[0], bar_rows[-1]
        pp_rows = {t.row for t in tokens
                   if t.yolo_class_name == "digit_main" and bar1_row < t.row < bar2_row}
        self.assertEqual(len(pp_rows), 3, f"Expected 3 PP rows between bars, got {sorted(pp_rows)}")

    def test_n3_result_row_is_last(self) -> None:
        # After compaction the result row is max(t.row) for digit_main tokens.
        _, tokens = layout_multiplication(12, 123, self.rng)
        result_row = max(t.row for t in tokens if t.yolo_class_name == "digit_main")
        result_toks = [t for t in tokens if t.row == result_row and t.yolo_class_name == "digit_main"]
        result_str = "".join(t.flattened_label.split("_")[1] for t in sorted(result_toks, key=lambda t: t.col))
        self.assertEqual(result_str, str(12 * 123))

    def test_equation_kind_is_multiply(self) -> None:
        kind, _ = layout_multiplication(3, 4, self.rng)
        self.assertEqual(kind, EquationKind.multiply)

    def test_result_cap_removed_large_product(self) -> None:
        # 999 × 999 = 998001 (6 digits) — must not raise
        kind, tokens = layout_multiplication(999, 999, self.rng)
        result_row = max(t.row for t in tokens if t.yolo_class_name == "digit_main")
        result_toks = sorted(
            [t for t in tokens if t.row == result_row and t.yolo_class_name == "digit_main"],
            key=lambda t: t.col,
        )
        result_str = "".join(t.flattened_label.split("_")[1] for t in result_toks)
        self.assertEqual(result_str, "998001")


from src.generation.layouts import DivStep, _div_steps


class TestDivSteps(unittest.TestCase):
    def test_exact_single_step(self) -> None:
        # 8 ÷ 4 = 2 R 0
        steps = _div_steps(8, 4)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].quotient_digit, 2)
        self.assertEqual(steps[0].product, 8)
        self.assertEqual(steps[0].remainder, 0)
        self.assertEqual(steps[0].right_col, 0)

    def test_two_step_division(self) -> None:
        # 96 ÷ 4 = 24 R 0
        steps = _div_steps(96, 4)
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0].quotient_digit, 2)   # 9÷4=2
        self.assertEqual(steps[0].product, 8)
        self.assertEqual(steps[0].remainder, 1)
        self.assertEqual(steps[0].right_col, 0)
        self.assertEqual(steps[1].quotient_digit, 4)   # 16÷4=4
        self.assertEqual(steps[1].product, 16)
        self.assertEqual(steps[1].remainder, 0)
        self.assertEqual(steps[1].right_col, 1)

    def test_four_step_division(self) -> None:
        # 9632 ÷ 4 = 2408
        steps = _div_steps(9632, 4)
        self.assertEqual(len(steps), 4)
        self.assertEqual(steps[0].quotient_digit, 2)
        self.assertEqual(steps[1].quotient_digit, 4)
        self.assertEqual(steps[2].quotient_digit, 0)
        self.assertEqual(steps[3].quotient_digit, 8)

    def test_zero_in_quotient(self) -> None:
        # 306 ÷ 3 = 102
        steps = _div_steps(306, 3)
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[1].quotient_digit, 0)
        self.assertEqual(steps[1].product, 0)

    def test_leading_bringdown(self) -> None:
        # 48 ÷ 6 = 8: first digit 4 < 6, so first step uses 48 (right_col=1)
        steps = _div_steps(48, 6)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].right_col, 1)

    def test_division_with_remainder(self) -> None:
        # 10 ÷ 3 = 3 R 1
        steps = _div_steps(10, 3)
        self.assertEqual(steps[-1].remainder, 1)


from src.generation.layouts import layout_long_division, _sample_division_layout


class TestLayoutLongDivision(unittest.TestCase):
    def _token_map(self, tokens):
        """Returns {(row, col): flattened_label} for quick lookup."""
        return {(t.row, t.col): t.flattened_label for t in tokens}

    def test_equation_kind_is_divide(self) -> None:
        rng = random.Random(0)
        kind, _ = layout_long_division(8, 4, rng)
        self.assertEqual(kind, EquationKind.divide)

    def test_simple_1step_quotient_token(self) -> None:
        # 8 ÷ 4 = 2: quotient digit 2 at row 0, col 0 (single-digit dividend → right_col=0)
        rng = random.Random(0)
        _, tokens = layout_long_division(8, 4, rng)
        tm = self._token_map(tokens)
        self.assertIn((0, 0), tm)
        self.assertEqual(tm[(0, 0)], "main_2")

    def test_dividend_at_row_1(self) -> None:
        rng = random.Random(0)
        _, tokens = layout_long_division(96, 4, rng)
        row1 = [t for t in tokens if t.row == 1]
        dividend_toks = [t for t in row1 if t.yolo_class_name == "digit_main" and t.col >= 0]
        # dividend 96 → two tokens at cols 0, 1
        digit_labels = "".join(t.flattened_label.split("_")[1] for t in sorted(dividend_toks, key=lambda t: t.col))
        self.assertEqual(digit_labels, "96")

    def test_divisor_at_negative_cols(self) -> None:
        rng = random.Random(0)
        _, tokens = layout_long_division(96, 4, rng)
        divisor_toks = [t for t in tokens if t.col < -1 and t.yolo_class_name == "digit_main"]
        # 1-digit divisor 4 at col -2
        self.assertTrue(len(divisor_toks) >= 1)
        self.assertTrue(all(t.col < 0 for t in divisor_toks))

    def test_div_bracket_at_col_neg1_row1(self) -> None:
        rng = random.Random(0)
        _, tokens = layout_long_division(96, 4, rng)
        bracket = [t for t in tokens if t.flattened_label == "div_bracket"]
        self.assertEqual(len(bracket), 1)
        self.assertEqual(bracket[0].row, 1)
        self.assertEqual(bracket[0].col, -1)
        self.assertEqual(bracket[0].yolo_class_name, "divide_bracket")

    def test_2step_has_correct_row_count(self) -> None:
        # 96 ÷ 4: 2 steps → total rows = 2 + 2*3 = 8 (rows 0..7)
        rng = random.Random(0)
        _, tokens = layout_long_division(96, 4, rng)
        max_row = max(t.row for t in tokens)
        self.assertEqual(max_row, 7)

    def test_4step_max_rows(self) -> None:
        # 9632 ÷ 4: 4 steps → total rows = 2 + 4*3 = 14, max row = 13
        rng = random.Random(0)
        _, tokens = layout_long_division(9632, 4, rng)
        max_row = max(t.row for t in tokens)
        self.assertEqual(max_row, 13)

    def test_zero_in_quotient_pp_is_zero(self) -> None:
        # 306 ÷ 3 = 102: step 1 (second step) has quotient digit 0
        rng = random.Random(0)
        _, tokens = layout_long_division(306, 3, rng)
        # step 1: PP at row 2+1*3 = 5
        pp_step1 = [t for t in tokens if t.row == 5 and t.yolo_class_name == "digit_main"]
        self.assertTrue(len(pp_step1) >= 1)
        self.assertEqual(pp_step1[0].flattened_label, "main_0")

    def test_exact_division_remainder_is_zero(self) -> None:
        # 8 ÷ 4: exact; final remainder row (row 4) token is main_0
        rng = random.Random(0)
        _, tokens = layout_long_division(8, 4, rng)
        final_row_toks = [t for t in tokens if t.row == 4 and t.yolo_class_name == "digit_main"]
        self.assertEqual(len(final_row_toks), 1)
        self.assertEqual(final_row_toks[0].flattened_label, "main_0")

    def test_sample_division_layout_returns_divide_kind(self) -> None:
        rng = random.Random(42)
        kind, tokens = _sample_division_layout(rng)
        self.assertEqual(kind, EquationKind.divide)
        self.assertTrue(len(tokens) > 0)


from src.generation.layouts import sample_layout


class TestSampleLayoutIncludesDivide(unittest.TestCase):
    def test_divide_kind_is_reachable(self) -> None:
        rng = random.Random(99)
        kinds_seen = set()
        for _ in range(200):
            kind, tokens = sample_layout(rng)
            kinds_seen.add(kind)
        self.assertIn(EquationKind.divide, kinds_seen)
        self.assertIn(EquationKind.add, kinds_seen)
        self.assertIn(EquationKind.subtract, kinds_seen)
        self.assertIn(EquationKind.multiply, kinds_seen)


from src.generation.synth_yolo import _apply_blank_result


class TestBlankResult(unittest.TestCase):
    def test_blank_result_removes_result_and_bar_rows(self) -> None:
        rng = random.Random(0)
        _, tokens = layout_addition(5, 3, rng)
        # addition: bar row=3, result row=4 — both must be removed
        result_toks_before = [t for t in tokens if t.row in {3, 4}]
        self.assertTrue(len(result_toks_before) > 0)

        blanked = _apply_blank_result(tokens, result_rows={3, 4})
        result_toks_after = [t for t in blanked if t.row in {3, 4}]
        self.assertEqual(result_toks_after, [])

    def test_non_result_rows_untouched(self) -> None:
        rng = random.Random(0)
        _, tokens = layout_addition(5, 3, rng)
        blanked = _apply_blank_result(tokens, result_rows={3, 4})
        non_result = [t for t in blanked if t.row not in {3, 4}]
        expected = [t for t in tokens if t.row not in {3, 4}]
        self.assertEqual(len(non_result), len(expected))

    def test_blank_result_removes_bar2_and_result_for_n2_multiply(self) -> None:
        # N×2: bar2=row 8, result=row 9 — both removed; bar1 (row 2) stays
        _, tokens = layout_multiplication(23, 45, random.Random(42))
        blanked = _apply_blank_result(tokens, result_rows={8, 9})
        self.assertEqual([t for t in blanked if t.row in {8, 9}], [])
        self.assertTrue(any(t.row == 2 for t in blanked))  # bar1 survives

    def test_blank_probability_roughly_20_percent(self) -> None:
        rng = random.Random(7)
        blanked_count = sum(
            1 for _ in range(500)
            if rng.random() < 0.20
        )
        self.assertGreater(blanked_count, 70)   # >> 0 (expected ~100)
        self.assertLess(blanked_count, 150)      # << 500


import numpy as np
import random as _random
from src.generation.strokes import draw_handwritten_bracket
from src.generation.handwriting_style import HandwritingStyle


class TestDivBracketRendering(unittest.TestCase):
    def _make_style(self) -> HandwritingStyle:
        return HandwritingStyle(
            rot_deg_min=0.0,
            rot_deg_max=0.0,
            jitter_min_px=0,
            jitter_max_px=0,
            glyph_broken_stroke_prob=0.0,
            crowding_factor=1.0,
        )

    def test_bracket_draws_pixels_on_canvas(self) -> None:
        canvas = np.full((200, 200), 255, dtype=np.uint8)
        # corner at (30, 20), horizontal arm RIGHT by 120, vertical arm DOWN by 80
        draw_handwritten_bracket(canvas, 30, 20, 120, 80, self._make_style(), _random.Random(42), 0.0)
        # horizontal arm: ink should appear to the right of corner_x=30
        self.assertTrue(np.any(canvas[15:30, 30:150] < 128))
        # vertical arm: ink should appear below corner_y=20
        self.assertTrue(np.any(canvas[20:100, 25:40] < 128))

    def test_bracket_returns_bounding_box(self) -> None:
        canvas = np.full((200, 200), 255, dtype=np.uint8)
        bbox = draw_handwritten_bracket(canvas, 30, 20, 120, 80, self._make_style(), _random.Random(42), 0.0)
        self.assertEqual(len(bbox), 4)
        min_x, min_y, max_x, max_y = bbox
        # bbox must be within canvas
        self.assertGreaterEqual(min_x, 0)
        self.assertGreaterEqual(min_y, 0)
        self.assertLessEqual(max_x, 200)
        self.assertLessEqual(max_y, 200)
        # horizontal arm extends right: max_x should be well to the right of corner
        self.assertGreater(max_x, 100)


import json
import tempfile
import shutil
from unittest import skipUnless

from src.core.config import DataPrepConfig

MANIFEST = Path(__file__).parents[1] / "data" / "generated" / "processed" / "yolo" / "symbol_assets_manifest.csv"


class TestEndToEndSmoke(unittest.TestCase):
    @skipUnless(MANIFEST.exists(), "symbol manifest not available — run validate first")
    def test_generated_scene_gt_schema(self) -> None:
        from src.generation.synth_yolo import build_synthetic_yolo_dataset
        from src.generation.layouts import SceneCase

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            tmp_config = DataPrepConfig.from_project_root(tmp_root)
            (tmp_config.yolo_dir).mkdir(parents=True, exist_ok=True)
            shutil.copy(MANIFEST, tmp_config.yolo_dir / "symbol_assets_manifest.csv")

            case = SceneCase.addition
            out_dir = tmp_root / "synth_run"
            out_dir.mkdir(parents=True, exist_ok=True)
            build_synthetic_yolo_dataset(tmp_config, case, split_name="train", images_per_split=20, out_dir=out_dir)

            gt_dir = out_dir / "train" / "ground_truth"
            gt_files = list(gt_dir.glob("*.gt.json"))
            self.assertGreater(len(gt_files), 0)

            for gt_path in gt_files:
                data = json.loads(gt_path.read_text())
                # scene-level keys
                self.assertIn("equation_type", data, f"missing equation_type in {gt_path.name}")
                self.assertIn("symbols", data, f"missing symbols in {gt_path.name}")
                self.assertNotIn("equation_kind", data)
                self.assertNotIn("tokens", data)

                for sym in data["symbols"]:
                    for key in ("fine_label", "row_index", "col_index", "bbox", "yolo_class"):
                        self.assertIn(key, sym, f"missing {key} in symbol of {gt_path.name}")
                    self.assertNotIn("flattened", sym)
                    self.assertNotIn("grid_row", sym)
                    self.assertNotIn("grid_col", sym)
                    self.assertNotIn("bbox_px", sym)
                    self.assertIsInstance(sym["bbox"], list)
                    self.assertEqual(len(sym["bbox"]), 4)

    @skipUnless(MANIFEST.exists(), "symbol manifest not available — run validate first")
    def test_divide_scenes_have_bracket_token(self) -> None:
        from src.generation.synth_yolo import build_synthetic_yolo_dataset
        from src.generation.layouts import SceneCase

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            tmp_config = DataPrepConfig.from_project_root(tmp_root)
            (tmp_config.yolo_dir).mkdir(parents=True, exist_ok=True)
            shutil.copy(MANIFEST, tmp_config.yolo_dir / "symbol_assets_manifest.csv")

            case = SceneCase.division_long
            out_dir = tmp_root / "synth_run"
            out_dir.mkdir(parents=True, exist_ok=True)
            build_synthetic_yolo_dataset(tmp_config, case, split_name="train", images_per_split=80, out_dir=out_dir)

            gt_dir = out_dir / "train" / "ground_truth"
            div_scenes = []
            for p in gt_dir.glob("*.gt.json"):
                d = json.loads(p.read_text())
                if d.get("equation_type") == "divide":
                    div_scenes.append(d)

            self.assertGreater(len(div_scenes), 0, "no divide scenes generated in 80 scenes")

            for scene in div_scenes:
                labels = {s["fine_label"] for s in scene["symbols"]}
                self.assertIn("div_bracket", labels, "div_bracket missing from divide scene")


from src.generation.layouts import (
    EquationKind, SceneCase,
    layout_short_division, layout_simple_division, layout_subtraction,
    sample_layout_for_case,
)


class TestNewLayoutsAndBorrowFix(unittest.TestCase):

    def test_layout_short_division_has_no_step_rows(self):
        """Short division must never emit step rows (row >= 3 is only valid for remainder)."""
        rng = random.Random(42)
        for _ in range(10):
            dividend = rng.randint(2, 99)
            divisor = rng.randint(1, 9)
            _, tokens = layout_short_division(dividend, divisor, rng)
            # row 2 = remainder (optional), no row >= 3 allowed
            step_toks = [t for t in tokens if t.row >= 3]
            self.assertEqual(step_toks, [], f"layout_short_division emitted row >= 3: {step_toks}")

    def test_layout_short_division_single_digit_divisor_tokens(self):
        """All divisor tokens in row 1 with negative cols are single glyphs (single digit divisor)."""
        rng = random.Random(7)
        _, tokens = layout_short_division(42, 7, rng)
        # col < -1 means divisor area; each such token should be a single digit
        divisor_toks = [t for t in tokens if t.row == 1 and t.col < -1]
        self.assertEqual(len(divisor_toks), 1, f"Expected 1 divisor token, got {len(divisor_toks)}")
        self.assertTrue(divisor_toks[0].glyph_key.isdigit())

    def test_layout_simple_division_uses_op_divide_token(self):
        """Exactly one token must have yolo_class_name='operator' and glyph_key='divide'."""
        rng = random.Random(42)
        _, tokens = layout_simple_division(18, 6, rng)
        op_divide = [t for t in tokens if t.yolo_class_name == "operator" and t.glyph_key == "divide"]
        self.assertEqual(len(op_divide), 1, f"Expected 1 op_divide token, got {len(op_divide)}")
        self.assertEqual(op_divide[0].flattened_label, "op_divide")

    def test_layout_simple_division_exact_result(self):
        """Result row (row 3) digits must equal the quotient."""
        rng = random.Random(42)
        a, b = 18, 6  # quotient = 3
        _, tokens = layout_simple_division(a, b, rng)
        result_toks = sorted([t for t in tokens if t.row == 3], key=lambda t: t.col)
        result_val = int("".join(t.glyph_key for t in result_toks))
        self.assertEqual(result_val, a // b, f"Result row has {result_val}, expected {a // b}")

    def test_layout_simple_division_no_carry_borrow_tokens(self):
        """Division simple must never emit carry or borrow tokens."""
        rng = random.Random(42)
        _, tokens = layout_simple_division(12, 4, rng)
        cb_toks = [t for t in tokens if t.yolo_class_name in ("digit_carry", "digit_borrow")]
        self.assertEqual(cb_toks, [], f"Found unexpected carry/borrow tokens: {cb_toks}")

    def test_borrow_1_token_at_borrower_columns(self):
        """layout_subtraction(100, 9) must emit a borrow_1 token at the units column (borrower)."""
        _, tokens = layout_subtraction(100, 9)
        borrow1_toks = [t for t in tokens if t.flattened_label == "borrow_1"]
        self.assertGreater(len(borrow1_toks), 0, "No borrow_1 tokens found")
        # All borrow_1 tokens must be at borrow_row=0
        for tok in borrow1_toks:
            self.assertEqual(tok.row, 0, f"borrow_1 token at wrong row: {tok.row}")
            self.assertEqual(tok.glyph_key, "1")
            self.assertEqual(tok.yolo_class_name, "digit_borrow")

    def test_borrow_no_tokens_when_no_borrow_needed(self):
        """layout_subtraction(5, 3) needs no borrowing — must emit zero borrow tokens."""
        _, tokens = layout_subtraction(5, 3)
        borrow_toks = [t for t in tokens if t.yolo_class_name == "digit_borrow"]
        self.assertEqual(borrow_toks, [], f"Unexpected borrow tokens for 5-3: {borrow_toks}")

    def test_scene_case_enum_has_7_members(self):
        """22 cases: 7 original + 6 B variants + 3 MC cases + 3 iter10 isolation cases + bare_digit_grid
        + 2 iter11 crowded double-digit cases (subtraction_crowded_borrow, addition_crowded_carry)."""
        self.assertEqual(len(list(SceneCase)), 22)
        # All original 7 cases must still be present
        original_7 = {
            "addition", "subtraction", "multiplication-simple", "multiplication-multi",
            "division-short", "division-long", "division-simple"
        }
        actual = {c.value for c in SceneCase}
        self.assertTrue(original_7.issubset(actual))

    def test_sample_layout_for_case_returns_correct_kind(self):
        """Each SceneCase must return the expected EquationKind."""
        rng = random.Random(42)
        expected_kinds = {
            SceneCase.addition: EquationKind.add,
            SceneCase.subtraction: EquationKind.subtract,
            SceneCase.multiplication_simple: EquationKind.multiply,
            SceneCase.multiplication_multi: EquationKind.multiply,
            SceneCase.division_short: EquationKind.divide,
            SceneCase.division_long: EquationKind.divide,
            SceneCase.division_simple: EquationKind.divide,
        }
        for case, expected in expected_kinds.items():
            kind, tokens = sample_layout_for_case(case, rng)
            self.assertEqual(kind, expected, f"{case}: got {kind}, expected {expected}")
            self.assertGreater(len(tokens), 0, f"{case}: returned empty token list")


from src.generation.layouts import layout_bounds, _sample_division_layout


class TestWorkstreamELayouts(unittest.TestCase):
    """Tests for Task 2: 4-digit operands, segment_kind, and canvas-fit guard."""

    def setUp(self) -> None:
        self.rng = random.Random(42)

    def test_4digit_layout_long_division_succeeds(self) -> None:
        rng = random.Random(0)
        kind, tokens = layout_long_division(1234, 12, rng)
        self.assertGreater(len(tokens), 0)
        self.assertEqual(kind, EquationKind.divide)

    def test_canvas_fit_guard_max_row_never_exceeds_15(self) -> None:
        rng = random.Random(7)
        for _ in range(1000):
            _, tokens = _sample_division_layout(rng)
            max_row = layout_bounds(tokens)[1]
            self.assertLessEqual(
                max_row, 15,
                f"layout_bounds max_row={max_row} exceeds canvas limit of 15",
            )

    def test_segment_kind_subtraction_bar_on_division_bars(self) -> None:
        rng = random.Random(0)
        _, tokens = layout_long_division(144, 12, rng)
        subtraction_bars = [t for t in tokens if t.segment_kind == "subtraction_bar"]
        self.assertGreater(len(subtraction_bars), 0, "no subtraction_bar tokens found")
        for tok in subtraction_bars:
            self.assertEqual(tok.flattened_label, "result_bar")
            self.assertEqual(tok.yolo_class_name, "result_bar")

    def test_segment_kind_none_on_main_digits(self) -> None:
        rng = random.Random(0)
        _, tokens = layout_long_division(144, 12, rng)
        main_tokens = [t for t in tokens if t.flattened_label.startswith("main_")]
        for tok in main_tokens:
            self.assertIsNone(tok.segment_kind, f"main token {tok} has unexpected segment_kind")

    def test_layout_multiplication_with_rng_produces_valid_token_count(self) -> None:
        # Smoke test: adding rng parameter must not break result correctness.
        rng = random.Random(99)
        _, tokens = layout_multiplication(12, 34, rng)
        result_row = max(t.row for t in tokens)
        result_toks = sorted(
            [t for t in tokens if t.row == result_row and t.yolo_class_name == "digit_main"],
            key=lambda t: t.col,
        )
        result_str = "".join(t.flattened_label.split("_")[1] for t in result_toks)
        self.assertEqual(result_str, str(12 * 34))


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# iter7 Workstream B + D tests
# ---------------------------------------------------------------------------

import random as _random

from src.generation.layouts import (
    SceneCase,
    EquationKind,
    layout_addition_no_bar,
    sample_layout_for_case,
)
from src.generation.completion_stages import valid_stages_for_case, CompletionStage


class TestIter7WorkstreamB(unittest.TestCase):

    def test_addition_op_right_operator_col(self) -> None:
        from src.generation.layouts import layout_addition
        rng = _random.Random(1)
        _, tokens = layout_addition(52, 7, rng, op_right=True)
        op_toks = [t for t in tokens if t.flattened_label == "op_plus"]
        self.assertEqual(len(op_toks), 1)
        op_col = op_toks[0].col
        # operator must be to the right of all digits in its row
        row_b = op_toks[0].row
        digit_cols_on_row_b = [t.col for t in tokens if t.row == row_b and t.yolo_class_name == "digit_main"]
        for dc in digit_cols_on_row_b:
            self.assertGreater(op_col, dc, f"op_col={op_col} not > digit col={dc}")

    def test_addition_no_bar_has_no_result_bar(self) -> None:
        rng = _random.Random(0)
        _, tokens = layout_addition_no_bar(52, 7, rng)
        bar_toks = [t for t in tokens if t.flattened_label == "result_bar"]
        self.assertEqual(len(bar_toks), 0)

    def test_new_cases_in_valid_stages_for_case(self) -> None:
        new_cases = [
            SceneCase.addition_op_right,
            SceneCase.addition_no_bar,
            SceneCase.subtraction_heavy_borrow,
            SceneCase.addition_dense_carries,
            SceneCase.addition_no_carries,
        ]
        valid_stage_values = {s.value for s in CompletionStage}
        for case in new_cases:
            stages = valid_stages_for_case(case)
            self.assertGreater(len(stages), 0, f"{case} has no valid stages")
            for s in stages:
                self.assertIn(s, valid_stage_values, f"{s} is not a valid CompletionStage for {case}")

    def test_sample_layout_for_case_exhaustive(self) -> None:
        """Every SceneCase must dispatch without raising AssertionError."""
        rng = _random.Random(42)
        for case in list(SceneCase):
            try:
                _kind, _tokens = sample_layout_for_case(case, rng)
            except AssertionError as exc:
                self.fail(f"sample_layout_for_case({case}) raised AssertionError: {exc}")



# ---------------------------------------------------------------------------
# Scene knob wiring tests (iter7)
# ---------------------------------------------------------------------------

import random as _scene_rng_mod
from src.core.run_config import SceneConfig
from src.generation.layouts import layout_addition, LayoutToken
from src.generation.synth_yolo import _apply_scene_knobs


def _addition_tokens() -> list:
    """Return a simple 3+4=7 token list with result bar and operator."""
    _, tokens = layout_addition(3, 4, _scene_rng_mod.Random(0))
    return list(tokens)


class TestSceneKnobsWired(unittest.TestCase):
    """Verify that scene knobs produce visible changes to the token list."""

    def test_wrong_result_prob_changes_result_digit(self) -> None:
        """With prob=1.0, a result-row main digit must be replaced."""
        tokens = _addition_tokens()
        # Identify original result row digits
        main_toks = [t for t in tokens if t.flattened_label.startswith("main_")]
        result_row = max(t.row for t in main_toks)
        original_result_glyphs = {t.glyph_key for t in tokens if t.row == result_row and t.flattened_label.startswith("main_")}

        cfg = SceneConfig(wrong_result_prob=1.0, missing_structural_prob=0.0, wrong_operator_prob=0.0)
        modified = _apply_scene_knobs(tokens, _scene_rng_mod.Random(99), cfg)

        new_result_glyphs = {t.glyph_key for t in modified if t.row == result_row and t.flattened_label.startswith("main_")}
        self.assertNotEqual(
            original_result_glyphs, new_result_glyphs,
            "wrong_result_prob=1.0 must change at least one result-row digit glyph",
        )

    def test_missing_structural_prob_drops_token(self) -> None:
        """With prob=1.0, either the result_bar or operator must be dropped."""
        tokens = _addition_tokens()
        original_count = len(tokens)
        cfg = SceneConfig(wrong_result_prob=0.0, missing_structural_prob=1.0, wrong_operator_prob=0.0)
        modified = _apply_scene_knobs(tokens, _scene_rng_mod.Random(42), cfg)
        self.assertLess(
            len(modified), original_count,
            "missing_structural_prob=1.0 must drop at least one structural token",
        )

    def test_missing_structural_drops_bar_or_operator(self) -> None:
        """The dropped token must be a result_bar or operator, not a digit."""
        tokens = _addition_tokens()
        cfg = SceneConfig(wrong_result_prob=0.0, missing_structural_prob=1.0, wrong_operator_prob=0.0)
        for seed in range(20):
            modified = _apply_scene_knobs(tokens, _scene_rng_mod.Random(seed), cfg)
            dropped = [t for t in tokens if t not in modified]
            for t in dropped:
                self.assertIn(
                    t.flattened_label if t.flattened_label == "result_bar" else t.yolo_class_name,
                    ("result_bar", "operator"),
                    f"Dropped token must be a structural token, got {t}",
                )

    def test_wrong_operator_prob_changes_operator(self) -> None:
        """With prob=1.0, the operator glyph_key must change."""
        tokens = _addition_tokens()
        original_ops = {t.glyph_key for t in tokens if t.yolo_class_name == "operator"}
        cfg = SceneConfig(wrong_result_prob=0.0, missing_structural_prob=0.0, wrong_operator_prob=1.0)
        modified = _apply_scene_knobs(tokens, _scene_rng_mod.Random(7), cfg)
        new_ops = {t.glyph_key for t in modified if t.yolo_class_name == "operator"}
        self.assertNotEqual(original_ops, new_ops, "wrong_operator_prob=1.0 must replace operator glyph")

    def test_force_carry_borrow_prob_removed(self) -> None:
        """SceneConfig must NOT have a force_carry_borrow_prob field."""
        cfg = SceneConfig()
        self.assertFalse(
            hasattr(cfg, "force_carry_borrow_prob"),
            "force_carry_borrow_prob was removed in iter7 — it was unwireable without semantic arithmetic",
        )

    def test_zero_probs_no_change(self) -> None:
        """With all probs=0, tokens must be unchanged."""
        tokens = _addition_tokens()
        cfg = SceneConfig(wrong_result_prob=0.0, missing_structural_prob=0.0, wrong_operator_prob=0.0)
        modified = _apply_scene_knobs(tokens, _scene_rng_mod.Random(0), cfg)
        self.assertEqual(len(tokens), len(modified))
        for orig, mod in zip(tokens, modified):
            self.assertEqual(orig.flattened_label, mod.flattened_label)


# ---------------------------------------------------------------------------
# A5: scene equation_type GT recomputed from FINAL (post-knob) tokens so a
# wrong-operator swap is reflected as-drawn (see correction-antipatterns A5).
# ---------------------------------------------------------------------------
from src.generation.layouts import (
    EquationKind as _EqKind,
    sample_layout_for_case as _sample_layout_for_case,
    SceneCase as _SceneCase,
    layout_multiplication as _layout_multiplication,
    layout_short_division as _layout_short_division,
)
from src.generation.synth_yolo import _eq_kind_from_final_tokens as _eq_kind_final

# Map each operator label to the kind it should yield once drawn.
_OP_LABEL_TO_KIND = {
    "op_plus": _EqKind.add,
    "op_minus": _EqKind.subtract,
    "op_times": _EqKind.multiply,
    "op_divide": _EqKind.divide,
}


def _primary_op_label(tokens: list) -> str:
    """Primary operator label in a token list, using layout precedence.

    Mirrors what a reader would call the equation: divide bracket / op_divide
    win, then op_times, then op_minus, then op_plus.
    """
    op_labels = {t.flattened_label for t in tokens if t.yolo_class_name == "operator"}
    if any(t.yolo_class_name == "divide_bracket" for t in tokens) or "op_divide" in op_labels:
        return "op_divide"
    for lbl in ("op_times", "op_minus", "op_plus"):
        if lbl in op_labels:
            return lbl
    return ""


class TestEqKindFromFinalTokensA5(unittest.TestCase):
    """A5: scene eq_type GT must follow the FINAL drawn operator after a swap."""

    def test_forced_swap_makes_eq_kind_match_drawn_operator(self) -> None:
        """wrong_operator_prob=1.0 on an addition layout flips the drawn
        operator; the recomputed scene kind must match that FINAL operator,
        not the original 'add' the layout sampled."""
        # Try several rng seeds so we cover every possible swap target
        # (op_plus -> minus / times / divide). For each, the recomputed kind
        # must equal the kind implied by the operator actually drawn.
        swap_kinds_seen = set()
        for seed in range(40):
            tokens = _addition_tokens()
            self.assertEqual(_primary_op_label(tokens), "op_plus")  # layout op
            cfg = SceneConfig(
                wrong_result_prob=0.0, missing_structural_prob=0.0, wrong_operator_prob=1.0
            )
            final = _apply_scene_knobs(tokens, _scene_rng_mod.Random(seed), cfg)
            drawn_label = _primary_op_label(final)
            self.assertNotEqual(drawn_label, "op_plus", "forced swap must change the operator")
            expected = _OP_LABEL_TO_KIND[drawn_label]
            derived = _eq_kind_final(tokens, final, _EqKind.add)
            self.assertEqual(
                derived, expected,
                f"seed={seed}: drawn op {drawn_label} should give {expected.value}, got {derived.value}",
            )
            swap_kinds_seen.add(derived.value)
        # The swap targets are uniform over the other 3 operators, so across 40
        # seeds we expect to have exercised more than one resulting kind.
        self.assertGreater(len(swap_kinds_seen), 1, f"only saw {swap_kinds_seen}")

    def test_no_swap_leaves_eq_kind_unchanged(self) -> None:
        """wrong_operator_prob=0.0 must leave the recomputed kind identical to
        the layout kind for every base SceneCase (the current-config no-op)."""
        no_swap_cfg = SceneConfig(
            wrong_result_prob=0.0, missing_structural_prob=0.0, wrong_operator_prob=0.0
        )
        for case in _SceneCase:
            for seed in range(25):
                rng = _scene_rng_mod.Random(seed * 31 + 1)
                layout_kind, tokens = _sample_layout_for_case(case, rng)
                if not tokens:
                    continue
                final = _apply_scene_knobs(list(tokens), rng, no_swap_cfg)
                derived = _eq_kind_final(list(tokens), final, layout_kind)
                self.assertEqual(
                    derived, layout_kind,
                    f"case={case.value} seed={seed}: no-swap must keep {layout_kind.value}, "
                    f"got {derived.value}",
                )

    def test_dropped_operator_keeps_layout_kind(self) -> None:
        """missing_structural dropping the op_times in a multiplication (leaving
        op_plus partial-product helpers) is a DROP, not a swap, so the kind
        must stay 'multiply' (matches pre-A5 behaviour, not 'add')."""
        # 12 x 34 forces partial-product op_plus helpers alongside the op_times.
        _, tokens = _layout_multiplication(12, 34, _scene_rng_mod.Random(0))
        op_labels = [t.flattened_label for t in tokens if t.yolo_class_name == "operator"]
        self.assertIn("op_times", op_labels)
        self.assertIn("op_plus", op_labels)  # PP helper present
        dropped = [t for t in tokens if t.flattened_label != "op_times"]
        derived = _eq_kind_final(list(tokens), dropped, _EqKind.multiply)
        self.assertEqual(derived, _EqKind.multiply)

    def test_ood_sentinel_never_rederived(self) -> None:
        """OOD scenes (fallback ood_unknown) must stay ood_unknown even when a
        divide_bracket token is present (standalone_bracket); a lone bracket is
        structure-only OOD, not a division equation."""
        _, tokens = _sample_layout_for_case(_SceneCase.standalone_bracket, _scene_rng_mod.Random(0))
        self.assertTrue(any(t.yolo_class_name == "divide_bracket" for t in tokens))
        derived = _eq_kind_final(list(tokens), list(tokens), _EqKind.ood_unknown)
        self.assertEqual(derived, _EqKind.ood_unknown)

    def test_division_bracket_swap_immune(self) -> None:
        """Division-by-bracket (short division) has no swappable op token for the
        bracket, so the kind stays 'divide' even under wrong_operator_prob=1.0."""
        layout_kind, tokens = _layout_short_division(56, 7, _scene_rng_mod.Random(0))
        self.assertEqual(layout_kind, _EqKind.divide)
        cfg = SceneConfig(
            wrong_result_prob=0.0, missing_structural_prob=0.0, wrong_operator_prob=1.0
        )
        final = _apply_scene_knobs(list(tokens), _scene_rng_mod.Random(3), cfg)
        derived = _eq_kind_final(list(tokens), final, layout_kind)
        self.assertEqual(derived, _EqKind.divide)
