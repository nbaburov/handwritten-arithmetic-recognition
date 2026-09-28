"""Tests for SceneCases added in W5 (subtraction_op_right, subtraction_no_bar,
multiplication_simple_op_right, multiplication_simple_no_bar) and coverage-axis cases
MC-3 (subtraction_heavy_borrow) and MC-4 (addition_dense_carries, addition_no_carries).
Also covers result_override parameter threading and the GAP-1 no-bar sampler fix.
"""
from __future__ import annotations

import random
import unittest

from src.generation.layouts import EquationKind, SceneCase, sample_layout_for_case


class TestSubtractionOpRight(unittest.TestCase):

    def test_returns_subtract_kind(self) -> None:
        rng = random.Random(0)
        kind, tokens = sample_layout_for_case(SceneCase.subtraction_op_right, rng)
        self.assertEqual(kind.value, "subtract")

    def test_operator_is_rightmost_on_bottom_row(self) -> None:
        for seed in range(5):
            rng = random.Random(seed)
            _, tokens = sample_layout_for_case(SceneCase.subtraction_op_right, rng)
            op_toks = [t for t in tokens if t.flattened_label == "op_minus"]
            self.assertEqual(len(op_toks), 1, "Must have exactly one op_minus")
            op_col = op_toks[0].col
            op_row = op_toks[0].row
            digit_cols = [
                t.col for t in tokens
                if t.row == op_row and t.yolo_class_name == "digit_main"
            ]
            if digit_cols:
                self.assertGreater(
                    op_col, max(digit_cols),
                    f"seed={seed}: op_col={op_col} must be right of max digit col={max(digit_cols)}"
                )

    def test_has_result_bar(self) -> None:
        rng = random.Random(1)
        _, tokens = sample_layout_for_case(SceneCase.subtraction_op_right, rng)
        bar_toks = [t for t in tokens if t.flattened_label == "result_bar"]
        self.assertGreater(len(bar_toks), 0, "subtraction_op_right must have a result_bar")

    def test_produces_valid_token_list(self) -> None:
        rng = random.Random(42)
        _, tokens = sample_layout_for_case(SceneCase.subtraction_op_right, rng)
        self.assertGreater(len(tokens), 0)
        for t in tokens:
            self.assertIsInstance(t.row, int)
            self.assertIsInstance(t.col, int)


class TestSubtractionNoBar(unittest.TestCase):

    def test_returns_subtract_kind(self) -> None:
        rng = random.Random(0)
        kind, tokens = sample_layout_for_case(SceneCase.subtraction_no_bar, rng)
        self.assertEqual(kind.value, "subtract")

    def test_no_result_bar_token(self) -> None:
        for seed in range(5):
            rng = random.Random(seed)
            _, tokens = sample_layout_for_case(SceneCase.subtraction_no_bar, rng)
            bar_toks = [t for t in tokens if t.flattened_label == "result_bar"]
            self.assertEqual(len(bar_toks), 0, f"seed={seed}: subtraction_no_bar must have no result_bar")

    def test_has_op_minus(self) -> None:
        rng = random.Random(2)
        _, tokens = sample_layout_for_case(SceneCase.subtraction_no_bar, rng)
        op_toks = [t for t in tokens if t.flattened_label == "op_minus"]
        self.assertEqual(len(op_toks), 1)

    def test_has_operand_tokens(self) -> None:
        rng = random.Random(3)
        _, tokens = sample_layout_for_case(SceneCase.subtraction_no_bar, rng)
        main_toks = [t for t in tokens if t.yolo_class_name == "digit_main"]
        self.assertGreater(len(main_toks), 0)


class TestMultiplicationSimpleOpRight(unittest.TestCase):

    def test_returns_multiply_kind(self) -> None:
        rng = random.Random(0)
        kind, tokens = sample_layout_for_case(SceneCase.multiplication_simple_op_right, rng)
        self.assertEqual(kind.value, "multiply")

    def test_operator_is_rightmost_on_multiplier_row(self) -> None:
        for seed in range(5):
            rng = random.Random(seed)
            _, tokens = sample_layout_for_case(SceneCase.multiplication_simple_op_right, rng)
            op_toks = [t for t in tokens if t.flattened_label == "op_times"]
            self.assertEqual(len(op_toks), 1, f"seed={seed}: must have exactly one op_times")
            op_col = op_toks[0].col
            op_row = op_toks[0].row
            digit_cols = [
                t.col for t in tokens
                if t.row == op_row and t.yolo_class_name == "digit_main"
            ]
            if digit_cols:
                self.assertGreater(
                    op_col, max(digit_cols),
                    f"seed={seed}: op_col={op_col} must be right of max digit col={max(digit_cols)}"
                )

    def test_has_result_bar(self) -> None:
        rng = random.Random(1)
        _, tokens = sample_layout_for_case(SceneCase.multiplication_simple_op_right, rng)
        bar_toks = [t for t in tokens if t.flattened_label == "result_bar"]
        self.assertGreater(len(bar_toks), 0)


class TestMultiplicationSimpleNoBar(unittest.TestCase):

    def test_returns_multiply_kind(self) -> None:
        rng = random.Random(0)
        kind, tokens = sample_layout_for_case(SceneCase.multiplication_simple_no_bar, rng)
        self.assertEqual(kind.value, "multiply")

    def test_no_result_bar_token(self) -> None:
        for seed in range(5):
            rng = random.Random(seed)
            _, tokens = sample_layout_for_case(SceneCase.multiplication_simple_no_bar, rng)
            bar_toks = [t for t in tokens if t.flattened_label == "result_bar"]
            self.assertEqual(len(bar_toks), 0, f"seed={seed}: no_bar must have no result_bar")

    def test_has_op_times(self) -> None:
        rng = random.Random(2)
        _, tokens = sample_layout_for_case(SceneCase.multiplication_simple_no_bar, rng)
        op_toks = [t for t in tokens if t.flattened_label == "op_times"]
        self.assertEqual(len(op_toks), 1)


class TestResultOverride(unittest.TestCase):

    def test_result_override_changes_result_digits(self) -> None:
        """Wrong-answer render must differ from correct-answer render."""
        rng_correct = random.Random(7)
        rng_wrong = random.Random(7)
        kind1, correct_tokens = sample_layout_for_case(SceneCase.addition, rng_correct)
        # Get correct result digits
        all_rows = sorted({t.row for t in correct_tokens if t.yolo_class_name == "digit_main"})
        result_row = all_rows[-1] if all_rows else 0
        correct_result = [t.flattened_label for t in correct_tokens if t.row == result_row]
        # Build wrong override: +1 mod 10 on each digit value
        override = [(int(lbl[-1]) + 1) % 10 for lbl in correct_result]
        kind2, wrong_tokens = sample_layout_for_case(SceneCase.addition, rng_wrong, result_override=override)
        wrong_result = [t.flattened_label for t in wrong_tokens if t.row == result_row]
        self.assertNotEqual(correct_result, wrong_result, "result_override must change result digits")

    def test_result_override_none_is_identity(self) -> None:
        """result_override=None must produce identical output to no argument."""
        rng1 = random.Random(5)
        rng2 = random.Random(5)
        _, tokens1 = sample_layout_for_case(SceneCase.subtraction, rng1, result_override=None)
        _, tokens2 = sample_layout_for_case(SceneCase.subtraction, rng2)
        self.assertEqual(
            [(t.flattened_label, t.row, t.col) for t in tokens1],
            [(t.flattened_label, t.row, t.col) for t in tokens2],
        )

    def test_result_override_does_not_change_non_result_rows(self) -> None:
        """result_override must only modify the result row, not operand rows."""
        rng_correct = random.Random(9)
        rng_wrong = random.Random(9)
        _, correct_tokens = sample_layout_for_case(SceneCase.addition, rng_correct)
        _, wrong_tokens = sample_layout_for_case(SceneCase.addition, rng_wrong, result_override=[9, 9, 9])
        # Operand rows should be unchanged
        all_rows = sorted({t.row for t in correct_tokens if t.yolo_class_name == "digit_main"})
        if len(all_rows) >= 2:
            operand_row = all_rows[0]
            correct_op = [t.flattened_label for t in correct_tokens if t.row == operand_row]
            wrong_op = [t.flattened_label for t in wrong_tokens if t.row == operand_row]
            self.assertEqual(correct_op, wrong_op, "Operand row must not change with result_override")


class TestNewSceneCasesInCompletionStages(unittest.TestCase):

    def test_new_cases_have_valid_stages(self) -> None:
        from src.generation.completion_stages import valid_stages_for_case, CompletionStage
        new_cases = [
            SceneCase.subtraction_op_right,
            SceneCase.subtraction_no_bar,
            SceneCase.multiplication_simple_op_right,
            SceneCase.multiplication_simple_no_bar,
        ]
        valid_stage_values = {s.value for s in CompletionStage}
        for case in new_cases:
            stages = valid_stages_for_case(case)
            self.assertGreater(len(stages), 0, f"{case.value} has no valid stages")
            for s in stages:
                self.assertIn(s, valid_stage_values, f"Stage '{s}' not valid for {case.value}")

    def test_no_bar_cases_include_partial_stages(self) -> None:
        """no_bar cases should allow partial stages (done_40, done_20) for examples renderer."""
        from src.generation.completion_stages import valid_stages_for_case
        for case in [SceneCase.subtraction_no_bar, SceneCase.multiplication_simple_no_bar]:
            stages = valid_stages_for_case(case)
            self.assertIn("done_40", stages, f"{case.value} must support done_40")

    def test_op_right_cases_include_full_stage(self) -> None:
        from src.generation.completion_stages import valid_stages_for_case
        for case in [SceneCase.subtraction_op_right, SceneCase.multiplication_simple_op_right]:
            stages = valid_stages_for_case(case)
            self.assertIn("full", stages, f"{case.value} must support full")


class TestMC3MC4SceneCases(unittest.TestCase):
    """Tests for MC-3 (subtraction_heavy_borrow) and MC-4 (addition_dense_carries,
    addition_no_carries) SceneCases added in the coverage-axis change set."""

    def _rng(self, seed: int = 42) -> random.Random:
        return random.Random(seed)

    # --- MC-3: subtraction_heavy_borrow ---

    def test_heavy_borrow_returns_subtract_kind(self) -> None:
        kind, tokens = sample_layout_for_case(SceneCase.subtraction_heavy_borrow, self._rng())
        self.assertEqual(kind, EquationKind.subtract)
        self.assertGreater(len(tokens), 0)

    def test_heavy_borrow_has_borrow_tokens(self) -> None:
        """At least 2 borrow tokens must appear in every heavy-borrow scene."""
        for seed in range(10):
            _, tokens = sample_layout_for_case(SceneCase.subtraction_heavy_borrow, self._rng(seed))
            borrow_count = sum(1 for t in tokens if t.yolo_class_name == "digit_borrow")
            self.assertGreaterEqual(
                borrow_count, 2,
                f"seed={seed}: expected ≥2 borrows, got {borrow_count}"
            )

    def test_heavy_borrow_operands_are_3_digits(self) -> None:
        """Both operands must be 3-digit (col indices 0-2 present on rows 1 and 2)."""
        _, tokens = sample_layout_for_case(SceneCase.subtraction_heavy_borrow, self._rng())
        row1_cols = {t.col for t in tokens if t.row == 1 and t.yolo_class_name == "digit_main"}
        row2_cols = {t.col for t in tokens if t.row == 2 and t.yolo_class_name == "digit_main"}
        self.assertEqual(len(row1_cols), 3, f"Top operand must have 3 digits, got cols={sorted(row1_cols)}")
        self.assertEqual(len(row2_cols), 3, f"Bottom operand must have 3 digits, got cols={sorted(row2_cols)}")

    def test_heavy_borrow_valid_stages(self) -> None:
        from src.generation.completion_stages import valid_stages_for_case, CompletionStage
        stages = valid_stages_for_case(SceneCase.subtraction_heavy_borrow)
        valid_stage_values = {s.value for s in CompletionStage}
        self.assertIn("full", stages)
        self.assertIn("done_60", stages)
        for s in stages:
            self.assertIn(s, valid_stage_values)

    def test_heavy_borrow_completion_stage_sampling(self) -> None:
        from src.generation.completion_stages import sample_completion_stage_for_case, CompletionStage
        rng = self._rng(7)
        _, tokens = sample_layout_for_case(SceneCase.subtraction_heavy_borrow, rng)
        stage, staged_tokens = sample_completion_stage_for_case(
            SceneCase.subtraction_heavy_borrow, tokens, rng
        )
        self.assertIsInstance(stage, CompletionStage)
        self.assertGreater(len(staged_tokens), 0)

    # --- MC-4a: addition_dense_carries ---

    def test_dense_carries_returns_add_kind(self) -> None:
        kind, tokens = sample_layout_for_case(SceneCase.addition_dense_carries, self._rng())
        self.assertEqual(kind, EquationKind.add)
        self.assertGreater(len(tokens), 0)

    def test_dense_carries_has_carry_tokens(self) -> None:
        """At least 2 carry tokens must appear in dense-carries scenes."""
        for seed in range(10):
            _, tokens = sample_layout_for_case(SceneCase.addition_dense_carries, self._rng(seed))
            carry_count = sum(1 for t in tokens if t.yolo_class_name == "digit_carry")
            self.assertGreaterEqual(
                carry_count, 2,
                f"seed={seed}: expected ≥2 carries, got {carry_count}"
            )

    def test_dense_carries_valid_stages(self) -> None:
        from src.generation.completion_stages import valid_stages_for_case
        stages = valid_stages_for_case(SceneCase.addition_dense_carries)
        self.assertIn("full", stages)
        self.assertIn("done_80", stages)

    # --- MC-4b: addition_no_carries ---

    def test_no_carries_returns_add_kind(self) -> None:
        kind, tokens = sample_layout_for_case(SceneCase.addition_no_carries, self._rng())
        self.assertEqual(kind, EquationKind.add)
        self.assertGreater(len(tokens), 0)

    def test_no_carries_has_zero_carry_tokens(self) -> None:
        """No carry tokens should appear in any no-carries scene."""
        for seed in range(10):
            _, tokens = sample_layout_for_case(SceneCase.addition_no_carries, self._rng(seed))
            carry_count = sum(1 for t in tokens if t.yolo_class_name == "digit_carry")
            self.assertEqual(
                carry_count, 0,
                f"seed={seed}: expected 0 carries, got {carry_count}"
            )

    def test_no_carries_valid_stages(self) -> None:
        from src.generation.completion_stages import valid_stages_for_case
        stages = valid_stages_for_case(SceneCase.addition_no_carries)
        self.assertIn("full", stages)
        self.assertIn("done_20", stages)

    def test_no_carries_completion_sampling_never_produces_carries(self) -> None:
        from src.generation.completion_stages import sample_completion_stage_for_case
        rng = self._rng(99)
        for seed in range(5):
            rng2 = random.Random(seed)
            _, tokens = sample_layout_for_case(SceneCase.addition_no_carries, rng2)
            _, staged = sample_completion_stage_for_case(SceneCase.addition_no_carries, tokens, rng2)
            carry_count = sum(1 for t in staged if t.yolo_class_name == "digit_carry")
            self.assertEqual(carry_count, 0, f"seed={seed}: staged tokens must not contain carries")

    # --- GAP-1: no-bar sampler produces partial stages ---

    def test_addition_no_bar_partial_stages_applied(self) -> None:
        """After GAP-1 fix: done_60/done_40/done_20 buckets must produce fewer tokens than full."""
        from src.generation.completion_stages import sample_completion_stage_for_case, CompletionStage
        rng_layout = random.Random(11)
        _, full_tokens = sample_layout_for_case(SceneCase.addition_no_bar, rng_layout)
        full_count = len(full_tokens)
        # Force done_40: weights = {done_40: 1.0}
        _, done40_tokens = sample_completion_stage_for_case(
            SceneCase.addition_no_bar, list(full_tokens), random.Random(0),
            weights={"done_40": 1.0}
        )
        self.assertLess(
            len(done40_tokens), full_count,
            f"done_40 should remove result row; got {len(done40_tokens)} vs full {full_count}"
        )
        # Force done_20
        _, done20_tokens = sample_completion_stage_for_case(
            SceneCase.addition_no_bar, list(full_tokens), random.Random(0),
            weights={"done_20": 1.0}
        )
        self.assertLessEqual(len(done20_tokens), len(done40_tokens),
            "done_20 should be no larger than done_40")


class TestBareDigits(unittest.TestCase):
    """Tests for SceneCase.bare_digits (iter10 W10-R-A)."""

    def _rng(self, seed: int = 0) -> random.Random:
        return random.Random(seed)

    def test_returns_ood_unknown_kind(self) -> None:
        rng = self._rng(0)
        kind, tokens = sample_layout_for_case(SceneCase.bare_digits, rng)
        self.assertEqual(kind, EquationKind.ood_unknown)

    def test_produces_only_digit_main_tokens(self) -> None:
        for seed in range(10):
            rng = self._rng(seed)
            _, tokens = sample_layout_for_case(SceneCase.bare_digits, rng)
            for t in tokens:
                self.assertEqual(
                    t.yolo_class_name, "digit_main",
                    f"seed={seed}: expected digit_main, got {t.yolo_class_name}",
                )
                self.assertNotIn(t.yolo_class_name, ("operator", "result_bar", "divide_bracket"))

    def test_digit_count_in_range_1_to_3(self) -> None:
        counts = set()
        for seed in range(20):
            rng = self._rng(seed)
            _, tokens = sample_layout_for_case(SceneCase.bare_digits, rng)
            n = len(tokens)
            self.assertGreaterEqual(n, 1, f"seed={seed}: expected >= 1 digit")
            self.assertLessEqual(n, 3, f"seed={seed}: expected <= 3 digits")
            counts.add(n)
        # Across 20 seeds, all three counts should appear (probabilistic but reliable)
        self.assertGreater(len(counts), 1, "Expected variance in digit count across seeds")

    def test_tokens_on_single_row(self) -> None:
        for seed in range(10):
            rng = self._rng(seed)
            _, tokens = sample_layout_for_case(SceneCase.bare_digits, rng)
            rows = {t.row for t in tokens}
            self.assertEqual(rows, {0}, f"seed={seed}: all tokens must be in row 0, got rows={rows}")

    def test_cols_are_consecutive(self) -> None:
        for seed in range(10):
            rng = self._rng(seed)
            _, tokens = sample_layout_for_case(SceneCase.bare_digits, rng)
            cols = sorted(t.col for t in tokens)
            expected = list(range(len(cols)))
            self.assertEqual(cols, expected, f"seed={seed}: cols must be consecutive 0..n-1")

    def test_valid_stages_returns_full_only(self) -> None:
        from src.generation.completion_stages import valid_stages_for_case
        stages = valid_stages_for_case(SceneCase.bare_digits)
        self.assertEqual(stages, ["full"])

    def test_completion_stage_sampling_returns_full(self) -> None:
        from src.generation.completion_stages import sample_completion_stage_for_case, CompletionStage
        rng = self._rng(0)
        _, tokens = sample_layout_for_case(SceneCase.bare_digits, rng)
        stage, staged_tokens = sample_completion_stage_for_case(SceneCase.bare_digits, tokens, rng)
        self.assertEqual(stage, CompletionStage.full)
        self.assertEqual(len(staged_tokens), len(tokens))

    def test_flattened_labels_are_valid_ontology_values(self) -> None:
        valid_labels = {f"main_{d}" for d in range(10)}
        for seed in range(10):
            rng = self._rng(seed)
            _, tokens = sample_layout_for_case(SceneCase.bare_digits, rng)
            for t in tokens:
                self.assertIn(
                    t.flattened_label, valid_labels,
                    f"seed={seed}: flattened_label '{t.flattened_label}' not in main_0..main_9",
                )

    def test_produces_non_empty_token_list(self) -> None:
        rng = self._rng(7)
        _, tokens = sample_layout_for_case(SceneCase.bare_digits, rng)
        self.assertGreater(len(tokens), 0)


class TestStandaloneBar(unittest.TestCase):
    """Tests for SceneCase.standalone_bar (iter10 W10-R-A)."""

    def _rng(self, seed: int = 0) -> random.Random:
        return random.Random(seed)

    def test_returns_ood_unknown_kind(self) -> None:
        rng = self._rng(0)
        kind, tokens = sample_layout_for_case(SceneCase.standalone_bar, rng)
        self.assertEqual(kind, EquationKind.ood_unknown)

    def test_produces_exactly_one_token(self) -> None:
        for seed in range(10):
            rng = self._rng(seed)
            _, tokens = sample_layout_for_case(SceneCase.standalone_bar, rng)
            self.assertEqual(len(tokens), 1, f"seed={seed}: standalone_bar must produce exactly 1 token")

    def test_token_is_result_bar(self) -> None:
        rng = self._rng(0)
        _, tokens = sample_layout_for_case(SceneCase.standalone_bar, rng)
        tok = tokens[0]
        self.assertEqual(tok.flattened_label, "result_bar")
        self.assertEqual(tok.yolo_class_name, "result_bar")

    def test_valid_stages_returns_full_only(self) -> None:
        from src.generation.completion_stages import valid_stages_for_case
        stages = valid_stages_for_case(SceneCase.standalone_bar)
        self.assertEqual(stages, ["full"])

    def test_completion_stage_returns_full(self) -> None:
        from src.generation.completion_stages import sample_completion_stage_for_case, CompletionStage
        rng = self._rng(0)
        _, tokens = sample_layout_for_case(SceneCase.standalone_bar, rng)
        stage, staged_tokens = sample_completion_stage_for_case(SceneCase.standalone_bar, tokens, rng)
        self.assertEqual(stage, CompletionStage.full)
        self.assertEqual(len(staged_tokens), 1)

    def test_no_digit_tokens(self) -> None:
        rng = self._rng(1)
        _, tokens = sample_layout_for_case(SceneCase.standalone_bar, rng)
        digit_toks = [t for t in tokens if t.yolo_class_name == "digit_main"]
        self.assertEqual(len(digit_toks), 0, "standalone_bar must have no digit_main tokens")


class TestStandaloneBracket(unittest.TestCase):
    """Tests for SceneCase.standalone_bracket (iter10 W10-R-A)."""

    def _rng(self, seed: int = 0) -> random.Random:
        return random.Random(seed)

    def test_returns_ood_unknown_kind(self) -> None:
        rng = self._rng(0)
        kind, tokens = sample_layout_for_case(SceneCase.standalone_bracket, rng)
        self.assertEqual(kind, EquationKind.ood_unknown)

    def test_produces_exactly_one_token(self) -> None:
        for seed in range(10):
            rng = self._rng(seed)
            _, tokens = sample_layout_for_case(SceneCase.standalone_bracket, rng)
            self.assertEqual(len(tokens), 1, f"seed={seed}: standalone_bracket must produce exactly 1 token")

    def test_token_is_div_bracket(self) -> None:
        rng = self._rng(0)
        _, tokens = sample_layout_for_case(SceneCase.standalone_bracket, rng)
        tok = tokens[0]
        self.assertEqual(tok.flattened_label, "div_bracket")
        self.assertEqual(tok.yolo_class_name, "divide_bracket")

    def test_valid_stages_returns_full_only(self) -> None:
        from src.generation.completion_stages import valid_stages_for_case
        stages = valid_stages_for_case(SceneCase.standalone_bracket)
        self.assertEqual(stages, ["full"])

    def test_completion_stage_returns_full(self) -> None:
        from src.generation.completion_stages import sample_completion_stage_for_case, CompletionStage
        rng = self._rng(0)
        _, tokens = sample_layout_for_case(SceneCase.standalone_bracket, rng)
        stage, staged_tokens = sample_completion_stage_for_case(SceneCase.standalone_bracket, tokens, rng)
        self.assertEqual(stage, CompletionStage.full)
        self.assertEqual(len(staged_tokens), 1)

    def test_no_digit_tokens(self) -> None:
        rng = self._rng(2)
        _, tokens = sample_layout_for_case(SceneCase.standalone_bracket, rng)
        digit_toks = [t for t in tokens if t.yolo_class_name == "digit_main"]
        self.assertEqual(len(digit_toks), 0, "standalone_bracket must have no digit_main tokens")


class TestBareDigitGrid(unittest.TestCase):
    """Tests for SceneCase.bare_digit_grid: a multi-row, multi-column block of bare main digits."""

    def _rng(self, seed: int = 0) -> random.Random:
        return random.Random(seed)

    def test_returns_ood_unknown_kind(self) -> None:
        rng = self._rng(0)
        kind, tokens = sample_layout_for_case(SceneCase.bare_digit_grid, rng)
        self.assertEqual(kind, EquationKind.ood_unknown)

    def test_produces_only_digit_main_tokens(self) -> None:
        for seed in range(10):
            rng = self._rng(seed)
            _, tokens = sample_layout_for_case(SceneCase.bare_digit_grid, rng)
            for t in tokens:
                self.assertEqual(
                    t.yolo_class_name, "digit_main",
                    f"seed={seed}: expected digit_main, got {t.yolo_class_name}",
                )

    def test_no_operator_bar_or_bracket(self) -> None:
        forbidden = {"operator", "result_bar", "divide_bracket"}
        for seed in range(10):
            rng = self._rng(seed)
            _, tokens = sample_layout_for_case(SceneCase.bare_digit_grid, rng)
            classes = {t.yolo_class_name for t in tokens}
            self.assertTrue(
                classes.isdisjoint(forbidden),
                f"seed={seed}: grid must contain no operator/bar/bracket, got {classes}",
            )

    def test_at_least_four_tokens(self) -> None:
        for seed in range(10):
            rng = self._rng(seed)
            _, tokens = sample_layout_for_case(SceneCase.bare_digit_grid, rng)
            self.assertGreaterEqual(
                len(tokens), 4,
                f"seed={seed}: a 2x2..3x3 grid must yield >= 4 tokens, got {len(tokens)}",
            )

    def test_spans_multiple_rows_and_cols(self) -> None:
        for seed in range(10):
            rng = self._rng(seed)
            _, tokens = sample_layout_for_case(SceneCase.bare_digit_grid, rng)
            rows = {t.row for t in tokens}
            cols = {t.col for t in tokens}
            self.assertGreaterEqual(
                len(rows), 2, f"seed={seed}: expected >= 2 distinct row indices, got {sorted(rows)}",
            )
            self.assertGreaterEqual(
                len(cols), 2, f"seed={seed}: expected >= 2 distinct col indices, got {sorted(cols)}",
            )

    def test_full_grid_invariants_in_one_scene(self) -> None:
        """Single-scene assertion covering all task requirements together."""
        rng = self._rng(3)
        kind, tokens = sample_layout_for_case(SceneCase.bare_digit_grid, rng)
        self.assertEqual(kind, EquationKind.ood_unknown)
        self.assertGreaterEqual(len(tokens), 4)
        self.assertGreaterEqual(len({t.row for t in tokens}), 2)
        self.assertGreaterEqual(len({t.col for t in tokens}), 2)
        for t in tokens:
            self.assertTrue(
                t.flattened_label.startswith("main_"),
                f"flattened_label '{t.flattened_label}' is not a main_ digit",
            )
            self.assertEqual(t.yolo_class_name, "digit_main")

    def test_flattened_labels_are_valid_ontology_values(self) -> None:
        valid_labels = {f"main_{d}" for d in range(10)}
        for seed in range(10):
            rng = self._rng(seed)
            _, tokens = sample_layout_for_case(SceneCase.bare_digit_grid, rng)
            for t in tokens:
                self.assertIn(
                    t.flattened_label, valid_labels,
                    f"seed={seed}: flattened_label '{t.flattened_label}' not in main_0..main_9",
                )

    def test_dims_in_range_2_to_3(self) -> None:
        for seed in range(20):
            rng = self._rng(seed)
            _, tokens = sample_layout_for_case(SceneCase.bare_digit_grid, rng)
            n_rows = len({t.row for t in tokens})
            n_cols = len({t.col for t in tokens})
            self.assertIn(n_rows, (2, 3), f"seed={seed}: n_rows={n_rows} out of {{2,3}}")
            self.assertIn(n_cols, (2, 3), f"seed={seed}: n_cols={n_cols} out of {{2,3}}")
            self.assertEqual(
                len(tokens), n_rows * n_cols,
                f"seed={seed}: token count {len(tokens)} != n_rows*n_cols {n_rows * n_cols}",
            )

    def test_valid_stages_returns_full_only(self) -> None:
        from src.generation.completion_stages import valid_stages_for_case
        stages = valid_stages_for_case(SceneCase.bare_digit_grid)
        self.assertEqual(stages, ["full"])

    def test_completion_stage_sampling_returns_full(self) -> None:
        from src.generation.completion_stages import sample_completion_stage_for_case, CompletionStage
        rng = self._rng(0)
        _, tokens = sample_layout_for_case(SceneCase.bare_digit_grid, rng)
        stage, staged_tokens = sample_completion_stage_for_case(SceneCase.bare_digit_grid, tokens, rng)
        self.assertEqual(stage, CompletionStage.full)
        self.assertEqual(len(staged_tokens), len(tokens))


class TestRegressionExistingCases(unittest.TestCase):
    """Guard: all 16 pre-existing SceneCases still dispatch correctly after iter10 additions."""

    _PRE_EXISTING_CASES = [
        SceneCase.addition,
        SceneCase.subtraction,
        SceneCase.multiplication_simple,
        SceneCase.multiplication_multi,
        SceneCase.division_short,
        SceneCase.division_long,
        SceneCase.division_simple,
        SceneCase.addition_op_right,
        SceneCase.addition_no_bar,
        SceneCase.subtraction_op_right,
        SceneCase.subtraction_no_bar,
        SceneCase.multiplication_simple_op_right,
        SceneCase.multiplication_simple_no_bar,
        SceneCase.subtraction_heavy_borrow,
        SceneCase.addition_dense_carries,
        SceneCase.addition_no_carries,
    ]

    def test_existing_scenecases_still_dispatch_correctly(self) -> None:
        for case in self._PRE_EXISTING_CASES:
            rng = random.Random(42)
            kind, tokens = sample_layout_for_case(case, rng)
            self.assertIsNotNone(kind, f"{case}: kind must not be None")
            self.assertGreater(len(tokens), 0, f"{case}: must produce non-empty token list")
            for t in tokens:
                self.assertIsInstance(t.row, int, f"{case}: token.row must be int")
                self.assertIsInstance(t.col, int, f"{case}: token.col must be int")


if __name__ == "__main__":
    unittest.main()
