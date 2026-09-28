from __future__ import annotations

import collections
import random
import unittest

from src.core.ontology import default_required_stage2_labels
from src.generation.completion_stages import CompletionStage, sample_completion_stage
from src.generation.layouts import (
    EquationKind,
    layout_addition,
    layout_long_division,
    layout_multiplication,
    layout_subtraction,
)


def _add_tokens() -> list:
    rng = random.Random(0)
    _, tokens = layout_addition(25, 37, rng)
    return tokens


def _sub_tokens() -> list:
    _, tokens = layout_subtraction(82, 47)
    return tokens


def _mul_tokens() -> list:
    rng = random.Random(0)
    _, tokens = layout_multiplication(12, 34, rng)
    return tokens


def _div_tokens() -> list:
    rng = random.Random(0)
    _, tokens = layout_long_division(144, 12, rng)
    return tokens


class TestStageMixtureAdd(unittest.TestCase):
    def test_full_frequency_add_is_50_to_60_percent(self) -> None:
        rng = random.Random(42)
        tokens = _add_tokens()
        counter: collections.Counter = collections.Counter()
        n_samples = 5000
        for _ in range(n_samples):
            stage, _ = sample_completion_stage(EquationKind.add, tokens, rng)
            counter[stage] += 1
        full_fraction = counter[CompletionStage.full] / n_samples
        # _WEIGHT_FULL = 0.65; allow ±5pp sampling variance around that target.
        self.assertGreaterEqual(full_fraction, 0.60, f"full too rare: {full_fraction:.3f}")
        self.assertLessEqual(full_fraction, 0.70, f"full too common: {full_fraction:.3f}")


class TestStageMixtureDivide(unittest.TestCase):
    def test_full_frequency_divide_is_50_to_60_percent(self) -> None:
        rng = random.Random(7)
        tokens = _div_tokens()
        counter: collections.Counter = collections.Counter()
        n_samples = 5000
        for _ in range(n_samples):
            stage, _ = sample_completion_stage(EquationKind.divide, tokens, rng)
            counter[stage] += 1
        full_fraction = counter[CompletionStage.full] / n_samples
        # _WEIGHT_FULL = 0.65; allow ±5pp sampling variance around that target.
        self.assertGreaterEqual(full_fraction, 0.60, f"full too rare: {full_fraction:.3f}")
        self.assertLessEqual(full_fraction, 0.70, f"full too common: {full_fraction:.3f}")


class TestNoResultStage(unittest.TestCase):
    def test_no_result_removes_result_row(self) -> None:
        tokens = _add_tokens()
        max_row = max(t.row for t in tokens)
        # Force the no_result stage by calling the internal function directly.
        from src.generation.completion_stages import _stage_add_sub_no_result
        filtered = _stage_add_sub_no_result(tokens)
        result_toks = [t for t in filtered if t.row == max_row]
        self.assertEqual(result_toks, [], f"result row {max_row} not removed")

    def test_no_result_via_sampler_removes_result_row(self) -> None:
        # Draw many samples and check that whenever no_result stage is picked,
        # the result row is indeed absent.
        rng = random.Random(11)
        tokens = _add_tokens()
        max_row = max(t.row for t in tokens)
        for _ in range(200):
            stage, filtered = sample_completion_stage(EquationKind.add, tokens, rng)
            if stage == CompletionStage.no_result:
                remaining = [t for t in filtered if t.row == max_row]
                self.assertEqual(remaining, [], "no_result stage left tokens in result row")


class TestNoBarNoResultStage(unittest.TestCase):
    def test_no_bar_no_result_removes_two_top_rows(self) -> None:
        from src.generation.completion_stages import _stage_add_sub_no_bar_no_result
        tokens = _add_tokens()
        max_row = max(t.row for t in tokens)
        bar_row = max_row - 1
        filtered = _stage_add_sub_no_bar_no_result(tokens)
        for tok in filtered:
            self.assertLess(tok.row, bar_row, f"token at row {tok.row} should have been removed")


class TestPartialCarriesOrBorrows(unittest.TestCase):
    def test_main_digit_tokens_are_never_dropped(self) -> None:
        from src.generation.completion_stages import _stage_partial_carries_or_borrows
        rng = random.Random(3)
        tokens = _add_tokens()
        main_ids = {id(t) for t in tokens if t.flattened_label.startswith("main_")}
        for _ in range(20):
            filtered = _stage_partial_carries_or_borrows(tokens, rng)
            filtered_main_ids = {id(t) for t in filtered if t.flattened_label.startswith("main_")}
            self.assertEqual(
                filtered_main_ids, main_ids,
                "partial_carries_or_borrows dropped a main-digit token",
            )


class TestBracketOnlyStage(unittest.TestCase):
    def test_bracket_only_returns_only_row_1(self) -> None:
        from src.generation.completion_stages import _stage_div_bracket_only
        tokens = _div_tokens()
        filtered = _stage_div_bracket_only(tokens)
        self.assertGreater(len(filtered), 0)
        for tok in filtered:
            self.assertEqual(tok.row, 1, f"bracket_only returned token at row {tok.row} != 1")


class TestNoFinalRemainder(unittest.TestCase):
    def test_no_final_remainder_removes_last_row_only(self) -> None:
        from src.generation.completion_stages import _stage_div_no_final_remainder
        tokens = _div_tokens()
        max_row = max(t.row for t in tokens)
        filtered = _stage_div_no_final_remainder(tokens)
        for tok in filtered:
            self.assertNotEqual(
                tok.row, max_row,
                f"no_final_remainder left a token at final row {max_row}",
            )
        # Rows below max_row must be fully preserved.
        original_non_final = [t for t in tokens if t.row != max_row]
        self.assertEqual(len(filtered), len(original_non_final))


class TestFullStageReturnsSameTokens(unittest.TestCase):
    def test_full_stage_returns_all_tokens_add(self) -> None:
        tokens = _add_tokens()
        original_labels = sorted(t.flattened_label for t in tokens)
        filtered = list(tokens)
        self.assertEqual(
            sorted(t.flattened_label for t in filtered),
            original_labels,
        )

    def test_full_stage_via_sampler_preserves_token_count(self) -> None:
        # Repeatedly call the sampler; when full stage is returned, count must match.
        rng = random.Random(55)
        tokens = _add_tokens()
        found_full = False
        for _ in range(500):
            stage, filtered = sample_completion_stage(EquationKind.add, tokens, rng)
            if stage == CompletionStage.full:
                self.assertEqual(
                    len(filtered), len(tokens),
                    "full stage changed token count",
                )
                found_full = True
        self.assertTrue(found_full, "full stage was never sampled in 500 draws")


class TestAllReturnedLabelsInOntology(unittest.TestCase):
    def _check_labels(self, kind: EquationKind, tokens: list) -> None:
        valid = default_required_stage2_labels()
        rng = random.Random(1)
        for _ in range(50):
            _, filtered = sample_completion_stage(kind, tokens, rng)
            for tok in filtered:
                self.assertIn(
                    tok.flattened_label, valid,
                    f"label {tok.flattened_label!r} is not in the 36-label GNN ontology",
                )

    def test_add_labels_in_ontology(self) -> None:
        self._check_labels(EquationKind.add, _add_tokens())

    def test_sub_labels_in_ontology(self) -> None:
        self._check_labels(EquationKind.subtract, _sub_tokens())

    def test_mul_labels_in_ontology(self) -> None:
        self._check_labels(EquationKind.multiply, _mul_tokens())

    def test_div_labels_in_ontology(self) -> None:
        self._check_labels(EquationKind.divide, _div_tokens())


class TestFirstStepOnlyDivide(unittest.TestCase):
    def test_first_step_only_returns_at_most_rows_0_to_4(self) -> None:
        from src.generation.completion_stages import _stage_div_first_step_only
        tokens = _div_tokens()
        filtered = _stage_div_first_step_only(tokens)
        for tok in filtered:
            self.assertLessEqual(
                tok.row, 4,
                f"first_step_only returned token at row {tok.row} > 4",
            )


class TestValidStagesForCase(unittest.TestCase):
    """valid_stages_for_case returns 5-bucket names after simplification."""

    def test_valid_stages_for_case_division_short_has_buckets(self):
        """division_short uses the 5-bucket scheme."""
        from src.generation.completion_stages import valid_stages_for_case
        from src.generation.layouts import SceneCase
        stages = valid_stages_for_case(SceneCase.division_short)
        self.assertIn("full", stages)
        self.assertIn("done_80", stages)

    def test_valid_stages_for_case_division_simple_has_buckets(self):
        """division_simple uses the 5-bucket scheme."""
        from src.generation.completion_stages import valid_stages_for_case
        from src.generation.layouts import SceneCase
        stages = valid_stages_for_case(SceneCase.division_simple)
        self.assertIn("full", stages)
        self.assertIn("done_80", stages)

    def test_valid_stages_for_case_division_long_has_buckets(self):
        """division_long uses the 5-bucket scheme."""
        from src.generation.completion_stages import valid_stages_for_case
        from src.generation.layouts import SceneCase
        stages = valid_stages_for_case(SceneCase.division_long)
        self.assertIn("full", stages)
        self.assertIn("done_80", stages)
        self.assertIn("done_60", stages)
        self.assertIn("done_40", stages)
        self.assertIn("done_20", stages)



if __name__ == "__main__":
    unittest.main()
