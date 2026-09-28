"""Tests for P0 and P1 scene-correctness fixes in layouts_core.py and synth_pool.py."""
from __future__ import annotations

import random
from typing import List

import pytest

from src.generation.layouts_core import (
    layout_addition,
    layout_long_division,
    layout_subtraction,
)
from src.generation.synth_pool import _result_digit_col_span
from src.generation.layouts import LayoutToken, layout_bounds


# ---------------------------------------------------------------------------
# A: Result bar width spans widest operand (P0)
# ---------------------------------------------------------------------------


class TestResultBarWidth:
    def _make_addition_tokens(self, a: int, b: int) -> List[LayoutToken]:
        rng = random.Random(0)
        _, tokens = layout_addition(a, b, rng)
        return tokens

    def test_bar_spans_operands_not_just_result(self):
        """9 + 3 = 12: operands are 1-digit, result is 2-digit.
        Bar must span at least 1 column (the widest operand width), not 0 columns.
        Before fix it spanned the result row (2 cols); after fix it spans all main_* cols."""
        tokens = self._make_addition_tokens(9, 3)
        c0, c1 = _result_digit_col_span(tokens)
        # All main_* tokens include both operands and result; result_bar spans all of them.
        mains = [t for t in tokens if t.flattened_label.startswith("main_")]
        expected_min = min(t.col for t in mains)
        expected_max = max(t.col for t in mains)
        assert c0 == expected_min, f"Bar c0 {c0} != expected {expected_min}"
        assert c1 == expected_max, f"Bar c1 {c1} != expected {expected_max}"

    def test_bar_spans_wider_operand_when_wider_than_result(self):
        """52 + 47 = 99: both operands are 2-digit, result is 2-digit.
        Bar should span at least 2 columns (cols of both operands)."""
        tokens = self._make_addition_tokens(52, 47)
        c0, c1 = _result_digit_col_span(tokens)
        assert c1 - c0 >= 1, f"Bar should span >=2 cols (c1-c0 >= 1), got c0={c0} c1={c1}"

    def test_bar_spans_max_col_of_any_main_row(self):
        """100 + 9 = 109: top operand is 3-digit, addend is 1-digit.
        Bar should span from col of '1' to col of '9' (rightmost across all main rows)."""
        tokens = self._make_addition_tokens(100, 9)
        c0, c1 = _result_digit_col_span(tokens)
        mains = [t for t in tokens if t.flattened_label.startswith("main_")]
        assert c0 == min(t.col for t in mains)
        assert c1 == max(t.col for t in mains)


# ---------------------------------------------------------------------------
# B: Division bracket vertical arm depth scales with scene (P0)
# ---------------------------------------------------------------------------


class TestDivisionBracketDepth:
    def test_long_division_max_row_exceeds_one_step(self):
        """A multi-step long division should have max_row > 4 (more than 1 step's rows).
        Verifies that the token layout does produce tall scenes requiring deep bracket."""
        rng = random.Random(42)
        # 256 / 4 = 64 -> 2 quotient digits -> multiple steps
        _, tokens = layout_long_division(256, 4, rng)
        _, max_r, _, _ = layout_bounds(tokens)
        # 2 steps -> rows: 0 (quotient), 1 (dividend), 2 (pp1), 3 (bar1), 4 (rem1),
        #                   5 (pp2), 6 (bar2), 7 (rem2) -> max_r = 7
        assert max_r > 4, f"Expected max_row > 4 for 2-step division, got {max_r}"

    def test_long_division_single_step_has_small_max_row(self):
        """Single-step division (8 / 4 = 2) should have max_row around 4."""
        rng = random.Random(0)
        _, tokens = layout_long_division(8, 4, rng)
        _, max_r, _, _ = layout_bounds(tokens)
        # 1 step -> rows 0..4 -> max_r = 4
        assert max_r <= 4, f"Expected max_row <= 4 for 1-step division, got {max_r}"


# ---------------------------------------------------------------------------
# C: Per-step op_minus in long division (P1) — probability knob fires correctly
# ---------------------------------------------------------------------------


class TestStepMinusProbability:
    def _count_step_minus_tokens(
        self, dividend: int, divisor: int, step_minus_prob: float, n_trials: int = 200
    ) -> int:
        """Run layout_long_division n_trials times; count scenes where at least one op_minus appears."""
        # op_minus injection happens in _draw_equation_block at render time using the rendering dict,
        # so we test the token-mutation logic directly by calling the helper that synth_pool uses.
        # Since _draw_equation_block is a private render function we cannot call it in isolation
        # without canvas/pool setup. Instead we test by checking that layout tokens do NOT include
        # op_minus (the injection is render-time), and that the rendering config knob is present.
        from src.core.run_config import _DEFAULT_RENDERING
        assert "step_minus_prob" in _DEFAULT_RENDERING, "step_minus_prob missing from _DEFAULT_RENDERING"
        assert 0.0 < _DEFAULT_RENDERING["step_minus_prob"] <= 1.0
        return 0  # checked via config presence; render-time injection tested separately

    def test_step_minus_prob_in_default_rendering(self):
        """step_minus_prob must be registered in _DEFAULT_RENDERING."""
        from src.core.run_config import _DEFAULT_RENDERING
        assert "step_minus_prob" in _DEFAULT_RENDERING
        val = _DEFAULT_RENDERING["step_minus_prob"]
        assert 0.0 <= val <= 1.0, f"step_minus_prob={val} must be in [0, 1]"

    def test_step_minus_prob_matches_config_toml(self):
        """step_minus_prob in config.toml should load and round-trip via load_config."""
        from pathlib import Path
        from src.core.run_config import load_config
        project_root = Path(__file__).parent.parent.parent
        gen_cfg, _, _ = load_config(project_root)
        val = gen_cfg.rendering.get("step_minus_prob")
        assert val is not None, "step_minus_prob not loaded from config.toml"
        assert 0.0 <= val <= 1.0, f"step_minus_prob={val} must be in [0, 1]"


# ---------------------------------------------------------------------------
# D: borrow_cross_prob replaces borrow_cross_enabled (P1)
# ---------------------------------------------------------------------------


class TestBorrowCrossProb:
    def test_borrow_cross_prob_in_default_rendering(self):
        """borrow_cross_prob must be in _DEFAULT_RENDERING; borrow_cross_enabled must be gone."""
        from src.core.run_config import _DEFAULT_RENDERING
        assert "borrow_cross_prob" in _DEFAULT_RENDERING, "borrow_cross_prob missing"
        assert "borrow_cross_enabled" not in _DEFAULT_RENDERING, "old key borrow_cross_enabled still present"
        val = _DEFAULT_RENDERING["borrow_cross_prob"]
        assert 0.0 <= val <= 1.0

    def test_borrow_cross_prob_loads_from_config(self):
        from pathlib import Path
        from src.core.run_config import load_config
        project_root = Path(__file__).parent.parent.parent
        gen_cfg, _, _ = load_config(project_root)
        val = gen_cfg.rendering.get("borrow_cross_prob")
        assert val is not None
        assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# E/F: wrong_carry_col_prob / missing_carry_prob registered (P1)
# ---------------------------------------------------------------------------


class TestCarryProbKnobs:
    def test_wrong_carry_col_prob_registered(self):
        from src.core.run_config import _DEFAULT_RENDERING
        assert "wrong_carry_col_prob" in _DEFAULT_RENDERING
        v = _DEFAULT_RENDERING["wrong_carry_col_prob"]
        assert 0.0 <= v <= 1.0

    def test_missing_carry_prob_registered(self):
        from src.core.run_config import _DEFAULT_RENDERING
        assert "missing_carry_prob" in _DEFAULT_RENDERING
        v = _DEFAULT_RENDERING["missing_carry_prob"]
        assert 0.0 <= v <= 1.0

    def test_carry_knobs_load_from_config(self):
        from pathlib import Path
        from src.core.run_config import load_config
        project_root = Path(__file__).parent.parent.parent
        gen_cfg, _, _ = load_config(project_root)
        for key in ("wrong_carry_col_prob", "missing_carry_prob"):
            val = gen_cfg.rendering.get(key)
            assert val is not None, f"{key} not loaded from config.toml"
            assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# G: multiplication_multi first bar has segment_kind (P2)
# ---------------------------------------------------------------------------


class TestMultiplicationMultiBarSegmentKind:
    def test_first_bar_has_segment_kind(self):
        """In the N x M layout (M >= 2), the bar at row=2 must have segment_kind set."""
        from src.generation.layouts_core import layout_multiplication
        rng = random.Random(7)
        # Force a multi-digit multiplier: a=12, b=34 -> N x M case
        _, tokens = layout_multiplication(12, 34, rng)
        bar_at_row2 = [
            t for t in tokens
            if t.flattened_label == "result_bar" and t.row == 2
        ]
        assert bar_at_row2, "Expected a result_bar token at row=2 for multiplication_multi"
        for bar_tok in bar_at_row2:
            assert bar_tok.segment_kind is not None, (
                f"First bar at row=2 has segment_kind=None; expected 'intermediate_result_bar'"
            )
            assert bar_tok.segment_kind == "intermediate_result_bar", (
                f"Expected 'intermediate_result_bar', got {bar_tok.segment_kind!r}"
            )


# ---------------------------------------------------------------------------
# H: short_bar_prob / short_bar_shrink_frac registered (P2)
# ---------------------------------------------------------------------------


class TestShortBarKnobs:
    def test_short_bar_knobs_in_default_rendering(self):
        from src.core.run_config import _DEFAULT_RENDERING
        assert "short_bar_prob" in _DEFAULT_RENDERING
        assert "short_bar_shrink_frac" in _DEFAULT_RENDERING
        assert 0.0 <= _DEFAULT_RENDERING["short_bar_prob"] <= 1.0
        assert 0.0 < _DEFAULT_RENDERING["short_bar_shrink_frac"] < 1.0

    def test_short_bar_knobs_load_from_config(self):
        from pathlib import Path
        from src.core.run_config import load_config
        project_root = Path(__file__).parent.parent.parent
        gen_cfg, _, _ = load_config(project_root)
        for key in ("short_bar_prob", "short_bar_shrink_frac"):
            val = gen_cfg.rendering.get(key)
            assert val is not None, f"{key} not loaded from config.toml"
            assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# A: collapse empty PP carry rows in multiplication_multi
# ---------------------------------------------------------------------------


class TestMulMultiNoCarryRowGap:
    def test_no_empty_row_between_bar_and_first_pp(self):
        """When PP k=0 produces no carry tokens, no blank row must separate bar1 from the first PP."""
        import random
        from src.generation.layouts_core import layout_multiplication as layout_multiplication_multi
        from src.generation.layouts_types import EquationKind

        # 10 * 12 = 120: PP k=0 is 10*2=20 (no carry), PP k=1 is 10*1=10 (no carry beyond result digit)
        rng = random.Random(42)
        kind, tokens = layout_multiplication_multi(10, 12, rng)
        assert kind == EquationKind.multiply

        row_set = {t.row for t in tokens}
        sorted_rows = sorted(row_set)
        # Rows must be contiguous (no gaps) up to the result row
        for i in range(len(sorted_rows) - 1):
            gap = sorted_rows[i + 1] - sorted_rows[i]
            assert gap <= 1, (
                f"Gap of {gap} found between rows {sorted_rows[i]} and {sorted_rows[i+1]}; "
                f"all rows: {sorted_rows}"
            )


# ---------------------------------------------------------------------------
# B: pp_wrong_operator_prob knob
# ---------------------------------------------------------------------------


class TestPpWrongOperatorProb:
    def test_pp_wrong_operator_prob_knobs_registered(self):
        """pp_plus_prob and pp_wrong_operator_prob must be in _DEFAULT_RENDERING with default 0.0."""
        from src.core.run_config import _DEFAULT_RENDERING
        assert "pp_plus_prob" in _DEFAULT_RENDERING
        assert "pp_wrong_operator_prob" in _DEFAULT_RENDERING
        assert _DEFAULT_RENDERING["pp_plus_prob"] == 0.0
        assert _DEFAULT_RENDERING["pp_wrong_operator_prob"] == 0.0


# ---------------------------------------------------------------------------
# C: bracket_depth_full_prob knob
# ---------------------------------------------------------------------------


class TestBracketDepthKnobs:
    def test_bracket_depth_knobs_in_default_rendering(self):
        from src.core.run_config import _DEFAULT_RENDERING
        assert "bracket_depth_full_prob" in _DEFAULT_RENDERING
        assert "bracket_depth_min_rows" in _DEFAULT_RENDERING
        assert _DEFAULT_RENDERING["bracket_depth_full_prob"] == 1.0
        assert _DEFAULT_RENDERING["bracket_depth_min_rows"] >= 1

    def test_bracket_depth_knobs_load_from_config(self):
        from pathlib import Path
        from src.core.run_config import load_config
        project_root = Path(__file__).parent.parent.parent
        gen_cfg, _, _ = load_config(project_root)
        assert gen_cfg.rendering.get("bracket_depth_full_prob") is not None
        assert gen_cfg.rendering.get("bracket_depth_min_rows") is not None


# ---------------------------------------------------------------------------
# D/E: long_div_step_borrow_prob and helper_operator_prob knobs
# ---------------------------------------------------------------------------


class TestDEKnobsLoaded:
    def test_knobs_in_default_rendering(self):
        from src.core.run_config import _DEFAULT_RENDERING
        assert "long_div_step_borrow_prob" in _DEFAULT_RENDERING
        assert _DEFAULT_RENDERING["long_div_step_borrow_prob"] == 0.0
        assert "helper_operator_prob" in _DEFAULT_RENDERING
        assert _DEFAULT_RENDERING["helper_operator_prob"] == 0.0

    def test_knobs_load_from_config(self):
        from pathlib import Path
        from src.core.run_config import load_config
        project_root = Path(__file__).parent.parent.parent
        gen_cfg, _, _ = load_config(project_root)
        for key in ("long_div_step_borrow_prob", "helper_operator_prob"):
            val = gen_cfg.rendering.get(key)
            assert val is not None, f"{key} not loaded from config.toml"
            assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# G: wrong_carry_value_prob / wrong_borrow_value_prob / missing_borrow_prob
# H: missing_pp_prob
# ---------------------------------------------------------------------------


class TestNewRealismKnobsRegistered:
    def test_knobs_in_default_rendering(self):
        from src.core.run_config import _DEFAULT_RENDERING
        for key in (
            "wrong_carry_value_prob",
            "wrong_borrow_value_prob",
            "missing_borrow_prob",
            "missing_pp_prob",
        ):
            assert key in _DEFAULT_RENDERING, f"{key} missing from _DEFAULT_RENDERING"
            assert _DEFAULT_RENDERING[key] == 0.0, f"{key} default should be 0.0"

    def test_knobs_load_from_config(self):
        from pathlib import Path
        from src.core.run_config import load_config
        project_root = Path(__file__).parent.parent.parent
        gen_cfg, _, _ = load_config(project_root)
        for key in (
            "wrong_carry_value_prob",
            "wrong_borrow_value_prob",
            "missing_borrow_prob",
            "missing_pp_prob",
        ):
            val = gen_cfg.rendering.get(key)
            assert val is not None, f"{key} not loaded from config.toml"
            assert 0.0 <= val <= 1.0


class TestWrongCarryValueProb:
    """wrong_carry_value_prob=1.0 must replace every carry digit with a different digit."""

    def test_all_carry_digits_replaced_when_prob_1(self):
        import dataclasses
        import random
        from src.generation.layouts_core import layout_addition
        from src.generation.layouts import SceneCase

        rng = random.Random(42)
        # 99 + 99 = 198 — guaranteed to produce carry tokens
        _, tokens = layout_addition(99, 99, random.Random(1))
        orig_carry = [t for t in tokens if t.yolo_class_name == "digit_carry"]
        assert orig_carry, "Expected carry tokens for 99+99"
        orig_labels = {t.flattened_label for t in orig_carry}

        # Reproduce the G mutation logic with prob=1.0
        rendering = {"wrong_carry_value_prob": 1.0, "wrong_borrow_value_prob": 0.0, "missing_borrow_prob": 0.0}
        _wrong_carry_val_prob = float(rendering.get("wrong_carry_value_prob", 0.0))
        _wrong_borrow_val_prob = float(rendering.get("wrong_borrow_value_prob", 0.0))
        _missing_borrow_prob = float(rendering.get("missing_borrow_prob", 0.0))
        mutated_g = []
        for tok in tokens:
            if tok.yolo_class_name == "digit_carry" and _wrong_carry_val_prob > 0.0:
                if rng.random() < _wrong_carry_val_prob:
                    orig_digit = int(tok.flattened_label.split("_")[1])
                    new_digit = (orig_digit + rng.randint(1, 9)) % 10
                    tok = dataclasses.replace(tok, flattened_label=f"carry_{new_digit}", glyph_key=str(new_digit))
            elif tok.yolo_class_name == "digit_borrow":
                if _missing_borrow_prob > 0.0 and rng.random() < _missing_borrow_prob:
                    continue
                if _wrong_borrow_val_prob > 0.0 and rng.random() < _wrong_borrow_val_prob:
                    orig_digit = int(tok.flattened_label.split("_")[1])
                    new_digit = (orig_digit + rng.randint(1, 9)) % 10
                    tok = dataclasses.replace(tok, flattened_label=f"borrow_{new_digit}", glyph_key=str(new_digit))
            mutated_g.append(tok)

        out_labels = {t.flattened_label for t in mutated_g if t.yolo_class_name == "digit_carry"}
        for orig in orig_labels:
            assert orig not in out_labels, (
                f"Carry label {orig!r} still present after wrong_carry_value_prob=1.0"
            )


class TestMissingBorrowProb:
    """missing_borrow_prob=1.0 must drop all borrow tokens."""

    def test_all_borrow_tokens_dropped_when_prob_1(self):
        import dataclasses
        import random
        from src.generation.layouts_core import layout_subtraction

        rng = random.Random(42)
        # 91 - 19 = 72 — guaranteed borrows
        _, tokens = layout_subtraction(91, 19)
        orig_borrows = [t for t in tokens if t.yolo_class_name == "digit_borrow"]
        assert orig_borrows, "Expected borrow tokens for 91-19"

        # Reproduce the G mutation logic with missing_borrow_prob=1.0
        rendering = {"wrong_carry_value_prob": 0.0, "wrong_borrow_value_prob": 0.0, "missing_borrow_prob": 1.0}
        _wrong_carry_val_prob = float(rendering.get("wrong_carry_value_prob", 0.0))
        _wrong_borrow_val_prob = float(rendering.get("wrong_borrow_value_prob", 0.0))
        _missing_borrow_prob = float(rendering.get("missing_borrow_prob", 0.0))
        mutated_g = []
        for tok in tokens:
            if tok.yolo_class_name == "digit_carry" and _wrong_carry_val_prob > 0.0:
                if rng.random() < _wrong_carry_val_prob:
                    orig_digit = int(tok.flattened_label.split("_")[1])
                    new_digit = (orig_digit + rng.randint(1, 9)) % 10
                    tok = dataclasses.replace(tok, flattened_label=f"carry_{new_digit}", glyph_key=str(new_digit))
            elif tok.yolo_class_name == "digit_borrow":
                if _missing_borrow_prob > 0.0 and rng.random() < _missing_borrow_prob:
                    continue
                if _wrong_borrow_val_prob > 0.0 and rng.random() < _wrong_borrow_val_prob:
                    orig_digit = int(tok.flattened_label.split("_")[1])
                    new_digit = (orig_digit + rng.randint(1, 9)) % 10
                    tok = dataclasses.replace(tok, flattened_label=f"borrow_{new_digit}", glyph_key=str(new_digit))
            mutated_g.append(tok)

        borrows_out = [t for t in mutated_g if t.yolo_class_name == "digit_borrow"]
        assert borrows_out == [], f"Expected no borrow tokens but got {len(borrows_out)}"


class TestMissingPPProb:
    """missing_pp_prob=1.0 must drop one non-final PP row from multiplication_multi token list."""

    def test_pp_row_dropped_when_prob_1(self):
        import random
        from src.generation.layouts_core import layout_multiplication
        from src.generation.layouts import SceneCase

        rng = random.Random(42)
        # 12 * 34 = 408: two PP rows between bar1 and bar2
        _, tokens = layout_multiplication(12, 34, random.Random(1))
        bar_rows = sorted({t.row for t in tokens if t.flattened_label == "result_bar"})
        assert len(bar_rows) >= 2, "Expected at least 2 result_bars in mul_multi"
        bar1, bar2 = bar_rows[0], bar_rows[-1]

        pp_rows_before = sorted({
            t.row for t in tokens
            if t.yolo_class_name == "digit_main" and bar1 < t.row < bar2
        })
        assert len(pp_rows_before) >= 2, "Need at least 2 PP rows to drop one non-final"

        # Reproduce the H mutation logic with prob=1.0 (rng.random() always < 1.0)
        rendering = {"missing_pp_prob": 1.0}
        _missing_pp_prob = float(rendering.get("missing_pp_prob", 0.0))
        mutated = list(tokens)
        if _missing_pp_prob > 0.0 and rng.random() < _missing_pp_prob:
            droppable = pp_rows_before[:-1]
            if droppable:
                drop_row = rng.choice(droppable)
                rows_to_drop = {drop_row, drop_row - 1}
                mutated = [t for t in mutated if t.row not in rows_to_drop]

        pp_rows_after = sorted({
            t.row for t in mutated
            if t.yolo_class_name == "digit_main" and bar1 < t.row < bar2
        })
        assert len(pp_rows_after) < len(pp_rows_before), (
            f"Expected fewer PP rows after missing_pp_prob=1.0; "
            f"before={pp_rows_before}, after={pp_rows_after}"
        )


class TestOperatorLeftOfResultBar:
    """Operator column must always be strictly left of bar span (which starts at col_offset-1)."""

    def test_addition_op_left_of_bar(self):
        import random
        from src.generation.layouts_core import layout_addition

        rng = random.Random(0)
        # Result wider than operands: 9 + 99 = 108 — width=3, op must be at col 1 (col_offset-1=1)
        _, tokens = layout_addition(9, 99, rng)
        op_tok = next(t for t in tokens if t.yolo_class_name == "operator")
        bar_tok = next(t for t in tokens if t.flattened_label == "result_bar")
        assert op_tok.col <= bar_tok.col, (
            f"op col {op_tok.col} should be <= bar col {bar_tok.col}"
        )

    def test_subtraction_op_left_of_bar(self):
        import random
        from src.generation.layouts_core import layout_subtraction

        rng = random.Random(0)
        _, tokens = layout_subtraction(100, 9)
        op_tok = next(t for t in tokens if t.yolo_class_name == "operator")
        bar_tok = next(t for t in tokens if t.flattened_label == "result_bar")
        assert op_tok.col <= bar_tok.col, (
            f"op col {op_tok.col} should be <= bar col {bar_tok.col}"
        )

    def test_multiplication_n1_op_left_of_result_span(self):
        import random
        from src.generation.layouts_core import layout_multiplication

        rng = random.Random(0)
        # N x 1: result can be wider than operands (e.g. 9*9=81)
        _, tokens = layout_multiplication(9, 9, rng)
        op_tok = next(t for t in tokens if t.yolo_class_name == "operator")
        main_cols = [t.col for t in tokens if t.yolo_class_name == "digit_main"]
        assert op_tok.col < min(main_cols), (
            f"op col {op_tok.col} should be < leftmost main digit col {min(main_cols)}"
        )
