from __future__ import annotations
import unittest
from src.parsing.assemble import (
    NodePrediction,
    assemble_json,
    _validate_equation_structure,
    _classify_gate,
    _GATE_OOD,
    _GATE_BARE_DIGITS,
    _GATE_VALID,
    _flag_spurious_carries,
    _deduplicate_operators,
    _apply_kind_override,
    _kind_low_confidence_reason,
    _apply_heuristic_overrides,
    _check_entropy_ood,
    _check_spatial_consistency,
    _ENTROPY_OOD_THRESHOLD,
    OOD_REASON_TOO_FEW_SYMBOLS,
    OOD_REASON_NO_DIGITS,
    OOD_REASON_NO_STRUCTURAL_TOKEN,
    OOD_REASON_TRUE_EMPTY,
    OOD_REASON_SINGLE_NON_DIGIT,
    OOD_REASON_HIGH_ENTROPY,
    OOD_EQUATION_KIND,
)


def _pred(fine_label, row_cluster_id, col_cluster_id, x0=0.0, y0=0.0, x1=10.0, y1=10.0, conf=0.95):
    return NodePrediction(
        fine_label=fine_label,
        row_cluster_id=row_cluster_id,
        col_cluster_id=col_cluster_id,
        within_row_ord=0,
        within_col_ord=0,
        x0=x0, y0=y0, x1=x1, y1=y1,
        confidence=conf,
    )


# Minimal valid scene: 1 digit + 1 operator satisfies all three invariants.
def _valid_pair():
    return [_pred("main_3", 0, 0), _pred("op_plus", 0, 1)]


class TestAssembleJson(unittest.TestCase):
    def test_schema_version_and_parser_mode(self) -> None:
        # Uses a valid scene (2+ symbols, digit + operator) so the gate passes.
        out = assemble_json(_valid_pair(), "add")
        self.assertEqual(out["schema_version"], 1)
        self.assertEqual(out["spatial_meta"]["parser_mode"], "gnn")

    def test_equation_kind_passthrough(self) -> None:
        # Valid scene: main_3 + op_plus. equation_kind is the recognized kind.
        out = assemble_json(_valid_pair(), "add")
        self.assertEqual(out["equation_kind"], "add")

    def test_rows_sorted_and_renumbered(self) -> None:
        # Two digits but no structural token -> would be OOD; add an operator to each row.
        preds = [
            _pred("main_5", 3, 0),
            _pred("op_plus", 3, 1),
            _pred("main_1", 0, 0),
            _pred("op_plus", 0, 1),
        ]
        out = assemble_json(preds, "add")
        self.assertEqual(out["row_count"], 2)
        self.assertEqual(out["rows"][0]["tokens"][0]["label"], "main_1")
        self.assertEqual(out["rows"][1]["tokens"][0]["label"], "main_5")

    def test_tokens_sorted_by_col_within_row(self) -> None:
        preds = [_pred("main_9", 0, 2), _pred("main_1", 0, 0), _pred("main_4", 0, 1),
                 _pred("op_plus", 0, 3)]
        out = assemble_json(preds, "add")
        labels = [t["label"] for t in out["rows"][0]["tokens"]]
        self.assertEqual(labels, ["main_1", "main_4", "main_9", "op_plus"])

    def test_grid_col_is_col_cluster_id_from_prediction(self) -> None:
        # carry_1 at grid_col=5, with a main_3 in same col so it is not flagged
        # (recognize-as-drawn keeps it regardless; this scene keeps it un-flagged).
        # carry y0=0, main y0=0 → equal, not strictly less → carry_conflict=False.
        preds = [_pred("carry_1", 0, 5), _pred("main_3", 0, 5), _pred("op_plus", 0, 0)]
        out = assemble_json(preds, "add")
        carry_token = next(t for t in out["rows"][0]["tokens"] if t["label"] == "carry_1")
        self.assertEqual(carry_token["grid_col"], 5)

    def test_within_row_ord_and_within_col_ord_in_token(self) -> None:
        pred = NodePrediction(
            fine_label="main_3", row_cluster_id=0, col_cluster_id=0,
            within_row_ord=3, within_col_ord=2,
            x0=0.0, y0=0.0, x1=10.0, y1=10.0, confidence=0.9,
        )
        # Add an operator so scene passes the structural gate.
        out = assemble_json([pred, _pred("op_plus", 0, 1)], "add")
        main_token = next(t for t in out["rows"][0]["tokens"] if t["label"] == "main_3")
        self.assertEqual(main_token["within_row_ord"], 3)
        self.assertEqual(main_token["within_col_ord"], 2)

    def test_slots_bucketing(self) -> None:
        # Recognize-as-drawn: carry_1 and borrow_2 are KEPT regardless of whether
        # they have a main sibling. Here each has a main_* in the same col, so they
        # are not flagged (carry y0=0 == main_min y0=0, filter uses strict <).
        preds = [
            _pred("main_3",     1, 0),           # main in col=0, y0=0
            _pred("main_4",     1, 1),           # main in col=1, y0=0 (anchor for borrow)
            _pred("carry_1",    0, 0),           # carry col=0, y0=0 == main_min → kept, not flagged
            _pred("borrow_2",   0, 1),           # borrow col=1, y0=0 == main_min → kept, not flagged
            _pred("op_plus",    1, 2),
            _pred("result_bar", 2, 0),
        ]
        slots = assemble_json(preds, "add")["slots"]
        self.assertEqual(len(slots["main_digits"]), 2)
        self.assertEqual(len(slots["carries"]), 1)
        self.assertEqual(len(slots["borrows"]), 1)
        self.assertEqual(len(slots["operators"]), 1)
        self.assertEqual(len(slots["structures"]), 1)

    def test_div_bracket_buckets_into_structures_on_valid_division_path(self) -> None:
        # Valid division scene (main + div_bracket + main): the gate passes and
        # the scene takes the success path. div_bracket parses to role="unknown"
        # (no digit suffix), so without the _STRUCTURAL_FINE_LABELS fallback it
        # would mis-bucket into main_digits. This asserts it lands in structures
        # (with kind=fine_label) and never in main_digits, mirroring the
        # bare_digits standalone_bracket route.
        preds = [_pred("main_8", 0, 0), _pred("div_bracket", 0, 1), _pred("main_4", 0, 2)]
        out = assemble_json(preds, "divide")
        self.assertEqual(out["equation_kind"], "divide")
        self.assertNotIn("ood_reason", out)
        slots = out["slots"]
        self.assertEqual(len(slots["structures"]), 1)
        self.assertEqual(slots["structures"][0]["label"], "div_bracket")
        self.assertEqual(slots["structures"][0]["kind"], "div_bracket")
        self.assertEqual(len(slots["main_digits"]), 2)
        self.assertFalse(any(t["label"] == "div_bracket" for t in slots["main_digits"]))

    def test_carry_slot_has_digit_field(self) -> None:
        # carry_7 below a main_3 in same column + op_plus → valid scene.
        # carry y0=50 > main y0=10, so carry is below main (not flagged).
        preds = [
            _pred("carry_7", 0, 0, y0=50.0, y1=60.0),
            _pred("main_3", 0, 0, y0=10.0, y1=20.0),
            _pred("op_plus", 0, 1),
        ]
        out = assemble_json(preds, "add")
        self.assertEqual(out["slots"]["carries"][0]["digit"], "7")

    def test_operator_slot_has_operator_field(self) -> None:
        # op_times + main_4 → valid scene.
        out = assemble_json([_pred("op_times", 0, 0), _pred("main_4", 0, 1)], "multiply")
        self.assertEqual(out["slots"]["operators"][0]["operator"], "times")

    def test_token_has_carry_conflict_field(self) -> None:
        # Every token carries a carry_conflict diagnostic field (default False).
        out = assemble_json(_valid_pair(), "add")
        for tok in out["rows"][0]["tokens"]:
            self.assertIn("carry_conflict", tok)
            self.assertFalse(tok["carry_conflict"])

    def test_empty_predictions_returns_ood(self) -> None:
        # W10: empty list now routes to true_empty (not too_few_symbols).
        out = assemble_json([], "add")
        self.assertEqual(out["schema_version"], 1)
        self.assertEqual(out["equation_kind"], OOD_EQUATION_KIND)
        self.assertEqual(out["ood_reason"], OOD_REASON_TRUE_EMPTY)
        self.assertEqual(out["rows"], [])
        self.assertEqual(out["slots"], [])
        self.assertEqual(out["spatial_meta"], {"empty": True})
        self.assertNotIn("row_count", out)


# ---------------------------------------------------------------------------
# Tests for _validate_equation_structure (unit-level)
# ---------------------------------------------------------------------------

class TestValidateEquationStructure(unittest.TestCase):
    def test_valid_full_scene(self) -> None:
        preds = [
            _pred("main_3", 0, 0),
            _pred("main_7", 0, 1),
            _pred("main_9", 1, 0),
            _pred("op_plus", 0, 2),
            _pred("result_bar", 2, 0),
        ]
        valid, reason = _validate_equation_structure(preds)
        self.assertTrue(valid)
        self.assertIsNone(reason)

    def test_empty_list_too_few_symbols(self) -> None:
        valid, reason = _validate_equation_structure([])
        self.assertFalse(valid)
        self.assertEqual(reason, OOD_REASON_TOO_FEW_SYMBOLS)

    def test_single_symbol_too_few_symbols(self) -> None:
        valid, reason = _validate_equation_structure([_pred("main_5", 0, 0)])
        self.assertFalse(valid)
        self.assertEqual(reason, OOD_REASON_TOO_FEW_SYMBOLS)

    def test_no_digits(self) -> None:
        # Two operators, zero digits.
        preds = [_pred("op_plus", 0, 0), _pred("op_minus", 0, 1)]
        valid, reason = _validate_equation_structure(preds)
        self.assertFalse(valid)
        self.assertEqual(reason, OOD_REASON_NO_DIGITS)

    def test_no_structural_token(self) -> None:
        # Three digits, no operator / bar / bracket.
        preds = [_pred("main_1", 0, 0), _pred("main_2", 0, 1), _pred("main_3", 0, 2)]
        valid, reason = _validate_equation_structure(preds)
        self.assertFalse(valid)
        self.assertEqual(reason, OOD_REASON_NO_STRUCTURAL_TOKEN)

    def test_minimum_valid_one_digit_one_operator(self) -> None:
        # Exactly 1 digit + 1 operator satisfies all invariants.
        preds = [_pred("main_4", 0, 0), _pred("op_plus", 0, 1)]
        valid, reason = _validate_equation_structure(preds)
        self.assertTrue(valid)
        self.assertIsNone(reason)

    def test_carry_counts_as_digit(self) -> None:
        preds = [_pred("carry_2", 0, 0), _pred("op_minus", 0, 1)]
        valid, reason = _validate_equation_structure(preds)
        self.assertTrue(valid)
        self.assertIsNone(reason)

    def test_borrow_counts_as_digit(self) -> None:
        preds = [_pred("borrow_3", 0, 0), _pred("result_bar", 1, 0)]
        valid, reason = _validate_equation_structure(preds)
        self.assertTrue(valid)
        self.assertIsNone(reason)

    def test_div_bracket_counts_as_structural(self) -> None:
        preds = [_pred("main_6", 0, 0), _pred("div_bracket", 0, 1)]
        valid, reason = _validate_equation_structure(preds)
        self.assertTrue(valid)
        self.assertIsNone(reason)

    def test_result_bar_counts_as_structural(self) -> None:
        preds = [_pred("main_8", 0, 0), _pred("result_bar", 1, 0)]
        valid, reason = _validate_equation_structure(preds)
        self.assertTrue(valid)
        self.assertIsNone(reason)


# ---------------------------------------------------------------------------
# Tests for assemble_json OOD paths
# ---------------------------------------------------------------------------

class TestAssembleJsonOodPaths(unittest.TestCase):
    def _assert_ood_shape(self, out: dict, expected_reason: str) -> None:
        self.assertEqual(out["schema_version"], 1)
        self.assertEqual(out["equation_kind"], OOD_EQUATION_KIND)
        self.assertEqual(out["ood_reason"], expected_reason)
        self.assertEqual(out["rows"], [])
        self.assertEqual(out["slots"], [])
        self.assertEqual(out["spatial_meta"], {"empty": True})

    def test_valid_scene_has_no_ood_reason(self) -> None:
        out = assemble_json(_valid_pair(), "add")
        self.assertNotIn("ood_reason", out)
        self.assertEqual(out["equation_kind"], "add")

    def test_ood_no_structural_token_now_bare_digits(self) -> None:
        # W10: digits-only scenes no longer OOD — they route to bare_digits.
        preds = [_pred("main_1", 0, 0), _pred("main_2", 0, 1), _pred("main_3", 0, 2)]
        out = assemble_json(preds, "add")
        self.assertEqual(out["equation_kind"], "bare_digits")
        self.assertNotIn("ood_reason", out)
        self.assertGreater(len(out["rows"]), 0)

    def test_ood_empty_now_true_empty(self) -> None:
        # W10: empty list routes to true_empty.
        out = assemble_json([], "add")
        self._assert_ood_shape(out, OOD_REASON_TRUE_EMPTY)

    def test_ood_no_digits_returns_unknown(self) -> None:
        preds = [_pred("op_plus", 0, 0), _pred("op_minus", 0, 1)]
        out = assemble_json(preds, "add")
        self._assert_ood_shape(out, OOD_REASON_NO_DIGITS)


# ---------------------------------------------------------------------------
# Rule 1: Spatial carry diagnostic (recognize-as-drawn — flag, never delete)
# ---------------------------------------------------------------------------

class TestFlagSpuriousCarries(unittest.TestCase):
    """The carry/borrow spatial check now FLAGS (carry_conflict=True) instead of
    deleting. Recognize-as-drawn: a drawn carry/borrow is always retained so it
    surfaces in the output; the flag is observability only."""

    def _make(self, label, col, y0, y1, row=0, conf=0.9):
        return NodePrediction(
            fine_label=label, row_cluster_id=row, col_cluster_id=col,
            within_row_ord=0, within_col_ord=0,
            x0=0.0, y0=y0, x1=10.0, y1=y1, confidence=conf,
        )

    def _find(self, preds, label):
        return next(p for p in preds if p.fine_label == label)

    def test_carry_above_all_mains_flagged_not_dropped(self) -> None:
        # carry y0=5, main y0=50 → carry is above (smaller y) → flagged, KEPT.
        preds = [
            self._make("carry_1", col=0, y0=5.0, y1=15.0),
            self._make("main_3", col=0, y0=50.0, y1=70.0),
            self._make("op_plus", col=1, y0=50.0, y1=70.0),
        ]
        result = _flag_spurious_carries(preds)
        labels = [p.fine_label for p in result]
        self.assertIn("carry_1", labels)  # retained, not deleted
        self.assertIn("main_3", labels)
        self.assertTrue(self._find(result, "carry_1").carry_conflict)
        self.assertFalse(self._find(result, "main_3").carry_conflict)

    def test_carry_below_main_kept_unflagged(self) -> None:
        # carry y0=60 > main y0=30 → carry is below → kept, NOT flagged.
        preds = [
            self._make("main_3", col=0, y0=30.0, y1=50.0),
            self._make("carry_1", col=0, y0=60.0, y1=75.0),
            self._make("op_plus", col=1, y0=30.0, y1=50.0),
        ]
        result = _flag_spurious_carries(preds)
        labels = [p.fine_label for p in result]
        self.assertIn("carry_1", labels)
        self.assertFalse(self._find(result, "carry_1").carry_conflict)

    def test_carry_no_mains_in_column_flagged_not_dropped(self) -> None:
        # carry in col=1, but no main_* in col=1 → flagged, KEPT (was deleted).
        preds = [
            self._make("main_3", col=0, y0=30.0, y1=50.0),
            self._make("carry_2", col=1, y0=10.0, y1=25.0),
            self._make("op_plus", col=2, y0=30.0, y1=50.0),
        ]
        result = _flag_spurious_carries(preds)
        labels = [p.fine_label for p in result]
        self.assertIn("carry_2", labels)  # retained
        self.assertTrue(self._find(result, "carry_2").carry_conflict)

    def test_borrow_above_mains_flagged_not_dropped(self) -> None:
        preds = [
            self._make("borrow_0", col=0, y0=5.0, y1=15.0),
            self._make("main_5", col=0, y0=50.0, y1=70.0),
            self._make("op_minus", col=1, y0=50.0, y1=70.0),
        ]
        result = _flag_spurious_carries(preds)
        labels = [p.fine_label for p in result]
        self.assertIn("borrow_0", labels)  # retained
        self.assertTrue(self._find(result, "borrow_0").carry_conflict)

    def test_length_preserved_nothing_deleted(self) -> None:
        # The flagger never changes the number of predictions.
        preds = [
            self._make("carry_1", col=9, y0=5.0, y1=15.0),   # spurious col
            self._make("main_3", col=0, y0=50.0, y1=70.0),
            self._make("op_plus", col=1, y0=50.0, y1=70.0),
        ]
        result = _flag_spurious_carries(preds)
        self.assertEqual(len(result), len(preds))

    def test_carry_no_main_in_column_retained_and_flagged_via_assemble(self) -> None:
        # EXPLICIT TASK TEST (1): a carry with no main in its column is RETAINED
        # in the output with carry_conflict=True (never silently removed).
        preds = [
            _pred("main_3", 0, 0, y0=50.0, y1=70.0),
            _pred("op_plus", 0, 1, y0=50.0, y1=70.0),
            # carry in an empty column with no main below it.
            _pred("carry_9", 0, 5, y0=5.0, y1=15.0),
        ]
        out = assemble_json(preds, "add")
        # Scene stays valid addition; the carry was NOT dropped.
        self.assertEqual(out["equation_kind"], "add")
        labels = [d["label"] for d in out["detections"]]
        self.assertIn("carry_9", labels)
        self.assertEqual(len(out["slots"]["carries"]), 1)
        carry_slot = out["slots"]["carries"][0]
        self.assertEqual(carry_slot["label"], "carry_9")
        self.assertTrue(carry_slot["carry_conflict"])

    def test_retained_spurious_carry_makes_lone_operator_scene_valid(self) -> None:
        # Previously the carry was deleted, leaving a lone op_plus → bare_digits.
        # Now the carry is retained (carry_conflict=True): carry (digit) + op_plus
        # (structural) = an equation-shaped scene, so equation_kind is the
        # recognized "add" and both tokens survive.
        preds = [
            NodePrediction(
                fine_label="carry_1", row_cluster_id=0, col_cluster_id=0,
                within_row_ord=0, within_col_ord=0,
                x0=0.0, y0=5.0, x1=10.0, y1=15.0, confidence=0.9,
            ),
            NodePrediction(
                fine_label="op_plus", row_cluster_id=0, col_cluster_id=1,
                within_row_ord=0, within_col_ord=0,
                x0=20.0, y0=50.0, x1=30.0, y1=60.0, confidence=0.9,
            ),
        ]
        out = assemble_json(preds, "add")
        self.assertEqual(out["equation_kind"], "add")
        self.assertNotIn("ood_reason", out)
        self.assertEqual(len(out["detections"]), 2)
        labels = sorted(d["label"] for d in out["detections"])
        self.assertEqual(labels, ["carry_1", "op_plus"])
        self.assertTrue(out["slots"]["carries"][0]["carry_conflict"])


# ---------------------------------------------------------------------------
# W1 Rule 2: Operator deduplication
# ---------------------------------------------------------------------------

class TestDeduplicateOperators(unittest.TestCase):
    def _op(self, label, row, x0, x1, conf=0.9):
        return NodePrediction(
            fine_label=label, row_cluster_id=row, col_cluster_id=0,
            within_row_ord=0, within_col_ord=0,
            x0=x0, y0=50.0, x1=x1, y1=60.0, confidence=conf,
        )

    def test_two_stacked_op_minus_same_row_overlapping_x_keeps_highest_conf(self) -> None:
        low = self._op("op_minus", row=0, x0=10.0, x1=30.0, conf=0.6)
        high = self._op("op_minus", row=0, x0=12.0, x1=28.0, conf=0.9)
        result = _deduplicate_operators([low, high])
        ops = [p for p in result if p.fine_label == "op_minus"]
        self.assertEqual(len(ops), 1)
        self.assertAlmostEqual(ops[0].confidence, 0.9)

    def test_two_op_minus_different_rows_both_kept(self) -> None:
        a = self._op("op_minus", row=0, x0=10.0, x1=30.0, conf=0.8)
        b = self._op("op_minus", row=1, x0=10.0, x1=30.0, conf=0.7)
        result = _deduplicate_operators([a, b])
        ops = [p for p in result if p.fine_label == "op_minus"]
        self.assertEqual(len(ops), 2)

    def test_two_op_minus_same_row_non_overlapping_both_kept(self) -> None:
        a = self._op("op_minus", row=0, x0=5.0, x1=15.0, conf=0.8)
        b = self._op("op_minus", row=0, x0=20.0, x1=30.0, conf=0.7)
        result = _deduplicate_operators([a, b])
        ops = [p for p in result if p.fine_label == "op_minus"]
        self.assertEqual(len(ops), 2)

    def test_non_operator_tokens_unchanged(self) -> None:
        preds = [
            NodePrediction(
                fine_label="main_3", row_cluster_id=0, col_cluster_id=0,
                within_row_ord=0, within_col_ord=0,
                x0=0.0, y0=0.0, x1=10.0, y1=10.0, confidence=0.9,
            ),
        ]
        result = _deduplicate_operators(preds)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].fine_label, "main_3")


# ---------------------------------------------------------------------------
# Rule 3: Kind passthrough (recognize-as-drawn — never relabel by operator vote)
# ---------------------------------------------------------------------------

class TestApplyKindOverride(unittest.TestCase):
    """_apply_kind_override is now a recognize-as-drawn passthrough: it returns
    the recognized equation_type unchanged. It no longer relabels a divide scene
    by operator vote, and never forces 'unknown' on a populated scene."""

    def _p(self, label):
        return NodePrediction(
            fine_label=label, row_cluster_id=0, col_cluster_id=0,
            within_row_ord=0, within_col_ord=0,
            x0=0.0, y0=0.0, x1=10.0, y1=10.0, confidence=0.9,
        )

    def test_divide_with_bracket_unchanged(self) -> None:
        preds = [self._p("div_bracket"), self._p("main_5")]
        result = _apply_kind_override(preds, "divide")
        self.assertEqual(result, "divide")

    def test_divide_no_bracket_op_plus_stays_divide(self) -> None:
        # Was relabelled to "add" by operator vote; now passthrough keeps "divide".
        preds = [self._p("main_5"), self._p("op_plus")]
        result = _apply_kind_override(preds, "divide")
        self.assertEqual(result, "divide")

    def test_divide_no_bracket_op_minus_stays_divide(self) -> None:
        preds = [self._p("main_5"), self._p("op_minus")]
        result = _apply_kind_override(preds, "divide")
        self.assertEqual(result, "divide")

    def test_divide_no_bracket_op_times_stays_divide(self) -> None:
        preds = [self._p("main_5"), self._p("op_times")]
        result = _apply_kind_override(preds, "divide")
        self.assertEqual(result, "divide")

    def test_divide_no_bracket_no_operator_stays_divide_not_unknown(self) -> None:
        # Was forced to 'unknown'; recognize-as-drawn keeps the recognized kind.
        preds = [self._p("main_5"), self._p("result_bar")]
        result = _apply_kind_override(preds, "divide")
        self.assertEqual(result, "divide")
        self.assertNotEqual(result, OOD_EQUATION_KIND)

    def test_non_divide_equation_type_unchanged(self) -> None:
        preds = [self._p("main_5"), self._p("op_plus")]
        self.assertEqual(_apply_kind_override(preds, "add"), "add")
        self.assertEqual(_apply_kind_override(preds, "subtract"), "subtract")

    def test_kind_low_confidence_reason_divide_without_bracket(self) -> None:
        # Diagnostic helper flags divide-with-no-bracket as a low-confidence
        # recognition WITHOUT mutating the kind.
        preds = [self._p("main_5"), self._p("op_plus")]
        self.assertEqual(
            _kind_low_confidence_reason(preds, "divide"), "divide_without_bracket"
        )

    def test_kind_low_confidence_reason_none_when_bracket_present(self) -> None:
        preds = [self._p("main_5"), self._p("div_bracket")]
        self.assertIsNone(_kind_low_confidence_reason(preds, "divide"))

    def test_kind_low_confidence_reason_none_for_non_divide(self) -> None:
        preds = [self._p("main_5"), self._p("op_plus")]
        self.assertIsNone(_kind_low_confidence_reason(preds, "add"))

    def test_kind_override_via_assemble_divide_kept_not_relabelled(self) -> None:
        # Full assembler integration: divide head + op_plus + no bracket. Old
        # behavior relabelled to "add"; recognize-as-drawn keeps "divide" and
        # surfaces the diagnostic kind_low_conf_reason instead.
        preds = [
            NodePrediction(
                fine_label="main_5", row_cluster_id=0, col_cluster_id=0,
                within_row_ord=0, within_col_ord=0,
                x0=0.0, y0=0.0, x1=10.0, y1=10.0, confidence=0.9,
            ),
            NodePrediction(
                fine_label="op_plus", row_cluster_id=0, col_cluster_id=1,
                within_row_ord=0, within_col_ord=0,
                x0=20.0, y0=0.0, x1=30.0, y1=10.0, confidence=0.9,
            ),
        ]
        out = assemble_json(preds, "divide")
        self.assertEqual(out["equation_kind"], "divide")
        self.assertNotIn("ood_reason", out)
        self.assertEqual(out["kind_low_conf_reason"], "divide_without_bracket")


# ---------------------------------------------------------------------------
# Rule 4: Entropy diagnostic — keep structure, demote kind to 'unknown'
# ---------------------------------------------------------------------------

import math as _math

class TestEntropyOodGate(unittest.TestCase):
    def _uniform_logits(self):
        # Uniform over 4 classes → max entropy → above threshold
        return [1.0, 1.0, 1.0, 1.0]

    def _peaked_logits(self):
        # Strongly peaked → low entropy → below threshold
        return [100.0, 0.0, 0.0, 0.0]

    def test_entropy_gate_fires_above_threshold(self) -> None:
        # Uniform logits → high entropy → diagnostic reason returned.
        reason = _check_entropy_ood(self._uniform_logits())
        self.assertEqual(reason, OOD_REASON_HIGH_ENTROPY)

    def test_entropy_gate_passes_peaked_logits(self) -> None:
        reason = _check_entropy_ood(self._peaked_logits())
        self.assertIsNone(reason)

    def test_entropy_gate_skipped_when_none(self) -> None:
        reason = _check_entropy_ood(None)
        self.assertIsNone(reason)

    def test_entropy_gate_via_assemble_high_entropy_keeps_structure(self) -> None:
        # Valid scene structurally, uniform logits → equation_kind demoted to
        # 'unknown' with an ood_reason diagnostic, but the recognized rows/slots
        # are KEPT (recognize-as-drawn), NOT blanked to [].
        preds = [
            NodePrediction(
                fine_label="main_5", row_cluster_id=0, col_cluster_id=0,
                within_row_ord=0, within_col_ord=0,
                x0=0.0, y0=0.0, x1=10.0, y1=10.0, confidence=0.9,
            ),
            NodePrediction(
                fine_label="op_plus", row_cluster_id=0, col_cluster_id=1,
                within_row_ord=0, within_col_ord=0,
                x0=20.0, y0=0.0, x1=30.0, y1=10.0, confidence=0.9,
            ),
        ]
        out = assemble_json(preds, "add", eq_type_logits=[1.0, 1.0, 1.0, 1.0])
        self.assertEqual(out["equation_kind"], OOD_EQUATION_KIND)
        self.assertEqual(out["ood_reason"], OOD_REASON_HIGH_ENTROPY)
        # Structure preserved, not blanked.
        self.assertNotEqual(out["rows"], [])
        self.assertGreater(len(out["rows"]), 0)
        self.assertIsInstance(out["slots"], dict)
        self.assertEqual(len(out["slots"]["main_digits"]), 1)
        self.assertEqual(len(out["slots"]["operators"]), 1)
        self.assertEqual(len(out["detections"]), 2)
        # spatial_meta keeps the normal success shape (not {"empty": True}).
        self.assertEqual(out["spatial_meta"]["parser_mode"], "gnn")

    def test_entropy_gate_via_assemble_none_logits_passes(self) -> None:
        preds = [
            NodePrediction(
                fine_label="main_5", row_cluster_id=0, col_cluster_id=0,
                within_row_ord=0, within_col_ord=0,
                x0=0.0, y0=0.0, x1=10.0, y1=10.0, confidence=0.9,
            ),
            NodePrediction(
                fine_label="op_plus", row_cluster_id=0, col_cluster_id=1,
                within_row_ord=0, within_col_ord=0,
                x0=20.0, y0=0.0, x1=30.0, y1=10.0, confidence=0.9,
            ),
        ]
        out = assemble_json(preds, "add", eq_type_logits=None)
        self.assertNotIn("ood_reason", out)
        self.assertEqual(out["equation_kind"], "add")


# ---------------------------------------------------------------------------
# W1 Rule 5: Spatial sanity check
# ---------------------------------------------------------------------------

class TestSpatialSanityCheck(unittest.TestCase):
    def _node(self, label, row, y0, y1):
        return NodePrediction(
            fine_label=label, row_cluster_id=row, col_cluster_id=0,
            within_row_ord=0, within_col_ord=0,
            x0=0.0, y0=y0, x1=10.0, y1=y1, confidence=0.9,
        )

    def test_no_conflict_monotonic_rows(self) -> None:
        # row=0 has lower y than row=1 → no conflict
        preds = [
            self._node("main_1", row=0, y0=10.0, y1=30.0),
            self._node("main_2", row=1, y0=50.0, y1=70.0),
        ]
        result, count = _check_spatial_consistency(preds)
        self.assertEqual(count, 0)
        self.assertFalse(any(p.spatial_conflict for p in result))

    def test_conflict_detected_when_row_order_inverted(self) -> None:
        # row=0 has HIGHER y than row=1 → conflict on both
        preds = [
            self._node("main_1", row=0, y0=80.0, y1=100.0),
            self._node("main_2", row=1, y0=10.0, y1=30.0),
        ]
        result, count = _check_spatial_consistency(preds)
        self.assertGreater(count, 0)
        self.assertTrue(any(p.spatial_conflict for p in result))

    def test_result_bar_excluded_from_monotonicity_check(self) -> None:
        # result_bar appears below digit rows but should not trigger conflict
        preds = [
            self._node("main_1", row=0, y0=10.0, y1=30.0),
            self._node("main_2", row=1, y0=50.0, y1=70.0),
            self._node("result_bar", row=2, y0=200.0, y1=210.0),
        ]
        result, count = _check_spatial_consistency(preds)
        self.assertEqual(count, 0)
        bar = next(p for p in result if p.fine_label == "result_bar")
        self.assertFalse(bar.spatial_conflict)

    def test_div_bracket_excluded(self) -> None:
        preds = [
            self._node("main_5", row=0, y0=10.0, y1=30.0),
            self._node("div_bracket", row=1, y0=200.0, y1=250.0),
        ]
        result, count = _check_spatial_consistency(preds)
        self.assertEqual(count, 0)

    def test_single_row_no_conflict(self) -> None:
        preds = [
            self._node("main_1", row=0, y0=10.0, y1=30.0),
            self._node("main_2", row=0, y0=12.0, y1=32.0),
        ]
        result, count = _check_spatial_consistency(preds)
        self.assertEqual(count, 0)

    def test_spatial_conflict_count_in_spatial_meta(self) -> None:
        # Integration: conflict detected → spatial_meta.spatial_conflict_count > 0
        preds = [
            NodePrediction(
                fine_label="main_1", row_cluster_id=0, col_cluster_id=0,
                within_row_ord=0, within_col_ord=0,
                x0=0.0, y0=80.0, x1=10.0, y1=100.0, confidence=0.9,
            ),
            NodePrediction(
                fine_label="main_2", row_cluster_id=1, col_cluster_id=1,
                within_row_ord=0, within_col_ord=0,
                x0=0.0, y0=10.0, x1=10.0, y1=30.0, confidence=0.9,
            ),
            NodePrediction(
                fine_label="op_plus", row_cluster_id=0, col_cluster_id=2,
                within_row_ord=0, within_col_ord=0,
                x0=20.0, y0=80.0, x1=30.0, y1=100.0, confidence=0.9,
            ),
        ]
        out = assemble_json(preds, "add")
        self.assertNotIn("ood_reason", out)
        self.assertIn("spatial_conflict_count", out["spatial_meta"])
        self.assertGreater(out["spatial_meta"]["spatial_conflict_count"], 0)

    def test_no_conflict_spatial_meta_zero(self) -> None:
        # Clean scene → spatial_conflict_count=0
        preds = [
            NodePrediction(
                fine_label="main_5", row_cluster_id=0, col_cluster_id=0,
                within_row_ord=0, within_col_ord=0,
                x0=0.0, y0=10.0, x1=10.0, y1=30.0, confidence=0.9,
            ),
            NodePrediction(
                fine_label="op_plus", row_cluster_id=0, col_cluster_id=1,
                within_row_ord=0, within_col_ord=0,
                x0=20.0, y0=10.0, x1=30.0, y1=30.0, confidence=0.9,
            ),
        ]
        out = assemble_json(preds, "add")
        self.assertEqual(out["spatial_meta"]["spatial_conflict_count"], 0)


# ---------------------------------------------------------------------------
# W10-ASSEMBLER-RELAX: gate routing tests
# ---------------------------------------------------------------------------

class TestW10ClassifyGate(unittest.TestCase):
    """Unit tests for _classify_gate routing logic."""

    def test_zero_detections_routes_to_true_empty(self) -> None:
        route, reason = _classify_gate([])
        self.assertEqual(route, _GATE_OOD)
        self.assertEqual(reason, OOD_REASON_TRUE_EMPTY)

    def test_single_digit_routes_to_bare_digits(self) -> None:
        route, reason = _classify_gate([_pred("main_3", 0, 0)])
        self.assertEqual(route, _GATE_BARE_DIGITS)
        self.assertIsNone(reason)

    def test_single_non_digit_operator_routes_to_bare_digits(self) -> None:
        # W10 gate relaxation: a lone non-digit token is real structure the GNN
        # placed, so it now routes to bare_digits (was single_non_digit OOD).
        route, reason = _classify_gate([_pred("op_plus", 0, 0)])
        self.assertEqual(route, _GATE_BARE_DIGITS)
        self.assertIsNone(reason)

    def test_single_result_bar_routes_to_bare_digits(self) -> None:
        # standalone_bar shape: a lone result_bar now routes to a populated
        # bare_digits payload rather than empty OOD.
        route, reason = _classify_gate([_pred("result_bar", 0, 0)])
        self.assertEqual(route, _GATE_BARE_DIGITS)
        self.assertIsNone(reason)

    def test_single_div_bracket_routes_to_bare_digits(self) -> None:
        # standalone_bracket shape: a lone div_bracket now routes to bare_digits.
        route, reason = _classify_gate([_pred("div_bracket", 0, 0)])
        self.assertEqual(route, _GATE_BARE_DIGITS)
        self.assertIsNone(reason)

    def test_multiple_no_digits_with_structural_routes_to_bare_digits(self) -> None:
        # Grouped standalone marks (two result_bars, no digits): the structural
        # token flips this digitless scene to bare_digits, not OOD.
        preds = [_pred("result_bar", 0, 0), _pred("result_bar", 1, 0)]
        route, reason = _classify_gate(preds)
        self.assertEqual(route, _GATE_BARE_DIGITS)
        self.assertIsNone(reason)

    def test_multiple_no_digits_routes_to_no_digits(self) -> None:
        preds = [_pred("op_plus", 0, 0), _pred("op_minus", 0, 1)]
        route, reason = _classify_gate(preds)
        self.assertEqual(route, _GATE_OOD)
        self.assertEqual(reason, OOD_REASON_NO_DIGITS)

    def test_digits_no_structural_routes_to_bare_digits(self) -> None:
        preds = [_pred("main_1", 0, 0), _pred("main_2", 0, 1)]
        route, reason = _classify_gate(preds)
        self.assertEqual(route, _GATE_BARE_DIGITS)
        self.assertIsNone(reason)

    def test_digits_with_structural_routes_to_valid(self) -> None:
        preds = [_pred("main_1", 0, 0), _pred("op_plus", 0, 1)]
        route, reason = _classify_gate(preds)
        self.assertEqual(route, _GATE_VALID)
        self.assertIsNone(reason)


class TestW10BareDigitsAssemble(unittest.TestCase):
    """Integration tests for bare_digits equation_kind via assemble_json."""

    def _assert_bare_digits(self, out: dict) -> None:
        self.assertEqual(out["schema_version"], 1)
        self.assertEqual(out["equation_kind"], "bare_digits")
        self.assertNotIn("ood_reason", out)
        self.assertIn("rows", out)
        self.assertIn("slots", out)
        self.assertIn("detections", out)

    def test_bare_digits_kind_for_n_digits_1(self) -> None:
        """Single digit → bare_digits with 1 row entry."""
        out = assemble_json([_pred("main_3", 0, 0)], "add")
        self._assert_bare_digits(out)
        self.assertEqual(len(out["rows"]), 1)
        self.assertEqual(len(out["rows"][0]["tokens"]), 1)

    def test_bare_digits_kind_for_n_digits_2(self) -> None:
        """Two digits → bare_digits with populated rows/slots."""
        preds = [_pred("main_2", 0, 0), _pred("main_4", 0, 1)]
        out = assemble_json(preds, "add")
        self._assert_bare_digits(out)
        self.assertEqual(out["row_count"], 1)
        self.assertEqual(len(out["slots"]["main_digits"]), 2)

    def test_bare_digits_kind_for_n_digits_5(self) -> None:
        """Five digits across two rows → bare_digits."""
        preds = [
            _pred("main_1", 0, 0),
            _pred("main_2", 0, 1),
            _pred("main_3", 0, 2),
            _pred("main_4", 1, 0),
            _pred("main_5", 1, 1),
        ]
        out = assemble_json(preds, "add")
        self._assert_bare_digits(out)
        self.assertEqual(out["row_count"], 2)
        self.assertEqual(len(out["slots"]["main_digits"]), 5)

    def test_bare_digits_kind_for_n_digits_10(self) -> None:
        """Ten digits across three rows → bare_digits."""
        preds = (
            [_pred(f"main_{i}", 0, i) for i in range(4)]
            + [_pred(f"main_{i}", 1, i) for i in range(4)]
            + [_pred("main_8", 2, 0), _pred("main_9", 2, 1)]
        )
        out = assemble_json(preds, "add")
        self._assert_bare_digits(out)
        self.assertEqual(len(out["slots"]["main_digits"]), 10)

    def test_true_empty_for_zero_detections(self) -> None:
        """Zero detections → OOD true_empty."""
        out = assemble_json([], "add")
        self.assertEqual(out["equation_kind"], OOD_EQUATION_KIND)
        self.assertEqual(out["ood_reason"], OOD_REASON_TRUE_EMPTY)
        self.assertEqual(out["rows"], [])

    def test_no_digits_still_ood(self) -> None:
        """Operators only (no digits) → OOD no_digits."""
        preds = [_pred("op_plus", 0, 0), _pred("op_minus", 0, 1), _pred("op_times", 0, 2)]
        out = assemble_json(preds, "add")
        self.assertEqual(out["equation_kind"], OOD_EQUATION_KIND)
        self.assertEqual(out["ood_reason"], OOD_REASON_NO_DIGITS)

    def test_bare_digits_row_col_match_cluster_by_coordinate(self) -> None:
        """row/col IDs in bare_digits output match the NodePrediction cluster IDs
        (which are assigned by _cluster_by_coordinate in the graph builder)."""
        preds = [
            _pred("main_1", row_cluster_id=0, col_cluster_id=0,
                  x0=10.0, y0=10.0, x1=40.0, y1=40.0),
            _pred("main_2", row_cluster_id=0, col_cluster_id=1,
                  x0=50.0, y0=10.0, x1=80.0, y1=40.0),
            _pred("main_3", row_cluster_id=1, col_cluster_id=0,
                  x0=10.0, y0=60.0, x1=40.0, y1=90.0),
        ]
        out = assemble_json(preds, "add")
        self._assert_bare_digits(out)
        # Row 0 should have 2 tokens; row 1 should have 1 token.
        self.assertEqual(len(out["rows"][0]["tokens"]), 2)
        self.assertEqual(len(out["rows"][1]["tokens"]), 1)
        # grid_col in tokens must match original col_cluster_id.
        row0_tokens = out["rows"][0]["tokens"]
        self.assertEqual(row0_tokens[0]["grid_col"], 0)
        self.assertEqual(row0_tokens[1]["grid_col"], 1)

    def test_existing_equation_paths_unchanged(self) -> None:
        """Regression: addition / subtraction / multiplication / division scenes
        keep their recognized kind (passthrough, no relabel)."""
        # addition
        preds = [_pred("main_3", 0, 0), _pred("op_plus", 0, 1), _pred("main_5", 0, 2)]
        out = assemble_json(preds, "add")
        self.assertEqual(out["equation_kind"], "add")
        self.assertNotIn("ood_reason", out)
        self.assertGreater(len(out["rows"]), 0)

        # division (div_bracket counts as structural)
        preds2 = [_pred("main_8", 0, 0), _pred("div_bracket", 0, 1), _pred("main_4", 0, 2)]
        out2 = assemble_json(preds2, "divide")
        self.assertEqual(out2["equation_kind"], "divide")

        # subtraction
        preds3 = [_pred("main_9", 0, 0), _pred("op_minus", 0, 1), _pred("main_3", 0, 2)]
        out3 = assemble_json(preds3, "subtract")
        self.assertEqual(out3["equation_kind"], "subtract")


# ---------------------------------------------------------------------------
# W10 gate relaxation: structure-only scenes (standalone_bar / standalone_bracket)
# emit populated rows/slots instead of empty OOD, while genuine garbage is
# still rejected.
# ---------------------------------------------------------------------------

class TestW10StructureOnlyRelaxation(unittest.TestCase):
    """A lone or grouped structural token (no digits) now produces a populated
    bare_digits payload so the GNN row/col/fine_label stay scorable, but
    digitless, structure-less scenes (operators only, zero detections) are
    still rejected as OOD to preserve anti-hallucination."""

    def test_standalone_bar_emits_populated_bare_digits(self) -> None:
        # The standalone_bar generator shape: a single result_bar, no digits.
        out = assemble_json([_pred("result_bar", 0, 0)], "unknown")
        self.assertEqual(out["equation_kind"], "bare_digits")
        self.assertNotIn("ood_reason", out)
        self.assertEqual(len(out["rows"]), 1)
        self.assertEqual(len(out["rows"][0]["tokens"]), 1)
        self.assertEqual(out["rows"][0]["tokens"][0]["label"], "result_bar")
        # result_bar parses to role="structure" → lands in the structures slot.
        self.assertEqual(len(out["slots"]["structures"]), 1)
        self.assertEqual(out["slots"]["structures"][0]["label"], "result_bar")
        self.assertEqual(len(out["slots"]["main_digits"]), 0)
        self.assertEqual(len(out["detections"]), 1)

    def test_standalone_bracket_emits_populated_bare_digits(self) -> None:
        # The standalone_bracket generator shape: a single div_bracket, no digits.
        # div_bracket parses to role="unknown"; the _STRUCTURAL_FINE_LABELS
        # fallback keeps it in the structures slot rather than main_digits.
        out = assemble_json([_pred("div_bracket", 0, 0)], "unknown")
        self.assertEqual(out["equation_kind"], "bare_digits")
        self.assertNotIn("ood_reason", out)
        self.assertEqual(len(out["rows"]), 1)
        self.assertEqual(len(out["slots"]["structures"]), 1)
        self.assertEqual(out["slots"]["structures"][0]["label"], "div_bracket")
        self.assertEqual(len(out["slots"]["main_digits"]), 0)
        self.assertEqual(len(out["detections"]), 1)

    def test_grouped_structural_no_digits_emits_bare_digits(self) -> None:
        # n >= 2, no digits, but a structural token present (two result_bars on
        # separate rows) → bare_digits, NOT OOD.
        preds = [_pred("result_bar", 0, 0), _pred("result_bar", 1, 0)]
        out = assemble_json(preds, "unknown")
        self.assertEqual(out["equation_kind"], "bare_digits")
        self.assertNotIn("ood_reason", out)
        self.assertEqual(len(out["slots"]["structures"]), 2)
        self.assertEqual(len(out["detections"]), 2)

    def test_mixed_structural_no_digits_emits_bare_digits(self) -> None:
        # n >= 2, no digits, mixed structural tokens (bar + bracket) → bare_digits.
        preds = [_pred("result_bar", 0, 0), _pred("div_bracket", 1, 0)]
        out = assemble_json(preds, "unknown")
        self.assertEqual(out["equation_kind"], "bare_digits")
        self.assertNotIn("ood_reason", out)
        self.assertEqual(len(out["slots"]["structures"]), 2)

    def test_operator_only_no_structural_still_ood(self) -> None:
        # Anti-hallucination guard: n >= 2, no digits, no structural token
        # (operators only) STILL routes to OOD no_digits.
        preds = [_pred("op_plus", 0, 0), _pred("op_minus", 0, 1)]
        out = assemble_json(preds, "add")
        self.assertEqual(out["equation_kind"], OOD_EQUATION_KIND)
        self.assertEqual(out["ood_reason"], OOD_REASON_NO_DIGITS)
        self.assertEqual(out["rows"], [])

    def test_zero_detections_still_true_empty_ood(self) -> None:
        # Anti-hallucination guard: zero detections STILL routes to true_empty OOD.
        out = assemble_json([], "add")
        self.assertEqual(out["equation_kind"], OOD_EQUATION_KIND)
        self.assertEqual(out["ood_reason"], OOD_REASON_TRUE_EMPTY)
        self.assertEqual(out["rows"], [])


# ---------------------------------------------------------------------------
# Anti-hallucination guards: explicit task tests (3) + (4) — must STAY OOD.
# These guard against inventing structure, NOT against drawn math being wrong.
# ---------------------------------------------------------------------------

class TestAntiHallucinationGuardsStayOod(unittest.TestCase):
    def test_zero_detections_still_ood(self) -> None:
        # EXPLICIT TASK TEST (3): zero detections STILL OOD (true_empty).
        out = assemble_json([], "add")
        self.assertEqual(out["equation_kind"], OOD_EQUATION_KIND)
        self.assertEqual(out["ood_reason"], OOD_REASON_TRUE_EMPTY)
        self.assertEqual(out["rows"], [])
        self.assertEqual(out["slots"], [])
        self.assertEqual(out["spatial_meta"], {"empty": True})

    def test_operator_only_still_ood(self) -> None:
        # EXPLICIT TASK TEST (4): operator-only scribble (>=2 tokens, no digit,
        # no structural mark) STILL OOD (no_digits).
        preds = [_pred("op_plus", 0, 0), _pred("op_minus", 0, 1)]
        out = assemble_json(preds, "add")
        self.assertEqual(out["equation_kind"], OOD_EQUATION_KIND)
        self.assertEqual(out["ood_reason"], OOD_REASON_NO_DIGITS)
        self.assertEqual(out["rows"], [])
        self.assertEqual(out["slots"], [])

    def test_operator_only_three_ops_still_ood(self) -> None:
        # Three operators, still no digit/mark → OOD no_digits.
        preds = [_pred("op_plus", 0, 0), _pred("op_minus", 0, 1), _pred("op_times", 0, 2)]
        out = assemble_json(preds, "subtract")
        self.assertEqual(out["equation_kind"], OOD_EQUATION_KIND)
        self.assertEqual(out["ood_reason"], OOD_REASON_NO_DIGITS)


# ---------------------------------------------------------------------------
# detections[] field tests (Change 2)
# ---------------------------------------------------------------------------

class TestDetectionsFieldPresent(unittest.TestCase):
    """assemble_json always emits a detections[] field in both OOD and success paths."""

    def _valid_preds(self):
        return [
            NodePrediction(
                fine_label="main_5", row_cluster_id=0, col_cluster_id=0,
                within_row_ord=0, within_col_ord=0,
                x0=10.0, y0=10.0, x1=50.0, y1=50.0, confidence=0.9,
            ),
            NodePrediction(
                fine_label="op_plus", row_cluster_id=0, col_cluster_id=1,
                within_row_ord=0, within_col_ord=0,
                x0=60.0, y0=10.0, x1=100.0, y1=50.0, confidence=0.9,
            ),
            NodePrediction(
                fine_label="main_3", row_cluster_id=0, col_cluster_id=2,
                within_row_ord=0, within_col_ord=0,
                x0=110.0, y0=10.0, x1=150.0, y1=50.0, confidence=0.85,
            ),
        ]

    def test_detections_present_on_success_path(self):
        """Successful assembly includes detections[] list."""
        out = assemble_json(self._valid_preds(), "add")
        self.assertIn("detections", out)
        self.assertIsInstance(out["detections"], list)
        self.assertGreater(len(out["detections"]), 0)

    def test_detections_entry_has_required_keys(self):
        """Each detections[] entry has label, bbox, row, col, conf."""
        out = assemble_json(self._valid_preds(), "add")
        for entry in out["detections"]:
            self.assertIn("label", entry)
            self.assertIn("bbox", entry)
            self.assertIn("row", entry)
            self.assertIn("col", entry)
            self.assertIn("conf", entry)
            self.assertIsInstance(entry["bbox"], list)
            self.assertEqual(len(entry["bbox"]), 4)
            self.assertIsInstance(entry["row"], int)
            self.assertIsInstance(entry["col"], int)
            self.assertIsInstance(entry["conf"], float)

    def test_detections_present_on_ood_too_few_symbols(self):
        """OOD output (true_empty) still includes detections[]."""
        out = assemble_json([], "add")
        self.assertEqual(out["equation_kind"], "unknown")
        self.assertIn("detections", out)
        self.assertIsInstance(out["detections"], list)
        self.assertEqual(len(out["detections"]), 0)

    def test_detections_present_on_bare_digits_no_structural_token(self):
        """W10: digits-only scene routes to bare_digits (not OOD); detections[] still present."""
        preds = [
            NodePrediction(
                fine_label="main_5", row_cluster_id=0, col_cluster_id=0,
                within_row_ord=0, within_col_ord=0,
                x0=0.0, y0=0.0, x1=10.0, y1=10.0, confidence=0.8,
            ),
            NodePrediction(
                fine_label="main_3", row_cluster_id=0, col_cluster_id=1,
                within_row_ord=0, within_col_ord=0,
                x0=20.0, y0=0.0, x1=30.0, y1=10.0, confidence=0.8,
            ),
            NodePrediction(
                fine_label="main_7", row_cluster_id=1, col_cluster_id=0,
                within_row_ord=0, within_col_ord=0,
                x0=0.0, y0=50.0, x1=10.0, y1=60.0, confidence=0.8,
            ),
        ]
        out = assemble_json(preds, "add")
        self.assertEqual(out["equation_kind"], "bare_digits")
        self.assertIn("detections", out)
        self.assertEqual(len(out["detections"]), 3)

    def test_detections_row_col_match_node_prediction(self):
        """detections[] entries carry row/col from NodePrediction cluster IDs."""
        preds = self._valid_preds()
        out = assemble_json(preds, "add")
        first = next(d for d in out["detections"] if d["label"] == "main_5")
        self.assertEqual(first["row"], 0)
        self.assertEqual(first["col"], 0)


# ---------------------------------------------------------------------------
# iter7 Integration 2: OOD path smoke tests (Workstream D)
# ---------------------------------------------------------------------------

class TestIter7OodAssemblePaths(unittest.TestCase):
    """Verify that the assembler emits the correct OOD shape for each ood_reason."""

    def _assert_ood(self, preds, expected_ood_reason: str) -> None:
        result = assemble_json(preds, "add")
        self.assertEqual(result["equation_kind"], "unknown",
                         f"Expected equation_kind='unknown', got {result['equation_kind']!r}")
        self.assertEqual(result.get("ood_reason"), expected_ood_reason,
                         f"Expected ood_reason={expected_ood_reason!r}, got {result.get('ood_reason')!r}")
        self.assertEqual(result["rows"], [],
                         f"Expected empty rows, got {result['rows']!r}")
        self.assertTrue(result["spatial_meta"].get("empty"),
                        f"Expected spatial_meta.empty=True")

    def test_true_empty_for_empty_list(self) -> None:
        """W10: empty prediction list → true_empty (was too_few_symbols)."""
        self._assert_ood([], OOD_REASON_TRUE_EMPTY)

    def test_single_non_digit_operator_now_bare_digits(self) -> None:
        """W10 gate relaxation: a lone non-digit token (operator) now routes to
        a populated bare_digits payload, not OOD."""
        preds = [_pred("op_plus", 0, 0)]
        result = assemble_json(preds, "add")
        self.assertEqual(result["equation_kind"], "bare_digits")
        self.assertNotIn("ood_reason", result)
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(len(result["detections"]), 1)

    def test_no_structural_token_digits_only_now_bare_digits(self) -> None:
        """W10: digits-only (no structural token) → bare_digits, NOT OOD."""
        preds = [_pred("main_5", 0, 0), _pred("main_3", 0, 1), _pred("main_7", 1, 0)]
        result = assemble_json(preds, "add")
        self.assertEqual(result["equation_kind"], "bare_digits")
        self.assertNotIn("ood_reason", result)
        self.assertGreater(len(result["rows"]), 0)

    def test_no_structural_token_digit_plus_carry_now_bare_digits(self) -> None:
        """W10: main digits + carry, no structural token → bare_digits, NOT OOD.
        Recognize-as-drawn: the carry (col 0 has a main below it) is retained."""
        preds = [
            _pred("main_5", 0, 0),
            _pred("main_3", 0, 1),
            _pred("carry_1", -1, 0),
        ]
        result = assemble_json(preds, "add")
        self.assertEqual(result["equation_kind"], "bare_digits")
        self.assertNotIn("ood_reason", result)
        # The carry is retained, not deleted.
        self.assertIn("carry_1", [d["label"] for d in result["detections"]])


# ---------------------------------------------------------------------------
# Recognize-as-drawn: equation_kind is the recognized kind; operator tokens
# NEVER flip it. The heuristic knob computes diagnostics only.
# ---------------------------------------------------------------------------

class TestRecognizeAsDrawnKind(unittest.TestCase):
    """equation_kind is whatever the GNN eq_type head recognized (passed in).
    The token-derived kind is recorded as a diagnostic (eq_kind_overridden_from
    / eq_kind_override_reason) but never mutates equation_kind. This is the +/-
    flip fix: a misread '+' detected as op_minus does NOT convert a drawn
    addition into a subtraction."""

    def test_addition_with_op_minus_present_does_not_flip(self) -> None:
        # EXPLICIT TASK TEST (2): an addition scene with an op_minus present does
        # NOT flip equation_kind to subtract — even with the diagnostic knob ON.
        preds = [
            _pred("main_5", 0, 0),
            _pred("op_minus", 0, 1, conf=0.85),
            _pred("main_2", 0, 2),
            _pred("result_bar", 1, 0),
        ]
        out = assemble_json(preds, "add", heuristic_enabled=True)
        self.assertEqual(out["equation_kind"], "add")
        # The disagreement is recorded as a diagnostic, not acted on.
        self.assertEqual(out["eq_kind_overridden_from"], "add")
        self.assertEqual(out["eq_kind_override_reason"], "op_minus_majority")

    def test_addition_with_op_minus_default_knob_off_no_diagnostic(self) -> None:
        # With the default (heuristic_enabled=False) the diagnostic disagreement
        # fields are not computed, but equation_kind is still the recognized kind.
        preds = [
            _pred("main_5", 0, 0),
            _pred("op_minus", 0, 1, conf=0.85),
            _pred("main_2", 0, 2),
            _pred("result_bar", 1, 0),
        ]
        out = assemble_json(preds, "add")
        self.assertEqual(out["equation_kind"], "add")
        self.assertIsNone(out["eq_kind_overridden_from"])
        self.assertIsNone(out["eq_kind_override_reason"])

    def test_div_bracket_present_does_not_override_add_kind(self) -> None:
        # head="add", div_bracket conf 0.45 (above threshold). The token signal
        # says divide, but equation_kind stays the recognized "add"; the
        # disagreement is a diagnostic only.
        preds = [
            _pred("main_3", 0, 0),
            _pred("div_bracket", 0, 1, conf=0.45),
            _pred("main_7", 0, 2),
        ]
        out = assemble_json(preds, "add", heuristic_enabled=True)
        self.assertEqual(out["equation_kind"], "add")
        self.assertEqual(out["eq_kind_overridden_from"], "add")
        self.assertEqual(out["eq_kind_override_reason"], "div_signal")
        self.assertAlmostEqual(out["div_bracket_max_conf"], 0.45, places=4)

    def test_op_times_present_does_not_override_add_kind(self) -> None:
        preds = [
            _pred("main_3", 0, 0),
            _pred("op_times", 0, 1, conf=0.75),
            _pred("main_4", 0, 2),
            _pred("result_bar", 1, 0),
        ]
        out = assemble_json(preds, "add", heuristic_enabled=True)
        self.assertEqual(out["equation_kind"], "add")
        self.assertEqual(out["eq_kind_overridden_from"], "add")
        self.assertEqual(out["eq_kind_override_reason"], "op_times_present")

    def test_tokens_agree_with_head_no_diagnostic(self) -> None:
        # head="subtract" + op_minus majority: tokens agree → no diagnostic.
        preds = [
            _pred("main_9", 0, 0),
            _pred("op_minus", 0, 1, conf=0.9),
            _pred("main_3", 0, 2),
            _pred("result_bar", 1, 0),
        ]
        out = assemble_json(preds, "subtract", heuristic_enabled=True)
        self.assertEqual(out["equation_kind"], "subtract")
        self.assertIsNone(out["eq_kind_overridden_from"])
        self.assertIsNone(out["eq_kind_override_reason"])

    def test_apply_heuristic_overrides_never_mutates_kind(self) -> None:
        # Unit-level: the helper always returns the head kind as its first value.
        preds = [
            _pred("main_5", 0, 0),
            _pred("op_minus", 0, 1),
            _pred("main_2", 0, 2),
        ]
        kind, overridden_from, reason, _conf, _counts = _apply_heuristic_overrides(
            preds, "add"
        )
        self.assertEqual(kind, "add")  # head kind, NOT "subtract"
        self.assertEqual(overridden_from, "add")
        self.assertEqual(reason, "op_minus_majority")

    def test_heuristic_disabled_no_diagnostic_fields(self) -> None:
        """heuristic_enabled=False: diagnostic disagreement fields stay None even
        with a div_bracket present; equation_kind stays the recognized kind."""
        preds = [
            _pred("main_3", 0, 0),
            _pred("div_bracket", 0, 1, conf=0.80),
            _pred("main_7", 0, 2),
        ]
        out = assemble_json(preds, "add", heuristic_enabled=False)
        self.assertEqual(out["equation_kind"], "add")
        self.assertIsNone(out["eq_kind_overridden_from"])
        self.assertIsNone(out["eq_kind_override_reason"])

    def test_observability_fields_always_present_on_success(self) -> None:
        """eq_kind_overridden_from, eq_kind_override_reason, div_bracket_max_conf,
        op_counts, kind_low_conf_reason in success output."""
        preds = [_pred("main_3", 0, 0), _pred("op_plus", 0, 1)]
        out = assemble_json(preds, "add")
        self.assertIn("eq_kind_overridden_from", out)
        self.assertIn("eq_kind_override_reason", out)
        self.assertIn("div_bracket_max_conf", out)
        self.assertIn("op_counts", out)
        self.assertIn("kind_low_conf_reason", out)
        self.assertIsInstance(out["op_counts"], dict)
        for key in ("op_plus", "op_minus", "op_times", "op_divide"):
            self.assertIn(key, out["op_counts"])

    def test_observability_fields_present_on_ood(self) -> None:
        """Observability fields are also included on OOD paths."""
        out = assemble_json([], "add")
        self.assertIn("div_bracket_max_conf", out)
        self.assertIn("op_counts", out)
        self.assertEqual(out["div_bracket_max_conf"], 0.0)
        self.assertEqual(out["op_counts"]["op_plus"], 0)

    def test_op_counts_correct_values(self) -> None:
        """op_counts reflects actual token counts in predictions (non-overlapping x spans)."""
        preds = [
            _pred("main_1", 0, 0),
            # Two op_plus at non-overlapping x positions — dedup keeps both.
            _pred("op_plus", 0, 1, x0=5.0, x1=15.0),
            _pred("op_plus", 0, 2, x0=25.0, x1=35.0),
            _pred("op_minus", 0, 3, x0=45.0, x1=55.0),
            _pred("result_bar", 1, 0),
        ]
        out = assemble_json(preds, "add")
        self.assertEqual(out["op_counts"]["op_plus"], 2)
        self.assertEqual(out["op_counts"]["op_minus"], 1)
        self.assertEqual(out["op_counts"]["op_times"], 0)
        self.assertEqual(out["op_counts"]["op_divide"], 0)


class TestNodePredictionGivenFlag(unittest.TestCase):
    """Foundation 1: given flag plumbing through NodePrediction and assembled tokens."""

    def test_node_prediction_given_defaults_false(self) -> None:
        p = _pred("main_3", 0, 0)
        self.assertFalse(p.given)

    def test_node_prediction_given_true_roundtrips(self) -> None:
        p = NodePrediction(
            fine_label="main_3",
            row_cluster_id=0,
            col_cluster_id=0,
            within_row_ord=0,
            within_col_ord=0,
            x0=0.0, y0=0.0, x1=10.0, y1=10.0,
            confidence=0.95,
            given=True,
        )
        self.assertTrue(p.given)

    def test_assembled_token_carries_given_false_by_default(self) -> None:
        out = assemble_json(_valid_pair(), "add")
        # all tokens in rows must have given=False when no given flag was set
        for row in out["rows"]:
            for token in row["tokens"]:
                self.assertIn("given", token)
                self.assertFalse(token["given"])

    def test_assembled_token_carries_given_true(self) -> None:
        # Build a minimal valid scene where the digit is marked given=True.
        digit = NodePrediction(
            fine_label="main_3",
            row_cluster_id=0,
            col_cluster_id=0,
            within_row_ord=0,
            within_col_ord=0,
            x0=0.0, y0=0.0, x1=10.0, y1=10.0,
            confidence=0.95,
            given=True,
        )
        op = NodePrediction(
            fine_label="op_plus",
            row_cluster_id=0,
            col_cluster_id=1,
            within_row_ord=0,
            within_col_ord=0,
            x0=20.0, y0=0.0, x1=30.0, y1=10.0,
            confidence=0.95,
            given=False,
        )
        out = assemble_json([digit, op], "add")
        tokens = out["rows"][0]["tokens"]
        # The digit token (col 0) should carry given=True
        digit_token = next(t for t in tokens if t["label"] == "main_3")
        op_token = next(t for t in tokens if t["label"] == "op_plus")
        self.assertTrue(digit_token["given"])
        self.assertFalse(op_token["given"])


if __name__ == "__main__":
    unittest.main()
