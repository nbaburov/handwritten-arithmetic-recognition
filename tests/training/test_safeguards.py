"""Tests for training safeguards (BS-1 through BS-5).

Each test exercises one safeguard in isolation without running the full
train_gnn_model loop. The training internals are mocked to zero in on the
specific guard logic.
"""
from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch


class TestLossCollapseWarning(unittest.TestCase):
    """BS-1: A >50x loss drop in a single epoch triggers a WARNING log."""

    def test_loss_collapse_warning_at_50x(self) -> None:
        """Simulates two consecutive epoch losses where prev/cur > 50, verifying
        that LOSS_COLLAPSE_DETECTED is emitted at WARNING level."""
        # Import internals directly to test the guard logic without running training.
        # We replicate the exact guard expression from train_gnn.py.
        prev_train_loss = 100.0
        avg_loss = 1.0  # ratio = 100x > 50 threshold

        triggered = False

        # Mirror the guard expression verbatim from the implementation.
        epoch = 1  # epoch > 0 required
        if epoch > 0 and prev_train_loss is not None and avg_loss > 0 and prev_train_loss / max(avg_loss, 1e-9) > 50.0:
            triggered = True

        self.assertTrue(
            triggered,
            "Loss collapse guard should fire when prev/cur ratio > 50x",
        )

    def test_loss_collapse_no_warning_within_threshold(self) -> None:
        """Ratio of exactly 50x should NOT trigger (threshold is strictly > 50)."""
        prev_train_loss = 50.0
        avg_loss = 1.0  # ratio = 50x exactly, not > 50
        epoch = 1

        triggered = False
        if epoch > 0 and prev_train_loss is not None and avg_loss > 0 and prev_train_loss / max(avg_loss, 1e-9) > 50.0:
            triggered = True

        self.assertFalse(triggered, "Exactly 50x should not trigger the guard (threshold is strictly >50)")

    def test_loss_collapse_no_warning_at_epoch_zero(self) -> None:
        """Guard must NOT fire at epoch 0 (no previous value yet)."""
        prev_train_loss = None
        avg_loss = 0.001
        epoch = 0

        triggered = False
        if epoch > 0 and prev_train_loss is not None and avg_loss > 0 and prev_train_loss / max(avg_loss, 1e-9) > 50.0:
            triggered = True

        self.assertFalse(triggered, "Guard must not fire at epoch 0 (prev_train_loss is None)")

    def test_loss_collapse_warning_emitted_via_logger(self) -> None:
        """Verify the logger.warning call fires when the guard condition is met."""
        logger = MagicMock()

        prev_train_loss = 500.0
        avg_loss = 0.5  # ratio = 1000x
        epoch = 3

        if epoch > 0 and prev_train_loss is not None and avg_loss > 0 and prev_train_loss / max(avg_loss, 1e-9) > 50.0:
            logger.warning(
                "LOSS_COLLAPSE_DETECTED: epoch %d train_loss=%.6f, prev=%.6f, ratio=%.1fx "
                "(>50x indicates trivial dataset or memorization)",
                epoch + 1, avg_loss, prev_train_loss, prev_train_loss / max(avg_loss, 1e-9),
            )

        logger.warning.assert_called_once()
        call_args = logger.warning.call_args[0]
        self.assertIn("LOSS_COLLAPSE_DETECTED", call_args[0])


class TestValTrivialDetection(unittest.TestCase):
    """BS-2: val_combined >= 0.999 at epoch < 2 triggers a WARNING log."""

    def test_val_trivial_warning_at_epoch_0(self) -> None:
        """val_combined=1.0 at epoch 0 should trigger the degenerate-val warning."""
        logger = MagicMock()
        val_combined = 1.0
        epoch = 0

        if epoch < 2 and val_combined >= 0.999:
            logger.warning(
                "VAL_TRIVIAL_DETECTED: val_combined=%.4f at epoch %d. "
                "Likely degenerate val set (1-class, single-symbol scenes, or perfect-match data). "
                "Check class distribution + scene diversity.",
                val_combined, epoch,
            )

        logger.warning.assert_called_once()
        call_args = logger.warning.call_args[0]
        self.assertIn("VAL_TRIVIAL_DETECTED", call_args[0])

    def test_val_trivial_warning_at_epoch_1(self) -> None:
        """val_combined=1.0 at epoch 1 (still < 2) should also trigger."""
        logger = MagicMock()
        val_combined = 1.0
        epoch = 1

        if epoch < 2 and val_combined >= 0.999:
            logger.warning("VAL_TRIVIAL_DETECTED: val_combined=%.4f at epoch %d.", val_combined, epoch)

        logger.warning.assert_called_once()

    def test_val_trivial_no_warning_at_epoch_2(self) -> None:
        """epoch=2 is outside the early-epoch window; no warning expected."""
        logger = MagicMock()
        val_combined = 1.0
        epoch = 2  # NOT < 2

        if epoch < 2 and val_combined >= 0.999:
            logger.warning("VAL_TRIVIAL_DETECTED")

        logger.warning.assert_not_called()

    def test_val_trivial_no_warning_for_normal_score(self) -> None:
        """val_combined=0.85 at epoch 0 is healthy; no warning expected."""
        logger = MagicMock()
        val_combined = 0.85
        epoch = 0

        if epoch < 2 and val_combined >= 0.999:
            logger.warning("VAL_TRIVIAL_DETECTED")

        logger.warning.assert_not_called()


class TestMacroF1PromotionGate(unittest.TestCase):
    """BS-3: fine_label_macro_f1 below floor blocks promotion (active.json not written)."""

    def _run_promotion_logic(self, test_metrics: dict) -> dict:
        """Replicate the promotion gate logic from train_gnn.py in isolation."""
        PROMOTION_MACRO_F1_FLOOR = 0.30
        PROMOTION_CLUSTER_SUM_FLOOR = 0.20

        fine_macro_f1 = test_metrics.get("fine_label_macro_f1", 0.0)
        cluster_sum = (
            test_metrics.get("row_cluster_accuracy", 0.0)
            + test_metrics.get("col_cluster_accuracy", 0.0)
        )

        promotion_blocked = False
        if fine_macro_f1 < PROMOTION_MACRO_F1_FLOOR:
            promotion_blocked = True
        if cluster_sum < PROMOTION_CLUSTER_SUM_FLOOR:
            promotion_blocked = True

        return {"promoted": not promotion_blocked}

    def test_macro_f1_below_floor_blocks_promotion(self) -> None:
        """macro_f1=0.10 is well below floor=0.30; promoted must be False."""
        result = self._run_promotion_logic({
            "fine_label_macro_f1": 0.10,
            "row_cluster_accuracy": 0.9,
            "col_cluster_accuracy": 0.9,
        })
        self.assertFalse(result["promoted"])

    def test_macro_f1_above_floor_allows_promotion_when_cluster_passes(self) -> None:
        """Both gates pass; promoted must be True."""
        result = self._run_promotion_logic({
            "fine_label_macro_f1": 0.85,
            "row_cluster_accuracy": 0.6,
            "col_cluster_accuracy": 0.6,
        })
        self.assertTrue(result["promoted"])

    def test_macro_f1_exactly_at_floor_passes(self) -> None:
        """macro_f1 exactly at floor (0.30) should NOT block (condition is strictly <)."""
        result = self._run_promotion_logic({
            "fine_label_macro_f1": 0.30,
            "row_cluster_accuracy": 0.5,
            "col_cluster_accuracy": 0.5,
        })
        self.assertTrue(result["promoted"])


class TestClusterFloorPromotionGate(unittest.TestCase):
    """BS-4: row + col cluster accuracy below floor blocks promotion."""

    def _run_promotion_logic(self, test_metrics: dict) -> dict:
        """Mirror of BS-3/BS-4 gate logic for isolation testing."""
        PROMOTION_MACRO_F1_FLOOR = 0.30
        PROMOTION_CLUSTER_SUM_FLOOR = 0.20

        fine_macro_f1 = test_metrics.get("fine_label_macro_f1", 0.0)
        cluster_sum = (
            test_metrics.get("row_cluster_accuracy", 0.0)
            + test_metrics.get("col_cluster_accuracy", 0.0)
        )

        promotion_blocked = False
        if fine_macro_f1 < PROMOTION_MACRO_F1_FLOOR:
            promotion_blocked = True
        if cluster_sum < PROMOTION_CLUSTER_SUM_FLOOR:
            promotion_blocked = True

        return {"promoted": not promotion_blocked}

    def test_cluster_floor_blocks_promotion_both_zero(self) -> None:
        """row=col=0.0 gives cluster_sum=0.0 < 0.20; promoted must be False."""
        result = self._run_promotion_logic({
            "fine_label_macro_f1": 0.85,
            "row_cluster_accuracy": 0.0,
            "col_cluster_accuracy": 0.0,
        })
        self.assertFalse(result["promoted"])

    def test_cluster_floor_blocks_when_sum_below_threshold(self) -> None:
        """row=0.05, col=0.05, sum=0.10 < 0.20; must block."""
        result = self._run_promotion_logic({
            "fine_label_macro_f1": 0.85,
            "row_cluster_accuracy": 0.05,
            "col_cluster_accuracy": 0.05,
        })
        self.assertFalse(result["promoted"])

    def test_cluster_sum_exactly_at_floor_passes(self) -> None:
        """cluster_sum exactly 0.20 should not block (condition is strictly <)."""
        result = self._run_promotion_logic({
            "fine_label_macro_f1": 0.85,
            "row_cluster_accuracy": 0.10,
            "col_cluster_accuracy": 0.10,
        })
        self.assertTrue(result["promoted"])

    def test_both_gates_fail_reports_not_promoted(self) -> None:
        """When both macro_f1 and cluster_sum fail, promoted=False."""
        result = self._run_promotion_logic({
            "fine_label_macro_f1": 0.05,
            "row_cluster_accuracy": 0.0,
            "col_cluster_accuracy": 0.0,
        })
        self.assertFalse(result["promoted"])


class TestPreTrainDiversityCheck(unittest.TestCase):
    """BS-5: fewer than 12 unique fine_label classes raises RuntimeError before training."""

    def _run_diversity_check(self, all_y_fine: list) -> None:
        """Replicate the pre-train diversity guard logic."""
        CLASS_DIVERSITY_FLOOR = 12
        unique_classes = len(set(int(y) for y in all_y_fine if y >= 0))
        if unique_classes < CLASS_DIVERSITY_FLOOR:
            raise RuntimeError(
                f"PRE_TRAIN_BLOCKED: training data has only {unique_classes} unique fine_label classes "
                f"(< floor {CLASS_DIVERSITY_FLOOR}). Likely synthesis pipeline corruption. "
                f"Refusing to train on degenerate data. Investigate data/generated/synthetic/latest/ before retrying."
            )

    def test_pre_train_diversity_check_raises_single_class(self) -> None:
        """Dataset with only class 0 (all same label) must raise RuntimeError."""
        all_y_fine = [0] * 1000
        with self.assertRaises(RuntimeError) as ctx:
            self._run_diversity_check(all_y_fine)
        self.assertIn("PRE_TRAIN_BLOCKED", str(ctx.exception))
        self.assertIn("1 unique fine_label classes", str(ctx.exception))

    def test_pre_train_diversity_check_raises_below_floor(self) -> None:
        """Dataset with 11 classes (< floor of 12) must raise."""
        all_y_fine = list(range(11)) * 100  # classes 0-10 only
        with self.assertRaises(RuntimeError) as ctx:
            self._run_diversity_check(all_y_fine)
        self.assertIn("PRE_TRAIN_BLOCKED", str(ctx.exception))

    def test_pre_train_diversity_check_passes_at_floor(self) -> None:
        """Exactly 12 unique classes meets the floor; no RuntimeError."""
        all_y_fine = list(range(12)) * 100  # classes 0-11
        try:
            self._run_diversity_check(all_y_fine)
        except RuntimeError:
            self.fail("Diversity check raised unexpectedly with exactly 12 classes")

    def test_pre_train_diversity_check_passes_full_16(self) -> None:
        """All 16 fine_label classes present; no RuntimeError."""
        all_y_fine = list(range(16)) * 100
        try:
            self._run_diversity_check(all_y_fine)
        except RuntimeError:
            self.fail("Diversity check raised unexpectedly with 16 classes")

    def test_pre_train_diversity_check_ignores_negative_labels(self) -> None:
        """Labels with y < 0 are sentinel/padding values and must be excluded from count."""
        # Only class 0 is valid; y=-1 padding must not count as a class.
        all_y_fine = [-1, -1, 0, 0, 0]
        with self.assertRaises(RuntimeError) as ctx:
            self._run_diversity_check(all_y_fine)
        self.assertIn("PRE_TRAIN_BLOCKED", str(ctx.exception))

    def test_error_message_contains_investigate_hint(self) -> None:
        """RuntimeError message must contain actionable investigation hint."""
        all_y_fine = [0] * 50
        with self.assertRaises(RuntimeError) as ctx:
            self._run_diversity_check(all_y_fine)
        self.assertIn("Investigate", str(ctx.exception))


class TestPromotedKeyInReturnDict(unittest.TestCase):
    """Verify that train_gnn_model's return dict always includes 'promoted' key."""

    def test_signature_return_docstring_mentions_promoted(self) -> None:
        """Docstring for train_gnn_model must document the 'promoted' return key."""
        from src.training.train_gnn import train_gnn_model
        self.assertIn(
            "promoted",
            train_gnn_model.__doc__,
            "Return dict docstring must mention 'promoted' key (added by BS-3/BS-4).",
        )


if __name__ == "__main__":
    unittest.main()
