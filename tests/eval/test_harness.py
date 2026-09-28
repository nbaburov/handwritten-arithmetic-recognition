"""Tests for src.eval.harness — F2 harness core.

TDD: tests document behaviour before implementation details.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.eval.harness import (
    _ASSEMBLER_TO_LABEL,
    AggregateMetric,
    RunRecord,
    SampleMetric,
    aggregate_by,
    compute_aggregate,
    compute_per_sample_metrics,
    wilson_interval,
)
from src.core.run_config import YoloInferenceConfig
from src.eval.labels import EQUATION_KINDS, SampleLabel, SymbolLabel
from src.parsing.assemble import OOD_REASON_NO_STRUCTURAL_TOKEN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample_label(
    equation_kind: str = "addition",
    symbols=None,
    sample: str = "test-sample",
) -> SampleLabel:
    """Build a minimal SampleLabel for testing."""
    return SampleLabel(
        schema_version=1,
        sample=sample,
        equation_kind=equation_kind,
        symbols=symbols,
    )


def _make_sample_metric(
    *,
    equation_kind_ok: bool = True,
    ood_triggered: bool = False,
    ood_honest: bool = False,
    matched: int | None = None,
    label_correct: int | None = None,
    recall: float | None = None,
    mean_iou_matched: float | None = None,
    row_acc: float | None = None,
    col_acc: float | None = None,
    n_pred: int | None = 0,
    n_gt: int | None = None,
    wall_ms: float = 10.0,
    name: str = "test-sample",
) -> SampleMetric:
    return SampleMetric(
        name=name,
        equation_kind_pred="add",
        equation_kind_gt="addition",
        equation_kind_ok=equation_kind_ok,
        ood_triggered=ood_triggered,
        ood_honest=ood_honest,
        ood_reason=None,
        n_pred=n_pred,
        n_gt=n_gt,
        matched=matched,
        label_correct=label_correct,
        recall=recall,
        mean_iou_matched=mean_iou_matched,
        row_acc=row_acc,
        col_acc=col_acc,
        wall_ms=wall_ms,
    )


# ---------------------------------------------------------------------------
# compute_per_sample_metrics tests
# ---------------------------------------------------------------------------

class TestComputePerSampleMetricsCorrectEqKind(unittest.TestCase):
    def test_compute_per_sample_metrics_correct_eq_kind(self):
        """pred equation_kind='add' maps to 'addition' == label → equation_kind_ok=True."""
        pred = {"equation_kind": "add"}
        label = _make_sample_label(equation_kind="addition", symbols=None)
        metric = compute_per_sample_metrics(pred, label, wall_ms=5.0)
        self.assertTrue(metric.equation_kind_ok)
        self.assertEqual(metric.name, "test-sample")
        self.assertAlmostEqual(metric.wall_ms, 5.0)


class TestComputePerSampleMetricsOodHonest(unittest.TestCase):
    def test_compute_per_sample_metrics_ood_honest_when_real_eq_exists(self):
        """OOD fires on real equation kind → equation_kind_ok=False AND ood_honest=True."""
        pred = {"equation_kind": "unknown", "ood_reason": OOD_REASON_NO_STRUCTURAL_TOKEN}
        label = _make_sample_label(equation_kind="addition", symbols=None)
        metric = compute_per_sample_metrics(pred, label, wall_ms=3.0)
        self.assertFalse(metric.equation_kind_ok)
        self.assertTrue(metric.ood_triggered)
        self.assertTrue(metric.ood_honest)


class TestComputePerSampleMetricsNoSymbolsNone(unittest.TestCase):
    def test_compute_per_sample_metrics_no_symbols_yields_none_spatial_metrics(self):
        """When label.symbols is None all spatial metrics are None."""
        pred = {"equation_kind": "add", "tokens": []}
        label = _make_sample_label(symbols=None)
        metric = compute_per_sample_metrics(pred, label, wall_ms=2.0)
        self.assertIsNone(metric.n_gt)
        self.assertIsNone(metric.matched)
        self.assertIsNone(metric.recall)
        self.assertIsNone(metric.mean_iou_matched)
        self.assertIsNone(metric.label_correct)


class TestComputePerSampleMetricsWithSymbols(unittest.TestCase):
    def test_compute_per_sample_metrics_with_symbols_uses_greedy_match_scene(self):
        """One GT symbol + one matching pred token → matched=1, label_correct=1."""
        sym = SymbolLabel(flattened="main_3", bbox_px=(10.0, 20.0, 50.0, 60.0))
        label = _make_sample_label(equation_kind="addition", symbols=(sym,))
        pred = {
            "equation_kind": "add",
            "tokens": [
                {
                    "flattened": "main_3",
                    "bbox_px": [15.0, 25.0, 45.0, 55.0],
                }
            ],
        }
        metric = compute_per_sample_metrics(pred, label, wall_ms=8.0)
        self.assertEqual(metric.n_gt, 1)
        self.assertEqual(metric.matched, 1)
        self.assertEqual(metric.label_correct, 1)
        self.assertIsNotNone(metric.recall)
        self.assertAlmostEqual(metric.recall, 1.0)


class TestComputePerSampleMetricsHandlesOodOutput(unittest.TestCase):
    def test_compute_per_sample_metrics_handles_ood_output_without_crashing(self):
        """OOD shape output (empty rows/slots/spatial_meta) does not crash."""
        pred = {
            "equation_kind": "unknown",
            "ood_reason": OOD_REASON_NO_STRUCTURAL_TOKEN,
            "rows": [],
            "slots": [],
            "spatial_meta": {"empty": True},
            "tokens": [],
        }
        label = _make_sample_label(equation_kind="addition", symbols=None)
        # Should not raise
        metric = compute_per_sample_metrics(pred, label, wall_ms=1.0)
        self.assertIsInstance(metric, SampleMetric)
        self.assertFalse(metric.equation_kind_ok)
        self.assertTrue(metric.ood_triggered)


# ---------------------------------------------------------------------------
# compute_aggregate tests
# ---------------------------------------------------------------------------

class TestComputeAggregateZeroSamplesRaises(unittest.TestCase):
    def test_compute_aggregate_zero_samples_raises(self):
        """Empty sample list raises ValueError."""
        with self.assertRaises(ValueError):
            compute_aggregate([])


class TestComputeAggregateAllCorrect(unittest.TestCase):
    def test_compute_aggregate_all_correct(self):
        """3 correct samples with no OOD → equation_kind_acc=1.0, ood_rate=0.0."""
        samples = [
            _make_sample_metric(equation_kind_ok=True, ood_triggered=False),
            _make_sample_metric(equation_kind_ok=True, ood_triggered=False),
            _make_sample_metric(equation_kind_ok=True, ood_triggered=False),
        ]
        agg = compute_aggregate(samples)
        self.assertEqual(agg.n_samples, 3)
        self.assertAlmostEqual(agg.equation_kind_acc, 1.0)
        self.assertAlmostEqual(agg.ood_rate, 0.0)
        self.assertAlmostEqual(agg.ood_honest_rate, 0.0)


class TestComputeAggregateSkipsNoneForIouMean(unittest.TestCase):
    def test_compute_aggregate_skips_none_for_iou_mean(self):
        """Mean IoU computed only over non-None samples; all-None gives None."""
        # Some with values, some without
        samples = [
            _make_sample_metric(mean_iou_matched=0.8),
            _make_sample_metric(mean_iou_matched=0.6),
            _make_sample_metric(mean_iou_matched=None),
        ]
        agg = compute_aggregate(samples)
        self.assertIsNotNone(agg.mean_iou_matched)
        self.assertAlmostEqual(agg.mean_iou_matched, 0.7, places=5)

        # All None
        samples_none = [
            _make_sample_metric(mean_iou_matched=None),
            _make_sample_metric(mean_iou_matched=None),
        ]
        agg_none = compute_aggregate(samples_none)
        self.assertIsNone(agg_none.mean_iou_matched)


# ---------------------------------------------------------------------------
# run_eval tests (mocked)
# ---------------------------------------------------------------------------

class TestRunEvalUsesAutoLoadNotLoad(unittest.TestCase):
    def test_run_eval_uses_auto_load_not_load(self):
        """run_eval calls auto_load(project_root), never load()."""
        from src.eval.harness import run_eval

        mock_session = MagicMock()
        mock_session.is_loaded.return_value = True
        mock_session._yolo_inference_config = YoloInferenceConfig()

        # predict returns a minimal valid output
        mock_session.predict.return_value = {
            "equation_kind": "add",
            "tokens": [],
        }

        import numpy as np

        fake_root = Path("/fake/project")

        # Patch away heavy imports and filesystem calls inside run_eval
        with patch("src.eval.harness.DataPrepConfig") as mock_cfg_cls, \
             patch("src.eval.harness.InferenceSession", return_value=mock_session), \
             patch("src.eval.harness.resolve_gnn_best_pt", return_value=Path("/fake/gnn.pt")), \
             patch("src.eval.harness.find_yolo_best_pt", return_value=Path("/fake/yolo.pt")), \
             patch("src.eval.harness._sha256_file", return_value="abcd1234"), \
             patch("src.eval.harness.load_label_sidecars") as mock_labels, \
             patch("src.eval.harness.preprocess_for_pipeline",
                   return_value=np.zeros((512, 512), dtype="uint8")), \
             patch("src.eval.harness.Image"), \
             patch.object(Path, "glob",
                          return_value=[Path("/fake/project/data/eval/bank/01-sample.png")]), \
             patch.dict(sys.modules, {
                 "torch": MagicMock(__version__="2.0.0"),
                 "ultralytics": MagicMock(__version__="8.0.0"),
             }):

            mock_cfg_cls.from_project_root.return_value = MagicMock()
            label = _make_sample_label(equation_kind="addition", symbols=None,
                                       sample="01-sample")
            mock_labels.return_value = {"01-sample": label}

            run_eval(fake_root)

        mock_session.auto_load.assert_called_once()
        mock_session.load.assert_not_called()


class TestRunEvalReturnsPinnedVersions(unittest.TestCase):
    def test_run_eval_returns_record_with_pinned_versions(self):
        """RunRecord.torch_version is non-empty; yolo_sha and gnn_sha are 8-char hex."""
        from src.eval.harness import run_eval
        import re

        mock_session = MagicMock()
        mock_session.is_loaded.return_value = True
        mock_session._yolo_inference_config = YoloInferenceConfig()
        mock_session.predict.return_value = {"equation_kind": "add", "tokens": []}

        import numpy as np

        fake_root = Path("/fake/project")
        label = _make_sample_label(equation_kind="addition", symbols=None, sample="s1")

        with patch("src.eval.harness.InferenceSession", return_value=mock_session), \
             patch("src.eval.harness.resolve_gnn_best_pt", return_value=Path("/fake/gnn.pt")), \
             patch("src.eval.harness.find_yolo_best_pt", return_value=Path("/fake/yolo.pt")), \
             patch("src.eval.harness._sha256_file", return_value="abcd1234ef567890"), \
             patch("src.eval.harness.load_label_sidecars", return_value={"s1": label}), \
             patch("src.eval.harness.preprocess_for_pipeline",
                   return_value=np.zeros((512, 512), dtype="uint8")), \
             patch("src.eval.harness.Image"), \
             patch("src.core.config.DataPrepConfig.from_project_root"), \
             patch.object(Path, "glob", return_value=[Path("/fake/project/data/eval/bank/s1.png")]), \
             patch.dict(sys.modules, {
                 "torch": MagicMock(__version__="2.3.1"),
                 "ultralytics": MagicMock(__version__="8.4.46"),
             }):

            record = run_eval(fake_root)

        self.assertIsInstance(record.torch_version, str)
        self.assertGreater(len(record.torch_version), 0)
        self.assertIsNotNone(record.gnn_sha)
        self.assertEqual(len(record.gnn_sha), 8)
        if record.yolo_sha is not None:
            self.assertEqual(len(record.yolo_sha), 8)
        hex_re = re.compile(r"^[0-9a-f]{8}$")
        self.assertRegex(record.gnn_sha, hex_re)


class TestRunEvalRaisesWhenWeightsMissing(unittest.TestCase):
    def test_run_eval_raises_when_weights_missing(self):
        """auto_load raising wraps into ValueError naming missing path."""
        from src.eval.harness import run_eval

        with patch("src.eval.harness.resolve_gnn_best_pt",
                   side_effect=FileNotFoundError("No GNN weights found")), \
             patch("src.core.config.DataPrepConfig.from_project_root"), \
             patch.dict(sys.modules, {
                 "torch": MagicMock(__version__="2.0.0"),
                 "ultralytics": MagicMock(__version__="8.0.0"),
             }):
            with self.assertRaises(ValueError) as ctx:
                run_eval(Path("/fake/project"))
            self.assertIn("GNN", str(ctx.exception))


# ---------------------------------------------------------------------------
# Mapping coverage sentinel
# ---------------------------------------------------------------------------

class TestAssemblerMappingCoversAllKinds(unittest.TestCase):
    def test_assembler_short_to_label_long_mapping_covers_all_kinds(self):
        """_ASSEMBLER_TO_LABEL.values() >= EQUATION_KINDS — drift sentinel."""
        mapped_long_names = set(_ASSEMBLER_TO_LABEL.values())
        self.assertTrue(
            mapped_long_names >= EQUATION_KINDS,
            f"EQUATION_KINDS not covered. Missing: {EQUATION_KINDS - mapped_long_names}",
        )


# ---------------------------------------------------------------------------
# Token key normalisation regression guard (assembler uses "label"/"bbox")
# ---------------------------------------------------------------------------

class TestComputePerSampleMetricsAssemblerTokenKeys(unittest.TestCase):
    """compute_per_sample_metrics must normalise assembler-format token keys.

    The assembler emits tokens with ``"label"`` and ``"bbox"`` keys, not
    ``"flattened"`` and ``"bbox_px"``.  The harness must accept both formats
    via a fallback chain so that per-symbol metrics are never silently zeroed.
    """

    def test_assembler_label_and_bbox_keys_yield_nonzero_match(self):
        """Assembler-format token (label+bbox) matches GT → matched=1."""
        sym = SymbolLabel(flattened="main_5", bbox_px=(20.0, 30.0, 80.0, 90.0))
        label = _make_sample_label(equation_kind="addition", symbols=(sym,))
        # Assembler key format: "label" + "bbox"
        pred = {
            "equation_kind": "add",
            "tokens": [
                {
                    "label": "main_5",
                    "bbox": [25.0, 35.0, 75.0, 85.0],
                    "confidence": 0.95,
                }
            ],
        }
        metric = compute_per_sample_metrics(pred, label, wall_ms=1.0)
        self.assertEqual(metric.n_gt, 1)
        self.assertEqual(metric.matched, 1)
        self.assertEqual(metric.label_correct, 1)
        self.assertAlmostEqual(metric.recall, 1.0)

    def test_legacy_flattened_and_bbox_px_keys_still_work(self):
        """Legacy format token (flattened+bbox_px) still matches GT correctly."""
        sym = SymbolLabel(flattened="op_plus", bbox_px=(10.0, 10.0, 60.0, 60.0))
        label = _make_sample_label(equation_kind="addition", symbols=(sym,))
        # Legacy key format: "flattened" + "bbox_px"
        pred = {
            "equation_kind": "add",
            "tokens": [
                {
                    "flattened": "op_plus",
                    "bbox_px": [12.0, 12.0, 58.0, 58.0],
                }
            ],
        }
        metric = compute_per_sample_metrics(pred, label, wall_ms=1.0)
        self.assertEqual(metric.matched, 1)
        self.assertEqual(metric.label_correct, 1)


# ---------------------------------------------------------------------------
# row_acc / col_acc computation tests
# ---------------------------------------------------------------------------

class TestRowColAccComputedFromDetections(unittest.TestCase):
    """compute_per_sample_metrics derives row_acc and col_acc from detections[]."""

    def _pred_with_detections(self, detections, tokens=None, eq_kind="add"):
        """Build a minimal pred dict with detections and tokens fields."""
        if tokens is None:
            tokens = [
                {"label": d["label"], "bbox": d["bbox"]}
                for d in detections
            ]
        return {
            "equation_kind": eq_kind,
            "tokens": tokens,
            "detections": detections,
        }

    def test_row_acc_1_when_all_pairs_match_row(self):
        """Single matched pair with correct row prediction gives row_acc=1.0."""
        # GT: one symbol in row 0 (top), one in row 1 (bottom)
        sym0 = SymbolLabel(flattened="main_1", bbox_px=(10.0, 10.0, 50.0, 50.0))
        sym1 = SymbolLabel(flattened="op_plus", bbox_px=(60.0, 10.0, 100.0, 50.0))
        label = _make_sample_label(equation_kind="addition", symbols=(sym0, sym1))

        # Two detections: row=0, col=0 and row=0, col=1 (same row — correct cluster)
        detections = [
            {"label": "main_1", "bbox": [10.0, 10.0, 50.0, 50.0], "row": 0, "col": 0, "conf": 0.9},
            {"label": "op_plus", "bbox": [60.0, 10.0, 100.0, 50.0], "row": 0, "col": 1, "conf": 0.9},
        ]
        pred = self._pred_with_detections(detections)
        metric = compute_per_sample_metrics(pred, label, wall_ms=1.0)
        # Both symbols have similar y-centroids so GT cluster assigns both to row 0.
        # Both predicted row=0. row_acc should be 1.0.
        self.assertIsNotNone(metric.row_acc)
        self.assertAlmostEqual(metric.row_acc, 1.0)

    def test_row_acc_0_when_no_pairs_match_row(self):
        """Matched pair with wrong row prediction gives row_acc=0.0."""
        sym0 = SymbolLabel(flattened="main_5", bbox_px=(10.0, 10.0, 50.0, 50.0))
        sym1 = SymbolLabel(flattened="main_3", bbox_px=(60.0, 300.0, 100.0, 340.0))
        label = _make_sample_label(equation_kind="addition", symbols=(sym0, sym1))

        # GT: sym0 y-center=30, sym1 y-center=320 — large gap, two distinct rows.
        # Predictions: both assigned row=0 (wrong for sym1).
        detections = [
            {"label": "main_5", "bbox": [10.0, 10.0, 50.0, 50.0], "row": 0, "col": 0, "conf": 0.9},
            {"label": "main_3", "bbox": [60.0, 300.0, 100.0, 340.0], "row": 0, "col": 1, "conf": 0.9},
        ]
        pred = self._pred_with_detections(detections)
        metric = compute_per_sample_metrics(pred, label, wall_ms=1.0)
        # sym0 GT row=0, pred row=0 — correct; sym1 GT row=1, pred row=0 — wrong.
        # row_acc = 0.5 (one of two pairs correct).
        self.assertIsNotNone(metric.row_acc)
        self.assertAlmostEqual(metric.row_acc, 0.5)

    def test_row_col_acc_none_when_no_symbols(self):
        """row_acc and col_acc are None when label.symbols is None."""
        pred = {"equation_kind": "add", "tokens": [], "detections": []}
        label = _make_sample_label(symbols=None)
        metric = compute_per_sample_metrics(pred, label, wall_ms=1.0)
        self.assertIsNone(metric.row_acc)
        self.assertIsNone(metric.col_acc)

    def test_row_col_acc_none_when_no_detections(self):
        """row_acc and col_acc are None when detections[] is absent."""
        sym = SymbolLabel(flattened="main_2", bbox_px=(10.0, 10.0, 50.0, 50.0))
        label = _make_sample_label(symbols=(sym,))
        pred = {
            "equation_kind": "add",
            "tokens": [{"label": "main_2", "bbox": [10.0, 10.0, 50.0, 50.0]}],
            # no "detections" key
        }
        metric = compute_per_sample_metrics(pred, label, wall_ms=1.0)
        self.assertIsNone(metric.row_acc)
        self.assertIsNone(metric.col_acc)


class TestAggregateRowColAccMacro(unittest.TestCase):
    """compute_aggregate correctly macro-averages row_acc and col_acc."""

    def test_macro_average_excludes_none(self):
        """row_acc_macro is mean of non-None row_acc values."""
        samples = [
            _make_sample_metric(row_acc=0.8, col_acc=0.6),
            _make_sample_metric(row_acc=0.6, col_acc=0.4),
            _make_sample_metric(row_acc=None, col_acc=None),
        ]
        agg = compute_aggregate(samples)
        self.assertIsNotNone(agg.row_acc_macro)
        self.assertAlmostEqual(agg.row_acc_macro, 0.7)
        self.assertIsNotNone(agg.col_acc_macro)
        self.assertAlmostEqual(agg.col_acc_macro, 0.5)
        self.assertEqual(agg.n_samples_with_spatial, 2)

    def test_macro_average_all_none_gives_none(self):
        """row_acc_macro is None when all samples have row_acc=None."""
        samples = [
            _make_sample_metric(row_acc=None, col_acc=None),
            _make_sample_metric(row_acc=None, col_acc=None),
        ]
        agg = compute_aggregate(samples)
        self.assertIsNone(agg.row_acc_macro)
        self.assertIsNone(agg.col_acc_macro)
        self.assertEqual(agg.n_samples_with_spatial, 0)


# ---------------------------------------------------------------------------
# Phase 0: wilson_interval tests
# ---------------------------------------------------------------------------

class TestWilsonInterval(unittest.TestCase):
    """wilson_interval returns correct bounds for known values."""

    def test_wilson_interval_zero_n_returns_zero_zero(self) -> None:
        lo, hi = wilson_interval(0, 0)
        self.assertEqual(lo, 0.0)
        self.assertEqual(hi, 0.0)

    def test_wilson_interval_8_of_10_approx(self) -> None:
        """8/10 successes: Wilson CI should be approximately (0.49, 0.94)."""
        lo, hi = wilson_interval(8, 10)
        self.assertAlmostEqual(lo, 0.49, delta=0.02)
        self.assertAlmostEqual(hi, 0.94, delta=0.02)

    def test_wilson_interval_all_success(self) -> None:
        """10/10: upper bound should be 1.0, lower should be > 0.7."""
        lo, hi = wilson_interval(10, 10)
        self.assertGreater(lo, 0.7)
        self.assertAlmostEqual(hi, 1.0, delta=0.001)

    def test_wilson_interval_zero_success(self) -> None:
        """0/10: lower bound should be 0.0, upper should be < 0.3."""
        lo, hi = wilson_interval(0, 10)
        self.assertAlmostEqual(lo, 0.0, delta=0.001)
        self.assertLess(hi, 0.3)

    def test_wilson_interval_bounds_in_range(self) -> None:
        """All outputs must be in [0, 1]."""
        for s, n in [(3, 7), (5, 5), (1, 100), (50, 100)]:
            lo, hi = wilson_interval(s, n)
            self.assertGreaterEqual(lo, 0.0, f"lo out of range for {s}/{n}")
            self.assertLessEqual(hi, 1.0, f"hi out of range for {s}/{n}")
            self.assertLessEqual(lo, hi, f"lo > hi for {s}/{n}")


# ---------------------------------------------------------------------------
# Phase 0: aggregate_by tests
# ---------------------------------------------------------------------------

class TestAggregateByEquationKind(unittest.TestCase):
    """aggregate_by('equation_kind') groups samples correctly."""

    def test_aggregate_by_equation_kind_groups(self) -> None:
        """Three samples: 2 addition, 1 subtraction — groups are correct."""
        samples = [
            _make_sample_metric(equation_kind_ok=True, name="a1"),
            _make_sample_metric(equation_kind_ok=True, name="a2"),
            _make_sample_metric(equation_kind_ok=False, name="s1"),
        ]
        # Patch equation_kind_gt to be different kinds
        import dataclasses
        s_add1 = dataclasses.replace(samples[0], equation_kind_gt="addition")
        s_add2 = dataclasses.replace(samples[1], equation_kind_gt="addition")
        s_sub = dataclasses.replace(samples[2], equation_kind_gt="subtraction", equation_kind_ok=False)

        result = aggregate_by([s_add1, s_add2, s_sub], "equation_kind")
        self.assertIn("addition", result)
        self.assertIn("subtraction", result)
        self.assertEqual(result["addition"].n_samples, 2)
        self.assertEqual(result["subtraction"].n_samples, 1)
        self.assertAlmostEqual(result["addition"].equation_kind_acc, 1.0)
        self.assertAlmostEqual(result["subtraction"].equation_kind_acc, 0.0)


class TestAggregateBySceneCase(unittest.TestCase):
    """aggregate_by('scene_case') groups samples by scene_case, skipping None."""

    def test_aggregate_by_scene_case_groups_correctly(self) -> None:
        import dataclasses
        s1 = dataclasses.replace(_make_sample_metric(equation_kind_ok=True), scene_case="addition")
        s2 = dataclasses.replace(_make_sample_metric(equation_kind_ok=True), scene_case="addition")
        s3 = dataclasses.replace(_make_sample_metric(equation_kind_ok=False), scene_case="division-short")
        s4 = dataclasses.replace(_make_sample_metric(equation_kind_ok=True), scene_case=None)

        result = aggregate_by([s1, s2, s3, s4], "scene_case")
        self.assertIn("addition", result)
        self.assertIn("division-short", result)
        self.assertNotIn(None, result)
        self.assertEqual(result["addition"].n_samples, 2)
        self.assertEqual(result["division-short"].n_samples, 1)

    def test_aggregate_by_all_none_scene_case_returns_empty(self) -> None:
        """When all scene_cases are None, result is empty dict."""
        import dataclasses
        samples = [
            dataclasses.replace(_make_sample_metric(), scene_case=None),
            dataclasses.replace(_make_sample_metric(), scene_case=None),
        ]
        result = aggregate_by(samples, "scene_case")
        self.assertEqual(result, {})

    def test_aggregate_by_invalid_key_raises(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_by([], "invalid_key")


# ---------------------------------------------------------------------------
# Phase 0: render_markdown_summary — per-kind and per-case tables
# ---------------------------------------------------------------------------

class TestRenderMarkdownSummaryPerKindTable(unittest.TestCase):
    """render_markdown_summary includes Per Equation-Kind table when data present."""

    def _make_record_with_groups(self, equation_kind_acc_add: float = 1.0) -> RunRecord:
        import dataclasses
        sample_add = dataclasses.replace(
            _make_sample_metric(equation_kind_ok=True),
            equation_kind_gt="addition",
        )
        sample_sub = dataclasses.replace(
            _make_sample_metric(equation_kind_ok=False),
            equation_kind_gt="subtraction",
        )
        from src.eval.harness import aggregate_by as _ab, compute_aggregate as _ca
        agg = _ca([sample_add, sample_sub])
        pek = _ab([sample_add, sample_sub], "equation_kind")
        psc: dict = {}
        from src.eval.harness import RunRecord, AggregateMetric
        return RunRecord(
            run_id="20260604T000000",
            config_id="test",
            config_snapshot={},
            torch_version="2.0",
            ultralytics_version="8.0",
            python_version="3.12",
            yolo_sha=None,
            gnn_sha="abc12345",
            aggregate=agg,
            samples=(sample_add, sample_sub),
            per_equation_kind=pek,
            per_scene_case=psc,
        )

    def test_per_equation_kind_table_present(self) -> None:
        import tempfile
        from src.eval.harness import render_markdown_summary
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "summary.md"
            rec = self._make_record_with_groups()
            render_markdown_summary(rec, None, out)
            content = out.read_text(encoding="utf-8")
            self.assertIn("Per Equation-Kind", content)
            self.assertIn("addition", content)
            self.assertIn("subtraction", content)

    def test_per_scene_case_table_omitted_when_empty(self) -> None:
        """When per_scene_case is empty dict, scene-case table is NOT rendered."""
        import tempfile
        from src.eval.harness import render_markdown_summary
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "summary.md"
            rec = self._make_record_with_groups()
            render_markdown_summary(rec, None, out)
            content = out.read_text(encoding="utf-8")
            self.assertNotIn("Per Scene-Case", content)

    def test_per_scene_case_table_present_when_populated(self) -> None:
        import dataclasses, tempfile
        from src.eval.harness import render_markdown_summary, aggregate_by, compute_aggregate, RunRecord
        s1 = dataclasses.replace(_make_sample_metric(equation_kind_ok=True), scene_case="addition")
        agg = compute_aggregate([s1])
        psc = aggregate_by([s1], "scene_case")
        rec = RunRecord(
            run_id="20260604T000001",
            config_id="test",
            config_snapshot={},
            torch_version="2.0",
            ultralytics_version="8.0",
            python_version="3.12",
            yolo_sha=None,
            gnn_sha="abc12345",
            aggregate=agg,
            samples=(s1,),
            per_equation_kind={},
            per_scene_case=psc,
        )
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "summary.md"
            render_markdown_summary(rec, None, out)
            content = out.read_text(encoding="utf-8")
            self.assertIn("Per Scene-Case", content)
            self.assertIn("addition", content)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
