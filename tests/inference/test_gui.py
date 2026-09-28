from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.inference.annotation import (
    annotation_colour,
    build_annotations,
    build_gt_annotations,
)
from src.inference.gt_loader import (
    GtSymbol,
    SceneScores,
    _iou,
    compute_scene_scores,
    load_scene,
    match_predictions,
)


def _tok(label: str, conf: float, bbox: list) -> dict:
    return {"label": label, "confidence": conf, "bbox": bbox, "row": 0, "col": 0}


class TestAnnotationColour(unittest.TestCase):
    def test_digit_main_returns_blue(self) -> None:
        self.assertEqual(annotation_colour("main_3", 0.99), "#3996ff")

    def test_digit_carry_returns_orange_red(self) -> None:
        self.assertEqual(annotation_colour("carry_1", 0.99), "#f56949")

    def test_digit_borrow_returns_amber(self) -> None:
        self.assertEqual(annotation_colour("borrow_2", 0.99), "#ed7000")

    def test_operator_returns_purple(self) -> None:
        self.assertEqual(annotation_colour("op_plus", 0.99), "#b672ff")

    def test_result_bar_returns_green(self) -> None:
        self.assertEqual(annotation_colour("result_bar", 0.99), "#39a849")

    def test_div_bracket_returns_light_blue(self) -> None:
        self.assertEqual(annotation_colour("div_bracket", 0.99), "#2297ff")

    def test_low_confidence_returns_amber_orange(self) -> None:
        self.assertEqual(annotation_colour("main_3", 0.49), "#ed7118")
        self.assertEqual(annotation_colour("carry_1", 0.0), "#ed7118")


class TestBuildAnnotations(unittest.TestCase):
    def test_returns_one_entry_per_token(self) -> None:
        tokens = [_tok("main_2", 0.99, [10, 20, 30, 40]), _tok("carry_1", 0.97, [5, 5, 15, 15])]
        annotations, color_map = build_annotations(tokens)
        self.assertEqual(len(annotations), 2)
        self.assertEqual(len(color_map), 2)

    def test_bbox_tuple_is_ints(self) -> None:
        tokens = [_tok("main_2", 0.99, [10.5, 20.1, 30.9, 40.0])]
        annotations, _ = build_annotations(tokens)
        bbox, _ = annotations[0]
        self.assertIsInstance(bbox[0], int)

    def test_label_format_includes_confidence(self) -> None:
        tokens = [_tok("carry_1", 0.97, [0, 0, 10, 10])]
        annotations, color_map = build_annotations(tokens)
        _, label = annotations[0]
        self.assertIn("carry_1", label)
        self.assertIn("0.97", label)

    def test_low_confidence_label_has_warning(self) -> None:
        tokens = [_tok("main_3", 0.45, [0, 0, 10, 10])]
        annotations, color_map = build_annotations(tokens)
        _, label = annotations[0]
        self.assertIn("⚠", label)

    def test_color_map_key_matches_annotation_label(self) -> None:
        tokens = [_tok("op_plus", 0.94, [0, 0, 10, 10])]
        annotations, color_map = build_annotations(tokens)
        _, label = annotations[0]
        self.assertIn(label, color_map)

    def test_empty_tokens_returns_empty(self) -> None:
        annotations, color_map = build_annotations([])
        self.assertEqual(annotations, [])
        self.assertEqual(color_map, {})


class TestBuildGtAnnotations(unittest.TestCase):
    def _gt(self, fine_label: str, bbox: list) -> GtSymbol:
        return GtSymbol(fine_label=fine_label, row_index=0, col_index=0, bbox=bbox, yolo_class="digit_main")

    def test_correct_match_is_green(self) -> None:
        tok = _tok("main_3", 0.98, [0, 0, 20, 20])
        gt = self._gt("main_3", [0, 0, 20, 20])
        annotations, color_map = build_gt_annotations([tok], [gt])
        _, label = annotations[0]
        self.assertIn("✓", label)
        color = list(color_map.values())[0]
        self.assertEqual(color, "#3fb950")

    def test_wrong_match_is_red(self) -> None:
        tok = _tok("main_1", 0.98, [0, 0, 20, 20])
        gt = self._gt("main_7", [0, 0, 20, 20])
        annotations, color_map = build_gt_annotations([tok], [gt])
        _, label = annotations[0]
        self.assertIn("✗", label)
        self.assertIn("main_7", label)
        color = list(color_map.values())[0]
        self.assertEqual(color, "#f85149")

    def test_missed_gt_is_amber(self) -> None:
        gt = self._gt("carry_1", [50, 50, 70, 70])
        annotations, color_map = build_gt_annotations([], [gt])
        _, label = annotations[0]
        self.assertIn("missing", label)
        color = list(color_map.values())[0]
        self.assertEqual(color, "#ffa657")

    def test_false_positive_is_orange(self) -> None:
        tok = _tok("main_5", 0.99, [90, 90, 110, 110])
        annotations, color_map = build_gt_annotations([tok], [])
        _, label = annotations[0]
        self.assertIn("extra", label)
        color = list(color_map.values())[0]
        self.assertEqual(color, "#f0883e")


class TestIou(unittest.TestCase):
    def test_identical_boxes_is_one(self) -> None:
        self.assertAlmostEqual(_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)

    def test_non_overlapping_is_zero(self) -> None:
        self.assertAlmostEqual(_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_half_overlap(self) -> None:
        iou = _iou([0, 0, 10, 10], [5, 0, 15, 10])
        self.assertAlmostEqual(iou, 1 / 3, places=5)


class TestLoadScene(unittest.TestCase):
    def _write_scene(self, tmp: Path) -> tuple[Path, Path]:
        img_path = tmp / "scene.png"
        Image.fromarray(np.full((128, 128), 200, dtype=np.uint8)).save(img_path)
        gt = {
            "equation_type": "add",
            "symbols": [
                {"fine_label": "main_2", "row_index": 1, "col_index": 2,
                 "bbox": [10, 20, 30, 40], "yolo_class": "digit_main"},
            ],
        }
        gt_path = tmp / "gt.json"
        gt_path.write_text(json.dumps(gt))
        return img_path, gt_path

    def test_returns_image_and_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            img_path, _ = self._write_scene(Path(d))
            image, symbols = load_scene(img_path)
            self.assertIsInstance(image, np.ndarray)
            self.assertEqual(len(symbols), 1)
            self.assertEqual(symbols[0].fine_label, "main_2")
            self.assertEqual(symbols[0].row_index, 1)

    def test_missing_gt_returns_none_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            img_path = Path(d) / "scene.png"
            Image.fromarray(np.full((64, 64), 255, dtype=np.uint8)).save(img_path)
            _, symbols = load_scene(img_path)
            self.assertIsNone(symbols)


class TestMatchPredictions(unittest.TestCase):
    def _tok(self, label: str, bbox: list) -> dict:
        return {"label": label, "confidence": 0.99, "bbox": bbox, "row": 0, "col": 0}

    def _gt(self, fine_label: str, bbox: list) -> GtSymbol:
        return GtSymbol(fine_label=fine_label, row_index=0, col_index=0, bbox=bbox, yolo_class="digit_main")

    def test_exact_match_is_correct(self) -> None:
        tok = self._tok("main_3", [0, 0, 20, 20])
        gt = self._gt("main_3", [0, 0, 20, 20])
        results = match_predictions([tok], [gt])
        self.assertTrue(results[0].is_correct)

    def test_wrong_label_is_not_correct(self) -> None:
        tok = self._tok("main_1", [0, 0, 20, 20])
        gt = self._gt("main_7", [0, 0, 20, 20])
        results = match_predictions([tok], [gt])
        self.assertFalse(results[0].is_correct)

    def test_non_overlapping_is_false_positive_and_missed(self) -> None:
        tok = self._tok("main_5", [0, 0, 10, 10])
        gt = self._gt("carry_1", [50, 50, 60, 60])
        results = match_predictions([tok], [gt])
        fp = [r for r in results if r.is_false_positive]
        missed = [r for r in results if r.is_missed]
        self.assertEqual(len(fp), 1)
        self.assertEqual(len(missed), 1)


class TestComputeSceneScores(unittest.TestCase):
    def _tok(self, label: str, row: int, col: int, bbox: list) -> dict:
        return {"label": label, "confidence": 0.99, "bbox": bbox, "row": row, "col": col}

    def _gt(self, fine_label: str, row: int, col: int, bbox: list) -> GtSymbol:
        return GtSymbol(fine_label=fine_label, row_index=row, col_index=col, bbox=bbox, yolo_class="digit_main")

    def test_all_correct_gives_exact_match(self) -> None:
        toks = [self._tok("main_3", 1, 2, [0, 0, 20, 20])]
        gts = [self._gt("main_3", 1, 2, [0, 0, 20, 20])]
        scores = compute_scene_scores(toks, gts)
        self.assertTrue(scores.exact_match)
        self.assertAlmostEqual(scores.token_f1, 1.0)
        self.assertAlmostEqual(scores.row_accuracy, 1.0)
        self.assertAlmostEqual(scores.col_accuracy, 1.0)

    def test_one_wrong_label_breaks_exact_match(self) -> None:
        toks = [self._tok("main_1", 1, 2, [0, 0, 20, 20])]
        gts = [self._gt("main_7", 1, 2, [0, 0, 20, 20])]
        scores = compute_scene_scores(toks, gts)
        self.assertFalse(scores.exact_match)
        self.assertEqual(scores.n_correct, 0)

    def test_empty_both_gives_exact_match(self) -> None:
        scores = compute_scene_scores([], [])
        self.assertTrue(scores.exact_match)


from src.inference.gui_components import (
    render_result_panel,
    render_score_bar,
    render_stage1_panel,
    render_stage2_panel,
    render_status_bar,
)


class TestGuiComponents(unittest.TestCase):
    def _tok(self, label: str, row: int = 0, col: int = 0) -> dict:
        return {"label": label, "confidence": 0.99, "bbox": [0, 0, 10, 10], "row": row, "col": col}

    def test_stage1_shows_detection_count(self) -> None:
        tokens = [self._tok("main_2"), self._tok("carry_1"), self._tok("main_5")]
        html = render_stage1_panel(tokens, latency_ms=3.2, backend="yolo")
        self.assertIn("3", html)
        self.assertIn("3.2", html)

    def test_stage1_components_fallback_label(self) -> None:
        html = render_stage1_panel([], latency_ms=1.0, backend="components")
        self.assertIn("Components", html)

    def test_stage1_groups_by_coarse_class(self) -> None:
        tokens = [self._tok("main_2"), self._tok("main_5"), self._tok("carry_1")]
        html = render_stage1_panel(tokens, latency_ms=1.0, backend="yolo")
        # Stage 1 renders human-readable coarse names and counts
        self.assertIn("digit", html)
        self.assertIn("x2", html)
        self.assertIn("carry", html)
        self.assertIn("x1", html)

    def test_stage2_shows_pipeline_label_gnn(self) -> None:
        html = render_stage2_panel([], latency_ms=5.0, pipeline_label="GNN")
        self.assertIn("GNN", html)

    def test_stage2_shows_pipeline_label_cnn(self) -> None:
        html = render_stage2_panel([], latency_ms=5.0, pipeline_label="CNN")
        self.assertIn("CNN", html)

    def test_stage2_shows_row_col(self) -> None:
        tokens = [
            {**self._tok("carry_1", row=0, col=3), "grid_col": 2, "within_row_ord": 1}
        ]
        html = render_stage2_panel(tokens, latency_ms=5.0, pipeline_label="GNN")
        # Stage 2 now renders row groups; row 0 -> "Row 1" (1-indexed)
        self.assertIn("Row 1", html)
        # row/col/ord details appear in tooltip title attribute
        self.assertIn("row 0", html)
        self.assertIn("col 3", html)
        self.assertIn("ord 1", html)

    def test_stage2_multi_row_grouping(self) -> None:
        tokens = [
            self._tok("main_3", row=0, col=0),
            self._tok("op_plus", row=0, col=1),
            self._tok("main_7", row=1, col=0),
            self._tok("result_bar", row=2, col=0),
        ]
        html = render_stage2_panel(tokens, latency_ms=5.0, pipeline_label="GNN")
        # Three rows -> Row 1, Row 2, Row 3
        self.assertIn("Row 1", html)
        self.assertIn("Row 2", html)
        self.assertIn("Row 3", html)
        # Glyphs rendered: digit 3, plus, digit 7, result_bar
        self.assertIn(">3<", html)
        self.assertIn(">+<", html)
        self.assertIn(">7<", html)
        self.assertIn("─────", html)

    def test_stage2_div_bracket_renders_bracket_chip(self) -> None:
        tokens = [self._tok("div_bracket", row=0, col=0)]
        html = render_stage2_panel(tokens, latency_ms=1.0, pipeline_label="GNN")
        # div_bracket uses a CSS-drawn bracket; chip carries the modifier class
        self.assertIn("engine-chip--bracket", html)
        # role colour is injected as --chip CSS property
        self.assertIn("--chip:", html)

    def test_result_panel_shows_equation_type(self) -> None:
        html = render_result_panel(equation_type="add", display_text="  1\n25\n+37\n──\n62", latency_ms=1.0)
        # Equation kind is shown in full ("add" -> "Addition") in the result body.
        self.assertIn("Addition", html)
        self.assertIn("Equation type", html)

    def test_result_panel_empty_shows_dash(self) -> None:
        html = render_result_panel(equation_type=None, display_text=None, latency_ms=0.0)
        self.assertIn("—", html)

    def test_result_panel_with_tokens_renders_rows(self) -> None:
        tokens = [
            self._tok("main_2", row=0, col=0),
            self._tok("op_plus", row=0, col=1),
            self._tok("main_5", row=0, col=2),
            self._tok("result_bar", row=1, col=0),
            self._tok("main_7", row=2, col=0),
        ]
        html = render_result_panel(equation_type="add", display_text="2+5=7", latency_ms=1.0, tokens=tokens)
        # Row groups rendered
        self.assertIn("Row 1", html)
        self.assertIn("Row 2", html)
        self.assertIn("Row 3", html)
        # Glyphs present
        self.assertIn(">2<", html)
        self.assertIn(">+<", html)
        self.assertIn("─────", html)

    def test_result_panel_ood_shows_text_fallback(self) -> None:
        tokens = [self._tok("main_3", row=0, col=0)]
        html = render_result_panel(equation_type="unknown", display_text="OOD scene", latency_ms=1.0, tokens=tokens)
        # OOD: should use monospace text block, not row chips
        self.assertIn("engine-result-eq", html)
        self.assertIn("OOD scene", html)
        self.assertNotIn("Row 1", html)

    def test_status_bar_fully_loaded(self) -> None:
        html = render_status_bar("yolo+gnn", "YOLO + GNN", "best.pt", None)
        self.assertIn("YOLO + GNN", html)
        self.assertIn("YOLO", html)

    def test_status_bar_no_models(self) -> None:
        html = render_status_bar(None, None, None, None)
        self.assertIn("No models", html)

    def test_score_bar_shows_metrics(self) -> None:
        scores = SceneScores(token_f1=0.86, row_accuracy=1.0, col_accuracy=0.75,
                             exact_match=False, n_correct=3, n_total=4)
        html = render_score_bar(scores)
        self.assertIn("3/4", html)
        self.assertIn("0.86", html)
        self.assertIn("1.00", html)
        self.assertIn("0.75", html)
        self.assertIn("✗", html)


from src.inference.run import InferenceSession, scan_stage1_models, scan_stage2_models


class TestRunPayloadDetections(unittest.TestCase):
    """_flatten_payload_tokens should flatten rows into a single tokens list."""

    def test_flatten_payload_tokens(self) -> None:
        from src.inference.run import _flatten_payload_tokens
        payload = {
            "rows": [
                {"row": 0, "tokens": [
                    {"label": "carry_1", "confidence": 0.97,
                     "bbox": [5, 5, 15, 15], "row": 0, "col": 3}
                ]},
                {"row": 1, "tokens": [
                    {"label": "main_2", "confidence": 0.99,
                     "bbox": [10, 20, 30, 40], "row": 1, "col": 2}
                ]},
            ]
        }
        tokens = _flatten_payload_tokens(payload)
        self.assertEqual(len(tokens), 2)
        self.assertEqual(tokens[0]["label"], "carry_1")
        self.assertEqual(tokens[1]["row"], 1)


class TestScanArtifacts(unittest.TestCase):
    def test_scan_stage1_finds_pt_files(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "artifacts" / "stage1").mkdir(parents=True)
            (root / "artifacts" / "stage1" / "best.pt").touch()
            results = scan_stage1_models(root)
            self.assertEqual(len(results), 1)
            self.assertIn("best.pt", results[0])

    def test_scan_stage1_empty_if_no_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            results = scan_stage1_models(Path(d))
            self.assertEqual(results, [])

    def test_scan_stage2_finds_keras_files(self) -> None:
        # scan_stage2_models now looks in artifacts/gnn/ (GNN pipeline replaced CNN/Keras).
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "artifacts" / "gnn").mkdir(parents=True)
            (root / "artifacts" / "gnn" / "best.pt").touch()
            results = scan_stage2_models(root)
            self.assertEqual(len(results), 1)
            self.assertIn("best.pt", results[0])

    def test_scan_stage2_finds_gnn_pt_files(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            gnn_dir = root / "artifacts" / "gnn" / "runs" / "run-001"
            gnn_dir.mkdir(parents=True)
            (gnn_dir / "best.pt").touch()
            results = scan_stage2_models(root)
            self.assertEqual(len(results), 1)
            self.assertIn("best.pt", results[0])


class TestInferenceSessionPipelineName(unittest.TestCase):
    def test_default_pipeline_name_is_none(self) -> None:
        session = InferenceSession()
        self.assertIsNone(session.pipeline_name)
        self.assertIsNone(session.pipeline_display)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# W10-GUI-RELAX: detection annotation and banner tests
# ---------------------------------------------------------------------------

from src.inference.annotation import build_detection_annotations
from src.inference.gui import _equation_kind_banner


class TestBuildDetectionAnnotations(unittest.TestCase):
    """build_detection_annotations produces R<row>C<col> · <label> · <conf> entries."""

    def _det(self, label: str, row: int, col: int, conf: float = 0.8) -> dict:
        return {"label": label, "bbox": [10.0, 10.0, 50.0, 50.0], "row": row, "col": col, "conf": conf}

    def test_empty_detections_returns_empty(self) -> None:
        annotations, color_map = build_detection_annotations([])
        self.assertEqual(annotations, [])
        self.assertEqual(color_map, {})

    def test_single_detection_label_format(self) -> None:
        annotations, color_map = build_detection_annotations([self._det("main_3", 0, 1, 0.75)])
        self.assertEqual(len(annotations), 1)
        label_str = annotations[0][1]
        self.assertIn("R0C1", label_str)
        # Label uses the display glyph (main_3 -> "3"), not the raw fine label.
        self.assertIn("3", label_str)
        self.assertIn("0.75", label_str)

    def test_different_rows_get_different_colors(self) -> None:
        dets = [self._det("main_1", 0, 0), self._det("main_2", 1, 0)]
        _, color_map = build_detection_annotations(dets)
        colours = list(color_map.values())
        self.assertNotEqual(colours[0], colours[1])

    def test_bbox_is_int_tuple(self) -> None:
        dets = [self._det("main_5", 0, 0)]
        annotations, _ = build_detection_annotations(dets)
        bbox = annotations[0][0]
        self.assertIsInstance(bbox, tuple)
        self.assertEqual(len(bbox), 4)
        for v in bbox:
            self.assertIsInstance(v, int)


class TestEquationKindBanner(unittest.TestCase):
    """_equation_kind_banner returns the correct banner text."""

    def test_bare_digits_banner(self) -> None:
        banner = _equation_kind_banner("bare_digits", None)
        self.assertIn("bare digits", banner.lower())

    def test_true_empty_banner(self) -> None:
        banner = _equation_kind_banner("unknown", "true_empty")
        self.assertIn("no detections", banner.lower())

    def test_other_ood_reason_banner(self) -> None:
        banner = _equation_kind_banner("unknown", "no_structural_token")
        self.assertIsNotNone(banner)
        self.assertIn("no structural token", banner.lower())

    def test_normal_equation_kind_no_banner(self) -> None:
        banner = _equation_kind_banner("add", None)
        self.assertIsNone(banner)

    def test_divide_kind_no_banner(self) -> None:
        banner = _equation_kind_banner("divide", None)
        self.assertIsNone(banner)


from src.inference.gui_components import render_cv_panel  # noqa: E402


class TestCvPanelCardStructure(unittest.TestCase):
    """render_cv_panel must use the shared _card wrapper, matching sibling stage panels."""

    def test_empty_uses_engine_card(self) -> None:
        html = render_cv_panel([], 0.0)
        self.assertIn('class="engine-card"', html)
        self.assertIn('class="engine-card-header"', html)
        self.assertIn('class="engine-card-body"', html)

    def test_empty_shows_no_extra_detections(self) -> None:
        html = render_cv_panel([], 0.0)
        self.assertIn("No extra detections", html)

    def test_with_detections_uses_engine_card(self) -> None:
        dets = [{"label": "digit_main", "confidence": 0.8, "bbox": [0, 0, 10, 10]}]
        html = render_cv_panel(dets, 5.0)
        self.assertIn('class="engine-card"', html)
        self.assertIn('class="engine-card-header"', html)
        self.assertIn('class="engine-card-body"', html)

    def test_cyan_accent_present(self) -> None:
        html = render_cv_panel([], 0.0)
        self.assertIn("#22d3ee", html)

    def test_skipped_uses_engine_card(self) -> None:
        html = render_cv_panel([], 0.0, skipped="complete")
        self.assertIn('class="engine-card"', html)
        self.assertIn("#22d3ee", html)
