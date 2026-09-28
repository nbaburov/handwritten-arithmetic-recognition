"""Tests for W2-GNN-BBOX-JITTER: training-time bbox jitter in SceneGraphDataset.

Covers:
- jitter=0.0 leaves coords unchanged
- jitter=2.0 with fixed seed shifts coords but stays within [0, 512]
- val/test datasets (training=False) never apply jitter regardless of jitter_px
- post-jitter invariants: x0 < x1 and y0 < y1 always hold
- post-jitter invariants: all coords stay in [0, 512]
- edge case: bbox whose edge is already at canvas boundary does not go out of bounds
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.parsing.detection import Detection
from src.training.train_gnn import _CANVAS_SIZE, _jitter_detections


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detection(x0: float, y0: float, x1: float, y1: float) -> Detection:
    return Detection(label="digit_main", confidence=1.0, x0=x0, y0=y0, x1=x1, y1=y1)


def _fixed_rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Unit tests for _jitter_detections
# ---------------------------------------------------------------------------

class TestJitterDetectionsNoJitter(unittest.TestCase):
    """jitter_px=0.0 must return coords identical to input."""

    def test_zero_jitter_preserves_coords(self) -> None:
        dets = [
            _make_detection(10.0, 20.0, 50.0, 60.0),
            _make_detection(100.0, 100.0, 200.0, 200.0),
        ]
        out = _jitter_detections(dets, jitter_px=0.0, rng=_fixed_rng(42))
        for orig, jittered in zip(dets, out):
            self.assertAlmostEqual(jittered.x0, orig.x0)
            self.assertAlmostEqual(jittered.y0, orig.y0)
            self.assertAlmostEqual(jittered.x1, orig.x1)
            self.assertAlmostEqual(jittered.y1, orig.y1)

    def test_zero_jitter_preserves_label_and_confidence(self) -> None:
        det = _make_detection(10.0, 10.0, 30.0, 30.0)
        out = _jitter_detections([det], jitter_px=0.0, rng=_fixed_rng(0))
        self.assertEqual(out[0].label, det.label)
        self.assertAlmostEqual(out[0].confidence, det.confidence)


class TestJitterDetectionsMagnitude(unittest.TestCase):
    """jitter_px=2.0 must shift coords but keep them within [0, 512]."""

    def _run_n_trials(self, n: int, jitter_px: float) -> None:
        """Run n independent jitter calls with fresh rng each time."""
        for seed in range(n):
            det = _make_detection(100.0, 100.0, 200.0, 200.0)
            out = _jitter_detections([det], jitter_px=jitter_px, rng=_fixed_rng(seed))
            jd = out[0]
            # All coords in [0, _CANVAS_SIZE]
            self.assertGreaterEqual(jd.x0, 0.0)
            self.assertGreaterEqual(jd.y0, 0.0)
            self.assertLessEqual(jd.x1, _CANVAS_SIZE)
            self.assertLessEqual(jd.y1, _CANVAS_SIZE)
            # No swap
            self.assertLess(jd.x0, jd.x1)
            self.assertLess(jd.y0, jd.y1)

    def test_2px_jitter_invariants_over_many_seeds(self) -> None:
        self._run_n_trials(n=200, jitter_px=2.0)

    def test_fixed_seed_produces_nonzero_shift(self) -> None:
        """With jitter_px=2.0 at a known seed, at least one coord must differ from the original."""
        det = _make_detection(100.0, 100.0, 200.0, 200.0)
        # Use seed=1 — deterministic; any non-trivial seed will produce a shift.
        out = _jitter_detections([det], jitter_px=2.0, rng=_fixed_rng(1))
        jd = out[0]
        coords_changed = any([
            abs(jd.x0 - det.x0) > 1e-9,
            abs(jd.y0 - det.y0) > 1e-9,
            abs(jd.x1 - det.x1) > 1e-9,
            abs(jd.y1 - det.y1) > 1e-9,
        ])
        self.assertTrue(coords_changed, "Expected at least one coord to shift with jitter_px=2.0")

    def test_shift_bounded_by_jitter_px_before_clamping(self) -> None:
        """For a bbox far from the canvas boundary the raw shift must be <= jitter_px."""
        jitter_px = 2.0
        det = _make_detection(100.0, 100.0, 200.0, 200.0)
        rng = _fixed_rng(7)
        out = _jitter_detections([det], jitter_px=jitter_px, rng=rng)
        jd = out[0]
        # For a bbox deep inside the canvas the clamp never fires, so the shift
        # is exactly the uniform draw which is bounded by jitter_px.
        self.assertLessEqual(abs(jd.x0 - det.x0), jitter_px + 1e-9)
        self.assertLessEqual(abs(jd.y0 - det.y0), jitter_px + 1e-9)
        self.assertLessEqual(abs(jd.x1 - det.x1), jitter_px + 1e-9)
        self.assertLessEqual(abs(jd.y1 - det.y1), jitter_px + 1e-9)


class TestJitterDetectionsBoundaryEdgeCases(unittest.TestCase):
    """Bboxes at canvas edges must not escape [0, 512] after jitter."""

    def test_bbox_at_origin_does_not_go_negative(self) -> None:
        det = _make_detection(0.0, 0.0, 5.0, 5.0)
        for seed in range(50):
            out = _jitter_detections([det], jitter_px=2.0, rng=_fixed_rng(seed))
            jd = out[0]
            self.assertGreaterEqual(jd.x0, 0.0)
            self.assertGreaterEqual(jd.y0, 0.0)

    def test_bbox_at_canvas_max_does_not_exceed_canvas(self) -> None:
        det = _make_detection(507.0, 507.0, 512.0, 512.0)
        for seed in range(50):
            out = _jitter_detections([det], jitter_px=2.0, rng=_fixed_rng(seed))
            jd = out[0]
            self.assertLessEqual(jd.x1, _CANVAS_SIZE)
            self.assertLessEqual(jd.y1, _CANVAS_SIZE)

    def test_swap_guard_enforced_when_edges_collapse(self) -> None:
        """A 1-pixel bbox near the canvas max can collapse after jitter; swap guard must hold."""
        det = _make_detection(511.0, 511.0, 512.0, 512.0)
        for seed in range(100):
            out = _jitter_detections([det], jitter_px=2.0, rng=_fixed_rng(seed))
            jd = out[0]
            self.assertLess(jd.x0, jd.x1, f"x swap violated at seed={seed}")
            self.assertLess(jd.y0, jd.y1, f"y swap violated at seed={seed}")
            self.assertLessEqual(jd.x1, _CANVAS_SIZE)
            self.assertLessEqual(jd.y1, _CANVAS_SIZE)


class TestJitterDetectionsEmptyInput(unittest.TestCase):
    def test_empty_list_returns_empty_list(self) -> None:
        out = _jitter_detections([], jitter_px=2.0, rng=_fixed_rng(0))
        self.assertEqual(out, [])


# ---------------------------------------------------------------------------
# Integration tests: SceneGraphDataset training vs val/test mode
# ---------------------------------------------------------------------------

def _write_minimal_scene(tmp_dir: Path, scene_id: str) -> tuple[Path, Path]:
    """Write a minimal synthetic scene image + ground-truth JSON to tmp_dir."""
    import numpy as np
    from PIL import Image

    # 512x512 white canvas with a tiny black square (the "digit")
    canvas = np.full((512, 512), 255, dtype=np.uint8)
    canvas[100:130, 100:130] = 0
    img_path = tmp_dir / f"{scene_id}.png"
    Image.fromarray(canvas).save(str(img_path))

    gt = {
        "equation_type": "add",
        "symbols": [
            {
                "fine_label": "main_1",
                "yolo_class": "digit_main",
                "bbox": [100.0, 100.0, 130.0, 130.0],
                "row_index": 0,
                "col_index": 0,
            },
            {
                "fine_label": "op_plus",
                "yolo_class": "operator",
                "bbox": [150.0, 100.0, 180.0, 130.0],
                "row_index": 0,
                "col_index": 1,
            },
            {
                "fine_label": "main_2",
                "yolo_class": "digit_main",
                "bbox": [200.0, 100.0, 230.0, 130.0],
                "row_index": 0,
                "col_index": 2,
            },
            {
                "fine_label": "result_bar",
                "yolo_class": "result_bar",
                "bbox": [100.0, 140.0, 240.0, 145.0],
                "row_index": 1,
                "col_index": 0,
            },
            {
                "fine_label": "main_3",
                "yolo_class": "digit_main",
                "bbox": [150.0, 160.0, 180.0, 190.0],
                "row_index": 2,
                "col_index": 0,
            },
        ],
    }
    gt_path = tmp_dir / f"{scene_id}_gt.json"
    gt_path.write_text(json.dumps(gt), encoding="utf-8")
    return img_path, gt_path


def _write_manifest(tmp_dir: Path, rows: list[tuple[Path, Path]]) -> Path:
    df = pd.DataFrame(
        [{"image": str(img), "ground_truth": str(gt)} for img, gt in rows]
    )
    csv_path = tmp_dir / "manifest.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


class TestSceneGraphDatasetJitterMode(unittest.TestCase):
    """SceneGraphDataset integration: training vs val/test jitter gating."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self._tmp_path = Path(self._tmp)
        img, gt = _write_minimal_scene(self._tmp_path, "scene_001")
        self._manifest = _write_manifest(self._tmp_path, [(img, gt)])

    def _get_bboxes(self, dataset) -> list[list[float]]:
        """Return bbox coords as list-of-lists from the first scene."""
        data = dataset.get(0)
        return data.bbox.tolist()

    def test_val_dataset_bboxes_match_ground_truth(self) -> None:
        """Val dataset (training=False, default) must return exact GT bboxes."""
        from src.training.train_gnn import SceneGraphDataset
        ds = SceneGraphDataset(self._manifest)  # default: training=False, jitter=0
        bboxes = self._get_bboxes(ds)
        expected_x0s = [100.0, 150.0, 200.0, 100.0, 150.0]
        actual_x0s = [b[0] for b in bboxes]
        for exp, act in zip(expected_x0s, actual_x0s):
            self.assertAlmostEqual(act, exp, places=3)

    def test_val_dataset_ignores_jitter_px_param(self) -> None:
        """Even with a nonzero bbox_jitter_px, val (training=False) must not jitter."""
        from src.training.train_gnn import SceneGraphDataset
        # training=False (default) so jitter_px has no effect
        ds_clean = SceneGraphDataset(self._manifest, bbox_jitter_px=0.0)
        ds_jitter_param = SceneGraphDataset(self._manifest, bbox_jitter_px=5.0)
        bboxes_clean = self._get_bboxes(ds_clean)
        bboxes_jitter_param = self._get_bboxes(ds_jitter_param)
        for b_clean, b_jp in zip(bboxes_clean, bboxes_jitter_param):
            for c, j in zip(b_clean, b_jp):
                self.assertAlmostEqual(c, j, places=6,
                    msg="Val dataset must not jitter even when bbox_jitter_px > 0")

    def test_training_dataset_zero_jitter_matches_val(self) -> None:
        """training=True but jitter_px=0.0 must produce identical bboxes to val."""
        from src.training.train_gnn import SceneGraphDataset
        ds_val = SceneGraphDataset(self._manifest)
        ds_train_no_jitter = SceneGraphDataset(
            self._manifest, bbox_jitter_px=0.0, training=True
        )
        bboxes_val = self._get_bboxes(ds_val)
        bboxes_train = self._get_bboxes(ds_train_no_jitter)
        for bv, bt in zip(bboxes_val, bboxes_train):
            for cv, ct in zip(bv, bt):
                self.assertAlmostEqual(cv, ct, places=6)

    def test_training_dataset_bbox_invariants_with_jitter(self) -> None:
        """training=True + jitter_px=2.0: all bboxes must satisfy x0<x1, y0<y1, all in [0,512]."""
        from src.training.train_gnn import SceneGraphDataset
        ds = SceneGraphDataset(
            self._manifest, bbox_jitter_px=2.0, training=True
        )
        # Run multiple get() calls to exercise different rng draws.
        for _ in range(20):
            bboxes = self._get_bboxes(ds)
            for b in bboxes:
                x0, y0, x1, y1 = b
                self.assertLess(x0, x1, f"x swap in bbox {b}")
                self.assertLess(y0, y1, f"y swap in bbox {b}")
                self.assertGreaterEqual(x0, 0.0)
                self.assertGreaterEqual(y0, 0.0)
                self.assertLessEqual(x1, _CANVAS_SIZE)
                self.assertLessEqual(y1, _CANVAS_SIZE)


# ---------------------------------------------------------------------------
# Config round-trip test
# ---------------------------------------------------------------------------

class TestGnnConfigBboxJitterPx(unittest.TestCase):
    """GnnConfig must carry bbox_jitter_px with default 0.0."""

    def test_default_is_zero(self) -> None:
        from src.core.run_config import GnnConfig
        cfg = GnnConfig()
        self.assertAlmostEqual(cfg.bbox_jitter_px, 0.0)

    def test_value_assigned_correctly(self) -> None:
        from src.core.run_config import GnnConfig
        cfg = GnnConfig(bbox_jitter_px=3.5)
        self.assertAlmostEqual(cfg.bbox_jitter_px, 3.5)


if __name__ == "__main__":
    unittest.main()
