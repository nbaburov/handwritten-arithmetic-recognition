"""Tests for the real+synthetic fine-tune dataset builder (prepare_realtrain).

These prove the merged dataset is ACTUALLY consumable by the real loaders, not
merely well-shaped:

* the merged GNN ``train_manifest.csv`` loads through the real ``SceneGraphDataset``
  and yields valid PyG ``Data`` for both synthetic and real rows;
* the eq_type vocabulary fix is exercised end-to-end (a real subtraction scene
  maps to idx 1, not the silent idx-0 "add" the old code produced);
* the merged YOLO ``dataset.yaml`` + dirs pass ultralytics' dataset checks.

The synthetic source is a tiny hand-built dataset (a few scenes per split) so the
test is hermetic and fast, exercising the same code paths the 44k real dataset uses.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.core.ontology import YOLO_CLASS_NAMES, YOLO_NAME_TO_ID
from src.data_pipeline.prepare_realtrain import (
    DEFAULT_REAL_OVERSAMPLE,
    build_finetune_dataset,
    discover_real_samples,
)
from src.setmaker.exporters import export_train
from src.setmaker.types import AnnotationDraft
from src.training.train_gnn import SceneGraphDataset

CANVAS = 512


def _blank_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((CANVAS, CANVAS), 255, dtype=np.uint8)).save(path)


def _synth_gt(eq_type: str, symbols: list) -> dict:
    """A synthetic-shape gt.json payload (SHORT eq vocab, like the generator)."""
    return {
        "equation_type": eq_type,
        "completion_stage": "full",
        "split": "train",
        "case": "addition",
        "symbols": symbols,
    }


def _sym(fine_label: str, yolo_class: str, r: int, c: int, bbox) -> dict:
    return {
        "fine_label": fine_label,
        "glyph_key": fine_label,
        "yolo_class": yolo_class,
        "row_index": r,
        "col_index": c,
        "bbox": list(bbox),
        "equation_idx": 0,
    }


def _yolo_txt(symbols: list) -> str:
    lines = []
    for s in symbols:
        cid = YOLO_NAME_TO_ID[s["yolo_class"]]
        x0, y0, x1, y1 = s["bbox"]
        cx = ((x0 + x1) / 2) / CANVAS
        cy = ((y0 + y1) / 2) / CANVAS
        bw = max((x1 - x0) / CANVAS, 1e-6)
        bh = max((y1 - y0) / CANVAS, 1e-6)
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines)


def _build_tiny_synth(root: Path, n_per_split: int = 4) -> Path:
    """Create a minimal synthetic run dir with train/val/test manifests + dirs.

    Returns the run dir. Also wires the ``data/generated/synthetic/latest`` pointer.
    """
    synth_base = root / "data" / "generated" / "synthetic"
    run = synth_base / "20990101T000000Z_testrun"
    for split in ("train", "val", "test"):
        (run / split / "images").mkdir(parents=True, exist_ok=True)
        (run / split / "labels").mkdir(parents=True, exist_ok=True)
        (run / split / "ground_truth").mkdir(parents=True, exist_ok=True)

    eq_cycle = ["add", "subtract", "multiply", "divide"]
    for split in ("train", "val", "test"):
        rows = []
        for i in range(n_per_split):
            eq = eq_cycle[i % len(eq_cycle)]
            stem = f"{eq}_{split}_{i:06d}"
            syms = [
                _sym("main_2", "digit_main", 0, 1, (100, 100, 140, 170)),
                _sym("op_plus", "operator", 1, 0, (60, 180, 100, 250)),
                _sym("main_3", "digit_main", 1, 1, (100, 180, 140, 250)),
                _sym("result_bar", "result_bar", 2, 1, (60, 260, 140, 272)),
            ]
            img_p = run / split / "images" / f"{stem}.png"
            lbl_p = run / split / "labels" / f"{stem}.txt"
            gt_p = run / split / "ground_truth" / f"{stem}.gt.json"
            _blank_png(img_p)
            lbl_p.write_text(_yolo_txt(syms), encoding="utf-8")
            gt_p.write_text(json.dumps(_synth_gt(eq, syms)), encoding="utf-8")
            rows.append({
                "image": str(img_p.resolve()),
                "label": str(lbl_p.resolve()),
                "ground_truth": str(gt_p.resolve()),
                "box_count": len(syms),
                "case": "addition",
            })
        pd.DataFrame(rows).to_csv(run / f"{split}_manifest.csv", index=False)

    # latest pointer
    latest = synth_base / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    os.symlink(run, latest)
    return run


def _export_real(root: Path) -> None:
    """Write a few teammate-shaped real triples via the REAL exporter (LONG vocab)."""
    (root / "data" / "setmaker" / "train").mkdir(parents=True, exist_ok=True)
    img = np.full((CANVAS, CANVAS), 255, dtype=np.uint8)

    def mk(stem, kind, case, fops):
        drafts = [
            AnnotationDraft(
                bbox_px=(50 + c * 45, 40 + r * 70, 50 + c * 45 + 40, 40 + r * 70 + 60),
                fine_label=fl, row_index=r, col_index=c, equation_idx=0,
                confidence=1.0, source="human", flagged=False,
            )
            for (fl, r, c) in fops
        ]
        export_train(stem, img, drafts, {"equation_type": kind, "case": case, "completion_stage": "full"}, root)

    mk("rt-add-1", "addition", "addition",
       [("main_2", 0, 1), ("op_plus", 1, 0), ("main_3", 1, 1), ("result_bar", 2, 1)])
    mk("rt-sub-1", "subtraction", "subtraction",
       [("main_9", 0, 1), ("op_minus", 1, 0), ("main_4", 1, 1), ("result_bar", 2, 1)])
    mk("rt-divlong-1", "division", "division-long",
       [("op_divide", 0, 0), ("main_6", 0, 1), ("div_bracket", 0, 2), ("main_2", 1, 1), ("result_bar", 2, 1)])


class TestDiscoverRealSamples(unittest.TestCase):
    def test_discovers_valid_triples(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _export_real(root)
            samples = discover_real_samples(root)
            self.assertEqual(len(samples), 3)
            kinds = {s.equation_type for s in samples}
            self.assertEqual(kinds, {"addition", "subtraction", "division"})
            for s in samples:
                self.assertTrue(s.png.is_file())
                self.assertTrue(s.gt.is_file())
                self.assertTrue(s.txt.is_file())
                self.assertGreater(s.n_symbols, 0)

    def test_empty_dir_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(discover_real_samples(Path(td)), [])

    def test_missing_companion_png_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            d = root / "data" / "setmaker" / "train"
            d.mkdir(parents=True)
            # gt + txt but no png
            (d / "orphan.gt.json").write_text(
                json.dumps(_synth_gt("add", [_sym("main_1", "digit_main", 0, 0, (1, 1, 20, 30))])),
                encoding="utf-8",
            )
            (d / "orphan.txt").write_text("0 0.5 0.5 0.1 0.1", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                discover_real_samples(root)

    def test_wrong_image_size_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            d = root / "data" / "setmaker" / "train"
            d.mkdir(parents=True)
            Image.fromarray(np.full((64, 64), 255, dtype=np.uint8)).save(d / "bad.png")
            (d / "bad.gt.json").write_text(
                json.dumps(_synth_gt("add", [_sym("main_1", "digit_main", 0, 0, (1, 1, 20, 30))])),
                encoding="utf-8",
            )
            (d / "bad.txt").write_text("0 0.5 0.5 0.1 0.1", encoding="utf-8")
            with self.assertRaises(ValueError):
                discover_real_samples(root)


class TestBuildFinetuneDataset(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self.synth_run = _build_tiny_synth(self.root, n_per_split=4)
        _export_real(self.root)

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_no_synthetic_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _export_real(root)  # real but no synthetic
            with self.assertRaises(RuntimeError):
                build_finetune_dataset(root)

    def test_no_real_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _build_tiny_synth(root)  # synthetic but no real
            with self.assertRaises(RuntimeError):
                build_finetune_dataset(root)

    def test_oversample_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            build_finetune_dataset(self.root, real_oversample=0)

    def test_summary_and_provenance(self) -> None:
        s = build_finetune_dataset(self.root, real_oversample=DEFAULT_REAL_OVERSAMPLE)
        self.assertEqual(s["real_sample_count"], 3)
        self.assertEqual(s["real_oversample"], DEFAULT_REAL_OVERSAMPLE)
        self.assertEqual(
            s["real_by_equation_kind"],
            {"addition": 1, "subtraction": 1, "division": 1},
        )
        # provenance file written
        prov = Path(s["run_dir"]) / "finetune_manifest.json"
        self.assertTrue(prov.is_file())
        loaded = json.loads(prov.read_text())
        self.assertEqual(loaded["real_sample_count"], 3)

    def test_merged_gnn_manifest_loads_through_scene_graph_dataset(self) -> None:
        s = build_finetune_dataset(self.root, real_oversample=3)
        man = Path(s["gnn_train_manifest"])
        df = pd.read_csv(man)
        # 4 synth train rows + 3 real * 3 oversample = 4 + 9 = 13
        self.assertEqual(len(df), 4 + 3 * 3)
        self.assertEqual(list(df.columns), ["image", "ground_truth"])

        ds = SceneGraphDataset(man)
        self.assertEqual(len(ds), 13)
        # every item builds a valid PyG Data with matching node/label counts
        for i in range(len(ds)):
            d = ds.get(i)
            self.assertEqual(d.x.shape[1], 13)
            self.assertEqual(d.x.shape[0], d.y_fine.shape[0])
            self.assertEqual(d.y_eq.shape[0], 1)

    def test_eq_type_fix_subtraction_maps_to_idx1(self) -> None:
        """The real subtraction scene must supervise eq_type as idx 1, not silent 0."""
        s = build_finetune_dataset(self.root, real_oversample=1)
        man = Path(s["gnn_train_manifest"])
        ds = SceneGraphDataset(man)
        eq_by_kind = {}
        for i in range(len(ds)):
            img, gt_path = ds._rows[i]  # noqa: SLF001 (white-box check of label wiring)
            gt = json.loads(Path(gt_path).read_text())
            eq_by_kind.setdefault(gt["equation_type"], set()).add(ds.get(i).y_eq.item())
        # real LONG-vocab kinds present and correctly mapped
        self.assertEqual(eq_by_kind.get("subtraction"), {1})
        self.assertEqual(eq_by_kind.get("addition"), {0})
        self.assertEqual(eq_by_kind.get("division"), {3})
        # synth SHORT-vocab still correct
        self.assertEqual(eq_by_kind.get("subtract"), {1})
        self.assertEqual(eq_by_kind.get("multiply"), {2})

    def test_val_test_manifests_are_pure_synthetic(self) -> None:
        s = build_finetune_dataset(self.root, real_oversample=3)
        for split in ("val", "test"):
            merged = pd.read_csv(s[f"gnn_{split}_manifest"])
            synth = pd.read_csv(self.synth_run / f"{split}_manifest.csv")
            self.assertEqual(len(merged), len(synth))

    def test_merged_yolo_dirs_valid(self) -> None:
        s = build_finetune_dataset(self.root, real_oversample=3)
        run = Path(s["run_dir"])
        # dataset.yaml exists with correct class count
        yaml_p = run / "dataset.yaml"
        self.assertTrue(yaml_p.is_file())
        text = yaml_p.read_text()
        self.assertIn(f"nc: {len(YOLO_CLASS_NAMES)}", text)
        self.assertIn(f"path: {run}", text)

        # every split has matching images + labels; real train images oversampled
        for split in ("train", "val", "test"):
            imgs = sorted((run / split / "images").glob("*.png"))
            lbls = sorted((run / split / "labels").glob("*.txt"))
            self.assertEqual(len(imgs), len(lbls), f"{split} img/label count mismatch")
            self.assertGreater(len(imgs), 0)

        real_imgs = list((run / "train" / "images").glob("rt-*__rt*.png"))
        self.assertEqual(len(real_imgs), s["yolo_real_train_pairs"])
        self.assertEqual(s["yolo_real_train_pairs"], 3 * 3)  # 3 real * 3 oversample

    def test_merged_yolo_loads_through_ultralytics(self) -> None:
        """ultralytics must scan the merged dir with zero corrupt labels."""
        try:
            from ultralytics.data.utils import check_det_dataset
            from ultralytics.data.dataset import YOLODataset
        except Exception as exc:  # pragma: no cover - ultralytics always present here
            self.skipTest(f"ultralytics unavailable: {exc}")

        s = build_finetune_dataset(self.root, real_oversample=2)
        run = Path(s["run_dir"])
        data = check_det_dataset(str(run / "dataset.yaml"))
        ds = YOLODataset(img_path=str(run / "train" / "images"), data=data, task="detect")
        # 4 synth + 3 real * 2 = 10 train items, all with at least one box
        self.assertEqual(len(ds.im_files), 4 + 3 * 2)
        with_bbox = sum(int(len(lab["bboxes"]) > 0) for lab in ds.labels)
        self.assertEqual(with_bbox, len(ds.labels))


if __name__ == "__main__":
    unittest.main()
