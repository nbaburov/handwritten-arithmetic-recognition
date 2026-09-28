from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.core.config import DataPrepConfig
from src.data_pipeline.validate_dataset import run_full_data_prep


def _write_png(path: Path, value: int) -> None:
    base = np.arange(28 * 28, dtype=np.uint16).reshape(28, 28)
    arr = ((base + value) % 256).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, arr, cmap="gray", vmin=0, vmax=255)


class TestIntegrationPrepare(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.project_root = Path(self.tmpdir.name)
        self.config = DataPrepConfig(
            project_root=self.project_root,
            data_pool_emnist_28_dir=self.project_root / "data" / "raw" / "pool_emnist_28",
            data_processed_dir=self.project_root / "data" / "generated" / "processed",
            reports_dir=self.project_root / "reports",
            required_labels=None,
        )

        classes = {"0": 18, "1": 18, "plus": 18}
        for label_idx, (label, count) in enumerate(classes.items()):
            for i in range(count):
                _write_png(
                    self.project_root / "data" / "raw" / "pool_emnist_28" / label / f"{label}_{i}.png",
                    (label_idx * 100 + i * 7 + len(label)) % 255,
                )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_full_prepare_outputs_and_leakage(self) -> None:
        run_full_data_prep(self.config)

        stage1 = self.project_root / "data" / "generated" / "processed" / "yolo"
        stage2 = self.project_root / "data" / "generated" / "processed" / "stage2"
        reports_prep = self.project_root / "reports" / "prep"

        expected_files = [
            stage1 / "symbol_assets_manifest.csv",
            stage1 / "symbol_class_balance.csv",
            stage2 / "manifests" / "train.csv",
            stage2 / "manifests" / "val.csv",
            stage2 / "manifests" / "test.csv",
            stage2 / "arrays" / "train.npz",
            stage2 / "arrays" / "val.npz",
            stage2 / "arrays" / "test.npz",
            reports_prep / "class-distribution.csv",
            reports_prep / "split-leakage-check.csv",
            reports_prep / "data-prep-summary.md",
        ]
        for file in expected_files:
            self.assertTrue(file.exists(), f"missing file: {file}")
            self.assertGreater(file.stat().st_size, 0, f"empty file: {file}")

        train = pd.read_csv(stage2 / "manifests" / "train.csv")
        val = pd.read_csv(stage2 / "manifests" / "val.csv")
        test = pd.read_csv(stage2 / "manifests" / "test.csv")
        for name, frame in (("train", train), ("val", val), ("test", test)):
            self.assertIn("yolo_class", frame.columns, f"missing yolo_class in {name}")
            self.assertIn("role", frame.columns, f"missing role in {name}")
            self.assertIn("glyph_key", frame.columns, f"missing glyph_key in {name}")
        sample = train.iloc[0]
        if sample["label"] == "0":
            self.assertEqual(sample["yolo_class"], "digit_main")
            self.assertEqual(sample["role"], "main")
            self.assertEqual(sample["glyph_key"], "0")
        elif sample["label"] == "plus":
            self.assertEqual(sample["yolo_class"], "operator")
            self.assertEqual(sample["role"], "operator")

        sym = pd.read_csv(stage1 / "symbol_assets_manifest.csv")
        self.assertIn("yolo_class", sym.columns)
        self.assertIn("role", sym.columns)
        self.assertIn("glyph_key", sym.columns)

        self.assertEqual(train["sha256"].isin(val["sha256"]).sum(), 0)
        self.assertEqual(train["sha256"].isin(test["sha256"]).sum(), 0)
        self.assertEqual(val["sha256"].isin(test["sha256"]).sum(), 0)

    def test_idempotent_manifest_order(self) -> None:
        run_full_data_prep(self.config)
        first = pd.read_csv(self.project_root / "data" / "generated" / "processed" / "stage2" / "manifests" / "train.csv")
        run_full_data_prep(self.config)
        second = pd.read_csv(self.project_root / "data" / "generated" / "processed" / "stage2" / "manifests" / "train.csv")
        self.assertListEqual(first["sha256"].tolist(), second["sha256"].tolist())


if __name__ == "__main__":
    unittest.main()
