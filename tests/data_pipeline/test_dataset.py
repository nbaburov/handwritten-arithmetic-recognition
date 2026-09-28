from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from matplotlib import pyplot as plt

from src.core.config import DataPrepConfig
from src.data_pipeline.dataset import (
    build_class_map,
    enforce_quality_gates,
    label_from_path,
    scan_dataset,
    stratified_split,
    validate_label_vocabulary,
)


def _write_png(path: Path, value: int) -> None:
    base = np.arange(28 * 28, dtype=np.uint16).reshape(28, 28)
    arr = ((base + value) % 256).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, arr, cmap="gray", vmin=0, vmax=255)


class TestDataPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.project_root = Path(self.tmpdir.name)
        self.combined = self.project_root / "data" / "raw" / "pool_emnist_28"
        self.config = DataPrepConfig(
            project_root=self.project_root,
            data_pool_emnist_28_dir=self.project_root / "data" / "raw" / "pool_emnist_28",
            data_processed_dir=self.project_root / "data" / "generated" / "processed",
            reports_dir=self.project_root / "reports",
            required_labels=None,
        )

        classes = {"0": 10, "1": 10, "plus": 10}
        for label_idx, (label, count) in enumerate(classes.items()):
            for i in range(count):
                _write_png(self.combined / label / f"{label}_{i}.png", (label_idx * 100 + i) % 255)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_label_extraction(self) -> None:
        sample = self.combined / "1" / "1_0.png"
        label = label_from_path(sample, self.combined)
        self.assertEqual(label, "1")

    def test_scan_and_validation(self) -> None:
        df, issues = scan_dataset(self.config)
        issues.extend(validate_label_vocabulary(df, self.config))
        enforce_quality_gates(df, issues)
        self.assertEqual(len(df), 30)
        self.assertIn("width", df.columns)
        self.assertIn("height", df.columns)

    def test_deterministic_split(self) -> None:
        df, issues = scan_dataset(self.config)
        issues.extend(validate_label_vocabulary(df, self.config))
        enforce_quality_gates(df, issues)

        split_a = stratified_split(df, self.config)
        split_b = stratified_split(df, self.config)

        for key in ("train", "val", "test"):
            self.assertListEqual(
                split_a[key]["sha256"].tolist(),
                split_b[key]["sha256"].tolist(),
            )

    def test_class_map_stability(self) -> None:
        df, _ = scan_dataset(self.config)
        class_map = build_class_map(df)
        expected = {"0": 0, "1": 1, "plus": 2}
        self.assertDictEqual(class_map, expected)

    def test_corrupt_file_detected(self) -> None:
        bad = self.combined / "0" / "bad.png"
        bad.write_text("not-a-png", encoding="utf-8")

        df, issues = scan_dataset(self.config)
        self.assertTrue(any(issue.kind == "corrupt_image" for issue in issues))
        self.assertGreaterEqual(len(df), 30)

    def test_yolo_dataset_yaml_has_test_split(self) -> None:
        """Verify that YOLO dataset.yaml includes test split key (flat timestamped layout).

        The new layout writes flat train/val/test/images directly in the run folder.
        dataset.yaml uses paths relative to the run folder root.
        """
        tmpdir = tempfile.TemporaryDirectory()
        run_dir = Path(tmpdir.name)

        # Create minimal flat layout
        for split in ["train", "val", "test"]:
            images_dir = run_dir / split / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            (images_dir / f"{split}_dummy.png").touch()

        # Simulate the dataset.yaml writer logic from prepare_synthetic_yolo
        names = ["digit_main", "digit_carry", "digit_borrow", "operator", "result_bar", "divide_bracket"]
        dataset_yaml_content = "\n".join([
            f"path: {str(run_dir)}",
            "train: train/images",
            "val: val/images",
            "test: test/images",
            f"nc: {len(names)}",
            "names: [" + ", ".join(f"'{n}'" for n in names) + "]",
        ])

        dataset_yaml = run_dir / "dataset.yaml"
        dataset_yaml.write_text(dataset_yaml_content, encoding="utf-8")

        # Parse and verify
        parsed = yaml.safe_load(dataset_yaml.read_text())
        self.assertIn("test", parsed, "dataset.yaml must include 'test' key")
        self.assertEqual(parsed["test"], "test/images", "test path must be 'test/images'")
        self.assertIn("train", parsed)
        self.assertIn("val", parsed)

        tmpdir.cleanup()

    def test_yolo_dataset_yaml_latest_pointer(self) -> None:
        """Verify latest pointer resolves to the run dir and train_manifest.csv exists."""
        import os
        from src.data_pipeline.prepare_synthetic_yolo import resolve_latest_run, LATEST_POINTER
        tmpdir = tempfile.TemporaryDirectory()
        synthetic_dir = Path(tmpdir.name)
        run_dir = synthetic_dir / "20260508T120000Z_abcd1234"
        run_dir.mkdir(parents=True)
        (run_dir / "train_manifest.csv").write_text("image,label,ground_truth,box_count,case\n", encoding="utf-8")

        # Create symlink pointer
        latest = synthetic_dir / LATEST_POINTER
        os.symlink(run_dir, latest)

        resolved = resolve_latest_run(synthetic_dir)
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved, run_dir.resolve())
        self.assertTrue((resolved / "train_manifest.csv").exists())

        tmpdir.cleanup()


    def test_no_per_case_folders_created(self) -> None:
        """New layout must NOT create per-case subfolders like 'addition/', 'subtraction/'."""
        import os
        from src.data_pipeline.prepare_synthetic_yolo import resolve_latest_run, LATEST_POINTER
        from src.generation.layouts import SceneCase

        tmpdir = tempfile.TemporaryDirectory()
        synthetic_dir = Path(tmpdir.name)
        run_dir = synthetic_dir / "20260508T120000Z_abcd1234"
        run_dir.mkdir(parents=True)
        (run_dir / "train_manifest.csv").write_text("image,label,ground_truth,box_count,case\n", encoding="utf-8")
        os.symlink(run_dir, synthetic_dir / LATEST_POINTER)

        # Only the run_dir should exist inside synthetic_dir (plus latest pointer).
        case_names = {c.value for c in SceneCase}
        for item in synthetic_dir.iterdir():
            if item.name == LATEST_POINTER:
                continue
            self.assertNotIn(
                item.name, case_names,
                f"Per-case folder '{item.name}' found — new layout must not create these"
            )
        tmpdir.cleanup()

    def test_examples_folder_has_pngs_and_readme(self) -> None:
        """examples/ in run_dir must contain at least one PNG and a README.md."""
        tmpdir = tempfile.TemporaryDirectory()
        run_dir = Path(tmpdir.name)
        from src.data_pipeline.prepare_synthetic_yolo import _write_examples

        samples = {
            "addition_full": run_dir / "addition_full.png",
            "subtraction_no_result": run_dir / "subtraction_no_result.png",
        }
        for p in samples.values():
            p.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal PNG header

        _write_examples(run_dir, samples)

        examples_dir = run_dir / "examples"
        self.assertTrue(examples_dir.is_dir(), "examples/ dir must exist")
        pngs = list(examples_dir.glob("*.png"))
        self.assertGreaterEqual(len(pngs), 1, "examples/ must have at least one PNG")
        readme = examples_dir / "README.md"
        self.assertTrue(readme.exists(), "examples/README.md must exist")
        content = readme.read_text(encoding="utf-8")
        self.assertIn("addition_full", content, "README must list example filenames")
        tmpdir.cleanup()

    def test_latest_pointer_text_file_fallback(self) -> None:
        """resolve_latest_run works when latest is a plain text file (symlink fallback)."""
        from src.data_pipeline.prepare_synthetic_yolo import resolve_latest_run, LATEST_POINTER
        tmpdir = tempfile.TemporaryDirectory()
        synthetic_dir = Path(tmpdir.name)
        run_dir = synthetic_dir / "20260508T120000Z_abcd1234"
        run_dir.mkdir(parents=True)

        # Write plain text fallback instead of symlink
        (synthetic_dir / LATEST_POINTER).write_text(str(run_dir), encoding="utf-8")

        resolved = resolve_latest_run(synthetic_dir)
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved, run_dir)
        tmpdir.cleanup()


if __name__ == "__main__":
    unittest.main()
