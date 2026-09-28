"""Tests that class_to_idx.json is written inside the run dir, not its parent.

Verifies fix C-3: the _top_dir assignment was changed from out_dir.parent to
out_dir so the file lands in the timestamped run directory alongside the
aggregate manifest rather than polluting the parent synthetic/ directory.
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import skipUnless

MANIFEST = (
    Path(__file__).parents[2]
    / "data"
    / "generated"
    / "processed"
    / "yolo"
    / "symbol_assets_manifest.csv"
)


class TestClassToIdxPath(unittest.TestCase):
    """Assert class_to_idx.json lands inside out_dir, not out_dir.parent."""

    @skipUnless(MANIFEST.exists(), "symbol manifest not available — run validate first")
    def test_class_to_idx_is_inside_run_dir(self) -> None:
        from src.core.config import DataPrepConfig
        from src.generation.layouts import SceneCase
        from src.generation.synth_yolo import build_synthetic_yolo_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            tmp_config = DataPrepConfig.from_project_root(tmp_root)
            tmp_config.yolo_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(MANIFEST, tmp_config.yolo_dir / "symbol_assets_manifest.csv")

            out_dir = tmp_root / "synth_run"
            out_dir.mkdir(parents=True, exist_ok=True)

            result = build_synthetic_yolo_dataset(
                tmp_config,
                SceneCase.addition,
                split_name="train",
                images_per_split=2,
                out_dir=out_dir,
            )

            class_to_idx_path = result["class_to_idx"]

            # C-3 fix: must be inside out_dir, not out_dir.parent.
            self.assertEqual(
                class_to_idx_path.parent,
                out_dir,
                msg=(
                    f"class_to_idx.json landed in {class_to_idx_path.parent} "
                    f"but expected it inside out_dir={out_dir}. "
                    f"C-3 fix (out_dir.parent → out_dir) may not have taken effect."
                ),
            )

            # The file must also exist.
            self.assertTrue(
                class_to_idx_path.exists(),
                msg=f"class_to_idx.json not found at {class_to_idx_path}.",
            )

    @skipUnless(MANIFEST.exists(), "symbol manifest not available — run validate first")
    def test_class_to_idx_not_written_to_parent(self) -> None:
        """Regression: old bug wrote to out_dir.parent (the synthetic/ folder)."""
        from src.core.config import DataPrepConfig
        from src.generation.layouts import SceneCase
        from src.generation.synth_yolo import build_synthetic_yolo_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            tmp_config = DataPrepConfig.from_project_root(tmp_root)
            tmp_config.yolo_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(MANIFEST, tmp_config.yolo_dir / "symbol_assets_manifest.csv")

            out_dir = tmp_root / "synth_run"
            out_dir.mkdir(parents=True, exist_ok=True)

            build_synthetic_yolo_dataset(
                tmp_config,
                SceneCase.addition,
                split_name="train",
                images_per_split=2,
                out_dir=out_dir,
            )

            # Pre-fix this file would appear at out_dir.parent / "class_to_idx.json".
            wrong_path = out_dir.parent / "class_to_idx.json"
            self.assertFalse(
                wrong_path.exists(),
                msg=(
                    f"class_to_idx.json found at the wrong location {wrong_path}. "
                    f"C-3 fix did not prevent the parent-dir write."
                ),
            )
