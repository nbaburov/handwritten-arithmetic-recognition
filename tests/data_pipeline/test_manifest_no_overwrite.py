"""Tests that per-case manifest paths are unique and aggregation covers all cases.

Verifies fix C-1: each (case, split) pair writes to a distinct manifest file
whose name contains case.value, preventing the overwrite bug where every case
clobbered the same file and only the last case's data survived in the aggregate.
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import skipUnless

import pandas as pd

MANIFEST = (
    Path(__file__).parents[2]
    / "data"
    / "generated"
    / "processed"
    / "yolo"
    / "symbol_assets_manifest.csv"
)


class TestManifestNoOverwrite(unittest.TestCase):
    """Run a tiny generation across 3 cases and assert manifest uniqueness."""

    @skipUnless(MANIFEST.exists(), "symbol manifest not available — run validate first")
    def test_per_case_manifest_paths_are_unique(self) -> None:
        from src.core.config import DataPrepConfig
        from src.generation.layouts import SceneCase
        from src.generation.synth_yolo import build_synthetic_yolo_dataset

        cases = [SceneCase.addition, SceneCase.subtraction, SceneCase.multiplication_simple]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            tmp_config = DataPrepConfig.from_project_root(tmp_root)
            tmp_config.yolo_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(MANIFEST, tmp_config.yolo_dir / "symbol_assets_manifest.csv")

            out_dir = tmp_root / "synth_run"
            out_dir.mkdir(parents=True, exist_ok=True)

            manifest_paths: list[Path] = []
            for case in cases:
                result = build_synthetic_yolo_dataset(
                    tmp_config,
                    case,
                    split_name="train",
                    images_per_split=2,
                    out_dir=out_dir,
                )
                manifest_paths.append(result["manifest"])

            # Each path must be distinct (C-1 fix: case.value prefix).
            self.assertEqual(
                len(manifest_paths),
                len(set(manifest_paths)),
                msg="Manifest paths are not unique — C-1 fix not effective.",
            )

            # Each manifest filename must contain the case value.
            for case, path in zip(cases, manifest_paths):
                self.assertIn(
                    case.value,
                    path.name,
                    msg=f"Manifest filename {path.name!r} does not contain case value {case.value!r}.",
                )

    @skipUnless(MANIFEST.exists(), "symbol manifest not available — run validate first")
    def test_aggregate_manifest_covers_all_cases(self) -> None:
        """After aggregation, combined manifest must have one row-group per case."""
        from src.core.config import DataPrepConfig
        from src.generation.layouts import SceneCase
        from src.generation.synth_yolo import build_synthetic_yolo_dataset

        cases = [SceneCase.addition, SceneCase.subtraction, SceneCase.multiplication_simple]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            tmp_config = DataPrepConfig.from_project_root(tmp_root)
            tmp_config.yolo_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(MANIFEST, tmp_config.yolo_dir / "symbol_assets_manifest.csv")

            out_dir = tmp_root / "synth_run"
            out_dir.mkdir(parents=True, exist_ok=True)

            per_case_manifests: list[Path] = []
            for case in cases:
                result = build_synthetic_yolo_dataset(
                    tmp_config,
                    case,
                    split_name="train",
                    images_per_split=2,
                    out_dir=out_dir,
                )
                per_case_manifests.append(result["manifest"])

            # Simulate the aggregation step in prepare_synthetic_yolo.py.
            dfs = [pd.read_csv(p) for p in per_case_manifests]
            combined = pd.concat(dfs, ignore_index=True)

            actual_cases = combined["case"].nunique()
            self.assertEqual(
                actual_cases,
                len(cases),
                msg=(
                    f"Aggregate manifest covers {actual_cases} case(s) "
                    f"but {len(cases)} were generated. "
                    f"Missing: {set(c.value for c in cases) - set(combined['case'].unique())}"
                ),
            )
