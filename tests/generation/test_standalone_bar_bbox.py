"""Tests for standalone_bar empty-label fix (iter10-R6 Step 3).

Bug: _apply_scene_knobs fired missing_structural_prob on standalone_bar scenes,
which have only one token (result_bar). Dropping it left tokens=[] -> empty
annotations -> 0-byte YOLO label file.

Fix: guard the drop so it only executes when len(tokens) > 1.

Two test classes:
  TestStandaloneBarLayoutMinWidth  -- pure layout-level, no manifest needed.
  TestStandaloneBarLabelNonempty   -- rendering-level, requires symbol manifest.
"""
from __future__ import annotations

import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import skipUnless

from src.generation.layouts import SceneCase, layout_standalone_bar
from src.generation.synth_yolo import _apply_scene_knobs


# ---------------------------------------------------------------------------
# Helper: manifest path (same guard used by test_generation.py)
# ---------------------------------------------------------------------------

MANIFEST = (
    Path(__file__).parents[1]
    / "data"
    / "generated"
    / "processed"
    / "yolo"
    / "symbol_assets_manifest.csv"
)


class TestStandaloneBarLayoutMinWidth(unittest.TestCase):
    """Layout-level regression: no standalone_bar layout produces a zero-token
    result when scene knobs with missing_structural_prob are applied.

    This is a pure-Python test — no pool manifest required.
    """

    def _rng(self, seed: int) -> random.Random:
        return random.Random(seed)

    def test_apply_scene_knobs_never_empties_standalone_bar_tokens(self) -> None:
        """_apply_scene_knobs with missing_structural_prob=1.0 must NOT drop the
        sole token when len(tokens) == 1 (standalone_bar scene).

        Runs 1000 seeds with missing_structural_prob forced to 1.0 so the guard
        fires on every iteration.
        """
        from src.core.run_config import SceneConfig

        scene_cfg = SceneConfig(missing_structural_prob=1.0)

        for seed in range(1000):
            rng = self._rng(seed)
            _, tokens = layout_standalone_bar(rng)
            self.assertEqual(
                len(tokens),
                1,
                f"seed={seed}: layout_standalone_bar must produce exactly 1 token before knobs",
            )
            rng2 = self._rng(seed + 10000)
            result_tokens = _apply_scene_knobs(tokens, rng2, scene_cfg)
            self.assertEqual(
                len(result_tokens),
                1,
                f"seed={seed}: _apply_scene_knobs emptied tokens for standalone_bar "
                f"(missing_structural_prob=1.0) — empty label file would be written",
            )
            self.assertEqual(
                result_tokens[0].flattened_label,
                "result_bar",
                f"seed={seed}: sole token must remain result_bar after knobs",
            )

    def test_normal_scenes_still_allow_structural_drop(self) -> None:
        """Ensure the guard does not block the drop for multi-token scenes.

        Uses a 2-token minimal scene (digit + result_bar). With
        missing_structural_prob=1.0 the result_bar must be dropped leaving 1 token.
        """
        from src.core.run_config import SceneConfig
        from src.generation.layouts import LayoutToken

        scene_cfg = SceneConfig(missing_structural_prob=1.0, wrong_result_prob=0.0, wrong_operator_prob=0.0)
        digit_tok = LayoutToken(
            row=0,
            col=0,
            flattened_label="main_3",
            glyph_key="3",
            yolo_class_name="digit_main",
            token_scale=1.0,
        )
        bar_tok = LayoutToken(
            row=1,
            col=0,
            flattened_label="result_bar",
            glyph_key="result_bar",
            yolo_class_name="result_bar",
            token_scale=1.0,
        )
        tokens = [digit_tok, bar_tok]
        rng = self._rng(42)
        result = _apply_scene_knobs(tokens, rng, scene_cfg)
        # With missing_structural_prob=1.0 the bar must have been dropped.
        self.assertEqual(len(result), 1, "2-token scene: result_bar should be dropped by knob")
        self.assertEqual(result[0].flattened_label, "main_3")


@skipUnless(MANIFEST.exists(), "symbol manifest not available — run validate first")
class TestStandaloneBarLabelNonempty(unittest.TestCase):
    """Rendering-level regression: all standalone_bar YOLO label files must be
    non-empty after the fix.

    Renders 100 standalone_bar scenes via build_synthetic_yolo_dataset and
    asserts no .txt label file is 0 bytes.

    Requires symbol_assets_manifest.csv — skipped in CI without it.
    """

    def test_all_label_files_nonempty(self) -> None:
        from src.core.config import DataPrepConfig
        from src.generation.synth_yolo import build_synthetic_yolo_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            tmp_config = DataPrepConfig.from_project_root(tmp_root)
            tmp_config.yolo_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(MANIFEST, tmp_config.yolo_dir / "symbol_assets_manifest.csv")

            out_dir = tmp_root / "synth_standalone_bar"
            out_dir.mkdir(parents=True, exist_ok=True)

            build_synthetic_yolo_dataset(
                tmp_config,
                SceneCase.standalone_bar,
                split_name="train",
                images_per_split=100,
                out_dir=out_dir,
            )

            labels_dir = out_dir / "train" / "labels"
            label_files = list(labels_dir.glob("*.txt"))
            self.assertGreater(len(label_files), 0, "No label files generated")

            empty_files = [f for f in label_files if f.stat().st_size == 0]
            self.assertEqual(
                len(empty_files),
                0,
                f"Found {len(empty_files)} empty label files: "
                + ", ".join(f.name for f in empty_files[:5]),
            )

            # Spot-check: each label file must contain a result_bar annotation (class 4).
            from src.core.ontology import YOLO_NAME_TO_ID
            bar_class_id = YOLO_NAME_TO_ID.get("result_bar")
            if bar_class_id is not None:
                for lf in label_files:
                    first_line = lf.read_text(encoding="utf-8").strip().splitlines()[0]
                    cls_id = int(first_line.split()[0])
                    self.assertEqual(
                        cls_id,
                        bar_class_id,
                        f"{lf.name}: expected class {bar_class_id} (result_bar), got {cls_id}",
                    )
