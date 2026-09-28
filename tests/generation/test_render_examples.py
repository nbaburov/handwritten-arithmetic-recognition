"""Tests for canonical examples folder produced by the render-examples subcommand.

These tests are skipped unless data/generated/synthetic/latest/examples/ exists.
Run them after: python -m src generate && python -m src render-examples
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.generation.layouts_types import SceneCase
from src.generation.completion_stages import valid_stages_for_case

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_EXAMPLES_DIR = _PROJECT_ROOT / "data" / "generated" / "synthetic" / "latest" / "examples"
_EXAMPLES_EXIST = _EXAMPLES_DIR.exists() and len(list(_EXAMPLES_DIR.glob("*.png"))) > 0


def _expected_render_count() -> int:
    """Compute expected canonical-example PNG count from SceneCase enum + valid stages.

    render-examples emits one PNG per (case, valid_stage) pair, so this is the
    authoritative count. Avoids hardcoded magic numbers that go stale every
    time new SceneCases are added (iter10 W10-ROBUSTNESS added 3 cases).
    """
    return sum(len(valid_stages_for_case(case)) for case in SceneCase)


EXPECTED_COUNT = _expected_render_count()


@unittest.skipUnless(_EXAMPLES_EXIST, "data/generated/synthetic/latest/examples/ not present — run generate + render first")
class TestCanonicalExamples(unittest.TestCase):

    def _examples_dir(self) -> Path:
        latest = _PROJECT_ROOT / "data" / "generated" / "synthetic" / "latest"
        self.assertTrue(latest.exists(), "data/generated/synthetic/latest symlink missing")
        resolved = latest.resolve()
        self.assertTrue(resolved.exists(), f"latest symlink dangling → {resolved}")
        return resolved / "examples"

    def test_png_count_matches_enum_derived_expected(self) -> None:
        examples_dir = self._examples_dir()
        pngs = list(examples_dir.glob("*.png"))
        self.assertEqual(
            len(pngs), EXPECTED_COUNT,
            f"Expected {EXPECTED_COUNT} PNGs (sum of valid_stages_for_case across SceneCase), found {len(pngs)}. "
            f"Run `python -m src render-examples` after `generate` to refresh the canonical gallery."
        )

    def test_manifest_count_matches_enum_derived_expected(self) -> None:
        examples_dir = self._examples_dir()
        manifest_path = examples_dir / "manifest.json"
        self.assertTrue(manifest_path.exists(), "manifest.json missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            len(manifest), EXPECTED_COUNT,
            f"manifest has {len(manifest)} entries, expected {EXPECTED_COUNT} (from valid_stages_for_case sum)"
        )

    def test_manifest_no_duplicate_case_names(self) -> None:
        examples_dir = self._examples_dir()
        manifest_path = examples_dir / "manifest.json"
        if not manifest_path.exists():
            self.skipTest("manifest.json missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(len(manifest), len(set(manifest.keys())), "Duplicate case names in manifest")

    def test_all_pngs_are_512x512_uint8(self) -> None:
        examples_dir = self._examples_dir()
        for png_path in sorted(examples_dir.glob("*.png")):
            arr = np.array(Image.open(png_path).convert("L"), dtype=np.uint8)
            self.assertEqual(
                arr.shape, (512, 512),
                f"{png_path.name}: expected (512, 512), got {arr.shape}"
            )
            self.assertEqual(arr.dtype, np.uint8, f"{png_path.name}: expected uint8")

    def test_all_pngs_pass_midtone_check(self) -> None:
        """All examples must be nearly binary (no erosion artifacts from stroke normalisation)."""
        examples_dir = self._examples_dir()
        failures = []
        for png_path in sorted(examples_dir.glob("*.png")):
            arr = np.array(Image.open(png_path).convert("L"), dtype=np.uint8)
            ink_px = int(np.sum(arr < 20))
            mid_px = int(np.sum((arr > 20) & (arr < 235)))
            ratio = mid_px / max(ink_px, 1)
            if ratio > 0.05:
                failures.append(f"{png_path.name}: mid/ink={ratio:.3f}")
        self.assertEqual([], failures, f"Midtone failures (expect binary images):\n" + "\n".join(failures))

    def test_readme_exists(self) -> None:
        examples_dir = self._examples_dir()
        readme = examples_dir / "README.md"
        self.assertTrue(readme.exists(), "README.md missing from examples dir")

    def test_manifest_each_entry_has_required_keys(self) -> None:
        examples_dir = self._examples_dir()
        manifest_path = examples_dir / "manifest.json"
        if not manifest_path.exists():
            self.skipTest("manifest.json missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required_keys = {"file", "scene_case", "completion_stage", "force_wrong_answer", "seed", "description"}
        for name, entry in manifest.items():
            missing = required_keys - set(entry.keys())
            self.assertEqual(set(), missing, f"Manifest entry '{name}' missing keys: {missing}")

    def test_manifest_png_files_match_on_disk(self) -> None:
        examples_dir = self._examples_dir()
        manifest_path = examples_dir / "manifest.json"
        if not manifest_path.exists():
            self.skipTest("manifest.json missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, entry in manifest.items():
            fname = entry.get("file", "")
            self.assertTrue(
                (examples_dir / fname).exists(),
                f"Manifest entry '{name}' references non-existent file: {fname}"
            )


class TestCanonicalCasesStructure(unittest.TestCase):
    """Tests that run without the examples dir present — validate CANONICAL_CASES list itself."""

    def _cases(self):
        from src.generation.render_examples import CANONICAL_CASES
        return CANONICAL_CASES

    def test_total_count(self) -> None:
        cases = self._cases()
        self.assertEqual(len(cases), EXPECTED_COUNT, f"Expected {EXPECTED_COUNT} canonical cases, got {len(cases)}")

    def test_no_duplicate_case_names(self) -> None:
        cases = self._cases()
        names = [c.case_name for c in cases]
        self.assertEqual(len(names), len(set(names)), "Duplicate case_name entries in CANONICAL_CASES")

    def test_rendering_overrides_field_exists_and_is_dict(self) -> None:
        """Every CanonicalCase has a rendering_overrides attribute that is a dict."""
        cases = self._cases()
        for c in cases:
            self.assertIsInstance(
                c.rendering_overrides, dict,
                f"Case '{c.case_name}' rendering_overrides is not a dict: {type(c.rendering_overrides)}"
            )

    def test_override_cases_have_non_empty_dicts(self) -> None:
        """Cases that are supposed to have overrides do have non-empty dicts."""
        cases = self._cases()
        by_name = {c.case_name: c for c in cases}
        # Spot-check a handful of known override cases
        override_cases = [
            "sub_full_with_borrow_no_crossout",
            "sub_full_with_borrow_full_crossout",
            "mul_multi_with_pp_plus",
            "div_long_with_step_minus",
            "div_long_with_shallow_bracket",
        ]
        for name in override_cases:
            self.assertIn(name, by_name, f"Expected case '{name}' not found in CANONICAL_CASES")
            self.assertTrue(
                len(by_name[name].rendering_overrides) > 0,
                f"Case '{name}' should have non-empty rendering_overrides"
            )

    def test_baseline_cases_have_empty_overrides(self) -> None:
        """Original 43 baseline cases must not have any rendering overrides."""
        cases = self._cases()
        by_name = {c.case_name: c for c in cases}
        baseline_cases = [
            "add_full_no_carry", "sub_full_no_borrow", "mul_simple_full_no_carry",
            "mul_multi_full", "div_short_full_with_remainder", "div_long_full",
        ]
        for name in baseline_cases:
            self.assertIn(name, by_name, f"Expected baseline case '{name}' not found")
            self.assertEqual(
                by_name[name].rendering_overrides, {},
                f"Baseline case '{name}' should have empty rendering_overrides"
            )


class TestRenderingOverridesMechanism(unittest.TestCase):
    """Functional tests confirming rendering_overrides produces visually distinct output.

    These tests do not require the examples/ folder to exist — they call render_one
    directly with two CanonicalCase instances that differ only in rendering_overrides
    and assert the resulting images are pixel-different.
    """

    def _render(self, case):
        """Render one CanonicalCase and return a (512, 512) uint8 numpy array."""
        from src.generation.render_examples import render_one
        from src.core.run_config import load_config
        import tempfile, shutil
        from pathlib import Path as _Path
        pr = _PROJECT_ROOT
        cfg = load_config(pr)
        symbol_manifest = pr / "data" / "generated" / "processed" / "yolo" / "symbol_manifest.csv"
        if not symbol_manifest.exists():
            self.skipTest("symbol_manifest.csv not present — run validate first")
        return render_one(case, pr, symbol_manifest, cfg.generation)

    def test_borrow_cross_prob_override_produces_different_pixels(self) -> None:
        """borrow_cross_prob=0.0 and =1.0 must produce different rendered images."""
        from src.generation.render_examples import CanonicalCase, _seed
        base_name = "sub_full_with_borrow"
        case_no_cross = CanonicalCase(
            case_name=base_name + "_no_cross_test",
            scene_case="subtraction",
            completion_stage="full",
            force_wrong_answer=False,
            description="test: borrow_cross_prob=0.0",
            rendering_overrides={"borrow_cross_prob": 0.0},
        )
        case_no_cross.seed = _seed(case_no_cross.case_name)

        case_full_cross = CanonicalCase(
            case_name=base_name + "_full_cross_test",
            scene_case="subtraction",
            completion_stage="full",
            force_wrong_answer=False,
            description="test: borrow_cross_prob=1.0",
            rendering_overrides={"borrow_cross_prob": 1.0},
        )
        case_full_cross.seed = _seed(case_no_cross.case_name)  # same seed for identical operands

        img_no_cross = self._render(case_no_cross)
        img_full_cross = self._render(case_full_cross)

        self.assertFalse(
            (img_no_cross == img_full_cross).all(),
            "borrow_cross_prob=0.0 and =1.0 produced identical images — override not applied",
        )

    def test_pp_plus_prob_override_produces_different_pixels(self) -> None:
        """pp_plus_prob=0.0 and =1.0 must produce different rendered images for mul_multi."""
        from src.generation.render_examples import CanonicalCase, _seed
        case_no_plus = CanonicalCase(
            case_name="mul_multi_no_pp_plus_test",
            scene_case="multiplication-multi",
            completion_stage="full",
            force_wrong_answer=False,
            description="test: pp_plus_prob=0.0",
            rendering_overrides={"pp_plus_prob": 0.0},
        )
        case_no_plus.seed = _seed(case_no_plus.case_name)

        case_with_plus = CanonicalCase(
            case_name="mul_multi_with_pp_plus_test",
            scene_case="multiplication-multi",
            completion_stage="full",
            force_wrong_answer=False,
            description="test: pp_plus_prob=1.0",
            rendering_overrides={"pp_plus_prob": 1.0},
        )
        case_with_plus.seed = _seed(case_no_plus.case_name)  # same seed for identical operands

        img_no_plus = self._render(case_no_plus)
        img_with_plus = self._render(case_with_plus)

        self.assertFalse(
            (img_no_plus == img_with_plus).all(),
            "pp_plus_prob=0.0 and =1.0 produced identical images — override not applied",
        )


if __name__ == "__main__":
    unittest.main()
