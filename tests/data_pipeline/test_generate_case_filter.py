from __future__ import annotations

"""Tests for the --cases CLI filter on the generate subcommand.

Coverage:
- parse_cases_arg with a valid multi-case string returns the expected SceneCase list.
- parse_cases_arg with an invalid name raises ValueError with a clear message.
- run_prepare_synthetic_yolo respects the cases filter: only the requested
  cases are passed to build_synthetic_yolo_dataset (verified via mock).
- When no cases argument is supplied, all 16 SceneCase values are used.
"""

import unittest
from unittest.mock import MagicMock, patch, call
from pathlib import Path

from src.generation.layouts_types import SceneCase


class TestParseCasesArg(unittest.TestCase):

    def test_valid_two_cases(self) -> None:
        from src.data_pipeline.prepare_synthetic_yolo import parse_cases_arg
        result = parse_cases_arg("multiplication-multi,subtraction")
        self.assertEqual(result, [SceneCase.multiplication_multi, SceneCase.subtraction])

    def test_valid_single_case(self) -> None:
        from src.data_pipeline.prepare_synthetic_yolo import parse_cases_arg
        result = parse_cases_arg("division-short")
        self.assertEqual(result, [SceneCase.division_short])

    def test_whitespace_stripped(self) -> None:
        from src.data_pipeline.prepare_synthetic_yolo import parse_cases_arg
        result = parse_cases_arg(" addition , subtraction ")
        self.assertEqual(result, [SceneCase.addition, SceneCase.subtraction])

    def test_invalid_name_raises_value_error(self) -> None:
        from src.data_pipeline.prepare_synthetic_yolo import parse_cases_arg
        with self.assertRaises(ValueError) as ctx:
            parse_cases_arg("multiplication-multi,not_a_real_case")
        self.assertIn("not_a_real_case", str(ctx.exception))

    def test_all_invalid_raises_value_error(self) -> None:
        from src.data_pipeline.prepare_synthetic_yolo import parse_cases_arg
        with self.assertRaises(ValueError):
            parse_cases_arg("totally_bogus")


class TestRunPrepareFilterPassthrough(unittest.TestCase):
    """run_prepare_synthetic_yolo must pass only the requested cases to build_synthetic_yolo_dataset."""

    def _make_minimal_config(self, tmp_dir: Path):
        """Return a DataPrepConfig pointing at a temp directory."""
        from src.core.config import DataPrepConfig
        cfg = MagicMock(spec=DataPrepConfig)
        cfg.synthetic_dir = tmp_dir / "synthetic"
        cfg.synthetic_dir.mkdir(parents=True, exist_ok=True)
        cfg.yolo_dir = tmp_dir / "yolo"
        cfg.yolo_dir.mkdir(parents=True, exist_ok=True)
        # create the manifest file that run_prepare_synthetic_yolo checks
        (cfg.yolo_dir / "symbol_assets_manifest.csv").write_text(
            "label,path\n", encoding="utf-8"
        )
        return cfg

    def _make_minimal_gen_cfg(self):
        from src.core.run_config import GenerationConfig, SceneConfig, PresetConfig
        return GenerationConfig(
            images_per_case=2,
            min_instances_per_label=None,
            topup_rounds=0,
            seed=0,
            split_train=0.8,
            split_val=0.1,
            split_test=0.1,
            case_weights={c.value: 1.0 for c in SceneCase},
            source_pool="auto",
            min_source_quality=0.0,
        )

    def _fake_build_result(self, tmp_dir: Path, case: SceneCase, split: str):
        """Return a minimal result dict that satisfies the caller."""
        import pandas as pd
        manifest = tmp_dir / "synthetic" / f"{case.value}_{split}_manifest.csv"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{
            "image": str(manifest.parent / f"{case.value}_{split}_000000.png"),
            "label": str(manifest.parent / f"{case.value}_{split}_000000.txt"),
            "ground_truth": str(manifest.parent / f"{case.value}_{split}_000000.gt.json"),
            "case": case.value,
        }]).to_csv(manifest, index=False)
        return {"manifest": manifest}

    def test_case_filter_limits_calls(self) -> None:
        import tempfile
        from src.data_pipeline.prepare_synthetic_yolo import run_prepare_synthetic_yolo

        requested_cases = [SceneCase.multiplication_multi, SceneCase.subtraction]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = self._make_minimal_config(tmp_path)
            gen_cfg = self._make_minimal_gen_cfg()

            captured_cases: list[SceneCase] = []

            def fake_build(dp_config, case, split_name, n, **kwargs):
                captured_cases.append(case)
                return self._fake_build_result(tmp_path, case, split_name)

            with patch("src.data_pipeline.prepare_synthetic_yolo.build_synthetic_yolo_dataset", side_effect=fake_build), \
                 patch("src.data_pipeline.prepare_synthetic_yolo.validate_yolo_labels"), \
                 patch("src.data_pipeline.prepare_synthetic_yolo.configure_run_logging"):
                run_prepare_synthetic_yolo(cfg, gen_cfg, cases=requested_cases)

            unique_cases = list(dict.fromkeys(captured_cases))  # preserve order, dedupe
            self.assertEqual(
                sorted(unique_cases, key=lambda c: c.value),
                sorted(requested_cases, key=lambda c: c.value),
                "Only the requested cases should be passed to build_synthetic_yolo_dataset",
            )

    def test_no_case_filter_uses_all_cases(self) -> None:
        import tempfile
        from src.data_pipeline.prepare_synthetic_yolo import run_prepare_synthetic_yolo

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = self._make_minimal_config(tmp_path)
            gen_cfg = self._make_minimal_gen_cfg()

            captured_cases: list[SceneCase] = []

            def fake_build(dp_config, case, split_name, n, **kwargs):
                captured_cases.append(case)
                return self._fake_build_result(tmp_path, case, split_name)

            with patch("src.data_pipeline.prepare_synthetic_yolo.build_synthetic_yolo_dataset", side_effect=fake_build), \
                 patch("src.data_pipeline.prepare_synthetic_yolo.validate_yolo_labels"), \
                 patch("src.data_pipeline.prepare_synthetic_yolo.configure_run_logging"):
                run_prepare_synthetic_yolo(cfg, gen_cfg, cases=None)

            unique_cases = set(captured_cases)
            all_cases = set(SceneCase)
            self.assertEqual(
                unique_cases,
                all_cases,
                "Without a filter, all 16 SceneCase values must be generated",
            )
