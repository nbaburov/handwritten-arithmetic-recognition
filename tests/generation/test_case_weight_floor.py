from __future__ import annotations

"""Tests for the case-weight negative-value floor guard.

Coverage:
- A negative weight in GenerationConfig.case_weights is clamped to 0.0.
- A warning is logged when clamping occurs.
- A zero weight passes through unchanged (no warning).
- A positive weight passes through unchanged (no warning).
"""

import logging
import unittest

from src.generation.layouts_types import SceneCase


class TestSafeCaseWeight(unittest.TestCase):

    def _make_gen_cfg(self, weights: dict) -> object:
        """Return a GenerationConfig-like object with the given case_weights dict."""
        from src.core.run_config import GenerationConfig
        return GenerationConfig(
            images_per_case=10,
            min_instances_per_label=None,
            topup_rounds=0,
            case_weights=weights,
        )

    def test_negative_weight_clamped_to_zero(self) -> None:
        from src.data_pipeline.prepare_synthetic_yolo import _safe_case_weight
        gen_cfg = self._make_gen_cfg({SceneCase.addition.value: -1.5})
        result = _safe_case_weight(gen_cfg, SceneCase.addition)
        self.assertEqual(result, 0.0)

    def test_negative_weight_emits_warning(self) -> None:
        from src.data_pipeline.prepare_synthetic_yolo import _safe_case_weight
        gen_cfg = self._make_gen_cfg({SceneCase.subtraction.value: -0.001})
        with self.assertLogs("src.data_pipeline.prepare_synthetic_yolo", level=logging.WARNING) as cm:
            _safe_case_weight(gen_cfg, SceneCase.subtraction)
        self.assertTrue(
            any("subtraction" in msg or "Negative" in msg for msg in cm.output),
            f"Expected a warning mentioning the case or 'Negative'; got: {cm.output}",
        )

    def test_zero_weight_passes_through(self) -> None:
        from src.data_pipeline.prepare_synthetic_yolo import _safe_case_weight
        gen_cfg = self._make_gen_cfg({SceneCase.division_short.value: 0.0})
        result = _safe_case_weight(gen_cfg, SceneCase.division_short)
        self.assertEqual(result, 0.0)

    def test_positive_weight_passes_through(self) -> None:
        from src.data_pipeline.prepare_synthetic_yolo import _safe_case_weight
        gen_cfg = self._make_gen_cfg({SceneCase.multiplication_multi.value: 2.0})
        result = _safe_case_weight(gen_cfg, SceneCase.multiplication_multi)
        self.assertEqual(result, 2.0)

    def test_missing_case_defaults_to_one(self) -> None:
        """A case absent from case_weights dict should default to 1.0."""
        from src.data_pipeline.prepare_synthetic_yolo import _safe_case_weight
        # Use a weights dict that does not contain division_long.
        gen_cfg = self._make_gen_cfg({SceneCase.addition.value: 1.0})
        result = _safe_case_weight(gen_cfg, SceneCase.division_long)
        self.assertEqual(result, 1.0)
