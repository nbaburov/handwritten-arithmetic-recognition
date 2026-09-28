"""Tests for operator-scale variance (iter10 R-B).

Verifies that the operator-scale sampling logic in synth_pool._draw_equation_block
respects the new operator_scale_min / operator_scale_max knobs and maintains
backward compatibility with the legacy scalar operator_scale key.
"""
from __future__ import annotations

import random
import unittest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _operator_scale_from_rendering(rendering: dict, rng: random.Random, token_scale: float) -> float:
    """Replicate the exact operator-scale resolution logic from synth_pool.py.

    This is a thin, isolated copy of the three-line branch so we can unit-test
    it without constructing a full canvas + pool.  If synth_pool.py changes the
    branch, this helper must be kept in sync.
    """
    _op_min = rendering.get("operator_scale_min", rendering.get("operator_scale", token_scale))
    _op_max = rendering.get("operator_scale_max", _op_min)
    return rng.uniform(_op_min, _op_max)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOperatorScaleRange(unittest.TestCase):
    """Operator scale sampled from [min, max] with the new knobs."""

    def _sample_scales(self, rendering: dict, n: int = 100, seed: int = 0) -> list[float]:
        rng = random.Random(seed)
        token_scale = 0.92  # OPERATOR_TOKEN_SCALE
        return [_operator_scale_from_rendering(rendering, rng, token_scale) for _ in range(n)]

    def test_range_knobs_produce_values_within_bounds(self) -> None:
        """With min/max set, every sampled value is in [0.65, 1.20]."""
        rendering = {"operator_scale_min": 0.65, "operator_scale_max": 1.20}
        scales = self._sample_scales(rendering, n=100)
        self.assertGreaterEqual(min(scales), 0.65, "observed scale below operator_scale_min")
        self.assertLessEqual(max(scales), 1.20, "observed scale above operator_scale_max")

    def test_range_knobs_produce_variance(self) -> None:
        """Across 100 draws the range is non-trivially wide (at least 0.3)."""
        rendering = {"operator_scale_min": 0.65, "operator_scale_max": 1.20}
        scales = self._sample_scales(rendering, n=100)
        observed_range = max(scales) - min(scales)
        self.assertGreater(observed_range, 0.3, "insufficient variance across 100 draws")

    def test_legacy_scalar_produces_no_variance(self) -> None:
        """With only operator_scale (no min/max), all 100 draws return the same value."""
        rendering = {"operator_scale": 1.0}
        scales = self._sample_scales(rendering, n=100)
        self.assertEqual(min(scales), max(scales), "legacy scalar path must be deterministic")
        self.assertAlmostEqual(scales[0], 1.0, places=9)

    def test_missing_all_keys_falls_back_to_token_scale(self) -> None:
        """When rendering has no operator_scale keys, the result equals token_scale."""
        rendering: dict = {}
        scales = self._sample_scales(rendering, n=100)
        token_scale = 0.92
        self.assertEqual(min(scales), max(scales))
        self.assertAlmostEqual(scales[0], token_scale, places=9)

    def test_min_without_max_collapses_to_scalar(self) -> None:
        """When only operator_scale_min is present, max defaults to min — no variance."""
        rendering = {"operator_scale_min": 0.80}
        scales = self._sample_scales(rendering, n=100)
        self.assertEqual(min(scales), max(scales))
        self.assertAlmostEqual(scales[0], 0.80, places=9)

    def test_precedence_min_max_overrides_legacy_scalar(self) -> None:
        """min/max takes precedence: if both old and new keys present, values come from min/max range."""
        rendering = {"operator_scale": 1.0, "operator_scale_min": 0.65, "operator_scale_max": 1.20}
        scales = self._sample_scales(rendering, n=100)
        # should see variance (not pinned to 1.0)
        self.assertGreater(max(scales) - min(scales), 0.1)
        # and stays within min/max bounds
        self.assertGreaterEqual(min(scales), 0.65)
        self.assertLessEqual(max(scales), 1.20)


if __name__ == "__main__":
    unittest.main()
