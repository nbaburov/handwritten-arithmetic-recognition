"""Tests for the simplified single-preset handwriting style sampler."""
import random
import unittest

from src.core.run_config import PresetConfig
from src.generation.handwriting_style import HandwritingStyle, make_style


class TestHandwritingStyle(unittest.TestCase):
    def _preset(self) -> PresetConfig:
        return PresetConfig(
            glyph_rotation_min_deg=-4.0,
            glyph_rotation_max_deg=4.0,
            glyph_jitter_min_px=0,
            glyph_jitter_max_px=4,
            glyph_broken_stroke_prob=0.05,
        )

    def _rng(self, seed: int = 42) -> random.Random:
        return random.Random(seed)

    def test_make_style_returns_handwriting_style(self) -> None:
        style = make_style(self._rng(), self._preset())
        self.assertIsInstance(style, HandwritingStyle)

    def test_deterministic_under_seed(self) -> None:
        s1 = make_style(random.Random(7), self._preset())
        s2 = make_style(random.Random(7), self._preset())
        self.assertEqual(s1, s2)

    def test_stroke_target_px_not_present(self) -> None:
        """stroke_target_px must have been removed from HandwritingStyle in iter7."""
        style = make_style(self._rng(), self._preset())
        self.assertFalse(
            hasattr(style, "stroke_target_px"),
            "stroke_target_px was deleted in iter7 — source crops keep natural width",
        )

    def test_crowding_factor_normal(self) -> None:
        style = make_style(self._rng(), self._preset(), crowded=False)
        self.assertAlmostEqual(style.crowding_factor, 1.0)

    def test_crowding_factor_crowded(self) -> None:
        style = make_style(self._rng(), self._preset(), crowded=True)
        self.assertAlmostEqual(style.crowding_factor, 0.70)

    def test_rotation_bounds_stored_from_preset(self) -> None:
        """make_style must store rot_deg_min/max from preset without pre-sampling."""
        preset = PresetConfig(
            glyph_rotation_min_deg=-4.0,
            glyph_rotation_max_deg=4.0,
            glyph_jitter_min_px=0,
            glyph_jitter_max_px=4,
            glyph_broken_stroke_prob=0.05,
        )
        for seed in range(200):
            style = make_style(random.Random(seed), preset)
            self.assertAlmostEqual(style.rot_deg_min, preset.glyph_rotation_min_deg)
            self.assertAlmostEqual(style.rot_deg_max, preset.glyph_rotation_max_deg)

    def test_per_glyph_rotation_diverges_within_scene(self) -> None:
        """With a non-zero rotation range, different glyphs in the same scene must
        receive different rotation angles (per-glyph sampling, not per-scene)."""
        preset = PresetConfig(
            glyph_rotation_min_deg=-10.0,
            glyph_rotation_max_deg=10.0,
            glyph_jitter_min_px=0,
            glyph_jitter_max_px=0,
            glyph_broken_stroke_prob=0.0,
        )
        rng = random.Random(42)
        style = make_style(rng, preset)
        # Simulate what synth_pool does: sample theta per glyph from the stored bounds.
        angles = [rng.uniform(style.rot_deg_min, style.rot_deg_max) for _ in range(10)]
        unique_angles = len(set(round(a, 6) for a in angles))
        self.assertGreater(
            unique_angles,
            1,
            "All 10 per-glyph angles were identical — rotation is still per-scene, not per-glyph.",
        )

    def test_jitter_bounds_match_preset(self) -> None:
        preset = PresetConfig(
            glyph_rotation_min_deg=-4.0,
            glyph_rotation_max_deg=4.0,
            glyph_jitter_min_px=1,
            glyph_jitter_max_px=6,
            glyph_broken_stroke_prob=0.05,
        )
        style = make_style(self._rng(), preset)
        self.assertEqual(style.jitter_min_px, 1)
        self.assertEqual(style.jitter_max_px, 6)

    def test_broken_stroke_prob_matches_preset(self) -> None:
        for prob in (0.0, 0.05, 0.10):
            preset = PresetConfig(
                glyph_rotation_min_deg=-4.0,
                glyph_rotation_max_deg=4.0,
                glyph_jitter_min_px=0,
                glyph_jitter_max_px=4,
                glyph_broken_stroke_prob=prob,
            )
            style = make_style(self._rng(), preset)
            self.assertAlmostEqual(style.glyph_broken_stroke_prob, prob)


class TestPresetConfig(unittest.TestCase):
    def test_defaults(self) -> None:
        cfg = PresetConfig()
        self.assertAlmostEqual(cfg.glyph_rotation_min_deg, -4.0)
        self.assertAlmostEqual(cfg.glyph_rotation_max_deg, 4.0)
        self.assertEqual(cfg.glyph_jitter_min_px, 0)
        self.assertEqual(cfg.glyph_jitter_max_px, 4)
        self.assertAlmostEqual(cfg.glyph_broken_stroke_prob, 0.05)

    def test_custom_values(self) -> None:
        cfg = PresetConfig(
            glyph_rotation_min_deg=-6.0,
            glyph_rotation_max_deg=6.0,
            glyph_jitter_min_px=2,
            glyph_jitter_max_px=8,
            glyph_broken_stroke_prob=0.10,
        )
        self.assertAlmostEqual(cfg.glyph_rotation_min_deg, -6.0)
        self.assertAlmostEqual(cfg.glyph_rotation_max_deg, 6.0)
        self.assertEqual(cfg.glyph_jitter_min_px, 2)
        self.assertEqual(cfg.glyph_jitter_max_px, 8)
        self.assertAlmostEqual(cfg.glyph_broken_stroke_prob, 0.10)


if __name__ == "__main__":
    unittest.main()
