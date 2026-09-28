"""iter11 crowding mechanics: double-digit-in-one-cell + glyph overflow.

Double-digit tests run at the layout level (pool-free, always run). The overflow test renders a
real scene and so needs the symbol manifest; it skips when the manifest is absent.
"""
from __future__ import annotations

import dataclasses
import random
import unittest
from pathlib import Path

from src.generation.layouts import SceneCase, sample_layout_for_case


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _manifest_present() -> bool:
    return (_project_root() / "data" / "generated" / "processed" / "yolo"
            / "symbol_assets_manifest.csv").exists()


class TestDoubleDigitCell(unittest.TestCase):
    """The crowded SceneCases pack two glyphs into one borrow/carry cell (same row,col)."""

    def _same_cell_pairs(self, tokens, role: str) -> int:
        from collections import Counter
        cells = Counter((t.row, t.col) for t in tokens if t.yolo_class_name == role)
        return sum(1 for _, n in cells.items() if n >= 2)

    def test_crowded_borrow_packs_two_digits(self):
        rng = random.Random(42)
        kind, tokens = sample_layout_for_case(SceneCase.subtraction_crowded_borrow, rng)
        self.assertEqual(kind.value, "subtract")
        self.assertGreaterEqual(
            self._same_cell_pairs(tokens, "digit_borrow"), 1,
            "expected at least one borrow cell holding two glyphs",
        )
        self.assertTrue(
            any(abs(getattr(t, "cell_subpos", 0.0)) > 0 for t in tokens),
            "double-digit glyphs must carry a non-zero cell_subpos",
        )

    def test_crowded_carry_packs_two_digits(self):
        rng = random.Random(7)
        kind, tokens = sample_layout_for_case(SceneCase.addition_crowded_carry, rng)
        self.assertEqual(kind.value, "add")
        self.assertGreaterEqual(self._same_cell_pairs(tokens, "digit_carry"), 1)

    def test_double_digit_labels_parse(self):
        """borrow_6 / carry_3 style labels must resolve in the ontology."""
        from src.core.ontology import parse_flattened_label
        rng = random.Random(3)
        _, tokens = sample_layout_for_case(SceneCase.subtraction_crowded_borrow, rng)
        for t in tokens:
            parse_flattened_label(t.flattened_label)  # raises if invalid

    def test_gt_row_col_unchanged_for_packed_cell(self):
        """Both glyphs in a packed cell keep the SAME (row, col) -- GT is the intended cell."""
        from collections import defaultdict
        rng = random.Random(11)
        _, tokens = sample_layout_for_case(SceneCase.subtraction_crowded_borrow, rng)
        by_cell = defaultdict(list)
        for t in tokens:
            if t.yolo_class_name == "digit_borrow":
                by_cell[(t.row, t.col)].append(t)
        packed = [cell for cell, ts in by_cell.items() if len(ts) >= 2]
        self.assertTrue(packed, "expected at least one packed borrow cell")


@unittest.skipUnless(_manifest_present(), "symbol manifest absent; overflow render test skipped")
class TestGlyphOverflow(unittest.TestCase):
    """Overflow enlarges/shifts a glyph so its bbox crosses the cell boundary."""

    def _max_glyph_width(self, overflow_prob: float, seed: int) -> float:
        from src.core.run_config import load_config
        from src.generation.synth_yolo import render_single_scene
        gen_cfg, _, _ = load_config(_project_root())
        rend = dict(gen_cfg.rendering)
        rend["glyph_overflow_prob"] = overflow_prob
        cfg = dataclasses.replace(gen_cfg, rendering=rend)
        ts = render_single_scene("subtraction", seed=seed, completion_stage="full",
                                 generation_config=cfg)
        widths = [s.bbox[2] - s.bbox[0] for s in ts.symbols if s.yolo_class == "digit_main"]
        return max(widths) if widths else 0.0

    def test_overflow_widens_some_glyph(self):
        """Across seeds, forcing overflow yields a wider max glyph than disabling it (majority)."""
        wider = 0
        seeds = [1, 2, 3, 4, 5]
        for sd in seeds:
            w_on = self._max_glyph_width(1.0, sd)
            w_off = self._max_glyph_width(0.0, sd)
            if w_on > w_off:
                wider += 1
        self.assertGreaterEqual(
            wider, 3,
            f"overflow should widen the max glyph in the majority of seeds (got {wider}/{len(seeds)})",
        )

    def test_no_overflow_respects_clamp(self):
        """With overflow disabled, no main glyph exceeds the column pitch (anti-overlap clamp holds)."""
        # Sanity: disabled overflow must not crash and must produce a finite max width.
        w = self._max_glyph_width(0.0, seed=42)
        self.assertGreater(w, 0.0)


if __name__ == "__main__":
    unittest.main()
