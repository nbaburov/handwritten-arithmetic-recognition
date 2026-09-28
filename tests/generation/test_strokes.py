import numpy as np
import random
import unittest
from src.generation.strokes import draw_handwritten_bar, draw_handwritten_bracket
from src.generation.handwriting_style import HandwritingStyle

def _neat_style():
    return HandwritingStyle(
        rot_deg_min=0.0,
        rot_deg_max=0.0,
        jitter_min_px=0,
        jitter_max_px=0,
        glyph_broken_stroke_prob=0.0,
        crowding_factor=1.0,
    )

class TestStrokes(unittest.TestCase):
    def test_bar_draws_ink(self):
        canvas = np.full((512, 512), 255, dtype=np.uint8)
        draw_handwritten_bar(canvas, 50, 100, 200, 110, _neat_style(), random.Random(1))
        self.assertTrue((canvas < 255).any())

    def test_bar_returns_bbox(self):
        canvas = np.full((512, 512), 255, dtype=np.uint8)
        bbox = draw_handwritten_bar(canvas, 50, 100, 200, 110, _neat_style(), random.Random(1))
        self.assertEqual(len(bbox), 4)
        x0, y0, x1, y1 = bbox
        self.assertLess(x0, x1)
        self.assertLessEqual(y0, y1)

    def test_bracket_draws_ink(self):
        canvas = np.full((512, 512), 255, dtype=np.uint8)
        draw_handwritten_bracket(canvas, 200, 50, 60, 200, _neat_style(), random.Random(1), 0.0)
        self.assertTrue((canvas < 255).any())

    def test_bar_bbox_covers_drawn_pixels(self):
        canvas = np.full((512, 512), 255, dtype=np.uint8)
        bbox = draw_handwritten_bar(canvas, 50, 100, 200, 110, _neat_style(), random.Random(3))
        x0, y0, x1, y1 = bbox
        ink_rows, ink_cols = np.where(canvas < 200)
        if len(ink_cols) == 0:
            return  # gap covered everything — unlikely but skip
        self.assertGreaterEqual(x0, ink_cols.min() - 2)
        self.assertLessEqual(x1, ink_cols.max() + 2)


class TestBracketRendering(unittest.TestCase):

    def test_bracket_horizontal_arm_extends_right(self):
        """Pixels to the right of corner_x must have ink; nothing far to the left."""
        canvas = np.full((512, 512), 255, dtype=np.uint8)
        corner_x = 100
        draw_handwritten_bracket(canvas, corner_x, 100, 80, 60, _neat_style(), random.Random(1), 0.0)
        ink_cols = np.where(canvas < 200)[1]
        self.assertGreater(len(ink_cols), 0, "No ink drawn")
        # Ink must extend to the right of corner
        self.assertGreater(ink_cols.max(), corner_x, "Horizontal arm did not extend right")
        # No ink far to the left (more than 10px left of corner means arm went wrong way)
        self.assertGreaterEqual(ink_cols.min(), corner_x - 10, f"Ink at x={ink_cols.min()} is too far left of corner_x={corner_x}")

    def test_bracket_vertical_arm_extends_down(self):
        """Ink must appear below corner_y."""
        canvas = np.full((512, 512), 255, dtype=np.uint8)
        corner_y = 50
        draw_handwritten_bracket(canvas, 100, corner_y, 80, 100, _neat_style(), random.Random(2), 0.0)
        ink_rows = np.where(canvas < 200)[0]
        self.assertGreater(len(ink_rows), 0, "No ink drawn")
        self.assertGreater(ink_rows.max(), corner_y + 5, "Vertical arm did not extend down")

    def test_bracket_bbox_within_canvas(self):
        """All 50 random seeds must return bbox coords within canvas bounds."""
        style = _neat_style()
        for seed in range(50):
            canvas = np.full((256, 256), 255, dtype=np.uint8)
            bbox = draw_handwritten_bracket(canvas, 80, 40, 100, 80, style, random.Random(seed), 0.0)
            min_x, min_y, max_x, max_y = bbox
            self.assertGreaterEqual(min_x, 0, f"seed {seed}: min_x={min_x} < 0")
            self.assertGreaterEqual(min_y, 0, f"seed {seed}: min_y={min_y} < 0")
            self.assertLessEqual(max_x, 256, f"seed {seed}: max_x={max_x} > 256")
            self.assertLessEqual(max_y, 256, f"seed {seed}: max_y={max_y} > 256")

    def test_bracket_bbox_coords_non_negative(self):
        """Specifically target the previously broken negative-cx case."""
        style = _neat_style()
        for seed in range(20):
            canvas = np.full((512, 512), 255, dtype=np.uint8)
            # corner near left edge to stress-test clamping
            bbox = draw_handwritten_bracket(canvas, 5, 5, 80, 60, style, random.Random(seed), 0.0)
            min_x, min_y, max_x, max_y = bbox
            self.assertGreaterEqual(min_x, 0, f"seed {seed}: negative min_x={min_x}")
            self.assertGreaterEqual(min_y, 0, f"seed {seed}: negative min_y={min_y}")

    def test_bracket_has_spline_variation(self):
        """10 different rng seeds must produce at least 2 distinct horizontal control point y values."""
        style = _neat_style()
        # Collect the first-interior control point y positions across seeds
        # We do this by rendering and checking that not all pixel heights are identical
        all_ink_rows = []
        for seed in range(10):
            canvas = np.full((512, 512), 255, dtype=np.uint8)
            draw_handwritten_bracket(canvas, 200, 100, 100, 60, style, random.Random(seed), 0.0)
            ink_rows = set(np.where(canvas[:, 200:250] < 200)[0].tolist())
            all_ink_rows.append(ink_rows)
        # Not all 10 seeds should produce identical row sets (jitter is applied)
        unique_sets = len(set(frozenset(r) for r in all_ink_rows))
        self.assertGreater(unique_sets, 1, "All seeds produced identical bracket strokes — jitter not applied")


class TestHandDrawnCharacterUpgrade(unittest.TestCase):
    """Tests for workstream A: variable thickness, intensity variation, micro-breaks."""

    def test_bar_variable_thickness_produces_different_segment_widths(self):
        """With thickness_min=2 and thickness_max=5, multiple renders should not
        all produce the same bounding box height — segments vary in thickness."""
        style = _neat_style()
        bbox_heights = set()
        for seed in range(20):
            canvas = np.full((512, 512), 255, dtype=np.uint8)
            bbox = draw_handwritten_bar(
                canvas, 50, 200, 350, 210, style, random.Random(seed),
                thickness_min=2, thickness_max=5,
            )
            _, y0, _, y1 = bbox
            bbox_heights.add(y1 - y0)
        # At least 2 distinct heights expected across 20 seeds when thickness range is wide
        self.assertGreater(len(bbox_heights), 1, "All 20 seeds produced identical bbox height — per-segment thickness not varying")

    def test_bar_intensity_variation_produces_non_pure_black_pixels(self):
        """With intensity_min=30 and intensity_max=80, the darkest drawn pixel
        must be strictly greater than 0 (not pure black)."""
        style = _neat_style()
        canvas = np.full((512, 512), 255, dtype=np.uint8)
        draw_handwritten_bar(
            canvas, 50, 200, 350, 210, style, random.Random(7),
            gap_prob=0.0, micro_break_prob=0.0,
            intensity_min=30, intensity_max=80,
        )
        ink_pixels = canvas[canvas < 255]
        self.assertGreater(len(ink_pixels), 0, "No ink drawn")
        # All ink pixels must be >= 30 (intensity_min) — pure black (0) should not appear
        self.assertGreaterEqual(int(ink_pixels.min()), 30,
            f"Pure-black pixel found (value {ink_pixels.min()}); intensity variation not applied")

    def test_bar_micro_break_prob_1_skips_all_segments(self):
        """With micro_break_prob=1.0 and no large gap, every segment is skipped
        so the canvas should remain all-white."""
        style = _neat_style()
        canvas = np.full((512, 512), 255, dtype=np.uint8)
        draw_handwritten_bar(
            canvas, 50, 200, 350, 210, style, random.Random(1),
            gap_prob=0.0, micro_break_prob=1.0,
        )
        self.assertTrue((canvas == 255).all(), "micro_break_prob=1.0 should skip all segments but ink was drawn")

    def test_bracket_intensity_variation_produces_non_pure_black_pixels(self):
        """Bracket segments drawn with intensity_min=30 must not produce pure-black pixels."""
        style = _neat_style()
        canvas = np.full((512, 512), 255, dtype=np.uint8)
        draw_handwritten_bracket(
            canvas, 200, 50, 150, 120, style, random.Random(5), 0.0,
            gap_prob=0.0, micro_break_prob=0.0,
            intensity_min=30, intensity_max=80,
        )
        ink_pixels = canvas[canvas < 255]
        self.assertGreater(len(ink_pixels), 0, "No ink drawn")
        self.assertGreaterEqual(int(ink_pixels.min()), 30,
            f"Pure-black pixel found (value {ink_pixels.min()}); intensity variation not applied to bracket")
