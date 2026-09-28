"""Tests for src/data_pipeline/preprocessing.py."""
from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from src.data_pipeline.preprocessing import (
    PreprocessConfig,
    contrast_normalise,
    estimate_stroke_width,
    ink_mask,
    normalise_stroke_width,
    preprocess_for_pipeline,
    resize_and_pad_to_square,
)


def _white_canvas(h: int, w: int) -> np.ndarray:
    return np.full((h, w), 255, dtype=np.uint8)


def _canvas_with_stripe(h: int, w: int, stripe_col: int, stripe_width: int) -> np.ndarray:
    """White canvas with a vertical black stripe of known width."""
    canvas = _white_canvas(h, w)
    half = stripe_width // 2
    cx = stripe_col
    canvas[:, max(0, cx - half): cx + half + 1] = 0
    return canvas


class TestResizeAndPad(unittest.TestCase):
    def test_output_shape_square_input(self) -> None:
        img = _white_canvas(512, 512)
        out = resize_and_pad_to_square(img, 512)
        self.assertEqual(out.shape, (512, 512))

    def test_output_shape_portrait_input(self) -> None:
        img = _white_canvas(500, 300)
        out = resize_and_pad_to_square(img, 512)
        self.assertEqual(out.shape, (512, 512))

    def test_output_shape_landscape_input(self) -> None:
        img = _white_canvas(300, 500)
        out = resize_and_pad_to_square(img, 512)
        self.assertEqual(out.shape, (512, 512))

    def test_padding_is_white(self) -> None:
        """Padded regions must be exactly 255 (white), not gray."""
        img = _white_canvas(300, 500)
        # Mark the content white to distinguish from padding
        out = resize_and_pad_to_square(img, 512)
        # Landscape image: padding appears top and bottom
        # new_h = int(300 * 512/500) = 307, pad_y = (512-307)//2 = 102
        self.assertEqual(int(out[0, 256]), 255, "Top pad row should be white")
        self.assertEqual(int(out[511, 256]), 255, "Bottom pad row should be white")

    def test_dtype_is_uint8(self) -> None:
        img = _white_canvas(200, 300)
        out = resize_and_pad_to_square(img, 512)
        self.assertEqual(out.dtype, np.uint8)

    def test_content_preserves_ink(self) -> None:
        """A dark square in the input must remain dark in the output."""
        img = _white_canvas(512, 512)
        img[200:300, 200:300] = 10
        out = resize_and_pad_to_square(img, 512)
        # Content region should have some dark pixels
        self.assertTrue(np.any(out < 50))


class TestContrastNormalise(unittest.TestCase):
    def test_all_white_unchanged(self) -> None:
        gray = np.full((100, 100), 255, dtype=np.uint8)
        out = contrast_normalise(gray, 1.0, 99.0)
        self.assertEqual(out.dtype, np.uint8)
        # All white — percentile lo == hi, so returns copy
        self.assertTrue(np.all(out >= 240))

    def test_ink_stays_dark_background_stays_bright(self) -> None:
        """Synthetic canvas with ink at 40 gray — after normalisation ink < 80, bg > 240."""
        gray = np.full((100, 100), 220, dtype=np.uint8)
        gray[30:70, 30:70] = 40
        out = contrast_normalise(gray, 1.0, 99.0)
        ink_pixels = out[30:70, 30:70]
        bg_pixels = out[:30, :30]
        self.assertTrue(np.all(ink_pixels < 80))
        self.assertTrue(np.all(bg_pixels > 240))

    def test_output_dtype(self) -> None:
        gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
        out = contrast_normalise(gray, 1.0, 99.0)
        self.assertEqual(out.dtype, np.uint8)

    def test_output_clipped_to_255(self) -> None:
        gray = np.full((50, 50), 128, dtype=np.uint8)
        gray[10, 10] = 0
        gray[40, 40] = 255
        out = contrast_normalise(gray, 1.0, 99.0)
        self.assertLessEqual(int(out.max()), 255)
        self.assertGreaterEqual(int(out.min()), 0)


class TestInkMask(unittest.TestCase):
    def test_pixels_below_threshold_are_true(self) -> None:
        gray = np.array([[100, 200, 210], [50, 199, 255]], dtype=np.uint8)
        mask = ink_mask(gray, 200)
        expected = np.array([[True, False, False], [True, True, False]])
        np.testing.assert_array_equal(mask, expected)

    def test_known_pixel_count(self) -> None:
        """Canvas with 10 ink pixels at known value."""
        gray = np.full((20, 20), 255, dtype=np.uint8)
        gray[5:7, 5:10] = 50  # 10 ink pixels
        mask = ink_mask(gray, 200)
        count = int(np.sum(mask))
        self.assertAlmostEqual(count, 10, delta=0)


class TestEstimateStrokeWidth(unittest.TestCase):
    def test_empty_mask_returns_zero(self) -> None:
        mask = np.zeros((100, 100), dtype=bool)
        self.assertAlmostEqual(estimate_stroke_width(mask), 0.0)

    def test_thick_stripe_estimates_correct_width(self) -> None:
        """A 10px-wide stripe should give estimate close to 10."""
        canvas = np.full((100, 100), 255, dtype=np.uint8)
        canvas[:, 45:55] = 0  # 10px wide vertical stripe
        mask = canvas < 200
        est = estimate_stroke_width(mask)
        self.assertGreater(est, 5.0)
        self.assertLess(est, 20.0)


class TestNormaliseStrokeWidth(unittest.TestCase):
    def test_output_shape_unchanged(self) -> None:
        gray = np.full((100, 100), 255, dtype=np.uint8)
        gray[40:60, 40:60] = 0
        mask = gray < 200
        out = normalise_stroke_width(gray, mask, target_px=2)
        self.assertEqual(out.shape, gray.shape)

    def test_output_dtype(self) -> None:
        gray = np.full((100, 100), 255, dtype=np.uint8)
        gray[45:55, 45:55] = 0
        mask = gray < 200
        out = normalise_stroke_width(gray, mask, target_px=2)
        self.assertEqual(out.dtype, np.uint8)

    def test_empty_mask_returns_copy(self) -> None:
        gray = np.full((50, 50), 255, dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=bool)
        out = normalise_stroke_width(gray, mask, target_px=2)
        np.testing.assert_array_equal(out, gray)


class TestPreprocessForPipeline(unittest.TestCase):
    def _simple_canvas(self, h: int = 384, w: int = 512) -> np.ndarray:
        canvas = np.full((h, w), 255, dtype=np.uint8)
        canvas[h // 3: 2 * h // 3, w // 4: 3 * w // 4] = 30
        return canvas

    def test_output_shape_square_input(self) -> None:
        out = preprocess_for_pipeline(self._simple_canvas(512, 512))
        self.assertEqual(out.shape, (512, 512))

    def test_output_shape_portrait_input(self) -> None:
        out = preprocess_for_pipeline(self._simple_canvas(800, 400))
        self.assertEqual(out.shape, (512, 512))

    def test_output_shape_landscape_input(self) -> None:
        out = preprocess_for_pipeline(self._simple_canvas(384, 512))
        self.assertEqual(out.shape, (512, 512))

    def test_output_dtype(self) -> None:
        out = preprocess_for_pipeline(self._simple_canvas())
        self.assertEqual(out.dtype, np.uint8)

    def test_white_padding_is_255(self) -> None:
        """Padded columns/rows must be white (255), not gray."""
        white = np.full((300, 512), 255, dtype=np.uint8)
        out = preprocess_for_pipeline(white)
        # Portrait 300×512 → long side is 512, short is 300
        # new_h = int(300*512/512) = 300, pad_y = (512-300)//2 = 106
        # Check top padding rows are 255
        self.assertTrue(np.all(out[:50, :] == 255), "Top padding rows should be white")
        self.assertTrue(np.all(out[-50:, :] == 255), "Bottom padding rows should be white")

    def test_accepts_pil_image(self) -> None:
        pil = Image.fromarray(self._simple_canvas()).convert("RGB")
        out = preprocess_for_pipeline(pil)
        self.assertEqual(out.shape, (512, 512))
        self.assertEqual(out.dtype, np.uint8)

    def test_accepts_rgba_array(self) -> None:
        h, w = 200, 300
        rgba = np.full((h, w, 4), 255, dtype=np.uint8)
        rgba[50:150, 50:250, :3] = 20
        rgba[50:150, 50:250, 3] = 200  # semi-transparent ink
        out = preprocess_for_pipeline(rgba)
        self.assertEqual(out.shape, (512, 512))

    def test_accepts_grayscale_array(self) -> None:
        gray = self._simple_canvas()
        out = preprocess_for_pipeline(gray)
        self.assertEqual(out.shape, (512, 512))

    def test_deterministic(self) -> None:
        """Same input must produce byte-identical output on two calls."""
        canvas = self._simple_canvas()
        out1 = preprocess_for_pipeline(canvas)
        out2 = preprocess_for_pipeline(canvas)
        np.testing.assert_array_equal(out1, out2)

    def test_custom_config(self) -> None:
        config = PreprocessConfig(target_size=512, stroke_target_px=1)
        out = preprocess_for_pipeline(self._simple_canvas(), config=config)
        self.assertEqual(out.shape, (512, 512))

    def test_blank_canvas_stays_blank(self) -> None:
        """A fully white canvas should remain mostly white after preprocessing."""
        white = np.full((512, 512), 255, dtype=np.uint8)
        out = preprocess_for_pipeline(white)
        self.assertTrue(np.mean(out) > 200, "Blank canvas should remain mostly white")

    def test_rgba_alpha_composite_onto_white(self) -> None:
        """RGBA input with transparent background should composite ink onto white."""
        h, w = 200, 200
        # Fully transparent background (alpha=0) with black ink at alpha=255
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = 255  # white pixels but transparent
        rgba[:, :, 3] = 0     # fully transparent
        rgba[50:150, 50:150, :3] = 0    # black ink
        rgba[50:150, 50:150, 3] = 255   # fully opaque
        out = preprocess_for_pipeline(rgba)
        self.assertEqual(out.shape, (512, 512))
        # Some dark ink pixels should be present
        self.assertTrue(np.any(out < 100))


if __name__ == "__main__":
    unittest.main()
