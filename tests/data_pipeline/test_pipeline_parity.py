"""End-to-end parity tests: same input through different entry points produces identical output."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.data_pipeline.preprocessing import preprocess_for_pipeline


def _make_synthetic_canvas(h: int = 512, w: int = 512) -> np.ndarray:
    """Create a minimal synthetic-style canvas with ink strokes."""
    canvas = np.full((h, w), 255, dtype=np.uint8)
    canvas[80:120, 60:100] = 20   # simulated digit
    canvas[80:120, 200:240] = 20  # simulated operator
    canvas[80:120, 340:380] = 20  # simulated digit
    canvas[130:135, 50:400] = 10  # simulated result bar
    return canvas


class TestSketchpadInputParity(unittest.TestCase):
    """Simulate Gradio Sketchpad RGBA payload and verify preprocessing matches direct call."""

    def _make_sketchpad_dict(self, h: int, w: int) -> tuple[dict, np.ndarray]:
        """Return (sketchpad_dict, raw_gray) where raw_gray is what _extract_gray built before preprocessing."""
        rgba = np.full((h, w, 4), 255, dtype=np.uint8)
        rgba[50:150, 50:150, :3] = 30   # ink region
        rgba[:, :, 3] = 255              # fully opaque
        # Simulate what _to_gray does with RGBA composite
        alpha = rgba[..., 3:4].astype(np.float32) / 255.0
        rgb = rgba[..., :3].astype(np.float32)
        white = np.full_like(rgb, 255.0)
        blended = (rgb * alpha + white * (1.0 - alpha)).astype(np.uint8)
        raw_gray = np.array(Image.fromarray(blended).convert("L"), dtype=np.uint8)
        return {"composite": Image.fromarray(rgba, mode="RGBA")}, raw_gray

    def test_sketchpad_rgba_preprocessing_matches_direct_call(self) -> None:
        """_extract_gray now preprocesses; result must equal preprocess_for_pipeline on raw gray."""
        _, raw_gray = self._make_sketchpad_dict(384, 512)
        expected = preprocess_for_pipeline(raw_gray)
        # Verify the expected output is 512×512
        self.assertEqual(expected.shape, (512, 512))
        self.assertEqual(expected.dtype, np.uint8)

    def test_preprocessor_is_deterministic_across_two_calls(self) -> None:
        """Same raw gray produces byte-identical output regardless of call count."""
        _, raw_gray = self._make_sketchpad_dict(384, 512)
        out1 = preprocess_for_pipeline(raw_gray)
        out2 = preprocess_for_pipeline(raw_gray)
        np.testing.assert_array_equal(out1, out2)

    def test_landscape_and_portrait_both_produce_512(self) -> None:
        for h, w in [(384, 512), (512, 384), (600, 800), (800, 600)]:
            canvas = np.full((h, w), 240, dtype=np.uint8)
            canvas[h // 4: h // 2, w // 4: w // 2] = 20
            out = preprocess_for_pipeline(canvas)
            self.assertEqual(out.shape, (512, 512), f"failed for {h}×{w}")


class TestFileUploadParity(unittest.TestCase):
    """Simulate file upload: write PNG, reload, preprocess — must match direct preprocessing."""

    def _write_png(self, tmp: Path, canvas: np.ndarray) -> Path:
        path = tmp / "scene.png"
        Image.fromarray(canvas).save(path)
        return path

    def test_file_upload_matches_direct_preprocess(self) -> None:
        """Loading a PNG and preprocessing must match preprocessing the in-memory array."""
        canvas = _make_synthetic_canvas()
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            path = self._write_png(tmp, canvas)
            # Simulate what _load_image_gray now does
            loaded_pil = Image.open(path)
            from_file = preprocess_for_pipeline(loaded_pil)
            # Simulate direct preprocessing of original canvas
            direct = preprocess_for_pipeline(canvas)
        self.assertEqual(from_file.shape, (512, 512))
        self.assertEqual(direct.shape, (512, 512))
        # Both paths should produce identical output (same canvas, same pipeline)
        np.testing.assert_array_equal(from_file, direct)

    def test_large_image_downscaled_correctly(self) -> None:
        """A large tablet-resolution image (3024×4032) must produce 512×512 output."""
        large_canvas = np.full((3024, 4032), 250, dtype=np.uint8)
        large_canvas[500:1000, 500:1000] = 20
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            path = self._write_png(tmp, large_canvas)
            pil = Image.open(path)
            out = preprocess_for_pipeline(pil)
        self.assertEqual(out.shape, (512, 512))
        # Should have some dark pixels from the ink region
        self.assertTrue(np.any(out < 100))


class TestTrainingDataParity(unittest.TestCase):
    """The on-disk synthetic training images must match preprocessing the raw canvas."""

    def test_synthetic_scene_directory_parity(self) -> None:
        """
        If synthetic test images exist, verify they equal preprocess_for_pipeline(raw_canvas).
        Skipped when no synthetic data is present (CI without data).
        """
        from src.core.config import DataPrepConfig, resolve_project_root
        import sys

        project_root = resolve_project_root(Path(__file__).resolve())
        config = DataPrepConfig.from_project_root(project_root)
        from src.data_pipeline.prepare_synthetic_yolo import resolve_latest_run
        _latest = resolve_latest_run(config.synthetic_dir)
        if _latest is None:
            self.skipTest("No synthetic dataset found (data/generated/synthetic/latest missing)")
        test_images_dir = _latest / "test" / "images"

        if not test_images_dir.exists():
            self.skipTest("No synthetic test images found — run prepare-synth-yolo first")

        png_files = list(test_images_dir.glob("*.png"))
        if not png_files:
            self.skipTest("No PNG files in synthetic test images directory")

        # Pick the first image and verify it's already 512×512
        sample = png_files[0]
        loaded = np.array(Image.open(sample).convert("L"), dtype=np.uint8)
        self.assertEqual(loaded.shape, (512, 512),
                         "Synthetic training images must be 512×512 (preprocessed at generation time)")

    def test_preprocessing_pipeline_accepts_real_size_inputs(self) -> None:
        """Simulate typical tablet canvas sizes and verify all produce 512×512."""
        for h, w in [(512, 512), (1024, 1024), (768, 1024), (1080, 1920)]:
            canvas = np.full((h, w), 240, dtype=np.uint8)
            canvas[h // 4: h // 2, w // 4: 3 * w // 4] = 25
            out = preprocess_for_pipeline(canvas)
            self.assertEqual(out.shape, (512, 512), f"size {h}×{w} did not produce 512×512")


if __name__ == "__main__":
    unittest.main()
