from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


def _write_label(d: Path, name: str, lines: list[str]) -> None:
    (d / name).write_text("\n".join(lines), encoding="utf-8")


class TestValidateYoloLabels(unittest.TestCase):

    def test_valid_labels_pass(self):
        """Known-good label files must produce empty error list."""
        from src.data_pipeline.validate_yolo_labels import validate_yolo_labels
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_label(d, "img0.txt", ["0 0.5 0.5 0.3 0.4"])
            _write_label(d, "img1.txt", ["3 0.1 0.9 0.05 0.08", "5 0.8 0.2 0.1 0.15"])
            errors = validate_yolo_labels(d)
            self.assertEqual(errors, [], f"Unexpected errors: {errors}")

    def test_negative_cx_raises(self):
        """Negative cx must raise RuntimeError."""
        from src.data_pipeline.validate_yolo_labels import validate_yolo_labels
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_label(d, "bad.txt", ["0 -0.02 0.5 0.1 0.1"])
            with self.assertRaises(RuntimeError):
                validate_yolo_labels(d)

    def test_cx_gt_1_raises(self):
        """cx > 1.0 must raise RuntimeError."""
        from src.data_pipeline.validate_yolo_labels import validate_yolo_labels
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_label(d, "bad.txt", ["0 1.1 0.5 0.1 0.1"])
            with self.assertRaises(RuntimeError):
                validate_yolo_labels(d)

    def test_empty_label_file_warns_not_raises(self):
        """Empty label file must not raise — some completion stages produce empty scenes."""
        from src.data_pipeline.validate_yolo_labels import validate_yolo_labels
        import logging
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_label(d, "empty.txt", [])
            # Must not raise
            try:
                errors = validate_yolo_labels(d)
            except Exception as e:
                self.fail(f"validate_yolo_labels raised on empty file: {e}")

    def test_invalid_class_id_raises(self):
        """class_id outside [0, n_classes-1] must raise RuntimeError."""
        from src.data_pipeline.validate_yolo_labels import validate_yolo_labels
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_label(d, "bad.txt", ["99 0.5 0.5 0.1 0.1"])
            with self.assertRaises(RuntimeError):
                validate_yolo_labels(d)


if __name__ == "__main__":
    unittest.main()
