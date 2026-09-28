"""Phase 0 / WS0c tests: per-class AP extraction from ultralytics val result.

Verifies:
- Correct index-to-name mapping using YOLO_CLASS_NAMES.
- Empty-array path writes {} and does not raise.
- Missing 'box' attribute writes {} and does not raise.
- All 6 coarse classes present when val result is fully populated.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np

from src.training.run import _extract_per_class_ap
from src.core.ontology import YOLO_CLASS_NAMES


def _make_val_result(ap_class_index, ap50_values, ap_values):
    """Build a minimal mock val result with populated box.ap_class_index/ap50/ap."""
    box = MagicMock()
    box.ap_class_index = list(ap_class_index)
    box.ap50 = np.array(ap50_values, dtype=float)
    box.ap = np.array(ap_values, dtype=float)

    val_result = MagicMock()
    val_result.box = box
    return val_result


class TestExtractPerClassApAllClasses(unittest.TestCase):
    """All 6 coarse class indices produce correctly named entries."""

    def test_all_six_classes_mapped(self) -> None:
        """When val result has all 6 class indices, all 6 names appear in output."""
        nc = len(YOLO_CLASS_NAMES)
        ap50_values = [0.90, 0.85, 0.80, 0.95, 0.92, 0.88]
        ap_values = [0.70, 0.65, 0.60, 0.75, 0.72, 0.68]
        val_result = _make_val_result(
            ap_class_index=list(range(nc)),
            ap50_values=ap50_values,
            ap_values=ap_values,
        )
        result = _extract_per_class_ap(val_result)
        self.assertEqual(len(result), nc)
        for name in YOLO_CLASS_NAMES:
            self.assertIn(name, result, f"Expected class '{name}' in per-class output")
            self.assertIn("ap50", result[name])
            self.assertIn("ap50_95", result[name])

    def test_ap50_values_correct(self) -> None:
        """AP50 values are correctly extracted per class index."""
        ap50_values = [0.91, 0.82, 0.73, 0.94, 0.87, 0.76]
        ap_values = [0.71, 0.62, 0.53, 0.74, 0.67, 0.56]
        val_result = _make_val_result(
            ap_class_index=list(range(len(YOLO_CLASS_NAMES))),
            ap50_values=ap50_values,
            ap_values=ap_values,
        )
        result = _extract_per_class_ap(val_result)
        for i, name in enumerate(YOLO_CLASS_NAMES):
            self.assertAlmostEqual(result[name]["ap50"], ap50_values[i], places=4,
                                   msg=f"ap50 mismatch for {name}")

    def test_partial_class_set(self) -> None:
        """Subset of class indices (e.g. only operator and result_bar) maps correctly."""
        # Only indices 3 (operator) and 4 (result_bar)
        val_result = _make_val_result(
            ap_class_index=[3, 4],
            ap50_values=[0.95, 0.88],
            ap_values=[0.75, 0.68],
        )
        result = _extract_per_class_ap(val_result)
        self.assertEqual(len(result), 2)
        self.assertIn("operator", result)
        self.assertIn("result_bar", result)
        self.assertNotIn("digit_main", result)


class TestExtractPerClassApEmptyArrays(unittest.TestCase):
    """Empty arrays write {} and do not raise."""

    def test_empty_ap_class_index_returns_empty_dict(self) -> None:
        """When ap_class_index is empty list, result is {} with no exception."""
        box = MagicMock()
        box.ap_class_index = []
        box.ap50 = np.array([])
        box.ap = np.array([])
        val_result = MagicMock()
        val_result.box = box
        result = _extract_per_class_ap(val_result)
        self.assertEqual(result, {})

    def test_missing_box_attribute_returns_empty_dict(self) -> None:
        """When val_result has no 'box' attribute, result is {} with no exception."""
        val_result = MagicMock(spec=[])  # no attributes
        result = _extract_per_class_ap(val_result)
        self.assertEqual(result, {})

    def test_none_box_returns_empty_dict(self) -> None:
        """When box is None, result is {} with no exception."""
        val_result = MagicMock()
        val_result.box = None
        result = _extract_per_class_ap(val_result)
        self.assertEqual(result, {})

    def test_exception_in_extraction_returns_empty_dict(self) -> None:
        """If box raises unexpectedly, result is {} with no exception propagated."""
        val_result = MagicMock()
        val_result.box = MagicMock()
        val_result.box.ap_class_index = [0]
        # Make ap50 raise on index access
        val_result.box.ap50 = MagicMock(side_effect=RuntimeError("mock error"))
        # Should not raise
        result = _extract_per_class_ap(val_result)
        self.assertIsInstance(result, dict)


class TestExtractPerClassApReturnTypes(unittest.TestCase):
    """Return values are plain Python floats, not numpy scalars."""

    def test_values_are_python_floats(self) -> None:
        val_result = _make_val_result(
            ap_class_index=[0],
            ap50_values=[0.92],
            ap_values=[0.72],
        )
        result = _extract_per_class_ap(val_result)
        self.assertEqual(len(result), 1)
        name = YOLO_CLASS_NAMES[0]
        self.assertIsInstance(result[name]["ap50"], float)
        self.assertIsInstance(result[name]["ap50_95"], float)


if __name__ == "__main__":
    unittest.main()
