"""Tests for the Detection dataclass — Foundation 1: given flag."""
from __future__ import annotations

import unittest

from src.parsing.detection import Detection


def _det(**kwargs) -> Detection:
    defaults = dict(label="digit_main", confidence=0.9, x0=10.0, y0=10.0, x1=50.0, y1=50.0)
    defaults.update(kwargs)
    return Detection(**defaults)


class TestDetectionGivenFlag(unittest.TestCase):
    def test_given_defaults_false(self) -> None:
        d = _det()
        self.assertFalse(d.given)

    def test_given_true_roundtrips(self) -> None:
        d = _det(given=True)
        self.assertTrue(d.given)

    def test_given_false_explicit(self) -> None:
        d = _det(given=False)
        self.assertFalse(d.given)

    def test_cx_cy_still_work(self) -> None:
        d = _det(x0=10.0, x1=50.0, y0=20.0, y1=60.0, given=True)
        self.assertAlmostEqual(d.cx, 30.0)
        self.assertAlmostEqual(d.cy, 40.0)

    def test_frozen(self) -> None:
        d = _det()
        with self.assertRaises((AttributeError, TypeError)):
            d.given = True  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
