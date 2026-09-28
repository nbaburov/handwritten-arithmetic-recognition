from __future__ import annotations

import unittest

from src.parsing.match_tokens import bbox_iou, greedy_match_scene


class TestEvalMetrics(unittest.TestCase):
    def test_iou_identical(self) -> None:
        box = [0.0, 0.0, 10.0, 10.0]
        self.assertAlmostEqual(bbox_iou(box, box), 1.0)

    def test_iou_disjoint(self) -> None:
        a = [0.0, 0.0, 5.0, 5.0]
        b = [10.0, 10.0, 15.0, 15.0]
        self.assertEqual(bbox_iou(a, b), 0.0)

    def test_iou_overlap(self) -> None:
        a = [0.0, 0.0, 10.0, 10.0]
        b = [5.0, 5.0, 15.0, 15.0]
        inter = 5.0 * 5.0
        area_a = area_b = 100.0
        expected = inter / (area_a + area_b - inter)
        self.assertAlmostEqual(bbox_iou(a, b), expected)

    def test_greedy_prefers_best_iou(self) -> None:
        gt = [{"flattened": "main_1", "bbox_px": [0.0, 0.0, 10.0, 10.0]}]
        pred = [{"flattened": "main_1", "bbox_px": [1.0, 1.0, 9.0, 9.0]}]
        m = greedy_match_scene(gt, pred, iou_thresh=0.3)
        self.assertEqual(m["matched"], 1)
        self.assertEqual(m["label_ok"], 1)

    def test_greedy_no_match_low_iou(self) -> None:
        gt = [{"flattened": "main_1", "bbox_px": [0.0, 0.0, 10.0, 10.0]}]
        pred = [{"flattened": "main_1", "bbox_px": [100.0, 100.0, 110.0, 110.0]}]
        m = greedy_match_scene(gt, pred, iou_thresh=0.5)
        self.assertEqual(m["matched"], 0)
        self.assertEqual(m["label_ok"], 0)


if __name__ == "__main__":
    unittest.main()
