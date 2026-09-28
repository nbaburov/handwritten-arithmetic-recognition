from __future__ import annotations

import unittest

from src.core.cluster_metrics import compute_scene_metrics


class TestGNNMetrics(unittest.TestCase):
    def test_perfect(self) -> None:
        preds = [("main_3", 0, 0), ("carry_1", 0, 1)]
        gts   = [("main_3", 0, 0), ("carry_1", 0, 1)]
        m = compute_scene_metrics(preds, gts, eq_pred="add", eq_gt="add")
        self.assertAlmostEqual(m["fine_label_accuracy"], 1.0)
        self.assertAlmostEqual(m["row_accuracy"], 1.0)
        self.assertAlmostEqual(m["col_accuracy"], 1.0)
        self.assertAlmostEqual(m["equation_type_accuracy"], 1.0)
        self.assertAlmostEqual(m["exact_scene_match"], 1.0)

    def test_all_wrong(self) -> None:
        m = compute_scene_metrics(
            [("main_9", 1, 5)], [("main_3", 0, 0)],
            eq_pred="subtract", eq_gt="add")
        self.assertAlmostEqual(m["fine_label_accuracy"], 0.0)
        self.assertAlmostEqual(m["exact_scene_match"], 0.0)

    def test_partial(self) -> None:
        preds = [("main_3", 0, 0), ("main_9", 0, 1)]
        gts   = [("main_3", 0, 0), ("carry_1", 0, 1)]
        m = compute_scene_metrics(preds, gts, eq_pred="add", eq_gt="add")
        self.assertAlmostEqual(m["fine_label_accuracy"], 0.5)
        self.assertAlmostEqual(m["exact_scene_match"], 0.0)


if __name__ == "__main__":
    unittest.main()
