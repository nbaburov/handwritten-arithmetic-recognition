from __future__ import annotations
import unittest
from src.core.ontology import YOLO_CLASS_NAMES, yolo_class_id_to_name

class TestOntology(unittest.TestCase):
    def test_divide_bracket_in_class_names(self) -> None:
        self.assertIn("divide_bracket", YOLO_CLASS_NAMES)

    def test_divide_bracket_is_sixth_class(self) -> None:
        self.assertEqual(YOLO_CLASS_NAMES[5], "divide_bracket")

    def test_yolo_class_id_to_name_valid(self) -> None:
        self.assertEqual(yolo_class_id_to_name(0), "digit_main")
        self.assertEqual(yolo_class_id_to_name(5), "divide_bracket")

    def test_yolo_class_id_to_name_out_of_range(self) -> None:
        self.assertIsNone(yolo_class_id_to_name(99))
        self.assertIsNone(yolo_class_id_to_name(-1))

    def test_excluded_labels_empty(self) -> None:
        from src.core.ontology import EXCLUDED_DATASET_LABELS
        self.assertEqual(len(EXCLUDED_DATASET_LABELS), 0)

    def test_stage2_labels_includes_div_bracket(self) -> None:
        from src.core.ontology import stage2_labels_ordered
        labels = stage2_labels_ordered()
        self.assertIn("div_bracket", labels)
        self.assertEqual(len(labels), 36)

if __name__ == "__main__":
    unittest.main()
