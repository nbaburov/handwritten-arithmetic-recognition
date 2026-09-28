from __future__ import annotations
import unittest
from src.modeling.edge_types import NUM_EDGE_TYPES, pair_edge_type

class TestPairEdgeType(unittest.TestCase):
    def test_returns_valid_index(self):
        for a in ("digit_main","digit_carry","digit_borrow","operator","result_bar","divide_bracket"):
            for b in ("digit_main","digit_carry","digit_borrow","operator","result_bar","divide_bracket"):
                idx = pair_edge_type(a, b)
                self.assertGreaterEqual(idx, 0)
                self.assertLess(idx, NUM_EDGE_TYPES)

    def test_symmetric(self):
        """pair_edge_type(a, b) == pair_edge_type(b, a) always."""
        for a in ("digit_main","digit_carry","operator","result_bar","divide_bracket"):
            for b in ("digit_main","digit_carry","operator","result_bar","divide_bracket"):
                self.assertEqual(pair_edge_type(a, b), pair_edge_type(b, a))

    def test_main_to_main_is_0(self):
        self.assertEqual(pair_edge_type("digit_main", "digit_main"), 0)

    def test_main_to_carry_is_1(self):
        self.assertEqual(pair_edge_type("digit_main", "digit_carry"), 1)
        self.assertEqual(pair_edge_type("digit_carry", "digit_main"), 1)

    def test_carry_to_carry_is_2(self):
        self.assertEqual(pair_edge_type("digit_carry", "digit_carry"), 2)

    def test_digit_to_operator_is_3(self):
        for d in ("digit_main","digit_carry","digit_borrow"):
            self.assertEqual(pair_edge_type(d, "operator"), 3)

    def test_digit_to_result_bar_is_4(self):
        self.assertEqual(pair_edge_type("digit_main", "result_bar"), 4)

    def test_digit_to_divide_bracket_is_5(self):
        self.assertEqual(pair_edge_type("digit_main", "divide_bracket"), 5)

    def test_operator_to_result_bar_is_6(self):
        self.assertEqual(pair_edge_type("operator", "result_bar"), 6)

    def test_unknown_pair_is_7(self):
        self.assertEqual(pair_edge_type("result_bar", "divide_bracket"), 7)

    def test_num_edge_types_is_8(self):
        self.assertEqual(NUM_EDGE_TYPES, 8)

if __name__ == "__main__":
    unittest.main()
