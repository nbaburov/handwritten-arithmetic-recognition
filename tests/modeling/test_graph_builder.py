"""Tests for the dense-graph build_graph contract (Workstream A — GNN Dense Attention)."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.modeling.edge_types import NUM_EDGE_TYPES, pair_edge_type
from src.modeling.graph_builder import build_graph
from src.parsing.detection import Detection


def _det(
    label: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    confidence: float = 0.9,
) -> Detection:
    return Detection(label=label, confidence=confidence, x0=x0, y0=y0, x1=x1, y1=y1)


class TestBuildGraph(unittest.TestCase):
    def setUp(self) -> None:
        self.gray = np.full((512, 512), 255, dtype=np.uint8)
        self.gray[10:50, 10:50] = 0
        self.gray[10:50, 100:140] = 0
        self.gray[300:340, 10:50] = 0

    # ------------------------------------------------------------------
    # Basic shape contracts
    # ------------------------------------------------------------------

    def test_node_count(self) -> None:
        dets = [
            _det("digit_main", 10, 10, 50, 50),
            _det("digit_carry", 100, 10, 140, 50),
            _det("operator", 10, 300, 50, 340),
        ]
        data = build_graph(dets, self.gray)
        self.assertEqual(data.num_nodes, 3)

    def test_crops_shape(self) -> None:
        dets = [_det("digit_main", 10, 10, 50, 50)]
        data = build_graph(dets, self.gray)
        self.assertEqual(data.crops.shape, (1, 1, 28, 28))

    def test_node_feature_dim(self) -> None:
        dets = [_det("digit_main", 10, 10, 50, 50)]
        data = build_graph(dets, self.gray)
        self.assertEqual(data.x.shape[1], 13)

    def test_dense_edge_count(self) -> None:
        dets = [
            _det("digit_main", 10, 10, 50, 50),
            _det("digit_carry", 100, 10, 140, 50),
            _det("operator", 10, 300, 50, 340),
        ]
        data = build_graph(dets, self.gray)
        N = 3
        self.assertEqual(data.edge_index.shape[1], N * (N - 1))

    def test_edge_attr_dim(self) -> None:
        dets = [
            _det("digit_main", 10, 10, 50, 50),
            _det("digit_carry", 100, 10, 140, 50),
            _det("operator", 10, 300, 50, 340),
        ]
        data = build_graph(dets, self.gray)
        self.assertEqual(data.edge_attr.shape[1], 16)

    def test_edge_type_shape_and_validity(self) -> None:
        dets = [
            _det("digit_main", 10, 10, 50, 50),
            _det("digit_carry", 100, 10, 140, 50),
            _det("operator", 10, 300, 50, 340),
        ]
        data = build_graph(dets, self.gray)
        N = 3
        E = N * (N - 1)
        self.assertEqual(data.edge_type.shape, (E,))
        self.assertTrue((data.edge_type >= 0).all())
        self.assertTrue((data.edge_type < NUM_EDGE_TYPES).all())

    # ------------------------------------------------------------------
    # node_mask
    # ------------------------------------------------------------------

    def test_node_mask_removes_nodes(self) -> None:
        dets = [
            _det("digit_main", 10, 10, 50, 50),
            _det("digit_carry", 100, 10, 140, 50),
            _det("operator", 10, 300, 50, 340),
        ]
        mask = torch.tensor([True, True, False], dtype=torch.bool)
        data = build_graph(dets, self.gray, node_mask=mask)
        self.assertEqual(data.num_nodes, 2)
        # 2 nodes → E = 2*(2-1) = 2
        self.assertEqual(data.edge_index.shape[1], 2)

    # ------------------------------------------------------------------
    # truncation
    # ------------------------------------------------------------------

    def test_max_nodes_truncation(self) -> None:
        # 90 detections with varied confidences — lowest 10 should be dropped.
        rng = np.random.default_rng(0)
        confidences = rng.uniform(0.1, 1.0, 90).tolist()
        dets = [
            _det("digit_main", 10, 10, 50, 50, confidence=c)
            for c in confidences
        ]
        data = build_graph(dets, self.gray)
        self.assertEqual(data.num_nodes, 80)
        self.assertTrue(data.truncated)
        # Verify the 10 dropped are the lowest-confidence ones.
        sorted_confs = sorted(confidences, reverse=True)
        threshold = sorted_confs[79]  # lowest kept confidence
        kept_confs = sorted(confidences, reverse=True)[:80]
        dropped_confs = sorted(confidences, reverse=True)[80:]
        # Every dropped detection must have confidence <= every kept one.
        if dropped_confs:
            self.assertLessEqual(max(dropped_confs), min(kept_confs) + 1e-9)

    def test_no_truncation_flag(self) -> None:
        dets = [
            _det("digit_main", 10, 10, 50, 50),
            _det("digit_carry", 100, 10, 140, 50),
            _det("operator", 10, 300, 50, 340),
        ]
        data = build_graph(dets, self.gray)
        self.assertFalse(data.truncated)

    # ------------------------------------------------------------------
    # bbox
    # ------------------------------------------------------------------

    def test_bbox_stored(self) -> None:
        dets = [_det("digit_main", 10.0, 20.0, 50.0, 60.0)]
        data = build_graph(dets, self.gray)
        self.assertEqual(data.bbox.shape, (1, 4))
        self.assertAlmostEqual(float(data.bbox[0, 0]), 10.0)
        self.assertAlmostEqual(float(data.bbox[0, 3]), 60.0)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_single_node_no_edges(self) -> None:
        dets = [_det("digit_main", 10, 10, 50, 50)]
        data = build_graph(dets, self.gray)
        self.assertEqual(data.edge_index.shape[1], 0)

    def test_rejects_non_512_input(self) -> None:
        bad_gray = np.full((128, 128), 255, dtype=np.uint8)
        dets = [_det("digit_main", 10, 10, 50, 50)]
        with self.assertRaises(ValueError) as ctx:
            build_graph(dets, bad_gray)
        self.assertIn("preprocess_for_pipeline", str(ctx.exception))

    # ------------------------------------------------------------------
    # Role features
    # ------------------------------------------------------------------

    def test_role_importance_operator(self) -> None:
        dets = [_det("operator", 10, 10, 50, 50)]
        data = build_graph(dets, self.gray)
        # Last feature (index 12) is role_importance; operator → 1.0.
        self.assertAlmostEqual(float(data.x[0, 12]), 1.0)

    # ------------------------------------------------------------------
    # Scene-median normalisation
    # ------------------------------------------------------------------

    def test_col_delta_norm_uses_scene_median(self) -> None:
        """col_delta_norm (feature index 7) must be normalised by the median
        width of digit_main nodes only, not by the overall median."""
        # Two digit_main nodes with a specific width, plus one digit_carry.
        # digit_main bounding box: x0=10, x1=50 → bw=40 → bw/512 ≈ 0.07813
        # digit_carry bounding box: x0=200, x1=210 → bw=10 → very different
        dets = [
            _det("digit_main", 10, 10, 50, 50),    # bw=40
            _det("digit_main", 70, 10, 110, 50),   # bw=40
            _det("digit_carry", 200, 10, 210, 50), # bw=10, should not affect median
        ]
        data = build_graph(dets, self.gray)

        # Median digit_main width in normalised coords = 40/512
        expected_median_w = 40.0 / 512.0

        # Find edge from node 2 (digit_carry at cx≈205/512) to node 0 (digit_main at cx≈30/512).
        # dx = pos[0,0] - pos[2,0] (for edge src=2, dst=0)
        src = data.edge_index[0].tolist()
        dst = data.edge_index[1].tolist()
        for e, (s, d) in enumerate(zip(src, dst)):
            if s == 2 and d == 0:
                dx = data.x[d, 0] - data.x[s, 0]
                expected_col_delta = float(dx) / (expected_median_w + 1e-6)
                actual_col_delta = float(data.edge_attr[e, 7])
                self.assertAlmostEqual(actual_col_delta, expected_col_delta, places=4)
                return
        self.fail("Expected edge (2→0) not found in edge_index")

    # ------------------------------------------------------------------
    # Edge-type symmetry
    # ------------------------------------------------------------------

    def test_edge_type_symmetry(self) -> None:
        self.assertEqual(
            pair_edge_type("digit_main", "digit_carry"),
            pair_edge_type("digit_carry", "digit_main"),
        )

    # ------------------------------------------------------------------
    # W-ARCH-2: scene virtual node
    # ------------------------------------------------------------------

    def test_virtual_node_absent_by_default(self) -> None:
        """Default build_graph must not append a virtual node."""
        dets = [_det("digit_main", 10, 10, 50, 50)]
        data = build_graph(dets, self.gray)
        self.assertFalse(getattr(data, "has_virtual_node", False))
        self.assertEqual(data.num_nodes, 1)

    def test_virtual_node_appended_when_enabled(self) -> None:
        """With use_scene_virtual_node=True, one virtual node is appended after real nodes."""
        dets = [
            _det("digit_main", 10, 10, 50, 50),
            _det("operator", 100, 10, 140, 50),
        ]
        data = build_graph(dets, self.gray, use_scene_virtual_node=True)
        self.assertEqual(data.num_nodes, 3)          # 2 real + 1 virtual
        self.assertTrue(data.has_virtual_node)

    def test_virtual_node_bbox_spans_canvas(self) -> None:
        """Virtual node bbox must be [0, 0, w_img, h_img] = [0, 0, 512, 512]."""
        dets = [_det("digit_main", 10, 10, 50, 50)]
        data = build_graph(dets, self.gray, use_scene_virtual_node=True)
        # Virtual node is last (index 1 for 1 real + 1 virtual)
        self.assertAlmostEqual(float(data.bbox[-1, 0]), 0.0)
        self.assertAlmostEqual(float(data.bbox[-1, 1]), 0.0)
        self.assertAlmostEqual(float(data.bbox[-1, 2]), 512.0)
        self.assertAlmostEqual(float(data.bbox[-1, 3]), 512.0)

    def test_virtual_node_feature_dim_unchanged(self) -> None:
        """x must still have 13 columns when virtual node is appended."""
        dets = [_det("digit_main", 10, 10, 50, 50)]
        data = build_graph(dets, self.gray, use_scene_virtual_node=True)
        self.assertEqual(data.x.shape[1], 13)

    def test_virtual_node_crop_shape(self) -> None:
        """Virtual node crop must be (1, 28, 28) all-zeros."""
        dets = [_det("digit_main", 10, 10, 50, 50)]
        data = build_graph(dets, self.gray, use_scene_virtual_node=True)
        vnode_crop = data.crops[-1]  # last node is virtual
        self.assertEqual(vnode_crop.shape, (1, 28, 28))
        self.assertTrue(torch.all(vnode_crop == 0).item())

    def test_max_nodes_truncation_with_virtual_node(self) -> None:
        """With virtual node enabled, 90 detections truncate to 80 real + 1 virtual = 81."""
        rng = np.random.default_rng(0)
        confidences = rng.uniform(0.1, 1.0, 90).tolist()
        dets = [
            _det("digit_main", 10, 10, 50, 50, confidence=c)
            for c in confidences
        ]
        data = build_graph(dets, self.gray, use_scene_virtual_node=True)
        self.assertEqual(data.num_nodes, 81)   # 80 real + 1 virtual
        self.assertTrue(data.truncated)
        self.assertTrue(data.has_virtual_node)


class TestBuildGraphGivenFlag(unittest.TestCase):
    """Foundation 1: given flag survives build_graph (sort + MAX_NODES truncation)."""

    def setUp(self) -> None:
        self.gray = np.full((512, 512), 255, dtype=np.uint8)

    def _given_det(self, x0: float, y0: float, x1: float, y1: float) -> Detection:
        return Detection(
            label="digit_main",
            confidence=1.0,
            x0=x0, y0=y0, x1=x1, y1=y1,
            given=True,
        )

    def _normal_det(self, x0: float, y0: float, x1: float, y1: float, conf: float = 0.5) -> Detection:
        return Detection(
            label="digit_main",
            confidence=conf,
            x0=x0, y0=y0, x1=x1, y1=y1,
            given=False,
        )

    def test_given_flag_default_false_in_data(self) -> None:
        dets = [self._normal_det(10, 10, 50, 50)]
        data = build_graph(dets, self.gray)
        self.assertEqual(data.given.shape, (1,))
        self.assertFalse(data.given[0].item())

    def test_given_flag_true_propagated_to_data(self) -> None:
        dets = [self._given_det(10, 10, 50, 50)]
        data = build_graph(dets, self.gray)
        self.assertEqual(data.given.shape, (1,))
        self.assertTrue(data.given[0].item())

    def test_given_flag_mixed_detections(self) -> None:
        # One given (conf=1.0) mixed with two normal detections.
        dets = [
            self._given_det(10, 10, 50, 50),
            self._normal_det(100, 10, 140, 50, conf=0.7),
            self._normal_det(200, 10, 240, 50, conf=0.5),
        ]
        data = build_graph(dets, self.gray)
        self.assertEqual(data.given.shape, (3,))
        # Exactly one given node.
        self.assertEqual(data.given.sum().item(), 1)

    def test_given_flag_survives_max_nodes_truncation(self) -> None:
        # Build 82 detections: 1 given with conf=1.0 + 81 normal with conf<1.
        # MAX_NODES defaults to 80. After truncation the given node must survive
        # because it has the highest confidence.
        given_det = self._given_det(10, 10, 50, 50)  # conf=1.0
        normal_dets = [
            self._normal_det(float(i * 5), 60, float(i * 5 + 4), 64, conf=0.1 + i * 0.001)
            for i in range(81)
        ]
        dets = [given_det] + normal_dets
        data = build_graph(dets, self.gray, max_nodes=80)
        self.assertTrue(data.truncated)
        self.assertEqual(data.num_nodes, 80)
        # The given node must be among the surviving 80.
        self.assertTrue(data.given.any().item(), "given node was truncated but should survive")

    def test_given_tensor_dtype_is_bool(self) -> None:
        dets = [self._given_det(10, 10, 50, 50), self._normal_det(100, 10, 140, 50)]
        data = build_graph(dets, self.gray)
        self.assertEqual(data.given.dtype, torch.bool)

    def test_virtual_node_given_flag_is_false(self) -> None:
        dets = [self._given_det(10, 10, 50, 50)]
        data = build_graph(dets, self.gray, use_scene_virtual_node=True)
        # 1 real + 1 virtual; virtual node's given flag must be False.
        self.assertEqual(data.num_nodes, 2)
        self.assertTrue(data.given[0].item())   # real given node
        self.assertFalse(data.given[1].item())  # virtual node


if __name__ == "__main__":
    unittest.main()
