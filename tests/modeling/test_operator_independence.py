"""W-ARCH-OPINDEP-TEST (iter9 Layer 1).

Verifies that row and column cluster assignment is operator-independent:
_cluster_by_coordinate and SymbolGNN.predict produce identical spatial cluster
IDs regardless of the YOLO role labels attached to nodes, because clustering
depends only on bbox geometry.
"""
from __future__ import annotations

import unittest
import torch
from torch_geometric.data import Data

from src.modeling.gnn import (
    _cluster_by_coordinate,
    CropBackbone,
    SymbolGNN,
)


def _make_predict_data(n: int, bbox: torch.Tensor, role_onehot: torch.Tensor) -> Data:
    """Build a minimal Data object accepted by SymbolGNN.predict.

    The spatial feature vector x is 13-dim:
      positions 0-4  : 5 geometric scalars (cx, cy, w, h, area -- all zeros here)
      positions 5-10 : 6-dim YOLO role one-hot
      positions 11-12: 2 additional geometric features (zeros)

    role_onehot is shape (N, 6).
    """
    x_geom_pre = torch.zeros(n, 5)
    x_geom_post = torch.zeros(n, 2)
    x = torch.cat([x_geom_pre, role_onehot, x_geom_post], dim=1)  # (N, 13)

    crops = torch.zeros(n, 1, 28, 28)

    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i)
                dst.append(j)
    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    E = len(src)
    edge_attr = torch.zeros(E, 16)
    edge_type = torch.zeros(E, dtype=torch.long)
    batch = torch.zeros(n, dtype=torch.long)

    return Data(
        crops=crops,
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type,
        bbox=bbox,
        batch=batch,
        num_nodes=n,
    )


def _all_operator_onehot(n: int) -> torch.Tensor:
    """All nodes labeled as 'operator' (YOLO class index 3)."""
    # YOLO_CLASS_NAMES order: digit_main(0), digit_carry(1), digit_borrow(2),
    #                          operator(3), result_bar(4), divide_bracket(5)
    t = torch.zeros(n, 6)
    t[:, 3] = 1.0
    return t


def _all_digit_main_onehot(n: int) -> torch.Tensor:
    """All nodes labeled as 'digit_main' (YOLO class index 0)."""
    t = torch.zeros(n, 6)
    t[:, 0] = 1.0
    return t


def _mixed_role_onehot(n: int) -> torch.Tensor:
    """Alternating digit_main / operator roles."""
    t = torch.zeros(n, 6)
    for i in range(n):
        col = 0 if i % 2 == 0 else 3
        t[i, col] = 1.0
    return t


def _make_2row_4col_bbox() -> torch.Tensor:
    """Construct 8 bboxes in a 2-row x 4-col layout.

    Row 0 (y ~ 10-30): nodes 0-3  at x positions 10, 50, 90, 130
    Row 1 (y ~ 80-100): nodes 4-7  at x positions 10, 50, 90, 130
    """
    bboxes = [
        [10,  10,  40,  30],
        [50,  10,  80,  30],
        [90,  10,  120, 30],
        [130, 10,  160, 30],
        [10,  80,  40,  100],
        [50,  80,  80,  100],
        [90,  80,  120, 100],
        [130, 80,  160, 100],
    ]
    return torch.tensor(bboxes, dtype=torch.float32)


class TestClusterByCoordinateSignature(unittest.TestCase):
    """Test 1: _cluster_by_coordinate takes only (bbox, axis) -- no label/role args."""

    def test_signature_no_label_arg(self) -> None:
        import inspect
        sig = inspect.signature(_cluster_by_coordinate)
        params = list(sig.parameters.keys())
        self.assertEqual(
            params,
            ["bbox", "axis"],
            msg=(
                "_cluster_by_coordinate must accept exactly (bbox, axis). "
                f"Got: {params}. Any addition of label/role params breaks "
                "operator-independence."
            ),
        )

    def test_return_type_is_list(self) -> None:
        bbox = _make_2row_4col_bbox()
        result = _cluster_by_coordinate(bbox, axis=1)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 8)


class TestClusterByCoordinateGeometryOnly(unittest.TestCase):
    """Test 2: clustering is purely geometric -- same bbox -> same clusters."""

    def setUp(self) -> None:
        self.bbox = _make_2row_4col_bbox()

    def test_row_clusters_split_correctly(self) -> None:
        """8 nodes in 2 clearly separated rows must yield exactly 2 distinct cluster IDs."""
        row_ids = _cluster_by_coordinate(self.bbox, axis=1)
        self.assertEqual(len(set(row_ids)), 2, msg=f"Expected 2 row clusters, got {set(row_ids)}")

    def test_col_clusters_split_correctly(self) -> None:
        """8 nodes in 4 clearly separated columns must yield exactly 4 distinct cluster IDs."""
        col_ids = _cluster_by_coordinate(self.bbox, axis=0)
        self.assertEqual(len(set(col_ids)), 4, msg=f"Expected 4 col clusters, got {set(col_ids)}")

    def test_identical_bbox_produces_identical_row_clusters(self) -> None:
        """Two calls with the same bbox tensor must return the same cluster list."""
        ids_a = _cluster_by_coordinate(self.bbox, axis=1)
        ids_b = _cluster_by_coordinate(self.bbox, axis=1)
        self.assertEqual(ids_a, ids_b)

    def test_row_clusters_group_rows(self) -> None:
        """Nodes 0-3 (row 0) and nodes 4-7 (row 1) must share their respective cluster IDs."""
        row_ids = _cluster_by_coordinate(self.bbox, axis=1)
        # All four nodes in the same row must have identical cluster ID.
        self.assertEqual(row_ids[0], row_ids[1])
        self.assertEqual(row_ids[1], row_ids[2])
        self.assertEqual(row_ids[2], row_ids[3])
        # Row 1 nodes must form a different cluster from row 0.
        self.assertNotEqual(row_ids[0], row_ids[4])
        # All row 1 nodes share a cluster ID.
        self.assertEqual(row_ids[4], row_ids[5])
        self.assertEqual(row_ids[5], row_ids[6])
        self.assertEqual(row_ids[6], row_ids[7])

    def test_col_clusters_group_columns(self) -> None:
        """Nodes at x~10, x~50, x~90, x~130 must form 4 distinct col cluster IDs."""
        col_ids = _cluster_by_coordinate(self.bbox, axis=0)
        # Node 0 and Node 4 share column x~10.
        self.assertEqual(col_ids[0], col_ids[4])
        # Node 1 and Node 5 share column x~50.
        self.assertEqual(col_ids[1], col_ids[5])
        # All four columns must differ.
        col_set = {col_ids[0], col_ids[1], col_ids[2], col_ids[3]}
        self.assertEqual(len(col_set), 4)


class TestSymbolGNNPredictOperatorIndependence(unittest.TestCase):
    """Test 3 + 4: SymbolGNN.predict gives identical row/col cluster IDs
    regardless of the YOLO role labels attached to nodes."""

    def setUp(self) -> None:
        self.model = SymbolGNN(CropBackbone())
        self.model.train(False)
        self.bbox = _make_2row_4col_bbox()
        n = self.bbox.shape[0]  # 8

        # Three graphs with identical bbox but different YOLO role labels.
        self.data_all_operator = _make_predict_data(
            n, self.bbox, _all_operator_onehot(n)
        )
        self.data_all_digit = _make_predict_data(
            n, self.bbox, _all_digit_main_onehot(n)
        )
        self.data_mixed = _make_predict_data(
            n, self.bbox, _mixed_role_onehot(n)
        )

    def _extract_cluster_ids(self, data: Data) -> tuple[list[int], list[int]]:
        preds, _eq, _logits = self.model.predict(data)
        row_ids = [p.row_cluster_id for p in preds]
        col_ids = [p.col_cluster_id for p in preds]
        return row_ids, col_ids

    def test_all_operator_vs_all_digit_row_clusters_identical(self) -> None:
        row_op, _ = self._extract_cluster_ids(self.data_all_operator)
        row_dg, _ = self._extract_cluster_ids(self.data_all_digit)
        self.assertEqual(
            row_op, row_dg,
            msg=(
                "row_cluster_ids differ between all-operator and all-digit role "
                "assignments with identical bbox. Clustering must be bbox-only."
            ),
        )

    def test_all_operator_vs_all_digit_col_clusters_identical(self) -> None:
        _, col_op = self._extract_cluster_ids(self.data_all_operator)
        _, col_dg = self._extract_cluster_ids(self.data_all_digit)
        self.assertEqual(
            col_op, col_dg,
            msg=(
                "col_cluster_ids differ between all-operator and all-digit role "
                "assignments with identical bbox."
            ),
        )

    def test_mixed_roles_row_clusters_same_as_all_digit(self) -> None:
        row_mx, _ = self._extract_cluster_ids(self.data_mixed)
        row_dg, _ = self._extract_cluster_ids(self.data_all_digit)
        self.assertEqual(row_mx, row_dg)

    def test_mixed_roles_col_clusters_same_as_all_digit(self) -> None:
        _, col_mx = self._extract_cluster_ids(self.data_mixed)
        _, col_dg = self._extract_cluster_ids(self.data_all_digit)
        self.assertEqual(col_mx, col_dg)

    def test_two_row_clusters_present_regardless_of_role(self) -> None:
        """2-row geometry always produces 2 row clusters regardless of role label."""
        for label, data in [
            ("all_operator", self.data_all_operator),
            ("all_digit", self.data_all_digit),
            ("mixed", self.data_mixed),
        ]:
            row_ids, _ = self._extract_cluster_ids(data)
            n_row_clusters = len(set(row_ids))
            self.assertEqual(
                n_row_clusters,
                2,
                msg=f"Expected 2 row clusters for {label}, got {n_row_clusters}: {row_ids}",
            )

    def test_four_col_clusters_present_regardless_of_role(self) -> None:
        """4-column geometry always produces 4 col clusters regardless of role."""
        for label, data in [
            ("all_operator", self.data_all_operator),
            ("all_digit", self.data_all_digit),
            ("mixed", self.data_mixed),
        ]:
            _, col_ids = self._extract_cluster_ids(data)
            n_col_clusters = len(set(col_ids))
            self.assertEqual(
                n_col_clusters,
                4,
                msg=f"Expected 4 col clusters for {label}, got {n_col_clusters}: {col_ids}",
            )


class TestClusterByCoordinateEdgeCases(unittest.TestCase):
    """Test 5: edge cases for _cluster_by_coordinate."""

    def test_empty_bbox_returns_empty_list(self) -> None:
        bbox = torch.zeros(0, 4)
        result = _cluster_by_coordinate(bbox, axis=0)
        self.assertEqual(result, [])

    def test_single_node_returns_zero(self) -> None:
        bbox = torch.tensor([[10.0, 20.0, 40.0, 50.0]])
        result = _cluster_by_coordinate(bbox, axis=1)
        self.assertEqual(result, [0])

    def test_all_identical_bboxes_single_cluster(self) -> None:
        """When all bboxes are identical the gap is 0, at or below gap_thresh.
        All nodes fall into cluster 0 (no gap exceeds threshold).
        """
        bbox = torch.tensor([[10.0, 20.0, 40.0, 50.0]] * 5)
        row_ids = _cluster_by_coordinate(bbox, axis=1)
        # All identical midpoints: no gap ever exceeds 0.5 * median_size.
        self.assertEqual(len(set(row_ids)), 1)

    def test_two_well_separated_nodes_two_clusters(self) -> None:
        """Two nodes far apart on an axis must each get their own cluster."""
        bbox = torch.tensor([
            [0.0,   0.0,  20.0, 20.0],
            [0.0, 200.0,  20.0, 220.0],
        ])
        row_ids = _cluster_by_coordinate(bbox, axis=1)
        self.assertNotEqual(row_ids[0], row_ids[1])
        self.assertEqual(len(set(row_ids)), 2)


if __name__ == "__main__":
    unittest.main()
