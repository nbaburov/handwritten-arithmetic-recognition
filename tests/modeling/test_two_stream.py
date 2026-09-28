"""Tests for the W-ARCH-1 two-stream GATv2 architecture (iter9).

Tests cover:
  - _spatial_x helper: shape contract and role-stripping behaviour.
  - Hard guarantee: spatial stream output is identical for two inputs that
    differ only in the role one-hot block (dims 5..10).
"""
from __future__ import annotations

import unittest

import torch
from torch_geometric.data import Data

from src.modeling.gnn import (
    CropBackbone,
    SPATIAL_STREAM_INPUT_DIM,
    SymbolGNN,
    _spatial_x,
)
from src.core.ontology import YOLO_CLASS_NAMES


def _make_full_graph(n: int, role_onehot: torch.Tensor) -> Data:
    """Build a minimal graph Data object with a specified role one-hot block.

    Args:
        n: Number of nodes.
        role_onehot: (N, 6) tensor for dims 5..10 of data.x.

    Returns:
        Data with fields crops, x, edge_index, edge_attr, edge_type, bbox, batch.
    """
    x_geom_pre = torch.zeros(n, 5)        # dims 0-4: geometry
    x_geom_post = torch.zeros(n, 2)       # dims 11-12: confidence + importance
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
    bbox = torch.zeros(n, 4)
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


class TestSpatialXHelper(unittest.TestCase):
    """Tests for the _spatial_x module-level helper function."""

    def test_output_shape(self) -> None:
        """_spatial_x must return (N, SPATIAL_STREAM_INPUT_DIM) for any N."""
        for n in (1, 5, 10):
            x = torch.randn(n, 13)
            out = _spatial_x(x)
            self.assertEqual(
                out.shape,
                (n, SPATIAL_STREAM_INPUT_DIM),
                msg=f"Expected ({n}, {SPATIAL_STREAM_INPUT_DIM}), got {out.shape}",
            )

    def test_keeps_geometry_dims_0_to_4(self) -> None:
        """Dims 0-4 of the output must equal dims 0-4 of the input."""
        n = 6
        x = torch.randn(n, 13)
        out = _spatial_x(x)
        torch.testing.assert_close(out[:, :5], x[:, :5])

    def test_keeps_confidence_and_importance_dims_11_12(self) -> None:
        """Dims 5-6 of the output must equal dims 11-12 of the input."""
        n = 6
        x = torch.randn(n, 13)
        out = _spatial_x(x)
        torch.testing.assert_close(out[:, 5:7], x[:, 11:13])

    def test_drops_role_onehot_dims_5_to_10(self) -> None:
        """The role one-hot block (dims 5..10) must not appear in the output.

        Changing only dims 5..10 in x must produce identical _spatial_x output.
        """
        n = 4
        x_base = torch.zeros(n, 13)
        x_base[:, 0] = 0.5   # some geometry signal

        x_perturbed = x_base.clone()
        x_perturbed[:, 5:11] = torch.randn(n, 6)  # mutate role one-hot block only

        torch.testing.assert_close(
            _spatial_x(x_base),
            _spatial_x(x_perturbed),
            msg="_spatial_x output changed when only role one-hot dims 5..10 changed",
        )

    def test_identical_geometry_different_roles_identical_output(self) -> None:
        """Two inputs with the same geometry but different YOLO role one-hots
        must produce identical _spatial_x outputs.

        This is the property that makes the spatial stream operator-independent.
        """
        n = 8
        geometry = torch.randn(n, 13)
        digit_x = geometry.clone()
        digit_x[:, 5:11] = 0.0
        digit_x[:, 5 + YOLO_CLASS_NAMES.index("digit_main")] = 1.0

        op_x = geometry.clone()
        op_x[:, 5:11] = 0.0
        op_x[:, 5 + YOLO_CLASS_NAMES.index("operator")] = 1.0

        torch.testing.assert_close(
            _spatial_x(digit_x),
            _spatial_x(op_x),
            msg=(
                "Identical geometry with different YOLO roles produced different "
                "_spatial_x output. Role one-hot must be stripped."
            ),
        )


class TestTwoStreamRowColOperatorIndependence(unittest.TestCase):
    """Hard guarantee: spatial stream output is role-invariant.

    Checks that row_cluster_emb and col_cluster_emb from SymbolGNN.forward
    are numerically identical when the only difference between two input
    graphs is the YOLO role one-hot block (dims 5..10).
    """

    def setUp(self) -> None:
        self.model = SymbolGNN(CropBackbone())
        self.model.eval()

    def test_row_cluster_emb_independent_of_role_onehot(self) -> None:
        """row_cluster_emb must be identical for two graphs that differ only in role one-hot."""
        n = 5
        shared_crops = torch.randn(n, 1, 28, 28)

        digit_role = torch.zeros(n, 6)
        digit_role[:, YOLO_CLASS_NAMES.index("digit_main")] = 1.0
        data_digit = _make_full_graph(n, digit_role)
        data_digit.crops = shared_crops

        op_role = torch.zeros(n, 6)
        op_role[:, YOLO_CLASS_NAMES.index("operator")] = 1.0
        data_op = _make_full_graph(n, op_role)
        data_op.crops = shared_crops.clone()

        with torch.no_grad():
            _, row_digit, _, _, _, _, _ = self.model(data_digit)
            _, row_op, _, _, _, _, _ = self.model(data_op)

        torch.testing.assert_close(
            row_digit, row_op, atol=1e-5, rtol=1e-5,
            msg=(
                "row_cluster_emb changed when only the YOLO role one-hot changed. "
                "The spatial stream must not receive role information."
            ),
        )

    def test_col_cluster_emb_independent_of_role_onehot(self) -> None:
        """col_cluster_emb must be identical for two graphs that differ only in role one-hot."""
        n = 5
        shared_crops = torch.randn(n, 1, 28, 28)

        digit_role = torch.zeros(n, 6)
        digit_role[:, YOLO_CLASS_NAMES.index("digit_main")] = 1.0
        data_digit = _make_full_graph(n, digit_role)
        data_digit.crops = shared_crops

        op_role = torch.zeros(n, 6)
        op_role[:, YOLO_CLASS_NAMES.index("operator")] = 1.0
        data_op = _make_full_graph(n, op_role)
        data_op.crops = shared_crops.clone()

        with torch.no_grad():
            _, _, _, col_digit, _, _, _ = self.model(data_digit)
            _, _, _, col_op, _, _, _ = self.model(data_op)

        torch.testing.assert_close(
            col_digit, col_op, atol=1e-5, rtol=1e-5,
            msg=(
                "col_cluster_emb changed when only the YOLO role one-hot changed. "
                "The spatial stream must not receive role information."
            ),
        )

    def test_fine_logits_differ_with_different_crops(self) -> None:
        """fine_logits must differ when crops differ, confirming the visual stream
        still carries role-discriminating signal through the backbone."""
        n = 4
        role = torch.zeros(n, 6)
        role[:, YOLO_CLASS_NAMES.index("digit_main")] = 1.0

        data_zero = _make_full_graph(n, role)
        data_zero.crops = torch.zeros(n, 1, 28, 28)

        data_ones = _make_full_graph(n, role.clone())
        data_ones.crops = torch.ones(n, 1, 28, 28)

        with torch.no_grad():
            fine_zero, *_ = self.model(data_zero)
            fine_ones, *_ = self.model(data_ones)

        self.assertFalse(
            torch.allclose(fine_zero, fine_ones),
            "fine_logits must differ when crops differ -- visual stream is not active",
        )

    def test_seven_tuple_output_shapes_preserved(self) -> None:
        """7-tuple output contract must be preserved by the two-stream architecture."""
        from src.modeling.gnn import (
            NUM_EDGE_TYPES, NUM_EQ_TYPES, NUM_FINE_LABELS,
            ROW_CLUSTER_DIM, COL_CLUSTER_DIM,
            WITHIN_ROW_ORDINAL_CLASSES, WITHIN_COL_ORDINAL_CLASSES,
        )
        n = 6
        role = torch.zeros(n, 6)
        role[:, 0] = 1.0
        data = _make_full_graph(n, role)
        data.crops = torch.randn(n, 1, 28, 28)
        E = data.edge_index.shape[1]

        with torch.no_grad():
            outputs = self.model(data)

        self.assertEqual(len(outputs), 7)
        fine, row_emb, row_ord, col_emb, col_ord, eq, edge_logits = outputs
        self.assertEqual(fine.shape, (n, NUM_FINE_LABELS))
        self.assertEqual(row_emb.shape, (n, ROW_CLUSTER_DIM))
        self.assertEqual(row_ord.shape, (n, WITHIN_ROW_ORDINAL_CLASSES))
        self.assertEqual(col_emb.shape, (n, COL_CLUSTER_DIM))
        self.assertEqual(col_ord.shape, (n, WITHIN_COL_ORDINAL_CLASSES))
        self.assertEqual(eq.shape, (1, NUM_EQ_TYPES))
        self.assertEqual(edge_logits.shape, (E, NUM_EDGE_TYPES))


if __name__ == "__main__":
    unittest.main()
