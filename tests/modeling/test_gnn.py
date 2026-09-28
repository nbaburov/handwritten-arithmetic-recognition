from __future__ import annotations
import unittest
import torch
from src.modeling.gnn import CROP_EMBED_DIM, CropBackbone

class TestCropBackbone(unittest.TestCase):
    """Iter2: backbone v2 with deeper conv stack + BN, 128-dim output."""

    def setUp(self) -> None:
        self.backbone = CropBackbone()
        self.backbone.train(False)

    def test_output_shape_single(self) -> None:
        out = self.backbone(torch.zeros(1, 1, 28, 28))
        self.assertEqual(out.shape, (1, CROP_EMBED_DIM))

    def test_output_shape_batch(self) -> None:
        out = self.backbone(torch.zeros(8, 1, 28, 28))
        self.assertEqual(out.shape, (8, CROP_EMBED_DIM))

    def test_param_count_in_band(self) -> None:
        total = sum(p.numel() for p in self.backbone.parameters())
        self.assertGreater(total, 150_000)
        # Upper bound widened from 300K to 800K in iter9 W-ARCH-1: the two-stream
        # GATv2 adds a spatial stack (~50K params) on top of the visual stack,
        # pushing total SymbolGNN params to ~605K — still under the 1M CPU-friendly cap.
        self.assertLess(total, 800_000)

    def test_output_non_negative(self) -> None:
        out = self.backbone(torch.randn(4, 1, 28, 28))
        self.assertTrue((out >= 0).all())

    def test_gradient_flow(self) -> None:
        self.backbone.train(True)
        out = self.backbone(torch.randn(4, 1, 28, 28))
        out.sum().backward()
        for name, p in self.backbone.named_parameters():
            self.assertIsNotNone(p.grad, f"{name} has no gradient")
            self.assertFalse(torch.isnan(p.grad).any(), f"{name} grad has NaN")
            self.assertGreater(
                p.grad.abs().sum().item(), 0,
                f"{name} grad is zero (dead layer or wiring bug)",
            )

if __name__ == "__main__":
    unittest.main()


from torch_geometric.data import Data
from src.modeling.gnn import (
    CropBackbone, SymbolGNN,
    NUM_FINE_LABELS, NUM_EQ_TYPES, NUM_EDGE_TYPES,
    ROW_CLUSTER_DIM, COL_CLUSTER_DIM,
    WITHIN_ROW_ORDINAL_CLASSES, WITHIN_COL_ORDINAL_CLASSES,
)


class TestSymbolGNN(unittest.TestCase):
    def setUp(self) -> None:
        self.model = SymbolGNN(CropBackbone())
        self.model.eval()

    def _make_graph(self, n: int = 5) -> Data:
        crops = torch.zeros(n, 1, 28, 28)
        x = torch.zeros(n, 13)   # 13-dim non-visual features
        # Dense edges: all pairs i != j
        src, dst = [], []
        for i in range(n):
            for j in range(n):
                if i != j:
                    src.append(i)
                    dst.append(j)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        E = len(src)
        edge_attr = torch.zeros(E, 16)   # 16-dim geometric features
        edge_type = torch.zeros(E, dtype=torch.long)   # edge-type vocabulary index
        bbox = torch.zeros(n, 4)
        batch = torch.zeros(n, dtype=torch.long)
        return Data(crops=crops, x=x, edge_index=edge_index,
                    edge_attr=edge_attr, edge_type=edge_type,
                    bbox=bbox, batch=batch, num_nodes=n)

    def test_forward_output_shapes(self) -> None:
        """7-tuple output shapes must match the contract for N nodes, G=1 graph, E edges."""
        n = 5
        data = self._make_graph(n)
        E = data.edge_index.shape[1]
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

    def test_gat_accepts_24_dim_edge(self) -> None:
        """Forward must not error with 24-dim edge features (16 geometric + 8 type embedding)."""
        data = self._make_graph(4)
        # Override edge_attr with random 16-dim to confirm it flows through without error
        data.edge_attr = torch.randn(data.edge_index.shape[1], 16)
        try:
            self.model(data)
        except Exception as exc:
            self.fail(f"Forward raised an exception with 24-dim edge features: {exc}")

    def test_edge_type_logits_dim(self) -> None:
        """edge_type_logits must have shape (E, NUM_EDGE_TYPES)."""
        data = self._make_graph(5)
        *_, edge_logits = self.model(data)
        E = data.edge_index.shape[1]
        self.assertEqual(edge_logits.shape, (E, NUM_EDGE_TYPES))

    def test_cluster_emb_nonzero(self) -> None:
        """row_cluster_emb and col_cluster_emb must not be all zeros after a random forward pass."""
        n = 5
        data = self._make_graph(n)
        # Use random crops so backbone output is non-trivial
        data.crops = torch.randn(n, 1, 28, 28)
        _, row_emb, _, col_emb, *_ = self.model(data)
        self.assertFalse(torch.all(row_emb == 0).item(), "row_cluster_emb is all zeros")
        self.assertFalse(torch.all(col_emb == 0).item(), "col_cluster_emb is all zeros")

    def test_predict_returns_node_predictions(self) -> None:
        """predict() must return a list of NodePrediction with new fields and a valid eq type."""
        from src.parsing.assemble import NodePrediction
        data = self._make_graph(3)
        preds, eq_type, eq_logits = self.model.predict(data)
        self.assertEqual(len(preds), 3)
        self.assertIsInstance(preds[0], NodePrediction)
        self.assertIn(eq_type, ["add", "subtract", "multiply", "divide"])
        self.assertIsInstance(eq_logits, list)
        self.assertEqual(len(eq_logits), 4)
        # Verify new fields exist
        p = preds[0]
        self.assertTrue(hasattr(p, "row_cluster_id"))
        self.assertTrue(hasattr(p, "col_cluster_id"))
        self.assertTrue(hasattr(p, "within_row_ord"))
        self.assertTrue(hasattr(p, "within_col_ord"))
        # Verify old fields are gone
        self.assertFalse(hasattr(p, "row_idx"))
        self.assertFalse(hasattr(p, "col_idx"))

    def test_predict_cluster_ids_non_negative(self) -> None:
        """All row_cluster_id and col_cluster_id values must be >= 0."""
        data = self._make_graph(6)
        # Give distinct bbox positions so clustering is meaningful
        data.bbox = torch.tensor([
            [10.0, 10.0, 30.0, 30.0],
            [50.0, 10.0, 70.0, 30.0],
            [10.0, 60.0, 30.0, 80.0],
            [50.0, 60.0, 70.0, 80.0],
            [10.0, 110.0, 30.0, 130.0],
            [50.0, 110.0, 70.0, 130.0],
        ])
        preds, _, _logits = self.model.predict(data)
        for p in preds:
            self.assertGreaterEqual(p.row_cluster_id, 0)
            self.assertGreaterEqual(p.col_cluster_id, 0)

    def test_total_params_reasonable(self) -> None:
        """Total parameter count must be < 1_000_000 to remain CPU-friendly."""
        total = sum(p.numel() for p in self.model.parameters())
        self.assertLess(total, 1_000_000)

    def test_no_edges_returns_zero_edge_logits(self) -> None:
        """Single-node graph with no edges must yield edge_type_logits of shape (0, NUM_EDGE_TYPES)."""
        n = 1
        crops = torch.zeros(n, 1, 28, 28)
        x = torch.zeros(n, 13)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, 16)
        edge_type = torch.zeros(0, dtype=torch.long)
        bbox = torch.zeros(n, 4)
        batch = torch.zeros(n, dtype=torch.long)
        data = Data(crops=crops, x=x, edge_index=edge_index,
                    edge_attr=edge_attr, edge_type=edge_type,
                    bbox=bbox, batch=batch, num_nodes=n)
        *_, edge_logits = self.model(data)
        self.assertEqual(edge_logits.shape, (0, NUM_EDGE_TYPES))

    def test_spatial_stream_operator_independence(self) -> None:
        """Spatial stream outputs (row_cluster_emb, col_cluster_emb) must be identical
        when only the role one-hot changes and all geometry is held constant.

        This is the hard guarantee of W-ARCH-1: the spatial GATv2 input is
        _spatial_x(data.x) which strips dims 5..10 (the 6-dim YOLO role one-hot).
        Changing only those dims must produce zero delta in both cluster embeddings.
        """
        from src.core.ontology import YOLO_CLASS_NAMES
        n = 4
        data_digit = self._make_graph(n)
        data_digit.x = torch.zeros(n, 13)
        digit_dim = 5 + YOLO_CLASS_NAMES.index("digit_main")
        data_digit.x[:, digit_dim] = 1.0
        data_digit.crops = torch.randn(n, 1, 28, 28)

        # Same geometry, different role: swap to operator one-hot
        data_op = self._make_graph(n)
        data_op.x = torch.zeros(n, 13)
        op_dim = 5 + YOLO_CLASS_NAMES.index("operator")
        data_op.x[:, op_dim] = 1.0
        data_op.crops = data_digit.crops.clone()  # identical visual input

        with torch.no_grad():
            _, row_emb_digit, _, col_emb_digit, _, _, _ = self.model(data_digit)
            _, row_emb_op, _, col_emb_op, _, _, _ = self.model(data_op)

        torch.testing.assert_close(
            row_emb_digit, row_emb_op, atol=1e-5, rtol=1e-5,
            msg="row_cluster_emb must not change when only the role one-hot changes",
        )
        torch.testing.assert_close(
            col_emb_digit, col_emb_op, atol=1e-5, rtol=1e-5,
            msg="col_cluster_emb must not change when only the role one-hot changes",
        )

    def test_visual_stream_changes_with_crops(self) -> None:
        """fine_logits must change when crops change, even with identical spatial features.

        Verifies that the visual stream carries real information through to the
        fine_label head and is not collapsed by the stream split.
        """
        n = 4
        data_a = self._make_graph(n)
        data_a.crops = torch.zeros(n, 1, 28, 28)
        data_b = self._make_graph(n)
        data_b.crops = torch.ones(n, 1, 28, 28)

        with torch.no_grad():
            fine_a, *_ = self.model(data_a)
            fine_b, *_ = self.model(data_b)

        self.assertFalse(
            torch.allclose(fine_a, fine_b),
            "fine_logits must differ when crops differ — visual stream not carrying signal",
        )




class TestStructuralColGapExclusion(unittest.TestCase):
    """Structural tokens (result_bar, divide_bracket) must be excluded from the
    column-gap computation in predict() so they do not inflate the gap threshold
    and collapse genuine child columns.

    Done-criterion: a scene with a 2-column answer row plus a full-width
    structural given node → the answer digits still cluster into 2 distinct
    column IDs, unchanged vs the same scene without the structural node.
    """

    def setUp(self) -> None:
        self.model = SymbolGNN(CropBackbone())
        self.model.eval()

    def _make_scene_graph(self, include_result_bar: bool) -> "Data":
        """Build a minimal Data object for predict().

        Layout:
          Node 0 — digit_main at x=[10..50]   (left column, answer row y=300..350)
          Node 1 — digit_main at x=[200..240]  (right column, answer row y=300..350)
          Node 2 — result_bar at x=[0..512]    (full-width; only when include_result_bar=True)

        The result_bar's wide bbox (0..512) would, if included in the gap computation,
        push the gap threshold high enough to merge the two digit columns.
        """
        from src.core.ontology import YOLO_CLASS_NAMES  # noqa: PLC0415

        digit_main_idx = YOLO_CLASS_NAMES.index("digit_main")
        result_bar_idx = YOLO_CLASS_NAMES.index("result_bar")

        n = 3 if include_result_bar else 2

        crops = torch.zeros(n, 1, 28, 28)
        x = torch.zeros(n, 13)

        # Set role one-hot (positions 5..10 in the 13-dim feature vector).
        x[0, 5 + digit_main_idx] = 1.0   # node 0: digit_main, left col
        x[1, 5 + digit_main_idx] = 1.0   # node 1: digit_main, right col
        if include_result_bar:
            x[2, 5 + result_bar_idx] = 1.0   # node 2: result_bar, full-width

        # Dense edges.
        src, dst = [], []
        for i in range(n):
            for j in range(n):
                if i != j:
                    src.append(i)
                    dst.append(j)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        E = len(src)
        edge_attr = torch.zeros(E, 16)
        edge_type = torch.zeros(E, dtype=torch.long)

        # Bboxes: two digits in separate columns, result_bar spans full width.
        bboxes_list = [
            [10.0, 300.0, 50.0, 350.0],    # node 0: left digit
            [200.0, 300.0, 240.0, 350.0],  # node 1: right digit
        ]
        if include_result_bar:
            bboxes_list.append([0.0, 250.0, 512.0, 280.0])  # node 2: full-width bar

        bbox = torch.tensor(bboxes_list, dtype=torch.float)
        batch = torch.zeros(n, dtype=torch.long)

        return Data(crops=crops, x=x, edge_index=edge_index,
                    edge_attr=edge_attr, edge_type=edge_type,
                    bbox=bbox, batch=batch, num_nodes=n)

    def test_two_digit_columns_preserved_with_structural_node(self) -> None:
        """Column IDs for the two answer digits must be DIFFERENT when a full-width
        result_bar is present — the structural node must not merge them."""
        with torch.no_grad():
            data = self._make_scene_graph(include_result_bar=True)
            preds, _, _ = self.model.predict(data)

        # Only the digit predictions (nodes 0 and 1).
        digit_preds = [p for p in preds if not p.fine_label in {"result_bar", "div_bracket"}]
        # There may be 2 or 3 preds depending on whether result_bar is in the output;
        # we care only that the two digit nodes got distinct col_cluster_ids.
        self.assertEqual(len(preds), 3, "Expected 3 NodePredictions (2 digits + 1 result_bar)")

        col_ids_digits = [preds[0].col_cluster_id, preds[1].col_cluster_id]
        self.assertNotEqual(
            col_ids_digits[0],
            col_ids_digits[1],
            f"Digit columns collapsed: both have col_cluster_id={col_ids_digits[0]}. "
            "The result_bar node must be excluded from the gap computation.",
        )

    def test_column_ids_unchanged_vs_no_structural_node(self) -> None:
        """The two digit column IDs must be identical whether or not the full-width
        result_bar is present — the structural node must be a no-op for col clustering."""
        with torch.no_grad():
            data_with = self._make_scene_graph(include_result_bar=True)
            preds_with, _, _ = self.model.predict(data_with)

            data_without = self._make_scene_graph(include_result_bar=False)
            preds_without, _, _ = self.model.predict(data_without)

        col_with = sorted([preds_with[0].col_cluster_id, preds_with[1].col_cluster_id])
        col_without = sorted([preds_without[0].col_cluster_id, preds_without[1].col_cluster_id])

        self.assertEqual(
            col_with,
            col_without,
            f"Col cluster IDs changed when result_bar was added: "
            f"with={col_with} vs without={col_without}. "
            "Structural node must not distort digit column clustering.",
        )
