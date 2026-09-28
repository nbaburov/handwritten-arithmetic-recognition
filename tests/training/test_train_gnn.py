from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.training.train_gnn import (
    EQ_TYPE_TO_IDX,
    FINE_LABEL_TO_IDX,
    SceneGraphDataset,
    _supervised_contrastive_loss,
)


def _write_fake_scene(
    tmp: Path,
    symbols=None,
    equation_type: str = "add",
) -> tuple[Path, Path]:
    img = Image.fromarray(np.full((512, 512), 255, dtype=np.uint8))
    img_path = tmp / "scene.png"
    img.save(img_path)
    if symbols is None:
        symbols = [
            {
                "bbox": [10, 10, 30, 30],
                "fine_label": "main_3",
                "yolo_class": "digit_main",
                "row_index": 0,
                "col_index": 0,
            },
            {
                "bbox": [40, 5, 60, 25],
                "fine_label": "carry_1",
                "yolo_class": "digit_carry",
                "row_index": 0,
                "col_index": 1,
            },
        ]
    gt = {"equation_type": equation_type, "symbols": symbols}
    gt_path = tmp / "gt.json"
    gt_path.write_text(json.dumps(gt))
    return img_path, gt_path


class TestSceneGraphDataset(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        img_path, gt_path = _write_fake_scene(self.tmp)
        manifest = self.tmp / "manifest.csv"
        manifest.write_text(f"image,ground_truth\n{img_path},{gt_path}\n")
        self.ds = SceneGraphDataset(manifest)

    def test_len(self) -> None:
        self.assertEqual(len(self.ds), 1)

    def test_node_count(self) -> None:
        self.assertEqual(self.ds[0].num_nodes, 2)

    def test_fine_labels(self) -> None:
        data = self.ds[0]
        # Iter2 ontology: 36-style "main_3"/"carry_1" map to 16-class "3"/"1".
        self.assertEqual(int(data.y_fine[0]), FINE_LABEL_TO_IDX["3"])
        self.assertEqual(int(data.y_fine[1]), FINE_LABEL_TO_IDX["1"])

    def test_within_row_ord(self) -> None:
        """Two tokens in the same row with col_indices 0 and 3 must have ordinal ranks [0, 1]."""
        tmp = Path(tempfile.mkdtemp())
        symbols = [
            {
                "bbox": [10, 10, 30, 30],
                "fine_label": "main_1",
                "yolo_class": "digit_main",
                "row_index": 0,
                "col_index": 0,
            },
            {
                "bbox": [100, 10, 120, 30],
                "fine_label": "main_2",
                "yolo_class": "digit_main",
                "row_index": 0,
                "col_index": 3,
            },
        ]
        img_path, gt_path = _write_fake_scene(tmp, symbols=symbols)
        manifest = tmp / "manifest.csv"
        manifest.write_text(f"image,ground_truth\n{img_path},{gt_path}\n")
        ds = SceneGraphDataset(manifest)
        data = ds[0]
        self.assertEqual(int(data.y_within_row_ord[0]), 0)
        self.assertEqual(int(data.y_within_row_ord[1]), 1)

    def test_within_col_ord(self) -> None:
        """Two tokens in the same col with row_indices 0 and 1 must have ordinal ranks [0, 1]."""
        tmp = Path(tempfile.mkdtemp())
        symbols = [
            {
                "bbox": [10, 10, 30, 30],
                "fine_label": "main_1",
                "yolo_class": "digit_main",
                "row_index": 0,
                "col_index": 2,
            },
            {
                "bbox": [10, 50, 30, 70],
                "fine_label": "main_2",
                "yolo_class": "digit_main",
                "row_index": 1,
                "col_index": 2,
            },
        ]
        img_path, gt_path = _write_fake_scene(tmp, symbols=symbols)
        manifest = tmp / "manifest.csv"
        manifest.write_text(f"image,ground_truth\n{img_path},{gt_path}\n")
        ds = SceneGraphDataset(manifest)
        data = ds[0]
        self.assertEqual(int(data.y_within_col_ord[0]), 0)
        self.assertEqual(int(data.y_within_col_ord[1]), 1)

    def test_y_row_cluster(self) -> None:
        """y_row_cluster must contain the raw row_index values from GT."""
        data = self.ds[0]
        # Both tokens have row_index=0 in the default fixture
        self.assertEqual(int(data.y_row_cluster[0]), 0)
        self.assertEqual(int(data.y_row_cluster[1]), 0)

    def test_y_col_cluster(self) -> None:
        """y_col_cluster must contain the raw col_index values from GT."""
        data = self.ds[0]
        # Default fixture: token 0 has col_index=0, token 1 has col_index=1
        self.assertEqual(int(data.y_col_cluster[0]), 0)
        self.assertEqual(int(data.y_col_cluster[1]), 1)

    def test_y_edge_type_shape(self) -> None:
        """y_edge_type must have the same shape as the graph's edge_type tensor."""
        data = self.ds[0]
        self.assertEqual(data.y_edge_type.shape, data.edge_type.shape)

    def test_equation_type_label(self) -> None:
        data = self.ds[0]
        self.assertEqual(int(data.y_eq[0]), EQ_TYPE_TO_IDX["add"])

    def test_seven_loss_terms_non_nan(self) -> None:
        """Train for 1 step on a 2-scene mini-dataset; combined loss must not be NaN."""
        import torch.nn.functional as F
        from torch_geometric.loader import DataLoader
        from src.modeling.gnn import (
            CropBackbone,
            NUM_FINE_LABELS,
            SymbolGNN,
        )
        from src.training.train_gnn import _supervised_contrastive_loss

        # Build 2 scenes
        tmp = Path(tempfile.mkdtemp())
        scenes = []
        for scene_i in range(2):
            s_tmp = tmp / f"scene{scene_i}"
            s_tmp.mkdir()
            img = Image.fromarray(np.full((512, 512), 255, dtype=np.uint8))
            img_path = s_tmp / "scene.png"
            img.save(img_path)
            symbols = [
                {
                    "bbox": [10, 10, 30, 30],
                    "fine_label": "main_3",
                    "yolo_class": "digit_main",
                    "row_index": 0,
                    "col_index": 0,
                },
                {
                    "bbox": [40, 5, 60, 25],
                    "fine_label": "carry_1",
                    "yolo_class": "digit_carry",
                    "row_index": 0,
                    "col_index": 1,
                },
            ]
            gt = {"equation_type": "add", "symbols": symbols}
            gt_path = s_tmp / "gt.json"
            gt_path.write_text(json.dumps(gt))
            scenes.append((img_path, gt_path))

        manifest = tmp / "manifest.csv"
        lines = ["image,ground_truth"] + [f"{img},{gt}" for img, gt in scenes]
        manifest.write_text("\n".join(lines) + "\n")

        ds = SceneGraphDataset(manifest)
        loader = DataLoader(ds, batch_size=2)
        batch = next(iter(loader))

        model = SymbolGNN(CropBackbone())
        model.train()

        fine_class_weights = torch.ones(NUM_FINE_LABELS)

        fl, row_emb, row_ord_logits, col_emb, col_ord_logits, eq_logits, et_logits = model(batch)

        node_keep_mask = torch.ones(batch.num_nodes, dtype=torch.bool)

        loss_fine    = F.cross_entropy(fl[node_keep_mask], batch.y_fine[node_keep_mask], weight=fine_class_weights)
        loss_row_cl  = _supervised_contrastive_loss(row_emb[node_keep_mask], batch.y_row_cluster[node_keep_mask])
        loss_row_ord = F.cross_entropy(row_ord_logits[node_keep_mask], batch.y_within_row_ord[node_keep_mask])
        loss_col_cl  = _supervised_contrastive_loss(col_emb[node_keep_mask], batch.y_col_cluster[node_keep_mask])
        loss_col_ord = F.cross_entropy(col_ord_logits[node_keep_mask], batch.y_within_col_ord[node_keep_mask])
        loss_eq      = F.cross_entropy(eq_logits, batch.y_eq)
        if et_logits.shape[0] > 0:
            loss_et = F.cross_entropy(et_logits, batch.y_edge_type)
        else:
            loss_et = torch.tensor(0.0)

        loss = (
            2.0 * loss_fine
            + loss_row_cl
            + loss_row_ord
            + loss_col_cl
            + loss_col_ord
            + loss_eq
            + 0.3 * loss_et
        )
        self.assertFalse(torch.isnan(loss).item(), f"Loss is NaN: {loss.item()}")

    def test_scene_dropout_mask(self) -> None:
        """With scene_dropout_prob=1.0, the keep mask should have 0 True entries."""
        scene_dropout_prob = 1.0
        num_nodes = 10
        node_keep_mask = torch.rand(num_nodes) > scene_dropout_prob
        self.assertEqual(int(node_keep_mask.sum()), 0)


class TestCombinedScore(unittest.TestCase):
    """Iter2: combined fine+eq early-stop metric — prevents silent overfit."""

    def test_symmetric_average(self) -> None:
        from src.training.train_gnn import combined_score
        self.assertAlmostEqual(combined_score(0.3, 0.9), combined_score(0.9, 0.3))

    def test_endpoints(self) -> None:
        from src.training.train_gnn import combined_score
        self.assertEqual(combined_score(0.0, 0.0), 0.0)
        self.assertEqual(combined_score(1.0, 1.0), 1.0)

    def test_strict_monotonic_in_each_arg(self) -> None:
        from src.training.train_gnn import combined_score
        self.assertGreater(combined_score(0.5, 0.5), combined_score(0.4, 0.5))
        self.assertGreater(combined_score(0.5, 0.5), combined_score(0.5, 0.4))

    def test_early_stop_prefers_balanced_over_fine_only(self) -> None:
        """Sequence where val_fine_acc creeps up but val_eq_acc collapses must
        NOT keep updating best.pt. Catches the iteration 1 overfit failure mode
        where fine-only tracking masked eq_type degradation."""
        from src.training.train_gnn import combined_score
        seq = [(0.40, 0.95), (0.45, 0.50), (0.50, 0.30)]
        scores = [combined_score(f, e) for f, e in seq]
        best_idx = scores.index(max(scores))
        self.assertEqual(best_idx, 0)


class TestPermutationInvariantClusterAccuracy(unittest.TestCase):
    """Tests for gnn_eval.permutation_invariant_cluster_accuracy (ARI-based metric)."""

    def setUp(self) -> None:
        from src.core.cluster_metrics import permutation_invariant_cluster_accuracy
        self.fn = permutation_invariant_cluster_accuracy

    def test_relabelled_clusters_perfect(self) -> None:
        """Swapped labels on an identical partition must give 1.0."""
        result = self.fn([0, 0, 1, 1], [1, 1, 0, 0])
        self.assertAlmostEqual(result, 1.0)

    def test_identity(self) -> None:
        """Identical sequences must give 1.0."""
        result = self.fn([0, 1, 2], [0, 1, 2])
        self.assertAlmostEqual(result, 1.0)

    def test_collapsed_prediction(self) -> None:
        """All-same prediction vs three distinct GT clusters must score below 0.5."""
        result = self.fn([0, 1, 2], [0, 0, 0])
        self.assertLess(result, 0.5)

    def test_empty_input(self) -> None:
        """Empty inputs (ARI undefined) must return 0.0."""
        result = self.fn([], [])
        self.assertEqual(result, 0.0)

    def test_single_element(self) -> None:
        """Single element (ARI undefined for <2 points) must return 0.0."""
        result = self.fn([0], [0])
        self.assertEqual(result, 0.0)

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched lengths must raise ValueError."""
        with self.assertRaises(ValueError):
            self.fn([0, 1], [0, 1, 2])

    def test_offset_columns_match(self) -> None:
        """GT col_index scene-global offset (1..4) vs predicted 0..3 must give 1.0.

        This is the exact iter1/iter2 failure mode: strict equality gives 0.0
        because indices never match, but ARI gives 1.0 because the partition
        is identical up to a relabelling.
        """
        result = self.fn([0, 1, 2, 3], [1, 2, 3, 4])
        self.assertAlmostEqual(result, 1.0)


if __name__ == "__main__":
    unittest.main()
