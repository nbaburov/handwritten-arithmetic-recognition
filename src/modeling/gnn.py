"""GATv2-based SymbolGNN with CropBackbone for joint symbol classification and spatial parsing."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


CROP_EMBED_DIM: int = 128
"""Output dimensionality of CropBackbone visual embeddings.

Iteration 2 bumped from 64 → 128 to address the fine_label capacity ceiling
observed in iteration 1 (val_fine_acc plateaued at 0.52 with 64-dim output).
"""


class CropBackbone(nn.Module):
    """
    28x28 grayscale crop -> 128-dim visual embedding.

    Six convolutional layers with batch normalisation:
      Conv(1->16, 3x3)   + BN + ReLU              -> (16, 26, 26)
      Conv(16->32, 3x3)  + BN + ReLU + MaxPool2x2 -> (32, 12, 12)
      Conv(32->64, 3x3)  + BN + ReLU              -> (64, 10, 10)
      Conv(64->64, 3x3)  + BN + ReLU + MaxPool2x2 -> (64, 4, 4)
      Conv(64->128, 3x3) + BN + ReLU              -> (128, 2, 2) = 512 flat
      Linear(512, 128) + ReLU                     -> 128-dim
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(512, CROP_EMBED_DIM)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = x.flatten(1)
        return F.relu(self.fc(x))


from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool

from ..core.ontology import (
    YOLO_CLASS_NAMES,
    full_label_from_gnn_and_yolo,
    gnn_fine_labels_ordered,
)
from .edge_types import NUM_EDGE_TYPES
from ..parsing.assemble import NodePrediction

_FINE_LABELS = gnn_fine_labels_ordered()
_EQ_TYPES = ["add", "subtract", "multiply", "divide"]

# YOLO coarse class indices for full-width structural tokens (result_bar, divide_bracket).
# These are excluded from the column-gap computation in predict() to prevent their
# wide bboxes from inflating the gap threshold and collapsing child columns.
# Derived from the ontology so it stays correct if YOLO_CLASS_NAMES is ever reordered.
_STRUCTURAL_COARSE_IDS: frozenset[int] = frozenset(
    i for i, name in enumerate(YOLO_CLASS_NAMES) if name in {"result_bar", "divide_bracket"}
)

NUM_FINE_LABELS = 16
NUM_EQ_TYPES = 4
SPATIAL_FEATURE_DIM = 13
NODE_DIM = CROP_EMBED_DIM + SPATIAL_FEATURE_DIM   # 128 visual + 13 spatial = 141
EDGE_GEOM_DIM = 16
EDGE_TYPE_EMB_DIM = 8
EDGE_DIM = EDGE_GEOM_DIM + EDGE_TYPE_EMB_DIM   # 24
ROW_CLUSTER_DIM = 16
COL_CLUSTER_DIM = 16
WITHIN_ROW_ORDINAL_CLASSES = 8
WITHIN_COL_ORDINAL_CLASSES = 8

# Two-stream architecture constants (W-ARCH-1)
SPATIAL_STREAM_INPUT_DIM: int = 7   # cx, cy, bw, bh, area (5 geo) + conf (1) + importance (1)
SPATIAL_STREAM_HIDDEN: int = 64     # per-head hidden dim in spatial GATv2 layer 1
SPATIAL_STREAM_HEADS: int = 2       # heads in spatial GATv2 layer 1; concat -> 128
SPATIAL_STREAM_OUT: int = 64        # spatial stream output after layer 2; same as visual stream out


def _spatial_x(x: Tensor) -> Tensor:
    """Extract the 7-dim geometry+meta block from the 13-dim node feature vector.

    Keeps: cx(0), cy(1), bw(2), bh(3), area(4), confidence(11), importance(12).
    Drops: role_onehot(5..10).
    This is the sole input to the spatial GATv2 stream. Stripping the role
    one-hot is the architectural guarantee that row/col/eq_type heads cannot
    receive operator class as direct input.
    """
    return torch.cat([x[:, :5], x[:, 11:13]], dim=1)  # (N, 7)


def _cluster_by_coordinate(bbox: Tensor, axis: int) -> list[int]:
    """Assign cluster IDs by sorting nodes on the given axis and grouping by gaps.

    axis=1: y-midpoints -> row clusters (top to bottom).
    axis=0: x-midpoints -> col clusters (left to right).
    Gap threshold: 0.5 x median bbox height (axis=1) or width (axis=0).
    """
    N = bbox.shape[0]
    if N == 0:
        return []
    mids = ((bbox[:, axis] + bbox[:, axis + 2]) / 2.0).tolist()
    sizes = (bbox[:, axis + 2] - bbox[:, axis]).abs().tolist()
    median_size = sorted(sizes)[len(sizes) // 2]
    gap_thresh = 0.5 * median_size

    order = sorted(range(N), key=lambda i: mids[i])
    cluster_ids = [0] * N
    current_cluster = 0
    for rank in range(1, len(order)):
        prev_idx = order[rank - 1]
        curr_idx = order[rank]
        if abs(mids[curr_idx] - mids[prev_idx]) > gap_thresh:
            current_cluster += 1
        cluster_ids[curr_idx] = current_cluster
    cluster_ids[order[0]] = 0
    return cluster_ids


class SymbolGNN(nn.Module):
    """
    Joint symbol classifier and spatial structure parser using dense GATv2 attention.

    Input: PyG Data with crops (N,1,28,28), x (N,13), edge_index, edge_attr (E,16),
           edge_type (E,), bbox (N,4).
    Output: 7-tuple (fine_logits, row_cluster_emb, within_row_ord_logits,
                      col_cluster_emb, within_col_ord_logits, eq_logits, edge_type_logits).
    """

    def __init__(self, backbone: CropBackbone, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = backbone
        self.dropout = dropout
        self.edge_type_emb = nn.Embedding(NUM_EDGE_TYPES, EDGE_TYPE_EMB_DIM)

        # --- Visual stream (fine_label + edge_type heads) ---
        # GATv2 layer 1: input NODE_DIM (141) -> 128 hidden * 4 heads = 512
        self.vis_gat1 = GATv2Conv(NODE_DIM, 128, heads=4, edge_dim=EDGE_DIM, concat=True)
        # GATv2 layer 2: 512 -> SPATIAL_STREAM_OUT (64), 1 head
        self.vis_gat2 = GATv2Conv(512, SPATIAL_STREAM_OUT, heads=1, edge_dim=EDGE_DIM, concat=False)

        # --- Spatial stream (row, col, eq_type heads) ---
        # Input: 7-dim geo+meta, NO role one-hot.
        # Gradient from row/col contrastive losses and eq_type CE cannot reach
        # the visual GATv2 stack via this path, severing the coupling identified
        # in the iter8-R2 post-mortem.
        # Layer 1: 7 -> SPATIAL_STREAM_HIDDEN (64) * SPATIAL_STREAM_HEADS (2) = 128
        self.spa_gat1 = GATv2Conv(
            SPATIAL_STREAM_INPUT_DIM,
            SPATIAL_STREAM_HIDDEN,
            heads=SPATIAL_STREAM_HEADS,
            edge_dim=EDGE_DIM,
            concat=True,
        )
        # Layer 2: 128 -> SPATIAL_STREAM_OUT (64), 1 head
        self.spa_gat2 = GATv2Conv(
            SPATIAL_STREAM_HIDDEN * SPATIAL_STREAM_HEADS,
            SPATIAL_STREAM_OUT,
            heads=1,
            edge_dim=EDGE_DIM,
            concat=False,
        )

        # --- Heads wired to their respective streams ---
        # Visual stream -> fine_label (role-aware)
        self.fine_label_head = nn.Linear(SPATIAL_STREAM_OUT, NUM_FINE_LABELS)

        # Spatial stream -> row/col cluster + ordinal (operator-independent)
        self.row_cluster_head = nn.Linear(SPATIAL_STREAM_OUT, ROW_CLUSTER_DIM)
        self.within_row_ord_head = nn.Linear(SPATIAL_STREAM_OUT, WITHIN_ROW_ORDINAL_CLASSES)
        self.col_cluster_head = nn.Linear(SPATIAL_STREAM_OUT, COL_CLUSTER_DIM)
        self.within_col_ord_head = nn.Linear(SPATIAL_STREAM_OUT, WITHIN_COL_ORDINAL_CLASSES)

        # Spatial stream -> eq_type (operator-independent layout signal)
        # eq_fc1 input dim is SPATIAL_STREAM_OUT (64), same as before.
        self.eq_fc1 = nn.Linear(SPATIAL_STREAM_OUT, 32)
        self.eq_fc2 = nn.Linear(32, NUM_EQ_TYPES)

        # Auxiliary edge-type classifier reads VISUAL stream (role-aware):
        # (src_emb | dst_emb | edge_feat) -> NUM_EDGE_TYPES
        # Input dim: SPATIAL_STREAM_OUT + SPATIAL_STREAM_OUT + EDGE_DIM = 64+64+24 = 152
        self.edge_head = nn.Sequential(
            nn.Linear(SPATIAL_STREAM_OUT + SPATIAL_STREAM_OUT + EDGE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_EDGE_TYPES),
        )

    def forward(self, data: Data) -> tuple[
        Tensor,  # fine_logits (N, 16)
        Tensor,  # row_cluster_emb (N, 16)
        Tensor,  # within_row_ord_logits (N, 8)
        Tensor,  # col_cluster_emb (N, 16)
        Tensor,  # within_col_ord_logits (N, 8)
        Tensor,  # eq_logits (G, 4)
        Tensor,  # edge_type_logits (E, 8) — or zeros(0, 8) if no edges
    ]:
        """Forward pass through the two-stream GATv2 graph network (W-ARCH-1)."""
        # --- Shared: backbone + edge features ---
        visual = self.backbone(data.crops)                             # (N, 128)
        h_vis_in = torch.cat([visual, data.x], dim=1)                 # (N, 141) full merged input
        h_spa_in = _spatial_x(data.x)                                 # (N, 7)  geo+meta only, NO role one-hot

        edge_emb = self.edge_type_emb(data.edge_type)                 # (E, 8)
        edge_feat = torch.cat([data.edge_attr, edge_emb], dim=1)      # (E, 24)

        # --- Visual stream (fine_label, edge_type) ---
        h_vis = F.elu(self.vis_gat1(h_vis_in, data.edge_index, edge_feat))  # (N, 512)
        h_vis = F.dropout(h_vis, p=self.dropout, training=self.training)
        h_vis = F.elu(self.vis_gat2(h_vis, data.edge_index, edge_feat))     # (N, 64)

        # --- Spatial stream (row, col, eq_type) ---
        # Spatial GATv2 receives NO role one-hot: row/col/eq_type gradients
        # cannot back-propagate into operator-label dimensions.
        h_spa = F.elu(self.spa_gat1(h_spa_in, data.edge_index, edge_feat))  # (N, 128)
        h_spa = F.dropout(h_spa, p=self.dropout, training=self.training)
        h_spa = F.elu(self.spa_gat2(h_spa, data.edge_index, edge_feat))     # (N, 64)

        # --- Heads wired to their streams ---
        fine_logits            = self.fine_label_head(h_vis)
        row_cluster_emb        = self.row_cluster_head(h_spa)
        within_row_ord_logits  = self.within_row_ord_head(h_spa)
        col_cluster_emb        = self.col_cluster_head(h_spa)
        within_col_ord_logits  = self.within_col_ord_head(h_spa)

        # eq_type head: spatial stream only.
        # W-ARCH-2 path: when a virtual scene node is present it is always the
        # last node per graph; extract its spatial embedding directly.
        # W-ARCH-1 path (default): use global_mean_pool over the spatial stream.
        batch = data.batch if data.batch is not None else torch.zeros(
            h_spa.shape[0], dtype=torch.long, device=h_spa.device
        )
        # Pass explicit graph count so empty-node graphs (OOD scribble) get a
        # zero row rather than being dropped -- keeps eq_logits aligned with y_eq.
        num_graphs = int(getattr(data, "num_graphs", batch.max().item() + 1 if h_spa.shape[0] > 0 else 1))
        # has_virtual_node may be a plain bool (single-graph inference) or a
        # batched boolean tensor (PyG DataLoader collation). Use bool() for the
        # scalar case and .any() for the tensor case.
        _hvn = getattr(data, "has_virtual_node", False)
        has_vnode: bool = bool(_hvn.any()) if hasattr(_hvn, "any") else bool(_hvn)
        if has_vnode:
            # The virtual node is always the last node per graph (appended last
            # in build_graph). Collect by finding the max-index node per graph.
            vnode_indices = torch.tensor(
                [int(torch.where(batch == g)[0].max().item()) for g in range(num_graphs)],
                dtype=torch.long,
                device=h_spa.device,
            )
            pooled = h_spa[vnode_indices]                              # (G, 64)
        else:
            pooled = global_mean_pool(h_spa, batch, size=num_graphs)  # (G, 64)
        eq_logits = self.eq_fc2(F.relu(self.eq_fc1(pooled)))          # (G, 4)

        # Edge-type auxiliary head reads VISUAL stream (role-aware)
        E = data.edge_index.shape[1]
        if E > 0:
            src_emb = h_vis[data.edge_index[0]]                        # (E, 64)
            dst_emb = h_vis[data.edge_index[1]]                        # (E, 64)
            edge_cat = torch.cat([src_emb, dst_emb, edge_feat], dim=1) # (E, 152)
            edge_type_logits = self.edge_head(edge_cat)                 # (E, 8)
        else:
            edge_type_logits = torch.zeros(0, NUM_EDGE_TYPES, device=h_vis.device)

        return (
            fine_logits,
            row_cluster_emb,
            within_row_ord_logits,
            col_cluster_emb,
            within_col_ord_logits,
            eq_logits,
            edge_type_logits,
        )

    def predict(self, data: Data) -> tuple[list[NodePrediction], str, list[float]]:
        """Single-graph inference. data.bbox must be (N, 4) pixel coordinates.

        Returns:
            (predictions, equation_type_str, eq_type_logits_as_list)
            eq_type_logits_as_list is a plain Python list of 4 floats (cpu),
            safe to pass directly to assemble_json.
        """
        with torch.no_grad():
            fl, row_emb, row_ord_logits, col_emb, col_ord_logits, eq_logits, _ = self.forward(data)

        fine_probs = F.softmax(fl, dim=1)

        # W-ARCH-2: strip the virtual scene node from per-node tensors before
        # assembling NodePrediction objects. The virtual node is always appended
        # last in build_graph, so slicing to [:N_real] is safe.
        # has_virtual_node may be a bool or a bool tensor depending on context.
        _hvn_pred = getattr(data, "has_virtual_node", False)
        has_vnode_pred: bool = bool(_hvn_pred.any()) if hasattr(_hvn_pred, "any") else bool(_hvn_pred)
        N_real = data.num_nodes - (1 if has_vnode_pred else 0)

        fine_idxs = fl.argmax(dim=1).tolist()[:N_real]
        within_row_ords = row_ord_logits.argmax(dim=1).tolist()[:N_real]
        within_col_ords = col_ord_logits.argmax(dim=1).tolist()[:N_real]
        eq_idx = int(eq_logits.argmax(dim=1).item())
        eq_type_logits: list[float] = eq_logits[0].cpu().tolist()

        # Cluster row and col IDs using real-node bbox coordinates only.
        # Pass data.bbox[:N_real] to exclude the canvas-spanning virtual bbox
        # from the gap-threshold computation.
        bbox_real = data.bbox[:N_real]

        # Recover YOLO coarse role BEFORE clustering so structural tokens (result_bar,
        # divide_bracket) can be excluded from the COLUMN gap computation.
        # Full-width result_bar and left-anchored div_bracket have large bbox widths
        # that inflate the median, raising the gap threshold and merging child columns.
        # Row clustering is NOT affected: structural tokens sit at valid y positions.
        role_block = data.x[:N_real, 5:5 + len(YOLO_CLASS_NAMES)]
        role_idxs = role_block.argmax(dim=1).tolist()

        structural_mask = [r in _STRUCTURAL_COARSE_IDS for r in role_idxs]
        non_structural_indices = [i for i, s in enumerate(structural_mask) if not s]

        row_cluster_ids = _cluster_by_coordinate(bbox_real, axis=1)  # y -> rows; all nodes OK

        # Column clustering: exclude structural tokens from the gap computation.
        if non_structural_indices and len(non_structural_indices) < len(role_idxs):
            bbox_non_structural = bbox_real[torch.tensor(non_structural_indices, dtype=torch.long)]
            col_ids_non_structural = _cluster_by_coordinate(bbox_non_structural, axis=0)
            # Map back: structural nodes get the col cluster of their x-nearest non-structural
            # neighbour (simple linear scan); this preserves a meaningful col_cluster_id
            # without letting them distort the gap computation.
            col_cluster_ids = [0] * len(role_idxs)
            bboxes_list = bbox_real.tolist()
            for rank, orig_i in enumerate(non_structural_indices):
                col_cluster_ids[orig_i] = col_ids_non_structural[rank]
            for i, s in enumerate(structural_mask):
                if not s:
                    continue
                # Assign the col cluster of the closest non-structural node by x-midpoint.
                cx_i = (bboxes_list[i][0] + bboxes_list[i][2]) / 2.0
                best_j, best_dist = non_structural_indices[0], float("inf")
                for j in non_structural_indices:
                    cx_j = (bboxes_list[j][0] + bboxes_list[j][2]) / 2.0
                    d = abs(cx_i - cx_j)
                    if d < best_dist:
                        best_dist, best_j = d, j
                col_cluster_ids[i] = col_cluster_ids[best_j]
        else:
            # All nodes are non-structural (normal case) or all are structural (degenerate):
            # run the standard full-set computation.
            col_cluster_ids = _cluster_by_coordinate(bbox_real, axis=0)  # x -> cols

        bboxes = data.bbox[:N_real].tolist()
        # Recover per-node given flags stored by graph_builder (default False when absent).
        given_tensor = getattr(data, "given", None)
        if given_tensor is not None:
            given_flags_pred: list[bool] = given_tensor[:N_real].tolist()
        else:
            given_flags_pred = [False] * N_real
        preds = [
            NodePrediction(
                fine_label=full_label_from_gnn_and_yolo(
                    _FINE_LABELS[fi],
                    YOLO_CLASS_NAMES[role_idxs[i]],
                ),
                row_cluster_id=row_cluster_ids[i],
                col_cluster_id=col_cluster_ids[i],
                within_row_ord=within_row_ords[i],
                within_col_ord=within_col_ords[i],
                x0=float(bboxes[i][0]),
                y0=float(bboxes[i][1]),
                x1=float(bboxes[i][2]),
                y1=float(bboxes[i][3]),
                confidence=float(fine_probs[i, fi]),
                given=given_flags_pred[i],
            )
            for i, fi in enumerate(fine_idxs)
        ]
        return preds, _EQ_TYPES[eq_idx], eq_type_logits
