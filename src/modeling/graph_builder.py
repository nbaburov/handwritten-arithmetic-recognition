"""Builds PyTorch Geometric Data graphs from YOLO detections and scene images for GNN input."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch_geometric.data import Data

from ..core.config import MAX_NODES
from ..core.ontology import YOLO_CLASS_NAMES
from ..parsing.detection import Detection
from .graph_features import build_edge_types, build_geometric_edge_features

_ROLE_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(YOLO_CLASS_NAMES)}
_N_COARSE: int = len(YOLO_CLASS_NAMES)

# Role importance scalars per YOLO coarse class.
_ROLE_IMPORTANCE: dict[str, float] = {
    "operator": 1.0,
    "result_bar": 1.0,
    "divide_bracket": 1.0,
    "digit_carry": 0.3,
    "digit_borrow": 0.3,
    "digit_main": 0.5,
}

# Index of "digit_main" in YOLO_CLASS_NAMES — used to filter medians.
_DIGIT_MAIN_IDX: int = YOLO_CLASS_NAMES.index("digit_main")


def _crop_tensor(gray: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> Tensor:
    """Extract a 28×28 normalised grayscale crop from the scene image."""
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        patch = np.zeros((1, 1), dtype=np.uint8)
    resized = np.array(
        Image.fromarray(patch).resize((28, 28), resample=Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    return torch.tensor(resized).unsqueeze(0)  # (1, 28, 28)


def _compute_scene_features(
    detections: list[Detection],
    w_img: int,
    h_img: int,
) -> list[float]:
    """Compute 8 scene-shape summary features for the virtual node (W-ARCH-2).

    All values are normalised to a [0, 1] range using fixed denominators.

    Channels (in order):
      0: total_nodes           -- count of real nodes, normalised by MAX_NODES
      1: result_bar_count      -- count of result_bar detections, normalised by 3
      2: div_bracket_count     -- count of divide_bracket detections, normalised by 2
      3: operator_count        -- count of operator detections, normalised by 4
      4: carry_count           -- count of digit_carry detections, normalised by 8
      5: borrow_count          -- count of digit_borrow detections, normalised by 8
      6: row_count_estimate    -- gap-based cluster count on y-midpoints, normalised by 6
      7: aspect_ratio_spread   -- std of (bw/bh) across all detections, normalised by 2
    """
    if not detections:
        return [0.0] * 8

    total_nodes = len(detections)
    result_bar_count = sum(1 for d in detections if d.label == "result_bar")
    div_bracket_count = sum(1 for d in detections if d.label == "divide_bracket")
    operator_count = sum(1 for d in detections if d.label == "operator")
    carry_count = sum(1 for d in detections if d.label == "digit_carry")
    borrow_count = sum(1 for d in detections if d.label == "digit_borrow")

    # Inline gap-based row count: mirrors _cluster_by_coordinate logic but
    # operates on raw detection y-midpoints to avoid a cross-module import.
    y_mids = [(d.y0 + d.y1) / 2.0 for d in detections]
    heights = [max(abs(d.y1 - d.y0), 1.0) for d in detections]
    sorted_heights = sorted(heights)
    median_h = sorted_heights[len(sorted_heights) // 2]
    gap_thresh = 0.5 * median_h
    sorted_mids = sorted(y_mids)
    row_count = 1
    for i in range(1, len(sorted_mids)):
        if abs(sorted_mids[i] - sorted_mids[i - 1]) > gap_thresh:
            row_count += 1

    # Aspect ratio spread: std of bw/bh across detections
    aspect_ratios: list[float] = []
    for d in detections:
        bw = max(abs(d.x1 - d.x0), 1.0)
        bh = max(abs(d.y1 - d.y0), 1.0)
        aspect_ratios.append(bw / bh)
    if len(aspect_ratios) > 1:
        mean_ar = sum(aspect_ratios) / len(aspect_ratios)
        variance = sum((a - mean_ar) ** 2 for a in aspect_ratios) / len(aspect_ratios)
        ar_std = variance ** 0.5
    else:
        ar_std = 0.0

    return [
        total_nodes / MAX_NODES,
        result_bar_count / 3.0,
        div_bracket_count / 2.0,
        operator_count / 4.0,
        carry_count / 8.0,
        borrow_count / 8.0,
        row_count / 6.0,
        ar_std / 2.0,
    ]


def build_graph(
    detections: list[Detection],
    gray: np.ndarray,
    *,
    node_mask: torch.BoolTensor | None = None,
    max_nodes: int = MAX_NODES,
    use_scene_virtual_node: bool = False,
) -> Data:
    """Build a PyG Data object from YOLO detections and the scene grayscale image.

    Constructs a fully-connected directed graph (all ordered pairs i!=j) with
    16-dimensional geometric edge features and 8-class edge-type labels.

    Args:
        detections: YOLO detections for a single scene.
        gray: (512, 512) uint8 grayscale scene image -- must have been produced
            by ``preprocess_for_pipeline``.
        node_mask: Optional boolean tensor (length = number of detections after
            truncation). True means keep the node. Applied after truncation.
        max_nodes: Hard cap on scene nodes; detections above this are dropped
            by ascending confidence (lowest first).
        use_scene_virtual_node: W-ARCH-2 flag (default False). When True, one
            virtual scene node is appended after truncation and node_mask.
            The virtual node carries 8-dim scene summary features (structural
            token counts, row estimate) in dims 0-7 and zeros in dims 8-12.
            Its crop is an all-zeros tile; its bbox spans the full canvas.
            The returned Data carries has_virtual_node=True when this path is
            active. The caller (SymbolGNN.predict) must strip the last
            NodePrediction before assembling the output JSON.

    Returns:
        PyG Data with fields:
            crops   (N, 1, 28, 28)  raw crops for CropBackbone
            x       (N, 13)         node features
            bbox    (N, 4)          pixel coords [x0, y0, x1, y1]
            edge_index (2, E)       dense directed edges
            edge_attr  (E, 16)      geometric edge features
            edge_type  (E,)         long -- edge-type vocabulary index
            num_nodes  int
            truncated  bool
            has_virtual_node  bool  (True only when use_scene_virtual_node=True)
    """
    if gray.shape != (512, 512):
        raise ValueError(
            f"gray must be (512, 512) — run preprocess_for_pipeline first. Got {gray.shape}."
        )

    h_img, w_img = gray.shape

    # --- truncation -------------------------------------------------------
    # Capture the full detection list before truncation for virtual node features.
    # Scene summary counts (structural tokens, row estimate) are more accurate
    # when computed from all detections, not just the top-max_nodes subset.
    detections_before_truncation = detections
    truncated: bool = False
    if len(detections) > max_nodes:
        # Sort descending by confidence; given=True detections have confidence=1.0
        # by convention so they naturally survive truncation.
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        detections = detections[:max_nodes]
        truncated = True

    # --- build raw node features ------------------------------------------
    crops: list[Tensor] = []
    node_feats: list[list[float]] = []
    bboxes: list[list[float]] = []
    roles: list[str] = []
    given_flags: list[bool] = []

    for det in detections:
        x0 = max(0, int(det.x0))
        y0 = max(0, int(det.y0))
        x1 = min(w_img, max(x0 + 1, int(det.x1)))
        y1 = min(h_img, max(y0 + 1, int(det.y1)))

        crops.append(_crop_tensor(gray, x0, y0, x1, y1))
        bboxes.append([det.x0, det.y0, det.x1, det.y1])

        cx = (det.x0 + det.x1) / 2.0
        cy = (det.y0 + det.y1) / 2.0
        bw = max(det.x1 - det.x0, 1.0)
        bh = max(det.y1 - det.y0, 1.0)

        role_idx = _ROLE_TO_IDX.get(det.label, 0)
        role_onehot: list[float] = [0.0] * _N_COARSE
        role_onehot[role_idx] = 1.0

        importance = _ROLE_IMPORTANCE.get(det.label, 0.5)

        node_feats.append([
            cx / w_img,
            cy / h_img,
            bw / w_img,
            bh / h_img,
            (bw * bh) / (w_img * h_img),
            *role_onehot,
            float(det.confidence),
            importance,
        ])
        roles.append(det.label)
        given_flags.append(det.given)

    # --- apply node_mask --------------------------------------------------
    if node_mask is not None:
        keep_indices = torch.where(node_mask)[0].tolist()
        crops = [crops[i] for i in keep_indices]
        node_feats = [node_feats[i] for i in keep_indices]
        bboxes = [bboxes[i] for i in keep_indices]
        roles = [roles[i] for i in keep_indices]
        given_flags = [given_flags[i] for i in keep_indices]

    N = len(crops)

    # --- W-ARCH-2: scene virtual node (appended after real nodes) ---------
    # MAX_NODES truncation above applies to real nodes only; the virtual node
    # is appended unconditionally when the flag is on, bringing total to N+1.
    if use_scene_virtual_node:
        scene_feats = _compute_scene_features(
            detections_before_truncation, w_img, h_img
        )
        # Virtual node feature vector: 13 dims matching real node layout.
        # Dims 0-7: scene summary features; dims 8-12: zero-padded.
        vnode_x: list[float] = scene_feats + [0.0] * (13 - len(scene_feats))
        crops.append(torch.zeros(1, 28, 28))          # zero-crop for CropBackbone
        node_feats.append(vnode_x)
        bboxes.append([0.0, 0.0, float(w_img), float(h_img)])  # full canvas bbox
        roles.append("__scene__")  # sentinel; falls back to edge-type 7 ("other")
        given_flags.append(False)  # virtual node is never a given token
        N += 1  # explicit increment; len(crops) now reflects the virtual append

    crops_t = torch.stack(crops) if N > 0 else torch.zeros(0, 1, 28, 28)
    x = torch.tensor(node_feats, dtype=torch.float32) if N > 0 else torch.zeros(0, 13)
    bbox_t = torch.tensor(bboxes, dtype=torch.float32) if N > 0 else torch.zeros(0, 4)
    given_t = torch.tensor(given_flags, dtype=torch.bool) if N > 0 else torch.zeros(0, dtype=torch.bool)

    # --- dense edges ------------------------------------------------------
    if N <= 1:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, 16, dtype=torch.float32)
        edge_type = torch.zeros(0, dtype=torch.long)
    else:
        # All ordered pairs (i, j) with i ≠ j — E = N*(N-1)
        src_list, dst_list = [], []
        for i in range(N):
            for j in range(N):
                if i != j:
                    src_list.append(i)
                    dst_list.append(j)
        src_t = torch.tensor(src_list, dtype=torch.long)
        dst_t = torch.tensor(dst_list, dtype=torch.long)
        edge_index = torch.stack([src_t, dst_t], dim=0)

        pos = x[:, :2]  # normalised cx, cy — first two node features

        # --- scene-median digit width/height for normalisation ------------
        role_onehot_block = x[:, 5:5 + _N_COARSE]  # (N, 6)
        role_indices = torch.argmax(role_onehot_block, dim=1)  # (N,)
        digit_main_mask = role_indices == _DIGIT_MAIN_IDX
        if digit_main_mask.any():
            digit_widths = x[digit_main_mask, 2]
            digit_heights = x[digit_main_mask, 3]
        else:
            digit_widths = x[:, 2]
            digit_heights = x[:, 3]

        median_digit_w = float(digit_widths.median())
        median_digit_h = float(digit_heights.median())

        edge_attr = build_geometric_edge_features(
            pos, x, src_t, dst_t, median_digit_w, median_digit_h
        )

        src_roles = [roles[i] for i in src_list]
        dst_roles = [roles[j] for j in dst_list]
        edge_type = build_edge_types(src_roles, dst_roles)

    return Data(
        crops=crops_t,
        x=x,
        bbox=bbox_t,
        given=given_t,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type,
        num_nodes=N,
        truncated=truncated,
        has_virtual_node=use_scene_virtual_node,
    )
