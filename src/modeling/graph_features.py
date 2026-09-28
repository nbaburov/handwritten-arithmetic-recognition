"""Pure functions for building edge features and edge types for the dense N×N graph."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .edge_types import NUM_EDGE_TYPES, pair_edge_type
from ..core.ontology import YOLO_CLASS_NAMES

# ---------------------------------------------------------------------------
# Role index constants derived from ontology — no magic numbers.
# ---------------------------------------------------------------------------
_OPERATOR_IDX: int = YOLO_CLASS_NAMES.index("operator")
_RESULT_BAR_IDX: int = YOLO_CLASS_NAMES.index("result_bar")
_DIVIDE_BRACKET_IDX: int = YOLO_CLASS_NAMES.index("divide_bracket")
_NUM_ROLES: int = len(YOLO_CLASS_NAMES)

# Indices within the node feature vector where the role one-hot starts/ends.
_ROLE_ONEHOT_START: int = 5
_ROLE_ONEHOT_END: int = _ROLE_ONEHOT_START + _NUM_ROLES  # exclusive → 11


def build_geometric_edge_features(
    pos: Tensor,
    x: Tensor,
    src: Tensor,
    dst: Tensor,
    median_digit_w: float,
    median_digit_h: float,
) -> Tensor:
    """Compute 16-dimensional geometric edge feature vectors.

    Encodes spatial relationship, size ratio, and role flags between every
    ordered source-destination pair.

    Args:
        pos: (N, 2) normalised centre-x, centre-y for each node.
        x: (N, 13) node feature matrix; indices 2 and 3 are normalised width
            and height; indices 5..10 are the role one-hot block.
        src: (E,) long tensor of source node indices.
        dst: (E,) long tensor of destination node indices.
        median_digit_w: scene-median digit width in normalised coordinates.
        median_digit_h: scene-median digit height in normalised coordinates.

    Returns:
        (E, 16) float32 tensor of edge features in the order documented in the
        module docstring.
    """
    dx = pos[dst, 0] - pos[src, 0]                                 # 0
    dy = pos[dst, 1] - pos[src, 1]                                 # 1
    dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)                   # 2
    w_ratio = x[dst, 2] / (x[src, 2] + 1e-6)                      # 3
    h_ratio = x[dst, 3] / (x[src, 3] + 1e-6)                      # 4
    angle_sin = dy / (dist + 1e-8)                                 # 5
    angle_cos = dx / (dist + 1e-8)                                 # 6
    col_delta_norm = dx / (median_digit_w + 1e-6)                  # 7
    row_delta_norm = dy / (median_digit_h + 1e-6)                  # 8
    size_ratio = (x[dst, 2] * x[dst, 3]) / (x[src, 2] * x[src, 3] + 1e-6)  # 9
    log_dist = torch.log(dist + 1e-8)                              # 10

    # Role classification from the one-hot block (indices 5..10 in x).
    role_onehot = x[:, _ROLE_ONEHOT_START:_ROLE_ONEHOT_END]       # (N, 6)
    role_idx = torch.argmax(role_onehot, dim=1)                    # (N,)

    src_is_operator = (role_idx[src] == _OPERATOR_IDX).float()                          # 11
    dst_is_operator = (role_idx[dst] == _OPERATOR_IDX).float()                          # 12
    src_is_bar_or_bracket = (
        (role_idx[src] == _RESULT_BAR_IDX) | (role_idx[src] == _DIVIDE_BRACKET_IDX)
    ).float()                                                                            # 13
    dst_is_bar_or_bracket = (
        (role_idx[dst] == _RESULT_BAR_IDX) | (role_idx[dst] == _DIVIDE_BRACKET_IDX)
    ).float()                                                                            # 14
    same_role_flag = (role_idx[src] == role_idx[dst]).float()                           # 15

    return torch.stack(
        [
            dx, dy, dist, w_ratio, h_ratio,
            angle_sin, angle_cos,
            col_delta_norm, row_delta_norm,
            size_ratio, log_dist,
            src_is_operator, dst_is_operator,
            src_is_bar_or_bracket, dst_is_bar_or_bracket,
            same_role_flag,
        ],
        dim=1,
    )


def build_edge_types(
    src_roles: list[str],
    dst_roles: list[str],
) -> Tensor:
    """Map per-edge YOLO role name pairs to edge-type indices.

    Args:
        src_roles: YOLO coarse class name for each source node (length E).
        dst_roles: YOLO coarse class name for each destination node (length E).

    Returns:
        (E,) long tensor with values in [0, NUM_EDGE_TYPES).
    """
    indices = [pair_edge_type(s, d) for s, d in zip(src_roles, dst_roles)]
    return torch.tensor(indices, dtype=torch.long)
