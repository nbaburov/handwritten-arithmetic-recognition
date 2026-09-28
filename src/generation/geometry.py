"""Composable distortion functions for glyph rendering.

Provides rotate_tile (per-glyph rotation with tight bbox crop).
"""
from __future__ import annotations


import cv2
import numpy as np


from ..core.ink import INK_BBOX_THRESHOLD as _INK_THRESHOLD  # noqa: N811 — local alias preserves existing usage

_MIN_THETA_DEG: float = -18.0
_MAX_THETA_DEG: float = 18.0


def rotate_tile(
    tile: np.ndarray,
    theta_deg: float,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Rotate a glyph tile and return the tight ink bounding box.

    Args:
        tile: uint8 grayscale array, white background (255), ink < 255.
        theta_deg: Rotation angle in degrees (positive = counterclockwise,
            per cv2 convention). Clipped to [-18, +18].

    Returns:
        A tuple ``(rotated_tile, (dx_offset, dy_offset))``.
        ``rotated_tile`` is the minimum tight bbox around ink pixels on a white
        canvas.  If the input has no ink, returns ``(tile, (0, 0))``.
        ``(dx_offset, dy_offset)`` is the pixel shift from the original
        tile's top-left to the ink-bbox top-left after rotation.
    """
    theta_deg = float(np.clip(theta_deg, _MIN_THETA_DEG, _MAX_THETA_DEG))

    ink_mask = tile < _INK_THRESHOLD
    if not ink_mask.any():
        return tile, (0, 0)

    h, w = tile.shape
    centre = (w / 2.0, h / 2.0)
    rot_matrix = cv2.getRotationMatrix2D(centre, theta_deg, 1.0)
    white_fill = 255
    rotated = cv2.warpAffine(
        tile,
        rot_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=white_fill,
    )

    rotated_ink = rotated < _INK_THRESHOLD
    if not rotated_ink.any():
        return tile, (0, 0)

    rows = np.any(rotated_ink, axis=1)
    cols = np.any(rotated_ink, axis=0)
    min_row, max_row = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    min_col, max_col = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))

    cropped = rotated[min_row: max_row + 1, min_col: max_col + 1].copy()
    dx_offset = min_col
    dy_offset = min_row
    return cropped, (dx_offset, dy_offset)


