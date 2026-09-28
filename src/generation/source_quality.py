"""Source crop quality scoring for the synthesis pipeline.

This module provides a single public function, ``score_crop``, that rates a
28x28 grayscale crop (uint8, white-background ink) on a 0..1 scale.  The
generator uses the score to reject poor-quality crops at runtime and retry
with a different random sample from the pool (up to a configurable limit).

Score components
----------------
- ink_density   : fraction of pixels with value < 200 (ink pixels). Captures
                  whether the crop has meaningful content.
- contrast      : normalised range (max - min) / 255.  Captures pen-pressure
                  richness; a washed-out grey crop scores low.
- connectivity  : ratio of the largest connected component of ink pixels to
                  the total ink pixel count.  Heavily fragmented crops score
                  low; a single solid stroke scores 1.0.

Final score = 0.4 * ink_density + 0.3 * contrast + 0.3 * connectivity
Threshold (default): reject if score < 0.4.

Design note: stroke widths are intentionally NOT normalised here -- that is
the job of the pool-cleanup script (Workstream A).  Do not add morphological
ops that would thicken strokes (see memory: track_b_thicker_strokes_failed).
"""

from __future__ import annotations

import numpy as np
import cv2

# Weight vector for the three components.
_W_INK = 0.4
_W_CONTRAST = 0.3
_W_CONNECT = 0.3

# Default rejection threshold (also set as config.toml [generation] min_source_quality).
DEFAULT_MIN_SOURCE_QUALITY: float = 0.4


def score_crop(tile: np.ndarray) -> float:
    """Return a quality score in [0, 1] for a grayscale crop.

    Parameters
    ----------
    tile:
        2-D uint8 numpy array, white-background (255 = white, 0 = black ink).
        Any size is accepted but 28x28 is the canonical input.

    Returns
    -------
    float in [0, 1].  0 means empty or constant-grey; 1 means clean dense
    single-stroke content.
    """
    if tile.ndim != 2:
        raise ValueError(f"score_crop expects a 2-D array, got shape {tile.shape}")
    if tile.size == 0:
        return 0.0

    tile_u8 = tile.astype(np.uint8)

    # --- ink density -------------------------------------------------------
    ink_mask = tile_u8 < 200  # True where ink pixel
    ink_count = int(np.count_nonzero(ink_mask))
    n_pixels = tile_u8.size
    ink_density = ink_count / n_pixels

    if ink_count == 0:
        # Completely empty crop -- all three components are 0.
        return 0.0

    # --- contrast ----------------------------------------------------------
    pmin = int(tile_u8.min())
    pmax = int(tile_u8.max())
    contrast = (pmax - pmin) / 255.0

    # --- connectivity -------------------------------------------------------
    # Work on a binary image: ink pixels = 255, background = 0.
    binary = (ink_mask.astype(np.uint8)) * 255
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if n_labels <= 1:
        # No foreground components at all (shouldn't happen given ink_count > 0).
        connectivity = 0.0
    else:
        # Component 0 is background; find the largest foreground component.
        fg_areas = stats[1:, cv2.CC_STAT_AREA]  # exclude background label
        largest = int(fg_areas.max())
        connectivity = largest / ink_count  # 1.0 if all ink is one component

    score = _W_INK * ink_density + _W_CONTRAST * contrast + _W_CONNECT * connectivity
    return float(min(1.0, max(0.0, score)))
