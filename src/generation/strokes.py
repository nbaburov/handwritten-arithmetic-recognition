"""Spline-based bar and bracket rendering for synthetic handwriting scenes.

All drawing functions operate in-place on a uint8 grayscale numpy canvas
(white background = 255) and return the tight bounding box of the drawn
stroke as ``(x0, y0, x1, y1)``.

All randomness is drawn exclusively from the caller-supplied
``rng: random.Random`` instance so results are reproducible.

Hand-drawn character model
--------------------------
To reduce the Frankenstein effect (clean programmatic lines next to imperfect
CROHME glyphs), bars and brackets apply four human-imperfection layers:

1. **Variable thickness per segment** — each segment samples its own thickness
   independently from [thickness_min, thickness_max].
2. **Intensity variation** — segments are drawn in a random dark-grey shade
   rather than pure black, mimicking ink-pressure variation.
3. **Micro-breaks** — with a small per-segment probability, a 1-2 px gap is
   inserted, accumulating a jerky/dotted hand-drawn look distinct from the
   single large gap controlled by ``gap_prob``.
4. **X + Y jitter on control points** — both axes are jittered so the stroke
   wanders slightly rather than running in a straight horizontal/vertical line.
"""
from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.generation.handwriting_style import HandwritingStyle

from ..core.ink import INK_DRAW_THRESHOLD as _INK_THRESHOLD  # noqa: N811 — local alias preserves existing usage

# Bar control-point count
_BAR_CONTROL_POINTS: int = 7

# Gap probability for mid-bar break (single large gap — separate from micro-breaks)
_BAR_GAP_PROB: float = 0.04
_BRACKET_GAP_PROB: float = 0.04


def _gauss_jitter(rng: random.Random, sigma: float) -> float:
    """Return a Gaussian sample with mean 0 and std ``sigma`` from ``rng``."""
    return rng.gauss(0.0, sigma)


def draw_handwritten_bar(
    canvas: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,  # noqa: ARG001  — y1 provided for bbox height reference only
    style: "HandwritingStyle",
    rng: random.Random,
    gap_prob: float = _BAR_GAP_PROB,
    thickness_min: int = 2,
    thickness_max: int = 3,
    intensity_min: int = 30,
    intensity_max: int = 80,
    micro_break_prob: float = 0.08,
    y_jitter_sigma: float = 3.0,
    x_jitter_sigma: float = 1.5,
) -> tuple[int, int, int, int]:
    """Draw a horizontal result bar from (x0, y0) to (x1, y0) on ``canvas``.

    The bar has per-segment variable thickness, ink intensity variation,
    micro-breaks, x+y jitter on control points, and an optional single mid-bar
    gap.  Drawing is in-place.

    Args:
        canvas: uint8 grayscale array (white = 255) to draw on.
        x0: Left x-coordinate of the bar.
        y0: Vertical position of the bar (y-centre).
        x1: Right x-coordinate of the bar.
        y1: Bottom of the surrounding bbox (used for height reference only).
        style: HandwritingStyle controlling ink darkness.
        rng: Seeded random.Random for all randomness.
        gap_prob: Probability of a single large mid-bar gap.
        thickness_min: Minimum stroke thickness (pixels) sampled per segment.
        thickness_max: Maximum stroke thickness (pixels) sampled per segment.
        intensity_min: Minimum ink darkness on 0-255 scale (0=black).
        intensity_max: Maximum ink darkness on 0-255 scale.
        micro_break_prob: Per-segment probability of a 1-2 px micro-gap.
        y_jitter_sigma: Gaussian sigma for per-control-point y displacement.
        x_jitter_sigma: Gaussian sigma for per-control-point x displacement.

    Returns:
        ``(min_x, min_y, max_x, max_y)`` tight bbox inclusive of stroke width.
    """
    # Compute a per-segment thickness and intensity pool up front
    max_thickness: int = max(thickness_min, thickness_max)
    half_max_thickness: int = max_thickness // 2

    # Endpoint jitter in x — scales with x_jitter_sigma (0 = perfectly straight endpoints)
    x_jitter_range: int = max(0, round(x_jitter_sigma * 2.0))
    if x_jitter_range > 0:
        x0_actual: int = x0 + rng.randint(-x_jitter_range, x_jitter_range)
        x1_actual: int = x1 + rng.randint(-x_jitter_range, x_jitter_range)
    else:
        x0_actual = x0
        x1_actual = x1

    # Optional single mid-bar gap
    apply_gap: bool = rng.random() < gap_prob
    gap_min_x: int = 0
    gap_max_x: int = 0
    if apply_gap:
        bar_len: int = abs(x1_actual - x0_actual)
        gap_width: int = rng.randint(2, 4)
        gap_start_frac: float = rng.uniform(0.42, 0.58)
        gap_min_x = x0_actual + int(gap_start_frac * bar_len)
        gap_max_x = gap_min_x + gap_width

    # Build control points along the bar with per-point x+y jitter
    control_pts: list[tuple[int, int]] = [
        (
            x0_actual + i * (x1_actual - x0_actual) // (_BAR_CONTROL_POINTS - 1)
            + round(_gauss_jitter(rng, x_jitter_sigma)),
            y0 + round(_gauss_jitter(rng, y_jitter_sigma)),
        )
        for i in range(_BAR_CONTROL_POINTS)
    ]

    # Track bbox extents
    min_x: int = min(p[0] for p in control_pts) - half_max_thickness
    max_x: int = max(p[0] for p in control_pts) + half_max_thickness
    min_y: int = min(p[1] for p in control_pts) - half_max_thickness
    max_y: int = max(p[1] for p in control_pts) + half_max_thickness

    # Draw segments between consecutive control points
    for idx in range(len(control_pts) - 1):
        px_a, py_a = control_pts[idx]
        px_b, py_b = control_pts[idx + 1]

        # Large gap check
        if apply_gap:
            seg_x_min = min(px_a, px_b)
            seg_x_max = max(px_a, px_b)
            if seg_x_max > gap_min_x and seg_x_min < gap_max_x:
                continue

        # Micro-break: skip this segment entirely
        if rng.random() < micro_break_prob:
            continue

        # Per-segment variable thickness
        seg_thickness: int = rng.randint(thickness_min, thickness_max)

        # Per-segment intensity variation (0=black, higher=lighter grey)
        seg_intensity: int = rng.randint(intensity_min, intensity_max)

        cv2.line(
            canvas,
            (px_a, py_a),
            (px_b, py_b),
            color=int(seg_intensity),
            thickness=seg_thickness,
        )

    return (min_x, min_y, max_x, max_y)


def draw_handwritten_bracket(
    canvas: np.ndarray,
    corner_x: int,
    corner_y: int,
    arm_dx: int,   # pixels to extend RIGHT (over dividend)
    arm_dy: int,   # pixels to extend DOWN
    style: "HandwritingStyle",
    rng: random.Random,
    row_slope_deg: float,
    gap_prob: float = _BRACKET_GAP_PROB,
    thickness_min: int = 2,
    thickness_max: int = 3,
    intensity_min: int = 30,
    intensity_max: int = 80,
    micro_break_prob: float = 0.08,
    h_y_jitter_sigma: float = 2.5,
    h_x_jitter_sigma: float = 1.0,
    v_x_jitter_sigma: float = 1.5,
) -> tuple[int, int, int, int]:
    """Draw a long-division bracket (⌐ shape) on ``canvas``.

    The horizontal arm extends RIGHT from corner_x over the dividend.
    The vertical arm extends DOWN from corner_x.
    Both arms apply per-segment variable thickness, intensity variation,
    micro-breaks, and x+y control-point jitter.

    Args:
        canvas: uint8 grayscale array (white = 255) to draw on.
        corner_x: x pixel of the inner corner (horizontal meets vertical).
        corner_y: y pixel of the inner corner.
        arm_dx: Length of the horizontal arm in pixels (extends RIGHT).
        arm_dy: Length of the vertical arm in pixels (extends DOWN).
        style: HandwritingStyle controlling ink darkness.
        rng: Seeded random.Random for all randomness.
        row_slope_deg: Tilt applied to the horizontal arm.
        gap_prob: Probability of a single large gap on the horizontal arm.
        thickness_min: Minimum stroke thickness sampled per segment.
        thickness_max: Maximum stroke thickness sampled per segment.
        intensity_min: Minimum ink darkness (0=black) per segment.
        intensity_max: Maximum ink darkness per segment.
        micro_break_prob: Per-segment probability of a 1-2 px micro-gap.
        h_y_jitter_sigma: Gaussian sigma for horizontal arm y jitter.
        h_x_jitter_sigma: Gaussian sigma for horizontal arm x jitter.
        v_x_jitter_sigma: Gaussian sigma for vertical arm x jitter.

    Returns:
        ``(min_x, min_y, max_x, max_y)`` tight bbox — always within canvas bounds.
    """
    canvas_h, canvas_w = canvas.shape[:2]
    max_thickness: int = max(thickness_min, thickness_max)
    half_max_thickness: int = max_thickness // 2

    # --- Horizontal arm: from (corner_x, corner_y) extending RIGHT by arm_dx ---
    _H_CONTROL_POINTS: int = 5
    slope_tan: float = math.tan(math.radians(row_slope_deg))

    h_ctrl: list[tuple[int, int]] = []
    for k in range(_H_CONTROL_POINTS):
        frac = k / (_H_CONTROL_POINTS - 1)
        px = corner_x + int(frac * arm_dx) + round(_gauss_jitter(rng, h_x_jitter_sigma))
        py = corner_y + round(slope_tan * frac * arm_dx) + round(_gauss_jitter(rng, h_y_jitter_sigma))
        h_ctrl.append((px, py))

    # Optional single large gap on horizontal arm
    apply_gap: bool = rng.random() < gap_prob
    gap_min_x: int = 0
    gap_max_x: int = 0
    if apply_gap and arm_dx > 8:
        gap_width: int = rng.randint(2, 4)
        gap_start_frac: float = rng.uniform(0.40, 0.60)
        gap_min_x = corner_x + int(gap_start_frac * arm_dx)
        gap_max_x = gap_min_x + gap_width

    for idx in range(len(h_ctrl) - 1):
        px_a, py_a = h_ctrl[idx]
        px_b, py_b = h_ctrl[idx + 1]
        if apply_gap:
            seg_x_min = min(px_a, px_b)
            seg_x_max = max(px_a, px_b)
            if seg_x_max > gap_min_x and seg_x_min < gap_max_x:
                continue
        if rng.random() < micro_break_prob:
            continue
        seg_thickness: int = rng.randint(thickness_min, thickness_max)
        seg_intensity: int = rng.randint(intensity_min, intensity_max)
        cv2.line(canvas, (px_a, py_a), (px_b, py_b), color=seg_intensity, thickness=seg_thickness)

    # --- Vertical arm: from (corner_x, corner_y) extending DOWN by arm_dy ---
    _V_CONTROL_POINTS: int = 4
    v_ctrl: list[tuple[int, int]] = []
    for k in range(_V_CONTROL_POINTS):
        frac = k / (_V_CONTROL_POINTS - 1)
        px = corner_x + round(_gauss_jitter(rng, v_x_jitter_sigma))
        py = corner_y + int(frac * arm_dy)
        v_ctrl.append((px, py))

    for idx in range(len(v_ctrl) - 1):
        px_a, py_a = v_ctrl[idx]
        px_b, py_b = v_ctrl[idx + 1]
        if rng.random() < micro_break_prob:
            continue
        seg_thickness = rng.randint(thickness_min, thickness_max)
        seg_intensity = rng.randint(intensity_min, intensity_max)
        cv2.line(canvas, (px_a, py_a), (px_b, py_b), color=seg_intensity, thickness=seg_thickness)

    # --- Compute bbox from all control points, then canvas-clamp ---
    all_x = [p[0] for p in h_ctrl] + [p[0] for p in v_ctrl]
    all_y = [p[1] for p in h_ctrl] + [p[1] for p in v_ctrl]
    min_x = max(0, min(all_x) - half_max_thickness)
    max_x = min(canvas_w, max(all_x) + half_max_thickness)
    min_y = max(0, min(all_y) - half_max_thickness)
    max_y = min(canvas_h, max(all_y) + half_max_thickness)

    return (min_x, min_y, max_x, max_y)
