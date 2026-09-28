"""Preprocessing pipeline shared by training-time synthetic generation and inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PreprocessConfig:
    target_size: int = 512
    stroke_target_px: int = 0  # 0 = skip normalisation (opt-in only). NOTE (iter11): scene-level norm
    # to 2px thins synthetic toward real (~2px) but ERODES soft grayscale glyphs into faint/fragmented
    # strokes on small crowded glyphs (probe-#48 risk). Kept off; the thickness fix is being decided
    # at the render-scale level instead.
    ink_threshold: int = 200
    contrast_clip_low: float = 1.0
    contrast_clip_high: float = 99.0


def resize_and_pad_to_square(img: np.ndarray, size: int) -> np.ndarray:
    """
    Resize so the longest edge equals size, then white-pad to size×size.

    Padding is symmetric (centered). Returns uint8 grayscale (size, size).
    """
    h, w = img.shape[:2]
    if h >= w:
        new_h = size
        new_w = max(1, int(w * size / h))
    else:
        new_w = size
        new_h = max(1, int(h * size / w))

    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

    result = np.full((size, size), 255, dtype=np.uint8)
    pad_y = (size - new_h) // 2
    pad_x = (size - new_w) // 2
    result[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = np.array(pil_img, dtype=np.uint8)
    return result


def contrast_normalise(gray: np.ndarray, clip_low: float, clip_high: float) -> np.ndarray:
    """Stretch clip_low–clip_high percentile range to [0, 255]. Returns uint8."""
    lo = float(np.percentile(gray, clip_low))
    hi = float(np.percentile(gray, clip_high))
    if hi <= lo:
        return gray.copy()
    stretched = (gray.astype(np.float32) - lo) / (hi - lo) * 255.0
    return np.clip(stretched, 0.0, 255.0).astype(np.uint8)


def ink_mask(gray: np.ndarray, threshold: int) -> np.ndarray:
    """Return boolean array where True = ink pixel (dark, below threshold)."""
    return gray < threshold


def estimate_stroke_width(mask: np.ndarray) -> float:
    """
    Estimate median stroke width via distance transform on ink mask.

    Returns 2 × median(distance transform over positive pixels), or 0.0 if mask is empty.
    """
    import cv2  # noqa: PLC0415

    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    positives = dist[mask]
    if positives.size == 0:
        return 0.0
    return float(2.0 * float(np.median(positives)))


def normalise_stroke_width(
    gray: np.ndarray, mask: np.ndarray, target_px: int
) -> np.ndarray:
    """
    Dilate or erode gray so median stroke width matches target_px.

    Operates on inverted-intensity image (bright = ink) so cv2 morphology acts on ink pixels.
    Opens before dilating to suppress speckle amplification.
    Returns uint8 grayscale.
    """
    import cv2  # noqa: PLC0415

    current_width = estimate_stroke_width(mask)
    if current_width <= 0:
        return gray.copy()

    delta = int(round(target_px - current_width))
    if delta == 0:
        return gray.copy()

    radius = abs(delta)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    ink = (255 - gray.astype(np.int16)).clip(0, 255).astype(np.uint8)

    if delta > 0:
        # Strokes too thin — open to remove speckle, then dilate
        ink_clean = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        ink_out = cv2.dilate(ink_clean, kernel)
    else:
        # Strokes too thick — erode
        ink_out = cv2.erode(ink, kernel)

    return (255 - ink_out.astype(np.int16)).clip(0, 255).astype(np.uint8)


def normalise_glyph_stroke_width(tile: np.ndarray, target_px: float) -> np.ndarray:
    """Normalise a single glyph tile to a target median stroke width.

    Tile must be uint8 grayscale, white background (255), black ink (0).
    target_px = desired median stroke width in pixels (e.g. 2.0).
    Returns adjusted uint8 tile (same shape, same white/black convention).

    Algorithm:
    1. Build ink mask (tile < INK_BINARIZE_THRESHOLD).
    2. If mask empty, return unchanged.
    3. Compute distance transform on mask → stroke half-widths.
    4. Median of nonzero distances × 2 = current median stroke width.
    5. delta = target_px - current_width.
    6. If |delta| <= 0.5: return unchanged (already close enough).
    7. If delta > 0.5: dilate ink by ceil(delta/2) iterations (thin → thicker).
    8. If delta < -0.5: erode ink by ceil(-delta/2) iterations (thick → thinner).
    9. Rebuild tile from updated mask (0 = ink, 255 = background).
    """
    import math as _math
    import cv2  # noqa: PLC0415

    from ..core.ink import INK_BINARIZE_THRESHOLD

    ink_mask = (tile < INK_BINARIZE_THRESHOLD).astype(np.uint8)
    if not ink_mask.any():
        return tile.copy()

    # estimate_stroke_width expects a boolean mask for correct numpy fancy-indexing
    current_width = estimate_stroke_width(ink_mask.astype(bool))
    if current_width <= 0:
        return tile.copy()

    delta = target_px - current_width
    if abs(delta) <= 0.5:
        return tile.copy()

    iterations = _math.ceil(abs(delta) / 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    if delta > 0:
        updated_mask = cv2.dilate(ink_mask, kernel, iterations=iterations)
    else:
        updated_mask = cv2.erode(ink_mask, kernel, iterations=iterations)

    return np.where(updated_mask > 0, 0, 255).astype(np.uint8)


def preprocess_for_pipeline(
    image: Union[np.ndarray, "Image.Image"],
    config: PreprocessConfig | None = None,
) -> np.ndarray:
    """
    Return a (512, 512) uint8 grayscale array consistent between train and inference.

    Accepts RGB, RGBA, L numpy arrays or PIL Images. RGBA alpha is composited onto white.

    Pipeline:
        1. Grayscale + alpha-composite onto white
        2. Resize longest-edge to target_size + white-pad to square
        3. Contrast normalise (1st–99th percentile → [0, 255])
        4. Stroke-width normalise to stroke_target_px via distance-transform morphology
    """
    if config is None:
        config = PreprocessConfig()

    # Step 1: grayscale + alpha-composite onto white
    if isinstance(image, Image.Image):
        pil = image
    else:
        pil = Image.fromarray(image)

    if pil.mode == "RGBA":
        arr = np.asarray(pil, dtype=np.uint8)
        alpha = arr[..., 3:4].astype(np.float32) / 255.0
        rgb = arr[..., :3].astype(np.float32)
        white = np.full_like(rgb, 255.0)
        blended = (rgb * alpha + white * (1.0 - alpha)).astype(np.uint8)
        pil = Image.fromarray(blended).convert("L")
    else:
        pil = pil.convert("L")

    gray = np.asarray(pil, dtype=np.uint8)

    # Step 2: resize longest-edge-to-target + white pad
    gray = resize_and_pad_to_square(gray, config.target_size)

    # Step 3: contrast normalise
    gray = contrast_normalise(gray, config.contrast_clip_low, config.contrast_clip_high)

    # Step 4: stroke-width normalise — opt-in only (stroke_target_px > 0)
    if config.stroke_target_px > 0:
        mask = ink_mask(gray, config.ink_threshold)
        gray = normalise_stroke_width(gray, mask, config.stroke_target_px)

    return gray
