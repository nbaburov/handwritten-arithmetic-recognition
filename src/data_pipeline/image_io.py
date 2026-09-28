from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_png_grayscale_uint8(path: Path) -> np.ndarray:
    """
    Load a PNG and return a (H, W, 1) uint8 array.

    This intentionally mirrors the existing behavior of the pipeline:
    - Grayscale input stays grayscale
    - RGB/RGBA is converted to L (single channel)
    """

    with Image.open(path) as img:
        # Ensure a single grayscale channel deterministically.
        img_l = img.convert("L")
        arr2d = np.asarray(img_l, dtype=np.uint8)

    # Expand channel dim to keep downstream shape stable: (H, W, 1)
    return np.expand_dims(arr2d, axis=-1)



def normalize_to_float32(image_uint8: "np.ndarray") -> "np.ndarray":
    """Normalize uint8 array to float32 [0, 1]."""
    if image_uint8.dtype != "uint8":
        image_uint8 = image_uint8.astype("uint8")
    return image_uint8.astype("float32") / 255.0


def flatten_label_indices(indices: "list[int]") -> "np.ndarray":
    """Convert a list of integer label indices to a numpy int64 array."""
    import numpy as _np
    return _np.array(indices, dtype=_np.int64)
