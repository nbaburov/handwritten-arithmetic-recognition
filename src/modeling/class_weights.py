from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def balanced_class_weights_sparse(y: np.ndarray, num_classes: int) -> Dict[int, float]:
    y_flat = np.asarray(y, dtype=np.int64).ravel()
    present = np.unique(y_flat)
    weights = compute_class_weight(class_weight="balanced", classes=present, y=y_flat)
    out: Dict[int, float] = {int(c): 1.0 for c in range(num_classes)}
    for c, w in zip(present, weights):
        out[int(c)] = float(w)
    return out
