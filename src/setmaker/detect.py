"""Detection bridge for the set-maker (WS-B).

Owns one job: turn a 512x512 preprocessed grayscale array into a
``List[Detection]`` for the set-maker prefill. It is a thin, single-purpose
wrapper over the canonical public detector ``src.inference.run.detect_boxes``
so detection logic lives in exactly one place (DRY): the active YOLO weights are
run under the ``[yolo.inference]`` NMS config (iou / agnostic_nms / max_det) and,
when no YOLO weights are present, the same connected-components fallback the full
inference pipeline uses applies.

This module does NOT own the GNN or JSON assembly. The set-maker takes fine
labels from the known target scene and needs detection only for box geometry, so
the classifier is never loaded here (``src.inference.run.detect_boxes`` loads the
YOLO stage only).

Zero detections are returned as an empty list, never ``None`` and never an
exception: an empty drawing, an all-white canvas, or weights that fire no boxes
all yield ``[]`` so the matcher and the ``/api/detect`` route can respond with an
empty draft plus a UI notice rather than failing (see the plan Error handling
section).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from ..inference.run import detect_boxes as _detect_boxes
from ..parsing.detection import Detection

# Re-export so ``from src.setmaker.detect import Detection`` works for callers
# (matcher, app) that build drafts from the detection contract.
__all__ = ["Detection", "detect_boxes"]

# Default YOLO confidence for set-maker prefill. Mirrors the canonical
# ``detect_boxes`` default (0.05): real handwriting is systematically
# under-confident versus synthetic input, so a low floor keeps faint strokes.
DEFAULT_CONF: float = 0.05

# Expected preprocessed-canvas shape; the underlying detector enforces this and
# raises ValueError on a mismatch. Named here for the matching test assertion.
EXPECTED_SHAPE = (512, 512)


def detect_boxes(
    gray: np.ndarray,
    *,
    project_root: Optional[Path] = None,
    conf: float = DEFAULT_CONF,
    tta: bool = True,
) -> List[Detection]:
    """Detect symbol boxes on a preprocessed image for the set-maker.

    Thin pass-through to ``src.inference.run.detect_boxes`` (single source of
    truth for detection): runs the active YOLO weights under the
    ``[yolo.inference]`` NMS config and returns the boxes with their coarse label
    and confidence, falling back to connected components when no YOLO weights are
    present. The GNN is not used; the set-maker supplies fine labels from the
    known target and needs detection for geometry only.

    Args:
        gray: Preprocessed ``(512, 512)`` uint8 grayscale array (the shape every
            set-maker image carries after ``preprocess_for_pipeline``).
        project_root: Project root for resolving the active weights and config;
            resolved from the inference module's location when omitted.
        conf: YOLO confidence threshold. Defaults low (``0.05``) because real
            handwriting is systematically under-confident versus synthetic input.
        tta: Enables test-time augmentation (horizontal flip + multi-scale),
            recommended for out-of-distribution real handwriting.

    Returns:
        A ``List[Detection]`` in 512 px space. Empty (``[]``) when nothing is
        detected; never ``None``.

    Raises:
        ValueError: If ``gray`` is not ``(512, 512)`` (propagated from the
            underlying detector so callers that skip preprocessing fail loudly).
    """
    detections = _detect_boxes(gray, project_root=project_root, conf=conf, tta=tta)
    # Contract guard: the bridge always hands the matcher a list. The underlying
    # detector already guarantees this, so this only documents and pins the
    # zero-detection-is-graceful behaviour the set-maker relies on.
    return list(detections)
