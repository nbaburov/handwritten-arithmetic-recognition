"""Inference pipeline: YOLO detection → GNN classification → JSON assembly."""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# Pipeline version stamped into inference payloads (consumed by the GUI).
_INFERENCE_VERSION = "73-carryassembler"

from ..core.artifact_paths import resolve_gnn_best_pt
from ..core.config import DataPrepConfig, MAX_NODES
from ..core.ontology import yolo_class_id_to_name
from ..core.run_config import (
    CvFusionConfig,
    YoloInferenceConfig,
    load_config,
    load_cv_fusion_config,
    load_parsing_config,
)
from ..data_pipeline.preprocessing import preprocess_for_pipeline
from .given_prior import GivenNode, compose_into_scene, merge_given, pin_given_labels
from ..parsing.assemble import assemble_json
from ..parsing.detection import Detection


# ── Image helpers ─────────────────────────────────────────────────────────────

def _load_image_gray(path: Path) -> np.ndarray:
    return preprocess_for_pipeline(Image.open(path))


# ── YOLO detection ────────────────────────────────────────────────────────────

def _run_yolo(
    gray: np.ndarray,
    yolo_model: Any,
    conf: float,
    tta: bool = False,
    *,
    iou: float = 0.3,
    agnostic_nms: bool = False,
    max_det: int = 80,
) -> List[Detection]:
    """Run YOLO on a grayscale image and return detections.

    Args:
        gray: Grayscale image array (512, 512) uint8.
        yolo_model: YOLO model instance.
        conf: Confidence threshold for YOLO predictions.
        tta: If True, enables ultralytics test-time augmentation (horizontal
            flip + multi-scale passes merged before NMS), approximately 2-3x
            slower; useful for out-of-distribution (real handwriting) input.
        iou: NMS IoU threshold; lower values enforce stricter deduplication.
        agnostic_nms: If True, collapses cross-class duplicates of the same stroke.
        max_det: Maximum detections per image; aligns with MAX_NODES.

    Returns:
        List of Detection objects with x0, y0, x1, y1 clipped to image bounds.
    """
    image_rgb = np.stack([gray, gray, gray], axis=-1)
    predict_kwargs: Dict[str, Any] = {
        "conf": conf,
        "verbose": False,
        "iou": iou,
        "agnostic_nms": agnostic_nms,
        "max_det": max_det,
    }
    if tta:
        predict_kwargs["augment"] = True
    results = yolo_model.predict(image_rgb, **predict_kwargs)
    if not results:
        return []
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return []
    xyxy = r0.boxes.xyxy.cpu().numpy()
    cls_ids = r0.boxes.cls.cpu().numpy().astype(int)
    confs = r0.boxes.conf.cpu().numpy()
    detections: List[Detection] = []
    h, w = gray.shape
    for i in range(len(xyxy)):
        x0, y0, x1, y1 = xyxy[i].tolist()
        cls_id = int(cls_ids[i])
        label = yolo_class_id_to_name(cls_id) or "digit_main"
        det_conf = float(confs[i])
        detections.append(Detection(
            label=label,
            confidence=det_conf,
            x0=max(0.0, float(x0)),
            y0=max(0.0, float(y0)),
            x1=min(float(w), float(x1)),
            y1=min(float(h), float(y1)),
        ))
    return detections


# ── Fallback: connected components ────────────────────────────────────────────

def _connected_components(binary: np.ndarray, min_pixels: int = 30) -> List[Tuple[int, int, int, int]]:
    """BFS flood-fill to find bounding boxes of connected ink regions."""
    from collections import deque
    h, w = binary.shape
    seen = np.zeros_like(binary, dtype=bool)
    boxes: List[Tuple[int, int, int, int]] = []
    for y in range(h):
        for x in range(w):
            if seen[y, x] or not binary[y, x]:
                continue
            q = deque([(x, y)])
            seen[y, x] = True
            xs, ys = [x], [y]
            while q:
                cx, cy = q.popleft()
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and not seen[ny, nx] and binary[ny, nx]:
                        seen[ny, nx] = True
                        xs.append(nx)
                        ys.append(ny)
                        q.append((nx, ny))
            if len(xs) >= min_pixels:
                boxes.append((min(xs), min(ys), max(xs) + 1, max(ys) + 1))
    return boxes


def _merge_nearby_boxes(
    boxes: List[Tuple[int, int, int, int]],
    gap: int = 14,
) -> List[Tuple[int, int, int, int]]:
    """Merge boxes whose bounding rectangles overlap when expanded by gap px."""
    if not boxes:
        return boxes
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        out: List[Tuple[int, int, int, int]] = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            ax0, ay0, ax1, ay1 = merged[i]
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                bx0, by0, bx1, by1 = merged[j]
                if bx0 - gap <= ax1 and bx1 + gap >= ax0 and \
                        by0 - gap <= ay1 and by1 + gap >= ay0:
                    ax0, ay0 = min(ax0, bx0), min(ay0, by0)
                    ax1, ay1 = max(ax1, bx1), max(ay1, by1)
                    used[j] = True
                    changed = True
            out.append((ax0, ay0, ax1, ay1))
            used[i] = True
        merged = out
    return merged


def _fallback_detections(gray: np.ndarray) -> List[Detection]:
    """Connected-components fallback when YOLO is unavailable."""
    binary = gray < 200
    raw = _connected_components(binary)
    boxes = _merge_nearby_boxes(raw, gap=14)
    return [
        Detection(label="digit_main", confidence=1.0,
                  x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1))
        for x0, y0, x1, y1 in boxes
    ]


# ── Public detection bridge (set-maker) ───────────────────────────────────────

# Module-level cache of the active YOLO model + resolved [yolo.inference] config
# so repeated detect_boxes calls (one per drawn scene in the set-maker) do not
# reload weights. Keyed by project root. Only populated once a model actually
# loads: when no YOLO weights exist yet we intentionally skip caching so that a
# long-lived server picks up a weights file added after start-up (the next call
# re-checks the filesystem). Until then detect_boxes uses the connected-
# components fallback.
_DETECT_BRIDGE_CACHE: Dict[str, "Tuple[Any, YoloInferenceConfig]"] = {}


def _detect_bridge_resources(
    project_root: Path,
) -> "Tuple[Optional[Any], YoloInferenceConfig]":
    """Resolve the active YOLO model + [yolo.inference] config (cached once loaded).

    Loads only the YOLO stage (no GNN): the set-maker prefill uses detection for
    pixel geometry and takes labels from the known target, so the classifier is
    never needed. Returns ``(None, cfg)`` when no YOLO weights are available so
    the caller falls back to connected components.

    The loaded model is cached per project root, but the ``None`` (no-weights)
    case is deliberately left uncached: re-resolving the config is cheap and it
    lets a weights file added while a server is running be detected on the next
    call instead of being shadowed by a stale cached ``None``.
    """
    from ..core.artifact_paths import find_yolo_best_pt  # noqa: PLC0415

    cache_key = str(project_root)
    cached = _DETECT_BRIDGE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    config = DataPrepConfig.from_project_root(project_root)
    _, yolo_cfg, _ = load_config(project_root)
    inference_cfg = yolo_cfg.inference
    yolo_path = find_yolo_best_pt(config)
    if yolo_path is None or not yolo_path.exists():
        # No weights yet: do not cache, so a later-added file is picked up.
        return (None, inference_cfg)

    from ultralytics import YOLO  # type: ignore  # noqa: PLC0415
    model = YOLO(str(yolo_path))
    value = (model, inference_cfg)
    _DETECT_BRIDGE_CACHE[cache_key] = value
    return value


def detect_boxes(
    gray: np.ndarray,
    *,
    project_root: Optional[Path] = None,
    conf: float = 0.05,
    tta: bool = True,
) -> List[Detection]:
    """Detect symbol boxes on a preprocessed image, returning ``List[Detection]``.

    Public wrapper over ``_run_yolo`` (DRY): runs the active YOLO weights under the
    ``[yolo.inference]`` config (NMS iou / agnostic_nms / max_det) and returns the
    detected boxes with their coarse label and confidence. The GNN is not used;
    the set-maker takes fine labels from the known target and needs detection only
    for geometry. When no YOLO weights are present, falls back to connected
    components (same fallback the full pipeline uses).

    Args:
        gray: Preprocessed (512, 512) uint8 grayscale array.
        project_root: Project root for resolving active weights + config; resolved
            from this file's location when omitted.
        conf: YOLO confidence threshold. Defaults low (0.05) because real
            handwriting is systematically under-confident versus synthetic input.
        tta: Enables test-time augmentation (flip + multi-scale), recommended for
            out-of-distribution real handwriting.

    Returns:
        A list of ``Detection`` objects in 512 px space.
    """
    from ..core.config import resolve_project_root  # noqa: PLC0415

    if gray.shape != (512, 512):
        raise ValueError(
            f"detect_boxes expects a (512, 512) preprocessed grayscale image; got {gray.shape}."
        )
    root = project_root if project_root is not None else resolve_project_root(Path(__file__))
    model, inference_cfg = _detect_bridge_resources(root)
    if model is None:
        return _fallback_detections(gray)
    return _run_yolo(
        gray,
        model,
        conf=conf,
        tta=tta,
        iou=inference_cfg.iou,
        agnostic_nms=inference_cfg.agnostic_nms,
        max_det=inference_cfg.max_det,
    )


# ── Main inference function ───────────────────────────────────────────────────

def _flatten_payload_tokens(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    tokens: List[Dict[str, Any]] = []
    for row in payload.get("rows", []):
        tokens.extend(row.get("tokens", []))
    return tokens


def _maybe_fuse_cv(
    gray: np.ndarray,
    yolo_dets: List[Detection],
    yolo_model: Any,
    cfg: CvFusionConfig,
) -> "tuple[List[Detection], Optional[Dict[str, Any]]]":
    """Apply the CV-fusion mode/gate, then fuse the classical-OpenCV detector's
    extras into YOLO's detections when warranted.

    mode "auto": skip the CV branch when YOLO already looks complete
    (>= gate_min_detections detections AND mean confidence >= gate_mean_conf),
    going straight to the GNN with yolo_dets alone.
    mode "on": always fuse.

    Returns ``(detections, info)``: the detection list to feed the GNN, plus
    observability metadata for the GUI (None when the branch was unavailable).
    Degrade-safe: if cv2 / the cv_fusion module is unavailable, YOLO's
    detections are returned unchanged. (mode "off" is handled by the caller.)
    """
    if cfg.mode == "auto":
        n = len(yolo_dets)
        mean_conf = sum(d.confidence for d in yolo_dets) / n if n else 0.0
        if n >= cfg.gate_min_detections and mean_conf >= cfg.gate_mean_conf:
            return yolo_dets, {
                "mode": "auto", "skipped": "gate_complete",
                "cv_boxes": 0, "added": 0, "detections": [],
            }
    try:
        from .cv_fusion import run_cv_fusion  # noqa: PLC0415
    except ImportError:
        logging.getLogger(__name__).warning(
            "cv_fusion unavailable (cv2 import failed); skipping CV branch"
        )
        return yolo_dets, None
    return run_cv_fusion(gray, yolo_dets, yolo_model, cfg)


def run_inference_with_models(
    gray: np.ndarray,
    *,
    gnn_model: Any,
    yolo_model: Optional[Any] = None,
    yolo_conf: float = 0.25,
    tta: bool = False,
    iou: float = 0.3,
    agnostic_nms: bool = False,
    max_det: int = 80,
    heuristic_enabled: bool = True,
    cv_fusion: Optional[CvFusionConfig] = None,
    prior: Optional[Sequence[GivenNode]] = None,
) -> Dict[str, Any]:
    """
    Full pipeline: YOLO (or fallback) → GNN → assemble_json.
    Returns the assembled payload dict plus timing and debug metadata.

    Args:
        tta: If True, enables ultralytics test-time augmentation (horizontal
            flip + multi-scale), approximately 2-3x slower; useful for
            out-of-distribution (real handwriting) input.
        iou: NMS IoU threshold forwarded to YOLO predict.
        agnostic_nms: If True, collapses cross-class duplicates of the same stroke.
        max_det: Maximum detections per image forwarded to YOLO predict.
        cv_fusion: When provided (and mode != "off"), fuses a classical-OpenCV
            detector's extras into YOLO's detections (catches symbols YOLO misses
            on unfinished scenes). Only runs when YOLO produced >= 1 detection;
            None disables the branch entirely (baseline behavior).
        prior: When provided, a sequence of GivenNode entries representing the
            pre-printed equation scaffold. YOLO runs on the original child image
            only (no double-detection of given tiles); given nodes are merged with
            child detections, painted into the graph-crop image, and their fine
            labels are pinned before assembly. When None, behaviour is byte-for-byte
            identical to the prior=None path (regression-critical).
    """
    # Optional: flatten a dark photo surround to white so the WHOLE pipeline
    # (YOLO, the CV branch, and the GNN's crops) sees an in-distribution white scene.
    if cv_fusion is not None and cv_fusion.flatten_background:
        try:
            from .cv_fusion import flatten_to_white_paper  # noqa: PLC0415
            gray = flatten_to_white_paper(gray)
        except ImportError:
            logging.getLogger(__name__).warning(
                "cv_fusion unavailable (cv2 import failed); skipping flatten_background"
            )

    # Stage 1: detect symbols
    cv_fusion_info: Optional[Dict[str, Any]] = None
    t0 = time.perf_counter()
    if yolo_model is not None:
        detections = _run_yolo(
            gray, yolo_model, yolo_conf, tta=tta,
            iou=iou, agnostic_nms=agnostic_nms, max_det=max_det,
        )
        used_yolo = True
        degraded = len(detections) == 0
        quality_reason = "yolo_empty" if degraded else None
        if degraded:
            detections = _fallback_detections(gray)
        elif cv_fusion is not None and cv_fusion.mode != "off":
            # Augment YOLO with the CV detector on regions YOLO did not claim.
            # Always use the preprocessed 512x512 gray — CV detections must share
            # the same coordinate frame as YOLO (both 512-space) so merged boxes
            # feed build_graph correctly. (C-1 fix: original_gray removed.)
            t_cv = time.perf_counter()
            detections, cv_fusion_info = _maybe_fuse_cv(gray, detections, yolo_model, cv_fusion)
            if cv_fusion_info is not None:
                cv_fusion_info["cv_ms"] = round((time.perf_counter() - t_cv) * 1000.0, 2)
    else:
        detections = _fallback_detections(gray)
        used_yolo = False
        degraded = True
        quality_reason = "no_yolo_model"
    yolo_ms = (time.perf_counter() - t0) * 1000.0

    # Given-equation prior merge: runs AFTER YOLO on the original child image so
    # given tiles are never double-detected. When prior is None this block is a
    # no-op and detections/gray are forwarded unchanged (regression-safe).
    gray_for_graph = gray
    known_fine: dict[tuple[int, int, int, int], str] = {}
    if prior is not None:
        mr = merge_given(detections, prior, MAX_NODES)
        detections = mr.detections
        gray_for_graph = compose_into_scene(gray, mr.compose_nodes)
        known_fine = mr.known_fine

    # Stage 2: GNN classification
    t1 = time.perf_counter()
    if detections:
        from ..modeling.graph_builder import build_graph  # noqa: PLC0415
        data = build_graph(detections, gray_for_graph)
        truncated = bool(getattr(data, 'truncated', False))
        node_preds, equation_type, eq_type_logits = gnn_model.predict(data)
        if prior is not None and known_fine:
            pin_given_labels(node_preds, known_fine)
    else:
        truncated = False
        node_preds, equation_type, eq_type_logits = [], "unknown", None
    stage2_ms = (time.perf_counter() - t1) * 1000.0

    # Assemble
    t2 = time.perf_counter()
    payload = assemble_json(
        node_preds, equation_type, eq_type_logits=eq_type_logits,
        heuristic_enabled=heuristic_enabled,
    )
    assemble_ms = (time.perf_counter() - t2) * 1000.0

    payload["inference_backend"] = "yolo+gnn" if used_yolo else "components+gnn"
    payload["quality"] = {"degraded": degraded, "reason": quality_reason}
    payload["truncated"] = truncated
    if truncated:
        payload["max_nodes_cap"] = MAX_NODES
    payload["timings"] = {
        "yolo_ms": round(yolo_ms, 2),
        "stage2_ms": round(stage2_ms, 2),
        "assemble_ms": round(assemble_ms, 3),
    }
    if cv_fusion_info is not None:
        payload["cv_fusion"] = cv_fusion_info
    payload["_version"] = _INFERENCE_VERSION  # Debug: verify code updates
    payload["tokens"] = _flatten_payload_tokens(payload)
    return payload


# ── Model scanning ────────────────────────────────────────────────────────────

def _load_run_labels(project_root: Path, stage: str) -> dict:
    """Human run labels from artifacts/run_labels.json (shared with docs/runs.md)."""
    f = project_root / "artifacts" / "run_labels.json"
    if not f.exists():
        return {}
    try:
        import json
        return json.loads(f.read_text()).get(stage, {})
    except Exception:
        return {}


def _tag_choice(name: str, parent: str, labels: dict) -> str:
    """Build a picker choice 'name (parent)  —  label'. Label is paren-free so the
    GUI resolver can still extract `parent` from the first parentheses group."""
    run_id = parent.rsplit("/", 1)[-1]
    label = labels.get(run_id)
    base = f"{name} ({parent})"
    return f"{base}  —  {label}" if label else base


def scan_yolo_models(project_root: Path) -> List[str]:
    d = project_root / "artifacts" / "yolo"
    if not d.exists():
        return []
    labels = _load_run_labels(project_root, "yolo")
    results: List[str] = []
    for p in sorted(d.glob("*.pt")):
        results.append(_tag_choice(p.name, p.parent.name, labels))
    # Per-run weights: list only each run's root best.pt (skip nested yolo/weights/ dups).
    for p in sorted(d.glob("runs/*/best.pt")):
        rel = p.relative_to(d)
        results.append(_tag_choice(p.name, str(rel.parent), labels))
    return results


def scan_stage1_models(project_root: Path) -> List[str]:
    """Scan artifacts/stage1/ for YOLO .pt weight files."""
    d = project_root / "artifacts" / "stage1"
    if not d.exists():
        return []
    return [f"{p.name} ({p.parent.name})" for p in sorted(d.rglob("*.pt"))]


def scan_stage2_models(project_root: Path) -> List[str]:
    d = project_root / "artifacts" / "gnn"
    if not d.exists():
        return []
    labels = _load_run_labels(project_root, "gnn")
    results: List[str] = []
    for p in sorted(d.glob("*.pt")):
        results.append(_tag_choice(p.name, p.parent.name, labels))
    for p in sorted(d.rglob("runs/**/*.pt")):
        rel = p.relative_to(d)
        results.append(_tag_choice(p.name, str(rel.parent), labels))
    return results


# ── InferenceSession ──────────────────────────────────────────────────────────

class InferenceSession:
    """Holds loaded YOLO + GNN models and runs inference."""

    def __init__(self) -> None:
        self._gnn_model: Optional[Any] = None
        self._yolo_model: Optional[Any] = None
        self._pipeline_name: Optional[str] = None
        self._pipeline_display: Optional[str] = None
        self._yolo_inference_config: YoloInferenceConfig = YoloInferenceConfig()
        self._assembler_heuristic_enabled: bool = True
        self._cv_fusion_config: CvFusionConfig = CvFusionConfig()

    def is_loaded(self) -> bool:
        return self._gnn_model is not None

    @property
    def pipeline_name(self) -> Optional[str]:
        return self._pipeline_name

    @property
    def pipeline_display(self) -> Optional[str]:
        return self._pipeline_display

    def load(self, *, gnn_weights: Path, yolo_weights: Optional[Path] = None) -> None:
        """Load GNN weights (required) and optional YOLO weights."""
        import torch  # noqa: PLC0415
        from ..modeling.gnn import CropBackbone, SymbolGNN  # noqa: PLC0415

        backbone = CropBackbone()
        model = SymbolGNN(backbone)
        state = torch.load(gnn_weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        self._gnn_model = model

        self._yolo_model = None
        if yolo_weights is not None and yolo_weights.exists():
            from ultralytics import YOLO  # type: ignore  # noqa: PLC0415
            self._yolo_model = YOLO(str(yolo_weights))

        has_yolo = self._yolo_model is not None
        self._pipeline_name = ("yolo" if has_yolo else "components") + "+gnn"
        self._pipeline_display = ("YOLO" if has_yolo else "Components") + " + GNN"

    def auto_load(self, project_root: Path) -> None:
        """Load best available models from artifacts/ and resolve YoloInferenceConfig from config.toml."""
        from ..core.artifact_paths import find_yolo_best_pt  # noqa: PLC0415
        config = DataPrepConfig.from_project_root(project_root)
        _, yolo_cfg, _ = load_config(project_root)
        self._yolo_inference_config = yolo_cfg.inference
        parsing_cfg = load_parsing_config(project_root)
        self._assembler_heuristic_enabled = parsing_cfg.assembler_heuristic_enabled
        self._cv_fusion_config = load_cv_fusion_config(project_root)
        try:
            gnn_path = resolve_gnn_best_pt(config)
        except FileNotFoundError:
            return  # no GNN weights available
        yolo_path = find_yolo_best_pt(config)
        self.load(gnn_weights=gnn_path, yolo_weights=yolo_path)

    def predict(
        self,
        image: np.ndarray,
        yolo_conf: float = 0.25,
        tta: bool = False,
        iou: Optional[float] = None,
        agnostic_nms: Optional[bool] = None,
        max_det: Optional[int] = None,
        cv_fusion: Optional[CvFusionConfig] = None,
        *,
        prior: Optional[Sequence[GivenNode]] = None,
    ) -> Dict[str, Any]:
        """Run the full inference pipeline on a preprocessed grayscale image.

        Args:
            image: Preprocessed (512, 512) uint8 grayscale array.
            yolo_conf: YOLO confidence threshold.
            tta: If True, enables ultralytics test-time augmentation (horizontal
                flip + multi-scale), approximately 2-3x slower; useful for
                out-of-distribution (real handwriting) input.
            iou: NMS IoU threshold override; falls back to session YoloInferenceConfig when None.
            agnostic_nms: Cross-class NMS override; falls back to session YoloInferenceConfig when None.
            max_det: Max detections override; falls back to session YoloInferenceConfig when None.
            cv_fusion: CvFusionConfig override; falls back to the session config
                (loaded from config.toml [inference.cv_fusion]) when None.
            prior: Optional given-equation prior (keyword-only). When provided,
                merge_given/compose_into_scene/pin_given_labels are applied before
                assembly. When None, behaviour is identical to today (regression-safe).
        """
        if not self.is_loaded():
            raise RuntimeError("No models loaded. Call load() or auto_load() first.")
        resolved_iou: float = iou if iou is not None else self._yolo_inference_config.iou
        resolved_agnostic_nms: bool = (
            agnostic_nms if agnostic_nms is not None else self._yolo_inference_config.agnostic_nms
        )
        resolved_max_det: int = (
            max_det if max_det is not None else self._yolo_inference_config.max_det
        )
        resolved_cv_fusion: CvFusionConfig = (
            cv_fusion if cv_fusion is not None else self._cv_fusion_config
        )
        return run_inference_with_models(
            image,
            gnn_model=self._gnn_model,
            yolo_model=self._yolo_model,
            yolo_conf=yolo_conf,
            tta=tta,
            iou=resolved_iou,
            agnostic_nms=resolved_agnostic_nms,
            max_det=resolved_max_det,
            heuristic_enabled=self._assembler_heuristic_enabled,
            cv_fusion=resolved_cv_fusion,
            prior=prior,
        )


# ── CLI entry point ───────────────────────────────────────────────────────────

def run_inference(
    image_path: Path,
    project_root: Path,
    yolo_conf: float = 0.25,
    tta: bool = False,
    iou: Optional[float] = None,
    agnostic_nms: Optional[bool] = None,
    max_det: Optional[int] = None,
    cv_fusion: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full inference pipeline on a single image file.

    Args:
        image_path: Path to the input PNG image.
        project_root: Project root directory for model resolution.
        yolo_conf: YOLO confidence threshold.
        tta: If True, enables ultralytics test-time augmentation (horizontal
            flip + multi-scale), approximately 2-3x slower; useful for
            out-of-distribution (real handwriting) input.
        iou: NMS IoU threshold override; falls back to config.toml when None.
        agnostic_nms: Cross-class NMS override; falls back to config.toml when None.
        max_det: Max detections override; falls back to config.toml when None.
        cv_fusion: Optional "on"|"auto"|"off" override of the config.toml
            [inference.cv_fusion] mode for this run; None uses config.toml.
    """
    session = InferenceSession()
    session.auto_load(project_root)
    gray = _load_image_gray(image_path)
    cv_fusion_cfg: Optional[CvFusionConfig] = None
    if cv_fusion is not None:
        from dataclasses import replace  # noqa: PLC0415
        mode = cv_fusion.lower()
        if mode not in {"on", "auto", "off"}:
            raise ValueError(f"cv_fusion must be one of on/auto/off, got {cv_fusion!r}")
        cv_fusion_cfg = replace(session._cv_fusion_config, mode=mode)
    return session.predict(
        gray, yolo_conf=yolo_conf, tta=tta,
        iou=iou, agnostic_nms=agnostic_nms, max_det=max_det,
        cv_fusion=cv_fusion_cfg,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run equation inference pipeline.")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--yolo-conf", type=float, default=0.25)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument(
        "--cv-fusion", choices=["on", "auto", "off"], default=None,
        help="Override config.toml [inference.cv_fusion] mode for this run "
             "(default: use config.toml, which ships as 'on').",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = run_inference(
        args.image, project_root=args.project_root, yolo_conf=args.yolo_conf,
        cv_fusion=args.cv_fusion,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
