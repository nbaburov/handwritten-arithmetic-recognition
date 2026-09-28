from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import gradio as gr
import numpy as np

from ..data_pipeline.preprocessing import preprocess_for_pipeline
from ..parsing.equation_display import format_parsed_equation_readable
from .annotation import (
    build_annotations,
    build_cv_annotations,
    build_detection_annotations,
    build_gt_annotations,
    build_yolo_annotations,
)
from .gt_loader import GtSymbol, compute_scene_scores, load_scene, match_predictions
from .gui_components import (
    render_cv_panel,
    render_result_panel,
    render_score_bar,
    render_yolo_panel,
    render_stage2_panel,
    render_status_bar,
)
from .run import (
    InferenceSession,
    scan_yolo_models,
    scan_stage2_models,
)
from .showcase import build_input_page, build_results_page
from .showcase_store import ShowcaseStore

_CSS_PATH = Path(__file__).parent / "static" / "gui.css"
_CSS = _CSS_PATH.read_text(encoding="utf-8") if _CSS_PATH.exists() else ""

_EMPTY_YOLO = render_yolo_panel([], 0.0, "yolo")
_EMPTY_STAGE2 = render_stage2_panel([], 0.0, "GNN")
_EMPTY_RESULT = render_result_panel(None, None, 0.0)
_EMPTY_CV = render_cv_panel([], 0.0)


def _placeholder_image() -> np.ndarray:
    return np.full((384, 512, 3), 248, dtype=np.uint8)


def _extract_gray(sketchpad_data: Any) -> Optional[np.ndarray]:
    """
    Extract a grayscale uint8 array from a Gradio Sketchpad payload.
    Sketchpad returns a dict with keys 'background', 'layers', 'composite'.
    We use 'composite' (merged result) falling back to 'layers[0]'.
    Ink is dark on white, so we invert if the canvas has a transparent/dark background.
    """
    if sketchpad_data is None:
        return None
    from PIL import Image as _Image

    def _to_gray(v: Any) -> Optional[np.ndarray]:
        if v is None:
            return None
        if isinstance(v, _Image.Image):
            arr = np.asarray(v.convert("RGBA"), dtype=np.uint8)
        elif isinstance(v, np.ndarray):
            arr = v
        else:
            return None
        if arr.ndim == 2:
            return arr.astype(np.uint8)
        if arr.ndim == 3:
            if arr.shape[2] == 4:
                # RGBA: composite alpha onto white background
                alpha = arr[..., 3:4].astype(np.float32) / 255.0
                rgb = arr[..., :3].astype(np.float32)
                white = np.full_like(rgb, 255.0)
                blended = (rgb * alpha + white * (1 - alpha)).astype(np.uint8)
                return np.array(_Image.fromarray(blended).convert("L"), dtype=np.uint8)
            return np.array(_Image.fromarray(arr).convert("L"), dtype=np.uint8)
        return None

    if isinstance(sketchpad_data, dict):
        # Try composite first, then layers[0], then background
        for key in ("composite", "image"):
            gray = _to_gray(sketchpad_data.get(key))
            if gray is not None:
                return preprocess_for_pipeline(gray)
        layers = sketchpad_data.get("layers")
        if layers:
            gray = _to_gray(layers[0])
            if gray is not None:
                return preprocess_for_pipeline(gray)
        gray = _to_gray(sketchpad_data.get("background"))
        if gray is not None:
            return preprocess_for_pipeline(gray)
    return None


def _is_blank(gray: np.ndarray, min_dark_pixels: int = 100) -> bool:
    """Return True if there are fewer than min_dark_pixels dark pixels in the image."""
    return bool(int(np.sum(gray < 200)) < min_dark_pixels)


def _equation_kind_banner(equation_kind: Optional[str], ood_reason: Optional[str]) -> Optional[str]:
    """Return a human-readable banner string for bare_digits or OOD scenes, or None for normal scenes."""
    if equation_kind == "bare_digits":
        return "Detected bare digits (no equation structure)"
    if equation_kind == "unknown":
        if ood_reason == "true_empty":
            return "No detections — try a different scene"
        if ood_reason:
            return f"Out-of-distribution: {ood_reason.replace('_', ' ')}"
    return None


def _run_and_build_outputs(
    session: InferenceSession,
    gray: np.ndarray,
    conf: float,
    gt_symbols: Optional[list[GtSymbol]],
    show_row_col: bool = False,
) -> tuple:
    try:
        payload = session.predict(image=gray, yolo_conf=conf)
    except Exception as exc:
        err = render_result_panel(None, f"Inference error: {str(exc)[:200]}", 0.0)
        placeholder = gr.update(value=(_placeholder_image(), []), color_map={})
        return placeholder, placeholder, _EMPTY_YOLO, _EMPTY_STAGE2, err, placeholder, None, "", gr.update(visible=False), placeholder, _EMPTY_CV

    tokens: list[dict] = payload.get("tokens", [])
    timings: dict = payload.get("timings", {})
    backend: str = payload.get("inference_backend", "components+gnn")
    yolo_backend = "yolo" if "yolo" in backend else "components"
    pipeline_label = "GNN"

    yolo_annotations, yolo_color_map = build_yolo_annotations(tokens)
    yolo_update = gr.update(value=(gray, yolo_annotations), color_map=yolo_color_map)

    equation_kind = payload.get("equation_kind") or payload.get("equation_type")
    ood_reason = payload.get("ood_reason")
    raw_detections: list[dict] = payload.get("detections", [])

    # GNN panel: use detection annotations when show_row_col is forced,
    # or when equation_kind is bare_digits/unknown but detections exist.
    use_detection_ann = (
        show_row_col
        or (equation_kind in ("bare_digits", "unknown") and bool(raw_detections) and not tokens)
    )
    if use_detection_ann and raw_detections:
        gnn_annotations, gnn_color_map = build_detection_annotations(raw_detections)
    else:
        gnn_annotations, gnn_color_map = build_annotations(tokens)
    gnn_update = gr.update(value=(gray, gnn_annotations), color_map=gnn_color_map)

    # OpenCV-fusion extras (boxes YOLO missed). The info dict nests results per
    # path: "cv_detector" (classical CV) and "second_yolo" (second YOLO pass).
    # Combine both so the panel shows every recovered detection. Falls back to the
    # legacy flat layout (top-level "detections") for older payloads.
    cv_info: dict = payload.get("cv_fusion") or {}
    cv_detector_info: dict = cv_info.get("cv_detector", {})
    second_yolo_info: dict = cv_info.get("second_yolo", {})
    cv_dets: list[dict] = (
        cv_detector_info.get("detections", [])
        + second_yolo_info.get("detections", [])
    ) or cv_info.get("detections", [])  # legacy fallback
    cv_annotations, cv_color_map = build_cv_annotations(cv_dets)
    cv_update = gr.update(value=(gray, cv_annotations), color_map=cv_color_map)
    cv_panel = render_cv_panel(
        cv_dets,
        cv_info.get("cv_ms", 0.0),
        cv_boxes=cv_detector_info.get("cv_boxes", cv_info.get("cv_boxes", 0)),
        cv_fallback=cv_detector_info.get("cv_fallback", cv_info.get("cv_fallback", 0)),
        skipped=cv_info.get("skipped"),
    )

    s1 = render_yolo_panel(tokens, timings.get("yolo_ms", 0.0), yolo_backend)
    s2 = render_stage2_panel(tokens, timings.get("stage2_ms", 0.0), pipeline_label)

    # Show version for debugging cache issues
    version = payload.get("_version", "?")

    banner = _equation_kind_banner(equation_kind, ood_reason)
    try:
        display_text = format_parsed_equation_readable(payload)
    except Exception:
        display_text = str(payload.get("rows", ""))
    if banner:
        display_text = f"{banner}\n\n{display_text}" if display_text else banner
    # Add version indicator for cache debugging
    display_text = f"{display_text}\n\n[v{version}]" if display_text else f"[v{version}]"
    result = render_result_panel(equation_kind, display_text, timings.get("parse_ms", 0.0), tokens=tokens)

    if gt_symbols is not None:
        gt_annotations, gt_color_map = build_gt_annotations(tokens, gt_symbols)
        gt_update = gr.update(value=(gray, gt_annotations), color_map=gt_color_map)
        scores = compute_scene_scores(tokens, gt_symbols)
        matches = match_predictions(tokens, gt_symbols)
        table_rows, idx = [], 1
        for m in matches:
            if not m.is_false_positive:
                pred = m.predicted_label or "—"
                exp = m.gt.fine_label if m.gt else "—"
                table_rows.append([idx, pred, exp, "✓" if m.is_correct else "✗"])
                idx += 1
        for m in matches:
            if m.is_false_positive:
                table_rows.append([idx, m.predicted_label, "—", "✗"])
                idx += 1
        return yolo_update, gnn_update, s1, s2, result, gt_update, table_rows, render_score_bar(scores), gr.update(visible=True), cv_update, cv_panel

    return yolo_update, gnn_update, s1, s2, result, gr.update(), None, "", gr.update(visible=False), cv_update, cv_panel


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Handwritten Arithmetic Recognition")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()

    session = InferenceSession()
    session.auto_load(project_root)

    run_id = uuid.uuid4().hex[:8]
    store = ShowcaseStore(
        showcase_root=project_root / "data" / "generated" / "showcase" / run_id,
    )

    initial_status = render_status_bar(
        session.pipeline_name,
        session.pipeline_display,
        str(project_root / "artifacts" / "yolo" / "best.pt")
            if session.pipeline_name and "yolo" in (session.pipeline_name or "") else None,
        str(project_root / "artifacts" / "gnn" / "best.pt")
            if session.pipeline_name else None,
    )

    yolo_choices = scan_yolo_models(project_root)
    stage2_choices = scan_stage2_models(project_root)

    with gr.Blocks(title="Handwritten Arithmetic Recognition") as demo:

        with gr.Tabs():

            # ── Main tab ──────────────────────────────────────────────────
            with gr.Tab("Main"):

                gt_state = gr.State(None)

                # ── Header ────────────────────────────────────────────────
                gr.HTML(
                    '<div class="engine-header">'
                    '<h1>Handwritten Arithmetic Recognition</h1>'
                    '<p>Column arithmetic symbol detection and parsing</p>'
                    '</div>'
                )

                # ── Quick links to the standalone presenter pages ─────────
                gr.HTML(
                    '<div class="engine-quicklinks">'
                    '<a class="engine-quicklink" href="/input" target="_blank" '
                    'rel="noopener">Open Input page</a>'
                    '<a class="engine-quicklink" href="/results" target="_blank" '
                    'rel="noopener">Open Results page</a>'
                    '</div>'
                )

                # ── Status bar ────────────────────────────────────────────
                status_html = gr.HTML(value=initial_status)

                # ── Model loader (collapsed by default) ───────────────────
                with gr.Accordion("Change models", open=False):
                    with gr.Row():
                        yolo_dd = gr.Dropdown(
                            choices=yolo_choices,
                            label="Stage 1 — YOLO weights (.pt)",
                            value=yolo_choices[0] if yolo_choices else None,
                            scale=1,
                        )
                        stage2_dd = gr.Dropdown(
                            choices=stage2_choices,
                            label="Stage 2 — GNN weights (.pt)",
                            value=stage2_choices[0] if stage2_choices else None,
                            scale=1,
                        )
                        load_models_btn = gr.Button("Load", size="sm", scale=0, min_width=80)

                # ── OpenCV hyperparameters (live-tunable) ─────────────────
                with gr.Accordion("OpenCV hyperparameters", open=False):
                    _cv = session._cv_fusion_config
                    gr.Markdown("""
                    **Detection tuning:**
                    - **min_area**: Lower = detect tinier symbols (carries, dots). Default 150.
                    - **min_side**: Lower = allow thinner strokes (thin operators, "1"). Default 10.
                    - **pad**: Extra pixels around each detected box. Default 4.
                    - **merge_overlap_ratio**: Merge boxes that overlap this much. Default 0.5.
                    - **proximity_merge_factor**: Merge boxes this close together. Lower = merge more. Default 0.25.
                    - **mask_dilate_px**: Expand YOLO bbox mask before CV detector. Default 5.
                    """)
                    with gr.Row():
                        cv_mode_dd = gr.Dropdown(["on", "auto", "off"], value=_cv.mode, label="mode (on/auto/off)")
                        cv_mask_yolo = gr.Checkbox(value=_cv.mask_yolo, label="mask_yolo (ignore YOLO regions)")
                        cv_tight = gr.Checkbox(value=_cv.cv_tight_mask, label="cv_tight_mask (mask only YOLO ink)")
                    with gr.Row():
                        cv_min_area = gr.Slider(5, 500, value=_cv.min_area, step=5, label="min_area (px²)")
                        cv_min_side = gr.Slider(1, 50, value=_cv.min_side, step=1, label="min_side (px)")
                    with gr.Row():
                        cv_pad = gr.Slider(0, 20, value=_cv.pad, step=1, label="pad (px)")
                        cv_dilate = gr.Slider(0, 20, value=_cv.mask_dilate_px, step=1, label="mask_dilate_px (rect mask)")
                        cv_tight_dilate = gr.Slider(0, 20, value=_cv.tight_mask_dilate_px, step=1, label="tight_mask_dilate_px")
                    with gr.Row():
                        cv_merge_overlap = gr.Slider(0.0, 1.0, value=_cv.merge_overlap_ratio, step=0.05, label="merge_overlap_ratio (1.0=no merge)")
                        cv_proximity = gr.Slider(0.0, 0.5, value=_cv.proximity_merge_factor, step=0.05, label="proximity_merge_factor (0=no merge)")
                    gr.Markdown("""
                    **Classification & merging:**
                    - **classify_conf**: Min YOLO confidence for CV crop classification. Default 0.25.
                    - **nms_iou**: Overlap threshold when merging CV+YOLO detections. Lower = stricter. Default 0.3.
                    - **classify_with_yolo**: Reclassify CV boxes with YOLO (better accuracy, 1 extra YOLO call).
                    - **keep_unclassified**: Keep CV boxes YOLO can't classify (sends to GNN as fallback).
                    """)
                    with gr.Row():
                        cv_classify = gr.Slider(0.05, 0.95, value=_cv.classify_conf, step=0.05, label="classify_conf")
                        cv_nms = gr.Slider(0.10, 0.90, value=_cv.nms_iou, step=0.05, label="nms_iou")
                        cv_classify_yolo = gr.Checkbox(value=_cv.classify_with_yolo, label="classify_with_yolo")
                        cv_keep_unclass = gr.Checkbox(value=_cv.keep_unclassified, label="keep_unclassified")
                    with gr.Row():
                        cv_flatten = gr.Checkbox(value=_cv.flatten_background, label="flatten_background (phone photos)")

                    def _apply_cv(mode: str, mask_yolo: bool, tight: bool, min_area: float, min_side: float,
                                  pad: float, dilate: float, tight_dilate: float, merge_overlap: float, proximity: float,
                                  classify: float, nms: float, classify_yolo: bool, keep_unclass: bool,
                                  flatten: bool) -> None:
                        from dataclasses import replace  # noqa: PLC0415
                        session._cv_fusion_config = replace(
                            session._cv_fusion_config,
                            mode=mode, mask_yolo=bool(mask_yolo), cv_tight_mask=bool(tight),
                            min_area=int(min_area), min_side=int(min_side), pad=int(pad),
                            mask_dilate_px=int(dilate), tight_mask_dilate_px=int(tight_dilate),
                            merge_overlap_ratio=float(merge_overlap), proximity_merge_factor=float(proximity),
                            classify_conf=float(classify), nms_iou=float(nms),
                            classify_with_yolo=bool(classify_yolo), keep_unclassified=bool(keep_unclass),
                            flatten_background=bool(flatten),
                        )

                    _cv_inputs = [cv_mode_dd, cv_mask_yolo, cv_tight, cv_min_area, cv_min_side, cv_pad,
                                  cv_dilate, cv_tight_dilate, cv_merge_overlap, cv_proximity, cv_classify, cv_nms,
                                  cv_classify_yolo, cv_keep_unclass, cv_flatten]
                    for _c in _cv_inputs:
                        _c.change(_apply_cv, inputs=_cv_inputs, outputs=None)

                # ── Main row ──────────────────────────────────────────────
                with gr.Row(equal_height=True):
                    # Left: canvas controls
                    with gr.Column(scale=1):
                        sketchpad = gr.Sketchpad(
                            label="Draw here",
                            height=384,
                            brush=gr.Brush(default_size=3, colors=["#000000"], color_mode="fixed"),
                        )
                        with gr.Row():
                            recognize_btn = gr.Button("Recognize", variant="primary", size="sm")
                            clear_btn = gr.Button("Clear", variant="secondary", size="sm")
                        with gr.Row():
                            mode_radio = gr.Radio(
                                choices=["Manual", "Live"],
                                value="Manual",
                                label="Mode",
                                scale=1,
                            )
                            conf_slider = gr.Slider(
                                minimum=0.05, maximum=0.90, value=0.25, step=0.05,
                                label="Confidence",
                                scale=2,
                            )
                        file_input = gr.File(
                            label="Load image (auto-loads gt.json if present)",
                            file_types=[".png", ".jpg"],
                        )

                    # YOLO coarse detections
                    with gr.Column(scale=1):
                        yolo_image = gr.AnnotatedImage(
                            label="YOLO — coarse boxes",
                            value=(_placeholder_image(), []),
                            height=384,
                        )

                    # OpenCV fusion — extra symbols YOLO missed (between YOLO and GNN)
                    with gr.Column(scale=1):
                        cv_image = gr.AnnotatedImage(
                            label="OpenCV — extra boxes",
                            value=(_placeholder_image(), []),
                            height=384,
                        )

                    # GNN fine-label detections
                    with gr.Column(scale=1):
                        gnn_image = gr.AnnotatedImage(
                            label="GNN — fine labels",
                            value=(_placeholder_image(), []),
                            height=384,
                        )

                # ── Stage panels (YOLO → CV-fusion → GNN → Result) ────────
                with gr.Row():
                    with gr.Column(elem_classes=["stage-col"]):
                        yolo_html = gr.HTML(value=_EMPTY_YOLO)
                    with gr.Column(elem_classes=["stage-col"]):
                        cv_html = gr.HTML(value=_EMPTY_CV)
                    with gr.Column(elem_classes=["stage-col"]):
                        stage2_html = gr.HTML(value=render_stage2_panel([], 0.0, "GNN"))
                    with gr.Column(elem_classes=["stage-col"]):
                        result_html = gr.HTML(value=_EMPTY_RESULT)

                # ── GT comparison (hidden until gt.json scene loaded) ──────
                with gr.Group(visible=False) as gt_section:
                    gr.HTML('<div class="engine-gt-header">Ground Truth Comparison</div>')
                    with gr.Row():
                        with gr.Column():
                            gt_image = gr.AnnotatedImage(label="GT overlay")
                        with gr.Column():
                            diff_table = gr.DataFrame(
                                headers=["#", "Predicted", "Expected", "Match"],
                                label="Per-symbol diff",
                            )
                            score_html = gr.HTML()

                # ── Event handlers ────────────────────────────────────────

                _OUTPUTS = [
                    yolo_image, gnn_image, yolo_html, stage2_html, result_html,
                    gt_image, diff_table, score_html, gt_section,
                    cv_image, cv_html,
                ]

                def on_recognize(sketchpad_data: Any, conf: float, gt_syms: Any) -> tuple:
                    gray = _extract_gray(sketchpad_data)
                    if gray is None or _is_blank(gray):
                        msg = render_result_panel(None, "Draw something first.", 0.0)
                        ph = gr.update(value=(_placeholder_image(), []), color_map={})
                        return ph, ph, _EMPTY_YOLO, _EMPTY_STAGE2, msg, gr.update(), None, "", gr.update(visible=False), ph, _EMPTY_CV
                    if not session.is_loaded():
                        msg = render_result_panel(None, "No models loaded.", 0.0)
                        ph = gr.update(value=(_placeholder_image(), []), color_map={})
                        return ph, ph, _EMPTY_YOLO, _EMPTY_STAGE2, msg, gr.update(), None, "", gr.update(visible=False), ph, _EMPTY_CV
                    return _run_and_build_outputs(session, gray, conf, gt_syms)

                _NOOP = (
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(),
                )

                def on_live_change(sketchpad_data: Any, mode: str, conf: float, gt_syms: Any) -> tuple:
                    if mode != "Live":
                        return _NOOP
                    now = time.monotonic()
                    if not hasattr(on_live_change, "_last") or now - on_live_change._last < 0.6:
                        return _NOOP
                    on_live_change._last = now
                    return on_recognize(sketchpad_data, conf, gt_syms)

                def on_clear() -> tuple:
                    ph = gr.update(value=(_placeholder_image(), []), color_map={})
                    return (
                        None,
                        ph, ph,
                        _EMPTY_YOLO, _EMPTY_STAGE2, _EMPTY_RESULT,
                        gr.update(visible=False),
                        None,
                        ph, _EMPTY_CV,
                    )

                def on_file_load(file_obj: Any, conf: float) -> tuple:
                    if file_obj is None:
                        ph = gr.update(value=(_placeholder_image(), []), color_map={})
                        return ph, ph, _EMPTY_YOLO, _EMPTY_STAGE2, _EMPTY_RESULT, None, gr.update(visible=False), gr.update(), None, "", ph, _EMPTY_CV
                    img_path = Path(file_obj.name)
                    gray, gt_syms = load_scene(img_path)
                    gray = preprocess_for_pipeline(gray)
                    if not session.is_loaded():
                        msg = render_result_panel(None, "No models loaded.", 0.0)
                        ph = gr.update(value=(gray, []), color_map={})
                        return ph, ph, _EMPTY_YOLO, _EMPTY_STAGE2, msg, gt_syms, gr.update(visible=False), gr.update(), None, "", ph, _EMPTY_CV
                    yolo_upd, gnn_upd, s1, s2, res, gt_upd, tbl, score, gt_vis, cv_upd, cv_panel = _run_and_build_outputs(session, gray, conf, gt_syms)
                    return yolo_upd, gnn_upd, s1, s2, res, gt_syms, gt_vis, gt_upd, tbl, score, cv_upd, cv_panel

                def _resolve_choice(choice: str, stage: str) -> Optional[Path]:
                    # choice looks like "best.pt (yolo)" / "best.pt (gnn)" (top-level)
                    # or "best.pt (runs/<run_id>)" (per-run). The paren disambiguates the
                    # exact run, so we must use it rather than globbing the first match.
                    if " (" not in choice:
                        return None
                    name = choice.split(" (")[0]
                    # Parent is the first parentheses group; a trailing "  —  label" tag
                    # (added by the picker for readability) is ignored.
                    paren = choice[choice.index(" (") + 2 : choice.index(")")]
                    base = project_root / "artifacts" / stage
                    cand = base / name if paren == stage else base / paren / name
                    return cand if cand.exists() and cand.suffix == ".pt" else None

                def on_load_models(s1_choice: Optional[str], s2_choice: Optional[str]) -> str:
                    if not s1_choice and not s2_choice:
                        return render_status_bar(None, None, None, None)
                    s1_path = _resolve_choice(s1_choice, "yolo") if s1_choice else None
                    s2_path = _resolve_choice(s2_choice, "gnn") if s2_choice else None
                    if s2_path is None or s2_path.suffix != ".pt":
                        return render_status_bar(None, None, None, None)
                    try:
                        session.load(yolo_weights=s1_path, gnn_weights=s2_path)
                        s1_label = "yolo" if s1_path else "components"
                        session._pipeline_name = f"{s1_label}+gnn"
                        session._pipeline_display = f"{'YOLO' if s1_path else 'Components'} + GNN"
                    except Exception as exc:
                        return (
                            f'<div class="engine-status">'
                            f'<div class="engine-status-dot engine-status-err"></div>'
                            f'<span class="engine-status-label">Load failed</span>'
                            f'<span class="engine-status-meta">{str(exc)[:120]}</span>'
                            f'</div>'
                        )
                    return render_status_bar(
                        session.pipeline_name, session.pipeline_display,
                        str(s1_path) if s1_path else None,
                        str(s2_path),
                    )

                # ── Wire events ───────────────────────────────────────────

                recognize_btn.click(
                    fn=on_recognize,
                    inputs=[sketchpad, conf_slider, gt_state],
                    outputs=_OUTPUTS,
                )
                sketchpad.change(
                    fn=on_live_change,
                    inputs=[sketchpad, mode_radio, conf_slider, gt_state],
                    outputs=_OUTPUTS,
                )
                clear_btn.click(
                    fn=on_clear,
                    outputs=[sketchpad, yolo_image, gnn_image, yolo_html, stage2_html, result_html, gt_section, gt_state, cv_image, cv_html],
                )
                file_input.change(
                    fn=on_file_load,
                    inputs=[file_input, conf_slider],
                    outputs=[yolo_image, gnn_image, yolo_html, stage2_html, result_html,
                             gt_state, gt_section, gt_image, diff_table, score_html, cv_image, cv_html],
                )
                load_models_btn.click(
                    fn=on_load_models,
                    inputs=[yolo_dd, stage2_dd],
                    outputs=[status_html],
                )

    # ── Build standalone presenter pages ─────────────────────────────────
    input_demo = build_input_page(session, store)
    results_demo = build_results_page(session, store)

    # ── Mount all three onto a single FastAPI app ─────────────────────────
    # Sub-path mounts must be registered before root "/" so Starlette routing
    # resolves them before the catch-all root app.
    # Starlette mount requires trailing slash for sub-apps; add plain redirects.
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import RedirectResponse
    from gradio import mount_gradio_app

    app = FastAPI()

    # Redirect bare paths to trailing-slash versions (Starlette mount behaviour).
    @app.get("/input")
    async def _redirect_input() -> RedirectResponse:
        return RedirectResponse(url="/input/", status_code=301)

    @app.get("/results")
    async def _redirect_results() -> RedirectResponse:
        return RedirectResponse(url="/results/", status_code=301)

    # Force dark mode on all mounts. Gradio 6.x toggles dark by adding
    # class "dark" to <body>. The js= param on mount_gradio_app runs after
    # page load, ensuring dark is applied regardless of OS preference.
    _FORCE_DARK_JS = (
        "document.querySelector('body').classList.add('dark');"
        " document.documentElement.setAttribute('data-color-scheme', 'dark');"
    )

    # Mount sub-path apps first, then root.
    mount_gradio_app(app, input_demo, path="/input", css=_CSS, theme=gr.themes.Soft(), js=_FORCE_DARK_JS)
    mount_gradio_app(app, results_demo, path="/results", css=_CSS, theme=gr.themes.Soft(), js=_FORCE_DARK_JS)
    mount_gradio_app(app, demo, path="/", css=_CSS, theme=gr.themes.Soft(), js=_FORCE_DARK_JS)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
