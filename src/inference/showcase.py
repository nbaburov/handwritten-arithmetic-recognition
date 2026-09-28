"""Standalone presenter pages for the recognition GUI.

Public API
----------
build_input_page(session, store) -> gr.Blocks
    Standalone freestanding Blocks with only the giant Sketchpad + Recognize/Clear.
    Mounted at /input.

build_results_page(session, store) -> gr.Blocks
    Standalone freestanding Blocks showing the latest result + feedback flow.
    Auto-polls the ShowcaseStore every 1.5 s via gr.Timer.
    Mounted at /results.

Shared pipeline logic lives in _run_pipeline_and_record().
"""
from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any, Optional

import gradio as gr
import numpy as np
from PIL import Image

from ..parsing.equation_display import format_parsed_equation_readable
from .annotation import build_annotations
from .run import InferenceSession
from .showcase_store import (
    ShowcaseStore,
    STATUS_OK,
    STATUS_BLANK,
    STATUS_NO_MODELS,
    STATUS_NO_DETECTIONS,
    STATUS_OOD,
    STATUS_ERROR,
    FEEDBACK_ENABLED_STATUSES,
)


# ── Small image helpers ───────────────────────────────────────────────────────

def _gray_to_png_bytes(gray: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(gray.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.stack([gray, gray, gray], axis=-1).astype(np.uint8)


def _placeholder_rgb() -> np.ndarray:
    return np.full((512, 512, 3), 248, dtype=np.uint8)


def _extract_tokens_with_conf(payload: dict) -> list:
    tokens = payload.get("tokens", [])
    return [
        {
            "label": t.get("label"),
            "yolo_conf": None,
            "gnn_conf": None,
            "confidence": t.get("confidence"),
            "bbox": t.get("bbox"),
        }
        for t in tokens
    ]


def _build_equation_html(readable: str) -> str:
    """Wrap the pre-aligned readable string in a <pre> block with CSS class."""
    import html as _html
    escaped = _html.escape(readable)
    return f'<pre class="engine-sc-pre">{escaped}</pre>'


def _build_tokens_dataframe(tokens: list[dict]) -> list[list]:
    """Build per-token DataFrame rows: [#, Predicted, Conf, Wrong?, Should be]."""
    rows = []
    for i, tok in enumerate(tokens):
        label = tok.get("label", "?")
        conf = tok.get("confidence")
        conf_str = f"{conf:.2f}" if conf is not None else ""
        rows.append([i, label, conf_str, "", ""])
    return rows


_ISSUE_ALIASES = {
    "wrong": "wrong_label", "w": "wrong_label", "wrong_label": "wrong_label",
    "yes": "wrong_label", "y": "wrong_label", "1": "wrong_label", "x": "wrong_label",
    "dup": "duplicate", "duplicate": "duplicate", "d": "duplicate",
    "extra": "spurious", "spurious": "spurious", "ghost": "spurious", "fake": "spurious",
    "split": "split", "merge": "merged",
}


def _parse_tokens_dataframe(df_value: Any) -> list[dict]:
    """Parse DataFrame state. Includes rows marked with an issue OR a should-be value.

    Issue column accepts: wrong / dup / extra / split / merge (or aliases).
    """
    out = []
    if df_value is None:
        return out
    try:
        rows = df_value.values.tolist() if hasattr(df_value, "values") else list(df_value)
    except Exception:
        return out
    for row in rows:
        if len(row) < 5:
            continue
        try:
            idx = int(row[0])
        except (ValueError, TypeError):
            continue
        predicted = str(row[1]) if row[1] is not None else "?"
        issue_raw = str(row[3]).strip().lower() if row[3] is not None else ""
        should_be = str(row[4]).strip() if row[4] is not None else ""
        issue_type = _ISSUE_ALIASES.get(issue_raw, "")
        if not issue_type and should_be:
            issue_type = "wrong_label"
        if not issue_type:
            continue
        out.append({
            "index": idx,
            "predicted_label": predicted,
            "user_says_wrong": True,
            "issue_type": issue_type,
            "should_be": should_be,
        })
    return out


def _parse_missed_tokens(text: str) -> list[dict]:
    """Parse missed-tokens textbox: one label per line, optional 'after #N' hint.

    Lines like 'main_5' or 'main_5 after #3' or 'op_plus near top'.
    """
    out = []
    if not text:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append({
            "index": -1,
            "predicted_label": "(none)",
            "user_says_wrong": True,
            "issue_type": "missed",
            "should_be": line.split()[0] if line else "",
            "note": line,
        })
    return out


# ── Shared pipeline runner ────────────────────────────────────────────────────

def _run_pipeline_and_record(
    sketchpad_data: Any,
    session: InferenceSession,
    store: ShowcaseStore,
) -> dict:
    """Run the pipeline for one sketchpad frame and persist the result.

    Returns a result dict with keys:
        status, sample_dir, kind_html, eq_html, thumb_value,
        fb_ok, tokens_with_conf, display_text
    All callers build their Gradio output tuples from this dict.
    """
    from .gui import _extract_gray, _is_blank  # avoid circular import

    gray = _extract_gray(sketchpad_data)
    is_blank = gray is None or _is_blank(gray)
    dark_px = 0 if gray is None else int(np.sum(gray < 200))

    png_bytes: Optional[bytes] = None
    if gray is not None:
        try:
            png_bytes = _gray_to_png_bytes(gray)
        except Exception:
            pass

    def _input_meta() -> dict:
        return {
            "shape": list(gray.shape) if gray is not None else [0, 0],
            "dark_pixel_count": dark_px,
            "is_blank": is_blank,
        }

    _BLANK_BANNER = '<div class="engine-sc-banner engine-sc-banner--neutral">—</div>'
    _PLACEHOLDER_ANNOTATED = (_placeholder_rgb(), [])

    def _noop(msg: str) -> dict:
        return {
            "status": STATUS_BLANK,
            "sample_dir": None,
            "kind_html": _BLANK_BANNER,
            "eq_html": f'<pre class="engine-sc-pre engine-sc-pre--empty">{msg}</pre>',
            "thumb_value": _PLACEHOLDER_ANNOTATED,
            "fb_ok": False,
            "tokens_with_conf": [],
            "display_text": msg,
        }

    if is_blank:
        store.record_prediction(
            gray_png_bytes=png_bytes,
            outcome={
                "status": STATUS_BLANK,
                "input": _input_meta(),
                "equation_kind": None,
                "ood_reason": None,
                "num_tokens": 0,
                "tokens_with_conf": [],
                "payload": {},
                "readable": "",
                "timings_ms": {},
            },
        )
        return _noop("Canvas is empty — draw something first.")

    if not session.is_loaded():
        store.record_prediction(
            gray_png_bytes=png_bytes,
            outcome={
                "status": STATUS_NO_MODELS,
                "input": _input_meta(),
                "equation_kind": None,
                "ood_reason": None,
                "num_tokens": 0,
                "tokens_with_conf": [],
                "payload": {},
                "readable": "",
                "timings_ms": {},
            },
        )
        return _noop("No models loaded.")

    t_total = time.perf_counter()
    try:
        payload = session.predict(image=gray, yolo_conf=0.05)
    except Exception as exc:
        total_ms = round((time.perf_counter() - t_total) * 1000.0, 2)
        err_str = str(exc)[:500]
        store.record_prediction(
            gray_png_bytes=png_bytes,
            outcome={
                "status": STATUS_ERROR,
                "input": _input_meta(),
                "equation_kind": None,
                "ood_reason": None,
                "num_tokens": 0,
                "tokens_with_conf": [],
                "error_detail": err_str,
                "payload": {},
                "readable": "",
                "timings_ms": {"total_ms": total_ms},
            },
        )
        return _noop(f"Error: {err_str}")

    total_ms = round((time.perf_counter() - t_total) * 1000.0, 2)
    timings = payload.get("timings", {})
    tokens = payload.get("tokens", [])
    num_tokens = len(tokens)
    equation_kind = payload.get("equation_kind") or "unknown"
    ood_reason = payload.get("ood_reason")

    if num_tokens == 0:
        status = STATUS_NO_DETECTIONS
    elif equation_kind == "unknown" or ood_reason:
        status = STATUS_OOD
    else:
        status = STATUS_OK

    try:
        readable = format_parsed_equation_readable(payload)
    except Exception:
        readable = str(payload.get("rows", ""))

    tokens_with_conf = _extract_tokens_with_conf(payload)

    sample_dir = store.record_prediction(
        gray_png_bytes=png_bytes,
        outcome={
            "status": status,
            "input": _input_meta(),
            "equation_kind": equation_kind,
            "ood_reason": ood_reason,
            "num_tokens": num_tokens,
            "tokens_with_conf": tokens_with_conf,
            "payload": payload,
            "readable": readable,
            "timings_ms": {
                "yolo_ms": timings.get("yolo_ms"),
                "stage2_ms": timings.get("stage2_ms"),
                "assemble_ms": timings.get("assemble_ms"),
                "total_ms": total_ms,
            },
        },
    )

    # Build annotated thumbnail.
    rgb_image = _gray_to_rgb(gray)
    valid_tokens = [
        t for t in tokens_with_conf
        if t.get("bbox") is not None and t.get("confidence") is not None
    ]
    try:
        annotations, _color_map = build_annotations(valid_tokens)
    except Exception:
        annotations = []
    thumb_value = (rgb_image, annotations)

    # Banner HTML.
    kind_css = "engine-sc-banner"
    if status == STATUS_OK:
        kind_css += " engine-sc-banner--ok"
    elif status == STATUS_OOD:
        kind_css += " engine-sc-banner--ood"
    else:
        kind_css += " engine-sc-banner--neutral"
    kind_html = f'<div class="{kind_css}">{equation_kind}</div>'

    if status == STATUS_NO_DETECTIONS:
        display_text = "Nothing recognised — try writing larger or clearer."
    elif status == STATUS_OOD:
        extra = f"\n\nCould not form a valid equation: {ood_reason}" if ood_reason else ""
        display_text = readable + extra
    else:
        display_text = readable

    eq_html = _build_equation_html(display_text)
    fb_ok = status in FEEDBACK_ENABLED_STATUSES

    return {
        "status": status,
        "sample_dir": sample_dir,
        "kind_html": kind_html,
        "eq_html": eq_html,
        "thumb_value": thumb_value,
        "fb_ok": fb_ok,
        "tokens_with_conf": tokens_with_conf,
        "display_text": display_text,
    }


# ── Existing Showcase tab (unchanged contract) ────────────────────────────────

def build_input_page(session: InferenceSession, store: ShowcaseStore) -> gr.Blocks:
    """Return a freestanding Blocks with only the giant Sketchpad.

    After Recognize the pipeline runs and writes to the shared store so the
    Results page can poll it. No result panel is shown here.
    CSS and theme are injected via mount_gradio_app in gui.py.
    """
    with gr.Blocks(
        title="Recognition — Input",
        elem_classes=["engine-input-page"],
    ) as blocks:
        with gr.Row(equal_height=True, elem_classes=["engine-input-row"]):
            with gr.Column(scale=3, min_width=0, elem_classes=["engine-input-left"]):
                gr.HTML('<div class="engine-input-page-header">Draw your arithmetic</div>')
                sk_input = gr.Sketchpad(
                    label="",
                    height=560,
                    brush=gr.Brush(
                        default_size=3,
                        colors=["#000000"],
                        color_mode="fixed",
                    ),
                    elem_classes=["engine-input-canvas"],
                )
                with gr.Row(elem_classes=["engine-input-btn-row"]):
                    inp_recognize_btn = gr.Button(
                        "Recognize",
                        variant="primary",
                        elem_classes=["engine-input-primary-btn"],
                    )
                    inp_clear_btn = gr.Button(
                        "Clear",
                        variant="secondary",
                        elem_classes=["engine-input-secondary-btn"],
                    )
                inp_status = gr.HTML(value="")

            with gr.Column(scale=2, min_width=0, elem_classes=["engine-input-right"]):
                gr.HTML('<div class="engine-input-page-header">What the model saw</div>')
                inp_thumb = gr.AnnotatedImage(
                    label="",
                    height=560,
                    value=(_placeholder_rgb(), []),
                    elem_classes=["engine-input-thumb"],
                )
        inp_last_seq = gr.State(0)

        def _load_thumb(sample_dir: Path) -> tuple:
            try:
                import json as _json
                pred = _json.loads((sample_dir / "prediction.json").read_text(encoding="utf-8"))
                tokens_with_conf = pred.get("tokens_with_conf", [])
                from PIL import Image as _PILImage
                pil_img = _PILImage.open(sample_dir / "input.png").convert("L")
                gray = np.array(pil_img, dtype=np.uint8)
                rgb = _gray_to_rgb(gray)
                valid_tokens = [
                    t for t in tokens_with_conf
                    if t.get("bbox") is not None and t.get("confidence") is not None
                ]
                try:
                    annotations, _ = build_annotations(valid_tokens)
                except Exception:
                    annotations = []
                return (rgb, annotations)
            except Exception:
                return (_placeholder_rgb(), [])

        def on_inp_recognize(sketchpad_data: Any) -> tuple:
            result = _run_pipeline_and_record(sketchpad_data, session, store)
            if result["sample_dir"] is None:
                return ('<span class="engine-inp-status-warn">Empty canvas</span>',
                        gr.update(), store.latest_seq)
            sample_dir = Path(result["sample_dir"])
            thumb = _load_thumb(sample_dir)
            return ('<span class="engine-inp-status-ok">Sent ✓</span>',
                    thumb, store.latest_seq)

        def on_inp_clear() -> tuple:
            return None, "", (_placeholder_rgb(), []), store.latest_seq

        def on_inp_poll(last_seq: int) -> tuple:
            current = store.latest_seq
            if current == last_seq:
                return last_seq, gr.update()
            sample_dir = store.get_latest_sample()
            if sample_dir is None:
                return current, gr.update()
            return current, _load_thumb(sample_dir)

        inp_recognize_btn.click(
            fn=on_inp_recognize,
            inputs=[sk_input],
            outputs=[inp_status, inp_thumb, inp_last_seq],
        )

        inp_clear_btn.click(
            fn=on_inp_clear,
            outputs=[sk_input, inp_status, inp_thumb, inp_last_seq],
        )

        inp_timer = gr.Timer(value=1.5)
        inp_timer.tick(
            fn=on_inp_poll,
            inputs=[inp_last_seq],
            outputs=[inp_last_seq, inp_thumb],
        )

    return blocks


# ── Standalone RESULTS page ───────────────────────────────────────────────────

def build_results_page(session: InferenceSession, store: ShowcaseStore) -> gr.Blocks:
    """Return a freestanding Blocks showing the latest result + feedback.

    Auto-polls the ShowcaseStore via gr.Timer(value=1.5).
    CSS and theme are injected via mount_gradio_app in gui.py.
    """
    _PLACEHOLDER_ANNOTATED = (_placeholder_rgb(), [])

    with gr.Blocks(
        title="Recognition — Results",
        elem_classes=["engine-results-page"],
    ) as blocks:
        # Holds the last seq we rendered — avoids redundant repaints.
        last_seq_state = gr.State(-1)
        # Holds the sample_dir string of whatever is currently displayed.
        res_sample_dir_state = gr.State(None)
        res_tokens_state = gr.State([])

        gr.HTML('<div class="engine-results-page-header">Latest Result</div>')

        with gr.Row():
            with gr.Column(scale=2):
                res_kind_banner = gr.HTML(
                    value='<div class="engine-sc-banner engine-sc-banner--neutral">Waiting for input…</div>'
                )
                res_result_html = gr.HTML(
                    value='<pre class="engine-sc-pre engine-sc-pre--empty">(no result yet)</pre>'
                )
                res_thumb = gr.AnnotatedImage(
                    label="What the model saw",
                    height=300,
                    value=_PLACEHOLDER_ANNOTATED,
                )

            with gr.Column(scale=1):
                with gr.Row(elem_classes=["engine-sc-hero-row"]):
                    res_yes_btn = gr.Button(
                        "Yes, that's right",
                        interactive=False,
                        elem_classes=["engine-sc-hero-yes"],
                    )
                    res_no_btn = gr.Button(
                        "No, something's wrong",
                        interactive=False,
                        elem_classes=["engine-sc-hero-no"],
                    )

                res_detail_group = gr.Group(visible=False)
                with res_detail_group:
                    gr.HTML(
                        '<div class="engine-sc-help">'
                        '<b>Issue</b> column — type one of: '
                        '<code>wrong</code> (wrong label), '
                        '<code>dup</code> (same symbol detected twice), '
                        '<code>extra</code> (model saw something not there), '
                        '<code>split</code> (one symbol split into two boxes), '
                        '<code>merge</code> (two symbols stuck together).<br>'
                        '<b>Should be</b> — the correct label, e.g. '
                        '<code>main_5</code>, <code>op_plus</code>, <code>op_minus</code>, '
                        '<code>op_times</code>, <code>op_divide</code>, '
                        '<code>carry_1</code>, <code>borrow_2</code>, '
                        '<code>result_bar</code>, <code>div_bracket</code>.'
                        '</div>'
                    )
                    res_token_df = gr.DataFrame(
                        headers=["#", "Predicted", "Conf", "Issue", "Should be"],
                        datatype=["number", "str", "str", "str", "str"],
                        value=[],
                        interactive=True,
                        label="Per-token review",
                        row_count=(1, "dynamic"),
                        col_count=(5, "fixed"),
                    )
                    res_missed_txt = gr.Textbox(
                        label="Missed symbols (one label per line — for things model didn't see)",
                        placeholder="main_5\nop_plus\ncarry_1",
                        lines=3,
                    )
                    res_intended_txt = gr.Textbox(
                        label="What did you intend? (optional)",
                        placeholder="e.g. 24 + 13 = 37",
                    )
                    res_update_btn = gr.Button("Update feedback", size="sm")

                res_status_txt = gr.HTML(value="")

        # ── Poll timer ────────────────────────────────────────────────────
        # gr.Timer fires every value seconds and triggers .tick events.
        timer = gr.Timer(value=1.5)

        # Outputs updated by the poll (seq state + all visible components).
        _POLL_OUTPUTS = [
            last_seq_state,
            res_sample_dir_state,
            res_tokens_state,
            res_kind_banner,
            res_result_html,
            res_thumb,
            res_yes_btn,
            res_no_btn,
            res_token_df,
        ]

        def _poll(last_seq: int) -> tuple:
            current_seq = store.latest_seq
            if current_seq == last_seq:
                # Nothing new — return no-op updates for everything.
                return (
                    last_seq,
                    gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(),
                )

            sample_dir = store.get_latest_sample()
            if sample_dir is None:
                return (
                    current_seq,
                    None, [],
                    gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(),
                )

            # Load the prediction.json to rebuild the display.
            import json
            pred_path = sample_dir / "prediction.json"
            try:
                pred = json.loads(pred_path.read_text(encoding="utf-8"))
            except Exception:
                return (
                    current_seq,
                    str(sample_dir), [],
                    gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(),
                )

            status = pred.get("status", "")
            equation_kind = pred.get("equation_kind") or "unknown"
            readable = pred.get("readable", "")
            ood_reason = pred.get("ood_reason")
            tokens_with_conf = pred.get("tokens_with_conf", [])

            # Reconstruct input image for thumbnail.
            img_path = sample_dir / "input.png"
            thumb_value = _PLACEHOLDER_ANNOTATED
            if img_path.exists() and img_path.stat().st_size > 0:
                try:
                    from PIL import Image as _PILImage
                    pil_img = _PILImage.open(img_path).convert("L")
                    gray = np.array(pil_img, dtype=np.uint8)
                    rgb = _gray_to_rgb(gray)
                    valid_tokens = [
                        t for t in tokens_with_conf
                        if t.get("bbox") is not None and t.get("confidence") is not None
                    ]
                    try:
                        annotations, _ = build_annotations(valid_tokens)
                    except Exception:
                        annotations = []
                    thumb_value = (rgb, annotations)
                except Exception:
                    pass

            # Banner.
            kind_css = "engine-sc-banner"
            if status == STATUS_OK:
                kind_css += " engine-sc-banner--ok"
            elif status == STATUS_OOD:
                kind_css += " engine-sc-banner--ood"
            else:
                kind_css += " engine-sc-banner--neutral"
            kind_html = f'<div class="{kind_css}">{equation_kind}</div>'

            if status == STATUS_NO_DETECTIONS:
                display_text = "Nothing recognised — try writing larger or clearer."
            elif status == STATUS_OOD:
                extra = f"\n\nCould not form a valid equation: {ood_reason}" if ood_reason else ""
                display_text = readable + extra
            elif status in (STATUS_BLANK, STATUS_NO_MODELS, STATUS_ERROR):
                display_text = f"({status})"
            else:
                display_text = readable

            eq_html = _build_equation_html(display_text)
            fb_ok = status in FEEDBACK_ENABLED_STATUSES
            token_rows = _build_tokens_dataframe(tokens_with_conf)

            return (
                current_seq,
                str(sample_dir),
                tokens_with_conf,
                kind_html,
                eq_html,
                thumb_value,
                gr.update(interactive=fb_ok),
                gr.update(interactive=fb_ok),
                gr.update(value=token_rows),
            )

        timer.tick(fn=_poll, inputs=[last_seq_state], outputs=_POLL_OUTPUTS)

        # ── Feedback handlers ─────────────────────────────────────────────

        def on_res_yes(sample_dir_str: Optional[str]) -> tuple:
            if not sample_dir_str:
                return gr.update(visible=False), "Nothing to save yet."
            try:
                store.record_feedback(
                    sample_dir=Path(sample_dir_str),
                    verdict="correct",
                    intended_text="",
                    per_token=None,
                )
                return gr.update(visible=False), '<span class="engine-sc-saved-ok">Saved as correct ✓</span>'
            except Exception as exc:
                return gr.update(visible=False), f"Save failed: {exc}"

        def on_res_no(sample_dir_str: Optional[str]) -> tuple:
            if not sample_dir_str:
                return gr.update(visible=False), "Nothing to save yet."
            try:
                store.record_feedback(
                    sample_dir=Path(sample_dir_str),
                    verdict="wrong",
                    intended_text="",
                    per_token=None,
                )
                return gr.update(visible=True), "Saved as wrong — add detail if you can"
            except Exception as exc:
                return gr.update(visible=False), f"Save failed: {exc}"

        def on_res_update(
            sample_dir_str: Optional[str],
            df_value: Any,
            intended: str,
            tokens_list: list,
            missed_text: str = "",
        ) -> str:
            if not sample_dir_str:
                return "Nothing to save yet."
            per_token = _parse_tokens_dataframe(df_value) + _parse_missed_tokens(missed_text)
            try:
                store.record_feedback(
                    sample_dir=Path(sample_dir_str),
                    verdict="wrong",
                    intended_text=intended or "",
                    per_token=per_token if per_token else None,
                )
                return "Updated"
            except Exception as exc:
                return f"Save failed: {exc}"

        res_yes_btn.click(
            fn=on_res_yes,
            inputs=[res_sample_dir_state],
            outputs=[res_detail_group, res_status_txt],
        )
        res_no_btn.click(
            fn=on_res_no,
            inputs=[res_sample_dir_state],
            outputs=[res_detail_group, res_status_txt],
        )
        res_update_btn.click(
            fn=on_res_update,
            inputs=[res_sample_dir_state, res_token_df, res_intended_txt, res_tokens_state, res_missed_txt],
            outputs=[res_status_txt],
        )

    return blocks
