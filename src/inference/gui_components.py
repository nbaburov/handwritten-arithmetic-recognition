from __future__ import annotations

from collections import Counter, defaultdict
from typing import Optional

from src.inference.annotation import annotation_colour, coarse_class
from src.inference.gt_loader import SceneScores
from src.inference.palette import (
    COARSE_DISPLAY_NAMES,
    ROLE_COLOURS,
    display_glyph,
    display_name,
    equation_kind_display,
)

_CV_ACCENT = "#22d3ee"  # cyan — distinguishes the OpenCV-fusion card from Stage 1/2


def _latency_badge(ms: float) -> str:
    """Consistent timing badge used in all three stage headers."""
    return f'<span class="engine-latency">{ms:.1f} ms</span>'


def _grid_col(tok: dict) -> int:
    """Place-value column index for a token (grid_col, falling back to col)."""
    v = tok.get("grid_col", tok.get("col", 0))
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _chip(tok: dict, result: bool = False) -> str:
    """One symbol as a role-coloured chip with a full hover tooltip. Colour comes from
    annotation_colour so the chip matches the image box for the same token."""
    label = tok["label"]
    glyph = display_glyph(label)
    conf = tok.get("confidence", 1.0)
    colour = annotation_colour(label, conf)
    conf_str = f"{conf:.2f}" if isinstance(conf, float) else str(conf)
    tip = (
        f"{display_name(label)} | row {tok.get('row', '?')} "
        f"col {tok.get('col', '?')} grid_col {tok.get('grid_col', '?')} "
        f"ord {tok.get('within_row_ord', '?')} conf {conf_str}"
    )
    is_struct = label in ("result_bar", "div_bracket")
    font = "18px" if is_struct else ("22px" if result else "20px")
    extra = " engine-chip--bracket" if label == "div_bracket" else (
        " engine-chip--bar" if label == "result_bar" else ""
    )
    rcls = " engine-chip--result" if result else ""
    return (
        f'<span class="engine-chip{rcls}{extra}" style="--chip:{colour};font-size:{font};" '
        f'title="{tip}">{glyph}</span>'
    )


def _render_symbol_grid(tokens: list[dict], result: bool = False) -> str:
    """Lay symbols out as a row x column grid aligned by predicted grid_col, so the
    place-value columns line up vertically (units under units, etc) and both the row
    and column structure the model predicts is visible. A row that is purely structural
    (a result bar) spans all columns."""
    rows_map: dict[int, list[dict]] = defaultdict(list)
    for tok in tokens:
        rows_map[tok.get("row", 0)].append(tok)
    ncols = max((_grid_col(t) for t in tokens), default=0) + 1

    cells = ['<div class="engine-grid-corner"></div>']
    cells += [f'<div class="engine-grid-collabel">C{c}</div>' for c in range(ncols)]

    for row_idx in sorted(rows_map.keys()):
        row_tokens = rows_map[row_idx]
        cells.append(f'<div class="engine-grid-rowlabel">Row {row_idx + 1}</div>')
        if all(t["label"] in ("result_bar", "div_bracket") for t in row_tokens):
            # Structural-only row: one cell spanning every column.
            chips = "".join(_chip(t, result) for t in row_tokens)
            cells.append(f'<div class="engine-grid-cell engine-grid-span">{chips}</div>')
            continue
        by_col: dict[int, list[dict]] = defaultdict(list)
        for t in sorted(row_tokens, key=lambda t: (_grid_col(t), t.get("within_row_ord", 0))):
            by_col[_grid_col(t)].append(t)
        for c in range(ncols):
            inner = "".join(_chip(t, result) for t in by_col.get(c, []))
            cells.append(f'<div class="engine-grid-cell">{inner}</div>')

    style = f"grid-template-columns: auto repeat({ncols}, minmax(34px, auto));"
    return f'<div class="engine-grid" style="{style}">' + "".join(cells) + "</div>"

# Chip box colours come from annotation_colour(label, conf) so the Stage 2 and Result
# chips use the EXACT colour the YOLO/GNN image boxes use for that token, including the
# low-confidence override. One colour source, no drift between panels and annotations.


def _card(header: str, body: str, accent: Optional[str] = None) -> str:
    border_top = f"border-top: 2px solid {accent};" if accent else ""
    return (
        f'<div class="engine-card" style="{border_top}">'
        f'<div class="engine-card-header">{header}</div>'
        f'<div class="engine-card-body">{body}</div>'
        f'</div>'
    )


def render_yolo_panel(tokens: list[dict], latency_ms: float, backend: str, *, degraded: bool = False) -> str:
    backend_label = "YOLO" if backend == "yolo" else "Components"
    header = f"Stage 1 — {backend_label} &nbsp;·&nbsp; {len(tokens)} detections {_latency_badge(latency_ms)}"

    body_parts = []
    if degraded:
        # Dark-mode safe warning: use secondary bg + subdued text, amber left-border for semantic signal
        body_parts.append(
            '<div style="margin-bottom:8px;padding:6px 8px;'
            'background:var(--engine-bg-secondary,#111827);'
            'border-left:3px solid #f59e0b;'
            'border-radius:4px;font-size:12px;color:var(--engine-text,#f9fafb);">'
            'No YOLO model — using connected-components fallback. '
            'Results may be unreliable. Train Stage 1 for accurate detection.'
            '</div>'
        )

    if not tokens:
        body_parts.append('<span class="empty">No detections</span>')
        return _card(header, "".join(body_parts))

    counts: Counter[str] = Counter(coarse_class(tok["label"]) for tok in tokens)
    body_parts += [
        f'<div class="engine-token">'
        f'<span class="engine-token-label" style="color:{annotation_colour(cls, 0.99)};">'
        f'{COARSE_DISPLAY_NAMES.get(cls, cls)}'
        f'</span>'
        f'<span class="engine-token-count">x{n}</span>'
        f'</div>'
        for cls, n in counts.most_common()
    ]
    return _card(header, "".join(body_parts))


render_stage1_panel = render_yolo_panel


def render_cv_panel(
    detections: list[dict],
    latency_ms: float,
    *,
    cv_boxes: int = 0,
    cv_fallback: int = 0,
    skipped: Optional[str] = None,
) -> str:
    """OpenCV-fusion panel: what the classical detector found in regions YOLO did
    not claim (and the coarse class the crop-classify pass assigned). Mirrors the
    Stage 1 panel layout, with a cyan accent to set it apart.

    detections: list of {label (coarse class), confidence, bbox}.
    cv_boxes: raw connected-components count before classification (for the subtitle).
    skipped: when set (mode "auto" gate), the branch was skipped as "complete".
    """
    header = (
        f"OpenCV after YOLO &nbsp;·&nbsp; {len(detections)} extra "
        f"{_latency_badge(latency_ms)}"
    )

    if skipped:
        body = (
            '<span class="empty">Skipped — scene looked complete '
            '(auto gate). Set cv_fusion mode to "on" to always run.</span>'
        )
        return _card(header, body, accent=_CV_ACCENT)

    if not detections:
        note = (
            f'<div style="font-size:11px;opacity:0.7;margin-bottom:4px;">'
            f'{cv_boxes} raw blob(s), none classified</div>' if cv_boxes else ""
        )
        return _card(header, note + '<span class="empty">No extra detections</span>',
                     accent=_CV_ACCENT)

    counts: Counter[str] = Counter(str(d.get("label", "digit_main")) for d in detections)
    sub = (
        f'<div style="font-size:11px;opacity:0.7;margin-bottom:4px;">'
        f'{len(detections) - cv_fallback} classified by YOLO · {cv_fallback} labelled by GNN</div>'
        if cv_fallback else ""
    )
    body_parts = [sub] + [
        f'<div class="engine-token">'
        f'<span class="engine-token-label" style="color:{ROLE_COLOURS.get(cls, ROLE_COLOURS["digit_main"])};">'
        f'{COARSE_DISPLAY_NAMES.get(cls, cls)}'
        f'</span>'
        f'<span class="engine-token-count">x{n}</span>'
        f'</div>'
        for cls, n in counts.most_common()
    ]
    return _card(header, "".join(body_parts), accent=_CV_ACCENT)


def render_stage2_panel(tokens: list[dict], latency_ms: float, pipeline_label: str) -> str:
    header = f"Stage 2 — {pipeline_label} &nbsp;·&nbsp; {len(tokens)} symbols {_latency_badge(latency_ms)}"

    if not tokens:
        return _card(header, '<span class="empty">No symbols</span>')

    return _card(header, _render_symbol_grid(tokens))


def render_result_panel(
    equation_type: Optional[str],
    display_text: Optional[str],
    latency_ms: float,
    tokens: Optional[list[dict]] = None,
) -> str:
    if equation_type is None or display_text is None:
        return _card("Result", '<span class="empty">—</span>')

    # Equation kind goes in the body as a full, untruncated line (the small uppercase
    # card header clipped longer kinds like "multiplication").
    header = f"Result {_latency_badge(latency_ms)}"
    kind_html = (
        '<div class="engine-result-kind">'
        '<span class="engine-result-kind-label">Equation type</span>'
        f'<span class="engine-result-kind-value">{equation_kind_display(equation_type)}</span>'
        '</div>'
    )

    # If tokens are provided and the equation is a recognised kind (not OOD/error),
    # render a row-grouped symbol view matching Stage 2 style.
    # Fall back to the monospace text block for OOD, error, and bare-text cases.
    is_ood = equation_type in ("unknown", "bare_digits", None)
    if tokens and not is_ood:
        body = _render_symbol_grid(tokens, result=True)
    else:
        # OOD / error / no tokens: monospace text fallback
        escaped = display_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        body = f'<div class="engine-result-eq">{escaped}</div>'

    return _card(header, kind_html + body)


def render_status_bar(
    pipeline_name: Optional[str],
    pipeline_display: Optional[str],
    yolo_path: Optional[str],
    stage2_path: Optional[str],
) -> str:
    if pipeline_name is None:
        return (
            '<div class="engine-status">'
            '<div class="engine-status-dot engine-status-err"></div>'
            '<span class="engine-status-label">No models loaded</span>'
            '<span class="engine-status-meta">Check artifacts/ directory</span>'
            '</div>'
        )

    has_yolo = "components" not in pipeline_name
    dot_cls = "engine-status-ok" if has_yolo else "engine-status-warn"
    s1_text = "YOLO" if has_yolo else "Connected-components fallback"
    s2_text = "GNN"

    return (
        f'<div class="engine-status">'
        f'<div class="engine-status-dot {dot_cls}"></div>'
        f'<span class="engine-status-label">{pipeline_display}</span>'
        f'<span class="engine-status-meta">Stage 1: {s1_text}</span>'
        f'<span class="engine-status-meta">Stage 2: {s2_text}</span>'
        f'</div>'
    )


def render_score_bar(scores: SceneScores) -> str:
    exact_cls = "ok" if scores.exact_match else "err"
    exact_sym = "✓ Exact match" if scores.exact_match else "✗ No exact match"
    return (
        f'<div class="engine-score">'
        f'<strong>{scores.n_correct}/{scores.n_total} correct</strong>'
        f'<span>Token F1: <strong>{scores.token_f1:.2f}</strong></span>'
        f'<span>Row acc: <strong>{scores.row_accuracy:.2f}</strong></span>'
        f'<span>Col acc: <strong>{scores.col_accuracy:.2f}</strong></span>'
        f'<span class="{exact_cls}">{exact_sym}</span>'
        f'</div>'
    )
