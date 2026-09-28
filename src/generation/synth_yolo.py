from __future__ import annotations

import collections
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from ..core.config import DataPrepConfig
from ..core.ontology import YOLO_CLASS_NAMES
from ..core.progress import progress_iter
from ..core.run_config import GenerationConfig, PresetConfig, SceneConfig
from ..data_pipeline.dataset import ensure_dirs, write_json
from .layouts import (
    LayoutToken, EquationKind, SceneCase, layout_bounds, sample_layout_for_case,
    OPERATOR_TOKEN_SCALE,
)
from .completion_stages import sample_completion_stage_for_case
from .handwriting_style import HandwritingStyle, make_style

# Pool helpers re-exported from synth_pool for back-compat (callers that
# previously imported from synth_yolo continue to work unchanged).
from .synth_pool import (  # noqa: F401
    _load_symbol_rows,
    _apply_broken_stroke,
    _paste_symbol,
    _tile_from_path,
    resolve_pool_root,
    _substitute_pool_path,
    _load_tile_with_quality,
    _result_bar_fallback_tile,
    _yolo_box,
    _sample_symbol,
    _result_digit_col_span,
    _bar_col_span,
    _draw_borrow_cross,
    _draw_equation_block,
)

# ---------------------------------------------------------------------------
# Utility (preserved for test compatibility)
# ---------------------------------------------------------------------------

def _apply_blank_result(
    tokens: list,
    result_rows: set,
) -> list:
    """Remove tokens on result rows (blank-result variant)."""
    return [t for t in tokens if t.row not in result_rows]


# ---------------------------------------------------------------------------
# Scene-level canvas rotation (section D)
# ---------------------------------------------------------------------------

def _rotate_canvas_and_bboxes(
    canvas: np.ndarray,
    bboxes_yolo: List[str],
    angle_deg: float,
    gt_tokens: List[Dict[str, Any]],
) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
    """Rotate the full canvas and transform all YOLO bboxes + gt pixel coords.

    Uses INTER_LINEAR interpolation and white background fill so corners stay
    white.  If angle_deg == 0.0 returns inputs unchanged.

    Args:
        canvas: uint8 grayscale (H, W).
        bboxes_yolo: List of YOLO annotation strings "class cx cy w h".
        angle_deg: Rotation angle in degrees (positive = counter-clockwise).
        gt_tokens: List of ground-truth token dicts with "bbox" key [x0,y0,x1,y1].

    Returns:
        (rotated_canvas, rotated_bboxes_yolo, rotated_gt_tokens)
    """
    if angle_deg == 0.0:
        return canvas, bboxes_yolo, gt_tokens

    H, W = canvas.shape
    cx, cy = W / 2.0, H / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    rotated = cv2.warpAffine(
        canvas, M, (W, H),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )

    def _transform_point(x: float, y: float) -> Tuple[float, float]:
        pt = np.array([x, y, 1.0])
        tx, ty = M @ pt
        return float(tx), float(ty)

    new_bboxes: List[str] = []
    for line in bboxes_yolo:
        parts = line.split()
        cls = parts[0]
        cx_n, cy_n, bw_n, bh_n = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        # Pixel corners
        px = cx_n * W
        py = cy_n * H
        hw = bw_n * W / 2.0
        hh = bh_n * H / 2.0
        corners = [(px - hw, py - hh), (px + hw, py - hh),
                   (px + hw, py + hh), (px - hw, py + hh)]
        txs, tys = zip(*[_transform_point(x, y) for x, y in corners])
        new_x0, new_x1 = min(txs), max(txs)
        new_y0, new_y1 = min(tys), max(tys)
        new_cx = ((new_x0 + new_x1) / 2.0) / W
        new_cy = ((new_y0 + new_y1) / 2.0) / H
        new_bw = max((new_x1 - new_x0) / W, 1e-6)
        new_bh = max((new_y1 - new_y0) / H, 1e-6)
        # Clamp to [0, 1]
        new_cx = float(np.clip(new_cx, 0.0, 1.0))
        new_cy = float(np.clip(new_cy, 0.0, 1.0))
        new_bw = float(np.clip(new_bw, 1e-6, 1.0))
        new_bh = float(np.clip(new_bh, 1e-6, 1.0))
        new_bboxes.append(f"{cls} {new_cx:.6f} {new_cy:.6f} {new_bw:.6f} {new_bh:.6f}")

    new_gt: List[Dict[str, Any]] = []
    for tok in gt_tokens:
        tok = dict(tok)
        if "bbox" in tok:
            x0, y0, x1, y1 = tok["bbox"]
            corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            txs, tys = zip(*[_transform_point(x, y) for x, y in corners])
            tok["bbox"] = [
                int(np.clip(min(txs), 0, W)),
                int(np.clip(min(tys), 0, H)),
                int(np.clip(max(txs), 0, W)),
                int(np.clip(max(tys), 0, H)),
            ]
        new_gt.append(tok)

    return rotated, new_bboxes, new_gt


_OPERATOR_LABELS = ("op_plus", "op_minus", "op_times", "op_divide")
_OPERATOR_GLYPH_KEYS = ("plus", "minus", "times", "divide")


def _primary_kind_from_tokens(tokens: List[LayoutToken]) -> Optional[EquationKind]:
    """Map a token list to its primary equation kind, or None if undecidable.

    Uses the same precedence the layouts encode:

    1. division wins if any divide_bracket structural token or op_divide
       operator is present (matches long/short/simple division layouts);
    2. otherwise op_times -> multiply (the primary operator in multiplication,
       even when partial-product op_plus tokens are also present);
    3. otherwise op_minus -> subtract;
    4. otherwise op_plus -> add.

    Returns None when no operator/structural token is present (caller decides
    the fallback).
    """
    has_bracket = any(t.yolo_class_name == "divide_bracket" for t in tokens)
    op_labels = {t.flattened_label for t in tokens if t.yolo_class_name == "operator"}
    if has_bracket or "op_divide" in op_labels:
        return EquationKind.divide
    if "op_times" in op_labels:
        return EquationKind.multiply
    if "op_minus" in op_labels:
        return EquationKind.subtract
    if "op_plus" in op_labels:
        return EquationKind.add
    return None


def _operator_multiset(tokens: List[LayoutToken]) -> "collections.Counter[str]":
    """Multiset of operator-token labels (yolo_class_name == 'operator')."""
    return collections.Counter(
        t.flattened_label for t in tokens if t.yolo_class_name == "operator"
    )


def _eq_kind_from_final_tokens(
    original_tokens: List[LayoutToken],
    final_tokens: List[LayoutToken],
    fallback: EquationKind,
) -> EquationKind:
    """Scene equation kind reflecting the FINAL drawn operator (A5 fix).

    ``_apply_scene_knobs`` can swap the drawn operator (when
    ``wrong_operator_prob > 0``), so the kind sampled from the layout no longer
    matches what is actually drawn. This recomputes the kind from the FINAL
    token list, but ONLY when ``_apply_scene_knobs`` performed an operator
    *swap*, detected as the operator multiset changing while its total count is
    preserved (``wrong_result`` only touches main digits; ``missing_structural``
    *drops* a token, lowering the count). In every other case the layout's own
    ``fallback`` kind is returned unchanged, so:

    - OOD sentinel scenes (``fallback == ood_unknown``: bare_digits,
      standalone_bar, standalone_bracket) stay OOD; a lone divide_bracket with
      no operands is structure-only OOD, not a division equation.
    - ``missing_structural`` dropping the lone op (or the op_times in a
      partial-product multiplication, leaving op_plus helpers) keeps the layout
      kind, matching the pre-A5 behaviour exactly.

    Net effect: a strict no-op whenever no operator was swapped, which makes it
    a no-op for the entire current config (``wrong_operator_prob == 0.0``).
    """
    if fallback == EquationKind.ood_unknown:
        return fallback
    before = _operator_multiset(original_tokens)
    after = _operator_multiset(final_tokens)
    # A swap preserves the operator count but changes which labels are present.
    swapped = sum(before.values()) == sum(after.values()) and before != after
    if not swapped:
        return fallback
    derived = _primary_kind_from_tokens(final_tokens)
    return derived if derived is not None else fallback


def _apply_scene_knobs(
    tokens: List[LayoutToken],
    rng: random.Random,
    scene_cfg: "SceneConfig",
) -> List[LayoutToken]:
    """Apply scene-level augmentations to the token list before rendering.

    Modifies tokens in place (returns a new list with replaced dataclass instances
    where needed since LayoutToken is frozen).

    Knobs wired:
    - wrong_result_prob: flip one result-row main-digit glyph to a different digit.
    - missing_structural_prob: drop the result_bar OR the operator token.
    - wrong_operator_prob: replace operator glyph_key/flattened_label with a different op.
    """
    # --- wrong_result_prob ---
    if scene_cfg.wrong_result_prob > 0 and rng.random() < scene_cfg.wrong_result_prob:
        main_toks = [t for t in tokens if t.flattened_label.startswith("main_")]
        if main_toks:
            result_row = max(t.row for t in main_toks)
            result_candidates = [t for t in tokens if t.row == result_row and t.flattened_label.startswith("main_")]
            if result_candidates:
                target = rng.choice(result_candidates)
                cur_digit = int(target.glyph_key) if target.glyph_key.isdigit() else 0
                new_digit = rng.choice([d for d in range(10) if d != cur_digit])
                replacement = LayoutToken(
                    row=target.row, col=target.col,
                    flattened_label=f"main_{new_digit}",
                    glyph_key=str(new_digit),
                    yolo_class_name=target.yolo_class_name,
                    token_scale=target.token_scale,
                    segment_kind=target.segment_kind,
                )
                tokens = [replacement if t is target else t for t in tokens]

    # --- missing_structural_prob ---
    if scene_cfg.missing_structural_prob > 0 and rng.random() < scene_cfg.missing_structural_prob:
        bars = [t for t in tokens if t.flattened_label == "result_bar"]
        ops = [t for t in tokens if t.yolo_class_name == "operator"]
        candidates = bars + ops
        # Guard: never drop a token when there is only one token in the scene.
        # standalone_bar / standalone_bracket scenes have a single structural token;
        # dropping it produces an empty tokens list which writes a 0-byte YOLO label file.
        if candidates and len(tokens) > 1:
            to_drop = rng.choice(candidates)
            tokens = [t for t in tokens if t is not to_drop]

    # --- wrong_operator_prob ---
    if scene_cfg.wrong_operator_prob > 0 and rng.random() < scene_cfg.wrong_operator_prob:
        ops = [t for t in tokens if t.yolo_class_name == "operator"]
        if ops:
            target = rng.choice(ops)
            cur_label = target.flattened_label
            alt_labels = [lbl for lbl in _OPERATOR_LABELS if lbl != cur_label]
            if alt_labels:
                new_label = rng.choice(alt_labels)
                new_glyph = _OPERATOR_GLYPH_KEYS[_OPERATOR_LABELS.index(new_label)]
                replacement = LayoutToken(
                    row=target.row, col=target.col,
                    flattened_label=new_label,
                    glyph_key=new_glyph,
                    yolo_class_name=target.yolo_class_name,
                    token_scale=target.token_scale,
                    segment_kind=target.segment_kind,
                )
                tokens = [replacement if t is target else t for t in tokens]

    return tokens


def _render_equations_on_canvas(
    canvas: np.ndarray,
    rng: random.Random,
    symbols_by_glyph_key: Dict[str, pd.DataFrame],
    symbols_by_glyph_key_fallback: Dict[str, pd.DataFrame],
    width: int,
    height: int,
    cell_w: int,
    cell_h: int,
    style: HandwritingStyle,
    case: SceneCase,
    completion_weights: dict | None = None,
    rendering: dict | None = None,
    scene_cfg: "SceneConfig | None" = None,
    pool_root: Optional[Path] = None,
    min_source_quality: float = 0.0,
) -> Tuple[List[str], List[Dict[str, Any]], List[str], List[str]]:
    _ = cell_w, cell_h
    annotations: List[str] = []
    gt_tokens: List[Dict[str, Any]] = []
    eq_kinds: List[str] = []
    stage_names: List[str] = []
    n_eq = 1
    _pad_min = int(rendering.get("page_padding_min_px", 2)) if rendering else 2
    _pad_max = int(rendering.get("page_padding_max_px", 10)) if rendering else 10
    _page_pad = rng.randint(_pad_min, max(_pad_min, _pad_max))
    margin_bottom = _page_pad
    top_pad = _page_pad

    if n_eq == 1:
        eq_kind, tokens = sample_layout_for_case(case, rng)
        if tokens:
            stage, tokens = sample_completion_stage_for_case(case, tokens, rng, weights=completion_weights)
            stage_names.append(stage.value)
            pre_knob_tokens = tokens
            if scene_cfg is not None:
                tokens = _apply_scene_knobs(tokens, rng, scene_cfg)
            # A5: derive eq kind from the FINAL drawn tokens so a wrong-operator
            # swap is reflected as-drawn. No-op when no operator was swapped.
            eq_kinds.append(_eq_kind_from_final_tokens(pre_knob_tokens, tokens, eq_kind).value)
            a, gt, _ = _draw_equation_block(
                canvas, rng,
                symbols_by_glyph_key, symbols_by_glyph_key_fallback,
                width, height, top_pad, height - top_pad - margin_bottom,
                tokens, 0,
                center_in_full_canvas_vertically=True,
                style=style, rendering=rendering, scene_case=case,
                pool_root=pool_root, min_source_quality=min_source_quality,
            )
            annotations.extend(a)
            gt_tokens.extend(gt)
    else:
        slot_h = max(80, (height - top_pad - margin_bottom) // n_eq)
        cursor = top_pad
        for eq_idx in range(n_eq):
            eq_kind, tokens = sample_layout_for_case(case, rng)
            if not tokens:
                continue
            stage, tokens = sample_completion_stage_for_case(case, tokens, rng, weights=completion_weights)
            stage_names.append(stage.value)
            pre_knob_tokens = tokens
            if scene_cfg is not None:
                tokens = _apply_scene_knobs(tokens, rng, scene_cfg)
            # A5: derive eq kind from the FINAL drawn tokens so a wrong-operator
            # swap is reflected as-drawn. No-op when no operator was swapped.
            eq_kinds.append(_eq_kind_from_final_tokens(pre_knob_tokens, tokens, eq_kind).value)
            a, gt, row_max = _draw_equation_block(
                canvas, rng,
                symbols_by_glyph_key, symbols_by_glyph_key_fallback,
                width, height, cursor, slot_h,
                tokens, eq_idx,
                center_in_full_canvas_vertically=False,
                style=style, rendering=rendering, scene_case=case,
                pool_root=pool_root, min_source_quality=min_source_quality,
            )
            annotations.extend(a)
            gt_tokens.extend(gt)
            cursor = min(height - margin_bottom, row_max + rng.randint(14, 26))
            if cursor + slot_h > height - margin_bottom and eq_idx + 1 < n_eq:
                break

    return annotations, gt_tokens, eq_kinds, stage_names


def build_synthetic_yolo_dataset(
    config: DataPrepConfig,
    case: SceneCase,
    split_name: str = "train",
    images_per_split: int = 1500,
    canvas_size: Tuple[int, int] = (512, 512),
    start_index: int = 0,
    completion_weights: dict | None = None,
    rendering: dict | None = None,
    preset_cfg: PresetConfig | None = None,
    scene_cfg: SceneConfig | None = None,
    out_dir: "Path | None" = None,
    source_pool: str = "auto",
    min_source_quality: float = 0.0,
) -> Dict[str, Path]:
    """Generate synthetic YOLO scenes for one case/split.

    out_dir: when provided, images/labels/ground_truth go into
    ``out_dir/{split_name}/{images,labels,ground_truth}``.
    Defaults to ``config.synthetic_dir / case.value`` (legacy layout).
    """
    symbol_manifest_path = config.yolo_dir / "symbol_assets_manifest.csv"
    symbol_df_full = _load_symbol_rows(symbol_manifest_path)
    if "glyph_key" not in symbol_df_full.columns:
        raise ValueError("symbol_assets_manifest.csv must include a glyph_key column (re-run validate).")

    fallback_by_glyph = {
        str(gk): frame.reset_index(drop=True)
        for gk, frame in symbol_df_full.groupby("glyph_key", dropna=False)
        if str(gk) != "" and str(gk).lower() != "nan"
    }

    symbol_df = symbol_df_full[symbol_df_full["split"] == split_name].reset_index(drop=True)
    if symbol_df.empty:
        raise ValueError(f"No rows for split '{split_name}' in symbol assets.")

    class_to_idx = {name: i for i, name in enumerate(YOLO_CLASS_NAMES)}
    rng = random.Random(config.random_seed + sum(ord(c) for c in split_name) + sum(ord(c) for c in case.value))

    # Support new flat timestamped layout (out_dir) or legacy per-case layout.
    _case_root = out_dir if out_dir is not None else (config.synthetic_dir / case.value)
    images_dir = _case_root / split_name / "images"
    labels_dir = _case_root / split_name / "labels"
    gt_dir = _case_root / split_name / "ground_truth"
    ensure_dirs([images_dir, labels_dir, gt_dir])

    symbols_by_glyph_key = {
        str(gk): frame.reset_index(drop=True)
        for gk, frame in symbol_df.groupby("glyph_key", dropna=False)
        if str(gk) != "" and str(gk).lower() != "nan"
    }

    _preset = preset_cfg if preset_cfg is not None else PresetConfig()
    _scene = scene_cfg if scene_cfg is not None else SceneConfig()

    manifest_rows: List[Dict[str, object]] = []
    width, height = canvas_size
    cell_w, cell_h = 28, 28

    # Resolve pool root once per dataset build (avoids per-crop filesystem checks).
    _project_root = config.yolo_dir.parent.parent.parent.parent  # data/generated/processed/yolo -> generated/processed -> generated -> data -> project root
    _pool_root: Optional[Path] = None
    if source_pool != "legacy":
        try:
            _pool_root = resolve_pool_root(_project_root, source_pool)
        except (RuntimeError, ValueError):
            _pool_root = None  # degrade gracefully to legacy pool
    _min_quality = float(min_source_quality)

    for i in progress_iter(
        range(start_index, start_index + images_per_split),
        total=images_per_split,
        desc=f"Synth {split_name} scenes",
    ):
        canvas = np.full((height, width), 255, dtype=np.uint8)

        # Scene-level crowding decision
        crowded = rng.random() < _scene.crowdness_prob

        # Sample style from single preset
        style = make_style(rng, _preset, _scene, crowded=crowded)

        annotations, gt_toks, eq_kinds, completion_stages = _render_equations_on_canvas(
            canvas, rng,
            symbols_by_glyph_key, fallback_by_glyph,
            width, height, cell_w, cell_h,
            style=style, case=case,
            completion_weights=completion_weights, rendering=rendering,
            scene_cfg=_scene,
            pool_root=_pool_root,
            min_source_quality=_min_quality,
        )

        # Scene-level rotation (section D) -- uses asymmetric min/max bounds from config.
        if _scene.scene_rotation_max_deg != 0.0 or _scene.scene_rotation_min_deg != 0.0:
            angle = rng.uniform(_scene.scene_rotation_min_deg, _scene.scene_rotation_max_deg)
            canvas, annotations, gt_toks = _rotate_canvas_and_bboxes(canvas, annotations, angle, gt_toks)

        image_name = f"{case.value}_{split_name}_{i:06d}.png"
        label_name = f"{case.value}_{split_name}_{i:06d}.txt"
        gt_name = f"{case.value}_{split_name}_{i:06d}.gt.json"

        from ..data_pipeline.preprocessing import preprocess_for_pipeline  # noqa: PLC0415
        Image.fromarray(preprocess_for_pipeline(canvas)).save(images_dir / image_name)
        (labels_dir / label_name).write_text("\n".join(annotations), encoding="utf-8")

        kind_primary = eq_kinds[0] if len(eq_kinds) == 1 else "multi"
        gt_payload: Dict[str, Any] = {
            "equation_type": kind_primary,
            "completion_stage": completion_stages[0] if completion_stages else "full",
            "split": split_name,
            "case": case.value,
            "symbols": gt_toks,
        }
        if len(eq_kinds) > 1:
            gt_payload["equation_kinds"] = eq_kinds
        (gt_dir / gt_name).write_text(json.dumps(gt_payload, indent=2), encoding="utf-8")

        manifest_rows.append({
            "image": str(images_dir / image_name),
            "label": str(labels_dir / label_name),
            "ground_truth": str(gt_dir / gt_name),
            "box_count": len(annotations),
            "case": case.value,
        })

    manifest_path = (
        _case_root / f"{case.value}_{split_name}_manifest.csv"
        if out_dir is not None
        else _case_root / f"{split_name}_manifest.csv"
    )
    new_df = pd.DataFrame(manifest_rows)
    if start_index > 0 and manifest_path.exists():
        new_df = pd.concat([pd.read_csv(manifest_path), new_df], ignore_index=True)
    new_df.to_csv(manifest_path, index=False)
    _top_dir = out_dir if out_dir is not None else config.synthetic_dir
    write_json(class_to_idx, _top_dir / "class_to_idx.json", sort_keys=False)
    write_json(
        {
            "split": split_name,
            "case": case.value,
            "images": images_per_split,
            "canvas_size": [width, height],
            "classes": list(YOLO_CLASS_NAMES),
            "yolo_class_count": len(YOLO_CLASS_NAMES),
        },
        _case_root / (
            f"{case.value}_{split_name}_meta.json"
            if out_dir is not None
            else f"{split_name}_meta.json"
        ),
    )

    return {"manifest": manifest_path, "class_to_idx": _top_dir / "class_to_idx.json"}


# ---------------------------------------------------------------------------
# Single-scene render wrapper (set-maker target generator)
# ---------------------------------------------------------------------------

# Module-level pool cache so the symbol DataFrames are loaded once across many
# single-scene renders (the set-maker generates one target per worklist item).
# Keyed by (manifest_path, source_pool); value mirrors the per-build pool setup
# in ``build_synthetic_yolo_dataset`` (symbols-by-glyph-key + glyph fallback +
# resolved pool root).
_SINGLE_SCENE_POOL_CACHE: Dict[Tuple[str, str], Tuple[
    Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Optional[Path]
]] = {}


def _load_single_scene_pool(
    config: DataPrepConfig,
    source_pool: str,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Optional[Path]]:
    """Load (and cache) the symbol pool for single-scene rendering.

    Loads the full symbol-assets manifest (no split filter — an interactive
    target generator does not partition by split) and groups it by glyph_key the
    same way ``build_synthetic_yolo_dataset`` does. Resolves the on-disk pool root
    once. Subsequent calls with the same manifest + source_pool reuse the cache.
    """
    symbol_manifest_path = config.yolo_dir / "symbol_assets_manifest.csv"
    cache_key = (str(symbol_manifest_path), str(source_pool))
    cached = _SINGLE_SCENE_POOL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    symbol_df_full = _load_symbol_rows(symbol_manifest_path)
    if "glyph_key" not in symbol_df_full.columns:
        raise ValueError(
            f"{symbol_manifest_path} must include a glyph_key column (re-run validate)."
        )
    symbols_by_glyph_key = {
        str(gk): frame.reset_index(drop=True)
        for gk, frame in symbol_df_full.groupby("glyph_key", dropna=False)
        if str(gk) != "" and str(gk).lower() != "nan"
    }
    if not symbols_by_glyph_key:
        raise ValueError(f"No symbol rows found in {symbol_manifest_path}.")
    # Fallback table is identical here (no split filter), reused for any missing key.
    fallback_by_glyph = symbols_by_glyph_key

    pool_root: Optional[Path] = None
    if source_pool != "legacy":
        try:
            pool_root = resolve_pool_root(config.project_root, source_pool)
        except (RuntimeError, ValueError):
            pool_root = None  # degrade gracefully to legacy pool

    value = (symbols_by_glyph_key, fallback_by_glyph, pool_root)
    _SINGLE_SCENE_POOL_CACHE[cache_key] = value
    return value


def render_single_scene(
    case: "SceneCase | str",
    seed: int,
    completion_stage: str,
    *,
    config: DataPrepConfig | None = None,
    generation_config: GenerationConfig | None = None,
):
    """Render exactly one synthetic scene in memory and return it as a TargetScene.

    Public wrapper over the same ``_render_equations_on_canvas`` path used by
    ``build_synthetic_yolo_dataset`` (DRY): the symbol pool is loaded once and
    cached, no files are written, no scene rotation is applied, and the scene
    holds a single equation (so every symbol carries ``equation_idx == 0``). The
    ``completion_stage`` is pinned by the caller (the matching bucket is forced to
    weight 1.0) rather than sampled randomly, so partial-exercise quotas are
    satisfiable. The ground-truth payload matches the per-symbol shape written by
    the dataset builder (``fine_label``/``glyph_key``/``yolo_class``/``row_index``/
    ``col_index``/``bbox``/``equation_idx``).

    Args:
        case: A ``SceneCase`` enum member or its string value (e.g. "addition").
        seed: Deterministic per-scene seed; the same seed reproduces the scene.
        completion_stage: A completion-stage bucket string ("full", "done_80",
            "done_60", "done_40", "done_20"). Forced via single-bucket weights.
        config: Optional ``DataPrepConfig``; resolved from the project root when
            omitted.
        generation_config: Optional ``GenerationConfig`` supplying preset/scene/
            rendering blocks and ``source_pool``; defaults are used when omitted.

    Returns:
        A ``TargetScene`` with its ``symbols`` list and a clean typeset
        ``reference`` string for the annotator.
    """
    from ..core.config import resolve_project_root  # noqa: PLC0415
    from ..data_pipeline.preprocessing import preprocess_for_pipeline  # noqa: PLC0415
    from ..inference.palette import display_glyph  # noqa: PLC0415
    from ..setmaker.types import TargetScene, TargetSymbol  # noqa: PLC0415

    scene_case = case if isinstance(case, SceneCase) else SceneCase(str(case))
    if config is None:
        config = DataPrepConfig.from_project_root(resolve_project_root(Path(__file__)))
    gen_cfg = generation_config if generation_config is not None else GenerationConfig()

    symbols_by_glyph_key, fallback_by_glyph, pool_root = _load_single_scene_pool(
        config, gen_cfg.source_pool
    )

    # Pin the completion stage by forcing its bucket to weight 1.0. The five
    # bucket keys match GenerationConfig.completion / config.toml [generation.
    # completion]; buckets the per-case sampler does not recognise fall through
    # to "full".
    pinned_weights = {b: 0.0 for b in ("full", "done_80", "done_60", "done_40", "done_20")}
    pinned_weights[completion_stage] = 1.0

    width = height = 512
    rng = random.Random(int(seed))
    canvas = np.full((height, width), 255, dtype=np.uint8)
    style = make_style(rng, gen_cfg.preset, gen_cfg.scene, crowded=False)

    # Single equation, pinned stage, no scene rotation (the wrapper never rotates).
    _annotations, gt_toks, eq_kinds, stage_names = _render_equations_on_canvas(
        canvas, rng,
        symbols_by_glyph_key, fallback_by_glyph,
        width, height, 28, 28,
        style=style, case=scene_case,
        completion_weights=pinned_weights, rendering=gen_cfg.rendering,
        scene_cfg=gen_cfg.scene,
        pool_root=pool_root,
        min_source_quality=float(gen_cfg.min_source_quality),
    )

    symbols = [
        TargetSymbol(
            fine_label=str(t["fine_label"]),
            glyph_key=str(t["glyph_key"]),
            yolo_class=str(t["yolo_class"]),
            row_index=int(t["row_index"]),
            col_index=int(t["col_index"]),
            equation_idx=int(t["equation_idx"]),
            bbox=tuple(float(v) for v in t["bbox"]),  # type: ignore[arg-type]
        )
        for t in gt_toks
    ]

    # Clean typeset reference: symbols laid out row-major, columns space-separated.
    reference = _typeset_reference(symbols, display_glyph)

    return TargetScene(
        case=scene_case.value,
        equation_type=eq_kinds[0] if eq_kinds else "ood_unknown",
        completion_stage=stage_names[0] if stage_names else "full",
        seed=int(seed),
        reference=reference,
        symbols=symbols,
    )


def _typeset_reference(symbols, glyph_fn) -> str:
    """Build a clean one-line-per-row typeset string from target symbols.

    Rows are ordered by ``row_index``; within a row, symbols are ordered by
    ``col_index`` and joined with spaces. Each symbol renders via ``glyph_fn``
    (the palette glyph map), so digits show as themselves and operators/structural
    tokens show their display glyph. Rows are joined with " / " to keep the
    reference a single short hint string.
    """
    if not symbols:
        return ""
    rows: Dict[int, List[Tuple[int, str]]] = {}
    for s in symbols:
        rows.setdefault(s.row_index, []).append((s.col_index, glyph_fn(s.fine_label)))
    row_strings: List[str] = []
    for r in sorted(rows):
        cols = sorted(rows[r], key=lambda c: c[0])
        row_strings.append(" ".join(g for _, g in cols).strip())
    return " / ".join(rs for rs in row_strings if rs)


# ---------------------------------------------------------------------------
# Pool-free target geometry (set-maker target generator)
# ---------------------------------------------------------------------------
#
# ``build_target_scene`` (below) produces the SAME ``TargetScene`` contract that
# ``render_single_scene`` returns, but WITHOUT loading the symbol pool or
# blitting any glyph pixels. A fresh clone of the repo (no ``data/raw`` pool, no
# ``symbol_assets_manifest.csv``) can therefore still generate targets for the
# annotator to redraw. The per-symbol bbox is derived from the LAYOUT grid
# geometry instead of from a throwaway render: the matcher aligns drafts to the
# target by centroid, so a per-cell rectangle is sufficient (tight glyph-pixel
# bboxes are not needed).


def _layout_token_rects(
    tokens: List[LayoutToken],
    rng: random.Random,
    rendering: dict | None,
    style: HandwritingStyle,
    canvas_w: int,
    canvas_h: int,
) -> List[Tuple[LayoutToken, Tuple[float, float, float, float]]]:
    """Compute one per-cell pixel rect per token from layout grid geometry alone.

    This is the *planning* half of ``synth_pool._draw_equation_block`` (the grid
    math at its lines 570-663) with the pixel-blitting half removed. It mirrors
    that math exactly so the placement of every token matches the synthetic
    renderer's intended layout, but it never touches the symbol pool:

    1. ``layout_bounds`` gives the token grid extents (min/max row + col);
    2. ``base_cell_px`` + sampled ``gap_x``/``gap_y`` give the column/row pitch
       (``x_step``/``y_step``), exactly as the renderer samples them;
    3. each token gets a layout-space rect centred on its grid cell, sized by
       ``base_cell_px * scale`` (result_bar spans its operand columns; carries
       and borrows sit slightly lower and smaller, matching the renderer);
    4. a single global scale ``s`` plus offset ``(ox, oy)`` centres the whole
       block on the 512x512 canvas (the ``center_in_full_canvas_vertically``
       branch the single-scene path always uses), and each rect is mapped to
       pixel space as ``(ox + s*x0, oy + s*y0, ox + s*x1, oy + s*y1)``.

    The rng draw order matches ``_draw_equation_block`` so the geometry is
    self-consistently deterministic for a fixed seed: ``gap_x``, ``gap_y``, then
    per carry/borrow token a vertical-jitter draw followed by a scale draw, per
    operator token a scale draw, then the final ``ox``/``oy`` nudge.

    Args:
        tokens: The (already completion-trimmed, scene-knob-applied) layout
            tokens for one equation. ``div_bracket`` is treated as a per-cell
            token like any glyph (its centroid sits on its grid cell, which is
            what the matcher uses).
        rng: Deterministic per-scene rng (draws mirror the renderer's order).
        rendering: The rendering knob dict (``GenerationConfig.rendering``);
            ``None`` falls back to the same hard defaults the renderer uses.
        style: The per-scene handwriting style (supplies ``crowding_factor``).
        canvas_w: Canvas width in pixels (512 for targets).
        canvas_h: Canvas height in pixels (512 for targets).

    Returns:
        A list of ``(token, (x0, y0, x1, y1))`` pairs in canvas pixel space,
        one per input token, in the input order. Empty when ``tokens`` is empty.
    """
    if not tokens:
        return []

    min_r, max_r, min_c, max_c = layout_bounds(tokens)
    base_cell = int(rendering.get("base_cell_px", 44)) if rendering else 44
    _gap_x_min = int(rendering.get("gap_x_min", 6)) if rendering else 6
    _gap_x_max = int(rendering.get("gap_x_max", 13)) if rendering else 13
    _gap_y_min = int(rendering.get("gap_y_min", 10)) if rendering else 10
    _gap_y_max = int(rendering.get("gap_y_max", 19)) if rendering else 19
    gap_x = rng.randint(_gap_x_min, _gap_x_max)
    gap_y = rng.randint(_gap_y_min, _gap_y_max)
    x_step = int(base_cell * style.crowding_factor) + gap_x
    y_step = base_cell + gap_y

    # Layout-space rects, one per token, in input order (so the returned list
    # lines up 1:1 with `tokens`).
    layout_rects: List[Tuple[float, float, float, float]] = []
    for tok in tokens:
        if tok.flattened_label == "result_bar":
            bar_c0, bar_c1 = _bar_col_span(tok, tokens)
            if bar_c0 > bar_c1:
                # Degenerate bar (no operand columns): fall back to its own cell
                # so every token still has a rect and the list stays 1:1.
                bar_c0 = bar_c1 = tok.col
            x_left_f = float((bar_c0 - min_c) * x_step)
            x_right_f = float((bar_c1 - min_c) * x_step + x_step)
            cy_mid = float((tok.row - min_r) * y_step + y_step * 0.5)
            bar_h = max(3.0, base_cell * 0.14)
            y_top_f = cy_mid - bar_h / 2.0
            y_bot_f = cy_mid + bar_h / 2.0
            layout_rects.append((x_left_f, y_top_f, x_right_f, y_bot_f))
        else:
            cx = (tok.col - min_c) * x_step + x_step / 2.0
            cy = (tok.row - min_r) * y_step + y_step / 2.0
            if tok.yolo_class_name in ("digit_carry", "digit_borrow"):
                cy += (0.18 + rng.uniform(-0.05, 0.05)) * y_step
            if rendering:
                if tok.yolo_class_name in ("digit_carry", "digit_borrow"):
                    _eff_scale = rng.uniform(
                        rendering.get("carry_borrow_scale_min", 0.50),
                        rendering.get("carry_borrow_scale_max", 0.70),
                    )
                elif tok.yolo_class_name == "operator":
                    _op_min = rendering.get("operator_scale_min", rendering.get("operator_scale", tok.token_scale))
                    _op_max = rendering.get("operator_scale_max", _op_min)
                    _eff_scale = rng.uniform(_op_min, _op_max)
                else:
                    _eff_scale = tok.token_scale
            else:
                _eff_scale = tok.token_scale
            tw = max(8.0, float(base_cell) * float(_eff_scale))
            th = tw
            x0 = cx - tw / 2.0
            y0 = cy - th / 2.0
            layout_rects.append((x0, y0, x0 + tw, y0 + th))

    bmin_x = min(r[0] for r in layout_rects)
    bmin_y = min(r[1] for r in layout_rects)
    bmax_x = max(r[2] for r in layout_rects)
    bmax_y = max(r[3] for r in layout_rects)
    bb_w = max(bmax_x - bmin_x, 1.0)
    bb_h = max(bmax_y - bmin_y, 1.0)

    # Single-equation centring (center_in_full_canvas_vertically=True branch),
    # which is the only branch the single-scene / target path uses.
    margin = 0.06
    s = min(
        (canvas_w * (1.0 - 2 * margin)) / bb_w,
        (canvas_h * (1.0 - 2 * margin)) / bb_h,
        5.5,
    )
    s = max(s, 0.45)
    ox = (canvas_w - s * bb_w) / 2.0 - s * bmin_x
    oy = (canvas_h - s * bb_h) / 2.0 - s * bmin_y
    ox += float(rng.randint(-2, 2))
    oy += float(rng.randint(-2, 2))

    out: List[Tuple[LayoutToken, Tuple[float, float, float, float]]] = []
    for tok, (x0, y0, x1, y1) in zip(tokens, layout_rects):
        px0 = ox + s * x0
        py0 = oy + s * y0
        px1 = ox + s * x1
        py1 = oy + s * y1
        # Clamp into the canvas and guarantee a non-degenerate box (x1 > x0,
        # y1 > y0) so every emitted bbox is a valid annotation target.
        px0 = float(min(max(px0, 0.0), float(canvas_w)))
        py0 = float(min(max(py0, 0.0), float(canvas_h)))
        px1 = float(min(max(px1, 0.0), float(canvas_w)))
        py1 = float(min(max(py1, 0.0), float(canvas_h)))
        if px1 <= px0:
            px1 = min(float(canvas_w), px0 + 1.0)
            if px1 <= px0:
                px0 = max(0.0, px1 - 1.0)
        if py1 <= py0:
            py1 = min(float(canvas_h), py0 + 1.0)
            if py1 <= py0:
                py0 = max(0.0, py1 - 1.0)
        out.append((tok, (px0, py0, px1, py1)))
    return out


def build_target_scene(
    case: "SceneCase | str",
    seed: int,
    completion_stage: str,
    gen_cfg: GenerationConfig | None = None,
):
    """Build one target scene from layout geometry alone (no symbol pool).

    Pool-free sibling of :func:`render_single_scene`: it runs the SAME token
    pipeline ``render_single_scene`` runs MINUS the pool load and the glyph
    blitting, then derives each symbol's bbox from the layout grid via
    :func:`_layout_token_rects`. The returned object is the SAME
    :class:`~src.setmaker.types.TargetScene` shape ``render_single_scene``
    returns (every ``TargetSymbol`` carries the seven keys
    ``fine_label``/``glyph_key``/``yolo_class``/``row_index``/``col_index``/
    ``equation_idx``/``bbox`` and the scene carries
    ``case``/``equation_type``/``completion_stage``/``seed``/``reference``/
    ``symbols``), so the set-maker exporters, matcher, and frontend are
    unchanged.

    Token pipeline (mirrors ``_render_equations_on_canvas`` for the single,
    n_eq == 1 equation the set-maker draws): consume the per-scene page-padding
    draw (kept so the layout sample lines up with the renderer for a given
    seed), ``sample_layout_for_case`` for the grid tokens, then the SAME
    completion-stage trimming (``sample_completion_stage_for_case`` with the
    completion bucket pinned to the caller's ``completion_stage``) and the SAME
    scene-knob transforms (``_apply_scene_knobs``). The equation kind is derived
    from the FINAL drawn tokens via ``_eq_kind_from_final_tokens`` exactly as the
    renderer does. No file is written and no pool path is read.

    Args:
        case: A ``SceneCase`` enum member or its string value (e.g. "addition").
        seed: Deterministic per-scene seed; the same seed reproduces the scene.
        completion_stage: A completion-stage bucket string ("full", "done_80",
            "done_60", "done_40", "done_20"). Pinned via single-bucket weights;
            an unrecognised bucket for the case falls through to "full".
        gen_cfg: Optional ``GenerationConfig`` supplying the preset/scene/
            rendering blocks; ``GenerationConfig()`` defaults are used when
            omitted (no ``config.toml`` or project root needed).

    Returns:
        A deterministic ``TargetScene`` for the given ``(case, seed,
        completion_stage)``, with bboxes in 512 px space, built without the pool.
    """
    from ..inference.palette import display_glyph  # noqa: PLC0415
    from ..setmaker.types import TargetScene, TargetSymbol  # noqa: PLC0415

    scene_case = case if isinstance(case, SceneCase) else SceneCase(str(case))
    cfg = gen_cfg if gen_cfg is not None else GenerationConfig()

    width = height = 512
    rng = random.Random(int(seed))
    # Style is needed only for crowding_factor in the grid pitch; no rng for
    # rotation is consumed here (make_style stores rotation bounds as-is).
    style = make_style(rng, cfg.preset, cfg.scene, crowded=False)

    # Pin the completion stage by forcing its bucket to weight 1.0 (same five
    # bucket keys as GenerationConfig.completion / [generation.completion]).
    pinned_weights = {b: 0.0 for b in ("full", "done_80", "done_60", "done_40", "done_20")}
    pinned_weights[completion_stage] = 1.0

    # --- Token pipeline (mirror of _render_equations_on_canvas, n_eq == 1) ---
    rendering = cfg.rendering
    _pad_min = int(rendering.get("page_padding_min_px", 2)) if rendering else 2
    _pad_max = int(rendering.get("page_padding_max_px", 10)) if rendering else 10
    # Consume the page-padding draw so the layout sample below lines up with the
    # renderer for a given seed (the value itself is not needed for geometry).
    _ = rng.randint(_pad_min, max(_pad_min, _pad_max))

    eq_kind, tokens = sample_layout_for_case(scene_case, rng)
    eq_kind_value = EquationKind.ood_unknown.value
    completion_stage_value = "full"
    symbols: List[Any] = []
    if tokens:
        stage, tokens = sample_completion_stage_for_case(
            scene_case, tokens, rng, weights=pinned_weights
        )
        completion_stage_value = stage.value
        pre_knob_tokens = tokens
        tokens = _apply_scene_knobs(tokens, rng, cfg.scene)
        eq_kind_value = _eq_kind_from_final_tokens(pre_knob_tokens, tokens, eq_kind).value

        rects = _layout_token_rects(tokens, rng, rendering, style, width, height)
        symbols = [
            TargetSymbol(
                fine_label=str(tok.flattened_label),
                glyph_key=str(tok.glyph_key),
                yolo_class=str(tok.yolo_class_name),
                row_index=int(tok.row),
                col_index=int(tok.col),
                equation_idx=0,
                bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
            )
            for tok, bbox in rects
        ]

    reference = _typeset_reference(symbols, display_glyph)

    return TargetScene(
        case=scene_case.value,
        equation_type=eq_kind_value,
        completion_stage=completion_stage_value,
        seed=int(seed),
        reference=reference,
        symbols=symbols,
    )
