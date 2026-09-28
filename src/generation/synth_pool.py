"""Pool loading and tile rendering helpers for synthetic YOLO scene generation.

Extracted from synth_yolo.py to keep that module focused on the dataset
orchestrator (build_synthetic_yolo_dataset). Callers that previously imported
these helpers from synth_yolo can now import from synth_pool directly, or
continue to use synth_yolo which re-exports them for back-compat.
"""
from __future__ import annotations

import dataclasses
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .handwriting_style import HandwritingStyle
from .layouts import CARRY_BORROW_TOKEN_SCALE, LayoutToken, SceneCase, layout_bounds
from .strokes import draw_handwritten_bar, draw_handwritten_bracket
from ..core.ontology import YOLO_NAME_TO_ID
from ..core.ink import INK_BINARIZE_THRESHOLD
from .geometry import rotate_tile


# ---------------------------------------------------------------------------
# Manifest loader
# ---------------------------------------------------------------------------


def _load_symbol_rows(symbol_manifest_path: Path) -> pd.DataFrame:
    if not symbol_manifest_path.exists():
        raise FileNotFoundError(f"Missing symbol manifest: {symbol_manifest_path}")
    df = pd.read_csv(
        symbol_manifest_path,
        dtype={"label": str, "split": str, "path": str, "glyph_key": str},
        low_memory=False,
    )
    required = {"path", "label", "split"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Symbol manifest missing columns: {sorted(missing)}")
    df["label"] = df["label"].astype(str)
    return df


# ---------------------------------------------------------------------------
# Per-glyph tile helpers
# ---------------------------------------------------------------------------


def _apply_broken_stroke(tile: np.ndarray, rng: random.Random, band_frac: float = 0.125) -> np.ndarray:
    """Erase a random horizontal band from the tile to simulate a stroke gap."""
    h, w = tile.shape
    band_h = max(1, int(h * band_frac))
    y_start = rng.randint(h // 4, max(h // 4 + 1, h - h // 4 - band_h))
    result = tile.copy()
    result[y_start:y_start + band_h, w // 5: w - w // 5] = 255
    return result


def _paste_symbol(
    canvas: np.ndarray,
    tile: np.ndarray,
    x0: int,
    y0: int,
    cell_w: int,
    cell_h: int,
    style: HandwritingStyle,
    rng: random.Random,
    morph_close_kernel: int = 3,
    glyph_stroke_target_px: float = 0.0,
    broken_stroke_band_frac: float = 0.125,
    binarize_threshold: int = INK_BINARIZE_THRESHOLD,
) -> Tuple[int, int, int, int]:
    # 1. Resize to cell size
    tile = np.array(Image.fromarray(tile).resize((cell_w, cell_h), resample=Image.Resampling.BICUBIC))
    # 2. Per-glyph broken-stroke (gap simulation)
    if rng.random() < style.glyph_broken_stroke_prob:
        tile = _apply_broken_stroke(tile, rng, band_frac=broken_stroke_band_frac)
    # 3. Rotation — sampled independently for every glyph
    theta = rng.uniform(style.rot_deg_min, style.rot_deg_max)
    if abs(theta) > 0.5:
        tile, (dx, dy) = rotate_tile(tile, theta)
        x0 += dx
        y0 += dy
        cell_w = tile.shape[1]
        cell_h = tile.shape[0]
    # 3b. Binarize: kill grey halos from BICUBIC interpolation, force solid black ink
    # Binarize + morphological close to fill interior gaps from bicubic upscale
    # Threshold 165 (INK_BINARIZE_THRESHOLD) is tighter than the 200 legacy value to
    # cut BICUBIC halo without clipping stroke body -- boldness regression fix.
    if binarize_threshold <= 0:
        # iter11 SOFT/RAW mode (binarize_threshold <= 0): keep the CROHME grayscale gradient (soft
        # antialiased edges, like the raw crop and like real handwriting after preprocess); only lift
        # the faint BICUBIC halo to pure white. No hard black/white binarisation, no thickening.
        halo_floor = 200
        tile = np.where(tile >= halo_floor, 255, tile).astype(np.uint8)
    else:
        # HARD binarise: cut grey BICUBIC halo and force solid black ink (legacy boldness-fix path).
        bin_mask = (tile < binarize_threshold).astype(np.uint8)
        if bin_mask.any() and morph_close_kernel >= 2:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close_kernel, morph_close_kernel))
            bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        tile = np.where(bin_mask > 0, 0, 255).astype(np.uint8)
    # 3c. Per-glyph stroke-width normalise (after binarize, before paste)
    if glyph_stroke_target_px > 0:
        from ..data_pipeline.preprocessing import normalise_glyph_stroke_width
        tile = normalise_glyph_stroke_width(tile, glyph_stroke_target_px)
    # 4. Clamp + paste
    x0 = max(0, x0)
    y0 = max(0, y0)
    y1 = min(y0 + cell_h, canvas.shape[0])
    x1 = min(x0 + cell_w, canvas.shape[1])
    actual_h = y1 - y0
    actual_w = x1 - x0
    if actual_h > 0 and actual_w > 0:
        canvas[y0:y1, x0:x1] = np.minimum(canvas[y0:y1, x0:x1], tile[:actual_h, :actual_w])
    return x0, y0, x0 + actual_w, y0 + actual_h


def _tile_from_path(symbol_path: Path) -> np.ndarray:
    return np.array(Image.open(symbol_path).convert("L"))


# ---------------------------------------------------------------------------
# Pool root resolution
# ---------------------------------------------------------------------------


def resolve_pool_root(project_root: Path, source_pool: str) -> Path:
    """Return the data root for symbol crops based on source_pool setting.

    Resolution order for "auto":
      pool_emnist_28_hires  (112x112 LANCZOS4+unsharp crops)
      pool_emnist_28_clean  (28x28 threshold+close normalised crops)
      pool_emnist_28  (legacy EMNIST+28px raw crops)

    For explicit settings ("hires", "clean", "legacy", "crohme") the named
    folder is used directly; if it does not exist a RuntimeError is raised.

    "crohme" points to data/raw/pool_crohme_128/ (49 740 PNGs, 14 label
    folders matching the pool-folder ontology, 128x128 grayscale white-bg).
    """
    data = project_root / "data" / "raw"
    candidates: dict[str, Path] = {
        "hires": data / "pool_emnist_28_hires",
        "clean": data / "pool_emnist_28_clean",
        "legacy": data / "pool_emnist_28",
        "crohme": data / "pool_crohme_128",
    }
    if source_pool == "auto":
        for key in ("hires", "clean", "legacy"):
            candidate = candidates[key]
            if candidate.exists() and any(candidate.iterdir()):
                return candidate
        return candidates["legacy"]  # ultimate fallback even if missing
    elif source_pool in candidates:
        p = candidates[source_pool]
        if not p.exists():
            raise RuntimeError(
                f"source_pool={source_pool!r} but folder does not exist: {p}\n"
                "Run the appropriate pool-preparation script first."
            )
        return p
    else:
        raise ValueError(
            f"Unknown source_pool value: {source_pool!r}. "
            "Valid values: 'auto', 'legacy', 'clean', 'hires', 'crohme'."
        )


def _substitute_pool_path(original_path: Path, pool_root: Path) -> Path:
    """Map a manifest path (under data/raw/pool_emnist_28/...) to the active pool root.

    The manifest stores absolute paths under data/raw/pool_emnist_28/<label>/<file>.
    When using pool_emnist_28_clean or pool_emnist_28_hires, substitute the parent root
    so the label subfolder and filename are preserved.
    """
    # original_path: .../data/raw/pool_emnist_28/<label>/<file>
    # We want: pool_root / <label> / <file>
    try:
        # Find the pool folder segment (startswith 'combined' or 'pool_emnist_28') and take everything after it.
        parts = original_path.parts
        for i, part in enumerate(parts):
            if part.startswith("combined") or part == "pool_emnist_28":
                return pool_root / Path(*parts[i + 1:])
    except Exception:
        pass
    # Fallback: assume last two parts are <label>/<file>
    return pool_root / original_path.parent.name / original_path.name


def _load_tile_with_quality(
    pool: pd.DataFrame,
    rng: random.Random,
    pool_root: Path,
    min_quality: float,
    max_retries: int = 5,
) -> np.ndarray:
    """Sample a crop from pool, score it, and retry up to max_retries times.

    Pool paths come from the symbol manifest and point to data/raw/pool_emnist_28/ (legacy pool).
    If pool_root is different (clean/hires), paths are substituted before load.
    Falls back to the best-scoring crop found if all retries fail quality gate.
    """
    from .source_quality import score_crop  # local import to avoid circular

    # Build a list of candidate row indices (shuffle to avoid always picking same crop).
    indices = list(range(len(pool)))
    rng.shuffle(indices)
    attempts = indices[:max(max_retries, len(indices))]

    best_tile: np.ndarray | None = None
    best_score = -1.0

    # STRICT MODE: only sample from pool_root. No silent fallback to manifest path.
    # If the manifest's filename doesn't exist in pool_root (e.g. CROHME naming
    # differs from EMNIST naming), sample a real file from pool_root/<label>/.
    # If pool_root/<label>/ has no files, raise loudly — we never want a
    # silent cross-pool leak.
    _label_index_cache: dict[str, list[Path]] = {}

    def _sample_from_pool_root(label: str) -> Path:
        if label not in _label_index_cache:
            label_dir = pool_root / label
            if not label_dir.exists():
                _label_index_cache[label] = []
            else:
                _label_index_cache[label] = sorted(label_dir.glob("*.png"))
        files = _label_index_cache[label]
        if not files:
            raise RuntimeError(
                f"Pool {pool_root.name!r} has no crops for label {label!r}. "
                f"Expected at {pool_root / label}. "
                f"Generation cannot proceed without falling back to a different pool. "
                f"Either (a) re-run validate / pool prep so this label is populated, "
                f"or (b) switch source_pool to one that has it."
            )
        return files[rng.randrange(len(files))]

    for idx in attempts[:max_retries]:
        row = pool.iloc[idx]
        raw_path = Path(row["path"])
        resolved = _substitute_pool_path(raw_path, pool_root)
        # Enforce: resolved MUST be under pool_root. No EMNIST fallback.
        if not (resolved.exists() and pool_root in resolved.parents):
            label = str(row.get("label", ""))
            resolved = _sample_from_pool_root(label)
        tile = _tile_from_path(resolved)
        score = score_crop(tile)
        if score > best_score:
            best_score = score
            best_tile = tile
        if score >= min_quality:
            return tile

    # All retries failed quality gate -- return best we found.
    assert best_tile is not None
    return best_tile


# ---------------------------------------------------------------------------
# Canvas-level drawing helpers
# ---------------------------------------------------------------------------


def _result_bar_fallback_tile() -> np.ndarray:
    arr = np.full((28, 28), 255, dtype=np.uint8)
    arr[12:16, 2:26] = 0
    return arr


def _yolo_box(x0: int, y0: int, x1: int, y1: int, width: int, height: int) -> Tuple[float, float, float, float]:
    cx = ((x0 + x1) / 2.0) / width
    cy = ((y0 + y1) / 2.0) / height
    bw = max((x1 - x0) / width, 1e-6)
    bh = max((y1 - y0) / height, 1e-6)
    return cx, cy, bw, bh


def _sample_symbol(pool: pd.DataFrame, rng: random.Random) -> pd.Series:
    idx = rng.randrange(len(pool))
    return pool.iloc[idx]


def _result_digit_col_span(tokens: List[LayoutToken]) -> Tuple[int, int]:
    # A: Bar spans the widest operand block (all main_* rows), not just the result row.
    mains = [t for t in tokens if t.flattened_label.startswith("main_")]
    if not mains:
        return 0, 0
    all_cols = [t.col for t in mains]
    return min(all_cols), max(all_cols)


def _bar_col_span(bar_tok: "LayoutToken", all_tokens: List["LayoutToken"]) -> Tuple[int, int]:
    """Compute col span for a result_bar token based on its segment_kind.

    - subtraction_bar / intermediate_result_bar: span the main_* tokens on the
      row directly above the bar (the partial product being subtracted).
    - Everything else (final_result_bar, None): fall back to full operand span.
    """
    if bar_tok.segment_kind in ("subtraction_bar", "intermediate_result_bar"):
        # Span across BOTH adjacent rows (above = subtractor/partial product,
        # below = remainder/partial dividend). Uses MAX width of the pair.
        row_above = bar_tok.row - 1
        row_below = bar_tok.row + 1
        cols = [
            t.col for t in all_tokens
            if t.row in (row_above, row_below) and t.flattened_label.startswith("main_")
        ]
        if cols:
            return min(cols), max(cols)
        # Fallback: neither adjacent row had main_* tokens — use full span.
    return _result_digit_col_span(all_tokens)


def _draw_borrow_cross(
    canvas: np.ndarray,
    bx0: int, by0: int, bx1: int, by1: int,
    rng: random.Random,
    canvas_h: int, canvas_w: int,
) -> None:
    cx = (bx0 + bx1) // 2
    cy = int(by0 + rng.uniform(0.35, 0.65) * (by1 - by0))
    angle_rad = math.radians(rng.uniform(-25.0, 25.0))
    half_len = max(3, int((bx1 - bx0) * rng.uniform(0.60, 0.95) / 2))
    x0_l = cx - int(half_len * math.cos(angle_rad))
    y0_l = cy - int(half_len * math.sin(angle_rad))
    x1_l = cx + int(half_len * math.cos(angle_rad))
    y1_l = cy + int(half_len * math.sin(angle_rad))
    thickness = rng.choices([1, 2, 3], weights=[2, 3, 1])[0]
    n_steps = max(abs(x1_l - x0_l), abs(y1_l - y0_l), 1) + 1
    xs = np.round(np.linspace(x0_l, x1_l, n_steps)).astype(int)
    ys = np.round(np.linspace(y0_l, y1_l, n_steps)).astype(int)
    for t_off in range(-(thickness // 2), thickness // 2 + 1):
        ys_t = np.clip(ys + t_off, 0, canvas_h - 1)
        xs_c = np.clip(xs, 0, canvas_w - 1)
        canvas[ys_t, xs_c] = np.minimum(canvas[ys_t, xs_c], np.uint8(0))


def _draw_equation_block(
    canvas: np.ndarray,
    rng: random.Random,
    symbols_by_glyph_key: Dict[str, pd.DataFrame],
    symbols_by_glyph_key_fallback: Dict[str, pd.DataFrame],
    canvas_w: int,
    canvas_h: int,
    slot_y0: int,
    slot_height: int,
    tokens: List[LayoutToken],
    eq_idx: int,
    *,
    center_in_full_canvas_vertically: bool,
    style: HandwritingStyle,
    rendering: dict | None = None,
    scene_case: SceneCase | None = None,
    pool_root: Optional[Path] = None,
    min_source_quality: float = 0.0,
) -> Tuple[List[str], List[Dict[str, Any]], int]:
    annotations: List[str] = []
    gt_tokens: List[Dict[str, Any]] = []
    if not tokens:
        return annotations, gt_tokens, slot_y0

    # --- Token mutations (C, E, F): applied before rendering ---

    # C: Per-step op_minus in long division (35% probability per step by default).
    if scene_case == SceneCase.division_long and rendering:
        _step_minus_prob = float(rendering.get("step_minus_prob", 0.35))
        if _step_minus_prob > 0.0:
            extra: List[LayoutToken] = []
            for bar_tok in tokens:
                if (
                    bar_tok.flattened_label == "result_bar"
                    and bar_tok.segment_kind == "subtraction_bar"
                    and rng.random() < _step_minus_prob
                ):
                    pp_row = bar_tok.row - 1
                    pp_cols = [t.col for t in tokens if t.row == pp_row and t.flattened_label.startswith("main_")]
                    if pp_cols:
                        minus_col = min(pp_cols) - 1
                        extra.append(LayoutToken(
                            row=pp_row,
                            col=minus_col,
                            flattened_label="op_minus",
                            glyph_key="minus",
                            yolo_class_name="operator",
                        ))
            tokens = list(tokens) + extra

    # D: long_div_step_borrow_prob — emit digit_borrow tokens for long-division subtraction steps.
    # For each subtraction step (identified by a subtraction_bar), look at the row above the bar
    # (the product/subtractor row) and the row two above (the working remainder row = minuend row).
    # Any column where the subtractor digit exceeds the minuend digit requires a borrow; emit a
    # digit_borrow token at that column in the minuend row with the requested probability.
    # Default 0.0 = no borrow tokens added (current behaviour).
    if rendering and scene_case == SceneCase.division_long:
        _ld_borrow_prob = float(rendering.get("long_div_step_borrow_prob", 0.0))
        if _ld_borrow_prob > 0.0:
            extra_borrow: list = []
            for bar_tok in tokens:
                if bar_tok.flattened_label != "result_bar" or bar_tok.segment_kind != "subtraction_bar":
                    continue
                if rng.random() >= _ld_borrow_prob:
                    continue
                subtractor_row = bar_tok.row - 1
                minuend_row_d = bar_tok.row - 2
                sub_digits = {t.col: int(t.flattened_label.split("_")[1])
                              for t in tokens if t.row == subtractor_row and t.flattened_label.startswith("main_")}
                min_digits = {t.col: int(t.flattened_label.split("_")[1])
                              for t in tokens if t.row == minuend_row_d and t.flattened_label.startswith("main_")}
                for col, sub_d in sub_digits.items():
                    min_d = min_digits.get(col, 0)
                    if sub_d > min_d:
                        extra_borrow.append(LayoutToken(
                            row=minuend_row_d,
                            col=col,
                            flattened_label=f"borrow_{min_d}",
                            glyph_key=str(min_d),
                            yolo_class_name="digit_borrow",
                            token_scale=CARRY_BORROW_TOKEN_SCALE,
                        ))
            tokens = list(tokens) + extra_borrow

    # B: pp_wrong_operator_prob — replace op_plus on PP rows with op_minus (student mistake).
    # B: pp_plus_prob — when 0.0 op_plus tokens on PP rows are dropped entirely (current default).
    #    Set >0 to keep/add op_plus; pp_wrong_operator_prob then governs mis-sign probability.
    if rendering and scene_case in (SceneCase.multiplication_multi,):
        _pp_plus_prob = float(rendering.get("pp_plus_prob", 0.0))
        _pp_wrong_op_prob = float(rendering.get("pp_wrong_operator_prob", 0.0))
        if _pp_plus_prob == 0.0:
            # Default: drop all PP op_plus tokens (current behaviour — no operator shown)
            tokens = [t for t in tokens if not (
                t.flattened_label == "op_plus"
                and any(
                    other.row == t.row and other.yolo_class_name == "digit_main"
                    for other in tokens
                )
            )]
        elif _pp_wrong_op_prob > 0.0:
            # Keep op_plus but optionally flip to op_minus per row
            mutated_b: list = []
            for t in tokens:
                if t.flattened_label == "op_plus" and rng.random() < _pp_wrong_op_prob:
                    t = dataclasses.replace(
                        t,
                        flattened_label="op_minus",
                        glyph_key="minus",
                    )
                mutated_b.append(t)
            tokens = mutated_b

    # E-new: helper_operator_prob — emit an extra op_plus or op_minus at left of intermediate rows.
    # Applies to long-division step rows (row above subtraction_bar) and mul-multi PP rows.
    # Default 0.0 = none added (current behaviour).
    if rendering:
        _helper_op_prob = float(rendering.get("helper_operator_prob", 0.0))
        if _helper_op_prob > 0.0:
            extra_helper: list = []
            # Collect rows that are "intermediate" (PP digit rows and long-div step product rows)
            intermediate_rows: set = set()
            for t in tokens:
                if t.flattened_label == "result_bar":
                    if t.segment_kind in ("subtraction_bar",):
                        intermediate_rows.add(t.row - 1)
                    elif t.segment_kind in ("intermediate_result_bar",):
                        # rows between bar1 (row 2) and bar2 are PP rows
                        pass
            # Also collect mul-multi PP rows by looking for op_plus/op_minus already there
            for t in tokens:
                if t.yolo_class_name == "operator" and t.flattened_label in ("op_plus", "op_minus"):
                    intermediate_rows.add(t.row)
            # Also collect rows with main digits that sit between two bars (PP in mul-multi)
            bar_rows = {t.row for t in tokens if t.flattened_label == "result_bar"}
            if bar_rows:
                bar_min = min(bar_rows)
                bar_max = max(bar_rows)
                for t in tokens:
                    if t.yolo_class_name == "digit_main" and bar_min < t.row < bar_max:
                        intermediate_rows.add(t.row)
            # Emit one helper op per qualifying row (not already having an operator)
            existing_op_rows = {t.row for t in tokens if t.yolo_class_name == "operator"}
            for row in intermediate_rows:
                if row in existing_op_rows:
                    continue
                if rng.random() >= _helper_op_prob:
                    continue
                row_main_cols = [t.col for t in tokens if t.row == row and t.yolo_class_name == "digit_main"]
                if not row_main_cols:
                    continue
                op_label = rng.choice(["op_plus", "op_minus"])
                op_glyph = "plus" if op_label == "op_plus" else "minus"
                extra_helper.append(LayoutToken(
                    row=row,
                    col=min(row_main_cols) - 1,
                    flattened_label=op_label,
                    glyph_key=op_glyph,
                    yolo_class_name="operator",
                    token_scale=1.0,
                ))
            tokens = list(tokens) + extra_helper

    # E: wrong_carry_col_prob — shift carry token col by ±1 with this probability.
    # F: missing_carry_prob — drop carry token entirely with this probability.
    if rendering:
        _wrong_carry_col_prob = float(rendering.get("wrong_carry_col_prob", 0.13))
        _missing_carry_prob = float(rendering.get("missing_carry_prob", 0.15))
        if _wrong_carry_col_prob > 0.0 or _missing_carry_prob > 0.0:
            mutated: List[LayoutToken] = []
            for tok in tokens:
                if tok.yolo_class_name == "digit_carry":
                    if rng.random() < _missing_carry_prob:
                        continue  # F: drop the token
                    if rng.random() < _wrong_carry_col_prob:
                        shift = rng.choice([-1, 1])
                        tok = dataclasses.replace(tok, col=tok.col + shift)  # E: shift col
                mutated.append(tok)
            tokens = mutated

    # G: wrong_carry_value_prob / wrong_borrow_value_prob / missing_borrow_prob
    if rendering:
        _wrong_carry_val_prob = float(rendering.get("wrong_carry_value_prob", 0.0))
        _wrong_borrow_val_prob = float(rendering.get("wrong_borrow_value_prob", 0.0))
        _missing_borrow_prob = float(rendering.get("missing_borrow_prob", 0.0))
        if _wrong_carry_val_prob > 0.0 or _wrong_borrow_val_prob > 0.0 or _missing_borrow_prob > 0.0:
            mutated_g: List[LayoutToken] = []
            for tok in tokens:
                if tok.yolo_class_name == "digit_carry" and _wrong_carry_val_prob > 0.0:
                    if rng.random() < _wrong_carry_val_prob:
                        orig_digit = int(tok.flattened_label.split("_")[1])
                        new_digit = (orig_digit + rng.randint(1, 9)) % 10
                        tok = dataclasses.replace(
                            tok,
                            flattened_label=f"carry_{new_digit}",
                            glyph_key=str(new_digit),
                        )
                elif tok.yolo_class_name == "digit_borrow":
                    if _missing_borrow_prob > 0.0 and rng.random() < _missing_borrow_prob:
                        continue  # drop borrow token entirely
                    if _wrong_borrow_val_prob > 0.0 and rng.random() < _wrong_borrow_val_prob:
                        orig_digit = int(tok.flattened_label.split("_")[1])
                        new_digit = (orig_digit + rng.randint(1, 9)) % 10
                        tok = dataclasses.replace(
                            tok,
                            flattened_label=f"borrow_{new_digit}",
                            glyph_key=str(new_digit),
                        )
                mutated_g.append(tok)
            tokens = mutated_g

    # H: missing_pp_prob — drop one non-final PP row in multiplication_multi scenes.
    if rendering and scene_case == SceneCase.multiplication_multi:
        _missing_pp_prob = float(rendering.get("missing_pp_prob", 0.0))
        if _missing_pp_prob > 0.0 and rng.random() < _missing_pp_prob:
            # Identify PP digit rows: rows between bar1 (row 2) and bar2 (last result_bar row).
            bar_rows = sorted({t.row for t in tokens if t.flattened_label == "result_bar"})
            if len(bar_rows) >= 2:
                bar1 = bar_rows[0]
                bar2 = bar_rows[-1]
                # PP digit rows are even rows between bar1 and bar2 that contain digit_main tokens.
                pp_rows = sorted({
                    t.row for t in tokens
                    if t.yolo_class_name == "digit_main" and bar1 < t.row < bar2
                })
                # Exclude the last PP row (final partial product just before bar2).
                droppable_pp_rows = pp_rows[:-1]
                if droppable_pp_rows:
                    drop_row = rng.choice(droppable_pp_rows)
                    # Also drop the carry row immediately above the PP row (drop_row - 1) if it exists.
                    rows_to_drop = {drop_row, drop_row - 1}
                    tokens = [t for t in tokens if t.row not in rows_to_drop]

    # --- End token mutations ---

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

    rects: List[Tuple[float, float, float, float]] = []

    def resolve_pool(pool_key: str) -> Optional[pd.DataFrame]:
        p = symbols_by_glyph_key.get(pool_key)
        if p is not None and len(p) > 0:
            return p
        return symbols_by_glyph_key_fallback.get(pool_key)

    symbol_plans: List[Tuple[LayoutToken, float, float, int, int]] = []
    bar_plans: List[Tuple[Tuple[float, float, float, float], LayoutToken]] = []

    borrow_cols: set = {t.col for t in tokens if t.yolo_class_name == "digit_borrow"}
    minuend_row = min_r + 1

    for tok in tokens:
        if tok.flattened_label == "result_bar":
            bar_c0, bar_c1 = _bar_col_span(tok, tokens)
            if bar_c0 > bar_c1:
                continue
            x_left_f = float((bar_c0 - min_c) * x_step)
            x_right_f = float((bar_c1 - min_c) * x_step + x_step)
            cy_mid = float((tok.row - min_r) * y_step + y_step * 0.5)
            bar_h = max(3.0, base_cell * 0.14)
            y_top_f = cy_mid - bar_h / 2.0
            y_bot_f = cy_mid + bar_h / 2.0
            plan = (x_left_f, y_top_f, x_right_f, y_bot_f)
            bar_plans.append((plan, tok))
            rects.append(plan)
        else:
            cx = (tok.col - min_c) * x_step + x_step / 2.0
            cy = (tok.row - min_r) * y_step + y_step / 2.0
            # iter11: sub-cell offset packs two digits in one cell (double-digit borrow/carry).
            # GT row/col unchanged; only the pixel centre shifts within the cell.
            cx += float(getattr(tok, "cell_subpos", 0.0)) * x_step
            if tok.yolo_class_name in ("digit_carry", "digit_borrow"):
                cy += (0.18 + rng.uniform(-0.05, 0.05)) * y_step  # shift DOWN: carries sit close to PP they help compute
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
            symbol_plans.append((tok, x0, y0, int(round(tw)), int(round(th))))
            rects.append((x0, y0, x0 + tw, y0 + th))

    bmin_x = min(r[0] for r in rects)
    bmin_y = min(r[1] for r in rects)
    bmax_x = max(r[2] for r in rects)
    bmax_y = max(r[3] for r in rects)
    bb_w = max(bmax_x - bmin_x, 1.0)
    bb_h = max(bmax_y - bmin_y, 1.0)

    margin = 0.06
    # iter11: cap the canvas-fill scale so glyphs render near native crop size (~no upscale) -> strokes
    # stay ~2px (matching measured real input) instead of bicubic-doubling to 4-6px. Lower = smaller,
    # thinner, sparser (more real-like); 5.5 was the legacy fill-the-canvas value.
    _max_s = float(rendering.get("max_render_scale", 5.5)) if rendering else 5.5
    if center_in_full_canvas_vertically:
        s = min(
            (canvas_w * (1.0 - 2 * margin)) / bb_w,
            (canvas_h * (1.0 - 2 * margin)) / bb_h,
            _max_s,
        )
        s = max(s, 0.45)
        ox = (canvas_w - s * bb_w) / 2.0 - s * bmin_x
        oy = (canvas_h - s * bb_h) / 2.0 - s * bmin_y
    else:
        s = min(
            (canvas_w * (1.0 - 2 * margin)) / bb_w,
            (max(60, slot_height) * (1.0 - 2 * margin)) / bb_h,
            _max_s,
        )
        s = max(s, 0.45)
        ox = (canvas_w - s * bb_w) / 2.0 - s * bmin_x
        oy = float(slot_y0) + (float(slot_height) - s * bb_h) / 2.0 - s * bmin_y

    ox += float(rng.randint(-2, 2))
    oy += float(rng.randint(-2, 2))

    row_max_y = slot_y0

    token_px_x: Dict[int, int] = {}
    for idx, (tok, x0, y0, tw_i, th_i) in enumerate(symbol_plans):
        px_x = int(ox + s * (x0 + tw_i / 2))
        token_px_x[idx] = px_x

    def _jitter(tok: LayoutToken) -> int:
        lo = max(0, style.jitter_min_px)
        hi = max(lo, style.jitter_max_px)
        return rng.randint(lo, hi) if hi > 0 else 0

    for bar_plan, bar_tok in bar_plans:
        x0_f, y0_f, x1_f, y1_f = bar_plan
        # H: Short-bar variance — ~15% of KS1 kids draw bar shorter than full width.
        _short_bar_prob = float(rendering.get("short_bar_prob", 0.15)) if rendering else 0.15
        _short_bar_shrink = float(rendering.get("short_bar_shrink_frac", 0.20)) if rendering else 0.20
        if bar_tok.segment_kind != "subtraction_bar" and rng.random() < _short_bar_prob:
            bar_width = x1_f - x0_f
            shrink = bar_width * _short_bar_shrink
            x0_f = x0_f + shrink
            x1_f = x1_f - shrink
        _gap_p = rendering.get("bar_gap_prob", 0.04) if rendering else 0.04
        _bar_t_min = int(rendering.get("bar_thickness_min", 2)) if rendering else 2
        _bar_t_max = int(rendering.get("bar_thickness_max", 3)) if rendering else 3
        _bar_i_min = int(rendering.get("bar_intensity_min", 30)) if rendering else 30
        _bar_i_max = int(rendering.get("bar_intensity_max", 80)) if rendering else 80
        _bar_mb = float(rendering.get("bar_micro_break_prob", 0.08)) if rendering else 0.08
        _bar_yj = float(rendering.get("bar_y_jitter_sigma", 3.0)) if rendering else 3.0
        _bar_xj = float(rendering.get("bar_x_jitter_sigma", 1.5)) if rendering else 1.5
        bx0, by0, bx1, by1 = draw_handwritten_bar(
            canvas,
            int(ox + s * x0_f), int(oy + s * y0_f),
            int(ox + s * x1_f), int(oy + s * y1_f),
            style, rng, gap_prob=_gap_p,
            thickness_min=_bar_t_min, thickness_max=_bar_t_max,
            intensity_min=_bar_i_min, intensity_max=_bar_i_max,
            micro_break_prob=_bar_mb,
            y_jitter_sigma=_bar_yj, x_jitter_sigma=_bar_xj,
        )
        yolo_id = YOLO_NAME_TO_ID[bar_tok.yolo_class_name]
        yolo = _yolo_box(bx0, by0, bx1, by1, canvas_w, canvas_h)
        annotations.append(f"{yolo_id} {yolo[0]:.6f} {yolo[1]:.6f} {yolo[2]:.6f} {yolo[3]:.6f}")
        gt_tokens.append({
            "fine_label": bar_tok.flattened_label,
            "glyph_key": bar_tok.glyph_key,
            "yolo_class": bar_tok.yolo_class_name,
            "row_index": bar_tok.row,
            "col_index": bar_tok.col,
            "bbox": [bx0, by0, bx1, by1],
            "equation_idx": eq_idx,
        })
        row_max_y = max(row_max_y, by1)

    for idx, (tok, x0, y0, tw_i, th_i) in enumerate(symbol_plans):
        if tok.flattened_label == "div_bracket":
            dividend_toks = [t for t in tokens if t.row == tok.row and t.col >= 0]
            if dividend_toks:
                max_div_col = max(t.col for t in dividend_toks)
                x_horiz_end_px = int(max(0, min(canvas_w, ox + s * ((max_div_col - min_c + 1) * x_step))))
            else:
                x_horiz_end_px = int(max(0, min(canvas_w, ox + s * ((max_c - min_c + 1) * x_step))))
            x_vert_px = int(max(0, min(canvas_w - 2, ox + s * ((tok.col - min_c) * x_step + x_step * 0.5))))
            y_top_px = int(max(0, min(canvas_h - 2, oy + s * (tok.row - min_r) * y_step)))
            # C: Variable bracket arm depth.
            # bracket_depth_full_prob=1.0 (default) → always extend to bottom row (original behaviour).
            # When < 1.0 a shorter arm depth is sampled between bracket_depth_min_rows and the
            # full span, giving the appearance of a student not finishing the bracket.
            _br_full_prob = float(rendering.get("bracket_depth_full_prob", 1.0)) if rendering else 1.0
            _br_min_rows = int(rendering.get("bracket_depth_min_rows", 2)) if rendering else 2
            full_depth_rows = max_r - min_r + 1
            if _br_full_prob < 1.0 and rng.random() >= _br_full_prob:
                max_rows_available = full_depth_rows
                short_depth_rows = rng.randint(
                    min(_br_min_rows, max_rows_available),
                    max_rows_available,
                )
            else:
                short_depth_rows = full_depth_rows
            y_bot_px = min(canvas_h, int(oy + s * short_depth_rows * y_step))
            _gap_p_br = rendering.get("bracket_gap_prob", 0.04) if rendering else 0.04
            _br_t_min = int(rendering.get("bracket_thickness_min", 2)) if rendering else 2
            _br_t_max = int(rendering.get("bracket_thickness_max", 3)) if rendering else 3
            _br_i_min = int(rendering.get("bracket_intensity_min", 30)) if rendering else 30
            _br_i_max = int(rendering.get("bracket_intensity_max", 80)) if rendering else 80
            _br_mb = float(rendering.get("bracket_micro_break_prob", 0.08)) if rendering else 0.08
            _br_hyj = float(rendering.get("bracket_h_y_jitter_sigma", 2.5)) if rendering else 2.5
            _br_hxj = float(rendering.get("bracket_h_x_jitter_sigma", 1.0)) if rendering else 1.0
            _br_vxj = float(rendering.get("bracket_v_x_jitter_sigma", 1.5)) if rendering else 1.5
            bx0, by0, bx1, by1 = draw_handwritten_bracket(
                canvas,
                corner_x=x_vert_px, corner_y=y_top_px,
                arm_dx=x_horiz_end_px - x_vert_px,
                arm_dy=y_bot_px - y_top_px,
                style=style, rng=rng,
                row_slope_deg=0.0,
                gap_prob=_gap_p_br,
                thickness_min=_br_t_min, thickness_max=_br_t_max,
                intensity_min=_br_i_min, intensity_max=_br_i_max,
                micro_break_prob=_br_mb,
                h_y_jitter_sigma=_br_hyj, h_x_jitter_sigma=_br_hxj,
                v_x_jitter_sigma=_br_vxj,
            )
            row_max_y = max(row_max_y, by1)
            yolo_id = YOLO_NAME_TO_ID[tok.yolo_class_name]
            yolo = _yolo_box(bx0, by0, bx1, by1, canvas_w, canvas_h)
            annotations.append(f"{yolo_id} {yolo[0]:.6f} {yolo[1]:.6f} {yolo[2]:.6f} {yolo[3]:.6f}")
            gt_tokens.append({
                "fine_label": tok.flattened_label,
                "glyph_key": tok.glyph_key,
                "yolo_class": tok.yolo_class_name,
                "row_index": tok.row,
                "col_index": tok.col,
                "bbox": [bx0, by0, bx1, by1],
                "equation_idx": eq_idx,
            })
            continue

        pool_key = str(tok.glyph_key)
        pool = resolve_pool(pool_key)
        if pool is not None and len(pool) > 0:
            if pool_root is not None and min_source_quality > 0.0:
                tile = _load_tile_with_quality(pool, rng, pool_root, min_source_quality)
            else:
                row = _sample_symbol(pool, rng)
                tile = _tile_from_path(Path(row["path"]))
        elif pool_key == "result_bar":
            tile = _result_bar_fallback_tile()
        else:
            raise RuntimeError(
                f"Synthetic YOLO: no symbol rows for glyph_key={pool_key!r} in split manifest. "
                "Ensure data/raw/pool_emnist_28 has digit/operator folders and re-run validate before generate."
            )

        # Cap scale_jitter so that the rendered glyph width (s * tw * sj) plus the
        # per-glyph jitter budget on both neighbours cannot exceed the column pitch.
        # Guarantee: pw + 2 * jitter_max + 1 <= s * x_step, i.e. adjacent glyph bboxes
        # never overlap regardless of how jitter breaks across the two neighbours.
        # Formula: sj_max = (s * x_step - 2 * jitter_max - 1) / (s * tw_i)
        #   -- derived by solving pw < s*x_step - 2*jitter_max - 1 for scale_jitter.
        # We clamp the result to [0.88, 1.08] to preserve enough scale variance while
        # guaranteeing clearance. When x_step is very small (compressed layout) the
        # upper bound falls below 0.92 and glyphs are drawn at 88-92% of cell size,
        # which is visually indistinguishable from the full range.
        _jitter_budget = 2 * max(0, style.jitter_max_px) + 1
        _sj_max = (s * float(x_step) - _jitter_budget) / max(1.0, s * float(tw_i))
        _sj_max = max(0.88, min(1.08, _sj_max))
        scale_jitter = rng.uniform(0.88, _sj_max)
        pw = max(4, int(s * tw_i * scale_jitter))
        ph = max(4, int(s * th_i * scale_jitter))
        px0 = int(ox + s * x0) + _jitter(tok)
        py0 = int(oy + s * y0) + _jitter(tok)
        px0 = max(0, min(px0, canvas_w - 2))
        py0 = max(0, min(py0, canvas_h - 2))

        # iter11 overflow: an eligible glyph bleeds into a neighbour cell (bbox crosses the pitch)
        # but its CENTROID stays inside the intended cell -- the explicit shift is < 0.5 pitch and the
        # scale growth is re-centred, so geometric row/col GT still holds. Bypasses the anti-overlap
        # clamp above on purpose (that clamp exists to prevent exactly this; overflow opts out).
        _ov_prob = float(rendering.get("glyph_overflow_prob", 0.0)) if rendering else 0.0
        if (
            _ov_prob > 0.0
            and tok.yolo_class_name in ("digit_main", "operator")
            and rng.random() < _ov_prob
        ):
            _ov_scale_max = float(rendering.get("glyph_overflow_scale_max", 1.35)) if rendering else 1.35
            _ov_shift = float(rendering.get("glyph_overflow_shift_frac", 0.30)) if rendering else 0.30
            _ov_scale = rng.uniform(1.10, max(1.10, _ov_scale_max))
            pw = max(4, int(s * tw_i * _ov_scale))
            ph = max(4, int(s * th_i * _ov_scale))
            # re-centre the scale growth so scaling alone never moves the centroid
            _ov_px = int(ox + s * x0) - (pw - int(s * tw_i)) // 2
            _ov_py = int(oy + s * y0) - (ph - int(s * th_i)) // 2
            # bleed toward a random neighbour on one axis; |shift| < 0.5 pitch keeps centroid in cell
            if rng.random() < 0.5:
                _ov_px += int(rng.uniform(-_ov_shift, _ov_shift) * s * float(x_step))
            else:
                _ov_py += int(rng.uniform(-_ov_shift, _ov_shift) * s * float(y_step))
            px0 = max(0, min(_ov_px + _jitter(tok), canvas_w - 2))
            py0 = max(0, min(_ov_py + _jitter(tok), canvas_h - 2))

        _morph_k = int(rendering.get("glyph_morph_close_kernel", 3)) if rendering else 3
        _stroke_tgt = float(rendering.get("glyph_stroke_target_px", 0.0)) if rendering else 0.0
        _band_frac = float(rendering.get("broken_stroke_band_frac", 0.125)) if rendering else 0.125
        _binar = int(rendering.get("glyph_binarize_threshold", INK_BINARIZE_THRESHOLD)) if rendering else INK_BINARIZE_THRESHOLD
        bx0, by0, bx1, by1 = _paste_symbol(
            canvas, tile, px0, py0, pw, ph, style, rng,
            morph_close_kernel=_morph_k,
            glyph_stroke_target_px=_stroke_tgt,
            broken_stroke_band_frac=_band_frac,
            binarize_threshold=_binar,
        )

        # D: borrow_cross_prob float (0.0–1.0); per-digit probability of drawing the cross.
        _borrow_cross_prob = float(rendering.get("borrow_cross_prob", 0.90)) if rendering else 0.90
        if (
            _borrow_cross_prob > 0.0
            and borrow_cols
            and tok.yolo_class_name == "digit_main"
            and tok.col in borrow_cols
            and tok.row == minuend_row
            and rng.random() < _borrow_cross_prob
        ):
            _draw_borrow_cross(canvas, bx0, by0, bx1, by1, rng, canvas_h, canvas_w)

        row_max_y = max(row_max_y, by1)
        yolo_id = YOLO_NAME_TO_ID[tok.yolo_class_name]
        yolo = _yolo_box(bx0, by0, bx1, by1, canvas_w, canvas_h)
        annotations.append(f"{yolo_id} {yolo[0]:.6f} {yolo[1]:.6f} {yolo[2]:.6f} {yolo[3]:.6f}")
        gt_tokens.append({
            "fine_label": tok.flattened_label,
            "glyph_key": tok.glyph_key,
            "yolo_class": tok.yolo_class_name,
            "row_index": tok.row,
            "col_index": tok.col,
            "bbox": [bx0, by0, bx1, by1],
            "equation_idx": eq_idx,
        })

    return annotations, gt_tokens, row_max_y
