"""Render the canonical example PNGs into data/generated/synthetic/latest/examples/.

Moved from scripts/render_canonical_examples.py; renamed to render_examples.py. Registered as ``python -m src render-examples``.

Each case has a deterministic seed and fixed operand overrides to produce a representative scene
(e.g. carry-free vs carry-present, wrong answer). Output:
  data/generated/synthetic/latest/examples/<case_name>.png
  data/generated/synthetic/latest/examples/manifest.json
  data/generated/synthetic/latest/examples/README.md

Run after:
  python -m src generate --project-root <project-root>

Usage:
  python -m src render-examples --project-root <project-root>
"""
from __future__ import annotations

import argparse
import json
import random as _random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Canonical case definitions (78 total -- hardcoded, stable list)
# ---------------------------------------------------------------------------

@dataclass
class CanonicalCase:
    case_name: str
    scene_case: str        # SceneCase.value string
    completion_stage: str  # CompletionStage.value or "full"
    force_wrong_answer: bool = False
    seed: int = 0          # filled automatically
    description: str = ""
    rendering_overrides: dict = None  # per-case knob overrides merged into gen_cfg.rendering; None = no overrides

    def __post_init__(self):
        if self.rendering_overrides is None:
            self.rendering_overrides = {}


def _seed(name: str) -> int:
    """Deterministic seed per case name, positive int."""
    return abs(hash(name)) % (2 ** 31)


CANONICAL_CASES: list[CanonicalCase] = [
    # ---------- Addition (8) ----------
    CanonicalCase("add_full_no_carry",          "addition",                       "full",    False, description="No carries"),
    CanonicalCase("add_full_with_carry",         "addition",                       "full",    False, description="With carry digits"),
    CanonicalCase("add_full_wrong_answer",        "addition",                       "full",    True,  description="Wrong result (faithful transcription)"),
    CanonicalCase("add_operator_left",           "addition",                       "full",    False, description="Operator left of bottom row (canonical)"),
    CanonicalCase("add_operator_right",          "addition_op_right",              "full",    False, description="Operator right of bottom operand"),
    CanonicalCase("add_no_bar",                  "addition_no_bar",                "full",    False, description="Operands + operator, no bar"),
    CanonicalCase("add_partial_operands_only",   "addition",                       "done_20", False, description="Top operand and operator only"),
    CanonicalCase("add_carries_above_operands",  "addition",                       "done_80", False, description="Carries visible, result partial"),
    # ---------- Subtraction (7) ----------
    CanonicalCase("sub_full_no_borrow",          "subtraction",                    "full",    False, description="No borrows"),
    CanonicalCase("sub_full_with_borrow",        "subtraction",                    "full",    False, description="With borrow digits"),
    CanonicalCase("sub_full_wrong_answer",       "subtraction",                    "full",    True,  description="Wrong result"),
    CanonicalCase("sub_operator_left",           "subtraction",                    "full",    False, description="Minus left of bottom row"),
    CanonicalCase("sub_operator_right",          "subtraction_op_right",           "full",    False, description="Minus right of bottom operand"),
    CanonicalCase("sub_no_bar",                  "subtraction_no_bar",             "full",    False, description="Operands + minus, no bar"),
    CanonicalCase("sub_partial_borrows_only",    "subtraction",                    "done_60", False, description="Borrows visible, result blank"),
    # ---------- Multiplication-simple (7) ----------
    CanonicalCase("mul_simple_full_no_carry",    "multiplication-simple",          "full",    False, description="Single-digit multiplier, no carry"),
    CanonicalCase("mul_simple_full_with_carry",  "multiplication-simple",          "full",    False, description="With carry"),
    CanonicalCase("mul_simple_full_wrong",       "multiplication-simple",          "full",    True,  description="Wrong result"),
    CanonicalCase("mul_simple_operator_left",    "multiplication-simple",          "full",    False, description="x left of multiplier"),
    CanonicalCase("mul_simple_operator_right",   "multiplication_simple_op_right", "full",    False, description="x right of multiplier"),
    CanonicalCase("mul_simple_no_bar",           "multiplication_simple_no_bar",   "full",    False, description="Operands + x, no bar"),
    CanonicalCase("mul_simple_partial",          "multiplication-simple",          "done_20", False, description="Operands only"),
    # ---------- Multiplication-multi (5) ----------
    CanonicalCase("mul_multi_full",              "multiplication-multi",           "full",    False, description="All partial products, sum bar, final result"),
    CanonicalCase("mul_multi_full_with_carries", "multiplication-multi",           "full",    False, description="With inter-PP carry rows"),
    CanonicalCase("mul_multi_wrong_final",       "multiplication-multi",           "full",    True,  description="Wrong final sum"),
    CanonicalCase("mul_multi_first_pp_only",     "multiplication-multi",           "done_20", False, description="First partial product only"),
    CanonicalCase("mul_multi_pp_no_final_sum",   "multiplication-multi",           "done_60", False, description="All partial products, no final sum"),
    # ---------- Division-short (4) ----------
    CanonicalCase("div_short_full_with_remainder","division-short",                "full",    False, description="Full short division with remainder"),
    CanonicalCase("div_short_full_multi_step",    "division-short",                "full",    False, description="Full with intermediate subtractions"),
    CanonicalCase("div_short_wrong_quotient",     "division-short",                "full",    True,  description="Wrong quotient"),
    CanonicalCase("div_short_first_step_only",    "division-short",                "done_40", False, description="Bracket + first quotient digit"),
    # ---------- Division-long (5) ----------
    CanonicalCase("div_long_full",               "division-long",                  "full",    False, description="Full staartdeling"),
    CanonicalCase("div_long_step_k_of_K",        "division-long",                  "done_60", False, description="First k of K subtraction steps"),
    CanonicalCase("div_long_bracket_dividend_only","division-long",                "done_20", False, description="Bracket + dividend + divisor, no work"),
    CanonicalCase("div_long_wrong_quotient",     "division-long",                  "full",    True,  description="Wrong quotient digit"),
    CanonicalCase("div_long_no_final_remainder", "division-long",                  "done_80", False, description="All steps done, final remainder missing"),
    # ---------- Division-simple (3) ----------
    CanonicalCase("div_simple_full",             "division-simple",                "full",    False, description="Full column-form simple division"),
    CanonicalCase("div_simple_wrong",            "division-simple",                "full",    True,  description="Wrong answer"),
    CanonicalCase("div_simple_partial",          "division-simple",                "done_40", False, description="Operands written, result blank"),
    # ---------- MC-3: subtraction with forced borrows (2) ----------
    CanonicalCase("sub_heavy_borrow_full",       "subtraction_heavy_borrow",       "full",    False, description="3-digit minus 3-digit with 2-3 forced borrows, all tokens"),
    CanonicalCase("sub_heavy_borrow_partial",    "subtraction_heavy_borrow",       "done_60", False, description="Borrows visible, result row blank"),
    # ---------- MC-4a: addition with 2-3 forced column carries (1) ----------
    CanonicalCase("add_dense_carries_full",      "addition_dense_carries",         "full",    False, description="3-digit plus 3-digit with carries in 2-3 columns"),
    # ---------- MC-4b: addition with zero carries baseline (1) ---------- ----------
    CanonicalCase("add_no_carries_full",         "addition_no_carries",            "full",    False, description="3-digit plus 3-digit with no carries in any column"),
    # ---------- iter11: crowded double-digit-in-one-cell variants (10: 5 stages x 2 cases) ----------
    CanonicalCase("sub_crowded_borrow_full",     "subtraction_crowded_borrow",     "full",    False, description="Borrow rendered as two digits in one cell (e.g. 16), all tokens"),
    CanonicalCase("sub_crowded_borrow_80",       "subtraction_crowded_borrow",     "done_80", False, description="Double-digit borrow, result row near-complete"),
    CanonicalCase("sub_crowded_borrow_60",       "subtraction_crowded_borrow",     "done_60", False, description="Double-digit borrow, result row blank"),
    CanonicalCase("sub_crowded_borrow_40",       "subtraction_crowded_borrow",     "done_40", False, description="Double-digit borrow, operands + operator only"),
    CanonicalCase("sub_crowded_borrow_20",       "subtraction_crowded_borrow",     "done_20", False, description="Double-digit borrow, most incomplete"),
    CanonicalCase("add_crowded_carry_full",      "addition_crowded_carry",         "full",    False, description="Carry rendered as two digits in one cell, all tokens"),
    CanonicalCase("add_crowded_carry_80",        "addition_crowded_carry",         "done_80", False, description="Double-digit carry, result row near-complete"),
    CanonicalCase("add_crowded_carry_60",        "addition_crowded_carry",         "done_60", False, description="Double-digit carry, result row blank"),
    CanonicalCase("add_crowded_carry_40",        "addition_crowded_carry",         "done_40", False, description="Double-digit carry, operands + operator only"),
    CanonicalCase("add_crowded_carry_20",        "addition_crowded_carry",         "done_20", False, description="Double-digit carry, most incomplete"),

    # ---------- Combination variants: addition (4) ----------
    CanonicalCase("add_full_op_right_with_carry", "addition_op_right", "full", False,
                  description="Operator right of bottom operand; operands sampled to produce carries"),
    CanonicalCase("add_full_op_right_no_carry",   "addition_op_right", "full", False,
                  description="Operator right of bottom operand; single-digit operands with no carry"),
    CanonicalCase("add_no_bar_with_carry",         "addition_no_bar",  "full", False,
                  description="No result bar; operands chosen to produce carry digits above"),
    CanonicalCase("add_partial_with_carry",        "addition",         "done_80", False,
                  description="Carries visible, result row partially written"),

    # ---------- Combination variants: subtraction (4) ----------
    CanonicalCase("sub_full_with_borrow_no_crossout",   "subtraction", "full", False,
                  description="Borrows present; cross-out stroke suppressed",
                  rendering_overrides={"borrow_cross_prob": 0.0}),
    CanonicalCase("sub_full_with_borrow_full_crossout",  "subtraction", "full", False,
                  description="Borrows present; every borrow column gets a cross-out stroke",
                  rendering_overrides={"borrow_cross_prob": 1.0}),
    CanonicalCase("sub_heavy_borrow_no_crossout",        "subtraction_heavy_borrow", "full", False,
                  description="3-digit forced borrows; cross-out stroke suppressed",
                  rendering_overrides={"borrow_cross_prob": 0.0}),
    CanonicalCase("sub_full_op_right_with_borrow",       "subtraction_op_right", "full", False,
                  description="Operator right of bottom operand with borrow digits visible"),

    # ---------- Combination variants: multiplication-simple (3) ----------
    CanonicalCase("mul_simple_op_right_with_carry",      "multiplication_simple_op_right", "full", False,
                  description="Operator right of multiplier; operand chosen to produce carry digits"),
    CanonicalCase("mul_simple_no_bar_with_carry",        "multiplication_simple_no_bar", "full", False,
                  description="No result bar; operands chosen to produce carry digits"),
    CanonicalCase("mul_simple_full_no_carry_simple_ops", "multiplication-simple", "full", False,
                  description="Single-digit by single-digit; no carry possible"),

    # ---------- Combination variants: multiplication-multi (4) ----------
    CanonicalCase("mul_multi_with_pp_plus",              "multiplication-multi", "full", False,
                  description="All PP rows show explicit plus operator",
                  rendering_overrides={"pp_plus_prob": 1.0}),
    CanonicalCase("mul_multi_with_pp_wrong_minus",       "multiplication-multi", "full", False,
                  description="All PP plus operators flipped to minus (student error)",
                  rendering_overrides={"pp_plus_prob": 1.0, "pp_wrong_operator_prob": 1.0}),
    CanonicalCase("mul_multi_with_pp_mixed",             "multiplication-multi", "full", False,
                  description="Half of PP plus operators flipped to minus",
                  rendering_overrides={"pp_plus_prob": 1.0, "pp_wrong_operator_prob": 0.5}),
    CanonicalCase("mul_multi_no_pp_operators",           "multiplication-multi", "full", False,
                  description="No plus operators on PP rows (current default; explicit reference case)",
                  rendering_overrides={"pp_plus_prob": 0.0}),

    # ---------- Combination variants: division-long (7) ----------
    CanonicalCase("div_long_with_step_minus",            "division-long", "full", False,
                  description="Every subtraction step shows an explicit minus operator",
                  rendering_overrides={"step_minus_prob": 1.0}),
    CanonicalCase("div_long_with_step_borrow",           "division-long", "full", False,
                  description="Step borrow tokens emitted whenever arithmetic needs borrowing",
                  rendering_overrides={"long_div_step_borrow_prob": 1.0}),
    CanonicalCase("div_long_with_step_minus_and_borrow", "division-long", "full", False,
                  description="Step minus operators and step borrow tokens both active",
                  rendering_overrides={"step_minus_prob": 1.0, "long_div_step_borrow_prob": 1.0}),
    CanonicalCase("div_long_with_helper_operator",       "division-long", "full", False,
                  description="Helper plus/minus operator inserted at left of intermediate rows",
                  rendering_overrides={"helper_operator_prob": 0.5}),
    CanonicalCase("div_long_with_shallow_bracket",       "division-long", "full", False,
                  description="Vertical bracket arm shortened; does not reach the last working row",
                  rendering_overrides={"bracket_depth_full_prob": 0.0, "bracket_depth_min_rows": 2}),
    CanonicalCase("div_long_missing_step_bar",           "division-long", "full", False,
                  description="Step result bars drawn short on both ends (missing/partial bar style)",
                  rendering_overrides={"short_bar_prob": 1.0, "short_bar_shrink_frac": 0.5}),
    CanonicalCase("div_long_partial_with_step_minus",    "division-long", "done_60", False,
                  description="Partial long division; completed steps show explicit minus operator",
                  rendering_overrides={"step_minus_prob": 1.0}),

    # ---------- Combination variants: division-short (3) ----------
    CanonicalCase("div_short_with_step_minus",           "division-short", "full", False,
                  description="Short division; every quotient step shows an explicit minus operator",
                  rendering_overrides={"step_minus_prob": 1.0}),
    CanonicalCase("div_short_with_step_borrow",          "division-short", "full", False,
                  description="Short division; step borrow tokens emitted when arithmetic needs borrowing",
                  rendering_overrides={"long_div_step_borrow_prob": 1.0}),
    CanonicalCase("div_short_with_step_minus_and_borrow","division-short", "full", False,
                  description="Short division; step minus and borrow tokens both active",
                  rendering_overrides={"step_minus_prob": 1.0, "long_div_step_borrow_prob": 1.0}),

    # ---------- Task 2: unexercised error knobs (5) ----------
    CanonicalCase("add_wrong_carry_col",     "addition",              "full", False,
                  description="Every carry token shifted one column left or right (wrong-col carry error)",
                  rendering_overrides={"wrong_carry_col_prob": 1.0}),
    CanonicalCase("add_missing_carry",       "addition",              "full", False,
                  description="Every carry token dropped entirely (child omits carry notation)",
                  rendering_overrides={"missing_carry_prob": 1.0}),
    CanonicalCase("sub_missing_borrow",      "subtraction",           "full", False,
                  description="Every borrow token dropped entirely (child omits borrow notation)",
                  rendering_overrides={"missing_borrow_prob": 1.0}),
    CanonicalCase("add_wrong_carry_value",   "addition",              "full", False,
                  description="Every carry digit replaced with a different digit (wrong carry value)",
                  rendering_overrides={"wrong_carry_value_prob": 1.0}),
    CanonicalCase("mul_multi_missing_pp_row","multiplication-multi",  "full", False,
                  description="One non-final partial-product row dropped (child skipped writing it)",
                  rendering_overrides={"missing_pp_prob": 1.0}),

    # ---------- Task 4: P2 combination cases (5) ----------
    # No addition-op-right-no-bar SceneCase exists; closest variant is addition_op_right.
    CanonicalCase("add_op_right_no_bar",          "addition_op_right",      "full", False,
                  description="Operator right of bottom operand; no result bar drawn (partial completion)"),
    CanonicalCase("sub_no_bar_with_borrow",        "subtraction_no_bar",     "full", False,
                  description="Subtraction with no result bar; borrow digits occur naturally from operands"),
    CanonicalCase("add_short_result_bar",          "addition",               "full", False,
                  description="Result bar drawn shorter than full operand width on both ends",
                  rendering_overrides={"short_bar_prob": 1.0}),
    CanonicalCase("mul_multi_partial_pp_plus",     "multiplication-multi",   "done_60", False,
                  description="Partial multi-digit multiplication; completed PP rows show explicit plus operator",
                  rendering_overrides={"pp_plus_prob": 1.0}),
    CanonicalCase("div_simple_wrong_partial",      "division-simple",        "done_40", True,
                  description="Simple division; operands written, result blank, wrong answer forced"),

    # ---------- iter10 W10-ROBUSTNESS: bare_digit_grid (1) ----------
    CanonicalCase("bare_digit_grid_full",          "bare_digit_grid",        "full", False,
                  description="Multi-row, multi-column block of bare digits; no operator, result bar, or bracket"),
]
assert len(CANONICAL_CASES) == 89, f"Expected 89 canonical cases, got {len(CANONICAL_CASES)}"

# Fill deterministic seeds
for _c in CANONICAL_CASES:
    if _c.seed == 0:
        _c.seed = _seed(_c.case_name)


# ---------------------------------------------------------------------------
# Override helpers
# ---------------------------------------------------------------------------

def _build_wrong_override(tokens: list) -> list[int]:
    """Return result-override digits (+1 mod 10 per digit) for wrong-answer rendering."""
    main_rows = sorted({t.row for t in tokens if t.yolo_class_name == "digit_main"})
    if not main_rows:
        return [9]
    result_row = main_rows[-1]
    result_toks = sorted(
        [t for t in tokens if t.row == result_row and t.yolo_class_name == "digit_main"],
        key=lambda t: t.col,
    )
    if not result_toks:
        return [9]
    return [(int(t.flattened_label[-1]) + 1) % 10 for t in result_toks]


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def render_one(
    case: CanonicalCase,
    project_root: Path,
    symbol_manifest_path: Path,
    gen_cfg,
) -> np.ndarray:
    """Render one canonical case. Returns (512, 512) uint8 grayscale array."""
    from src.generation.layouts import SceneCase, sample_layout_for_case
    from src.generation.completion_stages import sample_completion_stage_for_case
    from src.generation.handwriting_style import make_style
    from src.generation.synth_pool import (
        _draw_equation_block,
        _load_symbol_rows,
    )
    from src.data_pipeline.preprocessing import preprocess_for_pipeline

    rng_py = _random.Random(case.seed)
    rng_np = np.random.default_rng(case.seed)

    scene_case = SceneCase(case.scene_case)

    # Build wrong-answer override if needed (first pass to get result digits)
    result_override = None
    if case.force_wrong_answer:
        _, probe_tokens = sample_layout_for_case(scene_case, _random.Random(case.seed))
        result_override = _build_wrong_override(probe_tokens)

    # Sample layout
    _, tokens = sample_layout_for_case(scene_case, rng_py, result_override=result_override)
    if not tokens:
        raise ValueError(f"Layout for {case.scene_case} returned no tokens")

    # Apply completion stage by sampling with forced weights
    completion_weights = {case.completion_stage: 1.0}
    _stage, tokens = sample_completion_stage_for_case(
        scene_case, tokens, rng_py, weights=completion_weights
    )

    # Canvas
    W, H = 512, 512
    canvas = np.full((H, W), 255, dtype=np.uint8)

    # Style: easy_baseline (no jitter, no rotation)
    style = make_style(rng_np, gen_cfg.preset, crowded=False)

    # Load symbol pool
    symbol_df_full = _load_symbol_rows(symbol_manifest_path)
    symbol_df = symbol_df_full[symbol_df_full["split"] == "train"].reset_index(drop=True)
    symbols_by_glyph_key = {
        str(gk): frame.reset_index(drop=True)
        for gk, frame in symbol_df.groupby("glyph_key", dropna=False)
        if str(gk) != "" and str(gk).lower() != "nan"
    }
    fallback_by_glyph = {
        str(gk): frame.reset_index(drop=True)
        for gk, frame in symbol_df_full.groupby("glyph_key", dropna=False)
        if str(gk) != "" and str(gk).lower() != "nan"
    }

    pool_root: Path | None = None
    source_pool = gen_cfg.source_pool
    if source_pool != "legacy":
        candidate = project_root / "data" / f"combined_{source_pool}"
        if candidate.exists():
            pool_root = candidate
        elif (project_root / "data" / "raw" / "pool_emnist_28").exists():
            pool_root = project_root / "data" / "raw" / "pool_emnist_28"

    # Build effective rendering dict: base config merged with per-case overrides
    effective_rendering = dict(gen_cfg.rendering)
    if case.rendering_overrides:
        effective_rendering.update(case.rendering_overrides)

    # Draw
    _pad_min = int(effective_rendering.get("page_padding_min_px", 2))
    _pad_max = int(effective_rendering.get("page_padding_max_px", 10))
    _page_pad = rng_py.randint(_pad_min, max(_pad_min, _pad_max))
    top_pad, margin_bottom = _page_pad, _page_pad
    _draw_equation_block(
        canvas, rng_py,
        symbols_by_glyph_key, fallback_by_glyph,
        W, H, top_pad, H - top_pad - margin_bottom,
        tokens, 0,
        center_in_full_canvas_vertically=True,
        style=style,
        rendering=effective_rendering,
        scene_case=scene_case,
        pool_root=pool_root,
        min_source_quality=gen_cfg.min_source_quality,
    )

    return preprocess_for_pipeline(canvas)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render the canonical example PNGs.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent,
        help="Path to project root (default: grandparent of this file)",
    )
    args = parser.parse_args(argv)
    _run(args.project_root)


def _run(project_root: Path) -> None:
    from src.core.config import DataPrepConfig
    from src.core.run_config import load_config

    config = DataPrepConfig.from_project_root(project_root)
    gen_cfg, _, _ = load_config(project_root)

    # Verify symbol manifest exists
    symbol_manifest_path = config.yolo_dir / "symbol_assets_manifest.csv"
    if not symbol_manifest_path.exists():
        print(
            "ERROR: symbol_assets_manifest.csv not found. "
            "Run `python -m src validate --project-root <project-root>` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve latest symlink
    latest = project_root / "data" / "generated" / "synthetic" / "latest"
    if not latest.exists():
        print(
            "ERROR: data/generated/synthetic/latest does not exist. "
            "Run `python -m src generate --project-root <project-root>` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    resolved = latest.resolve()
    if not resolved.exists():
        print(
            f"ERROR: data/generated/synthetic/latest is a dangling symlink -> {resolved}. "
            "Regenerate with `python -m src generate --project-root <project-root>`.",
            file=sys.stderr,
        )
        sys.exit(1)

    examples_dir = resolved / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {}
    errors: list[tuple[str, str]] = []

    n_total = len(CANONICAL_CASES)
    for i, case in enumerate(CANONICAL_CASES):
        override_note = f", overrides={list(case.rendering_overrides)}" if case.rendering_overrides else ""
        print(
            f"  [{i+1:02d}/{n_total}] {case.case_name}  "
            f"({case.scene_case}, {case.completion_stage}"
            f"{', WRONG' if case.force_wrong_answer else ''}{override_note})"
        )
        try:
            arr = render_one(case, project_root, symbol_manifest_path, gen_cfg)
            out_path = examples_dir / f"{case.case_name}.png"
            Image.fromarray(arr).save(out_path)
            manifest[case.case_name] = {
                "file": f"{case.case_name}.png",
                "scene_case": case.scene_case,
                "completion_stage": case.completion_stage,
                "force_wrong_answer": case.force_wrong_answer,
                "seed": case.seed,
                "description": case.description,
                "rendering_overrides": case.rendering_overrides,
            }
        except Exception as exc:
            import traceback
            print(f"    ERROR: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            errors.append((case.case_name, str(exc)))

    # Write manifest
    manifest_path = examples_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Write README
    readme_lines = [
        "# Canonical Examples",
        "",
        f"One PNG per canonical arithmetic case. {n_total} files total.",
        "",
        "Regenerate with:",
        "```bash",
        "python -m src generate --project-root <project-root>",
        "python -m src render-examples --project-root <project-root>",
        "```",
        "",
        "See `docs/cases.md` for the full case specification.",
        "",
        "| case_name | scene_case | stage | wrong |",
        "|-----------|-----------|-------|-------|",
    ]
    for c in CANONICAL_CASES:
        readme_lines.append(
            f"| {c.case_name} | {c.scene_case} | {c.completion_stage} | {c.force_wrong_answer} |"
        )
    (examples_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    # Count PNGs
    png_count = len(list(examples_dir.glob("*.png")))
    print(f"\nGenerated {png_count} PNGs in {examples_dir}")
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for name, err in errors:
            print(f"  {name}: {err}")
    assert png_count == n_total, f"Expected {n_total} PNGs but got {png_count}."
    print(f"All {n_total} canonical examples rendered successfully.")


if __name__ == "__main__":
    main()
