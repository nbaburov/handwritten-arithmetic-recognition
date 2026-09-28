"""Build reports/knob_explorer/index.html with OFF/ON rendered pairs for every generation knob.

Run from project root:
    PYTHONPATH=. .venv/bin/python scripts/build_knob_explorer.py --project-root .
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import sys
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Knob catalogue
# ---------------------------------------------------------------------------

# Each entry: (toml_key, scope, type_, off_val, on_val, default_val, config_val, eli5, consumer)
KNOBS = [
    # ---- Scene-level ----
    (
        "scene_rotation_min_deg / scene_rotation_max_deg",
        "scene",
        "degrees",
        0.0, 4.0,
        "-4.0 / 4.0", "0.0 / 0.0",
        "Tilts the entire finished scene a few degrees, as if a child placed the paper at an angle on a table. "
        "The whole equation leans left or right together.",
        "src/generation/synth_yolo.py — scene-level affine applied post-draw",
    ),
    (
        "wrong_result_prob",
        "scene",
        "probability",
        0.0, 1.0,
        0.20, 0.0,
        "Makes the final answer wrong. A child sometimes writes 5+3=9 instead of 8. "
        "This knob controls how often generated scenes carry an incorrect result.",
        "src/generation/synth_yolo.py:_apply_scene_knobs — result digit override",
    ),
    (
        "missing_structural_prob",
        "scene",
        "probability",
        0.0, 1.0,
        0.15, 0.0,
        "Drops a key structural token (the result bar or division bracket) to mimic a child "
        "who has not yet drawn that part of the working.",
        "src/generation/synth_yolo.py:_apply_scene_knobs — token removal",
    ),
    (
        "wrong_operator_prob",
        "scene",
        "probability",
        0.0, 1.0,
        0.05, 0.0,
        "Replaces the operator symbol (e.g., swaps + for x) to simulate a child who copies "
        "the wrong sign from the question.",
        "src/generation/synth_yolo.py:_apply_scene_knobs — operator flip",
    ),
    (
        "crowdness_prob",
        "scene",
        "probability",
        0.0, 1.0,
        0.20, 0.0,
        "Makes the glyphs sit closer together, mimicking a child who writes in a cramped style. "
        "When active, the crowding_factor (0.70) squeezes the horizontal spacing.",
        "src/generation/handwriting_style.py — crowding_factor applied in _paste_symbol",
    ),
    (
        "crowding_factor",
        "scene",
        "float 0-1",
        1.0, 0.70,
        0.70, 0.70,
        "The scaling ratio applied to glyph cell spacing when crowded mode is active. "
        "0.70 means glyphs occupy 70% of the normal cell width, pushing digits closer.",
        "src/generation/handwriting_style.py:29 — crowding branch in make_style",
    ),
    # ---- Per-glyph ----
    (
        "glyph_rotation_min_deg / glyph_rotation_max_deg",
        "glyph",
        "degrees",
        0.0, 4.0,
        "-4.0 / 4.0", "0.0 / 0.0",
        "Rotates each individual digit a small random amount, as if a child doesn't hold the pen "
        "perfectly straight for every stroke. Each glyph gets its own tilt independently.",
        "src/generation/synth_pool.py:_paste_symbol — rotate_tile call",
    ),
    (
        "glyph_jitter_min_px / glyph_jitter_max_px",
        "glyph",
        "px",
        0, 4,
        "0 / 4", "0 / 0",
        "Nudges each glyph a random number of pixels from its ideal grid position. "
        "Real children never place digits perfectly on an invisible grid.",
        "src/generation/synth_pool.py:_draw_equation_block — jitter offset applied per token",
    ),
    (
        "glyph_broken_stroke_prob",
        "glyph",
        "probability",
        0.0, 1.0,
        0.05, 0.0,
        "Erases a thin horizontal band across one glyph, mimicking a skip or gap in the pen stroke "
        "when a child lifts the stylus mid-digit. Affects individual glyphs independently.",
        "src/generation/synth_pool.py:_paste_symbol:80 — _apply_broken_stroke",
    ),
    (
        "glyph_morph_close_kernel",
        "glyph",
        "int (px kernel)",
        0, 3,
        0, 0,
        "After resizing a crop to the cell size, a morphological close operation fills tiny gaps "
        "in the ink skeleton that BICUBIC interpolation introduces. 0 = disabled; 3 = 3x3 ellipse. "
        "Thickens glyphs slightly.",
        "src/generation/synth_pool.py:_paste_symbol:95 — morph close branch",
    ),
    (
        "glyph_stroke_target_px",
        "glyph",
        "float px (0=off)",
        0.0, 2.5,
        2.5, 2.5,
        "Normalises the stroke width of every glyph to this pixel thickness using a "
        "distance-transform morphology. 0 disables normalisation (CROHME native thickness is kept). "
        "Helps thin and thick pen styles look consistent.",
        "src/generation/synth_pool.py:_paste_symbol:100 — normalise_glyph_stroke_width call",
    ),
    (
        "broken_stroke_band_frac",
        "glyph",
        "float fraction (0.0–0.5)",
        0.05, 0.40,
        0.125, 0.125,
        "Controls how tall the erased band is when a broken stroke is applied to a glyph "
        "(triggered by glyph_broken_stroke_prob). A fraction of the tile height: 0.125 erases "
        "an eighth of the glyph; 0.40 removes nearly half. Larger values create more visible gaps.",
        "src/generation/synth_pool.py:_apply_broken_stroke — band_h = int(h * band_frac)",
    ),
    (
        "page_padding_min_px",
        "glyph",
        "int px",
        2, 40,
        2, 2,
        "Minimum top/bottom margin (pixels) inside the 512x512 canvas. "
        "Sampled uniformly with page_padding_max_px per scene to vary how much white space surrounds the equation.",
        "src/generation/synth_yolo.py + render_examples.py — rng.randint(min, max)",
    ),
    (
        "page_padding_max_px",
        "glyph",
        "int px",
        2, 80,
        10, 10,
        "Maximum top/bottom margin (pixels) inside the 512x512 canvas. "
        "Sampled uniformly with page_padding_min_px per scene. Large values push the equation toward the centre.",
        "src/generation/synth_yolo.py + render_examples.py — rng.randint(min, max)",
    ),
    (
        "carry_borrow_scale_min",
        "glyph",
        "float ratio",
        1.0, 0.50,
        0.50, 0.50,
        "Lower bound for per-token carry/borrow scale. "
        "Each carry or borrow glyph independently samples its size from [carry_borrow_scale_min, carry_borrow_scale_max]. "
        "0.50 means the smallest carry/borrow glyphs are 50% of the main glyph cell size.",
        "src/generation/synth_pool.py — rng.uniform(carry_borrow_scale_min, carry_borrow_scale_max) per token",
    ),
    (
        "carry_borrow_scale_max",
        "glyph",
        "float ratio",
        1.0, 0.70,
        0.70, 0.70,
        "Upper bound for per-token carry/borrow scale. "
        "Each carry or borrow glyph independently samples its size from [carry_borrow_scale_min, carry_borrow_scale_max]. "
        "0.70 means the largest carry/borrow glyphs are 70% of the main glyph cell size.",
        "src/generation/synth_pool.py — rng.uniform(carry_borrow_scale_min, carry_borrow_scale_max) per token",
    ),
    (
        "operator_scale",
        "glyph",
        "float ratio",
        1.0, 0.92,
        0.92, 0.92,
        "Slightly shrinks operator symbols (+, -, x, ÷) relative to digit cells. "
        "Operators in handwritten arithmetic are typically written a little smaller than digits.",
        "src/generation/synth_pool.py — cell size scaled for operator tokens",
    ),
    (
        "base_cell_px",
        "glyph",
        "int px",
        22, 44,
        44, 44,
        "The base pixel size of one glyph cell before any scaling. Larger values make digits "
        "bigger in the 512x512 scene. 44px produces numbers that look natural for tablet writing.",
        "src/generation/synth_pool.py:_draw_equation_block — cell_w/cell_h base",
    ),
    (
        "gap_x_min / gap_x_max",
        "glyph",
        "px range",
        0, 13,
        "6 / 13", "6 / 13",
        "Sets the random horizontal whitespace between adjacent glyph columns. "
        "A wider gap makes digits easier to separate; a narrower gap produces cramped writing.",
        "src/generation/synth_pool.py:_draw_equation_block — x-spacing sampled from this range",
    ),
    (
        "gap_y_min / gap_y_max",
        "glyph",
        "px range",
        0, 19,
        "10 / 19", "10 / 19",
        "Sets the random vertical whitespace between glyph rows. "
        "Children vary how much space they leave between operand rows and the result.",
        "src/generation/synth_pool.py:_draw_equation_block — y-spacing sampled from this range",
    ),
    # ---- Bar stroke ----
    (
        "bar_gap_prob",
        "glyph",
        "probability",
        0.0, 1.0,
        0.04, 0.0,
        "Leaves a visible gap in the middle of the result bar, as if the child lifted the pen "
        "mid-stroke while drawing the horizontal line under the operands.",
        "src/generation/strokes.py:draw_handwritten_bar:109 — apply_gap branch",
    ),
    (
        "bar_thickness_min / bar_thickness_max",
        "glyph",
        "px range",
        2, 4,
        "2 / 2", "2 / 2",
        "Controls how thick the result bar stroke is per segment. "
        "A range of 2-4 makes the bar look hand-drawn with varying pen pressure.",
        "src/generation/strokes.py:draw_handwritten_bar:152 — per-segment thickness",
    ),
    (
        "bar_intensity_min / bar_intensity_max",
        "glyph",
        "grey 0-255",
        0, 100,
        "0 / 0", "0 / 0",
        "Controls ink lightness of the bar segments. 0=pure black, 255=white. "
        "A non-zero range produces segments in medium grey, mimicking faded pen pressure.",
        "src/generation/strokes.py:draw_handwritten_bar:155 — per-segment intensity",
    ),
    (
        "bar_micro_break_prob",
        "glyph",
        "probability",
        0.0, 1.0,
        0.08, 0.0,
        "Skips individual short segments of the bar at random, creating tiny dotted gaps "
        "rather than one big gap. Real pen strokes can be jerky and produce this effect.",
        "src/generation/strokes.py:draw_handwritten_bar:148 — micro-break per segment",
    ),
    (
        "bar_y_jitter_sigma",
        "glyph",
        "px sigma",
        0.0, 4.0,
        3.0, 0.0,
        "Adds vertical wobble to the bar's control points so the line isn't perfectly horizontal. "
        "Real children draw wobbly lines rather than ruler-straight ones.",
        "src/generation/strokes.py:draw_handwritten_bar — y_jitter applied to control points",
    ),
    (
        "bar_x_jitter_sigma",
        "glyph",
        "px sigma",
        0.0, 3.0,
        1.5, 0.0,
        "Adds a small horizontal offset to the bar's endpoints, so the bar doesn't start and end "
        "exactly at the leftmost/rightmost digit. Mimics imprecise stroke start/end.",
        "src/generation/strokes.py:draw_handwritten_bar — x_jitter applied to endpoints",
    ),
    # ---- Bracket stroke ----
    (
        "bracket_gap_prob",
        "glyph",
        "probability",
        0.0, 1.0,
        0.04, 0.0,
        "Leaves a gap in the horizontal or vertical arm of the division bracket (the ⌐ shape). "
        "Similar to bar_gap_prob but for the bracket drawn in long/short division.",
        "src/generation/strokes.py:draw_handwritten_bracket — gap_prob applied per arm",
    ),
    (
        "bracket_thickness_min / bracket_thickness_max",
        "glyph",
        "px range",
        2, 4,
        "2 / 2", "2 / 2",
        "Controls per-segment stroke thickness for the division bracket arms. "
        "Variable thickness makes the bracket look hand-drawn.",
        "src/generation/strokes.py:draw_handwritten_bracket — thickness per segment",
    ),
    (
        "bracket_intensity_min / bracket_intensity_max",
        "glyph",
        "grey 0-255",
        0, 100,
        "0 / 0", "0 / 0",
        "Controls ink lightness for bracket segments. Non-zero values produce faded grey arms, "
        "mimicking inconsistent pen pressure across the bracket stroke.",
        "src/generation/strokes.py:draw_handwritten_bracket — intensity per segment",
    ),
    (
        "bracket_micro_break_prob",
        "glyph",
        "probability",
        0.0, 1.0,
        0.08, 0.0,
        "Randomly skips short bracket segments to create a jerky dotted appearance. "
        "Works the same as bar_micro_break_prob but applies to the bracket shape.",
        "src/generation/strokes.py:draw_handwritten_bracket — micro-break per segment",
    ),
    (
        "bracket_h_y_jitter_sigma",
        "glyph",
        "px sigma",
        0.0, 4.0,
        2.5, 0.0,
        "Adds vertical wobble to the horizontal arm of the division bracket. "
        "Without this the horizontal top of the bracket is a ruler-perfect line.",
        "src/generation/strokes.py:draw_handwritten_bracket — h_y_jitter on horizontal arm",
    ),
    (
        "bracket_h_x_jitter_sigma",
        "glyph",
        "px sigma",
        0.0, 3.0,
        1.0, 0.0,
        "Adds horizontal endpoint jitter to the bracket's horizontal arm. "
        "The bracket arm starts and ends at slightly random x positions.",
        "src/generation/strokes.py:draw_handwritten_bracket — h_x_jitter on horizontal arm",
    ),
    (
        "bracket_v_x_jitter_sigma",
        "glyph",
        "px sigma",
        0.0, 3.0,
        1.5, 0.0,
        "Adds horizontal wobble to the vertical arm of the division bracket. "
        "The descending stroke leans slightly rather than dropping straight down.",
        "src/generation/strokes.py:draw_handwritten_bracket — v_x_jitter on vertical arm",
    ),
    # ---- Error injection ----
    (
        "borrow_cross_prob",
        "error",
        "probability",
        0.0, 1.0,
        0.0, 0.0,
        "Draws a small cross-out stroke through the digit that was borrowed from, "
        "like a child who crosses out the top number when writing a borrow notation. "
        "Active per column that has a borrow token.",
        "src/generation/layouts.py — borrow cross-out stroke injection",
    ),
    (
        "step_minus_prob",
        "error",
        "probability",
        0.0, 1.0,
        0.0, 0.0,
        "Adds an explicit minus operator to the left of each subtraction step in long or short division. "
        "Some children write '- 24' under the working line rather than just '24'.",
        "src/generation/layouts.py — step minus token insertion",
    ),
    (
        "wrong_carry_col_prob",
        "error",
        "probability",
        0.0, 1.0,
        0.0, 0.0,
        "Shifts a carry digit one column to the left or right of where it should be. "
        "A child who misaligns their carry writes it above the wrong column.",
        "src/generation/layouts.py — carry token column offset",
    ),
    (
        "missing_carry_prob",
        "error",
        "probability",
        0.0, 1.0,
        0.0, 0.0,
        "Drops a carry digit entirely from the scene. "
        "Many children forget to write the carry when adding multi-digit numbers.",
        "src/generation/layouts.py — carry token removal",
    ),
    (
        "wrong_carry_value_prob",
        "error",
        "probability",
        0.0, 1.0,
        0.0, 0.0,
        "Replaces a carry digit with a different (wrong) digit. "
        "A child may write the wrong carry value (e.g., write 2 instead of 1 when adding).",
        "src/generation/layouts.py — carry digit label substitution",
    ),
    (
        "missing_borrow_prob",
        "error",
        "probability",
        0.0, 1.0,
        0.0, 0.0,
        "Drops a borrow digit entirely. Some children skip writing the borrow notation "
        "and just do the mental computation, leaving no visible token on paper.",
        "src/generation/layouts.py — borrow token removal",
    ),
    (
        "wrong_borrow_value_prob",
        "error",
        "probability",
        0.0, 1.0,
        0.0, 0.0,
        "Replaces a borrow digit with a different (wrong) digit. "
        "A child may write the wrong borrow value above the column.",
        "src/generation/layouts.py — borrow digit label substitution",
    ),
    (
        "short_bar_prob",
        "error",
        "probability",
        0.0, 1.0,
        0.0, 0.0,
        "Draws the result bar shorter than the full operand width on both ends. "
        "Children often draw a short underline that does not span all the digits.",
        "src/generation/strokes.py / layouts.py — bar x0/x1 shrunk by short_bar_shrink_frac",
    ),
    (
        "short_bar_shrink_frac",
        "error",
        "float 0-1",
        0.0, 0.5,
        0.20, 0.20,
        "How much of the bar is removed from each end when short_bar_prob fires. "
        "0.20 removes 20% from each side; 0.50 produces a very short centre stub.",
        "src/generation/strokes.py / layouts.py — x0/x1 offset amount",
    ),
    (
        "pp_plus_prob",
        "error",
        "probability",
        0.0, 1.0,
        0.0, 0.0,
        "Adds an explicit plus operator between partial-product rows in multi-digit multiplication. "
        "Some textbooks show '+ 120' style notation; others omit the operator.",
        "src/generation/layouts.py — pp plus token emission for multiplication_multi",
    ),
    (
        "pp_wrong_operator_prob",
        "error",
        "probability",
        0.0, 1.0,
        0.0, 0.0,
        "When a plus is shown on a partial-product row, flips it to a minus instead. "
        "Simulates a child who copies the wrong operator from a template.",
        "src/generation/layouts.py — pp operator flip",
    ),
    (
        "bracket_depth_full_prob",
        "error",
        "probability",
        0.0, 1.0,
        1.0, 1.0,
        "Controls whether the division bracket's vertical arm reaches the last working row. "
        "1.0 = arm always full length. 0.0 = arm always shortened (shallow bracket). "
        "Some children stop the bracket arm early.",
        "src/generation/layouts.py — bracket arm dy reduced when shortened",
    ),
    (
        "bracket_depth_min_rows",
        "error",
        "int rows",
        1, 4,
        2, 2,
        "When the bracket arm is shortened (bracket_depth_full_prob &lt; 1.0), this is the minimum "
        "number of rows the arm must still span. Prevents the arm from becoming too short to be meaningful.",
        "src/generation/layouts.py — bracket arm minimum row span",
    ),
    (
        "long_div_step_borrow_prob",
        "error",
        "probability",
        0.0, 1.0,
        0.0, 0.0,
        "Emits a borrow token above each subtraction step in long or short division that "
        "arithmetically requires borrowing. Mimics children who write borrow notation during division.",
        "src/generation/layouts.py — step borrow token emission",
    ),
    (
        "helper_operator_prob",
        "error",
        "probability",
        0.0, 1.0,
        0.0, 0.0,
        "Inserts a plus or minus operator at the left edge of each intermediate row in long division. "
        "Some teaching styles show explicit operators on every line of the working.",
        "src/generation/layouts.py — helper operator token per intermediate row",
    ),
    (
        "missing_pp_prob",
        "error",
        "probability",
        0.0, 1.0,
        0.0, 0.0,
        "Drops one non-final partial-product row from a multi-digit multiplication scene. "
        "Simulates a child who skipped writing one of the partial products.",
        "src/generation/layouts.py — PP row removal for multiplication_multi",
    ),
]

# Hardcoded effects (no config knob)
HARDCODED = [
    {
        "effect": "INK_BINARIZE_THRESHOLD = 200",
        "file": "src/core/ink.py:32",
        "eli5": "After a glyph crop is resized with BICUBIC interpolation it gains fuzzy grey halo pixels. "
                "Pixels darker than value 200 are forced to pure black; lighter pixels are erased to white. "
                "This threshold is hardcoded and cannot be adjusted without editing the source.",
        "should_be_knob": "Possibly yes — lower values (e.g., 165) cut more halo at the cost of clipping thin strokes.",
    },
    {
        "effect": "INK_BBOX_THRESHOLD = 250",
        "file": "src/core/ink.py:29",
        "eli5": "Defines what counts as 'ink' when computing a tight bounding box around a rotated glyph. "
                "Pixels below 250 are treated as ink. Hardcoded to accept very light grey as ink.",
        "should_be_knob": "No — this is an implementation detail of the rotation crop and is unlikely to need tuning.",
    },
    {
        "effect": "INK_DRAW_THRESHOLD = 200",
        "file": "src/core/ink.py:35",
        "eli5": "The threshold used by stroke drawing routines to decide what counts as drawn ink. "
                "Controls whether faint grey segments are considered part of the stroke.",
        "should_be_knob": "No — tightly coupled to bar/bracket drawing logic; changing it would require co-changing intensity ranges.",
    },
    {
        "effect": "_BAR_CONTROL_POINTS = 7",
        "file": "src/generation/strokes.py:40",
        "eli5": "The bar is drawn as a spline through 7 evenly-spaced control points. "
                "More control points would produce smoother waves; fewer would look angular. "
                "This is hardcoded.",
        "should_be_knob": "No — 7 points is already overkill for a short bar. Not a meaningful dial.",
    },
    {
        "effect": "_apply_broken_stroke band height (now broken_stroke_band_frac knob, default 0.125)",
        "file": "src/generation/synth_pool.py:_apply_broken_stroke",
        "eli5": "Previously hardcoded as tile_height / 8. Now exposed as the broken_stroke_band_frac knob in "
                "[generation.rendering]. Controls how tall the erased gap is relative to the glyph height.",
        "should_be_knob": "Yes — promoted to broken_stroke_band_frac in [generation.rendering].",
    },
    {
        "effect": "Glyph paste blending: minimum composite (np.minimum)",
        "file": "src/generation/synth_pool.py:111",
        "eli5": "When a glyph tile is pasted onto the canvas, each pixel is set to min(canvas, glyph). "
                "This means ink only darkens the canvas and never erases existing ink. "
                "Overlapping glyphs always merge visually. Not configurable.",
        "should_be_knob": "No — minimum composite is the correct blending for white-background ink; alternatives would produce unrealistic artefacts.",
    },
    {
        "effect": "glyph_rotation clip range: [-18, +18] degrees",
        "file": "src/generation/geometry.py:17-18",
        "eli5": "The per-glyph rotation is clamped to at most 18 degrees regardless of the config value. "
                "This prevents upside-down digits. The clip limits are hardcoded as module-level constants.",
        "should_be_knob": "No — extreme rotations would make digits unrecognisable. The 18-degree cap is a safety guard.",
    },
    {
        "effect": "top_pad / margin_bottom (now page_padding_px knob, default 10)",
        "file": "src/generation/render_examples.py + synth_yolo.py",
        "eli5": "Previously hardcoded as 10px. Now exposed as the page_padding_px knob in [generation.rendering]. "
                "Changing it controls how much white space appears above and below the drawn equation.",
        "should_be_knob": "Yes — promoted to page_padding_px in [generation.rendering].",
    },
    {
        "effect": "Completion stage drop_fraction range: 0.50 – 0.80 (partial carries/borrows)",
        "file": "src/generation/completion_stages.py:86",
        "eli5": "When the partial_carries_or_borrows stage fires, between 50% and 80% of carry/borrow tokens "
                "are randomly dropped. This range is hardcoded.",
        "should_be_knob": "Possibly — a 'partial_cb_drop_min/max' pair would give control over how incomplete the partial stages are.",
    },
]

# Knob pairs to render: (toml_key_slug, off_override_dict, on_override_dict, scene_case, completion_stage)
# We reuse the CANONICAL_CASES infrastructure but inject rendering overrides.
RENDER_PAIRS = [
    # ---- Scene-level ----
    # scene_rotation: slug must match KNOBS entry "scene_rotation_min_deg / scene_rotation_max_deg"
    # slug = "scene_rotation_min_deg"
    ("scene_rotation_min_deg", {"scene_rotation_min_deg": 0.0, "scene_rotation_max_deg": 0.0},
                                {"scene_rotation_min_deg": -8.0, "scene_rotation_max_deg": 0.0},
                                "addition", "full"),
    # slug = "scene_rotation_max_deg" — need a separate KNOBS entry? No: KNOBS compound entry gives one slug.
    # We add a supplementary pair for max_deg so that slug is also covered; both share card via build_html.
    # Actually max_deg has no KNOBS entry so this pair is only for the image assets referenced by min_deg card.
    # We use slug "scene_rotation_min_deg" to cover the combined knob (see KNOBS rename below).
    # crowdness_prob: slug = "crowdness_prob"
    ("crowdness_prob", {"crowdness_prob": 0.0},
                        {"crowdness_prob": 1.0},
                        "addition", "full"),
    # crowding_factor: slug = "crowding_factor"; force crowdness_prob=1.0 both sides so spacing is always crowded
    ("crowding_factor", {"crowdness_prob": 1.0, "crowding_factor": 0.95},
                         {"crowdness_prob": 1.0, "crowding_factor": 0.55},
                         "addition", "full"),
    # wrong result — scene knob, needs _apply_scene_knobs to fire
    ("wrong_result_prob", {"wrong_result_prob": 0.0},
                           {"wrong_result_prob": 1.0},
                           "addition", "full", "scene"),
    # missing structural — scene knob, use division-long (has structural tokens: bar + bracket)
    ("missing_structural_prob", {"missing_structural_prob": 0.0},
                                 {"missing_structural_prob": 1.0},
                                 "division-long", "full", "scene"),
    # wrong operator — scene knob
    ("wrong_operator_prob", {"wrong_operator_prob": 0.0},
                             {"wrong_operator_prob": 1.0},
                             "multiplication-multi", "full", "scene"),
    # ---- Per-glyph ----
    # glyph_rotation: slug = "glyph_rotation_min_deg"
    ("glyph_rotation_min_deg", {"glyph_rotation_min_deg": 0.0, "glyph_rotation_max_deg": 0.0},
                                {"glyph_rotation_min_deg": -15.0, "glyph_rotation_max_deg": 15.0},
                                "addition", "full"),
    # glyph_jitter: slug = "glyph_jitter_min_px"
    ("glyph_jitter_min_px", {"glyph_jitter_min_px": 0, "glyph_jitter_max_px": 0},
                              {"glyph_jitter_min_px": 0, "glyph_jitter_max_px": 4},
                              "addition", "full"),
    # broken stroke: slug = "glyph_broken_stroke_prob"
    ("glyph_broken_stroke_prob", {"glyph_broken_stroke_prob": 0.0},
                                  {"glyph_broken_stroke_prob": 1.0},
                                  "addition", "full"),
    # morph close kernel — larger kernel for visible effect
    ("glyph_morph_close_kernel", {"glyph_morph_close_kernel": 0},
                                  {"glyph_morph_close_kernel": 7},
                                  "addition", "full"),
    # stroke target — large range for visible effect
    ("glyph_stroke_target_px", {"glyph_stroke_target_px": 1.5},
                                 {"glyph_stroke_target_px": 4.5},
                                 "addition", "full"),
    # broken stroke band frac (needs broken_stroke_prob=1 so the gap always fires)
    ("broken_stroke_band_frac", {"glyph_broken_stroke_prob": 1.0, "broken_stroke_band_frac": 0.05},
                                 {"glyph_broken_stroke_prob": 1.0, "broken_stroke_band_frac": 0.40},
                                 "addition", "full"),
    # page padding range — use center_vertically=False so padding is respected
    ("page_padding_max_px", {"page_padding_min_px": 2, "page_padding_max_px": 2},
                              {"page_padding_min_px": 40, "page_padding_max_px": 80},
                              "addition", "full", "no_center"),
    # carry/borrow scale — use addition_dense_carries to guarantee carry tokens are present
    # OFF: both bounds at 1.0 (no shrink); ON: min=0.40, max=0.80 to show full per-token variance range
    ("carry_borrow_scale_min", {"carry_borrow_scale_min": 1.0, "carry_borrow_scale_max": 1.0},
                                {"carry_borrow_scale_min": 0.40, "carry_borrow_scale_max": 0.80},
                                "addition_dense_carries", "full"),
    # operator scale
    ("operator_scale", {"operator_scale": 1.0},
                        {"operator_scale": 0.62},
                        "addition", "full"),
    # base cell px
    ("base_cell_px", {"base_cell_px": 22},
                      {"base_cell_px": 55},
                      "addition", "full"),
    # gap_x: slug = "gap_x_min"
    ("gap_x_min", {"gap_x_min": 2, "gap_x_max": 2},
                   {"gap_x_min": 8, "gap_x_max": 12},
                   "addition", "full"),
    # gap_y: slug = "gap_y_min" — use addition_dense_carries for multi-row scene
    ("gap_y_min", {"gap_y_min": 2, "gap_y_max": 2},
                   {"gap_y_min": 8, "gap_y_max": 12},
                   "addition_dense_carries", "full"),
    # bar gap
    ("bar_gap_prob", {"bar_gap_prob": 0.0},
                      {"bar_gap_prob": 1.0},
                      "addition", "full"),
    # bar thickness: slug = "bar_thickness_min" — subtraction (forces bar visible)
    ("bar_thickness_min", {"bar_thickness_min": 1, "bar_thickness_max": 1},
                           {"bar_thickness_min": 5, "bar_thickness_max": 5},
                           "subtraction", "full"),
    # bar intensity: slug = "bar_intensity_min" — subtraction
    ("bar_intensity_min", {"bar_intensity_min": 0, "bar_intensity_max": 0},
                           {"bar_intensity_min": 150, "bar_intensity_max": 150},
                           "subtraction", "full"),
    # bar micro break
    ("bar_micro_break_prob", {"bar_micro_break_prob": 0.0},
                              {"bar_micro_break_prob": 1.0},
                              "addition", "full"),
    # bar y jitter — large sigma for visible wobble
    ("bar_y_jitter_sigma", {"bar_y_jitter_sigma": 0.0},
                            {"bar_y_jitter_sigma": 10.0},
                            "subtraction", "full"),
    # bar x jitter — large sigma so endpoints visibly shift
    ("bar_x_jitter_sigma", {"bar_x_jitter_sigma": 0.0},
                            {"bar_x_jitter_sigma": 25.0},
                            "addition", "full"),
    # short bar
    ("short_bar_prob", {"short_bar_prob": 0.0},
                        {"short_bar_prob": 1.0, "short_bar_shrink_frac": 0.40},
                        "addition", "full"),
    # short bar shrink frac — force short_bar_prob=1.0 both sides, vary frac dramatically
    ("short_bar_shrink_frac", {"short_bar_prob": 1.0, "short_bar_shrink_frac": 0.95},
                               {"short_bar_prob": 1.0, "short_bar_shrink_frac": 0.40},
                               "addition", "full"),
    # bracket gap: heavy sigma to guarantee visible gaps
    ("bracket_gap_prob", {"bracket_gap_prob": 0.0},
                          {"bracket_gap_prob": 1.0},
                          "division-long", "full"),
    # bracket thickness: slug = "bracket_thickness_min"
    ("bracket_thickness_min", {"bracket_thickness_min": 1, "bracket_thickness_max": 1},
                               {"bracket_thickness_min": 5, "bracket_thickness_max": 5},
                               "division-long", "full"),
    # bracket intensity: slug = "bracket_intensity_min"
    ("bracket_intensity_min", {"bracket_intensity_min": 0, "bracket_intensity_max": 0},
                               {"bracket_intensity_min": 150, "bracket_intensity_max": 150},
                               "division-long", "full"),
    # bracket micro break
    ("bracket_micro_break_prob", {"bracket_micro_break_prob": 0.0},
                                  {"bracket_micro_break_prob": 1.0},
                                  "division-long", "full"),
    # bracket h y jitter
    ("bracket_h_y_jitter_sigma", {"bracket_h_y_jitter_sigma": 0.0},
                                  {"bracket_h_y_jitter_sigma": 8.0},
                                  "division-long", "full"),
    # bracket h x jitter — very large sigma so endpoints visibly shift
    ("bracket_h_x_jitter_sigma", {"bracket_h_x_jitter_sigma": 0.0},
                                  {"bracket_h_x_jitter_sigma": 25.0},
                                  "division-long", "full"),
    # bracket v x jitter
    ("bracket_v_x_jitter_sigma", {"bracket_v_x_jitter_sigma": 0.0},
                                  {"bracket_v_x_jitter_sigma": 8.0},
                                  "division-long", "full"),
    # bracket depth full prob — force arm always shortened vs always full
    ("bracket_depth_full_prob", {"bracket_depth_full_prob": 1.0, "bracket_depth_min_rows": 1},
                                 {"bracket_depth_full_prob": 0.0, "bracket_depth_min_rows": 1},
                                 "division-long", "full"),
    # bracket depth min rows — arm shortened both sides; min=1 vs min=12 (near-full)
    ("bracket_depth_min_rows", {"bracket_depth_full_prob": 0.0, "bracket_depth_min_rows": 1},
                                {"bracket_depth_full_prob": 0.0, "bracket_depth_min_rows": 12},
                                "division-long", "full"),
    # ---- Error injection ----
    # borrow cross: subtraction_heavy_borrow guarantees borrow tokens present
    ("borrow_cross_prob", {"borrow_cross_prob": 0.0},
                           {"borrow_cross_prob": 1.0},
                           "subtraction_heavy_borrow", "full"),
    # step minus
    ("step_minus_prob", {"step_minus_prob": 0.0},
                         {"step_minus_prob": 1.0},
                         "division-long", "full"),
    # wrong carry col: addition_dense_carries guarantees carry tokens
    ("wrong_carry_col_prob", {"wrong_carry_col_prob": 0.0},
                              {"wrong_carry_col_prob": 1.0},
                              "addition_dense_carries", "full"),
    # missing carry — use addition_dense_carries to guarantee carry tokens are present
    ("missing_carry_prob", {"missing_carry_prob": 0.0},
                            {"missing_carry_prob": 1.0},
                            "addition_dense_carries", "full"),
    # wrong carry value
    ("wrong_carry_value_prob", {"wrong_carry_value_prob": 0.0},
                                {"wrong_carry_value_prob": 1.0},
                                "addition_dense_carries", "full"),
    # missing borrow: subtraction_heavy_borrow guarantees borrow tokens
    ("missing_borrow_prob", {"missing_borrow_prob": 0.0},
                             {"missing_borrow_prob": 1.0},
                             "subtraction_heavy_borrow", "full"),
    # wrong borrow value
    ("wrong_borrow_value_prob", {"wrong_borrow_value_prob": 0.0},
                                 {"wrong_borrow_value_prob": 1.0},
                                 "subtraction_heavy_borrow", "full"),
    # pp plus
    ("pp_plus_prob", {"pp_plus_prob": 0.0},
                      {"pp_plus_prob": 1.0},
                      "multiplication-multi", "full"),
    # pp wrong operator
    ("pp_wrong_operator_prob", {"pp_plus_prob": 1.0, "pp_wrong_operator_prob": 0.0},
                                {"pp_plus_prob": 1.0, "pp_wrong_operator_prob": 1.0},
                                "multiplication-multi", "full"),
    # long div step borrow
    ("long_div_step_borrow_prob", {"long_div_step_borrow_prob": 0.0},
                                   {"long_div_step_borrow_prob": 1.0},
                                   "division-long", "full"),
    # helper operator
    ("helper_operator_prob", {"helper_operator_prob": 0.0},
                              {"helper_operator_prob": 1.0},
                              "division-long", "full"),
    # missing pp
    ("missing_pp_prob", {"missing_pp_prob": 0.0},
                         {"missing_pp_prob": 1.0},
                         "multiplication-multi", "full"),
]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_pair(slug: str, off_overrides: dict, on_overrides: dict, scene_case_str: str,
                 completion_stage: str, project_root: Path, symbol_manifest_path: Path,
                 gen_cfg, imgs_dir: Path, render_flags: str = "") -> tuple[str, str]:
    """Render OFF and ON PNGs; return (off_rel_path, on_rel_path) relative to imgs_dir parent.

    render_flags:
      "scene"     — call _apply_scene_knobs so scene-level error injection fires
      "no_center" — use center_in_full_canvas_vertically=False so page_padding_px is respected
    """
    import random as _random
    from src.generation.layouts import SceneCase, sample_layout_for_case
    from src.generation.completion_stages import sample_completion_stage_for_case
    from src.generation.handwriting_style import make_style
    from src.generation.synth_pool import _draw_equation_block, _load_symbol_rows
    from src.generation.synth_yolo import _apply_scene_knobs
    from src.data_pipeline.preprocessing import preprocess_for_pipeline

    import hashlib as _hashlib
    seed = int(_hashlib.md5(slug.encode()).hexdigest()[:8], 16) % (2 ** 31)
    use_scene_knobs = "scene" in render_flags
    center_vertically = "no_center" not in render_flags

    def _render(overrides: dict, label: str) -> Path:
        rng_py = _random.Random(seed)
        rng_np = np.random.default_rng(seed)
        scene_case = SceneCase(scene_case_str)
        _, tokens = sample_layout_for_case(scene_case, rng_py)
        if not tokens:
            raise ValueError(f"No tokens for {scene_case_str}")
        comp_weights = {completion_stage: 1.0}
        _stage, tokens = sample_completion_stage_for_case(scene_case, tokens, rng_py, weights=comp_weights)
        W, H = 512, 512
        canvas = np.full((H, W), 255, dtype=np.uint8)
        # Build effective rendering
        effective = dict(gen_cfg.rendering)
        effective.update(overrides)
        # Patch scene and preset configs from overrides
        import dataclasses
        from src.core.run_config import SceneConfig, PresetConfig
        scene_overrides_keys = {"scene_rotation_min_deg", "scene_rotation_max_deg", "crowdness_prob", "crowding_factor",
                                "wrong_result_prob", "missing_structural_prob", "wrong_operator_prob"}
        preset_keys = {"glyph_rotation_min_deg", "glyph_rotation_max_deg", "glyph_jitter_min_px", "glyph_jitter_max_px", "glyph_broken_stroke_prob"}
        sc_kwargs = dataclasses.asdict(gen_cfg.scene)
        pc_kwargs = dataclasses.asdict(gen_cfg.preset)
        for k, v in overrides.items():
            if k in scene_overrides_keys:
                sc_kwargs[k] = v
            if k in preset_keys:
                pc_kwargs[k] = v
        patched_scene = SceneConfig(**sc_kwargs)
        patched_preset = PresetConfig(**pc_kwargs)
        style = make_style(rng_np, patched_preset, scene_cfg=patched_scene,
                           crowded=(rng_py.random() < patched_scene.crowdness_prob))
        # Apply scene-level token mutations (wrong_result, missing_structural, wrong_operator)
        if use_scene_knobs:
            tokens = _apply_scene_knobs(tokens, rng_py, patched_scene)
        # Load symbols
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
        pool_root = None
        source_pool = gen_cfg.source_pool
        if source_pool != "legacy":
            candidate = project_root / "data" / f"combined_{source_pool}"
            if candidate.exists():
                pool_root = candidate
        _pad_min = int(effective.get("page_padding_min_px", 2))
        _pad_max = int(effective.get("page_padding_max_px", 10))
        _page_pad = rng_py.randint(_pad_min, max(_pad_min, _pad_max))
        top_pad, margin_bottom = _page_pad, _page_pad
        _draw_equation_block(
            canvas, rng_py,
            symbols_by_glyph_key, fallback_by_glyph,
            W, H, top_pad, H - top_pad - margin_bottom,
            tokens, 0,
            center_in_full_canvas_vertically=center_vertically,
            style=style,
            rendering=effective,
            scene_case=scene_case,
            pool_root=pool_root,
            min_source_quality=gen_cfg.min_source_quality,
        )
        # Scene rotation
        rot_min = sc_kwargs.get("scene_rotation_min_deg", 0.0)
        rot_max = sc_kwargs.get("scene_rotation_max_deg", 0.0)
        if abs(rot_min) > 0.01 or abs(rot_max) > 0.01:
            angle = rng_py.uniform(rot_min, rot_max)
            import cv2
            center = (W // 2, H // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            canvas = cv2.warpAffine(canvas, M, (W, H), borderValue=255)
        result = preprocess_for_pipeline(canvas)
        out_path = imgs_dir / f"{slug}_{label}.png"
        Image.fromarray(result).save(out_path)
        return out_path

    off_path = _render(off_overrides, "off")
    on_path = _render(on_overrides, "on")
    return (f"imgs/{off_path.name}", f"imgs/{on_path.name}")


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       background: #fff; color: #111; padding: 2rem; max-width: 1400px; margin: 0 auto; }
h1 { font-size: 1.6rem; margin-bottom: 0.5rem; }
.subtitle { color: #555; margin-bottom: 2rem; font-size: 0.95rem; }
h2 { font-size: 1.2rem; margin: 2.5rem 0 1rem; border-bottom: 2px solid #e0e0e0; padding-bottom: 0.4rem; }
h3 { font-size: 0.95rem; margin: 1.5rem 0 0.6rem; border-bottom: 1px solid #eee; padding-bottom: 0.25rem; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(560px, 1fr)); gap: 1.5rem; }
.card { border: 1px solid #ddd; border-radius: 6px; padding: 1rem; background: #fafafa; }
.card-title { font-weight: 600; font-size: 1rem; font-family: monospace; margin-bottom: 0.4rem; }
.eli5 { font-size: 0.85rem; color: #333; margin-bottom: 0.75rem; line-height: 1.5; }
.images { display: flex; gap: 1rem; margin-bottom: 0.6rem; }
.img-block { text-align: center; }
.img-block img { width: 240px; height: 240px; object-fit: contain; border: 1px solid #ccc;
                 background: #fff; display: block; }
.img-label { font-size: 0.75rem; color: #666; margin-top: 3px; }
.meta { font-size: 0.75rem; color: #666; border-top: 1px solid #eee; padding-top: 0.5rem; }
.meta span { display: inline-block; margin-right: 1rem; }
.badge { display: inline-block; font-size: 0.68rem; padding: 1px 6px; border-radius: 10px;
         font-weight: 600; margin-right: 4px; }
.badge-scene { background: #e0f0ff; color: #1060a0; }
.badge-glyph { background: #e8ffe0; color: #206020; }
.badge-error { background: #fff0e0; color: #904010; }
.badge-completion { background: #f0e0ff; color: #602090; }
.hardcoded-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.hardcoded-table th { background: #f4f4f4; text-align: left; padding: 6px 10px; border-bottom: 2px solid #ccc; }
.hardcoded-table td { padding: 6px 10px; border-bottom: 1px solid #eee; vertical-align: top; }
.hardcoded-table tr:hover td { background: #fafafa; }
.knob-yes { color: #a05000; font-weight: 600; }
.knob-no  { color: #888; }
"""


def _badge(scope: str) -> str:
    classes = {"scene": "badge-scene", "glyph": "badge-glyph", "error": "badge-error",
                "completion": "badge-completion", "scene / error": "badge-error"}
    cls = classes.get(scope.split("/")[0].strip(), "badge-scene")
    return f'<span class="badge {cls}">{scope}</span>'


def build_html(knobs, render_map: dict, hardcoded: list, output_path: Path,
               pair_meta: dict | None = None, pair_diff: dict | None = None) -> None:
    scope_order = [("Scene-level manipulations", "scene"),
                   ("Per-glyph manipulations", "glyph"),
                   ("Error injection (realistic child mistakes)", "error"),
                   ("Completion stage mix", "completion")]

    sections = []
    for section_title, scope_key in scope_order:
        scoped = [k for k in knobs if scope_key in k[1]]
        if not scoped:
            continue
        cards_html = []
        for (name, scope, typ, off_val, on_val, default_val, config_val, eli5, consumer) in scoped:
            slug = name.split("/")[0].strip().replace(" ", "_")
            pair = render_map.get(slug, ("", ""))
            off_img = f'<img src="{pair[0]}" alt="OFF" loading="lazy">' if pair[0] else '<div style="width:240px;height:240px;background:#f5f5f5;display:flex;align-items:center;justify-content:center;color:#bbb;font-size:0.75rem">no image</div>'
            on_img = f'<img src="{pair[1]}" alt="ON" loading="lazy">' if pair[1] else '<div style="width:240px;height:240px;background:#f5f5f5;display:flex;align-items:center;justify-content:center;color:#bbb;font-size:0.75rem">no image</div>'
            meta_sc, meta_stage = (pair_meta or {}).get(slug, ("", ""))
            same_scene_note = (
                f'<span style="color:#1060a0">Same scene ({meta_sc}, stage: {meta_stage}), only <code>{name}</code> differs.</span><br>'
                if meta_sc else ""
            )
            diff_note = ""
            if pair_diff and slug in pair_diff:
                mad, mx, npx = pair_diff[slug]
                if mad == 0.0:
                    diff_note = '<span style="color:#c00;font-weight:600">Note: OFF and ON images are identical. This knob may not be exercised in this scene configuration.</span><br>'
                elif mad < 1.0 and npx > 0:
                    diff_note = f'<span style="color:#806000">Note: change is spatially concentrated (MAD={mad:.2f}, {npx} pixels differ at max&Delta;={mx:.0f}). Zoom in to the bar or bracket region to see the effect.</span><br>'
            card = f"""<div class="card">
  <div class="card-title">{_badge(scope)} {name}</div>
  <div class="eli5">{eli5}</div>
  <div class="images">
    <div class="img-block">{off_img}<div class="img-label">OFF ({off_val})</div></div>
    <div class="img-block">{on_img}<div class="img-label">ON ({on_val})</div></div>
  </div>
  <div class="meta">
    <span>type: {typ}</span>
    <span>default: {default_val}</span>
    <span>config.toml: {config_val}</span><br>
    {same_scene_note}{diff_note}
    <span style="color:#888">consumer: {consumer}</span>
  </div>
</div>"""
            cards_html.append(card)
        sections.append(f"<h2>{section_title}</h2>\n<div class='cards'>\n" + "\n".join(cards_html) + "\n</div>")

    # Completion section (no images)
    completion_html = """<h2>Completion stage mix</h2>
<p style="font-size:0.9rem;color:#555;margin-bottom:1rem">
  The completion mix controls how far through a problem a child has written. It is not a single knob but a
  probability distribution over 5 stages configured in <code>[generation.completion]</code>.
  65% of scenes are fully written; the remaining 35% are at various partial stages where
  later tokens (result digits, bars, carry rows) are progressively removed.
  This is a data-mix parameter rather than a visual manipulation.
</p>
<table class="hardcoded-table">
<tr><th>Stage</th><th>Default weight</th><th>Config value</th><th>What is removed</th></tr>
<tr><td>full</td><td>0.65</td><td>0.65</td><td>Nothing. All tokens present.</td></tr>
<tr><td>done_80</td><td>0.15</td><td>0.15</td><td>Last 20% of result-row digits dropped (child is almost done).</td></tr>
<tr><td>done_60</td><td>0.10</td><td>0.07</td><td>Last 40% of result-row digits dropped.</td></tr>
<tr><td>done_40</td><td>0.06</td><td>0.06</td><td>Entire result row and result bar dropped.</td></tr>
<tr><td>done_20</td><td>0.04</td><td>0.07</td><td>Operands only; nothing else written yet.</td></tr>
</table>"""
    sections.append(completion_html)

    # Hardcoded effects
    rows = []
    for h in hardcoded:
        yn = "Yes" if h["should_be_knob"].startswith("Yes") else ("Possibly" if h["should_be_knob"].startswith("Possibly") else "No")
        cls = "knob-yes" if yn in ("Yes", "Possibly") else "knob-no"
        rows.append(
            f"<tr><td><code>{h['effect']}</code></td><td><code>{h['file']}</code></td>"
            f"<td>{h['eli5']}</td><td class='{cls}'>{h['should_be_knob']}</td></tr>"
        )
    hardcoded_section = """<h2>Hardcoded effects (no config knob)</h2>
<p style="font-size:0.9rem;color:#555;margin-bottom:1rem">
  The following rendering behaviours are controlled by constants in source code rather than config.toml knobs.
</p>
<table class="hardcoded-table">
<tr><th>Effect</th><th>File:line</th><th>ELI5</th><th>Should become a knob?</th></tr>
""" + "\n".join(rows) + "\n</table>"
    sections.append(hardcoded_section)

    # Stats
    n_knobs = len(knobs)
    scope_counts = {}
    for k in knobs:
        s = k[1].split("/")[0].strip()
        scope_counts[s] = scope_counts.get(s, 0) + 1

    scope_summary = ", ".join(f"{v} {k}" for k, v in sorted(scope_counts.items()))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Generation Knob Explorer</title>
<style>{CSS}</style>
</head>
<body>
<h1>Generation Manipulation Knob Explorer</h1>
<p class="subtitle">
  One-shot reference for every generation manipulation parameter in the handwritten-arithmetic-recognition pipeline.
  Each knob shows an OFF (min/0) vs ON (max/1) rendered 512x512 scene side by side with a plain-language explanation.
  Total knobs documented: <strong>{n_knobs}</strong> ({scope_summary}).
</p>
{"".join(sections)}
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    PP = args.project_root.resolve()

    sys.path.insert(0, str(PP))

    from src.core.run_config import load_config
    from src.core.config import DataPrepConfig

    gen_cfg, _, _ = load_config(PP)
    config = DataPrepConfig.from_project_root(PP)
    symbol_manifest_path = config.yolo_dir / "symbol_assets_manifest.csv"

    if not symbol_manifest_path.exists():
        print(f"ERROR: symbol manifest not found at {symbol_manifest_path}", file=sys.stderr)
        sys.exit(1)

    imgs_dir = PP / "reports" / "knob_explorer" / "imgs"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    output_path = PP / "reports" / "knob_explorer" / "index.html"

    render_map: dict[str, tuple[str, str]] = {}
    total = len(RENDER_PAIRS)
    for i, entry in enumerate(RENDER_PAIRS):
        slug, off_ov, on_ov, sc, stage = entry[0], entry[1], entry[2], entry[3], entry[4]
        flags = entry[5] if len(entry) > 5 else ""
        print(f"  [{i+1:02d}/{total}] Rendering pair: {slug} ({sc}) flags={flags!r}")
        try:
            pair = _render_pair(slug, off_ov, on_ov, sc, stage, PP, symbol_manifest_path, gen_cfg, imgs_dir, render_flags=flags)
            render_map[slug] = pair
        except Exception as exc:
            import traceback
            print(f"    WARN: {slug} failed: {exc}")
            traceback.print_exc()
            render_map[slug] = ("", "")

    # Build pair_meta: slug → (scene_case, completion_stage) for same-scene contract note
    pair_meta = {entry[0]: (entry[3], entry[4]) for entry in RENDER_PAIRS}

    # Compute pixel diff stats for each rendered pair
    pair_diff: dict[str, tuple[float, float, int]] = {}  # slug → (MAD, maxdiff, npx)
    for slug, (off_p, on_p) in render_map.items():
        if off_p and on_p:
            try:
                off_full = PP / "reports" / "knob_explorer" / off_p
                on_full = PP / "reports" / "knob_explorer" / on_p
                if off_full.exists() and on_full.exists():
                    off_arr = np.array(Image.open(off_full).convert("L"), dtype=float)
                    on_arr = np.array(Image.open(on_full).convert("L"), dtype=float)
                    d = np.abs(off_arr - on_arr)
                    pair_diff[slug] = (float(d.mean()), float(d.max()), int((d > 0).sum()))
            except Exception:
                pass

    print("\nPer-pair MAD:")
    zero_mad_slugs = []
    for slug in sorted(pair_diff, key=lambda s: pair_diff[s][0]):
        mad, mx, npx = pair_diff[slug]
        flag = " <-- MAD<1 (spatially concentrated change)" if mad < 1.0 else ""
        if mad == 0.0:
            flag = " <-- ZERO DIFF"
            zero_mad_slugs.append(slug)
        print(f"  {mad:7.3f}  {slug}{flag}")
    if zero_mad_slugs:
        print(f"\nWARN: Zero-diff slugs: {zero_mad_slugs}")
    else:
        print("\nNo zero-diff pairs.")

    print("\nBuilding HTML...")
    build_html(KNOBS, render_map, HARDCODED, output_path, pair_meta=pair_meta, pair_diff=pair_diff)

    # Verification
    missing = []
    for slug, (off_p, on_p) in render_map.items():
        for rel in (off_p, on_p):
            if rel:
                full = PP / "reports" / "knob_explorer" / rel
                if not full.exists():
                    missing.append(str(full))

    print(f"\n=== Summary ===")
    print(f"Total knobs documented: {len(KNOBS)}")
    scope_counts = {}
    for k in KNOBS:
        s = k[1].split("/")[0].strip()
        scope_counts[s] = scope_counts.get(s, 0) + 1
    for scope, cnt in sorted(scope_counts.items()):
        print(f"  {scope}: {cnt}")
    print(f"Hardcoded effects flagged: {len(HARDCODED)}")
    print(f"Image pairs attempted: {len(RENDER_PAIRS)}")
    print(f"Images present: {len(list(imgs_dir.glob('*.png')))}")
    if missing:
        print(f"Missing images: {missing}")
    else:
        print("All image paths verified.")
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
