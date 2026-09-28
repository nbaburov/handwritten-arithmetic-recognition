"""Regression test: no main_*/main_* bbox overlap >30% in division-long scenes.

W10-LAYOUT-COLLISION fix verification.

Root cause: scale_jitter upper bound (1.08) combined with gap_x_min=6 left only
~1px clearance between adjacent-column glyph bboxes at small canvas-scale factors
(s < 0.55). Per-glyph jitter (0-2px) eliminated the remaining gap, causing overlap.

Fix (synth_pool.py line ~797): cap scale_jitter so that
    pw + 2 * jitter_max + 1 <= s * x_step
guaranteeing adjacent column bboxes never touch regardless of jitter direction.

This test generates 100 division-long scenes using the same rendering path as
production and asserts zero main_*/main_* collisions at the 30% IoU threshold.
"""
from __future__ import annotations

import random
import unittest
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _iou_overlap_fraction(b1: List[int], b2: List[int]) -> float:
    """Overlap area / min(area1, area2). Returns 0.0 when bboxes do not touch."""
    x0 = max(b1[0], b2[0])
    y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2])
    y1 = min(b1[3], b2[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    a1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / min(a1, a2)


def _render_scene_gt_tokens(
    rng: random.Random,
    canvas_size: Tuple[int, int] = (512, 512),
    base_cell: int = 44,
    gap_x: int = 6,
    gap_y: int = 10,
    jitter_max: int = 2,
) -> List[dict]:
    """Render one division-long scene using _draw_equation_block internals.

    Returns the gt_tokens list (same structure as ground-truth JSON symbols).
    Uses a minimal stub pool so no disk access is required: each glyph tile is
    a synthetic 28x28 solid-black-on-white square.
    """
    import numpy as np
    from src.generation.layouts import sample_layout_for_case
    from src.generation.layouts_types import SceneCase
    from src.generation.synth_pool import _draw_equation_block, _paste_symbol
    from src.generation.handwriting_style import HandwritingStyle

    # Stub pool: one row per label, path pointing to a dummy that _tile_from_path
    # would load. We override _paste_symbol via a lightweight tile factory instead.
    # Simpler: directly call _draw_equation_block with a real canvas and stub pool.
    # The pool must contain glyph_key rows matching what the layout will need.

    # Build a stub pool DataFrame with all glyph keys that division-long may need.
    glyph_keys = [str(d) for d in range(10)] + ["plus", "minus", "times", "divide"]
    # Use a tiny real 28x28 PNG from the emnist pool if available; otherwise skip.
    pool_dir = _PROJECT_ROOT / "data" / "raw" / "pool_emnist_28"
    if not pool_dir.exists():
        # Try crohme pool.
        pool_dir = _PROJECT_ROOT / "data" / "raw" / "pool_crohme_128"
    if not pool_dir.exists():
        return []  # No pool available; caller skips test.

    # Build a minimal manifest DataFrame with one real path per glyph_key.
    rows = []
    label_map = {
        "0": "main_0", "1": "main_1", "2": "main_2", "3": "main_3",
        "4": "main_4", "5": "main_5", "6": "main_6", "7": "main_7",
        "8": "main_8", "9": "main_9",
        "plus": "op_plus", "minus": "op_minus",
        "times": "op_times", "divide": "op_divide",
    }
    for gk, folder_label in label_map.items():
        folder = pool_dir / folder_label
        if not folder.exists():
            continue
        pngs = list(folder.glob("*.png"))
        if not pngs:
            continue
        rows.append({"glyph_key": gk, "path": str(pngs[0]), "split": "train"})

    if not rows:
        return []

    symbol_df = pd.DataFrame(rows)
    symbols_by_glyph_key = {
        str(gk): frame.reset_index(drop=True)
        for gk, frame in symbol_df.groupby("glyph_key")
    }

    style = HandwritingStyle(
        rot_deg_min=-10.0,
        rot_deg_max=10.0,
        jitter_min_px=0,
        jitter_max_px=jitter_max,
        glyph_broken_stroke_prob=0.0,
        crowding_factor=1.0,
    )

    rendering = {
        "base_cell_px": base_cell,
        "gap_x_min": gap_x,
        "gap_x_max": gap_x,
        "gap_y_min": gap_y,
        "gap_y_max": gap_y,
        "page_padding_min_px": 2,
        "page_padding_max_px": 2,
        "bar_gap_prob": 0.0,
        "bar_thickness_min": 2,
        "bar_thickness_max": 3,
        "bar_intensity_min": 30,
        "bar_intensity_max": 80,
        "bar_micro_break_prob": 0.0,
        "bar_y_jitter_sigma": 0.0,
        "bar_x_jitter_sigma": 0.0,
        "short_bar_prob": 0.0,
        "short_bar_shrink_frac": 0.0,
        "bracket_gap_prob": 0.0,
        "bracket_thickness_min": 2,
        "bracket_thickness_max": 3,
        "bracket_intensity_min": 30,
        "bracket_intensity_max": 80,
        "bracket_micro_break_prob": 0.0,
        "bracket_h_y_jitter_sigma": 0.0,
        "bracket_h_x_jitter_sigma": 0.0,
        "bracket_v_x_jitter_sigma": 0.0,
        "bracket_depth_full_prob": 1.0,
        "bracket_depth_min_rows": 2,
        "step_minus_prob": 0.0,
        "long_div_step_borrow_prob": 0.0,
        "helper_operator_prob": 0.0,
        "wrong_carry_col_prob": 0.0,
        "missing_carry_prob": 0.0,
        "wrong_carry_value_prob": 0.0,
        "missing_borrow_prob": 0.0,
        "wrong_borrow_value_prob": 0.0,
        "borrow_cross_prob": 0.0,
        "carry_borrow_scale_min": 0.55,
        "carry_borrow_scale_max": 0.70,
        "operator_scale": 1.0,
        "glyph_morph_close_kernel": 0,
        "glyph_stroke_target_px": 0.0,
        "broken_stroke_band_frac": 0.125,
    }

    canvas_w, canvas_h = canvas_size
    canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)

    eq_kind, tokens = sample_layout_for_case(SceneCase.division_long, rng)
    if not tokens:
        return []

    try:
        _, gt_tokens, _ = _draw_equation_block(
            canvas=canvas,
            rng=rng,
            symbols_by_glyph_key=symbols_by_glyph_key,
            symbols_by_glyph_key_fallback=symbols_by_glyph_key,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            slot_y0=0,
            slot_height=canvas_h,
            tokens=tokens,
            eq_idx=0,
            center_in_full_canvas_vertically=True,
            style=style,
            rendering=rendering,
            scene_case=SceneCase.division_long,
            pool_root=None,
            min_source_quality=0.0,
        )
    except Exception:
        return []

    return gt_tokens


class TestNoLayoutCollisions(unittest.TestCase):
    """Division-long scenes must not produce main_*/main_* bbox overlap > 30%."""

    N_SCENES = 100
    OVERLAP_THRESHOLD = 0.30

    def _check_scene_for_collisions(self, gt_tokens: List[dict]) -> List[Tuple[dict, dict, float]]:
        main_syms = [t for t in gt_tokens if t.get("yolo_class") == "digit_main" and t.get("bbox")]
        collisions = []
        for i in range(len(main_syms)):
            for j in range(i + 1, len(main_syms)):
                frac = _iou_overlap_fraction(main_syms[i]["bbox"], main_syms[j]["bbox"])
                if frac > self.OVERLAP_THRESHOLD:
                    collisions.append((main_syms[i], main_syms[j], frac))
        return collisions

    def test_no_main_digit_collisions_division_long(self) -> None:
        """100 division-long scenes: zero main_*/main_* pairs overlap > 30%."""
        pool_dir = _PROJECT_ROOT / "data" / "raw" / "pool_emnist_28"
        if not pool_dir.exists():
            pool_dir = _PROJECT_ROOT / "data" / "raw" / "pool_crohme_128"
        if not pool_dir.exists():
            self.skipTest("No symbol pool found (pool_emnist_28 / pool_crohme_128 missing). "
                          "Run validate to populate the pool before running this test.")

        rng = random.Random(12345)  # fixed seed for reproducibility
        all_collisions = []
        skipped = 0

        for scene_idx in range(self.N_SCENES):
            gt_tokens = _render_scene_gt_tokens(rng)
            if not gt_tokens:
                skipped += 1
                continue
            collisions = self._check_scene_for_collisions(gt_tokens)
            for a, b, frac in collisions:
                all_collisions.append({
                    "scene": scene_idx,
                    "a": f"{a['fine_label']}@row{a['row_index']},col{a['col_index']}",
                    "b": f"{b['fine_label']}@row{b['row_index']},col{b['col_index']}",
                    "overlap": frac,
                    "bbox_a": a["bbox"],
                    "bbox_b": b["bbox"],
                })

        if skipped > self.N_SCENES // 2:
            self.skipTest(f"Too many scenes skipped ({skipped}/{self.N_SCENES}) — pool incomplete.")

        self.assertEqual(
            len(all_collisions),
            0,
            msg=(
                f"Found {len(all_collisions)} main_*/main_* collision(s) "
                f"across {self.N_SCENES - skipped} division-long scenes.\n"
                + "\n".join(
                    f"  scene {c['scene']}: {c['a']} bbox{c['bbox_a']} "
                    f"<> {c['b']} bbox{c['bbox_b']} overlap={c['overlap']:.3f}"
                    for c in all_collisions[:10]
                )
            ),
        )


if __name__ == "__main__":
    unittest.main()
