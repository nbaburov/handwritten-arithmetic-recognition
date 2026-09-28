"""Handwriting style dataclass and single-preset sampler."""
from __future__ import annotations
import random
from dataclasses import dataclass
from ..core.run_config import PresetConfig, SceneConfig

@dataclass(frozen=True)
class HandwritingStyle:
    """Immutable per-scene style; rotation is sampled per-glyph at render time."""
    rot_deg_min: float
    rot_deg_max: float
    jitter_min_px: int
    jitter_max_px: int
    glyph_broken_stroke_prob: float
    crowding_factor: float

def make_style(
    rng: random.Random,
    preset_cfg: PresetConfig,
    scene_cfg: SceneConfig | None = None,
    *,
    crowded: bool = False,
) -> HandwritingStyle:
    """Build a HandwritingStyle from the single preset config.

    Rotation bounds are stored as-is; rng is NOT consumed here for rotation.
    Each glyph samples its own rotation at render time using these bounds.
    When crowded=True the crowding_factor is read from scene_cfg.crowding_factor
    (default 0.70) so the value is driven by config.toml rather than hardcoded.
    """
    crowding = (scene_cfg.crowding_factor if scene_cfg is not None else 0.70) if crowded else 1.0
    return HandwritingStyle(
        rot_deg_min=preset_cfg.glyph_rotation_min_deg,
        rot_deg_max=preset_cfg.glyph_rotation_max_deg,
        jitter_min_px=preset_cfg.glyph_jitter_min_px,
        jitter_max_px=preset_cfg.glyph_jitter_max_px,
        glyph_broken_stroke_prob=preset_cfg.glyph_broken_stroke_prob,
        crowding_factor=crowding,
    )
