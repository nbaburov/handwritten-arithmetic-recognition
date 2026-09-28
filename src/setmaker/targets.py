"""Equation-target generator wrapper for the set-maker tool.

This is the set-maker-facing facade over the generation layer's single-scene
render path. It produces one :class:`~src.setmaker.types.TargetScene` per
worklist item: the concrete equation the annotator is asked to redraw, carrying
the ground-truth symbol content and grid structure used by the matcher plus a
clean typeset ``reference`` string (NOT a synthetic handwriting render, since the
human supplies their own hand).

Design (DRY / single source of truth): the layout-only target geometry, the
completion-stage pinning, and the ``TargetScene`` construction all live in
``src/generation/synth_yolo.py`` (``build_target_scene``), the public wrapper
added so the set-maker never reaches into generator privates. ``generate_target``
is pool-free: it derives every symbol's bbox from the layout grid (not from a
rendered glyph), so no symbol pool (``data/raw`` crops) and no symbol manifest
(``symbol_assets_manifest.csv``) are read, and a fresh clone with no ``data/``
directory can still produce targets. This module delegates to that wrapper and
adds only what the set-maker needs on top: a one-time cache of the resolved
``GenerationConfig`` so ``config.toml`` is read once across the many targets
generated in a session. (``load_pool_once`` / ``_load_single_scene_pool`` remain
available for any caller that still wants the rendered pool, but
``generate_target`` no longer depends on them.)

The scene always holds a single equation (so every contained ``TargetSymbol``
carries ``equation_idx == 0``), and ``completion_stage`` is pinned by the caller
rather than sampled randomly, so partial-exercise quotas stay satisfiable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from ..core.config import DataPrepConfig, resolve_project_root
from ..core.run_config import GenerationConfig, SceneConfig, load_config
from ..generation.layouts_types import SceneCase
from ..generation.synth_yolo import _load_single_scene_pool, build_target_scene
from .types import TargetScene

# All error-injection / drop / corruption knobs in the rendering dict.  Every
# one of these is set to 0.0 in the clean-target path so the annotator always
# sees a *correct*, *complete* equation to copy.
_RENDERING_ERROR_KNOBS: tuple[str, ...] = (
    "wrong_carry_col_prob",
    "missing_carry_prob",
    "wrong_carry_value_prob",
    "wrong_borrow_value_prob",
    "missing_borrow_prob",
    "pp_wrong_operator_prob",
    "missing_pp_prob",
    "bar_gap_prob",
    "bar_micro_break_prob",
    "bracket_gap_prob",
    "bracket_micro_break_prob",
)

# Resolved-config cache: built once on the first ``load_pool_once`` /
# ``generate_target`` call and reused for every later target in the session so
# ``config.toml`` is parsed only once. Keyed by the resolved project-root string
# so a caller pointing at a different root (e.g. a test fixture) gets its own
# entry rather than silently reusing another root's config.
_CONFIG_CACHE: Dict[str, Tuple[DataPrepConfig, GenerationConfig]] = {}

# Type alias for the cached symbol pool returned by the generation layer:
# (symbols-by-glyph-key, glyph fallback table, resolved on-disk pool root).
PoolTriple = Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Optional[Path]]


def _resolve_configs(project_root: Optional[Path]) -> Tuple[DataPrepConfig, GenerationConfig]:
    """Resolve and cache ``(DataPrepConfig, GenerationConfig)`` for a project root.

    The project root is resolved from this file when not supplied. The pair is
    cached per resolved root so ``config.toml`` is read once per root across an
    interactive session.
    """
    root = (
        project_root.resolve()
        if project_root is not None
        else resolve_project_root(Path(__file__))
    )
    key = str(root)
    cached = _CONFIG_CACHE.get(key)
    if cached is not None:
        return cached

    config = DataPrepConfig.from_project_root(root)
    generation_config, _yolo_cfg, _gnn_cfg = load_config(root)
    pair = (config, generation_config)
    _CONFIG_CACHE[key] = pair
    return pair


def load_pool_once(
    project_root: "Path | None" = None,
    *,
    config: "DataPrepConfig | None" = None,
    generation_config: "GenerationConfig | None" = None,
) -> PoolTriple:
    """Load (and cache) the symbol pool once for single-scene target rendering.

    Warms the module-level symbol-pool cache in the generation layer
    (``_load_single_scene_pool``) so the symbol DataFrames are read from the
    150 MB manifest exactly once, then returns the cached pool triple
    ``(symbols_by_glyph_key, fallback_by_glyph, pool_root)``. Subsequent calls
    (and every :func:`generate_target` call) reuse the warmed cache.

    Args:
        project_root: Project root containing ``data/`` and ``config.toml``;
            resolved from this file when omitted. Ignored when ``config`` and
            ``generation_config`` are both supplied.
        config: Optional pre-built ``DataPrepConfig`` (locates the symbol
            manifest). When omitted it is resolved from ``project_root`` and
            cached.
        generation_config: Optional pre-built ``GenerationConfig`` (supplies
            ``source_pool``). When omitted it is loaded from ``config.toml`` and
            cached.

    Returns:
        The cached pool triple. ``pool_root`` is ``None`` when the on-disk pool
        could not be resolved (the renderer then degrades to the legacy pool).

    Raises:
        ValueError: When the symbol manifest is missing its ``glyph_key`` column
            or contains no usable symbol rows (re-run ``validate``). The
            file path is included in the message.
    """
    if config is None or generation_config is None:
        resolved_config, resolved_gen = _resolve_configs(project_root)
        config = config if config is not None else resolved_config
        generation_config = (
            generation_config if generation_config is not None else resolved_gen
        )
    return _load_single_scene_pool(config, generation_config.source_pool)


def _make_clean_generation_config(base: GenerationConfig) -> GenerationConfig:
    """Return a copy of *base* with all error-injection knobs set to 0.0.

    This is the config used for set-maker targets so the annotator always sees a
    *correct*, *complete* equation.  Legitimate handwriting-style variance (glyph
    rotation, jitter, stroke-width) is intentionally preserved — the goal is a
    clean equation, not a sterile one.

    The following groups of knobs are zeroed:

    - ``SceneConfig``: ``wrong_result_prob``, ``missing_structural_prob``,
      ``wrong_operator_prob`` (the three scene-level corruption flags).
    - Rendering dict: all keys listed in ``_RENDERING_ERROR_KNOBS`` (carry/borrow
      value errors, drop probabilities, bar/bracket gap/break probabilities,
      partial-product corruption knobs).

    The training generation path (``build_synthetic_yolo_dataset``) is never
    passed through this function; it always uses the real ``config.toml`` knobs.
    """
    clean_scene = SceneConfig(
        wrong_result_prob=0.0,
        missing_structural_prob=0.0,
        wrong_operator_prob=0.0,
        crowdness_prob=base.scene.crowdness_prob,
        scene_rotation_min_deg=base.scene.scene_rotation_min_deg,
        scene_rotation_max_deg=base.scene.scene_rotation_max_deg,
        crowding_factor=base.scene.crowding_factor,
    )
    clean_rendering = dict(base.rendering)
    for knob in _RENDERING_ERROR_KNOBS:
        if knob in clean_rendering:
            clean_rendering[knob] = 0.0

    # GenerationConfig is a mutable dataclass; build a fresh copy with the
    # cleaned scene and rendering blocks, keeping everything else identical.
    import dataclasses  # noqa: PLC0415 — local import to avoid circular at module level
    return dataclasses.replace(base, scene=clean_scene, rendering=clean_rendering)


def generate_target(
    case: "SceneCase | str",
    seed: int,
    completion_stage: str,
    *,
    project_root: "Path | None" = None,
    clean: bool = True,
) -> TargetScene:
    """Build one target equation for the annotator to redraw (pool-free).

    Delegates to ``build_target_scene``, the pool-free generation path: the
    target's symbol geometry is derived from the LAYOUT grid rather than from a
    rendered glyph image, so no symbol pool (``data/raw`` crops) and no symbol
    manifest (``symbol_assets_manifest.csv``) are read. A fresh clone with no
    ``data/`` directory can therefore still produce targets to draw. Only the
    ``GenerationConfig`` (preset/scene/rendering blocks from ``config.toml``, or
    its built-in defaults when the file is absent) is needed, and it is resolved
    once and cached for the session.

    The scene holds a single equation (``equation_idx == 0`` on every symbol),
    the ``completion_stage`` is pinned to the caller's value (never sampled), no
    disk writes happen, and no scene rotation is applied. The returned
    :class:`~src.setmaker.types.TargetScene` is the exact same shape the previous
    render path returned (the seven per-symbol keys plus the scene-level
    ``case``/``equation_type``/``completion_stage``/``seed``/``reference``/
    ``symbols``), so the matcher, exporters, and frontend are unchanged. ``bbox``
    is a per-cell layout rectangle in 512 px space; the matcher aligns drafts to
    the target by centroid, so tight glyph-pixel bboxes are not required.

    Args:
        case: A ``SceneCase`` enum member or its string value (e.g. "addition").
        seed: Deterministic per-scene seed; the same seed reproduces the scene.
        completion_stage: A completion-stage bucket string ("full", "done_80",
            "done_60", "done_40", "done_20"). Forced via single-bucket weights;
            an unrecognised bucket for the case falls through to "full".
        project_root: Project root containing ``config.toml``; resolved from this
            file when omitted. Used only to read ``config.toml`` (or fall back to
            defaults); the symbol pool / manifest are never read.
        clean: When ``True`` (default) all error-injection knobs are zeroed so
            the target is always a correct, complete equation.  Pass
            ``clean=False`` to keep the raw ``config.toml`` error knobs (e.g.
            for future "show a deliberately wrong target" mode).

    Returns:
        A deterministic ``TargetScene`` for the given ``(case, seed,
        completion_stage)``.
    """
    _config, generation_config = _resolve_configs(project_root)
    effective_gen_config = (
        _make_clean_generation_config(generation_config) if clean else generation_config
    )
    return build_target_scene(
        case,
        seed,
        completion_stage,
        gen_cfg=effective_gen_config,
    )


def _reset_target_cache() -> None:
    """Clear the resolved-config cache (test isolation only).

    The generation-layer symbol-pool cache is keyed by manifest path + source
    pool and is intentionally left intact; this only drops the per-root
    ``(DataPrepConfig, GenerationConfig)`` pairs cached here so a test can force a
    fresh config resolution.
    """
    _CONFIG_CACHE.clear()
