"""Tests for the set-maker equation-target generator wrapper (``targets.py``).

``generate_target`` is pool-free: it builds target geometry from the layout grid
(``build_target_scene``), not from a rendered glyph, so a fresh clone with no
``data/raw`` pool and no ``symbol_assets_manifest.csv`` can still produce targets.

Groups:

- Config-cache and clean-config logic tests (always on; no pool, no manifest).
- ``generate_target`` contract tests (always on; pool absent): every target is a
  schema-valid, deterministic ``TargetScene`` with all seven ``TargetSymbol``
  fields, bboxes within 512, a single equation per scene, and the pinned
  completion stage.
- A trip-wire test that runs ``generate_target`` with every symbol-pool /
  manifest loader patched to raise, proving no pool access happens.
- ``load_pool_once`` tests remain gated on the real manifest, since that helper
  (kept for callers that still want the rendered pool) genuinely needs it.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

import src.generation.synth_yolo as synth_yolo
from src.core.config import DataPrepConfig
from src.core.run_config import GenerationConfig
from src.generation.completion_stages import valid_stages_for_case
from src.generation.layouts_types import SceneCase
from src.setmaker import targets
from src.setmaker.targets import (
    PoolTriple,
    generate_target,
    load_pool_once,
)
from src.setmaker.types import TargetScene, TargetSymbol

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SYMBOL_MANIFEST = (
    _PROJECT_ROOT / "data" / "generated" / "processed" / "yolo" / "symbol_assets_manifest.csv"
)
_POOL_PRESENT = _SYMBOL_MANIFEST.exists()

_SYMBOL_FIELDS = (
    "fine_label",
    "glyph_key",
    "yolo_class",
    "row_index",
    "col_index",
    "equation_idx",
    "bbox",
)


class TestConfigCache(unittest.TestCase):
    """The resolved-config cache is reused per root and clears on reset."""

    def setUp(self) -> None:
        targets._reset_target_cache()

    def tearDown(self) -> None:
        targets._reset_target_cache()

    def test_resolve_configs_returns_cached_identity(self) -> None:
        first = targets._resolve_configs(None)
        second = targets._resolve_configs(None)
        # Same tuple and same contained config objects -> config.toml read once.
        self.assertIs(first, second)
        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])

    def test_resolve_configs_types(self) -> None:
        config, generation_config = targets._resolve_configs(_PROJECT_ROOT)
        self.assertIsInstance(config, DataPrepConfig)
        self.assertIsInstance(generation_config, GenerationConfig)

    def test_reset_clears_cache(self) -> None:
        before = targets._resolve_configs(None)
        targets._reset_target_cache()
        after = targets._resolve_configs(None)
        self.assertIsNot(before, after)

    def test_distinct_roots_get_distinct_entries(self) -> None:
        # A second, different root must not reuse the first root's cached pair.
        with tempfile.TemporaryDirectory() as d:
            other = targets._resolve_configs(Path(d))
        default = targets._resolve_configs(None)
        self.assertIsNot(other, default)
        self.assertNotEqual(other[0].project_root, default[0].project_root)


class TestLoadPoolBadManifest(unittest.TestCase):
    """``load_pool_once`` surfaces a clear ValueError on a malformed manifest."""

    def setUp(self) -> None:
        # The generation-layer pool cache is keyed by (manifest_path, source_pool);
        # clear it so a temp manifest forces a fresh load rather than a cache hit.
        synth_yolo._SINGLE_SCENE_POOL_CACHE.clear()
        targets._reset_target_cache()

    def tearDown(self) -> None:
        synth_yolo._SINGLE_SCENE_POOL_CACHE.clear()
        targets._reset_target_cache()

    def _write_manifest(self, root: Path, frame: pd.DataFrame) -> DataPrepConfig:
        yolo_dir = root / "data" / "generated" / "processed" / "yolo"
        yolo_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(yolo_dir / "symbol_assets_manifest.csv", index=False)
        return DataPrepConfig.from_project_root(root)

    def test_missing_glyph_key_column_raises_with_path(self) -> None:
        # Has the base columns _load_symbol_rows requires, but no glyph_key.
        frame = pd.DataFrame(
            {"path": ["x.png"], "label": ["main_0"], "split": ["train"]}
        )
        with tempfile.TemporaryDirectory() as d:
            config = self._write_manifest(Path(d), frame)
            with self.assertRaises(ValueError) as ctx:
                load_pool_once(config=config, generation_config=GenerationConfig())
        self.assertIn("symbol_assets_manifest.csv", str(ctx.exception))

    def test_empty_manifest_raises(self) -> None:
        # glyph_key column present but no usable rows.
        frame = pd.DataFrame(
            {"path": [], "label": [], "split": [], "glyph_key": []}
        )
        with tempfile.TemporaryDirectory() as d:
            config = self._write_manifest(Path(d), frame)
            with self.assertRaises(ValueError):
                load_pool_once(config=config, generation_config=GenerationConfig())


@unittest.skipUnless(
    _POOL_PRESENT,
    "symbol_assets_manifest.csv not present -- run `validate` to build the symbol pool",
)
class TestLoadPoolOnce(unittest.TestCase):
    """``load_pool_once`` warms and returns the shared symbol-pool triple."""

    def setUp(self) -> None:
        targets._reset_target_cache()

    def tearDown(self) -> None:
        targets._reset_target_cache()

    def test_returns_pool_triple(self) -> None:
        pool = load_pool_once(_PROJECT_ROOT)
        self.assertIsInstance(pool, tuple)
        self.assertEqual(len(pool), 3)
        symbols_by_glyph_key, fallback_by_glyph, _pool_root = pool
        self.assertIsInstance(symbols_by_glyph_key, dict)
        self.assertIsInstance(fallback_by_glyph, dict)
        self.assertGreater(len(symbols_by_glyph_key), 0)
        # Every value is a DataFrame of crop rows for that glyph key.
        for frame in symbols_by_glyph_key.values():
            self.assertIsInstance(frame, pd.DataFrame)

    def test_cache_warms_and_reuses(self) -> None:
        first = load_pool_once(_PROJECT_ROOT)
        second = load_pool_once(_PROJECT_ROOT)
        # Generation-layer pool cache means the identical object comes back.
        self.assertIs(first, second)

    def test_alias_matches_runtime_shape(self) -> None:
        # PoolTriple is the documented return alias; a real load satisfies it.
        pool: PoolTriple = load_pool_once(_PROJECT_ROOT)
        self.assertEqual(len(pool), 3)


class TestGenerateTarget(unittest.TestCase):
    """``generate_target`` returns a schema-valid, deterministic TargetScene.

    These run with the symbol pool ABSENT: target geometry is derived from the
    layout grid (``build_target_scene``), never from a rendered glyph, so no
    ``symbol_assets_manifest.csv`` and no ``data/raw`` pool are needed.
    """

    def setUp(self) -> None:
        targets._reset_target_cache()

    def tearDown(self) -> None:
        targets._reset_target_cache()

    def _scene(self, case: str = "addition", seed: int = 123, stage: str = "full") -> TargetScene:
        return generate_target(case, seed=seed, completion_stage=stage, project_root=_PROJECT_ROOT)

    def test_returns_target_scene(self) -> None:
        scene = self._scene()
        self.assertIsInstance(scene, TargetScene)
        self.assertIsInstance(scene.symbols, list)
        self.assertGreater(len(scene.symbols), 0)

    def test_every_symbol_has_all_seven_fields(self) -> None:
        scene = self._scene()
        for sym in scene.symbols:
            self.assertIsInstance(sym, TargetSymbol)
            for field_name in _SYMBOL_FIELDS:
                self.assertTrue(
                    hasattr(sym, field_name),
                    f"symbol missing field {field_name!r}",
                )

    def test_symbol_field_types(self) -> None:
        scene = self._scene()
        for sym in scene.symbols:
            self.assertIsInstance(sym.fine_label, str)
            self.assertIsInstance(sym.glyph_key, str)
            self.assertIsInstance(sym.yolo_class, str)
            self.assertIsInstance(sym.row_index, int)
            self.assertIsInstance(sym.col_index, int)
            self.assertIsInstance(sym.equation_idx, int)
            self.assertIsInstance(sym.bbox, tuple)
            self.assertEqual(len(sym.bbox), 4)
            for coord in sym.bbox:
                self.assertIsInstance(coord, float)

    def test_bbox_within_canvas(self) -> None:
        scene = self._scene()
        for sym in scene.symbols:
            x0, y0, x1, y1 = sym.bbox
            for coord in (x0, y0, x1, y1):
                self.assertGreaterEqual(coord, 0.0)
                self.assertLessEqual(coord, 512.0)
            # A box is non-degenerate: x1 > x0 and y1 > y0.
            self.assertGreater(x1, x0)
            self.assertGreater(y1, y0)

    def test_single_equation_all_equation_idx_zero(self) -> None:
        scene = self._scene()
        for sym in scene.symbols:
            self.assertEqual(sym.equation_idx, 0)

    def test_deterministic_for_fixed_seed(self) -> None:
        a = self._scene(seed=2026)
        b = self._scene(seed=2026)
        self.assertEqual(
            [s.fine_label for s in a.symbols],
            [s.fine_label for s in b.symbols],
        )
        self.assertEqual([s.bbox for s in a.symbols], [s.bbox for s in b.symbols])
        self.assertEqual([s.row_index for s in a.symbols], [s.row_index for s in b.symbols])
        self.assertEqual([s.col_index for s in a.symbols], [s.col_index for s in b.symbols])
        self.assertEqual(a.reference, b.reference)

    def test_different_seeds_differ(self) -> None:
        a = self._scene(seed=1)
        b = self._scene(seed=999)
        # Operand sampling gives natural variety across seeds (guards against a
        # silently constant scene). Compare the full label sequence.
        self.assertNotEqual(
            [s.fine_label for s in a.symbols],
            [s.fine_label for s in b.symbols],
        )

    def test_case_and_seed_roundtrip_onto_scene(self) -> None:
        scene = self._scene(case="addition", seed=77)
        self.assertEqual(scene.case, "addition")
        self.assertEqual(scene.seed, 77)

    def test_full_stage_pinned(self) -> None:
        scene = self._scene(stage="full")
        self.assertEqual(scene.completion_stage, "full")

    def test_non_full_stage_pinned(self) -> None:
        # Pick a valid non-full stage for addition and assert it is honored,
        # never randomly resampled to another bucket.
        stages = [s for s in valid_stages_for_case(SceneCase.addition) if s != "full"]
        self.assertTrue(stages, "addition has no non-full completion stage to test")
        target_stage = stages[0]
        scene = self._scene(stage=target_stage)
        self.assertEqual(scene.completion_stage, target_stage)

    def test_reference_is_non_empty_typeset_string(self) -> None:
        scene = self._scene()
        self.assertIsInstance(scene.reference, str)
        # A multi-symbol equation produces a non-empty clean hint string.
        self.assertNotEqual(scene.reference.strip(), "")

    def test_enum_case_accepted_and_matches_str(self) -> None:
        from_str = self._scene(case="subtraction", seed=5)
        from_enum = generate_target(
            SceneCase.subtraction, seed=5, completion_stage="full", project_root=_PROJECT_ROOT
        )
        self.assertEqual(from_enum.case, "subtraction")
        self.assertEqual(
            [s.fine_label for s in from_str.symbols],
            [s.fine_label for s in from_enum.symbols],
        )
        self.assertEqual(
            [s.bbox for s in from_str.symbols],
            [s.bbox for s in from_enum.symbols],
        )

    def test_yolo_class_in_ontology(self) -> None:
        from src.core.ontology import YOLO_CLASS_NAMES

        scene = self._scene()
        for sym in scene.symbols:
            self.assertIn(sym.yolo_class, YOLO_CLASS_NAMES)

    def test_structure_only_case_renders_with_grid_gt(self) -> None:
        # bare_digits is a real structure-only case (not a negative to discard):
        # it must still carry row/col/fine_label GT so the matcher flow works.
        scene = generate_target(
            "bare_digits", seed=7, completion_stage="full", project_root=_PROJECT_ROOT
        )
        self.assertGreater(len(scene.symbols), 0)
        for sym in scene.symbols:
            self.assertEqual(sym.equation_idx, 0)
            self.assertIsInstance(sym.row_index, int)
            self.assertIsInstance(sym.col_index, int)
            self.assertTrue(hasattr(sym, "fine_label"))


class TestCleanTargetConfig(unittest.TestCase):
    """``_make_clean_generation_config`` zeros all error-injection knobs."""

    def setUp(self) -> None:
        targets._reset_target_cache()

    def tearDown(self) -> None:
        targets._reset_target_cache()

    def test_scene_error_probs_zeroed(self) -> None:
        from src.core.run_config import GenerationConfig
        from src.setmaker.targets import _make_clean_generation_config

        base = GenerationConfig()
        clean = _make_clean_generation_config(base)
        self.assertEqual(clean.scene.wrong_result_prob, 0.0)
        self.assertEqual(clean.scene.missing_structural_prob, 0.0)
        self.assertEqual(clean.scene.wrong_operator_prob, 0.0)

    def test_rendering_error_knobs_zeroed(self) -> None:
        from src.core.run_config import GenerationConfig
        from src.setmaker.targets import (
            _RENDERING_ERROR_KNOBS,
            _make_clean_generation_config,
        )

        base = GenerationConfig()
        clean = _make_clean_generation_config(base)
        for knob in _RENDERING_ERROR_KNOBS:
            if knob in clean.rendering:
                self.assertEqual(
                    clean.rendering[knob],
                    0.0,
                    f"rendering knob {knob!r} should be 0.0 in clean config",
                )

    def test_style_variance_preserved(self) -> None:
        # Glyph rotation and jitter are NOT zeroed — they are legitimate
        # handwriting variance, not error injection.
        from src.core.run_config import GenerationConfig
        from src.setmaker.targets import _make_clean_generation_config

        base = GenerationConfig()
        clean = _make_clean_generation_config(base)
        # Preset is untouched (rotation/jitter survive).
        self.assertEqual(clean.preset, base.preset)
        # Scene crowdness / rotation are untouched.
        self.assertEqual(clean.scene.crowdness_prob, base.scene.crowdness_prob)
        self.assertEqual(clean.scene.crowding_factor, base.scene.crowding_factor)

    def test_clean_flag_default_true(self) -> None:
        # The generate_target signature must accept clean as a keyword arg
        # without a pool — just verify the parameter is accepted.
        import inspect
        sig = inspect.signature(targets.generate_target)
        self.assertIn("clean", sig.parameters)
        self.assertTrue(sig.parameters["clean"].default is True)

    def test_clean_param_default_is_true(self) -> None:
        # The default must be True so existing callers get clean targets without
        # updating their call sites.
        import inspect
        sig = inspect.signature(targets.generate_target)
        param = sig.parameters["clean"]
        self.assertTrue(param.default, "clean parameter default must be True")

    def test_base_config_untouched_by_clean(self) -> None:
        # Cleaning a config must not mutate the original (frozen SceneConfig + new dict).
        from src.core.run_config import GenerationConfig
        from src.setmaker.targets import _make_clean_generation_config

        base = GenerationConfig()
        orig_wrong_result = base.scene.wrong_result_prob
        _make_clean_generation_config(base)
        # SceneConfig is frozen=True so it cannot have been mutated.
        self.assertEqual(base.scene.wrong_result_prob, orig_wrong_result)


class TestCleanTargetRender(unittest.TestCase):
    """``generate_target(clean=True)`` produces valid, correct-looking scenes.

    Pool-free: the clean target geometry comes from the layout grid, so these
    pass with no symbol manifest and no ``data/raw`` pool present.
    """

    def setUp(self) -> None:
        targets._reset_target_cache()

    def tearDown(self) -> None:
        targets._reset_target_cache()

    def test_clean_target_returns_target_scene(self) -> None:
        scene = generate_target(
            "addition", seed=42, completion_stage="full",
            project_root=_PROJECT_ROOT, clean=True,
        )
        from src.setmaker.types import TargetScene
        self.assertIsInstance(scene, TargetScene)
        self.assertGreater(len(scene.symbols), 0)

    def test_clean_target_has_result_bar(self) -> None:
        # A fully-complete addition scene must contain a result_bar (the clean
        # path disables missing_structural_prob so it is never dropped).
        scene = generate_target(
            "addition", seed=1001, completion_stage="full",
            project_root=_PROJECT_ROOT, clean=True,
        )
        labels = {s.fine_label for s in scene.symbols}
        self.assertIn("result_bar", labels, "result_bar was dropped in a clean target")

    def test_clean_and_default_are_same(self) -> None:
        # Calling generate_target without clean= must give the same result as
        # explicitly passing clean=True.
        a = generate_target(
            "subtraction", seed=77, completion_stage="full",
            project_root=_PROJECT_ROOT,
        )
        b = generate_target(
            "subtraction", seed=77, completion_stage="full",
            project_root=_PROJECT_ROOT, clean=True,
        )
        self.assertEqual(
            [s.fine_label for s in a.symbols],
            [s.fine_label for s in b.symbols],
        )


class TestGenerateTargetPoolFree(unittest.TestCase):
    """``generate_target`` produces valid multi-symbol scenes without the pool.

    The headline guarantee of the decoupling: target generation reads neither the
    symbol manifest nor the ``data/raw`` crop pool. Every pool/manifest loader in
    the generation layer is patched to raise, so any accidental pool access fails
    the test loudly rather than silently passing on a dev machine that happens to
    have the pool on disk.
    """

    # SceneCases that must yield a multi-symbol grid the annotator can redraw.
    _MULTI_SYMBOL_CASES = ("addition", "subtraction", "division-long", "bare_digit_grid")

    def setUp(self) -> None:
        targets._reset_target_cache()
        synth_yolo._SINGLE_SCENE_POOL_CACHE.clear()
        # Trip-wire: any symbol-pool / manifest read raises. Patched on both the
        # generation-layer module and the synth_pool helpers it would call.
        import src.generation.synth_pool as synth_pool

        self._synth_pool = synth_pool
        self._saved = {
            (synth_yolo, "_load_single_scene_pool"): synth_yolo._load_single_scene_pool,
            (synth_pool, "_load_symbol_rows"): synth_pool._load_symbol_rows,
            (synth_pool, "resolve_pool_root"): synth_pool.resolve_pool_root,
            (synth_pool, "_sample_symbol"): synth_pool._sample_symbol,
            (synth_pool, "_tile_from_path"): synth_pool._tile_from_path,
        }

        def _boom(*_a, **_k):  # pragma: no cover - only fires on regression
            raise AssertionError("generate_target touched the symbol pool / manifest")

        for (mod, name) in self._saved:
            setattr(mod, name, _boom)

    def tearDown(self) -> None:
        for (mod, name), original in self._saved.items():
            setattr(mod, name, original)
        synth_yolo._SINGLE_SCENE_POOL_CACHE.clear()
        targets._reset_target_cache()

    def test_multi_symbol_cases_build_without_pool(self) -> None:
        for case in self._MULTI_SYMBOL_CASES:
            with self.subTest(case=case):
                scene = generate_target(
                    case, seed=123, completion_stage="full", project_root=_PROJECT_ROOT
                )
                self.assertIsInstance(scene, TargetScene)
                self.assertEqual(scene.case, case)
                # Multi-symbol: at least two symbols so row/col structure exists.
                self.assertGreaterEqual(
                    len(scene.symbols), 2, f"{case} produced too few symbols"
                )

    def test_bboxes_within_canvas_and_nondegenerate(self) -> None:
        for case in self._MULTI_SYMBOL_CASES:
            with self.subTest(case=case):
                scene = generate_target(
                    case, seed=321, completion_stage="full", project_root=_PROJECT_ROOT
                )
                for sym in scene.symbols:
                    x0, y0, x1, y1 = sym.bbox
                    for coord in (x0, y0, x1, y1):
                        self.assertIsInstance(coord, float)
                        self.assertGreaterEqual(coord, 0.0)
                        self.assertLessEqual(coord, 512.0)
                    self.assertGreater(x1, x0)
                    self.assertGreater(y1, y0)

    def test_row_col_are_ints_and_equation_idx_zero(self) -> None:
        for case in self._MULTI_SYMBOL_CASES:
            with self.subTest(case=case):
                scene = generate_target(
                    case, seed=55, completion_stage="full", project_root=_PROJECT_ROOT
                )
                rows = {s.row_index for s in scene.symbols}
                cols = {s.col_index for s in scene.symbols}
                for sym in scene.symbols:
                    self.assertIsInstance(sym.row_index, int)
                    self.assertIsInstance(sym.col_index, int)
                    self.assertEqual(sym.equation_idx, 0)
                    self.assertIsInstance(sym.fine_label, str)
                # A real grid spans more than one cell on at least one axis.
                self.assertTrue(
                    len(rows) > 1 or len(cols) > 1,
                    f"{case} collapsed to a single grid cell",
                )

    def test_all_seven_fields_present_without_pool(self) -> None:
        scene = generate_target(
            "addition", seed=9, completion_stage="full", project_root=_PROJECT_ROOT
        )
        self.assertGreater(len(scene.symbols), 0)
        for sym in scene.symbols:
            self.assertIsInstance(sym, TargetSymbol)
            for field_name in _SYMBOL_FIELDS:
                self.assertTrue(hasattr(sym, field_name), f"missing {field_name!r}")
            self.assertIsInstance(sym.bbox, tuple)
            self.assertEqual(len(sym.bbox), 4)

    def test_deterministic_without_pool(self) -> None:
        a = generate_target(
            "division-long", seed=2026, completion_stage="full", project_root=_PROJECT_ROOT
        )
        b = generate_target(
            "division-long", seed=2026, completion_stage="full", project_root=_PROJECT_ROOT
        )
        self.assertEqual([s.bbox for s in a.symbols], [s.bbox for s in b.symbols])
        self.assertEqual(
            [s.fine_label for s in a.symbols], [s.fine_label for s in b.symbols]
        )
        self.assertEqual(a.reference, b.reference)


class TestBuildTargetSceneNoConfig(unittest.TestCase):
    """``build_target_scene`` works with default config and no project root.

    This is the fresh-clone path: ``GenerationConfig()`` defaults, no
    ``config.toml`` read, no ``DataPrepConfig``, no pool. Imported directly from
    the generation layer so the contract is exercised at its source.
    """

    def test_returns_valid_scene_from_defaults(self) -> None:
        from src.generation.synth_yolo import build_target_scene

        for case in ("addition", "subtraction", "division-long", "bare_digit_grid"):
            with self.subTest(case=case):
                scene = build_target_scene(case, seed=1, completion_stage="full")
                self.assertIsInstance(scene, TargetScene)
                self.assertGreaterEqual(len(scene.symbols), 2)
                for sym in scene.symbols:
                    x0, y0, x1, y1 = sym.bbox
                    self.assertTrue(0.0 <= x0 <= 512.0 and 0.0 <= x1 <= 512.0)
                    self.assertTrue(0.0 <= y0 <= 512.0 and 0.0 <= y1 <= 512.0)
                    self.assertGreater(x1, x0)
                    self.assertGreater(y1, y0)
                    self.assertEqual(sym.equation_idx, 0)


if __name__ == "__main__":
    unittest.main()
