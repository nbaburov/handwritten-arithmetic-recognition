"""Tests for src.core.run_config — load_config and related dataclasses.

Each test exercises a distinct aspect of the loading logic so failures are
pinpointed to a single behaviour.  Tests write temporary TOML files to
isolated directories and never mutate the real config.toml.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.core.run_config import (
    GnnConfig,
    GenerationConfig,
    YoloConfig,
    YoloInferenceConfig,
    load_config,
)
from src.generation.layouts import SceneCase


class TestLoadConfigDefaults(unittest.TestCase):
    """load_config falls back to all defaults when config.toml is absent."""

    def test_no_config_toml_returns_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen, yolo, gnn = load_config(root)
            self.assertIsInstance(gen, GenerationConfig)
            self.assertIsInstance(yolo, YoloConfig)
            self.assertIsInstance(gnn, GnnConfig)
            # Spot-check key defaults
            self.assertEqual(gen.images_per_case, 1000)
            self.assertEqual(yolo.device, "cpu")
            self.assertEqual(gnn.epochs, 60)

    def test_empty_config_toml_returns_defaults(self) -> None:
        """An empty TOML file contains no keys and must not raise."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.toml").write_text("", encoding="utf-8")
            gen, yolo, gnn = load_config(root)
            self.assertEqual(gen.images_per_case, 1000)
            self.assertEqual(yolo.epochs, 50)
            self.assertEqual(gnn.lr, 0.001)


class TestLoadConfigPartial(unittest.TestCase):
    """Only the [yolo] section is present; generation and gnn must use defaults."""

    def test_partial_config_overrides_yolo_only(self) -> None:
        toml_text = """\
[yolo]
epochs = 99
batch = 8
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.toml").write_text(toml_text, encoding="utf-8")
            gen, yolo, gnn = load_config(root)
            # YOLO overridden
            self.assertEqual(yolo.epochs, 99)
            self.assertEqual(yolo.batch, 8)
            # Generation and GNN keep defaults
            self.assertEqual(gen.images_per_case, 1000)
            self.assertEqual(gnn.batch_size, 32)


class TestLoadConfigFull(unittest.TestCase):
    """A fully-populated config.toml populates all three dataclasses correctly."""

    _TOML = """\
[generation]
images_per_case = 500
min_instances_per_label = 200
equations_per_image = 2
topup_rounds = 4
seed = 7

[generation.preset]
glyph_rotation_min_deg = -6.0
glyph_rotation_max_deg = 6.0
glyph_jitter_min_px = 0
glyph_jitter_max_px = 6
glyph_broken_stroke_prob = 0.05

[generation.scene]
crowdness_prob = 0.30
scene_rotation_min_deg = -2.0
scene_rotation_max_deg = 2.0

[generation.case_weights]
addition = 2.0
subtraction = 1.5
multiplication_simple = 1.0
multiplication_multi = 0.5
division_short = 1.0
division_long = 1.0
division_simple = 1.0

[yolo]
model = "yolov8n.pt"
epochs = 25
batch = 32
image_size = 640
patience = 10
device = "mps"
workers = 4
cache = "ram"
rect = false

[yolo.augmentation]
degrees = 10.0
fliplr = 0.5

[gnn]
epochs = 30
batch_size = 64
lr = 0.01
dropout = 0.2
weight_decay = 1e-4
early_stop_patience = 5
lr_patience = 3
lr_factor = 0.3
min_lr = 1e-7
temperature = 0.05

[gnn.loss_weights]
fine_label = 3.0
edge_type = 0.1
"""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._root = Path(self._tmpdir.name)
        (self._root / "config.toml").write_text(self._TOML, encoding="utf-8")
        self._gen, self._yolo, self._gnn = load_config(self._root)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_generation_fields(self) -> None:
        self.assertEqual(self._gen.images_per_case, 500)
        self.assertEqual(self._gen.seed, 7)
        self.assertEqual(self._gen.topup_rounds, 4)
        self.assertAlmostEqual(self._gen.preset.glyph_rotation_min_deg, -6.0)
        self.assertAlmostEqual(self._gen.preset.glyph_rotation_max_deg, 6.0)
        self.assertEqual(self._gen.preset.glyph_jitter_max_px, 6)
        self.assertAlmostEqual(self._gen.preset.glyph_broken_stroke_prob, 0.05)
        self.assertAlmostEqual(self._gen.scene.crowdness_prob, 0.30)
        self.assertAlmostEqual(self._gen.scene.scene_rotation_min_deg, -2.0)
        self.assertAlmostEqual(self._gen.scene.scene_rotation_max_deg, 2.0)

    def test_case_weight_addition_overridden(self) -> None:
        self.assertAlmostEqual(self._gen.case_weights["addition"], 2.0)

    def test_case_weight_underscore_keys_normalised(self) -> None:
        # multiplication_simple in TOML must become multiplication-simple (SceneCase value)
        self.assertIn("multiplication-simple", self._gen.case_weights)
        self.assertNotIn("multiplication_simple", self._gen.case_weights)

    def test_yolo_fields(self) -> None:
        self.assertEqual(self._yolo.epochs, 25)
        self.assertEqual(self._yolo.batch, 32)
        self.assertFalse(self._yolo.rect)
        self.assertEqual(self._yolo.device, "mps")

    def test_yolo_augmentation_override_and_default_merge(self) -> None:
        # degrees overridden
        self.assertAlmostEqual(self._yolo.augmentation["degrees"], 10.0)
        # fliplr overridden
        self.assertAlmostEqual(self._yolo.augmentation["fliplr"], 0.5)
        # translate kept from default
        self.assertAlmostEqual(self._yolo.augmentation["translate"], 0.05)

    def test_gnn_fields(self) -> None:
        self.assertEqual(self._gnn.epochs, 30)
        self.assertAlmostEqual(self._gnn.lr, 0.01)
        self.assertAlmostEqual(self._gnn.dropout, 0.2)

    def test_gnn_loss_weights_override_and_default_merge(self) -> None:
        self.assertAlmostEqual(self._gnn.loss_weights["fine_label"], 3.0)
        self.assertAlmostEqual(self._gnn.loss_weights["edge_type"], 0.1)
        # row_contrastive kept from default
        self.assertAlmostEqual(self._gnn.loss_weights["row_contrastive"], 1.0)


class TestLoadConfigValidation(unittest.TestCase):
    """Invalid values in config.toml must raise ValueError with a clear message."""

    def _write_and_load(self, toml_text: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.toml").write_text(toml_text, encoding="utf-8")
            load_config(root)

    def test_unknown_case_weight_key_raises(self) -> None:
        toml_text = '[generation.case_weights]\nnonexistent_case = 1.0\n'
        with self.assertRaises(ValueError) as ctx:
            self._write_and_load(toml_text)
        self.assertIn("nonexistent_case", str(ctx.exception))


class TestLoadConfigFloatCoercion(unittest.TestCase):
    """Integer values in TOML for float-typed fields must be coerced to float."""

    def test_lr_integer_becomes_float(self) -> None:
        toml_text = "[gnn]\nlr = 1\n"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.toml").write_text(toml_text, encoding="utf-8")
            _, _, gnn = load_config(root)
            self.assertIsInstance(gnn.lr, float)
            self.assertAlmostEqual(gnn.lr, 1.0)


class TestLoadConfigCaseWeightsCoverage(unittest.TestCase):
    """All 7 SceneCase values must be present in case_weights after loading."""

    def test_all_scene_cases_present_with_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.toml").write_text("", encoding="utf-8")
            gen, _, _ = load_config(root)
            valid_values = {c.value for c in SceneCase}
            self.assertEqual(set(gen.case_weights.keys()), valid_values)

    def test_all_scene_cases_present_after_partial_override(self) -> None:
        toml_text = "[generation.case_weights]\naddition = 2.0\n"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.toml").write_text(toml_text, encoding="utf-8")
            gen, _, _ = load_config(root)
            valid_values = {c.value for c in SceneCase}
            self.assertEqual(set(gen.case_weights.keys()), valid_values)
            # addition overridden, others kept at 1.0
            self.assertAlmostEqual(gen.case_weights["addition"], 2.0)
            self.assertAlmostEqual(gen.case_weights["subtraction"], 1.0)


class TestYoloInferenceConfig(unittest.TestCase):
    """Tests for YoloInferenceConfig dataclass and its integration with YoloConfig."""

    def test_yolo_inference_defaults(self) -> None:
        """YoloInferenceConfig() has iou=0.3, agnostic_nms=False, max_det=80."""
        cfg = YoloInferenceConfig()
        self.assertAlmostEqual(cfg.iou, 0.3)
        self.assertFalse(cfg.agnostic_nms)
        self.assertEqual(cfg.max_det, 80)

    def test_yolo_config_has_inference_field(self) -> None:
        """YoloConfig().inference is YoloInferenceConfig with defaults."""
        cfg = YoloConfig()
        self.assertIsInstance(cfg.inference, YoloInferenceConfig)
        self.assertAlmostEqual(cfg.inference.iou, 0.3)
        self.assertFalse(cfg.inference.agnostic_nms)
        self.assertEqual(cfg.inference.max_det, 80)

    def test_load_config_parses_yolo_inference_block(self) -> None:
        """Given a toml with full [yolo.inference] block, parsed YoloConfig.inference reflects those values."""
        toml_text = """\
[yolo.inference]
iou = 0.4
agnostic_nms = false
max_det = 100
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.toml").write_text(toml_text, encoding="utf-8")
            _, yolo, _ = load_config(root)
            self.assertAlmostEqual(yolo.inference.iou, 0.4)
            self.assertFalse(yolo.inference.agnostic_nms)
            self.assertEqual(yolo.inference.max_det, 100)

    def test_load_config_yolo_inference_missing_block(self) -> None:
        """Toml without [yolo.inference] block uses YoloInferenceConfig() defaults."""
        toml_text = """\
[yolo]
epochs = 25
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.toml").write_text(toml_text, encoding="utf-8")
            _, yolo, _ = load_config(root)
            self.assertAlmostEqual(yolo.inference.iou, 0.3)
            self.assertFalse(yolo.inference.agnostic_nms)
            self.assertEqual(yolo.inference.max_det, 80)

    def test_load_config_yolo_inference_partial_block(self) -> None:
        """Toml with only iou set merges with defaults (other fields are defaults)."""
        toml_text = """\
[yolo.inference]
iou = 0.5
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.toml").write_text(toml_text, encoding="utf-8")
            _, yolo, _ = load_config(root)
            self.assertAlmostEqual(yolo.inference.iou, 0.5)
            self.assertFalse(yolo.inference.agnostic_nms)  # default
            self.assertEqual(yolo.inference.max_det, 80)  # default

    def test_yolo_inference_config_is_frozen(self) -> None:
        """Attempt to mutate YoloInferenceConfig raises FrozenInstanceError."""
        cfg = YoloInferenceConfig()
        with self.assertRaises(AttributeError):
            cfg.iou = 0.5  # type: ignore


if __name__ == "__main__":
    unittest.main()
