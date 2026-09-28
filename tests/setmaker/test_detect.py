"""Tests for the set-maker detection bridge (WS-B, ``src/setmaker/detect.py``).

The bridge is a thin pass-through to the canonical ``src.inference.run.detect_boxes``
(single source of truth for detection). These tests pin the contract the
set-maker relies on:

* a 512x512 grayscale array yields a ``List[Detection]`` (the WS-B done-definition);
* zero detections come back as ``[]`` (never ``None``, never an exception);
* the no-weights connected-components fallback is reachable through the bridge;
* the ``[yolo.inference]`` NMS config (iou / agnostic_nms / max_det) and the
  ``conf`` / ``tta`` arguments are threaded to the underlying YOLO predict call;
* a non-(512, 512) array raises ``ValueError`` (propagated from the detector).

Most tests control the detector by injecting a stub model + a known
``YoloInferenceConfig`` into the inference module's documented project-root cache
(``_DETECT_BRIDGE_CACHE``), so they exercise the real bridge code path without
loading any weights. ``setUp``/``tearDown`` snapshot and restore that cache so
tests never pollute one another or the live cache.
"""

from __future__ import annotations

import types
import unittest
from pathlib import Path

import numpy as np

from src.core.run_config import YoloInferenceConfig
from src.inference import run as inference_run
from src.parsing.detection import Detection
from src.setmaker.detect import DEFAULT_CONF, EXPECTED_SHAPE, detect_boxes


# A project-root key used only as a cache handle in the stub-injection tests; it
# is never touched on disk because the stub short-circuits weight resolution.
_FAKE_ROOT = Path("/tmp/setmaker-detect-test-root")


# ── Stub YOLO model (mirrors tests/inference/test_run.py conventions) ──────────

class _TensorLike:
    """Minimal ndarray wrapper exposing ``.cpu().numpy()`` for ``_run_yolo``."""

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def cpu(self) -> "_TensorLike":
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


class _EmptyBoxes:
    """Boxes object that looks empty to ``_run_yolo``'s ``len()`` guard."""

    def __len__(self) -> int:
        return 0

    xyxy = _TensorLike(np.zeros((0, 4), dtype=np.float32))
    cls = _TensorLike(np.zeros((0,), dtype=np.float32))
    conf = _TensorLike(np.zeros((0,), dtype=np.float32))


class _OneBox:
    """Boxes object with a single detection (cls id 0 -> ``digit_main``)."""

    def __len__(self) -> int:
        return 1

    xyxy = _TensorLike(np.array([[10.0, 12.0, 40.0, 44.0]], dtype=np.float32))
    cls = _TensorLike(np.array([0.0], dtype=np.float32))
    conf = _TensorLike(np.array([0.83], dtype=np.float32))


class _StubYoloModel:
    """Records the kwargs of the most recent ``predict`` call.

    ``boxes_factory`` builds the boxes object returned per call, so a single stub
    class serves both the empty-result and one-detection scenarios.
    """

    def __init__(self, boxes_factory=_EmptyBoxes) -> None:
        self._boxes_factory = boxes_factory
        self.last_predict_kwargs: dict = {}
        self.predict_calls: int = 0

    def predict(self, image: np.ndarray, **kwargs: object) -> list:  # type: ignore[override]
        self.predict_calls += 1
        self.last_predict_kwargs = dict(kwargs)
        return [types.SimpleNamespace(boxes=self._boxes_factory())]


def _make_gray(value: int = 200) -> np.ndarray:
    """A valid (512, 512) uint8 grayscale canvas."""
    return np.full((512, 512), value, dtype=np.uint8)


class _CacheInjectionTestCase(unittest.TestCase):
    """Base case that snapshots and restores the inference detect-bridge cache."""

    def setUp(self) -> None:
        self._cache_backup = dict(inference_run._DETECT_BRIDGE_CACHE)

    def tearDown(self) -> None:
        inference_run._DETECT_BRIDGE_CACHE.clear()
        inference_run._DETECT_BRIDGE_CACHE.update(self._cache_backup)

    def _inject(self, model, cfg: YoloInferenceConfig) -> None:
        """Pre-seed the bridge cache so detect_boxes uses ``model``/``cfg``."""
        inference_run._DETECT_BRIDGE_CACHE[str(_FAKE_ROOT)] = (model, cfg)


# ── Module surface / contract ─────────────────────────────────────────────────

class TestModuleSurface(unittest.TestCase):
    """The bridge re-exports the detection contract and sane set-maker defaults."""

    def test_detection_is_reexported(self) -> None:
        """``Detection`` re-exported from the bridge is the parsing Detection."""
        from src.setmaker.detect import Detection as BridgeDetection
        self.assertIs(BridgeDetection, Detection)

    def test_default_conf_is_low_floor(self) -> None:
        """Set-maker default conf is the 0.05 under-confidence floor."""
        self.assertAlmostEqual(DEFAULT_CONF, 0.05)

    def test_expected_shape_constant(self) -> None:
        """The documented expected canvas shape is (512, 512)."""
        self.assertEqual(EXPECTED_SHAPE, (512, 512))

    def test_default_signature_conf_and_tta(self) -> None:
        """detect_boxes defaults: conf == DEFAULT_CONF and tta is True."""
        import inspect
        sig = inspect.signature(detect_boxes)
        self.assertEqual(sig.parameters["conf"].default, DEFAULT_CONF)
        self.assertIs(sig.parameters["tta"].default, True)


# ── Shape guard ───────────────────────────────────────────────────────────────

class TestShapeGuard(unittest.TestCase):
    """A non-(512, 512) array fails loudly (propagated from the detector)."""

    def test_wrong_shape_raises_value_error(self) -> None:
        bad = np.zeros((256, 256), dtype=np.uint8)
        with self.assertRaises(ValueError):
            detect_boxes(bad, project_root=_FAKE_ROOT)

    def test_three_channel_input_raises_value_error(self) -> None:
        """An RGB-shaped array is rejected (the bridge wants grayscale 2-D)."""
        bad = np.zeros((512, 512, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            detect_boxes(bad, project_root=_FAKE_ROOT)


# ── Zero detections are graceful ──────────────────────────────────────────────

class TestZeroDetectionsGraceful(_CacheInjectionTestCase):
    """An empty YOLO result yields ``[]`` rather than ``None`` or an exception."""

    def test_empty_yolo_result_returns_empty_list(self) -> None:
        stub = _StubYoloModel(boxes_factory=_EmptyBoxes)
        self._inject(stub, YoloInferenceConfig(iou=0.3, agnostic_nms=False, max_det=80))
        out = detect_boxes(_make_gray(255), project_root=_FAKE_ROOT)
        self.assertIsInstance(out, list)
        self.assertEqual(out, [])

    def test_empty_result_is_a_real_list_not_none(self) -> None:
        stub = _StubYoloModel(boxes_factory=_EmptyBoxes)
        self._inject(stub, YoloInferenceConfig(iou=0.3, agnostic_nms=False, max_det=80))
        out = detect_boxes(_make_gray(255), project_root=_FAKE_ROOT)
        self.assertIsNotNone(out)


# ── Non-empty detection path ──────────────────────────────────────────────────

class TestDetectionResults(_CacheInjectionTestCase):
    """A non-empty YOLO result is mapped to ``Detection`` objects."""

    def test_single_detection_mapped(self) -> None:
        stub = _StubYoloModel(boxes_factory=_OneBox)
        self._inject(stub, YoloInferenceConfig(iou=0.3, agnostic_nms=False, max_det=80))
        out = detect_boxes(_make_gray(), project_root=_FAKE_ROOT)
        self.assertEqual(len(out), 1)
        det = out[0]
        self.assertIsInstance(det, Detection)
        self.assertEqual(det.label, "digit_main")
        self.assertAlmostEqual(det.confidence, 0.83, places=5)
        self.assertAlmostEqual(det.x0, 10.0)
        self.assertAlmostEqual(det.y1, 44.0)

    def test_returns_fresh_list_each_call(self) -> None:
        """Each call returns an independent list object (bridge wraps in list())."""
        stub = _StubYoloModel(boxes_factory=_OneBox)
        self._inject(stub, YoloInferenceConfig(iou=0.3, agnostic_nms=False, max_det=80))
        gray = _make_gray()
        out1 = detect_boxes(gray, project_root=_FAKE_ROOT)
        out2 = detect_boxes(gray, project_root=_FAKE_ROOT)
        self.assertIsNot(out1, out2)


# ── No-weights fallback path ──────────────────────────────────────────────────

class TestFallbackPath(_CacheInjectionTestCase):
    """With no YOLO model, the bridge reaches the connected-components fallback."""

    def test_no_model_uses_connected_components(self) -> None:
        # model=None tells the underlying detector to use the CC fallback.
        self._inject(None, YoloInferenceConfig(iou=0.3, agnostic_nms=False, max_det=80))
        # Draw a small dark square (ink < 200) so the fallback finds one blob.
        gray = _make_gray(255)
        gray[100:140, 100:140] = 0
        out = detect_boxes(gray, project_root=_FAKE_ROOT)
        self.assertIsInstance(out, list)
        self.assertGreaterEqual(len(out), 1)
        self.assertTrue(all(isinstance(d, Detection) for d in out))

    def test_no_model_blank_canvas_returns_empty(self) -> None:
        """No model + a blank canvas (no ink) yields ``[]`` gracefully."""
        self._inject(None, YoloInferenceConfig(iou=0.3, agnostic_nms=False, max_det=80))
        out = detect_boxes(_make_gray(255), project_root=_FAKE_ROOT)
        self.assertEqual(out, [])


# ── Config + argument threading ───────────────────────────────────────────────

class TestConfigThreading(_CacheInjectionTestCase):
    """``[yolo.inference]`` NMS config and conf/tta reach the YOLO predict call."""

    def test_inference_config_nms_kwargs_forwarded(self) -> None:
        stub = _StubYoloModel(boxes_factory=_EmptyBoxes)
        self._inject(stub, YoloInferenceConfig(iou=0.42, agnostic_nms=True, max_det=37))
        detect_boxes(_make_gray(), project_root=_FAKE_ROOT)
        kw = stub.last_predict_kwargs
        self.assertAlmostEqual(kw.get("iou"), 0.42)
        self.assertIs(kw.get("agnostic_nms"), True)
        self.assertEqual(kw.get("max_det"), 37)

    def test_default_conf_forwarded_to_predict(self) -> None:
        stub = _StubYoloModel(boxes_factory=_EmptyBoxes)
        self._inject(stub, YoloInferenceConfig(iou=0.3, agnostic_nms=False, max_det=80))
        detect_boxes(_make_gray(), project_root=_FAKE_ROOT)
        self.assertAlmostEqual(stub.last_predict_kwargs.get("conf"), DEFAULT_CONF)

    def test_explicit_conf_overrides_default(self) -> None:
        stub = _StubYoloModel(boxes_factory=_EmptyBoxes)
        self._inject(stub, YoloInferenceConfig(iou=0.3, agnostic_nms=False, max_det=80))
        detect_boxes(_make_gray(), project_root=_FAKE_ROOT, conf=0.5)
        self.assertAlmostEqual(stub.last_predict_kwargs.get("conf"), 0.5)

    def test_tta_true_passes_augment(self) -> None:
        stub = _StubYoloModel(boxes_factory=_EmptyBoxes)
        self._inject(stub, YoloInferenceConfig(iou=0.3, agnostic_nms=False, max_det=80))
        detect_boxes(_make_gray(), project_root=_FAKE_ROOT, tta=True)
        self.assertIs(stub.last_predict_kwargs.get("augment"), True)

    def test_tta_false_omits_augment(self) -> None:
        stub = _StubYoloModel(boxes_factory=_EmptyBoxes)
        self._inject(stub, YoloInferenceConfig(iou=0.3, agnostic_nms=False, max_det=80))
        detect_boxes(_make_gray(), project_root=_FAKE_ROOT, tta=False)
        self.assertNotIn("augment", stub.last_predict_kwargs)

    def test_default_tta_is_enabled(self) -> None:
        """The bridge default (no tta passed) enables augmentation."""
        stub = _StubYoloModel(boxes_factory=_EmptyBoxes)
        self._inject(stub, YoloInferenceConfig(iou=0.3, agnostic_nms=False, max_det=80))
        detect_boxes(_make_gray(), project_root=_FAKE_ROOT)
        self.assertIs(stub.last_predict_kwargs.get("augment"), True)


# ── Real preprocessed image (WS-B done-definition) ────────────────────────────

class TestRealPreprocessedImage(unittest.TestCase):
    """WS-B done-definition: detect_boxes returns ``List[Detection]`` on a real
    preprocessed image. Skips cleanly when weights or ultralytics are absent so
    the suite stays green on a fresh checkout; runs end-to-end when weights exist.
    """

    def test_real_image_returns_list_of_detections(self) -> None:
        from src.core.config import DataPrepConfig, resolve_project_root
        from src.core.artifact_paths import find_yolo_best_pt

        project_root = resolve_project_root(Path(inference_run.__file__))
        config = DataPrepConfig.from_project_root(project_root)
        weights = find_yolo_best_pt(config)
        if weights is None or not weights.exists():
            self.skipTest("no YOLO weights available; bridge real-image path covered by fallback tests")
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            self.skipTest("ultralytics not installed")

        sample = project_root / "data" / "eval" / "bank" / "01-addition-2digit-bar.png"
        if not sample.exists():
            self.skipTest("no real sample image available in data/eval/bank/")

        from PIL import Image
        from src.data_pipeline.preprocessing import preprocess_for_pipeline

        gray = preprocess_for_pipeline(Image.open(sample))
        self.assertEqual(gray.shape, (512, 512))
        # tta=False keeps this end-to-end check fast (single forward pass).
        out = detect_boxes(gray, project_root=project_root, tta=False)
        self.assertIsInstance(out, list)
        self.assertTrue(all(isinstance(d, Detection) for d in out))
        # A real addition scene must yield at least one detected box.
        self.assertGreater(len(out), 0)


if __name__ == "__main__":
    unittest.main()
