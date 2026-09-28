"""Tests for inference post-processing: cluster assignment, truncation, and assembler integration.

Covers Workstream D of the GNN Dense Attention plan.
"""

from __future__ import annotations

import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from src.modeling.gnn import _cluster_by_coordinate, CropBackbone, SymbolGNN
from src.parsing.assemble import NodePrediction, assemble_json
from src.inference.run import _run_yolo, run_inference_with_models


class TestClusterByCoordinate(unittest.TestCase):
    """Tests for the _cluster_by_coordinate helper in src.modeling.gnn."""

    def test_cluster_by_y_groups_rows(self) -> None:
        """Nodes with similar y-midpoints are assigned the same row cluster ID."""
        bboxes = [
            [10, 40, 50, 60],    # node 0 — row 0 (y-mid ~50)
            [100, 40, 140, 60],  # node 1 — row 0 (y-mid ~50)
            [10, 290, 50, 310],  # node 2 — row 1 (y-mid ~300)
            [100, 290, 140, 310], # node 3 — row 1 (y-mid ~300)
        ]
        bbox_t = torch.tensor(bboxes, dtype=torch.float32)
        cluster_ids = _cluster_by_coordinate(bbox_t, axis=1)

        self.assertEqual(len(cluster_ids), 4)
        # Nodes 0 and 1 must be in the same row cluster
        self.assertEqual(cluster_ids[0], cluster_ids[1])
        # Nodes 2 and 3 must be in the same row cluster
        self.assertEqual(cluster_ids[2], cluster_ids[3])
        # The two row clusters must be distinct
        self.assertNotEqual(cluster_ids[0], cluster_ids[2])

    def test_cluster_by_x_groups_cols(self) -> None:
        """Nodes with similar x-midpoints are assigned the same col cluster ID."""
        bboxes = [
            [10, 10, 50, 50],    # node 0 — col 0 (x-mid ~30)
            [180, 10, 220, 50],  # node 1 — col 1 (x-mid ~200)
            [10, 60, 50, 100],   # node 2 — col 0 (x-mid ~30)
            [180, 60, 220, 100], # node 3 — col 1 (x-mid ~200)
        ]
        bbox_t = torch.tensor(bboxes, dtype=torch.float32)
        cluster_ids = _cluster_by_coordinate(bbox_t, axis=0)

        self.assertEqual(len(cluster_ids), 4)
        # Nodes 0 and 2 must be in the same col cluster
        self.assertEqual(cluster_ids[0], cluster_ids[2])
        # Nodes 1 and 3 must be in the same col cluster
        self.assertEqual(cluster_ids[1], cluster_ids[3])
        # The two col clusters must be distinct
        self.assertNotEqual(cluster_ids[0], cluster_ids[1])

    def test_cluster_ids_nondecreasing_along_axis(self) -> None:
        """Cluster IDs are non-decreasing when nodes are sorted by x-midpoint."""
        # Three clearly separated x positions
        bboxes = [
            [10, 10, 50, 50],    # x-mid ~30
            [100, 10, 140, 50],  # x-mid ~120
            [200, 10, 240, 50],  # x-mid ~220
        ]
        bbox_t = torch.tensor(bboxes, dtype=torch.float32)
        cluster_ids = _cluster_by_coordinate(bbox_t, axis=0)

        # Sort cluster IDs by x-midpoint order (already sorted here)
        self.assertTrue(
            all(cluster_ids[i] <= cluster_ids[i + 1] for i in range(len(cluster_ids) - 1)),
            f"Cluster IDs not non-decreasing: {cluster_ids}",
        )

    def test_cluster_single_node(self) -> None:
        """A single-node scene returns cluster ID [0]."""
        bboxes = [[10, 10, 50, 50]]
        bbox_t = torch.tensor(bboxes, dtype=torch.float32)
        cluster_ids = _cluster_by_coordinate(bbox_t, axis=1)
        self.assertEqual(cluster_ids, [0])

    def test_cluster_empty_tensor(self) -> None:
        """An empty bbox tensor returns an empty list."""
        bbox_t = torch.zeros(0, 4, dtype=torch.float32)
        cluster_ids = _cluster_by_coordinate(bbox_t, axis=1)
        self.assertEqual(cluster_ids, [])


class TestTruncation(unittest.TestCase):
    """Tests for build_graph truncation behaviour."""

    def setUp(self) -> None:
        self.gray = np.full((512, 512), 255, dtype=np.uint8)

    def test_truncated_in_payload_when_over_max_nodes(self) -> None:
        """build_graph sets truncated=True when len(detections) > MAX_NODES (80)."""
        from src.modeling.graph_builder import build_graph
        from src.parsing.detection import Detection

        dets = [
            Detection(
                label="digit_main",
                confidence=0.9,
                x0=float(i * 5),
                y0=10.0,
                x1=float(i * 5 + 4),
                y1=14.0,
            )
            for i in range(90)
        ]
        data = build_graph(dets, self.gray)
        self.assertTrue(data.truncated)
        self.assertEqual(data.num_nodes, 80)

    def test_empty_scene_not_truncated(self) -> None:
        """A small scene (3 detections) is not truncated."""
        from src.modeling.graph_builder import build_graph
        from src.parsing.detection import Detection

        dets = [
            Detection(label="digit_main", confidence=0.9, x0=10.0, y0=10.0, x1=40.0, y1=40.0),
            Detection(label="digit_main", confidence=0.9, x0=60.0, y0=10.0, x1=90.0, y1=40.0),
            Detection(label="op_plus", confidence=0.9, x0=45.0, y0=10.0, x1=55.0, y1=40.0),
        ]
        data = build_graph(dets, self.gray)
        self.assertFalse(data.truncated)


class TestAssembleJsonClusterIds(unittest.TestCase):
    """Tests that assemble_json correctly maps NodePrediction cluster IDs to output tokens."""

    def test_assemble_json_uses_cluster_ids(self) -> None:
        """grid_col in output tokens must match col_cluster_id in NodePrediction."""
        preds = [
            NodePrediction(
                fine_label="main_3",
                row_cluster_id=0,
                col_cluster_id=0,
                within_row_ord=0,
                within_col_ord=0,
                x0=10.0,
                y0=10.0,
                x1=50.0,
                y1=50.0,
                confidence=0.9,
            ),
            NodePrediction(
                fine_label="main_7",
                row_cluster_id=0,
                col_cluster_id=1,
                within_row_ord=1,
                within_col_ord=0,
                x0=60.0,
                y0=10.0,
                x1=100.0,
                y1=50.0,
                confidence=0.9,
            ),
            NodePrediction(
                fine_label="op_plus",
                row_cluster_id=1,
                col_cluster_id=0,
                within_row_ord=0,
                within_col_ord=1,
                x0=10.0,
                y0=60.0,
                x1=50.0,
                y1=100.0,
                confidence=0.9,
            ),
        ]
        result = assemble_json(preds, "add")
        self.assertEqual(result["row_count"], 2)
        row0_tokens = result["rows"][0]["tokens"]
        row1_tokens = result["rows"][1]["tokens"]
        # First row has two tokens with grid_col 0 and 1
        self.assertEqual(row0_tokens[0]["grid_col"], 0)
        self.assertEqual(row0_tokens[1]["grid_col"], 1)
        # Second row has one token with grid_col 0
        self.assertEqual(row1_tokens[0]["grid_col"], 0)

    def test_assemble_json_within_row_ord_preserved(self) -> None:
        """within_row_ord values from NodePrediction are present in output tokens.

        An operator is included so the scene passes the structural validity gate
        (requires at least one digit and one structural token).
        """
        preds = [
            NodePrediction(
                fine_label="main_1",
                row_cluster_id=0,
                col_cluster_id=0,
                within_row_ord=0,
                within_col_ord=0,
                x0=10.0,
                y0=10.0,
                x1=40.0,
                y1=40.0,
                confidence=0.9,
            ),
            NodePrediction(
                fine_label="main_2",
                row_cluster_id=0,
                col_cluster_id=1,
                within_row_ord=1,
                within_col_ord=0,
                x0=60.0,
                y0=10.0,
                x1=90.0,
                y1=40.0,
                confidence=0.9,
            ),
            NodePrediction(
                fine_label="op_plus",
                row_cluster_id=0,
                col_cluster_id=2,
                within_row_ord=2,
                within_col_ord=0,
                x0=100.0,
                y0=10.0,
                x1=130.0,
                y1=40.0,
                confidence=0.9,
            ),
        ]
        result = assemble_json(preds, "add")
        tokens = result["rows"][0]["tokens"]
        self.assertEqual(tokens[0]["within_row_ord"], 0)
        self.assertEqual(tokens[1]["within_row_ord"], 1)

    def test_assemble_json_empty_predictions(self) -> None:
        """assemble_json with empty predictions returns an OOD sentinel dict.
        W10: reason is now true_empty (was too_few_symbols)."""
        result = assemble_json([], "add")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["equation_kind"], "unknown")
        self.assertEqual(result["ood_reason"], "true_empty")
        self.assertEqual(result["rows"], [])


class _TensorLike:
    """Minimal ndarray wrapper that exposes .cpu().numpy() for _run_yolo."""

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def cpu(self) -> "_TensorLike":
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


class _EmptyBoxes:
    """Minimal boxes object that looks empty to _run_yolo's len() guard."""

    def __len__(self) -> int:
        return 0

    xyxy = _TensorLike(np.zeros((0, 4), dtype=np.float32))
    cls = _TensorLike(np.zeros((0,), dtype=np.float32))
    conf = _TensorLike(np.zeros((0,), dtype=np.float32))


class _StubYoloModel:
    """Records the kwargs from the most recent predict() call.

    Returns a single result with zero detections so _run_yolo short-circuits
    at the ``len(r0.boxes) == 0`` guard without attempting tensor operations.
    """

    def __init__(self) -> None:
        self.last_predict_kwargs: dict = {}

    def predict(self, image: np.ndarray, **kwargs: object) -> list:  # type: ignore[override]
        self.last_predict_kwargs = dict(kwargs)
        result = types.SimpleNamespace(boxes=_EmptyBoxes())
        return [result]


class TestYoloTtaPlumbing(unittest.TestCase):
    """Verify the tta flag reaches yolo_model.predict with augment=True."""

    def _make_gray(self) -> np.ndarray:
        return np.full((512, 512), 200, dtype=np.uint8)

    def test_tta_true_passes_augment(self) -> None:
        """When tta=True, yolo_model.predict must receive augment=True."""
        stub = _StubYoloModel()
        _run_yolo(self._make_gray(), stub, conf=0.25, tta=True)
        self.assertIn("augment", stub.last_predict_kwargs)
        self.assertIs(stub.last_predict_kwargs["augment"], True)

    def test_tta_false_omits_augment_kwarg(self) -> None:
        """When tta=False, augment must NOT appear in predict kwargs at all."""
        stub = _StubYoloModel()
        _run_yolo(self._make_gray(), stub, conf=0.25, tta=False)
        self.assertNotIn("augment", stub.last_predict_kwargs)

    def test_run_inference_with_models_propagates_tta(self) -> None:
        """tta=True must reach _run_yolo (and thus yolo_model.predict) when
        called from run_inference_with_models."""

        class _StubGnn:
            def predict(self, data: object) -> tuple:  # type: ignore[override]
                return [], "unknown"

        stub_yolo = _StubYoloModel()
        stub_gnn = _StubGnn()
        run_inference_with_models(
            self._make_gray(),
            gnn_model=stub_gnn,
            yolo_model=stub_yolo,
            yolo_conf=0.25,
            tta=True,
        )
        self.assertIn("augment", stub_yolo.last_predict_kwargs)
        self.assertIs(stub_yolo.last_predict_kwargs["augment"], True)


class TestNmsPlumbing(unittest.TestCase):
    """Verify iou, agnostic_nms, and max_det are threaded through _run_yolo and InferenceSession."""

    def _make_gray(self) -> np.ndarray:
        return np.full((512, 512), 200, dtype=np.uint8)

    # ── _run_yolo-level tests ─────────────────────────────────────────────────

    def test_run_yolo_passes_iou_to_predict_kwargs(self) -> None:
        """iou kwarg is forwarded to yolo_model.predict."""
        stub = _StubYoloModel()
        _run_yolo(self._make_gray(), stub, conf=0.25, iou=0.4, agnostic_nms=True, max_det=50)
        self.assertIn("iou", stub.last_predict_kwargs)
        self.assertAlmostEqual(stub.last_predict_kwargs["iou"], 0.4)

    def test_run_yolo_passes_agnostic_nms_to_predict_kwargs(self) -> None:
        """agnostic_nms kwarg is forwarded to yolo_model.predict."""
        stub = _StubYoloModel()
        _run_yolo(self._make_gray(), stub, conf=0.25, iou=0.4, agnostic_nms=False, max_det=50)
        self.assertIn("agnostic_nms", stub.last_predict_kwargs)
        self.assertIs(stub.last_predict_kwargs["agnostic_nms"], False)

    def test_run_yolo_passes_max_det_to_predict_kwargs(self) -> None:
        """max_det kwarg is forwarded to yolo_model.predict."""
        stub = _StubYoloModel()
        _run_yolo(self._make_gray(), stub, conf=0.25, iou=0.4, agnostic_nms=True, max_det=50)
        self.assertIn("max_det", stub.last_predict_kwargs)
        self.assertEqual(stub.last_predict_kwargs["max_det"], 50)

    def test_run_yolo_default_iou_is_0_3(self) -> None:
        """_run_yolo default iou is 0.3 (matches YoloInferenceConfig default)."""
        stub = _StubYoloModel()
        _run_yolo(self._make_gray(), stub, conf=0.25)
        self.assertAlmostEqual(stub.last_predict_kwargs.get("iou"), 0.3)

    def test_run_yolo_default_agnostic_nms_is_false(self) -> None:
        """_run_yolo default agnostic_nms is False (matches YoloInferenceConfig.agnostic_nms=False; L-1 fix)."""
        stub = _StubYoloModel()
        _run_yolo(self._make_gray(), stub, conf=0.25)
        self.assertIs(stub.last_predict_kwargs.get("agnostic_nms"), False)

    def test_run_yolo_default_max_det_is_80(self) -> None:
        """_run_yolo default max_det is 80 (matches YoloInferenceConfig default)."""
        stub = _StubYoloModel()
        _run_yolo(self._make_gray(), stub, conf=0.25)
        self.assertEqual(stub.last_predict_kwargs.get("max_det"), 80)

    def test_run_yolo_with_tta_still_passes_nms_kwargs(self) -> None:
        """With tta=True, iou/agnostic_nms/max_det are still forwarded alongside augment=True."""
        stub = _StubYoloModel()
        _run_yolo(self._make_gray(), stub, conf=0.25, tta=True, iou=0.3, agnostic_nms=True, max_det=80)
        self.assertIs(stub.last_predict_kwargs.get("augment"), True)
        self.assertAlmostEqual(stub.last_predict_kwargs.get("iou"), 0.3)
        self.assertIs(stub.last_predict_kwargs.get("agnostic_nms"), True)
        self.assertEqual(stub.last_predict_kwargs.get("max_det"), 80)

    # ── InferenceSession.predict-level tests ──────────────────────────────────

    def _make_stub_session(self, iou: float = 0.5) -> tuple:
        """Return (session, stub_yolo) with a session pre-wired with a custom iou."""
        from src.core.run_config import YoloInferenceConfig
        from src.inference.run import InferenceSession

        class _StubGnn:
            def predict(self, data: object) -> tuple:  # type: ignore[override]
                return [], "unknown"

        stub_yolo = _StubYoloModel()
        session = InferenceSession()
        session._gnn_model = _StubGnn()
        session._yolo_model = stub_yolo
        session._yolo_inference_config = YoloInferenceConfig(iou=iou)
        return session, stub_yolo

    def test_inference_session_predict_uses_session_inference_config_when_none_passed(self) -> None:
        """When no iou kwarg is passed to session.predict, the session's YoloInferenceConfig.iou is used."""
        session, stub_yolo = self._make_stub_session(iou=0.5)
        session.predict(self._make_gray(), yolo_conf=0.05, tta=False)
        self.assertAlmostEqual(stub_yolo.last_predict_kwargs.get("iou"), 0.5)

    def test_inference_session_predict_call_site_iou_overrides_session_config(self) -> None:
        """An explicit iou= at the call site overrides the session's YoloInferenceConfig."""
        session, stub_yolo = self._make_stub_session(iou=0.5)
        session.predict(self._make_gray(), yolo_conf=0.05, tta=False, iou=0.2)
        self.assertAlmostEqual(stub_yolo.last_predict_kwargs.get("iou"), 0.2)

    def test_inference_session_predict_none_iou_falls_back_to_session_config(self) -> None:
        """Passing iou=None explicitly falls back to the session's YoloInferenceConfig.iou."""
        session, stub_yolo = self._make_stub_session(iou=0.5)
        session.predict(self._make_gray(), yolo_conf=0.05, tta=False, iou=None)
        self.assertAlmostEqual(stub_yolo.last_predict_kwargs.get("iou"), 0.5)


class TestDetectBridgeCache(unittest.TestCase):
    """_detect_bridge_resources must not cache the no-weights (None) case.

    Regression: caching ``(None, cfg)`` meant a long-lived server that started
    before any YOLO weights existed could never pick up a weights file added
    later -- the stale cached None shadowed it. Only a successfully loaded model
    may be cached; the None case re-checks the filesystem on every call.
    """

    def setUp(self) -> None:
        import src.inference.run as run_mod
        self._run_mod = run_mod
        # Isolate the module-level cache for this test.
        self._saved_cache = dict(run_mod._DETECT_BRIDGE_CACHE)
        run_mod._DETECT_BRIDGE_CACHE.clear()

    def tearDown(self) -> None:
        self._run_mod._DETECT_BRIDGE_CACHE.clear()
        self._run_mod._DETECT_BRIDGE_CACHE.update(self._saved_cache)

    def _patches(self, find_path_side):
        """Patch the three resolution dependencies of _detect_bridge_resources.

        ``find_path_side`` is passed straight to ``side_effect`` so each call can
        return a different path (simulating a weights file appearing later).
        Returns a context-manager-yielding tuple of the three mocks.
        """
        from src.core.run_config import YoloInferenceConfig

        yolo_cfg = types.SimpleNamespace(inference=YoloInferenceConfig())
        dp = mock.patch.object(
            self._run_mod.DataPrepConfig, "from_project_root",
            return_value=object(),
        )
        lc = mock.patch.object(
            self._run_mod, "load_config",
            return_value=(None, yolo_cfg, None),
        )
        fp = mock.patch(
            "src.core.artifact_paths.find_yolo_best_pt",
            side_effect=find_path_side,
        )
        return dp, lc, fp

    def test_none_case_not_cached_and_added_weights_picked_up(self) -> None:
        """First call (no weights) returns None and caches nothing; once a real
        weights file exists, the next call loads and caches it."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            missing = tmp / "missing.pt"
            real = tmp / "best.pt"
            real.write_bytes(b"")  # exists() must be True

            # find_yolo_best_pt returns the missing path first, then the real one.
            dp, lc, fp = self._patches([missing, real])
            yolo_loads: list[str] = []

            class _FakeYolo:
                def __init__(self, path: str) -> None:
                    yolo_loads.append(path)

            with dp, lc, fp, mock.patch("ultralytics.YOLO", _FakeYolo):
                # Call 1: no weights -> (None, cfg), nothing cached.
                model1, cfg1 = self._run_mod._detect_bridge_resources(tmp)
                self.assertIsNone(model1)
                self.assertNotIn(str(tmp), self._run_mod._DETECT_BRIDGE_CACHE)
                self.assertEqual(yolo_loads, [])

                # Call 2: weights now present -> model loaded and cached.
                model2, cfg2 = self._run_mod._detect_bridge_resources(tmp)
                self.assertIsNotNone(model2)
                self.assertIn(str(tmp), self._run_mod._DETECT_BRIDGE_CACHE)
                self.assertEqual(yolo_loads, [str(real)])

    def test_loaded_model_is_cached_no_reload(self) -> None:
        """Once a model loads, a second call returns the cached instance without
        re-invoking YOLO()."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            real = tmp / "best.pt"
            real.write_bytes(b"")

            dp, lc, fp = self._patches(lambda *_a, **_k: real)
            yolo_loads: list[str] = []

            class _FakeYolo:
                def __init__(self, path: str) -> None:
                    yolo_loads.append(path)

            with dp, lc, fp, mock.patch("ultralytics.YOLO", _FakeYolo):
                model1, _ = self._run_mod._detect_bridge_resources(tmp)
                model2, _ = self._run_mod._detect_bridge_resources(tmp)
                self.assertIs(model1, model2)  # same cached object
                self.assertEqual(yolo_loads, [str(real)])  # loaded exactly once


# ── Given-equation prior plumbing tests ──────────────────────────────────────


def _make_given_node(
    x0: int, y0: int, x1: int, y1: int,
    coarse: str = "operator", fine: str = "op_plus",
) -> "GivenNode":
    """Minimal GivenNode for prior tests (28x28 white tile)."""
    from src.inference.given_prior import GivenNode
    tile = np.full((28, 28), 255, dtype=np.uint8)
    return GivenNode(x0=x0, y0=y0, x1=x1, y1=y1,
                     coarse_label=coarse, fine_label=fine, tile=tile)


class TestPriorNoneRegression(unittest.TestCase):
    """prior=None must produce identical payload to a baseline run (regression gate)."""

    def _make_gray(self) -> np.ndarray:
        return np.full((512, 512), 200, dtype=np.uint8)

    def _make_stub_gnn(self, equation_type: str = "unknown"):
        """Return a GNN stub whose predict returns empty preds + fixed eq_type."""
        class _StubGnn:
            def predict(self, data):
                return [], equation_type, [1.0, 0.0, 0.0, 0.0]
        return _StubGnn()

    def test_prior_none_omitted_gives_same_result_as_explicit_none(self) -> None:
        """Calling run_inference_with_models without prior= and with prior=None
        must return equal payloads (equation_kind, quality, timings keys present)."""
        gray = self._make_gray()
        gnn = self._make_stub_gnn()

        result_omitted = run_inference_with_models(gray, gnn_model=gnn)
        result_none = run_inference_with_models(gray, gnn_model=gnn, prior=None)

        # Core structural fields must match.
        self.assertEqual(result_omitted["equation_kind"], result_none["equation_kind"])
        self.assertEqual(result_omitted["quality"], result_none["quality"])
        for key in ("schema_version", "rows", "tokens"):
            self.assertIn(key, result_omitted)
            self.assertIn(key, result_none)
            self.assertEqual(result_omitted[key], result_none[key])

    def test_prior_none_uses_fallback_path(self) -> None:
        """With no YOLO model and prior=None, quality.degraded must be True."""
        gray = self._make_gray()
        gnn = self._make_stub_gnn()
        result = run_inference_with_models(gray, gnn_model=gnn, prior=None)
        self.assertTrue(result["quality"]["degraded"])


class TestPriorSet(unittest.TestCase):
    """When prior is set, the given nodes are merged and pinned correctly."""

    def _make_gray(self) -> np.ndarray:
        return np.full((512, 512), 200, dtype=np.uint8)

    def test_prior_set_multiply_eq_kind_via_gnn_stub(self) -> None:
        """A sparse multiplication scene with operator+operands in the prior
        must yield equation_kind='multiply' when the GNN stub returns 'multiply'.

        The GNN stub returns NodePredictions (one digit + one structural token)
        so that assemble_json passes the structural gate and uses the GNN's
        equation_type='multiply' as equation_kind.
        """
        # Build a prior that contains operator + two operands + result_bar.
        prior = [
            _make_given_node(100, 50, 150, 100, "operator", "op_times"),
            _make_given_node(50, 50, 100, 100, "digit_main", "main_3"),
            _make_given_node(150, 50, 200, 100, "digit_main", "main_8"),
            _make_given_node(50, 110, 200, 115, "result_bar", "result_bar"),
        ]

        class _StubGnnMultiply:
            """Returns NodePredictions that pass the structural gate + 'multiply' eq type."""
            def predict(self, data):
                # Return one digit node + one operator to pass _classify_gate.
                preds = [
                    NodePrediction(
                        fine_label="main_3",
                        row_cluster_id=0, col_cluster_id=0,
                        within_row_ord=0, within_col_ord=0,
                        x0=50.0, y0=50.0, x1=100.0, y1=100.0, confidence=0.9,
                    ),
                    NodePrediction(
                        fine_label="op_times",
                        row_cluster_id=0, col_cluster_id=1,
                        within_row_ord=1, within_col_ord=0,
                        x0=100.0, y0=50.0, x1=150.0, y1=100.0, confidence=0.9,
                    ),
                    NodePrediction(
                        fine_label="main_8",
                        row_cluster_id=0, col_cluster_id=2,
                        within_row_ord=2, within_col_ord=0,
                        x0=150.0, y0=50.0, x1=200.0, y1=100.0, confidence=0.9,
                    ),
                    NodePrediction(
                        fine_label="result_bar",
                        row_cluster_id=1, col_cluster_id=0,
                        within_row_ord=0, within_col_ord=0,
                        x0=50.0, y0=110.0, x1=200.0, y1=115.0, confidence=0.9,
                    ),
                ]
                # High-confidence logits so the entropy OOD gate does not trigger.
                return preds, "multiply", [-10.0, -10.0, 10.0, -10.0]

        gray = self._make_gray()
        result = run_inference_with_models(
            gray,
            gnn_model=_StubGnnMultiply(),
            prior=prior,
        )
        self.assertEqual(result["equation_kind"], "multiply",
                         f"Expected 'multiply', got {result['equation_kind']!r}")

    def test_prior_set_does_not_call_yolo_on_given_tiles(self) -> None:
        """YOLO stub must see only the original gray (no tile composited into it)
        when prior is provided -- given tiles paint into gray_for_graph, not gray."""
        yolo_images_seen: list = []

        class _CapturingYolo(_StubYoloModel):
            """_StubYoloModel that also records the raw image array passed in."""
            def predict(self, image: np.ndarray, **kwargs: object) -> list:
                yolo_images_seen.append(image.copy() if isinstance(image, np.ndarray) else image)
                return super().predict(image, **kwargs)

        class _StubGnn:
            def predict(self, data):
                return [], "unknown", [1.0, 0.0, 0.0, 0.0]

        gray = self._make_gray()
        # Prior with a single operator tile painted near the center.
        prior = [_make_given_node(200, 200, 260, 260, "operator", "op_plus")]
        stub_yolo = _CapturingYolo()

        run_inference_with_models(
            gray, gnn_model=_StubGnn(), yolo_model=stub_yolo, prior=prior,
        )

        # YOLO must have been called exactly once, on the original gray.
        self.assertEqual(len(yolo_images_seen), 1,
                         f"Expected YOLO called once, called {len(yolo_images_seen)} times")
        # The image YOLO saw must be identical to the original (no tile painted in).
        # _run_yolo converts gray to RGB, so compare the luminance channel.
        yolo_img = yolo_images_seen[0]
        # If RGB: check all channels equal the gray fill value.
        if yolo_img.ndim == 3:
            np.testing.assert_array_equal(
                yolo_img[:, :, 0],
                np.full((512, 512), 200, dtype=np.uint8),
                err_msg="YOLO received a modified image; given tile should not affect YOLO input",
            )
        else:
            np.testing.assert_array_equal(yolo_img, gray,
                err_msg="YOLO received a modified image")

    def test_prior_set_inference_session_forwards_prior(self) -> None:
        """InferenceSession.predict must forward the prior kwarg to run_inference_with_models."""
        from src.core.run_config import YoloInferenceConfig
        from src.inference.run import InferenceSession

        received_prior = []

        original_fn = run_inference_with_models

        def _patched(**kwargs):
            received_prior.append(kwargs.get("prior"))
            return {"equation_kind": "unknown", "rows": [], "tokens": [],
                    "quality": {"degraded": True, "reason": "test"},
                    "timings": {}, "schema_version": 1, "slots": {},
                    "_version": "test", "truncated": False}

        class _StubGnn:
            def predict(self, data):
                return [], "unknown", None

        session = InferenceSession()
        session._gnn_model = _StubGnn()
        session._yolo_model = None
        session._yolo_inference_config = YoloInferenceConfig()

        prior = [_make_given_node(10, 10, 50, 50)]

        with mock.patch("src.inference.run.run_inference_with_models",
                        side_effect=lambda *a, **kw: _patched(**kw)):
            session.predict(np.full((512, 512), 200, dtype=np.uint8), prior=prior)

        self.assertEqual(len(received_prior), 1)
        self.assertIs(received_prior[0], prior)


if __name__ == "__main__":
    unittest.main()
