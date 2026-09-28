"""Tests that train_stage1 passes the required augmentation overrides to Ultralytics."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_config(tmp: Path):
    from src.core.config import DataPrepConfig
    from src.core.ontology import digit_pool_folder_labels
    return DataPrepConfig(
        project_root=tmp,
        data_pool_emnist_28_dir=tmp / "data" / "raw" / "pool_emnist_28",
        data_processed_dir=tmp / "data" / "generated" / "processed",
        reports_dir=tmp / "reports",
        required_labels=digit_pool_folder_labels(),
    )


def _setup_minimal_synth_dir(config) -> None:
    import os
    from src.data_pipeline.prepare_synthetic_yolo import LATEST_POINTER
    synth_dir = config.synthetic_dir
    synth_dir.mkdir(parents=True, exist_ok=True)
    # New layout: data inside a timestamped run folder; latest pointer resolves it.
    run_dir = synth_dir / "20260508T000000Z_testtest"
    run_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (run_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (run_dir / f"{split}_manifest.csv").write_text(
            "image,label,ground_truth,box_count,case\n", encoding="utf-8"
        )
    (run_dir / "train_manifest.csv").write_text(
        "image,label,ground_truth,box_count,case\n", encoding="utf-8"
    )
    latest = synth_dir / LATEST_POINTER
    if not latest.exists() and not latest.is_symlink():
        os.symlink(run_dir, latest)
    (config.artifacts_stage1_dir).mkdir(parents=True, exist_ok=True)


class TestTrainStage1AugmentationOverrides(unittest.TestCase):
    """Verify augmentation overrides are passed to model.train and recorded in run manifest."""

    def _run_with_mock_yolo(self, tmp: Path):
        config = _make_config(tmp)
        _setup_minimal_synth_dir(config)

        mock_result = MagicMock()
        mock_result.results_dict = {
            "metrics/mAP50(B)": 0.5,
            "metrics/mAP50-95(B)": 0.3,
        }
        mock_val_result = MagicMock()
        mock_val_result.results_dict = {
            "metrics/mAP50(B)": 0.5,
            "metrics/mAP50-95(B)": 0.3,
        }
        mock_model = MagicMock()
        mock_model.train.return_value = mock_result
        mock_model.val.return_value = mock_val_result

        with patch("ultralytics.YOLO", return_value=mock_model):
            from src.training.run import train_stage1
            train_stage1(config, epochs=1, image_size=512, batch=2, workers=0)

        return mock_model

    def test_model_train_called_with_fliplr_zero(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            mock_model = self._run_with_mock_yolo(Path(d))
        call_kwargs = mock_model.train.call_args.kwargs
        self.assertIn("fliplr", call_kwargs)
        self.assertEqual(call_kwargs["fliplr"], 0.0)

    def test_model_train_called_with_mosaic_zero(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            mock_model = self._run_with_mock_yolo(Path(d))
        call_kwargs = mock_model.train.call_args.kwargs
        self.assertIn("mosaic", call_kwargs)
        self.assertEqual(call_kwargs["mosaic"], 0.0)

    def test_model_train_called_with_flipud_zero(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            mock_model = self._run_with_mock_yolo(Path(d))
        call_kwargs = mock_model.train.call_args.kwargs
        self.assertEqual(call_kwargs.get("flipud"), 0.0)

    def test_model_train_called_with_hsv_h_zero(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            mock_model = self._run_with_mock_yolo(Path(d))
        call_kwargs = mock_model.train.call_args.kwargs
        self.assertEqual(call_kwargs.get("hsv_h"), 0.0)

    def test_model_train_called_with_hsv_s_low(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            mock_model = self._run_with_mock_yolo(Path(d))
        call_kwargs = mock_model.train.call_args.kwargs
        # hsv_s is kept low (0.05) rather than zero — mild saturation jitter is intentional.
        self.assertLessEqual(call_kwargs.get("hsv_s", 1.0), 0.1)

    def test_model_train_called_with_modest_degrees(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            mock_model = self._run_with_mock_yolo(Path(d))
        call_kwargs = mock_model.train.call_args.kwargs
        self.assertAlmostEqual(call_kwargs.get("degrees", 0.0), 5.0)

    def test_run_manifest_contains_augmentation_field(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            config = _make_config(tmp)
            _setup_minimal_synth_dir(config)

            mock_result = MagicMock()
            mock_result.results_dict = {"metrics/mAP50(B)": 0.4, "metrics/mAP50-95(B)": 0.2}
            mock_val_result = MagicMock()
            mock_val_result.results_dict = {"metrics/mAP50(B)": 0.4, "metrics/mAP50-95(B)": 0.2}
            mock_model = MagicMock()
            mock_model.train.return_value = mock_result
            mock_model.val.return_value = mock_val_result

            with patch("ultralytics.YOLO", return_value=mock_model):
                from src.training.run import train_stage1
                outputs = train_stage1(config, epochs=1, image_size=512, batch=2, workers=0)

            manifest_path = outputs["run_manifest"]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertIn("augmentation", manifest)
        aug = manifest["augmentation"]
        self.assertEqual(aug["fliplr"], 0.0)
        self.assertEqual(aug["mosaic"], 0.0)
        self.assertEqual(aug["flipud"], 0.0)
        self.assertEqual(aug["hsv_h"], 0.0)

    def test_run_manifest_augmentation_records_perspective_zero(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            config = _make_config(tmp)
            _setup_minimal_synth_dir(config)

            mock_result = MagicMock()
            mock_result.results_dict = {}
            mock_val_result = MagicMock()
            mock_val_result.results_dict = {}
            mock_model = MagicMock()
            mock_model.train.return_value = mock_result
            mock_model.val.return_value = mock_val_result

            with patch("ultralytics.YOLO", return_value=mock_model):
                from src.training.run import train_stage1
                outputs = train_stage1(config, epochs=1, image_size=512, batch=2, workers=0)

            manifest = json.loads(outputs["run_manifest"].read_text(encoding="utf-8"))

        self.assertEqual(manifest["augmentation"]["perspective"], 0.0)


if __name__ == "__main__":
    unittest.main()
