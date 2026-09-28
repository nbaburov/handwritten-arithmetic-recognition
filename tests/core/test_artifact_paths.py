from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from src.core.config import DataPrepConfig
from src.core.artifact_paths import gnn_runs_root, resolve_gnn_best_pt, write_active_gnn


class TestGNNArtifactPaths(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.config = DataPrepConfig.from_project_root(Path(self._tmp))

    def test_artifacts_gnn_dir_property(self) -> None:
        self.assertEqual(
            self.config.artifacts_gnn_dir,
            Path(self._tmp) / "artifacts" / "gnn",
        )

    def test_gnn_runs_root(self) -> None:
        self.assertEqual(
            gnn_runs_root(self.config),
            Path(self._tmp) / "artifacts" / "gnn" / "runs",
        )

    def test_resolve_gnn_best_pt_raises_when_missing(self) -> None:
        with self.assertRaises(FileNotFoundError):
            resolve_gnn_best_pt(self.config)

    def test_resolve_gnn_best_pt_finds_flat_file(self) -> None:
        """resolve_gnn_best_pt returns the flat best.pt when it exists."""
        gnn_dir = self.config.artifacts_gnn_dir
        gnn_dir.mkdir(parents=True, exist_ok=True)
        best = gnn_dir / "best.pt"
        best.write_bytes(b"fake weights")
        self.assertEqual(resolve_gnn_best_pt(self.config), best)

    def test_resolve_gnn_best_pt_uses_active_json_pointer(self) -> None:
        """resolve_gnn_best_pt follows the best_pt pointer in active.json."""
        gnn_dir = self.config.artifacts_gnn_dir
        gnn_dir.mkdir(parents=True, exist_ok=True)
        run_dir = gnn_dir / "runs" / "20260101T000000Z_abc"
        run_dir.mkdir(parents=True, exist_ok=True)
        weights = run_dir / "best.pt"
        weights.write_bytes(b"run weights")
        active = gnn_dir / "active.json"
        active.write_text(json.dumps({"best_pt": f"runs/20260101T000000Z_abc/best.pt"}))
        self.assertEqual(resolve_gnn_best_pt(self.config), weights)

    def test_write_active_gnn_creates_correct_json(self) -> None:
        """write_active_gnn writes active.json with required fields."""
        out = write_active_gnn(
            self.config,
            "run_001",
            best_pt_relative="runs/run_001/best.pt",
            run_manifest_relative="runs/run_001/run.json",
            extra={"metrics": {"test_fine_acc": 0.98}},
        )
        self.assertTrue(out.is_file())
        data = json.loads(out.read_text())
        self.assertEqual(data["stage"], "gnn")
        self.assertEqual(data["run_id"], "run_001")
        self.assertEqual(data["best_pt"], "runs/run_001/best.pt")
        self.assertEqual(data["run_manifest"], "runs/run_001/run.json")
        self.assertIn("updated_utc", data)
        self.assertEqual(data["metrics"]["test_fine_acc"], 0.98)


if __name__ == "__main__":
    unittest.main()
