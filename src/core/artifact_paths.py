from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .config import DataPrepConfig


def new_training_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{uuid.uuid4().hex[:8]}"


def try_git_revision(project_root: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def yolo_runs_root(config: DataPrepConfig) -> Path:
    return config.artifacts_yolo_dir / "runs"


def read_active_artifact(stage_dir: Path) -> Optional[Dict[str, Any]]:
    path = stage_dir / "active.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_active_yolo(
    config: DataPrepConfig,
    run_id: str,
    *,
    best_pt_relative: str,
    run_manifest_relative: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    payload: Dict[str, Any] = {
        "stage": "yolo",
        "run_id": run_id,
        "best_pt": best_pt_relative,
        "run_manifest": run_manifest_relative,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    out = config.artifacts_yolo_dir / "active.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def append_runs_index(runs_root: Path, entry: Dict[str, Any]) -> None:
    runs_root.mkdir(parents=True, exist_ok=True)
    index_path = runs_root / "index.json"
    data: Dict[str, Any] = {"runs": []}
    if index_path.is_file():
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {"runs": []}
    if "runs" not in data or not isinstance(data["runs"], list):
        data["runs"] = []
    data["runs"].append(entry)
    index_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def find_yolo_best_pt(config: DataPrepConfig) -> Optional[Path]:
    active = read_active_artifact(config.artifacts_yolo_dir)
    if active:
        rel = active.get("best_pt")
        if isinstance(rel, str):
            p = config.artifacts_yolo_dir / rel
            if p.is_file():
                return p
    base = config.artifacts_yolo_dir
    if not base.exists():
        return None
    flat = base / "best.pt"
    if flat.is_file():
        return flat
    for p in sorted(base.rglob("best.pt"), reverse=True):
        return p
    return None


def gnn_runs_root(config: DataPrepConfig) -> Path:
    return config.artifacts_gnn_dir / "runs"


def resolve_gnn_best_pt(config: DataPrepConfig) -> Path:
    active = read_active_artifact(config.artifacts_gnn_dir)
    if active:
        rel = active.get("best_pt")
        if isinstance(rel, str):
            p = config.artifacts_gnn_dir / rel
            if p.is_file():
                return p
    flat = config.artifacts_gnn_dir / "best.pt"
    if flat.is_file():
        return flat
    raise FileNotFoundError(
        f"No GNN model found. Train with: python -m src train --stage gnn. "
        f"Expected {flat} or active.json pointer."
    )


def write_active_gnn(
    config: DataPrepConfig,
    run_id: str,
    *,
    best_pt_relative: str,
    run_manifest_relative: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    payload: Dict[str, Any] = {
        "stage": "gnn",
        "run_id": run_id,
        "best_pt": best_pt_relative,
        "run_manifest": run_manifest_relative,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    out = config.artifacts_gnn_dir / "active.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
