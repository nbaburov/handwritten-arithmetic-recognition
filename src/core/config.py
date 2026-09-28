from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Optional, Tuple

from .ontology import digit_pool_folder_labels

MAX_NODES: int = 80  # Hard cap on nodes per scene; detections above this are truncated by confidence.


@dataclass(frozen=True)
class DataPrepConfig:
    project_root: Path
    data_pool_emnist_28_dir: Path
    data_processed_dir: Path
    reports_dir: Path
    random_seed: int = 42
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    expected_size: Tuple[int, int] = (28, 28)
    expected_channels: int = 1
    expected_extension: str = ".png"
    allowed_labels: Dict[str, int] = field(default_factory=dict)
    required_labels: Optional[FrozenSet[str]] = None
    min_samples_per_required_label: int = 2
    @property
    def yolo_dir(self) -> Path:
        return self.data_processed_dir / "yolo"

    @property
    def stage2_dir(self) -> Path:
        return self.data_processed_dir / "stage2"

    @property
    def reports_prep_dir(self) -> Path:
        return self.reports_dir / "prep"

    @property
    def reports_eval_dir(self) -> Path:
        return self.reports_dir / "eval"

    @property
    def artifacts_root(self) -> Path:
        return self.project_root / "artifacts"

    @property
    def artifacts_yolo_dir(self) -> Path:
        return self.artifacts_root / "yolo"

    @property
    def artifacts_stage1_dir(self) -> Path:
        """Alias for artifacts_yolo_dir; used by Stage 1 training helpers."""
        return self.artifacts_root / "yolo"

    @property
    def artifacts_stage2_dir(self) -> Path:
        return self.artifacts_root / "stage2"

    @property
    def artifacts_gnn_dir(self) -> Path:
        return self.artifacts_root / "gnn"

    @property
    def synthetic_dir(self) -> Path:
        return self.project_root / "data" / "generated" / "synthetic"

    @staticmethod
    def from_project_root(
        project_root: Path,
    ) -> "DataPrepConfig":
        data_pool_emnist_28_dir = project_root / "data" / "raw" / "pool_emnist_28"
        data_processed_dir = project_root / "data" / "generated" / "processed"
        reports_dir = project_root / "reports"

        allowed_labels: Dict[str, int] = {}
        required: FrozenSet[str] = digit_pool_folder_labels()

        return DataPrepConfig(
            project_root=project_root,
            data_pool_emnist_28_dir=data_pool_emnist_28_dir,
            data_processed_dir=data_processed_dir,
            reports_dir=reports_dir,
            allowed_labels=allowed_labels,
            required_labels=required,
        )


def resolve_project_root(current_file: Path) -> Path:
    here = current_file.resolve()
    for d in (here.parent, *here.parents):
        if (d / "data").is_dir() and (d / "src").is_dir():
            return d
    raise FileNotFoundError(
        f"Could not find project root (folder containing both data/ and src/); started from {here}"
    )
