from __future__ import annotations

from ..core.config import DataPrepConfig


def is_yolo_artifacts_complete(config: DataPrepConfig) -> bool:
    return (config.yolo_dir / "symbol_assets_manifest.csv").is_file()


def is_stage2_artifacts_complete(config: DataPrepConfig) -> bool:
    return (
        (config.stage2_dir / "arrays" / "train.npz").is_file()
        and (config.stage2_dir / "class_to_idx.json").is_file()
    )


def is_full_prep_complete(config: DataPrepConfig) -> bool:
    return is_yolo_artifacts_complete(config) and is_stage2_artifacts_complete(config)


def is_synthetic_yolo_complete(config: DataPrepConfig) -> bool:
    """True when a valid latest run folder exists with a train manifest."""
    from .prepare_synthetic_yolo import resolve_latest_run
    run_dir = resolve_latest_run(config.synthetic_dir)
    if run_dir is None:
        return False
    return (run_dir / "train_manifest.csv").is_file()
