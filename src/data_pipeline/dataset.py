from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..core.config import DataPrepConfig
from ..core.ontology import EXCLUDED_DATASET_LABELS
from ..core.progress import progress_iter
from .image_io import load_png_grayscale_uint8


@dataclass(frozen=True)
class DatasetIssue:
    kind: str
    path: str
    detail: str


def label_from_path(file_path: Path, pool_dir: Path) -> str:
    rel = file_path.relative_to(pool_dir)
    if len(rel.parts) < 2:
        raise ValueError(f"Unable to infer label from path: {file_path}")
    return rel.parts[0]


from ..core.hashing import sha256_file as file_sha256  # noqa: E402


def scan_dataset(config: DataPrepConfig) -> Tuple[pd.DataFrame, List[DatasetIssue]]:
    records: List[Dict[str, object]] = []
    issues: List[DatasetIssue] = []

    if not config.data_pool_emnist_28_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {config.data_pool_emnist_28_dir}")

    all_paths = sorted(config.data_pool_emnist_28_dir.rglob("*"))
    for file_path in progress_iter(all_paths, total=len(all_paths), desc="Scanning files"):
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        if ext != config.expected_extension:
            issues.append(
                DatasetIssue("invalid_extension", str(file_path), f"Expected {config.expected_extension}, got {ext}")
            )
            continue

        try:
            label = label_from_path(file_path=file_path, pool_dir=config.data_pool_emnist_28_dir)
        except ValueError as exc:
            issues.append(DatasetIssue("invalid_label_path", str(file_path), str(exc)))
            continue

        if label in EXCLUDED_DATASET_LABELS:
            continue

        image_hash = file_sha256(file_path)
        records.append(
            {
                "image_id": image_hash[:16],
                "path": str(file_path),
                "label": label,
                "sha256": image_hash,
            }
        )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df, issues

    shape_df, shape_issues = attach_image_metadata(df["path"].tolist(), config=config)
    issues.extend(shape_issues)

    merged = df.merge(shape_df, how="left", on="path")
    merged["is_valid"] = ~(merged["path"].isin([issue.path for issue in shape_issues]))
    return merged, issues


def attach_image_metadata(paths: Sequence[str], config: DataPrepConfig) -> Tuple[pd.DataFrame, List[DatasetIssue]]:
    rows: List[Dict[str, object]] = []
    issues: List[DatasetIssue] = []

    for path_str in progress_iter(paths, total=len(paths), desc="Validating images"):
        path = Path(path_str)
        try:
            image = load_png_grayscale_uint8(path)
            height, width, channels = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
        except Exception as exc:  # pragma: no cover - defensive parsing
            issues.append(DatasetIssue("corrupt_image", path_str, str(exc)))
            continue

        effective_channels = channels
        _ = effective_channels

        rows.append(
            {
                "path": path_str,
                "width": width,
                "height": height,
                "channels": effective_channels,
            }
        )

        if (width, height) != config.expected_size:
            issues.append(
                DatasetIssue(
                    "invalid_size",
                    path_str,
                    f"Expected {config.expected_size}, got {(width, height)}",
                )
            )
        if effective_channels != config.expected_channels:
            issues.append(
                DatasetIssue(
                    "invalid_channels",
                    path_str,
                    f"Expected {config.expected_channels}, got {channels}",
                )
            )

    return pd.DataFrame.from_records(rows), issues


def validate_label_vocabulary(df: pd.DataFrame, config: DataPrepConfig) -> List[DatasetIssue]:
    issues: List[DatasetIssue] = []
    label_set = set(df["label"].unique().tolist())
    labels = sorted(label_set)
    if config.required_labels:
        allowed = set(config.required_labels)
        unknown = [label for label in labels if label not in allowed]
        for label in unknown:
            issues.append(DatasetIssue("unknown_label", label, "Label not in required ontology"))
        for label in sorted(config.required_labels):
            if label not in label_set:
                issues.append(DatasetIssue("missing_required_label", label, "No samples for required ontology label"))
            else:
                count = int((df["label"] == label).sum())
                if count < config.min_samples_per_required_label:
                    issues.append(
                        DatasetIssue(
                            "insufficient_class_count",
                            label,
                            f"Need >= {config.min_samples_per_required_label} samples, got {count}",
                        )
                    )
    elif config.allowed_labels:
        unknown = [label for label in labels if label not in config.allowed_labels]
        for label in unknown:
            issues.append(DatasetIssue("unknown_label", label, "Label not in configured vocabulary"))

    label_counts = df["label"].value_counts()
    for label, count in label_counts.items():
        if count <= 0:
            issues.append(DatasetIssue("empty_class", label, "Class has zero samples"))
    return issues


def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    dup_mask = df.duplicated(subset=["sha256"], keep=False)
    return df.loc[dup_mask].sort_values(["sha256", "path"]).copy()


def enforce_quality_gates(df: pd.DataFrame, issues: Sequence[DatasetIssue]) -> None:
    issue_kinds = {issue.kind for issue in issues}
    hard_fail_kinds = {
        "corrupt_image",
        "invalid_size",
        "invalid_channels",
        "unknown_label",
        "invalid_label_path",
        "missing_required_label",
        "insufficient_class_count",
    }
    failures = issue_kinds.intersection(hard_fail_kinds)
    if failures:
        sample = [f"{issue.kind}: {issue.path} ({issue.detail})" for issue in issues[:10]]
        raise ValueError(
            "Dataset validation failed due to quality gate violations: "
            + ", ".join(sorted(failures))
            + ". Sample issues: "
            + " | ".join(sample)
        )
    if df.empty:
        raise ValueError("Dataset validation failed: no valid images found.")


def _validate_split_ratios(ratios: Tuple[float, float, float]) -> None:
    train, val, test = ratios
    if min(ratios) <= 0:
        raise ValueError("Split ratios must all be greater than zero.")
    if not np.isclose(train + val + test, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")


def _stratified_train_test_attach_rare_to_first(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split; labels with <2 rows in *df* are all kept in the first part."""
    counts = df["label"].value_counts()
    rare_mask = df["label"].map(counts) < 2
    rare_df = df.loc[rare_mask].copy()
    main_df = df.loc[~rare_mask].copy()

    if main_df.empty:
        empty = df.iloc[0:0].copy()
        if rare_df.empty:
            return empty, empty
        return rare_df.reset_index(drop=True), empty

    first, second = train_test_split(
        main_df,
        test_size=test_size,
        stratify=main_df["label"],
        random_state=random_state,
    )
    if not rare_df.empty:
        first = pd.concat([first, rare_df], ignore_index=True)
    return first, second


def stratified_split(df: pd.DataFrame, config: DataPrepConfig) -> Dict[str, pd.DataFrame]:
    _validate_split_ratios(config.split_ratios)
    train_ratio, val_ratio, test_ratio = config.split_ratios
    _ = train_ratio

    work_df = df.copy()
    work_df = work_df.drop_duplicates(subset=["sha256"], keep="first")
    work_df = work_df.sort_values("sha256").reset_index(drop=True)

    train_df, temp_df = _stratified_train_test_attach_rare_to_first(
        work_df,
        test_size=(val_ratio + test_ratio),
        random_state=config.random_seed,
    )
    val_size_in_temp = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = _stratified_train_test_attach_rare_to_first(
        temp_df,
        test_size=(1 - val_size_in_temp),
        random_state=config.random_seed,
    )

    return {
        "train": train_df.sort_values("sha256").reset_index(drop=True),
        "val": val_df.sort_values("sha256").reset_index(drop=True),
        "test": test_df.sort_values("sha256").reset_index(drop=True),
    }


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def write_split_manifests(split_map: Dict[str, pd.DataFrame], output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}
    for split_name, split_df in split_map.items():
        out_path = output_dir / f"{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        outputs[split_name] = out_path
    return outputs


def build_class_map(df: pd.DataFrame) -> Dict[str, int]:
    labels = sorted(df["label"].unique().tolist())
    return {label: idx for idx, label in enumerate(labels)}


def write_json(payload: Dict[str, object], target_path: Path, *, sort_keys: bool = True) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, indent=2, sort_keys=sort_keys), encoding="utf-8")
