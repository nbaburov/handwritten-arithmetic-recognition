from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..core.config import DataPrepConfig, resolve_project_root
from ..core.progress import progress_bar
from .prep_gates import is_full_prep_complete
from .dataset import detect_duplicates, scan_dataset, validate_label_vocabulary
from .prepare_stage1 import run_yolo_prepare
from .prepare_stage2 import run_stage2_prepare


def _split_leakage_report(stage2_manifest_dir: Path) -> pd.DataFrame:
    train = pd.read_csv(stage2_manifest_dir / "train.csv")
    val = pd.read_csv(stage2_manifest_dir / "val.csv")
    test = pd.read_csv(stage2_manifest_dir / "test.csv")

    def overlap(df_a: pd.DataFrame, df_b: pd.DataFrame, left: str, right: str) -> int:
        return int(df_a["sha256"].isin(df_b["sha256"]).sum())

    return pd.DataFrame(
        [
            {"pair": "train-val", "overlap_count": overlap(train, val, "train", "val")},
            {"pair": "train-test", "overlap_count": overlap(train, test, "train", "test")},
            {"pair": "val-test", "overlap_count": overlap(val, test, "val", "test")},
        ]
    )


def run_full_data_prep(config: DataPrepConfig) -> None:
    config.reports_prep_dir.mkdir(parents=True, exist_ok=True)
    with progress_bar(total=5, desc="Pipeline") as pipeline_bar:
        run_yolo_prepare(config)
        pipeline_bar.update(1)
        run_stage2_prepare(config)
        pipeline_bar.update(1)

        dataset_df, issues = scan_dataset(config)
        label_issues = validate_label_vocabulary(dataset_df, config)
        issues.extend(label_issues)
        duplicates = detect_duplicates(dataset_df)
        pipeline_bar.update(1)

        class_dist = (
            dataset_df.groupby("label", as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values("label")
        )
        class_dist.to_csv(config.reports_prep_dir / "class-distribution.csv", index=False)

        leakage = _split_leakage_report(config.stage2_dir / "manifests")
        leakage.to_csv(config.reports_prep_dir / "split-leakage-check.csv", index=False)
        pipeline_bar.update(1)

        summary_lines = [
            "# Data Preparation Summary",
            "",
            f"- total_images: {len(dataset_df)}",
            f"- class_count: {dataset_df['label'].nunique() if not dataset_df.empty else 0}",
            f"- duplicate_hash_rows: {len(duplicates)}",
            f"- issue_count: {len(issues)}",
            "",
            "## Leakage",
        ]
        for row in leakage.itertuples(index=False):
            summary_lines.append(f"- {row.pair}: {row.overlap_count}")

        (config.reports_prep_dir / "data-prep-summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

        if issues:
            issue_df = pd.DataFrame([issue.__dict__ for issue in issues])
            issue_df.to_csv(config.reports_prep_dir / "dataset-issues.csv", index=False)
        pipeline_bar.update(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full data preparation and validation reports.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=resolve_project_root(Path(__file__)),
        help="Path to project root containing data/ and reports/.",
    )
    parser.add_argument(
        "--skip-if-complete",
        action="store_true",
        help="Exit successfully if processed yolo+stage2 artifacts already exist (use --force to rebuild).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run full prep even if outputs exist (opposes --skip-if-complete).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DataPrepConfig.from_project_root(args.project_root)
    if args.skip_if_complete and not args.force and is_full_prep_complete(config):
        payload = {"skipped": True, "reason": "processed_data_already_present"}
        print(json.dumps(payload, indent=2))
        return
    run_full_data_prep(config)


if __name__ == "__main__":
    main()
