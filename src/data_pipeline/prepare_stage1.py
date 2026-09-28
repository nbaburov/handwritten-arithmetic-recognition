from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd

from ..core.config import DataPrepConfig, resolve_project_root
from .prep_gates import is_yolo_artifacts_complete
from ..core.ontology import glyph_key_for_pool, parse_flattened_label, safe_flattened_to_yolo_class_name
from .dataset import (
    enforce_quality_gates,
    ensure_dirs,
    scan_dataset,
    stratified_split,
    validate_label_vocabulary,
    write_json,
    write_split_manifests,
)


def build_symbol_assets_manifest(split_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for split_name, split_df in split_map.items():
        tmp = split_df.copy()
        tmp["split"] = split_name
        rows.append(tmp)
    all_rows = pd.concat(rows, axis=0, ignore_index=True)
    all_rows["asset_id"] = all_rows["sha256"].str.slice(0, 20)
    all_rows["source_type"] = "combined_symbol"
    all_rows["yolo_class"] = all_rows["label"].astype(str).map(safe_flattened_to_yolo_class_name)
    roles = []
    glyphs = []
    for lab in all_rows["label"].astype(str):
        role, _ = parse_flattened_label(lab)
        roles.append(role)
        try:
            glyphs.append(glyph_key_for_pool(lab))
        except ValueError:
            glyphs.append("")
    all_rows["role"] = roles
    all_rows["glyph_key"] = glyphs
    return all_rows.sort_values(["split", "label", "sha256"]).reset_index(drop=True)


def run_yolo_prepare(config: DataPrepConfig) -> Dict[str, Path]:
    ensure_dirs([config.data_processed_dir, config.yolo_dir])
    df, issues = scan_dataset(config)
    issues.extend(validate_label_vocabulary(df, config))
    enforce_quality_gates(df, issues)

    split_map = stratified_split(df, config)
    manifests = write_split_manifests(split_map=split_map, output_dir=config.yolo_dir / "splits")

    symbol_manifest = build_symbol_assets_manifest(split_map)
    symbol_manifest_path = config.yolo_dir / "symbol_assets_manifest.csv"
    symbol_manifest.to_csv(symbol_manifest_path, index=False)

    class_counts = (
        symbol_manifest.groupby(["split", "label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["split", "label"])
    )
    class_counts_path = config.yolo_dir / "symbol_class_balance.csv"
    class_counts.to_csv(class_counts_path, index=False)

    write_json(
        {
            "split_counts": {split: int(len(frame)) for split, frame in split_map.items()},
            "class_count": int(symbol_manifest["label"].nunique()),
            "record_count": int(len(symbol_manifest)),
        },
        config.yolo_dir / "yolo_meta.json",
    )

    outputs: Dict[str, Path] = {}
    outputs.update(manifests)
    outputs["symbol_assets_manifest"] = symbol_manifest_path
    outputs["symbol_class_balance"] = class_counts_path
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Stage 1 symbol asset manifests.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=resolve_project_root(Path(__file__)),
        help="Path to project root containing data/ and reports/.",
    )
    parser.add_argument(
        "--skip-if-complete",
        action="store_true",
        help="Skip if symbol_assets_manifest.csv already exists (use --force to rebuild).",
    )
    parser.add_argument("--force", action="store_true", help="Run even if yolo outputs exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DataPrepConfig.from_project_root(args.project_root)
    if args.skip_if_complete and not args.force and is_yolo_artifacts_complete(config):
        print(json.dumps({"skipped": True, "reason": "yolo_artifacts_present"}, indent=2))
        return
    run_yolo_prepare(config)


if __name__ == "__main__":
    main()
