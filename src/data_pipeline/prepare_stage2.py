from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from ..core.config import DataPrepConfig, resolve_project_root
from ..core.ontology import glyph_key_for_pool, parse_flattened_label, safe_flattened_to_yolo_class_name
from ..core.progress import progress_iter
from .prep_gates import is_stage2_artifacts_complete
from .dataset import (
    build_class_map,
    detect_duplicates,
    enforce_quality_gates,
    ensure_dirs,
    scan_dataset,
    stratified_split,
    validate_label_vocabulary,
    write_json,
    write_split_manifests,
)
from .image_io import flatten_label_indices, load_png_grayscale_uint8 as load_png_grayscale, normalize_to_float32


def enrich_manifest_with_ontology(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["yolo_class"] = out["label"].astype(str).map(safe_flattened_to_yolo_class_name)
    roles, glyphs = [], []
    for lab in out["label"].astype(str):
        role, _ = parse_flattened_label(lab)
        roles.append(role)
        try:
            glyphs.append(glyph_key_for_pool(lab))
        except ValueError:
            glyphs.append("")
    out["role"] = roles
    out["glyph_key"] = glyphs
    return out


def export_stage2_tensors(
    split_map: Dict[str, pd.DataFrame],
    class_map: Dict[str, int],
    output_dir: Path,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}

    for split_name, split_df in split_map.items():
        images: list[np.ndarray] = []
        labels: list[int] = []
        image_ids: list[str] = []

        rows = list(split_df.itertuples(index=False))
        for row in progress_iter(rows, total=len(rows), desc=f"Stage2 {split_name} export"):
            arr = load_png_grayscale(Path(row.path))
            images.append(normalize_to_float32(arr))
            labels.append(class_map[row.label])
            image_ids.append(row.image_id)

        x = np.stack(images, axis=0).astype(np.float32)
        y = flatten_label_indices(labels)
        ids = np.array(image_ids, dtype=object)

        out_file = output_dir / f"{split_name}.npz"
        np.savez_compressed(out_file, x=x, y=y, image_id=ids)
        outputs[split_name] = out_file

    return outputs


def run_stage2_prepare(config: DataPrepConfig) -> Dict[str, Path]:
    ensure_dirs([config.data_processed_dir, config.stage2_dir])
    df, issues = scan_dataset(config)
    issues.extend(validate_label_vocabulary(df, config))
    enforce_quality_gates(df, issues)

    duplicates_df = detect_duplicates(df)
    split_map = stratified_split(df, config)
    split_map = {name: enrich_manifest_with_ontology(part) for name, part in split_map.items()}
    manifests = write_split_manifests(split_map=split_map, output_dir=config.stage2_dir / "manifests")
    class_map = build_class_map(df)
    tensors = export_stage2_tensors(split_map=split_map, class_map=class_map, output_dir=config.stage2_dir / "arrays")

    write_json(class_map, config.stage2_dir / "class_to_idx.json", sort_keys=False)
    write_json(
        {
            "rows": int(len(df)),
            "classes": sorted(class_map.keys()),
            "split_counts": {split: int(len(frame)) for split, frame in split_map.items()},
            "duplicate_count": int(len(duplicates_df)),
        },
        config.stage2_dir / "dataset_meta.json",
    )

    if not duplicates_df.empty:
        duplicates_df.to_csv(config.stage2_dir / "duplicates.csv", index=False)

    outputs: Dict[str, Path] = {}
    outputs.update(manifests)
    outputs.update({f"{k}_npz": v for k, v in tensors.items()})
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Stage 2 classification dataset artifacts.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=resolve_project_root(Path(__file__)),
        help="Path to project root containing data/ and reports/.",
    )
    parser.add_argument(
        "--skip-if-complete",
        action="store_true",
        help="Skip if stage2 NPZ arrays and class_to_idx already exist (use --force to rebuild).",
    )
    parser.add_argument("--force", action="store_true", help="Run even if stage2 outputs exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DataPrepConfig.from_project_root(args.project_root)
    if args.skip_if_complete and not args.force and is_stage2_artifacts_complete(config):
        print(json.dumps({"skipped": True, "reason": "stage2_artifacts_present"}, indent=2))
        return
    run_stage2_prepare(config)


if __name__ == "__main__":
    main()
