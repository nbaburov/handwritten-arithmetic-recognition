from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

from ..core.config import DataPrepConfig, resolve_project_root
from ..core.logging_setup import configure_run_logging
from ..core.ontology import YOLO_CLASS_NAMES
from ..core.run_config import GenerationConfig, load_config
from ..generation.layouts import SceneCase
from ..generation.synth_yolo import build_synthetic_yolo_dataset
from .prep_gates import is_synthetic_yolo_complete
from .validate_yolo_labels import validate_yolo_labels

# Mapping from CLI-friendly hyphenated names to SceneCase enum values.
_SCENE_CASE_BY_VALUE: dict[str, SceneCase] = {c.value: c for c in SceneCase}


def parse_cases_arg(raw: str) -> list[SceneCase]:
    """Parse a comma-separated list of SceneCase value strings.

    Raises ValueError with a clear message if any name is not a valid
    SceneCase value.
    """
    names = [n.strip() for n in raw.split(",") if n.strip()]
    result: list[SceneCase] = []
    for name in names:
        if name not in _SCENE_CASE_BY_VALUE:
            valid = sorted(_SCENE_CASE_BY_VALUE.keys())
            raise ValueError(
                f"Unknown case name: {name!r}. Valid values: {valid}"
            )
        result.append(_SCENE_CASE_BY_VALUE[name])
    if not result:
        raise ValueError("--cases argument produced an empty list after parsing.")
    return result

logger = logging.getLogger(__name__)

LATEST_POINTER = "latest"  # name of the symlink/text file at data/generated/synthetic/latest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run_id() -> str:
    """Return a timestamped run-id: <UTC_TIMESTAMP>Z_<8char_hash>."""
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    h = hashlib.sha256(ts.encode()).hexdigest()[:8]
    return f"{ts}_{h}"


def _write_latest_pointer(synthetic_dir: Path, run_dir: Path) -> None:
    """Create/update data/generated/synthetic/latest symlink pointing at run_dir."""
    latest = synthetic_dir / LATEST_POINTER
    # Remove stale pointer (symlink or regular file).
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    try:
        os.symlink(run_dir, latest)
    except OSError:
        # Fallback: plain text file with the absolute path.
        latest.write_text(str(run_dir), encoding="utf-8")


def resolve_latest_run(synthetic_dir: Path) -> Optional[Path]:
    """Return the Path referenced by the latest pointer, or None if absent."""
    latest = synthetic_dir / LATEST_POINTER
    if latest.is_symlink():
        target = latest.resolve()
        return target if target.is_dir() else None
    if latest.is_file():
        target = Path(latest.read_text(encoding="utf-8").strip())
        return target if target.is_dir() else None
    return None


def _count_fine_labels(manifest_path: Path) -> Dict[str, int]:
    """Count fine label instances across all GT JSONs in a manifest."""
    import pandas as pd
    counts: Dict[str, int] = {}
    df = pd.read_csv(manifest_path)
    for gt_path in df["ground_truth"]:
        gt = json.loads(Path(gt_path).read_text(encoding="utf-8"))
        for tok in gt.get("symbols", []):
            lbl = tok.get("fine_label", "")
            if lbl:
                counts[lbl] = counts.get(lbl, 0) + 1
    return counts


def _write_dataset_yaml(run_dir: Path, names: List[str]) -> Path:
    """Write dataset.yaml with relative paths so the file is portable."""
    content = "\n".join([
        f"path: {str(run_dir)}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        f"nc: {len(names)}",
        "names: [" + ", ".join(f"'{n}'" for n in names) + "]",
    ])
    yaml_path = run_dir / "dataset.yaml"
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def _write_examples(
    run_dir: Path,
    case_completion_samples: Dict[str, Path],
) -> None:
    """Write examples/ folder — one PNG per (case × completion_stage) combo.

    ``case_completion_samples`` maps ``"<case>_<completion>"`` to a source PNG path.
    """
    examples_dir = run_dir / "examples"
    examples_dir.mkdir(exist_ok=True)

    readme_lines = [
        "# examples/",
        "",
        "One rendered scene per (case × completion-stage) combination.",
        "Images are taken from the training set.",
        "",
        "| filename | case | completion_stage |",
        "|----------|------|-----------------|",
    ]

    for key, src_png in sorted(case_completion_samples.items()):
        dst = examples_dir / f"{key}.png"
        shutil.copy2(src_png, dst)
        parts = key.split("_", 1)
        case_name = parts[0] if len(parts) == 2 else key
        stage_name = parts[1] if len(parts) == 2 else "full"
        readme_lines.append(f"| {key}.png | {case_name} | {stage_name} |")

    (examples_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def _safe_case_weight(gen_cfg: GenerationConfig, case: SceneCase) -> float:
    """Return the configured weight for *case*, clamped to >= 0.

    Logs a warning and clamps to 0.0 if a negative value is found in the
    config -- negative weights are meaningless and would produce negative
    image counts.
    """
    raw = gen_cfg.case_weights.get(case.value, 1.0)
    if raw < 0.0:
        logger.warning(
            "Negative case_weight %r for case %r; clamping to 0.0.",
            raw,
            case.value,
        )
        return 0.0
    return raw


def run_prepare_synthetic_yolo(
    config: DataPrepConfig,
    gen_cfg: GenerationConfig,
    cases: Optional[List[SceneCase]] = None,
) -> Dict[str, Path]:
    """Generate synthetic YOLO scenes and write to a timestamped run folder.

    Parameters
    ----------
    cases:
        Subset of SceneCase values to generate.  When *None* (the default)
        all 16 cases are generated -- existing behaviour is preserved.

    Returns
    -------
    Dict mapping split name to the aggregate manifest CSV inside the run folder.
    The run folder is registered in ``data/generated/synthetic/latest``.
    """
    run_id = _make_run_id()
    run_dir = config.synthetic_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "run.log"
    configure_run_logging(log_path)
    logger.info("Generate run %s → %s", run_id, run_dir)

    symbol_manifest = config.yolo_dir / "symbol_assets_manifest.csv"
    if not symbol_manifest.exists():
        raise FileNotFoundError(
            f"Missing {symbol_manifest}. Run validate before generate."
        )

    min_instances_per_label = gen_cfg.min_instances_per_label
    topup_rounds = gen_cfg.topup_rounds

    all_cases: List[SceneCase] = list(cases) if cases is not None else list(SceneCase)
    if not all_cases:
        raise ValueError("cases list is empty; nothing to generate.")
    if cases is not None:
        logger.info("Case filter active: generating %d case(s): %s", len(all_cases), [c.value for c in all_cases])
    train_ratio = gen_cfg.split_train
    val_ratio = gen_cfg.split_val
    test_ratio = gen_cfg.split_test

    # Per-case manifests keyed by split.  Each manifest lives inside run_dir.
    case_manifests: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}
    # Track one sample PNG per (case × completion_stage) for examples/.
    examples_samples: Dict[str, Path] = {}

    for case in all_cases:
        per_case = max(1, round(gen_cfg.images_per_case * _safe_case_weight(gen_cfg, case)))
        per_case_val = max(1, round(per_case * val_ratio / train_ratio))
        per_case_test = max(1, round(per_case * test_ratio / train_ratio))

        for split_name, n in [("train", per_case), ("val", per_case_val), ("test", per_case_test)]:
            result = build_synthetic_yolo_dataset(
                config, case, split_name, n,
                completion_weights=gen_cfg.completion,
                rendering=gen_cfg.rendering,
                preset_cfg=gen_cfg.preset,
                scene_cfg=gen_cfg.scene,
                out_dir=run_dir,
                source_pool=gen_cfg.source_pool,
                min_source_quality=gen_cfg.min_source_quality,
            )
            manifest = result["manifest"]
            case_manifests[split_name].append(manifest)
            # H-2: validate runs on the shared flat labels dir (all cases co-located).
            # Per-case manifests now have unique paths (case.value prefix via C-1 fix),
            # so orphan risk is zero — label files are named {case}_{split}_{i:06d}.txt
            # and the manifest rows carry absolute paths.
            # TODO: extend validate_yolo_labels to accept a manifest path and
            #       cross-check that every label file referenced in the manifest
            #       exists and every file in labels_dir is referenced by some manifest.
            validate_yolo_labels(run_dir / split_name / "labels")

            # Collect example images from training split only.
            if split_name == "train":
                _collect_examples(manifest, examples_samples)

    # Class-conditional top-up.
    if min_instances_per_label is not None:
        import pandas as _pd
        for _round in range(topup_rounds):
            global_counts: Dict[str, int] = {}
            for mp in case_manifests["train"]:
                for lbl, cnt in _count_fine_labels(mp).items():
                    global_counts[lbl] = global_counts.get(lbl, 0) + cnt

            deficit_labels = {lbl for lbl, cnt in global_counts.items() if cnt < min_instances_per_label}
            if not deficit_labels:
                break

            for idx, case in enumerate(all_cases):
                case_counts = _count_fine_labels(case_manifests["train"][idx])
                case_deficit = deficit_labels & set(case_counts.keys())
                if not case_deficit:
                    continue
                per_case = max(1, round(gen_cfg.images_per_case * _safe_case_weight(gen_cfg, case)))
                additional = per_case
                current_n = len(_pd.read_csv(case_manifests["train"][idx]))
                result = build_synthetic_yolo_dataset(
                    config, case, "train", additional,
                        start_index=current_n,
                    completion_weights=gen_cfg.completion,
                    rendering=gen_cfg.rendering,
                    preset_cfg=gen_cfg.preset,
                    scene_cfg=gen_cfg.scene,
                    out_dir=run_dir,
                    source_pool=gen_cfg.source_pool,
                    min_source_quality=gen_cfg.min_source_quality,
                )
                case_manifests["train"][idx] = result["manifest"]
                validate_yolo_labels(run_dir / "train" / "labels")

    # Write aggregate manifests, dataset.yaml, class_to_idx.json.
    import pandas as pd
    outputs: Dict[str, Path] = {}
    for split_name in ("train", "val", "test"):
        dfs = [pd.read_csv(p) for p in case_manifests[split_name]]
        combined = pd.concat(dfs, ignore_index=True)
        # Verify aggregation captured every case that was generated.
        expected_cases = len(all_cases)
        actual_cases = combined["case"].nunique() if "case" in combined.columns else 0
        if actual_cases != expected_cases:
            raise RuntimeError(
                f"Aggregate {split_name}_manifest.csv covers {actual_cases} case(s) "
                f"but {expected_cases} were generated. "
                f"Missing cases: {set(c.value for c in all_cases) - set(combined['case'].unique())}"
            )
        agg_path = run_dir / f"{split_name}_manifest.csv"
        combined.to_csv(agg_path, index=False)
        outputs[split_name] = agg_path

    _write_dataset_yaml(run_dir, list(YOLO_CLASS_NAMES))
    class_to_idx = {name: i for i, name in enumerate(YOLO_CLASS_NAMES)}
    (run_dir / "class_to_idx.json").write_text(
        json.dumps(class_to_idx, indent=2), encoding="utf-8"
    )

    # Write examples folder.
    _write_examples(run_dir, examples_samples)

    # Register latest pointer.
    _write_latest_pointer(config.synthetic_dir, run_dir)
    logger.info("Latest pointer → %s", run_dir)

    return outputs


def _collect_examples(manifest_path: Path, samples: Dict[str, Path]) -> None:
    """Scan manifest GT JSONs; record one PNG per (case × completion_stage) combo."""
    import pandas as pd
    try:
        df = pd.read_csv(manifest_path)
    except Exception:
        return
    for _, row in df.iterrows():
        gt_path = Path(str(row.get("ground_truth", "")))
        img_path = Path(str(row.get("image", "")))
        if not gt_path.is_file() or not img_path.is_file():
            continue
        try:
            gt = json.loads(gt_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        case_val = gt.get("case", "unknown")
        stage = gt.get("completion_stage", "full")
        key = f"{case_val}_{stage}"
        if key not in samples:
            samples[key] = img_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic YOLO training scenes under data/generated/synthetic/<run_id>/.",
    )
    parser.add_argument("--project-root", type=Path, default=resolve_project_root(Path(__file__)))
    parser.add_argument(
        "--skip-if-complete",
        action="store_true",
        help="Skip if latest pointer already exists.",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate even if synth outputs exist.")
    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        metavar="CASE1,CASE2,...",
        help=(
            "Comma-separated list of SceneCase values to generate "
            "(e.g. multiplication-multi,subtraction,division-short). "
            "When absent all 16 cases are generated."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DataPrepConfig.from_project_root(args.project_root)
    if args.skip_if_complete and not args.force and is_synthetic_yolo_complete(config):
        print(json.dumps({"skipped": True, "reason": "synthetic_yolo_present"}, indent=2))
        return
    gen_cfg, _, _ = load_config(args.project_root)

    cases: Optional[List[SceneCase]] = None
    if args.cases is not None:
        cases = parse_cases_arg(args.cases)

    outputs = run_prepare_synthetic_yolo(config, gen_cfg, cases=cases)
    run_dir = resolve_latest_run(config.synthetic_dir)
    print(json.dumps({
        "run_dir": str(run_dir),
        "outputs": {k: str(v) for k, v in outputs.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
