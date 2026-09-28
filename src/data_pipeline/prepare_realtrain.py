"""Build a merged fine-tune dataset from teammate-drawn real scenes + synthetic.

Registered as ``python -m src prepare-realtrain``.

Teammate-drawn TRAIN scenes are exported by the set-maker into
``data/setmaker/train/`` as a triple per sample::

    <stem>.png      preprocessed 512x512 grayscale image
    <stem>.gt.json  ground-truth shape (equation_type + symbols[fine_label, bbox,
                    yolo_class, row_index, col_index, ...])
    <stem>.txt      YOLO label lines (coarse class id + box normalised by 512)

Until now those files had ZERO consumers: both trainers hardwire
``resolve_latest_run(config.synthetic_dir)`` and read only the synthetic
``{train,val,test}_manifest.csv``. This module bridges that gap. It scans the
real triples, validates each one through the SAME loaders the trainers use, and
writes a single merged dataset under ``data/generated/finetune/<run_id>/`` that
both stages can consume:

* GNN  -- ``train_manifest.csv`` = synthetic train rows + real rows repeated
  ``real_oversample`` times; ``val_manifest.csv`` / ``test_manifest.csv`` copied
  verbatim from the synthetic run (val/test stay pure synthetic so the held-out
  measurement is unchanged).
* YOLO -- ``train/{images,labels}`` + ``val/{images,labels}`` +
  ``test/{images,labels}`` plus ``dataset.yaml``. Images are SYMLINKED (large,
  read-only) and labels are COPIED (tiny). The real train triples are added
  ``real_oversample`` times under distinct ``<stem>__rtNN`` names. Empirically,
  ultralytics' ``img2label_paths`` derives the label path from the listed image
  path string, so a symlinked image at ``<merged>/train/images/x.png`` plus a
  real label at ``<merged>/train/labels/x.txt`` is matched correctly.

The default synthetic-only training path is untouched: the trainers call into
their normal ``resolve_latest_run`` flow unless explicitly pointed at a merged
dir via ``--finetune-from-real`` / ``--data-root``.

Oversampling rationale: the real set is tiny next to ~44k synthetic train rows,
so repeating each real row ``real_oversample`` times (default 3, configurable via
``config.toml [finetune] real_oversample`` or ``--real-oversample``) raises the
gradient signal from real operators without drowning the synthetic prior. It is a
warm-start fine-tune, not a from-scratch run, so a modest multiplier is enough.

Usage::

    python -m src prepare-realtrain [--project-root <path>] [--real-oversample N]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.config import DataPrepConfig, resolve_project_root
from ..core.ontology import YOLO_CLASS_NAMES
from .prepare_synthetic_yolo import resolve_latest_run
from .validate_yolo_labels import validate_yolo_labels

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANVAS_SIZE = 512  # all preprocessed scenes are 512x512
GT_SUFFIX = ".gt.json"
TXT_SUFFIX = ".txt"
DEFAULT_REAL_OVERSAMPLE = 3
FINETUNE_DIRNAME = "finetune"  # under data/generated/


# ---------------------------------------------------------------------------
# Real-sample discovery + validation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RealSample:
    """One validated teammate-drawn train triple (paths are absolute)."""

    stem: str
    png: Path
    gt: Path
    txt: Path
    equation_type: str
    scene_case: Optional[str]
    n_symbols: int


def setmaker_train_dir(project_root: Path) -> Path:
    """Return ``<project_root>/data/setmaker/train`` (single source of truth)."""
    return project_root / "data" / "setmaker" / "train"


def finetune_root(project_root: Path) -> Path:
    """Return ``<project_root>/data/generated/finetune`` (parent of all runs)."""
    return project_root / "data" / "generated" / FINETUNE_DIRNAME


def _validate_png_512(png: Path) -> None:
    """Raise ValueError if the PNG is not a readable 512x512 image."""
    from PIL import Image  # noqa: PLC0415 (local import keeps module import cheap)

    with Image.open(png) as im:
        if im.size != (CANVAS_SIZE, CANVAS_SIZE):
            raise ValueError(
                f"{png.name}: image size {im.size} != ({CANVAS_SIZE}, {CANVAS_SIZE}). "
                "Real train PNGs must be preprocessed to 512x512 (set-maker does this)."
            )


def _validate_gt(gt: Path) -> Tuple[str, Optional[str], int]:
    """Parse + structurally validate one gt.json.

    Returns ``(equation_type, scene_case, n_symbols)``. Raises ValueError on any
    contract violation the GNN loader would later trip over.
    """
    payload: Dict[str, Any] = json.loads(gt.read_text(encoding="utf-8"))
    eq_type = payload.get("equation_type")
    if not isinstance(eq_type, str) or not eq_type:
        raise ValueError(f"{gt.name}: missing or empty 'equation_type'.")
    symbols = payload.get("symbols")
    if not isinstance(symbols, list):
        raise ValueError(f"{gt.name}: 'symbols' must be a list, got {type(symbols).__name__}.")
    for i, tok in enumerate(symbols):
        if "fine_label" not in tok:
            raise ValueError(f"{gt.name} symbol[{i}]: missing required 'fine_label'.")
        bbox = tok.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(f"{gt.name} symbol[{i}]: 'bbox' must be a 4-element list.")
    scene_case = payload.get("case")
    if scene_case is not None and not isinstance(scene_case, str):
        scene_case = None
    return eq_type, scene_case, len(symbols)


def discover_real_samples(project_root: Path) -> List[RealSample]:
    """Scan ``data/setmaker/train`` for complete, valid triples.

    A sample is included only when all three artifacts exist and every loader
    contract passes (gt.json structurally valid, PNG is 512x512, YOLO label file
    parses). Validation is fail-loud per sample: a single bad triple raises rather
    than silently shrinking the set, so the caller sees corruption immediately.
    """
    train_dir = setmaker_train_dir(project_root)
    if not train_dir.is_dir():
        return []

    samples: List[RealSample] = []
    for gt in sorted(train_dir.glob(f"*{GT_SUFFIX}")):
        stem = gt.name[: -len(GT_SUFFIX)]
        png = train_dir / f"{stem}.png"
        txt = train_dir / f"{stem}{TXT_SUFFIX}"
        if not png.is_file():
            raise FileNotFoundError(f"{gt.name}: companion PNG {png.name} missing.")
        if not txt.is_file():
            raise FileNotFoundError(f"{gt.name}: companion label {txt.name} missing.")
        _validate_png_512(png)
        eq_type, scene_case, n_sym = _validate_gt(gt)
        samples.append(
            RealSample(
                stem=stem,
                png=png.resolve(),
                gt=gt.resolve(),
                txt=txt.resolve(),
                equation_type=eq_type,
                scene_case=scene_case,
                n_symbols=n_sym,
            )
        )

    # Validate every real YOLO label in one batch (same checker the synth pipeline
    # uses post-generation). It raises RuntimeError on any coordinate/class fault.
    if samples:
        validate_yolo_labels(train_dir, n_classes=len(YOLO_CLASS_NAMES))
    return samples


# ---------------------------------------------------------------------------
# Merged GNN manifest
# ---------------------------------------------------------------------------

def _build_gnn_manifest(
    synth_root: Path,
    real_samples: List[RealSample],
    out_dir: Path,
    real_oversample: int,
) -> Dict[str, Path]:
    """Write merged train + verbatim val/test manifests; return their paths.

    The merged ``train_manifest.csv`` carries only the two columns the GNN loader
    consumes (``image``, ``ground_truth``); extra synthetic columns are dropped so
    the schema is uniform across synthetic and real rows. ``val_manifest.csv`` and
    ``test_manifest.csv`` are copied byte-for-byte from the synthetic run so the
    held-out measurement is unaffected by fine-tuning.
    """
    import pandas as pd  # noqa: PLC0415

    synth_train = pd.read_csv(synth_root / "train_manifest.csv")[["image", "ground_truth"]]
    real_rows = pd.DataFrame(
        [{"image": str(s.png), "ground_truth": str(s.gt)} for s in real_samples]
    )
    if not real_rows.empty and real_oversample > 1:
        real_rows = pd.concat([real_rows] * real_oversample, ignore_index=True)

    merged_train = pd.concat([synth_train, real_rows], ignore_index=True)
    train_path = out_dir / "train_manifest.csv"
    merged_train.to_csv(train_path, index=False)

    # val/test verbatim (pure synthetic)
    val_path = out_dir / "val_manifest.csv"
    test_path = out_dir / "test_manifest.csv"
    shutil.copy2(synth_root / "val_manifest.csv", val_path)
    shutil.copy2(synth_root / "test_manifest.csv", test_path)

    return {"train": train_path, "val": val_path, "test": test_path}


# ---------------------------------------------------------------------------
# Merged YOLO dataset dir
# ---------------------------------------------------------------------------

def _symlink_or_copy_image(src: Path, dst: Path) -> None:
    """Symlink ``src`` (absolute) to ``dst``; fall back to copy if symlinks fail."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _link_split_from_synth(synth_root: Path, out_dir: Path, split: str) -> int:
    """Mirror one synthetic split into the merged dir (symlink imgs, copy labels).

    Returns the number of image/label pairs linked. Labels are copied (not
    symlinked) so ultralytics finds them at the merged path regardless of how it
    stringifies the symlinked image path.
    """
    src_imgs = synth_root / split / "images"
    src_lbls = synth_root / split / "labels"
    dst_imgs = out_dir / split / "images"
    dst_lbls = out_dir / split / "labels"
    dst_imgs.mkdir(parents=True, exist_ok=True)
    dst_lbls.mkdir(parents=True, exist_ok=True)

    count = 0
    for img in sorted(src_imgs.glob("*.png")):
        lbl = src_lbls / f"{img.stem}{TXT_SUFFIX}"
        if not lbl.is_file():
            # synthetic generator always writes a label (possibly empty); skip
            # defensively rather than fail the whole merge on one stray image.
            log.warning("synthetic image %s has no label; skipped", img.name)
            continue
        _symlink_or_copy_image(img.resolve(), dst_imgs / img.name)
        shutil.copy2(lbl, dst_lbls / lbl.name)
        count += 1
    return count


def _add_real_to_yolo_train(
    out_dir: Path, real_samples: List[RealSample], real_oversample: int
) -> int:
    """Add real train triples into the merged YOLO train split, oversampled.

    Each real sample is added ``real_oversample`` times under distinct
    ``<stem>__rtNN`` names so the loader treats them as separate items. Returns the
    total number of (real x oversample) image/label pairs written.
    """
    dst_imgs = out_dir / "train" / "images"
    dst_lbls = out_dir / "train" / "labels"
    dst_imgs.mkdir(parents=True, exist_ok=True)
    dst_lbls.mkdir(parents=True, exist_ok=True)

    count = 0
    reps = max(1, real_oversample)
    for s in real_samples:
        for k in range(reps):
            name = f"{s.stem}__rt{k:02d}"
            _symlink_or_copy_image(s.png, dst_imgs / f"{name}.png")
            shutil.copy2(s.txt, dst_lbls / f"{name}{TXT_SUFFIX}")
            count += 1
    return count


def _write_dataset_yaml(out_dir: Path) -> Path:
    """Write the merged dataset.yaml (absolute ``path`` for this machine)."""
    names = list(YOLO_CLASS_NAMES)
    content = "\n".join([
        f"path: {out_dir}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        f"nc: {len(names)}",
        "names: [" + ", ".join(f"'{n}'" for n in names) + "]",
    ])
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def _build_yolo_dataset(
    synth_root: Path,
    real_samples: List[RealSample],
    out_dir: Path,
    real_oversample: int,
) -> Dict[str, Any]:
    """Build the merged YOLO dataset dir; return a counts + paths summary."""
    n_train = _link_split_from_synth(synth_root, out_dir, "train")
    n_val = _link_split_from_synth(synth_root, out_dir, "val")
    n_test = _link_split_from_synth(synth_root, out_dir, "test")
    n_real = _add_real_to_yolo_train(out_dir, real_samples, real_oversample)
    yaml_path = _write_dataset_yaml(out_dir)

    # Re-validate the merged train labels as a final integrity gate.
    validate_yolo_labels(out_dir / "train" / "labels", n_classes=len(YOLO_CLASS_NAMES))

    return {
        "dataset_yaml": yaml_path,
        "synth_train_pairs": n_train,
        "synth_val_pairs": n_val,
        "synth_test_pairs": n_test,
        "real_train_pairs": n_real,
    }


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def build_finetune_dataset(
    project_root: Path,
    *,
    real_oversample: int = DEFAULT_REAL_OVERSAMPLE,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Scan real train triples + synthetic latest, write a merged fine-tune dataset.

    Args:
        project_root: Project root containing ``data/``.
        real_oversample: How many times each real row/triple is repeated in the
            merged train split (GNN manifest + YOLO train dir). Must be >= 1.
        out_dir: Optional explicit output dir; defaults to
            ``data/generated/finetune/<run_id>/``.

    Returns:
        A summary dict with ``run_dir``, the GNN manifest paths
        (``gnn_train_manifest`` / ``gnn_val_manifest`` / ``gnn_test_manifest``),
        the YOLO ``dataset_yaml`` path, the real-sample count, and per-split pair
        counts. Also written to ``<run_dir>/finetune_manifest.json`` for provenance.

    Raises:
        RuntimeError: If no synthetic dataset exists, or no valid real samples are
            found (fine-tuning with zero real data is a no-op the caller must fix).
        ValueError / FileNotFoundError: On any invalid real triple (fail-loud).
    """
    if real_oversample < 1:
        raise ValueError(f"real_oversample must be >= 1, got {real_oversample}")

    config = DataPrepConfig.from_project_root(project_root)
    synth_root = resolve_latest_run(config.synthetic_dir)
    if synth_root is None:
        raise RuntimeError(
            "No synthetic dataset found (data/generated/synthetic/latest missing). "
            "Run: python -m src generate --project-root <project>"
        )
    for split in ("train", "val", "test"):
        if not (synth_root / f"{split}_manifest.csv").exists():
            raise RuntimeError(
                f"Synthetic run {synth_root.name} missing {split}_manifest.csv. "
                "Regenerate with: python -m src generate."
            )

    real_samples = discover_real_samples(project_root)
    if not real_samples:
        raise RuntimeError(
            f"No valid real train samples in {setmaker_train_dir(project_root)}. "
            "Draw + export scenes in the set-maker (mode=train) first."
        )

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if out_dir is None:
        out_dir = finetune_root(project_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    gnn_paths = _build_gnn_manifest(synth_root, real_samples, out_dir, real_oversample)
    yolo_summary = _build_yolo_dataset(synth_root, real_samples, out_dir, real_oversample)

    # Per-equation-kind tally of the real contribution (gap-targeting visibility).
    kind_counts: Dict[str, int] = {}
    case_counts: Dict[str, int] = {}
    for s in real_samples:
        kind_counts[s.equation_type] = kind_counts.get(s.equation_type, 0) + 1
        if s.scene_case:
            case_counts[s.scene_case] = case_counts.get(s.scene_case, 0) + 1

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(out_dir),
        "synthetic_source": str(synth_root),
        "real_oversample": real_oversample,
        "real_sample_count": len(real_samples),
        "real_by_equation_kind": kind_counts,
        "real_by_scene_case": case_counts,
        "gnn_train_manifest": str(gnn_paths["train"]),
        "gnn_val_manifest": str(gnn_paths["val"]),
        "gnn_test_manifest": str(gnn_paths["test"]),
        "dataset_yaml": str(yolo_summary["dataset_yaml"]),
        "yolo_synth_train_pairs": yolo_summary["synth_train_pairs"],
        "yolo_synth_val_pairs": yolo_summary["synth_val_pairs"],
        "yolo_synth_test_pairs": yolo_summary["synth_test_pairs"],
        "yolo_real_train_pairs": yolo_summary["real_train_pairs"],
    }
    (out_dir / "finetune_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log.info(
        "Merged fine-tune dataset at %s (real=%d x%d, synth_train=%d)",
        out_dir, len(real_samples), real_oversample, yolo_summary["synth_train_pairs"],
    )
    return summary


def _resolve_real_oversample(project_root: Path, cli_value: Optional[int]) -> int:
    """CLI value wins; else config.toml [finetune] real_oversample; else default."""
    if cli_value is not None:
        return cli_value
    import tomllib  # noqa: PLC0415 (stdlib in 3.11+)

    toml_path = project_root / "config.toml"
    if toml_path.exists():
        with toml_path.open("rb") as f:
            raw = tomllib.load(f)
        ft = raw.get("finetune", {})
        return int(ft.get("real_oversample", DEFAULT_REAL_OVERSAMPLE))
    return DEFAULT_REAL_OVERSAMPLE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a merged real+synthetic fine-tune dataset for GNN/YOLO."
    )
    parser.add_argument(
        "--project-root", type=Path, default=resolve_project_root(Path(__file__))
    )
    parser.add_argument(
        "--real-oversample",
        type=int,
        default=None,
        help=(
            "Times each real train row/triple is repeated in the merged train split "
            f"(default: config.toml [finetune] real_oversample or {DEFAULT_REAL_OVERSAMPLE})."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    oversample = _resolve_real_oversample(args.project_root, args.real_oversample)
    summary = build_finetune_dataset(args.project_root, real_oversample=oversample)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
