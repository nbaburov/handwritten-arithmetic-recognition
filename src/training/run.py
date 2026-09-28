from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict

from ..core.artifact_paths import (
    append_runs_index,
    new_training_run_id,
    yolo_runs_root,
    write_active_yolo,
    try_git_revision,
)
from ..core.config import DataPrepConfig, resolve_project_root
from ..core.logging_setup import configure_run_logging
from ..core.ontology import YOLO_CLASS_NAMES
from ..core.run_config import YoloConfig, load_config


def _load_class_map(path: Path) -> Dict[str, int]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_per_class_ap(val_result: Any) -> Dict[str, Any]:
    """Extract per-class AP50 and AP50-95 from an ultralytics val result object.

    Parameters
    ----------
    val_result:
        Object returned by ``YOLO.val()``. Expected to have a ``box``
        attribute of type ``ultralytics.utils.metrics.Metric`` with
        ``ap_class_index`` (list of int), ``ap50`` (ndarray), and ``ap``
        (ndarray) attributes.

    Returns
    -------
    dict mapping coarse class name to ``{"ap50": float, "ap50_95": float}``.
    Returns ``{}`` and logs a warning when arrays are empty or extraction fails.

    Notes
    -----
    Verified attribute names on ultralytics 8.4.54:
    - ``box.ap_class_index`` — list[int], shape (nc,)
    - ``box.ap50`` — ndarray | list, shape (nc,), from all_ap[:, 0]
    - ``box.ap`` — ndarray | list, shape (nc,), from all_ap.mean(1)
    """
    import logging as _log
    _logger = _log.getLogger(__name__)

    try:
        box = getattr(val_result, "box", None)
        if box is None:
            _logger.warning("Per-class AP: val_result has no 'box' attribute; writing {}")
            return {}

        ap_class_index = getattr(box, "ap_class_index", None)
        ap50_arr = getattr(box, "ap50", None)
        ap_arr = getattr(box, "ap", None)

        if (
            ap_class_index is None
            or ap50_arr is None
            or ap_arr is None
            or len(ap_class_index) == 0
        ):
            _logger.warning("Per-class AP: arrays empty or missing; writing {}")
            return {}

        result: Dict[str, Any] = {}
        for i, cls_id in enumerate(ap_class_index):
            name = YOLO_CLASS_NAMES[cls_id] if 0 <= cls_id < len(YOLO_CLASS_NAMES) else f"class_{cls_id}"
            try:
                ap50_val = float(ap50_arr[i])
                ap_val = float(ap_arr[i])
            except (IndexError, TypeError, ValueError) as exc:
                _logger.warning("Per-class AP: index %d extraction failed: %s", i, exc)
                continue
            result[name] = {"ap50": ap50_val, "ap50_95": ap_val}
        return result
    except Exception as exc:
        _logger.warning("Per-class AP extraction failed unexpectedly: %s", exc)
        return {}


def train_yolo(
    config: DataPrepConfig,
    yolo_cfg: YoloConfig,
    data_root_override: Any = None,
) -> Dict[str, Any]:
    """Train a YOLOv8 model for Stage 1 bounding-box detection.

    Parameters
    ----------
    config:
        DataPrepConfig carrying dataset and artifact paths.
    yolo_cfg:
        YoloConfig with all training hyperparameters.

    Returns
    -------
    dict with keys: run_id, run_dir, report, best_pt, run_manifest, deploy_best.
    """
    if data_root_override is not None:
        # Fine-tune path: a pre-built merged real+synthetic YOLO dir
        # (src/data_pipeline/prepare_realtrain.py) with the same layout as a
        # synthetic run -- train/val/test images+labels + dataset.yaml.
        synth_root = Path(data_root_override)
        if not synth_root.is_dir():
            raise FileNotFoundError(
                f"--finetune-from-real dir not found: {synth_root}. "
                "Build it with: python -m src prepare-realtrain."
            )
    else:
        from ..data_pipeline.prepare_synthetic_yolo import resolve_latest_run  # noqa: PLC0415
        _latest = resolve_latest_run(config.synthetic_dir)
        if _latest is None:
            raise FileNotFoundError(
                "No synthetic dataset found (data/generated/synthetic/latest missing). "
                "Run: python -m src generate --project-root <project>"
            )
        synth_root = _latest
    train_manifest = synth_root / "train_manifest.csv"
    if not train_manifest.exists():
        raise FileNotFoundError(
            f"Missing YOLO dataset manifest ({train_manifest}). "
            "Run: python -m src generate (or prepare-realtrain for fine-tune)."
        )

    # MPS TAL loss crashes on PyTorch 2.11.0; CPU is stable on M4 Pro.
    # YoloConfig.device defaults to "cpu" for this reason. Override only if
    # a non-cpu device is explicitly requested in config or via CLI.
    dev = yolo_cfg.device if yolo_cfg.device != "cpu" else "cpu"

    # Delete stale YOLO label cache files before every run so tensor shapes are
    # always consistent with the current dataset layout on disk.
    for cache_file in synth_root.rglob("labels.cache"):
        cache_file.unlink()

    deploy_dir = config.artifacts_yolo_dir
    deploy_dir.mkdir(parents=True, exist_ok=True)
    run_id = new_training_run_id()
    run_dir = yolo_runs_root(config) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = configure_run_logging(run_dir / "run.log")
    logger.info("Starting YOLO training run %s", run_id)
    logger.info("Config: %s", yolo_cfg)

    report_path = run_dir / "eval.md"

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception:
        fallback = {
            "backend": "fallback",
            "reason": "ultralytics unavailable",
            "trained": False,
            "next_step": "pip install ultralytics torch and rerun yolo training",
        }
        out = run_dir / "fallback_detector.json"
        out.write_text(json.dumps(fallback, indent=2), encoding="utf-8")
        report_path.write_text(
            "# Stage 1 Evaluation\n\n- backend: fallback\n- trained: false\n- mAP50: 0.0\n",
            encoding="utf-8",
        )
        return {"yolo_fallback": out, "report": report_path, "run_dir": run_dir}

    # New flat layout: dataset.yaml is pre-written by generate into the run folder.
    # Use it directly; re-write only to ensure path is absolute for this machine.
    dataset_yaml = synth_root / "dataset.yaml"
    names = list(YOLO_CLASS_NAMES)
    if not dataset_yaml.exists() or True:  # always refresh path so relocates work
        dataset_yaml_content = "\n".join([
            f"path: {str(synth_root)}",
            "train: train/images",
            "val: val/images",
            "test: test/images",
            f"nc: {len(names)}",
            "names: [" + ", ".join(f"\'{n}\'" for n in names) + "]",
        ])
        dataset_yaml.write_text(dataset_yaml_content, encoding="utf-8")
    logger.info("Dataset YAML: %s", dataset_yaml)

    yolo_name = "yolo"
    local_pretrained = config.artifacts_root / "pretrained" / yolo_cfg.model
    yolo_weights = str(local_pretrained) if local_pretrained.is_file() else yolo_cfg.model

    model = YOLO(yolo_weights)

    def _flush_mps(trainer: Any) -> None:
        """Flush MPS allocator cache between epochs to prevent buffer fragmentation crash."""
        try:
            import torch
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass

    model.add_callback("on_train_epoch_end", _flush_mps)

    result = model.train(
        data=str(dataset_yaml),
        epochs=yolo_cfg.epochs,
        imgsz=yolo_cfg.image_size,
        project=str(run_dir),
        name=yolo_name,
        batch=yolo_cfg.batch,
        workers=yolo_cfg.workers,
        device=dev,
        cache=yolo_cfg.cache,
        patience=yolo_cfg.patience,
        rect=yolo_cfg.rect,
        optimizer=yolo_cfg.optimizer,
        lr0=yolo_cfg.lr0,
        lrf=yolo_cfg.lrf,
        momentum=yolo_cfg.momentum,
        weight_decay=yolo_cfg.weight_decay,
        warmup_epochs=yolo_cfg.warmup_epochs,
        box=yolo_cfg.box,
        cls=yolo_cfg.cls,
        dfl=yolo_cfg.dfl,
        amp=yolo_cfg.amp,
        **yolo_cfg.augmentation,
    )
    logger.info("YOLO training complete")

    metrics = getattr(result, "results_dict", {})
    report_lines = ["# Stage 1 Evaluation", "", "- backend: ultralytics"]
    map50_v: Any = 0.0
    map5095_v: Any = 0.0
    val_per_class: Dict[str, Any] = {}
    if metrics:
        map50_v = metrics.get("metrics/mAP50(B)", 0.0)
        map5095_v = metrics.get("metrics/mAP50-95(B)", 0.0)
        report_lines.append(f"- mAP50: {map50_v}")
        report_lines.append(f"- mAP50_95: {map5095_v}")
    # Extract per-class AP from the val validator embedded in the training result.
    # The training result's validator holds the final-epoch val metrics.
    try:
        val_validator = getattr(result, "validator", None)
        val_result_for_cls = getattr(val_validator, "metrics", None) if val_validator else None
        if val_result_for_cls is not None:
            val_per_class = _extract_per_class_ap(val_result_for_cls)
        else:
            logger.warning("Per-class AP (val): trainer.validator.metrics not available; writing {}")
    except Exception as _exc:
        logger.warning("Per-class AP (val) extraction failed: %s", _exc)

    # Test-set evaluation pass
    test_metrics: Dict[str, Any] = {"mAP50": 0.0, "mAP50_95": 0.0}
    test_per_class: Dict[str, Any] = {}
    try:
        test_result = model.val(data=str(dataset_yaml), split="test")
        test_dict = getattr(test_result, "results_dict", {})
        if test_dict:
            test_metrics = {
                "mAP50": test_dict.get("metrics/mAP50(B)", 0.0),
                "mAP50_95": test_dict.get("metrics/mAP50-95(B)", 0.0),
            }
        test_per_class = _extract_per_class_ap(test_result)
        logger.info("Test metrics: %s", test_metrics)
        logger.info("Test per-class AP: %s", test_per_class)
    except Exception as exc:
        logger.warning("Test-set eval failed: %s", exc)

    # Append test metrics and per-class AP table to eval.md
    report_lines.append(f"- test_mAP50: {test_metrics.get('mAP50', 0.0)}")
    report_lines.append(f"- test_mAP50_95: {test_metrics.get('mAP50_95', 0.0)}")
    if val_per_class or test_per_class:
        report_lines.append("")
        report_lines.append("## Per-Class AP")
        report_lines.append("")
        report_lines.append("| class | val_ap50 | val_ap50_95 | test_ap50 | test_ap50_95 |")
        report_lines.append("|---|---|---|---|---|")
        all_classes = sorted(set(list(val_per_class.keys()) + list(test_per_class.keys())))
        for cls_name in all_classes:
            v = val_per_class.get(cls_name, {})
            t = test_per_class.get(cls_name, {})
            v50 = f"{v.get('ap50', 0.0):.4f}" if v else "n/a"
            v5095 = f"{v.get('ap50_95', 0.0):.4f}" if v else "n/a"
            t50 = f"{t.get('ap50', 0.0):.4f}" if t else "n/a"
            t5095 = f"{t.get('ap50_95', 0.0):.4f}" if t else "n/a"
            report_lines.append(f"| {cls_name} | {v50} | {v5095} | {t50} | {t5095} |")
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    trained = list(run_dir.rglob("best.pt"))
    best_src = max(trained, key=lambda p: p.stat().st_mtime) if trained else None
    stable_pt = run_dir / "best.pt"
    if best_src is not None and best_src.resolve() != stable_pt.resolve():
        shutil.copy2(best_src, stable_pt)
        best_pt_path = stable_pt
    elif best_src is not None:
        best_pt_path = best_src
    else:
        best_pt_path = stable_pt

    rel_run = run_dir.relative_to(deploy_dir).as_posix()
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "stage": "yolo",
        "run_type": "scratch",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_revision": try_git_revision(config.project_root),
        "hyperparams": {
            "model": yolo_cfg.model,
            "epochs": yolo_cfg.epochs,
            "image_size": yolo_cfg.image_size,
            "batch": yolo_cfg.batch,
            "workers": yolo_cfg.workers,
            "device": dev,
            "cache": yolo_cfg.cache,
            "patience": yolo_cfg.patience,
            "rect": yolo_cfg.rect,
        },
        "augmentation": yolo_cfg.augmentation,
        # val metrics come from the training run's final validation pass;
        # test metrics come from the held-out test split evaluated after training.
        # per_class maps coarse class name to {"ap50": float, "ap50_95": float}.
        "metrics": {
            "val": {"mAP50": map50_v, "mAP50_95": map5095_v, "per_class": val_per_class},
            "test": {**test_metrics, "per_class": test_per_class},
        },
        "ultralytics_weights": str(best_src) if best_src else None,
        "best_pt": str(best_pt_path) if best_pt_path.is_file() else None,
    }
    manifest_path = run_dir / "run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("run.json written to %s", manifest_path)

    if best_pt_path.is_file():
        shutil.copy2(best_pt_path, deploy_dir / "best.pt")
        best_rel = f"{rel_run}/{best_pt_path.name}"
        write_active_yolo(
            config,
            run_id,
            best_pt_relative=best_rel,
            run_manifest_relative=f"{rel_run}/run.json",
            extra={"metrics": manifest["metrics"], "run_type": manifest["run_type"]},
        )
        append_runs_index(
            yolo_runs_root(config),
            {
                "run_id": run_id,
                "created_utc": manifest["created_utc"],
                "mAP50": map50_v,
                "run_dir": rel_run,
            },
        )

    logger.info("YOLO run %s complete; best_pt=%s", run_id, best_pt_path)

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "report": report_path,
        "best_pt": best_pt_path if best_pt_path.is_file() else None,
        "run_manifest": manifest_path,
        "deploy_best": deploy_dir / "best.pt" if (deploy_dir / "best.pt").is_file() else None,
    }


def train_stage1(
    config: DataPrepConfig,
    epochs: int = 50,
    image_size: int = 512,
    batch: int = 16,
    workers: int = 0,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Convenience wrapper used by tests and scripts.

    Builds a YoloConfig from individual kwargs and delegates to train_yolo.
    """
    yolo_cfg = YoloConfig(
        epochs=epochs,
        image_size=image_size,
        batch=batch,
        workers=workers,
        device=device,
    )
    return train_yolo(config, yolo_cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train yolo/gnn models for equation recognition.")
    parser.add_argument("--project-root", type=Path, default=resolve_project_root(Path(__file__)))
    parser.add_argument("--stage", type=str, required=True, choices=["yolo", "gnn"])
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Max training epochs (overrides config.toml [yolo] or [gnn] epochs).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="YOLO batch size (overrides config.toml [yolo] batch).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="YOLO input image size (overrides config.toml [yolo] image_size).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="YOLO device: cpu, mps, 0 (overrides config.toml [yolo] device).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="YOLO weights to start from (e.g. yolov8n.pt, yolov8s.pt, yolov8m.pt).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Dataloader workers (overrides config.toml [yolo] workers).",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        choices=["ram", "disk", "false"],
        help="Cache mode: disk (default, MPS-safe), ram (fastest), false (safest).",
    )
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        dest="init_from",
        help=(
            "GNN only: run_id of a previously trained GNN run to warm-start from "
            "(resolves to artifacts/gnn/runs/<run_id>/best.pt). "
            "Useful for fine-tuning on new data; combine with --lr at ~1/10 of the "
            "original learning rate (e.g. --lr 0.0001)."
        ),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=(
            "GNN only: override the learning rate from config.toml [gnn] lr. "
            "Useful for warm-start fine-tunes (typical: original_lr / 10)."
        ),
    )
    parser.add_argument(
        "--finetune-from-real",
        type=Path,
        default=None,
        dest="finetune_from_real",
        help=(
            "Path to a merged real+synthetic dataset dir built by "
            "`python -m src prepare-realtrain` (e.g. data/generated/finetune/<ts>). "
            "When set, training consumes that dir instead of the synthetic latest "
            "pointer. Pair with warm-start: GNN --init-from <active_run_id>, "
            "YOLO --model <active best.pt>. Omit for the default synthetic-only run."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        dest="data_root",
        help="Alias for --finetune-from-real (explicit merged dataset dir).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, yolo_cfg, gnn_cfg = load_config(args.project_root)
    config = DataPrepConfig.from_project_root(args.project_root)
    start = time.perf_counter()

    # --finetune-from-real and --data-root are aliases; either selects the merged
    # real+synthetic dataset dir. Validate existence up front so a typo fails fast.
    data_root_override: Path | None = args.finetune_from_real or args.data_root
    if data_root_override is not None and not Path(data_root_override).is_dir():
        raise FileNotFoundError(
            f"--finetune-from-real/--data-root not a directory: {data_root_override}. "
            "Build it with: python -m src prepare-realtrain."
        )

    if args.stage == "gnn":
        from .train_gnn import train_gnn_model
        # Apply CLI overrides to GNN config if provided.
        gnn_overrides: Dict[str, Any] = {}
        if args.epochs is not None:
            gnn_overrides["epochs"] = args.epochs
        if args.lr is not None:
            gnn_overrides["lr"] = args.lr
        if gnn_overrides:
            gnn_cfg = dataclasses.replace(gnn_cfg, **gnn_overrides)

        # Resolve --init-from run_id to the checkpoint path.
        init_weights_path: Path | None = None
        if args.init_from is not None:
            candidate = config.artifacts_gnn_dir / "runs" / args.init_from / "best.pt"
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"--init-from run '{args.init_from}' has no best.pt at "
                    f"{candidate}. Check run_id or train GNN first."
                )
            init_weights_path = candidate

        outputs = train_gnn_model(
            config, gnn_cfg, init_weights_path=init_weights_path,
            device=args.device, data_root_override=data_root_override,
        )
    else:
        # Build override kwargs for only the args that were explicitly passed.
        overrides: Dict[str, Any] = {}
        if args.epochs is not None:
            overrides["epochs"] = args.epochs
        if args.batch is not None:
            overrides["batch"] = args.batch
        if args.image_size is not None:
            overrides["image_size"] = args.image_size
        if args.device is not None:
            overrides["device"] = args.device
        if args.model is not None:
            overrides["model"] = args.model
        if args.workers is not None:
            overrides["workers"] = args.workers
        if args.cache is not None:
            overrides["cache"] = False if args.cache == "false" else args.cache

        if overrides:
            yolo_cfg = dataclasses.replace(yolo_cfg, **overrides)

        outputs = train_yolo(config, yolo_cfg, data_root_override=data_root_override)

    elapsed = time.perf_counter() - start
    out_json = {k: (str(v) if v is not None else None) for k, v in outputs.items()}
    print(json.dumps({"stage": args.stage, "elapsed_seconds": round(elapsed, 2), "outputs": out_json}, indent=2))


if __name__ == "__main__":
    main()
