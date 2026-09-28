"""CLI entry point for the iter6 evaluation harness.

Moved from scripts/eval_harness.py. Registered as ``python -m src eval``.

Usage
-----
PYTHONPATH="$PP" .venv/bin/python -m src eval --project-root "$PP"
PYTHONPATH="$PP" .venv/bin/python -m src eval --project-root "$PP" --config-id baseline
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.eval.harness import (
    load_baseline_record,
    render_markdown_summary,
    run_eval,
    write_jsonl_record,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the evaluation harness over data/eval/bank/."
    )
    parser.add_argument(
        "--project-root",
        required=True,
        type=Path,
        help="Absolute path to the project root (contains config.toml).",
    )
    parser.add_argument(
        "--config-id",
        default="baseline",
        help="Identifier tag stored with the run record (default: baseline).",
    )
    parser.add_argument(
        "--samples-dir",
        default=None,
        type=Path,
        help="Override the eval samples directory (default: data/eval/bank).",
    )
    parser.add_argument(
        "--yolo-run-id",
        default=None,
        dest="yolo_run_id",
        help="Eval a specific YOLO run (artifacts/yolo/runs/<id>/best.pt) instead of the active "
             "pointer. Use to compare a freshly-trained run vs baseline without touching active.json.",
    )
    parser.add_argument(
        "--gnn-run-id",
        default=None,
        dest="gnn_run_id",
        help="Eval a specific GNN run (artifacts/gnn/runs/<id>/best.pt) instead of the active pointer.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    args = _parse_args(argv)
    project_root: Path = args.project_root.resolve()
    config_id: str = args.config_id

    try:
        samples_dir = args.samples_dir.resolve() if args.samples_dir else None
        record = run_eval(
            project_root,
            config_id,
            samples_dir=samples_dir,
            yolo_run_id=args.yolo_run_id,
            gnn_run_id=args.gnn_run_id,
        )

        from src.core.config import DataPrepConfig
        cfg = DataPrepConfig.from_project_root(project_root)
        eval_dir = cfg.reports_eval_dir
        eval_dir.mkdir(parents=True, exist_ok=True)

        history_path = eval_dir / "history.jsonl"
        summary_path = eval_dir / "latest.md"

        baseline = load_baseline_record(history_path, config_id)
        write_jsonl_record(record, history_path)
        render_markdown_summary(record, baseline, summary_path)

        agg = record.aggregate
        n = agg.n_samples
        eq_acc = agg.equation_kind_acc if agg.equation_kind_acc is not None else float("nan")
        ood = agg.ood_rate if agg.ood_rate is not None else float("nan")
        wall_ms = int(agg.total_wall_ms)
        print(
            f"Eval {config_id}: {n} samples, "
            f"equation_kind_acc={eq_acc:.3f}, "
            f"ood_rate={ood:.3f}, "
            f"total_wall_ms={wall_ms}"
        )
        return 0

    except Exception:
        logging.exception("harness failed")
        raise


if __name__ == "__main__":
    sys.exit(main())
