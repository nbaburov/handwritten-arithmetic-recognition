from __future__ import annotations

import importlib
import sys
from typing import Dict

_COMMAND_MODULES: Dict[str, str] = {
    "validate":           "src.data_pipeline.validate_dataset",
    "generate":           "src.data_pipeline.prepare_synthetic_yolo",
    "train":              "src.training.run",
    "infer":              "src.inference.run",
    "gui":                "src.inference.gui",
    "setmaker":           "src.setmaker.app",
    "demo":               "src.demo.app",
    "eval":               "src.eval.run",
    "render-examples":    "src.generation.render_examples",
    "prepare-pool-clean": "src.data_pipeline.prepare_pool_clean",
    "prepare-pool-hires": "src.data_pipeline.prepare_pool_hires",
    "prepare-realtrain":  "src.data_pipeline.prepare_realtrain",
}


def _usage(stream) -> None:
    stream.write(
        "usage: python -m src COMMAND [ARGS...]\n\n"
        "Run from the project root (folder containing data/, src/, reports/) or set PYTHONPATH accordingly.\n\n"
        "Commands:\n"
    )
    for name in sorted(_COMMAND_MODULES):
        stream.write(f"  {name}\n")


def main() -> None:
    argv = sys.argv[1:]
    if not argv:
        _usage(sys.stderr)
        sys.exit(2)
    if argv[0] in ("-h", "--help"):
        _usage(sys.stdout)
        return
    cmd, *rest = argv
    mod_name = _COMMAND_MODULES.get(cmd)
    if not mod_name:
        sys.stderr.write(f"unknown command: {cmd!r}\n\n")
        _usage(sys.stderr)
        sys.exit(2)
    old = sys.argv
    try:
        sys.argv = [f"src {cmd}"] + rest
        importlib.import_module(mod_name).main()
    finally:
        sys.argv = old
