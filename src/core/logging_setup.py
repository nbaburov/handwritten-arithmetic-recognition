"""Per-run file logging setup. Call once per operation (generate/train) at startup."""
from __future__ import annotations

import logging
from pathlib import Path

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s — %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_run_logging(log_path: Path) -> logging.Logger:
    """Attach a FileHandler to the root logger for this run.

    Creates parent directories if they do not exist. Never raises — logging
    failures are swallowed so they cannot crash a training run.

    Args:
        log_path: Full path for the .log file (e.g. artifacts/yolo/runs/abc/run.log).

    Returns:
        The root logger with the FileHandler attached.
    """
    root = logging.getLogger()
    if root.level == logging.WARNING or root.level == logging.NOTSET:
        root.setLevel(logging.INFO)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)
        )
        root.addHandler(handler)
        stream = logging.StreamHandler()
        stream.setLevel(logging.INFO)
        stream.setFormatter(
            logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)
        )
        root.addHandler(stream)
    except Exception as exc:
        logging.warning("configure_run_logging: could not attach file handler: %s", exc)
    return root
