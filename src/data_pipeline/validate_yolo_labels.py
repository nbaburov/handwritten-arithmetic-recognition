from __future__ import annotations

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def validate_yolo_labels(labels_dir: Path, n_classes: int = 6) -> List[str]:
    """Validate all YOLO label files under labels_dir.

    Reads every .txt file, parses each line as ``class_id cx cy w h``,
    and collects violations. Raises RuntimeError if any violations found.
    Logs a warning for empty label files (valid — some completion stages
    produce scenes with no visible tokens).

    Args:
        labels_dir: Directory containing .txt YOLO label files.
        n_classes: Number of valid class IDs (0..n_classes-1).

    Returns:
        Empty list if all labels are valid.

    Raises:
        RuntimeError: If any coordinate or class violations are found.
        OSError: If labels_dir does not exist or cannot be read.
    """
    violations: List[str] = []

    label_files = sorted(labels_dir.glob("*.txt"))
    for label_file in label_files:
        lines = label_file.read_text(encoding="utf-8").splitlines()
        lines = [ln.strip() for ln in lines if ln.strip()]

        if not lines:
            logger.warning("Empty label file (no annotations): %s", label_file.name)
            continue

        for line_no, line in enumerate(lines, start=1):
            parts = line.split()
            if len(parts) != 5:
                violations.append(
                    f"{label_file.name}:{line_no}: expected 5 fields, got {len(parts)}: {line!r}"
                )
                continue
            try:
                class_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                violations.append(
                    f"{label_file.name}:{line_no}: cannot parse fields: {line!r}"
                )
                continue

            if class_id < 0 or class_id >= n_classes:
                violations.append(
                    f"{label_file.name}:{line_no}: class_id={class_id} outside [0, {n_classes - 1}]"
                )
            if not (0.0 <= cx <= 1.0):
                violations.append(
                    f"{label_file.name}:{line_no}: cx={cx} outside [0, 1]"
                )
            if not (0.0 <= cy <= 1.0):
                violations.append(
                    f"{label_file.name}:{line_no}: cy={cy} outside [0, 1]"
                )
            if w <= 0 or w > 1.0:
                violations.append(
                    f"{label_file.name}:{line_no}: w={w} not in (0, 1]"
                )
            if h <= 0 or h > 1.0:
                violations.append(
                    f"{label_file.name}:{line_no}: h={h} not in (0, 1]"
                )

    if violations:
        raise RuntimeError(
            f"YOLO label validation failed with {len(violations)} violation(s):\n"
            + "\n".join(violations)
        )

    return []
