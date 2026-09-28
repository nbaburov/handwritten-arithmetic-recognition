"""Workstream A: pre-process the source crop pool once and save to data/raw/pool_emnist_28_clean/.

Moved from scripts/clean_source_pool.py. Registered as ``python -m src prepare-pool-clean``.

For every PNG in data/raw/pool_emnist_28/<label>/:
  1. Load as grayscale.
  2. Threshold (ink < 200 -> 0, else 255) to eliminate grey wash.
  3. Morphological close (3x3 ellipse) to fill interior gaps.
  4. Tight bounding-box crop + re-pad to 28x28 (centred, white background).
  5. Save to data/raw/pool_emnist_28_clean/<label>/<filename>.

IMPORTANT: data/raw/pool_emnist_28/ is READ-ONLY. This module never modifies it.
Stroke widths are preserved -- no erosion/dilation that would inflate stroke
width (see track_b_thicker_strokes_failed memory note).  The close op only
fills interior gaps, not expands stroke boundaries.

Output:
  data/raw/pool_emnist_28_clean/<label>/*.png  -- normalised 28x28 crops
  data/raw/pool_emnist_28_clean/MANIFEST.json  -- timestamp + md5 tree hash of source

Usage:
  python -m src prepare-pool-clean [--project-root <path>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TILE_SIZE = 28
CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
INK_THRESHOLD = 200  # pixels < this value are treated as ink


# ---------------------------------------------------------------------------
# Per-crop processing
# ---------------------------------------------------------------------------

def process_crop(src_path: Path) -> np.ndarray | None:
    """Load, clean, and return a 28x28 uint8 grayscale tile, or None if no ink."""
    img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        log.warning("Could not read %s -- skipped", src_path)
        return None

    # Step 1: threshold to binary (0 = ink, 255 = background)
    _, binary = cv2.threshold(img, INK_THRESHOLD - 1, 255, cv2.THRESH_BINARY)
    # binary is 255 where background, 0 where ink.
    # Invert to get ink = 255 for morphological ops.
    ink = cv2.bitwise_not(binary)  # 255 = ink, 0 = background

    if not np.any(ink):
        return None  # No ink pixels -- reject

    # Step 2: morphological close to fill interior gaps (does NOT expand strokes
    # outward; MORPH_CLOSE = dilate then erode, fills holes inside strokes).
    ink_closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, CLOSE_KERNEL)

    # Step 3: tight bounding box around ink
    coords = cv2.findNonZero(ink_closed)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    ink_crop = ink_closed[y : y + h, x : x + w]

    # Step 4: re-pad to TILE_SIZE x TILE_SIZE (centred, white background)
    out = np.full((TILE_SIZE, TILE_SIZE), 0, dtype=np.uint8)  # ink=0 bg for padding
    scale = min((TILE_SIZE - 2) / max(w, 1), (TILE_SIZE - 2) / max(h, 1))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(ink_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_x = (TILE_SIZE - new_w) // 2
    pad_y = (TILE_SIZE - new_h) // 2
    out[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    # Invert back: 255 = background, 0 = ink
    return cv2.bitwise_not(out)


# ---------------------------------------------------------------------------
# Directory-level MD5 tree hash (for MANIFEST.json provenance)
# ---------------------------------------------------------------------------

def _md5_tree(root: Path) -> str:
    """Return a hex digest representing the content of all PNGs under root."""
    h = hashlib.md5()
    for p in sorted(root.rglob("*.png")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-process source crop pool.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent,
        help="Root of the handwritten-arithmetic-recognition project.",
    )
    args = parser.parse_args()

    project_root: Path = args.project_root.resolve()
    src_root = project_root / "data" / "raw" / "pool_emnist_28"
    dst_root = project_root / "data" / "raw" / "pool_emnist_28_clean"

    if not src_root.exists():
        log.error("Source pool not found: %s", src_root)
        sys.exit(1)

    log.info("Source pool : %s", src_root)
    log.info("Clean pool  : %s", dst_root)

    total_processed = 0
    total_rejected = 0
    total_saved = 0
    label_counts: dict[str, dict[str, int]] = {}

    label_dirs = sorted(d for d in src_root.iterdir() if d.is_dir())
    if not label_dirs:
        log.error("No label subdirectories found under %s", src_root)
        sys.exit(1)

    for label_dir in label_dirs:
        label = label_dir.name
        dst_label = dst_root / label
        dst_label.mkdir(parents=True, exist_ok=True)
        counts = {"processed": 0, "rejected": 0, "saved": 0}

        pngs = list(label_dir.glob("*.png"))
        if not pngs:
            log.warning("Label %s: no PNGs found", label)
            continue

        for src_path in pngs:
            counts["processed"] += 1
            cleaned = process_crop(src_path)
            if cleaned is None:
                counts["rejected"] += 1
                log.debug("Rejected (no ink): %s", src_path.name)
                continue
            dst_path = dst_label / src_path.name
            cv2.imwrite(str(dst_path), cleaned)
            counts["saved"] += 1

        label_counts[label] = counts
        total_processed += counts["processed"]
        total_rejected += counts["rejected"]
        total_saved += counts["saved"]
        log.info(
            "  %-20s  processed=%d  rejected=%d  saved=%d",
            label, counts["processed"], counts["rejected"], counts["saved"],
        )

    # Write MANIFEST.json
    log.info("Computing source tree MD5 (may take a moment)...")
    source_md5 = _md5_tree(src_root)
    manifest = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(src_root),
        "clean_dir": str(dst_root),
        "source_md5": source_md5,
        "tile_size": TILE_SIZE,
        "ink_threshold": INK_THRESHOLD,
        "totals": {
            "processed": total_processed,
            "rejected": total_rejected,
            "saved": total_saved,
        },
        "per_label": label_counts,
    }
    manifest_path = dst_root / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    log.info("")
    log.info("Done.")
    log.info("  Total processed : %d", total_processed)
    log.info("  Total rejected  : %d  (no ink)", total_rejected)
    log.info("  Total saved     : %d", total_saved)
    log.info("  Manifest        : %s", manifest_path)


if __name__ == "__main__":
    main()
