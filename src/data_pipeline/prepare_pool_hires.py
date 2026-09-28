"""Workstream C: upsample cleaned source crops 28x28 -> 112x112.

Moved from scripts/supersample_source_pool.py. Registered as ``python -m src prepare-pool-hires``.

Input:  data/raw/pool_emnist_28_clean/<label>/*.png   (produced by prepare-pool-clean)
Output: data/raw/pool_emnist_28_hires/<label>/*.png   (112x112 uint8 grayscale)

Super-resolution method
-----------------------
Real-ESRGAN was evaluated but requires torch + custom CUDA kernels that
conflict with this project's pinned torch+ultralytics versions.  Attempting
to install basicsr/realesrgan into the project venv risks breaking YOLO
training.  Decision: fall back to LANCZOS4 upsampling + unsharp-mask
sharpening, which consistently outperforms naive BICUBIC for thin handwriting
strokes (Lanczos preserves high-frequency edge information; unsharp-mask
recovers apparent stroke crispness lost in 4x resize).

Method: cv2.resize(INTER_LANCZOS4) + Gaussian unsharp-mask (kernel 3x3,
strength 1.5).  Output is clamped to [0, 255] uint8.

This approach is:
  - Deterministic (no model weights needed)
  - Fully CPU-bound (no GPU dependency)
  - Compatible with the project venv
  - Produces visibly sharper glyphs than 28->cell BICUBIC in the generator

Usage:
  python -m src prepare-pool-hires [--project-root <path>]
                                    [--scale 4]
                                    [--workers <n>]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Default scale: 28 * 4 = 112
DEFAULT_SCALE = 4
UNSHARP_SIGMA = 1.0
UNSHARP_STRENGTH = 1.5  # controls how aggressively edges are sharpened


# ---------------------------------------------------------------------------
# Per-crop processing
# ---------------------------------------------------------------------------

def _unsharp_mask(img: np.ndarray, sigma: float, strength: float) -> np.ndarray:
    """Apply unsharp masking to recover edge crispness after large upscale."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def _upsample_crop(src_path: Path, dst_path: Path, scale: int) -> float:
    """Upsample one crop.  Returns wall-clock seconds taken."""
    t0 = time.perf_counter()
    img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    h, w = img.shape
    out_h, out_w = h * scale, w * scale
    upsampled = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
    sharpened = _unsharp_mask(upsampled, UNSHARP_SIGMA, UNSHARP_STRENGTH)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), sharpened)
    return time.perf_counter() - t0


def _worker(args: tuple[Path, Path, int]) -> float:
    src, dst, scale = args
    return _upsample_crop(src, dst, scale)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upsample cleaned crops to hires pool via LANCZOS4 + unsharp mask."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent,
        help="Root of the handwritten-arithmetic-recognition project.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=DEFAULT_SCALE,
        help=f"Integer upscale factor (default {DEFAULT_SCALE}: 28->112).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes (default 4).",
    )
    args = parser.parse_args()

    project_root: Path = args.project_root.resolve()
    src_root = project_root / "data" / "raw" / "pool_emnist_28_clean"
    dst_root = project_root / "data" / "raw" / "pool_emnist_28_hires"
    scale = args.scale
    n_workers = args.workers

    if not src_root.exists():
        log.error(
            "Clean pool not found: %s\n"
            "Run `python -m src prepare-pool-clean` first.",
            src_root,
        )
        sys.exit(1)

    log.info("Clean pool  : %s", src_root)
    log.info("Hires pool  : %s", dst_root)
    log.info("Scale       : %dx (%dpx -> %dpx)", scale, 28, 28 * scale)
    log.info("Workers     : %d", n_workers)
    log.info(
        "Method      : LANCZOS4 + unsharp mask (sigma=%.1f, strength=%.1f)",
        UNSHARP_SIGMA, UNSHARP_STRENGTH,
    )

    # Collect all (src, dst) pairs
    tasks: list[tuple[Path, Path, int]] = []
    label_dirs = sorted(d for d in src_root.iterdir() if d.is_dir())
    for label_dir in label_dirs:
        label = label_dir.name
        dst_label = dst_root / label
        for src_path in sorted(label_dir.glob("*.png")):
            dst_path = dst_label / src_path.name
            tasks.append((src_path, dst_path, scale))

    if not tasks:
        log.error("No PNG crops found under %s", src_root)
        sys.exit(1)

    log.info("Total crops : %d", len(tasks))
    wall_start = time.perf_counter()

    total_secs = 0.0
    done = 0
    if n_workers <= 1:
        for task in tasks:
            total_secs += _worker(task)
            done += 1
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_worker, t): t for t in tasks}
            for fut in as_completed(futures):
                total_secs += fut.result()
                done += 1
                if done % 5000 == 0:
                    log.info("  %d / %d crops done", done, len(tasks))

    wall_elapsed = time.perf_counter() - wall_start
    avg_ms = (total_secs / max(done, 1)) * 1000

    log.info("")
    log.info("Done.")
    log.info("  Crops processed : %d", done)
    log.info("  Wall time       : %.1f s", wall_elapsed)
    log.info("  Avg per crop    : %.2f ms", avg_ms)
    log.info("  Output dir      : %s", dst_root)

    # Quick size report
    hires_pngs = list(dst_root.rglob("*.png"))
    total_mb = sum(p.stat().st_size for p in hires_pngs) / (1024 * 1024)
    log.info("  Output files    : %d  (%.1f MB)", len(hires_pngs), total_mb)


if __name__ == "__main__":
    main()
