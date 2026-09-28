"""Dual export for the set-maker tool (WS-E).

Owns one responsibility: persisting one finished, human-confirmed scene to disk
in the artifacts each mode needs. Two public entry points, one per mode:

    export_eval(stem, png, drafts, meta, project_root)
        -> data/eval/real/<stem>.png   (preprocessed 512x512 grayscale)
         + data/eval/real/<stem>.label.json
             (the iter6 eval sidecar schema validated through the
              ``src/eval/labels.py`` rules; captures only bbox + fine label)

    export_train(stem, png, drafts, meta, project_root)
        -> data/setmaker/train/<stem>.png   (preprocessed 512x512 grayscale)
         + data/setmaker/train/<stem>.gt.json
             (the synthetic ground-truth shape written by
              ``src/generation/synth_pool.py`` / ``synth_yolo.py``)
         + data/setmaker/train/<stem>.txt
             (YOLO label lines: coarse class id via the ontology + box
              normalised by 512)

Design rules (kept identical to the rest of the package):

- The PNG written is the PREPROCESSED 512x512 image, never the raw canvas, so the
  on-disk image matches exactly what the model sees and every box coordinate the
  exporter writes is in 512 px space. The same ``preprocess_for_pipeline`` the
  dataset builder calls runs here (idempotent on an already-512 grayscale), so a
  caller may pass either the raw drawing or an already-preprocessed array. Draft
  boxes are produced in 512 space upstream (the matcher works on the preprocessed
  canvas), so they are written through unchanged.
- Validation runs before any write and follows the project fail-loud-with-path
  convention: every coordinate, label, equation kind / type, and scene_case is
  checked against the same constants ``src/eval/labels.py`` and
  ``src/core/ontology.py`` expose, and any violation raises ``ValueError`` naming
  the offending field. Nothing is written when validation fails.
- Writes are atomic. Each artifact is written to a sibling ``*.tmp`` file and then
  ``os.replace``-d over its target; for the multi-file train export all temp files
  are written first and only then renamed, so a crash mid-export never leaves a
  half-written set.
- A repeated stem never overwrites. The app's stem allocator only picks a free
  index when the client omits the stem; a client-supplied one is passed straight
  through, so before any write the exporter bumps a colliding stem to the first
  free ``-NNN`` suffix (a candidate is free only when none of that mode's
  artifacts exist for it). This protects a prior labelled sample from being
  silently clobbered by a duplicate stem.
- Stems are sanitised before any filesystem use: a client-supplied stem
  containing a path separator, a ``..`` segment, a leading dot, or whitespace is
  rejected, so ``/api/save`` input can never escape the export directory.
- ``equation_type`` in the train GT mirrors the synthetic out-of-distribution
  value (``ood_unknown``) for structure-only / no-equation scenes; how the GNN
  eq_type loss treats it is a deferred training concern and out of scope here.

The module does NOT own split dirs, manifests, or the dataset loader (all
deferred per the plan); it produces correct standalone files only. The project
root is passed in by the caller (the app resolves it once); this module never
discovers paths on its own, which keeps it a typed, injectable boundary tests can
point at a temporary directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from PIL import Image

from ..core.ontology import (
    YOLO_CLASS_NAMES,
    YOLO_NAME_TO_ID,
    flattened_to_yolo_class_name,
    stage2_labels_ordered,
)
from ..data_pipeline.preprocessing import preprocess_for_pipeline
from ..eval.labels import ERROR_KINDS, EQUATION_KINDS, SCENE_CASES, load_label_sidecars
from .types import AnnotationDraft

# Re-export the canonical loader so a caller wanting to confirm an export is
# round-trip valid (``from src.setmaker.exporters import load_label_sidecars``)
# reads it from one place.
__all__ = [
    "EVAL_DIR_PARTS",
    "TRAIN_DIR_PARTS",
    "EVAL_LABEL_SUFFIX",
    "TRAIN_GT_SUFFIX",
    "TRAIN_TXT_SUFFIX",
    "OOD_EQUATION_TYPE",
    "TRAIN_EQUATION_TYPES",
    "eval_dir",
    "train_dir",
    "sanitize_stem",
    "export_eval",
    "export_train",
    "load_label_sidecars",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Output directories relative to the project root. Tuples (not joined strings) so
# the path is built with ``Path`` joins on any OS.
EVAL_DIR_PARTS: tuple[str, ...] = ("data", "eval", "real")
TRAIN_DIR_PARTS: tuple[str, ...] = ("data", "setmaker", "train")

# Compound filename suffixes for the per-stem artifacts.
EVAL_LABEL_SUFFIX: str = ".label.json"
TRAIN_GT_SUFFIX: str = ".gt.json"
TRAIN_TXT_SUFFIX: str = ".txt"

# Canvas edge length (px). The preprocessor's output and the YOLO normalisation
# divisor are both this value; kept as a named constant rather than a literal 512.
_CANVAS_SIZE: float = 512.0

# Schema tag for the eval sidecar; must equal ``src/eval/labels.py`` expectation.
_EVAL_SCHEMA_VERSION: int = 1

# Synthetic out-of-distribution equation type, mirrored into the train GT for
# structure-only / no-equation scenes (matches ``render_single_scene``'s default
# of ``"ood_unknown"`` when a scene carries no equation kind).
OOD_EQUATION_TYPE: str = "ood_unknown"

# Valid ``equation_type`` strings the train GT may carry: every real equation kind
# the synthetic generator emits plus the structure-only OOD sentinel. (The eval
# sidecar uses the narrower ``EQUATION_KINDS`` set from ``src/eval/labels.py``.)
TRAIN_EQUATION_TYPES: frozenset[str] = frozenset(EQUATION_KINDS) | {OOD_EQUATION_TYPE}

# 36-class fine-label ontology (single source of truth). A draft's ``fine_label``
# and a sidecar ``flattened`` must both be a member of this set.
_VALID_FINE_LABELS: frozenset[str] = frozenset(stage2_labels_ordered())

# Coarse YOLO class names (single source of truth) — a draft's ``yolo_class`` must
# be one of these for the train ``.txt`` class-id mapping.
_VALID_YOLO_CLASSES: frozenset[str] = frozenset(YOLO_CLASS_NAMES)


# ---------------------------------------------------------------------------
# Paths + stem sanitisation
# ---------------------------------------------------------------------------

def eval_dir(project_root: Path) -> Path:
    """Return ``<project_root>/data/eval/real`` (not created here)."""
    return Path(project_root).joinpath(*EVAL_DIR_PARTS)


def train_dir(project_root: Path) -> Path:
    """Return ``<project_root>/data/setmaker/train`` (not created here)."""
    return Path(project_root).joinpath(*TRAIN_DIR_PARTS)


def _free_stem(out_dir: Path, stem: str, suffixes: Sequence[str]) -> str:
    """Return ``stem`` if free in ``out_dir``, else the first ``-NNN``-bumped stem.

    A client-supplied stem can repeat (the app's ``_next_stem`` only allocates a
    free index when the client omits the stem; a supplied one is passed straight
    through). Writing through ``os.replace`` would then silently clobber a prior
    labelled sample, so this guard finds a free name before any write.

    ``suffixes`` lists the per-mode artifacts that share the stem (eval: the PNG
    and the ``.label.json``; train: the PNG, the ``.gt.json`` and the ``.txt``). A
    candidate stem is considered free only when *none* of its artifacts exist, so
    a half-written set still bumps rather than partially overwriting one of its
    siblings. When ``stem`` itself is free it is returned unchanged (the common
    no-collision path), so existing filenames stay exactly what the caller asked
    for. On collision the suffix ``-001``, ``-002``, ... is appended to the
    sanitised stem (themselves sanitisation-safe: digits and a hyphen only) and
    the first free candidate is returned. The search is bounded only by the number
    of existing collisions, which in practice is tiny.
    """

    def _taken(candidate: str) -> bool:
        return any((out_dir / f"{candidate}{suffix}").exists() for suffix in suffixes)

    if not _taken(stem):
        return stem
    index = 1
    while True:
        candidate = f"{stem}-{index:03d}"
        if not _taken(candidate):
            return candidate
        index += 1


def sanitize_stem(stem: str) -> str:
    """Validate a client-supplied stem and return it unchanged when safe.

    ``/api/save`` supplies the stem, so it is untrusted: a value containing a path
    separator (``/`` or ``\\``), a ``..`` segment, a leading dot, leading/trailing
    whitespace, or a NUL byte could escape the export directory or create a hidden
    file. Any such stem raises ``ValueError`` naming the offending value; a clean
    stem is returned as-is (no silent rewriting, so the on-disk filename is exactly
    what the caller asked for).
    """
    if not isinstance(stem, str) or not stem:
        raise ValueError(f"stem must be a non-empty string, got {stem!r}")
    if stem != stem.strip():
        raise ValueError(f"stem must not have leading/trailing whitespace: {stem!r}")
    if "/" in stem or "\\" in stem:
        raise ValueError(f"stem must not contain a path separator: {stem!r}")
    if "\x00" in stem:
        raise ValueError(f"stem must not contain a NUL byte: {stem!r}")
    if stem.startswith("."):
        raise ValueError(f"stem must not start with a dot: {stem!r}")
    # Reject a literal parent-directory token anywhere; ".." is caught by the
    # separator + leading-dot checks above for path forms, but a bare ".." stem
    # has no separator, so guard it explicitly.
    if ".." in stem:
        raise ValueError(f"stem must not contain '..': {stem!r}")
    # Reject interior control characters (0x01-0x1F, excluding 0x00 already
    # checked above). Such characters are invisible in most terminals and could
    # corrupt filenames on POSIX systems or trigger unexpected shell behaviour.
    if any(ord(ch) < 0x20 for ch in stem):
        raise ValueError(
            f"stem must not contain ASCII control characters (< 0x20): {stem!r}"
        )
    # Reject Windows reserved device basenames. These names cannot be used as
    # filenames on Windows and would silently fail or redirect I/O there.
    _WIN_RESERVED = frozenset({
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    })
    # Compare the stem (or its first dot-separated component) case-insensitively.
    if stem.upper().split(".")[0] in _WIN_RESERVED:
        raise ValueError(
            f"stem must not be a Windows reserved device name: {stem!r}"
        )
    return stem


# ---------------------------------------------------------------------------
# Shared validation helpers (the ``src/eval/labels.py`` rules, applied pre-write)
# ---------------------------------------------------------------------------

def _validate_bbox(bbox: Sequence[float], where: str) -> tuple[float, float, float, float]:
    """Validate a ``[x0, y0, x1, y1]`` box in 512 px space.

    Applies the identical rule ``src/eval/labels.py`` enforces on a sidecar box:
    exactly four numbers with ``0 <= x0 < x1 <= 512`` and ``0 <= y0 < y1 <= 512``.
    ``where`` identifies the offending draft in the error message.
    """
    if len(bbox) != 4:
        raise ValueError(f"{where}: bbox must have exactly 4 elements, got {len(bbox)}")
    x0, y0, x1, y1 = (float(v) for v in bbox)
    if not (0.0 <= x0 < x1 <= _CANVAS_SIZE):
        raise ValueError(
            f"{where}: bbox x-coords must satisfy 0 <= x0 < x1 <= 512, "
            f"got x0={x0}, x1={x1}"
        )
    if not (0.0 <= y0 < y1 <= _CANVAS_SIZE):
        raise ValueError(
            f"{where}: bbox y-coords must satisfy 0 <= y0 < y1 <= 512, "
            f"got y0={y0}, y1={y1}"
        )
    return (x0, y0, x1, y1)


def _validate_fine_label(label: str, where: str) -> str:
    """Validate a 36-class fine label (== sidecar ``flattened``)."""
    if label not in _VALID_FINE_LABELS:
        raise ValueError(
            f"{where}: unknown fine_label '{label}'; must be one of the "
            f"36-class ontology"
        )
    return label


def _yolo_class_for_fine_label(fine_label: str, where: str) -> str:
    """Derive the coarse YOLO class name from a 36-class fine label.

    ``AnnotationDraft`` carries only the fine label (the human edits that), so the
    coarse class the train ``.txt`` needs is derived here, in one place, from the
    ontology's ``flattened_to_yolo_class_name``. That function does not handle the
    long-division anchor (fine ``div_bracket`` -> YOLO ``divide_bracket``), so that
    one mapping is added explicitly; the result is asserted to be a real
    ``YOLO_CLASS_NAMES`` member so a future ontology drift fails loud here.
    """
    if fine_label == "div_bracket":
        name = "divide_bracket"
    else:
        try:
            name = flattened_to_yolo_class_name(fine_label)
        except ValueError as exc:
            raise ValueError(
                f"{where}: cannot map fine_label '{fine_label}' to a YOLO class: {exc}"
            ) from exc
    if name not in _VALID_YOLO_CLASSES:
        raise ValueError(
            f"{where}: derived yolo_class '{name}' for fine_label '{fine_label}' "
            f"is not one of {sorted(_VALID_YOLO_CLASSES)}"
        )
    return name


def _require_drafts(drafts: Sequence[AnnotationDraft], where: str) -> None:
    """Reject an empty draft list before any file is written.

    A finished scene must carry at least one box; exporting zero boxes would write
    an empty-symbols sidecar / GT that no downstream consumer wants. Surfaced as a
    ``ValueError`` so ``/api/save`` returns 422 rather than writing a useless pair.
    """
    if not drafts:
        raise ValueError(f"{where}: cannot export a scene with zero drafts")


# ---------------------------------------------------------------------------
# Image + atomic write helpers
# ---------------------------------------------------------------------------

def _preprocessed_512(png: "np.ndarray | Image.Image", where: str) -> np.ndarray:
    """Return the canonical (512, 512) uint8 grayscale array to write to disk.

    The caller must supply an image that is already preprocessed to 512x512
    grayscale (the same space bboxes live in). If a numpy array is passed that
    does not have shape (512, 512), a ``ValueError`` is raised naming the field
    before any preprocessing is attempted. This prevents silent image/bbox frame
    desync: a raw canvas in a different coordinate space would write bboxes that
    point at the wrong pixels on the written PNG.

    Pillow images are accepted and converted through ``preprocess_for_pipeline``
    (they may originate from a data-URL decode which returns RGBA; the
    preprocessor composites onto white). The output shape is asserted after
    preprocessing so any pipeline deviation still fails loud here.
    """
    if png is None:
        raise ValueError(f"{where}: image is required, got None")
    # Guard: a numpy array that is not already (512, 512) grayscale would produce
    # bboxes in the wrong coordinate frame. Reject it before preprocessing so the
    # caller cannot accidentally pass a raw drawing in a different resolution.
    if isinstance(png, np.ndarray) and png.shape != (int(_CANVAS_SIZE), int(_CANVAS_SIZE)):
        raise ValueError(
            f"{where}: image must be a (512, 512) grayscale numpy array; "
            f"got shape {png.shape}. Pass the result of preprocess_for_pipeline() "
            f"or a PIL image so the pixel frame matches the bbox coordinate space."
        )
    gray = preprocess_for_pipeline(png)
    if gray.shape != (int(_CANVAS_SIZE), int(_CANVAS_SIZE)):
        raise ValueError(
            f"{where}: preprocessed image must be {int(_CANVAS_SIZE)}x"
            f"{int(_CANVAS_SIZE)}, got {gray.shape}"
        )
    return gray


def _atomic_write_text(target: Path, text: str) -> None:
    """Write ``text`` to ``target`` atomically (temp file then ``os.replace``)."""
    tmp = target.with_name(f"{target.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, target)


def _atomic_write_png(target: Path, gray: np.ndarray) -> None:
    """Write a grayscale array as a PNG to ``target`` atomically."""
    tmp = target.with_name(f"{target.name}.tmp")
    Image.fromarray(gray).save(tmp, format="PNG")
    os.replace(tmp, target)


# ---------------------------------------------------------------------------
# YOLO label line (mirrors synth_pool.py exactly)
# ---------------------------------------------------------------------------

def _yolo_line(yolo_class: str, bbox: tuple[float, float, float, float]) -> str:
    """Build one YOLO ``.txt`` line for a box on the 512 canvas.

    Format and precision mirror ``src/generation/synth_pool.py`` exactly:
    ``"<class_id> <cx> <cy> <w> <h>"`` with six decimals, coordinates normalised by
    512, and the class id taken from the ontology's coarse-name map so a future
    ontology reorder propagates here without a code change.
    """
    cls_id = YOLO_NAME_TO_ID[yolo_class]
    x0, y0, x1, y1 = bbox
    cx = ((x0 + x1) / 2.0) / _CANVAS_SIZE
    cy = ((y0 + y1) / 2.0) / _CANVAS_SIZE
    bw = max((x1 - x0) / _CANVAS_SIZE, 1e-6)
    bh = max((y1 - y0) / _CANVAS_SIZE, 1e-6)
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


# ---------------------------------------------------------------------------
# Eval export
# ---------------------------------------------------------------------------

def export_eval(
    stem: str,
    png: "np.ndarray | Image.Image",
    drafts: Sequence[AnnotationDraft],
    meta: Mapping[str, Any],
    project_root: Path,
) -> Dict[str, Path]:
    """Persist one finished EVAL scene: preprocessed PNG + validated sidecar.

    Writes ``data/eval/real/<stem>.png`` (the preprocessed 512x512 grayscale) and
    ``data/eval/real/<stem>.label.json`` (the iter6 eval sidecar). Eval captures
    only what the harness reads: each draft contributes a ``{flattened, bbox_px}``
    symbol; row/col/equation_idx are inferred geometrically at eval time and are
    deliberately not written.

    ``meta`` supplies the scene-level fields:
        equation_kind : str   required; must be in ``EQUATION_KINDS``.
        scene_case    : str   optional; when present must be in ``SCENE_CASES``.
        error_kind    : str   optional; when present must be in ``ERROR_KINDS``.
            Tags how the scene is wrong as-drawn (e.g. ``"missing_carry"``); the
            system records the error, it never corrects it. The key is omitted
            from the sidecar when ``None`` or absent.

    The sidecar is validated against the ``src/eval/labels.py`` rules before any
    write (schema version, label vocabulary, bbox bounds, equation kind, scene
    case) and, after writing, is loaded back through ``load_label_sidecars`` so a
    caller that only calls this function still gets the loader's guarantee. Writes
    are atomic (temp then rename) and the stem is sanitised first. A stem that
    already names an output on disk is auto-bumped to the first free ``-NNN``
    suffix rather than overwriting a prior labelled sample; the returned paths
    carry the name actually written.

    Args:
        stem: Output filename stem (no extension), e.g. ``"re-addition-001"``.
        png: The drawing as a numpy array or PIL image; preprocessed to 512x512
            before saving.
        drafts: The human-confirmed boxes for this scene (must be non-empty).
        meta: Scene-level metadata (see above).
        project_root: Project root containing ``data/``.

    Returns:
        Mapping with ``"png"`` and ``"label"`` keys -> the written paths.

    Raises:
        ValueError: On an unsafe stem, an empty draft list, an invalid bbox /
            fine_label, a missing or invalid ``equation_kind``, or an invalid
            ``scene_case``. The message names the offending field; nothing is
            written when validation fails.
    """
    safe = sanitize_stem(stem)
    where = f"export_eval[{safe}]"
    _require_drafts(drafts, where)

    equation_kind = meta.get("equation_kind")
    if equation_kind not in EQUATION_KINDS:
        raise ValueError(
            f"{where}: equation_kind {equation_kind!r} is not valid; must be one "
            f"of {sorted(EQUATION_KINDS)}"
        )

    scene_case = meta.get("scene_case")
    if scene_case is not None and scene_case not in SCENE_CASES:
        raise ValueError(
            f"{where}: scene_case {scene_case!r} is not valid; must be one of "
            f"{sorted(SCENE_CASES)}"
        )

    # As-drawn error tag (optional). Validated against the same canonical set the
    # eval loader enforces so a bad value fails here, before any write, rather
    # than at eval time. None / absent => the key is omitted from the sidecar.
    error_kind = meta.get("error_kind")
    if error_kind is not None and error_kind not in ERROR_KINDS:
        raise ValueError(
            f"{where}: error_kind {error_kind!r} is not valid; must be one of "
            f"{sorted(ERROR_KINDS)}"
        )

    # Build + validate the symbol list (eval scope: flattened + bbox only).
    symbols: List[Dict[str, Any]] = []
    for i, draft in enumerate(drafts):
        d_where = f"{where} draft[{i}]"
        flattened = _validate_fine_label(draft.fine_label, d_where)
        x0, y0, x1, y1 = _validate_bbox(draft.bbox_px, d_where)
        symbols.append({"flattened": flattened, "bbox_px": [x0, y0, x1, y1]})

    gray = _preprocessed_512(png, where)

    out_dir = eval_dir(project_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the final stem against the dir: if the sanitised stem (or any of its
    # two artifacts) already exists, bump to the first free ``-NNN`` so a repeated
    # client-supplied stem never silently overwrites a prior labelled sample. The
    # sidecar's ``sample`` field and both filenames use the resolved stem so the
    # round-trip loader finds it under the name actually written.
    safe = _free_stem(out_dir, safe, (".png", EVAL_LABEL_SUFFIX))
    png_path = out_dir / f"{safe}.png"
    label_path = out_dir / f"{safe}{EVAL_LABEL_SUFFIX}"

    sidecar: Dict[str, Any] = {
        "schema_version": _EVAL_SCHEMA_VERSION,
        "sample": safe,
        "equation_kind": equation_kind,
        "symbols": symbols,
    }
    if scene_case is not None:
        sidecar["scene_case"] = scene_case
    if error_kind is not None:
        sidecar["error_kind"] = error_kind

    _atomic_write_png(png_path, gray)
    _atomic_write_text(label_path, json.dumps(sidecar, indent=2))

    # Round-trip through the canonical loader: scoped to this stem's sidecar so a
    # standalone caller still inherits the loader's validation guarantee, and a
    # bug in the constructed payload fails here rather than at eval time.
    _verify_sidecar_loads(label_path, safe)

    return {"png": png_path, "label": label_path}


def _verify_sidecar_loads(label_path: Path, stem: str) -> None:
    """Load the just-written sidecar via ``load_label_sidecars`` and assert it parsed.

    The loader scans a directory, so it is pointed at the sidecar's own directory;
    the written stem must appear in the result. Any parse / validation failure
    surfaces as the loader's ``ValueError`` (path included), keeping the eval
    sidecar's on-disk guarantee identical to what the harness later relies on.
    """
    loaded = load_label_sidecars(label_path.parent)
    if stem not in loaded:
        raise ValueError(
            f"{label_path}: sidecar did not load back under stem '{stem}' "
            f"(loaded stems: {sorted(loaded)})"
        )


# ---------------------------------------------------------------------------
# Train export
# ---------------------------------------------------------------------------

def export_train(
    stem: str,
    png: "np.ndarray | Image.Image",
    drafts: Sequence[AnnotationDraft],
    meta: Mapping[str, Any],
    project_root: Path,
) -> Dict[str, Path]:
    """Persist one finished TRAIN scene: preprocessed PNG + GT json + YOLO txt.

    Writes three artifacts under ``data/setmaker/train/``:
        ``<stem>.png``     the preprocessed 512x512 grayscale image;
        ``<stem>.gt.json`` the synthetic ground-truth shape (top-level
            ``equation_type`` / ``completion_stage`` / ``split`` / ``case`` /
            ``symbols``, each symbol carrying the seven keys
            ``fine_label`` / ``glyph_key`` / ``yolo_class`` / ``row_index`` /
            ``col_index`` / ``bbox`` / ``equation_idx``);
        ``<stem>.txt``     one YOLO line per box (coarse class id via the ontology
            plus the box normalised by 512), format-identical to the synthetic
            generator's labels.

    Train captures the full draft: ``row_index`` / ``col_index`` /
    ``equation_idx`` are written (unlike eval). ``meta`` supplies the scene-level
    fields:
        equation_type    : str   required; in ``TRAIN_EQUATION_TYPES``
                                  (a real kind or the ``ood_unknown`` sentinel,
                                  which structure-only scenes mirror).
        completion_stage : str   optional; defaults to ``"full"``.
        case             : str   optional SceneCase value; when present must be in
                                  ``SCENE_CASES``.
        split            : str   optional; defaults to ``"train"``.
        glyph_key        : per-draft glyph keys are read from ``meta["glyph_keys"]``
                                  when supplied (one per draft, same order); absent
                                  -> derived from the fine label.

    Every box's ``fine_label`` is validated against the 36-class ontology and the
    bbox against the 512 bounds before any write; the coarse ``yolo_class`` is
    derived from the fine label (the draft carries no coarse class). All three temp
    files are written first and then renamed, so a failure mid-export never leaves
    a partial set on disk. The stem is sanitised first; a stem that already names
    any of its three outputs on disk is auto-bumped to the first free ``-NNN``
    suffix rather than overwriting a prior labelled sample (the returned paths
    carry the name actually written).

    Args:
        stem: Output filename stem (no extension), e.g. ``"rt-addition-001"``.
        png: The drawing as a numpy array or PIL image; preprocessed to 512x512.
        drafts: The human-confirmed boxes (must be non-empty).
        meta: Scene-level metadata (see above).
        project_root: Project root containing ``data/``.

    Returns:
        Mapping with ``"png"``, ``"gt"`` and ``"txt"`` keys -> the written paths.

    Raises:
        ValueError: On an unsafe stem, an empty draft list, an invalid bbox or
            fine_label (incl. a fine label with no coarse-class mapping), an
            invalid ``equation_type`` or ``case``, or a ``glyph_keys`` list whose
            length does not match ``drafts``. The message names the offending
            field; nothing is written when validation fails.
    """
    safe = sanitize_stem(stem)
    where = f"export_train[{safe}]"
    _require_drafts(drafts, where)

    equation_type = meta.get("equation_type")
    if equation_type not in TRAIN_EQUATION_TYPES:
        raise ValueError(
            f"{where}: equation_type {equation_type!r} is not valid; must be one "
            f"of {sorted(TRAIN_EQUATION_TYPES)}"
        )

    # As-drawn error tag (optional, mirrors eval). Validated against ERROR_KINDS
    # before any write; None / absent -> key omitted from the GT (matches the
    # worklist counter which uses .get() and treats absent as "no error").
    error_kind = meta.get("error_kind")
    if error_kind is not None and error_kind not in ERROR_KINDS:
        raise ValueError(
            f"{where}: error_kind {error_kind!r} is not valid; must be one of "
            f"{sorted(ERROR_KINDS)}"
        )

    completion_stage = meta.get("completion_stage", "full")
    if not isinstance(completion_stage, str) or not completion_stage:
        raise ValueError(
            f"{where}: completion_stage must be a non-empty string, "
            f"got {completion_stage!r}"
        )

    case = meta.get("case")
    if case is not None and case not in SCENE_CASES:
        raise ValueError(
            f"{where}: case {case!r} is not valid; must be one of "
            f"{sorted(SCENE_CASES)}"
        )

    split = meta.get("split", "train")
    if not isinstance(split, str) or not split:
        raise ValueError(f"{where}: split must be a non-empty string, got {split!r}")

    glyph_keys = meta.get("glyph_keys")
    if glyph_keys is not None and len(glyph_keys) != len(drafts):
        raise ValueError(
            f"{where}: glyph_keys length {len(glyph_keys)} does not match "
            f"drafts length {len(drafts)}"
        )

    # Build + validate the GT symbols and the YOLO lines together so a single pass
    # validates every box once and the two artifacts stay in lockstep.
    gt_symbols: List[Dict[str, Any]] = []
    yolo_lines: List[str] = []
    for i, draft in enumerate(drafts):
        d_where = f"{where} draft[{i}]"
        fine_label = _validate_fine_label(draft.fine_label, d_where)
        yolo_class = _yolo_class_for_fine_label(fine_label, d_where)
        x0, y0, x1, y1 = _validate_bbox(draft.bbox_px, d_where)
        glyph_key = (
            str(glyph_keys[i]) if glyph_keys is not None
            else _glyph_key_from_fine_label(fine_label)
        )
        gt_symbols.append({
            "fine_label": fine_label,
            "glyph_key": glyph_key,
            "yolo_class": yolo_class,
            "row_index": int(draft.row_index),
            "col_index": int(draft.col_index),
            "bbox": [x0, y0, x1, y1],
            "equation_idx": int(draft.equation_idx),
        })
        yolo_lines.append(_yolo_line(yolo_class, (x0, y0, x1, y1)))

    gt_payload: Dict[str, Any] = {
        "equation_type": equation_type,
        "completion_stage": completion_stage,
        "split": split,
        "case": case,
        "symbols": gt_symbols,
    }
    if error_kind is not None:
        gt_payload["error_kind"] = error_kind

    gray = _preprocessed_512(png, where)

    out_dir = train_dir(project_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the final stem against the dir: if the sanitised stem (or any of its
    # three artifacts) already exists, bump to the first free ``-NNN`` so a
    # repeated client-supplied stem never silently overwrites a prior labelled
    # sample. The train GT carries no embedded stem, so only the paths change.
    safe = _free_stem(out_dir, safe, (".png", TRAIN_GT_SUFFIX, TRAIN_TXT_SUFFIX))
    png_path = out_dir / f"{safe}.png"
    gt_path = out_dir / f"{safe}{TRAIN_GT_SUFFIX}"
    txt_path = out_dir / f"{safe}{TRAIN_TXT_SUFFIX}"

    # Write all three temps first, then rename, so a failure leaves nothing
    # partial: a PNG without its labels (or vice versa) never appears on disk.
    png_tmp = png_path.with_name(f"{png_path.name}.tmp")
    gt_tmp = gt_path.with_name(f"{gt_path.name}.tmp")
    txt_tmp = txt_path.with_name(f"{txt_path.name}.tmp")
    Image.fromarray(gray).save(png_tmp, format="PNG")
    gt_tmp.write_text(json.dumps(gt_payload, indent=2), encoding="utf-8")
    txt_tmp.write_text("\n".join(yolo_lines), encoding="utf-8")
    os.replace(png_tmp, png_path)
    os.replace(gt_tmp, gt_path)
    os.replace(txt_tmp, txt_path)

    return {"png": png_path, "gt": gt_path, "txt": txt_path}


def _glyph_key_from_fine_label(fine_label: str) -> str:
    """Derive a pool glyph key from a 36-class fine label.

    Mirrors ``src/core/ontology.glyph_key_for_pool`` for the labels the set-maker
    can produce. ``div_bracket`` has no physical crop folder (it is drawn
    programmatically), so it maps to its own name rather than raising, keeping the
    GT payload writable for long-division targets.
    """
    from ..core.ontology import glyph_key_for_pool  # noqa: PLC0415

    try:
        return glyph_key_for_pool(fine_label)
    except ValueError:
        # Programmatically-drawn structural tokens (e.g. div_bracket) have no pool
        # folder; use the label itself as the key so the GT stays writable.
        return fine_label
