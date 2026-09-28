"""Tests for the set-maker dual exporter (``exporters.py``, WS-E).

These are always-on, data-free tests: they build ``AnnotationDraft`` objects by
hand and write into a ``tempfile.TemporaryDirectory`` standing in for the project
root, so no symbol pool, weights, or network access is needed.

Coverage mirrors the plan's ``test_exporters.py`` line:

- eval sidecar passes ``load_label_sidecars`` (the canonical loader);
- train ``.txt`` lines parse as valid YOLO (class id in the ontology range,
  coords in ``[0, 1]``);
- train ``.gt.json`` matches the synthetic ground-truth shape;
- writes are atomic (temp-then-rename: no ``*.tmp`` left, re-export overwrites
  cleanly);
- the YOLO class-id mapping is asserted against ``src/core/ontology.py``
  (the pre-mortem's corrupt-label failure mode);
- stems are sanitised (no path traversal) before any write;
- the PNG written is the preprocessed 512x512 image;
- structure-only scenes carry ``equation_type == "ood_unknown"`` in the GT.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.core.ontology import YOLO_CLASS_NAMES, YOLO_NAME_TO_ID, stage2_labels_ordered
from src.eval.labels import ERROR_KINDS, EQUATION_KINDS, SampleLabel, load_label_sidecars
from src.setmaker import exporters
from src.setmaker.exporters import (
    OOD_EQUATION_TYPE,
    TRAIN_EQUATION_TYPES,
    _free_stem,
    eval_dir,
    export_eval,
    export_train,
    sanitize_stem,
    train_dir,
)
from src.setmaker.types import AnnotationDraft


def _draft(
    *,
    bbox=(100.0, 100.0, 160.0, 180.0),
    fine_label="main_3",
    row_index=0,
    col_index=0,
    equation_idx=0,
    confidence=0.9,
    source="matched",
    flagged=False,
) -> AnnotationDraft:
    """Build one valid ``AnnotationDraft`` with sensible defaults.

    ``AnnotationDraft`` carries no coarse ``yolo_class`` (the human edits only the
    fine label); the train exporter derives the coarse class from ``fine_label``.
    """
    return AnnotationDraft(
        bbox_px=bbox,
        fine_label=fine_label,
        row_index=row_index,
        col_index=col_index,
        equation_idx=equation_idx,
        confidence=confidence,
        source=source,
        flagged=flagged,
    )


def _addition_drafts() -> list[AnnotationDraft]:
    """A small but realistic addition scene: 3 + 4 with a result bar and result 7.

    Covers three coarse classes once the exporter derives them from the fine
    labels: ``digit_main`` (the digits), ``operator`` (the plus), ``result_bar``.
    """
    return [
        _draft(bbox=(40.0, 40.0, 90.0, 110.0), fine_label="main_3", row_index=0, col_index=0),
        _draft(bbox=(100.0, 40.0, 150.0, 110.0), fine_label="op_plus", row_index=0, col_index=1),
        _draft(bbox=(160.0, 40.0, 210.0, 110.0), fine_label="main_4", row_index=0, col_index=2),
        _draft(bbox=(40.0, 120.0, 210.0, 128.0), fine_label="result_bar", row_index=1, col_index=0),
        _draft(bbox=(160.0, 140.0, 210.0, 210.0), fine_label="main_7", row_index=2, col_index=2),
    ]


def _canvas(size: int = 512) -> np.ndarray:
    """A white grayscale canvas with a little ink so preprocessing is well-defined.

    The contrast-normalisation step needs a non-flat histogram; a couple of dark
    rectangles give it ink without mattering to the label assertions.
    """
    arr = np.full((size, size), 255, dtype=np.uint8)
    arr[50:110, 40:90] = 0
    arr[50:110, 160:210] = 0
    return arr


class TestSanitizeStem(unittest.TestCase):
    """The stem is client-supplied, so traversal / hidden-file forms are rejected."""

    def test_clean_stem_passes_unchanged(self) -> None:
        self.assertEqual(sanitize_stem("re-addition-001"), "re-addition-001")
        self.assertEqual(sanitize_stem("rt-division_long-042"), "rt-division_long-042")

    def test_empty_stem_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sanitize_stem("")

    def test_forward_slash_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sanitize_stem("a/b")

    def test_backslash_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sanitize_stem("a\\b")

    def test_parent_traversal_rejected(self) -> None:
        for bad in ("..", "../etc/passwd", "foo/../bar", "a..b"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    sanitize_stem(bad)

    def test_leading_dot_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sanitize_stem(".hidden")

    def test_whitespace_padding_rejected(self) -> None:
        for bad in (" lead", "trail ", "mid space\t"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    sanitize_stem(bad)

    def test_nul_byte_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sanitize_stem("a\x00b")

    def test_interior_control_chars_rejected(self) -> None:
        # Fix 4: interior control chars (0x01-0x1F) must be rejected. 0x00
        # was already caught; this covers the rest of the range.
        for code in (0x01, 0x09, 0x0A, 0x0D, 0x1F):
            bad = "stem" + chr(code) + "name"
            with self.subTest(code=hex(code), bad=repr(bad)):
                with self.assertRaises(ValueError):
                    sanitize_stem(bad)

    def test_windows_reserved_names_rejected(self) -> None:
        # Fix 4: Windows reserved device basenames must be rejected regardless
        # of case, so exports never silently redirect I/O on Windows.
        reserved = [
            "CON", "PRN", "AUX", "NUL",
            "COM1", "COM9", "LPT1", "LPT9",
            "con", "nul", "lpt3",            # lower-case variants
        ]
        for name in reserved:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    sanitize_stem(name)

    def test_non_reserved_names_not_blocked(self) -> None:
        # Sanity: names that look similar but are not reserved must pass.
        for ok in ("CONMAN", "NULL", "COM", "LPT", "com10", "lpt10"):
            with self.subTest(ok=ok):
                self.assertEqual(sanitize_stem(ok), ok)


class TestExportPaths(unittest.TestCase):
    """The output directories resolve under the given project root."""

    def test_eval_dir(self) -> None:
        root = Path("/tmp/pr")
        self.assertEqual(eval_dir(root), root / "data" / "eval" / "real")

    def test_train_dir(self) -> None:
        root = Path("/tmp/pr")
        self.assertEqual(train_dir(root), root / "data" / "setmaker" / "train")


class TestExportEval(unittest.TestCase):
    """``export_eval`` writes a preprocessed PNG + a loader-valid sidecar."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _export(self, stem="re-addition-001", drafts=None, meta=None):
        drafts = drafts if drafts is not None else _addition_drafts()
        meta = meta if meta is not None else {"equation_kind": "addition"}
        return export_eval(stem, _canvas(), drafts, meta, self.root)

    def test_writes_png_and_label(self) -> None:
        out = self._export()
        self.assertTrue(out["png"].exists())
        self.assertTrue(out["label"].exists())
        self.assertEqual(out["png"].name, "re-addition-001.png")
        self.assertEqual(out["label"].name, "re-addition-001.label.json")

    def test_sidecar_loads_via_load_label_sidecars(self) -> None:
        # The headline guarantee: the written sidecar passes the canonical loader.
        out = self._export()
        loaded = load_label_sidecars(out["label"].parent)
        self.assertIn("re-addition-001", loaded)
        label = loaded["re-addition-001"]
        self.assertIsInstance(label, SampleLabel)
        self.assertEqual(label.equation_kind, "addition")
        self.assertEqual(label.schema_version, 1)
        self.assertIsNotNone(label.symbols)
        self.assertEqual(len(label.symbols), len(_addition_drafts()))

    def test_sidecar_captures_only_flattened_and_bbox(self) -> None:
        # Eval scope: no row/col/equation_idx in the per-symbol payload.
        out = self._export()
        doc = json.loads(out["label"].read_text())
        for sym in doc["symbols"]:
            self.assertEqual(set(sym.keys()), {"flattened", "bbox_px"})
            self.assertEqual(len(sym["bbox_px"]), 4)

    def test_sidecar_sample_matches_stem(self) -> None:
        out = self._export(stem="re-subtraction-007", meta={"equation_kind": "subtraction"})
        doc = json.loads(out["label"].read_text())
        self.assertEqual(doc["sample"], "re-subtraction-007")

    def test_scene_case_written_when_valid(self) -> None:
        out = self._export(meta={"equation_kind": "addition", "scene_case": "addition_dense_carries"})
        doc = json.loads(out["label"].read_text())
        self.assertEqual(doc["scene_case"], "addition_dense_carries")
        # And it still loads (scene_case validated by the loader too).
        loaded = load_label_sidecars(out["label"].parent)
        self.assertEqual(loaded["re-addition-001"].scene_case, "addition_dense_carries")

    def test_scene_case_omitted_when_absent(self) -> None:
        out = self._export()
        doc = json.loads(out["label"].read_text())
        self.assertNotIn("scene_case", doc)

    def test_invalid_scene_case_rejected_and_nothing_written(self) -> None:
        with self.assertRaises(ValueError):
            self._export(meta={"equation_kind": "addition", "scene_case": "not_a_case"})
        self.assertFalse((eval_dir(self.root) / "re-addition-001.label.json").exists())

    def test_missing_equation_kind_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._export(meta={})

    def test_invalid_equation_kind_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._export(meta={"equation_kind": "calculus"})

    def test_bare_digits_equation_kind_allowed(self) -> None:
        # bare_digits is a real structure-only eval kind (in EQUATION_KINDS).
        out = self._export(
            stem="re-bare_digits-001",
            drafts=[_draft(fine_label="main_5"), _draft(bbox=(200.0, 100.0, 260.0, 180.0), fine_label="main_2")],
            meta={"equation_kind": "bare_digits", "scene_case": "bare_digits"},
        )
        loaded = load_label_sidecars(out["label"].parent)
        self.assertEqual(loaded["re-bare_digits-001"].equation_kind, "bare_digits")

    def test_error_kind_written_when_valid(self) -> None:
        # As-drawn error tag (gap 2): a valid error_kind lands in the sidecar and
        # survives the canonical loader round-trip.
        out = self._export(
            meta={"equation_kind": "addition", "scene_case": "addition", "error_kind": "missing_carry"}
        )
        doc = json.loads(out["label"].read_text())
        self.assertEqual(doc["error_kind"], "missing_carry")
        loaded = load_label_sidecars(out["label"].parent)
        self.assertEqual(loaded["re-addition-001"].error_kind, "missing_carry")

    def test_every_error_kind_value_accepted(self) -> None:
        # Each canonical ERROR_KINDS member exports + loads back to itself.
        for i, kind in enumerate(sorted(ERROR_KINDS)):
            stem = f"re-addition-{i:03d}"
            out = self._export(stem=stem, meta={"equation_kind": "addition", "error_kind": kind})
            loaded = load_label_sidecars(out["label"].parent)
            self.assertEqual(loaded[stem].error_kind, kind)

    def test_error_kind_omitted_when_absent(self) -> None:
        # No error_kind in meta -> the key is absent from the sidecar (clean scene).
        out = self._export()
        doc = json.loads(out["label"].read_text())
        self.assertNotIn("error_kind", doc)

    def test_error_kind_omitted_when_none(self) -> None:
        # An explicit None is treated like absent: the key is omitted entirely.
        out = self._export(meta={"equation_kind": "addition", "error_kind": None})
        doc = json.loads(out["label"].read_text())
        self.assertNotIn("error_kind", doc)

    def test_invalid_error_kind_rejected_and_nothing_written(self) -> None:
        # A bad error_kind fails before any write (same contract as scene_case).
        with self.assertRaises(ValueError):
            self._export(meta={"equation_kind": "addition", "error_kind": "not_a_real_error"})
        self.assertFalse((eval_dir(self.root) / "re-addition-001.label.json").exists())
        self.assertFalse((eval_dir(self.root) / "re-addition-001.png").exists())

    def test_empty_drafts_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._export(drafts=[])

    def test_unknown_fine_label_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._export(drafts=[_draft(fine_label="main_99")])

    def test_out_of_bounds_bbox_rejected(self) -> None:
        for bad in [
            (-1.0, 10.0, 50.0, 80.0),       # x0 < 0
            (10.0, 10.0, 600.0, 80.0),      # x1 > 512
            (50.0, 10.0, 40.0, 80.0),       # x1 < x0
            (10.0, 80.0, 50.0, 40.0),       # y1 < y0
        ]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self._export(drafts=[_draft(bbox=bad)])

    def test_png_is_512_grayscale(self) -> None:
        out = self._export()
        with Image.open(out["png"]) as im:
            self.assertEqual(im.size, (512, 512))
            self.assertEqual(im.mode, "L")

    def test_non_512_numpy_array_rejected_before_write(self) -> None:
        # A non-512 numpy array is rejected with ValueError naming the field
        # (fix 1: prevents silent image/bbox frame desync). The exporter no
        # longer silently resizes an arbitrarily-sized array; bboxes are in
        # 512px space so the image must already be in that space.
        raw = np.full((300, 220), 255, dtype=np.uint8)
        raw[20:80, 20:80] = 0
        with self.assertRaises(ValueError) as ctx:
            export_eval("re-addition-002", raw, _addition_drafts(),
                        {"equation_kind": "addition"}, self.root)
        self.assertIn("512", str(ctx.exception))
        # Nothing written: the guard fires before any file operation.
        self.assertFalse((eval_dir(self.root) / "re-addition-002.png").exists())

    def test_accepts_pil_image(self) -> None:
        pil = Image.fromarray(_canvas())
        out = export_eval("re-addition-003", pil, _addition_drafts(),
                          {"equation_kind": "addition"}, self.root)
        self.assertTrue(out["png"].exists())

    def test_no_tmp_files_left(self) -> None:
        self._export()
        leftovers = list(eval_dir(self.root).glob("*.tmp"))
        self.assertEqual(leftovers, [])

    def test_reexport_same_stem_does_not_overwrite(self) -> None:
        # Collision bump (data-corruption guard): a second save with the SAME stem
        # must NOT clobber the first. The first sample keeps its content and name;
        # the second is written under a bumped ``-NNN`` stem. No temp is left.
        first = self._export(meta={"equation_kind": "addition"})
        second = self._export(meta={"equation_kind": "subtraction"})

        # First sample untouched: original name, original content.
        self.assertEqual(first["label"].name, "re-addition-001.label.json")
        first_doc = json.loads(first["label"].read_text())
        self.assertEqual(first_doc["equation_kind"], "addition")

        # Second sample landed on a bumped stem with the new content.
        self.assertNotEqual(second["label"], first["label"])
        self.assertEqual(second["label"].name, "re-addition-001-001.label.json")
        self.assertEqual(second["png"].name, "re-addition-001-001.png")
        second_doc = json.loads(second["label"].read_text())
        self.assertEqual(second_doc["equation_kind"], "subtraction")
        # The bumped sidecar's ``sample`` matches the bumped stem (loader key).
        self.assertEqual(second_doc["sample"], "re-addition-001-001")

        # Both samples exist; no temp residue.
        self.assertTrue(first["label"].exists())
        self.assertTrue(second["label"].exists())
        self.assertEqual(list(eval_dir(self.root).glob("*.tmp")), [])

    def test_unsafe_stem_rejected_before_write(self) -> None:
        with self.assertRaises(ValueError):
            export_eval("../escape", _canvas(), _addition_drafts(),
                        {"equation_kind": "addition"}, self.root)
        # The directory must not even exist if nothing was ever written.
        self.assertFalse(any(eval_dir(self.root).glob("*")) if eval_dir(self.root).exists() else False)


class TestExportTrain(unittest.TestCase):
    """``export_train`` writes a preprocessed PNG + GT json + YOLO txt."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _export(self, stem="rt-addition-001", drafts=None, meta=None):
        drafts = drafts if drafts is not None else _addition_drafts()
        meta = meta if meta is not None else {
            "equation_type": "addition",
            "completion_stage": "full",
            "case": "addition",
            "split": "train",
        }
        return export_train(stem, _canvas(), drafts, meta, self.root)

    def test_writes_three_artifacts(self) -> None:
        out = self._export()
        for key in ("png", "gt", "txt"):
            self.assertTrue(out[key].exists(), f"{key} not written")
        self.assertEqual(out["png"].name, "rt-addition-001.png")
        self.assertEqual(out["gt"].name, "rt-addition-001.gt.json")
        self.assertEqual(out["txt"].name, "rt-addition-001.txt")

    def test_gt_top_level_shape_matches_synthetic(self) -> None:
        out = self._export()
        doc = json.loads(out["gt"].read_text())
        self.assertEqual(
            set(doc.keys()),
            {"equation_type", "completion_stage", "split", "case", "symbols"},
        )
        self.assertEqual(doc["equation_type"], "addition")
        self.assertEqual(doc["completion_stage"], "full")
        self.assertEqual(doc["split"], "train")
        self.assertEqual(doc["case"], "addition")

    def test_gt_symbol_has_seven_synthetic_keys(self) -> None:
        out = self._export()
        doc = json.loads(out["gt"].read_text())
        expected = {
            "fine_label", "glyph_key", "yolo_class",
            "row_index", "col_index", "bbox", "equation_idx",
        }
        self.assertEqual(len(doc["symbols"]), len(_addition_drafts()))
        for sym in doc["symbols"]:
            self.assertEqual(set(sym.keys()), expected)
            self.assertEqual(len(sym["bbox"]), 4)
            self.assertIsInstance(sym["row_index"], int)
            self.assertIsInstance(sym["col_index"], int)
            self.assertIsInstance(sym["equation_idx"], int)

    def test_gt_fine_labels_in_36_class_ontology(self) -> None:
        out = self._export()
        doc = json.loads(out["gt"].read_text())
        valid = set(stage2_labels_ordered())
        for sym in doc["symbols"]:
            self.assertIn(sym["fine_label"], valid)

    def test_gt_glyph_key_derived_from_fine_label(self) -> None:
        # No glyph_keys supplied -> derived: a digit's key is the bare digit.
        out = self._export(
            stem="rt-bare-001",
            drafts=[_draft(fine_label="main_5")],
            meta={"equation_type": "bare_digits", "case": "bare_digits"},
        )
        doc = json.loads(out["gt"].read_text())
        self.assertEqual(doc["symbols"][0]["glyph_key"], "5")

    def test_gt_glyph_keys_from_meta_used_when_supplied(self) -> None:
        drafts = [_draft(fine_label="main_3"), _draft(bbox=(200.0, 40.0, 250.0, 110.0), fine_label="main_4")]
        out = self._export(
            stem="rt-gk-001",
            drafts=drafts,
            meta={"equation_type": "bare_digits", "glyph_keys": ["3", "4"]},
        )
        doc = json.loads(out["gt"].read_text())
        self.assertEqual([s["glyph_key"] for s in doc["symbols"]], ["3", "4"])

    def test_glyph_keys_length_mismatch_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._export(meta={"equation_type": "addition", "glyph_keys": ["3"]})

    def test_txt_lines_parse_as_valid_yolo(self) -> None:
        out = self._export()
        lines = out["txt"].read_text().splitlines()
        self.assertEqual(len(lines), len(_addition_drafts()))
        n_classes = len(YOLO_CLASS_NAMES)
        for line in lines:
            parts = line.split()
            self.assertEqual(len(parts), 5)
            cls_id = int(parts[0])
            self.assertGreaterEqual(cls_id, 0)
            self.assertLess(cls_id, n_classes)
            cx, cy, bw, bh = (float(p) for p in parts[1:])
            for v in (cx, cy, bw, bh):
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)
            # A box has positive extent.
            self.assertGreater(bw, 0.0)
            self.assertGreater(bh, 0.0)

    def test_txt_class_ids_match_ontology_mapping(self) -> None:
        # The pre-mortem failure mode: wrong class-id order vs ontology. The coarse
        # class is derived from each draft's fine label; assert the written id
        # equals YOLO_NAME_TO_ID for that derived coarse class, in order.
        from src.core.ontology import flattened_to_yolo_class_name

        drafts = _addition_drafts()
        out = self._export(drafts=drafts)
        lines = out["txt"].read_text().splitlines()
        for draft, line in zip(drafts, lines):
            written_id = int(line.split()[0])
            expected_coarse = flattened_to_yolo_class_name(draft.fine_label)
            self.assertEqual(written_id, YOLO_NAME_TO_ID[expected_coarse])

    def test_txt_normalisation_is_by_512(self) -> None:
        # A single known box: cx/cy/w/h must equal the hand-computed /512 values.
        draft = _draft(bbox=(128.0, 256.0, 256.0, 384.0), fine_label="main_1")
        out = self._export(stem="rt-one-001", drafts=[draft],
                           meta={"equation_type": "bare_digits"})
        cx, cy, bw, bh = (float(p) for p in out["txt"].read_text().split()[1:])
        self.assertAlmostEqual(cx, ((128.0 + 256.0) / 2.0) / 512.0, places=6)
        self.assertAlmostEqual(cy, ((256.0 + 384.0) / 2.0) / 512.0, places=6)
        self.assertAlmostEqual(bw, (256.0 - 128.0) / 512.0, places=6)
        self.assertAlmostEqual(bh, (384.0 - 256.0) / 512.0, places=6)

    def test_structure_only_equation_type_ood_unknown(self) -> None:
        # Structure-only scene: equation_type mirrors the synthetic OOD value.
        out = self._export(
            stem="rt-standalone_bar-001",
            drafts=[_draft(bbox=(40.0, 120.0, 210.0, 128.0), fine_label="result_bar",
                           row_index=0, col_index=0)],
            meta={"equation_type": OOD_EQUATION_TYPE, "case": "standalone_bar"},
        )
        doc = json.loads(out["gt"].read_text())
        self.assertEqual(doc["equation_type"], OOD_EQUATION_TYPE)
        self.assertIn(OOD_EQUATION_TYPE, TRAIN_EQUATION_TYPES)

    def test_default_completion_stage_and_split(self) -> None:
        out = self._export(stem="rt-defaults-001",
                           meta={"equation_type": "addition"})
        doc = json.loads(out["gt"].read_text())
        self.assertEqual(doc["completion_stage"], "full")
        self.assertEqual(doc["split"], "train")
        self.assertIsNone(doc["case"])

    def test_invalid_equation_type_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._export(meta={"equation_type": "not_a_kind"})

    def test_invalid_case_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._export(meta={"equation_type": "addition", "case": "not_a_case"})

    def test_div_bracket_fine_label_derives_divide_bracket_coarse(self) -> None:
        # The long-division anchor: fine "div_bracket" -> YOLO "divide_bracket".
        # flattened_to_yolo_class_name raises on it, so the exporter maps it
        # explicitly; assert the GT yolo_class and the .txt class id are correct.
        out = self._export(
            stem="rt-division-long-001",
            drafts=[_draft(bbox=(40.0, 40.0, 120.0, 200.0), fine_label="div_bracket")],
            meta={"equation_type": "division", "case": "division-long"},
        )
        doc = json.loads(out["gt"].read_text())
        self.assertEqual(doc["symbols"][0]["yolo_class"], "divide_bracket")
        written_id = int(out["txt"].read_text().split()[0])
        self.assertEqual(written_id, YOLO_NAME_TO_ID["divide_bracket"])

    def test_unknown_fine_label_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._export(drafts=[_draft(fine_label="zzz")])

    def test_out_of_bounds_bbox_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._export(drafts=[_draft(bbox=(10.0, 10.0, 9.0, 80.0))])

    def test_empty_drafts_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._export(drafts=[])

    def test_png_is_512_grayscale(self) -> None:
        out = self._export()
        with Image.open(out["png"]) as im:
            self.assertEqual(im.size, (512, 512))
            self.assertEqual(im.mode, "L")

    def test_no_tmp_files_left(self) -> None:
        self._export()
        self.assertEqual(list(train_dir(self.root).glob("*.tmp")), [])

    def test_reexport_same_stem_does_not_overwrite(self) -> None:
        # Collision bump (data-corruption guard): a second save with the SAME stem
        # must NOT clobber any of the first scene's three artifacts. The first keeps
        # its name + content; the second lands on a bumped ``-NNN`` stem.
        first = self._export(meta={"equation_type": "addition"})
        second = self._export(meta={"equation_type": "subtraction"})

        # First scene untouched.
        first_doc = json.loads(first["gt"].read_text())
        self.assertEqual(first_doc["equation_type"], "addition")
        self.assertEqual(first["gt"].name, "rt-addition-001.gt.json")

        # Second scene on a bumped stem; all three artifacts renamed together.
        self.assertEqual(second["gt"].name, "rt-addition-001-001.gt.json")
        self.assertEqual(second["png"].name, "rt-addition-001-001.png")
        self.assertEqual(second["txt"].name, "rt-addition-001-001.txt")
        for key in ("png", "gt", "txt"):
            self.assertNotEqual(second[key], first[key])
            self.assertTrue(first[key].exists())
            self.assertTrue(second[key].exists())
        second_doc = json.loads(second["gt"].read_text())
        self.assertEqual(second_doc["equation_type"], "subtraction")

        self.assertEqual(list(train_dir(self.root).glob("*.tmp")), [])

    def test_unsafe_stem_rejected_before_write(self) -> None:
        with self.assertRaises(ValueError):
            export_train("a/b", _canvas(), _addition_drafts(),
                         {"equation_type": "addition"}, self.root)


class TestFreeStemHelper(unittest.TestCase):
    """``_free_stem`` finds a non-clobbering name before any write."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_free_stem_unchanged_when_nothing_exists(self) -> None:
        self.assertEqual(_free_stem(self.dir, "re-addition-001", (".png", ".label.json")), "re-addition-001")

    def test_bumps_to_001_on_collision(self) -> None:
        (self.dir / "re-addition-001.png").write_bytes(b"x")
        (self.dir / "re-addition-001.label.json").write_text("{}")
        self.assertEqual(_free_stem(self.dir, "re-addition-001", (".png", ".label.json")), "re-addition-001-001")

    def test_bumps_past_existing_indexed_stems(self) -> None:
        # Base + -001 + -002 already taken -> next free is -003.
        for stem in ("re-addition-001", "re-addition-001-001", "re-addition-001-002"):
            (self.dir / f"{stem}.png").write_bytes(b"x")
            (self.dir / f"{stem}.label.json").write_text("{}")
        self.assertEqual(_free_stem(self.dir, "re-addition-001", (".png", ".label.json")), "re-addition-001-003")

    def test_partial_artifact_presence_still_collides(self) -> None:
        # Only ONE of the two artifacts present is enough to treat the stem as taken,
        # so a crash-orphaned sibling can never be half-overwritten.
        (self.dir / "rt-addition-001.txt").write_text("0 0.5 0.5 0.1 0.1")
        self.assertEqual(
            _free_stem(self.dir, "rt-addition-001", (".png", ".gt.json", ".txt")),
            "rt-addition-001-001",
        )


class TestSaveTwiceDoesNotOverwrite(unittest.TestCase):
    """End-to-end: a duplicate client-supplied stem never corrupts a prior sample."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_eval_three_saves_same_stem_yield_three_distinct_samples(self) -> None:
        kinds = ["addition", "subtraction", "multiplication"]
        outs = [
            export_eval("re-addition-001", _canvas(), _addition_drafts(), {"equation_kind": k}, self.root)
            for k in kinds
        ]
        names = [o["label"].name for o in outs]
        self.assertEqual(
            names,
            ["re-addition-001.label.json", "re-addition-001-001.label.json", "re-addition-001-002.label.json"],
        )
        # Each on-disk sidecar kept its own content (no clobber across the chain).
        for out, kind in zip(outs, kinds):
            self.assertEqual(json.loads(out["label"].read_text())["equation_kind"], kind)
        # All three load under their own stems via the canonical loader.
        loaded = load_label_sidecars(eval_dir(self.root))
        self.assertEqual(
            {"re-addition-001", "re-addition-001-001", "re-addition-001-002"},
            set(loaded),
        )

    def test_train_three_saves_same_stem_yield_three_distinct_sample_sets(self) -> None:
        types_ = ["addition", "subtraction", "multiplication"]
        outs = [
            export_train("rt-addition-001", _canvas(), _addition_drafts(), {"equation_type": t}, self.root)
            for t in types_
        ]
        gt_names = [o["gt"].name for o in outs]
        self.assertEqual(
            gt_names,
            ["rt-addition-001.gt.json", "rt-addition-001-001.gt.json", "rt-addition-001-002.gt.json"],
        )
        for out, etype in zip(outs, types_):
            self.assertEqual(json.loads(out["gt"].read_text())["equation_type"], etype)
        # 3 samples x 3 artifacts each = 9 files, no temp residue.
        self.assertEqual(len(list(train_dir(self.root).glob("rt-addition-001*"))), 9)
        self.assertEqual(list(train_dir(self.root).glob("*.tmp")), [])

    def test_bumped_stem_is_still_sanitisation_safe(self) -> None:
        # The bump appends only "-NNN", so the result passes sanitize_stem too.
        out = export_eval("re-addition-001", _canvas(), _addition_drafts(),
                          {"equation_kind": "addition"}, self.root)
        export_eval("re-addition-001", _canvas(), _addition_drafts(),
                    {"equation_kind": "addition"}, self.root)
        bumped = (eval_dir(self.root) / "re-addition-001-001.png").stem
        self.assertEqual(sanitize_stem(bumped), bumped)
        self.assertTrue(out["png"].exists())


class TestTrainExportErrorKind(unittest.TestCase):
    """export_train writes / omits / rejects error_kind correctly (mirrors eval)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _export(self, stem="rt-addition-001", meta=None):
        base_meta = {
            "equation_type": "addition",
            "completion_stage": "full",
            "case": "addition",
            "split": "train",
        }
        if meta is not None:
            base_meta.update(meta)
        return export_train(stem, _canvas(), _addition_drafts(), base_meta, self.root)

    def test_error_kind_written_when_valid(self) -> None:
        # A valid error_kind lands in the train GT under the "error_kind" key.
        out = self._export(meta={"error_kind": "wrong_operator"})
        doc = json.loads(out["gt"].read_text())
        self.assertEqual(doc["error_kind"], "wrong_operator")

    def test_every_error_kind_value_accepted(self) -> None:
        from src.eval.labels import ERROR_KINDS
        for i, kind in enumerate(sorted(ERROR_KINDS)):
            stem = f"rt-err-{i:03d}"
            out = self._export(stem=stem, meta={"error_kind": kind})
            doc = json.loads(out["gt"].read_text())
            self.assertEqual(doc["error_kind"], kind)

    def test_error_kind_omitted_when_absent(self) -> None:
        # No error_kind in meta -> key must not appear in the GT.
        out = self._export()
        doc = json.loads(out["gt"].read_text())
        self.assertNotIn("error_kind", doc)

    def test_error_kind_omitted_when_none(self) -> None:
        out = self._export(meta={"error_kind": None})
        doc = json.loads(out["gt"].read_text())
        self.assertNotIn("error_kind", doc)

    def test_invalid_error_kind_rejected_and_nothing_written(self) -> None:
        # A bad error_kind raises ValueError before any file is written.
        with self.assertRaises(ValueError) as ctx:
            self._export(meta={"error_kind": "not_a_real_error"})
        self.assertIn("error_kind", str(ctx.exception))
        train_d = self.root / "data" / "setmaker" / "train"
        if train_d.exists():
            self.assertFalse(list(train_d.glob("*.gt.json")))

    def test_normal_train_gt_shape_unchanged(self) -> None:
        # A normal (no error_kind) export keeps only the original 5 top-level keys.
        out = self._export()
        doc = json.loads(out["gt"].read_text())
        self.assertEqual(
            set(doc.keys()),
            {"equation_type", "completion_stage", "split", "case", "symbols"},
        )


class TestTrainEvalConstantsAlignWithLabels(unittest.TestCase):
    """The exporter's accepted sets stay in lockstep with src/eval/labels.py."""

    def test_train_equation_types_superset_of_equation_kinds(self) -> None:
        # Every eval equation kind is a valid train equation_type, plus the OOD one.
        self.assertTrue(EQUATION_KINDS.issubset(TRAIN_EQUATION_TYPES))
        self.assertIn(OOD_EQUATION_TYPE, TRAIN_EQUATION_TYPES)
        self.assertEqual(TRAIN_EQUATION_TYPES, frozenset(EQUATION_KINDS) | {OOD_EQUATION_TYPE})

    def test_module_exposes_documented_symbols(self) -> None:
        for name in ("export_eval", "export_train", "sanitize_stem",
                     "eval_dir", "train_dir", "OOD_EQUATION_TYPE"):
            self.assertTrue(hasattr(exporters, name), f"missing public symbol {name}")


if __name__ == "__main__":
    unittest.main()
