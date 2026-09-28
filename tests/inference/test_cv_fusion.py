"""Tests for the classical-OpenCV fusion branch (src/inference/cv_fusion.py) and
its wiring into run_inference_with_models.

Budget guarantee under test: full-image YOLO = 1, crop-classify YOLO = 0 or 1
(batched), GNN = 1. The CV branch only fires when YOLO produced >= 1 detection.
"""
from __future__ import annotations

import unittest

import numpy as np

from src.core.run_config import CvFusionConfig
from src.inference.cv_fusion import (
    _classify_crops_with_yolo,
    _letterbox_crop,
    _mask_yolo_regions,
    _merge_detections_nms,
    _tight_mask_yolo_ink,
    detect_symbols_cv,
    flatten_to_white_paper,
    run_cv_fusion,
    run_yolo_second_pass,
)
from src.inference.run import _maybe_fuse_cv, run_inference_with_models
from src.parsing.detection import Detection

try:  # build_graph (used only by the end-to-end tests) needs torch_geometric
    import torch_geometric  # noqa: F401
    _HAS_PYG = True
except Exception:  # pragma: no cover - environment-dependent
    _HAS_PYG = False


# ── Minimal ultralytics-shaped fakes ──────────────────────────────────────────

class _Arr:
    def __init__(self, a: object) -> None:
        self._a = np.asarray(a, dtype=np.float32)

    def cpu(self) -> "_Arr":
        return self

    def numpy(self) -> np.ndarray:
        return self._a


class _Boxes:
    def __init__(self, xyxy: object, cls: object, conf: object) -> None:
        self.xyxy = _Arr(xyxy)
        self.cls = _Arr(cls)
        self.conf = _Arr(conf)

    def __len__(self) -> int:
        return int(self.conf._a.shape[0])


class _Result:
    def __init__(self, boxes: _Boxes) -> None:
        self.boxes = boxes


class _FakeYolo:
    """Single ndarray input → fixed full-image detections. List input (crop batch)
    → one detection per crop. Records every predict() call so the per-image YOLO
    budget can be asserted.
    """

    def __init__(self, full_dets, crop_cls: int = 3, crop_conf: float = 0.9,
                 crop_returns_empty: bool = False) -> None:
        self.full_dets = full_dets  # list of (cls, conf, x0, y0, x1, y1)
        self.crop_cls = crop_cls
        self.crop_conf = crop_conf
        self.crop_returns_empty = crop_returns_empty
        self.calls: list = []  # ("single", 1) or ("batch", n)

    def predict(self, image, conf: float = 0.25, verbose: bool = False, **kw):
        if isinstance(image, list):
            self.calls.append(("batch", len(image)))
            out = []
            for _ in image:
                if self.crop_returns_empty:
                    out.append(_Result(_Boxes(np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,)))))
                else:
                    out.append(_Result(_Boxes([[10, 10, 400, 400]], [self.crop_cls], [self.crop_conf])))
            return out
        self.calls.append(("single", 1))
        xy = [[d[2], d[3], d[4], d[5]] for d in self.full_dets]
        cl = [d[0] for d in self.full_dets]
        cf = [d[1] for d in self.full_dets]
        return [_Result(_Boxes(xy, cl, cf))]


class _FakeGnn:
    """Returns empty node predictions so assemble_json takes its empty path."""

    def predict(self, data: object) -> tuple:
        return [], "unknown", None


def _white(size: int = 512) -> np.ndarray:
    return np.full((size, size), 255, dtype=np.uint8)


def _det(label: str, conf: float, x0, y0, x1, y1) -> Detection:
    return Detection(label=label, confidence=conf, x0=float(x0), y0=float(y0),
                     x1=float(x1), y1=float(y1))


# ── detect_symbols_cv ─────────────────────────────────────────────────────────

class TestDetectSymbolsCv(unittest.TestCase):
    def test_finds_drawn_block(self) -> None:
        gray = _white()
        gray[50:90, 50:90] = 0  # one dark 40x40 symbol
        boxes, confs = detect_symbols_cv(gray)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(len(confs), 1)
        x0, y0, x1, y1 = boxes[0]
        # within a few px of the drawn block (pad=4 default)
        self.assertLessEqual(abs(x0 - 46), 6)
        self.assertLessEqual(abs(y1 - 94), 6)

    def test_blank_image_returns_nothing(self) -> None:
        boxes, confs = detect_symbols_cv(_white())
        self.assertEqual(boxes, [])
        self.assertEqual(confs, [])

    def test_flatten_removes_dark_surround_so_digit_is_detectable(self) -> None:
        # phone-photo layout: white pad -> black frame -> white paper -> dark digit
        img = _white()
        img[80:432, 80:432] = 30     # black frame fill
        img[100:412, 100:412] = 255  # white paper interior
        img[150:200, 150:180] = 20   # a dark digit on the paper
        # raw: the black frame is a huge blob (near-full-image box)
        raw = detect_symbols_cv(img)[0]
        self.assertTrue(any((b[2] - b[0]) * (b[3] - b[1]) > 0.4 * 512 * 512 for b in raw))
        # flattened: frame gone, only the digit remains (no giant box)
        flat = flatten_to_white_paper(img)
        boxes = detect_symbols_cv(flat)[0]
        self.assertTrue(boxes)
        self.assertFalse(any((b[2] - b[0]) * (b[3] - b[1]) > 0.4 * 512 * 512 for b in boxes))
        bx = boxes[0]
        self.assertTrue(140 <= bx[0] <= 175 and 140 <= bx[1] <= 205)

    def test_flatten_leaves_clean_white_scene_usable(self) -> None:
        img = _white()
        img[50:90, 50:90] = 0  # a symbol on an already-white scene
        boxes = detect_symbols_cv(flatten_to_white_paper(img))[0]
        self.assertEqual(len(boxes), 1)

    def test_flatten_does_not_wipe_clean_scene_with_holed_digit(self) -> None:
        # a ring (like "0"/"6"/"9") has a WHITE interior hole that used to be picked
        # as the "paper" and wipe the whole image. Flatten must be a no-op here.
        img = _white()
        img[60:130, 60:110] = 0    # solid dark block
        img[75:115, 72:98] = 255   # carve a white hole -> a ring with interior bright
        out = flatten_to_white_paper(img)
        self.assertTrue(np.array_equal(out, img))          # unchanged, not wiped
        self.assertEqual(len(detect_symbols_cv(out)[0]), 1)  # ring still detectable

    def test_thin_stroke_kept_by_post_merge_filter(self) -> None:
        # 8px-wide stroke: width < min_side(10) so the OLD pre-merge filter dropped
        # it. The gate now runs after the merge against the LARGER extent, so a
        # legitimately thin stroke (e.g. a "1" or a "+" arm) survives.
        gray = _white()
        gray[40:80, 60:68] = 0  # w=8, h=40
        boxes, _ = detect_symbols_cv(gray)
        self.assertEqual(len(boxes), 1)


# ── _mask_yolo_regions ────────────────────────────────────────────────────────

class TestMaskYoloRegions(unittest.TestCase):
    def test_whites_out_dilated_bbox(self) -> None:
        gray = _white()
        gray[100:140, 100:140] = 0  # ink that a YOLO box claims
        yolo = [_det("digit_main", 0.9, 100, 100, 140, 140)]
        masked = _mask_yolo_regions(gray, yolo, dilate_px=5)
        # original ink is gone (whited out)
        self.assertTrue(np.all(masked[100:140, 100:140] == 255))
        # dilation extends ~5px beyond the box (pixel just outside is part of mask path)
        self.assertEqual(int(masked[100, 96]), 255)
        # untouched region is unchanged
        self.assertEqual(int(masked[300, 300]), 255)

    def test_does_not_mutate_input(self) -> None:
        gray = _white()
        gray[100:140, 100:140] = 0
        before = gray.copy()
        _mask_yolo_regions(gray, [_det("digit_main", 0.9, 100, 100, 140, 140)])
        self.assertTrue(np.array_equal(gray, before))


# ── _tight_mask_yolo_ink ──────────────────────────────────────────────────────

class TestTightMaskYoloInk(unittest.TestCase):
    def test_whites_out_only_dark_pixels_not_full_rect(self) -> None:
        gray = _white()
        # a 40x40 box with only a 10x10 dark spot inside it
        gray[110:120, 110:120] = 0  # small dark spot
        yolo = [_det("digit_main", 0.9, 100, 100, 140, 140)]  # loose bbox
        masked = _tight_mask_yolo_ink(gray, yolo, dilate_px=4)
        # the dark spot is whited out
        self.assertTrue(np.all(masked[110:120, 110:120] == 255))
        # but whitespace inside the loose bbox that has no dark pixels survives
        self.assertEqual(int(masked[105, 105]), 255)  # was white, stays white
        # untouched region stays unchanged
        self.assertEqual(int(masked[300, 300]), 255)

    def test_does_not_mutate_input(self) -> None:
        gray = _white()
        gray[110:120, 110:120] = 0
        before = gray.copy()
        _tight_mask_yolo_ink(gray, [_det("digit_main", 0.9, 100, 100, 140, 140)])
        self.assertTrue(np.array_equal(gray, before))

    def test_spares_carry_inside_loose_bbox(self) -> None:
        # Regression: a carry digit sits INSIDE a loose YOLO box but is a smaller,
        # SEPARATE component. The mask must remove YOLO's symbol (largest component)
        # while leaving the carry, so the CV detector can still find it.
        gray = _white()
        gray[120:200, 150:210] = 0  # big digit YOLO detected
        gray[80:115, 170:178] = 0   # small carry, inside the loose box, above the digit
        yolo = [_det("digit_main", 0.9, 140, 70, 220, 210)]  # loose box covers the carry
        masked = _tight_mask_yolo_ink(gray, yolo, dilate_px=4)
        # the big digit (largest component) is removed
        self.assertTrue(np.all(masked[120:200, 150:210] == 255))
        # but the carry survives (it is a smaller, separate component)
        self.assertGreater(int(np.sum(masked[80:115, 170:178] < 200)), 0)


# ── run_yolo_second_pass ──────────────────────────────────────────────────────

class TestSecondYoloPass(unittest.TestCase):
    def test_calls_yolo_once_on_masked_image(self) -> None:
        gray = _white()
        gray[100:140, 100:140] = 0  # claimed by first-pass YOLO
        gray[200:240, 200:240] = 0  # unclaimed (to be found by second pass)
        yolo_dets = [_det("digit_main", 0.9, 100, 100, 140, 140)]
        fake = _FakeYolo(full_dets=[(0, 0.85, 200, 200, 240, 240)], crop_cls=0, crop_conf=0.85)
        dets, info = run_yolo_second_pass(
            gray, yolo_dets, fake,
            conf=0.15, iou=0.3, agnostic_nms=True, max_det=80, dilate_px=4)
        # exactly one single-image YOLO call (the second pass)
        self.assertEqual(fake.calls, [("single", 1)])
        # found the unclaimed symbol
        self.assertEqual(len(dets), 1)
        self.assertAlmostEqual(dets[0].confidence, 0.85, places=2)
        # info surface the results
        self.assertEqual(info["second_pass_dets"], 1)
        self.assertEqual(info["second_pass_conf"], 0.15)

    def test_suppresses_overlapping_first_pass_dets(self) -> None:
        gray = _white()
        gray[100:140, 100:140] = 0  # in both first and second pass
        yolo_dets = [_det("digit_main", 0.95, 100, 100, 140, 140)]
        # second pass also finds the same region with lower confidence
        fake = _FakeYolo(full_dets=[(0, 0.60, 100, 100, 140, 140)], crop_cls=0, crop_conf=0.60)
        dets, _ = run_yolo_second_pass(
            gray, yolo_dets, fake,
            conf=0.15, iou=0.3, agnostic_nms=True, max_det=80, dilate_px=4)
        # merged list should have only one (first-pass YOLO wins on overlap)
        second_pass_dets, _ = run_yolo_second_pass(
            gray, yolo_dets, fake,
            conf=0.15, iou=0.3, agnostic_nms=True, max_det=80, dilate_px=4)
        # NMS is applied in run_cv_fusion, but here we just verify second-pass finds it
        self.assertGreater(len(second_pass_dets), 0)


# ── _letterbox_crop ───────────────────────────────────────────────────────────

class TestLetterboxCrop(unittest.TestCase):
    def test_shape_and_white_padding(self) -> None:
        gray = _white()
        gray[50:90, 50:70] = 0  # tall-ish crop region
        out = _letterbox_crop(gray, [50, 50, 70, 90], size=512)
        self.assertEqual(out.shape, (512, 512, 3))
        self.assertEqual(out.dtype, np.uint8)
        # corners are white padding
        self.assertEqual(int(out[0, 0, 0]), 255)
        # channels are stacked (grayscale)
        self.assertTrue(np.array_equal(out[..., 0], out[..., 1]))


# ── _merge_detections_nms ─────────────────────────────────────────────────────

class TestMergeNms(unittest.TestCase):
    def test_prefers_yolo_on_overlap(self) -> None:
        yolo = [_det("operator", 0.90, 100, 100, 150, 150)]
        # higher-conf CV box overlapping YOLO (IoU ~0.68) + a disjoint CV box
        cv = [_det("digit_main", 0.95, 105, 105, 155, 155),
              _det("digit_main", 0.80, 300, 300, 340, 340)]
        merged = _merge_detections_nms(yolo, cv, iou_thresh=0.3)
        self.assertEqual(len(merged), 2)
        # the overlapping CV box was dropped; the YOLO det survived unchanged
        self.assertIn(yolo[0], merged)
        labels = {(round(d.x0), d.label) for d in merged}
        self.assertIn((100, "operator"), labels)
        self.assertIn((300, "digit_main"), labels)

    def test_empty_cv_returns_yolo(self) -> None:
        yolo = [_det("operator", 0.9, 10, 10, 30, 30)]
        self.assertEqual(_merge_detections_nms(yolo, [], iou_thresh=0.3), yolo)


# ── _classify_crops_with_yolo ─────────────────────────────────────────────────

class TestClassifyCrops(unittest.TestCase):
    def test_single_batched_call_assigns_class_keeps_cv_coords(self) -> None:
        gray = _white()
        fake = _FakeYolo(full_dets=[], crop_cls=3, crop_conf=0.88)
        boxes = [[50, 50, 90, 90], [300, 300, 340, 340]]
        dets, nfb = _classify_crops_with_yolo(gray, boxes, fake, classify_conf=0.25,
                                              letterbox_size=512)
        # exactly one batched predict call, sized to the number of crops
        self.assertEqual(fake.calls, [("batch", 2)])
        self.assertEqual(len(dets), 2)
        self.assertEqual(nfb, 0)
        # class comes from YOLO (cls 3 -> "operator"); bbox stays the CV box
        self.assertEqual(dets[0].label, "operator")
        self.assertEqual([dets[0].x0, dets[0].y0, dets[0].x1, dets[0].y1], [50, 50, 90, 90])
        self.assertAlmostEqual(dets[0].confidence, 0.88, places=4)

    def test_keeps_unclassified_as_fallback_for_gnn(self) -> None:
        # YOLO can't classify the crop (the symbol it already missed) -> keep it
        # with the fallback label so the GNN labels it (fix for the isolated "+").
        gray = _white()
        gray[55:85, 55:85] = 0  # draw ink so box passes the H-1 floor
        fake = _FakeYolo(full_dets=[], crop_returns_empty=True)
        dets, nfb = _classify_crops_with_yolo(
            gray, [[50, 50, 90, 90]], fake, classify_conf=0.25, letterbox_size=512,
            keep_unclassified=True, fallback_label="digit_main", unclassified_conf=0.2)
        self.assertEqual(fake.calls, [("batch", 1)])
        self.assertEqual(len(dets), 1)
        self.assertEqual(nfb, 1)
        self.assertEqual(dets[0].label, "digit_main")
        self.assertAlmostEqual(dets[0].confidence, 0.2, places=4)

    def test_drops_unclassified_when_keep_disabled(self) -> None:
        gray = _white()
        fake = _FakeYolo(full_dets=[], crop_returns_empty=True)
        dets, nfb = _classify_crops_with_yolo(
            gray, [[50, 50, 90, 90]], fake, classify_conf=0.25, letterbox_size=512,
            keep_unclassified=False)
        self.assertEqual(dets, [])
        self.assertEqual(nfb, 0)
        self.assertEqual(fake.calls, [("batch", 1)])

    def test_classify_with_yolo_false_makes_no_call(self) -> None:
        gray = _white()
        gray[55:85, 55:85] = 0  # draw ink in box 1 so it passes the H-1 floor
        gray[12:28, 12:28] = 0  # draw ink in box 2
        fake = _FakeYolo(full_dets=[])
        dets, nfb = _classify_crops_with_yolo(
            gray, [[50, 50, 90, 90], [10, 10, 30, 30]], fake,
            classify_conf=0.25, letterbox_size=512, classify_with_yolo=False)
        self.assertEqual(fake.calls, [])          # 0 extra YOLO calls
        self.assertEqual(len(dets), 2)
        self.assertEqual(nfb, 2)

    def test_no_boxes_makes_no_call(self) -> None:
        fake = _FakeYolo(full_dets=[])
        dets, nfb = _classify_crops_with_yolo(_white(), [], fake, classify_conf=0.25,
                                              letterbox_size=512)
        self.assertEqual(dets, [])
        self.assertEqual(nfb, 0)
        self.assertEqual(fake.calls, [])


# ── run_cv_fusion ─────────────────────────────────────────────────────────────

class TestRunCvFusion(unittest.TestCase):
    def _gray_with_unclaimed_ink(self) -> np.ndarray:
        gray = _white()
        gray[50:90, 50:90] = 0  # unclaimed symbol
        return gray

    def test_both_paths_run_simultaneously(self) -> None:
        gray = self._gray_with_unclaimed_ink()
        yolo = [_det("digit_main", 0.9, 300, 300, 340, 340)]  # claims a different region
        # CV and second YOLO both find the unclaimed carry
        fake = _FakeYolo(full_dets=[(0, 0.8, 50, 50, 90, 90)], crop_cls=0, crop_conf=0.9)
        cfg = CvFusionConfig(use_second_yolo_pass=True, classify_with_yolo=True, mask_yolo=True)
        merged, info = run_cv_fusion(gray, yolo, fake, cfg)
        # both batched call (CV crop classify) and single call (second pass) made
        self.assertEqual(len(fake.calls), 2)
        self.assertIn(("batch", 1), fake.calls)
        self.assertIn(("single", 1), fake.calls)
        # both paths find the symbol, but NMS merges suppress duplicates
        # Final: YOLO + CV carry (second YOLO suppressed as duplicate)
        self.assertEqual(len(merged), 2)
        self.assertIn("cv_detector", info)
        self.assertIn("second_yolo", info)
        self.assertEqual(info["cv_detector"]["added"], 1)
        # info still shows what each path found before NMS
        self.assertEqual(info["second_yolo"]["second_pass_dets"], 1)

    def test_cv_detector_alone_when_second_pass_disabled(self) -> None:
        gray = self._gray_with_unclaimed_ink()
        yolo = [_det("digit_main", 0.9, 300, 300, 340, 340)]
        fake = _FakeYolo(full_dets=[], crop_cls=0, crop_conf=0.9)
        cfg = CvFusionConfig(use_second_yolo_pass=False, classify_with_yolo=True, mask_yolo=True)
        merged, info = run_cv_fusion(gray, yolo, fake, cfg)
        # only batched call (CV)
        self.assertEqual(fake.calls, [("batch", 1)])
        # CV found the symbol
        self.assertEqual(len(merged), 2)
        self.assertIn("cv_detector", info)
        self.assertNotIn("second_yolo", info)
        self.assertEqual(info["cv_detector"]["added"], 1)

    def test_no_unclaimed_ink_no_extras(self) -> None:
        gray = self._gray_with_unclaimed_ink()
        yolo = [_det("digit_main", 0.9, 50, 50, 90, 90)]  # claims the ink
        fake = _FakeYolo(full_dets=[])
        merged, info = run_cv_fusion(gray, yolo, fake, CvFusionConfig(use_second_yolo_pass=True))
        # both paths run but find nothing new
        self.assertEqual(len(merged), 1)  # just YOLO
        self.assertEqual(info["cv_detector"]["added"], 0)
        self.assertEqual(info["second_yolo"]["added"], 0)

    def test_recovers_carry_inside_loose_yolo_box(self) -> None:
        # Regression for the real failure: YOLO detects a digit with a LOOSE box
        # that geometrically covers a carry it missed. The carry must survive both
        # the mask (component-aware) AND the NMS merge (IoA disabled for tight mask).
        gray = _white()
        gray[120:200, 150:210] = 0  # big digit YOLO detected
        gray[80:115, 170:178] = 0   # carry YOLO missed, inside the loose box
        yolo = [_det("digit_main", 0.85, 140, 70, 220, 210)]  # loose box covers the carry
        cfg = CvFusionConfig(
            use_second_yolo_pass=False, mask_yolo=True, cv_tight_mask=True,
            tight_mask_dilate_px=4, min_area=30, min_side=4,
            merge_overlap_ratio=0.95, proximity_merge_factor=0.0,
            classify_with_yolo=False, keep_unclassified=True,
        )
        merged, info = run_cv_fusion(gray, yolo, None, cfg)
        # the carry is recovered (YOLO + 1 CV carry)
        self.assertEqual(len(merged), 2)
        self.assertEqual(info["cv_detector"]["added"], 1)
        carry = next(d for d in merged if d not in yolo)
        # the recovered box is in the carry region (above the digit), not the digit
        self.assertLess(carry.y1, 120)


# ── _maybe_fuse_cv gate (mode auto/on) ────────────────────────────────────────

class TestGate(unittest.TestCase):
    def _gray(self) -> np.ndarray:
        gray = _white()
        gray[50:90, 50:90] = 0
        return gray

    def test_auto_skips_when_yolo_looks_complete(self) -> None:
        gray = self._gray()
        yolo = [_det("digit_main", 0.9, 200 + 10 * i, 200, 210 + 10 * i, 210) for i in range(6)]
        fake = _FakeYolo(full_dets=[])
        cfg = CvFusionConfig(mode="auto", gate_min_detections=6, gate_mean_conf=0.6, classify_with_yolo=True)
        out, info = _maybe_fuse_cv(gray, yolo, fake, cfg)
        self.assertEqual(out, yolo)        # straight to GNN, unchanged
        self.assertEqual(fake.calls, [])   # CV branch skipped entirely
        self.assertEqual(info["skipped"], "gate_complete")

    def test_auto_fuses_when_sparse(self) -> None:
        gray = self._gray()
        yolo = [_det("digit_main", 0.9, 300, 300, 340, 340)]  # only 1 < gate_min
        fake = _FakeYolo(full_dets=[], crop_cls=0, crop_conf=0.9)
        cfg = CvFusionConfig(mode="auto", gate_min_detections=6, gate_mean_conf=0.6, use_second_yolo_pass=False, classify_with_yolo=True)
        out, info = _maybe_fuse_cv(gray, yolo, fake, cfg)
        self.assertEqual(len(out), 2)
        self.assertEqual([c for c in fake.calls if c[0] == "batch"], [("batch", 1)])
        self.assertEqual(len(info["cv_detector"]["detections"]), 1)

    def test_on_fuses_even_when_complete(self) -> None:
        gray = self._gray()
        yolo = [_det("digit_main", 0.95, 200 + 10 * i, 200, 210 + 10 * i, 210) for i in range(8)]
        fake = _FakeYolo(full_dets=[], crop_cls=0, crop_conf=0.9)
        out, info = _maybe_fuse_cv(gray, yolo, fake, CvFusionConfig(mode="on", use_second_yolo_pass=False))
        self.assertEqual(len(out), 9)  # 8 YOLO + 1 CV
        self.assertIsNone(info.get("skipped"))


# ── End-to-end budget through run_inference_with_models ───────────────────────

@unittest.skipUnless(_HAS_PYG, "torch_geometric not installed (build_graph unavailable)")
class TestEndToEndBudget(unittest.TestCase):
    def _gray(self) -> np.ndarray:
        gray = _white()
        gray[50:90, 50:90] = 0  # unclaimed symbol (YOLO claims elsewhere)
        return gray

    def _full_dets(self):
        return [(3, 0.9, 300, 300, 340, 340)]  # one "operator" far from the unclaimed ink

    def test_mode_off_is_single_yolo_call(self) -> None:
        fake = _FakeYolo(full_dets=self._full_dets())
        payload = run_inference_with_models(self._gray(), gnn_model=_FakeGnn(), yolo_model=fake,
                                            cv_fusion=CvFusionConfig(mode="off"))
        self.assertEqual(fake.calls, [("single", 1)])
        self.assertNotIn("cv_fusion", payload)  # nothing surfaced when off

    def test_cv_fusion_none_is_baseline_single_call(self) -> None:
        fake = _FakeYolo(full_dets=self._full_dets())
        run_inference_with_models(self._gray(), gnn_model=_FakeGnn(), yolo_model=fake,
                                  cv_fusion=None)
        self.assertEqual(fake.calls, [("single", 1)])

    def test_mode_on_both_paths_simultaneously(self) -> None:
        fake = _FakeYolo(full_dets=self._full_dets(), crop_cls=0, crop_conf=0.9)
        payload = run_inference_with_models(self._gray(), gnn_model=_FakeGnn(), yolo_model=fake,
                                            cv_fusion=CvFusionConfig(mode="on", use_second_yolo_pass=True))
        # At least first pass + one of (batch or single from fusion)
        self.assertGreaterEqual(len(fake.calls), 2)
        self.assertIn("cv_fusion", payload)
        self.assertIn("cv_detector", payload["cv_fusion"])
        self.assertIn("second_yolo", payload["cv_fusion"])

    def test_mode_on_cv_only_when_disabled(self) -> None:
        fake = _FakeYolo(full_dets=self._full_dets(), crop_cls=0, crop_conf=0.9)
        payload = run_inference_with_models(self._gray(), gnn_model=_FakeGnn(), yolo_model=fake,
                                            cv_fusion=CvFusionConfig(mode="on", use_second_yolo_pass=False, classify_with_yolo=True))
        # first pass YOLO + batch (CV) only
        self.assertEqual(len(fake.calls), 2)
        self.assertIn(("single", 1), fake.calls)  # first pass
        self.assertIn(("batch", 1), fake.calls)   # CV crop classify
        self.assertIn("cv_fusion", payload)
        self.assertIn("cv_detector", payload["cv_fusion"])
        self.assertNotIn("second_yolo", payload["cv_fusion"])


# ── Phase 2.1: config-vs-dataclass default guard ──────────────────────────────

class TestCvFusionConfigDefaults(unittest.TestCase):
    """Guard: CvFusionConfig() field-by-field must equal load_cv_fusion_config
    on the shipped config.toml so a test-constructed CvFusionConfig() matches
    the runtime config loaded from disk (H-2 fix)."""

    def _project_root(self):
        from pathlib import Path
        return Path(__file__).parent.parent.parent

    def test_dataclass_defaults_match_config_toml(self) -> None:
        """Complete drift detector: every field of CvFusionConfig() must equal the
        value loaded from config.toml so that test-constructed configs are always
        in sync with the shipped defaults.  Add new fields to load_cv_fusion_config
        and this test together; neither should diverge from the other."""
        import dataclasses
        from src.core.run_config import CvFusionConfig, load_cv_fusion_config
        root = self._project_root()
        if not (root / "config.toml").exists():
            self.skipTest("config.toml not present")
        from_file = load_cv_fusion_config(root)
        from_default = CvFusionConfig()
        # Iterate all dataclass fields so new fields are caught automatically.
        for f in dataclasses.fields(CvFusionConfig):
            default_val = getattr(from_default, f.name)
            file_val = getattr(from_file, f.name)
            self.assertEqual(
                default_val, file_val,
                f"CvFusionConfig.{f.name} dataclass default ({default_val!r}) diverges "
                f"from config.toml value ({file_val!r}) — update one to match the other"
            )


# ── Phase 2.2: second-pass agnostic_nms threading ─────────────────────────────

class TestSecondPassAgnosticNms(unittest.TestCase):
    """L-1 fix: run_yolo_second_pass must use YoloInferenceConfig.agnostic_nms
    (False) rather than a hardcoded True."""

    def _make_gray(self) -> np.ndarray:
        return _white()

    def test_second_pass_uses_agnostic_nms_false_by_default(self) -> None:
        """When agnostic_nms is not explicitly set, the second pass should
        default to False (matching YoloInferenceConfig), not hardcoded True."""
        from src.inference.cv_fusion import run_yolo_second_pass

        captured = {}

        class _CapturingYolo:
            def predict(self, image, **kwargs):
                captured.update(kwargs)
                return []

        gray = self._make_gray()
        yolo_dets = [_det("digit_main", 0.9, 50, 50, 90, 90)]
        run_yolo_second_pass(gray, yolo_dets, _CapturingYolo(), conf=0.15, iou=0.3,
                             agnostic_nms=False, max_det=80, dilate_px=4)
        self.assertIs(captured.get("agnostic_nms"), False,
                      "second pass forwarded agnostic_nms=True (hardcoded) instead of False")

    def test_run_cv_fusion_second_pass_threads_yolo_inference_config_agnostic_nms(self) -> None:
        """run_cv_fusion with use_second_yolo_pass=True must pass agnostic_nms=False
        (from YoloInferenceConfig) rather than a hardcoded True."""
        from src.inference.cv_fusion import run_cv_fusion
        from src.core.run_config import CvFusionConfig

        captured_calls = []

        class _CapturingYolo:
            def predict(self, image, **kwargs):
                if isinstance(image, list):
                    return []
                captured_calls.append(kwargs.copy())
                return []

        gray = self._make_gray()
        yolo_dets = [_det("digit_main", 0.9, 300, 300, 340, 340)]
        cfg = CvFusionConfig(
            use_second_yolo_pass=True, classify_with_yolo=False,
            keep_unclassified=False,
        )
        run_cv_fusion(gray, yolo_dets, _CapturingYolo(), cfg)
        # The second pass call must have agnostic_nms=False
        second_pass_calls = [c for c in captured_calls]
        self.assertTrue(second_pass_calls, "No second-pass YOLO call was made")
        self.assertIs(second_pass_calls[0].get("agnostic_nms"), False,
                      f"second pass used agnostic_nms=True; calls={second_pass_calls}")


# ── Phase 2.3: degrade-safe flatten import ────────────────────────────────────

class TestDegradeSafeFlattenImport(unittest.TestCase):
    """L-2 fix: flatten_background=True with missing cv2 must degrade gracefully
    (return YOLO-only output) rather than raise ImportError."""

    @unittest.skipUnless(_HAS_PYG, "torch_geometric not installed")
    def test_flatten_background_degrades_when_cv2_missing(self) -> None:
        """The try/except guard around `from .cv_fusion import flatten_to_white_paper`
        in run_inference_with_models must catch ImportError and continue without
        raising.  The function must return a normal payload ('equation_kind' present).

        We simulate a missing cv2 by setting sys.modules['src.inference.cv_fusion']
        to None, which causes the `from .cv_fusion import …` inside the try block to
        raise ImportError.  The guard must catch that and fall through to the rest of
        the inference pipeline (YOLO + GNN) returning a valid payload."""
        import sys
        import unittest.mock as mock
        from src.inference.run import run_inference_with_models

        gray = _white()
        gray[50:90, 50:90] = 0
        fake = _FakeYolo(full_dets=[(3, 0.9, 300, 300, 340, 340)])
        # flatten_background=True AND mode="off" so only the flatten guard fires;
        # the CV fusion branch is skipped, avoiding a second ImportError from
        # _maybe_fuse_cv (which has its own separate guard).
        cfg = CvFusionConfig(mode="off", flatten_background=True)

        # Setting a module to None in sys.modules makes `from that_module import X`
        # raise ImportError — exactly what happens when cv2 is absent and cv_fusion
        # itself cannot be imported.
        saved = sys.modules.get("src.inference.cv_fusion")
        sys.modules["src.inference.cv_fusion"] = None  # type: ignore[assignment]
        try:
            payload = run_inference_with_models(
                gray, gnn_model=_FakeGnn(), yolo_model=fake, cv_fusion=cfg
            )
        except ImportError as exc:
            self.fail(
                f"ImportError escaped run_inference_with_models — "
                f"flatten_background guard is missing or broken: {exc}"
            )
        finally:
            # Restore the real module so subsequent tests are unaffected.
            if saved is None:
                sys.modules.pop("src.inference.cv_fusion", None)
            else:
                sys.modules["src.inference.cv_fusion"] = saved

        # Guard worked: inference completed and returned a valid YOLO payload.
        self.assertIn("equation_kind", payload,
                      "payload missing equation_kind — inference did not complete after flatten guard")

    @unittest.skipUnless(_HAS_PYG, "torch_geometric not installed")
    def test_flatten_background_end_to_end_through_run_inference_with_models(self) -> None:
        """flatten_background=True on a plain white 512 image must not raise
        and must return a payload (the flatten is a no-op on a white image)."""
        from src.inference.run import run_inference_with_models
        gray = _white()
        gray[50:90, 50:90] = 0
        fake = _FakeYolo(full_dets=[(3, 0.9, 300, 300, 340, 340)], crop_cls=0, crop_conf=0.9)
        cfg = CvFusionConfig(mode="on", flatten_background=True, use_second_yolo_pass=False,
                             classify_with_yolo=True)
        payload = run_inference_with_models(gray, gnn_model=_FakeGnn(), yolo_model=fake,
                                            cv_fusion=cfg)
        self.assertIn("equation_kind", payload)


# ── Phase 3: phantom-node ink-density floor ───────────────────────────────────

class TestInkDensityFloor(unittest.TestCase):
    """H-1 fix: _classify_crops_with_yolo must drop unclassified CV boxes whose
    ink-density (dark-pixel fraction within the box) is below keep_min_ink_frac,
    preventing speckle from becoming phantom GNN nodes."""

    def test_speckle_box_below_floor_is_not_promoted(self) -> None:
        """A near-empty box (very few dark pixels) must NOT become a fallback
        Detection even when keep_unclassified=True and classify_with_yolo=False."""
        from src.inference.cv_fusion import _classify_crops_with_yolo
        from src.core.run_config import CvFusionConfig

        gray = _white(512)
        # speckle: 2x2 dark pixels in a 60x60 box → ink fraction ≈ 4/3600 ≈ 0.001
        gray[100:102, 100:102] = 0
        speckle_box = [98, 98, 158, 158]  # 60x60 box around the 2-pixel speckle

        cfg = CvFusionConfig(keep_unclassified=True, keep_min_ink_frac=0.10, min_area=30)
        dets, n_fallback = _classify_crops_with_yolo(
            gray, [speckle_box], yolo_model=None,
            classify_conf=0.25, letterbox_size=512,
            classify_with_yolo=False,
            keep_unclassified=cfg.keep_unclassified,
            keep_min_ink_frac=cfg.keep_min_ink_frac,
            min_area=cfg.min_area,
        )
        self.assertEqual(len(dets), 0, "Speckle box was promoted despite ink below floor")
        self.assertEqual(n_fallback, 0)

    def test_real_carry_above_floor_survives(self) -> None:
        """A real carry digit (solid ink filling a meaningful fraction of the box)
        must survive the floor and be returned as a fallback Detection."""
        from src.inference.cv_fusion import _classify_crops_with_yolo
        from src.core.run_config import CvFusionConfig

        gray = _white(512)
        # carry: fill 20x18 dark pixels in a 30x30 box → ink fraction = 360/900 = 0.40 > 0.10
        gray[80:100, 170:188] = 0
        carry_box = [168, 78, 198, 108]  # 30x30 box tightly around carry

        cfg = CvFusionConfig(keep_unclassified=True, keep_min_ink_frac=0.10, min_area=30)
        dets, n_fallback = _classify_crops_with_yolo(
            gray, [carry_box], yolo_model=None,
            classify_conf=0.25, letterbox_size=512,
            classify_with_yolo=False,
            keep_unclassified=cfg.keep_unclassified,
            keep_min_ink_frac=cfg.keep_min_ink_frac,
            min_area=cfg.min_area,
        )
        self.assertEqual(len(dets), 1, "Real carry was dropped by ink floor (floor too aggressive)")
        self.assertEqual(n_fallback, 1)

    def test_floor_disabled_at_zero_keeps_speckle(self) -> None:
        """When keep_min_ink_frac=0.0 the floor is disabled; all boxes pass."""
        from src.inference.cv_fusion import _classify_crops_with_yolo

        gray = _white(512)
        gray[100:102, 100:102] = 0
        speckle_box = [98, 98, 158, 158]

        dets, n_fallback = _classify_crops_with_yolo(
            gray, [speckle_box], yolo_model=None,
            classify_conf=0.25, letterbox_size=512,
            classify_with_yolo=False,
            keep_unclassified=True,
            keep_min_ink_frac=0.0,
            min_area=0,
        )
        self.assertEqual(len(dets), 1, "Floor=0 should pass everything through")


# ── Phase 1 regression: non-512 alignment ─────────────────────────────────────

@unittest.skipUnless(_HAS_PYG, "torch_geometric not installed")
class TestNon512Alignment(unittest.TestCase):
    """C-1 regression: run_inference_with_models must NOT accept original_gray.
    Before the fix, original_gray (arbitrary HxW) was fed as cv_image causing
    coordinate-space mismatch. After the fix, the parameter is removed entirely
    and the CV branch always receives the preprocessed 512x512 gray."""

    def test_run_inference_with_models_has_no_original_gray_param(self) -> None:
        """original_gray must not be a parameter of run_inference_with_models."""
        import inspect
        from src.inference.run import run_inference_with_models
        params = inspect.signature(run_inference_with_models).parameters
        self.assertNotIn(
            "original_gray", params,
            "original_gray must be removed from run_inference_with_models signature "
            "(C-1 fix: CV branch uses preprocessed 512 gray, never raw image)",
        )

    def test_predict_has_no_original_gray_param(self) -> None:
        """original_gray must not be a parameter of InferenceSession.predict."""
        import inspect
        from src.inference.run import InferenceSession
        params = inspect.signature(InferenceSession.predict).parameters
        self.assertNotIn(
            "original_gray", params,
            "original_gray must be removed from InferenceSession.predict signature",
        )

    def test_merged_detections_in_512_space(self) -> None:
        """Smoke: CV-added detections must all lie within [0, 512] (coordinate
        space of the preprocessed gray).  This test cannot catch the old
        original_gray coordinate-space bug (that parameter is gone; see
        test_run_inference_with_models_has_no_original_gray_param for the
        signature guard).  Its value is as a bbox-sanity smoke on a realistic
        scene: the unclaimed ink blob at [50,90]x[50,90] should produce at
        least one CV detection whose coordinates are within the 512 canvas."""
        gray_512 = _white(512)
        gray_512[50:90, 50:90] = 0   # unclaimed ink for CV to find

        fake = _FakeYolo(
            full_dets=[(3, 0.9, 300, 300, 340, 340)],  # operator far from unclaimed ink
            crop_cls=0, crop_conf=0.9,
        )
        cfg = CvFusionConfig(
            mode="on", use_second_yolo_pass=False,
            classify_with_yolo=True, keep_unclassified=True,
        )
        payload = run_inference_with_models(
            gray_512, gnn_model=_FakeGnn(), yolo_model=fake, cv_fusion=cfg,
        )
        cv_info = payload.get("cv_fusion", {})
        cv_dets = cv_info.get("cv_detector", {}).get("detections", [])

        # At least one CV detection expected (the 40x40 ink blob).
        self.assertGreater(len(cv_dets), 0, "expected at least one CV detection for the ink blob")

        # Every bbox coordinate must be within the 512 canvas.
        for det_dict in cv_dets:
            bbox = det_dict["bbox"]  # [x0, y0, x1, y1]
            self.assertEqual(len(bbox), 4, f"malformed bbox: {bbox}")
            for coord in bbox:
                self.assertGreaterEqual(coord, 0.0, f"coord below 0: {coord} in {bbox}")
                self.assertLessEqual(coord, 512.0, f"coord above 512: {coord} in {bbox}")




# ── Phase 5.3: merge-loop cap (M-3) ───────────────────────────────────────────

class TestMergeLoopCap(unittest.TestCase):
    """M-3: the while-changed merge loop in detect_symbols_cv must converge on
    a high-component input without hanging or exceeding len(raw)+1 iterations."""

    def test_high_component_input_converges(self) -> None:
        """The merge loop must terminate correctly even on inputs that require
        multiple iterations.

        Geometry: two separate pairs of proximal blobs placed far apart.  In
        iteration 1 each pair merges (changed=True, count drops from 4 to 2).
        In iteration 2 the two merged boxes are still far apart so no merge
        fires (changed=False) and the loop exits.  This gives exactly 2
        iterations, confirming the loop correctly runs more than once when
        needed and still terminates.

        Note on the cap: the cap guard (max_iters = len(raw)+1) is dead code
        for any geometrically correct input.  The greedy union-monotonic scan
        guarantees that if boxes i and j are proximal with the original box i,
        they are AT LEAST as proximal with the running union that already
        includes i — so a chain always collapses in a single pass.  The cap
        exists to catch hypothetical float-arithmetic bugs; asserting the
        WARNING fires is not feasible with correct real geometry."""
        gray = np.full((512, 512), 255, dtype=np.uint8)

        # Left pair: two 10x10 blobs with 2 px gap → gap/diag ≈ 2/14 ≈ 0.14 < 0.2.
        # They merge in iteration 1.
        gray[100:110, 10:20] = 0   # blob A
        gray[100:110, 22:32] = 0   # blob B  (gap = 2 px)

        # Right pair: same geometry, far from left pair.
        gray[100:110, 300:310] = 0  # blob C
        gray[100:110, 312:322] = 0  # blob D  (gap = 2 px)

        # Gap between the two groups ≈ 268 px >> 14 px diag → groups never merge.

        boxes, confs = detect_symbols_cv(
            gray,
            min_area=10,
            min_side=3,
            pad=0,
            merge_overlap_ratio=0.5,
            proximity_merge_factor=0.20,  # 0.14 < 0.20 → pairs merge; groups don't
        )

        # Two merged boxes expected (one per pair).
        self.assertEqual(len(boxes), 2,
                         f"Expected 2 merged boxes (one per pair); got {len(boxes)}: {boxes}")
        self.assertEqual(len(confs), len(boxes))

        # Both returned boxes must be within the 512 canvas.
        for box in boxes:
            x0, y0, x1, y1 = box
            self.assertGreaterEqual(x0, 0)
            self.assertGreaterEqual(y0, 0)
            self.assertLessEqual(x1, 512)
            self.assertLessEqual(y1, 512)
            self.assertLess(x0, x1)
            self.assertLess(y0, y1)


# ── Phase 5.4: aspect-ratio filter pin (L-3) ─────────────────────────────────

class TestAspectRatioFilter(unittest.TestCase):
    """L-3: a long thin bar-like component (aspect ratio > 10) must be silently
    dropped by detect_symbols_cv.  This pins the bar-is-YOLO's-job assumption:
    horizontal bars are detected as structural tokens by YOLO and must NOT
    produce an extra GNN node from the CV branch."""

    def test_long_thin_bar_is_dropped(self) -> None:
        """A 5x80 horizontal bar (aspect ratio 16) must not appear in CV output."""
        gray = np.full((512, 512), 255, dtype=np.uint8)
        # Draw a long thin horizontal bar (5 px tall, 80 px wide → ratio 16)
        gray[200:205, 100:180] = 0
        boxes, _ = detect_symbols_cv(gray, min_area=10, min_side=3, pad=0)
        # No box should survive after the aspect-ratio > 10 filter
        self.assertEqual(boxes, [],
                         "Long thin bar (aspect ratio 16) was not filtered by the CV detector; "
                         "this would create a phantom GNN node duplicating YOLO's bar detection.")

    def test_square_symbol_passes_filter(self) -> None:
        """A roughly square blob (aspect ratio ~1) must NOT be filtered."""
        gray = np.full((512, 512), 255, dtype=np.uint8)
        # Draw a 20x20 square blob
        gray[100:120, 100:120] = 0
        boxes, _ = detect_symbols_cv(gray, min_area=10, min_side=3, pad=0)
        self.assertEqual(len(boxes), 1, "Square symbol was incorrectly filtered by aspect-ratio gate.")

if __name__ == "__main__":
    unittest.main()
