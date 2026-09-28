"""Run-time configuration for the handwritten-arithmetic-recognition pipeline."""
from __future__ import annotations
import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from ..generation.layouts import SceneCase
_VALID_CASES: set[str] = {c.value for c in SceneCase}
_DEFAULT_CASE_WEIGHTS: dict[str, float] = {c.value: 1.0 for c in SceneCase}
_DEFAULT_COMPLETION: dict[str, float] = {"full":0.65,"done_80":0.15,"done_60":0.10,"done_40":0.06,"done_20":0.04}
_DEFAULT_SCENE: dict[str, Any] = {"wrong_result_prob":0.20,"missing_structural_prob":0.15,"wrong_operator_prob":0.05,"crowdness_prob":0.20,"scene_rotation_min_deg":-4.0,"scene_rotation_max_deg":4.0,"crowding_factor":0.70}
_DEFAULT_PRESET: dict[str, Any] = {"glyph_rotation_min_deg":-4.0,"glyph_rotation_max_deg":4.0,"glyph_jitter_min_px":0,"glyph_jitter_max_px":4,"glyph_broken_stroke_prob":0.05}
_DEFAULT_RENDERING: dict[str, float] = {
    "carry_borrow_scale_min": 0.50,
    "carry_borrow_scale_max": 0.70,
    "operator_scale": 1.0,          # legacy scalar fallback (used when min/max absent)
    "operator_scale_min": 0.65,     # iter10 R-B: range replaces scalar for variance
    "operator_scale_max": 1.20,
    "glyph_stroke_target_px": 2.0,  # iter11 FINAL (2026-06-22, user pick): normalise glyphs to 2.0px (real-matched, clean uniform "proven" look). 0=natural variance (rejected).
    "glyph_binarize_threshold": 165,  # iter11 FINAL: proven hard-binarise; glyph_stroke_target_px resets final width to 2px. 0=soft (AVOID, contrast fattens).
    "max_render_scale": 5.5,          # iter11 FINAL: legacy fill-canvas (normalise handles thickness); lower for smaller/sparser glyphs.
    # iter11 crowding (2026-06-22): two mechanics that model real tight kid handwriting.
    # overflow = a glyph enlarged/shifted so its bbox bleeds into a neighbour cell (H or V);
    # the centroid stays inside the intended cell (shift < 0.5*pitch) so geometric row/col GT holds.
    # double-digit = a borrow/carry rendered as two glyphs packed in ONE cell (same row,col), e.g. "16".
    "glyph_overflow_prob": 0.20,        # per main/operand glyph: probability it overflows a neighbour
    "glyph_overflow_scale_max": 1.35,   # max scale multiplier when overflow fires
    "glyph_overflow_shift_frac": 0.30,  # max centroid shift toward neighbour as fraction of pitch (<0.5 invariant)
    "double_digit_cell_prob": 0.30,     # per borrow/carry token: probability it packs two glyphs in one cell
    # bar
    "bar_gap_prob": 0.0,
    "bar_thickness_min": 2,
    "bar_thickness_max": 2,
    "bar_intensity_min": 0,
    "bar_intensity_max": 0,
    "bar_micro_break_prob": 0.0,
    "bar_y_jitter_sigma": 0.0,
    "bar_x_jitter_sigma": 0.0,
    # bracket
    "bracket_gap_prob": 0.0,
    "bracket_thickness_min": 2,
    "bracket_thickness_max": 2,
    "bracket_intensity_min": 0,
    "bracket_intensity_max": 0,
    "bracket_micro_break_prob": 0.0,
    "bracket_h_y_jitter_sigma": 0.0,
    "bracket_h_x_jitter_sigma": 0.0,
    "bracket_v_x_jitter_sigma": 0.0,
    # layout geometry
    "base_cell_px": 44,
    "gap_x_min": 6,
    "gap_x_max": 13,
    "gap_y_min": 10,
    "gap_y_max": 19,
    # glyph post-process (after BICUBIC resize + binarize)
    "glyph_morph_close_kernel": 0,   # 0 = disabled, 2 = 2x2, 3 = 3x3, etc. Thickens glyphs by filling halos.
    "broken_stroke_band_frac": 0.125,  # height of the erased band as a fraction of tile height (default 1/8)
    "page_padding_min_px": 2,          # minimum top/bottom canvas margin (px); sampled per scene
    "page_padding_max_px": 10,         # maximum top/bottom canvas margin (px); sampled per scene
    # subtraction borrow cross-out (D: per-digit probability; replaces binary borrow_cross_enabled)
    "borrow_cross_prob": 0.0,        # probability per borrow column of drawing the cross-out stroke
    # long division per-step minus operator (C)
    "step_minus_prob": 0.0,          # probability of emitting op_minus left of each step's product row
    # carry notation errors (E, F)
    "wrong_carry_col_prob": 0.0,     # probability of shifting a carry token one col left or right
    "missing_carry_prob": 0.0,       # probability of dropping a carry token entirely
    # result bar length variance (H)
    "short_bar_prob": 0.0,           # probability of drawing a shorter-than-full-width result bar
    "short_bar_shrink_frac": 0.20,   # fraction of bar width removed from each end when short
    # B: partial-product operator knobs (multiplication_multi)
    "pp_plus_prob": 0.0,             # probability of keeping/adding op_plus on PP rows (0=drop all)
    "pp_wrong_operator_prob": 0.0,   # probability of flipping PP op_plus → op_minus (student mistake)
    # C: variable bracket vertical arm depth (division)
    "bracket_depth_full_prob": 1.0,  # probability arm reaches last working row (1.0 = always full)
    "bracket_depth_min_rows": 2,     # minimum row span when arm is shortened
    # D: long-division step borrow/carry tokens
    "long_div_step_borrow_prob": 0.0,  # probability of emitting digit_borrow per step needing borrow
    # E: helper operator at left of intermediate rows
    "helper_operator_prob": 0.0,     # probability of inserting helper op_plus/op_minus per row
    # G: carry/borrow value errors
    "wrong_carry_value_prob": 0.0,   # probability of replacing a carry digit with a different digit
    "wrong_borrow_value_prob": 0.0,  # probability of replacing a borrow digit with a different digit
    "missing_borrow_prob": 0.0,      # probability of dropping a borrow token entirely
    # H: missing partial-product row (multiplication_multi only)
    "missing_pp_prob": 0.0,          # probability of dropping one non-final PP row per scene
}
_DEFAULT_AUGMENTATION: dict[str, float] = {"degrees":5.0,"translate":0.05,"scale":0.4,"hsv_s":0.05,"hsv_v":0.3,"fliplr":0.0,"flipud":0.0,"mosaic":0.0,"mixup":0.0,"copy_paste":0.0,"hsv_h":0.0,"perspective":0.0}
_DEFAULT_LOSS_WEIGHTS: dict[str, float] = {"fine_label":2.0,"row_contrastive":1.0,"row_ordinal":1.0,"col_contrastive":1.0,"col_ordinal":1.0,"eq_type":1.0,"edge_type":0.3}

@dataclass(frozen=True)
class PresetConfig:
    glyph_rotation_min_deg: float = -4.0
    glyph_rotation_max_deg: float = 4.0
    glyph_jitter_min_px: int = 0
    glyph_jitter_max_px: int = 4
    glyph_broken_stroke_prob: float = 0.05

@dataclass(frozen=True)
class SceneConfig:
    wrong_result_prob: float = 0.20
    missing_structural_prob: float = 0.15
    wrong_operator_prob: float = 0.05
    crowdness_prob: float = 0.20
    scene_rotation_min_deg: float = -4.0
    scene_rotation_max_deg: float = 4.0
    crowding_factor: float = 0.70

@dataclass
class GenerationConfig:
    images_per_case: int = 1000
    min_instances_per_label: int = 400
    topup_rounds: int = 8
    seed: int = 42
    split_train: float = 0.80
    split_val: float = 0.10
    split_test: float = 0.10
    case_weights: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_CASE_WEIGHTS))
    completion: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_COMPLETION))
    scene: SceneConfig = field(default_factory=SceneConfig)
    preset: PresetConfig = field(default_factory=PresetConfig)
    rendering: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_RENDERING))
    source_pool: str = "auto"  # "auto" | "legacy" | "clean" | "hires" | "crohme"
    min_source_quality: float = 0.4  # crops scoring below this are rejected at runtime

@dataclass(frozen=True)
class YoloInferenceConfig:
    iou: float = 0.3
    agnostic_nms: bool = False
    max_det: int = 80

@dataclass
class YoloConfig:
    model: str = "yolov8s.pt"
    epochs: int = 50
    batch: int = 16
    image_size: int = 512
    patience: int = 20
    device: str = "cpu"
    workers: int = 0
    cache: str = "disk"
    rect: bool = True
    optimizer: str = "auto"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    amp: bool = True
    augmentation: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_AUGMENTATION))
    inference: YoloInferenceConfig = field(default_factory=YoloInferenceConfig)

@dataclass
class GnnConfig:
    epochs: int = 60
    batch_size: int = 32
    lr: float = 0.001
    dropout: float = 0.1
    weight_decay: float = 0.0
    early_stop_patience: int = 10
    lr_patience: int = 5
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    temperature: float = 0.1
    scene_dropout_prob: float = 0.15
    loss_weights: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_LOSS_WEIGHTS))
    # W-ARCH-2: scene virtual node (default OFF; enable in config.toml [gnn] to ablate)
    use_scene_virtual_node: bool = False
    # W2-GNN-BBOX-JITTER: training-time bbox jitter (px) applied to spatial stream inputs.
    # 0.0 = disabled (default). Val/test datasets always see clean coords regardless of this value.
    bbox_jitter_px: float = 0.0

@dataclass
class ParsingConfig:
    """Assembler / parsing configuration (config.toml [parsing]).

    assembler_heuristic_enabled is diagnostic-only: it toggles whether the
    operator-disagreement observability fields are computed. It NEVER mutates
    the recognized equation_kind (recognize-as-drawn). Default OFF.
    """
    assembler_heuristic_enabled: bool = False


def load_parsing_config(project_root: Path) -> "ParsingConfig":
    """Read [parsing] section from config.toml. Returns defaults if absent."""
    toml_path = project_root / "config.toml"
    if not toml_path.exists():
        return ParsingConfig()
    with toml_path.open("rb") as f:
        raw: dict[str, Any] = tomllib.load(f)
    rp: dict[str, Any] = raw.get("parsing", {})
    return ParsingConfig(
        assembler_heuristic_enabled=bool(rp.get("assembler_heuristic_enabled", False)),
    )


@dataclass(frozen=True)
class CvFusionConfig:
    """Classical-OpenCV complementary detector fusion (config.toml [inference.cv_fusion]).

    YOLO is trained on finished scenes and misses symbols in unfinished ones. The
    CV branch runs a connected-component detector over regions YOLO did *not*
    claim, reclassifies the extras with a single batched YOLO pass, and merges
    them into YOLO's detections. It augments YOLO; it never replaces it.
    """
    mode: str = "on"                    # "on" (always fuse) | "auto" (gated) | "off"
    flatten_background: bool = False    # flatten a dark photo surround to white before the WHOLE pipeline (YOLO+CV+GNN) — for phone photos of paper
    gate_min_detections: int = 6        # auto-mode: skip CV when YOLO returns >= this many ...
    gate_mean_conf: float = 0.6         # ... high-confidence detections (mean conf >= this)
    use_second_yolo_pass: bool = False  # True → run second YOLO on tight-masked image; False → run CV detector (matches config.toml default)
    second_pass_conf: float = 0.15      # confidence threshold for second YOLO pass (lower than first-pass)
    tight_mask_dilate_px: int = 4       # dilation (px) of tight-mask before second YOLO pass
    classify_conf: float = 0.25         # drop a CV crop if its YOLO reclassification scores below this
    classify_with_yolo: bool = True     # True → reclassify CV crops with YOLO; False → skip (0 extra YOLO calls). Matches config.toml default.
    keep_unclassified: bool = False     # keep CV boxes YOLO can't classify, tagged with fallback_label, for the GNN
    keep_min_ink_frac: float = 0.10     # minimum dark-pixel fraction inside a box for it to pass as a fallback node; 0.0 disables the floor (H-1 fix)
    fallback_label: str = "digit_main"  # coarse label hint for kept-but-unclassified CV boxes
    unclassified_conf: float = 0.20     # confidence assigned to kept-but-unclassified CV boxes
    mask_yolo: bool = True              # False → run OpenCV on the FULL image independently of YOLO; overlaps resolved by NMS (ties → YOLO)
    cv_tight_mask: bool = True          # True → mask only YOLO's actual ink pixels (tight) so carries inside loose bboxes survive; False → rectangle mask
    mask_dilate_px: int = 5             # dilation (px) of the YOLO-bbox mask before running CV (only when mask_yolo and not cv_tight_mask)
    nms_iou: float = 0.3                # class-agnostic NMS IoU on the merged union
    letterbox_size: int = 512           # YOLO input size for the batched crop-classify pass
    # detect_symbols_cv knobs (passed straight through to the detector)
    min_area: int = 30                  # matches config.toml shipped value (was 20 in dataclass, 30 in config)
    min_side: int = 4
    pad: int = 4
    merge_overlap_ratio: float = 0.95   # near-1.0 = only merge near-complete overlaps (matches config.toml shipped value)
    proximity_merge_factor: float = 0.0  # 0.0 = NO proximity merge → no mega-blobs gluing symbols (matches config.toml shipped value)


def load_cv_fusion_config(project_root: Path) -> "CvFusionConfig":
    """Read [inference.cv_fusion] from config.toml. Returns defaults if absent."""
    toml_path = project_root / "config.toml"
    if not toml_path.exists():
        return CvFusionConfig()
    with toml_path.open("rb") as f:
        raw: dict[str, Any] = tomllib.load(f)
    section: dict[str, Any] = raw.get("inference", {}).get("cv_fusion", {})
    d = CvFusionConfig()
    mode = str(section.get("mode", d.mode)).lower()
    if mode not in {"on", "auto", "off"}:
        logging.warning("[inference.cv_fusion] invalid mode %r; using %r", mode, d.mode)
        mode = d.mode
    return CvFusionConfig(
        mode=mode,
        flatten_background=bool(section.get("flatten_background", d.flatten_background)),
        gate_min_detections=int(section.get("gate_min_detections", d.gate_min_detections)),
        gate_mean_conf=float(section.get("gate_mean_conf", d.gate_mean_conf)),
        use_second_yolo_pass=bool(section.get("use_second_yolo_pass", d.use_second_yolo_pass)),
        second_pass_conf=float(section.get("second_pass_conf", d.second_pass_conf)),
        tight_mask_dilate_px=int(section.get("tight_mask_dilate_px", d.tight_mask_dilate_px)),
        classify_conf=float(section.get("classify_conf", d.classify_conf)),
        classify_with_yolo=bool(section.get("classify_with_yolo", d.classify_with_yolo)),
        keep_unclassified=bool(section.get("keep_unclassified", d.keep_unclassified)),
        keep_min_ink_frac=float(section.get("keep_min_ink_frac", d.keep_min_ink_frac)),
        fallback_label=str(section.get("fallback_label", d.fallback_label)),
        unclassified_conf=float(section.get("unclassified_conf", d.unclassified_conf)),
        mask_yolo=bool(section.get("mask_yolo", d.mask_yolo)),
        cv_tight_mask=bool(section.get("cv_tight_mask", d.cv_tight_mask)),
        mask_dilate_px=int(section.get("mask_dilate_px", d.mask_dilate_px)),
        nms_iou=float(section.get("nms_iou", d.nms_iou)),
        letterbox_size=int(section.get("letterbox_size", d.letterbox_size)),
        min_area=int(section.get("min_area", d.min_area)),
        min_side=int(section.get("min_side", d.min_side)),
        pad=int(section.get("pad", d.pad)),
        merge_overlap_ratio=float(section.get("merge_overlap_ratio", d.merge_overlap_ratio)),
        proximity_merge_factor=float(section.get("proximity_merge_factor", d.proximity_merge_factor)),
    )


@dataclass(frozen=True)
class GradingConfig:
    """Grading-engine client config (config.toml [grading]).

    ``mode`` selects the client; only ``"mock"`` (the in-process grader) ships.
    ``audience`` is the curriculum selector passed when an exercise is created.
    """
    mode: str = "mock"
    audience: str = "uk_KS2"
    # Exercise UUIDs to offer in the demo picker, in picker order. Each is derived
    # at select time from its captured create + solution fixtures.
    exercise_ids: list[str] = field(default_factory=list)


def load_grading_config(project_root: Path) -> "GradingConfig":
    """Read [grading] from config.toml. Returns mock defaults if absent."""
    toml_path = project_root / "config.toml"
    if not toml_path.exists():
        return GradingConfig()
    with toml_path.open("rb") as f:
        raw: dict[str, Any] = tomllib.load(f)
    section: dict[str, Any] = raw.get("grading", {})
    d = GradingConfig()
    mode = str(section.get("mode", d.mode)).lower()
    if mode not in {"mock"}:
        logging.warning("[grading] invalid mode %r; using %r", mode, d.mode)
        mode = d.mode
    raw_ids = section.get("exercise_ids", d.exercise_ids)
    exercise_ids = [str(eid) for eid in raw_ids] if isinstance(raw_ids, list) else d.exercise_ids
    return GradingConfig(
        mode=mode,
        audience=str(section.get("audience", d.audience)),
        exercise_ids=exercise_ids,
    )


@dataclass(frozen=True)
class DemoConfig:
    """Demo app configuration (config.toml [demo]).

    ``given_prior_enabled``: when True, the evaluate_route builds a GivenNode
    prior from the active exercise's scaffold and passes it to InferenceSession.predict
    so the GNN sees given tiles in the scene. Ships OFF (False); promotion is gated
    on empirical eval showing positive eq_kind and row/col deltas.
    """

    given_prior_enabled: bool = False


def load_demo_config(project_root: Path) -> "DemoConfig":
    """Read [demo] from config.toml. Returns defaults if absent."""
    toml_path = project_root / "config.toml"
    if not toml_path.exists():
        return DemoConfig()
    with toml_path.open("rb") as f:
        raw: dict[str, Any] = tomllib.load(f)
    section: dict[str, Any] = raw.get("demo", {})
    return DemoConfig(
        given_prior_enabled=bool(section.get("given_prior_enabled", False)),
    )


def _normalise_case_key(key: str) -> str:
    if key in _VALID_CASES:
        return key
    return key.replace("_", "-")

def load_config(project_root: Path) -> tuple[GenerationConfig, YoloConfig, GnnConfig]:
    toml_path = project_root / "config.toml"
    if not toml_path.exists():
        logging.warning("config.toml not found at %s", toml_path)
        return GenerationConfig(), YoloConfig(), GnnConfig()
    with toml_path.open("rb") as f:
        raw: dict[str, Any] = tomllib.load(f)
    g = GenerationConfig()
    rg: dict[str, Any] = raw.get("generation", {})
    cw: dict[str, float] = dict(_DEFAULT_CASE_WEIGHTS)
    for k, v in rg.get("case_weights", {}).items():
        n = _normalise_case_key(k)
        if n not in _VALID_CASES:
            raise ValueError(f"Unknown case_weight key: {k!r}. Valid: {sorted(_VALID_CASES)}")
        cw[n] = float(v)
    comp: dict[str, float] = dict(_DEFAULT_COMPLETION)
    for k, v in rg.get("completion", {}).items():
        if k in comp:
            comp[k] = float(v)
    _comp_sum = sum(comp.values())
    if abs(_comp_sum - 1.0) >= 1e-6:
        raise ValueError(
            f"[generation.completion] weights must sum to 1.0, got {_comp_sum:.8f}. "
            f"Values: {comp}"
        )
    rs = rg.get("scene", {})
    sc = SceneConfig(wrong_result_prob=float(rs.get("wrong_result_prob",0.20)),missing_structural_prob=float(rs.get("missing_structural_prob",0.15)),wrong_operator_prob=float(rs.get("wrong_operator_prob",0.05)),crowdness_prob=float(rs.get("crowdness_prob",0.20)),scene_rotation_min_deg=float(rs.get("scene_rotation_min_deg",-4.0)),scene_rotation_max_deg=float(rs.get("scene_rotation_max_deg",4.0)),crowding_factor=float(rs.get("crowding_factor",0.70)))
    rp = rg.get("preset", {})
    pc = PresetConfig(glyph_rotation_min_deg=float(rp.get("glyph_rotation_min_deg",-4.0)),glyph_rotation_max_deg=float(rp.get("glyph_rotation_max_deg",4.0)),glyph_jitter_min_px=int(rp.get("glyph_jitter_min_px",0)),glyph_jitter_max_px=int(rp.get("glyph_jitter_max_px",4)),glyph_broken_stroke_prob=float(rp.get("glyph_broken_stroke_prob",0.05)))
    rend: dict[str, float] = dict(_DEFAULT_RENDERING)
    for k, v in rg.get("rendering", {}).items():
        if k in rend:
            rend[k] = float(v)
    gen_cfg = GenerationConfig(images_per_case=int(rg.get("images_per_case",g.images_per_case)),min_instances_per_label=int(rg.get("min_instances_per_label",g.min_instances_per_label)),topup_rounds=int(rg.get("topup_rounds",g.topup_rounds)),seed=int(rg.get("seed",g.seed)),case_weights=cw,split_train=float(rg.get("split_train",g.split_train)),split_val=float(rg.get("split_val",g.split_val)),split_test=float(rg.get("split_test",g.split_test)),completion=comp,scene=sc,preset=pc,rendering=rend,source_pool=str(rg.get("source_pool",g.source_pool)),min_source_quality=float(rg.get("min_source_quality",g.min_source_quality)))
    yd = YoloConfig()
    ry: dict[str, Any] = raw.get("yolo", {})
    aug: dict[str, float] = dict(_DEFAULT_AUGMENTATION)
    for k, v in ry.get("augmentation", {}).items():
        if k in aug:
            aug[k] = float(v)
    ri = ry.get("inference", {})
    inf = YoloInferenceConfig(iou=float(ri.get("iou",0.3)),agnostic_nms=bool(ri.get("agnostic_nms",False)),max_det=int(ri.get("max_det",80)))
    yolo_cfg = YoloConfig(model=str(ry.get("model",yd.model)),epochs=int(ry.get("epochs",yd.epochs)),batch=int(ry.get("batch",yd.batch)),image_size=int(ry.get("image_size",yd.image_size)),patience=int(ry.get("patience",yd.patience)),device=str(ry.get("device",yd.device)),workers=int(ry.get("workers",yd.workers)),cache=str(ry.get("cache",yd.cache)),rect=bool(ry.get("rect",yd.rect)),optimizer=str(ry.get("optimizer",yd.optimizer)),lr0=float(ry.get("lr0",yd.lr0)),lrf=float(ry.get("lrf",yd.lrf)),momentum=float(ry.get("momentum",yd.momentum)),weight_decay=float(ry.get("weight_decay",yd.weight_decay)),warmup_epochs=float(ry.get("warmup_epochs",yd.warmup_epochs)),box=float(ry.get("box",yd.box)),cls=float(ry.get("cls",yd.cls)),dfl=float(ry.get("dfl",yd.dfl)),amp=bool(ry.get("amp",yd.amp)),augmentation=aug,inference=inf)
    gd = GnnConfig()
    rn: dict[str, Any] = raw.get("gnn", {})
    lw: dict[str, float] = dict(_DEFAULT_LOSS_WEIGHTS)
    for k, v in rn.get("loss_weights", {}).items():
        if k in lw:
            lw[k] = float(v)
    gnn_cfg = GnnConfig(epochs=int(rn.get("epochs",gd.epochs)),batch_size=int(rn.get("batch_size",gd.batch_size)),lr=float(rn.get("lr",gd.lr)),dropout=float(rn.get("dropout",gd.dropout)),weight_decay=float(rn.get("weight_decay",gd.weight_decay)),early_stop_patience=int(rn.get("early_stop_patience",gd.early_stop_patience)),lr_patience=int(rn.get("lr_patience",gd.lr_patience)),lr_factor=float(rn.get("lr_factor",gd.lr_factor)),min_lr=float(rn.get("min_lr",gd.min_lr)),temperature=float(rn.get("temperature",gd.temperature)),scene_dropout_prob=float(rn.get("scene_dropout_prob",gd.scene_dropout_prob)),loss_weights=lw,use_scene_virtual_node=bool(rn.get("use_scene_virtual_node",gd.use_scene_virtual_node)),bbox_jitter_px=float(rn.get("bbox_jitter_px",gd.bbox_jitter_px)))
    return gen_cfg, yolo_cfg, gnn_cfg
