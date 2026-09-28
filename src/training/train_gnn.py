from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from ..core.artifact_paths import (
    append_runs_index,
    gnn_runs_root,
    new_training_run_id,
    try_git_revision,
    write_active_gnn,
)
from ..core.config import DataPrepConfig
from ..core.logging_setup import configure_run_logging
from ..core.ontology import gnn_fine_labels_ordered, gnn_label_from_full
from ..core.run_config import GnnConfig
from ..core.cluster_metrics import permutation_invariant_cluster_accuracy
from ..modeling.gnn import (
    CropBackbone,
    NUM_FINE_LABELS,
    WITHIN_ROW_ORDINAL_CLASSES,
    WITHIN_COL_ORDINAL_CLASSES,
    SymbolGNN,
    _cluster_by_coordinate,
)
from ..modeling.graph_builder import build_graph
from ..parsing.detection import Detection

_FINE_LABELS = gnn_fine_labels_ordered()
FINE_LABEL_TO_IDX: Dict[str, int] = {lbl: i for i, lbl in enumerate(_FINE_LABELS)}


def _full_to_gnn_idx(full_label: str) -> int:
    """Map a 36-style fine label (from synthetic GT JSON) to a 16-class GNN index.

    Unknown labels fall back to index 0 (digit "0") so downstream training
    code never raises on a malformed scene.
    """
    gnn_lbl = gnn_label_from_full(full_label)
    if gnn_lbl is None:
        return 0
    return FINE_LABEL_TO_IDX.get(gnn_lbl, 0)

_EQ_TYPES = ["add", "subtract", "multiply", "divide"]
EQ_TYPE_TO_IDX: Dict[str, int] = {eq: i for i, eq in enumerate(_EQ_TYPES)}

# Two vocabularies reach this loader. Synthetic GT writes the SHORT form
# ("add"/"subtract"/"multiply"/"divide"); the setmaker train exporter writes the
# LONG EQUATION_KINDS form ("addition"/"subtraction"/"multiplication"/"division").
# This alias normalises both onto the SHORT keys EQ_TYPE_TO_IDX expects. Without
# it, every long-form (real) scene falls through `.get(..., 0)` to idx 0 ("add"),
# silently teaching the eq_type head that all real scenes are addition.
_EQ_KIND_ALIAS: Dict[str, str] = {
    "addition": "add",
    "subtraction": "subtract",
    "multiplication": "multiply",
    "division": "divide",
    "add": "add",
    "subtract": "subtract",
    "multiply": "multiply",
    "divide": "divide",
}
# Scenes with no single operator label to supervise. These map to the masking
# sentinel -1 so cross_entropy(ignore_index=-1) drops them from the eq_type loss.
# "bare_digits"/"ood_unknown" come from the exporter + synthetic OOD scenes;
# "unknown" is the legacy synthetic sentinel preserved for back-compat.
_EQ_MASKED_KINDS: frozenset[str] = frozenset(
    {"unknown", "ood_unknown", "bare_digits", "bare_digit_grid"}
)


def _supervised_contrastive_loss(
    emb: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Supervised NT-Xent: same-cluster pairs are positive, different-cluster are negative.

    Returns scalar loss. Returns zero tensor if all nodes are in the same cluster or N < 2.
    """
    N = emb.shape[0]
    if N < 2:
        return torch.tensor(0.0, requires_grad=True, device=emb.device)

    # L2-normalise embeddings
    emb = F.normalize(emb, dim=1)  # (N, D)

    # Cosine similarity matrix
    sim = torch.mm(emb, emb.T) / temperature  # (N, N)

    # Mask diagonal
    diag_mask = torch.eye(N, dtype=torch.bool, device=emb.device)

    # Positive pairs: same label, not diagonal
    label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N)
    pos_mask = label_eq & ~diag_mask

    # If no positive pairs exist (all different clusters), return 0
    if not pos_mask.any():
        return torch.tensor(0.0, requires_grad=True, device=emb.device)

    # Log-sum-exp over negatives per anchor
    sim_exp = torch.exp(sim - sim.max(dim=1, keepdim=True).values)
    sim_exp = sim_exp * ~diag_mask  # zero diagonal

    numerator = (sim_exp * pos_mask).sum(dim=1)  # (N,)
    denominator = sim_exp.sum(dim=1)  # (N,)

    # Only compute loss for anchors that have at least one positive
    has_pos = pos_mask.any(dim=1)  # (N,)
    if not has_pos.any():
        return torch.tensor(0.0, requires_grad=True, device=emb.device)

    loss = -torch.log(numerator[has_pos] / (denominator[has_pos] + 1e-8))
    return loss.mean()


_CANVAS_SIZE: float = 512.0


def _jitter_detections(
    detections: List[Detection],
    jitter_px: float,
    rng: np.random.Generator,
) -> List[Detection]:
    """Return a new list of Detections with independent per-bbox coordinate jitter.

    Each of the four coordinates (x0, y0, x1, y1) is shifted by a uniform
    random draw in [-jitter_px, +jitter_px].  After jitter, two invariants are
    enforced in this order:

    1. Canvas clamp: all coords are clamped to [0, _CANVAS_SIZE].
    2. Swap guard: x1 is bumped to max(x1, x0 + 1) and y1 to max(y1, y0 + 1)
       so the bbox always has at least 1-pixel extent even when both edges are
       pushed to the same boundary.
    """
    result: List[Detection] = []
    for det in detections:
        dx0, dy0, dx1, dy1 = rng.uniform(-jitter_px, jitter_px, size=4)
        x0 = float(np.clip(det.x0 + dx0, 0.0, _CANVAS_SIZE))
        y0 = float(np.clip(det.y0 + dy0, 0.0, _CANVAS_SIZE))
        x1 = float(np.clip(det.x1 + dx1, 0.0, _CANVAS_SIZE))
        y1 = float(np.clip(det.y1 + dy1, 0.0, _CANVAS_SIZE))
        # Swap guard + canvas re-clamp (two-step to handle boundary collisions).
        # Step 1: if x1 <= x0, push x1 up; if that would exceed the canvas,
        #         pull x0 down instead so both coords stay inside [0, CANVAS].
        if x1 <= x0:
            x1 = x0 + 1.0
            if x1 > _CANVAS_SIZE:
                x1 = _CANVAS_SIZE
                x0 = _CANVAS_SIZE - 1.0
        # Step 2: same logic for y axis.
        if y1 <= y0:
            y1 = y0 + 1.0
            if y1 > _CANVAS_SIZE:
                y1 = _CANVAS_SIZE
                y0 = _CANVAS_SIZE - 1.0
        result.append(Detection(
            label=det.label,
            confidence=det.confidence,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
        ))
    return result


class SceneGraphDataset(Dataset):
    """
    One item = one arithmetic scene graph.
    Manifest CSV columns: image, ground_truth.
    gt.json format: {equation_type, symbols: [{fine_label, bbox, yolo_class, row_index, col_index}]}
    row_index and col_index are used directly as cluster IDs for contrastive learning.
    Within-row and within-col ordinal ranks are derived from sorting by col_index and row_index.
    """

    def __init__(
        self,
        manifest_csv: Path,
        *,
        bbox_jitter_px: float = 0.0,
        training: bool = False,
    ) -> None:
        super().__init__()
        df = pd.read_csv(manifest_csv)
        self._rows = df[["image", "ground_truth"]].dropna().values.tolist()
        self._bbox_jitter_px = bbox_jitter_px
        self._training = training

    def len(self) -> int:
        return len(self._rows)

    def compute_scene_weights(self) -> list:
        """Per-scene sampling weight = max inverse frequency of fine labels in scene.

        Scenes that contain rare labels (borrow_*, carry_*, div_bracket) get higher
        weight so WeightedRandomSampler draws them more often.
        """
        from collections import Counter
        global_counts: Counter = Counter()
        all_scene_labels: list = []
        for _, gt_path in self._rows:
            gt = json.loads(Path(gt_path).read_text(encoding="utf-8"))
            labels = [
                _full_to_gnn_idx(tok.get("fine_label", ""))
                for tok in gt.get("symbols", [])
            ]
            all_scene_labels.append(labels)
            global_counts.update(labels)

        total = sum(global_counts.values()) or 1
        weights = []
        for labels in all_scene_labels:
            if not labels:
                weights.append(1.0)
            else:
                # Scene weight = max rarity of any label it contains
                weights.append(max(total / global_counts[lbl] for lbl in labels))
        return weights

    def get(self, idx: int) -> Data:
        img_path, gt_path = self._rows[idx]
        gt: Dict[str, Any] = json.loads(Path(gt_path).read_text(encoding="utf-8"))
        gray = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
        tokens = gt.get("symbols", [])

        detections = [
            Detection(
                label=tok.get("yolo_class", "digit_main"),
                confidence=1.0,
                x0=float(tok["bbox"][0]),
                y0=float(tok["bbox"][1]),
                x1=float(tok["bbox"][2]),
                y1=float(tok["bbox"][3]),
            )
            for tok in tokens
        ]

        if self._training and self._bbox_jitter_px > 0.0:
            rng = np.random.default_rng()
            detections = _jitter_detections(detections, self._bbox_jitter_px, rng)

        data = build_graph(detections, gray)
        data.y_fine = torch.tensor(
            [_full_to_gnn_idx(tok["fine_label"]) for tok in tokens],
            dtype=torch.long,
        )

        # y_row_cluster: cluster ID per node = absolute row_index (unique per row)
        data.y_row_cluster = torch.tensor(
            [int(tok.get("row_index", 0)) for tok in tokens], dtype=torch.long
        )

        # y_col_cluster: cluster ID per node = absolute col_index (unique per col)
        data.y_col_cluster = torch.tensor(
            [int(tok.get("col_index", 0)) for tok in tokens], dtype=torch.long
        )

        # y_within_row_ord: rank of this node within its row, ordered by col_index asc
        row_to_cols: Dict[int, list] = {}
        for i, tok in enumerate(tokens):
            r = int(tok.get("row_index", 0))
            c = int(tok.get("col_index", 0))
            row_to_cols.setdefault(r, []).append((c, i))
        within_row_ord = [0] * len(tokens)
        for r, pairs in row_to_cols.items():
            for rank, (c, i) in enumerate(sorted(pairs, key=lambda x: x[0])):
                within_row_ord[i] = min(rank, WITHIN_ROW_ORDINAL_CLASSES - 1)
        data.y_within_row_ord = torch.tensor(within_row_ord, dtype=torch.long)

        # y_within_col_ord: rank of this node within its col, ordered by row_index asc
        col_to_rows: Dict[int, list] = {}
        for i, tok in enumerate(tokens):
            r = int(tok.get("row_index", 0))
            c = int(tok.get("col_index", 0))
            col_to_rows.setdefault(c, []).append((r, i))
        within_col_ord = [0] * len(tokens)
        for c, pairs in col_to_rows.items():
            for rank, (r, i) in enumerate(sorted(pairs, key=lambda x: x[0])):
                within_col_ord[i] = min(rank, WITHIN_COL_ORDINAL_CLASSES - 1)
        data.y_within_col_ord = torch.tensor(within_col_ord, dtype=torch.long)

        # y_edge_type: ground truth edge types come from the graph builder (already computed)
        data.y_edge_type = data.edge_type.clone()

        # eq_type label: normalise the SHORT (synthetic) and LONG (setmaker) vocabularies
        # onto EQ_TYPE_TO_IDX, and mask structure-only / OOD scenes with sentinel -1
        # (ignore_index=-1 in cross_entropy) so they never supervise the operator head.
        eq_type_str = gt.get("equation_type", "add")
        if eq_type_str in _EQ_MASKED_KINDS:
            eq_type_idx = -1
        else:
            eq_type_idx = EQ_TYPE_TO_IDX.get(_EQ_KIND_ALIAS.get(eq_type_str, eq_type_str), 0)
        data.y_eq = torch.tensor([eq_type_idx], dtype=torch.long)
        return data


def _fine_acc(model: SymbolGNN, loader: DataLoader, device: str = "cpu") -> float:
    """Per-node fine-label accuracy across the loader."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            fl, *_ = model(batch)
            correct += int((fl.argmax(dim=1) == batch.y_fine).sum())
            total += int(batch.y_fine.shape[0])
    return correct / total if total > 0 else 0.0


def _eq_type_acc(model: SymbolGNN, loader: DataLoader, device: str = "cpu") -> float:
    """Per-graph equation-type accuracy across the loader.

    OOD scenes (y_eq == -1) are excluded from accuracy computation.
    """
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, _, _, _, _, eq_logits, _ = model(batch)
            valid = batch.y_eq != -1
            if not valid.any():
                continue
            correct += int((eq_logits.argmax(dim=1)[valid] == batch.y_eq[valid]).sum())
            total += int(valid.sum())
    return correct / total if total > 0 else 0.0


def combined_score(fine_acc: float, eq_acc: float) -> float:
    """Equal-weighted average of fine-label and equation-type accuracies.

    Used as the early-stop tracking metric to prevent silent overfit on
    fine_label alone (iteration 1 failure mode where eq_type generalisation
    degraded while val_fine_acc inched upward).
    """
    return 0.5 * fine_acc + 0.5 * eq_acc


def compute_test_metrics(
    model: SymbolGNN,
    test_loader: SceneGraphDataset,
    device: str = "cpu",
) -> Dict[str, float]:
    """Compute test-set metrics for a trained GNN model.

    Args:
        model: Trained SymbolGNN (already loaded with weights, on correct device).
        test_loader: SceneGraphDataset for the test split (iterated scene-by-scene).
        device: Torch device string; used to move data tensors if needed.

    Returns:
        Dict with keys: fine_label_accuracy, fine_label_macro_f1,
        row_cluster_accuracy (mean per-scene ARI), col_cluster_accuracy
        (mean per-scene ARI), within_row_ord_accuracy, within_col_ord_accuracy,
        equation_type_accuracy, exact_scene_match.
    """
    model.eval()
    all_fine_p: List[int] = []
    all_fine_g: List[int] = []
    exact_matches = eq_correct = eq_scenes = 0
    row_cl_scene_aris: List[float] = []
    col_cl_scene_aris: List[float] = []
    row_ord_ok = row_ord_tot = 0
    col_ord_ok = col_ord_tot = 0

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            fl, row_emb, row_ord_logits, col_emb, col_ord_logits, eq_logits, et_logits = model(data)
            fp = fl.argmax(dim=1)
            eq_p = int(eq_logits.argmax(dim=1).item())

            all_fine_p.extend(fp.tolist())
            all_fine_g.extend(data.y_fine.tolist())

            # Row cluster: per-scene ARI (permutation-invariant)
            pred_row = _cluster_by_coordinate(data.bbox, axis=1)
            row_cl_scene_aris.append(
                permutation_invariant_cluster_accuracy(pred_row, data.y_row_cluster.tolist())
            )

            # Col cluster: per-scene ARI (permutation-invariant)
            pred_col = _cluster_by_coordinate(data.bbox, axis=0)
            col_cl_scene_aris.append(
                permutation_invariant_cluster_accuracy(pred_col, data.y_col_cluster.tolist())
            )

            rp = row_ord_logits.argmax(dim=1)
            row_ord_ok += int((rp == data.y_within_row_ord).sum())
            row_ord_tot += len(data.y_within_row_ord)

            cp = col_ord_logits.argmax(dim=1)
            col_ord_ok += int((cp == data.y_within_col_ord).sum())
            col_ord_tot += len(data.y_within_col_ord)

            if int(data.y_eq[0]) != -1:  # skip OOD scenes
                eq_correct += int(eq_p == int(data.y_eq[0]))
                eq_scenes += 1

            pred_row_t = torch.tensor(pred_row, dtype=torch.long)
            if (fp.cpu() == data.y_fine.cpu()).all() and (pred_row_t == data.y_row_cluster.cpu()).all():
                exact_matches += 1

    n_test = len(test_loader)
    fine_acc = float(np.mean(np.array(all_fine_p) == np.array(all_fine_g))) if all_fine_p else 0.0
    clf_report = classification_report(
        all_fine_g, all_fine_p,
        labels=list(range(len(_FINE_LABELS))),
        target_names=_FINE_LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {
        "fine_label_accuracy":     fine_acc,
        "fine_label_macro_f1":     float(clf_report["macro avg"]["f1-score"]),
        "row_cluster_accuracy":    float(np.mean(row_cl_scene_aris)) if row_cl_scene_aris else 0.0,
        "col_cluster_accuracy":    float(np.mean(col_cl_scene_aris)) if col_cl_scene_aris else 0.0,
        "within_row_ord_accuracy": row_ord_ok / row_ord_tot if row_ord_tot else 0.0,
        "within_col_ord_accuracy": col_ord_ok / col_ord_tot if col_ord_tot else 0.0,
        "equation_type_accuracy":  eq_correct / eq_scenes if eq_scenes else 0.0,
        "exact_scene_match":       exact_matches / n_test if n_test else 0.0,
    }


def train_gnn_model(
    config: DataPrepConfig,
    gnn_cfg: GnnConfig,
    init_weights_path: Optional[Path] = None,
    device: Optional[str] = None,
    data_root_override: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train the GNN model using the given DataPrepConfig and GnnConfig.

    Args:
        config: Data paths and pipeline settings.
        gnn_cfg: All GNN hyperparameters (epochs, lr, dropout, loss weights, etc.).
        init_weights_path: Optional path to a .pt checkpoint to warm-start from.
            When set the model is initialised from these weights before training
            begins (useful for fine-tuning a previously trained GNN on new data).
            Pass the absolute path; the file must exist or a FileNotFoundError is
            raised immediately.

    Returns:
        Dict with keys: run_id, run_dir, best_model, report, run_manifest, promoted.
            promoted=False when promotion gates (BS-3/BS-4) block active.json update.
    """
    if data_root_override is not None:
        # Fine-tune path: consume a pre-built merged real+synthetic dataset
        # (see src/data_pipeline/prepare_realtrain.py). Same manifest layout as a
        # synthetic run, so nothing else in this function changes.
        synth_root = Path(data_root_override)
        if not synth_root.is_dir():
            raise FileNotFoundError(
                f"--finetune-from-real dir not found: {synth_root}. "
                "Build it with: python -m src prepare-realtrain."
            )
    else:
        from ..data_pipeline.prepare_synthetic_yolo import resolve_latest_run  # noqa: PLC0415
        _latest = resolve_latest_run(config.synthetic_dir)
        if _latest is None:
            raise FileNotFoundError(
                "No synthetic dataset found (data/generated/synthetic/latest missing). "
                "Run: python -m src generate --project-root <project>"
            )
        synth_root = _latest
    for split in ("train", "val", "test"):
        p = synth_root / f"{split}_manifest.csv"
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run: python -m src generate --project-root <project> "
                "(or prepare-realtrain for a fine-tune dataset)."
            )

    train_ds = SceneGraphDataset(
        synth_root / "train_manifest.csv",
        bbox_jitter_px=gnn_cfg.bbox_jitter_px,
        training=True,
    )
    val_ds   = SceneGraphDataset(synth_root / "val_manifest.csv")
    test_ds  = SceneGraphDataset(synth_root / "test_manifest.csv")

    from torch.utils.data import WeightedRandomSampler
    scene_weights = train_ds.compute_scene_weights()
    sampler = WeightedRandomSampler(scene_weights, num_samples=len(scene_weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=gnn_cfg.batch_size, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=gnn_cfg.batch_size)

    # Resolve training device: explicit override > auto-detect (CUDA > CPU).
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Compute class weights for fine_label from training data to handle
    # imbalance (carries/borrows are rarer than main digits).
    from ..modeling.class_weights import balanced_class_weights_sparse
    all_y_fine = []
    for data in train_ds:
        all_y_fine.extend(data.y_fine.tolist())

    # BS-5: Pre-train unique class count check. Refuse to train on degenerate data
    # that lacks enough label diversity -- a symptom of synthesis pipeline corruption.
    CLASS_DIVERSITY_FLOOR = 12  # out of NUM_FINE_LABELS (16)
    unique_classes = len(set(int(y) for y in all_y_fine if y >= 0))
    if unique_classes < CLASS_DIVERSITY_FLOOR:
        raise RuntimeError(
            f"PRE_TRAIN_BLOCKED: training data has only {unique_classes} unique fine_label classes "
            f"(< floor {CLASS_DIVERSITY_FLOOR}). Likely synthesis pipeline corruption. "
            f"Refusing to train on degenerate data. Investigate data/generated/synthetic/latest/ before retrying."
        )

    cw_dict = balanced_class_weights_sparse(
        np.array(all_y_fine, dtype=int), num_classes=NUM_FINE_LABELS
    )
    fine_class_weights = torch.tensor(
        [cw_dict.get(i, 1.0) for i in range(NUM_FINE_LABELS)], dtype=torch.float32
    ).to(device)

    # SymbolGNN accepts dropout — pass gnn_cfg.dropout
    model = SymbolGNN(CropBackbone(), dropout=gnn_cfg.dropout)
    model = model.to(device)

    # Warm-start: load checkpoint weights before training if requested.
    run_type = "scratch"
    init_from_run_id: Optional[str] = None
    if init_weights_path is not None:
        if not init_weights_path.is_file():
            raise FileNotFoundError(
                f"--init-from checkpoint not found: {init_weights_path}. "
                "Provide a valid run_id whose best.pt exists under artifacts/gnn/runs/."
            )
        # strict=False allows partial warm-start from R2 checkpoints whose
        # state_dict has gat1/gat2 keys rather than the iter9 vis_gat1/vis_gat2/
        # spa_gat1/spa_gat2 split. Matching keys load; new spatial stream keys
        # stay randomly initialised.
        model.load_state_dict(
            torch.load(init_weights_path, map_location="cpu"), strict=False
        )
        model = model.to(device)
        run_type = "finetune"
        # Extract run_id from the parent directory name when the path is inside
        # artifacts/gnn/runs/<run_id>/best.pt; fall back to the full path string.
        try:
            init_from_run_id = init_weights_path.parent.name
        except Exception:
            init_from_run_id = str(init_weights_path)
        # logger not yet configured at this point (configure_run_logging runs later);
        # print to stdout so nohup captures it.
        print(f"[warm-start] loaded weights from {init_weights_path} (run_type=finetune, init_from={init_from_run_id})", flush=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=gnn_cfg.lr, weight_decay=gnn_cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=gnn_cfg.lr_patience,
        factor=gnn_cfg.lr_factor,
        min_lr=gnn_cfg.min_lr,
    )

    scene_dropout_prob = gnn_cfg.scene_dropout_prob

    # Loss weights come from config (replaces previous hardcoded dict)
    loss_weights: Dict[str, float] = gnn_cfg.loss_weights

    run_id     = new_training_run_id()
    deploy_dir = config.artifacts_gnn_dir
    run_dir    = gnn_runs_root(config) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    deploy_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt  = run_dir / "best.pt"

    # Set up file logging for this run
    logger = configure_run_logging(run_dir / "run.log")
    logger.info("Starting GNN training run %s", run_id)
    logger.info("Config: %s", gnn_cfg)
    logger.info("GNN training device: %s", device)

    best_val_combined = 0.0
    best_val_fine_acc = 0.0
    best_val_eq_acc = 0.0
    patience_ctr = 0
    prev_train_loss: Optional[float] = None  # BS-1: loss collapse detection

    for epoch in range(gnn_cfg.epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Scene dropout: each node has scene_dropout_prob chance of being masked.
            # Masked nodes are excluded from per-node losses (fine label, clusters, ordinals).
            # Edge losses use all edges for simplicity — graph topology is fixed in the batch.
            if model.training and scene_dropout_prob > 0:
                node_keep_mask = torch.rand(batch.num_nodes, device=device) > scene_dropout_prob
            else:
                node_keep_mask = torch.ones(batch.num_nodes, dtype=torch.bool, device=device)

            fl, row_emb, row_ord_logits, col_emb, col_ord_logits, eq_logits, et_logits = model(batch)

            # Per-node losses -- applied only to kept nodes
            kept_fl            = fl[node_keep_mask]
            kept_y_fine        = batch.y_fine[node_keep_mask]
            kept_row_emb       = row_emb[node_keep_mask]
            kept_y_row_cluster = batch.y_row_cluster[node_keep_mask]
            kept_row_ord       = row_ord_logits[node_keep_mask]
            kept_y_row_ord     = batch.y_within_row_ord[node_keep_mask]
            kept_col_emb       = col_emb[node_keep_mask]
            kept_y_col_cluster = batch.y_col_cluster[node_keep_mask]
            kept_col_ord       = col_ord_logits[node_keep_mask]
            kept_y_col_ord     = batch.y_within_col_ord[node_keep_mask]

            if kept_fl.shape[0] == 0:
                # All nodes dropped -- skip this batch step
                continue

            loss_fine    = F.cross_entropy(kept_fl, kept_y_fine, weight=fine_class_weights)
            loss_row_cl  = _supervised_contrastive_loss(
                kept_row_emb, kept_y_row_cluster, temperature=gnn_cfg.temperature
            )
            loss_row_ord = F.cross_entropy(kept_row_ord, kept_y_row_ord)
            loss_col_cl  = _supervised_contrastive_loss(
                kept_col_emb, kept_y_col_cluster, temperature=gnn_cfg.temperature
            )
            loss_col_ord = F.cross_entropy(kept_col_ord, kept_y_col_ord)
            loss_eq      = F.cross_entropy(eq_logits, batch.y_eq, ignore_index=-1)

            # Edge type loss (all edges -- no masking by kept nodes)
            if et_logits.shape[0] > 0:
                loss_et = F.cross_entropy(et_logits, batch.y_edge_type)
            else:
                loss_et = torch.tensor(0.0, device=device)

            loss = (
                loss_weights["fine_label"]       * loss_fine
                + loss_weights["row_contrastive"]  * loss_row_cl
                + loss_weights["row_ordinal"]      * loss_row_ord
                + loss_weights["col_contrastive"]  * loss_col_cl
                + loss_weights["col_ordinal"]      * loss_col_ord
                + loss_weights["eq_type"]          * loss_eq
                + loss_weights["edge_type"]        * loss_et
            )
            loss.backward()
            optimizer.step()
            epoch_loss_sum += float(loss.item())
            epoch_batches += 1

        val_fine = _fine_acc(model, val_loader, device=device)
        val_eq = _eq_type_acc(model, val_loader, device=device)
        val_combined = combined_score(val_fine, val_eq)
        scheduler.step(val_combined)

        avg_loss = epoch_loss_sum / epoch_batches if epoch_batches > 0 else 0.0

        # BS-1: Loss collapse detection. A >50x drop in a single epoch indicates a
        # trivial dataset, vanishing-gradient memorization, or degenerate batches.
        if epoch > 0 and prev_train_loss is not None and avg_loss > 0 and prev_train_loss / max(avg_loss, 1e-9) > 50.0:
            logger.warning(
                "LOSS_COLLAPSE_DETECTED: epoch %d train_loss=%.6f, prev=%.6f, ratio=%.1fx "
                "(>50x indicates trivial dataset or memorization)",
                epoch + 1, avg_loss, prev_train_loss, prev_train_loss / max(avg_loss, 1e-9),
            )
        prev_train_loss = avg_loss

        logger.info(
            "Epoch %d/%d -- train_loss=%.4f val_fine_acc=%.4f val_eq_acc=%.4f val_combined=%.4f",
            epoch + 1, gnn_cfg.epochs, avg_loss, val_fine, val_eq, val_combined,
        )

        # BS-2: val_combined=1.0 at epoch <= 1 is almost certainly a degenerate val set
        # (single-class, all-same-symbol scenes, or accidental perfect-match data).
        if epoch < 2 and val_combined >= 0.999:
            logger.warning(
                "VAL_TRIVIAL_DETECTED: val_combined=%.4f at epoch %d. "
                "Likely degenerate val set (1-class, single-symbol scenes, or perfect-match data). "
                "Check class distribution + scene diversity.",
                val_combined, epoch,
            )

        if val_combined > best_val_combined:
            best_val_combined = val_combined
            best_val_fine_acc = val_fine
            best_val_eq_acc = val_eq
            patience_ctr = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience_ctr += 1
            if patience_ctr >= gnn_cfg.early_stop_patience:
                logger.info(
                    "Early stopping at epoch %d (no val improvement for %d epochs)",
                    epoch + 1, gnn_cfg.early_stop_patience,
                )
                break

    logger.info(
        "Training complete. Best val_combined=%.4f (fine_acc=%.4f, eq_acc=%.4f)",
        best_val_combined, best_val_fine_acc, best_val_eq_acc,
    )

    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    # ------------------------------------------------------------------ #
    # Test-set evaluation (delegates to compute_test_metrics)              #
    # ------------------------------------------------------------------ #
    logger.info("Starting test-set evaluation")
    test_metrics = compute_test_metrics(model, test_ds, device=device)
    logger.info("Test eval complete: %s", test_metrics)

    # ------------------------------------------------------------------ #
    # Report and manifest                                                   #
    # ------------------------------------------------------------------ #
    report_suffix = ".md"
    report_filename = "eval" + report_suffix
    report_path = run_dir / report_filename
    report_path.write_text(
        f"# GNN Eval\n\n"
        f"- best_val_combined: {best_val_combined:.6f}\n"
        f"- best_val_fine_acc: {best_val_fine_acc:.6f}\n"
        f"- best_val_eq_acc: {best_val_eq_acc:.6f}\n"
        f"- test_fine_label_accuracy: {test_metrics['fine_label_accuracy']:.6f}\n"
        f"- test_fine_label_macro_f1: {test_metrics['fine_label_macro_f1']:.6f}\n"
        f"- test_row_cluster_accuracy: {test_metrics['row_cluster_accuracy']:.6f}\n"
        f"- test_col_cluster_accuracy: {test_metrics['col_cluster_accuracy']:.6f}\n"
        f"- test_within_row_ord_accuracy: {test_metrics['within_row_ord_accuracy']:.6f}\n"
        f"- test_within_col_ord_accuracy: {test_metrics['within_col_ord_accuracy']:.6f}\n"
        f"- test_equation_type_accuracy: {test_metrics['equation_type_accuracy']:.6f}\n"
        f"- test_exact_scene_match: {test_metrics['exact_scene_match']:.6f}\n",
        encoding="utf-8",
    )
    shutil.copy2(best_ckpt, deploy_dir / "best.pt")

    rel_run = run_dir.relative_to(deploy_dir).as_posix()
    run_manifest: Dict[str, Any] = {
        "run_id": run_id,
        "stage": "gnn",
        "run_type": run_type,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_revision": try_git_revision(config.project_root),
        "hyperparams": {
            "epochs_requested": gnn_cfg.epochs,
            "batch_size": gnn_cfg.batch_size,
            "lr": gnn_cfg.lr,
            "dropout": gnn_cfg.dropout,
            "weight_decay": gnn_cfg.weight_decay,
            "early_stop_patience": gnn_cfg.early_stop_patience,
            "lr_patience": gnn_cfg.lr_patience,
            "lr_factor": gnn_cfg.lr_factor,
            "min_lr": gnn_cfg.min_lr,
            "temperature": gnn_cfg.temperature,
            "scene_dropout_prob": gnn_cfg.scene_dropout_prob,
        },
        "loss_weights": loss_weights,
        "metrics": {
            "val": {
                "fine_label_accuracy": best_val_fine_acc,
                "eq_type_accuracy": best_val_eq_acc,
                "combined_score": best_val_combined,
            },
            "test": test_metrics,
        },
    }
    if init_from_run_id is not None:
        run_manifest["init_from_run_id"] = init_from_run_id
    manifest_path = run_dir / "run.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    # BS-3 + BS-4: Pre-promotion quality gates. Both must pass or active.json is
    # NOT updated. The run dir and weights are preserved for diagnosis.
    PROMOTION_MACRO_F1_FLOOR = 0.30
    PROMOTION_CLUSTER_SUM_FLOOR = 0.20

    fine_macro_f1 = test_metrics.get("fine_label_macro_f1", 0.0)
    cluster_sum = (
        test_metrics.get("row_cluster_accuracy", 0.0)
        + test_metrics.get("col_cluster_accuracy", 0.0)
    )

    promotion_blocked = False
    if fine_macro_f1 < PROMOTION_MACRO_F1_FLOOR:
        logger.error(
            "PROMOTION_BLOCKED: fine_label_macro_f1=%.4f < floor %.2f. "
            "active.json NOT updated. Run dir preserved for diagnosis.",
            fine_macro_f1, PROMOTION_MACRO_F1_FLOOR,
        )
        promotion_blocked = True

    if cluster_sum < PROMOTION_CLUSTER_SUM_FLOOR:
        logger.error(
            "PROMOTION_BLOCKED: row_cluster_accuracy + col_cluster_accuracy = %.4f < floor %.2f. "
            "Both spatial heads degenerate. active.json NOT updated.",
            cluster_sum, PROMOTION_CLUSTER_SUM_FLOOR,
        )
        promotion_blocked = True

    promoted = not promotion_blocked
    if promoted:
        write_active_gnn(
            config,
            run_id,
            best_pt_relative=f"{rel_run}/best.pt",
            run_manifest_relative=f"{rel_run}/run.json",
            extra={"metrics": run_manifest["metrics"], "run_type": run_type},
        )

    append_runs_index(
        gnn_runs_root(config),
        {
            "run_id": run_id,
            "test_fine_acc": test_metrics["fine_label_accuracy"],
            "run_dir": rel_run,
        },
    )

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "best_model": deploy_dir / "best.pt" if promoted else None,
        "report": report_path,
        "run_manifest": manifest_path,
        "promoted": promoted,
    }
