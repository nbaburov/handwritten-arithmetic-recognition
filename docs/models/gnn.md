# GNN Stage 2 — Symbol Classifier and Spatial Parser

## What it is

Stage 2 takes the bounding boxes from YOLO and answers: **which digit is this? which row and column does it sit in? what kind of equation is this?**

It is a **GNN — Graph Neural Network**. Not a CNN. Not a transformer. A graph network.

- **Graph** = a set of nodes connected by edges. Here each node is one detected symbol (one bounding box). Edges connect nearby symbols spatially.
- **Neural Network on a graph** = instead of processing pixels in a grid (CNN) or tokens in a sequence (transformer), the network passes messages between nodes along edges to let each node learn from its neighbours.

Total parameters: roughly **500K** (larger than the original 278K due to the dense graph and new heads, still far fewer than YOLO's 3.2M).

The model is split into two parts that run together:

1. **CropBackbone** — a small CNN that reads the pixel crop of each symbol and turns it into a 128-number vector describing what it looks like visually.
2. **SymbolGNN** — a graph network that takes those visual vectors plus spatial position info and classifies each symbol in context.

---

## Iter2 capacity upgrade rationale

The original CropBackbone (iter1) had 4 layers and produced a 64-dim embedding. Fine-label accuracy hit a ceiling at 0.51 — the "trio collapse" problem: `main_5`, `carry_5`, and `borrow_5` are visually identical 28×28 crops, so the 36-class design forced the visual embedding to encode role, which it could not do reliably from pixel content alone.

The fix was a two-part redesign:

1. **16-class ontology** — the fine_label head predicts visual class only (10 digits + 4 operators + result_bar + div_bracket). Role (main/carry/borrow) is encoded in the 13-dim spatial feature vector, not the fine_label output. This removes the trio collapse entirely.
2. **CropBackbone v2** — 6 conv layers producing a 128-dim embedding (up from 4 layers / 64-dim). This lifted fine_label accuracy from 0.51 to 0.95.

---

## Why a graph at all

YOLO gives you isolated boxes. It does not know that the "3" above another "3" is a carry digit, not a main digit. The only way to know is context: where is it relative to other symbols?

A graph encodes that spatial context. Edges connect every ordered pair of nodes (dense N×N graph). The GNN updates each node's representation by aggregating information from its connected neighbours. After two rounds of this, each node has seen information from across the scene and can use position, size, and visual appearance of nearby symbols to disambiguate its own label.

A CNN or MLP operating per-symbol independently cannot do this. A transformer could, but would be overkill for a small fixed-size scene.

---

## Step 1: Building the graph

Before any neural network runs, the raw YOLO detections are converted into a graph object. This happens in `graph_builder.py`.

For each detected symbol (node), three things are computed:

### Pixel crop
The bounding box region is cut out of the grayscale image and resized to exactly **28×28 pixels**. This is the raw image patch that the CropBackbone will read.

### Node feature vector (13 numbers)
Each node gets a 13-dimensional feature vector of non-visual, geometric properties:

| Feature | Meaning |
|---------|---------|
| `cx / image_width` | Horizontal centre position, normalised 0–1 |
| `cy / image_height` | Vertical centre position, normalised 0–1 |
| `bw / image_width` | Box width, normalised 0–1 |
| `bh / image_height` | Box height, normalised 0–1 |
| `(bw × bh) / (img_w × img_h)` | Relative area of the box |
| 6× one-hot YOLO class | Which of the 6 YOLO coarse classes YOLO assigned (e.g. digit_main=1,0,0,0,0,0) |
| `confidence` | YOLO detection confidence score |
| `role_importance` | Scalar: 1.0 for operator/bar/bracket, 0.5 for main digits, 0.3 for carry/borrow |

One-hot means: 6 numbers, all zero except a 1 at the index of the assigned class. It tells the GNN the coarse type without assuming the embedding already knows it.

Normalising positions and sizes to 0–1 makes the features scale-invariant — the model sees the same numbers regardless of image resolution. The role-importance scalar gives the GNN an explicit signal about structural anchor nodes (operators, result bars) before any message passing.

### Edges (dense N×N graph)
Edges connect **every ordered pair of distinct nodes** — a fully connected directed graph with N×(N−1) edges. There is no k-NN distance cutoff. This lets the attention mechanism learn which relationships matter rather than hard-coding spatial proximity. Scenes with more than 80 nodes are truncated to the 80 highest-confidence detections (`data.truncated = True`).

Each edge has a **24-dimensional feature**: 16 geometric dimensions plus an 8-dimensional learned edge-type embedding.

**16 geometric dimensions:**

| Dim | Feature | Meaning |
|-----|---------|---------|
| 0 | `dx` | Normalised horizontal distance (src→dst) |
| 1 | `dy` | Normalised vertical distance |
| 2 | `dist` | Euclidean distance between centres |
| 3 | `w_ratio` | dst width / src width |
| 4 | `h_ratio` | dst height / src height |
| 5 | `angle_sin` | sin of the angle from src to dst |
| 6 | `angle_cos` | cos of the angle |
| 7 | `col_delta_norm` | dx normalised by median digit width in the scene |
| 8 | `row_delta_norm` | dy normalised by median digit height in the scene |
| 9 | `size_ratio` | dst area / src area |
| 10 | `log_dist` | log(dist) |
| 11 | `src_is_operator` | 1 if source role is operator |
| 12 | `dst_is_operator` | 1 if destination role is operator |
| 13 | `src_is_bar_or_bracket` | 1 if source is result_bar or divide_bracket |
| 14 | `dst_is_bar_or_bracket` | 1 if destination is result_bar or divide_bracket |
| 15 | `same_role_flag` | 1 if both nodes share the same YOLO coarse class |

**8-dim edge-type embedding:** the YOLO coarse classes of the two endpoint nodes are mapped to one of 8 vocabulary types (digit–digit, digit–carry, digit–operator, etc.) and looked up in a learned `nn.Embedding(8, 8)`. This embedding is concatenated with the 16 geometric dims before attention scoring, so the GNN can learn different attention patterns per relationship type.

Scene-median normalisation (`col_delta_norm`, `row_delta_norm`) uses only `digit_main` nodes to compute the median. This makes the spatial prior robust to the presence of small carry/borrow symbols or large structural elements.

---

## Step 2: CropBackbone v2 — visual embedding per symbol

A 6-layer CNN that reads the 28×28 pixel crop of each symbol and outputs a **128-number vector** summarising what it looks like visually (CROP_EMBED_DIM = 128).

Architecture (6 layers with batch normalisation):

| Layer | Operation | Input shape | Output shape |
|-------|-----------|------------|--------------|
| conv1 | Conv2d(1→16, 3×3) + BN + ReLU | (1, 28, 28) | (16, 26, 26) |
| conv2 | Conv2d(16→32, 3×3) + BN + ReLU | (16, 26, 26) | (32, 24, 24) |
| pool1 | MaxPool2d(2×2) | (32, 24, 24) | (32, 12, 12) |
| conv3 | Conv2d(32→64, 3×3) + BN + ReLU | (32, 12, 12) | (64, 10, 10) |
| conv4 | Conv2d(64→64, 3×3) + BN + ReLU | (64, 10, 10) | (64, 8, 8) |
| pool2 | MaxPool2d(2×2) | (64, 8, 8) | (64, 4, 4) |
| conv5 | Conv2d(64→128, 3×3) + BN + ReLU | (64, 4, 4) | (128, 2, 2) = 512 flat |
| fc | Linear(512→128) + ReLU | (512,) | (128,) |

**BatchNorm** after each conv: normalises each output channel to have mean≈0 and std≈1 across the batch. Stabilises training and reduces sensitivity to learning rate.

**MaxPool2d(2×2)** = take the maximum value in each 2×2 block. Halves the spatial size. Makes the representation robust to small shifts.

**ReLU** = activation: set all negative values to zero. Without this the whole network is linear and cannot learn curves or loops.

Output: one 128-dim vector per symbol. This is the visual description of the symbol's appearance.

No pretrained weights. This backbone is **trained from scratch** together with the GNN.

---

## Step 3: SymbolGNN — two-stream graph attention over neighbours

The full GNN does not run a single shared stack of GATv2 layers. It splits into **two independent streams** that each process the same graph but see different slices of the node features. This split is the architectural fix for a coupling problem found in earlier iterations: when one shared stream fed every head, the row/column/equation-type predictions could read the operator class (carried in the role one-hot) and leaned on it instead of on pure geometry. Separating the streams severs that path by construction.

The two streams are:

1. **Visual stream** — sees the full **141-dim node input** (the 128-dim CropBackbone visual vector concatenated with all 13 geometric node features, NODE_DIM = 141). It feeds the `fine_label` head and the auxiliary `edge_type` head, both of which are allowed to be role-aware.
2. **Spatial stream** — sees only a **7-dim subset** of the node features (the 5 geometric coordinates `cx, cy, bw, bh, area`, plus `confidence` and `role_importance`). The 6-dim role one-hot is stripped out before the stream ever sees it (`_spatial_x`, SPATIAL_STREAM_INPUT_DIM = 7). It feeds the `row_cluster`, `within_row_ord`, `col_cluster`, `within_col_ord`, and `equation_type` heads. Because the role one-hot is absent, gradients from these heads cannot back-propagate into operator-label dimensions, so the spatial predictions are **operator-independent by design**.

Both streams use the same 24-dim edge feature (16 geometric + 8-dim edge-type embedding) and run over the same dense N×N graph.

### What GATv2 is

GATv2 = Graph Attention Network version 2. A graph convolution layer with learnable attention.

**Graph convolution** means: each node collects messages from its neighbours and aggregates them. The aggregated message updates the node's representation. This is repeated for each layer.

**Attention** means: not all neighbours are equally important. The layer learns a scalar **attention weight** (0–1) for each edge. A neighbour with high attention weight contributes more to the update. The weights are computed from the features of both source and target nodes plus the edge attributes, so the model learns which geometric relationships matter (e.g. a small symbol directly above gets high attention weight when deciding if the current node is a main digit).

**GATv2 vs GAT**: in the original GAT, the attention score is a static linear function. In GATv2, the two node vectors are concatenated before computing the score, making it fully dynamic — the importance of a neighbour depends jointly on both its features and the receiving node's features.

### Visual stream

Two GATv2 layers, `vis_gat1` and `vis_gat2`.

```
vis_gat1: GATv2Conv(141 → 128, heads=4, edge_dim=24, concat=True)   # -> 512
vis_gat2: GATv2Conv(512 → 64,  heads=1, edge_dim=24, concat=False)  # -> 64
```

- `vis_gat1` input: each node has all 141 features. 4 independent attention heads, each producing 128 features; `concat=True` joins them into **512 features per node**.
- `vis_gat2` input: those 512 features. 1 head, 64 output features; `concat=False` with 1 head means the output is just the 64-dim vector.
- Edge features (24-dim) are fed into the attention score computation at both layers.
- Activation: ELU (similar to ReLU but smooth at zero, avoids dead neurons).
- Dropout applied between the two layers: randomly zeros a fraction of features during training to prevent overfitting.

**Multiple heads** = the model learns 4 different attention patterns simultaneously. One head might specialise in attending to symbols in the same row. Another might specialise in carry/borrow relationships. Concatenating them gives the next layer access to all 4 patterns.

The output is a **64-dim visual embedding per node** that feeds the `fine_label` and `edge_type` heads.

### Spatial stream

Two GATv2 layers, `spa_gat1` and `spa_gat2`, with their own weights independent of the visual stream.

```
spa_gat1: GATv2Conv(7 → 64,   heads=2, edge_dim=24, concat=True)   # -> 128
spa_gat2: GATv2Conv(128 → 64, heads=1, edge_dim=24, concat=False)  # -> 64
```

- `spa_gat1` input: the 7-dim geometry-plus-meta subset only, no role one-hot. 2 attention heads, each producing 64 features; `concat=True` joins them into **128 features per node** (SPATIAL_STREAM_HIDDEN = 64, SPATIAL_STREAM_HEADS = 2).
- `spa_gat2` input: those 128 features. 1 head, 64 output features; `concat=False` (SPATIAL_STREAM_OUT = 64).
- Same 24-dim edge feature and same ELU activation and dropout pattern as the visual stream.

The output is a **64-dim spatial embedding per node** that feeds the `row_cluster`, `within_row_ord`, `col_cluster`, `within_col_ord`, and `equation_type` heads.

After two rounds of message passing in each stream, every node carries two 64-dim vectors: a role-aware visual one and an operator-independent spatial one. Each encodes information aggregated from every other symbol in the scene (not just k nearest), but along the feature dimensions that stream was allowed to see.

---

## Step 4: Output heads — what is predicted

Seven heads in total. Each reads from the 64-dim embedding of its assigned stream: the `fine_label` and `edge_type` heads read the visual stream; the row, column, and equation-type heads read the spatial stream. All are trained simultaneously.

| Head | Reads | Architecture | Output | What it predicts |
|------|-------|-------------|--------|-----------------|
| `fine_label_head` | visual | Linear(64→16) | 16 logits per node | Which of 16 visual classes: 0–9 (digit), op_plus/minus/times/divide, result_bar, div_bracket. Role (main/carry/borrow) is encoded in the spatial features, not here. |
| `row_cluster_head` | spatial | Linear(64→16) | 16-dim embedding per node | Continuous embedding trained to cluster same-row nodes together (NT-Xent contrastive) |
| `within_row_ord_head` | spatial | Linear(64→8) | 8 logits per node | Left-to-right ordinal position within the predicted row (0–7) |
| `col_cluster_head` | spatial | Linear(64→16) | 16-dim embedding per node | Continuous embedding trained to cluster same-column nodes together |
| `within_col_ord_head` | spatial | Linear(64→8) | 8 logits per node | Top-to-bottom ordinal position within the predicted column (0–7) |
| `eq_fc1+eq_fc2` | spatial | Linear(64→32)+ReLU+Linear(32→4) | 4 logits per graph | Which equation type: add, subtract, multiply, divide |
| `edge_head` | visual | Linear(152→64)+ReLU+Linear(64→8) | 8 logits per edge | Auxiliary: which of 8 edge-type vocabulary classes this edge represents |

**Logits** = raw unnormalised scores. Higher = more confident. During inference, the highest logit wins (argmax). During training, softmax converts them to probabilities and cross-entropy loss penalises wrong predictions.

### The scene virtual node and the equation-type head

The equation-type head needs one vector for the whole scene, not one per node. How it obtains that vector depends on the **scene virtual node** flag (`config.toml [gnn] use_scene_virtual_node`, enabled in the shipped config.toml; the dataclass fallback is off).

A scene virtual node is a synthetic extra node appended to the graph after the real nodes. It carries an 8-dim scene-summary feature vector (symbol counts, density, maximum coordinates) rather than a pixel crop, and it is wired with edges to every real node, so during message passing it accumulates a global picture of the scene. Real nodes are first truncated to MAX_NODES = 80; the virtual node is appended after that, making the final node count 81.

- **Virtual node enabled (default):** the equation-type head reads the spatial-stream embedding of the virtual node directly. Because that node is always the last one per graph, the forward pass picks out the max-index node for each graph and feeds its 64-dim spatial vector into `eq_fc1+eq_fc2`. One prediction per image.
- **Virtual node disabled (ablation only):** there is no dedicated scene node, so the head falls back to `global_mean_pool` over the spatial-stream embeddings of all real nodes, averaging them into one 64-dim scene vector before classification.

Either way the equation-type head reads the spatial stream, so it stays operator-independent: it judges equation type from layout, not from which operator symbol was detected.

**16 fine labels** = 10 digit classes (0–9) + 4 operators (op_plus, op_minus, op_times, op_divide) + result_bar + div_bracket. This is a visual classification: the digit class "7" covers main_7, carry_7, and borrow_7 alike. Role disambiguation happens in the assembler using the YOLO coarse class and spatial position, not the fine_label head.

**Row/col cluster heads** replace the old absolute row/col classifier heads. Instead of predicting a hard-coded row number, the model emits a continuous embedding that is trained (via contrastive loss) to place same-row nodes close together and different-row nodes far apart. At inference, row assignment uses gap-based clustering on bbox y-coordinates (`_cluster_by_coordinate(bbox, axis=1)`) — not the cluster embeddings directly. The within-ordinal heads then resolve the left-to-right or top-to-bottom ordering within each group.

**Edge head** (auxiliary) concatenates src/dst node embeddings with the 24-dim edge feature (64+64+24 = 152) and classifies the edge-type vocabulary. This auxiliary loss at weight 0.3 encourages the model to learn the 8-token edge-type vocabulary explicitly.

---

## Training: from scratch, no pretrained weights

The GNN is trained entirely from scratch on synthetic data. There is no pretrained checkpoint to start from.

**Data source**: the same synthetic dataset used for YOLO. Each training sample is one full arithmetic scene: the image, the ground-truth bounding boxes, and a JSON with the fine label, row, and column for every symbol.

**Dataset split**: train / val / test (same manifests as YOLO stage).

**Loss function**: seven terms, weighted and summed.

```
total_loss = 2.0 × CE(fine_label)
           + NT-Xent(row_cluster_emb, y_row_cluster)
           + CE(within_row_ord)
           + NT-Xent(col_cluster_emb, y_col_cluster)
           + CE(within_col_ord)
           + CE(equation_type)
           + 0.3 × CE(edge_type)
```

**Cross-entropy loss (CE)**: given the model's probability distribution over classes and the true class, CE = -log(probability assigned to the correct class). If the model is confident and right, the loss is near zero. If it is confident and wrong, the loss is large.

The fine label loss is **weighted by 2.0** because it is the most important prediction.

**NT-Xent (supervised contrastive) loss**: used for the row and column cluster embeddings. Same-row nodes are pulled together in embedding space; different-row nodes are pushed apart. Temperature = 0.1. This replaces the old absolute-row cross-entropy head and removes the hard-coded row/column grid assumption.

**Edge-type loss** (weight 0.3) is auxiliary — it encourages the GNN to learn explicit relationship types between node pairs. If it degrades fine-label accuracy by more than 1 point it can be reduced to 0.1 or disabled; the weight is recorded in the run manifest for tracking.

**Scene dropout** (15%): during training, 15% of nodes are randomly masked out of each batch step. The loss is computed only on surviving nodes. This forces robustness to partial scenes (not all symbols visible).

**Class weights on fine_label**: carry and borrow digits are rarer than main digits in the synthetic data. Without correction, the model would mostly see main digits and learn to ignore carries. Balanced class weights are computed from the training set: rarer classes get higher weight, so their loss contribution is scaled up proportionally. Computed using `sklearn`'s `compute_class_weight("balanced", ...)`.

**Optimiser**: Adam with `lr=0.001`. Adam adapts the learning rate per parameter using estimates of gradient mean and variance, converging faster than plain SGD on sparse updates like those in graph networks.

**Learning rate scheduler**: `ReduceLROnPlateau` with patience=5, factor=0.5. If val fine-label accuracy does not improve for 5 epochs, the learning rate is halved (down to minimum 1e-6). This avoids overshooting near convergence.

**Early stopping**: if val fine-label accuracy does not improve for 10 consecutive epochs, training halts. Best weights are saved and restored.

---

## Hyperparameters (active run)

| Param | Value | Meaning |
|-------|-------|---------|
| `epochs_requested` | 60 | Max epochs before forced stop |
| `batch_size` | 32 | Scenes per gradient update |
| `lr` | 0.001 | Initial learning rate |
| `early_stop_patience` | 8 | Epochs without val improvement before stop |
| `lr_patience` | 5 | Epochs without val improvement before halving LR |
| `dropout` | 0.2 | Fraction of features randomly zeroed during training |
| `scene_dropout_prob` | 0.15 | Fraction of nodes masked per training step |
| `max_nodes` | 80 | Max nodes per scene (truncated by confidence) |
| `NT-Xent temperature` | 0.1 | Contrastive loss temperature for cluster embeddings |
| `edge_type_loss_weight` | 0.3 | Weight of auxiliary edge-type CE term |

---

## Results

The active GNN is run `20260622T123233Z_7dfc6c79` (iter11 cloud, warm-start fine-tune from iter10-R6 `14df2140`, early-stopped at epoch 25). The live pointer is `artifacts/gnn/active.json`; the full run registry and lineage are in `docs/runs.md`. Test-split metrics on synthetic data:

| Metric | Value (active run) |
|--------|--------------------|
| fine_acc | 0.9875 |
| fine_macro_f1 | 0.9901 |
| row | 0.9348 |
| col | 0.7280 |
| within_row_ord | 0.7445 |
| within_col_ord | 0.7546 |
| eq_type | 0.9053 |
| exact_scene | 0.5071 |

**Iter11 promotion.** The warm-start fine-tune from iter10-R6 was promoted on real-data evaluation (190-scene set): row_acc 0.854 (+0.5pp) and col_acc 0.792 (+6.2pp) improved significantly on real handwriting. Synthetic-test eq_type (0.9053 vs 0.9006) rose slightly, and row-cluster embeddings strengthened (0.9348 vs 0.9209). The prior iter10-R6 is retained as rollback target.

Per-run detail is also written to `artifacts/gnn/runs/<id>/eval.md`. Secondary note: if the synthetic dataset is regenerated, run `train --stage gnn` to produce a fresh checkpoint, then update `artifacts/gnn/active.json` and `docs/runs.md` to point at the new run.

---

## Full data flow summary

```
YOLO detections (N boxes with coarse labels, max 80 kept by confidence)
  |
  v
build_graph()
  +- crop each box -> resize to 28x28
  +- compute 13-dim geometric features per node (incl. role_importance)
  +- connect all N*(N-1) ordered pairs with 16-dim edge attributes + edge_type long tensor
  |
  v
CropBackbone v2 (per node, in parallel)
  Conv->BN->Conv->BN->Pool->Conv->BN->Conv->BN->Pool->Conv->BN->Flatten->Linear -> 128-dim visual vector
  |
  v
edge_type_emb(edge_type) -> 8-dim embedding
cat(edge_attr_16, emb_8) -> 24-dim edge feature
(optional) append scene virtual node (8-dim summary) -> node 81 after MAX_NODES=80 truncation
  |
  +======================== two independent streams ========================+
  |                                                                          |
  v VISUAL STREAM                                  v SPATIAL STREAM          |
  full 141-dim node input                          7-dim subset (no role 1-hot)
  (visual 128 + geometric 13)                      (cx,cy,bw,bh,area,conf,importance)
  |                                                |
  vis_gat1: 141 -> 512 (4 heads x 128)             spa_gat1: 7 -> 128 (2 heads x 64)
  | Dropout                                        | Dropout
  vis_gat2: 512 -> 64 (1 head)                     spa_gat2: 128 -> 64 (1 head)
  |                                                |
  +- fine_label_head -> 16 logits -> visual class  +- row_cluster_head    -> 16-dim embedding (contrastive)
  +- edge_head (aux)  -> 8 logits per edge          +- within_row_ord_head -> 8 logits -> ordinal in row
                                                    +- col_cluster_head    -> 16-dim embedding (contrastive)
                                                    +- within_col_ord_head -> 8 logits -> ordinal in column
                                                    +- eq_type (spatial): if virtual node -> read its
                                                       embedding directly; else global_mean_pool over
                                                       real nodes -> eq_fc1+eq_fc2 -> 4 logits
  |
  v (inference only)
_cluster_by_coordinate(bbox, axis=1) -> row_cluster_id per node
_cluster_by_coordinate(bbox, axis=0) -> col_cluster_id per node
  |
  v
NodePrediction(fine_label, row_cluster_id, col_cluster_id,
               within_row_ord, within_col_ord, bbox, confidence)
  |
  v
Assembler -> structured JSON (schema_version=1, rows, slots)
```

---

## No transformers. No encoders. No self-attention over a sequence.

GATv2 uses attention, but it is **graph attention** — scores over edges between spatially nearby nodes, not a global all-to-all attention matrix over a sequence. There are no positional embeddings, no query/key/value projection matrices in the transformer sense, no layer norm, no feed-forward sublayers. It is a graph network with local learned attention weights, which is a structurally different and much lighter mechanism.
