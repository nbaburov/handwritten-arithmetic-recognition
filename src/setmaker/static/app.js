/*
 * set-maker annotator — single-screen draw -> detect -> fix -> save loop (WS-H).
 *
 * Vanilla JS over the vendored Konva global (no framework, no build step). The
 * file is organised top-down as small single-purpose units:
 *
 *   1. constants + label vocabulary (mirrors src/inference/palette.py)
 *   2. API client (the six routes; mode is a query param on next/detect/save/
 *      progress and a body field on /api/mode — matches src/setmaker/app.py)
 *   3. app state (one mutable object; never reach into the DOM for truth)
 *   4. Konva stage: a freehand draw layer + a box layer with one shared Transformer
 *   5. drafts <-> boxes <-> chips rendering
 *   6. inspector + keyboard (digit/operator keys set labels, arrows move
 *      selection, Enter saves and advances)
 *   7. the loop wiring (mode start, next, detect, save, progress)
 *
 * Coordinate contract: the stage is exactly 512x512, the same space the backend
 * preprocesses to, so every box x/y is already a bbox_px value. Nothing is scaled.
 */

(function () {
  "use strict";

  // ---------------------------------------------------------------- 1. consts

  const CANVAS = 512; // px; matches preprocess_for_pipeline output + bbox space.

  // Fine-label vocabulary = the 36-class stage-2 ontology
  // (src/core/ontology.py stage2_labels_ordered): role-prefixed digits
  // (main_/carry_/borrow_ 0..9), 4 operators, and 2 structural tokens. This is
  // the EXACT set every backend boundary uses (matcher target labels, the eval
  // sidecar loader, the train ground-truth, and both exporters' validation), so
  // it is the only set that round-trips. The previous 16-class set silently
  // failed every save because the exporter rejects bare digits like "2".
  const DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
  const DIGIT_ROLES = ["main", "carry", "borrow"];
  const OPERATORS = ["op_plus", "op_minus", "op_times", "op_divide"];
  const STRUCTURAL = ["result_bar", "div_bracket"];

  // As-drawn error tags, mirroring src/eval/labels.py ERROR_KINDS (kept in sync by
  // hand; the backend re-validates on save, so a drift here fails loudly there).
  // EVAL only: the human draws a deliberate error, then tags which kind it is.
  // The empty string is the "(none)" sentinel and is never sent to the backend.
  const ERROR_KIND_NONE = "";
  const ERROR_KINDS = [
    "missing_borrow",
    "missing_carry",
    "spurious_borrow",
    "spurious_carry",
    "wrong_borrow",
    "wrong_carry",
    "wrong_digit",
    "wrong_operator",
    "wrong_result",
    "other",
  ];

  // Picker groups (label-namespace partition) used to build the inspector's
  // grouped <select>: digits split by role, then operators, then structural.
  const LABEL_GROUPS = [
    ["main digit", DIGITS.map((d) => "main_" + d)],
    ["carry digit", DIGITS.map((d) => "carry_" + d)],
    ["borrow digit", DIGITS.map((d) => "borrow_" + d)],
    ["operator", OPERATORS.slice()],
    ["structural", STRUCTURAL.slice()],
  ];

  // Flat vocabulary in group order (the only label set the exporters accept).
  const FINE_LABELS = LABEL_GROUPS.reduce((acc, g) => acc.concat(g[1]), []);

  // Base glyph + human name per token, mirroring src/inference/palette.py
  // GLYPH_MAP (the inference-GUI single source of truth: + − × ÷, result bar,
  // bracket). palette.GLYPH_MAP has no carry_/borrow_ rows, so those are derived
  // here: the digit glyph carries a small role marker (ᶜ carry, ᵇ borrow) so a
  // carry/borrow box is visually distinct from a main digit on the canvas chip.
  const OP_GLYPH = {
    "op_plus": ["+", "plus"], "op_minus": ["−", "minus"],
    "op_times": ["×", "times"], "op_divide": ["÷", "divide"],
  };
  const STRUCT_GLYPH = {
    "result_bar": ["─", "result bar"], "div_bracket": ["┐", "division bracket"],
  };
  const ROLE_MARK = { main: "", carry: "ᶜ", borrow: "ᵇ" };

  // Build GLYPH for all 36 labels (+ the empty/unlabelled sentinel).
  const GLYPH = { "": ["?", "unlabelled"] };
  DIGIT_ROLES.forEach((role) => {
    DIGITS.forEach((d) => {
      GLYPH[role + "_" + d] = [d + ROLE_MARK[role], role + " " + d];
    });
  });
  Object.assign(GLYPH, OP_GLYPH, STRUCT_GLYPH);

  // Single-key bindings: digit keys label the selected box as a MAIN digit (the
  // common case). Operator keys set operators. Two mnemonic keys set structural
  // tokens. Carry/borrow are set via a two-key sequence: press c then a digit for
  // carry_<digit>; press v then a digit for borrow_<digit> (FIX 5).
  const KEY_TO_LABEL = {
    "0": "main_0", "1": "main_1", "2": "main_2", "3": "main_3", "4": "main_4",
    "5": "main_5", "6": "main_6", "7": "main_7", "8": "main_8", "9": "main_9",
    "+": "op_plus", "=": "op_plus",
    "-": "op_minus", "_": "op_minus",
    "*": "op_times",
    "/": "op_divide", ":": "op_divide",
    "b": "result_bar", "B": "result_bar",
    "k": "div_bracket", "K": "div_bracket",
  };

  // Colours (subset of palette.py): normal box, selected box, flagged box.
  const COL = { box: "#3996ff", selected: "#e3b341", flag: "#ed7118" };

  function glyphOf(label) { return (GLYPH[label] || [label, label])[0]; }
  function nameOf(label) { return (GLYPH[label] || [label, label])[1]; }

  // ---------------------------------------------------------------- 2. API

  // The active mode rides as a query param on every route except /api/mode, where
  // it is the body. Errors carry the backend's `detail` string (422/503) so the
  // banner can show the offending field rather than a generic failure.
  const api = {
    async _json(res) {
      let body = null;
      try { body = await res.json(); } catch (_e) { /* non-JSON error body */ }
      if (!res.ok) {
        const detail = (body && (body.detail || body.message)) || res.statusText;
        const err = new Error(detail);
        err.status = res.status;
        throw err;
      }
      return body;
    },
    async setMode(mode, roundBudget) {
      const res = await fetch("/api/mode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode, round_budget: roundBudget }),
      });
      return this._json(res);
    },
    async next(mode) {
      const res = await fetch("/api/next?mode=" + encodeURIComponent(mode));
      return this._json(res);
    },
    async detect(mode, imagePng) {
      const res = await fetch("/api/detect?mode=" + encodeURIComponent(mode), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_png: imagePng }),
      });
      return this._json(res);
    },
    async save(mode, payload) {
      const res = await fetch("/api/save?mode=" + encodeURIComponent(mode), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      return this._json(res);
    },
    async progress(mode) {
      const res = await fetch("/api/progress?mode=" + encodeURIComponent(mode));
      return this._json(res);
    },
  };

  // ---------------------------------------------------------------- 3. state

  // One source of truth for the screen. `drafts` is the editable model; Konva
  // rects and HTML chips are projections of it, rebuilt whenever it changes.
  const state = {
    mode: null,        // "eval" | "train" | null
    item: null,        // current worklist item from /api/next
    target: null,      // current TargetScene
    // Scene-kind fields echoed straight back on save so the backend never has to
    // regenerate the target at save time (that double-generation was a 503 risk
    // and re-ran the symbol pool). Captured from /api/next; sent on /api/save.
    sceneEqKind: null,   // eval equation_kind (EQUATION_KINDS member)
    sceneEqType: null,   // train equation_type (TRAIN_EQUATION_TYPES member)
    sceneCase: null,     // scene_case / case for the scene
    sceneStage: null,    // completion_stage for the scene
    errorKind: null,     // EVAL: chosen as-drawn error tag (ERROR_KINDS) or null
    directive: "normal", // "normal" | "error" — from the current worklist item
    drafts: [],        // array of draft objects (DraftModel shape)
    selected: -1,      // index into drafts, or -1
    drawing: false,
    busy: false,       // a request is in flight; gate the loop buttons
    resumable: {},     // mode -> scenes-left, from the page-load session probe
  };

  // ---------------------------------------------------------------- DOM refs

  const el = {
    modeEval: document.getElementById("mode-eval"),
    modeTrain: document.getElementById("mode-train"),
    budgetWrap: document.getElementById("budget-wrap"),
    roundBudget: document.getElementById("round-budget"),
    start: document.getElementById("mode-start"),
    resume: document.getElementById("mode-resume"),
    progressFill: document.getElementById("progress-fill"),
    progressText: document.getElementById("progress-text"),
    banner: document.getElementById("banner"),
    targetRef: document.getElementById("target-ref"),
    targetMeta: document.getElementById("target-meta"),
    errorKindWrap: document.getElementById("error-kind-wrap"),
    errorKind: document.getElementById("error-kind"),
    errorKindHint: document.getElementById("error-kind-hint"),
    errorDirectiveBanner: document.getElementById("error-directive-banner"),
    // B3 help-panel sub-blocks toggled by mode (the panel itself is always shown).
    helpGrid: document.getElementById("help-grid"),
    helpErrorEval: document.getElementById("help-error-eval"),
    helpErrorTrain: document.getElementById("help-error-train"),
    inspector: document.getElementById("inspector"),
    btnDetect: document.getElementById("btn-detect"),
    btnAdd: document.getElementById("btn-add"),
    btnClearDraw: document.getElementById("btn-clear-draw"),
    btnUndo: document.getElementById("btn-undo"),
    btnEraser: document.getElementById("btn-eraser"),
    btnSave: document.getElementById("btn-save"),
    penSize: document.getElementById("pen-size"),
    stageHost: document.getElementById("stage"),
    chipLayer: document.getElementById("chip-layer"),
  };

  function banner(msg, kind) {
    if (!msg) { el.banner.hidden = true; el.banner.textContent = ""; return; }
    el.banner.hidden = false;
    el.banner.textContent = msg;
    el.banner.dataset.kind = kind || "notice";
  }

  // ---------------------------------------------------------------- 4. Konva

  // Two layers on a 512x512 stage. drawLayer holds the freehand strokes the human
  // makes (flattened to a PNG for detect/save). boxLayer holds one Konva.Rect per
  // draft plus a single shared Transformer that attaches to the selected rect to
  // give drag + resize handles for free.
  const stage = new Konva.Stage({ container: el.stageHost, width: CANVAS, height: CANVAS });
  const drawLayer = new Konva.Layer();
  const boxLayer = new Konva.Layer();
  stage.add(drawLayer);
  stage.add(boxLayer);

  const transformer = new Konva.Transformer({
    rotateEnabled: false,
    keepRatio: false,
    borderStroke: COL.selected,
    anchorStroke: COL.selected,
    anchorFill: "#161b22",
    anchorSize: 9,
    ignoreStroke: true,
    boundBoxFunc: (oldBox, newBox) => (newBox.width < 4 || newBox.height < 4 ? oldBox : newBox),
  });
  boxLayer.add(transformer);

  let rects = []; // Konva.Rect[], parallel to state.drafts

  // -- freehand drawing (Pointer Events => stylus, pen, touch, and mouse) -----
  // Pointer Events (not mouse-only) so a stylus / drawing tablet works, which is
  // the whole point: real-handwriting proxy, not mouse scribbles. A new stroke
  // begins only when the pointer goes down on empty canvas (not on a box).
  let currentLine = null;
  let eraseMode = false;      // when true, clicks on strokes delete them; drawing is suppressed
  const penColor = "#101418"; // near-black; preprocessing composites onto white.

  function pointerPos() {
    const p = stage.getPointerPosition();
    return p ? [p.x, p.y] : null;
  }

  stage.on("pointerdown", (e) => {
    // Clicking a box selects it (handled per-rect); empty canvas starts a stroke.
    if (e.target !== stage) return;
    if (!state.mode) return;
    selectDraft(-1);
    const pos = pointerPos();
    if (!pos) return;
    // Eraser mode: do not start a new stroke (clicks are handled per-line).
    if (eraseMode) return;
    state.drawing = true;
    const penPx = Number(el.penSize.value) || 3;
    const line = new Konva.Line({
      points: pos,
      stroke: penColor,
      strokeWidth: penPx,
      lineCap: "round",
      lineJoin: "round",
      tension: 0,
      // Widen the invisible hit area so clicks near (but not pixel-perfect on)
      // a thin stroke still register for eraser deletion.
      hitStrokeWidth: Math.max(penPx, 12),
    });
    // Capture reference in closure so the handler works after currentLine is cleared.
    line.on("click tap", () => {
      if (!eraseMode) return;
      line.destroy();
      drawLayer.draw();
      syncButtons();
    });
    currentLine = line;
    drawLayer.add(currentLine);
  });

  stage.on("pointermove", () => {
    if (!state.drawing || !currentLine) return;
    const pos = pointerPos();
    if (!pos) return;
    currentLine.points(currentLine.points().concat(pos));
    drawLayer.batchDraw();
  });

  function endStroke() {
    if (!state.drawing) return;
    state.drawing = false;
    currentLine = null;
    syncButtons();
  }
  stage.on("pointerup", endStroke);
  stage.on("pointercancel", endStroke);
  // Pointer leaving the stage mid-stroke should still end it cleanly.
  el.stageHost.addEventListener("pointerleave", endStroke);

  function hasDrawing() { return drawLayer.getChildren().length > 0; }

  function clearDrawing() {
    drawLayer.destroyChildren();
    drawLayer.draw();
    setEraseMode(false);   // fresh scene always starts in pen mode
    syncButtons();
  }

  // Remove the most recently drawn stroke. No-op if the draw layer is empty.
  function undoLastStroke() {
    const children = drawLayer.getChildren();
    if (children.length === 0) return;
    children[children.length - 1].destroy();
    drawLayer.draw();
    syncButtons();
  }

  // Apply or remove eraser mode and keep button + cursor in sync.
  function setEraseMode(on) {
    eraseMode = on;
    // Visual: set aria-pressed on the button so the CSS active rule fires.
    if (el.btnEraser) el.btnEraser.setAttribute("aria-pressed", String(on));
    // Cursor: crosshair in eraser mode to signal "click to delete", else default.
    const stageContainer = stage.container();
    stageContainer.style.cursor = on ? "crosshair" : "";
  }

  function toggleEraser() {
    setEraseMode(!eraseMode);
  }

  // Flatten the draw layer alone to a base64 PNG (boxes excluded; the backend
  // wants the human's strokes, not the annotation overlay). The data-URL prefix
  // is accepted by app.py's _decode_png.
  function drawingDataUrl() {
    return drawLayer.toDataURL({ pixelRatio: 1, width: CANVAS, height: CANVAS, x: 0, y: 0 });
  }

  // ---------------------------------------------------------------- 5. boxes

  function clampBox(x0, y0, x1, y1) {
    // Keep boxes inside the 512 frame and normalise to x0<x1, y0<y1 so the
    // exporter's 0 <= a < b <= 512 rule is satisfied client-side too.
    let a = Math.max(0, Math.min(x0, x1));
    let b = Math.max(0, Math.min(y0, y1));
    let c = Math.min(CANVAS, Math.max(x0, x1));
    let d = Math.min(CANVAS, Math.max(y0, y1));
    if (c - a < 2) c = Math.min(CANVAS, a + 2);
    if (d - b < 2) d = Math.min(CANVAS, b + 2);
    return [a, b, c, d];
  }

  function boxColor(draft, isSelected) {
    if (isSelected) return COL.selected;
    return draft.flagged ? COL.flag : COL.box;
  }

  // Rebuild every rect + chip from state.drafts. Called after any model change.
  function renderBoxes() {
    transformer.nodes([]);
    rects.forEach((r) => r.destroy());
    rects = [];
    el.chipLayer.innerHTML = "";

    state.drafts.forEach((draft, idx) => {
      const [x0, y0, x1, y1] = draft.bbox_px;
      const rect = new Konva.Rect({
        x: x0, y: y0, width: x1 - x0, height: y1 - y0,
        stroke: boxColor(draft, idx === state.selected),
        strokeWidth: idx === state.selected ? 2.5 : 1.8,
        fill: "rgba(57,150,255,0.06)",
        draggable: true,
        name: "box",
      });
      rect.on("mousedown touchstart", (e) => { e.cancelBubble = true; selectDraft(idx); });
      rect.on("dragmove transform", () => writeBackRect(idx, rect));
      rect.on("dragend transformend", () => { writeBackRect(idx, rect); positionChip(idx); });
      boxLayer.add(rect);
      rects.push(rect);
      makeChip(draft, idx);
    });

    applySelectionVisual();
    boxLayer.draw();
    state.drafts.forEach((_d, idx) => positionChip(idx));
  }

  // Pull a rect's geometry (after drag/resize) back into the draft model. The
  // Transformer scales the node, so bake scale into width/height then reset it.
  function writeBackRect(idx, rect) {
    const sx = rect.scaleX();
    const sy = rect.scaleY();
    const w = rect.width() * sx;
    const h = rect.height() * sy;
    rect.scaleX(1);
    rect.scaleY(1);
    rect.width(w);
    rect.height(h);
    const [a, b, c, d] = clampBox(rect.x(), rect.y(), rect.x() + w, rect.y() + h);
    rect.position({ x: a, y: b });
    rect.width(c - a);
    rect.height(d - b);
    state.drafts[idx].bbox_px = [a, b, c, d];
    positionChip(idx);
  }

  // -- chip overlay (per-box attribute label, plain HTML over the canvas) ------
  function makeChip(draft, idx) {
    const chip = document.createElement("div");
    chip.className = "sm-chip";
    chip.dataset.idx = String(idx);
    chip.style.setProperty("--chip", draft.flagged ? COL.flag : COL.box);
    chip.dataset.flagged = String(!!draft.flagged);
    chip.addEventListener("mousedown", (e) => { e.stopPropagation(); selectDraft(idx); });
    el.chipLayer.appendChild(chip);
    paintChip(idx);
  }

  function paintChip(idx) {
    const chip = el.chipLayer.querySelector('.sm-chip[data-idx="' + idx + '"]');
    if (!chip) return;
    const d = state.drafts[idx];
    const lbl = d.fine_label ? glyphOf(d.fine_label) : "?";
    let html = '<span class="sm-chip-glyph">' + escapeHtml(lbl) + "</span>";
    if (state.mode === "train") {
      html += '<span class="sm-chip-src">r' + d.row_index + "c" + d.col_index + "</span>";
    } else {
      html += '<span class="sm-chip-src">' + escapeHtml(d.source) + "</span>";
    }
    if (d.flagged) html += '<span class="sm-chip-flag" title="flagged">⚠</span>';
    chip.innerHTML = html;
    chip.style.setProperty("--chip", d.flagged ? COL.flag : COL.box);
    chip.dataset.flagged = String(!!d.flagged);
  }

  function positionChip(idx) {
    const chip = el.chipLayer.querySelector('.sm-chip[data-idx="' + idx + '"]');
    if (!chip) return;
    const [x0, y0] = state.drafts[idx].bbox_px;
    chip.style.left = Math.max(0, x0) + "px";
    chip.style.top = Math.max(12, y0 - 2) + "px";
  }

  function applySelectionVisual() {
    rects.forEach((r, i) => {
      const d = state.drafts[i];
      r.stroke(boxColor(d, i === state.selected));
      r.strokeWidth(i === state.selected ? 2.5 : 1.8);
    });
    if (state.selected >= 0 && rects[state.selected]) {
      transformer.nodes([rects[state.selected]]);
    } else {
      transformer.nodes([]);
    }
    boxLayer.batchDraw();
  }

  // ---------------------------------------------------------------- selection

  function selectDraft(idx) {
    state.selected = idx >= 0 && idx < state.drafts.length ? idx : -1;
    applySelectionVisual();
    renderInspector();
  }

  function moveSelection(delta) {
    if (state.drafts.length === 0) return;
    let i = state.selected < 0 ? (delta > 0 ? 0 : state.drafts.length - 1) : state.selected + delta;
    if (i < 0) i = state.drafts.length - 1;
    if (i >= state.drafts.length) i = 0;
    selectDraft(i);
  }

  // Any human touch (label edit or clearing a flag) makes the box human-owned.
  // Provenance becomes "manual" regardless of its prior value ("matched" /
  // "detected" / "manual"), because the human just took responsibility for it; a
  // prior "matched" box whose label the human corrected must not stay "matched".
  function markConfirmed(d) {
    d.source = "manual";
  }

  function setLabelOnSelected(label) {
    if (state.selected < 0) return;
    const d = state.drafts[state.selected];
    d.fine_label = label;
    // A human-set label clears the flag and marks the box manual-confirmed: the
    // flag exists to demand review, and the review just happened.
    d.flagged = false;
    markConfirmed(d);
    paintChip(state.selected);
    positionChip(state.selected);
    applySelectionVisual();
    renderInspector();
  }

  function toggleFlag() {
    if (state.selected < 0) return;
    const d = state.drafts[state.selected];
    d.flagged = !d.flagged;
    // Clearing a flag is a human review decision -> the box is now manual-owned.
    if (!d.flagged) markConfirmed(d);
    paintChip(state.selected);
    applySelectionVisual();
    renderInspector();
  }

  function deleteSelected() {
    if (state.selected < 0) return;
    state.drafts.splice(state.selected, 1);
    state.selected = -1;
    renderBoxes();
    renderInspector();
  }

  function addBox() {
    // Drop a small unlabelled box in the centre for the human to place + label.
    const c = CANVAS / 2;
    state.drafts.push({
      bbox_px: [c - 20, c - 24, c + 20, c + 24],
      fine_label: "",
      row_index: -1, col_index: -1, equation_idx: 0,
      confidence: 0.0, source: "manual", flagged: true,
    });
    renderBoxes();
    selectDraft(state.drafts.length - 1);
  }

  // ---------------------------------------------------------------- 6. inspector

  // The inspector edits the selected draft. EVAL hides row/col/equation_idx (the
  // eval exporter ignores them — they are inferred geometrically at eval time);
  // TRAIN shows them as editable number fields (the human-confirmed grid).
  function renderInspector() {
    if (state.selected < 0) {
      el.inspector.innerHTML = '<p class="sm-empty">Nothing selected. Click a box, or draw and press Detect.</p>';
      return;
    }
    const d = state.drafts[state.selected];
    // Grouped picker: <optgroup> per label namespace (main/carry/borrow/op/
    // structural) so the 36 options are scannable. Every backend-emitted label
    // therefore appears and is selectable; the leading "(none)" clears the label.
    const groups = LABEL_GROUPS.map(([groupName, labels]) => {
      const opts = labels.map(
        (l) => '<option value="' + l + '"' + (l === d.fine_label ? " selected" : "") +
          ">" + escapeHtml(glyphOf(l)) + "  " + escapeHtml(nameOf(l)) + "</option>"
      ).join("");
      return '<optgroup label="' + escapeHtml(groupName) + '">' + opts + "</optgroup>";
    }).join("");
    const showGrid = state.mode === "train";

    el.inspector.innerHTML = [
      '<div class="sm-insp-row">',
      '  <div class="sm-insp-glyph">' + escapeHtml(d.fine_label ? glyphOf(d.fine_label) : "?") + "</div>",
      '  <div class="sm-insp-meta">',
      "    <strong>" + escapeHtml(d.fine_label ? nameOf(d.fine_label) : "unlabelled") + "</strong><br>",
      "    source " + escapeHtml(d.source) + " &middot; conf " + d.confidence.toFixed(2),
      "  </div>",
      "</div>",
      '<div class="sm-field"><label for="insp-label">fine label</label>',
      '  <select id="insp-label"><option value=""' + (d.fine_label ? "" : " selected") + ">(none)</option>" + groups + "</select></div>",
      showGrid ? gridFieldsHtml(d) : "",
      '<div class="sm-field sm-field--flag"><label><input type="checkbox" id="insp-flag"' +
        (d.flagged ? " checked" : "") + "> flagged (needs review)</label></div>",
      '<div class="sm-insp-actions">',
      '  <button class="sm-btn" id="insp-del" type="button">Delete box</button>',
      "</div>",
    ].join("\n");

    document.getElementById("insp-label").addEventListener("change", (e) => {
      const v = e.target.value;
      // Clearing to "(none)" is still a human edit, so the box becomes manual.
      if (v) { setLabelOnSelected(v); } else { d.fine_label = ""; markConfirmed(d); paintChip(state.selected); renderInspector(); }
    });
    document.getElementById("insp-flag").addEventListener("change", (e) => {
      d.flagged = e.target.checked;
      // Unchecking the flag in the inspector is a review decision -> manual.
      if (!d.flagged) markConfirmed(d);
      paintChip(state.selected); applySelectionVisual(); renderInspector();
    });
    document.getElementById("insp-del").addEventListener("click", deleteSelected);
    if (showGrid) bindGridFields(d);
  }

  // TRAIN inspector grid fields, each with a short inline hint directly under it.
  // The hints restate the 0-based counting rule at the point of edit so the human
  // does not have to open the Help panel for every box. They mirror the longer
  // explanations in #help-grid (single source of the wording is the help panel;
  // these are the compressed reminders).
  function gridFieldsHtml(d) {
    return [
      '<div class="sm-field"><label for="insp-row">row index</label>',
      '  <input id="insp-row" type="number" step="1" value="' + d.row_index + '"></div>',
      '<p class="sm-field-hint">Horizontal line, 0 = top. Result is its own row below the operands; a carry/borrow shares the row above its digits.</p>',
      '<div class="sm-field"><label for="insp-col">col index</label>',
      '  <input id="insp-col" type="number" step="1" value="' + d.col_index + '"></div>',
      '<p class="sm-field-hint">Place-value column, 0 = leftmost, units on the right. Vertically aligned digits share a column.</p>',
      '<div class="sm-field"><label for="insp-eq">equation idx</label>',
      '  <input id="insp-eq" type="number" step="1" value="' + d.equation_idx + '"></div>',
      '<p class="sm-field-hint">0 for a single equation (almost always 0 here).</p>',
    ].join("\n");
  }

  function bindGridFields(d) {
    const bind = (id, key) => {
      const node = document.getElementById(id);
      if (!node) return;
      node.addEventListener("change", (e) => {
        const n = parseInt(e.target.value, 10);
        d[key] = Number.isFinite(n) ? n : d[key];
        paintChip(state.selected);
      });
    };
    bind("insp-row", "row_index");
    bind("insp-col", "col_index");
    bind("insp-eq", "equation_idx");
  }

  // ---------------------------------------------------------------- keyboard

  // pendingRole: tracks the active carry/borrow prefix key (FIX 5).
  // When "c" is pressed, pendingRole = "carry"; when "v" is pressed,
  // pendingRole = "borrow". The next digit key applies carry_<digit> or
  // borrow_<digit>. Any non-digit key clears the pending role.
  let pendingRole = null;

  // Keyboard-first: digit/operator keys label the selected box, arrows move the
  // selection, Enter saves and advances. Skipped while typing in a form field so
  // numbers in the inspector inputs do not get hijacked.
  document.addEventListener("keydown", (e) => {
    const tag = (e.target && e.target.tagName) || "";
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;

    // Ctrl+Z / Cmd+Z: undo last stroke. Handled before the modifier guard so
    // the standard undo chord works even though other modifier combos are blocked.
    if ((e.ctrlKey || e.metaKey) && !e.altKey && e.key === "z") {
      e.preventDefault();
      undoLastStroke();
      return;
    }

    if (e.metaKey || e.ctrlKey || e.altKey) return;

    if (e.key === "Enter") { pendingRole = null; e.preventDefault(); doSave(); return; }
    if (e.key === "ArrowRight" || e.key === "ArrowDown") { pendingRole = null; e.preventDefault(); moveSelection(1); return; }
    if (e.key === "ArrowLeft" || e.key === "ArrowUp") { pendingRole = null; e.preventDefault(); moveSelection(-1); return; }
    // Delete only (not Backspace): Backspace is a destructive accidental trigger
    // when the focus is not in a field, and there is no undo for a box deletion.
    if (e.key === "Delete") { pendingRole = null; e.preventDefault(); deleteSelected(); return; }
    if (e.key === "f" || e.key === "F") { pendingRole = null; e.preventDefault(); toggleFlag(); return; }
    if (e.key === "n" || e.key === "N") { pendingRole = null; e.preventDefault(); if (state.mode) addBox(); return; }
    if (e.key === "d" || e.key === "D") { pendingRole = null; e.preventDefault(); doDetect(); return; }
    if (e.key === "e" || e.key === "E") { pendingRole = null; e.preventDefault(); if (state.mode) toggleEraser(); return; }

    // FIX 5: two-key carry/borrow sequence.
    // "c" arms carry mode; "v" arms borrow mode. The subsequent digit applies the role.
    if (e.key === "c" || e.key === "C") { e.preventDefault(); pendingRole = "carry"; return; }
    if (e.key === "v" || e.key === "V") { e.preventDefault(); pendingRole = "borrow"; return; }

    // Digit keys: if a carry/borrow role is pending apply it, else use KEY_TO_LABEL.
    const isDigit = e.key >= "0" && e.key <= "9";
    if (isDigit && pendingRole) {
      e.preventDefault();
      setLabelOnSelected(pendingRole + "_" + e.key);
      pendingRole = null;
      return;
    }

    if (Object.prototype.hasOwnProperty.call(KEY_TO_LABEL, e.key)) {
      pendingRole = null;
      e.preventDefault();
      setLabelOnSelected(KEY_TO_LABEL[e.key]);
    } else {
      // Any unhandled key clears the pending role.
      pendingRole = null;
    }
  });

  // ---------------------------------------------------------------- 7. loop

  function syncButtons() {
    const live = !!state.mode && !state.busy;
    el.btnDetect.disabled = !live || !hasDrawing();
    el.btnAdd.disabled = !live;
    el.btnClearDraw.disabled = !live || !hasDrawing();
    if (el.btnUndo) el.btnUndo.disabled = !live || !hasDrawing();
    if (el.btnEraser) el.btnEraser.disabled = !live;
    // For error-directive items, Save is blocked until a non-(none) error_kind is chosen
    // regardless of mode (eval and train both require the tag on directive="error" items).
    const errorKindMissing = state.directive === "error" &&
      (!el.errorKind || !el.errorKind.value || el.errorKind.value === ERROR_KIND_NONE);
    el.btnSave.disabled = !live || errorKindMissing;
    el.start.disabled = state.busy || (!el.modeEval.dataset.armed && !el.modeTrain.dataset.armed);
    if (el.resume) el.resume.disabled = state.busy;
  }

  function armMode(mode) {
    el.modeEval.setAttribute("aria-pressed", String(mode === "eval"));
    el.modeTrain.setAttribute("aria-pressed", String(mode === "train"));
    el.modeEval.dataset.armed = mode === "eval" ? "1" : "";
    el.modeTrain.dataset.armed = mode === "train" ? "1" : "";
    el.budgetWrap.hidden = mode !== "train";
    // Offer Resume only when the armed mode has a persisted session with scenes
    // still left. Start always rebuilds the worklist at cursor 0; Resume reads
    // the saved cursor instead, so prior in-progress work is not discarded.
    const left = (state.resumable && state.resumable[mode]) || 0;
    if (el.resume) {
      el.resume.hidden = !(mode && left > 0);
      el.resume.textContent = left > 0 ? "Resume (" + left + " left)" : "Resume";
    }
    syncButtons();
  }

  // Build a 2-D grid HTML string from the target symbols list.
  // Layout rules:
  //   - Rows are stacked top-to-bottom by row_index.
  //   - Within each row, cells are placed at their col_index.
  //   - carry_* / borrow_* fine_labels render as small raised superscripts.
  //   - result_bar spans all occupied columns as a horizontal rule.
  //   - div_bracket renders as the "┐" glyph.
  //   - All other labels pass through GLYPH (digits, operators).
  //   - Empty cells are rendered as invisible spacers so columns align.
  function buildTargetGrid(symbols) {
    if (!symbols || symbols.length === 0) return "<em>no symbols</em>";

    // Gather per-row buckets keyed by col_index.
    // Separate carries/borrows (superscript row above their main row) from the
    // main-row tokens so they can be rendered raised.
    const mainRows = {};   // row_index -> { col_index -> symbol }
    const superRows = {};  // row_index -> { col_index -> symbol }  (carry/borrow)

    symbols.forEach((sym) => {
      const lbl = sym.fine_label || "";
      const isCarryBorrow = lbl.startsWith("carry_") || lbl.startsWith("borrow_");
      const bucket = isCarryBorrow ? superRows : mainRows;
      if (!bucket[sym.row_index]) bucket[sym.row_index] = {};
      // Last writer wins on exact collision (shouldn't happen in clean data).
      bucket[sym.row_index][sym.col_index] = sym;
    });

    // Collect all row indices that need rendering, sorted ascending.
    const allRowIdxs = Array.from(
      new Set([...Object.keys(mainRows), ...Object.keys(superRows)].map(Number))
    ).sort((a, b) => a - b);

    // Global column span across all rows so every row is the same width.
    let minCol = Infinity, maxCol = -Infinity;
    symbols.forEach((sym) => {
      if (sym.col_index < minCol) minCol = sym.col_index;
      if (sym.col_index > maxCol) maxCol = sym.col_index;
    });
    if (minCol === Infinity) return "<em>no symbols</em>";
    const colSpan = maxCol - minCol + 1;

    function cellHtml(sym, colIdx) {
      if (!sym) {
        // Empty spacer — invisible but takes up the cell.
        return '<td class="tg-cell tg-cell--empty"> </td>';
      }
      const lbl = sym.fine_label || "";

      // result_bar: span all columns as a thin rule.
      if (lbl === "result_bar") {
        return '<td class="tg-cell tg-cell--bar" colspan="' + colSpan + '"><hr class="tg-bar"></td>';
      }

      // div_bracket: drawn by CSS as a vertical wall; the overline (vinculum)
      // over the dividend is added by the main-row loop (tg-cell--dividend).
      if (lbl === "div_bracket") {
        return '<td class="tg-cell tg-cell--bracket">&nbsp;</td>';
      }

      // carry / borrow (in the super-row): digit as raised small text.
      const isCB = lbl.startsWith("carry_") || lbl.startsWith("borrow_");
      if (isCB) {
        const digit = lbl.split("_")[1] || "?";
        const cls = lbl.startsWith("carry_") ? "tg-carry" : "tg-borrow";
        return '<td class="tg-cell tg-cell--super"><sup class="' + cls + '">' +
          escapeHtml(digit) + "</sup></td>";
      }

      // Normal token: digit or operator glyph.
      const glyph = glyphOf(lbl);
      return '<td class="tg-cell">' + escapeHtml(glyph) + "</td>";
    }

    let html = '<table class="tg-grid">';

    allRowIdxs.forEach((rowIdx) => {
      const superBucket = superRows[rowIdx] || {};
      const mainBucket  = mainRows[rowIdx]  || {};

      // Only emit the superscript row if there is at least one carry/borrow here.
      if (Object.keys(superBucket).length > 0) {
        html += '<tr class="tg-row tg-row--super">';
        for (let c = minCol; c <= maxCol; c++) {
          html += cellHtml(superBucket[c] || null, c);
        }
        html += "</tr>";
      }

      // Main row — result_bar is special: if present, emit a single spanning cell
      // for the entire row (first result_bar col wins).
      const barSym = Object.values(mainBucket).find((s) => s.fine_label === "result_bar");
      if (barSym) {
        html += '<tr class="tg-row tg-row--bar">';
        html += cellHtml(barSym, minCol);  // spanning cell handles the full width
        html += "</tr>";
      } else {
        // Division bracket: cells to its right get an overline (the vinculum),
        // so the dividend reads as sitting under a proper division bracket.
        // A bracket with no dividend to its right (standalone_bracket) is drawn
        // as a self-contained bracket glyph (left vertical + overline stub).
        let bracketCol = null;
        for (const [cc, sObj] of Object.entries(mainBucket)) {
          if (sObj && sObj.fine_label === "div_bracket") { bracketCol = Number(cc); break; }
        }
        let hasDividend = false;
        if (bracketCol !== null) {
          for (const [cc, sObj] of Object.entries(mainBucket)) {
            if (sObj && Number(cc) > bracketCol && sObj.fine_label !== "div_bracket") {
              hasDividend = true; break;
            }
          }
        }
        html += '<tr class="tg-row">';
        for (let c = minCol; c <= maxCol; c++) {
          let cell = cellHtml(mainBucket[c] || null, c);
          if (bracketCol !== null && c === bracketCol && !hasDividend) {
            cell = cell.replace('tg-cell--bracket', 'tg-cell--bracket tg-cell--bracket-solo');
          } else if (bracketCol !== null && hasDividend && c > bracketCol) {
            const extra = (c === bracketCol + 1)
              ? "tg-cell--dividend tg-cell--dividend-first"
              : "tg-cell--dividend";
            cell = cell.replace('class="tg-cell', 'class="tg-cell ' + extra);
          }
          html += cell;
        }
        html += "</tr>";
      }
    });

    html += "</table>";
    return html;
  }

  // ---------------------------------------------------------------- error kind

  // Populate the EVAL error-kind <select> once: a "(none)" sentinel first, then
  // every ERROR_KINDS tag. Idempotent (clears first) so it is safe to call again.
  function initErrorKindSelect() {
    if (!el.errorKind) return;
    const opts = ['<option value="">(none)</option>'];
    ERROR_KINDS.forEach((k) => {
      opts.push('<option value="' + escapeHtml(k) + '">' + escapeHtml(k) + "</option>");
    });
    el.errorKind.innerHTML = opts.join("");
    el.errorKind.value = ERROR_KIND_NONE;
  }

  // The error-kind tag applies to both eval and train error-directive items. When
  // the current item has directive="error", the control is forced visible AND
  // marked required (shown with a visual cue). For normal (non-error) items in
  // train mode the control is hidden; for normal eval items it is visible but
  // optional (allows tagging incidental errors on clean-redraw items).
  function updateErrorKindVisibility() {
    if (!el.errorKindWrap) return;
    const isEval = state.mode === "eval";
    const isError = state.directive === "error";
    // Show for eval (always) OR for any error-directive item (eval or train).
    el.errorKindWrap.hidden = !isEval && !isError;
    // Visual required marker: add/remove a CSS class the stylesheet can style.
    if (el.errorKindWrap) {
      if (isError) {
        el.errorKindWrap.classList.add("sm-error-kind--required");
      } else {
        el.errorKindWrap.classList.remove("sm-error-kind--required");
      }
    }
    // Error-directive banner: prominent instruction shown above the canvas.
    if (el.errorDirectiveBanner) {
      el.errorDirectiveBanner.hidden = !isError;
    }
    // Inline hint follows the dropdown visibility (shown when wrap is visible).
    if (el.errorKindHint) el.errorKindHint.hidden = !isEval && !isError;
    updateHelpForMode();
  }

  // Toggle the mode-sensitive blocks inside the always-visible Help panel.
  // row/col/equation_idx are TRAIN-only inspector fields, so their explanations
  // are shown in train mode and before any mode is chosen (a first-time reader
  // benefits), and hidden in eval where those fields do not exist. The error
  // paragraph switches between the eval wording (draw-the-mistake + tag it) and
  // the train wording (label the wrong symbols as-drawn, no sidecar tag).
  function updateHelpForMode() {
    const isTrain = state.mode === "train";
    const isEval = state.mode === "eval";
    if (el.helpGrid) el.helpGrid.hidden = isEval;        // shown for train + no-mode
    if (el.helpErrorEval) el.helpErrorEval.hidden = isTrain;
    if (el.helpErrorTrain) el.helpErrorTrain.hidden = !isTrain;
  }

  // Reset the selection to "(none)" between scenes so an error tag never leaks
  // from one drawn scene onto the next (each scene is tagged on its own merits).
  // directive is reset by loadNext from the incoming item; this only clears the
  // dropdown value so the human starts fresh on the next scene.
  function resetErrorKind() {
    state.errorKind = null;
    if (el.errorKind) el.errorKind.value = ERROR_KIND_NONE;
    // Re-evaluate Save gating now that the error kind is cleared.
    syncButtons();
  }

  function renderTarget() {
    if (!state.target) {
      el.targetRef.innerHTML = state.mode
        ? "<em>Worklist complete for this mode.</em>"
        : "<em>No worklist. Pick a mode and press Start.</em>";
      el.targetMeta.innerHTML = "";
      return;
    }
    // 2-D grid render when symbols are available; fall back to the flat
    // reference string for older targets that carry no symbols array.
    const syms = state.target.symbols;
    if (syms && syms.length > 0) {
      el.targetRef.innerHTML = buildTargetGrid(syms);
    } else {
      el.targetRef.textContent = state.target.reference || "(no reference string)";
    }
    const rows = [
      ["case", state.target.case],
      ["type", state.target.equation_type],
      ["stage", state.target.completion_stage],
      ["symbols", String((state.target.symbols || []).length)],
      ["seed", String(state.target.seed)],
    ];
    el.targetMeta.innerHTML = rows
      .map(([k, v]) => "<dt>" + escapeHtml(k) + "</dt><dd>" + escapeHtml(v) + "</dd>")
      .join("");
  }

  function renderProgress(p) {
    if (!p || typeof p.done !== "number" || typeof p.total !== "number") {
      el.progressText.textContent = state.mode ? state.mode : "no worklist";
      return;
    }
    const pct = p.total > 0 ? Math.round((p.done / p.total) * 100) : 0;
    el.progressFill.style.width = pct + "%";
    const left = typeof p.left === "number" ? p.left : Math.max(0, p.total - p.done);
    el.progressText.textContent = p.done + " / " + p.total + " done · " + left + " left · " + pct + "%";
  }

  async function refreshProgress() {
    if (!state.mode) return;
    try { renderProgress(await api.progress(state.mode)); }
    catch (err) { /* progress is non-critical; keep the last shown value */ }
  }

  async function startMode(mode) {
    if (state.busy) return;
    state.busy = true; syncButtons();
    banner("Building " + mode + " worklist…", "notice");
    try {
      const budget = mode === "train" ? Math.max(1, parseInt(el.roundBudget.value, 10) || 20) : undefined;
      const res = await api.setMode(mode, budget);
      state.mode = mode;
      updateErrorKindVisibility();
      banner(res.warning || ("Planned " + res.planned + " " + mode + " scene(s)."), res.warning ? "warn" : "ok");
      renderProgress(res.progress);
      await loadNext();
    } catch (err) {
      banner("Could not start " + mode + " mode: " + err.message, "error");
    } finally {
      state.busy = false; syncButtons();
    }
  }

  // Resume the persisted worklist for ``mode`` WITHOUT rebuilding it. /api/mode
  // would re-plan at cursor 0 and overwrite the saved cursor; resume just adopts
  // the mode and reads the next outstanding scene the backend already has on
  // disk (state.load trims the completed prefix), so prior work is preserved.
  async function resumeMode(mode) {
    if (state.busy) return;
    state.busy = true; syncButtons();
    banner("Resuming " + mode + " session…", "notice");
    try {
      state.mode = mode;
      updateErrorKindVisibility();
      await refreshProgress();
      await loadNext();
    } catch (err) {
      banner("Could not resume " + mode + " mode: " + err.message, "error");
    } finally {
      state.busy = false; syncButtons();
    }
  }

  // On page load, probe both modes' persisted sessions so the UI can offer
  // Resume. /api/progress reads the trimmed session, so ``overall.pending`` is
  // the scene count still outstanding; a mode with none is not resumable.
  async function probeResumable() {
    const found = {};
    await Promise.all(["eval", "train"].map(async (mode) => {
      try {
        const p = await api.progress(mode);
        const overall = (p && p.overall) || {};
        const pending = typeof overall.pending === "number"
          ? overall.pending
          : Math.max(0, (overall.total || 0) - (overall.done || 0));
        if (pending > 0) found[mode] = pending;
      } catch (_e) { /* no session for this mode; not resumable */ }
    }));
    state.resumable = found;
    const modes = Object.keys(found);
    if (modes.length) {
      const parts = modes.map((m) => m + " (" + found[m] + " left)").join(", ");
      banner("Resumable session: " + parts + ". Arm a mode and click Resume to continue, or Start to rebuild.", "notice");
    }
    // Re-evaluate the Resume button for whatever mode is already armed.
    const armed = el.modeTrain.dataset.armed ? "train" : (el.modeEval.dataset.armed ? "eval" : null);
    if (armed) armMode(armed);
  }

  async function loadNext() {
    state.busy = true; syncButtons();
    try {
      const res = await api.next(state.mode);
      clearDrawing();
      state.drafts = [];
      state.selected = -1;
      // New scene: clear any error-kind tag so it never carries over from the
      // previous drawing (each scene's error is tagged independently).
      resetErrorKind();
      renderBoxes();
      renderInspector();
      if (res.done || res.complete) {
        state.item = null;
        state.target = null;
        state.sceneEqKind = null;
        state.sceneEqType = null;
        state.sceneCase = null;
        state.sceneStage = null;
        state.directive = "normal";
        updateErrorKindVisibility();
        renderTarget();
        // Disable the draw/detect/save controls — there is nothing left to draw.
        // The mode selector stays active so the human can switch modes.
        if (res.progress) renderProgress(res.progress.overall);
        banner("Eval set complete — nothing left to draw.", "ok");
        // Explicitly disable action buttons; syncButtons will keep them disabled
        // because state.target is null (doSave guards on state.target).
        syncButtons();
      } else {
        state.item = res.item;
        state.target = res.target;
        // Echo-on-save fields: prefer the route's resolved scene kinds, falling
        // back to the target / item so an older backend still works.
        state.sceneEqKind = res.equation_kind || null;
        state.sceneEqType = res.equation_type || (res.target && res.target.equation_type) || null;
        state.sceneCase = (res.item && res.item.case) || (res.target && res.target.case) || null;
        state.sceneStage = (res.item && res.item.completion_stage) ||
          (res.target && res.target.completion_stage) || null;
        state.directive = (res.item && res.item.directive) || "normal";
        updateErrorKindVisibility();
        renderTarget();
        const sceneMsg = state.directive === "error"
          ? "ERROR SCENE — draw with a deliberate mistake. Scene " + (res.cursor + 1) + " of " + res.total + "."
          : "Draw the target, then press Detect (d). Scene " + (res.cursor + 1) + " of " + res.total + ".";
        banner(sceneMsg, state.directive === "error" ? "warn" : "notice");
      }
    } catch (err) {
      // 503 = pool unavailable (run validate first); show the backend's detail.
      banner("Could not load next scene: " + err.message, "error");
    } finally {
      state.busy = false; syncButtons();
    }
  }

  async function doDetect() {
    if (!state.mode || state.busy) return;
    if (!hasDrawing()) { banner("Draw something first, then press Detect.", "notice"); return; }
    state.busy = true; syncButtons();
    banner("Detecting boxes…", "notice");
    try {
      const res = await api.detect(state.mode, drawingDataUrl());
      state.drafts = (res.drafts || []).map(normaliseDraft);
      state.selected = -1;
      renderBoxes();
      // Auto-select the first flagged box so the human starts where review is due.
      const firstFlag = state.drafts.findIndex((d) => d.flagged);
      selectDraft(firstFlag >= 0 ? firstFlag : (state.drafts.length ? 0 : -1));
      const flagged = state.drafts.filter((d) => d.flagged).length;
      banner(
        res.notice ||
          (state.drafts.length + " box(es) · " + flagged + " flagged for review · " +
            res.detection_count + " detected vs " + res.target_symbol_count + " target."),
        res.notice ? "notice" : (flagged ? "warn" : "ok")
      );
    } catch (err) {
      banner("Detect failed: " + err.message, "error");
    } finally {
      state.busy = false; syncButtons();
    }
  }

  // Coerce a server draft into the local shape, defaulting any missing field so
  // the editor never touches undefined (the server may omit grid fields in eval).
  function normaliseDraft(d) {
    const b = d.bbox_px || [0, 0, 10, 10];
    return {
      bbox_px: clampBox(b[0], b[1], b[2], b[3]),
      fine_label: d.fine_label || "",
      row_index: Number.isFinite(d.row_index) ? d.row_index : -1,
      col_index: Number.isFinite(d.col_index) ? d.col_index : -1,
      equation_idx: Number.isFinite(d.equation_idx) ? d.equation_idx : 0,
      confidence: Number.isFinite(d.confidence) ? d.confidence : 0.0,
      source: d.source || "detected",
      flagged: !!d.flagged,
    };
  }

  async function doSave() {
    if (!state.mode || state.busy) return;
    if (!state.target) { banner("Nothing to save: the worklist is complete.", "notice"); return; }
    // Guard: error-directive items require a non-(none) error_kind before saving,
    // in both eval and train mode.
    if (state.directive === "error") {
      const chosen = el.errorKind ? el.errorKind.value : "";
      if (!chosen || chosen === ERROR_KIND_NONE) {
        banner("Choose an error kind in the dropdown before saving this error scene.", "warn");
        if (el.errorKind) el.errorKind.focus();
        return;
      }
    }
    // Guard the common mistake: saving boxes that still have no label.
    const unlabelled = state.drafts.filter((d) => !d.fine_label).length;
    if (unlabelled > 0) {
      banner(unlabelled + " box(es) still have no label. Label them (digit/operator keys) or delete them before saving.", "warn");
      const first = state.drafts.findIndex((d) => !d.fine_label);
      if (first >= 0) selectDraft(first);
      return;
    }
    // doSave owns busy only for the /api/save call itself. Once the save
    // succeeds, busy is released before calling loadNext so that loadNext is
    // the sole owner of busy during the auto-advance. This prevents a double-
    // release in doSave's finally that would fire after loadNext has already
    // cleared busy, and keeps the ownership of each async step unambiguous.
    state.busy = true; syncButtons();
    banner("Saving…", "notice");
    let saveRes;
    try {
      // Echo the scene-kind fields received on /api/next so the backend writes
      // them straight through instead of regenerating the target at save time.
      const payload = {
        image_png: drawingDataUrl(),
        drafts: state.drafts.map(toServerDraft),
      };
      if (state.sceneEqKind != null) payload.equation_kind = state.sceneEqKind;
      if (state.sceneEqType != null) payload.equation_type = state.sceneEqType;
      if (state.sceneCase != null) { payload.scene_case = state.sceneCase; payload.case = state.sceneCase; }
      if (state.sceneStage != null) payload.completion_stage = state.sceneStage;
      // Attach the chosen as-drawn error tag for both eval and train when a real
      // kind is picked (directive="error" items in either mode). "(none)" is the
      // empty sentinel and is left off so the backend omits it from the artifact.
      if (el.errorKind) {
        const chosen = el.errorKind.value || ERROR_KIND_NONE;
        if (chosen !== ERROR_KIND_NONE) payload.error_kind = chosen;
      }
      saveRes = await api.save(state.mode, payload);
    } catch (err) {
      // 422 = export validation failure (bad bbox / unknown label); nothing written.
      banner("Save rejected: " + err.message, "error");
      state.busy = false; syncButtons();
      return;
    }
    // Save succeeded — release busy before auto-advance so loadNext is the
    // sole busy owner for the next step.
    state.busy = false; syncButtons();
    renderProgress(saveRes.progress);
    banner("Saved " + saveRes.stem + ". Loading next…", "ok");
    await loadNext();           // AUTO-ADVANCE; loadNext manages busy itself
    await refreshProgress();
  }

  // Project a local draft to the exact DraftModel the backend validates.
  function toServerDraft(d) {
    return {
      bbox_px: d.bbox_px.map((v) => Math.round(v * 100) / 100),
      fine_label: d.fine_label,
      row_index: d.row_index,
      col_index: d.col_index,
      equation_idx: d.equation_idx,
      confidence: d.confidence,
      source: d.source,
      flagged: d.flagged,
    };
  }

  // ---------------------------------------------------------------- util

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  // ---------------------------------------------------------------- wire-up

  function armedMode() {
    return el.modeTrain.dataset.armed ? "train" : (el.modeEval.dataset.armed ? "eval" : null);
  }

  el.modeEval.addEventListener("click", () => armMode("eval"));
  el.modeTrain.addEventListener("click", () => armMode("train"));
  el.start.addEventListener("click", () => {
    const mode = armedMode();
    if (!mode) return;
    // Starting rebuilds the worklist, so this mode is no longer resume-from-saved.
    delete state.resumable[mode];
    armMode(mode);
    startMode(mode);
  });
  if (el.resume) {
    el.resume.addEventListener("click", () => {
      const mode = armedMode();
      if (!mode || !(state.resumable[mode] > 0)) return;
      delete state.resumable[mode];
      armMode(mode);
      resumeMode(mode);
    });
  }
  el.btnDetect.addEventListener("click", doDetect);
  el.btnAdd.addEventListener("click", addBox);
  el.btnClearDraw.addEventListener("click", clearDrawing);
  if (el.btnUndo) el.btnUndo.addEventListener("click", undoLastStroke);
  if (el.btnEraser) el.btnEraser.addEventListener("click", toggleEraser);
  el.btnSave.addEventListener("click", doSave);
  if (el.errorKind) {
    // Keep state in sync with the live <select> so the value survives focus
    // changes; doSave still reads the element directly as the source of truth.
    // Also call syncButtons so the Save button re-enables the moment a valid
    // error kind is chosen (for error-directive items it was blocked until now).
    el.errorKind.addEventListener("change", () => {
      const v = el.errorKind.value || ERROR_KIND_NONE;
      state.errorKind = v === ERROR_KIND_NONE ? null : v;
      syncButtons();
    });
  }

  initErrorKindSelect();
  updateErrorKindVisibility();
  renderBoxes();
  renderInspector();
  syncButtons();
  // Probe persisted sessions so Resume can be offered before any mode is started.
  probeResumable();
})();
