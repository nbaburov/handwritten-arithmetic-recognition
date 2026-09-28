/*
 * feedback demo: draw -> Check -> recognize -> grade -> on-canvas popups.
 *
 * Vanilla JS over the vendored Konva global (no framework, no build step).
 * File sections:
 *   1. constants + STYLE block (Konva-shape equivalent of CSS :root tokens;
 *      single source of truth for all Konva shape properties)
 *   2. API client (GET /api/exercises, POST /api/exercise/select,
 *      POST /api/evaluate, GET /api/score)
 *   3. app state (one mutable object; the DOM is a projection, never truth)
 *   4. Konva stage: five layers: gridLayer (writing guide), scaffoldLayer
 *      (pre-printed problem), drawLayer (freehand), glyphLayer (recognition
 *      overlay at real bboxes), popupLayer (status markers + callouts)
 *   5. freehand drawing + flatten-to-512-base64-PNG
 *   6. select / evaluate flow + result-panel + popup rendering
 *   7. wiring (picker, Check, Reset)
 *
 * Coordinate contract: the stage is 512x512, matching the backend's preprocessed
 * image space, so every pixel_rect and glyph bbox is already in the correct frame.
 * Nothing is scaled.
 *
 * Model-first: overlays and popup anchors use the REAL glyph bboxes from the
 * YOLO+GNN pipeline (token.bbox / popup.pixelRect). The guide grid is a writing
 * aid only; recognition and feedback placement never depend on it. Colours come
 * from token.color (derived by palette.py / annotation.py on the backend).
 */
(function () {
  "use strict";

  // ---------------------------------------------------------------- 1. consts

  const CANVAS = 512; // px; matches preprocess_for_pipeline output.

  // Hide (not delete) the dashed FILL/borrow guide boxes that mark where to
  // write. Flip to true to bring the guide cells back. Printed problem glyphs
  // (operands, operator, bar, bracket) are unaffected.
  const SHOW_GUIDE_BOXES = false;

  // STYLE: single source of truth for all Konva shape properties (five layers).
  // CSS :root tokens cover HTML chrome; STYLE covers the canvas.
  const STYLE = {
    // gridLayer: faint writing guide
    grid: {
      stroke: "rgba(0,0,0,0.13)", // --dk-grid-line
      strokeWidth: 1,
      dash: [4, 4],
      opacity: 1,
      listening: false,
    },

    // scaffoldLayer: pre-printed problem
    scaffold: {
      digit: {
        fontFamily: '"SF Mono", ui-monospace, Menlo, monospace',
        fontSize: 40, // ~64 px cell
        fill: "#1a1814", // --dk-scaffold
        opacity: 0.85,
        listening: false,
      },
      bar: {
        stroke: "#1a1814",
        strokeWidth: 3,
        opacity: 0.85,
        listening: false,
      },
    },

    // drawLayer: child freehand strokes
    ink: {
      stroke: "#1a1814",
      strokeWidth: 4,
      lineCap: "round",
      lineJoin: "round",
      globalCompositeOperation: "source-over",
      listening: false,
    },

    // Eraser: destination-out cuts through existing ink. White stroke is a
    // no-op under destination-out (alpha only); 18 px for comfortable erasing.
    eraser: {
      stroke: "rgba(255,255,255,1)",
      strokeWidth: 18,
      lineCap: "round",
      lineJoin: "round",
      globalCompositeOperation: "destination-out",
      listening: false,
    },

    // glyphLayer: stroked rect at the REAL pipeline bbox (not grid-snapped);
    // dashed when disagreement=true (soft guide cue, never moved).
    glyphRect: {
      strokeWidth: 2,
      fill: "transparent",
      cornerRadius: 3,
      listening: false,
      opacity: 0.85,
    },
    glyphLabel: {
      fontFamily: '"SF Mono", ui-monospace, Menlo, monospace',
      fontSize: 11,
      listening: false,
      opacity: 0.9,
    },

    // popupLayer: status markers and callout bubbles
    statusRect: {
      ok: { fill: "rgba(22,163,74,0.14)", stroke: "#16a34a", strokeWidth: 2, cornerRadius: 8, opacity: 1 },
      missing: { fill: "rgba(217,119,6,0.14)", stroke: "#d97706", strokeWidth: 2, cornerRadius: 8, opacity: 1, dash: [5, 4] },
      error: { fill: "rgba(220,38,38,0.14)", stroke: "#dc2626", strokeWidth: 2, cornerRadius: 8, opacity: 1 },
      hint: { fill: "rgba(124,58,237,0.12)", stroke: "#7c3aed", strokeWidth: 2, cornerRadius: 8, opacity: 1, dash: [6, 3] },
    },
  };

  // Icon set: stroke-based inline SVG, 18px, currentColor. Used in the feedback
  // panel HTML. One function per semantic role so callers stay declarative.
  // All icons are aria-hidden="true"; the surrounding text carries the meaning.
  const ICONS = {
    // success / check (green context)
    correct: '<svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true"><path d="M3 9.5l4 4 8-8" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    // mistake / error (red context)
    mistake: '<svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true"><circle cx="9" cy="9" r="7" stroke="currentColor" stroke-width="1.8"/><path d="M9 5.5v4M9 12.5v.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
    // missing / incomplete (amber context)
    incomplete: '<svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true"><circle cx="9" cy="9" r="7" stroke="currentColor" stroke-width="1.8"/><path d="M9 5.5v4M9 12.5v.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
    // missing step / hint (purple context)
    missing_step: '<svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true"><path d="M4 9h10M10 5l4 4-4 4" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    // error detail (used in message lines)
    error_detail: '<svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true"><circle cx="7" cy="7" r="5.5" stroke="currentColor" stroke-width="1.6"/><path d="M7 4.5v2.5M7 9.5v.3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>',
    // hint detail (used in hint lines)
    hint_detail: '<svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true"><path d="M3 7h8M8 4l3 3-3 3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    // error/alert for banner (engine unavailable)
    alert: '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M8 2L2 13h12L8 2Z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/><path d="M8 7v3M8 12v.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>',
    // retry affordance
    retry: '<svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true"><path d="M2 7a5 5 0 1 1 1.5 3.6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M2 11V7h4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  };

  // _iconEl(key): creates a span holding the SVG string. Callers append it to
  // DOM elements; aria-hidden is embedded in each SVG string above.
  function _iconEl(key) {
    const span = document.createElement("span");
    span.className = "dk-icon";
    span.innerHTML = ICONS[key] || "";
    return span;
  }

  // Focus highlight: stroke-only, status-appropriate. One gentle pulse via CSS;
  // the Konva rect is never filled.
  const FOCUS_STYLE = {
    mistake:      { stroke: "rgba(220,38,38,0.55)",   strokeWidth: 2, cornerRadius: 8, fill: "transparent", dash: undefined },
    missing_step: { stroke: "rgba(124,58,237,0.45)",  strokeWidth: 2, cornerRadius: 8, fill: "transparent", dash: [5, 4] },
    incomplete:   { stroke: "rgba(217,119,6,0.45)",   strokeWidth: 2, cornerRadius: 8, fill: "transparent", dash: [5, 4] },
    ok:           { stroke: "rgba(22,163,74,0.35)",   strokeWidth: 2, cornerRadius: 8, fill: "transparent", dash: undefined },
  };

  // ---------------------------------------------------------------- 2. API
  // GET  /api/exercises         -> { exercises: [...] }
  // POST /api/exercise/select   -> { calibration, scaffold, sessionId, refId, marksTotal }
  // POST /api/evaluate          -> { recognized, popups, timings, opencv_used,
  //                                  progress, marks, feedback, warning? }
  //   recognized: { equation_kind, equation_label, row_count, summary,
  //                 tokens: [{index, label, glyph, role, bbox, row, col,
  //                           confidence, color, low_conf, disagreement}],
  //                 disagreements: [{token_id, char, recognizer_row,
  //                                  recognizer_col, guide_row, guide_col}] }
  //   popups: [{pixelRect:[l,t,r,b], status, message, kind, symbolIds, anchor}]
  //   timings: { total_ms, yolo_ms, stage2_ms, assemble_ms }
  // GET  /api/score             -> { finished, marksTotal, marksEarned, penalties }
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
    async exercises() {
      const res = await fetch("/api/exercises");
      return this._json(res);
    },
    async select(exerciseId) {
      const res = await fetch("/api/exercise/select", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ exercise_id: exerciseId }),
      });
      return this._json(res);
    },
    async evaluate(imagePng) {
      const res = await fetch("/api/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_png: imagePng }),
      });
      return this._json(res);
    },
    async score() {
      const res = await fetch("/api/score");
      return this._json(res);
    },
  };

  // ---------------------------------------------------------------- 3. state

  const state = {
    exercises: [],     // bank list from /api/exercises
    selectedId: null,  // active exercise id
    calibration: null, // GridCalibration-shaped object from select (normalised)
    busy: false,       // a request is in flight; gate the buttons
    tool: "pen",       // "pen" | "eraser"
  };

  // ---------------------------------------------------------------- DOM refs

  const el = {
    picker: document.getElementById("exercise-picker"),
    banner: document.getElementById("banner"),
    prompt: document.getElementById("prompt"),
    stageHost: document.getElementById("stage"),
    stageFrame: document.querySelector(".dk-stage-host"),
    popupHost: document.getElementById("popup-layer"),
    btnCheck: document.getElementById("btn-check"),
    btnReset: document.getElementById("btn-reset"),
    btnPen: document.getElementById("btn-pen"),
    btnEraser: document.getElementById("btn-eraser"),
    recognizedEq: document.getElementById("recognized-equation"),
    recognizedKind: document.getElementById("recognized-kind"),
    recognizedCount: document.getElementById("recognized-count"),
    recognizedTokens: document.getElementById("recognized-tokens"),
    timeTotal: document.getElementById("time-total"),
    opencvBadge: document.getElementById("opencv-badge"),
    timingsDetail: document.getElementById("timings-detail"),
    timingsList: document.getElementById("timings-list"),
    engineExchange: document.getElementById("engine-exchange"),
    engineSent: document.getElementById("engine-sent"),
    engineReceived: document.getElementById("engine-received"),
    marks: document.getElementById("marks"),
    progressFill: document.getElementById("progress-fill"),
    progressText: document.getElementById("progress-text"),
    reconcile: document.getElementById("reconcile-warning"),
    feedbackCard: document.getElementById("feedback-card"),
    feedbackList: document.getElementById("feedback-list"),
    feedbackEmpty: document.getElementById("feedback-empty"),
  };

  function banner(msg, kind) {
    if (!msg) { el.banner.hidden = true; el.banner.textContent = ""; return; }
    el.banner.hidden = false;
    el.banner.textContent = msg;
    el.banner.dataset.kind = kind || "notice";
  }

  // ---------------------------------------------------------------- 4. Konva
  // Five layers (bottom to top): gridLayer (writing guide, visual only),
  // scaffoldLayer (pre-printed), drawLayer (freehand, flattened for evaluate),
  // glyphLayer (recognition overlay at real bboxes), popupLayer (status markers;
  // HTML callouts live in #popup-layer above).
  const stage = new Konva.Stage({ container: el.stageHost, width: CANVAS, height: CANVAS });
  const gridLayer = new Konva.Layer({ listening: false });
  const scaffoldLayer = new Konva.Layer({ listening: false });
  const drawLayer = new Konva.Layer({ listening: true });
  const glyphLayer = new Konva.Layer({ listening: true });
  const popupLayer = new Konva.Layer({ listening: false });
  stage.add(gridLayer);
  stage.add(scaffoldLayer);
  stage.add(drawLayer);
  stage.add(glyphLayer);
  stage.add(popupLayer);

  // -- freehand drawing (Pointer Events: stylus, pen, touch, mouse) ------
  let currentLine = null;
  // Per-symbol fade: Konva nodes from renderScaffold that carry an engineCellRect.
  // Populated by renderScaffold; cleared and restored to full opacity by clearDrawing.
  // Canvas bubble proximity-fade registry: {el, left, top, w, h} for each live bubble.
  // Cleared in clearFeedback(); populated by _placeCanvasPopup().
  let _activeBubbles = [];
  let scaffoldNodes = [];

  function pointerPos() {
    const p = stage.getPointerPosition();
    return p ? [p.x, p.y] : null;
  }

  // Fade any scaffold glyph the live pen point is inside. Single-point test so
  // the fade fires the instant the pen enters the glyph's cell, not on stroke
  // end. The _fading flag prevents restarting the tween on every move event.
  function _fadeScaffoldAtPoint(x, y) {
    if (scaffoldNodes.length === 0) return;
    const PAD = 4;
    for (const node of scaffoldNodes) {
      const r = node.getAttr("engineCellRect");
      if (!r) continue;
      const [left, top, right, bottom] = r;
      const inside = x >= left - PAD && x <= right + PAD && y >= top - PAD && y <= bottom + PAD;
      if (inside && !node.getAttr("_fading") && node.opacity() > 0.12) {
        node.setAttr("_fading", true);
        new Konva.Tween({
          node,
          duration: 0.15,
          opacity: 0.12,
          easing: Konva.Easings.EaseInOut,
        }).play();
      }
    }
  }

  stage.on("pointerdown", (e) => {
    if (e.target !== stage && e.target.getLayer() !== drawLayer) {
      // allow starting a stroke over the stage or any non-interactive layer
    }
    if (!state.selectedId || state.busy) return;
    const pos = pointerPos();
    if (!pos) return;

    const isEraser = state.tool === "eraser";
    const styleProps = isEraser ? STYLE.eraser : STYLE.ink;
    const line = new Konva.Line({
      points: pos,
      stroke: styleProps.stroke,
      strokeWidth: styleProps.strokeWidth,
      lineCap: styleProps.lineCap,
      lineJoin: styleProps.lineJoin,
      globalCompositeOperation: styleProps.globalCompositeOperation,
      tension: 0,
    });
    currentLine = line;
    drawLayer.add(currentLine);

    if (!isEraser) {
      clearFeedback();
      _fadeScaffoldAtPoint(pos[0], pos[1]);
    }
  });

  stage.on("pointermove", () => {
    if (!currentLine) return;
    const pos = pointerPos();
    if (!pos) return;
    currentLine.points(currentLine.points().concat(pos));
    drawLayer.batchDraw();
    // Live per-symbol fade: ink strokes only (eraser never fades scaffold).
    if (currentLine.globalCompositeOperation() !== "destination-out") {
      _fadeScaffoldAtPoint(pos[0], pos[1]);
    }
  });

  function endStroke() {
    if (!currentLine) return;
    // Scaffold fade now happens live during the stroke (pointerdown/move via
    // _fadeScaffoldAtPoint); nothing to do here but finalise button state.
    currentLine = null;
    syncButtons();
  }
  stage.on("pointerup", endStroke);
  stage.on("pointercancel", endStroke);
  el.stageHost.addEventListener("pointerleave", endStroke);

  // Proximity-fade: when the pointer hovers over a canvas bubble, dim it so the
  // child can see the content beneath. Implemented via JS geometry (not CSS hover)
  // because bubbles have pointer-events:none — the browser never fires mouse events
  // on them, so we must manually hit-test the raw pointer position.
  // A separate DOM listener (not Konva stage event) is used so this fires even when
  // no stroke is active (Konva's stage pointermove bails early when !currentLine).
  el.stageHost.addEventListener("pointermove", (e) => {
    if (_activeBubbles.length === 0) return;
    const rect = el.stageHost.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    // Scale from CSS pixels to Konva 512 space if host is scaled (currently 1:1).
    const scaleX = CANVAS / rect.width;
    const scaleY = CANVAS / rect.height;
    const kx = px * scaleX;
    const ky = py * scaleY;
    for (const b of _activeBubbles) {
      const inside = kx >= b.left && kx <= b.left + b.w && ky >= b.top && ky <= b.top + b.h;
      b.el.classList.toggle("dk-popup--dimmed", inside);
    }
  });

  function hasDrawing() { return drawLayer.getChildren().length > 0; }

  // Switch between "pen" and "eraser" tools without clearing the drawing.
  // Updates state, button aria-pressed states, and the stage cursor class.
  function switchTool(tool) {
    state.tool = tool;
    const isEraser = tool === "eraser";
    el.btnPen.setAttribute("aria-pressed", String(!isEraser));
    el.btnEraser.setAttribute("aria-pressed", String(isEraser));
    el.btnPen.classList.toggle("dk-tool-btn--active", !isEraser);
    el.btnEraser.classList.toggle("dk-tool-btn--active", isEraser);
    if (el.stageFrame) {
      el.stageFrame.classList.toggle("dk-stage-host--eraser", isEraser);
    }
  }

  function clearDrawing() {
    drawLayer.destroyChildren();
    drawLayer.draw();
    // Restore all per-symbol scaffold nodes to full opacity.
    _restoreScaffoldOpacity();
    clearFeedback();
    syncButtons();
  }

  function _restoreScaffoldOpacity() {
    for (const node of scaffoldNodes) {
      const initial = node.getAttr("engineInitialOpacity");
      node.opacity(initial !== undefined ? initial : 1);
      node.setAttr("_fading", false);
    }
    if (scaffoldNodes.length > 0) scaffoldLayer.batchDraw();
  }

  // Flatten draw layer only to base64 PNG (grid/scaffold/glyphs/popups excluded).
  function drawingDataUrl() {
    return drawLayer.toDataURL({ pixelRatio: 1, width: CANVAS, height: CANVAS, x: 0, y: 0 });
  }

  // -- scaffold + guide grid --------------------------------------------------

  function normaliseCalibration(raw) {
    if (!raw || typeof raw !== "object") return null;
    const pick = (a, b, d) => {
      if (raw[a] !== undefined && raw[a] !== null) return raw[a];
      if (raw[b] !== undefined && raw[b] !== null) return raw[b];
      return d;
    };
    const gw = Number(pick("grid_width", "gridWidth", 0));
    const gh = Number(pick("grid_height", "gridHeight", 0));
    if (!gw || !gh) return null;
    return {
      gridWidth: gw,
      gridHeight: gh,
      cellW: Number(pick("cell_w_px", "cellWPx", CANVAS / gw)),
      cellH: Number(pick("cell_h_px", "cellHPx", CANVAS / gh)),
      originX: Number(pick("origin_x_px", "originXPx", 0)),
      originY: Number(pick("origin_y_px", "originYPx", 0)),
    };
  }

  // Faint writing guide. Phase-aligned to calibration cells so exercise ink sits
  // on the lines; tiles the full 512 frame regardless of exercise position.
  // Never drives recognition or feedback placement.
  function renderGuideGrid(cal) {
    gridLayer.destroyChildren();
    if (!cal) { gridLayer.draw(); return; }
    const cw = cal.cellW, ch = cal.cellH;
    if (!(cw > 0) || !(ch > 0)) { gridLayer.draw(); return; }
    const phaseX = ((cal.originX % cw) + cw) % cw;
    const phaseY = ((cal.originY % ch) + ch) % ch;
    for (let x = phaseX; x <= CANVAS + 0.5; x += cw) {
      gridLayer.add(new Konva.Line(Object.assign({ points: [x, 0, x, CANVAS] }, STYLE.grid)));
    }
    for (let y = phaseY; y <= CANVAS + 0.5; y += ch) {
      gridLayer.add(new Konva.Line(Object.assign({ points: [0, y, CANVAS, y] }, STYLE.grid)));
    }
    gridLayer.draw();
  }

  function renderScaffold(tokens, cal) {
    scaffoldLayer.destroyChildren();
    // Reset per-symbol fade tracker for the new exercise.
    scaffoldNodes = [];
    const list = Array.isArray(tokens) ? tokens : [];
    for (const tok of list) {
      const rect = scaffoldRect(tok, cal);
      if (!rect) continue;
      const [left, top, right, bottom] = rect;
      const ch = String(tok.char !== undefined ? tok.char : (tok.c !== undefined ? tok.c : (tok.glyph !== undefined ? tok.glyph : "")));

      // Placeholder cells (FILL): dashed empty outline showing where to write.
      // Not fadeable -- the child should draw INTO these cells.
      if (tok.kind === "placeholder" || (Array.isArray(tok.tags) && tok.tags.includes("FILL"))) {
        if (!SHOW_GUIDE_BOXES) continue; // hidden (kept behind flag, not deleted)
        const w = right - left;
        const h = bottom - top;
        // Borrow / carry cells are OPTIONAL working: draw them smaller and in a
        // softer violet so they read as "show your borrow here", distinct from
        // the answer line the child must fill.
        const isBorrow = tok.role === "borrow" || tok.role === "carry";
        const inset = isBorrow ? Math.min(w, h) * 0.22 : 3;
        scaffoldLayer.add(new Konva.Rect({
          x: left + inset,
          y: top + inset,
          width: w - inset * 2,
          height: h - inset * 2,
          stroke: isBorrow ? "rgba(124,58,237,0.30)" : "rgba(100,120,200,0.35)",
          strokeWidth: 1.5,
          dash: isBorrow ? [3, 3] : [5, 4],
          fill: isBorrow ? "rgba(124,58,237,0.03)" : "rgba(100,120,200,0.04)",
          cornerRadius: 4,
          listening: false,
        }));
        continue;
      }

      const isDivBracket = (tok.kind === "div_bracket" || tok.label === "div_bracket");
      if (isDivBracket) {
        // Bus-stop L: vertical at x=left, horizontal at y=top over the dividend.
        // Register two line nodes covering the combined bracket rect for fade detection.
        const vLine = new Konva.Line(Object.assign({ points: [left, top, left, bottom] }, STYLE.scaffold.bar));
        const hLine = new Konva.Line(Object.assign({ points: [left, top, right, top] }, STYLE.scaffold.bar));
        const bracketOpacity = vLine.opacity();
        vLine.setAttr("engineCellRect", [left, top, right, bottom]);
        vLine.setAttr("engineInitialOpacity", bracketOpacity);
        vLine.setAttr("engineCellKind", "bracket");
        hLine.setAttr("engineCellRect", [left, top, right, bottom]);
        hLine.setAttr("engineInitialOpacity", bracketOpacity);
        hLine.setAttr("engineCellKind", "bracket");
        scaffoldLayer.add(vLine);
        scaffoldLayer.add(hLine);
        scaffoldNodes.push(vLine, hLine);
        continue;
      }
      const isBar = (tok.kind === "bar" || tok.label === "result_bar" || tok.type === "bar" || ch === "─" || ch === "_");
      if (isBar) {
        const y = (top + bottom) / 2;
        const barLine = new Konva.Line(Object.assign({ points: [left, y, right, y] }, STYLE.scaffold.bar));
        barLine.setAttr("engineCellRect", [left, top, right, bottom]);
        barLine.setAttr("engineInitialOpacity", barLine.opacity());
        barLine.setAttr("engineCellKind", "bar");
        scaffoldLayer.add(barLine);
        scaffoldNodes.push(barLine);
        continue;
      }
      if (!ch) continue;
      const text = new Konva.Text(Object.assign({}, STYLE.scaffold.digit, {
        text: ch,
        x: left,
        y: top,
        width: right - left,
        height: bottom - top,
        align: "center",
        verticalAlign: "middle",
      }));
      text.setAttr("engineCellRect", [left, top, right, bottom]);
      text.setAttr("engineInitialOpacity", text.opacity());
      text.setAttr("engineCellKind", "digit");
      scaffoldLayer.add(text);
      scaffoldNodes.push(text);
    }
    scaffoldLayer.draw();
  }

  function scaffoldRect(tok, cal) {
    const pr = tok.pixelRect || tok.pixel_rect || tok.bbox;
    if (Array.isArray(pr) && pr.length === 4) {
      return [Number(pr[0]), Number(pr[1]), Number(pr[2]), Number(pr[3])];
    }
    if (!cal || tok.x === undefined || tok.y === undefined) return null;
    return cellToPixels(Number(tok.x), Number(tok.y), cal);
  }

  // engine grid (x,y) -> 512-space rect. y is bottom-origin (larger y = higher row),
  // so row = (gridHeight - 1 - y). Mirrors grid_mapping.cell_to_pixels.
  function cellToPixels(x, y, cal) {
    const col = x;
    const row = cal.gridHeight - 1 - y;
    const left = cal.originX + col * cal.cellW;
    const top = cal.originY + row * cal.cellH;
    return [left, top, left + cal.cellW, top + cal.cellH];
  }

  // -- recognition overlay + minimal feedback ---------------------------------

  // tokenId -> Konva.Rect for focus highlights; cleared on reset/exercise swap.
  const glyphBoxNodes = new Map();

  // Build tokenId -> token map from the recognized block so renderMinimalFeedback
  // can resolve focus[].symbolId to a real bbox.
  function buildIdToToken(recognized) {
    glyphLayer.destroyChildren();
    glyphBoxNodes.clear();
    glyphLayer.draw();
    const tokens = Array.isArray(recognized && recognized.tokens) ? recognized.tokens : [];
    const idToToken = new Map(); // tokenId ("T<N-1>") -> token object
    for (const tok of tokens) {
      const idx = tok.index !== undefined ? tok.index : "";
      if (idx !== "") {
        idToToken.set("T" + (Number(idx) - 1), tok);
      }
    }
    return idToToken;
  }

  // Derive a minimal feedback object from the raw popups array (fallback when
  // the backend does not include a feedback field). outcome/headline/focus/checkAnchor.
  function deriveFeedback(popups, finished) {
    const list = Array.isArray(popups) ? popups : [];

    if (finished) {
      // Correct: find centroid of ok rects for the check-mark anchor.
      let cx = 0, cy = 0, n = 0;
      for (const p of list) {
        const r = p.pixelRect || p.pixel_rect;
        if (Array.isArray(r) && r.length === 4) {
          cx += (Number(r[0]) + Number(r[2])) / 2;
          cy += (Number(r[1]) + Number(r[3])) / 2;
          n++;
        }
      }
      return {
        outcome: "correct",
        headline: "Correct",
        messages: [],
        hint: null,
        focus: [],
        checkAnchor: n > 0 ? [cx / n, cy / n] : [CANVAS * 0.82, CANVAS * 0.72],
      };
    }

    // Highest-priority actionable popup: error > hint > missing. "ok" cells are
    // given/correct context, never the verdict while the exercise is unfinished.
    const rank = { error: 3, hint: 2, missing: 1 };
    let primary = null;
    for (const p of list) {
      const s = String(p.status || "").toLowerCase();
      if (!(s in rank)) continue;
      if (!primary || rank[s] > (rank[String(primary.status || "").toLowerCase()] || 0)) {
        primary = p;
      }
    }

    if (!primary) {
      return { outcome: "incomplete", headline: "Not finished", messages: [], hint: null, focus: [], checkAnchor: null };
    }

    const ps = String(primary.status || "").toLowerCase();
    let outcome, headline;

    // Verdict labels only (no guidance nudges); detail comes from engine messages.
    if (ps === "error") {
      outcome = "mistake";
      headline = "Not quite";
    } else if (ps === "hint") {
      outcome = "missing_step";
      headline = "Not finished";
    } else {
      outcome = "incomplete";
      headline = "Not finished";
    }

    const rect = primary.pixelRect || primary.pixel_rect;
    const focus = [];
    if (Array.isArray(rect) && rect.length === 4) {
      focus.push({
        pixelRect: rect.map(Number),
        symbolId: null,
        kind: ps === "error" ? "error" : "missing",
      });
    }

    // Include popup message if available (comes from backend _render_message).
    const messages = primary.message
      ? [{ text: primary.message, status: ps }]
      : [];

    return { outcome, headline, messages, hint: null, focus, checkAnchor: null };
  }

  // Post-Check feedback: focus highlight(s) on canvas + one headline line in panel.
  // "correct" shows a subtle check mark near the answer region; all other outcomes
  // draw one soft rounded outline around the primary focus cell.
  // Draw one on-canvas highlight rect (engine-positioned).
  // kind: "error" -> red solid; "missing" -> amber dashed; "hint" -> purple dashed.
  function _drawFocusRect(rect, kind) {
    if (!Array.isArray(rect) || rect.length !== 4) return;
    const [left, top, right, bottom] = rect;
    const w = right - left;
    const h = bottom - top;
    if (w <= 0 || h <= 0) return;
    const styles = {
      error:   { stroke: "rgba(220,38,38,0.80)",  strokeWidth: 2.5, cornerRadius: 8, fill: "rgba(220,38,38,0.08)",   dash: undefined },
      missing: { stroke: "rgba(217,119,6,0.80)",  strokeWidth: 2,   cornerRadius: 8, fill: "rgba(217,119,6,0.06)",   dash: [5, 4] },
      hint:    { stroke: "rgba(124,58,237,0.70)", strokeWidth: 2,   cornerRadius: 8, fill: "rgba(124,58,237,0.05)",  dash: [6, 3] },
    };
    const style = styles[kind] || styles.error;
    const kRect = new Konva.Rect(Object.assign({}, style, {
      x: left, y: top, width: w, height: h, listening: false,
    }));
    popupLayer.add(kRect);
    kRect.opacity(0);
    new Konva.Tween({ node: kRect, duration: 0.28, opacity: 1, easing: Konva.Easings.EaseOut }).play();
  }

  // Place one feedback bubble on the canvas overlay, anchored to a 512-space
  // pixelRect. Uses a best-of-N candidate search: generates candidates in 8
  // directions around the anchor, scores each by overlap cost with all content
  // obstacles + already-placed bubbles + off-canvas penalty, and picks the
  // minimum-cost candidate. Tail (--dk-tail-x / --below) always points at the
  // anchor centre. pointer-events:none so the child can always draw underneath.
  //
  // `obstacles` is an array of {left, top, w, h} rects covering ALL content
  // (scaffold digits, bars, brackets, child's drawn glyph bboxes).
  // `placed` is an array of {left, top, w, h, el} for already-placed bubbles;
  // this function appends to it on success and registers the bubble in _activeBubbles.
  function _placeCanvasPopup(text, status, pixelRect, obstacles, placed) {
    if (!text || !Array.isArray(pixelRect) || pixelRect.length !== 4) return;
    const [l, t, r, b] = pixelRect;
    const cx = (l + r) / 2;
    const cy = (t + b) / 2;

    const pop = document.createElement("div");
    pop.className = "dk-popup dk-popup--" + status + " dk-popup--canvas";
    const ic = _iconEl(status === "hint" ? "hint_detail" : "error_detail");
    ic.className = "dk-icon dk-popup-icon";
    const tx = document.createElement("span");
    tx.className = "dk-popup-text";
    tx.textContent = text;
    pop.appendChild(ic);
    pop.appendChild(tx);

    // Measure off-screen before positioning.
    pop.style.left = "0px";
    pop.style.top = "0px";
    pop.style.visibility = "hidden";
    el.popupHost.appendChild(pop);
    const w = pop.offsetWidth || 150;
    const h = pop.offsetHeight || 60;
    const GAP = 10, M = 4;

    // ----------------------------------------------------------------
    // Candidate generation: 8 directions, each anchored so the tail
    // points at (cx, cy). The horizontal centre / vertical centre of the
    // bubble aligns with the anchor in the perpendicular axis.
    // "below=true" means the tail points UP (bubble is below the anchor).
    const candidates = [
      // above (preferred — tail points down at the anchor)
      { left: cx - w / 2, top: t - h - GAP, side: "above" },
      // below — tail points up
      { left: cx - w / 2, top: b + GAP,     side: "below" },
      // bubble to the RIGHT of the anchor — tail on its LEFT edge points left
      { left: r + GAP,    top: cy - h / 2,  side: "rightOfAnchor" },
      // bubble to the LEFT of the anchor — tail on its RIGHT edge points right
      { left: l - w - GAP, top: cy - h / 2, side: "leftOfAnchor" },
      // diagonal corners keep a vertical tail (tail-x still points at the anchor)
      { left: r,          top: t - h - GAP, side: "above" },
      { left: l - w,      top: t - h - GAP, side: "above" },
      { left: r,          top: b + GAP,     side: "below" },
      { left: l - w,      top: b + GAP,     side: "below" },
    ];

    // ----------------------------------------------------------------
    // Scoring helpers.

    // Intersection area of two axis-aligned rects (0 if no overlap).
    function _overlapArea(al, at, aw, ah, bl, bt, bw, bh) {
      const ox = Math.max(0, Math.min(al + aw, bl + bw) - Math.max(al, bl));
      const oy = Math.max(0, Math.min(at + ah, bt + bh) - Math.max(at, bt));
      return ox * oy;
    }

    // How many pixels of the bubble rect are outside [0, CANVAS].
    function _offCanvasArea(cl, ct) {
      const right = cl + w, bottom = ct + h;
      const ox = Math.max(0, -cl) + Math.max(0, right - CANVAS);
      const oy = Math.max(0, -ct) + Math.max(0, bottom - CANVAS);
      // Penalise off-canvas heavily — any clipping is very undesirable.
      return (ox * h + oy * w + ox * oy) * 4;
    }

    // Euclidean distance from bubble centre to anchor centre (prefer proximity).
    function _distancePenalty(cl, ct) {
      const bcx = cl + w / 2, bcy = ct + h / 2;
      return Math.sqrt((bcx - cx) ** 2 + (bcy - cy) ** 2) * 0.05;
    }

    function _scoreCandidate(cl, ct) {
      let cost = 0;
      // Obstacle overlap (given digits, bars, brackets, child's drawn ink).
      for (const o of obstacles) {
        cost += _overlapArea(cl, ct, w, h, o.left, o.top, o.w, o.h);
      }
      // The ANCHOR cell (the number this feedback is about) must stay visible.
      // Covering it is far worse than covering any other content, so weight it
      // heavily: the bubble is pushed off the number (above/below/clear side)
      // even when that means overlapping a bar or an empty box instead.
      cost += _overlapArea(cl, ct, w, h, l, t, r - l, b - t) * 50;
      // Already-placed bubble overlap (weighted extra to avoid stacking).
      for (const p of placed) {
        cost += _overlapArea(cl, ct, w, h, p.left, p.top, p.w, p.h) * 2;
      }
      // Off-canvas penalty.
      cost += _offCanvasArea(cl, ct);
      // Proximity preference (slight bias toward close placement).
      cost += _distancePenalty(cl, ct);
      return cost;
    }

    // ----------------------------------------------------------------
    // Find best candidate (min cost).
    let bestLeft = candidates[0].left;
    let bestTop = candidates[0].top;
    let bestSide = candidates[0].side;
    let bestCost = Infinity;

    for (const c of candidates) {
      // Clamp horizontal into canvas before scoring (preserves tail alignment).
      const cl = Math.max(M, Math.min(c.left, CANVAS - w - M));
      // Clamp BOTH axes before scoring so we score the ACHIEVABLE position:
      // the picked candidate (and the ANCHOR penalty) then reflect where the
      // bubble actually lands after the mandatory on-canvas clamp.
      const ct = Math.max(M, Math.min(c.top, CANVAS - h - M));
      const cost = _scoreCandidate(cl, ct);
      if (cost < bestCost) {
        bestCost = cost;
        bestLeft = cl;
        bestTop = ct;
        bestSide = c.side;
      }
    }

    // Final vertical clamp: never clip off the canvas.
    bestTop = Math.max(M, Math.min(bestTop, CANVAS - h - M));

    pop.style.left = bestLeft + "px";
    pop.style.top = bestTop + "px";

    // ----------------------------------------------------------------
    // Tail: derived from the FINAL geometry (not the winning candidate label),
    // so it always points at the anchor from whichever bubble edge faces it.
    // Pick the edge the anchor sits FURTHEST beyond; ties prefer a vertical tail.
    const bRight = bestLeft + w, bBottom = bestTop + h;
    const dTop = bestTop - cy;     // >0 when the anchor is above the bubble
    const dBottom = cy - bBottom;  // >0 when below
    const dLeft = bestLeft - cx;   // >0 when the anchor is left of the bubble
    const dRight = cx - bRight;    // >0 when right
    const maxOut = Math.max(dTop, dBottom, dLeft, dRight);

    pop.classList.remove("dk-popup--below", "dk-popup--tail-left", "dk-popup--tail-right");
    if (maxOut > 0 && maxOut === dLeft) {
      // anchor to the LEFT -> tail on the bubble's LEFT edge, pointing left
      const tailY = Math.max(14, Math.min(cy - bestTop, h - 14));
      pop.style.setProperty("--dk-tail-y", tailY + "px");
      pop.classList.add("dk-popup--tail-left");
    } else if (maxOut > 0 && maxOut === dRight) {
      // anchor to the RIGHT -> tail on the bubble's RIGHT edge, pointing right
      const tailY = Math.max(14, Math.min(cy - bestTop, h - 14));
      pop.style.setProperty("--dk-tail-y", tailY + "px");
      pop.classList.add("dk-popup--tail-right");
    } else {
      // vertical tail: anchor above -> tail up (--below); else tail down.
      const tailX = Math.max(14, Math.min(cx - bestLeft, w - 14));
      pop.style.setProperty("--dk-tail-x", tailX + "px");
      pop.classList.toggle("dk-popup--below", cy < bestTop);
    }
    pop.style.visibility = "visible";

    // Register for collision avoidance by subsequent bubbles and proximity fade.
    const entry = { left: bestLeft, top: bestTop, w, h, el: pop };
    placed.push(entry);
    _activeBubbles.push(entry);
  }

  // Render the engine's feedback. The result CARD holds only the verdict (headline). The
  // detailed feedback messages and the hint float ON the paper as bubbles over
  // the glyph each one refers to. ERROR/MISSING cells get a highlight ring too.
  // Messages with no resolvable position fall back into the panel so nothing is
  // ever lost. "correct" shows a check mark.
  // recognized: the resp.recognized object from the evaluate response.
  // Used to seed obstacle rects with the child's drawn glyph bboxes so bubbles
  // avoid landing on top of written answers, borrows, and carries.
  function renderMinimalFeedback(feedback, recognized) {
    _hideEmptyState();
    popupLayer.destroyChildren();
    el.popupHost.innerHTML = "";
    _activeBubbles = [];

    const outcome = feedback.outcome || "incomplete";

    if (outcome === "correct") {
      // SVG check mark rendered as an HTML overlay so it uses the icon set.
      const anchor = feedback.checkAnchor || [CANVAS * 0.82, CANVAS * 0.72];
      const checkDiv = document.createElement("div");
      checkDiv.className = "dk-canvas-check";
      checkDiv.style.left = anchor[0] + "px";
      checkDiv.style.top = anchor[1] + "px";
      checkDiv.innerHTML = ICONS.correct;
      checkDiv.setAttribute("aria-hidden", "true");
      el.popupHost.appendChild(checkDiv);
    } else {
      // Highlight every engine-flagged cell on the canvas.
      const focusList = Array.isArray(feedback.focus) ? feedback.focus : [];
      for (const f of focusList) {
        _drawFocusRect(f.pixelRect, f.kind || "error");
      }
      if (feedback.hint && feedback.hint.pixelRect) {
        _drawFocusRect(feedback.hint.pixelRect, "hint");
      }
    }
    popupLayer.draw();

    // --- Canvas feedback bubbles: feedback messages ALWAYS float on the paper ---
    // When engine gives a message no position, anchor it to the first flagged cell
    // (else the canvas centre) so it still floats next to the work, never the
    // panel. The panel holds only the verdict headline.

    // Build the full obstacle list for bubble placement:
    //   (a) scaffold given-digit rects (pre-printed operands / operator)
    //   (b) scaffold bar and bracket rects (structural lines)
    //   (c) the child's drawn glyph bboxes from resp.recognized.tokens[].bbox
    // Using ALL scaffold kinds here (not just "digit") because bars span the
    // full width and brackets span the dividend area — bubbles above or below
    // those rows should not land on them either.
    // scaffoldNodes carries engineCellKind = "digit"|"bar"|"bracket".
    const obstacles = [];
    for (const node of scaffoldNodes) {
      const r = node.getAttr("engineCellRect");
      if (!Array.isArray(r) || r.length !== 4) continue;
      const [sl, st, sr, sb] = r;
      const kind = node.getAttr("engineCellKind");
      // Bar/bracket are thin lines — use a modest vertical expansion so
      // bubbles don't float immediately on top of them, but don't exaggerate
      // to avoid pushing bubbles too far away.
      const expand = kind === "digit" ? 0 : 6;
      obstacles.push({ left: sl - expand, top: st - expand, w: (sr - sl) + expand * 2, h: (sb - st) + expand * 2 });
    }
    // Child's drawn glyph bboxes (512-space [x0,y0,x1,y1]).
    // Skip given=true tokens: they are pre-printed scaffold, not child ink, so
    // they must not perturb bubble placement for the child's feedback messages.
    const recTokens = Array.isArray(recognized && recognized.tokens) ? recognized.tokens : [];
    for (const tok of recTokens) {
      if (tok.given === true) continue;
      const bb = tok.bbox;
      if (!Array.isArray(bb) || bb.length !== 4) continue;
      const [bx0, by0, bx1, by1] = bb;
      if (bx1 <= bx0 || by1 <= by0) continue;
      obstacles.push({ left: bx0, top: by0, w: bx1 - bx0, h: by1 - by0 });
    }

    // `placed` accumulates placed-bubble rects for inter-bubble collision avoidance.
    // (It also drives _activeBubbles via _placeCanvasPopup.)
    const placed = [];
    const focusList = Array.isArray(feedback.focus) ? feedback.focus : [];
    const fallbackRect =
      (focusList[0] && focusList[0].pixelRect) ||
      (feedback.hint && feedback.hint.pixelRect) ||
      [CANVAS * 0.5 - 70, CANVAS * 0.42 - 22, CANVAS * 0.5 + 70, CANVAS * 0.42 + 22];

    const messages = Array.isArray(feedback.messages) ? feedback.messages : [];
    for (const m of messages) {
      if (!m.text) continue;
      const rect = Array.isArray(m.pixelRect) ? m.pixelRect : fallbackRect;
      _placeCanvasPopup(m.text, "error", rect, obstacles, placed);
    }
    if (feedback.hint && feedback.hint.text) {
      const rect = Array.isArray(feedback.hint.pixelRect) ? feedback.hint.pixelRect : fallbackRect;
      _placeCanvasPopup(feedback.hint.text, "hint", rect, obstacles, placed);
    }

    // --- Panel: verdict headline only ---
    el.feedbackList.innerHTML = "";
    const headlineIconKey = { correct: "correct", mistake: "mistake", incomplete: "incomplete", missing_step: "missing_step" };
    const li = document.createElement("li");
    li.className = "dk-feedback-headline dk-feedback-headline--" + outcome;
    const icon = _iconEl(headlineIconKey[outcome] || "incomplete");
    icon.className = "dk-icon dk-feedback-headline-icon";
    const msg = document.createElement("span");
    msg.className = "dk-feedback-headline-msg";
    msg.textContent = feedback.headline || "";
    li.appendChild(icon);
    li.appendChild(msg);
    el.feedbackList.appendChild(li);

    el.feedbackCard.hidden = false;
  }

  // Clear all grading overlays (called on a new stroke, reset, or exercise swap).
  function clearFeedback() {
    glyphLayer.destroyChildren();
    glyphBoxNodes.clear();
    glyphLayer.draw();
    popupLayer.destroyChildren();
    popupLayer.draw();
    el.popupHost.innerHTML = "";
    el.feedbackList.innerHTML = "";
    el.feedbackCard.hidden = true;
    _activeBubbles = [];
    // Restore empty state so the column is never blank between exercises.
    if (el.feedbackEmpty) el.feedbackEmpty.hidden = false;
  }

  // ---------------------------------------------------------------- 5. panel

  // Render the result panel from an EvaluateResponse. Recognized card shows
  // equation_label (prominent), kind pill, symbol/row count, and a column-
  // aligned token grid coloured by token.color (from palette.py / annotation.py).
  function renderPanel(resp) {
    const recognized = resp.recognized || {};
    const eqKind = recognized.equation_kind || recognized.equationKind || "";
    // Heading: prefer the active exercise's title (known regardless of model's
    // partial-ink classification) so a child drawing only the answer does not
    // see "Bare digits" inside a known exercise. Fall back to the recognizer
    // label only when no exercise is selected.
    const activeEx = state.exercises.find((e) => e.exerciseId === state.selectedId);
    const exerciseTitle = activeEx && (activeEx.title || activeEx.prompt);
    const eqLabel = exerciseTitle
      || recognized.equation_label || recognized.equationLabel || prettyKind(eqKind) || "";
    const rowCount = recognized.row_count !== undefined ? recognized.row_count
                   : (recognized.rowCount !== undefined ? recognized.rowCount : null);

    // Prominent equation label (e.g. "Subtraction")
    el.recognizedEq.textContent = eqLabel;
    el.recognizedEq.removeAttribute("data-empty");

    // Kind pill mirrors eqLabel (operator-derived) so it never contradicts the heading.
    if (eqLabel && eqLabel !== "") {
      el.recognizedKind.textContent = eqLabel;
      el.recognizedKind.hidden = false;
    } else {
      el.recognizedKind.hidden = true;
    }

    const tokens = Array.isArray(recognized.tokens) ? recognized.tokens : [];
    if (el.recognizedCount) {
      if (tokens.length > 0) {
        const rowPart = rowCount !== null
          ? ", " + rowCount + " row" + (rowCount !== 1 ? "s" : "")
          : "";
        el.recognizedCount.textContent = tokens.length + " symbol" + (tokens.length !== 1 ? "s" : "") + rowPart;
        el.recognizedCount.hidden = false;
      } else {
        el.recognizedCount.textContent = "";
        el.recognizedCount.hidden = true;
      }
    }

    // Column-aligned token grid: placed by (row, grid_col) so place-value columns
    // align vertically. grid_col is the scene-global cluster id from the GNN;
    // falls back to tok.col on older payloads.
    el.recognizedTokens.innerHTML = "";
    if (tokens.length > 0) {
      const rowMap = new Map();   // row -> Map<gridCol, token>
      let minCol = Infinity, maxCol = -Infinity;

      for (const tok of tokens) {
        const r = tok.row !== undefined ? tok.row : 0;
        const gc = tok.grid_col !== undefined ? tok.grid_col
                 : (tok.col !== undefined ? tok.col : 0);
        if (!rowMap.has(r)) rowMap.set(r, new Map());
        // If two tokens share a grid_col in the same row (rare), keep both by
        // appending to an array under that slot.
        const slotMap = rowMap.get(r);
        if (!slotMap.has(gc)) slotMap.set(gc, []);
        slotMap.get(gc).push(tok);
        if (gc < minCol) minCol = gc;
        if (gc > maxCol) maxCol = gc;
      }

      if (!isFinite(minCol)) { minCol = 0; maxCol = 0; }

      // One CSS grid column per grid_col slot (0-based: colIndex = gc - minCol).
      const colCount = maxCol - minCol + 1;
      const grid = document.createElement("div");
      grid.className = "dk-token-grid";
      grid.style.gridTemplateColumns = "repeat(" + colCount + ", 1fr)";

      const sortedRows = Array.from(rowMap.keys()).sort((a, b) => a - b);

      for (const r of sortedRows) {
        const slotMap = rowMap.get(r);
        for (let gc = minCol; gc <= maxCol; gc++) {
          const cell = document.createElement("div");
          cell.className = "dk-token-cell";
          cell.style.gridColumn = String(gc - minCol + 1); // 1-based CSS grid column

          const toks = slotMap.get(gc);
          if (toks && toks.length > 0) {
            for (const tok of toks) {
              const chip = document.createElement("span");
              chip.className = "dk-chip";
              const g = tok.glyph !== undefined ? tok.glyph
                      : (tok.label !== undefined ? tok.label : "?");
              chip.textContent = String(g);
              if (tok.color) {
                chip.style.color = tok.color;
                chip.style.borderColor = tok.color + "55"; // ~33% alpha
              }
              if (tok.low_conf) {
                chip.title = "low confidence";
                chip.classList.add("dk-chip--warn");
              }
              cell.appendChild(chip);
            }
          }
          grid.appendChild(cell);
        }
      }
      el.recognizedTokens.appendChild(grid);
    }

    const timings = resp.timings || {};
    const total = timings.total_ms !== undefined ? timings.total_ms : timings.totalMs;
    el.timeTotal.textContent = (total !== undefined ? Math.round(total) + " ms" : "not run");
    renderTimings(timings);
    const used = !!resp.opencv_used;
    el.opencvBadge.textContent = "OpenCV: " + (used ? "used" : "not used");
    el.opencvBadge.className = "dk-badge " + (used ? "dk-badge--on" : "dk-badge--off");

    const marks = resp.marks || {};
    const earned = marks.earned !== undefined ? marks.earned : 0;
    const totalMarks = marks.total !== undefined ? marks.total : 0;
    el.marks.textContent = earned + " / " + totalMarks + " correct";
    const progress = Number(resp.progress || 0);
    const pct = Math.max(0, Math.min(100, Math.round(progress * 100)));
    el.progressFill.style.width = pct + "%";
    el.progressText.textContent = pct + "%";
    // Update ARIA progressbar state.
    const progressEl = el.progressFill.parentElement && el.progressFill.parentElement.parentElement;
    if (progressEl && progressEl.getAttribute("role") === "progressbar") {
      progressEl.setAttribute("aria-valuenow", String(pct));
    }

    const disagreements = countDisagreements(resp);
    if (disagreements > 0) {
      el.reconcile.hidden = false;
      el.reconcile.textContent =
        "Some symbols were placed slightly off-guide, but the recognizer understood them.";
    } else {
      el.reconcile.hidden = true;
    }
  }

  function renderTimings(timings) {
    const subs = [
      ["yolo_ms", "detection"],
      ["stage2_ms", "structure"],
      ["assemble_ms", "assemble"],
      ["engine_ms", "grading engine API"],
    ];
    el.timingsList.innerHTML = "";
    let any = false;
    for (const [key, label] of subs) {
      const v = timings[key];
      if (v === undefined) continue;
      any = true;
      const dt = document.createElement("dt");
      dt.textContent = label;
      const dd = document.createElement("dd");
      dd.textContent = Math.round(v) + " ms";
      el.timingsList.appendChild(dt);
      el.timingsList.appendChild(dd);
    }
    el.timingsDetail.hidden = !any;
  }

  function renderDevDebug(resp) {
    if (!resp.debug || !el.engineExchange) return;
    const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
    const safeClass = (s) => String(s).toLowerCase().replace(/[^a-z0-9-]/g, "");

    // -- helper: build a positional grid HTML string from a list of cells -----
    // cells: [{x:N, y:N, char:str, cssClass:str, title:str}]
    // engine y is bottom-origin (larger y = higher up) so we render higher y at the top.
    function _buildEngineGrid(cells) {
      if (!cells.length) return "<em>empty</em>";
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (const c of cells) {
        if (c.x < minX) minX = c.x;
        if (c.x > maxX) maxX = c.x;
        if (c.y < minY) minY = c.y;
        if (c.y > maxY) maxY = c.y;
      }
      const cols = maxX - minX + 1;
      const rows = maxY - minY + 1;
      // Build a lookup: (x,y) -> cell
      const lookup = new Map();
      for (const c of cells) lookup.set(c.x + "," + c.y, c);

      // One column per engine x; one row per engine y rendered top-to-bottom (higher y = top).
      let gridCells = "";
      for (let gy = maxY; gy >= minY; gy--) {
        for (let gx = minX; gx <= maxX; gx++) {
          const cell = lookup.get(gx + "," + gy);
          if (cell) {
            const titleAttr = cell.title ? ` title="${esc(cell.title)}"` : "";
            gridCells += `<div class="dk-engine-cell ${esc(cell.cssClass)}"${titleAttr}>${esc(cell.char)}</div>`;
          } else {
            gridCells += `<div class="dk-engine-cell"></div>`;
          }
        }
      }
      return `<div class="dk-engine-grid-wrap"><div class="dk-engine-grid" style="grid-template-columns:repeat(${cols},22px)">${gridCells}</div></div>`;
    }

    // -- SENT grid ------------------------------------------------------------
    const sent = resp.debug.engine_sent || {};
    const tokens = Array.isArray(sent.tokens) ? sent.tokens : [];
    let sentHtml = "";
    if (sent.session_id) {
      sentHtml += `<div class="dk-engine-kv"><span class="dk-engine-key">session</span><code class="dk-engine-val">${esc(sent.session_id)}</code></div>`;
    }
    if (tokens.length) {
      // Map each token to a grid cell; role drives CSS class (fixed vs generated).
      const cells = tokens.map((t) => {
        const role = String(t.role || "").toLowerCase();
        const cssClass = role === "fixed" ? "dk-engine-cell--fixed" : "dk-engine-cell--generated";
        const ch = (t.c !== undefined && t.c !== null && String(t.c) !== "") ? String(t.c) : "_";
        return {
          x: Number(t.x),
          y: Number(t.y),
          char: ch,
          cssClass,
          title: String(t.id) + " " + String(t.role || ""),  // escaped once in _buildEngineGrid
        };
      });
      sentHtml += _buildEngineGrid(cells);
      sentHtml += `<div class="dk-engine-legend">`;
      sentHtml += `<div class="dk-engine-legend-item"><div class="dk-engine-legend-dot" style="background:var(--dk-text-sub);opacity:0.45"></div>fixed (given)</div>`;
      sentHtml += `<div class="dk-engine-legend-item"><div class="dk-engine-legend-dot" style="background:var(--dk-border)"></div>generated (child)</div>`;
      sentHtml += `</div>`;
    } else {
      sentHtml += "<em>nothing sent</em>";
    }
    el.engineSent.innerHTML = sentHtml;

    // -- RECEIVED grid --------------------------------------------------------
    const rcv = resp.debug.engine_received || {};
    let rcvHtml = "";
    const pct = Math.round((Number(rcv.progress) || 0) * 100);
    rcvHtml += `<div class="dk-engine-kv"><span class="dk-engine-key">progress</span><code class="dk-engine-val">${esc(pct)}%</code></div>`;
    const elements = Array.isArray(rcv.elements) ? rcv.elements : [];
    if (elements.length) {
      // Build id → token lookup from the SENT tokens so we can resolve each
      // element's constituent ids to their individual (x,y) grid positions and
      // display characters. engine groups multi-cell tokens into ONE element
      // (e.g. operand "256" → ids:[G7,G8,G9], each id at its own (x,y)). We
      // must emit one RECEIVED cell per id, not one cell per element, so the
      // RECEIVED grid mirrors the SENT grid's cell coverage.
      const tokMap = new Map();
      for (const t of tokens) tokMap.set(String(t.id), t);

      const cells = [];
      let droppedNoPos = 0;  // MISSING elements with neither ids nor pos (debug-only)
      for (const e of elements) {
        const st = String(e.status || "").toLowerCase();
        const cssClass = "dk-engine-cell--" + safeClass(st);
        const idsStr = Array.isArray(e.ids) ? e.ids.join(",") : "";  // escaped once in _buildEngineGrid

        if (Array.isArray(e.ids) && e.ids.length > 0) {
          // Expand: one cell per id, each at its own (x,y) from SENT tokens.
          for (const id of e.ids) {
            const tok = tokMap.get(String(id));
            if (!tok || tok.x === undefined || tok.y === undefined) continue;
            const ch = (tok.c !== undefined && tok.c !== null && String(tok.c) !== "")
              ? String(tok.c) : "·";
            cells.push({
              x: Number(tok.x),
              y: Number(tok.y),
              char: ch,
              cssClass,
              title: st + " [" + idsStr + "]",  // escaped once in _buildEngineGrid
            });
          }
        } else {
          // MISSING element with no ids: render a single placeholder at e.pos.
          if (!Array.isArray(e.pos) || e.pos.length < 2) { droppedNoPos++; continue; }
          cells.push({
            x: Number(e.pos[0]),
            y: Number(e.pos[1]),
            char: "·",
            cssClass,
            title: st,  // escaped once in _buildEngineGrid
          });
        }
      }
      if (cells.length) {
        rcvHtml += _buildEngineGrid(cells);
        rcvHtml += `<div class="dk-engine-legend">`;
        rcvHtml += `<div class="dk-engine-legend-item"><div class="dk-engine-legend-dot" style="background:var(--dk-ok)"></div>OK</div>`;
        rcvHtml += `<div class="dk-engine-legend-item"><div class="dk-engine-legend-dot" style="background:var(--dk-error)"></div>error</div>`;
        rcvHtml += `<div class="dk-engine-legend-item"><div class="dk-engine-legend-dot" style="background:var(--dk-missing)"></div>missing</div>`;
        rcvHtml += `</div>`;
      } else {
        rcvHtml += "<em>no graded cells</em>";
      }
      if (droppedNoPos > 0) {
        // Never silently undercount: surface positionless MISSING elements.
        rcvHtml += `<div class="dk-engine-note">${esc(String(droppedNoPos))} MISSING (no position)</div>`;
      }
    }
    const feedbacks = Array.isArray(rcv.feedback) ? rcv.feedback : [];
    if (feedbacks.length) {
      const rows = feedbacks.map((f) => `<tr><td>${esc(f.type)}</td></tr>`).join("");
      rcvHtml += `<div class="dk-engine-subsection">feedback</div><table class="dk-engine-table"><thead><tr><th>messageType</th></tr></thead><tbody>${rows}</tbody></table>`;
    }
    if (rcv.hint) {
      rcvHtml += `<div class="dk-engine-kv"><span class="dk-engine-key">hint</span><code class="dk-engine-val">${esc(rcv.hint.type)}</code></div>`;
    }
    el.engineReceived.innerHTML = rcvHtml;
    el.engineExchange.hidden = false;
  }

  function countDisagreements(resp) {
    const recognized = resp.recognized || {};
    const dl = recognized.disagreements;
    if (Array.isArray(dl)) return dl.length;
    const tokens = recognized.tokens || [];
    return tokens.filter((t) => t.disagreement).length;
  }

  function prettyKind(kind) {
    return String(kind).replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  }

  // ---------------------------------------------------------------- 6. flow

  function renderPicker() {
    el.picker.innerHTML = "";
    for (const ex of state.exercises) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "dk-pill";
      btn.textContent = ex.title || ex.prompt || ex.exerciseId;
      btn.setAttribute("aria-pressed", String(ex.exerciseId === state.selectedId));
      btn.addEventListener("click", () => selectExercise(ex.exerciseId));
      el.picker.appendChild(btn);
    }
  }

  async function loadExercises() {
    try {
      const list = await api.exercises();
      state.exercises = Array.isArray(list) ? list : (list && list.exercises) || [];
      renderPicker();
      if (state.exercises.length > 0) {
        await selectExercise(state.exercises[0].exerciseId);
      }
    } catch (err) {
      banner("Could not load exercises. Refresh the page to try again.", "error");
    }
  }

  async function selectExercise(exerciseId) {
    if (state.busy) return;
    setBusy(true);
    try {
      const resp = await api.select(exerciseId);
      state.selectedId = exerciseId;
      const ex = state.exercises.find((e) => e.exerciseId === exerciseId);
      el.prompt.textContent = (ex && (ex.prompt || ex.title)) || (resp && resp.exercise && (resp.exercise.prompt || resp.exercise.title)) || "Draw your working.";
      state.calibration = normaliseCalibration(resp && (resp.calibration || resp.guideGrid));
      renderGuideGrid(state.calibration);
      renderScaffold(resp && (resp.scaffold || resp.scaffoldTokens), state.calibration);
      clearDrawing();
      resetPanel();
      banner(null);
      renderPicker();
    } catch (err) {
      banner("Could not start that exercise. Try picking it again.", "error");
    } finally {
      setBusy(false);
      syncButtons();
    }
  }

  // Hide the pre-Check empty state. Called once on the first result or error.
  function _hideEmptyState() {
    if (el.feedbackEmpty) el.feedbackEmpty.hidden = true;
  }

  // Show a recoverable error state in the feedback card: icon + message + retry button.
  // Used for network failures (502) and unreadable-drawing (422) -- not for the banner.
  function showFeedbackError(message, onRetry) {
    _hideEmptyState();
    el.feedbackList.innerHTML = "";
    const li = document.createElement("li");
    li.className = "dk-feedback-error";
    const iconWrap = _iconEl("alert");
    iconWrap.className = "dk-icon dk-feedback-error-icon";
    const textWrap = document.createElement("div");
    textWrap.className = "dk-feedback-error-body";
    const p = document.createElement("p");
    p.className = "dk-feedback-error-msg";
    p.textContent = message;
    const retryBtn = document.createElement("button");
    retryBtn.type = "button";
    retryBtn.className = "dk-feedback-error-retry";
    const retryIcon = _iconEl("retry");
    retryIcon.className = "dk-icon";
    const retryLabel = document.createElement("span");
    retryLabel.textContent = "Try again";
    retryBtn.appendChild(retryIcon);
    retryBtn.appendChild(retryLabel);
    retryBtn.addEventListener("click", () => { banner(null); onRetry(); });
    textWrap.appendChild(p);
    textWrap.appendChild(retryBtn);
    li.appendChild(iconWrap);
    li.appendChild(textWrap);
    el.feedbackList.appendChild(li);
    el.feedbackCard.hidden = false;
  }

  async function check() {
    if (state.busy || !state.selectedId) return;
    if (!hasDrawing()) {
      banner("Write your answer on the canvas, then tap Check.", "notice");
      return;
    }
    setBusy(true);
    setCheckLoading(true);
    el.btnCheck.setAttribute("aria-label", "Checking your work");
    banner(null);
    try {
      const resp = await api.evaluate(drawingDataUrl());
      buildIdToToken(resp.recognized || {});
      const finished = !!(resp.marks && resp.marks.finished) || Number(resp.progress || 0) >= 1.0;
      // Prefer the backend's feedback field; fall back to popup-derived for
      // older responses that predate the feedback field.
      const feedback = resp.feedback || deriveFeedback(resp.popups, finished);
      // Show a soft guidance banner for empty-canvas or unreadable-drawing paths.
      if (resp.warning) {
        const isReadingError = resp.warning.toLowerCase().includes("draw") ||
                               resp.warning.toLowerCase().includes("read");
        if (isReadingError) {
          banner("We could not read that. Try writing a little bigger and press Check again.", "warn");
        } else {
          banner(resp.warning, "warn");
        }
      }
      renderMinimalFeedback(feedback, resp.recognized || {});
      renderPanel(resp);
      renderDevDebug(resp);
    } catch (err) {
      const status = err.status;
      if (status === 422) {
        showFeedbackError(
          "We could not read that drawing. Try writing the answer a little bigger.",
          check,
        );
      } else if (status === 502 || status >= 500) {
        showFeedbackError(
          "We could not reach the marker. Check your connection and try again.",
          check,
        );
      } else if (!navigator.onLine) {
        showFeedbackError("No connection. Reconnect and try again.", check);
      } else {
        showFeedbackError("Something went wrong. Try again in a moment.", check);
      }
    } finally {
      el.btnCheck.setAttribute("aria-label", "Check");
      setCheckLoading(false);
      setBusy(false);
      syncButtons();
    }
  }

  function resetPanel() {
    el.recognizedEq.textContent = "";
    el.recognizedEq.setAttribute("data-empty", "true");
    el.recognizedKind.hidden = true;
    if (el.recognizedCount) { el.recognizedCount.textContent = ""; el.recognizedCount.hidden = true; }
    el.recognizedTokens.innerHTML = "";
    el.timeTotal.textContent = "Waiting";
    el.opencvBadge.textContent = "OpenCV";
    el.opencvBadge.className = "dk-badge dk-badge--off";
    el.timingsDetail.hidden = true;
    el.marks.textContent = "No result yet";
    el.progressFill.style.width = "0%";
    el.progressText.textContent = "0%";
    el.reconcile.hidden = true;
    el.feedbackList.innerHTML = "";
    el.feedbackCard.hidden = true;
    if (el.engineExchange) el.engineExchange.hidden = true;
  }

  // ---------------------------------------------------------------- 7. wiring

  function setBusy(on) { state.busy = on; }
  function setCheckLoading(on) { el.btnCheck.dataset.loading = String(on); }

  function syncButtons() {
    const ready = !!state.selectedId && !state.busy;
    el.btnCheck.disabled = !(ready && hasDrawing());
    el.btnReset.disabled = !(ready && hasDrawing());
    el.btnPen.disabled = !ready;
    el.btnEraser.disabled = !ready;
  }

  el.btnCheck.addEventListener("click", check);
  el.btnReset.addEventListener("click", () => { clearDrawing(); resetPanel(); banner(null); });
  el.btnPen.addEventListener("click", () => switchTool("pen"));
  el.btnEraser.addEventListener("click", () => switchTool("eraser"));

  // boot
  loadExercises();
  syncButtons();
})();
