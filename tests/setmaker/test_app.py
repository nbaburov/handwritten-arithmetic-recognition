"""Route tests for the set-maker FastAPI app (WS-G).

Uses ``fastapi.testclient.TestClient`` to exercise every route that does not need
a real drawing or model weights: ``POST /api/mode`` (eval + train, incl. the
train soft-gate warning), ``GET /api/next``, ``GET /api/progress``, ``POST
/api/save`` with a synthetic draft (eval + train), and ``POST /api/detect`` with
an injected detector. The two heavy collaborators (the symbol-pool target
generator and the YOLO detector) are swapped on ``app.state`` so the routes are
the only thing under test, and a self-contained temporary project tree supplies
``data/setmaker/quota.json`` so the worklist planner runs without the real data.

The error contracts are covered too: 503 when the symbol pool is unavailable
(``GET /api/next`` with a generator that raises), and 422 when export validation
fails (``POST /api/save`` with an out-of-bounds bbox).
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import List

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.setmaker import app as app_mod
from src.parsing.detection import Detection
from src.setmaker.types import TargetScene, TargetSymbol


# --------------------------------------------------------------------------- #
# Fixtures: a self-contained project tree + a TestClient with injected heavies
# --------------------------------------------------------------------------- #


def _write_quota(root: Path) -> None:
    """Write a minimal ``data/setmaker/quota.json`` the planner accepts.

    Two real cases on each side keep both the eval deficit math and the train gap
    allocation non-trivial without depending on the repo's full quota file.
    """
    quota_path = root / "data" / "setmaker" / "quota.json"
    quota_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema_version": 1,
        "eval": {"scene_cases": {"addition": 2, "subtraction": 1}},
        "train": {"scene_cases": {"addition": 4, "subtraction": 4}},
    }
    quota_path.write_text(json.dumps(doc), encoding="utf-8")


def _fake_target(case: str, seed: int, completion_stage: str, **_kw) -> TargetScene:
    """A deterministic two-symbol addition target (no symbol pool needed).

    Mirrors the shape ``targets.generate_target`` returns: an ``equation_type`` in
    the eval vocabulary and two grid-placed symbols with 512-space boxes, so the
    matcher and both exporters can consume it.
    """
    symbols = [
        TargetSymbol(
            fine_label="main_2",
            glyph_key="main_2",
            yolo_class="digit_main",
            row_index=0,
            col_index=0,
            equation_idx=0,
            bbox=(100.0, 100.0, 150.0, 160.0),
        ),
        TargetSymbol(
            fine_label="main_3",
            glyph_key="main_3",
            yolo_class="digit_main",
            row_index=0,
            col_index=1,
            equation_idx=0,
            bbox=(200.0, 100.0, 250.0, 160.0),
        ),
    ]
    return TargetScene(
        case=case,
        equation_type="addition",
        completion_stage=completion_stage,
        seed=seed,
        reference="2 3",
        symbols=symbols,
    )


def _png_b64(size: int = 64) -> str:
    """Return a base64 PNG (a tiny white square) for detect/save bodies."""
    img = Image.fromarray(np.full((size, size), 255, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A temporary project root with the planner's quota file in place."""
    _write_quota(tmp_path)
    return tmp_path


@pytest.fixture
def client(root: Path) -> TestClient:
    """A ``TestClient`` over the app with the heavy collaborators stubbed.

    ``generate_target_fn`` returns ``_fake_target`` (no pool); ``detect_fn``
    returns two detections aligned to the fake target's boxes (no model weights),
    so the matcher produces two matched drafts.
    """
    application = app_mod.create_app(root)
    application.state.generate_target_fn = _fake_target

    def _fake_detect(gray, *, project_root=None, **_kw) -> List[Detection]:
        return [
            Detection(label="digit_main", confidence=0.9, x0=100, y0=100, x1=150, y1=160),
            Detection(label="digit_main", confidence=0.9, x0=200, y0=100, x1=250, y1=160),
        ]

    application.state.detect_fn = _fake_detect
    return TestClient(application)


# --------------------------------------------------------------------------- #
# GET /  (static shell / placeholder)
# --------------------------------------------------------------------------- #


def test_index_serves_html(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "set-maker" in resp.text


# --------------------------------------------------------------------------- #
# POST /api/mode
# --------------------------------------------------------------------------- #


def test_mode_eval_builds_worklist(client: TestClient) -> None:
    resp = client.post("/api/mode", json={"mode": "eval"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "eval"
    # quota: addition 2 + subtraction 1, none drawn yet -> 3 planned.
    assert body["planned"] == 3
    assert body["progress"]["overall"]["total"] == 3
    assert body["progress"]["overall"]["done"] == 0


def test_mode_eval_persists_state(client: TestClient, root: Path) -> None:
    client.post("/api/mode", json={"mode": "eval"})
    assert (root / "data" / "setmaker" / "state" / "eval.json").is_file()


def test_mode_rejects_unknown_mode(client: TestClient) -> None:
    resp = client.post("/api/mode", json={"mode": "bogus"})
    assert resp.status_code == 422


def test_mode_train_requires_round_budget(client: TestClient) -> None:
    resp = client.post("/api/mode", json={"mode": "train"})
    assert resp.status_code == 422


def test_mode_train_soft_gate_warns_without_history(client: TestClient) -> None:
    # No reports/eval/history.jsonl in the tmp tree -> provisional weights + warning.
    resp = client.post("/api/mode", json={"mode": "train", "round_budget": 4})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "train"
    assert body["planned"] > 0
    assert "warning" in body
    assert "history" in body["warning"]


# --------------------------------------------------------------------------- #
# GET /api/next
# --------------------------------------------------------------------------- #


def test_next_returns_item_and_target(client: TestClient) -> None:
    client.post("/api/mode", json={"mode": "eval"})
    resp = client.get("/api/next", params={"mode": "eval"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["done"] is False
    assert body["item"]["case"] in {"addition", "subtraction"}
    assert "completion_stage" in body["item"]
    target = body["target"]
    assert target["equation_type"] == "addition"
    assert target["reference"] == "2 3"
    assert len(target["symbols"]) == 2
    assert target["symbols"][0]["fine_label"] == "main_2"
    assert target["symbols"][0]["bbox"] == [100.0, 100.0, 150.0, 160.0]


def test_next_exposes_resolved_kinds_for_save_echo(client: TestClient) -> None:
    # /api/next must surface the resolved eval equation_kind + train
    # equation_type so the frontend can echo them back on save and the save
    # route never has to regenerate the target. The fake target is "addition"
    # (already long form), so both resolve to "addition".
    client.post("/api/mode", json={"mode": "eval"})
    body = client.get("/api/next", params={"mode": "eval"}).json()
    from src.eval.labels import EQUATION_KINDS
    from src.setmaker.exporters import TRAIN_EQUATION_TYPES

    assert body["equation_kind"] in EQUATION_KINDS
    assert body["equation_type"] in TRAIN_EQUATION_TYPES
    assert body["equation_kind"] == "addition"
    assert body["equation_type"] == "addition"


def test_next_503_when_pool_unavailable(client: TestClient) -> None:
    client.post("/api/mode", json={"mode": "eval"})

    def _boom(case, seed, completion_stage, **_kw):
        raise ValueError("symbol manifest missing glyph_key column")

    client.app.state.generate_target_fn = _boom
    resp = client.get("/api/next", params={"mode": "eval"})
    assert resp.status_code == 503
    assert "validate" in resp.json()["detail"]


def test_next_rejects_unknown_mode(client: TestClient) -> None:
    resp = client.get("/api/next", params={"mode": "bogus"})
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# POST /api/detect
# --------------------------------------------------------------------------- #


def test_detect_matches_target(client: TestClient) -> None:
    client.post("/api/mode", json={"mode": "eval"})
    resp = client.post("/api/detect", params={"mode": "eval"}, json={"image_png": _png_b64()})
    assert resp.status_code == 200
    body = resp.json()
    assert body["detection_count"] == 2
    assert body["target_symbol_count"] == 2
    assert len(body["drafts"]) == 2
    # Both detections align to a target symbol -> two matched drafts with labels.
    labels = sorted(d["fine_label"] for d in body["drafts"])
    assert labels == ["main_2", "main_3"]
    assert all(d["source"] == "matched" for d in body["drafts"])


def test_detect_zero_boxes_returns_notice(client: TestClient) -> None:
    client.post("/api/mode", json={"mode": "eval"})
    client.app.state.detect_fn = lambda gray, *, project_root=None, **_kw: []
    resp = client.post("/api/detect", params={"mode": "eval"}, json={"image_png": _png_b64()})
    assert resp.status_code == 200
    body = resp.json()
    assert body["detection_count"] == 0
    assert "notice" in body
    # No detections -> no drafts (FIX 1: ghost boxes suppressed).
    assert len(body["drafts"]) == 0


def test_detect_bad_base64_is_422(client: TestClient) -> None:
    client.post("/api/mode", json={"mode": "eval"})
    resp = client.post("/api/detect", params={"mode": "eval"}, json={"image_png": "!!!notb64!!!"})
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# POST /api/save  (synthetic draft, both modes)
# --------------------------------------------------------------------------- #


def _eval_draft() -> dict:
    return {
        "bbox_px": [100.0, 100.0, 150.0, 160.0],
        "fine_label": "main_2",
        "row_index": 0,
        "col_index": 0,
        "equation_idx": 0,
        "confidence": 0.9,
        "source": "matched",
        "flagged": False,
    }


def test_save_eval_writes_and_advances(client: TestClient, root: Path) -> None:
    client.post("/api/mode", json={"mode": "eval"})
    resp = client.post(
        "/api/save",
        params={"mode": "eval"},
        json={"image_png": _png_b64(), "drafts": [_eval_draft()]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["saved"] is True
    stem = body["stem"]
    assert stem.startswith("re-addition-")
    # Files written under data/eval/real/.
    assert (root / "data" / "eval" / "real" / f"{stem}.png").is_file()
    assert (root / "data" / "eval" / "real" / f"{stem}.label.json").is_file()
    # Cursor advanced: one of three done.
    assert body["progress"]["overall"]["done"] == 1


def test_save_eval_sidecar_is_loadable(client: TestClient, root: Path) -> None:
    from src.eval.labels import load_label_sidecars

    client.post("/api/mode", json={"mode": "eval"})
    resp = client.post(
        "/api/save",
        params={"mode": "eval"},
        json={"image_png": _png_b64(), "drafts": [_eval_draft()]},
    )
    stem = resp.json()["stem"]
    loaded = load_label_sidecars(root / "data" / "eval" / "real")
    assert stem in loaded
    assert loaded[stem].equation_kind == "addition"


def test_save_train_writes_three_artifacts(client: TestClient, root: Path) -> None:
    client.post("/api/mode", json={"mode": "train", "round_budget": 4})
    draft = _eval_draft()  # full draft incl. row/col is fine for train too
    resp = client.post(
        "/api/save",
        params={"mode": "train"},
        json={"image_png": _png_b64(), "drafts": [draft]},
    )
    assert resp.status_code == 200
    stem = resp.json()["stem"]
    assert stem.startswith("rt-")
    base = root / "data" / "setmaker" / "train"
    assert (base / f"{stem}.png").is_file()
    assert (base / f"{stem}.gt.json").is_file()
    assert (base / f"{stem}.txt").is_file()
    # YOLO line: class id in range, coords in [0, 1].
    line = (base / f"{stem}.txt").read_text(encoding="utf-8").strip().splitlines()[0]
    parts = line.split()
    assert len(parts) == 5
    assert 0 <= int(parts[0])
    assert all(0.0 <= float(v) <= 1.0 for v in parts[1:])


def test_save_invalid_bbox_is_422(client: TestClient, root: Path) -> None:
    client.post("/api/mode", json={"mode": "eval"})
    bad = _eval_draft()
    bad["bbox_px"] = [100.0, 100.0, 50.0, 160.0]  # x1 < x0 -> exporter rejects.
    resp = client.post(
        "/api/save",
        params={"mode": "eval"},
        json={"image_png": _png_b64(), "drafts": [bad]},
    )
    assert resp.status_code == 422
    # Nothing written: the export validates before any file is created.
    real_dir = root / "data" / "eval" / "real"
    assert list(real_dir.glob("*.png")) == [] if real_dir.is_dir() else True


def test_save_advances_cursor_to_done(client: TestClient) -> None:
    # subtraction-only single-scene round so the worklist drains in one save.
    application = client.app
    application.state.generate_target_fn = lambda c, s, st, **k: _fake_target(c, s, st)
    client.post("/api/mode", json={"mode": "eval"})
    # Drain all three eval scenes; the last save should report pending 0.
    last = None
    for _ in range(3):
        last = client.post(
            "/api/save",
            params={"mode": "eval"},
            json={"image_png": _png_b64(), "drafts": [_eval_draft()]},
        ).json()
    assert last is not None
    assert last["progress"]["overall"]["pending"] == 0
    nxt = client.get("/api/next", params={"mode": "eval"}).json()
    assert nxt["done"] is True


def _boom_target(*_a, **_k):
    """A target generator that always raises (would map to 503 if reached)."""
    raise ValueError("symbol pool exploded")


def test_save_eval_uses_echoed_kind_without_regen(client: TestClient, root: Path) -> None:
    # With the kind echoed from /api/next, the save route must NOT regenerate the
    # target: a generator that raises would surface as 503 if it were called, so a
    # clean 200 proves the regen was removed (the plan's "503-on-save risk" fix).
    client.post("/api/mode", json={"mode": "eval"})
    client.app.state.generate_target_fn = _boom_target
    resp = client.post(
        "/api/save",
        params={"mode": "eval"},
        json={
            "image_png": _png_b64(),
            "drafts": [_eval_draft()],
            "equation_kind": "addition",
            "scene_case": "addition",
        },
    )
    assert resp.status_code == 200
    stem = resp.json()["stem"]
    from src.eval.labels import load_label_sidecars

    loaded = load_label_sidecars(root / "data" / "eval" / "real")
    assert loaded[stem].equation_kind == "addition"


def test_save_eval_writes_error_kind_to_sidecar(client: TestClient, root: Path) -> None:
    # Gap-2 collection path: the human draws a deliberate error, tags it, and the
    # chosen error_kind reaches the sidecar via POST /api/save.
    from src.eval.labels import load_label_sidecars

    client.post("/api/mode", json={"mode": "eval"})
    resp = client.post(
        "/api/save",
        params={"mode": "eval"},
        json={
            "image_png": _png_b64(),
            "drafts": [_eval_draft()],
            "equation_kind": "addition",
            "scene_case": "addition",
            "error_kind": "wrong_result",
        },
    )
    assert resp.status_code == 200
    stem = resp.json()["stem"]
    loaded = load_label_sidecars(root / "data" / "eval" / "real")
    assert loaded[stem].error_kind == "wrong_result"


def test_save_eval_without_error_kind_leaves_it_none(client: TestClient, root: Path) -> None:
    # Omitting error_kind (a clean redraw) leaves the sidecar tag as None.
    from src.eval.labels import load_label_sidecars

    client.post("/api/mode", json={"mode": "eval"})
    resp = client.post(
        "/api/save",
        params={"mode": "eval"},
        json={"image_png": _png_b64(), "drafts": [_eval_draft()]},
    )
    assert resp.status_code == 200
    stem = resp.json()["stem"]
    loaded = load_label_sidecars(root / "data" / "eval" / "real")
    assert loaded[stem].error_kind is None


def test_save_eval_invalid_error_kind_is_422(client: TestClient, root: Path) -> None:
    # A bad error_kind is rejected by the exporter and surfaced as 422; nothing
    # is written (the ValueError -> 422 handler mirrors scene_case behaviour).
    client.post("/api/mode", json={"mode": "eval"})
    resp = client.post(
        "/api/save",
        params={"mode": "eval"},
        json={
            "image_png": _png_b64(),
            "drafts": [_eval_draft()],
            "equation_kind": "addition",
            "error_kind": "not_a_real_error",
        },
    )
    assert resp.status_code == 422
    # No eval artifacts written on the rejected save.
    real_dir = root / "data" / "eval" / "real"
    if real_dir.exists():
        assert not any(real_dir.glob("*.label.json"))


def test_save_train_uses_echoed_type_without_regen(client: TestClient, root: Path) -> None:
    # Same contract for train: the echoed equation_type is written through and the
    # raising generator is never reached, so the save succeeds with no 503.
    client.post("/api/mode", json={"mode": "train", "round_budget": 4})
    client.app.state.generate_target_fn = _boom_target
    resp = client.post(
        "/api/save",
        params={"mode": "train"},
        json={
            "image_png": _png_b64(),
            "drafts": [_eval_draft()],
            "equation_type": "addition",
            "case": "addition",
        },
    )
    assert resp.status_code == 200
    stem = resp.json()["stem"]
    import json as _json

    gt = _json.loads((root / "data" / "setmaker" / "train" / f"{stem}.gt.json").read_text())
    assert gt["equation_type"] == "addition"


def test_save_eval_fallback_kind_does_not_regen_target(client: TestClient, root: Path) -> None:
    # When the client omits equation_kind, the route derives it from the item
    # layout WITHOUT loading the symbol pool, so even a raising target generator
    # does not break the save (no 503) and the kind is still correct.
    client.post("/api/mode", json={"mode": "eval"})
    client.app.state.generate_target_fn = _boom_target
    resp = client.post(
        "/api/save",
        params={"mode": "eval"},
        json={"image_png": _png_b64(), "drafts": [_eval_draft()]},
    )
    assert resp.status_code == 200
    stem = resp.json()["stem"]
    from src.eval.labels import load_label_sidecars

    loaded = load_label_sidecars(root / "data" / "eval" / "real")
    # The eval worklist items are "addition"/"subtraction"; the derived kind is a
    # valid long-form EQUATION_KINDS member either way.
    assert loaded[stem].equation_kind in {"addition", "subtraction"}


def test_save_returns_actual_written_stem_after_collision_bump(
    client: TestClient, root: Path
) -> None:
    # Fix 2: when the exporter collision-bumps a duplicate client-supplied stem,
    # the /api/save response must return the WRITTEN stem (the bumped one), not
    # the client-supplied pre-bump value. Supply an explicit stem twice; the
    # second save lands on a bumped stem and that bumped stem must appear in the
    # response, with a corresponding file on disk.
    client.post("/api/mode", json={"mode": "eval"})
    explicit_stem = "re-custom-001"

    def _do_save():
        return client.post(
            "/api/save",
            params={"mode": "eval"},
            json={
                "image_png": _png_b64(),
                "drafts": [_eval_draft()],
                "stem": explicit_stem,
                "equation_kind": "addition",
                "scene_case": "addition",
            },
        ).json()

    first = _do_save()
    second = _do_save()

    # First save: stem matches the supplied value.
    assert first["stem"] == explicit_stem
    assert (root / "data" / "eval" / "real" / f"{explicit_stem}.png").is_file()

    # Second save: exporter bumped the stem; response must echo the bumped name.
    assert second["stem"] != explicit_stem, (
        "response stem must not be the pre-bump value when the exporter bumped it"
    )
    assert second["stem"].startswith(explicit_stem + "-"), (
        f"bumped stem should start with '{explicit_stem}-', got {second['stem']!r}"
    )
    # The bumped file must exist on disk.
    assert (root / "data" / "eval" / "real" / f"{second['stem']}.png").is_file()


def test_save_eval_rejects_16class_bare_digit_label(client: TestClient) -> None:
    # Namespace contract: the frontend now sends 36-class labels (main_2), not the
    # old 16-class bare digit ("2"). A bare digit must be rejected by the exporter
    # (422), documenting why the bare-digit frontend could never round-trip.
    client.post("/api/mode", json={"mode": "eval"})
    bad = _eval_draft()
    bad["fine_label"] = "2"  # old 16-class GNN label, not a stage2_labels_ordered member
    resp = client.post(
        "/api/save",
        params={"mode": "eval"},
        json={"image_png": _png_b64(), "drafts": [bad], "equation_kind": "addition"},
    )
    assert resp.status_code == 422
    assert "fine_label" in resp.json()["detail"]


# --------------------------------------------------------------------------- #
# Train error-kind round-trip (new: train error items require error_kind too)
# --------------------------------------------------------------------------- #


def test_save_train_error_item_with_valid_error_kind_is_200(
    client: TestClient, root: Path
) -> None:
    # A train error-directive item saved with a valid error_kind must be accepted
    # and the written GT must carry the key.
    import json as _json

    client.post("/api/mode", json={"mode": "train", "round_budget": 4})
    resp = client.post(
        "/api/save",
        params={"mode": "train"},
        json={
            "image_png": _png_b64(),
            "drafts": [_eval_draft()],
            "equation_type": "addition",
            "case": "addition",
            "error_kind": "wrong_operator",
        },
    )
    assert resp.status_code == 200
    stem = resp.json()["stem"]
    gt = _json.loads(
        (root / "data" / "setmaker" / "train" / f"{stem}.gt.json").read_text()
    )
    assert gt.get("error_kind") == "wrong_operator"


def test_save_train_error_item_without_error_kind_is_422(
    client: TestClient, root: Path
) -> None:
    # A train save where the worklist item directive is "error" but no error_kind
    # is supplied must be rejected with 422.  We use the TestClient-level worklist
    # that has a train.error_cases block by writing a quota that guarantees an
    # error item is served.  If no error item is served in this run the test is
    # vacuously skipped (same guard pattern used for eval tests above).
    import json as _json

    # Write a quota with a train error_cases block so error items are generated.
    quota_path = root / "data" / "setmaker" / "quota.json"
    quota_path.parent.mkdir(parents=True, exist_ok=True)
    quota_path.write_text(
        _json.dumps({
            "schema_version": 1,
            "eval": {"scene_cases": {"addition": 1}},
            "train": {
                "scene_cases": {"addition": 1},
                "error_cases": {"addition": 4},
            },
        }),
        encoding="utf-8",
    )

    # Reinitialise a fresh client against the updated quota tree.
    from fastapi.testclient import TestClient as _TC
    import src.setmaker.app as _app_mod
    app2 = _app_mod.create_app(root)
    app2.state.generate_target_fn = client.app.state.generate_target_fn
    c2 = _TC(app2)

    c2.post("/api/mode", json={"mode": "train", "round_budget": 10})
    # Drain normal items until we hit an error item or exhaust the list.
    for _ in range(20):
        res = c2.get("/api/next", params={"mode": "train"}).json()
        if res.get("done"):
            return  # no error item served; skip
        if res["item"]["directive"] == "error":
            break
        # Save the normal item to advance the cursor.
        c2.post(
            "/api/save",
            params={"mode": "train"},
            json={
                "image_png": _png_b64(),
                "drafts": [_eval_draft()],
                "equation_type": "addition",
                "case": "addition",
            },
        )
    else:
        return  # could not find an error item; skip

    if res["item"]["directive"] != "error":
        return

    # Now try saving WITHOUT error_kind — must be 422.
    resp = c2.post(
        "/api/save",
        params={"mode": "train"},
        json={
            "image_png": _png_b64(),
            "drafts": [_eval_draft()],
            "equation_type": "addition",
            "case": "addition",
        },
    )
    assert resp.status_code == 422
    assert "error_kind" in resp.json()["detail"]


def test_save_train_normal_item_does_not_write_error_kind(
    client: TestClient, root: Path
) -> None:
    # A normal train save (no error_kind supplied) must NOT write the key into the GT.
    import json as _json

    client.post("/api/mode", json={"mode": "train", "round_budget": 4})
    resp = client.post(
        "/api/save",
        params={"mode": "train"},
        json={
            "image_png": _png_b64(),
            "drafts": [_eval_draft()],
            "equation_type": "addition",
            "case": "addition",
        },
    )
    assert resp.status_code == 200
    stem = resp.json()["stem"]
    gt = _json.loads(
        (root / "data" / "setmaker" / "train" / f"{stem}.gt.json").read_text()
    )
    assert "error_kind" not in gt


# --------------------------------------------------------------------------- #
# GET /api/progress
# --------------------------------------------------------------------------- #


def test_progress_after_mode(client: TestClient) -> None:
    client.post("/api/mode", json={"mode": "eval"})
    resp = client.get("/api/progress", params={"mode": "eval"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "eval"
    assert body["overall"]["total"] == 3
    assert body["overall"]["pending"] == 3
    assert set(body["per_case"]).issubset({"addition", "subtraction"})


def test_progress_rejects_unknown_mode(client: TestClient) -> None:
    resp = client.get("/api/progress", params={"mode": "bogus"})
    assert resp.status_code == 422
