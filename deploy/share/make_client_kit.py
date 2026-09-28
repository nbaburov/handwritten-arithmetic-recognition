#!/usr/bin/env python3
"""Build a client-facing model kit: friendly-named weights + installer + README.

The client already has the git repo (code, config, active.json pointers, eval bank)
but not the gitignored *.pt weights. This script packs every unique weight on disk
under a human-readable name, and ships an install.sh that copies each one back to the
exact repo-relative path(s) active.json and the run dirs expect. Unzip + bash install.sh
+ run, with no need to decode timestamped run ids.

    python3 deploy/share/make_client_kit.py

Produces: deploy/share/har_client_models.zip
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO / "artifacts"
STAGE_DIR = REPO / "deploy" / "share" / "client_kit"
ZIP_OUT = REPO / "deploy" / "share" / "har_client_models.zip"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def slugify(text: str) -> str:
    text = text.replace("·", "-")  # middle dot -> hyphen
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")
    return re.sub(r"-+", "-", text)


def stage_of(rel: str) -> str:
    if rel.startswith("artifacts/yolo/"):
        return "yolo"
    if rel.startswith("artifacts/gnn/"):
        return "gnn"
    if rel.startswith("artifacts/pretrained/"):
        return "pretrained"
    return "other"


def run_id_of(rel: str) -> str | None:
    m = re.search(r"/runs/([^/]+)/", rel)
    return m.group(1) if m else None


def main() -> None:
    active_yolo = load_json(ARTIFACTS / "yolo" / "active.json")
    active_gnn = load_json(ARTIFACTS / "gnn" / "active.json")
    labels = load_json(ARTIFACTS / "run_labels.json")

    active_run = {"yolo": active_yolo.get("run_id"), "gnn": active_gnn.get("run_id")}
    metrics_by_run = {
        active_yolo.get("run_id"): active_yolo.get("metrics", {}),
        active_gnn.get("run_id"): active_gnn.get("metrics", {}),
    }
    # Non-active runs only carry a one-line summary in index.json; fold it in as a test dict.
    for stage, key in (("yolo", "mAP50"), ("gnn", "test_fine_acc")):
        for entry in load_json(ARTIFACTS / stage / "runs" / "index.json").get("runs", []):
            rid = entry.get("run_id")
            if rid and rid not in metrics_by_run:
                field = "mAP50" if stage == "yolo" else "fine_label_accuracy"
                metrics_by_run[rid] = {"test": {field: entry.get(key)}}

    # Group every weight on disk by content hash -> list of repo-relative destinations.
    blobs: dict[str, dict] = {}
    for pt in sorted(ARTIFACTS.rglob("*.pt")):
        rel = pt.relative_to(REPO).as_posix()
        digest = sha256(pt)
        blobs.setdefault(digest, {"size": pt.stat().st_size, "dests": []})
        blobs[digest]["dests"].append(rel)

    # Assign each unique blob a friendly name + human description. Classify off the
    # canonical `runs/<id>/best.pt` destination (a flat `artifacts/<stage>/best.pt` has no
    # run id, and a `weights/last.pt` is a training checkpoint, not the promoted best).
    records = []
    for digest, info in blobs.items():
        dests = sorted(info["dests"])
        stage = stage_of(dests[0])

        canonical = next(
            (d for d in dests if re.search(r"/runs/[^/]+/best\.pt$", d)), None
        )
        run_id = run_id_of(canonical) if canonical else None
        is_active = run_id is not None and run_id == active_run.get(stage)

        if stage == "pretrained":
            name = "yolo_pretrained_yolov8n_coco_base"
            desc = "Ultralytics YOLOv8n COCO pretraining base. Only needed to retrain Stage 1; not used at inference."
            role = "base"
        elif canonical and labels.get(stage, {}).get(run_id):
            label = labels[stage][run_id]
            role = "ACTIVE" if is_active else "rollback"
            name = f"{stage}_{slugify(label)}"
            desc = f"{label}. " + (
                "Promoted, default model the pipeline loads via active.json."
                if is_active
                else "Retained fallback; not loaded unless you repoint active.json."
            )
        elif any(d.endswith("weights/last.pt") for d in dests):
            tr_run = run_id_of(next(d for d in dests if d.endswith("weights/last.pt")))
            name = f"{stage}_training_last_epoch_{tr_run or ''}".rstrip("_")
            desc = "Last-epoch training checkpoint (not the best epoch). Diagnostic only; do not run with this."
            role = "training-intermediate"
        else:
            name = f"{stage}_{run_id or 'unlabeled'}"
            desc = "Weight on disk with no run label."
            role = "other"

        records.append(
            {
                "name": name + ".pt",
                "sha256": digest,
                "size": info["size"],
                "stage": stage,
                "run_id": run_id,
                "role": role,
                "active": is_active,
                "dests": dests,
                "desc": desc,
                "metrics": metrics_by_run.get(run_id, {}),
            }
        )

    # Active first, then rollbacks, then base/training.
    role_order = {"ACTIVE": 0, "rollback": 1, "base": 2, "training-intermediate": 3, "other": 4}
    records.sort(key=lambda r: (role_order.get(r["role"], 9), r["stage"], r["name"]))

    # Stage the kit.
    if STAGE_DIR.exists():
        shutil.rmtree(STAGE_DIR)
    weights_dir = STAGE_DIR / "weights"
    weights_dir.mkdir(parents=True)

    for r in records:
        src = REPO / r["dests"][0]
        shutil.copy2(src, weights_dir / r["name"])

    (STAGE_DIR / "MANIFEST.json").write_text(json.dumps(records, indent=2) + "\n")
    (STAGE_DIR / "install.sh").write_text(render_install(records))
    (STAGE_DIR / "README.md").write_text(render_readme(records, active_yolo, active_gnn))

    # Zip it (store the kit under a top-level folder so it extracts cleanly).
    if ZIP_OUT.exists():
        ZIP_OUT.unlink()
    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(STAGE_DIR.rglob("*")):
            if path.is_file():
                zf.write(path, Path("har_client_models") / path.relative_to(STAGE_DIR))

    total = sum(r["size"] for r in records)
    print(f"Unique weights: {len(records)}  (from {sum(len(r['dests']) for r in records)} files on disk)")
    for r in records:
        print(f"  {r['role']:20} {r['name']:46} {r['size']/1e6:5.1f} MB -> {len(r['dests'])} path(s)")
    print(f"\nKit staged : {STAGE_DIR.relative_to(REPO)}")
    print(f"Zip created: {ZIP_OUT.relative_to(REPO)} ({ZIP_OUT.stat().st_size/1e6:.1f} MB, {total/1e6:.1f} MB uncompressed)")


def render_install(records: list[dict]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "# Plug the model weights into the repo. Run this from anywhere; it finds the repo root via git.",
        "#   bash install.sh",
        "set -euo pipefail",
        "",
        'KIT_DIR="$(cd "$(dirname "$0")" && pwd)"',
        'REPO_ROOT="$(git -C "$KIT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"',
        'if [ -z "$REPO_ROOT" ]; then',
        '  REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"',
        "fi",
        'if [ -z "$REPO_ROOT" ] || [ ! -f "$REPO_ROOT/artifacts/yolo/active.json" ]; then',
        '  echo "ERROR: run this from inside the handwritten-arithmetic-recognition git repo." >&2',
        '  echo "       (cd into the cloned repo, then: bash <path-to-kit>/install.sh)" >&2',
        "  exit 1",
        "fi",
        'echo "Repo root: $REPO_ROOT"',
        "",
    ]
    for r in records:
        lines.append(f'# {r["name"]}  ({r["role"]})')
        for dest in r["dests"]:
            lines.append(f'mkdir -p "$REPO_ROOT/$(dirname "{dest}")"')
            lines.append(f'cp "$KIT_DIR/weights/{r["name"]}" "$REPO_ROOT/{dest}"')
            lines.append(f'echo "  placed {dest}"')
        lines.append("")
    lines += [
        'echo ""',
        'echo "Done. Verify the active weights resolved:"',
        'echo "  ls -lh $REPO_ROOT/artifacts/yolo/runs/*/best.pt $REPO_ROOT/artifacts/gnn/runs/*/best.pt"',
        'echo ""',
        'echo "Run the GUI:"',
        'echo "  cd $REPO_ROOT && PYTHONPATH=\\"\\$(pwd)\\" .venv/bin/python -m src gui --project-root \\"\\$(pwd)\\""',
        "",
    ]
    return "\n".join(lines)


def _fmt_metrics(stage: str, m: dict) -> str:
    test = m.get("test", {})
    if not test:
        return ""
    if stage == "yolo":
        fields = [("mAP50", "mAP50"), ("mAP50_95", "mAP50-95")]
    else:
        fields = [
            ("fine_label_accuracy", "fine_acc"),
            ("row_cluster_accuracy", "row"),
            ("col_cluster_accuracy", "col"),
            ("equation_type_accuracy", "eq_type"),
        ]
    # Only print fields actually present (index.json carries a one-line subset).
    parts = [f"{label} {test[key]:.4f}" for key, label in fields if test.get(key) is not None]
    return "test " + ", ".join(parts) if parts else ""


def render_readme(records: list[dict], active_yolo: dict, active_gnn: dict) -> str:
    lines = [
        "# Model kit — handwritten-arithmetic-recognition",
        "",
        "You already have the git repo (code, config, `active.json` pointers, eval bank).",
        "This kit holds only the model weights, which are gitignored (`*.pt`) and never committed.",
        "",
        "## Quick start",
        "",
        "```bash",
        "# 1. cd into the cloned repo",
        "cd handwritten-arithmetic-recognition",
        "",
        "# 2. plug the weights into place (copies each .pt to the path the repo expects)",
        "bash /path/to/har_client_models/install.sh",
        "",
        "# 3. set up the env if you have not already",
        "python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt",
        "",
        "# 4. run the GUI (draw / upload a handwritten sum)",
        'PYTHONPATH="$(pwd)" .venv/bin/python -m src gui --project-root "$(pwd)"',
        "# open http://127.0.0.1:7860",
        "",
        "# or the feedback demo",
        'PYTHONPATH="$(pwd)" .venv/bin/python -m src demo --project-root "$(pwd)"',
        "```",
        "",
        "`install.sh` reads nothing it should not: it copies each named weight to the exact",
        "repo-relative path(s) it came from, so `active.json` resolves the right model with no edits.",
        "",
        "## What loads when you run",
        "",
        "The pipeline always loads the two models flagged ACTIVE below; `active.json` points at them.",
        "Everything else is a labelled fallback or a training artifact and is ignored unless you",
        "deliberately repoint `active.json`.",
        "",
        "| File in `weights/` | Role | Size | Goes to | Metrics |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in records:
        dest_str = "<br>".join(f"`{d}`" for d in r["dests"])
        metric_str = _fmt_metrics(r["stage"], r["metrics"]) or "—"
        lines.append(
            f"| `{r['name']}` | {r['role']} | {r['size']/1e6:.1f} MB | {dest_str} | {metric_str} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- The flat `artifacts/yolo/best.pt` and `artifacts/gnn/best.pt` are *stale fallbacks*",
        "  (older iter7 / GNN-B weights). They are only used if `active.json` is missing, so the",
        "  installer restores them for fidelity but they are not what runs.",
        "- The pretrained `yolov8n` base and the `last_epoch` checkpoint are for retraining and",
        "  diagnostics only; do not run inference with them.",
        "- To regenerate this kit after promoting new models: `python3 deploy/share/make_client_kit.py`.",
        "- `MANIFEST.json` lists every file with its sha256 and all destinations for verification.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
