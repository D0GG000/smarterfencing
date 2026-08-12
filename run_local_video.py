#!/usr/bin/env python3
"""Local bout inference for diagnosing attack-type (proximity model).

Interactive ROI pick (OpenCV), then:
  1) 2D extract  2) 3D lift  3) touch + attack predict

Example:
  cd /home/jordan/fencing-mmpose-dev3/app
  export WORKSPACE_ROOT=/home/jordan/fencing-mmpose-dev3/local_workspace
  /home/jordan/miniconda3/envs/mmpose-env/bin/python run_local_video.py \\
    --video /mnt/c/Users/jorda/Downloads/IMG_0298.MOV

Reuse saved boxes:
  python run_local_video.py --video ... --selections /path/to/selections.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace


def _box_dict(x1, y1, x2, y2):
    return {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}


def pick_selections(video_path: str, frame_index: int = 0) -> dict:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idx = max(0, min(frame_index, max(total - 1, 0)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise SystemExit("Could not read selection frame from video")

    h, w = frame.shape[:2]
    print(f"Selection frame {idx}: {w}x{h}")
    print("For each box: drag ROI, ENTER/SPACE confirm, C cancel.")

    labels = [
        ("fencer1", "Fencer 1 (LEFT person)"),
        ("fencer2", "Fencer 2 (RIGHT person)"),
        ("fencer1_light", "Fencer 1 score LIGHT"),
        ("fencer2_light", "Fencer 2 score LIGHT"),
    ]
    out = {
        "video_width": w,
        "video_height": h,
        "light_mode": "static",
        "template_frame_index": idx,
    }
    for key, title in labels:
        vis = frame.copy()
        cv2.putText(
            vis,
            title,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        r = cv2.selectROI(title, vis, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(title)
        x, y, bw, bh = [int(v) for v in r]
        if bw <= 1 or bh <= 1:
            raise SystemExit(f"Empty ROI for {key}")
        out[key] = _box_dict(x, y, x + bw, y + bh)
        print(f"  {key}: {out[key]}")
    return out


def make_app(workspace: str, attack_model: str, touch_model: str | None):
    # Must set before importing workspace_paths / demo.
    os.environ["WORKSPACE_ROOT"] = workspace
    os.environ.setdefault("UPLOAD_DIR", os.path.join(workspace, "uploads"))
    os.environ.setdefault("OUTPUT_2D", os.path.join(workspace, "unlabeled"))
    os.environ.setdefault("OUTPUT_3D", os.path.join(workspace, "3d_outputs"))
    os.environ.setdefault("WORKSPACE_TMP", os.path.join(workspace, "tmp"))
    if attack_model:
        os.environ["ATTACK_MODEL_PATH"] = attack_model
    if touch_model:
        os.environ["MODEL_PATH"] = touch_model

    from workspace_paths import OUTPUT_2D, OUTPUT_3D, UPLOAD_DIR, ensure_workspace_dirs
    from mmpose_paths import attack_classifier_default_path, touch_classifier_default_path

    ensure_workspace_dirs()
    app = SimpleNamespace()
    app.config = {
        "OUTPUT_2D": OUTPUT_2D,
        "OUTPUT_3D": OUTPUT_3D,
        "UPLOAD_DIR": UPLOAD_DIR,
        "ATTACK_MODEL_PATH": os.environ.get("ATTACK_MODEL_PATH") or attack_classifier_default_path(),
        "MODEL_PATH": os.environ.get("MODEL_PATH") or touch_classifier_default_path(),
    }
    return app


def dump_attack_table(results: dict) -> None:
    preds = (results or {}).get("predictions") or []
    print("\n" + "=" * 72)
    print("ATTACK PREDICTIONS")
    print("=" * 72)
    if not preds:
        print("(no predictions)")
        return
    for i, r in enumerate(preds):
        touch = r.get("touch") or r.get("touch_id") or r.get("id") or f"#{i}"
        ap = r.get("attack_prediction")
        conf = r.get("attack_confidence") or []
        probs = ""
        if conf and len(conf) >= 3:
            probs = f"  lunge={conf[0]*100:.0f}% fleche={conf[1]*100:.0f}% other={conf[2]*100:.0f}%"
        print(f"{i+1:2d}. {touch}: {ap}{probs}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Local fencing pipeline (attack diagnose)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--selections", default=None, help="JSON with fencer/light boxes")
    ap.add_argument("--save-selections", default=None)
    ap.add_argument("--frame-index", type=int, default=0)
    ap.add_argument(
        "--workspace",
        default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "local_workspace")
        ),
    )
    ap.add_argument(
        "--attack-model",
        default=os.path.join(os.path.dirname(__file__), "best_attack_3d_proximity_winrobust.pth"),
    )
    ap.add_argument("--touch-model", default=None)
    ap.add_argument("--skip-arm", action="store_true", help="Unused; arm pass is inside pipeline")
    args = ap.parse_args()

    video = os.path.abspath(args.video)
    if not os.path.isfile(video):
        print(f"Missing video: {video}", file=sys.stderr)
        return 1

    if args.selections:
        with open(args.selections, "r", encoding="utf-8") as f:
            selections = json.load(f)
        print(f"Loaded selections: {args.selections}")
    else:
        selections = pick_selections(video, frame_index=args.frame_index)

    save_path = args.save_selections
    if not save_path:
        save_path = os.path.join(args.workspace, "selections_IMG_0298.json")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(selections, f, indent=2)
    print(f"Saved selections: {save_path}")

    app = make_app(args.workspace, args.attack_model, args.touch_model)
    print(f"WORKSPACE: {args.workspace}")
    print(f"ATTACK_MODEL: {app.config['ATTACK_MODEL_PATH']}")
    print(f"TOUCH_MODEL:  {app.config['MODEL_PATH']}")

    # Import after env/workspace is configured.
    from demo import run_full_pipeline, pipeline_state

    run_full_pipeline(app, video, selections)
    results = pipeline_state.get("results") or {}
    out_json = os.path.join(args.workspace, "results_IMG_0298.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Wrote {out_json}")
    dump_attack_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
