#!/usr/bin/env python3
"""
Manual-touch diagnose using the SAME extract/track logic as demo.py.

Requires USER-drawn fencer boxes (no auto-pair override).
Window matches demo: touch-29 .. touch+5 (attack uses last 30 of that clip).

  cd /home/jordan/fencing-mmpose-dev3/app
  export WORKSPACE_ROOT=/home/jordan/fencing-mmpose-dev3/local_workspace
  ../opt/conda/envs/mmpose-env/bin/python run_manual_touches.py \\
    --video /mnt/c/Users/jorda/Downloads/IMG_0298.MOV \\
    --selections /mnt/c/Users/jorda/Desktop/fencing/selections_IMG_0298_user.json \\
    --scorer fencer2 --frames 77,612,979,1528
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def log(msg: str) -> None:
    print(msg, flush=True)


def _box(sel: dict, key: str, sx: float, sy: float) -> Tuple[int, int, int, int]:
    b = sel[key]
    return (
        int(float(b["x1"]) * sx),
        int(float(b["y1"]) * sy),
        int(float(b["x2"]) * sx),
        int(float(b["y2"]) * sy),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--selections", required=True, help="User-drawn fencer boxes JSON")
    ap.add_argument("--frames", required=True, help="comma-separated touch frame indices")
    ap.add_argument("--scorer", default="fencer2")
    ap.add_argument(
        "--workspace",
        default="/home/jordan/fencing-mmpose-dev3/local_workspace",
    )
    ap.add_argument(
        "--attack-model",
        default="/home/jordan/fencing-mmpose-dev3/app/best_attack_3d_proximity_winrobust.pth",
    )
    # Match demo.extract_frames_before_touch exactly (end on light)
    ap.add_argument("--pre-light", type=int, default=29)
    ap.add_argument("--post-light", type=int, default=0)
    ap.add_argument(
        "--job-id",
        default="IMG_0298_manual",
        help="Output folder name under unlabeled/ and 3d_outputs/",
    )
    args = ap.parse_args()

    frames = [int(x.strip()) for x in args.frames.split(",") if x.strip()]
    if not frames:
        log("No frames")
        return 1

    os.environ["WORKSPACE_ROOT"] = os.path.abspath(args.workspace)
    os.environ["OUTPUT_2D"] = os.path.join(args.workspace, "unlabeled")
    os.environ["OUTPUT_3D"] = os.path.join(args.workspace, "3d_outputs")
    os.environ["WORKSPACE_TMP"] = os.path.join(args.workspace, "tmp")
    os.environ["ATTACK_MODEL_PATH"] = args.attack_model

    app_dir = os.path.dirname(os.path.abspath(__file__))
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    from workspace_paths import OUTPUT_2D, OUTPUT_3D, ensure_workspace_dirs
    from fencing_inference import (
        ensure_pose_stack,
        infer_pose,
        vertical_ref_from_fencer_boxes,
        get_fencer_pair_indices,
        extract_keypoints_dict,
    )
    from demo import enforce_fencer1_left_in_touch_folder
    from attack_classifier import (
        AttackClassifier,
        CLASSES,
        POSE_DIM,
        load_attack_3d_batch,
    )

    ensure_workspace_dirs()
    with open(args.selections, "r", encoding="utf-8") as f:
        selections = json.load(f)

    video = os.path.abspath(args.video)
    job_id = args.job_id
    out2 = os.path.join(OUTPUT_2D, job_id)
    out3 = os.path.join(OUTPUT_3D, job_id)
    if os.path.isdir(out2):
        shutil.rmtree(out2)
    if os.path.isdir(out3):
        shutil.rmtree(out3)
    os.makedirs(out2, exist_ok=True)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        log(f"Cannot open {video}")
        return 1
    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30)

    # Same scaling as demo.run_full_pipeline
    sw = float(selections.get("video_width") or native_w)
    sh = float(selections.get("video_height") or native_h)
    sx, sy = native_w / sw, native_h / sh
    f1_box = _box(selections, "fencer1", sx, sy)
    f2_box = _box(selections, "fencer2", sx, sy)

    # Same left/right swap as demo
    f1cx = (f1_box[0] + f1_box[2]) / 2
    f2cx = (f2_box[0] + f2_box[2]) / 2
    if f1cx > f2cx:
        log("WARNING: F1/F2 boxes reversed on X; swapping so F1 is left")
        f1_box, f2_box = f2_box, f1_box
    fencer1_side = "left"

    vertical_y0, vertical_y1 = vertical_ref_from_fencer_boxes(
        f1_box, f2_box, native_h
    )

    log(f"Video {native_w}x{native_h} @ {fps:.1f}fps n={total}")
    log(f"USER boxes (no auto): f1={list(f1_box)} f2={list(f2_box)}")
    log(f"Touches: scorer={args.scorer} frames={frames}")
    log(
        f"Window: -{args.pre_light} .. +{args.post_light} "
        f"(demo default -29..+5)"
    )
    log(f"Vertical band y=[{vertical_y0}, {vertical_y1}]")
    log(
        "TRACK rules: gate=ViTPose upper+lower in vertical band; "
        "identity=ref-box match then area L/R fallback (same as demo get_fencer_ids)"
    )

    ensure_pose_stack(log)

    def get_fencer_ids(results, audit_out=None):
        # Exact same call signature as demo.extract_frames_before_touch
        return get_fencer_pair_indices(
            results,
            fencer1_side,
            vertical_y0,
            vertical_y1,
            native_h,
            native_w,
            fencer1_ref_box=f1_box,
            fencer2_ref_box=f2_box,
            debug_log=None,
            audit_out=audit_out,
        )

    def extract_one(touch_frame: int, scoring_fencer: str) -> Optional[str]:
        # Mirror demo.extract_frames_before_touch
        start = max(0, touch_frame - args.pre_light)
        end = min(total - 1, touch_frame + args.post_light)
        light_frame_seq = touch_frame - start + 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        group_id = f"{scoring_fencer}_score_{ts}_frame{touch_frame}"
        group_dir = os.path.join(out2, group_id)
        os.makedirs(group_dir, exist_ok=True)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames_bgr: List[Any] = []
        for _ in range(end - start + 1):
            ok, fr = cap.read()
            if ok:
                frames_bgr.append(fr)
        if not frames_bgr:
            log(f"  FAIL {touch_frame}: no frames read")
            return None

        log(
            f"  Pose {len(frames_bgr)} frames for touch@{touch_frame} "
            f"(light_seq={light_frame_seq}, end=+{args.post_light}) ..."
        )
        saved = 0
        methods: Dict[str, int] = {}
        track_audit_frames = []
        for idx, fr in enumerate(frames_bgr):
            frame_num = idx + 1
            results = infer_pose(
                fr,
                vertical_ref_y0=vertical_y0,
                vertical_ref_y1=vertical_y1,
            )
            frame_audit: Dict[str, Any] = {
                "seq": frame_num,
                "video_frame_index": start + idx,
                "n_bboxes": len((results or {}).get("bboxes") or []),
            }
            if not results or not results.get("bboxes"):
                frame_audit["status"] = "no_dets"
                track_audit_frames.append(frame_audit)
                continue
            pair_audit: Dict[str, Any] = {}
            f1_id, f2_id = get_fencer_ids(results, audit_out=pair_audit)
            frame_audit["pair"] = {
                "method": pair_audit.get("method"),
                "f1_idx": pair_audit.get("f1_idx"),
                "f2_idx": pair_audit.get("f2_idx"),
                "ok": pair_audit.get("ok"),
                "fail_reason": pair_audit.get("fail_reason"),
                "gate_idxs": pair_audit.get("gate_idxs"),
            }
            method = pair_audit.get("method") or "none"
            methods[method] = methods.get(method, 0) + 1
            if f1_id is None or f2_id is None:
                frame_audit["status"] = "no_pair"
                track_audit_frames.append(frame_audit)
                continue
            f1_kp = extract_keypoints_dict(results, f1_id)
            f2_kp = extract_keypoints_dict(results, f2_id)
            if not f1_kp or not f2_kp:
                frame_audit["status"] = "null_kp"
                track_audit_frames.append(frame_audit)
                continue

            boxes = results.get("bboxes") or []
            if f1_id < len(boxes):
                frame_audit["f1_bbox"] = [round(float(v), 1) for v in boxes[f1_id][:4]]
            if f2_id < len(boxes):
                frame_audit["f2_bbox"] = [round(float(v), 1) for v in boxes[f2_id][:4]]
            frame_audit["status"] = "saved"
            frame_audit["method"] = method
            track_audit_frames.append(frame_audit)

            cv2.imwrite(os.path.join(group_dir, f"frame_{frame_num}.jpg"), fr)
            with open(
                os.path.join(group_dir, f"frame_{frame_num}_keypoints.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {
                        "frame_index": frame_num,
                        "video_frame_index": start + idx,
                        "light_frame_seq": light_frame_seq,
                        "frame_width": native_w,
                        "frame_height": native_h,
                        "scoring_fencer": scoring_fencer,
                        "fencer1_keypoints": f1_kp,
                        "fencer2_keypoints": f2_kp,
                        "fencer1_det_index": f1_id,
                        "fencer2_det_index": f2_id,
                        "track_method": method,
                        "group_id": group_id,
                        "video_name": job_id,
                    },
                    f,
                    indent=2,
                )
            saved += 1

        with open(os.path.join(group_dir, "track_audit.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "touch_frame": touch_frame,
                    "light_frame_seq": light_frame_seq,
                    "extract_start_frame": start,
                    "extract_end_frame": end,
                    "scoring_fencer": scoring_fencer,
                    "fencer1_box": list(map(float, f1_box)),
                    "fencer2_box": list(map(float, f2_box)),
                    "vertical_gate": [vertical_y0, vertical_y1],
                    "pair_methods": methods,
                    "frames": track_audit_frames,
                },
                f,
                indent=2,
            )

        swapped = enforce_fencer1_left_in_touch_folder(group_dir)
        log(
            f"  saved {saved}/{len(frames_bgr)} methods={methods} "
            f"identity_swap={swapped}"
        )
        return group_dir if saved else None

    log("=== STEP 1: Extract 2D (demo-equivalent) ===")
    folders = []
    for fr in frames:
        d = extract_one(fr, args.scorer)
        if d:
            folders.append(d)
    cap.release()
    if not folders:
        log("No touches extracted")
        return 1

    app = SimpleNamespace()
    app.config = {
        "OUTPUT_2D": OUTPUT_2D,
        "OUTPUT_3D": OUTPUT_3D,
        "ATTACK_MODEL_PATH": args.attack_model,
    }

    import demo as demo_mod

    demo_mod.log = log

    log("=== STEP 2: 3D lift ===")
    demo_mod.run_3d_lifting(app, OUTPUT_2D, OUTPUT_3D, job_id)

    log("=== STEP 3: Attack only (skip touch clf SSL) ===")
    clf = AttackClassifier(
        args.attack_model,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    results = []
    files_3d = sorted(
        f for f in os.listdir(out3) if f.endswith("_3d.json")
    )
    for fn in files_3d:
        touch = fn.replace("_3d.json", "")
        p3 = os.path.join(out3, fn)
        p2 = os.path.join(out2, touch)
        x = load_attack_3d_batch(p3, p2)
        pose = x[:, :POSE_DIM]
        fk = pose[:, 6]
        with torch.no_grad():
            logits = clf.model(
                torch.FloatTensor(x).unsqueeze(0).to(clf.device)
            )
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        idx = int(probs.argmax())
        touch_frame = None
        if "_frame" in touch:
            try:
                touch_frame = int(touch.split("_frame")[-1])
            except ValueError:
                pass
        feats = {
            "min_front_knee_deg": float(np.degrees(np.min(fk))),
            "end_front_knee_deg": float(np.degrees(fk[-1])),
            "peak_knee_frame": int(np.argmin(fk) + 1),
        }
        entry = {
            "touch": touch,
            "touch_frame": touch_frame,
            "attack_prediction": CLASSES[idx],
            "attack_confidence": [float(p) for p in probs],
            "features_3d": feats,
        }
        # attach track methods
        audit_p = os.path.join(p2, "track_audit.json")
        if os.path.isfile(audit_p):
            with open(audit_p, encoding="utf-8") as f:
                audit = json.load(f)
            entry["pair_methods"] = audit.get("pair_methods")
            entry["fencer1_box"] = audit.get("fencer1_box")
            entry["fencer2_box"] = audit.get("fencer2_box")
        results.append(entry)
        log(
            f"  frame {touch_frame}: {CLASSES[idx]}  "
            f"L={probs[0]*100:.1f}% F={probs[1]*100:.1f}% O={probs[2]*100:.1f}%  "
            f"knee3d_min={feats['min_front_knee_deg']:.1f}° "
            f"peak@{feats['peak_knee_frame']} methods={entry.get('pair_methods')}"
        )

    out_json = os.path.join(args.workspace, "results_IMG_0298_manual.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"Wrote {out_json}")

    # Identity overlays at light frame (seq = light_frame_seq)
    ov_dir = os.path.join(args.workspace, "identity_overlays_user")
    os.makedirs(ov_dir, exist_ok=True)
    for entry in results:
        touch = entry["touch"]
        p2 = os.path.join(out2, touch)
        audit_p = os.path.join(p2, "track_audit.json")
        if not os.path.isfile(audit_p):
            continue
        with open(audit_p, encoding="utf-8") as f:
            audit = json.load(f)
        light_seq = int(audit["light_frame_seq"])
        jp = os.path.join(p2, f"frame_{light_seq}_keypoints.json")
        img_p = os.path.join(p2, f"frame_{light_seq}.jpg")
        if not (os.path.isfile(jp) and os.path.isfile(img_p)):
            # fall back to last saved
            kps = sorted(
                [
                    n
                    for n in os.listdir(p2)
                    if n.startswith("frame_") and n.endswith("_keypoints.json")
                ],
                key=lambda n: int(n.split("_")[1]),
            )
            if not kps:
                continue
            jp = os.path.join(p2, kps[-1])
            light_seq = int(kps[-1].split("_")[1])
            img_p = os.path.join(p2, f"frame_{light_seq}.jpg")
        with open(jp, encoding="utf-8") as f:
            data = json.load(f)
        img = cv2.imread(img_p)
        if img is None:
            continue

        def draw(kp, color, label):
            pts = []
            for name in (
                "nose",
                "left_shoulder",
                "right_shoulder",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
                "left_wrist",
                "right_wrist",
            ):
                p = (kp or {}).get(name)
                if p:
                    xy = (int(p[0]), int(p[1]))
                    cv2.circle(img, xy, 5, color, -1)
                    pts.append(xy)
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                cv2.rectangle(
                    img,
                    (min(xs) - 8, min(ys) - 8),
                    (max(xs) + 8, max(ys) + 8),
                    color,
                    2,
                )
                cv2.putText(
                    img,
                    label,
                    (min(xs), max(20, min(ys) - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )

        draw(data.get("fencer1_keypoints"), (0, 255, 0), "F1")
        draw(data.get("fencer2_keypoints"), (0, 0, 255), "F2")
        # also draw user ref boxes
        for box, col in ((f1_box, (0, 200, 0)), (f2_box, (0, 0, 200))):
            cv2.rectangle(
                img, (box[0], box[1]), (box[2], box[3]), col, 1
            )
        outp = os.path.join(
            ov_dir, f"f{entry.get('touch_frame')}_light_seq{light_seq}.jpg"
        )
        cv2.imwrite(outp, img)
        log(f"  overlay {outp}")

    log("\n" + "=" * 72)
    log("ATTACK RESULTS (user boxes, demo window -29..+5)")
    log("=" * 72)
    for i, r in enumerate(results):
        conf = r.get("attack_confidence") or []
        probs = ""
        if len(conf) >= 3:
            probs = f"  L={conf[0]*100:.0f}% F={conf[1]*100:.0f}% O={conf[2]*100:.0f}%"
        feats = r.get("features_3d") or {}
        log(
            f"{i+1}. frame={r.get('touch_frame')}  "
            f"attack={r.get('attack_prediction')}{probs}  "
            f"knee3d_min={feats.get('min_front_knee_deg')}  "
            f"methods={r.get('pair_methods')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
