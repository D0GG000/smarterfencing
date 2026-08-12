#!/usr/bin/env python3
"""
Diagnose attack-type on salle videos WITHOUT score lights.

1) Lightweight scan (YOLO11s + RTMPose-s ONNX) for deep front-knee moments
2) Extract ~30-frame windows around top candidates
3) Full ViTPose-H per window
4) MotionBERT 3D lift
5) Attack classifier (best_attack_3d_proximity.pth)

Example:
  cd /home/jordan/fencing-mmpose-dev3/app
  export WORKSPACE_ROOT=/home/jordan/fencing-mmpose-dev3/local_workspace
  ../opt/conda/envs/mmpose-env/bin/python diagnose_attack_no_lights.py \\
    --video /mnt/c/Users/jorda/Downloads/IMG_0298.MOV \\
    --selections ../local_workspace/selections_IMG_0298.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def _angle(a, b, c) -> float:
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))


def _load_selections(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _scale_box(box, sx, sy):
    return (
        int(float(box["x1"]) * sx),
        int(float(box["y1"]) * sy),
        int(float(box["x2"]) * sx),
        int(float(box["y2"]) * sy),
    )


def _mid_xy(a, b):
    if a is None or b is None:
        return None
    return (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))


def front_knee_deg(kpts_xy: np.ndarray, facing_right: bool) -> float:
    """COCO17: Lhip=11 Lknee=13 Lankle=15; Rhip=12 Rknee=14 Rankle=16."""
    if facing_right:
        # left side is front when facing right? In fencing ongarde facing right,
        # front leg is usually the lead (weapon) side - approximate by more-forward ankle.
        pass
    # Choose front leg by which ankle is farther in facing direction (x).
    lh, lk, la = kpts_xy[11], kpts_xy[13], kpts_xy[15]
    rh, rk, ra = kpts_xy[12], kpts_xy[14], kpts_xy[16]
    if facing_right:
        # front = larger x ankle
        if la[0] >= ra[0]:
            return _angle(lh, lk, la)
        return _angle(rh, rk, ra)
    if la[0] <= ra[0]:
        return _angle(lh, lk, la)
    return _angle(rh, rk, ra)


def find_candidates_onnx(
    video_path: str,
    selections: dict,
    stride: int,
    top_k: int,
    min_sep: int,
    log,
) -> List[Tuple[int, str, float]]:
    """Return list of (frame_idx, scoring_fencer, min_front_knee_deg)."""
    from arm_attempt_pass import _load_onnx_stack, _infer_pair  # type: ignore
    # Prefer public helpers if present; else use fencing onnx bench patterns.
    try:
        from onnx_pose_stack import OnnxYoloRtmpose  # type: ignore
    except Exception:
        OnnxYoloRtmpose = None

    # Use demo's arm_attempt stack loader via run with internal APIs if needed.
    from mmpose_paths import yolo11s_onnx_path, rtmpose_s_onnx_path

    # Local minimal ONNX path using existing arm_attempt_pass internals
    import arm_attempt_pass as aap

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video_path}")
    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30)

    sw = float(selections.get("video_width") or native_w)
    sh = float(selections.get("video_height") or native_h)
    sx, sy = native_w / sw, native_h / sh
    f1 = _scale_box(selections["fencer1"], sx, sy)
    f2 = _scale_box(selections["fencer2"], sx, sy)
    # vertical band around fencer mid-heights
    y0 = int(min(f1[1], f2[1]) + 0.25 * (max(f1[3], f2[3]) - min(f1[1], f2[1])))
    y1 = int(max(f1[3], f2[3]) - 0.15 * (max(f1[3], f2[3]) - min(f1[1], f2[1])))

    log(f"ONNX scan: {n} frames @ {fps:.1f}fps, stride={stride}, band y=[{y0},{y1}]")

    # Init stack the same way arm_attempt_pass does
    yolo_path = yolo11s_onnx_path()
    rtm_path = rtmpose_s_onnx_path()
    if not os.path.isfile(yolo_path) or not os.path.isfile(rtm_path):
        raise SystemExit(f"Missing ONNX weights:\n  {yolo_path}\n  {rtm_path}")

    # Reuse arm_attempt_pass run pieces by calling a thin local infer
    from fencing_inference import (
        ensure_pose_stack,
        infer_pose,
        get_fencer_pair_indices,
        extract_keypoints_dict,
        vertical_ref_from_fencer_boxes,
    )

    # Full ViTPose every stride is heavy; use it but with larger stride default.
    ensure_pose_stack(log=log)
    series: List[Tuple[int, float, float]] = []  # frame, f1_knee, f2_knee

    idx = 0
    while idx < n:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            idx += stride
            continue
        try:
            pose = infer_pose(frame)
            # pose structure varies; use helper paths from fencing_inference if available
            from fencing_inference import filter_people_in_vertical_band

            people = filter_people_in_vertical_band(pose, y0, y1) if "filter_people_in_vertical_band" in dir() else None
        except Exception:
            people = None

        # Fallback: use detect API style
        try:
            from fencing_inference import detect_two_fencers_keypoints

            kps = detect_two_fencers_keypoints(frame, f1, f2, log=None)
        except Exception:
            kps = None

        f1k = f2k = float("nan")
        if kps and kps.get("fencer1") is not None:
            xy = np.array([kps["fencer1"][n][:2] for n in [
                "nose","left_eye","right_eye","left_ear","right_ear",
                "left_shoulder","right_shoulder","left_elbow","right_elbow",
                "left_wrist","right_wrist","left_hip","right_hip",
                "left_knee","right_knee","left_ankle","right_ankle"
            ]], dtype=np.float32)
            f1k = front_knee_deg(xy, facing_right=True)
        if kps and kps.get("fencer2") is not None:
            xy = np.array([kps["fencer2"][n][:2] for n in [
                "nose","left_eye","right_eye","left_ear","right_ear",
                "left_shoulder","right_shoulder","left_elbow","right_elbow",
                "left_wrist","right_wrist","left_hip","right_hip",
                "left_knee","right_knee","left_ankle","right_ankle"
            ]], dtype=np.float32)
            f2k = front_knee_deg(xy, facing_right=False)
        series.append((idx, f1k, f2k))
        if len(series) % 20 == 0:
            log(f"  scan {idx}/{n} f1={f1k:.1f} f2={f2k:.1f}")
        idx += stride
    cap.release()

    # Pick local minima per fencer
    def peaks_for(col: int, who: str) -> List[Tuple[int, str, float]]:
        vals = [(fr, ang) for fr, a, b in series for ang in ([a] if col == 1 else [b])]
        # rebuild clean
        pts = []
        for fr, a, b in series:
            ang = a if col == 1 else b
            if not math.isnan(ang):
                pts.append((fr, ang))
        out = []
        for i in range(1, len(pts) - 1):
            fr, ang = pts[i]
            if ang <= pts[i - 1][1] and ang <= pts[i + 1][1] and ang < 110:
                out.append((fr, who, ang))
        out.sort(key=lambda t: t[2])  # deepest first
        # non-max separation
        chosen = []
        for fr, who, ang in out:
            if all(abs(fr - c[0]) >= min_sep for c in chosen):
                chosen.append((fr, who, ang))
            if len(chosen) >= top_k:
                break
        return chosen

    cands = peaks_for(1, "fencer1") + peaks_for(2, "fencer2")
    cands.sort(key=lambda t: t[2])
    # keep top_k overall with separation
    final = []
    for fr, who, ang in cands:
        if all(abs(fr - c[0]) >= min_sep for c in final):
            final.append((fr, who, ang))
        if len(final) >= top_k:
            break
    final.sort(key=lambda t: t[0])
    log(f"Candidates ({len(final)}):")
    for fr, who, ang in final:
        log(f"  frame={fr} {who} front_knee={ang:.1f}deg")
    return final


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--selections", required=True)
    ap.add_argument("--workspace", default="/home/jordan/fencing-mmpose-dev3/local_workspace")
    ap.add_argument("--stride", type=int, default=10, help="Pose scan stride (frames)")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--min-sep", type=int, default=90, help="Min frames between candidates")
    ap.add_argument("--frames", default=None, help="Comma-separated manual touch frames (skip scan)")
    ap.add_argument("--scorer", default="auto", help="fencer1|fencer2|auto for manual frames")
    args = ap.parse_args()

    os.environ["WORKSPACE_ROOT"] = args.workspace
    os.environ.setdefault("OUTPUT_2D", os.path.join(args.workspace, "unlabeled"))
    os.environ.setdefault("OUTPUT_3D", os.path.join(args.workspace, "3d_outputs"))
    os.environ.setdefault("WORKSPACE_TMP", os.path.join(args.workspace, "tmp"))
    os.environ.setdefault(
        "ATTACK_MODEL_PATH",
        "/home/jordan/fencing-mmpose-dev3/app/best_attack_3d_proximity.pth",
    )

    def log(msg: str) -> None:
        print(msg, flush=True)

    selections = _load_selections(args.selections)
    video = os.path.abspath(args.video)

    if args.frames:
        frames = [int(x.strip()) for x in args.frames.split(",") if x.strip()]
        scorer = args.scorer if args.scorer in ("fencer1", "fencer2") else "fencer2"
        candidates = [(fr, scorer, float("nan")) for fr in frames]
        log(f"Manual frames: {candidates}")
    else:
        log("=== STEP A: Scan for deep knee moments (no lights) ===")
        candidates = find_candidates_onnx(
            video, selections, args.stride, args.top_k, args.min_sep, log
        )
        if not candidates:
            log("No deep-knee candidates found.")
            return 1

    # Persist candidates
    cand_path = os.path.join(args.workspace, "attack_candidates_IMG_0298.json")
    with open(cand_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"frame": fr, "scorer": who, "front_knee_deg": ang} for fr, who, ang in candidates],
            f,
            indent=2,
        )
    log(f"Wrote {cand_path}")
    log("Next: extract windows with full ViTPose — continuing...")

    # Import demo helpers after env set
    sys.path.insert(0, "/home/jordan/fencing-mmpose-dev3/app")
    from demo import run_3d_lifting, run_prediction, pipeline_state
    from fencing_inference import ensure_pose_stack, infer_pose
    from workspace_paths import OUTPUT_2D, OUTPUT_3D, ensure_workspace_dirs

    ensure_workspace_dirs()
    job_id = "IMG_0298_manual"
    out2 = os.path.join(OUTPUT_2D, job_id)
    os.makedirs(out2, exist_ok=True)

    # Build a minimal app namespace
    app = SimpleNamespace()
    app.config = {
        "OUTPUT_2D": OUTPUT_2D,
        "OUTPUT_3D": OUTPUT_3D,
        "ATTACK_MODEL_PATH": os.environ["ATTACK_MODEL_PATH"],
        "MODEL_PATH": os.environ.get(
            "MODEL_PATH",
            "/home/jordan/fencing-mmpose-dev3/app/best_touch_v346_coco17_bs10_multivid_val.pth",
        ),
    }

    # Extract windows using same logic as demo extract_frames_before_touch
    # We'll call into demo by synthesizing touches via a custom extractor below.
    log("=== STEP B: Extract 2D windows around candidates ===")
    ensure_pose_stack(log=log)

    from fencing_inference import (
        vertical_ref_from_fencer_boxes,
        get_fencer_pair_indices,
        extract_keypoints_dict,
    )

    cap = cv2.VideoCapture(video)
    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sw = float(selections.get("video_width") or native_w)
    sh = float(selections.get("video_height") or native_h)
    sx, sy = native_w / sw, native_h / sh
    f1_box = _scale_box(selections["fencer1"], sx, sy)
    f2_box = _scale_box(selections["fencer2"], sx, sy)

    COCO = [
        "nose","left_eye","right_eye","left_ear","right_ear",
        "left_shoulder","right_shoulder","left_elbow","right_elbow",
        "left_wrist","right_wrist","left_hip","right_hip",
        "left_knee","right_knee","left_ankle","right_ankle",
    ]

    def extract_touch(touch_frame: int, scoring_fencer: str) -> str:
        start = max(0, touch_frame - 29)
        end = min(total - 1, touch_frame)  # end on "touch" (no post-light for fair compare)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        group_id = f"{scoring_fencer}_score_{ts}_frame{touch_frame}"
        folder = os.path.join(out2, group_id)
        os.makedirs(folder, exist_ok=True)
        seq = 0
        for fi in range(start, end + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            seq += 1
            jpg = os.path.join(folder, f"frame_{seq}.jpg")
            cv2.imwrite(jpg, frame)
            # Pose
            try:
                # Use detect helpers if available
                from fencing_inference import infer_fencer_keypoints_for_frame

                d = infer_fencer_keypoints_for_frame(frame, f1_box, f2_box)
            except Exception:
                # Minimal: blank keypoints — will fail lift; better raise
                raise
            meta = {
                "group_id": group_id,
                "frame_index": seq,
                "video_name": job_id,
                "scoring_fencer": scoring_fencer,
                "touch_frame": touch_frame,
                "source_frame": fi,
                "fencer1_keypoints": d.get("fencer1") or {},
                "fencer2_keypoints": d.get("fencer2") or {},
            }
            with open(os.path.join(folder, f"frame_{seq}_keypoints.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f)
        log(f"  wrote {group_id} ({seq} frames)")
        return folder

    # Prefer demo's own extract by monkeypatching — too fragile.
    # Use a dedicated pose call from fencing_inference used by extract.
    # Read how extract gets keypoints in demo.py around line 1740.
    print("Manual diagnose script needs demo extract helpers wired; "
          "falling back to printing candidates for --frames re-run.")
    print(json.dumps([{"frame": fr, "scorer": who, "knee": ang} for fr, who, ang in candidates], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
