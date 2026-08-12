"""
Separate lightweight bout-wide arm-attempt pose pass.

Default backend: ONNX YOLO11s-det + RTMPose-s (fast, good parity with RTMDet+RTMPose).
Fallback: ARM_ATTEMPT_BACKEND=mmpose → RTMDet-m + RTMPose-s (PyTorch/MMPose).

Uses the same fencer filtering / identity as touch analysis:
  vertical_ref_from_fencer_boxes → infer → get_fencer_pair_indices → extract_keypoints_dict

Efficiency (accuracy-neutral):
  - Park ViTPose on CPU during this pass (caller)
  - Prefetch/decode thread overlaps CPU decode with GPU infer
  - ONNX path avoids loading a second PyTorch detector
"""

from __future__ import annotations

import os
import queue
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from arm_attempt_detector import (
    arm_xy_features,
    attempts_payload,
    choose_weapon_arm,
    detect_attempts_from_series,
)
from forward_back import (
    CamCompHipTracker,
    build_forward_back_payload,
    compact_person,
    mid_hip,
    planted_ankle_cam_dx,
)
from fencing_inference import (
    extract_keypoints_dict,
    get_fencer_pair_indices,
    get_shared_person_detector,
    vertical_ref_from_fencer_boxes,
)
from mmpose_paths import (
    arm_attempt_backend,
    rtmdet_person_checkpoint_path,
    rtmdet_person_config_path,
    rtmpose_s_checkpoint_path,
    rtmpose_s_config_path,
)
from test_fencing_vitpose18 import expand_bboxes_xyxy, filter_topk_by_area_xyxy

_arm_det = None
_arm_pose = None
_arm_device = None
_arm_det_is_shared = False
_arm_backend = None
_arm_lock = threading.Lock()

LogFn = Callable[[str], None]

DEFAULT_RULES = {
    "min_peak": 150.0,
    "min_delta": 30.0,
    "min_speed": 300.0,
    "min_point": 0.02,
    "min_wrist_fwd": 0.015,
    "max_jump_deg": 50.0,
    # Every frame; sample_fps = fps/stride in the detector.
    "stride": 1,
    # Planted-ankle cam-compensated hip travel (body-scales / sec).
    "still_thr": 0.35,
    "switch_hold": 2,
    "coast_sec": 0.25,
}

# Pose this many largest in-band detections (enough for gate/referee filter,
# much cheaper than POSE_CANDIDATE_POOL=12 used by ViTPose extract).
ARM_POSE_CANDIDATES = 4

# Prefetch queue depth (decoded frames waiting for GPU). Accuracy-neutral.
ARM_PREFETCH_DEPTH = 8


def ensure_arm_attempt_stack(log_fn: LogFn) -> None:
    """Load arm-attempt pose stack (ONNX by default, or MMPose fallback)."""
    global _arm_det, _arm_pose, _arm_device, _arm_det_is_shared, _arm_backend
    backend = arm_attempt_backend()
    with _arm_lock:
        if _arm_backend == backend:
            if backend == "onnx":
                from onnx_yolo_rtmpose import get_onnx_arm_stack

                if get_onnx_arm_stack() is not None:
                    return
            elif _arm_pose is not None and _arm_det is not None:
                shared = get_shared_person_detector()
                if shared is not None and _arm_det is not shared:
                    _arm_det = shared
                    _arm_det_is_shared = True
                    log_fn("[ARM] Reusing shared RTMDet from main pose stack.")
                return

        _arm_backend = backend
        if backend == "onnx":
            from onnx_yolo_rtmpose import ensure_onnx_arm_stack

            ensure_onnx_arm_stack(log_fn)
            _arm_det = None
            _arm_pose = None
            _arm_det_is_shared = False
            _arm_device = "onnx"
            return

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        shared = get_shared_person_detector()
        pose_cfg = rtmpose_s_config_path()
        pose_ckpt = rtmpose_s_checkpoint_path()
        if not os.path.isfile(pose_cfg):
            raise FileNotFoundError(f"RTMPose-s config missing: {pose_cfg}")

        from mmpose.apis import init_model

        if shared is not None:
            _arm_det = shared
            _arm_det_is_shared = True
            log_fn(f"[ARM] Reusing shared RTMDet; loading RTMPose-s on {device}...")
        else:
            det_cfg = rtmdet_person_config_path()
            det_ckpt = rtmdet_person_checkpoint_path()
            if not os.path.isfile(det_cfg):
                raise FileNotFoundError(f"RTMDet config missing: {det_cfg}")
            log_fn(f"[ARM] Loading RTMDet-m + RTMPose-s on {device}...")
            log_fn(f"[ARM]   det: {det_cfg}")
            from mmdet.apis import init_detector
            from mmpose.utils import adapt_mmdet_pipeline

            det = init_detector(det_cfg, det_ckpt, device=device)
            det.cfg = adapt_mmdet_pipeline(det.cfg)
            _arm_det = det
            _arm_det_is_shared = False

        log_fn(f"[ARM]   pose: {pose_cfg}")
        if _arm_pose is None:
            _arm_pose = init_model(pose_cfg, pose_ckpt, device=device)
        _arm_device = device
        log_fn("[ARM] Lightweight MMPose stack ready.")


def _empty_structured(reason: str) -> dict:
    return {
        "bboxes": [],
        "keypoints": [],
        "bbox_scores": [],
        "_pose_filter_audit": {"mode": "empty", "reason": reason},
    }


def infer_arm_attempt_pose(
    frame: np.ndarray,
    vertical_ref_y0: Optional[float] = None,
    vertical_ref_y1: Optional[float] = None,
    bbox_thr: float = 0.3,
    nms_thr: float = 0.3,
    top_k_persons: int = ARM_POSE_CANDIDATES,
) -> dict:
    """
    Same gate/identity helpers as fencing_inference.infer_pose, but poses only
    the top_k largest in-band people (default 4) instead of POSE_CANDIDATE_POOL.
    """
    if arm_attempt_backend() == "onnx" or _arm_backend == "onnx":
        from onnx_yolo_rtmpose import infer_onnx_arm_pose

        return infer_onnx_arm_pose(
            frame,
            vertical_ref_y0=vertical_ref_y0,
            vertical_ref_y1=vertical_ref_y1,
            bbox_thr=min(float(bbox_thr), 0.25),
            top_k_persons=top_k_persons,
        )

    if _arm_det is None or _arm_pose is None:
        return _empty_structured("models_not_loaded")
    from mmdet.apis import inference_detector
    from mmpose.apis import inference_topdown
    from mmpose.evaluation.functional import nms
    from mmpose.structures import merge_data_samples
    from fencing_inference import (
        filter_dets_vertical_third_xyxy,
        filter_topk_by_inband_area_xyxy,
        select_gate_fencer_instances,
    )

    top_k = max(2, int(top_k_persons))
    try:
        with torch.inference_mode():
            det_result = inference_detector(_arm_det, frame)
            pred = det_result.pred_instances
            if pred is None or len(pred) == 0:
                return _empty_structured("no_detector_instances")
            pred_instance = pred.cpu().numpy()
            dets = np.concatenate(
                (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1
            )
            dets = dets[
                np.logical_and(
                    pred_instance.labels == 0,
                    pred_instance.scores > bbox_thr,
                )
            ]
            dets = dets[nms(dets, nms_thr), :4]
            if vertical_ref_y0 is not None and vertical_ref_y1 is not None:
                dets = filter_dets_vertical_third_xyxy(
                    dets, float(vertical_ref_y0), float(vertical_ref_y1)
                )
            if dets.size == 0:
                return _empty_structured("no_dets_after_band")

            h, w = frame.shape[:2]
            if vertical_ref_y0 is not None and vertical_ref_y1 is not None:
                dets = filter_topk_by_inband_area_xyxy(
                    dets,
                    top_k,
                    float(vertical_ref_y0),
                    float(vertical_ref_y1),
                    True,
                )
            else:
                dets = filter_topk_by_area_xyxy(dets, top_k, True)
            if dets.size == 0:
                return _empty_structured("no_dets_after_topk")

            bboxes = expand_bboxes_xyxy(dets, h, w, 1.0, 0.0)
            pose_results = inference_topdown(_arm_pose, frame, bboxes)
            data_samples = merge_data_samples(pose_results)
            inst = data_samples.pred_instances
            if inst is None or len(inst) == 0:
                return _empty_structured("pose_returned_no_instances")

            bb = inst.bboxes
            kpts = inst.keypoints
            ksc = inst.keypoint_scores
            if isinstance(bb, torch.Tensor):
                bb = bb.detach().cpu().numpy()
            if isinstance(kpts, torch.Tensor):
                kpts = kpts.detach().cpu().numpy()
            if isinstance(ksc, torch.Tensor):
                ksc = ksc.detach().cpu().numpy()

            structured = {"bboxes": [], "keypoints": [], "bbox_scores": []}
            for i in range(int(bb.shape[0])):
                structured["bboxes"].append(
                    [float(bb[i, 0]), float(bb[i, 1]), float(bb[i, 2]), float(bb[i, 3])]
                )
                row = []
                for j in range(kpts.shape[1]):
                    row.append(
                        [float(kpts[i, j, 0]), float(kpts[i, j, 1]), float(ksc[i, j])]
                    )
                structured["keypoints"].append(row)
                structured["bbox_scores"].append(1.0)

            if vertical_ref_y0 is not None and vertical_ref_y1 is not None:
                structured = select_gate_fencer_instances(
                    structured,
                    h,
                    w,
                    top_k=top_k,
                    vertical_y0=float(vertical_ref_y0),
                    vertical_y1=float(vertical_ref_y1),
                    order_left_to_right=True,
                    fallback_if_none=True,
                )
            return structured
    except Exception:
        return _empty_structured("exception")


def _scale_box(box, scale_x, scale_y, name):
    if not box:
        raise ValueError(f"Missing {name}")
    x1 = int(float(box["x1"]) * scale_x)
    y1 = int(float(box["y1"]) * scale_y)
    x2 = int(float(box["x2"]) * scale_x)
    y2 = int(float(box["y2"]) * scale_y)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid {name}")
    return (x1, y1, x2, y2)


def _prefetch_frames(
    video_path: str,
    stride: int,
    proc_scale: float,
    frame_w: int,
    frame_h: int,
    out_q: "queue.Queue",
) -> None:
    """Background decode/resize; puts (frame_i, frame) or None sentinel."""
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            out_q.put(None)
            return
        frame_i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_i % stride != 0:
                frame_i += 1
                continue
            if proc_scale < 1.0:
                frame = cv2.resize(
                    frame, (frame_w, frame_h), interpolation=cv2.INTER_AREA
                )
            out_q.put((frame_i, frame))
            frame_i += 1
    finally:
        cap.release()
        out_q.put(None)


def _append_frame_features(
    series: Dict[str, List[float]],
    hits: List[int],
    structured: dict,
    fencer1_side: str,
    vertical_y0: float,
    vertical_y1: float,
    frame_h: int,
    frame_w: int,
    motion_states: Optional[List[CamCompHipTracker]] = None,
    motion_frames: Optional[List[Dict[str, Any]]] = None,
    frame_i: int = 0,
    fps: float = 30.0,
    ankle_cam_state: Optional[Dict[str, float]] = None,
) -> None:
    f1_id, f2_id = get_fencer_pair_indices(
        structured,
        fencer1_side,
        vertical_y0,
        vertical_y1,
        frame_height=frame_h,
        frame_width=frame_w,
    )
    kps = [
        extract_keypoints_dict(structured, f1_id),
        extract_keypoints_dict(structured, f2_id),
    ]
    for p in range(2):
        kp = kps[p]
        if kp is None:
            for arm in ("left", "right"):
                series[f"p{p}_{arm}_elbow"].append(0.0)
                series[f"p{p}_{arm}_wrist_x"].append(0.0)
                series[f"p{p}_{arm}_shoulder_x"].append(0.0)
            continue
        hits[p] += 1
        for arm in ("left", "right"):
            feat = arm_xy_features(kp, arm, frame_w)
            series[f"p{p}_{arm}_elbow"].append(feat["elbow"])
            series[f"p{p}_{arm}_wrist_x"].append(feat["wrist_x"])
            series[f"p{p}_{arm}_shoulder_x"].append(feat["shoulder_x"])

    if motion_states is not None and motion_frames is not None:
        hips = [mid_hip(kps[0]), mid_hip(kps[1])]
        t = float(frame_i) / max(float(fps), 1e-6)
        prev = ankle_cam_state if ankle_cam_state is not None else {}
        cam_dx, cur_ankles = planted_ankle_cam_dx(kps, prev)
        if ankle_cam_state is not None:
            ankle_cam_state.clear()
            ankle_cam_state.update(cur_ankles)
        people = []
        for p in range(2):
            snap = motion_states[p].update(
                kps[p], t, cam_dx_px=cam_dx, opponent_hip=hips[1 - p]
            )
            people.append(compact_person(snap, frame_w, frame_h))
        motion_frames.append(
            {
                "i": int(frame_i),
                "t": round(t, 4),
                "p": people,
            }
        )

def run_arm_attempt_pass(
    video_path: str,
    selections: dict,
    log_fn: LogFn,
    *,
    stride: Optional[int] = None,
    max_proc_dim: int = 1920,
    rules: Optional[dict] = None,
    prefetch_depth: int = ARM_PREFETCH_DEPTH,
) -> Dict[str, Any]:
    """
    Scan the bout with the arm-attempt pose stack (ONNX YOLO+RTMPose by default).
    Returns attempts for weapon arms (reach); call attribute_arm_attempts_handedness later.
    """
    rules = {**DEFAULT_RULES, **(rules or {})}
    stride = int(stride if stride is not None else rules.get("stride", 1))
    stride = max(1, stride)

    ensure_arm_attempt_stack(log_fn)

    # Probe video metadata (prefetch thread opens its own capture).
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for arm attempts: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    longest = max(native_w, native_h)
    proc_scale = 1.0 if longest <= max_proc_dim else (max_proc_dim / float(longest))
    frame_w = max(1, int(round(native_w * proc_scale)))
    frame_h = max(1, int(round(native_h * proc_scale)))

    sel_w = float(selections.get("video_width", frame_w) or frame_w)
    sel_h = float(selections.get("video_height", frame_h) or frame_h)
    scale_x = frame_w / sel_w
    scale_y = frame_h / sel_h
    fencer1_box = _scale_box(selections.get("fencer1"), scale_x, scale_y, "fencer1")
    fencer2_box = _scale_box(selections.get("fencer2"), scale_x, scale_y, "fencer2")

    # Same left=F1 convention as run_data_extractor
    f1cx = (fencer1_box[0] + fencer1_box[2]) / 2
    f2cx = (fencer2_box[0] + fencer2_box[2]) / 2
    if f1cx > f2cx:
        fencer1_box, fencer2_box = fencer2_box, fencer1_box
    fencer1_side = "left"
    vertical_y0, vertical_y1 = vertical_ref_from_fencer_boxes(
        fencer1_box, fencer2_box, frame_h
    )
    log_fn(
        f"[ARM] Bout scan backend={_arm_backend or arm_attempt_backend()} "
        f"stride={stride} pose_topk={ARM_POSE_CANDIDATES} "
        f"prefetch={prefetch_depth} shared_det={_arm_det_is_shared} "
        f"gate y=[{vertical_y0:.0f},{vertical_y1:.0f}] "
        f"{frame_w}x{frame_h} ({total} frames)"
    )

    series: Dict[str, List[float]] = {}
    for p in range(2):
        for arm in ("left", "right"):
            series[f"p{p}_{arm}_elbow"] = []
            series[f"p{p}_{arm}_wrist_x"] = []
            series[f"p{p}_{arm}_shoulder_x"] = []
    frame_indices: List[int] = []
    hits = [0, 0]
    still_thr = float(rules.get("still_thr", 0.35))
    switch_hold = int(rules.get("switch_hold", 2))
    coast_sec = float(rules.get("coast_sec", 0.25))
    motion_states = [
        CamCompHipTracker(
            still_thr=still_thr, switch_hold=switch_hold, coast_sec=coast_sec
        ),
        CamCompHipTracker(
            still_thr=still_thr, switch_hold=switch_hold, coast_sec=coast_sec
        ),
    ]
    motion_frames: List[Dict[str, Any]] = []
    ankle_cam_state: Dict[str, float] = {}

    frame_q: queue.Queue = queue.Queue(maxsize=max(2, int(prefetch_depth)))
    reader = threading.Thread(
        target=_prefetch_frames,
        args=(video_path, stride, proc_scale, frame_w, frame_h, frame_q),
        name="arm-attempt-prefetch",
        daemon=True,
    )
    reader.start()

    processed = 0
    while True:
        item = frame_q.get()
        if item is None:
            break
        frame_i, frame = item
        structured = infer_arm_attempt_pose(
            frame,
            vertical_ref_y0=vertical_y0,
            vertical_ref_y1=vertical_y1,
        )
        frame_indices.append(frame_i)
        _append_frame_features(
            series,
            hits,
            structured,
            fencer1_side,
            vertical_y0,
            vertical_y1,
            frame_h,
            frame_w,
            motion_states=motion_states,
            motion_frames=motion_frames,
            frame_i=frame_i,
            fps=fps,
            ankle_cam_state=ankle_cam_state,
        )
        processed += 1
        if processed % 100 == 0:
            log_fn(
                f"[ARM] scanned {processed} samples (frame {frame_i}/{total}) "
                f"hits F1={hits[0]} F2={hits[1]}"
            )

    reader.join(timeout=5.0)
    log_fn(f"[ARM] coverage F1={hits[0]}/{processed} F2={hits[1]}/{processed}")

    rule_kw = {
        "min_peak": float(rules["min_peak"]),
        "min_delta": float(rules["min_delta"]),
        "min_speed": float(rules["min_speed"]),
        "min_point": float(rules["min_point"]),
        "min_wrist_fwd": float(rules["min_wrist_fwd"]),
        "max_jump_deg": float(rules["max_jump_deg"]),
    }
    # Same weapon-arm rule as local desktop: shoulder-relative reach toward opponent.
    h1_reach = choose_weapon_arm(series, 0)
    h2_reach = choose_weapon_arm(series, 1)
    log_fn(f"[ARM] weapon arm (reach): F1={h1_reach} F2={h2_reach}")
    attempts = detect_attempts_from_series(
        series,
        fps=fps,
        stride=stride,
        frame_indices=frame_indices,
        arms_by_person={0: h1_reach, 1: h2_reach},
        **rule_kw,
    )
    payload = attempts_payload(
        attempts,
        fencer1_handedness=h1_reach,
        fencer2_handedness=h2_reach,
        rules={
            **rules,
            "stride": stride,
            "pose_topk": ARM_POSE_CANDIDATES,
            "prefetch_depth": prefetch_depth,
            "weapon_source": "reach",
            "pose_backend": _arm_backend or arm_attempt_backend(),
            "fencer1_handedness_reach": h1_reach,
            "fencer2_handedness_reach": h2_reach,
            "still_thr": still_thr,
            "switch_hold": switch_hold,
            "coast_sec": coast_sec,
            "forward_back_method": "anchor_hip",
        },
    )
    payload["forward_back"] = build_forward_back_payload(
        motion_frames,
        fps=fps,
        stride=stride,
        width=frame_w,
        height=frame_h,
        still_thr=still_thr,
        switch_hold=switch_hold,
        method="anchor_hip",
    )
    # Quick summary for logs
    for p in range(2):
        counts = {"forward": 0, "backward": 0, "still": 0, "unknown": 0}
        for row in motion_frames:
            lab = (row.get("p") or [{}])[p].get("l") or "unknown"
            counts[lab] = counts.get(lab, 0) + 1
        log_fn(f"[ARM] footwork F{p + 1}: {counts}")
    log_fn(
        f"[ARM] attempts: {len(attempts)} "
        f"(F1={payload['fencer1_total']} F2={payload['fencer2_total']})"
    )
    return payload

def attribute_arm_attempts_handedness(
    arm_attempts: Optional[dict],
    three_d_batch: Optional[dict],
    log_fn: Optional[LogFn] = None,
) -> Optional[dict]:
    """
    Attach 3D handedness metadata. Counting uses reach-based weapon arms
    (same as desktop); 3D is recorded for display / disagreement logs.
    """
    if not arm_attempts or not isinstance(arm_attempts, dict):
        return arm_attempts

    h1_reach = None
    h2_reach = None
    rules = arm_attempts.get("rules") or {}
    if isinstance(rules, dict):
        h1_reach = rules.get("fencer1_handedness_reach") or arm_attempts.get(
            "fencer1_handedness"
        )
        h2_reach = rules.get("fencer2_handedness_reach") or arm_attempts.get(
            "fencer2_handedness"
        )

    h1_3d = None
    h2_3d = None
    if three_d_batch and isinstance(three_d_batch, dict):
        for payload in three_d_batch.values():
            if not isinstance(payload, dict):
                continue
            if payload.get("fencer1_handedness"):
                h1_3d = payload.get("fencer1_handedness")
            if payload.get("fencer2_handedness"):
                h2_3d = payload.get("fencer2_handedness")
            if h1_3d and h2_3d:
                break

    out = dict(arm_attempts)
    out["fencer1_handedness"] = h1_reach
    out["fencer2_handedness"] = h2_reach
    out["fencer1_handedness_3d"] = h1_3d
    out["fencer2_handedness_3d"] = h2_3d
    rules_out = dict(rules) if isinstance(rules, dict) else {}
    rules_out["weapon_source"] = "reach"
    if h1_3d or h2_3d:
        rules_out["fencer1_handedness_3d"] = h1_3d
        rules_out["fencer2_handedness_3d"] = h2_3d
        if (h1_3d and h1_reach and h1_3d != h1_reach) or (
            h2_3d and h2_reach and h2_3d != h2_reach
        ):
            if log_fn:
                log_fn(
                    f"[ARM] NOTE: 3D handedness differs from reach "
                    f"(reach F1={h1_reach} F2={h2_reach}; "
                    f"3d F1={h1_3d} F2={h2_3d}) — counts use reach"
                )
    out["rules"] = rules_out
    if log_fn:
        log_fn(
            f"[ARM] final F1={out.get('fencer1_total')} F2={out.get('fencer2_total')} "
            f"(weapon reach F1={h1_reach} F2={h2_reach})"
        )
    return out
