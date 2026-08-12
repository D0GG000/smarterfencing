"""
ONNX YOLO11s person detect + RTMPose-s (SimCC) pose.

Used by the bout-wide arm-attempt pass. Self-contained (no rtmlib): letterbox
YOLO + affine crop + SimCC decode. Returns the same structured dict shape as
MMPose top-down infer so fencer gating stays unchanged.
"""

from __future__ import annotations

import os
import threading
from typing import List, Optional, Tuple

import cv2
import numpy as np

from mmpose_paths import yolo11s_onnx_path, rtmpose_s_onnx_path

POSE_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
POSE_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

_stack = None
_stack_lock = threading.Lock()


def _ort_providers() -> List:
    import onnxruntime as ort

    avail = set(ort.get_available_providers())
    if "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def bbox_xyxy2cs(
    bbox: np.ndarray, padding: float = 1.25
) -> Tuple[np.ndarray, np.ndarray]:
    bbox = np.asarray(bbox[:4], dtype=np.float32)
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]
    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding
    if dim == 1:
        return center[0], scale[0]
    return center, scale


def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    return np.array([[cs, -sn], [sn, cs]]) @ pt


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direction = a - b
    return b + np.r_[-direction[1], direction[0]]


def get_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: Tuple[int, int],
) -> np.ndarray:
    src_w = float(scale[0])
    dst_w = float(output_size[0])
    dst_h = float(output_size[1])
    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0.0, src_w * -0.5]), rot_rad)
    dst_dir = np.array([0.0, dst_w * -0.5])
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = dst[0, :] + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def top_down_affine(
    input_size_wh: Tuple[int, int],
    scale: np.ndarray,
    center: np.ndarray,
    img: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop/resize person bbox to model input (w, h) = (192, 256) for RTMPose-s."""
    w, h = int(input_size_wh[0]), int(input_size_wh[1])
    aspect_ratio = w / float(h)
    scale = np.asarray(scale, dtype=np.float32).reshape(2).copy()
    b_w, b_h = float(scale[0]), float(scale[1])
    if b_w > b_h * aspect_ratio:
        scale = np.array([b_w, b_w / aspect_ratio], dtype=np.float32)
    else:
        scale = np.array([b_h * aspect_ratio, b_h], dtype=np.float32)
    warp_mat = get_warp_matrix(center, scale, 0.0, output_size=(w, h))
    out = cv2.warpAffine(img, warp_mat, (w, h), flags=cv2.INTER_LINEAR)
    return out, scale


def get_simcc_maximum(
    simcc_x: np.ndarray, simcc_y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if simcc_x.ndim == 2:
        simcc_x = simcc_x[None]
        simcc_y = simcc_y[None]
    n, k, _ = simcc_x.shape
    x = simcc_x.reshape(n * k, -1)
    y = simcc_y.reshape(n * k, -1)
    x_locs = np.argmax(x, axis=1)
    y_locs = np.argmax(y, axis=1)
    scores = np.minimum(np.amax(x, axis=1), np.amax(y, axis=1))
    locs = np.stack([x_locs, y_locs], axis=-1).astype(np.float32).reshape(n, k, 2)
    scores = scores.astype(np.float32).reshape(n, k)
    scores[scores <= 0] = 0
    return locs, scores


def letterbox_yolo(
    img_bgr: np.ndarray, size: int = 640
) -> Tuple[np.ndarray, float]:
    h, w = img_bgr.shape[:2]
    ratio = min(size / h, size / w)
    nh, nw = int(round(h * ratio)), int(round(w * ratio))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    padded[:nh, :nw] = resized
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)[None], ratio


def nms_xyxy(
    boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45
) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


class YoloRtmOnnxStack:
    def __init__(self, det_path: Optional[str] = None, pose_path: Optional[str] = None):
        import onnxruntime as ort

        det_path = det_path or yolo11s_onnx_path()
        pose_path = pose_path or rtmpose_s_onnx_path()
        if not os.path.isfile(det_path):
            raise FileNotFoundError(f"YOLO11s ONNX missing: {det_path}")
        if not os.path.isfile(pose_path):
            raise FileNotFoundError(f"RTMPose-s ONNX missing: {pose_path}")

        providers = _ort_providers()
        self.det = ort.InferenceSession(det_path, providers=providers)
        self.pose = ort.InferenceSession(pose_path, providers=providers)
        self.det_in = self.det.get_inputs()[0].name
        self.pose_in = self.pose.get_inputs()[0].name
        self.det_path = det_path
        self.pose_path = pose_path
        self.providers = list(self.det.get_providers())

    def detect(
        self,
        frame_bgr: np.ndarray,
        score_thr: float = 0.25,
        top_k: int = 8,
        imgsz: int = 640,
        nms_iou: float = 0.45,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x, ratio = letterbox_yolo(frame_bgr, imgsz)
        out = self.det.run(None, {self.det_in: x})[0]
        pred = out[0].T
        boxes_cxcywh = pred[:, :4]
        person_scores = pred[:, 4:][:, 0]
        keep = person_scores >= score_thr
        if not np.any(keep):
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
        boxes_cxcywh = boxes_cxcywh[keep]
        person_scores = person_scores[keep]
        cx, cy, bw, bh = boxes_cxcywh.T
        boxes = (
            np.stack(
                [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0],
                axis=1,
            )
            / max(ratio, 1e-6)
        )
        h, w = frame_bgr.shape[:2]
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
        idx = nms_xyxy(boxes, person_scores, iou_thr=nms_iou)
        boxes = boxes[idx]
        scores = person_scores[idx].astype(np.float32)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = np.argsort(-areas)[: max(1, int(top_k))]
        return boxes[order].astype(np.float32), scores[order]

    def pose_one(
        self, frame_bgr: np.ndarray, bbox: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        center, scale = bbox_xyxy2cs(bbox, padding=1.25)
        resized, scale = top_down_affine((192, 256), scale, center, frame_bgr)
        x = (resized.astype(np.float32) - POSE_MEAN) / POSE_STD
        x = x.transpose(2, 0, 1)[None]
        simcc_x, simcc_y = self.pose.run(None, {self.pose_in: x})
        locs, scores = get_simcc_maximum(simcc_x, simcc_y)
        kpts = locs[0] / 2.0
        kpts = kpts / np.array([192.0, 256.0], dtype=np.float32) * scale
        kpts = kpts + center - scale / 2
        return kpts.astype(np.float64), scores[0].astype(np.float64)


def get_onnx_arm_stack() -> Optional[YoloRtmOnnxStack]:
    return _stack


def ensure_onnx_arm_stack(log_fn=None) -> YoloRtmOnnxStack:
    global _stack
    with _stack_lock:
        if _stack is not None:
            return _stack
        det = yolo11s_onnx_path()
        pose = rtmpose_s_onnx_path()
        if log_fn:
            log_fn(f"[ARM] Loading ONNX YOLO11s-det + RTMPose-s...")
            log_fn(f"[ARM]   det: {det}")
            log_fn(f"[ARM]   pose: {pose}")
        _stack = YoloRtmOnnxStack(det_path=det, pose_path=pose)
        if log_fn:
            log_fn(f"[ARM] ONNX providers={_stack.providers}")
            log_fn("[ARM] Lightweight ONNX pose stack ready.")
        return _stack


def infer_onnx_arm_pose(
    frame: np.ndarray,
    *,
    vertical_ref_y0: Optional[float] = None,
    vertical_ref_y1: Optional[float] = None,
    bbox_thr: float = 0.25,
    top_k_persons: int = 4,
    min_kpt_score: float = 0.15,
) -> dict:
    """
    Detect + pose; apply the same vertical-band / top-k / gate helpers as MMPose path.
    """
    from fencing_inference import (
        filter_dets_vertical_third_xyxy,
        filter_topk_by_inband_area_xyxy,
        select_gate_fencer_instances,
    )
    from test_fencing_vitpose18 import filter_topk_by_area_xyxy

    stack = ensure_onnx_arm_stack()
    top_k = max(2, int(top_k_persons))
    try:
        boxes, scores = stack.detect(frame, score_thr=bbox_thr, top_k=max(top_k, 8))
        if boxes.size == 0:
            return {
                "bboxes": [],
                "keypoints": [],
                "bbox_scores": [],
                "_pose_filter_audit": {"mode": "empty", "reason": "no_detector_instances"},
            }
        dets = boxes.astype(np.float32)
        if vertical_ref_y0 is not None and vertical_ref_y1 is not None:
            dets = filter_dets_vertical_third_xyxy(
                dets, float(vertical_ref_y0), float(vertical_ref_y1)
            )
        if dets.size == 0:
            return {
                "bboxes": [],
                "keypoints": [],
                "bbox_scores": [],
                "_pose_filter_audit": {"mode": "empty", "reason": "no_dets_after_band"},
            }
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
            return {
                "bboxes": [],
                "keypoints": [],
                "bbox_scores": [],
                "_pose_filter_audit": {"mode": "empty", "reason": "no_dets_after_topk"},
            }

        structured = {"bboxes": [], "keypoints": [], "bbox_scores": []}
        # Keep score aligned with filtered dets by re-matching areas (order preserved).
        for i in range(int(dets.shape[0])):
            bbox = dets[i]
            kpts, ksc = stack.pose_one(frame, bbox)
            if float(np.mean(ksc)) < min_kpt_score:
                continue
            structured["bboxes"].append(
                [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
            )
            row = []
            for j in range(min(17, len(kpts))):
                row.append([float(kpts[j, 0]), float(kpts[j, 1]), float(ksc[j])])
            structured["keypoints"].append(row)
            structured["bbox_scores"].append(1.0)

        if not structured["bboxes"]:
            return {
                "bboxes": [],
                "keypoints": [],
                "bbox_scores": [],
                "_pose_filter_audit": {
                    "mode": "empty",
                    "reason": "pose_returned_no_instances",
                },
            }

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
        return {
            "bboxes": [],
            "keypoints": [],
            "bbox_scores": [],
            "_pose_filter_audit": {"mode": "empty", "reason": "exception"},
        }
