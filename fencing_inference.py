"""
RTMDet + COCO-17 ViTPose-H with vertical-band gating.

Used by demo pipeline and /api/detect-frame.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from mmpose_paths import (
    mmpose_root,
    rtmdet_person_checkpoint_path,
    rtmdet_person_config_path,
    vitpose_coco17_config_path,
    vitpose_h_checkpoint_path,
)
from test_fencing_vitpose18 import (
    expand_bboxes_xyxy,
    filter_topk_by_area_xyxy,
)

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

_POSE_CONFIG = vitpose_coco17_config_path()
_POSE_CHECKPOINT = vitpose_h_checkpoint_path()
_MMPOSE_ROOT = mmpose_root()
_DET_CONFIG = rtmdet_person_config_path()

_pose_detector = None
_pose_estimator = None
_pose_device = None
_pose_init_lock = threading.Lock()

COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# Fraction of bbox height that must lie inside [y0,y1] (strictly > this value).
VERTICAL_BAND_MIN_HEIGHT_OVERLAP = 0.7

# When selecting two fencers inside the vertical band, pose more candidates than
# needed so a large partial body (e.g. referee torso at the strip front) can be
# dropped after keypoint visibility checks.
POSE_CANDIDATE_POOL = 12

# Reject tiny false people that still pass the vertical gate (hallucinated
# keypoints in a ~40x100 box). Old "2 largest bodies" avoided this; gate+nearest
# ref alone does not. Prefer height/area vs the UI ref — width is not a hard cut
# because a left-facing fencer can be a tall narrow silhouette.
REF_SIZE_MIN_HEIGHT_FRAC = 0.55
REF_SIZE_MIN_AREA_FRAC = 0.20
# Among size-ok candidates, reject anyone farther than this fraction of the
# F1↔F2 ref separation (stops far opposite-strip bodies from winning F2).
# Slightly loose so a real opponent who steps inward from the setup box still
# competes against a smaller bystander glued to the stale F2 ref.
REF_MATCH_MAX_DIST_FRAC = 0.60
REF_MATCH_MAX_DIST_MIN_PX = 140.0
# Drop clearly smaller same-side bodies once a full-size candidate exists
# (sideline person ~0.3× the real fencer area).
PEER_AREA_MIN_FRAC = 0.55
# F1/F2 should be similar on-camera scale (same strip depth).
PARTNER_AREA_MIN_FRAC = 0.40

# Bbox within this fraction of a frame edge is treated as clipped / cut off.
FRAME_EDGE_MARGIN_FRAC = 0.02

# Minimum keypoint confidence to count a joint as visible.
KEYPOINT_VISIBLE_CONF = 0.3

# Knees/ankles must meet this stricter threshold (pose often hallucinates legs on torsos).
LEG_KEYPOINT_VISIBLE_CONF = 0.4

# When the bbox is clipped at the bottom, require higher leg confidence.
LEG_KEYPOINT_CLIPPED_CONF = 0.5

# Full fencer on strip should span at least this fraction of frame height.
# Crouching en-garde boxes are often ~18–25% of frame height.
MIN_STRIP_FENCER_HEIGHT_FRAC = 0.18

# Bottom-of-frame torso (referee at strip front): box top sits too low in the frame.
# Pose often hallucinates knees/ankles on these, so geometry must veto them.
PARTIAL_BODY_MAX_TOP_FRAC = 0.35

# Wide bottom-clipped boxes are almost always torsos, not standing fencers.
PARTIAL_BODY_MAX_ASPECT_WH = 0.75

_logger = logging.getLogger(__name__)

# COCO-17 indices for body-segment visibility.
_HEAD_INDICES = (0, 1, 2, 3, 4)
_SHOULDER_INDICES = (5, 6)
_HIP_INDICES = (11, 12)
_LOWER_LEG_INDICES = (13, 14, 15, 16)  # knees + ankles
_LOWER_BODY_INDICES = _HIP_INDICES + _LOWER_LEG_INDICES  # hips, knees, ankles
_LOWER_LEG_NAMES = ("left_knee", "right_knee", "left_ankle", "right_ankle")
_LOWER_BODY_NAMES = (
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# Fraction of confident lower-body joints that must lie in the vertical gate.
# "Most" — forgiving when one leg joint flickers off.
LOWER_BODY_IN_GATE_MIN_FRAC = 0.5


def bbox_area_in_vertical_band_xyxy(
    xyxy, band_y0: float, band_y1: float
) -> float:
    """Area of the axis-aligned bbox ∩ horizontal strip [band_y0, band_y1] (pixels²)."""
    x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
    w = max(0.0, x2 - x1)
    iy0 = max(y1, band_y0)
    iy1 = min(y2, band_y1)
    h = max(0.0, iy1 - iy0)
    return w * h


def _any_keypoint_visible(
    keypoints_row, indices: tuple, conf_thresh: float = KEYPOINT_VISIBLE_CONF
) -> bool:
    for i in indices:
        if i >= len(keypoints_row):
            continue
        if float(keypoints_row[i][2]) >= conf_thresh:
            return True
    return False


def _keypoint_confident_and_in_frame(
    keypoints_row,
    index: int,
    frame_height: int,
    frame_width: int,
    conf_thresh: float,
) -> bool:
    """A joint counts only when confidence is high enough and (x,y) lies inside the frame."""
    if index >= len(keypoints_row):
        return False
    x, y, conf = (
        float(keypoints_row[index][0]),
        float(keypoints_row[index][1]),
        float(keypoints_row[index][2]),
    )
    if conf < conf_thresh:
        return False
    if frame_height > 0:
        m_h = FRAME_EDGE_MARGIN_FRAC * float(frame_height)
        if y < m_h or y > float(frame_height) - m_h:
            return False
    if frame_width > 0:
        m_w = FRAME_EDGE_MARGIN_FRAC * float(frame_width)
        if x < m_w or x > float(frame_width) - m_w:
            return False
    return True


def _lower_leg_keypoints_trusted(
    keypoints_row,
    frame_height: int,
    frame_width: int,
    conf_thresh: float = LEG_KEYPOINT_VISIBLE_CONF,
) -> bool:
    """True when at least one knee/ankle has sufficient confidence and is inside the frame."""
    for i in _LOWER_LEG_INDICES:
        if _keypoint_confident_and_in_frame(
            keypoints_row, i, frame_height, frame_width, conf_thresh
        ):
            return True
    return False


def _leg_confidence_snapshot(keypoints_row) -> dict:
    snap = {}
    for i, name in zip(_LOWER_LEG_INDICES, _LOWER_LEG_NAMES):
        if i < len(keypoints_row):
            snap[name] = round(float(keypoints_row[i][2]), 3)
    return snap


def bbox_clipped_at_frame_edge(
    xyxy,
    frame_height: int,
    frame_width: int,
    margin_frac: float = FRAME_EDGE_MARGIN_FRAC,
) -> tuple:
    """Return (top, bottom, left, right) booleans for bbox touching frame edges."""
    if frame_height <= 0 or frame_width <= 0:
        return False, False, False, False
    x1, y1, x2, y2 = map(float, xyxy[:4])
    m_h = margin_frac * float(frame_height)
    m_w = margin_frac * float(frame_width)
    return (
        y1 <= m_h,
        y2 >= float(frame_height) - m_h,
        x1 <= m_w,
        x2 >= float(frame_width) - m_w,
    )


def keypoints_mostly_full_body(
    keypoints_row,
    frame_height: int,
    frame_width: int,
    conf_thresh: float = KEYPOINT_VISIBLE_CONF,
    leg_conf_thresh: float = LEG_KEYPOINT_VISIBLE_CONF,
) -> bool:
    """True when upper body and at least one in-frame knee/ankle pass confidence gates."""
    has_upper = _any_keypoint_visible(
        keypoints_row, _HEAD_INDICES, conf_thresh
    ) or _any_keypoint_visible(keypoints_row, _SHOULDER_INDICES, conf_thresh)
    has_legs = _lower_leg_keypoints_trusted(
        keypoints_row, frame_height, frame_width, leg_conf_thresh
    )
    return has_upper and has_legs


def _keypoint_y_in_band(y: float, band_y0: float, band_y1: float) -> bool:
    return float(band_y0) <= y <= float(band_y1)


def _any_keypoint_in_vertical_band(
    keypoints_row,
    indices: tuple,
    band_y0: float,
    band_y1: float,
    conf_thresh: float = KEYPOINT_VISIBLE_CONF,
) -> bool:
    for i in indices:
        if i >= len(keypoints_row):
            continue
        x, y, conf = (
            float(keypoints_row[i][0]),
            float(keypoints_row[i][1]),
            float(keypoints_row[i][2]),
        )
        if conf >= conf_thresh and _keypoint_y_in_band(y, band_y0, band_y1):
            return True
    return False


def _lower_body_gate_stats(
    keypoints_row,
    band_y0: float,
    band_y1: float,
    conf_thresh: float = KEYPOINT_VISIBLE_CONF,
) -> dict:
    """How much of the lower body (hips/knees/ankles) sits in the vertical gate."""
    confident = []
    in_gate = []
    leg_in_gate = False
    for i, name in zip(_LOWER_BODY_INDICES, _LOWER_BODY_NAMES):
        if i >= len(keypoints_row):
            continue
        x, y, conf = (
            float(keypoints_row[i][0]),
            float(keypoints_row[i][1]),
            float(keypoints_row[i][2]),
        )
        if conf < conf_thresh:
            continue
        entry = {
            "name": name,
            "index": i,
            "x": round(x, 1),
            "y": round(y, 1),
            "conf": round(conf, 3),
            "in_band": _keypoint_y_in_band(y, band_y0, band_y1),
        }
        confident.append(entry)
        if entry["in_band"]:
            in_gate.append(entry)
            if i in _LOWER_LEG_INDICES:
                leg_in_gate = True
    n_conf = len(confident)
    n_in = len(in_gate)
    frac = (n_in / n_conf) if n_conf > 0 else 0.0
    # Most of the confident lower-body joints in the gate, and at least one
    # knee/ankle in the gate (hips alone are not enough — referee torsos).
    most_in_gate = n_conf > 0 and frac >= LOWER_BODY_IN_GATE_MIN_FRAC and leg_in_gate
    return {
        "ok": most_in_gate,
        "n_confident": n_conf,
        "n_in_gate": n_in,
        "frac_in_gate": round(frac, 3),
        "leg_in_gate": leg_in_gate,
        "confident": confident,
        "in_gate": in_gate,
    }


def body_segments_visible_in_gate(
    keypoints_row,
    band_y0: float,
    band_y1: float,
    conf_thresh: float = KEYPOINT_VISIBLE_CONF,
    leg_conf_thresh: float = KEYPOINT_VISIBLE_CONF,
) -> Tuple[bool, bool]:
    """Return (upper_in_gate, most_of_lower_in_gate) from ViTPose confidences.

    Lower body is forgiving: hips/knees/ankles with conf >= threshold; at least
    half of those confident joints must lie in the gate, including one knee/ankle.
    """
    del leg_conf_thresh  # lower-body gate uses conf_thresh for all lower joints
    has_upper = _any_keypoint_in_vertical_band(
        keypoints_row, _HEAD_INDICES, band_y0, band_y1, conf_thresh
    ) or _any_keypoint_in_vertical_band(
        keypoints_row, _SHOULDER_INDICES, band_y0, band_y1, conf_thresh
    )
    lower_stats = _lower_body_gate_stats(
        keypoints_row, band_y0, band_y1, conf_thresh
    )
    return has_upper, lower_stats["ok"]


def is_gate_fencer_candidate(
    xyxy,
    keypoints_row,
    band_y0: float,
    band_y1: float,
) -> bool:
    """Strip fencer: upper body + most of lower body ViTPose joints in the gate.

    Bbox area is not used — identity/filtering is keypoint-based.
    """
    del xyxy
    has_upper, has_lower = body_segments_visible_in_gate(
        keypoints_row, band_y0, band_y1
    )
    return has_upper and has_lower


def keypoint_center_in_gate(
    keypoints_row,
    band_y0: float,
    band_y1: float,
    conf_thresh: float = KEYPOINT_VISIBLE_CONF,
) -> Optional[Tuple[float, float]]:
    """Mean (x, y) of confident joints whose y lies in the vertical gate."""
    xs, ys = [], []
    for i in range(min(len(keypoints_row), 17)):
        x, y, conf = (
            float(keypoints_row[i][0]),
            float(keypoints_row[i][1]),
            float(keypoints_row[i][2]),
        )
        if conf >= conf_thresh and _keypoint_y_in_band(y, band_y0, band_y1):
            xs.append(x)
            ys.append(y)
    if not xs:
        return None
    return sum(xs) / len(xs), sum(ys) / len(ys)


def gate_keypoint_support(
    keypoints_row,
    band_y0: float,
    band_y1: float,
) -> int:
    """Count confident joints inside the gate (for ranking, not bbox area)."""
    n = 0
    for i in range(min(len(keypoints_row), 17)):
        conf = float(keypoints_row[i][2])
        y = float(keypoints_row[i][1])
        if conf >= KEYPOINT_VISIBLE_CONF and _keypoint_y_in_band(y, band_y0, band_y1):
            n += 1
    return n


def _best_keypoint_in_band(
    keypoints_row,
    indices: tuple,
    band_y0: float,
    band_y1: float,
) -> Optional[dict]:
    """Highest-confidence joint among indices whose y lies in the vertical gate."""
    best = None
    for i in indices:
        if i >= len(keypoints_row):
            continue
        x, y, conf = (
            float(keypoints_row[i][0]),
            float(keypoints_row[i][1]),
            float(keypoints_row[i][2]),
        )
        in_band = _keypoint_y_in_band(y, band_y0, band_y1)
        entry = {
            "index": i,
            "x": round(x, 1),
            "y": round(y, 1),
            "conf": round(conf, 3),
            "in_band": in_band,
        }
        if best is None or conf > best["conf"]:
            best = entry
    return best


def explain_gate_visibility(
    xyxy,
    keypoints_row,
    band_y0: float,
    band_y1: float,
) -> dict:
    """Audit of upper/lower ViTPose visibility inside the vertical gate."""
    x1, y1, x2, y2 = map(float, xyxy[:4])
    inband = bbox_area_in_vertical_band_xyxy(xyxy, band_y0, band_y1)
    full_area = _bbox_area_xyxy(xyxy)
    has_upper = _any_keypoint_in_vertical_band(
        keypoints_row, _HEAD_INDICES, band_y0, band_y1, KEYPOINT_VISIBLE_CONF
    ) or _any_keypoint_in_vertical_band(
        keypoints_row, _SHOULDER_INDICES, band_y0, band_y1, KEYPOINT_VISIBLE_CONF
    )
    lower_stats = _lower_body_gate_stats(
        keypoints_row, band_y0, band_y1, KEYPOINT_VISIBLE_CONF
    )
    has_lower = lower_stats["ok"]
    best_upper = _best_keypoint_in_band(
        keypoints_row, _HEAD_INDICES + _SHOULDER_INDICES, band_y0, band_y1
    )
    best_lower = _best_keypoint_in_band(
        keypoints_row, _LOWER_BODY_INDICES, band_y0, band_y1
    )
    leg_confs = _leg_confidence_snapshot(keypoints_row)
    gate_ok = has_upper and has_lower
    kp_center = keypoint_center_in_gate(keypoints_row, band_y0, band_y1)
    kp_support = gate_keypoint_support(keypoints_row, band_y0, band_y1)
    reasons = []
    if not has_upper:
        reasons.append("missing_upper_in_gate")
    if not has_lower:
        if lower_stats["n_confident"] == 0:
            reasons.append("no_confident_lower_body")
        elif not lower_stats["leg_in_gate"]:
            reasons.append("no_knee_or_ankle_in_gate")
        elif lower_stats["frac_in_gate"] < LOWER_BODY_IN_GATE_MIN_FRAC:
            reasons.append("most_lower_body_outside_gate")
        else:
            reasons.append("missing_lower_in_gate")
    return {
        "gate_ok": gate_ok,
        "reasons": reasons,
        "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        "inband_area": round(inband, 1),
        "full_area": round(full_area, 1),
        "inband_frac": round(inband / max(full_area, 1e-6), 3),
        "upper_in_gate": has_upper,
        "lower_in_gate": has_lower,
        "lower_n_confident": lower_stats["n_confident"],
        "lower_n_in_gate": lower_stats["n_in_gate"],
        "lower_frac_in_gate": lower_stats["frac_in_gate"],
        "leg_in_gate": lower_stats["leg_in_gate"],
        "lower_joints": lower_stats["confident"],
        "kp_support": kp_support,
        "kp_center": (
            [round(kp_center[0], 1), round(kp_center[1], 1)] if kp_center else None
        ),
        "best_upper": best_upper,
        "best_lower": best_lower,
        "leg_confs": leg_confs,
    }


def build_gate_debug_payload(
    structured: dict,
    band_y0: float,
    band_y1: float,
) -> List[dict]:
    boxes = structured.get("bboxes") or []
    keypoints = structured.get("keypoints") or []
    out = []
    for i, box in enumerate(boxes):
        row = keypoints[i] if i < len(keypoints) else []
        audit = explain_gate_visibility(box, row, band_y0, band_y1)
        audit["index"] = i
        out.append(audit)
    return out


def explain_body_visibility(
    xyxy,
    keypoints_row,
    frame_height: int,
    frame_width: int,
) -> dict:
    """Human-readable audit of why a detection passed or failed the full-body gate."""
    x1, y1, x2, y2 = map(float, xyxy[:4])
    bh = max(0.0, y2 - y1)
    bw = max(0.0, x2 - x1)
    top_clip, bottom_clip, left_clip, right_clip = bbox_clipped_at_frame_edge(
        xyxy, frame_height, frame_width
    )
    has_upper = _any_keypoint_visible(
        keypoints_row, _HEAD_INDICES, KEYPOINT_VISIBLE_CONF
    ) or _any_keypoint_visible(keypoints_row, _SHOULDER_INDICES, KEYPOINT_VISIBLE_CONF)
    aspect = bw / max(bh, 1e-6)
    has_head = _any_keypoint_visible(keypoints_row, _HEAD_INDICES, KEYPOINT_VISIBLE_CONF)
    leg_conf_thresh = (
        LEG_KEYPOINT_CLIPPED_CONF if bottom_clip else LEG_KEYPOINT_VISIBLE_CONF
    )
    has_legs = _lower_leg_keypoints_trusted(
        keypoints_row, frame_height, frame_width, leg_conf_thresh
    )
    leg_confs = _leg_confidence_snapshot(keypoints_row)
    height_frac = bh / float(frame_height) if frame_height > 0 else 0.0
    partial_geom = bbox_looks_like_partial_foreground(
        xyxy, frame_height, frame_width
    )
    ok = is_mostly_full_body_visible(
        xyxy, keypoints_row, frame_height, frame_width
    )
    reasons = []
    if partial_geom:
        width_frac = bw / float(frame_width) if frame_width > 0 else 0.0
        height_frac_box = bh / float(frame_height) if frame_height > 0 else 0.0
        if bottom_clip and (y1 / float(frame_height) > PARTIAL_BODY_MAX_TOP_FRAC):
            reasons.append("bbox_bottom_partial_too_low")
        if bottom_clip and (left_clip or right_clip):
            reasons.append("bbox_corner_partial")
        if bottom_clip and width_frac > 0.22:
            reasons.append("bbox_wide_bottom_partial")
        if bottom_clip and height_frac_box > 0.50 and width_frac > 0.15:
            reasons.append("bbox_fills_frame_partial")
        if aspect > PARTIAL_BODY_MAX_ASPECT_WH and (
            bottom_clip or left_clip or right_clip
        ):
            reasons.append("bbox_wide_edge_partial")
    if not has_upper:
        reasons.append("missing_upper_body")
    if not has_legs:
        reasons.append("missing_trusted_knee_or_ankle")
        if leg_confs and max(leg_confs.values(), default=0.0) < leg_conf_thresh:
            reasons.append("leg_conf_below_threshold")
        elif leg_confs:
            reasons.append("leg_keypoints_outside_frame_or_low_conf")
    if frame_height > 0 and bh < MIN_STRIP_FENCER_HEIGHT_FRAC * frame_height:
        reasons.append("bbox_too_short")
    if top_clip and not has_head:
        reasons.append("top_clipped_no_head")
    if bottom_clip and not has_legs:
        reasons.append("bottom_clipped_no_trusted_legs")
    return {
        "full_body": ok,
        "reasons": reasons,
        "bbox_xyxy": [x1, y1, x2, y2],
        "bbox_h_frac": round(height_frac, 3),
        "aspect_wh": round(aspect, 3),
        "leg_conf_thresh": leg_conf_thresh,
        "leg_confs": leg_confs,
        "clips": {
            "top": top_clip,
            "bottom": bottom_clip,
            "left": left_clip,
            "right": right_clip,
        },
        "kp": {
            "upper": has_upper,
            "head": has_head,
            "legs": has_legs,
        },
    }


def bbox_looks_like_partial_foreground(
    xyxy,
    frame_height: int,
    frame_width: int,
) -> bool:
    """True for frame-edge partial bodies (referee / close-up spectator), bbox only.

    Pose invents high-conf legs on these. Strip fencers are usually fully inside the
    frame or only lightly touch the bottom without also hugging a side edge.
    """
    if frame_height <= 0 or frame_width <= 0:
        return False
    x1, y1, x2, y2 = map(float, xyxy[:4])
    bh = max(0.0, y2 - y1)
    bw = max(0.0, x2 - x1)
    if bh <= 0.0:
        return True
    top_clip, bottom_clip, left_clip, right_clip = bbox_clipped_at_frame_edge(
        xyxy, frame_height, frame_width
    )
    aspect = bw / bh
    top_frac = y1 / float(frame_height)
    width_frac = bw / float(frame_width)
    height_frac = bh / float(frame_height)
    # Classic mid-frame torso sitting on the bottom edge.
    if bottom_clip and top_frac > PARTIAL_BODY_MAX_TOP_FRAC:
        return True
    # Close-up person cut off at bottom AND a side (fills a corner of the frame).
    # This catches foreground officials whose head is high (top_frac small) so the
    # rule above does not fire.
    if bottom_clip and (left_clip or right_clip):
        return True
    # Bottom-clipped and very wide in-frame — foreground body, not a strip fencer.
    if bottom_clip and width_frac > 0.22:
        return True
    # Tall bottom-clipped body that also dominates width (fills the frame).
    if bottom_clip and height_frac > 0.50 and width_frac > 0.15:
        return True
    # Wide body pressed against a frame edge is a cropped torso.
    if aspect > PARTIAL_BODY_MAX_ASPECT_WH and (bottom_clip or left_clip or right_clip):
        return True
    return False


def is_mostly_full_body_visible(
    xyxy,
    keypoints_row,
    frame_height: int,
    frame_width: int,
    conf_thresh: float = KEYPOINT_VISIBLE_CONF,
) -> bool:
    """Reject partial bodies for auto-select (bbox geometry + pose confidences).

    Bbox geometry is applied first: pose often hallucinates legs on a referee torso
    at the strip front, so keypoint confidence alone is not enough.
    """
    if bbox_looks_like_partial_foreground(xyxy, frame_height, frame_width):
        return False

    _, bottom_clip, _, _ = bbox_clipped_at_frame_edge(
        xyxy, frame_height, frame_width
    )
    leg_conf_thresh = (
        LEG_KEYPOINT_CLIPPED_CONF if bottom_clip else LEG_KEYPOINT_VISIBLE_CONF
    )
    if not keypoints_mostly_full_body(
        keypoints_row,
        frame_height,
        frame_width,
        conf_thresh,
        leg_conf_thresh,
    ):
        return False

    bh = max(0.0, float(xyxy[3]) - float(xyxy[1]))
    if frame_height > 0 and bh < MIN_STRIP_FENCER_HEIGHT_FRAC * frame_height:
        return False

    top_clip, _, _, _ = bbox_clipped_at_frame_edge(
        xyxy, frame_height, frame_width
    )
    if top_clip and not _any_keypoint_visible(
        keypoints_row, _HEAD_INDICES, conf_thresh
    ):
        return False
    return True


def _bbox_area_xyxy(xyxy) -> float:
    x1, y1, x2, y2 = map(float, xyxy[:4])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def select_full_body_instances(
    structured: dict,
    frame_height: int,
    frame_width: int,
    top_k: int,
    vertical_y0: Optional[float] = None,
    vertical_y1: Optional[float] = None,
    order_left_to_right: bool = False,
    fallback_if_none: bool = True,
) -> dict:
    """Drop partial bodies, then keep top_k instances ranked by bbox (or in-band) area."""
    bboxes = structured.get("bboxes") or []
    keypoints = structured.get("keypoints") or []
    scores = structured.get("bbox_scores") or []
    if not bboxes:
        return structured

    def rank_area(i: int) -> float:
        if vertical_y0 is not None and vertical_y1 is not None:
            return bbox_area_in_vertical_band_xyxy(
                bboxes[i], float(vertical_y0), float(vertical_y1)
            )
        return _bbox_area_xyxy(bboxes[i])

    full_body_idxs = [
        i
        for i in range(len(bboxes))
        if i < len(keypoints)
        and is_mostly_full_body_visible(
            bboxes[i], keypoints[i], frame_height, frame_width
        )
    ]
    if full_body_idxs:
        ranked = sorted(full_body_idxs, key=rank_area, reverse=True)
    elif fallback_if_none:
        ranked = sorted(range(len(bboxes)), key=rank_area, reverse=True)
    else:
        ranked = []

    ranked = ranked[:top_k]
    if order_left_to_right and len(ranked) > 1:
        ranked = sorted(
            ranked,
            key=lambda i: (bboxes[i][0] + bboxes[i][2]) / 2.0,
        )
    ranked_set = set(ranked)
    dropped = [
        {"index": i, "full_body": i in full_body_idxs}
        for i in range(len(bboxes))
        if i not in ranked_set
    ]
    return {
        "bboxes": [bboxes[i] for i in ranked],
        "keypoints": [keypoints[i] for i in ranked],
        "bbox_scores": [scores[i] for i in ranked] if scores else [],
        "_pose_filter_audit": {
            "mode": "full_body",
            "n_before": len(bboxes),
            "n_after": len(ranked),
            "kept_idxs": list(ranked),
            "full_body_idxs": full_body_idxs,
            "used_fallback": not bool(full_body_idxs) and fallback_if_none,
            "dropped": dropped,
        },
    }


def select_gate_fencer_instances(
    structured: dict,
    frame_height: int,
    frame_width: int,
    top_k: int,
    vertical_y0: float,
    vertical_y1: float,
    order_left_to_right: bool = False,
    fallback_if_none: bool = True,
) -> dict:
    """Pipeline pool: keep people with upper+lower ViTPose joints inside the gate.

    Ranked by in-gate keypoint support (not bbox area). Bboxes are only carriers
    for the pose instances.
    """
    del frame_height, frame_width
    bboxes = structured.get("bboxes") or []
    keypoints = structured.get("keypoints") or []
    scores = structured.get("bbox_scores") or []
    if not bboxes:
        return structured

    y0, y1 = float(vertical_y0), float(vertical_y1)
    gate_audits = build_gate_debug_payload(structured, y0, y1)
    gate_idxs = [g["index"] for g in gate_audits if g["gate_ok"]]

    def rank_kp(i: int) -> int:
        return gate_keypoint_support(keypoints[i], y0, y1) if i < len(keypoints) else 0

    used_fallback = False
    if gate_idxs:
        ranked = sorted(gate_idxs, key=rank_kp, reverse=True)
    elif fallback_if_none:
        used_fallback = True
        ranked = sorted(range(len(bboxes)), key=rank_kp, reverse=True)
    else:
        ranked = []

    ranked = ranked[:top_k]
    if order_left_to_right and len(ranked) > 1:
        def kp_cx(i: int) -> float:
            c = keypoint_center_in_gate(keypoints[i], y0, y1) if i < len(keypoints) else None
            if c is not None:
                return c[0]
            return (float(bboxes[i][0]) + float(bboxes[i][2])) / 2.0

        ranked = sorted(ranked, key=kp_cx)

    ranked_set = set(ranked)
    dropped = []
    for g in gate_audits:
        if g["index"] not in ranked_set:
            dropped.append(
                {
                    "index": g["index"],
                    "gate_ok": g["gate_ok"],
                    "reasons": g["reasons"],
                    "kp_support": g["kp_support"],
                    "kp_center": g["kp_center"],
                    "bbox_xyxy": g["bbox_xyxy"],
                }
            )
    return {
        "bboxes": [bboxes[i] for i in ranked],
        "keypoints": [keypoints[i] for i in ranked],
        "bbox_scores": [scores[i] for i in ranked] if scores else [],
        "_pose_filter_audit": {
            "mode": "gate_keypoints",
            "n_before": len(bboxes),
            "n_after": len(ranked),
            "kept_idxs": list(ranked),
            "gate_idxs": gate_idxs,
            "used_fallback": used_fallback,
            "dropped": dropped,
            "gate_dets_before": gate_audits,
        },
    }


def filter_topk_by_inband_area_xyxy(
    bboxes: np.ndarray,
    k: int,
    band_y0: float,
    band_y1: float,
    order_left_to_right: bool,
) -> np.ndarray:
    """Keep k detections with largest area inside [band_y0, band_y1]; optional L→R order."""
    if bboxes.size == 0 or k <= 0:
        return bboxes
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    w = np.maximum(0.0, x2 - x1)
    iy0 = np.maximum(y1, band_y0)
    iy1 = np.minimum(y2, band_y1)
    h = np.maximum(0.0, iy1 - iy0)
    areas = w * h
    order = np.argsort(-areas, kind="stable")
    order = order[: min(k, len(order))]
    sel = bboxes[order].copy()
    if order_left_to_right and len(sel) > 1:
        cx = (sel[:, 0] + sel[:, 2]) / 2.0
        sel = sel[np.argsort(cx, kind="stable")]
    return sel


def filter_dets_vertical_third_xyxy(dets: np.ndarray, y0: float, y1: float) -> np.ndarray:
    """Keep dets whose center-y is in [y0,y1] and >70% of bbox height overlaps the band.

    If <2 qualify, return dets unchanged.
    """
    if dets.size == 0:
        return dets
    t = dets[:, 1]
    b = dets[:, 3]
    cy = (t + b) / 2.0
    bh = np.maximum(b - t, 1e-6)
    overlap = np.maximum(0.0, np.minimum(b, y1) - np.maximum(t, y0))
    ok = (cy >= y0) & (cy <= y1) & (overlap / bh > VERTICAL_BAND_MIN_HEIGHT_OVERLAP)
    out = dets[ok]
    if out.size == 0 or out.shape[0] < 2:
        return dets
    return out


def indices_vertical_third_boxes(
    boxes: List, y0: Optional[float], y1: Optional[float]
) -> List[int]:
    """Indices passing vertical band on pose bboxes; if <2 qualify, all indices (same as dataExtractor)."""
    if not boxes:
        return []
    if y0 is None or y1 is None:
        return list(range(len(boxes)))

    def bbox_in_band(xyxy) -> bool:
        t = float(xyxy[1])
        b = float(xyxy[3])
        cy = (t + b) / 2.0
        if cy < y0 or cy > y1:
            return False
        bh = max(b - t, 1e-6)
        overlap = max(0.0, min(b, y1) - max(t, y0))
        return overlap / bh > VERTICAL_BAND_MIN_HEIGHT_OVERLAP

    idxs = [i for i, b in enumerate(boxes) if bbox_in_band(b)]
    if len(idxs) < 2:
        return list(range(len(boxes)))
    return idxs


def vertical_ref_from_fencer_boxes(
    box1: Optional[tuple],
    box2: Optional[tuple],
    frame_height: int,
    min_span_frac: float = 1.0 / 3.0,
) -> Tuple[Optional[float], Optional[float]]:
    """Band [y0,y1] from two fencer xyxy boxes (dataExtractor _set_vertical_ref_from_selection)."""
    if not box1 or not box2 or frame_height <= 0:
        return None, None
    cy1 = (float(box1[1]) + float(box1[3])) / 2.0
    cy2 = (float(box2[1]) + float(box2[3])) / 2.0
    lo = min(cy1, cy2)
    hi = max(cy1, cy2)
    mid = (lo + hi) / 2.0
    min_span = min_span_frac * float(frame_height)
    total_span = max(hi - lo, min_span)
    y0 = mid - total_span / 2.0
    y1 = mid + total_span / 2.0
    y0 = max(0.0, y0)
    y1 = min(float(frame_height), y1)
    return y0, y1


def _bbox_center_xyxy(xyxy) -> Tuple[float, float]:
    x1, y1, x2, y2 = map(float, xyxy[:4])
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _bbox_iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = map(float, a[:4])
    bx1, by1, bx2, by2 = map(float, b[:4])
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _bbox_wh_area(xyxy) -> Tuple[float, float, float]:
    x1, y1, x2, y2 = map(float, xyxy[:4])
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return w, h, w * h


def _size_ok_vs_reference(det_box, ref_box) -> Tuple[bool, float, float, float]:
    """True when det is not a tiny ghost vs the UI selection box.

    Returns (ok, height_frac, width_frac, area_frac).
    """
    dw, dh, da = _bbox_wh_area(det_box)
    rw, rh, ra = _bbox_wh_area(ref_box)
    h_frac = dh / max(rh, 1.0)
    w_frac = dw / max(rw, 1.0)
    a_frac = da / max(ra, 1.0)
    ok = h_frac >= REF_SIZE_MIN_HEIGHT_FRAC and a_frac >= REF_SIZE_MIN_AREA_FRAC
    return ok, h_frac, w_frac, a_frac


def _score_detection_vs_reference(
    det_box,
    keypoints_row,
    ref_box,
    frame_height: int,
    frame_width: int,
    vertical_y0: Optional[float] = None,
    vertical_y1: Optional[float] = None,
    *,
    max_dist_px: Optional[float] = None,
    require_left_of_x: Optional[float] = None,
    require_right_of_x: Optional[float] = None,
) -> dict:
    """Score one detection against a UI reference using in-gate ViTPose joints.

    Eligible only when upper+lower keypoints lie in the vertical gate *and* the
    bbox is a plausible size vs the UI selection (blocks tiny gate-ok ghosts).
    Among those, distance still matters, but area vs the UI ref also counts so a
    full-size opponent who moved inward beats a smaller bystander near the box.
    """
    ref_cx, ref_cy = _bbox_center_xyxy(ref_box)
    diag = math.hypot(float(frame_width or 1), float(frame_height or 1))
    gate_ok = False
    kp_center = None
    kp_support = 0
    upper_in_gate = False
    lower_in_gate = False
    if vertical_y0 is not None and vertical_y1 is not None and keypoints_row:
        y0, y1 = float(vertical_y0), float(vertical_y1)
        upper_in_gate, lower_in_gate = body_segments_visible_in_gate(
            keypoints_row, y0, y1
        )
        gate_ok = upper_in_gate and lower_in_gate
        kp_center = keypoint_center_in_gate(keypoints_row, y0, y1)
        kp_support = gate_keypoint_support(keypoints_row, y0, y1)

    size_ok, h_frac, w_frac, a_frac = _size_ok_vs_reference(det_box, ref_box)

    if kp_center is not None:
        cx, cy = kp_center
    else:
        cx, cy = _bbox_center_xyxy(det_box)
    dist = math.hypot(cx - ref_cx, cy - ref_cy)
    iou = _bbox_iou_xyxy(det_box, ref_box)

    side_ok = True
    if require_left_of_x is not None and cx >= float(require_left_of_x):
        side_ok = False
    if require_right_of_x is not None and cx <= float(require_right_of_x):
        side_ok = False
    near_ok = True
    if max_dist_px is not None and dist > float(max_dist_px):
        near_ok = False

    # Match needs gate + size + correct half-strip + not too far from the UI box.
    eligible = bool(gate_ok and size_ok and side_ok and near_ok)

    if not eligible:
        score = -1e6 + iou - (dist / max(diag, 1.0)) + min(a_frac, 1.0) * 0.1
    else:
        # Near the setup box still helps; area vs that box lets a larger inward-
        # moved fencer outscore a skinny sideline person glued to the ref.
        score = (
            -(dist / max(diag, 1.0)) * 5.0
            + iou * 2.0
            + (kp_support / 17.0) * 0.35
            + min(h_frac, 1.2) * 0.45
            + min(a_frac, 1.5) * 1.1
        )
    return {
        "score": round(score, 4),
        "gate_ok": gate_ok,
        "size_ok": size_ok,
        "side_ok": side_ok,
        "near_ok": near_ok,
        "eligible": eligible,
        "height_frac": round(h_frac, 3),
        "width_frac": round(w_frac, 3),
        "area_frac": round(a_frac, 3),
        "upper_in_gate": upper_in_gate,
        "lower_in_gate": lower_in_gate,
        "kp_support": kp_support,
        "kp_center": [round(cx, 1), round(cy, 1)],
        "dist_px": round(dist, 1),
        "iou": round(iou, 4),
        "det_xyxy": [round(float(v), 1) for v in det_box[:4]],
    }


def _peer_cull_eligible_by_area(scores: List[dict], boxes: List) -> None:
    """Among eligible same-role candidates, drop ones much smaller than the largest."""
    elig = [s for s in scores if s.get("eligible")]
    if len(elig) < 2:
        for s in scores:
            s["peer_size_ok"] = bool(s.get("eligible"))
        return
    max_a = max(_bbox_wh_area(boxes[s["index"]])[2] for s in elig)
    for s in scores:
        if not s.get("eligible"):
            s["peer_size_ok"] = False
            continue
        a = _bbox_wh_area(boxes[s["index"]])[2]
        if a < PEER_AREA_MIN_FRAC * max(max_a, 1.0):
            s["eligible"] = False
            s["peer_size_ok"] = False
        else:
            s["peer_size_ok"] = True


def _match_index_to_reference_box(
    ref_box,
    candidate_idxs: List[int],
    boxes: List,
    keypoints: List,
    frame_height: int,
    frame_width: int,
    exclude: Optional[int] = None,
    vertical_y0: Optional[float] = None,
    vertical_y1: Optional[float] = None,
    score_details_out: Optional[List[dict]] = None,
) -> Optional[int]:
    """Pick detection closest to a UI reference box (IoU, distance, in-band area)."""
    if not ref_box or not candidate_idxs:
        return None
    best_idx = None
    best_score = -1e18
    for i in candidate_idxs:
        if i == exclude:
            continue
        row = keypoints[i] if i < len(keypoints) else []
        detail = _score_detection_vs_reference(
            boxes[i],
            row,
            ref_box,
            frame_height,
            frame_width,
            vertical_y0=vertical_y0,
            vertical_y1=vertical_y1,
        )
        detail["index"] = i
        detail["excluded"] = False
        if score_details_out is not None:
            score_details_out.append(detail)
        if detail["score"] > best_score:
            best_score = detail["score"]
            best_idx = i
    return best_idx


def build_detection_debug_payload(
    structured: dict, frame_height: int, frame_width: int
) -> List[dict]:
    """Per-detection visibility audit for API logs / UI troubleshooting."""
    boxes = structured.get("bboxes") or []
    keypoints = structured.get("keypoints") or []
    out = []
    for i, box in enumerate(boxes):
        row = keypoints[i] if i < len(keypoints) else []
        audit = explain_body_visibility(box, row, frame_height, frame_width)
        audit["index"] = i
        out.append(audit)
    return out


def suggest_auto_fencer_pair(
    structured: dict,
    frame_height: int,
    frame_width: int,
    min_score: float = 0.30,
) -> dict:
    """Pick left/right strip fencers for UI auto-select (ViTPose full_body primary).

    Returns {success, fencer1_index, fencer2_index, confidence, reason, candidates}.
    fencer1 = left, fencer2 = right (screen order).
    """
    boxes = structured.get("bboxes") or []
    keypoints = structured.get("keypoints") or []
    fh = int(frame_height or 0)
    fw = int(frame_width or 0)
    empty = {
        "success": False,
        "fencer1_index": None,
        "fencer2_index": None,
        "confidence": 0.0,
        "reason": "no_detections",
        "candidates": [],
    }
    if len(boxes) < 2 or fh <= 0 or fw <= 0:
        empty["reason"] = "too_few_detections"
        return empty

    candidates = []
    for i, box in enumerate(boxes):
        row = keypoints[i] if i < len(keypoints) else []
        audit = explain_body_visibility(box, row, fh, fw)
        if not audit.get("full_body"):
            continue
        # Geometry veto only for classic bottom-of-frame partials (referee).
        if bbox_looks_like_partial_foreground(box, fh, fw):
            continue
        x1, y1, x2, y2 = map(float, box[:4])
        candidates.append(
            {
                "index": i,
                "box": [x1, y1, x2, y2],
                "area": max(0.0, x2 - x1) * max(0.0, y2 - y1),
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
                "height": max(0.0, y2 - y1),
                "h_frac": audit.get("bbox_h_frac"),
                "reasons": audit.get("reasons") or [],
            }
        )

    cand_summary = [
        {
            "index": c["index"],
            "h_frac": c["h_frac"],
            "cx": round(c["cx"], 1),
            "cy": round(c["cy"], 1),
        }
        for c in candidates
    ]
    if len(candidates) < 2:
        return {
            "success": False,
            "fencer1_index": None,
            "fencer2_index": None,
            "confidence": 0.0,
            "reason": f"full_body_lt2({len(candidates)})",
            "candidates": cand_summary,
        }

    max_area = max(c["area"] for c in candidates) or 1.0
    best = None
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            left, right = (a, b) if a["cx"] <= b["cx"] else (b, a)
            vert_overlap = max(
                0.0, min(a["box"][3], b["box"][3]) - max(a["box"][1], b["box"][1])
            )
            min_h = max(1.0, min(a["height"], b["height"]))
            overlap_ratio = vert_overlap / min_h
            center_y_offset = abs(a["cy"] - b["cy"])
            avg_h = max(1.0, (a["height"] + b["height"]) / 2.0)
            alignment = max(0.0, 1.0 - center_y_offset / avg_h)
            height_balance = min(a["height"], b["height"]) / max(
                a["height"], b["height"]
            )
            area_score = (a["area"] + b["area"]) / (2.0 * max_area)
            sep = abs(a["cx"] - b["cx"]) / float(fw)
            # Prefer on-strip separation (not two people on top of each other,
            # and not one in the far stands vs one on strip).
            if sep < 0.06:
                separation_score = sep / 0.06 * 0.3
            elif sep > 0.75:
                separation_score = max(0.0, 1.0 - (sep - 0.75) / 0.25)
            else:
                separation_score = min(1.0, sep * 2.0)
            # Prefer bodies whose centers sit in the middle/lower frame (strip).
            strip_cy = 1.0 - abs(((a["cy"] + b["cy"]) / 2.0) / fh - 0.55) / 0.55
            strip_cy = max(0.0, min(1.0, strip_cy))
            score = (
                0.22 * area_score
                + 0.26 * overlap_ratio
                + 0.18 * alignment
                + 0.12 * height_balance
                + 0.12 * separation_score
                + 0.10 * strip_cy
            )
            if best is None or score > best["score"]:
                best = {
                    "left": left,
                    "right": right,
                    "score": score,
                }

    if best is None:
        return {
            "success": False,
            "fencer1_index": None,
            "fencer2_index": None,
            "confidence": 0.0,
            "reason": "no_pair",
            "candidates": cand_summary,
        }

    conf = max(0.0, min(1.0, best["score"]))
    ok = conf >= min_score
    return {
        "success": ok,
        "fencer1_index": best["left"]["index"] if ok else None,
        "fencer2_index": best["right"]["index"] if ok else None,
        "fencer1_box": best["left"]["box"] if ok else None,
        "fencer2_box": best["right"]["box"] if ok else None,
        "confidence": conf,
        "reason": "ok" if ok else f"score_below_min({conf:.3f}<{min_score})",
        "candidates": cand_summary,
    }


def get_fencer_pair_indices(
    structured: dict,
    fencer1_side: str,
    vertical_y0: Optional[float],
    vertical_y1: Optional[float],
    frame_height: Optional[int] = None,
    frame_width: Optional[int] = None,
    fencer1_ref_box: Optional[tuple] = None,
    fencer2_ref_box: Optional[tuple] = None,
    debug_log: Optional[Callable[[str], None]] = None,
    audit_out: Optional[dict] = None,
) -> Tuple[Optional[int], Optional[int]]:
    """Map UI fencer1/fencer2 from in-gate full bodies.

    Filter: upper + lower keypoints inside the vertical gate (full body showing).
    Identity: prefer matching each UI setup box (size + proximity vs ref), else
    fall back to the two largest in-gate bodies by bbox area (left/right = F1/F2).
    """
    audit: dict = {
        "method": None,
        "f1_idx": None,
        "f2_idx": None,
        "ok": False,
        "fail_reason": None,
        "n_dets": 0,
        "gate_idxs": [],
        "gate_dets": [],
        "pose_filter": structured.get("_pose_filter_audit"),
        "ref_match": {},
        "kp_rank": [],
    }

    def _emit(msg: str) -> None:
        if debug_log:
            debug_log(msg)

    def _finish(
        f1: Optional[int], f2: Optional[int], method: Optional[str], fail: Optional[str]
    ) -> Tuple[Optional[int], Optional[int]]:
        audit["f1_idx"] = f1
        audit["f2_idx"] = f2
        audit["method"] = method
        audit["fail_reason"] = fail
        audit["ok"] = f1 is not None and f2 is not None and f1 != f2
        if audit_out is not None:
            audit_out.clear()
            audit_out.update(audit)
        if audit["ok"]:
            _emit(
                f"TRACK pair ok method={method} f1_idx={f1} f2_idx={f2} "
                f"n_dets={audit['n_dets']} gate_idxs={audit['gate_idxs']}"
            )
        else:
            _emit(
                f"TRACK pair FAIL reason={fail} method={method} "
                f"f1_idx={f1} f2_idx={f2} n_dets={audit['n_dets']} "
                f"gate_idxs={audit['gate_idxs']}"
            )
        return f1, f2

    if not structured or not structured.get("bboxes"):
        return _finish(None, None, None, "no_bboxes")
    boxes = structured["bboxes"]
    keypoints = structured.get("keypoints") or []
    audit["n_dets"] = len(boxes)
    if len(boxes) < 2:
        return _finish(None, None, None, f"too_few_dets({len(boxes)})")

    fh = int(frame_height or 0)
    fw = int(frame_width or 0)
    audit["frame_hw"] = [fh, fw]
    audit["vertical_gate"] = [
        None if vertical_y0 is None else float(vertical_y0),
        None if vertical_y1 is None else float(vertical_y1),
    ]
    audit["fencer1_side"] = fencer1_side

    if vertical_y0 is None or vertical_y1 is None:
        _emit("TRACK warning: no vertical gate set")
        return _finish(None, None, None, "no_vertical_gate")

    y0, y1 = float(vertical_y0), float(vertical_y1)
    audit["gate_dets"] = build_gate_debug_payload(structured, y0, y1)
    for g in audit["gate_dets"]:
        _emit(
            f"TRACK det[{g['index']}] gate_ok={g['gate_ok']} "
            f"upper={g['upper_in_gate']} lower={g['lower_in_gate']} "
            f"lower_in_gate={g['lower_n_in_gate']}/{g['lower_n_confident']} "
            f"(frac={g['lower_frac_in_gate']}, leg={g['leg_in_gate']}) "
            f"kp_support={g['kp_support']} kp_center={g['kp_center']} "
            f"reasons={g['reasons']} bbox={g['bbox_xyxy']} "
            f"lower_joints={g['lower_joints']}"
        )

    if structured.get("_pose_filter_audit"):
        pf = structured["_pose_filter_audit"]
        _emit(
            f"TRACK pose_filter mode={pf.get('mode')} "
            f"before={pf.get('n_before')} after={pf.get('n_after')} "
            f"kept={pf.get('kept_idxs')} dropped={pf.get('dropped')}"
        )

    # Full body in vertical band (upper + lower joints).
    gate_idxs = [
        i
        for i in range(len(boxes))
        if i < len(keypoints)
        and is_gate_fencer_candidate(boxes[i], keypoints[i], y0, y1)
    ]
    audit["gate_idxs"] = list(gate_idxs)
    _emit(f"TRACK gate_keypoint filter idxs={gate_idxs} (of {len(boxes)} dets)")
    if len(gate_idxs) < 2:
        return _finish(None, None, None, f"gate_kp_lt2({gate_idxs})")

    # Preferred identity: match each UI setup box (gate + size vs ref + near box).
    # Falls back to two-largest-in-gate when refs are missing.
    if fencer1_ref_box is not None and fencer2_ref_box is not None and fh > 0 and fw > 0:
        ref1 = fencer1_ref_box
        ref2 = fencer2_ref_box
        r1cx, _ = _bbox_center_xyxy(ref1)
        r2cx, _ = _bbox_center_xyxy(ref2)
        ref_sep = abs(r1cx - r2cx)
        max_dist = max(REF_MATCH_MAX_DIST_MIN_PX, REF_MATCH_MAX_DIST_FRAC * ref_sep)

        f1_details: List[dict] = []
        f2_details: List[dict] = []
        for i in gate_idxs:
            row = keypoints[i] if i < len(keypoints) else []
            d1 = _score_detection_vs_reference(
                boxes[i],
                row,
                ref1,
                fh,
                fw,
                vertical_y0=y0,
                vertical_y1=y1,
                max_dist_px=max_dist,
                require_left_of_x=(r2cx if r1cx < r2cx else None),
                require_right_of_x=(r2cx if r1cx > r2cx else None),
            )
            d1["index"] = i
            f1_details.append(d1)
            d2 = _score_detection_vs_reference(
                boxes[i],
                row,
                ref2,
                fh,
                fw,
                vertical_y0=y0,
                vertical_y1=y1,
                max_dist_px=max_dist,
                require_left_of_x=(r1cx if r2cx < r1cx else None),
                require_right_of_x=(r1cx if r2cx > r1cx else None),
            )
            d2["index"] = i
            f2_details.append(d2)

        _peer_cull_eligible_by_area(f1_details, boxes)
        _peer_cull_eligible_by_area(f2_details, boxes)
        audit["ref_match"] = {
            "max_dist_px": round(max_dist, 1),
            "f1": f1_details,
            "f2": f2_details,
        }

        def _best(details: List[dict], exclude: Optional[int] = None) -> Optional[int]:
            best_i, best_s = None, -1e18
            for d in details:
                if d["index"] == exclude:
                    continue
                if not d.get("eligible"):
                    continue
                if d["score"] > best_s:
                    best_s = d["score"]
                    best_i = d["index"]
            return best_i

        f1_idx = _best(f1_details)
        f2_idx = _best(f2_details, exclude=f1_idx)
        if f2_idx is None and f1_idx is not None:
            # Retry F2 without excluding if needed after peer cull emptied it
            f2_idx = _best(f2_details, exclude=f1_idx)

        # Partner scale: drop if one is tiny vs the other
        if f1_idx is not None and f2_idx is not None:
            a1 = _bbox_wh_area(boxes[f1_idx])[2]
            a2 = _bbox_wh_area(boxes[f2_idx])[2]
            lo, hi = min(a1, a2), max(a1, a2)
            if hi > 0 and lo < PARTNER_AREA_MIN_FRAC * hi:
                # Keep the one that better matches its own ref; drop the other
                s1 = next((d["score"] for d in f1_details if d["index"] == f1_idx), -1e9)
                s2 = next((d["score"] for d in f2_details if d["index"] == f2_idx), -1e9)
                if s1 >= s2:
                    f2_idx = _best(f2_details, exclude=f1_idx)
                    # if still bad, clear pair
                    if f2_idx is not None:
                        a2 = _bbox_wh_area(boxes[f2_idx])[2]
                        if min(a1, a2) < PARTNER_AREA_MIN_FRAC * max(a1, a2):
                            f2_idx = None
                else:
                    f1_idx = _best(f1_details, exclude=f2_idx)
                    if f1_idx is not None:
                        a1 = _bbox_wh_area(boxes[f1_idx])[2]
                        if min(a1, a2) < PARTNER_AREA_MIN_FRAC * max(a1, a2):
                            f1_idx = None

        for d in f1_details:
            if d.get("eligible"):
                _emit(
                    f"TRACK ref_f1 det[{d['index']}] score={d['score']} "
                    f"dist={d['dist_px']} size_ok={d['size_ok']} gate={d['gate_ok']}"
                )
        for d in f2_details:
            if d.get("eligible"):
                _emit(
                    f"TRACK ref_f2 det[{d['index']}] score={d['score']} "
                    f"dist={d['dist_px']} size_ok={d['size_ok']} gate={d['gate_ok']}"
                )

        if f1_idx is not None and f2_idx is not None and f1_idx != f2_idx:
            return _finish(f1_idx, f2_idx, "gate_keypoints_ref_match", None)
        _emit(
            f"TRACK ref_match incomplete f1={f1_idx} f2={f2_idx}; "
            "falling back to area L/R"
        )

    # Fallback: two biggest full-body people in the band → left/right = F1/F2.
    def _area_i(i: int) -> float:
        return _bbox_wh_area(boxes[i])[2]

    ranked = sorted(gate_idxs, key=_area_i, reverse=True)
    if ranked:
        largest_h = _bbox_wh_area(boxes[ranked[0]])[1]
        sized = [
            i
            for i in ranked
            if _bbox_wh_area(boxes[i])[1] >= REF_SIZE_MIN_HEIGHT_FRAC * max(largest_h, 1.0)
        ]
        if len(sized) >= 2:
            ranked = sized
    audit["kp_rank"] = [
        {
            "index": i,
            "kp_support": gate_keypoint_support(keypoints[i], y0, y1),
            "kp_center": list(keypoint_center_in_gate(keypoints[i], y0, y1) or ()),
            "bbox_area": round(_area_i(i), 1),
        }
        for i in ranked
    ]
    for entry in audit["kp_rank"]:
        _emit(
            f"TRACK area_rank det[{entry['index']}] area={entry['bbox_area']} "
            f"support={entry['kp_support']} center={entry['kp_center']}"
        )
    if len(ranked) < 2:
        return _finish(None, None, "gate_keypoints_lr", f"kp_rank_lt2({ranked})")
    top2 = ranked[:2]

    def _cx(i: int) -> float:
        c = keypoint_center_in_gate(keypoints[i], y0, y1)
        if c is not None:
            return c[0]
        return (float(boxes[i][0]) + float(boxes[i][2])) / 2.0

    positions = [(i, _cx(i)) for i in top2]
    positions.sort(key=lambda x: x[1])
    left_idx, right_idx = positions[0][0], positions[1][0]
    if fencer1_side == "left":
        return _finish(left_idx, right_idx, "gate_keypoints_lr", None)
    return _finish(right_idx, left_idx, "gate_keypoints_lr", None)


def ensure_pose_stack(log_fn) -> None:
    global _pose_detector, _pose_estimator, _pose_device
    with _pose_init_lock:
        if _pose_estimator is not None:
            # Arm-attempt pass may have parked ViTPose on CPU for VRAM; restore.
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            if _pose_device != device:
                try:
                    _pose_estimator.to(device)
                    if _pose_detector is not None:
                        _pose_detector.to(device)
                    _pose_device = device
                    log_fn(f"Pose stack restored to {device}.")
                except Exception as exc:
                    log_fn(f"WARNING: could not restore pose stack to {device}: {exc}")
            return
        ckpt = _POSE_CHECKPOINT.strip()
        if not ckpt or (not ckpt.startswith("http") and not os.path.isfile(ckpt)):
            raise FileNotFoundError(
                f"COCO-17 ViTPose-H checkpoint not found. "
                f"Tried VITPOSE_H_CHECKPOINT / checkpoints dir: '{ckpt}'. "
                "Download the OpenMMLab ViTPose-H COCO weights or set VITPOSE_H_CHECKPOINT."
            )
        if not os.path.isfile(_POSE_CONFIG):
            raise FileNotFoundError(f"Pose config missing: {_POSE_CONFIG}")
        if not os.path.isfile(_DET_CONFIG):
            raise FileNotFoundError(
                f"MMDet config missing: {_DET_CONFIG}. "
                f"Ensure app/mmpose is present or set MMPOSE_ROOT (current root: {_MMPOSE_ROOT})."
            )
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        log_fn(f"Loading RTMDet + COCO-17 ViTPose-H on {device}...")
        log_fn(f"  Pose config: {_POSE_CONFIG}")
        log_fn(f"  Pose weights: {ckpt}")
        det_ckpt = rtmdet_person_checkpoint_path()
        log_fn(f"  Detector weights: {det_ckpt}")
        from mmdet.apis import init_detector
        from mmpose.apis import init_model as init_pose_estimator
        from mmpose.utils import adapt_mmdet_pipeline

        det = init_detector(_DET_CONFIG, det_ckpt, device=device)
        det.cfg = adapt_mmdet_pipeline(det.cfg)
        pose = init_pose_estimator(_POSE_CONFIG, ckpt, device=device)
        _pose_detector = det
        _pose_estimator = pose
        _pose_device = device
        log_fn("Pose stack ready (COCO-17 ViTPose-H).")


def get_shared_person_detector():
    """RTMDet instance from the main pose stack, or None if not loaded."""
    return _pose_detector


def park_vitpose_to_cpu(log_fn=None) -> bool:
    """
    Move ViTPose-H off GPU to free VRAM for the arm-attempt RTMPose pass.
    Leaves the shared RTMDet on device so arm detection can reuse it.
    Returns True if anything was parked.
    """
    global _pose_estimator, _pose_device
    if _pose_estimator is None:
        return False
    if not torch.cuda.is_available():
        return False
    if str(_pose_device).startswith("cpu"):
        return False
    try:
        _pose_estimator.to("cpu")
        _pose_device = "cpu"
        torch.cuda.empty_cache()
        if log_fn:
            log_fn("[ARM] Parked ViTPose-H on CPU to free VRAM for RTMPose.")
        return True
    except Exception as exc:
        if log_fn:
            log_fn(f"[ARM] WARNING: could not park ViTPose: {exc}")
        return False


def unpark_vitpose_to_gpu(log_fn=None) -> bool:
    """Restore ViTPose-H to CUDA after the arm-attempt pass (next job / reuse)."""
    global _pose_estimator, _pose_detector, _pose_device
    if _pose_estimator is None:
        return False
    if not torch.cuda.is_available():
        return False
    if str(_pose_device).startswith("cuda"):
        return False
    try:
        device = "cuda:0"
        _pose_estimator.to(device)
        if _pose_detector is not None:
            _pose_detector.to(device)
        _pose_device = device
        if log_fn:
            log_fn("[ARM] Restored ViTPose-H to GPU.")
        return True
    except Exception as exc:
        if log_fn:
            log_fn(f"[ARM] WARNING: could not restore ViTPose: {exc}")
        return False


# Backward-compatible alias (legacy fencing18 name).
ensure_fencing18_stack = ensure_pose_stack


def infer_pose(
    frame: np.ndarray,
    vertical_ref_y0: Optional[float] = None,
    vertical_ref_y1: Optional[float] = None,
    bbox_scale: float = 1.0,
    bbox_pad: float = 0.0,
    bbox_thr: float = 0.3,
    nms_thr: float = 0.3,
    top_k_persons: int = 2,
    order_fencers_lr: bool = True,
    det_cat_id: int = 0,
    filter_partial_bodies: bool = False,
    include_debug: bool = False,
) -> dict:
    """
    Person detection + COCO-17 ViTPose-H on one BGR frame.
    Returns {'bboxes': [], 'keypoints': [[[x,y,sc], ...17]], 'bbox_scores': []}.
    """
    def _empty(reason: str, **extra) -> dict:
        out = {
            "bboxes": [],
            "keypoints": [],
            "bbox_scores": [],
            "_pose_filter_audit": {"mode": "empty", "reason": reason, **extra},
        }
        return out

    if _pose_detector is None or _pose_estimator is None:
        return _empty("models_not_loaded")
    from mmdet.apis import inference_detector
    from mmpose.apis import inference_topdown
    from mmpose.evaluation.functional import nms
    from mmpose.structures import merge_data_samples

    try:
        det_result = inference_detector(_pose_detector, frame)
        pred = det_result.pred_instances
        if pred is None or len(pred) == 0:
            return _empty("no_detector_instances")
        pred_instance = pred.cpu().numpy()
        dets = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        n_raw = int(dets.shape[0])
        dets = dets[
            np.logical_and(
                pred_instance.labels == det_cat_id,
                pred_instance.scores > bbox_thr,
            )
        ]
        n_after_score = int(dets.shape[0])
        dets = dets[nms(dets, nms_thr), :4]
        n_after_nms = int(dets.shape[0])
        if vertical_ref_y0 is not None and vertical_ref_y1 is not None:
            dets = filter_dets_vertical_third_xyxy(
                dets, float(vertical_ref_y0), float(vertical_ref_y1)
            )
        n_after_band = int(dets.shape[0]) if dets.size else 0
        if dets.size == 0:
            return _empty(
                "no_dets_after_band_or_nms",
                n_raw=n_raw,
                n_after_score=n_after_score,
                n_after_nms=n_after_nms,
                n_after_band=n_after_band,
            )

        h, w = frame.shape[:2]
        use_body_filter = filter_partial_bodies or (
            vertical_ref_y0 is not None
            and vertical_ref_y1 is not None
            and top_k_persons <= 2
        )
        fallback_if_no_full_body = not filter_partial_bodies
        use_vertical_band_pool = (
            vertical_ref_y0 is not None
            and vertical_ref_y1 is not None
            and top_k_persons <= 2
        )
        det_pool = (
            max(top_k_persons, POSE_CANDIDATE_POOL)
            if use_vertical_band_pool
            else top_k_persons
        )

        if vertical_ref_y0 is not None and vertical_ref_y1 is not None:
            dets = filter_topk_by_inband_area_xyxy(
                dets,
                det_pool,
                float(vertical_ref_y0),
                float(vertical_ref_y1),
                order_fencers_lr,
            )
        else:
            dets = filter_topk_by_area_xyxy(dets, det_pool, order_fencers_lr)
        if dets.size == 0:
            return _empty(
                "no_dets_after_topk",
                n_raw=n_raw,
                n_after_band=n_after_band,
                det_pool=det_pool,
            )

        bboxes = expand_bboxes_xyxy(dets, h, w, bbox_scale, bbox_pad)
        pose_results = inference_topdown(_pose_estimator, frame, bboxes)
        data_samples = merge_data_samples(pose_results)
        inst = data_samples.pred_instances
        if inst is None or len(inst) == 0:
            return _empty("pose_returned_no_instances", n_pose_inputs=int(bboxes.shape[0]))

        bb = inst.bboxes
        kpts = inst.keypoints
        ksc = inst.keypoint_scores
        if isinstance(bb, torch.Tensor):
            bb = bb.detach().cpu().numpy()
        if isinstance(kpts, torch.Tensor):
            kpts = kpts.detach().cpu().numpy()
        if isinstance(ksc, torch.Tensor):
            ksc = ksc.detach().cpu().numpy()

        n = int(bb.shape[0])
        bbox_scores = None
        if hasattr(inst, "bbox_scores") and inst.bbox_scores is not None:
            bs = inst.bbox_scores
            bbox_scores = bs.detach().cpu().numpy() if isinstance(bs, torch.Tensor) else np.asarray(bs)

        structured_results = {"bboxes": [], "keypoints": [], "bbox_scores": []}
        for i in range(n):
            structured_results["bboxes"].append(
                [float(bb[i, 0]), float(bb[i, 1]), float(bb[i, 2]), float(bb[i, 3])]
            )
            row = []
            for j in range(kpts.shape[1]):
                row.append(
                    [
                        float(kpts[i, j, 0]),
                        float(kpts[i, j, 1]),
                        float(ksc[i, j]),
                    ]
                )
            structured_results["keypoints"].append(row)
            if bbox_scores is not None:
                structured_results["bbox_scores"].append(float(bbox_scores[i]))
            else:
                structured_results["bbox_scores"].append(1.0)

        if use_body_filter and structured_results["bboxes"]:
            debug_before = build_detection_debug_payload(structured_results, h, w)
            if filter_partial_bodies:
                filtered = select_full_body_instances(
                    structured_results,
                    h,
                    w,
                    top_k_persons,
                    vertical_y0=vertical_ref_y0,
                    vertical_y1=vertical_ref_y1,
                    order_left_to_right=order_fencers_lr,
                    fallback_if_none=False,
                )
            elif use_vertical_band_pool:
                # Pipeline: keep people with upper+lower ViTPose joints in the gate.
                # Identity is keypoint-center vs initial selection, not bbox area.
                filtered = select_gate_fencer_instances(
                    structured_results,
                    h,
                    w,
                    top_k=det_pool,
                    vertical_y0=float(vertical_ref_y0),
                    vertical_y1=float(vertical_ref_y1),
                    order_left_to_right=order_fencers_lr,
                    fallback_if_none=True,
                )
            else:
                filtered = select_full_body_instances(
                    structured_results,
                    h,
                    w,
                    top_k_persons,
                    vertical_y0=vertical_ref_y0,
                    vertical_y1=vertical_ref_y1,
                    order_left_to_right=order_fencers_lr,
                    fallback_if_none=fallback_if_no_full_body,
                )
            filtered["_debug_before_filter"] = debug_before
            filtered["_debug_after_filter"] = build_detection_debug_payload(
                filtered, h, w
            )
            if vertical_ref_y0 is not None and vertical_ref_y1 is not None:
                filtered["_debug_gate_before"] = build_gate_debug_payload(
                    structured_results,
                    float(vertical_ref_y0),
                    float(vertical_ref_y1),
                )
            structured_results = filtered

        return structured_results
    except Exception as exc:
        _logger.exception("infer_pose failed")
        return _empty("exception", error=str(exc))


# Backward-compatible alias (legacy fencing18 name).
infer_pose_fencing18 = infer_pose


def extract_keypoints_dict(structured: dict, fencer_id: Optional[int]) -> Optional[dict]:
    """JSON-friendly dict: COCO-17 joints as name -> [x, y]."""
    if (
        not structured
        or not structured.get("keypoints")
        or fencer_id is None
        or fencer_id >= len(structured["keypoints"])
    ):
        return None
    kpts = structured["keypoints"][fencer_id]
    labeled = {}
    for i, name in enumerate(COCO_KEYPOINT_NAMES):
        if i >= len(kpts):
            break
        x, y, conf = kpts[i][0], kpts[i][1], kpts[i][2]
        labeled[name] = [float(x), float(y)]
    return labeled


# Backward-compatible alias (legacy fencing18 name).
extract_fencing18_keypoints_dict = extract_keypoints_dict
