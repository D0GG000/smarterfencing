"""
Bout-wide 2D arm-attempt detection (lightweight rules).

Operates on COCO-17 keypoint dicts from the production fencer filter
(left=F1, right=F2). Weapon arm is applied later from 3D handedness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# COCO-17 names used by extract_keypoints_dict
L_SH, R_SH = "left_shoulder", "right_shoulder"
L_EL, R_EL = "left_elbow", "right_elbow"
L_WR, R_WR = "left_wrist", "right_wrist"


@dataclass
class ArmAttempt:
    fencer: str  # "fencer1" | "fencer2"
    arm: str  # "left" | "right"
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    peak_angle: float
    delta_deg: float
    peak_speed_deg_s: float
    point_toward: float
    wrist_forward: float
    source: str = "2d_onnx_yolo_rtmpose"


def _xy(kp: Optional[dict], name: str) -> Optional[np.ndarray]:
    if not kp or name not in kp:
        return None
    v = kp[name]
    if v is None or len(v) < 2:
        return None
    return np.array([float(v[0]), float(v[1])], dtype=np.float64)


def elbow_angle_deg(kp: Optional[dict], side: str) -> float:
    """Interior elbow angle in degrees; 0 if missing."""
    if side == "left":
        sh, el, wr = _xy(kp, L_SH), _xy(kp, L_EL), _xy(kp, L_WR)
    else:
        sh, el, wr = _xy(kp, R_SH), _xy(kp, R_EL), _xy(kp, R_WR)
    if sh is None or el is None or wr is None:
        return 0.0
    ba, bc = sh - el, wr - el
    na, nc = float(np.linalg.norm(ba)), float(np.linalg.norm(bc))
    if na < 1e-8 or nc < 1e-8:
        return 0.0
    cos = float(np.dot(ba, bc) / (na * nc))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cos)))))


def arm_xy_features(kp: Optional[dict], side: str, frame_w: int) -> Dict[str, float]:
    """Normalized wrist/shoulder x + elbow angle for one arm."""
    w = max(int(frame_w), 1)
    if side == "left":
        sh, wr = _xy(kp, L_SH), _xy(kp, L_WR)
    else:
        sh, wr = _xy(kp, R_SH), _xy(kp, R_WR)
    out = {"elbow": 0.0, "wrist_x": 0.0, "shoulder_x": 0.0}
    if sh is None or wr is None:
        return out
    out["elbow"] = elbow_angle_deg(kp, side)
    out["wrist_x"] = float(wr[0] / w)
    out["shoulder_x"] = float(sh[0] / w)
    return out


def _elbow_nan(elbows: Sequence[float]) -> np.ndarray:
    arr = np.asarray(elbows, dtype=np.float64).copy()
    arr[arr <= 1.0] = np.nan
    return arr


def _rolling_median_nan(arr: np.ndarray, win: int = 3) -> np.ndarray:
    out = arr.copy()
    r = max(0, win // 2)
    for i in range(len(arr)):
        w = arr[max(0, i - r) : i + r + 1]
        w = w[np.isfinite(w)]
        if w.size:
            out[i] = float(np.median(w))
    return out


def _opening_speeds(
    smooth: np.ndarray,
    sample_fps: float,
    span: int = 2,
    max_jump_deg: float = 50.0,
) -> np.ndarray:
    n = len(smooth)
    speeds = np.full(n, np.nan, dtype=np.float64)
    span = max(1, int(span))
    for i in range(span, n):
        a0, a1 = smooth[i - span], smooth[i]
        if not (np.isfinite(a0) and np.isfinite(a1)):
            continue
        bad = False
        for j in range(i - span + 1, i + 1):
            if not (np.isfinite(smooth[j - 1]) and np.isfinite(smooth[j])):
                bad = True
                break
            if abs(float(smooth[j] - smooth[j - 1])) > max_jump_deg:
                bad = True
                break
        if bad:
            continue
        speeds[i] = (float(a1) - float(a0)) * float(sample_fps) / float(span)
    return speeds


def _body_relative_wrist_fwd(
    wx: Sequence[float],
    sx: Sequence[float],
    onset0: int,
    i: int,
    toward: float,
) -> float:
    rel: List[float] = []
    for j in range(onset0, i + 1):
        if float(wx[j]) <= 0 and float(sx[j]) <= 0:
            continue
        rel.append(toward * (float(wx[j]) - float(sx[j])))
    if len(rel) < 2:
        return 0.0
    return float(max(rel) - min(rel))


def detect_attempts_for_arm(
    elbows: Sequence[float],
    wrist_x: Sequence[float],
    shoulder_x: Sequence[float],
    *,
    fencer: str,
    arm: str,
    fps: float,
    stride: int,
    toward: float,
    min_peak: float = 150.0,
    min_delta: float = 30.0,
    min_speed: float = 300.0,
    min_point: float = 0.02,
    min_wrist_fwd: float = 0.015,
    lookback_sec: float = 0.40,
    cooldown_sec: float = 0.40,
    speed_span: int = 2,
    max_jump_deg: float = 50.0,
) -> List[ArmAttempt]:
    sample_fps = fps / max(1, stride)
    lookback = max(2, int(round(lookback_sec * sample_fps)))
    cooldown = max(1, int(round(cooldown_sec * sample_fps)))
    n = min(len(elbows), len(wrist_x), len(shoulder_x))
    smooth = _rolling_median_nan(_elbow_nan(elbows[:n]), win=3)
    speeds = _opening_speeds(smooth, sample_fps, span=speed_span, max_jump_deg=max_jump_deg)
    attempts: List[ArmAttempt] = []
    last = -10**9

    for i in range(lookback, n):
        if i - last < cooldown:
            continue
        if not np.isfinite(smooth[i]):
            continue
        onset0 = i - lookback
        window = smooth[onset0 : i + 1]
        if int(np.sum(np.isfinite(window))) < max(3, lookback // 2):
            continue
        early_cut = max(1, len(window) // 3)
        late_start = max(0, len(window) * 2 // 3)
        early = window[:early_cut]
        late = window[late_start:]
        early_f = early[np.isfinite(early)]
        late_f = late[np.isfinite(late)]
        if early_f.size == 0 or late_f.size == 0:
            continue
        pre_min = float(np.min(early_f))
        peak = float(np.max(late_f))
        if peak < min_peak:
            continue
        delta = peak - pre_min
        if delta < min_delta:
            continue
        sw = speeds[onset0 : i + 1]
        sw_f = sw[np.isfinite(sw)]
        if sw_f.size == 0 or float(np.max(sw_f)) < min_speed:
            continue
        near = sw[max(0, len(sw) - max(2, lookback // 3)) :]
        near_f = near[np.isfinite(near)]
        if near_f.size == 0 or float(np.max(near_f)) < min_speed * 0.5:
            continue
        if float(wrist_x[i]) <= 0 and float(shoulder_x[i]) <= 0:
            continue
        point = toward * (float(wrist_x[i]) - float(shoulder_x[i]))
        if point < min_point:
            continue
        wrist_fwd = _body_relative_wrist_fwd(wrist_x, shoulder_x, onset0, i, toward)
        if wrist_fwd < min_wrist_fwd:
            continue

        pre_rel = int(np.nanargmin(early)) if np.any(np.isfinite(early)) else 0
        peak_rel = (
            late_start + int(np.nanargmax(late)) if np.any(np.isfinite(late)) else (i - onset0)
        )
        pre_idx = onset0 + pre_rel
        peak_idx = onset0 + peak_rel
        if peak_idx < pre_idx:
            peak_idx = i
        # Map sample index → source frame (stride sampling)
        sf = int(pre_idx * stride)
        ef = int(peak_idx * stride)
        attempts.append(
            ArmAttempt(
                fencer=fencer,
                arm=arm,
                start_frame=sf,
                end_frame=ef,
                start_sec=sf / fps,
                end_sec=ef / fps,
                peak_angle=peak,
                delta_deg=delta,
                peak_speed_deg_s=float(np.max(sw_f)),
                point_toward=point,
                wrist_forward=wrist_fwd,
            )
        )
        last = i
    return attempts


def choose_weapon_arm(series: Dict[str, List[float]], person: int) -> str:
    """
    Pick arm with more shoulder-relative reach toward the opponent (pan-robust).
    Same rule as desktop test_arm_attempts_2d.py.
    """
    toward = 1.0 if person == 0 else -1.0
    left_wx = series.get(f"p{person}_left_wrist_x") or []
    right_wx = series.get(f"p{person}_right_wrist_x") or []
    left_sx = series.get(f"p{person}_left_shoulder_x") or []
    right_sx = series.get(f"p{person}_right_shoulder_x") or []
    left_el = series.get(f"p{person}_left_elbow") or []
    right_el = series.get(f"p{person}_right_elbow") or []
    vals_l: List[float] = []
    vals_r: List[float] = []
    n = min(
        len(left_wx),
        len(right_wx),
        len(left_sx),
        len(right_sx),
        len(left_el),
        len(right_el),
    )
    for i in range(n):
        if left_el[i] <= 1 and right_el[i] <= 1:
            continue
        if left_wx[i] <= 0 and left_sx[i] <= 0 and right_wx[i] <= 0 and right_sx[i] <= 0:
            continue
        vals_l.append(toward * (float(left_wx[i]) - float(left_sx[i])))
        vals_r.append(toward * (float(right_wx[i]) - float(right_sx[i])))
    if not vals_l or not vals_r:
        return "right" if person == 0 else "left"
    mean_l, mean_r = float(np.mean(vals_l)), float(np.mean(vals_r))
    return "right" if mean_r >= mean_l else "left"


def detect_attempts_from_series(
    series: Dict[str, List[float]],
    fps: float,
    stride: int,
    frame_indices: Optional[Sequence[int]] = None,
    arms_by_person: Optional[Dict[int, str]] = None,
    **rule_kwargs,
) -> List[ArmAttempt]:
    """
    series keys: p0|p1 _{left|right}_{elbow,wrist_x,shoulder_x}
    person 0 = fencer1 (left), toward +x; person 1 = fencer2, toward -x.

    Series samples are spaced ``stride`` video frames apart. Temporal math
    (lookback / speed) must use sample_fps = fps/stride. Frame numbers are
    remapped via ``frame_indices`` when provided.
    """
    stride = max(1, int(stride))
    attempts: List[ArmAttempt] = []
    for person, fencer in ((0, "fencer1"), (1, "fencer2")):
        toward = 1.0 if person == 0 else -1.0
        if arms_by_person is not None:
            arms = [arms_by_person.get(person, "right")]
        else:
            arms = ["left", "right"]
        for arm in arms:
            if arm not in ("left", "right"):
                continue
            el = series.get(f"p{person}_{arm}_elbow") or []
            wx = series.get(f"p{person}_{arm}_wrist_x") or []
            sx = series.get(f"p{person}_{arm}_shoulder_x") or []
            # Series is already one row per sampled frame: use stride for
            # sample_fps only, then map sample index → source frame.
            found = detect_attempts_for_arm(
                el, wx, sx,
                fencer=fencer,
                arm=arm,
                fps=fps,
                stride=stride,
                toward=toward,
                **rule_kwargs,
            )
            if frame_indices is not None and len(frame_indices) > 0:
                for a in found:
                    # detect_attempts_for_arm stored sample_idx * stride
                    si = int(a.start_frame // stride)
                    ei = int(a.end_frame // stride)
                    si = max(0, min(len(frame_indices) - 1, si))
                    ei = max(0, min(len(frame_indices) - 1, ei))
                    a.start_frame = int(frame_indices[si])
                    a.end_frame = int(frame_indices[ei])
                    a.start_sec = a.start_frame / fps
                    a.end_sec = a.end_frame / fps
            attempts.extend(found)
    attempts.sort(key=lambda a: (a.start_frame, a.fencer))
    return attempts


def attribute_weapon_arm(
    attempts: List[ArmAttempt],
    fencer1_handedness: Optional[str],
    fencer2_handedness: Optional[str],
) -> List[ArmAttempt]:
    """Keep only the weapon-arm attempts for each fencer."""
    h1 = (fencer1_handedness or "").lower()
    h2 = (fencer2_handedness or "").lower()
    if h1 not in ("left", "right"):
        h1 = "right"
    if h2 not in ("left", "right"):
        h2 = "right"
    out = []
    for a in attempts:
        want = h1 if a.fencer == "fencer1" else h2
        if a.arm == want:
            out.append(a)
    return out


def attempts_payload(
    attempts: List[ArmAttempt],
    *,
    fencer1_handedness: Optional[str] = None,
    fencer2_handedness: Optional[str] = None,
    rules: Optional[dict] = None,
) -> Dict[str, Any]:
    items = []
    for i, a in enumerate(attempts):
        d = asdict(a)
        d["id"] = f"{a.fencer}_attempt_frame{a.end_frame}_{i}"
        items.append(d)
    c1 = sum(1 for a in attempts if a.fencer == "fencer1")
    c2 = sum(1 for a in attempts if a.fencer == "fencer2")
    return {
        "fencer1_total": c1,
        "fencer2_total": c2,
        "fencer1_handedness": fencer1_handedness,
        "fencer2_handedness": fencer2_handedness,
        "rules": rules or {},
        "items": items,
    }
