"""
Person-relative forward / backward along the strip.

Production labels use planted-ankle camera compensation: world hip travel =
image hip dx − median ankle dx (planted feet ≈ world-stationary). Facing is
toward the opponent. Designed for the same COCO-17 dicts as arm attempts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

L_SH, R_SH = "left_shoulder", "right_shoulder"
L_HIP, R_HIP = "left_hip", "right_hip"
L_ANK, R_ANK = "left_ankle", "right_ankle"
NOSE = "nose"


def _xy(kp: Optional[dict], name: str) -> Optional[np.ndarray]:
    if not kp or name not in kp:
        return None
    v = kp[name]
    if v is None or len(v) < 2:
        return None
    return np.array([float(v[0]), float(v[1])], dtype=np.float64)


def _unit(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return None
    return (v / n).astype(np.float64)


def mid_hip(kp: Optional[dict]) -> Optional[np.ndarray]:
    lh, rh = _xy(kp, L_HIP), _xy(kp, R_HIP)
    if lh is None or rh is None:
        return None
    return (0.5 * (lh + rh)).astype(np.float64)


def body_scale(kp: Optional[dict]) -> float:
    lh, rh = _xy(kp, L_HIP), _xy(kp, R_HIP)
    ls, rs = _xy(kp, L_SH), _xy(kp, R_SH)
    if lh is not None and rh is not None and ls is not None and rs is not None:
        mh = 0.5 * (lh + rh)
        ms = 0.5 * (ls + rs)
        t = float(np.linalg.norm(ms - mh))
        if t > 1.0:
            return t
    if lh is not None and rh is not None:
        w = float(np.linalg.norm(lh - rh))
        if w > 1.0:
            return w
    return 80.0


def estimate_facing(
    kp: Optional[dict],
    toward_x: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Horizontal facing; prefer toward opponent when known."""
    origin = mid_hip(kp)
    if origin is None:
        ls, rs = _xy(kp, L_SH), _xy(kp, R_SH)
        if ls is not None and rs is not None:
            origin = (0.5 * (ls + rs)).astype(np.float64)
    if origin is None:
        return None

    if toward_x is not None:
        dx = float(toward_x) - float(origin[0])
        if abs(dx) >= 2.0:
            return np.array([1.0 if dx > 0 else -1.0, 0.0], dtype=np.float64)

    right_w = 0.0
    left_w = 0.0

    def _vote(dx: float, weight: float) -> None:
        nonlocal right_w, left_w
        if abs(dx) < 2.0:
            return
        if dx > 0:
            right_w += weight * abs(dx)
        else:
            left_w += weight * abs(dx)

    nose = _xy(kp, NOSE)
    if nose is not None:
        _vote(float(nose[0] - origin[0]), 2.0)
    ls, rs = _xy(kp, L_SH), _xy(kp, R_SH)
    if ls is not None and rs is not None:
        _vote(float(0.5 * (ls[0] + rs[0]) - origin[0]), 0.75)
    anks = [a for a in (_xy(kp, L_ANK), _xy(kp, R_ANK)) if a is not None]
    if anks:
        lead = max(anks, key=lambda a: abs(float(a[0] - origin[0])))
        _vote(float(lead[0] - origin[0]), 1.25)

    if right_w < 1e-6 and left_w < 1e-6:
        return None
    return np.array([1.0 if right_w >= left_w else -1.0, 0.0], dtype=np.float64)


def _ankle_pair(kp: Optional[dict]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    return _xy(kp, L_ANK), _xy(kp, R_ANK)


def planted_ankle_cam_dx(
    kps: List[Optional[dict]],
    prev_ankle_x: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    """
    Estimate horizontal camera pan (px) from median ankle image displacement.

    Planted feet are approximately world-stationary, so their image motion is
    the pan. Returns (cam_dx_px, updated_ankle_x_map).
    """
    cur: Dict[str, float] = {}
    for pi, kp in enumerate(kps[:2]):
        la, ra = _ankle_pair(kp)
        if la is not None:
            cur[f"{pi}l"] = float(la[0])
        if ra is not None:
            cur[f"{pi}r"] = float(ra[0])
    dxs = [cur[k] - prev_ankle_x[k] for k in cur if k in prev_ankle_x]
    cam_dx = float(np.median(dxs)) if len(dxs) >= 2 else 0.0
    return cam_dx, cur


class PersonMotionState:
    def __init__(
        self,
        still_thr: float = 0.18,
        face_ema: float = 0.12,
        hip_ema: float = 0.35,
        speed_ema: float = 0.45,
        switch_hold: int = 3,
        coast_sec: float = 0.25,
    ):
        self.still_thr = float(still_thr)
        self.face_ema = float(face_ema)
        self.hip_ema = float(hip_ema)
        self.speed_ema = float(speed_ema)
        self.switch_hold = int(switch_hold)
        self.coast_sec = float(coast_sec)
        self.facing: Optional[np.ndarray] = None
        self.hip_smooth: Optional[np.ndarray] = None
        self._vel: Optional[np.ndarray] = None
        self._signed_smooth = 0.0
        self.t: Optional[float] = None
        self.label = "unknown"
        self.signed_speed = 0.0
        self._pending: Optional[str] = None
        self._pending_n = 0
        self._seen_t: Optional[float] = None

    def update(
        self,
        kp: Optional[dict],
        t: float,
        opponent_hip: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        if kp is None:
            # Pose dropouts cluster on the scoring action (deep lunge/flèche, the
            # two bodies overlapping), so blanking on the first miss reads as
            # "unknown" exactly when the touch lands. Hold the last label for a
            # short window; only sustained losses (cuts, replays) go unknown.
            if not self._coasting(t):
                self.label = "unknown"
                self.signed_speed = 0.0
                self._pending = None
                self._pending_n = 0
            return self.snapshot()

        self._seen_t = t
        hip_raw = mid_hip(kp)
        toward_x = None if opponent_hip is None else float(opponent_hip[0])
        face = estimate_facing(kp, toward_x=toward_x)
        scale = body_scale(kp)
        locked = toward_x is not None and face is not None

        if face is not None:
            if self.facing is None:
                self.facing = face.copy()
            elif locked:
                a = max(self.face_ema, 0.45)
                blended = _unit(
                    np.array(
                        [(1.0 - a) * self.facing[0] + a * face[0], 0.0],
                        dtype=np.float64,
                    )
                )
                if blended is not None:
                    self.facing = blended
            else:
                if float(np.dot(face, self.facing)) < 0.0:
                    face = -face
                a = self.face_ema
                blended = _unit(
                    np.array(
                        [(1.0 - a) * self.facing[0] + a * face[0], 0.0],
                        dtype=np.float64,
                    )
                )
                if blended is not None:
                    self.facing = blended

        if hip_raw is not None:
            if self.hip_smooth is None or self.t is None:
                self.hip_smooth = hip_raw.copy()
                self.t = t
            else:
                dt = max(t - self.t, 1e-4)
                a = self.hip_ema
                prev = self.hip_smooth
                self.hip_smooth = (1.0 - a) * self.hip_smooth + a * hip_raw
                raw_v = (self.hip_smooth - prev) / dt
                if self._vel is None:
                    self._vel = raw_v
                else:
                    self._vel = (1.0 - a) * self._vel + a * raw_v
                self.t = t

        label = "still"
        signed = 0.0
        if self._vel is not None and self.facing is not None and scale > 1.0:
            vel_h = np.array([float(self._vel[0]), 0.0], dtype=np.float64)
            signed_raw = float(np.dot(vel_h, self.facing)) / scale
            b = self.speed_ema
            self._signed_smooth = (1.0 - b) * self._signed_smooth + b * signed_raw
            signed = self._signed_smooth
            if signed > self.still_thr:
                label = "forward"
            elif signed < -self.still_thr:
                label = "backward"
            else:
                label = "still"
        elif self.facing is None:
            label = "unknown"

        self.signed_speed = signed
        self._commit_label(label)
        return self.snapshot(hip_raw=hip_raw)

    def _coasting(self, t: float) -> bool:
        if self.label == "unknown" or self._seen_t is None:
            return False
        return (float(t) - self._seen_t) <= self.coast_sec

    def _commit_label(self, label: str) -> None:
        if label == self.label:
            self._pending = None
            self._pending_n = 0
            return
        if label == "unknown":
            self.label = label
            self._pending = None
            self._pending_n = 0
            return
        if self._pending != label:
            self._pending = label
            self._pending_n = 1
        else:
            self._pending_n += 1
        if self._pending_n >= self.switch_hold or self.label == "unknown":
            self.label = label
            self._pending = None
            self._pending_n = 0

    def snapshot(self, hip_raw: Optional[np.ndarray] = None) -> Dict[str, Any]:
        hip_draw = hip_raw if hip_raw is not None else self.hip_smooth
        vel_u = None
        if self._vel is not None:
            vel_u = _unit(np.array([float(self._vel[0]), 0.0], dtype=np.float64))
        return {
            "label": self.label,
            "signed_speed": round(float(self.signed_speed), 4),
            "facing": None
            if self.facing is None
            else [round(float(self.facing[0]), 5), round(float(self.facing[1]), 5)],
            "vel_dir": None
            if vel_u is None
            else [round(float(vel_u[0]), 5), round(float(vel_u[1]), 5)],
            "hip": None
            if hip_draw is None
            else [round(float(hip_draw[0]), 2), round(float(hip_draw[1]), 2)],
        }


class CamCompHipTracker:
    """
    Forward/back from hip travel after removing camera pan.

    Caller supplies cam_dx_px (typically planted_ankle_cam_dx). No FSM:
    world_dx = hip_dx − cam_dx; signed along facing with hysteresis.
    """

    def __init__(
        self,
        still_thr: float = 0.35,
        exit_ratio: float = 0.55,
        speed_ema: float = 0.45,
        switch_hold: int = 2,
        coast_sec: float = 0.25,
        face_ema: float = 0.12,
    ):
        self.still_thr = float(still_thr)
        self.exit_ratio = float(exit_ratio)
        self.speed_ema = float(speed_ema)
        self.switch_hold = int(switch_hold)
        self.coast_sec = float(coast_sec)
        self.face_ema = float(face_ema)

        self.facing: Optional[np.ndarray] = None
        self.hip_smooth: Optional[np.ndarray] = None
        self.t: Optional[float] = None
        self.label = "unknown"
        self.signed_speed = 0.0
        self.signal_source = "anchor_hip"
        self.world_x = 0.0
        self._hip_x: Optional[float] = None
        self._v = 0.0
        self._pending: Optional[str] = None
        self._pending_n = 0
        self._seen_t: Optional[float] = None

    def update(
        self,
        kp: Optional[dict],
        t: float,
        cam_dx_px: float = 0.0,
        opponent_hip: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        if kp is None:
            if not self._coasting(t):
                self.label = "unknown"
                self.signed_speed = 0.0
                self._hip_x = None
                self._v = 0.0
                self._pending = None
                self._pending_n = 0
            return self.snapshot()

        self._seen_t = t
        hip = mid_hip(kp)
        scale = body_scale(kp)
        toward_x = None if opponent_hip is None else float(opponent_hip[0])
        face = estimate_facing(kp, toward_x=toward_x)
        if face is not None:
            if self.facing is None:
                self.facing = face.copy()
            else:
                if toward_x is None and float(np.dot(face, self.facing)) < 0.0:
                    face = -face
                a = max(self.face_ema, 0.45) if toward_x is not None else self.face_ema
                blended = _unit(
                    np.array(
                        [(1.0 - a) * self.facing[0] + a * face[0], 0.0],
                        dtype=np.float64,
                    )
                )
                if blended is not None:
                    self.facing = blended

        dt = None if self.t is None else max(t - self.t, 1e-4)
        self.t = t
        if hip is not None:
            self.hip_smooth = (
                hip if self.hip_smooth is None else (0.5 * self.hip_smooth + 0.5 * hip)
            )

        label = "still"
        if (
            hip is not None
            and self._hip_x is not None
            and dt is not None
            and scale > 1.0
            and self.facing is not None
        ):
            dx_world = (float(hip[0]) - self._hip_x) - float(cam_dx_px)
            self.world_x += dx_world
            v = (dx_world / dt) / scale
            self._v = (1.0 - self.speed_ema) * self._v + self.speed_ema * v
            sign = 1.0 if float(self.facing[0]) >= 0.0 else -1.0
            signed = self._v * sign
            self.signed_speed = round(float(signed), 4)

            thr_in = self.still_thr
            thr_out = self.still_thr * self.exit_ratio
            cur = self.label
            if cur == "forward":
                if signed > thr_out:
                    label = "forward"
                elif signed < -thr_in:
                    label = "backward"
            elif cur == "backward":
                if signed < -thr_out:
                    label = "backward"
                elif signed > thr_in:
                    label = "forward"
            else:
                if signed > thr_in:
                    label = "forward"
                elif signed < -thr_in:
                    label = "backward"
        elif self.facing is None:
            label = "unknown"
        if hip is not None:
            self._hip_x = float(hip[0])

        self._commit_label(label)
        return self.snapshot()

    def _coasting(self, t: float) -> bool:
        if self.label == "unknown" or self._seen_t is None:
            return False
        return (float(t) - self._seen_t) <= self.coast_sec

    def _commit_label(self, label: str) -> None:
        if label == self.label:
            self._pending = None
            self._pending_n = 0
            return
        if label == "unknown":
            self.label = label
            self._pending = None
            self._pending_n = 0
            return
        if self._pending != label:
            self._pending = label
            self._pending_n = 1
        else:
            self._pending_n += 1
        if self._pending_n >= self.switch_hold or self.label == "unknown":
            self.label = label
            self._pending = None
            self._pending_n = 0

    def snapshot(self) -> Dict[str, Any]:
        hip = self.hip_smooth
        vel_u = None
        if abs(self._v) > 1e-9 and self.facing is not None:
            # Image-horizontal unit in the signed travel direction for overlay.
            sx = 1.0 if (self._v * float(self.facing[0])) >= 0.0 else -1.0
            vel_u = [sx, 0.0]
        return {
            "label": self.label,
            "signed_speed": round(float(self.signed_speed), 4),
            "signal_source": self.signal_source,
            "world_x": round(float(self.world_x), 1),
            "facing": None
            if self.facing is None
            else [round(float(self.facing[0]), 5), round(float(self.facing[1]), 5)],
            "vel_dir": vel_u,
            "hip": None
            if hip is None
            else [round(float(hip[0]), 2), round(float(hip[1]), 2)],
        }


def compact_person(snap: Dict[str, Any], frame_w: int, frame_h: int) -> Dict[str, Any]:
    """Normalized overlay record for one fencer."""
    w = max(int(frame_w), 1)
    h = max(int(frame_h), 1)
    hip = snap.get("hip")
    return {
        "l": snap.get("label") or "unknown",
        "s": snap.get("signed_speed") or 0.0,
        "fx": None if not snap.get("facing") else float(snap["facing"][0]),
        "vx": None if not snap.get("vel_dir") else float(snap["vel_dir"][0]),
        "hx": None if hip is None else round(float(hip[0]) / w, 4),
        "hy": None if hip is None else round(float(hip[1]) / h, 4),
    }


def build_forward_back_payload(
    frames: List[Dict[str, Any]],
    *,
    fps: float,
    stride: int,
    width: int,
    height: int,
    still_thr: float,
    switch_hold: int,
    method: str = "anchor_hip",
) -> Dict[str, Any]:
    return {
        "fps": float(fps),
        "stride": int(stride),
        "width": int(width),
        "height": int(height),
        "still_thr": float(still_thr),
        "switch_hold": int(switch_hold),
        "method": str(method),
        "sample_count": len(frames),
        "frames": frames,
    }


# ---------------------------------------------------------------------------
# Pre-touch spatial aggressor from forward / back series
# ---------------------------------------------------------------------------
# Who was advancing *more* toward the opponent before the light — comparative
# footwork pressure, not attack-type / priority. Complementary to the spatial
# "pressing touch" (engagement third) heuristic.
DEFAULT_PRE_TOUCH_WINDOW_SEC = 5.0
# Drop the last half-second before the light so the scoring action itself
# (lunge/flèche blur, overlap) does not dominate net distance.
DEFAULT_PRE_TOUCH_END_CUT_SEC = 0.5
# Relative margin on forward_frac for calling the exchange even.
DEFAULT_EVEN_MARGIN = 0.25
# Absolute floor: if both forward_frac are below this, call even.
DEFAULT_STILL_FLOOR = 0.10
# Min usable samples in the window (unknown-dropped).
DEFAULT_MIN_SAMPLES = 3


def touch_frame_from_name(touch_name: str) -> Optional[int]:
    if not touch_name:
        return None
    # ..._frame12345 or ..._frame12345_...
    marker = "_frame"
    idx = str(touch_name).rfind(marker)
    if idx < 0:
        return None
    tail = str(touch_name)[idx + len(marker) :]
    digits = []
    for ch in tail:
        if ch.isdigit():
            digits.append(ch)
        elif digits:
            break
    if not digits:
        return None
    try:
        return int("".join(digits))
    except ValueError:
        return None


# Back-compat alias
_touch_frame_from_name = touch_frame_from_name


def _person_window_stats(
    rows: List[Dict[str, Any]],
    person_idx: int,
    still_thr: float,
    fps: float = 30.0,
) -> Dict[str, Any]:
    """Per-person overlay stats over the pre-touch window.

    Primary score is forward_frac (share of known frames labeled forward).
    Also keeps integrated signed speed (net_disp) as a diagnostic.
    """
    del still_thr  # reserved for callers / future gates
    fps = float(fps) if fps and fps > 1e-3 else 30.0
    default_dt = 1.0 / fps

    labels = {"forward": 0, "backward": 0, "still": 0, "unknown": 0}
    # (t, signed_speed) for known samples; also track hips for endpoint check.
    samples: List[Tuple[float, float]] = []
    hip_xs: List[Tuple[float, float, float]] = []  # t, hx, fx

    for row in rows:
        people = row.get("p") or []
        try:
            t = float(row.get("t"))
        except (TypeError, ValueError):
            t = None
        if person_idx >= len(people):
            labels["unknown"] += 1
            continue
        person = people[person_idx] or {}
        lab = str(person.get("l") or "unknown")
        if lab not in labels:
            lab = "unknown"
        labels[lab] += 1
        if lab == "unknown":
            continue
        try:
            signed = float(person.get("s") or 0.0)
        except (TypeError, ValueError):
            continue
        if t is None:
            # Fall back to frame index if present.
            try:
                t = float(row.get("i")) / fps
            except (TypeError, ValueError):
                t = float(len(samples)) * default_dt
        samples.append((t, signed))
        hx = person.get("hx")
        fx = person.get("fx")
        try:
            if hx is not None and fx is not None:
                hip_xs.append((t, float(hx), float(fx)))
        except (TypeError, ValueError):
            pass

    known = labels["forward"] + labels["backward"] + labels["still"]
    n = len(samples)
    mean_signed = float(np.mean([s for _, s in samples])) if n else 0.0

    # Trapezoidal integrate signed speed → net displacement (body-lengths).
    net_disp = 0.0
    path_len = 0.0  # total |distance| traveled along facing (not net)
    if n == 1:
        net_disp = samples[0][1] * default_dt
        path_len = abs(net_disp)
    elif n > 1:
        for i in range(1, n):
            t0, s0 = samples[i - 1]
            t1, s1 = samples[i]
            dt = t1 - t0
            if dt <= 1e-6 or dt > 1.0:
                dt = default_dt
            step = 0.5 * (s0 + s1) * dt
            net_disp += step
            path_len += abs(step)

    # Optional: hip endpoint in frame-widths toward facing (sanity / display).
    hip_net_frac = 0.0
    if len(hip_xs) >= 2:
        _t0, hx0, fx0 = hip_xs[0]
        _t1, hx1, fx1 = hip_xs[-1]
        fx = fx1 if abs(fx1) >= abs(fx0) else fx0
        if abs(fx) > 0.01:
            # Positive when hip moved in facing direction (toward opponent).
            hip_net_frac = (hx1 - hx0) * (1.0 if fx > 0 else -1.0)

    return {
        "samples": known,
        "forward_n": labels["forward"],
        "backward_n": labels["backward"],
        "still_n": labels["still"],
        "unknown_n": labels["unknown"],
        "forward_frac": round(
            (labels["forward"] / known) if known else 0.0, 4
        ),
        "mean_signed": round(mean_signed, 4),
        # Secondary / diagnostic: integrated signed speed (body-lengths).
        "net_disp": round(float(net_disp), 4),
        "path_len": round(float(path_len), 4),
        "hip_net_frac": round(float(hip_net_frac), 4),
        # Primary comparator: share of known frames labeled forward.
        "advance_score": round(
            (labels["forward"] / known) if known else 0.0, 4
        ),
        "retreat_score": round(
            (labels["backward"] / known) if known else 0.0, 4
        ),
    }


def _pick_spatial_aggressor(
    f1: Dict[str, Any],
    f2: Dict[str, Any],
    *,
    even_margin: float,
    still_floor: float,
    min_samples: int,
) -> str:
    """Who spent more of the window labeled forward (higher forward_frac)."""
    if f1["samples"] < min_samples and f2["samples"] < min_samples:
        return "unclear"
    m1 = float(f1.get("forward_frac", f1.get("advance_score", 0.0)))
    m2 = float(f2.get("forward_frac", f2.get("advance_score", 0.0)))
    # Neither advancing much of the time.
    if m1 < still_floor and m2 < still_floor:
        return "even"
    delta = m1 - m2
    scale = max(m1, m2, still_floor)
    if abs(delta) / scale <= even_margin:
        return "even"
    return "fencer1" if delta > 0 else "fencer2"


def score_pre_touch_aggressor(
    frames: List[Dict[str, Any]],
    light_frame: int,
    *,
    fps: float,
    window_sec: float = DEFAULT_PRE_TOUCH_WINDOW_SEC,
    end_cut_sec: float = DEFAULT_PRE_TOUCH_END_CUT_SEC,
    still_thr: float = 0.18,
    even_margin: float = DEFAULT_EVEN_MARGIN,
    still_floor: float = DEFAULT_STILL_FLOOR,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    # Deprecated kwargs kept so older callers don't break.
    advance_thr: float = 0.0,
    both_margin: float = 0.0,
) -> Dict[str, Any]:
    """
    Who spent more time advancing toward the opponent in
    [light - window, light - end_cut).

    Comparator = fraction of known overlay frames labeled forward.
    Not an attack/priority call.
    """
    del advance_thr, both_margin
    fps = float(fps) if fps and fps > 1e-3 else 30.0
    window_sec = float(window_sec) if window_sec and window_sec > 1e-6 else DEFAULT_PRE_TOUCH_WINDOW_SEC
    end_cut_sec = float(end_cut_sec) if end_cut_sec and end_cut_sec > 0.0 else 0.0
    if end_cut_sec >= window_sec:
        end_cut_sec = max(0.0, window_sec * 0.1)
    t_light = float(light_frame) / fps
    t0 = t_light - window_sec
    t1 = t_light - end_cut_sec

    # Prefer frame index when present (robust to minor fps drift).
    indexed = [row for row in frames if row.get("i") is not None]
    if indexed:
        i0 = max(0, int(light_frame) - int(round(window_sec * fps)))
        i1 = max(i0, int(light_frame) - int(round(end_cut_sec * fps)))
        rows = [row for row in indexed if i0 <= int(row["i"]) < i1]
    else:
        rows = [
            row
            for row in frames
            if t0 <= float(row.get("t") or 0.0) < t1
        ]

    f1 = _person_window_stats(rows, 0, still_thr, fps=fps)
    f2 = _person_window_stats(rows, 1, still_thr, fps=fps)
    aggressor = _pick_spatial_aggressor(
        f1,
        f2,
        even_margin=even_margin,
        still_floor=still_floor,
        min_samples=min_samples,
    )
    return {
        "aggressor": aggressor,
        "window_sec": round(window_sec, 3),
        "end_cut_sec": round(end_cut_sec, 3),
        "light_frame": int(light_frame),
        "light_t": round(t_light, 4),
        "sample_count": len(rows),
        "fencer1": f1,
        "fencer2": f2,
        "rationale": (
            f"Who advanced more of the time: forward-label share from the "
            f"footwork overlay, from {window_sec:.2f}s to {end_cut_sec:.2f}s "
            f"before the light (last {end_cut_sec:.2f}s excluded)."
        ),
    }


def annotate_pre_touch_aggressors(
    forward_back: Optional[Dict[str, Any]],
    touches: List[Dict[str, Any]],
    *,
    window_sec: float = DEFAULT_PRE_TOUCH_WINDOW_SEC,
    end_cut_sec: float = DEFAULT_PRE_TOUCH_END_CUT_SEC,
    even_margin: float = DEFAULT_EVEN_MARGIN,
    still_floor: float = DEFAULT_STILL_FLOOR,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    advance_thr: float = 0.0,
    both_margin: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """
    Score each touch against the bout-wide forward_back series.

    `touches` items need `touch` (id) and preferably `frame` (int). Frame is
    also parsed from the touch id (..._frameN).
    """
    del advance_thr, both_margin
    if not forward_back or not isinstance(forward_back, dict):
        return None
    frames = forward_back.get("frames")
    if not isinstance(frames, list) or not frames:
        return None

    fps = float(forward_back.get("fps") or 30.0)
    still_thr = float(forward_back.get("still_thr") or 0.18)

    by_touch: Dict[str, Dict[str, Any]] = {}
    counts = {
        "fencer1": 0,
        "fencer2": 0,
        "even": 0,
        "unclear": 0,
    }
    for item in touches or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("touch") or item.get("name") or "")
        if not name:
            continue
        frame = item.get("frame")
        if frame is None:
            frame = _touch_frame_from_name(name)
        if frame is None:
            continue
        try:
            light_frame = int(frame)
        except (TypeError, ValueError):
            continue
        scored = score_pre_touch_aggressor(
            frames,
            light_frame,
            fps=fps,
            window_sec=window_sec,
            end_cut_sec=end_cut_sec,
            still_thr=still_thr,
            even_margin=even_margin,
            still_floor=still_floor,
            min_samples=min_samples,
        )
        scored["scoring_fencer"] = item.get("fencer") or item.get("scoring_fencer")
        by_touch[name] = scored
        key = scored["aggressor"]
        if key in counts:
            counts[key] += 1

    used = sum(counts.values())
    if used == 0:
        return None

    if counts["fencer1"] > counts["fencer2"]:
        main = "fencer1"
    elif counts["fencer2"] > counts["fencer1"]:
        main = "fencer2"
    else:
        main = "even"

    return {
        "window_sec": float(window_sec),
        "end_cut_sec": float(end_cut_sec),
        "even_margin": float(even_margin),
        "still_floor": float(still_floor),
        "by_touch": by_touch,
        "fencer1_pre_touch_aggression": counts["fencer1"],
        "fencer2_pre_touch_aggression": counts["fencer2"],
        "even": counts["even"],
        "unclear": counts["unclear"],
        # Back-compat keys for older UI snippets
        "both_advancing": 0,
        "neither_advancing": counts["even"],
        "touches_scored": used,
        "main_footwork_aggressor": main,
        "rationale": (
            "Spatial aggressor = who spent more of the pre-touch window labeled "
            f"forward (overlay time share) from {window_sec:.2f}s to "
            f"{end_cut_sec:.2f}s before each light (last {end_cut_sec:.2f}s cut "
            "to avoid touch-action noise). Bout main uses sole-winner touch counts."
        ),
    }
