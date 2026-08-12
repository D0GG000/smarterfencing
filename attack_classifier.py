"""
Attack-type classifier inference (lunge / fleche / other) from scorer 3D pose geometry.

Matches annotate_attack/TrainingAttack3D.py winrobust checkpoint
(best_attack_3d_proximity_winrobust.pth): 28-D local pose + 15-D motion,
BiGRU with temporal attention.

Extract may still include post-light frames on older jobs. Attack scoring
truncates through the light frame, then keeps the last ATTACK_WINDOW_SEC
of real time (default 0.5s) and resamples to 30 steps.

App layout:
  OUTPUT_3D/{job_id}/{touch}_3d.json
  OUTPUT_2D/{job_id}/{touch}/frame_*_keypoints.json
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose_paths import attack_classifier_default_path

CLASSES = ["lunge", "fleche", "other"]
POSE_DIM = 28
MOTION_DIM = 15
INPUT_DIM = POSE_DIM + MOTION_DIM
DEFAULT_NUM_FRAMES = 30
DEFAULT_ATTACK_WINDOW_SEC = float(os.environ.get("ATTACK_WINDOW_SEC", "0.5"))
DEFAULT_ATTACK_FPS = float(os.environ.get("ATTACK_ASSUME_FPS", "30"))
# Legacy extracts used touch..+5; drop this many trailing frames when light_frame_seq
# is missing. New extracts end on the light (see demo.extract_frames_before_touch).
DEFAULT_POST_LIGHT_FRAMES = int(os.environ.get("ATTACK_POST_LIGHT_FRAMES", "5"))

J = {
    "pelvis": 0,
    "right_hip": 1,
    "right_knee": 2,
    "right_ankle": 3,
    "left_hip": 4,
    "left_knee": 5,
    "left_ankle": 6,
    "spine": 7,
    "thorax": 8,
    "neck": 9,
    "head": 10,
    "left_shoulder": 11,
    "left_elbow": 12,
    "left_wrist": 13,
    "right_shoulder": 14,
    "right_elbow": 15,
    "right_wrist": 16,
}


def default_model_path() -> str:
    return attack_classifier_default_path()


def _pt(kp: dict, key: str) -> Tuple[float, float]:
    try:
        p = kp[key]
        return float(p[0]), float(p[1])
    except (KeyError, TypeError, IndexError, ValueError):
        return 0.0, 0.0


def _mid(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def infer_weapon_side_frame(s2d: dict, t2d: dict) -> Optional[str]:
    lw = _pt(s2d, "left_wrist")
    rw = _pt(s2d, "right_wrist")
    if lw == (0.0, 0.0) and rw == (0.0, 0.0):
        return None
    sh = _mid(_pt(s2d, "left_hip"), _pt(s2d, "right_hip"))
    dh = _mid(_pt(t2d, "left_hip"), _pt(t2d, "right_hip"))
    if abs(dh[0] - sh[0]) < 1e-3:
        return "left" if _dist(lw, dh) <= _dist(rw, dh) else "right"
    if dh[0] > sh[0]:
        return "right" if rw[0] > lw[0] else "left"
    return "left" if lw[0] < rw[0] else "right"


def infer_weapon_side_clip(pairs: List[Tuple[dict, dict]]) -> str:
    votes = {"left": 0, "right": 0}
    for s2d, t2d in pairs:
        side = infer_weapon_side_frame(s2d, t2d)
        if side:
            votes[side] += 1
    if votes["left"] == votes["right"]:
        left_d, right_d, n = 0.0, 0.0, 0
        for s2d, t2d in pairs:
            dh = _mid(_pt(t2d, "left_hip"), _pt(t2d, "right_hip"))
            lw, rw = _pt(s2d, "left_wrist"), _pt(s2d, "right_wrist")
            left_d += _dist(lw, dh)
            right_d += _dist(rw, dh)
            n += 1
        if n and left_d <= right_d:
            return "left"
        return "right"
    return "left" if votes["left"] > votes["right"] else "right"


def angle3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


def normalize_3d(seq: np.ndarray) -> np.ndarray:
    scales = [np.linalg.norm(f[J["left_knee"]] - f[J["left_shoulder"]]) + 1e-6 for f in seq]
    return seq / np.median(scales)


def center_pelvis(seq: np.ndarray) -> np.ndarray:
    out = seq.copy()
    for i in range(len(out)):
        out[i] = out[i] - out[i, J["pelvis"]]
    return out


def pelvis_trajectory(seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pel = seq[:, J["pelvis"], :].astype(np.float32)
    from_start = pel - pel[0:1]
    vel = np.zeros_like(from_start)
    if len(pel) > 1:
        vel[1:] = pel[1:] - pel[:-1]
    return from_start, vel


def _side_joints(side: str) -> Dict[str, str]:
    if side == "left":
        return {
            "shoulder": "left_shoulder",
            "elbow": "left_elbow",
            "wrist": "left_wrist",
            "hip": "left_hip",
            "knee": "left_knee",
            "ankle": "left_ankle",
        }
    return {
        "shoulder": "right_shoulder",
        "elbow": "right_elbow",
        "wrist": "right_wrist",
        "hip": "right_hip",
        "knee": "right_knee",
        "ankle": "right_ankle",
    }


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.zeros(3, dtype=np.float32)
    return (v / n).astype(np.float32)


def frame_features(frame: np.ndarray, weapon_side: str) -> List[float]:
    wj = _side_joints(weapon_side)
    oj = _side_joints("left" if weapon_side == "right" else "right")

    def g(name: str) -> np.ndarray:
        return frame[J[name]]

    pelvis = g("pelvis")
    thorax = g("thorax")
    l_hip, r_hip = g("left_hip"), g("right_hip")
    l_knee, r_knee = g("left_knee"), g("right_knee")
    l_ankle, r_ankle = g("left_ankle"), g("right_ankle")
    w_wrist = g(wj["wrist"])
    w_shoulder = g(wj["shoulder"])
    o_shoulder = g(oj["shoulder"])
    w_elbow = g(wj["elbow"])
    o_elbow = g(oj["elbow"])
    w_ankle = g(wj["ankle"])
    o_ankle = g(oj["ankle"])

    weapon_arm = angle3(w_shoulder, w_elbow, w_wrist)
    off_arm = angle3(o_shoulder, o_elbow, g(oj["wrist"]))
    left_knee = angle3(l_hip, l_knee, l_ankle)
    right_knee = angle3(r_hip, r_knee, r_ankle)
    weapon_knee = angle3(g(wj["hip"]), g(wj["knee"]), w_ankle)
    off_knee = angle3(g(oj["hip"]), g(oj["knee"]), o_ankle)
    left_hip_flex = angle3(g("left_shoulder"), l_hip, l_knee)
    right_hip_flex = angle3(g("right_shoulder"), r_hip, r_knee)
    torso = angle3(l_hip, pelvis, thorax)
    head_tilt = angle3(thorax, g("neck"), g("head"))
    up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    spine = thorax - pelvis
    spine_n = _unit(spine)
    torso_lean = float(np.arccos(np.clip(np.dot(spine_n, up), -1.0, 1.0)))

    fwd = np.array([w_wrist[0] - pelvis[0], 0.0, w_wrist[2] - pelvis[2]], dtype=np.float32)
    if float(np.linalg.norm(fwd)) < 1e-4:
        fwd = np.array([w_ankle[0] - o_ankle[0], 0.0, w_ankle[2] - o_ankle[2]], dtype=np.float32)
    fwd = _unit(fwd)

    l_proj = float(np.dot(l_ankle - pelvis, fwd))
    r_proj = float(np.dot(r_ankle - pelvis, fwd))
    if l_proj >= r_proj:
        front_knee, rear_knee = left_knee, right_knee
        front_hip_flex, rear_hip_flex = left_hip_flex, right_hip_flex
        front_ankle, rear_ankle = l_ankle, r_ankle
    else:
        front_knee, rear_knee = right_knee, left_knee
        front_hip_flex, rear_hip_flex = right_hip_flex, left_hip_flex
        front_ankle, rear_ankle = r_ankle, l_ankle

    front_knee_bend = front_knee
    rear_knee_extend = rear_knee
    knee_bend_asym = front_knee - rear_knee

    ankle_spread = float(np.linalg.norm(l_ankle - r_ankle))
    ankle_horiz_sep = float(math.hypot(l_ankle[0] - r_ankle[0], l_ankle[2] - r_ankle[2]))
    ankle_fwd_sep = float(np.dot(front_ankle - rear_ankle, fwd))
    hip_width = float(np.linalg.norm(l_hip - r_hip)) + 1e-6
    hip_lat = float(l_hip[0] - r_hip[0])
    ankle_lat = float(l_ankle[0] - r_ankle[0])
    ankle_lat_ratio = float(ankle_lat / (abs(hip_lat) + 1e-6))
    feet_near_stack = float(1.0 / (1.0 + abs(ankle_lat_ratio)))
    feet_together = float(1.0 / (1.0 + ankle_horiz_sep / hip_width))
    narrow_stance = float(ankle_horiz_sep / hip_width)

    weapon_reach = float(np.linalg.norm(w_wrist - w_shoulder))
    weapon_wrist_h = float(w_wrist[1] - pelvis[1])
    weapon_wrist_fwd = float(np.dot(w_wrist - pelvis, fwd))
    weapon_ankle_fwd = float(np.dot(w_ankle - pelvis, fwd))
    spine_len = float(np.linalg.norm(spine))
    wrist_from_pelvis = float(np.linalg.norm(w_wrist - pelvis))

    return [
        weapon_arm,
        off_arm,
        weapon_knee,
        off_knee,
        left_knee,
        right_knee,
        front_knee_bend,
        rear_knee_extend,
        knee_bend_asym,
        left_hip_flex,
        right_hip_flex,
        front_hip_flex,
        rear_hip_flex,
        torso,
        head_tilt,
        torso_lean,
        weapon_reach,
        weapon_wrist_h,
        weapon_wrist_fwd,
        weapon_ankle_fwd,
        ankle_spread,
        ankle_horiz_sep,
        ankle_fwd_sep,
        feet_near_stack,
        feet_together,
        narrow_stance,
        spine_len,
        wrist_from_pelvis,
    ]


def _scorer_target_names(touch_folder: str, meta: dict) -> Tuple[str, str]:
    sf = str(meta.get("scoring_fencer", "")).lower()
    if sf == "fencer1":
        return "fencer1", "fencer2"
    if sf == "fencer2":
        return "fencer2", "fencer1"
    uid = os.path.basename(touch_folder).lower()
    if uid.startswith("fencer2"):
        return "fencer2", "fencer1"
    return "fencer1", "fencer2"


def _resolve_attack_fps(meta: dict, data3d: dict) -> float:
    for src in (meta, data3d):
        if not isinstance(src, dict):
            continue
        for key in ("fps", "video_fps", "source_fps"):
            try:
                v = float(src.get(key))
                if v > 1.0:
                    return v
            except (TypeError, ValueError):
                pass
    return DEFAULT_ATTACK_FPS


def _temporal_resample(seq: np.ndarray, num_frames: int) -> np.ndarray:
    """Fit a T-frame sequence to num_frames by index resampling (repeat/skip)."""
    n = len(seq)
    if n <= 0:
        raise ValueError("empty sequence")
    if n == num_frames:
        return seq
    idxs = np.round(np.linspace(0, n - 1, num=num_frames)).astype(int)
    return seq[idxs]


def _attack_keep_count(fps: float, window_sec: float, num_frames: int) -> int:
    """How many trailing source frames to keep before resampling to num_frames.

    Do not cap at num_frames: a 1.0s window @ 30fps is 30 source frames that
    still resample to num_frames. Capping hid longer real-time context.
    """
    del num_frames  # kept for call-site compatibility
    fps = float(fps) if fps and fps > 1e-3 else DEFAULT_ATTACK_FPS
    window_sec = (
        float(window_sec) if window_sec and window_sec > 1e-6 else DEFAULT_ATTACK_WINDOW_SEC
    )
    keep = int(round(window_sec * fps))
    return max(1, keep)


# Trailing windows (seconds) scanned at inference to find the attack motion
# even when the extract includes extra prep/recovery frames.
DEFAULT_SEARCH_WINDOWS_SEC = (0.40, 0.50, 0.60, 0.75, 1.00)


def _truncate_seq_to_light(
    scorer_seq: np.ndarray,
    kp_files: List[str],
    meta: dict,
    post_light_frames: int = DEFAULT_POST_LIGHT_FRAMES,
) -> Tuple[np.ndarray, List[str]]:
    """Keep frames through the light only (drop post-light recovery)."""
    n = len(scorer_seq)
    if n < 1:
        return scorer_seq, kp_files

    light_seq = None
    try:
        v = meta.get("light_frame_seq")
        if v is not None:
            light_seq = int(v)
    except (TypeError, ValueError):
        light_seq = None

    if light_seq is not None and 1 <= light_seq <= n:
        cut = light_seq
    elif post_light_frames > 0 and n > post_light_frames:
        cut = n - post_light_frames
    else:
        cut = n

    scorer_seq = scorer_seq[:cut]
    if kp_files and len(kp_files) > cut:
        kp_files = kp_files[:cut]
    return scorer_seq, kp_files


def load_attack_3d_batch(
    path_3d: str,
    touch_2d_folder: str,
    num_frames: int = DEFAULT_NUM_FRAMES,
    window_sec: Optional[float] = None,
    fps: Optional[float] = None,
) -> np.ndarray:
    with open(path_3d, "r", encoding="utf-8") as f:
        data3d = json.load(f)

    meta_path = os.path.join(touch_2d_folder, "frame_1_keypoints.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta0 = json.load(f)

    scorer_name, target_name = _scorer_target_names(touch_2d_folder, meta0)
    scorer_key = f"{scorer_name}_keypoints"
    target_key = f"{target_name}_keypoints"

    scorer_seq = np.array(data3d[f"{scorer_name}_3d"], dtype=np.float32)

    kp_files = sorted(
        [
            os.path.join(touch_2d_folder, name)
            for name in os.listdir(touch_2d_folder)
            if name.startswith("frame_") and name.endswith("_keypoints.json")
        ],
        key=lambda p: int(os.path.basename(p).split("_")[1]),
    )

    # Drop post-light frames before the real-time attack window.
    scorer_seq, kp_files = _truncate_seq_to_light(scorer_seq, kp_files, meta0)
    n = len(scorer_seq)

    use_fps = float(fps) if fps is not None else _resolve_attack_fps(meta0, data3d)
    use_window = (
        float(window_sec) if window_sec is not None else DEFAULT_ATTACK_WINDOW_SEC
    )
    keep = _attack_keep_count(use_fps, use_window, num_frames)
    keep = min(keep, n) if n else keep

    # Trailing ~window_sec of pre-light+light, then resample to num_frames.
    if n > keep:
        scorer_seq = scorer_seq[-keep:]
    scorer_seq = _temporal_resample(scorer_seq, num_frames)

    if len(kp_files) > keep:
        kp_files = kp_files[-keep:]
    if kp_files and len(kp_files) != num_frames:
        idxs = np.round(np.linspace(0, len(kp_files) - 1, num=num_frames)).astype(int)
        kp_files = [kp_files[i] for i in idxs]

    pairs: List[Tuple[dict, dict]] = []
    for jpath in kp_files:
        with open(jpath, "r", encoding="utf-8") as f:
            d2d = json.load(f)
        if scorer_key in d2d and target_key in d2d:
            pairs.append((d2d[scorer_key], d2d[target_key]))

    weapon_side = infer_weapon_side_clip(pairs) if pairs else "right"

    scorer_seq = normalize_3d(scorer_seq)
    pel_from_start, pel_vel = pelvis_trajectory(scorer_seq)
    local_seq = center_pelvis(scorer_seq)

    pose_rows: List[List[float]] = []
    for i in range(num_frames):
        pose_rows.append(frame_features(local_seq[i], weapon_side))
    pose = np.array(pose_rows, dtype=np.float32)

    delta_idx = [6, 8, 11, 16, 21, 22]
    pose_delta = np.zeros((pose.shape[0], len(delta_idx)), dtype=np.float32)
    pose_delta[1:] = pose[1:, delta_idx] - pose[:-1, delta_idx]

    ankle_horiz = pose[:, 21]
    ankle_fwd = pose[:, 22]
    clip_min_horiz = float(np.min(ankle_horiz))
    clip_min_fwd = float(np.min(ankle_fwd))
    clip_close_amount = float(max(0.0, float(ankle_horiz[0]) - clip_min_horiz))
    clip_foot = np.tile(
        np.array([clip_min_horiz, clip_min_fwd, clip_close_amount], dtype=np.float32),
        (pose.shape[0], 1),
    )

    motion = np.concatenate(
        [pel_from_start, pel_vel, pose_delta, clip_foot],
        axis=-1,
    ).astype(np.float32)
    return np.concatenate([pose, motion], axis=-1)


class Attack3DClassifier(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, num_classes: int = len(CLASSES), dropout: float = 0.35):
        super().__init__()
        self.embed = nn.Linear(input_dim, 64)
        self.gru = nn.GRU(
            64,
            48,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.attn = nn.Linear(96, 1)
        self.fc = nn.Sequential(
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(48, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.embed(x))
        h, _ = self.gru(h)
        w = F.softmax(self.attn(h), dim=1)
        pooled = torch.sum(w * h, dim=1)
        return self.fc(pooled)


@dataclass
class AttackPrediction:
    label: str
    class_index: int
    confidence: float
    probabilities: Tuple[float, ...]


class AttackClassifier:
    def __init__(self, weights_path: str, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        try:
            ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(weights_path, map_location=self.device)
        self.classes = list(ckpt.get("classes", CLASSES))
        self.num_frames = int(ckpt.get("num_frames", DEFAULT_NUM_FRAMES))
        input_dim = int(ckpt.get("input_dim", INPUT_DIM))
        self.model = Attack3DClassifier(input_dim=input_dim, num_classes=len(self.classes)).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, path_3d: str, path_2d: str) -> AttackPrediction:
        """Single trailing ATTACK_WINDOW_SEC crop (default 0.5s)."""
        x = load_attack_3d_batch(path_3d, path_2d, num_frames=self.num_frames)
        logits = self.model(torch.FloatTensor(x).unsqueeze(0).to(self.device))
        probs = F.softmax(logits, dim=1)[0]
        idx = int(probs.argmax().item())
        return AttackPrediction(
            label=self.classes[idx],
            class_index=idx,
            confidence=float(probs[idx].item()),
            probabilities=tuple(float(probs[i].item()) for i in range(len(self.classes))),
        )

    @torch.no_grad()
    def predict_window(
        self,
        path_3d: str,
        path_2d: str,
        window_sec: float,
    ) -> AttackPrediction:
        x = load_attack_3d_batch(
            path_3d, path_2d, num_frames=self.num_frames, window_sec=window_sec
        )
        logits = self.model(torch.FloatTensor(x).unsqueeze(0).to(self.device))
        probs = F.softmax(logits, dim=1)[0]
        idx = int(probs.argmax().item())
        return AttackPrediction(
            label=self.classes[idx],
            class_index=idx,
            confidence=float(probs[idx].item()),
            probabilities=tuple(float(probs[i].item()) for i in range(len(self.classes))),
        )

    @torch.no_grad()
    def predict_search(
        self,
        path_3d: str,
        path_2d: str,
        windows_sec: Optional[Tuple[float, ...]] = None,
    ) -> AttackPrediction:
        """Score several trailing windows; pick the one with strongest attack evidence.

        Uses max(p_lunge, p_fleche) so extra prep frames in a long crop do not
        force a single bad window — a shorter window can still surface the motion.
        """
        windows = windows_sec or DEFAULT_SEARCH_WINDOWS_SEC
        best: Optional[AttackPrediction] = None
        best_attack = -1.0
        for w in windows:
            pred = self.predict_window(path_3d, path_2d, float(w))
            # classes: lunge=0, fleche=1, other=2
            attack_score = max(pred.probabilities[0], pred.probabilities[1])
            if attack_score > best_attack:
                best_attack = attack_score
                best = pred
        assert best is not None
        return best
