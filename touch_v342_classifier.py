"""
Touch classifier v3.42 inference for the app pipeline.

v3.42 = v3.40 masked_late vision + expanded relative geometry (58-D),
with auto weapon-arm side from 2D pose (default).

Expects per-touch folders with frame_*.jpg and frame_*_keypoints.json
from fencing-18 pose (bellguard keypoint included).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_CLASSES = ["chest", "abdomen", "arm", "leg"]
_NUM_FRAMES = 8
_IMG_SIZE = 224
_MASK_GRAY = 0.45
_GEOM_DIM = 58
_SIDE_X_MARGIN_SCALE = 1.45

DEFENDER_GEOM_KEYS = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
DEFENDER_BOX_KEYS = [
    "left_shoulder", "right_shoulder", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def default_model_path() -> str:
    return os.environ.get(
        "MODEL_PATH",
        os.path.join(_APP_DIR, "best_touch_v342_expanded_autoweapon_masked_late.pth"),
    )


def _resolve_touch_folder(path: str) -> str:
    bn = os.path.basename(path).lower()
    if bn.endswith("_keypoints.json") and bn.startswith("frame_"):
        return os.path.dirname(path)
    fn = os.path.basename(path)
    uid = fn.replace("_3d.json", "").replace("_3D.json", "")
    touchtype3d_dir = os.path.dirname(os.path.dirname(path))
    dataset_root = os.path.dirname(touchtype3d_dir)
    touch_type_3d = os.path.basename(touchtype3d_dir).lower()
    touch_type_2d = touch_type_3d.replace("3d", "2d")
    video_name = os.path.basename(os.path.dirname(path))
    return os.path.join(dataset_root, touch_type_2d, video_name, uid)


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


def _pt(kp: dict, key: str) -> Tuple[float, float]:
    try:
        p = kp[key]
        return float(p[0]), float(p[1])
    except (KeyError, TypeError, IndexError, ValueError):
        return 0.0, 0.0


def _mid(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def _clamp_box(x1, y1, x2, y2, img_w, img_h) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(img_w - 1, x1)))
    y1 = int(max(0, min(img_h - 1, y1)))
    x2 = int(max(0, min(img_w, x2)))
    y2 = int(max(0, min(img_h, y2)))
    if x2 <= x1:
        x2 = min(img_w, x1 + 2)
    if y2 <= y1:
        y2 = min(img_h, y1 + 2)
    return x1, y1, x2, y2


def _points_bbox(pts, pad_x, pad_y, img_w, img_h):
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return _clamp_box(min(xs) - pad_x, min(ys) - pad_y, max(xs) + pad_x, max(ys) + pad_y, img_w, img_h)


def _union_boxes(a, b, img_w, img_h):
    if a is None:
        return b
    if b is None:
        return a
    return _clamp_box(min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]), img_w, img_h)


def build_weapon_defender_boxes(s2d, t2d, weapon_side, scorer_wrist_key, def_height, img_w, img_h):
    dh = max(def_height, 1.0)
    dpts = []
    for k in DEFENDER_BOX_KEYS:
        if k in t2d:
            dpts.append((float(t2d[k][0]), float(t2d[k][1])))
    pad_d_x = max(28.0, 0.14 * dh) * _SIDE_X_MARGIN_SCALE
    pad_d_y = max(24.0, 0.12 * dh)
    def_box = _points_bbox(dpts, pad_d_x, pad_d_y, img_w, img_h)

    elbow_k = "right_elbow" if str(weapon_side).lower() == "right" else "left_elbow"
    wpts = []
    for k in (elbow_k, scorer_wrist_key):
        if k in s2d:
            wpts.append((float(s2d[k][0]), float(s2d[k][1])))
    if "bellguard" in s2d:
        wpts.append((float(s2d["bellguard"][0]), float(s2d["bellguard"][1])))
    slack = 0.22 * dh
    pad_w_x = (max(72.0, 0.48 * dh) + slack) * _SIDE_X_MARGIN_SCALE
    pad_w_y = max(64.0, 0.40 * dh) + slack
    wep_box = _points_bbox(wpts, pad_w_x, pad_w_y, img_w, img_h)
    if wep_box is None and scorer_wrist_key in s2d:
        wx, wy = float(s2d[scorer_wrist_key][0]), float(s2d[scorer_wrist_key][1])
        r = max(80.0, 0.45 * dh)
        wep_box = _clamp_box(wx - r * _SIDE_X_MARGIN_SCALE, wy - r, wx + r * _SIDE_X_MARGIN_SCALE, wy + r, img_w, img_h)
    return wep_box, def_box, _union_boxes(wep_box, def_box, img_w, img_h)


def _apply_interaction_mask(crop_rgb, roi, wep_box, def_box, mask_mode: str):
    if mask_mode == "none":
        return crop_rgb
    x1, y1, x2, y2 = roi
    ch, cw = crop_rgb.shape[:2]
    mask = np.zeros((ch, cw), dtype=np.float32)
    for box in (wep_box, def_box):
        if box is None:
            continue
        bx1 = max(0, int(box[0]) - x1)
        by1 = max(0, int(box[1]) - y1)
        bx2 = min(cw, int(box[2]) - x1)
        by2 = min(ch, int(box[3]) - y1)
        if bx2 > bx1 and by2 > by1:
            mask[by1:by2, bx1:bx2] = 1.0
    if mask.max() < 0.5:
        return crop_rgb
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=3.0, sigmaY=3.0)
    mask = np.clip(mask, 0.0, 1.0)[..., None]
    gray = np.full_like(crop_rgb, int(_MASK_GRAY * 255), dtype=np.uint8)
    out = crop_rgb.astype(np.float32) * mask + gray.astype(np.float32) * (1.0 - mask)
    return np.clip(out, 0, 255).astype(np.uint8)


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


def build_expanded_touch_geom(t2d: dict, s2d: dict, weapon_side: str) -> List[float]:
    ls = _pt(t2d, "left_shoulder")
    rs = _pt(t2d, "right_shoulder")
    la = _pt(t2d, "left_ankle")
    ra = _pt(t2d, "right_ankle")
    hip = _mid(_pt(t2d, "left_hip"), _pt(t2d, "right_hip"))
    shoulder = _mid(ls, rs)
    ankle = _mid(la, ra)
    torso_h = max(_dist(shoulder, ankle), 1.0)

    def norm(p):
        return ((p[0] - hip[0]) / torso_h, (p[1] - hip[1]) / torso_h)

    weapon_side = "right" if weapon_side == "right" else "left"
    off_side = "left" if weapon_side == "right" else "right"
    wrist_key = f"{weapon_side}_wrist"
    sw = norm(_pt(s2d, wrist_key))
    chest = norm(shoulder)
    abdomen = norm(hip)

    row: List[float] = []
    for k in DEFENDER_GEOM_KEYS:
        row.extend(norm(_pt(t2d, k)))
    for k in (f"{weapon_side}_shoulder", f"{weapon_side}_elbow", wrist_key):
        row.extend(norm(_pt(s2d, k)))
    if "bellguard" in s2d:
        row.extend(norm(_pt(s2d, "bellguard")))
    else:
        row.extend(sw)
    row.extend(norm(_pt(s2d, f"{off_side}_shoulder")))
    for k in (f"{weapon_side}_hip", f"{weapon_side}_knee", f"{weapon_side}_ankle"):
        row.extend(norm(_pt(s2d, k)))

    zone_dists = [_dist(sw, chest), _dist(sw, abdomen)]
    for k in DEFENDER_GEOM_KEYS:
        zone_dists.append(_dist(sw, norm(_pt(t2d, k))))
    row.extend(zone_dists)

    arm_keys = ("left_elbow", "right_elbow", "left_wrist", "right_wrist")
    leg_keys = ("left_knee", "right_knee", "left_ankle", "right_ankle")
    row.append(min(_dist(sw, norm(_pt(t2d, k))) for k in arm_keys))
    row.append(min(_dist(sw, norm(_pt(t2d, k))) for k in leg_keys))
    row.append(sw[1] - chest[1])
    row.append(sw[0] - chest[0])
    assert len(row) == _GEOM_DIM
    return row


def _iter_touch_frames(touch_folder: str, scorer_key: str, target_key: str):
    pairs = []
    for i in range(1, 31):
        jpath = os.path.join(touch_folder, f"frame_{i}_keypoints.json")
        if i > 1 and not os.path.exists(jpath):
            break
        if not os.path.exists(jpath):
            break
        ipath = os.path.join(touch_folder, f"frame_{i}.jpg")
        if not os.path.isfile(ipath):
            continue
        img_bgr = cv2.imread(ipath)
        if img_bgr is None:
            continue
        import json

        with open(jpath, "r", encoding="utf-8") as f:
            d2d = json.load(f)
        if scorer_key not in d2d or target_key not in d2d:
            continue
        pairs.append((i, d2d[scorer_key], d2d[target_key], img_bgr))
    return pairs


def load_expanded_maskedcrop_batch(
    path: str,
    num_frames: int = _NUM_FRAMES,
    mask_mode: str = "interaction",
) -> Tuple[np.ndarray, np.ndarray]:
    touch_folder = _resolve_touch_folder(path)
    anchor = os.path.join(touch_folder, "frame_1_keypoints.json")
    import json

    with open(anchor, "r", encoding="utf-8") as f:
        meta0 = json.load(f)
    scorer_name, target_name = _scorer_target_names(touch_folder, meta0)
    scorer_key = f"{scorer_name}_keypoints"
    target_key = f"{target_name}_keypoints"

    raw = _iter_touch_frames(touch_folder, scorer_key, target_key)
    if not raw:
        return (
            np.zeros((num_frames, _GEOM_DIM), dtype=np.float32),
            np.zeros((num_frames, 3, _IMG_SIZE, _IMG_SIZE), dtype=np.float32),
        )

    weapon_side = infer_weapon_side_clip([(s, t) for _, s, t, _ in raw])
    scorer_wrist_key = f"{weapon_side}_wrist"

    geom_rows: List[List[float]] = []
    timeline: List[int] = []
    last_geom = None
    for fi, s2d, t2d, _ in raw:
        if scorer_wrist_key not in s2d:
            if last_geom is not None:
                geom_rows.append(last_geom.copy())
                timeline.append(fi)
            continue
        row = build_expanded_touch_geom(t2d, s2d, weapon_side)
        last_geom = row
        geom_rows.append(row)
        timeline.append(fi)

    while len(geom_rows) < 30 and last_geom is not None:
        geom_rows.append(last_geom.copy())
        timeline.append(timeline[-1])

    geom_rows = geom_rows[-num_frames:]
    timeline = timeline[-num_frames:]
    if len(geom_rows) < num_frames and last_geom is not None:
        pad = num_frames - len(geom_rows)
        geom_rows = [last_geom.copy()] * pad + geom_rows
        timeline = [timeline[0] if timeline else 1] * pad + timeline

    geom = np.array(geom_rows, dtype=np.float32)
    imgs = np.zeros((num_frames, 3, _IMG_SIZE, _IMG_SIZE), dtype=np.float32)
    frame_by_i = {fi: (s2d, t2d, img) for fi, s2d, t2d, img in raw}

    for t_idx, fi in enumerate(timeline):
        if fi not in frame_by_i:
            continue
        s2d, t2d, img_bgr = frame_by_i[fi]
        if scorer_wrist_key not in s2d:
            continue
        h_img, w_img = img_bgr.shape[:2]
        try:
            def_ank_y = (t2d["left_ankle"][1] + t2d["right_ankle"][1]) / 2
            def_sh_y = (t2d["left_shoulder"][1] + t2d["right_shoulder"][1]) / 2
            def_h = abs(def_sh_y - def_ank_y) + 1e-6
        except (KeyError, TypeError):
            continue
        wep_box, def_box, roi = build_weapon_defender_boxes(
            s2d, t2d, weapon_side, scorer_wrist_key, def_h, w_img, h_img,
        )
        if roi is None:
            continue
        x1, y1, x2, y2 = roi
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        ch, cw = crop_rgb.shape[:2]
        max_crop_side = 512
        scale = 1.0
        if max(ch, cw) > max_crop_side:
            scale = max_crop_side / float(max(ch, cw))
            nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
            crop_rgb = cv2.resize(crop_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
            ch, cw = crop_rgb.shape[:2]

        def _box_in_crop(box):
            if box is None:
                return None
            return (
                int((box[0] - x1) * scale),
                int((box[1] - y1) * scale),
                int((box[2] - x1) * scale),
                int((box[3] - y1) * scale),
            )

        crop_rgb = _apply_interaction_mask(
            crop_rgb, (0, 0, cw, ch), _box_in_crop(wep_box), _box_in_crop(def_box), mask_mode,
        )
        crop_resized = cv2.resize(crop_rgb, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        imgs[t_idx] = crop_resized.transpose(2, 0, 1).astype(np.float32) / 255.0

    return geom, imgs


class LateFusionClassifier(nn.Module):
    def __init__(self, geom_dim: int = _GEOM_DIM, num_classes: int = 4, geom_weight: float = 0.65, dropout: float = 0.35):
        super().__init__()
        self.geom_weight = geom_weight
        self.geom_enc = nn.Sequential(
            nn.Linear(geom_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout),
        )
        self.geom_gru = nn.GRU(64, 40, batch_first=True, bidirectional=True)
        self.geom_attn = nn.Linear(80, 1)
        self.geom_fc = nn.Sequential(nn.Linear(80, 48), nn.GELU(), nn.Dropout(dropout), nn.Linear(48, num_classes))

        from torchvision import models

        try:
            from torchvision.models import MobileNet_V3_Small_Weights as W
            backbone = models.mobilenet_v3_small(weights=W.IMAGENET1K_V1)
        except Exception:
            backbone = models.mobilenet_v3_small(pretrained=True)
        self.backbone = backbone.features
        for p in self.backbone.parameters():
            p.requires_grad = False
        self._freeze_backbone = True
        self.img_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_proj = nn.Linear(576, 48)
        self.img_gru = nn.GRU(48, 32, batch_first=True, bidirectional=True)
        self.img_attn = nn.Linear(64, 1)
        self.img_fc = nn.Sequential(
            nn.Linear(64, 48), nn.GELU(), nn.Dropout(dropout), nn.Linear(48, num_classes),
        )
        self.register_buffer("_imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self._freeze_backbone:
            self.backbone.eval()
        return self

    def _pool_gru(self, seq, gru, attn):
        h, _ = gru(seq)
        wv = F.softmax(attn(h), dim=1)
        return torch.sum(wv * h, dim=1)

    def forward_geom_only(self, geom):
        h = self.geom_enc(geom)
        return self.geom_fc(self._pool_gru(h, self.geom_gru, self.geom_attn))

    def forward_img_only(self, images):
        b, t, c, h, w = images.shape
        x = (images.view(b * t, c, h, w) - self._imagenet_mean) / self._imagenet_std
        z = self.img_proj(self.img_pool(self.backbone(x)).flatten(1)).view(b, t, -1)
        return self.img_fc(self._pool_gru(z, self.img_gru, self.img_attn))

    def forward(self, geom, images):
        lg = self.forward_geom_only(geom)
        li = self.forward_img_only(images)
        w = self.geom_weight
        return w * lg + (1.0 - w) * li


@dataclass
class TouchPrediction:
    label: str
    class_index: int
    confidence: float
    probabilities: Tuple[float, ...]


class TouchV342Classifier:
    def __init__(self, weights_path: str, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = LateFusionClassifier(geom_dim=_GEOM_DIM, geom_weight=0.65, dropout=0.35).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    @torch.no_grad()
    def predict_path(self, clip_path: str) -> TouchPrediction:
        geom_np, img_np = load_expanded_maskedcrop_batch(clip_path, num_frames=_NUM_FRAMES)
        geom = torch.FloatTensor(geom_np).unsqueeze(0).to(self.device)
        imgs = torch.FloatTensor(img_np).unsqueeze(0).to(self.device)
        logits = self.model(geom, imgs)
        probs = F.softmax(logits, dim=1)[0]
        idx = int(probs.argmax().item())
        return TouchPrediction(
            label=_CLASSES[idx],
            class_index=idx,
            confidence=float(probs[idx].item()),
            probabilities=tuple(float(probs[i].item()) for i in range(len(_CLASSES))),
        )
