"""
Track a user-drawn scoring-apparatus rectangle across panning camera motion.

Fencer 1 is always the left half of the apparatus; fencer 2 is the right half.

Primary: ORB feature matching (handles small angle / lighting changes).
Validator: template score in a tight window (rejects false jumps).
Spatial gates: scoreboard vertical band + max step size from last good lock.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]

TEMPLATE_GOOD_THRESH = 0.48
TEMPLATE_OK_THRESH = 0.36
RECOVERY_TEMPLATE_THRESH = 0.50
RECOVERY_CONF_TRIGGER = 0.2
ORB_MIN_GOOD_MATCHES = 8
ORB_RATIO = 0.75
SEARCH_RADIUS_MIN_PX = 56


def split_apparatus_to_lights(apparatus: Box) -> Tuple[Box, Box]:
    x1, y1, x2, y2 = apparatus
    mid = (x1 + x2) // 2
    return (x1, y1, mid, y2), (mid, y1, x2, y2)


def interpolate_box(frame_a: int, box_a: Box, frame_b: int, box_b: Box, frame_mid: int) -> Box:
    if frame_b <= frame_a:
        return box_a
    t = (frame_mid - frame_a) / float(frame_b - frame_a)
    t = max(0.0, min(1.0, t))
    out = []
    for i in range(4):
        out.append(int(round(box_a[i] + t * (box_b[i] - box_a[i]))))
    return tuple(out)  # type: ignore[return-value]


def _clamp_box(box: Box, frame_w: int, frame_h: int, min_size: int = 8) -> Box:
    x1, y1, x2, y2 = box
    w = max(min_size, x2 - x1)
    h = max(min_size, y2 - y1)
    x1 = max(0, min(frame_w - w, x1))
    y1 = max(0, min(frame_h - h, y1))
    return (int(x1), int(y1), int(x1 + w), int(y1 + h))


def _box_shift(a: Box, b: Box) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _box_size(box: Box) -> Tuple[int, int]:
    return box[2] - box[0], box[3] - box[1]


def _crop_gray(gray: np.ndarray, box: Box) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = box
    h, w = gray.shape[:2]
    x1c = max(0, min(w - 1, x1))
    y1c = max(0, min(h - 1, y1))
    x2c = max(x1c + 1, min(w, x2))
    y2c = max(y1c + 1, min(h, y2))
    patch = gray[y1c:y2c, x1c:x2c]
    return patch if patch.size else None


class ScoreLightTracker:
    """Follow scoring apparatus via ORB features + validated template matching."""

    def __init__(self, frame_bgr: np.ndarray, apparatus_box: Box):
        self.frame_h, self.frame_w = frame_bgr.shape[:2]
        self.initial_apparatus = _clamp_box(apparatus_box, self.frame_w, self.frame_h)
        self.current_apparatus = self.initial_apparatus
        self._anchor_apparatus = self.initial_apparatus
        self._anchor_confidence = 1.0
        self.confidence = 1.0
        self.last_tpl_score = 1.0
        self.last_orb_matches = 0
        self.last_method = "init"
        self._was_lights_on = False

        self._box_w, self._box_h = _box_size(self.initial_apparatus)
        self._search_radius = max(SEARCH_RADIUS_MIN_PX, int(self.frame_w * 0.12))
        self._max_step = max(28, int(self.frame_w * 0.045))
        self._max_recovery_shift = max(80, int(self.frame_w * 0.40))

        cy = (self.initial_apparatus[1] + self.initial_apparatus[3]) // 2
        y_slack = max(self._box_h, int(self.frame_h * 0.10))
        self._y_min = max(0, cy - y_slack)
        self._y_max = min(self.frame_h, cy + y_slack)

        gray = self._to_gray(frame_bgr)
        self._template = self._build_template(gray, self.initial_apparatus)
        self._orb = cv2.ORB_create(nfeatures=400, scaleFactor=1.2, nlevels=8, edgeThreshold=12)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._kp_ref, self._des_ref = self._extract_orb_patch(gray, self.initial_apparatus)

    @staticmethod
    def _to_gray(frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def _build_template(self, gray: np.ndarray, box: Box) -> Optional[np.ndarray]:
        patch = _crop_gray(gray, box)
        if patch is None or patch.shape[0] < 6 or patch.shape[1] < 6:
            return None
        edges = cv2.Canny(patch, 50, 150)
        return cv2.addWeighted(patch, 0.55, edges, 0.45, 0).astype(np.float32)

    def _extract_orb_patch(self, gray: np.ndarray, box: Box) -> Tuple[list, Optional[np.ndarray]]:
        margin = max(12, self._search_radius // 2)
        x1 = max(0, box[0] - margin)
        y1 = max(0, box[1] - margin)
        x2 = min(self.frame_w, box[2] + margin)
        y2 = min(self.frame_h, box[3] + margin)
        patch = gray[y1:y2, x1:x2]
        if patch.size == 0:
            return [], None
        kp, des = self._orb.detectAndCompute(patch, None)
        self._orb_patch_origin = (x1, y1)
        self._orb_box_origin = (box[0], box[1])
        return kp or [], des

    def _in_y_band(self, box: Box) -> bool:
        cy = (box[1] + box[3]) // 2
        return self._y_min <= cy <= self._y_max

    def _score_template_at(self, gray: np.ndarray, box: Box) -> float:
        if self._template is None:
            return 0.0
        th, tw = self._template.shape[:2]
        if box[2] - box[0] != tw or box[3] - box[1] != th:
            box = _clamp_box(box, self.frame_w, self.frame_h)
        patch = _crop_gray(gray, box)
        if patch is None or patch.shape[0] != th or patch.shape[1] != tw:
            return 0.0
        edges = cv2.Canny(patch, 50, 150)
        blend = cv2.addWeighted(patch, 0.55, edges, 0.45, 0).astype(np.float32)
        res = cv2.matchTemplate(blend, self._template, cv2.TM_CCOEFF_NORMED)
        return float(res[0, 0]) if res.size else 0.0

    def _match_template_near(
        self,
        gray: np.ndarray,
        hint: Box,
        *,
        radius_scale: float = 1.0,
    ) -> Tuple[Optional[Box], float]:
        if self._template is None:
            return None, 0.0
        th, tw = self._template.shape[:2]
        cx = (hint[0] + hint[2]) // 2
        cy = (hint[1] + hint[3]) // 2
        r = int(self._search_radius * radius_scale)
        sx1 = max(0, cx - r - tw // 2)
        sy1 = max(0, cy - r - th // 2)
        sx2 = min(self.frame_w, cx + r + tw // 2)
        sy2 = min(self.frame_h, cy + r + th // 2)
        if sx2 - sx1 <= tw or sy2 - sy1 <= th:
            return None, 0.0
        search = gray[sy1:sy2, sx1:sx2].astype(np.float32)
        res = cv2.matchTemplate(search, self._template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        box = (sx1 + max_loc[0], sy1 + max_loc[1], sx1 + max_loc[0] + tw, sy1 + max_loc[1] + th)
        return _clamp_box(box, self.frame_w, self.frame_h), float(max_val)

    def _track_orb(
        self,
        gray: np.ndarray,
        hint: Box,
        *,
        radius_scale: float = 1.0,
    ) -> Tuple[Optional[Box], int]:
        if self._des_ref is None or len(self._kp_ref) < ORB_MIN_GOOD_MATCHES:
            return None, 0

        cx = (hint[0] + hint[2]) // 2
        cy = (hint[1] + hint[3]) // 2
        r = int(self._search_radius * radius_scale) + max(self._box_w, self._box_h) // 2
        sx1 = max(0, cx - r)
        sy1 = max(0, cy - r)
        sx2 = min(self.frame_w, cx + r)
        sy2 = min(self.frame_h, cy + r)
        patch = gray[sy1:sy2, sx1:sx2]
        if patch.size == 0:
            return None, 0

        kp2, des2 = self._orb.detectAndCompute(patch, None)
        if des2 is None or len(kp2 or []) < ORB_MIN_GOOD_MATCHES:
            return None, 0

        pairs = self._matcher.knnMatch(self._des_ref, des2, k=2)
        good = []
        for pair in pairs:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ORB_RATIO * n.distance:
                good.append(m)

        if len(good) < ORB_MIN_GOOD_MATCHES:
            return None, len(good)

        dxs: List[float] = []
        dys: List[float] = []
        for m in good:
            ref_pt = self._kp_ref[m.queryIdx].pt
            cur_pt = kp2[m.trainIdx].pt
            ref_frame = (
                self._orb_patch_origin[0] + ref_pt[0],
                self._orb_patch_origin[1] + ref_pt[1],
            )
            cur_frame = (sx1 + cur_pt[0], sy1 + cur_pt[1])
            dxs.append(cur_frame[0] - ref_frame[0])
            dys.append(cur_frame[1] - ref_frame[1])

        dx = float(np.median(dxs))
        dy = float(np.median(dys))
        w, h = self._box_w, self._box_h
        ix1, iy1 = self.initial_apparatus[0], self.initial_apparatus[1]
        box = _clamp_box(
            (
                int(round(ix1 + dx)),
                int(round(iy1 + dy)),
                int(round(ix1 + dx + w)),
                int(round(iy1 + dy + h)),
            ),
            self.frame_w,
            self.frame_h,
        )
        return box, len(good)

    def _accept_candidate(
        self,
        gray: np.ndarray,
        candidate: Box,
        *,
        from_hint: Box,
        allow_large_shift: bool = False,
    ) -> Tuple[bool, float]:
        if not self._in_y_band(candidate):
            return False, 0.0
        step = _box_shift(candidate, from_hint)
        if not allow_large_shift and step > self._max_step:
            return False, 0.0
        if allow_large_shift and _box_shift(candidate, self._anchor_apparatus) > self._max_recovery_shift:
            return False, 0.0
        score = self._score_template_at(gray, candidate)
        return score >= TEMPLATE_OK_THRESH, score

    def _refresh_template(self, gray: np.ndarray, box: Box, score: float) -> None:
        if score < TEMPLATE_GOOD_THRESH:
            return
        new_tpl = self._build_template(gray, box)
        if new_tpl is None or self._template is None:
            return
        if new_tpl.shape != self._template.shape:
            return
        self._template = (0.88 * self._template + 0.12 * new_tpl).astype(np.float32)

    def _commit(self, box: Box, score: float, method: str, orb_matches: int = 0) -> None:
        self.current_apparatus = box
        self.confidence = score
        self.last_tpl_score = score
        self.last_orb_matches = orb_matches
        self.last_method = method
        if score >= 0.45:
            self._anchor_apparatus = box
            self._anchor_confidence = score

    def _search_in_scoreboard_band(self, gray: np.ndarray) -> Tuple[Optional[Box], float]:
        if self._template is None:
            return None, 0.0
        th, tw = self._template.shape[:2]
        sy1 = max(0, self._y_min - th // 2)
        sy2 = min(self.frame_h, self._y_max + th // 2)
        if sy2 - sy1 < th:
            return None, 0.0
        band = gray[sy1:sy2, :].astype(np.float32)
        res = cv2.matchTemplate(band, self._template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        box = (max_loc[0], sy1 + max_loc[1], max_loc[0] + tw, sy1 + max_loc[1] + th)
        return _clamp_box(box, self.frame_w, self.frame_h), float(max_val)

    def _update_recovery_mode(self, gray: np.ndarray) -> Box:
        best_box: Optional[Box] = None
        best_score = 0.0
        best_orb = 0

        orb_box, orb_n = self._track_orb(gray, self._anchor_apparatus, radius_scale=2.5)
        if orb_box is not None:
            ok, score = self._accept_candidate(
                gray, orb_box, from_hint=self.current_apparatus, allow_large_shift=True
            )
            if ok and score > best_score:
                best_box, best_score, best_orb = orb_box, score, orb_n

        tpl_box, tpl_score = self._search_in_scoreboard_band(gray)
        if tpl_box is not None and tpl_score >= RECOVERY_TEMPLATE_THRESH:
            ok, vscore = self._accept_candidate(
                gray, tpl_box, from_hint=self.current_apparatus, allow_large_shift=True
            )
            if ok and vscore > best_score:
                best_box, best_score, best_orb = tpl_box, vscore, best_orb

        self.last_tpl_score = max(best_score, tpl_score if tpl_box else 0.0)
        self.last_orb_matches = best_orb

        if best_box is not None and best_score >= RECOVERY_TEMPLATE_THRESH:
            self._commit(best_box, best_score, "recovery", best_orb)
            return self.current_apparatus

        self.last_method = "recovery_try"
        self.confidence = max(0.06, self.confidence * 0.98)
        return self.current_apparatus

    def _update_lights_on(self, gray: np.ndarray) -> Box:
        self.current_apparatus = self._anchor_apparatus
        self.confidence = max(RECOVERY_CONF_TRIGGER + 0.02, self._anchor_confidence * 0.92)
        self.last_method = "lights_hold"
        self.last_tpl_score = self._score_template_at(gray, self._anchor_apparatus)
        return self.current_apparatus

    def update(self, frame_bgr: np.ndarray, *, lights_on: bool = False) -> Box:
        gray = self._to_gray(frame_bgr)

        if lights_on and not self._was_lights_on:
            self._anchor_apparatus = self.current_apparatus
            self._anchor_confidence = max(self.confidence, self.last_tpl_score, 0.5)
        self._was_lights_on = lights_on

        if lights_on:
            return self._update_lights_on(gray)

        if self.confidence < RECOVERY_CONF_TRIGGER:
            return self._update_recovery_mode(gray)

        hint = self.current_apparatus
        orb_box, orb_n = self._track_orb(gray, hint, radius_scale=1.2)
        self.last_orb_matches = orb_n

        candidates: List[Tuple[Box, str, int]] = []
        if orb_box is not None:
            candidates.append((orb_box, "orb", orb_n))

        tpl_box, tpl_score = self._match_template_near(gray, hint, radius_scale=1.0)
        if tpl_box is not None:
            candidates.append((tpl_box, "template", orb_n))

        best_box: Optional[Box] = None
        best_score = 0.0
        best_method = "hold"
        best_orb = orb_n

        for box, method, om in candidates:
            ok, score = self._accept_candidate(gray, box, from_hint=hint, allow_large_shift=False)
            if not ok:
                continue
            combined = score
            if method == "orb" and om >= ORB_MIN_GOOD_MATCHES:
                combined += 0.04
            if combined > best_score:
                best_box, best_score, best_method, best_orb = box, score, method, om

        if best_box is not None and best_score >= TEMPLATE_OK_THRESH:
            self._commit(best_box, best_score, best_method, best_orb)
            self._refresh_template(gray, best_box, best_score)
        else:
            self.confidence = max(0.08, self.confidence * 0.94)
            self.last_tpl_score = max(
                tpl_score,
                self._score_template_at(gray, hint) if hint else 0.0,
            )
            self.last_method = "hold"
            if self.confidence < RECOVERY_CONF_TRIGGER:
                return self._update_recovery_mode(gray)

        return self.current_apparatus

    def light_boxes(self) -> Tuple[Box, Box]:
        return split_apparatus_to_lights(self.current_apparatus)

    def snapshot(self) -> Dict[str, object]:
        f1, f2 = self.light_boxes()
        return {
            "apparatus": self.current_apparatus,
            "fencer1_light": f1,
            "fencer2_light": f2,
            "confidence": round(float(self.confidence), 3),
            "tpl_score": round(float(self.last_tpl_score), 3),
            "orb_matches": int(self.last_orb_matches),
            "method": self.last_method,
        }
