"""
ODTrack wrapper for scoreboard apparatus tracking.

Plain single-tracker design: fixed init box size, center follows the model.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from odtrack_engine import ODTrackEngine, ensure_odtrack_model
from ostrack_utils import xyxy_to_xywh

Box = Tuple[int, int, int, int]

CONF_SMOOTH = 0.30


def split_apparatus_to_lights(apparatus: Box) -> Tuple[Box, Box]:
    x1, y1, x2, y2 = apparatus
    mid = (x1 + x2) // 2
    return (x1, y1, mid, y2), (mid, y1, x2, y2)


def _clamp_box(box: Box, frame_w: int, frame_h: int, min_size: int = 8) -> Box:
    x1, y1, x2, y2 = box
    w = max(min_size, x2 - x1)
    h = max(min_size, y2 - y1)
    x1 = max(0, min(frame_w - w, x1))
    y1 = max(0, min(frame_h - h, y1))
    return int(x1), int(y1), int(x1 + w), int(y1 + h)


class ODTrackScoreboardTracker:
    """Fixed-size apparatus box driven by ODTrack temporal memory."""

    def __init__(self, frame_bgr: np.ndarray, apparatus_box: Box):
        self.frame_h, self.frame_w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = _clamp_box(apparatus_box, self.frame_w, self.frame_h)
        self._box_w = max(8, x2 - x1)
        self._box_h = max(8, y2 - y1)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        self.initial_apparatus = self._fixed_at_center(cx, cy)
        self.current_apparatus = self.initial_apparatus
        self._anchor_apparatus = self.initial_apparatus
        self._anchor_confidence = 1.0
        self.confidence = 0.85
        self.last_tpl_score = 0.85
        self.last_color_score = 0.85
        self.last_orb_matches = 0
        self.last_method = "init"
        self.inference_provider = "CPU"

        model_path = ensure_odtrack_model()
        self._engine = ODTrackEngine(model_path)
        self.inference_provider = self._engine.provider
        init_xywh = xyxy_to_xywh(self.initial_apparatus)
        self._engine.initialize(frame_bgr, init_xywh)
        self.last_method = "odtrack_init"

    def _fixed_at_center(self, cx: int, cy: int) -> Box:
        return _clamp_box(
            (
                cx - self._box_w // 2,
                cy - self._box_h // 2,
                cx - self._box_w // 2 + self._box_w,
                cy - self._box_h // 2 + self._box_h,
            ),
            self.frame_w,
            self.frame_h,
        )

    def _fixed_from_xywh(self, state: Tuple[float, float, float, float]) -> Box:
        cx = int(round(state[0] + state[2] * 0.5))
        cy = int(round(state[1] + state[3] * 0.5))
        return self._fixed_at_center(cx, cy)

    def _smooth(self, sample: float) -> None:
        self.confidence = (1.0 - CONF_SMOOTH) * self.confidence + CONF_SMOOTH * sample

    def update(self, frame_bgr: np.ndarray, *, lights_on: bool = False) -> Box:
        if lights_on:
            self.current_apparatus = self._anchor_apparatus
            self.confidence = max(0.25, self._anchor_confidence * 0.96)
            self.last_method = "lights_hold"
            return self.current_apparatus

        state, score = self._engine.track(frame_bgr)
        box = self._fixed_from_xywh(tuple(state))
        self.current_apparatus = box
        self.last_tpl_score = score
        self.last_color_score = score
        self.last_orb_matches = int(min(99, score * 100))
        self.last_method = "odtrack"
        self._smooth(min(0.95, max(0.1, score)))

        if score >= 0.25:
            self._anchor_apparatus = box
            self._anchor_confidence = self.confidence

        return box

    def light_boxes(self) -> Tuple[Box, Box]:
        return split_apparatus_to_lights(self.current_apparatus)

    def snapshot(self) -> Dict[str, object]:
        return {
            "apparatus": self.current_apparatus,
            "fencer1_light": self.light_boxes()[0],
            "fencer2_light": self.light_boxes()[1],
            "confidence": round(float(self.confidence), 3),
            "tpl_score": round(float(self.last_tpl_score), 3),
            "color_score": round(float(self.last_color_score), 3),
            "orb_matches": int(self.last_orb_matches),
            "method": self.last_method,
            "backend": "odtrack",
            "provider": self.inference_provider,
        }
