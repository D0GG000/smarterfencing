"""
Scoreboard tracking for SmarterFencing pipeline.

Default: ODTrack (GPU temporal memory). Fallback: legacy ORB ScoreLightTracker.

Set TRACKER_BACKEND=legacy to force the original tracker.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Dict, Tuple

import numpy as np

from score_light_tracker import (
    ScoreLightTracker as _LegacyScoreLightTracker,
    interpolate_box,
    split_apparatus_to_lights,
)

Box = Tuple[int, int, int, int]
_log = logging.getLogger(__name__)


class TrackerBackend(str, Enum):
    ODTRACK = "odtrack"
    LEGACY = "legacy"


def _default_backend() -> str:
    return os.environ.get("TRACKER_BACKEND", TrackerBackend.ODTRACK.value).strip().lower()


class ScoreboardTracker:
    """Unified tracker API for demo.py."""

    def __init__(
        self,
        frame_bgr: np.ndarray,
        apparatus_box: Box,
        *,
        backend: str | None = None,
    ):
        requested = (backend or _default_backend()).strip().lower()
        try:
            self.backend = TrackerBackend(requested)
        except ValueError:
            self.backend = TrackerBackend.ODTRACK

        self._odtrack = None
        self._legacy = None
        self._was_lights_on = False

        if self.backend == TrackerBackend.ODTRACK:
            try:
                from odtrack_tracker import ODTrackScoreboardTracker

                self._odtrack = ODTrackScoreboardTracker(frame_bgr, apparatus_box)
                return
            except Exception as exc:
                _log.warning("ODTrack init failed (%s); using legacy tracker", exc)

        self.backend = TrackerBackend.LEGACY
        self._legacy = _LegacyScoreLightTracker(frame_bgr, apparatus_box)

    def _active(self):
        return self._odtrack if self._odtrack is not None else self._legacy

    @property
    def current_apparatus(self) -> Box:
        return self._active().current_apparatus

    @property
    def confidence(self) -> float:
        return self._active().confidence

    @property
    def last_tpl_score(self) -> float:
        return self._active().last_tpl_score

    @property
    def last_color_score(self) -> float:
        return self._active().last_color_score

    @property
    def last_orb_matches(self) -> int:
        return self._active().last_orb_matches

    @property
    def last_method(self) -> str:
        return self._active().last_method

    def update(self, frame_bgr: np.ndarray, *, lights_on: bool = False) -> Box:
        if self._odtrack is not None:
            if lights_on and not self._was_lights_on:
                self._odtrack._anchor_apparatus = self._odtrack.current_apparatus
                self._odtrack._anchor_confidence = max(self._odtrack.confidence, 0.5)
            self._was_lights_on = lights_on
            return self._odtrack.update(frame_bgr, lights_on=lights_on)

        self._was_lights_on = lights_on
        return self._legacy.update(frame_bgr, lights_on=lights_on)

    def light_boxes(self) -> Tuple[Box, Box]:
        return self._active().light_boxes()

    def snapshot(self) -> Dict[str, object]:
        snap = dict(self._active().snapshot())
        snap["backend"] = self.backend.value
        return snap

    @staticmethod
    def available_backends() -> list[str]:
        return [TrackerBackend.ODTRACK.value, TrackerBackend.LEGACY.value]


# Backward-compatible alias used by older imports.
ScoreLightTracker = ScoreboardTracker

__all__ = [
    "ScoreboardTracker",
    "ScoreLightTracker",
    "TrackerBackend",
    "interpolate_box",
    "split_apparatus_to_lights",
]
