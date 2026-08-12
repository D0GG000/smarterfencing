"""Helpers for tracker bbox conversions."""

from __future__ import annotations

from typing import Tuple


def xyxy_to_xywh(box: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return float(x1), float(y1), float(x2 - x1), float(y2 - y1)


def xywh_to_xyxy(state: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    x, y, w, h = state
    return int(x), int(y), int(x + w), int(y + h)
