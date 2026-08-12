"""Tests for score apparatus tracking helpers."""

import numpy as np

from score_light_tracker import (
    ScoreLightTracker,
    interpolate_box,
    split_apparatus_to_lights,
)


def test_split_apparatus_left_right():
    f1, f2 = split_apparatus_to_lights((100, 50, 200, 120))
    assert f1 == (100, 50, 150, 120)
    assert f2 == (150, 50, 200, 120)


def test_interpolate_box_midpoint():
    a = (0, 0, 100, 50)
    b = (20, 10, 120, 60)
    mid = interpolate_box(0, a, 10, b, 5)
    assert mid == (10, 5, 110, 55)


def test_tracker_update_with_track_points_no_ambiguous_truth():
    h, w = 240, 320
    frame0 = np.zeros((h, w, 3), dtype=np.uint8)
    frame0[120:180, 40:140] = 80
    tracker = ScoreLightTracker(frame0, (40, 120, 140, 180))
    frame1 = frame0.copy()
    frame1[120:180, 50:150] = 80
    tracker.update(frame1, lights_on=False)
    assert tracker.current_apparatus is not None


def test_tracker_follows_horizontal_shift():
    h, w = 240, 320
    frame0 = np.zeros((h, w, 3), dtype=np.uint8)
    cv2 = __import__("cv2")
    frame0[120:180, 40:140] = (80, 80, 80)
    frame0[130:170, 50:130] = (180, 180, 180)

    tracker = ScoreLightTracker(frame0, (40, 120, 140, 180))
    start = tracker.current_apparatus

    frame1 = np.zeros_like(frame0)
    frame1[120:180, 55:155] = frame0[120:180, 40:140]
    frame1[130:170, 65:145] = frame0[130:170, 50:130]

    shifted = tracker.update(frame1, lights_on=False)
    dx = shifted[0] - start[0]
    dy = shifted[1] - start[1]
    assert dx > 5
    assert abs(dy) <= 3

    f1, f2 = tracker.light_boxes()
    assert f1[0] < f2[0]
    assert f1[2] == f2[0]


def test_recovery_after_confidence_drops():
    h, w = 240, 320
    cv2 = __import__("cv2")
    frame0 = np.zeros((h, w, 3), dtype=np.uint8)
    frame0[40:90, 80:180] = (90, 90, 90)
    frame0[50:80, 95:165] = (200, 200, 200)

    tracker = ScoreLightTracker(frame0, (80, 40, 180, 90))
    tracker.confidence = 0.1

    frame1 = np.zeros_like(frame0)
    frame1[40:90, 110:210] = frame0[40:90, 80:180]
    frame1[50:80, 125:195] = frame0[50:80, 95:165]

    recovered = tracker.update(frame1, lights_on=False)
    assert recovered[0] > tracker.initial_apparatus[0] + 10
    assert tracker.last_method in ("recovery", "orb", "template")


def test_low_confidence_shows_recovery_try_not_hold():
    h, w = 240, 320
    frame0 = np.zeros((h, w, 3), dtype=np.uint8)
    frame0[40:90, 80:180] = 80
    tracker = ScoreLightTracker(frame0, (80, 40, 180, 90))
    tracker.confidence = 0.08
    frame1 = np.zeros_like(frame0)
    tracker.update(frame1, lights_on=False)
    assert tracker.last_method != "hold"
    assert tracker.last_method.startswith("recovery")


def test_lights_on_holds_anchor_without_jumping():
    h, w = 240, 320
    frame0 = np.zeros((h, w, 3), dtype=np.uint8)
    frame0[40:90, 80:180] = 80
    tracker = ScoreLightTracker(frame0, (80, 40, 180, 90))
    anchor = tracker.current_apparatus

    frame_lit = frame0.copy()
    frame_lit[40:90, 80:180] = (0, 255, 0)

    out = tracker.update(frame_lit, lights_on=True)
    assert out == anchor
    assert tracker.last_method == "lights_hold"
