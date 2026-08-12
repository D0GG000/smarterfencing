"""Unit tests for partial-body / referee filtering in fencing_inference."""

from fencing_inference import (
    bbox_clipped_at_frame_edge,
    body_segments_visible_in_gate,
    get_fencer_pair_indices,
    is_gate_fencer_candidate,
    is_mostly_full_body_visible,
    keypoints_mostly_full_body,
    select_full_body_instances,
    suggest_auto_fencer_pair,
)


def _blank_keypoints():
    return [[0.0, 0.0, 0.0] for _ in range(17)]


def _set_kp(kpts, index, conf=0.9, x=500.0, y=500.0):
    kpts[index] = [float(x), float(y), conf]


def test_keypoints_mostly_full_body_requires_upper_and_lower():
    frame_h, frame_w = 1080, 1920
    kpts = _blank_keypoints()
    assert not keypoints_mostly_full_body(kpts, frame_h, frame_w)

    _set_kp(kpts, 0)  # nose
    _set_kp(kpts, 5)  # left shoulder
    assert not keypoints_mostly_full_body(kpts, frame_h, frame_w)

    _set_kp(kpts, 11)  # hip alone is not enough
    assert not keypoints_mostly_full_body(kpts, frame_h, frame_w)

    _set_kp(kpts, 15, conf=0.45, y=900.0)  # left ankle
    assert keypoints_mostly_full_body(kpts, frame_h, frame_w)


def test_bottom_clipped_torso_without_legs_is_rejected():
    frame_h, frame_w = 1080, 1920
    # Large torso at the bottom edge (referee at strip front).
    bbox = [800.0, 520.0, 1120.0, 1080.0]
    kpts = _blank_keypoints()
    _set_kp(kpts, 5)
    _set_kp(kpts, 6)
    _set_kp(kpts, 11)
    _set_kp(kpts, 12)

    top, bottom, _, _ = bbox_clipped_at_frame_edge(bbox, frame_h, frame_w)
    assert bottom
    assert not is_mostly_full_body_visible(bbox, kpts, frame_h, frame_w)


def test_bottom_clipped_referee_with_hallucinated_legs_is_rejected():
    """Pose often invents high-conf ankles on a referee torso; bbox geometry must veto."""
    frame_h, frame_w = 1080, 1920
    bbox = [800.0, 520.0, 1120.0, 1080.0]
    kpts = _blank_keypoints()
    _set_kp(kpts, 0, y=540.0)
    _set_kp(kpts, 5, y=560.0)
    _set_kp(kpts, 6, y=560.0)
    _set_kp(kpts, 15, conf=0.9, y=1000.0)
    _set_kp(kpts, 16, conf=0.9, y=1000.0)

    assert not is_mostly_full_body_visible(bbox, kpts, frame_h, frame_w)


def test_corner_foreground_person_with_head_high_is_rejected():
    """Close-up official: head near top, cut off at bottom+right — old veto missed this."""
    frame_h, frame_w = 1080, 1920
    # Fills right side from near top to bottom (like brown-jacket spectator).
    bbox = [1400.0, 80.0, 1910.0, 1080.0]
    kpts = _blank_keypoints()
    _set_kp(kpts, 0, x=1600.0, y=120.0)
    _set_kp(kpts, 5, x=1550.0, y=200.0)
    _set_kp(kpts, 6, x=1650.0, y=200.0)
    _set_kp(kpts, 15, conf=0.9, x=1580.0, y=1000.0)
    _set_kp(kpts, 16, conf=0.9, x=1620.0, y=1000.0)

    assert not is_mostly_full_body_visible(bbox, kpts, frame_h, frame_w)


def test_high_conf_leg_below_frame_is_rejected():
    frame_h, frame_w = 1080, 1920
    bbox = [800.0, 520.0, 1120.0, 1080.0]
    kpts = _blank_keypoints()
    _set_kp(kpts, 5)
    _set_kp(kpts, 6)
    # Hallucinated ankle below visible frame with high confidence.
    _set_kp(kpts, 15, conf=0.95, y=1090.0)

    assert not is_mostly_full_body_visible(bbox, kpts, frame_h, frame_w)


def test_wide_torso_at_right_edge_without_trusted_legs_is_rejected():
    frame_h, frame_w = 1080, 1920
    bbox = [1200.0, 200.0, 1910.0, 1060.0]
    kpts = _blank_keypoints()
    _set_kp(kpts, 5)
    _set_kp(kpts, 6)

    assert not is_mostly_full_body_visible(bbox, kpts, frame_h, frame_w)


def test_full_fencer_with_visible_ankles_at_bottom_edge_is_kept():
    frame_h, frame_w = 1080, 1920
    bbox = [300.0, 120.0, 520.0, 1080.0]
    kpts = _blank_keypoints()
    _set_kp(kpts, 0)
    _set_kp(kpts, 5)
    _set_kp(kpts, 15, conf=0.55, y=1000.0)
    _set_kp(kpts, 16, conf=0.55, y=1000.0)

    assert is_mostly_full_body_visible(bbox, kpts, frame_h, frame_w)


def test_gate_keypoints_reject_referee_without_legs_in_band():
    band_y0, band_y1 = 200.0, 900.0
    referee_kpts = _blank_keypoints()
    _set_kp(referee_kpts, 5, y=600.0)
    _set_kp(referee_kpts, 6, y=620.0)
    _set_kp(referee_kpts, 11, y=700.0)
    _set_kp(referee_kpts, 12, y=720.0)

    fencer_kpts = _blank_keypoints()
    _set_kp(fencer_kpts, 0, y=250.0)
    _set_kp(fencer_kpts, 5, y=300.0)
    _set_kp(fencer_kpts, 15, conf=0.45, y=850.0)

    # Huge bbox does not matter — hips only, no knee/ankle in gate.
    referee_bbox = [100.0, 200.0, 900.0, 900.0]
    fencer_bbox = [200.0, 150.0, 420.0, 1050.0]

    assert not is_gate_fencer_candidate(referee_bbox, referee_kpts, band_y0, band_y1)
    assert is_gate_fencer_candidate(fencer_bbox, fencer_kpts, band_y0, band_y1)


def test_gate_lower_body_forgives_one_leg_outside():
    """Most lower-body joints in gate is enough (one ankle may flicker out)."""
    band_y0, band_y1 = 200.0, 900.0
    kpts = _blank_keypoints()
    _set_kp(kpts, 5, y=300.0)
    _set_kp(kpts, 11, conf=0.5, y=500.0)  # hip in gate
    _set_kp(kpts, 12, conf=0.5, y=500.0)  # hip in gate
    _set_kp(kpts, 15, conf=0.5, y=850.0)  # ankle in gate
    _set_kp(kpts, 16, conf=0.5, y=950.0)  # ankle outside gate
    # 3/4 confident lower joints in gate (>= 50%) and has a knee/ankle in gate.
    assert is_gate_fencer_candidate([200, 150, 420, 1050], kpts, band_y0, band_y1)


def test_gate_lower_body_rejects_when_most_legs_outside():
    band_y0, band_y1 = 200.0, 900.0
    kpts = _blank_keypoints()
    _set_kp(kpts, 5, y=300.0)
    _set_kp(kpts, 15, conf=0.5, y=850.0)  # one ankle in gate
    _set_kp(kpts, 16, conf=0.5, y=950.0)  # outside
    _set_kp(kpts, 13, conf=0.5, y=960.0)  # outside
    # 1/3 < 50% in gate.
    assert not is_gate_fencer_candidate([200, 150, 420, 1050], kpts, band_y0, band_y1)


def test_pairing_uses_gate_keypoints_not_bbox_area():
    """Large referee bbox loses to smaller fencers with upper+lower in gate."""
    frame_h, frame_w = 1080, 1920
    vertical_y0, vertical_y1 = 200.0, 900.0

    referee_kpts = _blank_keypoints()
    _set_kp(referee_kpts, 5, x=800.0, y=600.0)
    _set_kp(referee_kpts, 6, x=820.0, y=600.0)
    # No knees/ankles in gate.

    f1_kpts = _blank_keypoints()
    _set_kp(f1_kpts, 0, x=300.0, y=250.0)
    _set_kp(f1_kpts, 5, x=300.0, y=300.0)
    _set_kp(f1_kpts, 15, conf=0.45, x=300.0, y=850.0)

    f2_kpts = _blank_keypoints()
    _set_kp(f2_kpts, 0, x=1500.0, y=260.0)
    _set_kp(f2_kpts, 6, x=1500.0, y=310.0)
    _set_kp(f2_kpts, 16, conf=0.45, x=1500.0, y=850.0)

    f1_box = [200.0, 150.0, 420.0, 1050.0]
    f2_box = [1450.0, 160.0, 1680.0, 1040.0]
    referee_box = [100.0, 200.0, 1100.0, 900.0]  # huge in-band area

    structured = {
        "bboxes": [referee_box, f1_box, f2_box],
        "keypoints": [referee_kpts, f1_kpts, f2_kpts],
    }

    f1_idx, f2_idx = get_fencer_pair_indices(
        structured,
        "left",
        vertical_y0,
        vertical_y1,
        frame_height=frame_h,
        frame_width=frame_w,
        fencer1_ref_box=f1_box,
        fencer2_ref_box=f2_box,
    )
    assert f1_idx == 1
    assert f2_idx == 2


def test_pairing_fails_when_leg_keypoint_missing_in_gate():
    """Keypoints are the main filter: missing lower body in gate drops the person."""
    frame_h, frame_w = 1080, 1920
    vertical_y0, vertical_y1 = 200.0, 900.0

    f1_kpts = _blank_keypoints()
    _set_kp(f1_kpts, 0, x=300.0, y=250.0)
    _set_kp(f1_kpts, 5, x=300.0, y=300.0)
    # No ankle/knee in gate.

    f2_kpts = _blank_keypoints()
    _set_kp(f2_kpts, 0, x=1500.0, y=260.0)
    _set_kp(f2_kpts, 6, x=1500.0, y=310.0)
    _set_kp(f2_kpts, 16, conf=0.45, x=1500.0, y=850.0)

    f1_box = [200.0, 150.0, 420.0, 1050.0]
    f2_box = [1450.0, 160.0, 1680.0, 1040.0]
    structured = {
        "bboxes": [f1_box, f2_box],
        "keypoints": [f1_kpts, f2_kpts],
    }

    f1_idx, f2_idx = get_fencer_pair_indices(
        structured,
        "left",
        vertical_y0,
        vertical_y1,
        frame_height=frame_h,
        frame_width=frame_w,
        fencer1_ref_box=f1_box,
        fencer2_ref_box=f2_box,
    )
    assert f1_idx is None
    assert f2_idx is None


def test_body_segments_visible_in_gate_requires_upper_and_most_lower():
    band_y0, band_y1 = 200.0, 900.0
    kpts = _blank_keypoints()
    assert body_segments_visible_in_gate(kpts, band_y0, band_y1) == (False, False)

    _set_kp(kpts, 5, y=300.0)
    assert body_segments_visible_in_gate(kpts, band_y0, band_y1) == (True, False)

    _set_kp(kpts, 15, conf=0.45, y=850.0)
    assert body_segments_visible_in_gate(kpts, band_y0, band_y1) == (True, True)

    # Single confident ankle outside gate → no lower body in gate.
    _set_kp(kpts, 15, conf=0.45, y=950.0)
    assert body_segments_visible_in_gate(kpts, band_y0, band_y1) == (True, False)


def test_crouching_fencer_height_passes_full_body():
    frame_h, frame_w = 1080, 1920
    # ~20% of frame height (old threshold 0.28 rejected these).
    bbox = [300.0, 500.0, 480.0, 720.0]
    kpts = _blank_keypoints()
    _set_kp(kpts, 0, y=520.0)
    _set_kp(kpts, 5, y=540.0)
    _set_kp(kpts, 15, conf=0.55, y=700.0)
    assert is_mostly_full_body_visible(bbox, kpts, frame_h, frame_w)


def test_suggest_auto_fencer_pair_picks_left_right_full_body():
    frame_h, frame_w = 1080, 1920
    f1_kpts = _blank_keypoints()
    _set_kp(f1_kpts, 0, x=300.0, y=200.0)
    _set_kp(f1_kpts, 5, x=300.0, y=250.0)
    _set_kp(f1_kpts, 15, conf=0.55, x=300.0, y=900.0)
    f2_kpts = _blank_keypoints()
    _set_kp(f2_kpts, 0, x=1500.0, y=200.0)
    _set_kp(f2_kpts, 6, x=1500.0, y=250.0)
    _set_kp(f2_kpts, 16, conf=0.55, x=1500.0, y=900.0)
    referee_kpts = _blank_keypoints()
    _set_kp(referee_kpts, 5, y=600.0)
    _set_kp(referee_kpts, 6, y=600.0)

    structured = {
        "bboxes": [
            [800.0, 520.0, 1120.0, 1080.0],  # referee partial
            [200.0, 150.0, 420.0, 1000.0],
            [1450.0, 160.0, 1680.0, 1000.0],
        ],
        "keypoints": [referee_kpts, f1_kpts, f2_kpts],
    }
    pick = suggest_auto_fencer_pair(structured, frame_h, frame_w)
    assert pick["success"]
    assert pick["fencer1_index"] == 1
    assert pick["fencer2_index"] == 2


def test_select_full_body_instances_drops_partial_for_auto_detect():
    frame_h, frame_w = 1080, 1920
    referee_kpts = _blank_keypoints()
    _set_kp(referee_kpts, 5)
    _set_kp(referee_kpts, 6)
    _set_kp(referee_kpts, 11)
    _set_kp(referee_kpts, 12)

    f1_kpts = _blank_keypoints()
    _set_kp(f1_kpts, 0)
    _set_kp(f1_kpts, 5)
    _set_kp(f1_kpts, 15, conf=0.45, y=900.0)

    structured = {
        "bboxes": [
            [800.0, 500.0, 1150.0, 1080.0],
            [200.0, 150.0, 420.0, 1050.0],
        ],
        "keypoints": [referee_kpts, f1_kpts],
        "bbox_scores": [0.95, 0.9],
    }

    out = select_full_body_instances(
        structured,
        frame_h,
        frame_w,
        top_k=32,
        fallback_if_none=False,
    )
    assert len(out["bboxes"]) == 1
    assert out["bboxes"][0] == structured["bboxes"][1]
