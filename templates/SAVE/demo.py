# demo.py
import os, json, glob, shutil, uuid, threading, queue, logging
from pathlib import Path

import numpy as np
import torch
import cv2

import boto3
from botocore.config import Config

from flask import Blueprint, render_template, request, jsonify, send_from_directory, current_app

# -------------------------
# Demo blueprint
# -------------------------
demo_bp = Blueprint("demo", __name__)

# -------------------------
# Demo globals (only demo.py owns these)
# -------------------------
log_queue = queue.Queue()

pipeline_state = {
    "current_step": "idle",
    "error": None,
    "results": [],
    "3d_results": None,
    "fps": 30,
}
current_selections = None

pose_model = None  # cached MMPoseInferencer

# -------------------------
# Logging helper
# -------------------------
def log(msg: str):
    try:
        log_queue.put(msg)
    except Exception:
        pass
    logging.info(msg)

# -------------------------
# Config helpers (read from env, app.config, defaults)
# -------------------------
def _get_cfg(app, key, default=None):
    return app.config.get(key, os.environ.get(key, default))

def _ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

# -------------------------
# Optional: do heavy registry init only if demo is registered
# -------------------------
def _init_mm_registry_once():
    # Safe to call multiple times; keep it idempotent.
    from mmpose.utils import register_all_modules as register_mmpose
    from mmdet.utils import register_all_modules as register_mmdet
    from mmengine.registry import init_default_scope

    register_mmpose(init_default_scope=False)
    register_mmdet(init_default_scope=False)
    init_default_scope("mmpose")

# -------------------------
# R2 client
# -------------------------
def r2_client(app):
    r2_account_id = _get_cfg(app, "R2_ACCOUNT_ID")
    r2_bucket = _get_cfg(app, "R2_BUCKET", "smarterfencing-videos")

    if not r2_account_id:
        raise RuntimeError("Missing R2_ACCOUNT_ID")

    return (
        boto3.client(
            "s3",
            endpoint_url=f"https://{r2_account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=_get_cfg(app, "R2_ACCESS_KEY_ID"),
            aws_secret_access_key=_get_cfg(app, "R2_SECRET_ACCESS_KEY"),
            region_name="auto",
            config=Config(signature_version="s3v4"),
        ),
        r2_bucket,
    )

# -------------------------
# Pose model caching
# -------------------------
def get_pose_model():
    global pose_model
    if pose_model is None:
        from mmpose.apis import MMPoseInferencer
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        log(f"Loading pose model on {device}...")
        pose_model = MMPoseInferencer("vitpose-h", device=device)
        log("Pose model loaded!")
    return pose_model

# ============================================================
# ROUTES (demo pipeline)
# ============================================================

@demo_bp.route("/demo")
def demo():
    return render_template("demo.html")

@demo_bp.get("/api/get-3d-data")
def get_3d_data():
    app = current_app

    video = request.args.get("video")
    touch = request.args.get("touch")
    job_id = request.args.get("job_id")

    if not video or not job_id or not touch:
        return jsonify({"error": "Missing video/job_id or touch parameter"}), 400

    output_3d = _get_cfg(app, "OUTPUT_3D", "./3d_outputs")
    json_path = os.path.join(output_3d, job_id, f"{touch}_3d.json")

    if not os.path.exists(json_path):
        return jsonify({"error": f"3D data not found: {json_path}"}), 404

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def build_3d_batch(app, job_id):
    """
    Returns { touch_name: <json dict from *_3d.json> }
    """
    output_3d = _get_cfg(app, "OUTPUT_3D", "./3d_outputs")
    job_dir = os.path.join(output_3d, job_id)  # <-- must match where you write 3D files

    batch = {}
    if not os.path.isdir(job_dir):
        log(f"No 3D dir found: {job_dir}")
        return batch

    for p in glob.glob(os.path.join(job_dir, "*_3d.json")):
        touch_name = os.path.basename(p).replace("_3d.json", "")
        try:
            with open(p, "r") as f:
                batch[touch_name] = json.load(f)
        except Exception as e:
            log(f"Failed loading {p}: {e}")

    return batch

def pipeline_runner(app_obj, object_key, local_video_path, selections):
    """
    Runs in a background thread.
    Assumes r2_client() is the app.py version (no app arg) OR import the right one.
    """
    global pipeline_state

    try:
        pipeline_state["current_step"] = "downloading"
        log(f"Downloading from R2: {object_key} -> {local_video_path}")

        s3, bucket = r2_client(app_obj)
        s3.download_file(bucket, object_key, local_video_path)
        log("video downloaded from R2")

        pipeline_state["current_step"] = "extracting"
        run_full_pipeline(app_obj, local_video_path, selections)   # <-- app.py signature
        log(f"done with run_full_pipeline")

    except Exception as e:
        pipeline_state["current_step"] = "error"
        pipeline_state["error"] = str(e)
        log(f"ERROR in pipeline thread: {e}")

def run_full_pipeline(app, video_path, selections):
    global pipeline_state

    output_2d = _get_cfg(app, "OUTPUT_2D", "./unlabeled")
    output_3d = _get_cfg(app, "OUTPUT_3D", "./3d_outputs")

    job_id = os.path.splitext(os.path.basename(video_path))[0]

    try:
        pipeline_state["current_step"] = "extracting"
        log("=== STEP 1: Extracting 2D Keypoints ===")
        video_fps = run_data_extractor(app, video_path, output_2d, selections, job_id)
        pipeline_state["fps"] = video_fps
        log("2D extraction complete!")

        pipeline_state["current_step"] = "lifting"
        log("=== STEP 2: Lifting to 3D ===")
        run_3d_lifting(app, output_2d, output_3d, job_id)
        log("3D lifting complete!")

        pipeline_state["current_step"] = "predicting"
        log("=== STEP 3: Predicting Touches ===")
        results = run_prediction(app, output_2d, output_3d, job_id)

        pipeline_state["current_step"] = "complete"
        pipeline_state["results"] = results
        pipeline_state["3d_results"] = build_3d_batch(app, job_id)
        log(f"=== COMPLETE: {len(results)} touches analyzed ===")

    except Exception as e:
        pipeline_state["current_step"] = "error"
        pipeline_state["error"] = str(e)
        log(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

def analyze_fencer_sequence(pose_sequence):
    """
    Comprehensive pose analysis for a single fencer across 30 frames.
    Extracts joint angles and fencing-specific metrics.
    Each fencer is analyzed independently - no opponent data required.

    H36M Joint Indices (17-joint MotionBERT format):
    0: pelvis, 1: right_hip, 2: right_knee, 3: right_ankle,
    4: left_hip, 5: left_knee, 6: left_ankle, 7: spine,
    8: thorax, 9: neck, 10: head,
    11: left_shoulder, 12: left_elbow, 13: left_wrist,
    14: right_shoulder, 15: right_elbow, 16: right_wrist
    """

    def to_native(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_native(v) for v in obj]
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def calculate_3d_angle(a, b, c):
        """Calculate 3D angle at point b."""
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)

        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def calculate_joint_angles(frame):
        """Extract all joint angles for the current frame."""
        # H36M indices mapping
        pelvis = frame[0]
        right_hip = frame[1]
        right_knee = frame[2]
        right_ankle = frame[3]
        left_hip = frame[4]
        left_knee = frame[5]
        left_ankle = frame[6]
        spine = frame[7]
        thorax = frame[8]
        neck = frame[9]
        head = frame[10]
        left_shoulder = frame[11]
        left_elbow = frame[12]
        left_wrist = frame[13]
        right_shoulder = frame[14]
        right_elbow = frame[15]
        right_wrist = frame[16]

        # Upper body angles
        shoulder_width_angle = calculate_3d_angle(left_shoulder, thorax, right_shoulder)
        trunk_angle = calculate_3d_angle(left_hip, spine, neck)

        # Arm extensions (both arms for dynamic handedness)
        right_arm_extension = calculate_3d_angle(right_shoulder, right_elbow, right_wrist)
        left_arm_extension = calculate_3d_angle(left_shoulder, left_elbow, left_wrist)

        # Shoulder angles (arm to body connection)
        right_shoulder_angle = calculate_3d_angle(thorax, right_shoulder, right_elbow)
        left_shoulder_angle = calculate_3d_angle(thorax, left_shoulder, left_elbow)

        # Hip angles
        right_hip_angle = calculate_3d_angle(pelvis, right_hip, right_knee)
        left_hip_angle = calculate_3d_angle(pelvis, left_hip, left_knee)

        # Knee angles
        right_knee_angle = calculate_3d_angle(right_hip, right_knee, right_ankle)
        left_knee_angle = calculate_3d_angle(left_hip, left_knee, left_ankle)

        return {
            # Upper body
            'shoulder_width_angle': shoulder_width_angle,
            'trunk_angle': trunk_angle,

            # Arm extensions (both for handedness toggle)
            'right_arm_extension': right_arm_extension,
            'left_arm_extension': left_arm_extension,

            # Shoulders (arm to body)
            'right_shoulder_angle': right_shoulder_angle,
            'left_shoulder_angle': left_shoulder_angle,

            # Hips
            'right_hip_angle': right_hip_angle,
            'left_hip_angle': left_hip_angle,

            # Knees
            'right_knee_angle': right_knee_angle,
            'left_knee_angle': left_knee_angle,

            # Binary flags (based on weapon hand - updated dynamically in UI)
            'right_arm_extended': right_arm_extension > 150,
            'right_arm_bent': right_arm_extension < 160,
            'left_arm_extended': left_arm_extension > 150,
            'left_arm_bent': left_arm_extension < 160,
            'right_leg_bent': right_knee_angle < 170,
            'left_leg_bent': left_knee_angle < 170,
            'deep_lunge': left_knee_angle < 100 if left_knee_angle > 0 else False
        }

    def calculate_movement_metrics(pose_sequence):
        """Extract temporal movement characteristics across the 30 frames."""
        if len(pose_sequence) < 2:
            return None

        movements = {
            'frame_displacements': [],
            'acceleration_magnitude': []
        }

        prev_frame = pose_sequence[0]

        for i, frame in enumerate(pose_sequence):
            # Calculate displacement from previous frame
            displacement = np.linalg.norm(frame - prev_frame)
            movements['frame_displacements'].append(float(displacement))
            prev_frame = frame

        # Calculate accelerations
        displacements = movements['frame_displacements']
        if len(displacements) > 2:
            accelerations = [0]  # First frame has no acceleration
            for i in range(1, len(displacements)):
                acc = displacements[i] - displacements[i-1]
                accelerations.append(float(acc))
            movements['acceleration_magnitude'] = accelerations

        # Summary statistics
        movements['summary'] = {
            'avg_displacement': float(np.mean(displacements)),
            'max_displacement': float(max(displacements)),
            'min_displacement': float(min(displacements)),
            'std_displacement': float(np.std(displacements)) if len(displacements) > 1 else 0,
            'movement_intensity': float(np.mean([abs(a) for a in movements['acceleration_magnitude']]))
        }

        return movements

    # Main analysis: process all 30 frames
    frame_analyses = []
    for i, frame in enumerate(pose_sequence):
        frame_analysis = {
            'frame_index': i + 1,  # 1-indexed for display
            'timestamp': i / 30.0,  # Assuming 30fps
            'angles': calculate_joint_angles(frame)
        }
        frame_analyses.append(frame_analysis)

    # Calculate overall sequence statistics
    all_angles = [f['angles'] for f in frame_analyses]

    # Angle statistics (including both arm extensions for handedness support)
    angle_keys = ['right_arm_extension', 'left_arm_extension', 'shoulder_width_angle', 'trunk_angle',
                  'right_shoulder_angle', 'left_shoulder_angle',
                  'right_hip_angle', 'left_hip_angle', 'right_knee_angle', 'left_knee_angle']
    angle_stats = {}
    for key in angle_keys:
        angle_stats[key] = {
            'mean': float(np.mean([a[key] for a in all_angles])),
            'std': float(np.std([a[key] for a in all_angles])),
            'min': float(np.min([a[key] for a in all_angles])),
            'max': float(np.max([a[key] for a in all_angles]))
        }

    # Movement analysis
    movement = calculate_movement_metrics(pose_sequence)

    # Sequence-level findings (based on right arm by default, can be overridden by UI)
    sequence_findings = {
        'max_extension_frame': int(np.argmax([a['right_arm_extension'] for a in all_angles])) + 1,
        'max_extension_angle': float(np.max([a['right_arm_extension'] for a in all_angles]))
    }

    return to_native({
        'per_frame_analysis': frame_analyses,
        'angle_statistics': angle_stats,
        'movement_analysis': movement,
        'sequence_findings': sequence_findings,
        'total_frames': len(pose_sequence)
    })


def detect_and_update_handedness(app, output_3d, video_name):
    """
    Analyze all touches to detect each fencer's dominant hand.

    Uses facing direction and which hand ends up in front to determine handedness:
    - Fencer 1 (left side of video): faces RIGHT, opponent is in POSITIVE Z direction
    - Fencer 2 (right side of video): faces LEFT, opponent is in NEGATIVE Z direction

    The hand that is MORE FORWARD toward the opponent is the weapon hand.
    """
    video_output_dir = os.path.join(output_3d, video_name)
    if not os.path.exists(video_output_dir):
        log(f"[HANDEDNESS] No output directory found for {video_name}")
        return

    # Collect front-hand data from all touches
    f1_left_forward = 0
    f1_right_forward = 0
    f2_left_forward = 0
    f2_right_forward = 0
    f1_total_frames = 0
    f2_total_frames = 0

    json_files = [f for f in os.listdir(video_output_dir) if f.endswith('_3d.json')]

    log(f"[HANDEDNESS] Processing {len(json_files)} touch files for handedness detection")

    for json_file in json_files:
        json_path = os.path.join(video_output_dir, json_file)
        touch_name = json_file.replace('_3d.json', '')

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            f1_3d = data.get('fencer1_3d', [])
            f2_3d = data.get('fencer2_3d', [])

            if not f1_3d or not f2_3d:
                continue

            for frame_idx, (f1_frame, f2_frame) in enumerate(zip(f1_3d, f2_3d)):
                try:
                    # H36M indices: 13=left_wrist, 16=right_wrist (0-indexed in array)
                    f1_left_wrist = np.array(f1_frame[13])
                    f1_right_wrist = np.array(f1_frame[16])
                    f1_left_x = f1_left_wrist[0]
                    f1_right_x = f1_right_wrist[0]

                    f2_left_wrist = np.array(f2_frame[13])
                    f2_right_wrist = np.array(f2_frame[16])
                    f2_left_x = f2_left_wrist[0]
                    f2_right_x = f2_right_wrist[0]

                    # Fencer 1 (LEFT side, faces RIGHT): weapon hand has LARGER X
                    if f1_right_x > f1_left_x:
                        f1_right_forward += 1
                    else:
                        f1_left_forward += 1
                    f1_total_frames += 1

                    # Fencer 2 (RIGHT side, faces LEFT): weapon hand has SMALLER X
                    if f2_right_x < f2_left_x:
                        f2_right_forward += 1
                    else:
                        f2_left_forward += 1
                    f2_total_frames += 1

                except Exception as e:
                    continue

        except Exception as e:
            log(f"[HANDEDNESS] Could not process {json_file}: {e}")
            continue

    # Determine handedness
    f1_handedness = 'right' if f1_right_forward >= f1_left_forward else 'left'
    f2_handedness = 'right' if f2_right_forward >= f2_left_forward else 'left'

    log(f"[HANDEDNESS] Fencer 1: {f1_handedness.upper()}-handed ({f1_right_forward}/{f1_total_frames} right-forward)")
    log(f"[HANDEDNESS] Fencer 2: {f2_handedness.upper()}-handed ({f2_right_forward}/{f2_total_frames} right-forward)")

    # Update all JSON files with detected handedness
    for json_file in json_files:
        json_path = os.path.join(video_output_dir, json_file)
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            data['fencer1_handedness'] = f1_handedness
            data['fencer2_handedness'] = f2_handedness
            data['handedness_debug'] = {
                'fencer1': {
                    'right_forward': int(f1_right_forward),
                    'left_forward': int(f1_left_forward),
                    'total_frames': int(f1_total_frames)
                },
                'fencer2': {
                    'right_forward': int(f2_right_forward),
                    'left_forward': int(f2_left_forward),
                    'total_frames': int(f2_total_frames)
                }
            }

            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            log(f"[HANDEDNESS] Could not update {json_file}: {e}")


@demo_bp.get("/api/status")
def get_status():
    # Only return pipeline state to the client whose job is currently being processed.
    # This prevents User B's "Analyze Another Video" (or job start) from resetting User A's view.
    from job_queue_worker import get_current_job_id
    requested_job_id = request.args.get("job_id")
    current_job_id = get_current_job_id()
    if requested_job_id and current_job_id and requested_job_id != current_job_id:
        return jsonify({
            "current_step": "idle",
            "error": None,
            "results": [],
            "fps": 30,
            "logs": [],
        })

    logs = []
    while not log_queue.empty():
        try:
            logs.append(log_queue.get_nowait())
        except Exception:
            break

    # pop 3d_results so it is sent ONLY ONCE
    three_d_results = pipeline_state.pop("3d_results", None)

    resp = {
        "current_step": pipeline_state.get("current_step"),
        "error": pipeline_state.get("error"),
        "results": pipeline_state.get("results", []),
        "fps": pipeline_state.get("fps", 30),
        "logs": logs,
    }

    if three_d_results is not None:
        resp["3d_results"] = three_d_results

    return jsonify(resp)

@demo_bp.post("/api/reset")
def reset():
    log(f"Reset requested")
    from job_queue_worker import get_current_job_id
    if get_current_job_id() is not None:
        # Another user's job is still processing; do not wipe global state
        return jsonify({"success": True, "skipped": True})
    global pipeline_state
    pipeline_state = {"current_step": "idle", "error": None, "results": [], "3d_results": None, "fps": 30}
    return jsonify({"success": True})

@demo_bp.route("/result")
def result():
    return render_template("result.html")

# -------------------------
# Your existing implementations go here with one change:
# add `app` param and use app-config paths instead of module globals
# -------------------------

def run_data_extractor(app, video_path, output_dir, selections, job_id):
    """Extract 2D keypoints using browser-provided selections"""
    from mmpose.apis import MMPoseInferencer

    KEYPOINT_LABELS = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    video_output_dir = os.path.join(output_dir, job_id)
    os.makedirs(video_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    log(f"Video: {job_id}, {frame_width}x{frame_height}, {fps:.1f}fps, {total_frames} frames")

    # Scale selections from browser canvas to actual video resolution
    sel_width = selections.get('video_width', frame_width)
    sel_height = selections.get('video_height', frame_height)
    scale_x = frame_width / sel_width
    scale_y = frame_height / sel_height

    def scale_box(box):
        if not box:
            return None
        return (
            int(box['x1'] * scale_x),
            int(box['y1'] * scale_y),
            int(box['x2'] * scale_x),
            int(box['y2'] * scale_y)
        )

    fencer1_box = scale_box(selections.get('fencer1'))
    fencer2_box = scale_box(selections.get('fencer2'))
    fencer1_light = scale_box(selections.get('fencer1_light'))
    fencer2_light = scale_box(selections.get('fencer2_light'))

    # Determine sides based on box positions
    f1_center_x = (fencer1_box[0] + fencer1_box[2]) / 2 if fencer1_box else 0
    f2_center_x = (fencer2_box[0] + fencer2_box[2]) / 2 if fencer2_box else frame_width
    fencer1_side = 'left' if f1_center_x < f2_center_x else 'right'

    log(f"Fencer 1 side: {fencer1_side}")

    # Load pose model
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    log(f"Loading pose model on {device}...")
    pose_model = get_pose_model()
    log("Model loaded!")

    def run_pose_inference(frame):
        try:
            result = next(pose_model(frame, show=False))
            predictions = result.get('predictions', [[]])[0]
            results = {'bboxes': [], 'keypoints': [], 'bbox_scores': []}
            for pred in predictions:
                bbox = pred.get('bbox', [[]])[0] if pred.get('bbox') else None
                keypoints = pred.get('keypoints', [])
                keypoint_scores = pred.get('keypoint_scores', [])
                if bbox and keypoints:
                    results['bboxes'].append(bbox)
                    kpts = [[kp[0], kp[1], sc] for kp, sc in zip(keypoints, keypoint_scores)]
                    results['keypoints'].append(kpts)
                    results['bbox_scores'].append(pred.get('bbox_score', 0.0))
            return results
        except:
            return None

    # Threshold for "light on" (fraction of box pixels that must be that color); lower = more forgiving
    LIGHT_ON_RATIO = 0.35

    def get_light_ratios(frame, box):
        """Return fraction of region that is green and red (0-1). Uses the full selected box. HSV ranges are wide to accept multiple shades."""
        if not box:
            return {"green": 0.0, "red": 0.0}
        x1, y1, x2, y2 = box
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return {"green": 0.0, "red": 0.0}
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        # Red: wide H (wraparound), relaxed S/V for dim or off-shade LEDs
        red_mask = cv2.inRange(hsv, np.array([0, 80, 50]), np.array([12, 255, 255])) + \
                   cv2.inRange(hsv, np.array([168, 80, 50]), np.array([180, 255, 255]))
        # Green: wide H (yellow-green to green), relaxed S/V
        green_mask = cv2.inRange(hsv, np.array([25, 60, 60]), np.array([95, 255, 255]))
        total = region.shape[0] * region.shape[1]
        return {
            "green": cv2.countNonZero(green_mask) / total,
            "red": cv2.countNonZero(red_mask) / total
        }

    def _check_light_any(frame, box):
        """Discovery only: on if either color exceeds threshold. No color param."""
        r = get_light_ratios(frame, box)
        if r["green"] > LIGHT_ON_RATIO or r["red"] > LIGHT_ON_RATIO:
            return "on"
        return "off"

    def check_score_light(frame, box, color):
        """Color-specific: on only if the given color exceeds threshold. color must be 'green' or 'red'."""
        if not box:
            return "off"
        r = get_light_ratios(frame, box)
        if color == "green" and r["green"] > LIGHT_ON_RATIO:
            return "on"
        if color == "red" and r["red"] > LIGHT_ON_RATIO:
            return "on"
        return "off"

    def find_exact_light_on_frame(off_frame, on_frame, score_box, color=None):
        """Binary search to find exact frame where light turns on. color=None uses _check_light_any (discovery)."""
        left = off_frame
        right = on_frame
        exact_frame = on_frame

        log(f"Binary search between frames {off_frame} (OFF) and {on_frame} (ON)")

        while left <= right:
            mid = (left + right) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ret, frame = cap.read()

            if not ret:
                break

            if color is None:
                light_state = _check_light_any(frame, score_box)
            else:
                light_state = check_score_light(frame, score_box, color)

            if light_state == "on":
                exact_frame = mid
                right = mid - 1
            else:
                left = mid + 1

        log(f"Found exact frame: {exact_frame}")
        return exact_frame

    def get_fencer_ids(results):
        if not results or not results['bboxes'] or len(results['bboxes']) < 2:
            return None, None
        boxes = results['bboxes']
        areas = [(i, (b[2]-b[0])*(b[3]-b[1])) for i, b in enumerate(boxes)]
        areas.sort(key=lambda x: x[1], reverse=True)
        top2 = [areas[0][0], areas[1][0]] if len(areas) >= 2 else [areas[0][0], areas[0][0]]
        pos = [(i, (boxes[i][0]+boxes[i][2])/2) for i in top2]
        pos.sort(key=lambda x: x[1])
        left_idx, right_idx = pos[0][0], pos[1][0]
        return (left_idx, right_idx) if fencer1_side == 'left' else (right_idx, left_idx)

    def extract_keypoints(results, fencer_id):
        if not results or not results['keypoints'] or fencer_id is None or fencer_id >= len(results['keypoints']):
            return None
        kpts = results['keypoints'][fencer_id]
        return {KEYPOINT_LABELS[i]: [kpts[i][0], kpts[i][1], float(kpts[i][2])] for i in range(17)}

    def extract_frames_before_touch(touch_frame, scoring_fencer):
        from datetime import datetime
        start = max(0, touch_frame - 29)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        group_id = f"{scoring_fencer}_score_{timestamp}_frame{touch_frame}"
        group_dir = os.path.join(video_output_dir, group_id)
        os.makedirs(group_dir, exist_ok=True)

        # 1. Seek once, read all frames sequentially (faster than random seeking)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(touch_frame - start + 1):
            ret, fr = cap.read()
            if ret:
                frames.append(fr)

        if not frames:
            log(f"Touch at frame {touch_frame}: no frames could be read")
            return 0

        # 2. Batch inference - single call for all frames (much faster on GPU)
        log(f"Running batch pose inference on {len(frames)} frames...")
        all_results = []
        for result in pose_model(frames, show=False):
            predictions = result.get('predictions', [[]])[0]
            res = {'bboxes': [], 'keypoints': [], 'bbox_scores': []}
            for pred in predictions:
                bbox = pred.get('bbox', [[]])[0] if pred.get('bbox') else None
                keypoints = pred.get('keypoints', [])
                keypoint_scores = pred.get('keypoint_scores', [])
                if bbox and keypoints:
                    res['bboxes'].append(bbox)
                    kpts = [[kp[0], kp[1], sc] for kp, sc in zip(keypoints, keypoint_scores)]
                    res['keypoints'].append(kpts)
                    res['bbox_scores'].append(pred.get('bbox_score', 0.0))
            all_results.append(res)

        # 3. Save results - write keypoints only for frames with detections (frame numbers may have gaps)
        saved = 0
        for idx, (fr, results) in enumerate(zip(frames, all_results)):
            if results and results['bboxes']:
                f1_id, f2_id = get_fencer_ids(results)
                f1_kp = extract_keypoints(results, f1_id)
                f2_kp = extract_keypoints(results, f2_id)

                frame_num = idx + 1
                cv2.imwrite(os.path.join(group_dir, f"frame_{frame_num}.jpg"), fr)
                with open(os.path.join(group_dir, f"frame_{frame_num}_keypoints.json"), "w") as f:
                    json.dump({
                        "frame_index": frame_num,
                        "scoring_fencer": scoring_fencer,
                        "fencer1_keypoints": f1_kp,
                        "fencer2_keypoints": f2_kp
                    }, f, indent=2)
                saved += 1

        log(f"Touch at frame {touch_frame}: saved {saved} frames")
        return saved

    # Scan for touches; assign which box is green vs red from first activation (real = earlier/bolder, bleed = later/weaker)
    log("Scanning for score lights...")
    prev_f1_light = "off"
    prev_f2_light = "off"
    fencer1_light_color = None  # "green" or "red", set after first touch
    fencer2_light_color = None
    sample_interval = 50
    current_frame = 0
    touches_found = 0

    def assign_light_colors(frame, box1, box2):
        """Set fencer1_light_color and fencer2_light_color from ratios at this frame. Real light is bolder and dominant."""
        nonlocal fencer1_light_color, fencer2_light_color
        r1 = get_light_ratios(frame, box1)
        r2 = get_light_ratios(frame, box2)
        s1 = max(r1["green"], r1["red"])
        s2 = max(r2["green"], r2["red"])
        dom1 = "green" if r1["green"] > r1["red"] and r1["green"] > LIGHT_ON_RATIO else ("red" if r1["red"] > LIGHT_ON_RATIO else None)
        dom2 = "green" if r2["green"] > r2["red"] and r2["green"] > LIGHT_ON_RATIO else ("red" if r2["red"] > LIGHT_ON_RATIO else None)
        if s1 >= s2 and dom1:
            fencer1_light_color = dom1
            fencer2_light_color = "red" if dom1 == "green" else "green"
        elif s2 >= s1 and dom2:
            fencer2_light_color = dom2
            fencer1_light_color = "red" if dom2 == "green" else "green"
        elif dom1:
            fencer1_light_color = dom1
            fencer2_light_color = "red" if dom1 == "green" else "green"
        elif dom2:
            fencer2_light_color = dom2
            fencer1_light_color = "red" if dom2 == "green" else "green"
        else:
            fencer1_light_color = "green"
            fencer2_light_color = "red"
        log(f"Assigned: Fencer 1 light = {fencer1_light_color}, Fencer 2 light = {fencer2_light_color}")

    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, fr = cap.read()
        if not ret:
            break

        if fencer1_light_color is None:
            f1_light = _check_light_any(fr, fencer1_light)
            f2_light = _check_light_any(fr, fencer2_light)
        else:
            f1_light = check_score_light(fr, fencer1_light, fencer1_light_color)
            f2_light = check_score_light(fr, fencer2_light, fencer2_light_color)

        f1_new_touch = f1_light == "on" and prev_f1_light == "off"
        f2_new_touch = f2_light == "on" and prev_f2_light == "off"

        if f1_new_touch or f2_new_touch:
            search_start = max(0, current_frame - sample_interval)

            if fencer1_light_color is None:
                # First touch: assign colors from ratios at current frame (real light is bolder)
                assign_light_colors(fr, fencer1_light, fencer2_light)
                # Re-evaluate who actually touched with color-specific check (drops false bleed)
                f1_new_touch = check_score_light(fr, fencer1_light, fencer1_light_color) == "on" and prev_f1_light == "off"
                f2_new_touch = check_score_light(fr, fencer2_light, fencer2_light_color) == "on" and prev_f2_light == "off"

            # Handle detected touch(es) with color-specific exact frame
            if f1_new_touch:
                exact_frame = find_exact_light_on_frame(search_start, current_frame, fencer1_light, fencer1_light_color)
                log(f"Touch detected for Fencer 1 at exact frame {exact_frame}")
                extract_frames_before_touch(exact_frame, "fencer1")
                touches_found += 1
                prev_f1_light = "on"

            if f2_new_touch:
                exact_frame = find_exact_light_on_frame(search_start, current_frame, fencer2_light, fencer2_light_color)
                log(f"Touch detected for Fencer 2 at exact frame {exact_frame}")
                extract_frames_before_touch(exact_frame, "fencer2")
                touches_found += 1
                prev_f2_light = "on"

            # If only one light triggered, check next 30 frames for the other (color-specific)
            if f1_new_touch != f2_new_touch and fencer1_light_color is not None:
                for check_frame in range(current_frame, min(current_frame + 30, total_frames)):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, check_frame)
                    ret, check_fr = cap.read()
                    if not ret:
                        break
                    if f1_new_touch and prev_f2_light == "off":
                        if check_score_light(check_fr, fencer2_light, fencer2_light_color) == "on":
                            exact_frame = find_exact_light_on_frame(search_start, check_frame, fencer2_light, fencer2_light_color)
                            log(f"Touch detected for Fencer 2 at exact frame {exact_frame} (double touch)")
                            extract_frames_before_touch(exact_frame, "fencer2")
                            touches_found += 1
                            prev_f2_light = "on"
                            break
                    if f2_new_touch and prev_f1_light == "off":
                        if check_score_light(check_fr, fencer1_light, fencer1_light_color) == "on":
                            exact_frame = find_exact_light_on_frame(search_start, check_frame, fencer1_light, fencer1_light_color)
                            log(f"Touch detected for Fencer 1 at exact frame {exact_frame} (double touch)")
                            extract_frames_before_touch(exact_frame, "fencer1")
                            touches_found += 1
                            prev_f1_light = "on"
                            break

            # Skip ahead past the light duration
            current_frame += 150
            # Wait for both lights to turn off (color-specific)
            while current_frame < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, fr = cap.read()
                if not ret:
                    break
                f1_off = check_score_light(fr, fencer1_light, fencer1_light_color) == "off"
                f2_off = check_score_light(fr, fencer2_light, fencer2_light_color) == "off"
                if f1_off:
                    prev_f1_light = "off"
                if f2_off:
                    prev_f2_light = "off"
                if f1_off and f2_off:
                    break
                current_frame += 1
            continue

        prev_f1_light = f1_light
        prev_f2_light = f2_light
        current_frame += sample_interval

        if current_frame % 500 == 0:
            log(f"Progress: {current_frame}/{total_frames} frames")

    cap.release()
    log(f"Extraction complete: {touches_found} touches found")

    if touches_found == 0:
        raise Exception("No touches detected. Check your score light box selections.")

    return fps
def run_3d_lifting(app, input_2d, output_3d, job_id):
    """Lift 2D keypoints to 3D for the given job_id only (avoids re-processing other jobs)."""
    from mmengine.structures import InstanceData
    from mmpose.structures import PoseDataSample
    from mmpose.apis import init_model, inference_pose_lifter_model
    from mmpose.apis.inference_3d import convert_keypoint_definition

    COCO_ORDER = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    # Only process sequences for this job (avoid re-lifting and accumulating other jobs' data)
    job_2d_dir = os.path.join(input_2d, job_id)
    sequences = []
    if os.path.isdir(job_2d_dir):
        for tf in glob.glob(os.path.join(job_2d_dir, "*")):
            if os.path.isdir(tf):
                sequences.append(tf)

    if not sequences:
        log("No sequences found")
        return

    log(f"Found {len(sequences)} sequences to lift")

    config = "mmpose/configs/body_3d_keypoint/motionbert/h36m/motionbert_dstformer-ft-243frm_8xb32-120e_h36m.py"
    checkpoint = "motionbert_ft_h36m-d80af323_20230531.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = init_model(config, checkpoint, device=device).to(torch.float32)
    model.backbone.register_forward_pre_hook(lambda m, i: (i[0].float(),))

    for seq_folder in sequences:
        touch_name = os.path.basename(seq_folder)
        job_id = os.path.basename(os.path.dirname(seq_folder))

        log(f"seq: {touch_name} - {job_id}")

        try:
            files = sorted(glob.glob(os.path.join(seq_folder, "frame_*_keypoints.json")),
                          key=lambda p: int(os.path.basename(p).split("_")[1]))
            if not files:
                continue

            def load_seq(fencer_key):
                seq = []
                for f in files[:30]:
                    with open(f) as fp:
                        d = json.load(fp)
                    kpts = [d[fencer_key][n] for n in COCO_ORDER]
                    arr = np.array(kpts, dtype=np.float32)
                    # Use only x,y for 3D lifting (confidence in JSON for future use)
                    if arr.shape[-1] >= 3:
                        arr = arr[..., :2]
                    seq.append(arr)
                return np.stack(seq)

            f1_2d = load_seq("fencer1_keypoints")
            f2_2d = load_seq("fencer2_keypoints")

            def build_samples(seq):
                kpts = convert_keypoint_definition(seq, 'coco', 'h36m')
                samples = []
                for t in range(len(kpts)):
                    s = PoseDataSample()
                    s.gt_instances = InstanceData()
                    s.pred_instances = InstanceData()
                    s.pred_instances.keypoints = kpts[t][None].astype(np.float32)
                    xs, ys = kpts[t][:,0], kpts[t][:,1]
                    s.pred_instances.bboxes = np.array([[xs.min(), ys.min(), xs.max(), ys.max()]], dtype=np.float32)
                    s.track_id = 0
                    samples.append(s)
                return [samples]

            out1 = inference_pose_lifter_model(model, build_samples(f1_2d), with_track_id=False,
                                               image_size=(1920,1080), norm_pose_2d=True)
            out2 = inference_pose_lifter_model(model, build_samples(f2_2d), with_track_id=False,
                                               image_size=(1920,1080), norm_pose_2d=True)

            def extract_3d(out):
                arr = []
                for s in out:
                    k = s.pred_instances.keypoints
                    if isinstance(k, torch.Tensor):
                        k = k.detach().cpu().numpy()
                    while k.ndim > 2:
                        k = np.squeeze(k, 0)
                    arr.append(k)
                return np.stack(arr)

            # Extract raw 3D data
            f1_3d_raw = extract_3d(out1)
            f2_3d_raw = extract_3d(out2)

            out_dir = os.path.join(output_3d, job_id)
            os.makedirs(out_dir, exist_ok=True)

            # Save 3D data with per-frame analysis
            with open(os.path.join(out_dir, f"{touch_name}_3d.json"), "w") as f:
                json.dump({
                    "num_frames": len(f1_3d_raw),
                    "fencer1_3d": f1_3d_raw.tolist(),
                    "fencer2_3d": f2_3d_raw.tolist(),
                    "fencer1_analysis": analyze_fencer_sequence(f1_3d_raw),
                    "fencer2_analysis": analyze_fencer_sequence(f2_3d_raw)
                }, f)

            log(f"Lifted: {touch_name}")

        except Exception as e:
            log(f"Failed {touch_name}: {e}")

    # After all sequences are processed, detect handedness for this job
    if sequences:
        detect_and_update_handedness(app, output_3d, job_id)

def run_prediction(app, path_2d, path_3d, job_id):
    model_path = _get_cfg(app, "MODEL_PATH", "./best_touch_unbiased.pth")
    import torch.nn as nn
    import torch.nn.functional as F

    class FencingFeatureExtractor:
        JOINTS = {'left_shoulder':5,'right_shoulder':6,'left_elbow':7,'right_elbow':8,
                  'left_wrist':9,'right_wrist':10,'left_hip':11,'right_hip':12,
                  'left_knee':13,'right_knee':14,'left_ankle':15,'right_ankle':16}

        @staticmethod
        def angle(frame, a, b, c):
            j = FencingFeatureExtractor.JOINTS
            pa, pb, pc = frame[j[a]], frame[j[b]], frame[j[c]]
            v1, v2 = pa - pb, pc - pb
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                return 0.0
            return np.arccos(np.clip(np.dot(v1,v2)/(n1*n2), -1, 1))

        @staticmethod
        def normalize_3d(seq):
            scales = [np.linalg.norm(f[5]-f[11])+1e-6 for f in seq]
            return seq / np.median(scales)

        @staticmethod
        def extract_3d_features(sf, tf):
            a = FencingFeatureExtractor.angle
            return [a(sf,'right_shoulder','right_elbow','right_wrist'),
                    a(sf,'left_shoulder','left_elbow','left_wrist'),
                    a(sf,'right_elbow','right_shoulder','right_hip'),
                    a(sf,'left_elbow','left_shoulder','left_hip'),
                    a(sf,'right_hip','right_knee','right_ankle'),
                    a(sf,'left_hip','left_knee','left_ankle'),
                    a(tf,'right_elbow','right_shoulder','right_hip'),
                    a(tf,'left_elbow','left_shoulder','left_hip')]

    class FencingTouchClassifier(nn.Module):
        def __init__(self, input_dim, num_classes=4, dropout=0.3):
            super().__init__()
            self.embed = nn.Linear(input_dim, 72)
            self.gru = nn.GRU(72, 48, batch_first=True, bidirectional=True)
            self.attn = nn.Linear(96, 1)
            self.fc = nn.Sequential(
                nn.Linear(96, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes)
            )

        def forward(self, x):
            x = F.gelu(self.embed(x))
            x, _ = self.gru(x)
            w = F.softmax(self.attn(x), dim=1)
            x = torch.sum(w * x, dim=1)
            return self.fc(x)

    def mirror_features(features):
        """Mirror features for TTA (Test-Time Augmentation)."""
        mirrored = features.copy()
        # Swap left/right pairs
        mirrored[:, [0, 1]] = mirrored[:, [1, 0]]
        mirrored[:, [2, 3]] = mirrored[:, [3, 2]]
        mirrored[:, [4, 5]] = mirrored[:, [5, 4]]
        mirrored[:, [6, 7]] = mirrored[:, [7, 6]]
        mirrored[:, [8, 9]] = mirrored[:, [9, 8]]
        mirrored[:, [10, 11]] = mirrored[:, [11, 10]]
        mirrored[:, [12, 13]] = mirrored[:, [13, 12]]
        mirrored[:, [12, 13]] *= -1
        return mirrored

    def predict_with_tta(model, features, device):
        """Predict with test-time augmentation using mirrored features."""
        model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0).to(device)
            logits1 = model(x)
            prob1 = F.softmax(logits1, dim=1)

            x_mirror = torch.FloatTensor(mirror_features(features)).unsqueeze(0).to(device)
            logits2 = model(x_mirror)
            prob2 = F.softmax(logits2, dim=1)

            avg_prob = (prob1 + prob2) / 2
            pred_class = avg_prob.argmax(1).item()
            confidence = avg_prob.max().item()

            return pred_class, confidence, avg_prob.cpu().numpy()[0]

    results = []
    classes = ["chest", "abdomen", "arm", "leg"]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model once outside the loop
    model = None

    # Only process the current job's folder (avoid returning hits from previous jobs)
    video_folder = os.path.join(path_3d, job_id)
    if not os.path.isdir(video_folder):
        return results

    files_3d = sorted([f for f in os.listdir(video_folder) if f.endswith("_3d.json")])

    for f in files_3d:
        p3d = os.path.join(video_folder, f)
        touch_name = f.replace("_3d.json", "")
        p2d = os.path.join(path_2d, job_id, touch_name)

        if not os.path.isdir(p2d):
            continue

        try:
            with open(p3d) as fp:
                d3d = json.load(fp)

            scorer = "fencer1" if f.startswith("fencer1") else "fencer2"
            target = "fencer2" if scorer == "fencer1" else "fencer1"

            scorer_seq = np.array(d3d[f"{scorer}_3d"], dtype=np.float32)
            target_seq = np.array(d3d[f"{target}_3d"], dtype=np.float32)

            if len(scorer_seq) < 30:
                pad = 30 - len(scorer_seq)
                scorer_seq = np.concatenate([scorer_seq, np.repeat(scorer_seq[-1:], pad, 0)])
                target_seq = np.concatenate([target_seq, np.repeat(target_seq[-1:], pad, 0)])
            else:
                scorer_seq, target_seq = scorer_seq[-30:], target_seq[-30:]

            scorer_seq = FencingFeatureExtractor.normalize_3d(scorer_seq)
            target_seq = FencingFeatureExtractor.normalize_3d(target_seq)

            # Use whatever 2D keypoint files exist (sorted by frame number), not assumed frame_1..frame_30
            files_2d = sorted(
                glob.glob(os.path.join(p2d, "frame_*_keypoints.json")),
                key=lambda p: int(os.path.basename(p).split("_")[1])
            )
            if len(files_2d) < 30:
                log(f"Prediction skip {touch_name}: only {len(files_2d)} 2D frames (need 30)")
                continue

            features = []
            last_valid = None

            for i in range(30):
                fp_path = files_2d[i]
                with open(fp_path) as ff:
                    d2d = json.load(ff)

                s2d = d2d.get(f"{scorer}_keypoints")
                t2d = d2d.get(f"{target}_keypoints")
                if not s2d or not t2d:
                    features.append(last_valid)
                    continue

                sf, tf = scorer_seq[i], target_seq[i]
                f3d = FencingFeatureExtractor.extract_3d_features(sf, tf)

                ank_y = (t2d["left_ankle"][1]+t2d["right_ankle"][1])/2
                hip_y = (t2d["left_hip"][1]+t2d["right_hip"][1])/2
                sh_y = (t2d["left_shoulder"][1]+t2d["right_shoulder"][1])/2
                hip_x = (t2d["left_hip"][0]+t2d["right_hip"][0])/2
                bw = abs(t2d["left_shoulder"][0]-t2d["right_shoulder"][0])+1e-6
                leg_n = abs(hip_y-ank_y)+1e-6
                torso_n = abs(sh_y-hip_y)+1e-6

                rw, lw = s2d["right_wrist"], s2d["left_wrist"]

                last_valid = f3d + [
                    (rw[1]-ank_y)/leg_n, (lw[1]-ank_y)/leg_n,
                    (rw[1]-hip_y)/torso_n, (lw[1]-hip_y)/torso_n,
                    (rw[0]-hip_x)/bw, (lw[0]-hip_x)/bw,
                    abs(((s2d["left_shoulder"][1]+s2d["right_shoulder"][1])/2)-
                        ((s2d["left_ankle"][1]+s2d["right_ankle"][1])/2))/(abs(sh_y-ank_y)+1e-6),
                    np.linalg.norm(((np.array(s2d["left_hip"])+np.array(s2d["right_hip"]))/2)-
                                   ((np.array(t2d["left_hip"])+np.array(t2d["right_hip"]))/2))/bw
                ]
                features.append(last_valid)

            features_arr = np.array(features, dtype=np.float32)

            # Load model once (lazy initialization) - fixed input_dim=16 to match best_touch_unbiased.pth
            if model is None:
                log(f"Loading FencingTouchClassifier")
                model = FencingTouchClassifier(input_dim=16)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                model.eval()

            # Use TTA for more robust predictions
            pred, confidence, all_probs = predict_with_tta(model, features_arr, device)

            results.append({
                "video": job_id,
                "touch": touch_name,
                "prediction": classes[pred],
                "confidence": all_probs.tolist()
            })

            log(f"Prediction: {touch_name} -> {classes[pred]} ({confidence:.1%})")

        except Exception as e:
            log(f"Prediction failed for {touch_name}: {e}")

    return results

# ============================================================
# One public entrypoint app.py calls
# ============================================================
def register_demo(app):
    """
    Call this from app.py once.
    Sets up demo config defaults, initializes heavy registries, creates dirs, and registers routes.
    """
    _init_mm_registry_once()

    # default dirs (override in app.config if you want)
    app.config.setdefault("UPLOAD_DIR", "./uploads")
    app.config.setdefault("OUTPUT_2D", "./unlabeled")
    app.config.setdefault("OUTPUT_3D", "./3d_outputs")
    app.config.setdefault("MODEL_PATH", "./best_touch_unbiased.pth")
    _ensure_dirs(app.config["UPLOAD_DIR"], app.config["OUTPUT_2D"], app.config["OUTPUT_3D"])

    app.register_blueprint(demo_bp)

