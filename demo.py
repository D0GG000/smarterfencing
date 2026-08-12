# demo.py
import os, json, glob, shutil, uuid, threading, queue, logging, subprocess, math
from pathlib import Path

import numpy as np
import torch
import cv2

import boto3
from botocore.config import Config

from flask import Blueprint, render_template, request, jsonify, send_from_directory, current_app, session

from workspace_paths import OUTPUT_2D, OUTPUT_3D, UPLOAD_DIR, ensure_workspace_dirs, tmp_path
from scoreboard_tracker import (
    ScoreboardTracker,
    interpolate_box,
    split_apparatus_to_lights,
)

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
    "3d_results_durable": None,
    "spatial_touch_summary": None,
    "spatial_touch_summary_durable": None,
    "arm_attempts": None,
    "arm_attempts_durable": None,
    "highlight_reel_key": None,
    "fps": 30,
}
current_selections = None

# -------------------------
# Logging helper
# -------------------------
def log(msg: str):
    try:
        log_queue.put(msg)
    except Exception:
        pass
    # Always emit to stdout so RunPod / container logs capture TRACK diagnostics.
    print(msg, flush=True)
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

# ============================================================
# ROUTES (demo pipeline)
# ============================================================

@demo_bp.route("/demo")
def demo():
    return render_template("demo.html", clipper_mode=False)


@demo_bp.route("/clipper")
def clipper():
    return render_template("demo.html", clipper_mode=True)


@demo_bp.route("/past-results")
def past_results():
    return render_template("past_results.html")


@demo_bp.route("/archetypes")
def archetypes_page():
    return render_template("archetypes.html")


@demo_bp.get("/api/get-3d-data")
def get_3d_data():
    app = current_app

    uid = session.get("user_id")
    if not uid:
        return jsonify({"error": "Authentication required"}), 401

    video = request.args.get("video")
    touch = request.args.get("touch")
    job_id = request.args.get("job_id")

    if not video or not job_id or not touch:
        return jsonify({"error": "Missing video/job_id or touch parameter"}), 400

    from job_queue_models import UserJob

    if not UserJob.query.filter_by(job_id=job_id, user_id=uid).first():
        return jsonify({"error": "Job not found"}), 404

    output_3d = _get_cfg(app, "OUTPUT_3D", OUTPUT_3D)
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
    output_3d = _get_cfg(app, "OUTPUT_3D", OUTPUT_3D)
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
        if os.path.isfile(local_video_path) and os.path.getsize(local_video_path) > 0:
            log(f"Using cached local video: {local_video_path}")
        else:
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

    output_2d = _get_cfg(app, "OUTPUT_2D", OUTPUT_2D)
    output_3d = _get_cfg(app, "OUTPUT_3D", OUTPUT_3D)

    job_id = os.path.splitext(os.path.basename(video_path))[0]

    try:
        pipeline_state["current_step"] = "extracting"
        log("=== STEP 1: Extracting 2D Keypoints ===")
        video_path = ensure_processable_video(video_path)
        video_fps = run_data_extractor(app, video_path, output_2d, selections, job_id)
        pipeline_state["fps"] = video_fps
        log("2D extraction complete!")

        # Separate lightweight bout-wide arm-attempt pass (RTMPose-s), same fencer filter.
        pipeline_state["current_step"] = "arm_attempts"
        log("=== STEP 1b: Arm-attempt scan (RTMPose-s, separate) ===")
        try:
            from fencing_inference import park_vitpose_to_cpu, unpark_vitpose_to_gpu
            from arm_attempt_pass import run_arm_attempt_pass

            park_vitpose_to_cpu(log)
            try:
                arm_attempts = run_arm_attempt_pass(
                    video_path, selections, log, max_proc_dim=MAX_PROC_DIM
                )
            finally:
                unpark_vitpose_to_gpu(log)
            pipeline_state["arm_attempts"] = arm_attempts
            # Durable: /api/status may pop "arm_attempts" during later steps.
            pipeline_state["arm_attempts_durable"] = arm_attempts
        except Exception as arm_exc:
            log(f"[ARM] WARNING: arm-attempt pass failed (continuing): {arm_exc}")
            import traceback
            traceback.print_exc()
            try:
                from fencing_inference import unpark_vitpose_to_gpu

                unpark_vitpose_to_gpu(log)
            except Exception:
                pass
            arm_attempts = {
                "fencer1_total": 0,
                "fencer2_total": 0,
                "items": [],
                "error": str(arm_exc),
            }
            pipeline_state["arm_attempts"] = arm_attempts
            pipeline_state["arm_attempts_durable"] = arm_attempts

        pipeline_state["current_step"] = "lifting"
        log("=== STEP 2: Lifting to 3D ===")
        run_3d_lifting(app, output_2d, output_3d, job_id)
        log("3D lifting complete!")

        pipeline_state["current_step"] = "predicting"
        log("=== STEP 3: Predicting Touches ===")
        results = run_prediction(app, output_2d, output_3d, job_id)

        pipeline_state["current_step"] = "complete"
        pipeline_state["results"] = results
        three_d_batch = build_3d_batch(app, job_id)
        # Durable copy: /api/status pops "3d_results" for one-shot UI delivery; the
        # queue worker must still persist 3D/angles into job.results_json / email.
        pipeline_state["3d_results"] = three_d_batch
        pipeline_state["3d_results_durable"] = three_d_batch
        pipeline_state["spatial_touch_summary"] = summarize_touch_spatial_from_batch(
            three_d_batch
        )
        pipeline_state["spatial_touch_summary_durable"] = pipeline_state[
            "spatial_touch_summary"
        ]
        # Weapon-arm filter using 3D handedness (prefer durable if status popped live copy)
        try:
            from arm_attempt_pass import attribute_arm_attempts_handedness

            provisional = (
                pipeline_state.get("arm_attempts_durable")
                or pipeline_state.get("arm_attempts")
            )
            attributed = attribute_arm_attempts_handedness(
                provisional,
                three_d_batch,
                log_fn=log,
            )
            pipeline_state["arm_attempts"] = attributed
            pipeline_state["arm_attempts_durable"] = attributed
        except Exception as arm_exc:
            log(f"[ARM] WARNING: handedness attribute failed: {arm_exc}")
            if pipeline_state.get("arm_attempts_durable") is None:
                pipeline_state["arm_attempts_durable"] = pipeline_state.get(
                    "arm_attempts"
                )

        # Pre-touch footwork aggressor from forward/back bout scan.
        try:
            arm_payload = (
                pipeline_state.get("arm_attempts_durable")
                or pipeline_state.get("arm_attempts")
            )
            annotated = attach_pre_touch_aggressors(
                arm_payload, results, three_d_batch, log_fn=log
            )
            if annotated is not None:
                pipeline_state["arm_attempts"] = annotated
                pipeline_state["arm_attempts_durable"] = annotated
        except Exception as fb_exc:
            log(f"[FOOTWORK] WARNING: pre-touch aggressor failed: {fb_exc}")

        log(f"=== COMPLETE: {len(results)} touches analyzed ===")
        if three_d_batch:
            for touch_name, payload in three_d_batch.items():
                f1 = payload.get("fencer1_3d") or []
                f2 = payload.get("fencer2_3d") or []
                has_a1 = bool(payload.get("fencer1_analysis"))
                has_a2 = bool(payload.get("fencer2_analysis"))
                log(
                    f"TRACK 3D stored {touch_name}: "
                    f"f1_frames={len(f1)} f2_frames={len(f2)} "
                    f"f1_analysis={has_a1} f2_analysis={has_a2}"
                )

    except Exception as e:
        pipeline_state["current_step"] = "error"
        pipeline_state["error"] = str(e)
        log(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

# ============================================================
# CLIPPER pipeline (touch detection + highlight reel, no pose/3D)
# ============================================================

# Highlight clip window around each touch (seconds before/after the light frame).
CLIP_PRE_SEC = 1.5
CLIP_POST_SEC = 1.0

# Cap the working resolution. Some phone clips report 4K (or larger) dimensions
# which can exhaust GPU memory during pose inference and choke VideoWriter/codecs,
# making both the analyzer and the clipper fail. Downscaling the long side to this
# keeps detail high enough for pose/light detection while staying robust.
MAX_PROC_DIM = int(os.environ.get("MAX_PROC_DIM", "1920"))

# Cap working frame rate. Phone clips at 60/120 fps inflate frame counts and
# analysis cost without helping fencing detection. Anything above this is
# decimated once in ensure_processable_video(); lower rates are left alone.
MAX_PROC_FPS = int(os.environ.get("MAX_PROC_FPS", "30"))


def _proc_scale(width, height, max_dim=MAX_PROC_DIM):
    """Scale factor (<= 1.0) to fit (width, height) within max_dim on the long side."""
    longest = max(int(width or 0), int(height or 0))
    if longest <= 0 or longest <= max_dim:
        return 1.0
    return max_dim / float(longest)


def _downscale_frame(frame, scale):
    if scale >= 1.0 or frame is None:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(
        frame, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


_H264_FFMPEG = None  # cache: (ffmpeg_path, encoder_name) or (None, None)


def _find_h264_ffmpeg():
    """Find an ffmpeg binary that actually runs and has an H.264 encoder.

    The first `ffmpeg` on PATH can be unusable: in this deployment the conda
    build is missing a shared lib (libopenh264.so.5) and OpenCV's bundled
    encoder is only the hardware `h264_v4l2m2m`, which has no device. So we
    probe candidates, require `-version` to succeed, and pick a software H.264
    encoder we know the build supports. Result is cached.
    """
    global _H264_FFMPEG
    if _H264_FFMPEG is not None:
        return _H264_FFMPEG

    candidates = []
    which = shutil.which("ffmpeg")
    if which:
        candidates.append(which)
    for p in ("/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/bin/ffmpeg"):
        if p not in candidates and os.path.exists(p):
            candidates.append(p)

    # CPU encoders first for reliability (nvenc can be listed but fail at runtime).
    preferred = ("libx264", "libopenh264", "h264_nvenc", "h264", "mpeg4")
    for ff in candidates:
        try:
            if subprocess.run(
                [ff, "-hide_banner", "-version"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ).returncode != 0:
                continue  # broken build (e.g. missing shared library)
            enc = subprocess.run(
                [ff, "-hide_banner", "-encoders"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            listing = enc.stdout.decode("utf-8", "ignore") if enc.stdout else ""
            for name in preferred:
                # encoder lines look like " V..... libx264   ..."
                if any(name == tok for tok in listing.split()):
                    _H264_FFMPEG = (ff, name)
                    log(f"Highlight reel encoder: {ff} -> {name}")
                    return _H264_FFMPEG
        except Exception as e:
            log(f"ffmpeg probe failed for {ff}: {e}")
            continue

    _H264_FFMPEG = (None, None)
    log("Highlight reel: no working ffmpeg H.264 encoder found")
    return _H264_FFMPEG


def _opencv_can_decode(path):
    """True if OpenCV can open AND actually read a frame from `path`."""
    cap = cv2.VideoCapture(path)
    ok = False
    try:
        if cap.isOpened():
            for _ in range(15):
                ret, fr = cap.read()
                if ret and fr is not None:
                    ok = True
                    break
    finally:
        cap.release()
    return ok


def video_meta(video_path):
    """Return (total_frames, fps) for a video path."""
    cap = cv2.VideoCapture(video_path)
    try:
        total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if fps <= 1e-3:
            fps = 30.0
        return total, fps
    finally:
        cap.release()


def read_video_frame_bgr(video_path, frame_index):
    """Read a specific frame reliably (H.264 seeking often fails with OpenCV alone).

    Returns (frame_bgr, total_frames, fps).
    Prefer ffmpeg timestamp seek; fall back to OpenCV POS_MSEC / sequential grab.
    """
    total_frames, fps = video_meta(video_path)
    frame_index = int(max(0, min(total_frames - 1, int(frame_index))))
    t_sec = frame_index / fps

    ffmpeg, _enc = _find_h264_ffmpeg()
    if ffmpeg:
        out_jpg = tmp_path(f"frame_extract_{os.getpid()}_{frame_index}.jpg")
        try:
            # -ss before -i: fast keyframe seek (good enough for template scrubbing)
            cmd = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{t_sec:.4f}",
                "-i", video_path,
                "-frames:v", "1",
                "-q:v", "2",
                out_jpg,
            ]
            proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if proc.returncode == 0 and os.path.isfile(out_jpg) and os.path.getsize(out_jpg) > 0:
                frame = cv2.imread(out_jpg)
                if frame is not None:
                    return frame, total_frames, fps
        finally:
            try:
                if os.path.exists(out_jpg):
                    os.remove(out_jpg)
            except Exception:
                pass

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        # Time-based seek is usually better than POS_FRAMES on H.264.
        cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
        ret, frame = cap.read()
        if ret and frame is not None and frame_index > 0:
            # Reject obvious "stuck on first frame" when we expected a later frame:
            # OpenCV often ignores seek and still returns frame 0 with POS=1.
            reported = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
            if reported <= 1 and frame_index > 5:
                ret = False
                frame = None
        if ret and frame is not None:
            return frame, total_frames, fps

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        reported = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
        if ret and frame is not None and not (reported <= 1 and frame_index > 5):
            return frame, total_frames, fps

        # Last resort: sequential decode (ok for early frames / short clips).
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame = None
        for i in range(frame_index + 1):
            ret, candidate = cap.read()
            if not ret or candidate is None:
                break
            frame = candidate
        if frame is None:
            raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
        return frame, total_frames, fps
    finally:
        cap.release()


def ensure_processable_video(video_path):
    """Return a path OpenCV can decode at <= MAX_PROC_FPS, transcoding once if needed.

    Some sources (notably 4K/HEVC or 10-bit phone clips) open but fail to decode
    with OpenCV's bundled ffmpeg, which breaks both the analyzer and the clipper.
    High-FPS clips (60/120) inflate frame counts and analysis cost. When either
    happens we transcode once to H.264 (fps capped, scaled to 1920 when re-encoding
    for decode) and process that copy instead. Videos that already decode at
    <= MAX_PROC_FPS are returned unchanged, so normal clips incur no extra work.
    """
    can_decode = _opencv_can_decode(video_path)
    try:
        _, src_fps = video_meta(video_path)
    except Exception:
        src_fps = 0.0
    needs_fps_cap = src_fps > float(MAX_PROC_FPS) + 1e-3

    if can_decode and not needs_fps_cap:
        return video_path

    ffmpeg, encoder = _find_h264_ffmpeg()
    if not ffmpeg:
        if not can_decode:
            log("Video not decodable by OpenCV and no working ffmpeg to transcode it")
        elif needs_fps_cap:
            log(
                f"Video is {src_fps:.1f}fps (>{MAX_PROC_FPS}) but no working ffmpeg "
                "to cap it; using original"
            )
        return video_path

    encoder = encoder or "libx264"
    norm_path = video_path + ".norm.mp4"
    vf_parts = []
    if needs_fps_cap:
        vf_parts.append(f"fps={MAX_PROC_FPS}")
    if not can_decode:
        vf_parts.append("scale=1920:-2")
    cmd = [ffmpeg, "-y", "-i", video_path]
    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]
    cmd += ["-c:v", encoder]
    if encoder == "h264_nvenc":
        cmd += ["-preset", "p4", "-rc", "vbr", "-cq", "23"]
    else:
        cmd += ["-preset", "veryfast", "-crf", "23"]
    cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", norm_path]

    reasons = []
    if not can_decode:
        reasons.append("undecodable by OpenCV")
    if needs_fps_cap:
        reasons.append(f"{src_fps:.1f}fps > {MAX_PROC_FPS}")
    log(f"Video normalize ({', '.join(reasons)}); transcoding with {encoder} -> {norm_path}")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", "ignore")[-500:] if e.stderr else ""
        log(f"Video normalize transcode failed; using original. {err}")
        return video_path
    except Exception as e:
        log(f"Video normalize transcode error ({e}); using original")
        return video_path

    if (
        os.path.exists(norm_path)
        and os.path.getsize(norm_path) > 0
        and _opencv_can_decode(norm_path)
    ):
        return norm_path
    log("Normalized copy still not decodable; using original")
    return video_path


def build_highlight_reel(video_path, touches, fps, out_path):
    """Stitch a short clip around each detected touch into one mp4.

    Reads the exact frame windows with OpenCV (precise cut points), writes a
    temporary video, then transcodes to browser-friendly H.264 with ffmpeg when
    available. Returns out_path on success, else None.
    """
    if not touches:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"Highlight reel: could not open {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = fps if fps and fps > 0 else 30.0

    # Cap reel resolution so 4K sources still encode reliably.
    scale = _proc_scale(src_width, src_height)
    width = max(1, int(round(src_width * scale)))
    height = max(1, int(round(src_height * scale)))
    if scale < 1.0:
        log(f"Highlight reel: downscaling {src_width}x{src_height} -> {width}x{height}")

    pre = max(1, int(round(CLIP_PRE_SEC * out_fps)))
    post = max(1, int(round(CLIP_POST_SEC * out_fps)))

    raw_path = out_path + ".raw.mp4"
    writer = None
    for fourcc_name in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        candidate = cv2.VideoWriter(raw_path, fourcc, out_fps, (width, height))
        if candidate.isOpened():
            writer = candidate
            log(f"Highlight reel writer using fourcc={fourcc_name}")
            break
        candidate.release()

    if writer is None:
        cap.release()
        log("Highlight reel: no usable VideoWriter codec")
        return None

    frames_written = 0
    for t in sorted(touches, key=lambda x: x.get("frame", 0)):
        frame_no = int(t.get("frame", 0))
        start = max(0, frame_no - pre)
        end = min(total_frames - 1, frame_no + post) if total_frames > 0 else frame_no + post
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(start, end + 1):
            ret, fr = cap.read()
            if not ret:
                break
            if scale < 1.0:
                fr = cv2.resize(fr, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(fr)
            frames_written += 1

    writer.release()
    cap.release()

    if frames_written == 0:
        try:
            if os.path.exists(raw_path):
                os.remove(raw_path)
        except Exception:
            pass
        return None

    # Transcode to a widely playable mp4 (H.264 + faststart). The raw OpenCV
    # output is mp4v (MPEG-4 Part 2), which browsers cannot play, so a working
    # H.264 transcode is required for the reel to actually load/download.
    ffmpeg, encoder = _find_h264_ffmpeg()
    if ffmpeg and encoder:
        cmd = [ffmpeg, "-y", "-i", raw_path, "-c:v", encoder]
        if encoder == "h264_nvenc":
            cmd += ["-preset", "p4", "-rc", "vbr", "-cq", "23"]
        elif encoder in ("libx264", "libopenh264", "h264"):
            cmd += ["-preset", "veryfast", "-crf", "23"]
        cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", out_path]
        try:
            subprocess.run(
                cmd, check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            try:
                os.remove(raw_path)
            except Exception:
                pass
            log(f"Highlight reel transcoded with {encoder} ({frames_written} frames)")
            return out_path
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode("utf-8", "ignore")[-500:] if e.stderr else ""
            log(f"Highlight reel {encoder} transcode failed; using raw output. {err}")
        except Exception as e:
            log(f"Highlight reel ffmpeg transcode failed ({e}); using raw output")

    # Fallback: use the raw OpenCV output directly.
    try:
        if os.path.exists(out_path):
            os.remove(out_path)
    except Exception:
        pass
    os.replace(raw_path, out_path)
    log(f"Highlight reel written without transcode ({frames_written} frames)")
    return out_path


def run_clipper_pipeline(app, video_path, selections, job_id):
    """Detect touches and build a highlight reel (no pose/3D/prediction)."""
    global pipeline_state

    output_2d = _get_cfg(app, "OUTPUT_2D", OUTPUT_2D)
    workspace_tmp = _get_cfg(app, "WORKSPACE_TMP", None)

    try:
        from touch_ref_util import assign_touch_refs

        pipeline_state["current_step"] = "detecting"
        log("=== CLIPPER STEP 1: Detecting touches ===")
        video_path = ensure_processable_video(video_path)
        touches = []
        video_fps = run_data_extractor(
            app, video_path, output_2d, selections, job_id,
            detect_only=True, touches_out=touches,
        )
        pipeline_state["fps"] = video_fps
        log(f"Detected {len(touches)} touches")

        pipeline_state["current_step"] = "clipping"
        log("=== CLIPPER STEP 2: Building highlight reel ===")
        reel_dir = workspace_tmp or os.path.dirname(video_path)
        os.makedirs(reel_dir, exist_ok=True)
        reel_path = os.path.join(reel_dir, f"{job_id}_highlights.mp4")
        built = build_highlight_reel(video_path, touches, video_fps, reel_path)

        highlight_key = None
        if built and os.path.exists(built) and os.path.getsize(built) > 0:
            highlight_key = f"highlights/{job_id}.mp4"
            log(f"Uploading highlight reel to R2: {highlight_key}")
            s3, bucket = r2_client(app)
            s3.upload_file(
                built, bucket, highlight_key,
                ExtraArgs={"ContentType": "video/mp4"},
            )
            try:
                os.remove(built)
            except Exception:
                pass
        else:
            log("No highlight reel produced (no playable clips).")

        predictions = [
            {"touch": t["touch"], "fencer": t["fencer"], "frame": t["frame"]}
            for t in touches
        ]
        assign_touch_refs(predictions, job_id)

        pipeline_state["current_step"] = "complete"
        pipeline_state["results"] = predictions
        pipeline_state["3d_results"] = None
        pipeline_state["spatial_touch_summary"] = None
        pipeline_state["highlight_reel_key"] = highlight_key
        log(f"=== CLIPPER COMPLETE: {len(predictions)} touches ===")

    except Exception as e:
        pipeline_state["current_step"] = "error"
        pipeline_state["error"] = str(e)
        log(f"ERROR (clipper): {str(e)}")
        import traceback
        traceback.print_exc()


def clipper_pipeline_runner(app_obj, object_key, local_video_path, selections, job_id):
    """Runs in a background thread: download video then run the clipper pipeline."""
    global pipeline_state
    try:
        pipeline_state["current_step"] = "downloading"
        if os.path.isfile(local_video_path) and os.path.getsize(local_video_path) > 0:
            log(f"Using cached local video: {local_video_path}")
        else:
            log(f"Downloading from R2: {object_key} -> {local_video_path}")
            s3, bucket = r2_client(app_obj)
            s3.download_file(bucket, object_key, local_video_path)
            log("video downloaded from R2")

        run_clipper_pipeline(app_obj, local_video_path, selections, job_id)
    except Exception as e:
        pipeline_state["current_step"] = "error"
        pipeline_state["error"] = str(e)
        log(f"ERROR in clipper thread: {e}")


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
        """Convert numpy types to native Python types for JSON serialization.

        Non-finite floats become null — browsers reject `NaN` in JSON.parse and
        that left the 3D viewer stuck on the default origin-only joint (one dot).
        """
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_native(v) for v in obj]
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            v = float(obj)
            return v if math.isfinite(v) else None
        elif isinstance(obj, np.ndarray):
            return to_native(obj.tolist())
        else:
            return obj

    def calculate_3d_angle(a, b, c):
        """Calculate 3D angle at point b."""
        ba = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
        bc = np.asarray(c, dtype=np.float64) - np.asarray(b, dtype=np.float64)
        na = np.linalg.norm(ba)
        nc = np.linalg.norm(bc)
        if na < 1e-8 or nc < 1e-8:
            return 0.0
        cos_angle = float(np.dot(ba, bc) / (na * nc))
        cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
        ang = float(np.degrees(np.arccos(cos_angle)))
        if not np.isfinite(ang):
            return 0.0
        return ang

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

        # Summary statistics (never emit NaN — browsers reject it in JSON.parse)
        acc = movements['acceleration_magnitude']
        movements['summary'] = {
            'avg_displacement': float(np.mean(displacements)) if displacements else 0.0,
            'max_displacement': float(max(displacements)) if displacements else 0.0,
            'min_displacement': float(min(displacements)) if displacements else 0.0,
            'std_displacement': float(np.std(displacements)) if len(displacements) > 1 else 0.0,
            'movement_intensity': float(np.mean([abs(a) for a in acc])) if acc else 0.0,
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
    from job_queue_models import UserJob

    uid = session.get("user_id")
    if not uid:
        return jsonify({"error": "Authentication required"}), 401

    requested_job_id = request.args.get("job_id")
    if requested_job_id:
        if not UserJob.query.filter_by(job_id=requested_job_id, user_id=uid).first():
            return jsonify({
                "current_step": "idle",
                "error": None,
                "results": [],
                "fps": 30,
                "logs": [],
            })

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
    spatial_touch_summary = pipeline_state.pop("spatial_touch_summary", None)
    arm_attempts = pipeline_state.pop("arm_attempts", None)
    highlight_reel_key = pipeline_state.pop("highlight_reel_key", None)

    resp = {
        "current_step": pipeline_state.get("current_step"),
        "error": pipeline_state.get("error"),
        "results": pipeline_state.get("results", []),
        "fps": pipeline_state.get("fps", 30),
        "logs": logs,
    }

    if three_d_results is not None:
        resp["3d_results"] = three_d_results
    if spatial_touch_summary is not None:
        resp["spatial_touch_summary"] = spatial_touch_summary
    if arm_attempts is not None:
        resp["arm_attempts"] = arm_attempts
    if highlight_reel_key is not None:
        from r2_urls import video_playback_url
        resp["highlight_reel_url"] = video_playback_url(highlight_reel_key)

    return jsonify(resp)

@demo_bp.post("/api/reset")
def reset():
    log(f"Reset requested")
    if not session.get("user_id"):
        return jsonify({"success": False, "error": "Authentication required"}), 401
    from job_queue_worker import get_current_job_id
    if get_current_job_id() is not None:
        # Another user's job is still processing; do not wipe global state
        return jsonify({"success": True, "skipped": True})
    global pipeline_state
    pipeline_state = {
        "current_step": "idle",
        "error": None,
        "results": [],
        "3d_results": None,
        "3d_results_durable": None,
        "spatial_touch_summary": None,
        "spatial_touch_summary_durable": None,
        "arm_attempts": None,
        "arm_attempts_durable": None,
        "highlight_reel_key": None,
        "fps": 30,
    }
    return jsonify({"success": True})

@demo_bp.route("/result")
def result():
    return render_template("result.html")

# -------------------------
# Your existing implementations go here with one change:
# add `app` param and use app-config paths instead of module globals
# -------------------------

def run_data_extractor(app, video_path, output_dir, selections, job_id, detect_only=False, touches_out=None):
    """Extract 2D keypoints (RTMDet + COCO-17 ViTPose-H, vertical band gating).

    When detect_only=True, the score-light scan still runs (touch detection),
    but per-touch pose extraction is skipped. Detected touches are appended to
    touches_out as {"touch": <id>, "fencer": <key>, "frame": <int>}. This powers
    the Clipper (highlight-reel) flow, which needs touch timing but not poses.
    """
    from fencing_inference import (
        ensure_pose_stack,
        infer_pose,
        vertical_ref_from_fencer_boxes,
        get_fencer_pair_indices,
        extract_keypoints_dict,
    )

    video_output_dir = os.path.join(output_dir, job_id)
    os.makedirs(video_output_dir, exist_ok=True)
    log(f"Job output dir: {video_output_dir}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Cap processing resolution (4K phone clips can OOM pose inference / stall decode).
    proc_scale = _proc_scale(native_width, native_height)
    frame_width = max(1, int(round(native_width * proc_scale)))
    frame_height = max(1, int(round(native_height * proc_scale)))
    if proc_scale < 1.0:
        log(f"Downscaling {native_width}x{native_height} -> {frame_width}x{frame_height} for processing")

    def _read_proc():
        """cap.read() but resized to the capped processing resolution."""
        ok, fr = cap.read()
        if ok and proc_scale < 1.0 and fr is not None:
            fr = cv2.resize(fr, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
        return ok, fr

    log(f"Video: {job_id}, {frame_width}x{frame_height}, {fps:.1f}fps, {total_frames} frames")

    if not isinstance(selections, dict):
        raise Exception("Missing selections payload. Please re-run and mark fencers/lights.")

    # Scale selections from browser canvas to actual video resolution
    sel_width = selections.get('video_width', frame_width)
    sel_height = selections.get('video_height', frame_height)
    try:
        sel_width = float(sel_width)
        sel_height = float(sel_height)
    except (TypeError, ValueError):
        raise Exception("Invalid selection canvas size. Please re-run selection.")
    if sel_width <= 0 or sel_height <= 0:
        raise Exception("Selection canvas size must be positive.")
    scale_x = frame_width / sel_width
    scale_y = frame_height / sel_height

    def scale_box(box, name):
        if not box:
            raise Exception(f"Missing {name} selection.")
        try:
            x1 = int(float(box['x1']) * scale_x)
            y1 = int(float(box['y1']) * scale_y)
            x2 = int(float(box['x2']) * scale_x)
            y2 = int(float(box['y2']) * scale_y)
        except (KeyError, TypeError, ValueError):
            raise Exception(f"Invalid {name} selection format.")
        if x2 <= x1 or y2 <= y1:
            raise Exception(f"Invalid {name} selection dimensions.")
        return (x1, y1, x2, y2)

    # Clipper (detect_only) finds touches purely from the scoreboard lights, so
    # fencer boxes are optional there. Full analysis still requires them.
    if detect_only:
        fencer1_box = scale_box(selections.get('fencer1'), 'fencer1') if selections.get('fencer1') else None
        fencer2_box = scale_box(selections.get('fencer2'), 'fencer2') if selections.get('fencer2') else None
    else:
        fencer1_box = scale_box(selections.get('fencer1'), 'fencer1')
        fencer2_box = scale_box(selections.get('fencer2'), 'fencer2')
    light_mode = str(selections.get("light_mode") or "static").strip().lower()
    if light_mode not in ("static", "tracking"):
        light_mode = "static"

    score_apparatus = None
    light_tracker = None
    apparatus_history = {}

    if light_mode == "tracking":
        score_apparatus = scale_box(selections.get("score_apparatus"), "score_apparatus")
        fencer1_light, fencer2_light = split_apparatus_to_lights(score_apparatus)
        log(f"Light mode: tracking (apparatus -> left=Fencer1, right=Fencer2)")
    else:
        fencer1_light = scale_box(selections.get('fencer1_light'), 'fencer1_light')
        fencer2_light = scale_box(selections.get('fencer2_light'), 'fencer2_light')
        log("Light mode: static")

    try:
        template_frame_index = int(selections.get("template_frame_index", 0))
    except (TypeError, ValueError):
        template_frame_index = 0
    if light_mode != "tracking":
        template_frame_index = 0
    else:
        template_frame_index = max(0, min(max(total_frames - 1, 0), template_frame_index))
        log(f"Tracking template frame index: {template_frame_index}")

    def _box_str(box):
        x1, y1, x2, y2 = box
        return f"({x1},{y1})-({x2},{y2}) size={x2 - x1}x{y2 - y1}"

    raw_f1_light = selections.get('fencer1_light') or {}
    raw_f2_light = selections.get('fencer2_light') or {}
    raw_apparatus = selections.get("score_apparatus") or {}
    log("=== Score light box placement ===")
    log(
        f"Video native={native_width}x{native_height}, processing={frame_width}x{frame_height}, "
        f"proc_scale={proc_scale:.4f}"
    )
    log(
        f"Selection canvas={sel_width:g}x{sel_height:g}, scale=({scale_x:.4f}, {scale_y:.4f})"
    )
    if abs(scale_x - scale_y) > 0.015:
        log(
            f"WARNING: Uneven X/Y scale ({scale_x:.4f} vs {scale_y:.4f}) — "
            "UI boxes may not land where they appear if canvas aspect ratio differs from processed video."
        )
    if int(round(sel_width)) != frame_width or int(round(sel_height)) != frame_height:
        log(
            f"NOTE: Selection canvas ({int(round(sel_width))}x{int(round(sel_height))}) "
            f"!= processing frame ({frame_width}x{frame_height}); boxes are rescaled."
        )
    if light_mode == "tracking":
        log(f"score_apparatus UI={raw_apparatus} -> processed {_box_str(score_apparatus)}")
    log(f"fencer1_light UI={raw_f1_light} -> processed {_box_str(fencer1_light)}")
    log(f"fencer2_light UI={raw_f2_light} -> processed {_box_str(fencer2_light)}")
    for name, box in (("fencer1_light", fencer1_light), ("fencer2_light", fencer2_light)):
        x1, y1, x2, y2 = box
        if x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
            log(
                f"WARNING: {name} extends outside processed frame "
                f"({frame_width}x{frame_height}) — coordinates will be clipped during sampling."
            )
        if (x2 - x1) * (y2 - y1) < 64:
            log(f"WARNING: {name} is very small ({x2 - x1}x{y2 - y1} px) — may miss lights.")

    # Fencer 1 is always the left person on screen.
    # If the UI labels were reversed, swap BOTH people and their lamp boxes so
    # "Fencer 1 lamp" stays with whoever the user attached to the Fencer 1 label.
    # Static mode still allows Fencer 1's lamp on the left OR right of the board
    # when people are already left/right correct — we do not reorder lamps by X.
    f1_center_x = (fencer1_box[0] + fencer1_box[2]) / 2 if fencer1_box else 0
    f2_center_x = (fencer2_box[0] + fencer2_box[2]) / 2 if fencer2_box else frame_width
    if fencer1_box and fencer2_box and f1_center_x > f2_center_x:
        log(
            "WARNING: Fencer selections were right/left reversed; "
            "swapping people AND lamp boxes so Fencer 1 is the left fencer "
            "(each fencer keeps the lamp the user labeled for them)."
        )
        fencer1_box, fencer2_box = fencer2_box, fencer1_box
        fencer1_light, fencer2_light = fencer2_light, fencer1_light
        f1_center_x, f2_center_x = f2_center_x, f1_center_x
    fencer1_side = "left"

    log(f"Fencer 1 side: {fencer1_side} (always left fencer; lights follow UI / apparatus)")
    if light_mode == "static":
        l1cx = (fencer1_light[0] + fencer1_light[2]) / 2
        l2cx = (fencer2_light[0] + fencer2_light[2]) / 2
        if l1cx > l2cx:
            log(
                "NOTE: Fencer 1 lamp is to the right of Fencer 2 lamp on the scoreboard — "
                "allowed. Scoring attribution still follows these labeled lamp boxes."
            )

    vertical_y0, vertical_y1 = (None, None)
    if fencer1_box and fencer2_box:
        vertical_y0, vertical_y1 = vertical_ref_from_fencer_boxes(
            fencer1_box, fencer2_box, frame_height
        )
    if vertical_y0 is not None and vertical_y1 is not None:
        log(
            f"Vertical band (fencer gate): y in [{vertical_y0:.0f}, {vertical_y1:.0f}] px"
        )
    log(
        f"TRACK setup fencer1_box={list(map(float, fencer1_box)) if fencer1_box else None} "
        f"fencer2_box={list(map(float, fencer2_box)) if fencer2_box else None} "
        f"fencer1_side={fencer1_side} frame={frame_width}x{frame_height}"
    )
    log(
        "TRACK rules: filter=ViTPose upper+lower in vertical gate (full body); "
        "identity=2 largest in-gate bodies by area, left=F1 right=F2"
    )

    if not detect_only:
        ensure_pose_stack(log)

    def run_pose_inference(frame):
        return infer_pose(
            frame,
            vertical_ref_y0=vertical_y0,
            vertical_ref_y1=vertical_y1,
        )

    # Threshold for "light on" (fraction of box pixels that must be that color)
    LIGHT_ON_RATIO = 0.16
    # Baseline calibration uses frame 0 only (resting glow before touches)
    LIGHT_DELTA = 0.10
    # Never require more than this fraction (guards poisoned baselines)
    LIGHT_ON_THRESHOLD_MAX = 0.88
    # High S/V so muddy scoreboard UI / ambient noise do not count as LEDs.
    # Hue kept wide enough for orange-red and chartreuse fencing lights.
    RED_HSV_LO1 = (0, 90, 90)
    RED_HSV_HI1 = (12, 255, 255)
    RED_HSV_LO2 = (168, 90, 90)
    RED_HSV_HI2 = (180, 255, 255)
    GREEN_HSV_LO = (32, 90, 90)
    GREEN_HSV_HI = (88, 255, 255)
    CORE_MIN_SATURATION = 90
    LIGHT_TOUCH_COOLDOWN_SEC = 5
    light_cooldown_frames = max(1, int(LIGHT_TOUCH_COOLDOWN_SEC * fps))
    last_touch_frame = {"fencer1": None, "fencer2": None}

    def _touch_allowed(fencer_key, frame_num):
        last = last_touch_frame[fencer_key]
        if last is None:
            return True
        return (frame_num - last) >= light_cooldown_frames

    def _record_touch(fencer_key, frame_num):
        last_touch_frame[fencer_key] = frame_num

    def _register_touch(fencer_key, exact_frame, note=""):
        """Record a touch if outside cooldown; returns True if registered."""
        nonlocal touches_found
        fencer_label = fencer_key.replace("fencer", "Fencer ")
        if not _touch_allowed(fencer_key, exact_frame):
            log(
                f"Skipping {fencer_label} touch at frame {exact_frame} "
                f"(within {LIGHT_TOUCH_COOLDOWN_SEC}s cooldown)"
            )
            return False
        suffix = f" {note}" if note else ""
        log(f"Touch detected for {fencer_label} at exact frame {exact_frame}{suffix}")
        if detect_only:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            touch_id = f"{fencer_key}_score_{timestamp}_frame{int(exact_frame)}"
            if touches_out is not None:
                touches_out.append(
                    {"touch": touch_id, "fencer": fencer_key, "frame": int(exact_frame)}
                )
        else:
            extract_frames_before_touch(exact_frame, fencer_key)
        touches_found += 1
        _record_touch(fencer_key, exact_frame)
        return True

    def get_light_metrics(frame, box):
        """Return full-box and bright-pixel color fractions for red/green."""
        empty = {"green": 0.0, "red": 0.0, "green_bright": 0.0, "red_bright": 0.0}
        if not box:
            return empty
        x1, y1, x2, y2 = box
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return empty
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        s = hsv[..., 1]
        v = hsv[..., 2]
        red_mask = cv2.inRange(hsv, np.array(RED_HSV_LO1), np.array(RED_HSV_HI1)) + \
                   cv2.inRange(hsv, np.array(RED_HSV_LO2), np.array(RED_HSV_HI2))
        green_mask = cv2.inRange(hsv, np.array(GREEN_HSV_LO), np.array(GREEN_HSV_HI))
        total = region.shape[0] * region.shape[1]
        bright_thr = max(int(np.percentile(v, 55)), 55)
        bright = v >= bright_thr
        bright_count = int(np.count_nonzero(bright))
        red_bool = red_mask > 0
        green_bool = green_mask > 0
        metrics = {
            "green": float(np.count_nonzero(green_bool)) / total,
            "red": float(np.count_nonzero(red_bool)) / total,
        }
        if bright_count > 0:
            metrics["green_bright"] = float(np.count_nonzero(green_bool & bright)) / bright_count
            metrics["red_bright"] = float(np.count_nonzero(red_bool & bright)) / bright_count
        else:
            metrics["green_bright"] = 0.0
            metrics["red_bright"] = 0.0
        return metrics

    def get_light_ratios(frame, box):
        """Backward-compatible ratio dict used by color assignment fallbacks."""
        m = get_light_metrics(frame, box)
        return {"green": m["green"], "red": m["red"]}

    def _effective_color_signal(metrics, color):
        """Use the stronger of full-box or bright-pixel signal (helps large boxes)."""
        if color == "green":
            return max(metrics["green"], metrics["green_bright"])
        return max(metrics["red"], metrics["red_bright"])

    def _light_on_threshold(baseline):
        """Minimum signal to count as on: resting baseline + delta, with floor and ceiling."""
        raw = max(float(baseline) + LIGHT_DELTA, LIGHT_ON_RATIO)
        return min(raw, LIGHT_ON_THRESHOLD_MAX)

    def _color_is_on(metrics, color, baseline):
        signal = _effective_color_signal(metrics, color)
        if signal > _light_on_threshold(baseline):
            return True
        # Relative rise when resting bleed is high but below the hard cap
        if baseline >= 0.18 and signal >= LIGHT_ON_RATIO:
            relative = baseline + max(LIGHT_DELTA * 0.6, baseline * 0.10)
            return signal > min(relative, LIGHT_ON_THRESHOLD_MAX)
        return False

    def _check_light_any(frame, box, baselines):
        """Discovery only: on if either color exceeds its baseline-adjusted threshold."""
        m = get_light_metrics(frame, box)
        if _color_is_on(m, "green", baselines["green"]) or _color_is_on(m, "red", baselines["red"]):
            return "on"
        return "off"

    def check_score_light(frame, box, color, baselines):
        """Color-specific: on only if the given color exceeds its baseline-adjusted threshold."""
        if not box:
            return "off"
        m = get_light_metrics(frame, box)
        if _color_is_on(m, color, baselines[color]):
            return "on"
        return "off"

    def calibrate_light_baselines():
        """Learn resting red/green bleed from the tracking template frame (or frame 0)."""
        cap.set(cv2.CAP_PROP_POS_FRAMES, template_frame_index)
        ret, fr = _read_proc()
        if not ret:
            log(
                f"WARNING: Could not read frame {template_frame_index} for light baseline; "
                "using zero baselines"
            )
            return {"green": 0.0, "red": 0.0}, {"green": 0.0, "red": 0.0}

        m1 = get_light_metrics(fr, fencer1_light)
        m2 = get_light_metrics(fr, fencer2_light)

        def _baseline_from_metrics(metrics, label):
            base = {
                "green": _effective_color_signal(metrics, "green"),
                "red": _effective_color_signal(metrics, "red"),
            }
            for color in ("green", "red"):
                if base[color] > 0.35:
                    log(
                        f"WARNING: Baseline {label} {color} on frame {template_frame_index} is already high "
                        f"({base[color]:.3f}) — start video with lights off if detection fails."
                    )
            return base

        f1_base = _baseline_from_metrics(m1, "fencer1")
        f2_base = _baseline_from_metrics(m2, "fencer2")
        log(
            "Light baselines (template frame "
            f"{template_frame_index}): "
            f"Fencer1 green={f1_base['green']:.3f} red={f1_base['red']:.3f} "
            f"(on>{_light_on_threshold(f1_base['green']):.3f}/{_light_on_threshold(f1_base['red']):.3f}), "
            f"Fencer2 green={f2_base['green']:.3f} red={f2_base['red']:.3f} "
            f"(on>{_light_on_threshold(f2_base['green']):.3f}/{_light_on_threshold(f2_base['red']):.3f})"
        )
        return f1_base, f2_base

    def _save_light_box_debug(label, frame, baselines1=None, baselines2=None):
        """Write overlay + per-box crops so RunPod logs/paths can verify placement."""
        if frame is None:
            log(f"Skipping debug image '{label}': frame unavailable")
            return
        vis = frame.copy()
        for name, box, color, baselines in (
            ("fencer1_light", fencer1_light, (0, 255, 255), baselines1),
            ("fencer2_light", fencer2_light, (0, 200, 255), baselines2),
        ):
            x1, y1, x2, y2 = box
            x1c = max(0, min(frame_width - 1, x1))
            y1c = max(0, min(frame_height - 1, y1))
            x2c = max(x1c + 1, min(frame_width, x2))
            y2c = max(y1c + 1, min(frame_height, y2))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            m = get_light_metrics(frame, box)
            g_sig = _effective_color_signal(m, "green")
            r_sig = _effective_color_signal(m, "red")
            cv2.putText(
                vis, name, (x1, max(16, y1 - 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )
            if baselines:
                cv2.putText(
                    vis,
                    f"g={g_sig:.2f}>{_light_on_threshold(baselines['green']):.2f} "
                    f"r={r_sig:.2f}>{_light_on_threshold(baselines['red']):.2f}",
                    (x1, max(36, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    vis, f"g={g_sig:.2f} r={r_sig:.2f}",
                    (x1, max(36, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
                )
            crop = frame[y1c:y2c, x1c:x2c]
            if crop.size > 0:
                crop_path = os.path.join(video_output_dir, f"debug_{label}_{name}_crop.jpg")
                cv2.imwrite(crop_path, crop)
                log(
                    f"Saved light crop: {crop_path} ({crop.shape[1]}x{crop.shape[0]} px) "
                    f"signals g={g_sig:.3f} r={r_sig:.3f}"
                )
        overlay_path = os.path.join(video_output_dir, f"debug_{label}_light_overlay.jpg")
        cv2.imwrite(overlay_path, vis)
        log(f"Saved light overlay: {overlay_path} ({frame_width}x{frame_height})")

    def find_exact_light_on_frame(off_frame, on_frame, fencer_key, baselines, color=None):
        """Binary search to find exact frame where light turns on. color=None uses _check_light_any (discovery)."""
        left = off_frame
        right = on_frame
        exact_frame = on_frame

        log(f"Binary search between frames {off_frame} (OFF) and {on_frame} (ON) for {fencer_key}")

        def _score_box_at(mid_frame):
            if light_tracker is None or score_apparatus is None:
                return fencer1_light if fencer_key == "fencer1" else fencer2_light
            app_off = apparatus_history.get(off_frame, score_apparatus)
            app_on = apparatus_history.get(on_frame, light_tracker.current_apparatus)
            app_mid = interpolate_box(off_frame, app_off, on_frame, app_on, mid_frame)
            f1_box, f2_box = split_apparatus_to_lights(app_mid)
            return f1_box if fencer_key == "fencer1" else f2_box

        while left <= right:
            mid = (left + right) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ret, frame = _read_proc()

            if not ret:
                break

            score_box = _score_box_at(mid)
            if color is None:
                light_state = _check_light_any(frame, score_box, baselines)
            else:
                light_state = check_score_light(frame, score_box, color, baselines)

            if light_state == "on":
                exact_frame = mid
                right = mid - 1
            else:
                left = mid + 1

        log(f"Found exact frame: {exact_frame}")
        return exact_frame

    def get_fencer_ids(results, audit_out=None):
        return get_fencer_pair_indices(
            results,
            fencer1_side,
            vertical_y0,
            vertical_y1,
            frame_height,
            frame_width,
            fencer1_ref_box=fencer1_box,
            fencer2_ref_box=fencer2_box,
            debug_log=log,
            audit_out=audit_out,
        )

    def extract_keypoints(results, fencer_id):
        return extract_keypoints_dict(results, fencer_id)

    def extract_frames_before_touch(touch_frame, scoring_fencer):
        from datetime import datetime
        # Location model uses seq 1..light (≤30). Attack type uses trailing
        # ATTACK_WINDOW_SEC through the light (no post-light frames).
        start = max(0, touch_frame - 29)
        end = touch_frame  # end on light (no +5 post-light)
        light_frame_seq = touch_frame - start + 1  # 1-based seq of the light frame
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        group_id = f"{scoring_fencer}_score_{timestamp}_frame{touch_frame}"
        group_dir = os.path.join(video_output_dir, group_id)
        os.makedirs(group_dir, exist_ok=True)

        # 1. Seek once, read all frames sequentially (faster than random seeking)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(end - start + 1):
            ret, fr = _read_proc()
            if ret:
                frames.append(fr)

        if not frames:
            log(f"Touch at frame {touch_frame}: no frames could be read")
            return 0

        # 2. Per-frame ViTPose-H pose (vertical band matches setup fencers)
        log(f"Running ViTPose-H pose on {len(frames)} frames "
            f"(light_seq={light_frame_seq}, end=on light)...")
        all_results = [run_pose_inference(fr) for fr in frames]

        # 3. Save results - write keypoints only for frames with detections (frame numbers may have gaps)
        saved = 0
        skipped_no_pair = 0
        skipped_null_kp = 0
        skipped_no_dets = 0
        track_audit_frames = []
        for idx, (fr, results) in enumerate(zip(frames, all_results)):
            frame_num = idx + 1
            frame_audit = {
                "seq": frame_num,
                "touch_frame": touch_frame,
                "light_frame_seq": light_frame_seq,
                "video_frame_index": start + idx,
                "n_bboxes": len((results or {}).get("bboxes") or []),
            }
            if not results or not results.get('bboxes'):
                skipped_no_dets += 1
                frame_audit["status"] = "no_dets"
                frame_audit["pose_filter"] = (results or {}).get("_pose_filter_audit")
                frame_audit["debug_before"] = (results or {}).get("_debug_before_filter")
                track_audit_frames.append(frame_audit)
                log(
                    f"TRACK touch={touch_frame} seq={frame_num}: no detections "
                    f"(pose_filter={(results or {}).get('_pose_filter_audit')})"
                )
                continue
            pair_audit = {}
            f1_id, f2_id = get_fencer_ids(results, audit_out=pair_audit)
            frame_audit["pair"] = pair_audit
            frame_audit["f1_id"] = f1_id
            frame_audit["f2_id"] = f2_id
            if f1_id is None or f2_id is None:
                skipped_no_pair += 1
                frame_audit["status"] = "no_pair"
                track_audit_frames.append(frame_audit)
                continue
            f1_kp = extract_keypoints(results, f1_id)
            f2_kp = extract_keypoints(results, f2_id)
            if not f1_kp or not f2_kp:
                skipped_null_kp += 1
                frame_audit["status"] = "null_kp"
                frame_audit["f1_kp_ok"] = bool(f1_kp)
                frame_audit["f2_kp_ok"] = bool(f2_kp)
                track_audit_frames.append(frame_audit)
                log(
                    f"Touch frame {touch_frame} seq {frame_num}: "
                    f"null keypoints (f1_id={f1_id}, f2_id={f2_id})"
                )
                continue

            frame_audit["status"] = "saved"
            frame_audit["method"] = pair_audit.get("method")
            # Compact chosen-box snapshot for quick scan of identity switches.
            boxes = results.get("bboxes") or []
            if f1_id < len(boxes):
                frame_audit["f1_bbox"] = [round(float(v), 1) for v in boxes[f1_id][:4]]
            if f2_id < len(boxes):
                frame_audit["f2_bbox"] = [round(float(v), 1) for v in boxes[f2_id][:4]]
            track_audit_frames.append(frame_audit)

            cv2.imwrite(os.path.join(group_dir, f"frame_{frame_num}.jpg"), fr)
            with open(os.path.join(group_dir, f"frame_{frame_num}_keypoints.json"), "w") as f:
                json.dump({
                    "frame_index": frame_num,
                    "video_frame_index": start + idx,
                    "light_frame_seq": light_frame_seq,
                    "frame_width": frame_width,
                    "frame_height": frame_height,
                    "fps": float(fps) if fps else None,
                    "scoring_fencer": scoring_fencer,
                    "fencer1_keypoints": f1_kp,
                    "fencer2_keypoints": f2_kp,
                    "fencer1_det_index": f1_id,
                    "fencer2_det_index": f2_id,
                    "track_method": pair_audit.get("method"),
                }, f, indent=2)
            saved += 1

        audit_path = os.path.join(group_dir, "track_audit.json")
        with open(audit_path, "w") as f:
            json.dump(
                {
                    "touch_frame": touch_frame,
                    "light_frame_seq": light_frame_seq,
                    "extract_start_frame": start,
                    "extract_end_frame": end,
                    "scoring_fencer": scoring_fencer,
                    "fencer1_box": list(map(float, fencer1_box)) if fencer1_box else None,
                    "fencer2_box": list(map(float, fencer2_box)) if fencer2_box else None,
                    "vertical_gate": [
                        None if vertical_y0 is None else float(vertical_y0),
                        None if vertical_y1 is None else float(vertical_y1),
                    ],
                    "fencer1_side": fencer1_side,
                    "frame_width": frame_width,
                    "frame_height": frame_height,
                    "saved": saved,
                    "skipped_no_dets": skipped_no_dets,
                    "skipped_no_pair": skipped_no_pair,
                    "skipped_null_kp": skipped_null_kp,
                    "frames": track_audit_frames,
                },
                f,
                indent=2,
            )
        log(f"TRACK audit written: {audit_path}")

        # Summarize identity stability / failure modes for the job log.
        methods = {}
        fail_reasons = {}
        for fa in track_audit_frames:
            st = fa.get("status")
            if st == "saved":
                m = fa.get("method") or "unknown"
                methods[m] = methods.get(m, 0) + 1
            else:
                reason = st
                pair = fa.get("pair") or {}
                if pair.get("fail_reason"):
                    reason = f"{st}:{pair['fail_reason']}"
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
        summary = (
            f"TRACK SUMMARY touch_frame={touch_frame} scoring={scoring_fencer} "
            f"saved={saved} no_dets={skipped_no_dets} no_pair={skipped_no_pair} "
            f"null_kp={skipped_null_kp} methods={methods} fails={fail_reasons}"
        )
        log(summary)
        track_summaries.append(
            {
                "touch_frame": touch_frame,
                "scoring_fencer": scoring_fencer,
                "saved": saved,
                "skipped_no_dets": skipped_no_dets,
                "skipped_no_pair": skipped_no_pair,
                "skipped_null_kp": skipped_null_kp,
                "methods": methods,
                "fails": fail_reasons,
                "audit_path": audit_path,
                # One-line-per-frame status for RunPod (no full score tables).
                "frame_status": [
                    {
                        "seq": fa.get("seq"),
                        "status": fa.get("status"),
                        "f1_id": fa.get("f1_id"),
                        "f2_id": fa.get("f2_id"),
                        "method": (fa.get("pair") or {}).get("method") or fa.get("method"),
                        "fail_reason": (fa.get("pair") or {}).get("fail_reason"),
                        "n_bboxes": fa.get("n_bboxes"),
                        "f1_bbox": fa.get("f1_bbox"),
                        "f2_bbox": fa.get("f2_bbox"),
                    }
                    for fa in track_audit_frames
                ],
            }
        )
        for fs in track_summaries[-1]["frame_status"]:
            log(
                f"TRACK frame touch={touch_frame} seq={fs['seq']} status={fs['status']} "
                f"n_bboxes={fs['n_bboxes']} f1={fs['f1_id']} f2={fs['f2_id']} "
                f"method={fs['method']} fail={fs['fail_reason']} "
                f"f1_bbox={fs['f1_bbox']} f2_bbox={fs['f2_bbox']}"
            )
        # Keep fencer1_keypoints = left person across the clip. Tracking fallbacks can
        # flip identities when people cross; without this the scorer skeleton / attack /
        # area models read the wrong person when lamps aren't mirrored to strip sides.
        swapped = enforce_fencer1_left_in_touch_folder(group_dir)
        if swapped:
            log(f"TRACK identity fix: swapped fencer1/fencer2 keypoints in {group_id} (F1 must be left)")
        return saved

    track_summaries = []

    # Light boxes are user-assigned: fencer1_light / fencer2_light follow UI labels (not screen left/right).

    # Scan for touches; assign which box is green vs red from first activation (real = earlier/bolder, bleed = later/weaker)
    log("Scanning for score lights...")
    prev_f1_light = "off"
    prev_f2_light = "off"
    fencer1_light_color = None  # "green" or "red", set after first touch
    fencer2_light_color = None
    sample_interval = 1 if light_mode == "tracking" else 25
    if light_mode == "tracking":
        log(f"Tracking mode: scanning every {sample_interval} frame(s) for apparatus follow")
    current_frame = template_frame_index if light_mode == "tracking" else 0
    touches_found = 0

    def _compute_color_core_strength(frame, box, color):
        """Return brightness-weighted strength for the given color ('red' or 'green') within the box."""
        if not box:
            return 0.0
        x1, y1, x2, y2 = box
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return 0.0

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        if color == "red":
            mask = cv2.inRange(hsv, np.array(RED_HSV_LO1), np.array(RED_HSV_HI1)) + \
                   cv2.inRange(hsv, np.array(RED_HSV_LO2), np.array(RED_HSV_HI2))
        else:
            mask = cv2.inRange(hsv, np.array(GREEN_HSV_LO), np.array(GREEN_HSV_HI))

        v = hsv[..., 2]
        s = hsv[..., 1]
        core_mask = (mask > 0) & (s >= CORE_MIN_SATURATION)
        vals = v[core_mask]
        if vals.size < 10:
            return 0.0

        thr = np.percentile(vals, 95)
        core = vals[vals >= thr]
        if core.size == 0:
            return 0.0

        # Combine how bright and how concentrated the core is
        return float(core.mean() * core.size)

    def assign_light_colors(frame, box1, box2, baselines1, baselines2):
        """Set fencer1_light_color and fencer2_light_color from brightness-weighted core strengths."""
        nonlocal fencer1_light_color, fencer2_light_color

        strengths = [
            ("fencer1", "green", _compute_color_core_strength(frame, box1, "green")),
            ("fencer1", "red", _compute_color_core_strength(frame, box1, "red")),
            ("fencer2", "green", _compute_color_core_strength(frame, box2, "green")),
            ("fencer2", "red", _compute_color_core_strength(frame, box2, "red")),
        ]

        best_fencer, best_color, best_strength = max(strengths, key=lambda t: t[2])

        if best_strength <= 0.0:
            # Fallback to ratio-based assignment if core strengths are not usable
            m1 = get_light_metrics(frame, box1)
            m2 = get_light_metrics(frame, box2)
            r1 = get_light_ratios(frame, box1)
            r2 = get_light_ratios(frame, box2)
            s1 = max(r1["green"], r1["red"])
            s2 = max(r2["green"], r2["red"])
            dom1 = "green" if r1["green"] > r1["red"] and _color_is_on(m1, "green", baselines1["green"]) else (
                "red" if _color_is_on(m1, "red", baselines1["red"]) else None
            )
            dom2 = "green" if r2["green"] > r2["red"] and _color_is_on(m2, "green", baselines2["green"]) else (
                "red" if _color_is_on(m2, "red", baselines2["red"]) else None
            )
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
            log(f"Assigned (fallback ratios): Fencer 1 light = {fencer1_light_color}, Fencer 2 light = {fencer2_light_color}")
            return

        if best_fencer == "fencer1":
            fencer1_light_color = best_color
            fencer2_light_color = "red" if best_color == "green" else "green"
        else:
            fencer2_light_color = best_color
            fencer1_light_color = "red" if best_color == "green" else "green"

        log(f"Assigned (core strength): Fencer 1 light = {fencer1_light_color}, Fencer 2 light = {fencer2_light_color}; strengths={strengths}")

    fencer1_light_baselines, fencer2_light_baselines = calibrate_light_baselines()
    cap.set(cv2.CAP_PROP_POS_FRAMES, template_frame_index)
    ret, baseline_debug_fr = _read_proc()
    if ret and light_mode == "tracking" and score_apparatus is not None:
        light_tracker = ScoreboardTracker(baseline_debug_fr, score_apparatus)
        apparatus_history[template_frame_index] = score_apparatus
        backend = light_tracker.backend.value
        log(
            f"Score light tracker ({backend}) initialized @ frame {template_frame_index} "
            f"apparatus={_box_str(score_apparatus)}"
        )
    if ret:
        _save_light_box_debug(
            f"baseline_frame{template_frame_index}",
            baseline_debug_fr,
            fencer1_light_baselines,
            fencer2_light_baselines,
        )

    peak_signals = {
        "fencer1": {"green": 0.0, "red": 0.0, "frame": 0},
        "fencer2": {"green": 0.0, "red": 0.0, "frame": 0},
    }

    def _record_peak_signals(frame_num, frame):
        for fencer_key, box, baselines in (
            ("fencer1", fencer1_light, fencer1_light_baselines),
            ("fencer2", fencer2_light, fencer2_light_baselines),
        ):
            m = get_light_metrics(frame, box)
            for color in ("green", "red"):
                signal = _effective_color_signal(m, color)
                if signal > peak_signals[fencer_key][color]:
                    peak_signals[fencer_key][color] = signal
                    peak_signals[fencer_key]["frame"] = frame_num

    def _largest_touch_delta(frame):
        """Pick the fencer/color with the largest rise above baseline at this frame."""
        best = None
        best_delta = 0.0
        for fencer_key, box, baselines in (
            ("fencer1", fencer1_light, fencer1_light_baselines),
            ("fencer2", fencer2_light, fencer2_light_baselines),
        ):
            m = get_light_metrics(frame, box)
            for color in ("green", "red"):
                signal = _effective_color_signal(m, color)
                delta = signal - float(baselines[color])
                if delta > best_delta:
                    best_delta = delta
                    best = (fencer_key, color, signal, float(baselines[color]))
        return best, best_delta

    def _update_tracked_light_boxes(frame_num, frame, lights_on=False):
        nonlocal fencer1_light, fencer2_light
        if light_tracker is None:
            return
        apparatus = light_tracker.update(frame, lights_on=lights_on)
        apparatus_history[frame_num] = apparatus
        fencer1_light, fencer2_light = light_tracker.light_boxes()
        if frame_num % 500 == 0:
            snap = light_tracker.snapshot()
            log(
                f"Tracker @ frame {frame_num}: confidence={snap['confidence']} "
                f"apparatus={_box_str(apparatus)}"
            )

    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, fr = _read_proc()
        if not ret:
            break

        lights_on_now = (
            fencer1_light_color is not None
            and (prev_f1_light == "on" or prev_f2_light == "on")
        )
        _update_tracked_light_boxes(current_frame, fr, lights_on=lights_on_now)

        if fencer1_light_color is None:
            f1_light = _check_light_any(fr, fencer1_light, fencer1_light_baselines)
            f2_light = _check_light_any(fr, fencer2_light, fencer2_light_baselines)
        else:
            f1_light = check_score_light(fr, fencer1_light, fencer1_light_color, fencer1_light_baselines)
            f2_light = check_score_light(fr, fencer2_light, fencer2_light_color, fencer2_light_baselines)

        _record_peak_signals(current_frame, fr)

        f1_new_touch = f1_light == "on" and prev_f1_light == "off"
        f2_new_touch = f2_light == "on" and prev_f2_light == "off"

        if f1_new_touch or f2_new_touch:
            search_start = max(0, current_frame - sample_interval)

            if fencer1_light_color is None:
                # First touch: assign colors from ratios at current frame (real light is bolder)
                assign_light_colors(fr, fencer1_light, fencer2_light, fencer1_light_baselines, fencer2_light_baselines)
                # Re-evaluate who actually touched with color-specific check (drops false bleed)
                f1_new_touch = check_score_light(fr, fencer1_light, fencer1_light_color, fencer1_light_baselines) == "on" and prev_f1_light == "off"
                f2_new_touch = check_score_light(fr, fencer2_light, fencer2_light_color, fencer2_light_baselines) == "on" and prev_f2_light == "off"
                if not f1_new_touch and not f2_new_touch:
                    best, best_delta = _largest_touch_delta(fr)
                    if best and best_delta >= LIGHT_DELTA:
                        touch_fencer, touch_color, signal, baseline = best
                        log(
                            "First-touch fallback: "
                            f"{touch_fencer} {touch_color} signal={signal:.3f} baseline={baseline:.3f} delta={best_delta:.3f}"
                        )
                        if touch_fencer == "fencer1":
                            fencer1_light_color = touch_color
                            fencer2_light_color = "red" if touch_color == "green" else "green"
                            f1_new_touch = True
                        else:
                            fencer2_light_color = touch_color
                            fencer1_light_color = "red" if touch_color == "green" else "green"
                            f2_new_touch = True

            # Handle detected touch(es) with color-specific exact frame
            if f1_new_touch:
                exact_frame = find_exact_light_on_frame(
                    search_start, current_frame, "fencer1", fencer1_light_baselines, fencer1_light_color
                )
                _register_touch("fencer1", exact_frame)
                prev_f1_light = "on"

            if f2_new_touch:
                exact_frame = find_exact_light_on_frame(
                    search_start, current_frame, "fencer2", fencer2_light_baselines, fencer2_light_color
                )
                _register_touch("fencer2", exact_frame)
                prev_f2_light = "on"

            # If only one light triggered, check next 30 frames for the other (color-specific)
            if f1_new_touch != f2_new_touch and fencer1_light_color is not None:
                for check_frame in range(current_frame, min(current_frame + 30, total_frames)):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, check_frame)
                    ret, check_fr = _read_proc()
                    if not ret:
                        break
                    _update_tracked_light_boxes(
                        check_frame,
                        check_fr,
                        lights_on=(prev_f1_light == "on" or prev_f2_light == "on"),
                    )
                    if f1_new_touch and prev_f2_light == "off":
                        if check_score_light(check_fr, fencer2_light, fencer2_light_color, fencer2_light_baselines) == "on":
                            exact_frame = find_exact_light_on_frame(
                                search_start, check_frame, "fencer2", fencer2_light_baselines, fencer2_light_color
                            )
                            _register_touch("fencer2", exact_frame, note="(double touch)")
                            prev_f2_light = "on"
                            break
                    if f2_new_touch and prev_f1_light == "off":
                        if check_score_light(check_fr, fencer1_light, fencer1_light_color, fencer1_light_baselines) == "on":
                            exact_frame = find_exact_light_on_frame(
                                search_start, check_frame, "fencer1", fencer1_light_baselines, fencer1_light_color
                            )
                            _register_touch("fencer1", exact_frame, note="(double touch)")
                            prev_f1_light = "on"
                            break

            # Skip ahead past the light duration
            current_frame += 150
            # Wait for both lights to turn off (color-specific)
            while current_frame < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, fr = _read_proc()
                if not ret:
                    break
                _update_tracked_light_boxes(
                    current_frame,
                    fr,
                    lights_on=(prev_f1_light == "on" or prev_f2_light == "on"),
                )
                f1_off = check_score_light(fr, fencer1_light, fencer1_light_color, fencer1_light_baselines) == "off"
                f2_off = check_score_light(fr, fencer2_light, fencer2_light_color, fencer2_light_baselines) == "off"
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
            m1 = get_light_metrics(fr, fencer1_light)
            m2 = get_light_metrics(fr, fencer2_light)
            log(
                f"Progress {current_frame}/{total_frames}: "
                f"f1 g={_effective_color_signal(m1, 'green'):.3f} r={_effective_color_signal(m1, 'red'):.3f}, "
                f"f2 g={_effective_color_signal(m2, 'green'):.3f} r={_effective_color_signal(m2, 'red'):.3f}"
            )

    cap.release()
    log("========== TRACK SUMMARY (search RunPod log for TRACK) ==========")
    if not track_summaries:
        log("TRACK SUMMARY: no per-touch pose tracking ran (detect_only or no touches)")
    for ts in track_summaries:
        log(
            f"TRACK SUMMARY touch_frame={ts['touch_frame']} scoring={ts['scoring_fencer']} "
            f"saved={ts['saved']} no_dets={ts['skipped_no_dets']} "
            f"no_pair={ts['skipped_no_pair']} null_kp={ts['skipped_null_kp']} "
            f"methods={ts['methods']} fails={ts['fails']}"
        )
        for fs in ts["frame_status"]:
            log(
                f"TRACK frame touch={ts['touch_frame']} seq={fs['seq']} "
                f"status={fs['status']} n_bboxes={fs['n_bboxes']} "
                f"f1={fs['f1_id']} f2={fs['f2_id']} method={fs['method']} "
                f"fail={fs['fail_reason']} f1_bbox={fs['f1_bbox']} f2_bbox={fs['f2_bbox']}"
            )
    log("========== END TRACK SUMMARY ==========")
    log(f"Extraction complete: {touches_found} touches found")

    if touches_found == 0:
        for fencer_key in ("fencer1", "fencer2"):
            baselines = fencer1_light_baselines if fencer_key == "fencer1" else fencer2_light_baselines
            peaks = peak_signals[fencer_key]
            log(
                f"Peak signals {fencer_key} @ frame {peaks['frame']}: "
                f"green={peaks['green']:.3f} (on>{_light_on_threshold(baselines['green']):.3f}), "
                f"red={peaks['red']:.3f} (on>{_light_on_threshold(baselines['red']):.3f})"
            )
        peak_frame = max(peak_signals["fencer1"]["frame"], peak_signals["fencer2"]["frame"])
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, peak_frame)
            ret, peak_fr = cap.read()
            if ret and proc_scale < 1.0 and peak_fr is not None:
                peak_fr = cv2.resize(
                    peak_fr, (frame_width, frame_height), interpolation=cv2.INTER_AREA
                )
            if ret:
                log(f"Saving peak-frame placement debug @ frame {peak_frame}")
                _save_light_box_debug(
                    f"peak_frame{peak_frame}",
                    peak_fr,
                    fencer1_light_baselines,
                    fencer2_light_baselines,
                )
            cap.release()
        raise Exception("No touches detected. Check your score light box selections.")

    return fps


def _kp_center_x(kp_dict):
    """Mean joint x from a keypoints dict; None if empty."""
    if not kp_dict or not isinstance(kp_dict, dict):
        return None
    xs = []
    for k, v in kp_dict.items():
        if k.endswith("_conf"):
            continue
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            continue
        try:
            xs.append(float(v[0]))
        except (TypeError, ValueError):
            continue
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def enforce_fencer1_left_in_touch_folder(seq_folder: str) -> bool:
    """Swap fencer1/fencer2 keypoints in a touch clip if F1 is not the left person.

    Pose tracking can flip identities when fencers cross. Attack / area / 3D animation
    all key off `fencer1_*` vs `fencer2_*` plus `scoring_fencer`, so a flip attributes
    the wrong skeleton to the scorer — especially noticeable when the scorer's lamp is
    on the opposite side of the scoreboard from where they stand.
    """
    paths = sorted(
        glob.glob(os.path.join(seq_folder, "frame_*_keypoints.json")),
        key=lambda p: int(os.path.basename(p).split("_")[1]),
    )
    if not paths:
        return False

    f1_xs, f2_xs = [], []
    for path in paths:
        try:
            with open(path) as fp:
                d = json.load(fp)
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        c1 = _kp_center_x(d.get("fencer1_keypoints"))
        c2 = _kp_center_x(d.get("fencer2_keypoints"))
        if c1 is not None:
            f1_xs.append(c1)
        if c2 is not None:
            f2_xs.append(c2)

    if not f1_xs or not f2_xs:
        return False

    med1 = float(sorted(f1_xs)[len(f1_xs) // 2])
    med2 = float(sorted(f2_xs)[len(f2_xs) // 2])
    if med1 <= med2:
        return False

    for path in paths:
        try:
            with open(path) as fp:
                d = json.load(fp)
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        d["fencer1_keypoints"], d["fencer2_keypoints"] = (
            d.get("fencer2_keypoints"),
            d.get("fencer1_keypoints"),
        )
        if "fencer1_det_index" in d or "fencer2_det_index" in d:
            d["fencer1_det_index"], d["fencer2_det_index"] = (
                d.get("fencer2_det_index"),
                d.get("fencer1_det_index"),
            )
        d["identity_swapped_to_f1_left"] = True
        try:
            with open(path, "w") as fp:
                json.dump(d, fp, indent=2)
        except OSError:
            continue
    return True


def _bbox_xyxy_from_kp_dict(kp_dict):
    """Tight axis-aligned bbox from joint dict (name -> [x, y])."""
    if not kp_dict or not isinstance(kp_dict, dict):
        return None
    xs, ys = [], []
    for k, v in kp_dict.items():
        if k.endswith("_conf"):
            continue
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            continue
        try:
            x, y = float(v[0]), float(v[1])
        except (TypeError, ValueError):
            continue
        xs.append(x)
        ys.append(y)
    if len(xs) < 3:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _infer_frame_size_fallback_last_frame(d_keypoints_json):
    """
    Legacy fallback when frame_width/height were not saved: estimate from one frame's
    pose boxes (underestimates real width if bodies don't reach frame edges — re-run extraction).
    """
    b1 = _bbox_xyxy_from_kp_dict(d_keypoints_json.get("fencer1_keypoints") or {})
    b2 = _bbox_xyxy_from_kp_dict(d_keypoints_json.get("fencer2_keypoints") or {})
    if not b1 or not b2:
        return None, None
    max_x = max(b1[2], b2[2])
    max_y = max(b1[3], b2[3])
    w = int(max(max_x * 1.08, 640))
    h = int(max(max_y * 1.08, 360))
    return w, h


def _horizontal_third(engage_mid_x, frame_w):
    """Return 'left' | 'center' | 'right' for x in [0, frame_w)."""
    if frame_w <= 0:
        return "center"
    t = frame_w / 3.0
    if engage_mid_x < t:
        return "left"
    if engage_mid_x < 2.0 * t:
        return "center"
    return "right"


def _pressing_touch_for_scorer_thirds(scoring_fencer, f1_left_of_f2, engage_mid_x, frame_w):
    """
    True if engagement center lies in the opponent's horizontal third (deep attack heuristic).
    """
    if frame_w <= 0:
        return False
    region = _horizontal_third(engage_mid_x, frame_w)
    if scoring_fencer == "fencer1":
        if f1_left_of_f2:
            return region == "right"
        return region == "left"
    if scoring_fencer == "fencer2":
        if f1_left_of_f2:
            return region == "left"
        return region == "right"
    return False


def compute_touch_spatial_meta(seq_folder, files_sorted):
    """
    Per-touch spatial summary using 2D pose bboxes at the light frame (not post-light).
    Engagement x = average of attacker (scorer) and defender bbox center x.
    Screen split: horizontal thirds (left / center / right). Frame size from video metadata
    saved in keypoints JSON (stable); avoid inferring from keypoint spread alone.
    """
    if not files_sorted:
        return None

    def _frame_num(path):
        try:
            return int(os.path.basename(path).split("_")[1])
        except (IndexError, ValueError):
            return -1

    light_seq = None
    scoring_fencer = None
    for path in files_sorted:
        try:
            with open(path) as fp:
                d = json.load(fp)
            if light_seq is None and d.get("light_frame_seq") is not None:
                light_seq = int(d["light_frame_seq"])
            sf = d.get("scoring_fencer")
            if scoring_fencer is None and sf in ("fencer1", "fencer2"):
                scoring_fencer = sf
            if light_seq is not None and scoring_fencer is not None:
                break
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue

    if scoring_fencer is None:
        for path in (files_sorted[-1], files_sorted[0]):
            try:
                with open(path) as fp:
                    d = json.load(fp)
                sf = d.get("scoring_fencer")
                if sf in ("fencer1", "fencer2"):
                    scoring_fencer = sf
                    break
            except (OSError, json.JSONDecodeError, TypeError):
                continue
    if not scoring_fencer:
        return None

    # Prefer the light frame; fall back to frame_30 / last pre-post frame / last file.
    light_path = None
    if light_seq is not None:
        candidate = os.path.join(seq_folder, f"frame_{light_seq}_keypoints.json")
        if os.path.isfile(candidate):
            light_path = candidate
        else:
            pre = [p for p in files_sorted if _frame_num(p) <= light_seq]
            if pre:
                light_path = pre[-1]
    if light_path is None:
        p30 = os.path.join(seq_folder, "frame_30_keypoints.json")
        if os.path.isfile(p30):
            light_path = p30
        else:
            pre30 = [p for p in files_sorted if _frame_num(p) <= 30]
            light_path = pre30[-1] if pre30 else files_sorted[-1]

    try:
        with open(light_path) as fp:
            d_last = json.load(fp)
    except (OSError, json.JSONDecodeError, TypeError):
        return None

    b1 = _bbox_xyxy_from_kp_dict(d_last.get("fencer1_keypoints") or {})
    b2 = _bbox_xyxy_from_kp_dict(d_last.get("fencer2_keypoints") or {})
    if not b1 or not b2:
        return None

    c1x = (b1[0] + b1[2]) / 2.0
    c2x = (b2[0] + b2[2]) / 2.0
    f1_left_of_f2 = c1x < c2x

    if scoring_fencer == "fencer1":
        attacker_cx, defender_cx = c1x, c2x
    else:
        attacker_cx, defender_cx = c2x, c1x

    engage_mid_x = (attacker_cx + defender_cx) / 2.0

    fw = d_last.get("frame_width")
    fh = d_last.get("frame_height")
    if isinstance(fw, (int, float)) and isinstance(fh, (int, float)) and fw > 0 and fh > 0:
        frame_w, frame_h = int(fw), int(fh)
    else:
        fb_w, fb_h = _infer_frame_size_fallback_last_frame(d_last)
        if fb_w is None:
            return None
        frame_w, frame_h = fb_w, fb_h

    touch_region = _horizontal_third(engage_mid_x, frame_w)
    pressing = _pressing_touch_for_scorer_thirds(scoring_fencer, f1_left_of_f2, engage_mid_x, frame_w)

    # Back-compat: map thirds to old left/right for callers expecting two buckets only
    touch_screen_side = touch_region if touch_region != "center" else "center"

    return {
        "scoring_fencer": scoring_fencer,
        "touch_screen_region": touch_region,
        "touch_screen_side": touch_screen_side,
        "engage_mid_x": round(float(engage_mid_x), 1),
        "frame_width": int(frame_w),
        "frame_height": int(frame_h),
        "attacker_bbox_center_x": round(float(attacker_cx), 1),
        "defender_bbox_center_x": round(float(defender_cx), 1),
        "fencer1_bbox_center_x": round(float(c1x), 1),
        "fencer2_bbox_center_x": round(float(c2x), 1),
        "fencer1_left_of_fencer2": bool(f1_left_of_f2),
        "pressing_touch": bool(pressing),
        "light_frame_seq": int(light_seq) if light_seq is not None else _frame_num(light_path),
    }


def attach_pre_touch_aggressors(arm_attempts, results, three_d_batch, log_fn=None):
    """
    Score who advanced toward the opponent before each light using the
    bout-wide forward_back series. Mutates arm_attempts and three_d_batch.
    """
    _log = log_fn or (lambda _m: None)
    if not arm_attempts or not isinstance(arm_attempts, dict):
        return arm_attempts
    fb = arm_attempts.get("forward_back")
    if not fb or not isinstance(fb, dict):
        return arm_attempts

    from forward_back import annotate_pre_touch_aggressors, touch_frame_from_name

    touches = []
    for r in results or []:
        if not isinstance(r, dict):
            continue
        name = r.get("touch")
        if not name:
            continue
        frame = r.get("frame")
        if frame is None:
            frame = touch_frame_from_name(str(name))
        fencer = None
        low = str(name).lower()
        if low.startswith("fencer1"):
            fencer = "fencer1"
        elif low.startswith("fencer2"):
            fencer = "fencer2"
        touches.append({"touch": name, "frame": frame, "fencer": fencer})

    summary = annotate_pre_touch_aggressors(fb, touches)
    if not summary:
        _log("[FOOTWORK] pre-touch aggressor: no scorable touches")
        return arm_attempts

    arm_attempts["pre_touch_aggressor"] = summary
    by_touch = summary.get("by_touch") or {}

    # Attach onto each 3D payload's touch_spatial for the per-touch UI row.
    if isinstance(three_d_batch, dict):
        for touch_name, scored in by_touch.items():
            payload = three_d_batch.get(touch_name)
            if not isinstance(payload, dict):
                continue
            ts = payload.get("touch_spatial")
            if not isinstance(ts, dict):
                ts = {}
                payload["touch_spatial"] = ts
            ts["pre_touch_aggressor"] = scored.get("aggressor")
            ts["pre_touch_footwork"] = {
                "window_sec": scored.get("window_sec"),
                "sample_count": scored.get("sample_count"),
                "fencer1": scored.get("fencer1"),
                "fencer2": scored.get("fencer2"),
            }

    _log(
        f"[FOOTWORK] pre-touch spatial aggressor: scored={summary.get('touches_scored')} "
        f"F1={summary.get('fencer1_pre_touch_aggression')} "
        f"F2={summary.get('fencer2_pre_touch_aggression')} "
        f"even={summary.get('even')} "
        f"main={summary.get('main_footwork_aggressor')}"
    )
    return arm_attempts


def summarize_touch_spatial_from_batch(batch_3d):
    """Aggregate touch_spatial across all touches in a 3D batch dict."""
    if not batch_3d:
        return None
    left_t = center_t = right_t = 0
    f1_press = f2_press = 0
    f1_score = f2_score = 0
    used = 0
    for _name, data in batch_3d.items():
        if not isinstance(data, dict):
            continue
        ts = data.get("touch_spatial")
        if not isinstance(ts, dict):
            continue
        region = ts.get("touch_screen_region")
        if not region:
            side = ts.get("touch_screen_side")
            if side == "center":
                region = "center"
            elif side == "left":
                region = "left"
            elif side == "right":
                region = "right"
            else:
                region = "center"
        if region == "left":
            left_t += 1
        elif region == "center":
            center_t += 1
        elif region == "right":
            right_t += 1
        if ts.get("pressing_touch"):
            if ts.get("scoring_fencer") == "fencer1":
                f1_press += 1
            elif ts.get("scoring_fencer") == "fencer2":
                f2_press += 1
        if ts.get("scoring_fencer") == "fencer1":
            f1_score += 1
        elif ts.get("scoring_fencer") == "fencer2":
            f2_score += 1
        used += 1

    if used == 0:
        return None

    if f1_press > f2_press:
        main = "fencer1"
    elif f2_press > f1_press:
        main = "fencer2"
    elif f1_score > f2_score:
        main = "fencer1"
    elif f2_score > f1_score:
        main = "fencer2"
    else:
        main = "even"

    return {
        "touches_left_third": left_t,
        "touches_center_third": center_t,
        "touches_right_third": right_t,
        "fencer1_pressing_touches": f1_press,
        "fencer2_pressing_touches": f2_press,
        "fencer1_scoring_touches": f1_score,
        "fencer2_scoring_touches": f2_score,
        "main_aggressor": main,
        "touches_with_spatial": used,
        "rationale": (
            "Engagement x is the average of scorer and opponent pose box centers; "
            "horizontal thirds use the real video size. Main aggressor uses pressing "
            "touches (in opponent's third), then total scores."
        ),
    }


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

    from mmpose_paths import motionbert_checkpoint_path

    config = "mmpose/configs/body_3d_keypoint/motionbert/h36m/motionbert_dstformer-ft-243frm_8xb32-120e_h36m.py"
    checkpoint = motionbert_checkpoint_path()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = init_model(config, checkpoint, device=device).to(torch.float32)
    model.backbone.register_forward_pre_hook(lambda m, i: (i[0].float(),))

    for seq_folder in sequences:
        touch_name = os.path.basename(seq_folder)
        job_id = os.path.basename(os.path.dirname(seq_folder))

        log(f"seq: {touch_name} - {job_id}")
        if enforce_fencer1_left_in_touch_folder(seq_folder):
            log(f"TRACK identity fix before 3D: swapped fencer1/fencer2 in {touch_name}")

        try:
            files = sorted(glob.glob(os.path.join(seq_folder, "frame_*_keypoints.json")),
                          key=lambda p: int(os.path.basename(p).split("_")[1]))
            if not files:
                continue

            def load_seq(fencer_key):
                seq = []
                # Lift full extract window (pre-light through light).
                # Location still reads frames 1..light; attack type uses the last 30.
                for f in files:
                    with open(f) as fp:
                        d = json.load(fp)
                    kp_dict = d.get(fencer_key)
                    if not kp_dict or not isinstance(kp_dict, dict):
                        raise ValueError(
                            f"Missing {fencer_key} in {os.path.basename(f)}"
                        )
                    kpts = [kp_dict[n] for n in COCO_ORDER]
                    arr = np.array(kpts, dtype=np.float32)
                    # Use only x,y for 3D lifting (confidence in JSON for future use)
                    if arr.shape[-1] >= 3:
                        arr = arr[..., :2]
                    seq.append(arr)
                return np.stack(seq)

            f1_2d = load_seq("fencer1_keypoints")
            f2_2d = load_seq("fencer2_keypoints")

            def _xy_span(seq_2d):
                # seq: (T, 17, 2) — tiny span means collapsed/identical joints → 3D is a dot.
                xs = seq_2d[..., 0]
                ys = seq_2d[..., 1]
                return float(xs.max() - xs.min()), float(ys.max() - ys.min())

            f1_span = _xy_span(f1_2d)
            f2_span = _xy_span(f2_2d)
            log(
                f"TRACK 3D input {touch_name}: frames={len(f1_2d)} "
                f"f1_xy_span={f1_span} f2_xy_span={f2_span}"
            )
            if min(f1_span) < 5.0 or min(f2_span) < 5.0:
                log(
                    f"TRACK 3D WARNING {touch_name}: 2D joints nearly collapsed "
                    f"(f1_span={f1_span}, f2_span={f2_span}) — 3D viewer may show a single point. "
                    "Usually wrong auto-select (foreground person) or failed gate tracking."
                )

            touch_spatial = compute_touch_spatial_meta(seq_folder, files)

            # Use the processing resolution stored with the 2D keypoints (not a
            # hardcoded 1920x1080 — downscaled clips break lifting otherwise).
            with open(files[0]) as fp0:
                meta0 = json.load(fp0)
            img_w = int(meta0.get("frame_width") or 1920)
            img_h = int(meta0.get("frame_height") or 1080)
            log(f"TRACK 3D image_size=({img_w},{img_h}) from 2D meta")

            def build_samples(seq):
                """One list entry per video frame, each with a single person.

                inference_pose_lifter_model expects
                List[frame][person] — NOT one frame containing all timesteps as
                separate people (that yields broken temporal context).
                """
                kpts = convert_keypoint_definition(seq, "coco", "h36m")
                frames = []
                for t in range(len(kpts)):
                    s = PoseDataSample()
                    s.gt_instances = InstanceData()
                    s.pred_instances = InstanceData()
                    s.pred_instances.keypoints = kpts[t][None].astype(np.float32)
                    xs, ys = kpts[t][:, 0], kpts[t][:, 1]
                    s.pred_instances.bboxes = np.array(
                        [[xs.min(), ys.min(), xs.max(), ys.max()]],
                        dtype=np.float32,
                    )
                    s.track_id = 0
                    frames.append([s])
                return frames

            def _to_Tk3(k):
                """Normalize lifter output to (T, K, 3) — no axis remap.

                Viewer applies the historical -Y/-Z display transform; handedness
                also depends on the lifter's native X sign.
                """
                if isinstance(k, torch.Tensor):
                    k = k.detach().cpu().numpy()
                k = np.asarray(k, dtype=np.float32)
                while k.ndim > 3:
                    k = np.squeeze(k, axis=0)
                if k.ndim == 2:
                    k = k[None, ...]
                if k.shape[-1] < 3:
                    pad = np.zeros(k.shape[:-1] + (3 - k.shape[-1],), dtype=np.float32)
                    k = np.concatenate([k, pad], axis=-1)
                return k

            def extract_3d(out, n_frames):
                """Return (T, 17, 3) from lifter outputs."""
                if not out:
                    raise ValueError("pose lifter returned no samples")
                # Prefer a single sample that already contains the full sequence.
                if len(out) == 1:
                    k = _to_Tk3(out[0].pred_instances.keypoints)
                    if k.shape[0] == n_frames:
                        return k
                    if k.shape[0] == 1 and n_frames == 1:
                        return k
                    if k.shape[0] == 1 and n_frames > 1:
                        raise ValueError(
                            f"lifter returned 1 frame, expected {n_frames}"
                        )
                arr = []
                for s in out:
                    k = _to_Tk3(s.pred_instances.keypoints)
                    arr.append(k[0] if k.shape[0] == 1 else k[k.shape[0] // 2])
                stacked = np.stack(arr, axis=0)
                if stacked.shape[0] != n_frames:
                    raise ValueError(
                        f"lifter returned {stacked.shape[0]} frame(s), expected {n_frames}"
                    )
                return stacked

            def lift_sequence(seq_2d):
                """Lift with a proper frame list; fall back to per-frame if needed."""
                n_frames = int(seq_2d.shape[0])
                # Per-frame lift is reliable for short clips (touch windows are ~30f).
                frames_out = []
                for t in range(n_frames):
                    out_t = inference_pose_lifter_model(
                        model,
                        build_samples(seq_2d[t : t + 1]),
                        with_track_id=False,
                        image_size=(img_w, img_h),
                        norm_pose_2d=True,
                    )
                    k = extract_3d(out_t, 1)
                    frames_out.append(k[0])
                return np.stack(frames_out, axis=0)

            # COCO-17 L/R swap for horizontal flip before lift.
            _coco_flip = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
            # H36M-17 L/R swap (MotionBERT / convert_keypoint_definition order).
            _h36m_flip = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]

            def mirror_2d_coco(seq_2d):
                """Image-flip 2D + swap L/R so a left-facing fencer looks right-facing."""
                out = np.asarray(seq_2d, dtype=np.float32).copy()
                out[..., 0] = float(img_w - 1) - out[..., 0]
                return out[:, _coco_flip, :]

            def unmirror_3d_h36m(seq_3d):
                """Undo mirror_2d_coco in MotionBERT space (flip X, swap L/R)."""
                out = np.asarray(seq_3d, dtype=np.float32).copy()
                out[..., 0] *= -1.0
                return out[:, _h36m_flip, :]

            f1_3d_raw = lift_sequence(f1_2d)
            # Fencer 2 faces left (toward center). MotionBERT is biased to
            # right-facing poses like F1; lift a mirrored F2 then unmirror.
            f2_3d_raw = unmirror_3d_h36m(lift_sequence(mirror_2d_coco(f2_2d)))
            log(f"TRACK 3D: F2 lifted via mirror-before-lift (left-facing TTA)")

            def _xyz_span(seq_3d):
                return (
                    float(seq_3d[..., 0].max() - seq_3d[..., 0].min()),
                    float(seq_3d[..., 1].max() - seq_3d[..., 1].min()),
                    float(seq_3d[..., 2].max() - seq_3d[..., 2].min()),
                )

            log(
                f"TRACK 3D output {touch_name}: shape_f1={tuple(f1_3d_raw.shape)} "
                f"shape_f2={tuple(f2_3d_raw.shape)} "
                f"f1_xyz_span={_xyz_span(f1_3d_raw)} f2_xyz_span={_xyz_span(f2_3d_raw)}"
            )
            if min(_xyz_span(f1_3d_raw)) < 1e-3 or min(_xyz_span(f2_3d_raw)) < 1e-3:
                log(
                    f"TRACK 3D WARNING {touch_name}: lifted skeleton is collapsed "
                    f"in at least one axis"
                )

            out_dir = os.path.join(output_3d, job_id)
            os.makedirs(out_dir, exist_ok=True)

            # Save 3D data with per-frame analysis
            out_payload = {
                "num_frames": len(f1_3d_raw),
                "fps": float(pipeline_state.get("fps") or 0) or None,
                "fencer1_3d": f1_3d_raw.tolist(),
                "fencer2_3d": f2_3d_raw.tolist(),
                "fencer1_analysis": analyze_fencer_sequence(f1_3d_raw),
                "fencer2_analysis": analyze_fencer_sequence(f2_3d_raw),
            }
            if touch_spatial:
                out_payload["touch_spatial"] = touch_spatial
            with open(os.path.join(out_dir, f"{touch_name}_3d.json"), "w") as f:
                # allow_nan=False: browsers cannot JSON.parse NaN (breaks 3D/angles UI).
                json.dump(out_payload, f, allow_nan=False)

            log(f"Lifted: {touch_name}")

        except Exception as e:
            log(f"Failed {touch_name}: {e}")

    # After all sequences are processed, detect handedness for this job
    if sequences:
        detect_and_update_handedness(app, output_3d, job_id)

def run_prediction(app, path_2d, path_3d, job_id):
    """Touch classifier v3.46 + attack-type classifier (lunge/fleche/other)."""
    from attack_classifier import AttackClassifier, default_model_path as default_attack_model_path
    from touch_v346_classifier import TouchV346Classifier, default_model_path

    model_path = _get_cfg(app, "MODEL_PATH", default_model_path())
    attack_model_path = _get_cfg(app, "ATTACK_MODEL_PATH", default_attack_model_path())
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    video_folder = os.path.join(path_3d, job_id)
    if not os.path.isdir(video_folder):
        return results

    files_3d = sorted([f for f in os.listdir(video_folder) if f.endswith("_3d.json")])
    if not files_3d:
        return results

    if not os.path.isfile(model_path):
        log(f"Prediction: touch model file not found: {model_path}")
        return results

    clf = TouchV346Classifier(model_path, device=device)
    log(f"Touch classifier v3.46 (masked_late + expanded geom, COCO-17 ViTPose): {model_path}")

    attack_clf = None
    if os.path.isfile(attack_model_path):
        attack_clf = AttackClassifier(attack_model_path, device=device)
        log(f"Attack classifier (3D geometry): {attack_model_path}")
    else:
        log(f"Attack classifier skipped (model not found): {attack_model_path}")

    for f in files_3d:
        p3d = os.path.join(video_folder, f)
        touch_name = f.replace("_3d.json", "")
        p2d = os.path.join(path_2d, job_id, touch_name)

        if not os.path.isdir(p2d):
            continue

        try:
            pred = clf.predict_path(p2d)
            entry = {
                "video": job_id,
                "touch": touch_name,
                "prediction": pred.label,
                "confidence": list(pred.probabilities),
            }
            if attack_clf is not None:
                try:
                    apred = attack_clf.predict(path_3d=p3d, path_2d=p2d)
                    entry["attack_prediction"] = apred.label
                    entry["attack_confidence"] = list(apred.probabilities)
                    log(
                        f"Attack: {touch_name} -> {apred.label} ({apred.confidence:.1%})"
                    )
                except Exception as e:
                    log(f"Attack prediction failed for {touch_name}: {e}")
            results.append(entry)
            log(f"Prediction: {touch_name} -> {pred.label} ({pred.confidence:.1%})")
        except Exception as e:
            log(f"Prediction failed for {touch_name}: {e}")

    from touch_ref_util import assign_touch_refs
    assign_touch_refs(results, job_id)
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
    app.config.setdefault("UPLOAD_DIR", UPLOAD_DIR)
    app.config.setdefault("OUTPUT_2D", OUTPUT_2D)
    app.config.setdefault("OUTPUT_3D", OUTPUT_3D)
    app.config.setdefault("MODEL_PATH", "./best_touch_v346_coco17_bs10_multivid_val.pth")
    app.config.setdefault("ATTACK_MODEL_PATH", "./best_attack_3d_proximity_winrobust.pth")
    ensure_workspace_dirs()

    app.register_blueprint(demo_bp)

