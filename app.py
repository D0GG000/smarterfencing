""" Fencing Analysis Pipeline - Flask Backend
Automatic pipeline: Upload video -> Extract 2D -> Lift to 3D -> Predict
Selections come from browser UI (no OpenCV window needed)
"""

import os
import json
import threading
import queue
import glob
import shutil
import numpy as np
import torch
import cv2
import logging, sys

from werkzeug.middleware.proxy_fix import ProxyFix

import os, uuid
from pathlib import Path
from datetime import datetime, timedelta

from models import db, Comment, Post
from blog import blog_bp, register_blog_cli

import boto3
from botocore.config import Config

from demo import (
    register_demo, r2_client, pipeline_runner,
    _proc_scale, _downscale_frame, ensure_processable_video,
    read_video_frame_bgr,
)
from fencing_inference import (
    build_detection_debug_payload,
    ensure_pose_stack,
    infer_pose,
    suggest_auto_fencer_pair,
)

from job_queue_models import SiteUser, UserJob, TrackedFencer, get_queue_stats
from job_queue_worker import register_queue
from queue_routes import register_queue_routes
from email_routes import register_email
from auth_routes import register_auth
from fencer_routes import register_fencer_routes
from touch_share_routes import register_touch_share_routes
from community_routes import register_community_routes
from db_schema import ensure_extended_schema
from workspace_paths import (
    UPLOAD_DIR,
    OUTPUT_2D,
    OUTPUT_3D,
    WORKSPACE_TMP,
    default_database_url,
    ensure_workspace_dirs,
    tmp_path,
)
from version import __version__

R2_ACCOUNT_ID = os.environ["R2_ACCOUNT_ID"]
R2_BUCKET = os.environ.get("R2_BUCKET", "smarterfencing-videos")

# Set cache directory to current project directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
torch.hub.set_dir(PROJECT_DIR)
os.environ['TORCH_HOME'] = PROJECT_DIR
os.environ['XDG_CACHE_HOME'] = PROJECT_DIR

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s %(levelname)s %(message)s",
)

def log(msg):
    log_queue.put(msg)
    logging.info(msg)

from flask import Flask, request, jsonify, render_template, send_from_directory, make_response, g, session
from flask_cors import CORS
from werkzeug.utils import secure_filename

###>>>>>>>>>>

from mmpose.utils import register_all_modules as register_mmpose
from mmdet.utils import register_all_modules as register_mmdet
from mmengine.registry import init_default_scope

# ---- registry init (top of app.py) ----
from mmpose.utils import register_all_modules as register_mmpose
from mmdet.utils import register_all_modules as register_mmdet
from mmengine.registry import init_default_scope

register_mmpose(init_default_scope=False)
register_mmdet(init_default_scope=False)

# Keep default scope as mmpose for the overall app
init_default_scope('mmpose')

###>>>>>>>>>>

# -------------------------
# Cookie/User Tracking Helpers
# -------------------------
COOKIE_NAME = 'sf_user_id'
COOKIE_MAX_AGE = 60 * 60 * 24 * 365  # 1 year

def ensure_user_exists(user_id):
    """Ensure user exists in database."""
    user = SiteUser.query.get(user_id)
    if not user:
        user = SiteUser(id=user_id)
        db.session.add(user)
        db.session.commit()
    else:
        # Update last_seen
        user.last_seen = datetime.utcnow()
        db.session.commit()
    return user


def create_app():
    #app = Flask(__name__)
    app = Flask(__name__, static_folder="static", static_url_path="/static")

    db_url = default_database_url()
    app.config["SQLALCHEMY_DATABASE_URI"] = db_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "connect_args": {"timeout": int(os.environ.get("SQLITE_BUSY_TIMEOUT_MS", "5000")) / 1000},
    }
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-only-change-in-production")
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=365)
    app.config["SESSION_COOKIE_SECURE"] = os.environ.get("SESSION_COOKIE_SECURE", "0") == "1"
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["PREFERRED_URL_SCHEME"] = os.environ.get("PREFERRED_URL_SCHEME", "https")

    # Ensure persistent DB directory exists
    if db_url.startswith("sqlite:////"):
        Path(db_url.replace("sqlite:////", "/")).parent.mkdir(
            parents=True, exist_ok=True
        )

    # Persistent data on /workspace (network volume)
    ensure_workspace_dirs()
    app.config["UPLOAD_DIR"] = UPLOAD_DIR
    app.config["OUTPUT_2D"] = OUTPUT_2D
    app.config["OUTPUT_3D"] = OUTPUT_3D
    app.config["WORKSPACE_TMP"] = WORKSPACE_TMP

    db.init_app(app)

    with app.app_context():
        from db_concurrency import configure_sqlite

        configure_sqlite(db.engine)
        db.create_all()
        ensure_extended_schema(db.engine)

    # Register blog
    app.register_blueprint(blog_bp)
    register_blog_cli(app)

   # register demo pipeline endpoints
    register_demo(app)

    # Register email routes
    register_email(app)

    # Make R2_ACCOUNT_ID available in templates
    app.config['R2_PUBLIC_URL'] = os.environ.get('R2_PUBLIC_URL', '')

    register_queue_routes(app)
    register_queue(app)  # Starts background worker

    from admin_tasks import start_admin_task_worker

    start_admin_task_worker(app)
    register_auth(app)
    register_fencer_routes(app)
    register_touch_share_routes(app)
    register_community_routes(app)

    # -------------------------
    # Session user + legacy anonymous cookie (for job migration on first login)
    # -------------------------
    @app.before_request
    def before_request_user_tracking():
        """Logged-in user id for jobs/APIs; optional sf_user_id cookie when logged out."""
        # Static assets must stay cookieless — set_cookie + DB writes break
        # Flask/gunicorn send_file for bots (Google favicon crawler, etc.).
        if request.endpoint == "static":
            return
        g.user_id = session.get("user_id")
        if not g.user_id and not request.cookies.get(COOKIE_NAME):
            g.new_user_id = str(uuid.uuid4())

    @app.after_request
    def after_request_set_cookie(response):
        """Set legacy cookie for anonymous visitors (OAuth/job migration)."""
        if request.endpoint == "static":
            return response
        if hasattr(g, "new_user_id"):
            response.set_cookie(
                COOKIE_NAME,
                g.new_user_id,
                max_age=COOKIE_MAX_AGE,
                secure=app.config["SESSION_COOKIE_SECURE"],
                httponly=True,
                samesite="Lax",
            )
            ensure_user_exists(g.new_user_id)
        return response

    return app

app = create_app()

app.wsgi_app = ProxyFix(
    app.wsgi_app,
    x_for=1,
    x_proto=1,
    x_host=1,
    x_port=1,
)

CORS(app, resources={
    r"/upload*": {
        "origins": [
            "https://smarterfencing.ai",
            "https://www.smarterfencing.ai"
        ]
    },
    r"/api/*": {
        "origins": "*"   # ok ONLY if these endpoints are truly public
    }
})

# Pipeline directories (persistent; see workspace_paths.py)
ensure_workspace_dirs()

# Global state
pipeline_state = {
    'current_step': 'idle',
    'error': None,
    'results': [],
    'fps': 30
}
log_queue = queue.Queue()
current_selections = None

@app.route('/')
def index():
    log(f"Index requested")
    return render_template('index.html')


@app.route('/robots.txt')
def robots_txt():
    body = "\n".join([
        "User-agent: *",
        "Allow: /",
        "Allow: /blog",
        "Allow: /archetypes",
        "Disallow: /demo",
        "Disallow: /clipper",
        "Disallow: /result",
        "Disallow: /past-results",
        "Disallow: /inbox",
        "Disallow: /profile",
        "Disallow: /admin",
        "Disallow: /reset-password",
        "Disallow: /api/",
        "Disallow: /community",
        "Disallow: /fencers",
        "",
        "Sitemap: https://smarterfencing.ai/sitemap.xml",
        "",
    ])
    resp = make_response(body)
    resp.headers["Content-Type"] = "text/plain; charset=utf-8"
    resp.headers["Cache-Control"] = "public, max-age=3600"
    return resp


@app.route('/sitemap.xml')
def sitemap_xml():
    site = "https://smarterfencing.ai"
    urls = [
        (f"{site}/", "1.0", "weekly"),
        (f"{site}/blog", "0.8", "weekly"),
        (f"{site}/archetypes", "0.7", "monthly"),
    ]
    try:
        posts = (
            Post.query
            .filter_by(published=True)
            .order_by(Post.published_at.desc())
            .all()
        )
        for post in posts:
            lastmod = ""
            if post.published_at:
                lastmod = post.published_at.date().isoformat()
            slug = (post.slug or "").strip()
            if not slug:
                continue
            urls.append((f"{site}/blog/{slug}", "0.6", "monthly", lastmod))
    except Exception:
        logging.exception("sitemap: failed to load blog posts")

    def xml_escape(value: str) -> str:
        return (
            str(value)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]
    for entry in urls:
        loc, priority, changefreq = entry[0], entry[1], entry[2]
        lastmod = entry[3] if len(entry) > 3 else ""
        lines.append("  <url>")
        lines.append(f"    <loc>{xml_escape(loc)}</loc>")
        if lastmod:
            lines.append(f"    <lastmod>{xml_escape(lastmod)}</lastmod>")
        lines.append(f"    <changefreq>{xml_escape(changefreq)}</changefreq>")
        lines.append(f"    <priority>{xml_escape(priority)}</priority>")
        lines.append("  </url>")
    lines.append("</urlset>")
    lines.append("")

    resp = make_response("\n".join(lines))
    resp.headers["Content-Type"] = "application/xml; charset=utf-8"
    resp.headers["Cache-Control"] = "public, max-age=3600"
    return resp


@app.route('/ping', methods=['GET'])
def ping():
    log(f"Ping requested")
    resp = make_response("OK", 200)
    resp.headers["X-App-Version"] = __version__
    return resp


@app.get("/api/version")
def app_version():
    """Public build version for deploy verification."""
    return jsonify({"app": "smarterfencing", "version": __version__})

@app.post("/api/uploads/init")
def uploads_init():
    log("Video upload requested")
    uid = session.get("user_id")
    if not uid:
        return jsonify({"error": "Authentication required"}), 401

    # Do not wipe OUTPUT_2D/OUTPUT_3D here: other jobs may still be using them.
    # Per-job dirs are cleaned in job_queue_worker after each job completes.

    body = request.get_json(silent=True) or {}
    filename = body.get("filename", "video.mp4")
    content_type = body.get("content_type") or "video/mp4"

    job_id = uuid.uuid4().hex[:12]

    ext = ""
    if "." in filename:
        ext = "." + filename.rsplit(".", 1)[1].lower()
    if not ext:
        ext = ".mp4"

    object_key = f"uploads/{job_id}{ext}"

    s3, bucket = r2_client(app)
    upload_url = s3.generate_presigned_url(
        ClientMethod="put_object",
        Params={"Bucket": bucket, "Key": object_key, "ContentType": content_type},
        ExpiresIn=15 * 60,
    )

    # -------------------------
    # Associate job with user
    # -------------------------
    user_id = uid
    ensure_user_exists(user_id)

    user_job = UserJob(
        user_id=user_id,
        job_id=job_id,
        r2_object_key=object_key,
        filename=filename
    )
    db.session.add(user_job)
    db.session.commit()
    log(f"Job {job_id} associated with user {user_id[:8]}...")

    return jsonify({"job_id": job_id, "object_key": object_key, "upload_url": upload_url})

@app.get("/api/detect-frame")
def detect_frame():

    log("detect-frame requested")

    uid = session.get("user_id")
    if not uid:
        return jsonify({"success": False, "error": "Authentication required"}), 401

    object_key = request.args.get("video")
    job_id = request.args.get("job_id")
    # Clipper only needs the scoreboard lights, so skip fencer pose detection.
    lights_only = request.args.get("lights_only") in ("1", "true", "True", "yes")
    frame_index_arg = request.args.get("frame_index")
    try:
        requested_frame = int(frame_index_arg) if frame_index_arg is not None else 0
    except (TypeError, ValueError):
        requested_frame = 0
    if not object_key:
        return jsonify({"success": False, "error": "No video specified"}), 400
    if not job_id:
        return jsonify({"success": False, "error": "No job id specified"}), 400

    job = UserJob.query.filter_by(job_id=job_id, user_id=uid).first()
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404

    upload_dir = UPLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)

    local_video_path = tmp_path(f"{job_id}.mp4")

    try:
        s3, bucket = r2_client(app)
        if not (os.path.isfile(local_video_path) and os.path.getsize(local_video_path) > 1024):
            s3.download_file(bucket, object_key, local_video_path)
            log("video downloaded from R2")
        else:
            log("using cached local video for detect-frame")

        # 4K/HEVC clips can open but fail to decode with OpenCV; transcode if so.
        proc_video_path = ensure_processable_video(local_video_path)

        if frame_index_arg is not None:
            frame, total_frames, fps = read_video_frame_bgr(proc_video_path, requested_frame)
            frame_index = max(0, min(total_frames - 1, requested_frame))
        else:
            # Some encodings/keyframe layouts fail on the very first read; try a few
            # frames before giving up so we still get a usable preview frame.
            frame = None
            total_frames, fps = 1, 30.0
            frame_index = 0
            for i in range(15):
                try:
                    frame, total_frames, fps = read_video_frame_bgr(proc_video_path, i)
                    frame_index = i
                    break
                except Exception:
                    continue
        if frame is None:
            return jsonify({"success": False, "error": "Could not read video frame"}), 400

        # Cap preview/selection resolution so 4K clips don't blow up pose inference.
        # Selections are made in this space and the pipeline downscales to match.
        frame = _downscale_frame(frame, _proc_scale(frame.shape[1], frame.shape[0]))
        h, w = frame.shape[:2]

        boxes = []
        fencer_debug = []
        auto_fencers = None
        if not lights_only:
            ensure_pose_stack(log)
            # First preview frame: RTMDet + ViTPose-H (COCO-17). full_body flags drive
            # auto-select; all boxes are still returned for manual override.
            log(
                f"AUTOSELECT: running ViTPose-H on preview frame {frame_index} "
                f"({w}x{h}) for fencer box selection"
            )
            structured = infer_pose(
                frame,
                top_k_persons=32,
                order_fencers_lr=False,
            )
            fencer_debug = build_detection_debug_payload(structured, h, w)
            boxes = [list(map(float, b)) for b in structured.get("bboxes", [])]
            full_body_n = sum(1 for e in fencer_debug if e.get("full_body"))
            log(
                f"AUTOSELECT: {len(boxes)} ViTPose detections "
                f"({full_body_n} pass full_body for auto-select)"
            )
            for entry in fencer_debug:
                status = "FULL" if entry.get("full_body") else "partial"
                log(
                    f"AUTOSELECT {status} idx={entry.get('index')} "
                    f"h_frac={entry.get('bbox_h_frac')} "
                    f"aspect={entry.get('aspect_wh')} "
                    f"clips={entry.get('clips')} "
                    f"kp={entry.get('kp')} "
                    f"leg_confs={entry.get('leg_confs')} "
                    f"reasons={entry.get('reasons')} "
                    f"bbox={entry.get('bbox_xyxy')}"
                )
            auto_fencers = suggest_auto_fencer_pair(structured, h, w)
            log(
                f"AUTOSELECT pair success={auto_fencers.get('success')} "
                f"reason={auto_fencers.get('reason')} "
                f"conf={auto_fencers.get('confidence')} "
                f"f1_idx={auto_fencers.get('fencer1_index')} "
                f"f2_idx={auto_fencers.get('fencer2_index')} "
                f"candidates={auto_fencers.get('candidates')}"
            )

        frame_filename = f"{job_id}_frame_{frame_index}.jpg"
        frame_path = os.path.join(upload_dir, frame_filename)
        cv2.imwrite(frame_path, frame)

        return jsonify(
            {
                "success": True,
                "boxes": boxes,
                "width": w,
                "height": h,
                "frame_url": f"/api/frame/{frame_filename}",
                "video": object_key,
                "fencer_debug": fencer_debug,
                "auto_fencers": auto_fencers,
                "total_frames": total_frames,
                "fps": fps,
                "frame_index": frame_index,
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.get("/api/preview-frame")
def preview_frame():
    """Return a specific downscaled preview frame for template scrubbing."""
    uid = session.get("user_id")
    if not uid:
        return jsonify({"success": False, "error": "Authentication required"}), 401

    object_key = request.args.get("video")
    job_id = request.args.get("job_id")
    if not object_key:
        return jsonify({"success": False, "error": "No video specified"}), 400
    if not job_id:
        return jsonify({"success": False, "error": "No job id specified"}), 400

    try:
        frame_index = int(request.args.get("frame_index", 0))
    except (TypeError, ValueError):
        frame_index = 0

    job = UserJob.query.filter_by(job_id=job_id, user_id=uid).first()
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404

    upload_dir = UPLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)
    local_video_path = tmp_path(f"{job_id}.mp4")

    try:
        s3, bucket = r2_client(app)
        # Keep the local copy so scrubbing does not re-download the whole video
        # on every slider move (previous code deleted it after each request).
        if not (os.path.isfile(local_video_path) and os.path.getsize(local_video_path) > 1024):
            s3.download_file(bucket, object_key, local_video_path)
        proc_video_path = ensure_processable_video(local_video_path)

        frame, total_frames, fps = read_video_frame_bgr(proc_video_path, frame_index)
        frame_index = max(0, min(total_frames - 1, frame_index))

        frame = _downscale_frame(frame, _proc_scale(frame.shape[1], frame.shape[0]))
        h, w = frame.shape[:2]
        frame_filename = f"{job_id}_preview_{frame_index}.jpg"
        frame_path = os.path.join(upload_dir, frame_filename)
        cv2.imwrite(frame_path, frame)

        return jsonify(
            {
                "success": True,
                "width": w,
                "height": h,
                "frame_url": f"/api/frame/{frame_filename}",
                "frame_index": frame_index,
                "total_frames": total_frames,
                "fps": fps,
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.get("/api/frame/<filename>")
def serve_frame(filename):
    upload_dir = UPLOAD_DIR
    return send_from_directory(upload_dir, filename)

@app.post("/api/run-pipeline")
def run_pipeline():
    """Submit job to queue instead of running directly."""
    from job_queue_worker import add_to_queue, QueueBusyError

    uid = session.get("user_id")
    if not uid:
        return jsonify({"success": False, "error": "Authentication required"}), 401

    data = request.get_json(silent=True) or {}
    job_id = data.get("job_id")
    object_key = data.get("video")
    selections = data.get("selections")
    job_type = data.get("job_type", "analysis")
    if job_type not in ("analysis", "clipper"):
        job_type = "analysis"

    if not job_id:
        return jsonify({"success": False, "error": "No job_id specified"}), 400
    if not object_key:
        return jsonify({"success": False, "error": "No video specified"}), 400

    from queue_routes import _validate_selections_payload
    sel_error = _validate_selections_payload(selections, require_fencers=(job_type != "clipper"))
    if sel_error:
        return jsonify({"success": False, "error": sel_error}), 400

    job = UserJob.query.filter_by(job_id=job_id, user_id=uid).first()
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404

    job.job_type = job_type
    db.session.commit()

    try:
        queue_info = add_to_queue(job_id, selections)

        return jsonify({
            "success": True,
            "message": "Job added to queue",
            "status": "queued",
            "queue_position": queue_info['position'],
            "estimated_wait_minutes": queue_info['estimated_wait_minutes']
        })

    except QueueBusyError as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "code": "queue_busy",
            "queue_length": e.queue_length,
            "queue_max_length": e.max_length,
        }), 503
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# -------------------------
# User API Endpoints
# -------------------------
@app.route('/api/user', methods=['GET'])
def get_user_info():
    """Get current user info."""
    user_id = g.user_id
    if not user_id:
        return jsonify({'user': None, 'job_count': 0})

    user = SiteUser.query.get(user_id)

    if not user:
        return jsonify({'user': None, 'job_count': 0})

    return jsonify({
        'user': user.to_dict(),
        'job_count': user.jobs.count()
    })

@app.route('/api/user/email', methods=['POST'])
def set_user_email():
    """Set or update user email (optional)."""
    user_id = g.user_id
    if not user_id:
        return jsonify({'success': False, 'error': 'Authentication required'}), 401
    data = request.get_json(silent=True) or {}
    email = data.get('email', '').strip()

    if not email:
        return jsonify({'success': False, 'error': 'No email provided'}), 400

    # Basic email validation
    if '@' not in email or '.' not in email:
        return jsonify({'success': False, 'error': 'Invalid email format'}), 400

    user = ensure_user_exists(user_id)
    user.email = email
    db.session.commit()

    log(f"Email set for user {user_id[:8]}...")
    return jsonify({'success': True, 'email': email})

@app.route('/api/user/jobs', methods=['GET'])
def get_user_jobs():
    """Get all jobs for current user."""
    user_id = g.user_id
    if not user_id:
        return jsonify({'error': 'Authentication required'}), 401
    jobs = UserJob.query.filter_by(user_id=user_id).order_by(UserJob.created_at.desc()).all()

    return jsonify({
        'jobs': [job.to_dict() for job in jobs]
    })

@app.get("/api/job-status/<job_id>")
def job_status(job_id):
    from job_queue_worker import get_job_status
    uid = session.get("user_id")
    if not uid:
        return jsonify({"error": "Authentication required"}), 401
    status = get_job_status(job_id, user_id=uid)
    if not status:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(status)

# -------------------------
# Admin Endpoints (protect in production!)
# -------------------------
from functools import wraps

ADMIN_TOKEN = (os.environ.get('ADMIN_TOKEN') or '').strip()

def require_admin(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        expected = (os.environ.get("ADMIN_TOKEN") or "").strip()
        token = request.args.get("token") or ""
        if not expected or token != expected:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


def _admin_list_limit(default=500, maximum=2000):
    try:
        return max(1, min(int(request.args.get("limit", default)), maximum))
    except (TypeError, ValueError):
        return default


@app.route('/api/admin/summary')
@require_admin
def admin_summary():
    """Fast dashboard counts without loading full tables."""
    from sqlalchemy import func

    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    total_users = db.session.query(func.count(SiteUser.id)).scalar() or 0
    users_with_email = (
        db.session.query(func.count(SiteUser.id))
        .filter(SiteUser.email.isnot(None), SiteUser.email != "")
        .scalar()
        or 0
    )
    total_jobs = db.session.query(func.count(UserJob.id)).scalar() or 0
    today_jobs = (
        db.session.query(func.count(UserJob.id))
        .filter(UserJob.created_at >= today_start)
        .scalar()
        or 0
    )
    pending_comments = (
        db.session.query(func.count(Comment.id))
        .filter(Comment.approved.is_(False))
        .scalar()
        or 0
    )

    return jsonify({
        "total_users": total_users,
        "users_with_email": users_with_email,
        "total_jobs": total_jobs,
        "today_jobs": today_jobs,
        "pending_comments": pending_comments,
    })


@app.route('/api/admin/users')
@require_admin
def admin_list_users():
    """List recent users with job counts (bounded, no N+1)."""
    from sqlalchemy import func

    limit = _admin_list_limit()
    total = db.session.query(func.count(SiteUser.id)).scalar() or 0

    rows = (
        db.session.query(
            SiteUser,
            func.count(UserJob.id).label("job_count"),
        )
        .outerjoin(UserJob, UserJob.user_id == SiteUser.id)
        .group_by(SiteUser.id)
        .order_by(SiteUser.last_seen.desc())
        .limit(limit)
        .all()
    )

    return jsonify({
        "total": total,
        "limit": limit,
        "truncated": total > limit,
        "users": [{
            "id": u.id,
            "email": u.email,
            "username": u.username,
            "display_name": u.display_name,
            "created_at": u.created_at.isoformat() if u.created_at else None,
            "last_seen": u.last_seen.isoformat() if u.last_seen else None,
            "job_count": int(job_count or 0),
        } for u, job_count in rows],
    })


@app.get("/api/admin/user-lookup")
@require_admin
def admin_user_lookup():
    """Find user(s) by email, username, or id prefix."""
    from admin_user_util import build_user_detail, find_users_by_query, user_association_counts

    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify({"success": False, "error": "q parameter required"}), 400

    users = find_users_by_query(q)
    if not users:
        return jsonify({"success": False, "error": "No matching users"}), 404

    if len(users) == 1:
        payload = build_user_detail(users[0])
        payload["success"] = True
        payload["matches"] = 1
        return jsonify(payload)

    return jsonify({
        "success": True,
        "matches": len(users),
        "users": [
            {
                **_user_admin_summary(u),
                "counts": user_association_counts(u.id),
            }
            for u in users
        ],
    })


def _user_admin_summary(user):
    return {
        "id": user.id,
        "email": user.email,
        "username": user.username,
        "display_name": user.display_name,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "last_seen": user.last_seen.isoformat() if user.last_seen else None,
    }


@app.get("/api/admin/users/<user_id>/detail")
@require_admin
def admin_user_detail(user_id):
    """Full association report for one user."""
    from admin_user_util import build_user_detail

    user = SiteUser.query.get(user_id)
    if not user:
        return jsonify({"success": False, "error": "User not found"}), 404

    payload = build_user_detail(user)
    payload["success"] = True
    return jsonify(payload)


@app.delete("/api/admin/users/<user_id>")
@require_admin
def admin_delete_user(user_id):
    """Delete a user and all associated DB rows; R2 cleanup runs in background."""
    from admin_tasks import delete_job_r2_objects, enqueue_admin_task
    from admin_user_util import (
        delete_user_and_all_data,
        user_association_counts,
        user_has_active_processing_job,
    )

    user = SiteUser.query.get(user_id)
    if not user:
        return jsonify({"success": False, "error": "User not found"}), 404

    active = user_has_active_processing_job(user_id)
    if active:
        return jsonify({
            "success": False,
            "error": f"Cannot delete user while job {active.job_id} is processing",
        }), 400

    counts = user_association_counts(user_id)
    try:

        def enqueue_r2(job_id, object_key):
            enqueue_admin_task(app, delete_job_r2_objects, job_id, object_key)

        deleted = delete_user_and_all_data(user_id, enqueue_r2_cleanup=enqueue_r2)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        log(f"Admin user purge failed for {user_id[:8]}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

    log(f"Admin purged user {user_id[:8]}...")
    return jsonify({
        "success": True,
        "user_id": user_id,
        "counts_before": counts,
        "deleted": deleted,
        "message": "User deleted. Video storage cleanup continues in background.",
    })

@app.get("/api/admin/jobs")
def admin_jobs():
    """Get recent jobs (bounded list + total count)."""
    from sqlalchemy import func

    expected = (os.environ.get("ADMIN_TOKEN") or "").strip()
    token = request.args.get("token") or ""
    if not expected or token != expected:
        return jsonify({"error": "Unauthorized"}), 401

    limit = _admin_list_limit()
    total = db.session.query(func.count(UserJob.id)).scalar() or 0

    jobs = (
        UserJob.query.order_by(UserJob.created_at.desc())
        .limit(limit)
        .all()
    )

    return jsonify({
        "total": total,
        "limit": limit,
        "truncated": total > limit,
        "jobs": [{
            "job_id": j.job_id,
            "user_id": j.user_id,
            "user_email": j.user.email if j.user else None,
            "username": j.user.username if j.user else None,
            "filename": j.filename,
            "file_size": j.file_size,
            "object_key": j.r2_object_key,
            "status": j.status,
            "created_at": j.created_at.isoformat() if j.created_at else None,
            "completed_at": j.completed_at.isoformat() if hasattr(j, 'completed_at') and j.completed_at else None
        } for j in jobs]
    })


@app.get("/api/admin/queue-live")
def admin_queue_live():
    """Live queue snapshot (worker state + waiting jobs)."""
    expected = (os.environ.get("ADMIN_TOKEN") or "").strip()
    token = request.args.get("token") or ""
    if not expected or token != expected:
        return jsonify({"error": "Unauthorized"}), 401

    from queue_state import get_admin_queue_snapshot

    return jsonify(get_admin_queue_snapshot())


@app.delete("/api/admin/delete-job/<job_id>")
def admin_delete_job(job_id):
    """Delete a job row quickly; R2 cleanup runs in the background."""
    from job_queue_worker import get_current_job_id
    from admin_tasks import delete_job_r2_objects, enqueue_admin_task

    expected = (os.environ.get("ADMIN_TOKEN") or "").strip()
    token = request.args.get("token") or ""
    if not expected or token != expected:
        return jsonify({"error": "Unauthorized"}), 401

    job = UserJob.query.filter_by(job_id=job_id).first()
    if not job:
        return jsonify({"error": "Job not found"}), 404

    if job.status == "processing" and get_current_job_id() == job_id:
        return jsonify({"error": "Cannot delete a job that is currently processing"}), 400

    try:
        object_key = job.r2_object_key
        db.session.delete(job)
        db.session.commit()

        enqueue_admin_task(app, delete_job_r2_objects, job_id, object_key)

        return jsonify({
            "success": True,
            "message": f"Job {job_id} deleted (storage cleanup in background)",
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.post("/api/admin/jobs/<job_id>/assign-user")
@require_admin
def admin_assign_job_user(job_id):
    """Assign a job to a different user (admin handoff)."""
    from admin_user_util import assign_job_to_user, find_users_by_query

    data = request.get_json(silent=True) or {}
    user_id = (data.get("user_id") or "").strip()
    query = (data.get("query") or "").strip()

    if not user_id and not query:
        return jsonify({"success": False, "error": "user_id or query required"}), 400

    if not user_id:
        users = find_users_by_query(query)
        if not users:
            return jsonify({"success": False, "error": "No matching users"}), 404
        if len(users) > 1:
            return jsonify({
                "success": False,
                "error": "Multiple users match; pick one or pass user_id",
                "users": [_user_admin_summary(u) for u in users],
            }), 409
        user_id = users[0].id

    try:
        result = assign_job_to_user(job_id, user_id)
        log(
            f"Admin assigned job {job_id} from {result['previous_user_id'][:8]} "
            f"to {user_id[:8]}"
        )
        return jsonify({"success": True, **result})
    except LookupError as e:
        return jsonify({"success": False, "error": str(e)}), 404
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        db.session.rollback()
        log(f"Admin assign job failed for {job_id}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/admin/video-url/<job_id>')
@require_admin
def get_video_url(job_id):
    """Generate a presigned URL to view a video."""
    job = UserJob.query.filter_by(job_id=job_id).first()
    if not job or not job.r2_object_key:
        return jsonify({'error': 'Job not found'}), 404

    s3, bucket = r2_client(app)
    url = s3.generate_presigned_url(
        ClientMethod='get_object',
        Params={'Bucket': bucket, 'Key': job.r2_object_key},
        ExpiresIn=3600  # 1 hour
    )
    return jsonify({'url': url})

@app.route('/api/admin/cleanup-users', methods=['DELETE'])
@require_admin
def cleanup_empty_users():
    """Queue deletion of users with no jobs and no email (background, batched)."""
    from admin_tasks import cleanup_empty_users_batched, enqueue_admin_task

    enqueue_admin_task(app, cleanup_empty_users_batched)

    return jsonify({
        "success": True,
        "accepted": True,
        "message": "Cleanup started in background; analyzer takes priority",
    }), 202


@app.route('/api/admin/anonymous-user-cleanup-stats')
@require_admin
def admin_anonymous_user_cleanup_stats():
    """Cumulative idle cleanup stats and remaining eligible users."""
    from user_cleanup import get_cleanup_stats

    return jsonify(get_cleanup_stats())


@app.route('/api/admin/anonymous-user-cleanup-stats/reset', methods=['POST'])
@require_admin
def admin_reset_anonymous_user_cleanup_stats():
    """Reset the cumulative idle cleanup counter to zero."""
    from user_cleanup import reset_cleaned_total

    reset_cleaned_total()
    log("Reset anonymous user cleanup counter")
    return jsonify({'success': True, 'cleaned_total': 0})


@app.route('/api/admin/stale-job-cleanup-stats')
@require_admin
def admin_stale_job_cleanup_stats():
    """Stale queued job cleanup stats."""
    from job_cleanup import get_stale_job_cleanup_stats

    return jsonify(get_stale_job_cleanup_stats())


@app.route('/api/admin/stale-job-cleanup-stats/reset', methods=['POST'])
@require_admin
def admin_reset_stale_job_cleanup_stats():
    """Reset the startup abandoned-job cleanup counter to zero."""
    from job_cleanup import reset_cleaned_total

    reset_cleaned_total()
    log("Reset stale job cleanup counter")
    return jsonify({'success': True, 'stale_jobs_cleaned_total': 0})


@app.route('/admin')
def admin_page():
    return render_template('admin.html', app_version=__version__)

@app.get("/api/admin/result-url/<job_id>")
def admin_result_url(job_id):
    """Get result page URL for a completed job (same as email link)."""
    import os
    import json
    from urllib.parse import urlencode

    expected = (os.environ.get("ADMIN_TOKEN") or "").strip()
    token = request.args.get("token") or ""
    if not expected or token != expected:
        return jsonify({"error": "Unauthorized"}), 401

    # Get the job from database
    job = UserJob.query.filter_by(job_id=job_id).first()
    if not job:
        return jsonify({"error": "Job not found"}), 404

    if job.status != 'complete':
        return jsonify({"error": f"Job not complete (status: {job.status})"}), 400

    # Get R2 public URL base
    r2_public_url = os.environ.get("R2_PUBLIC_URL", "")
    if not r2_public_url:
        return jsonify({"error": "R2_PUBLIC_URL not configured"}), 500

    video_url = f"{r2_public_url}/{job.r2_object_key}"
    # Use correct path: results/{job_id}_results.json
    data_url = f"{r2_public_url}/results/{job_id}_results.json"

    # Build result page URL with query parameters
    base_url = os.environ.get("BASE_URL", request.host_url.rstrip('/'))
    params = urlencode({
        'job_id': job_id,
        'video': video_url,
        'data': data_url
    })
    result_url = f"{base_url}/result?{params}"

    return jsonify({
        "url": result_url,
        "video_url": video_url,
        "data_url": data_url
    })

# =============================================================
# List all comments with user info
# =============================================================

@app.route('/api/admin/comments', methods=['GET'])
@require_admin
def admin_list_comments():
    """List recent comments with user and post info (bounded)."""
    limit = _admin_list_limit(default=200, maximum=1000)

    rows = (
        db.session.query(Comment, Post, SiteUser)
        .join(Post, Comment.post_id == Post.id)
        .outerjoin(SiteUser, Comment.user_id == SiteUser.id)
        .order_by(Comment.created_at.desc())
        .limit(limit)
        .all()
    )

    from sqlalchemy import func
    total = db.session.query(func.count(Comment.id)).scalar() or 0

    result = []
    for c, post, user in rows:
        result.append({
            "id": c.id,
            "post_id": c.post_id,
            "post_title": post.title if post else "Unknown Post",
            "user_id": c.user_id,
            "user_email": user.email if user else None,
            "author_name": c.author_name,
            "body": c.body,
            "approved": c.approved,
            "created_at": c.created_at.isoformat() if c.created_at else None,
        })

    return jsonify({
        "total": total,
        "limit": limit,
        "truncated": total > limit,
        "comments": result,
    })

# =============================================================
# Approve a comment
# =============================================================

@app.route('/api/admin/comments/<int:comment_id>/approve', methods=['POST'])
@require_admin
def admin_approve_comment(comment_id):
    """Approve a comment to show on the blog."""
    comment = Comment.query.get(comment_id)
    if not comment:
        return jsonify({'error': 'Comment not found'}), 404

    comment.approved = True
    db.session.commit()

    return jsonify({
        'success': True,
        'message': f'Comment {comment_id} approved'
    })


# =============================================================
# Unapprove a comment (hide from blog)
# =============================================================

@app.route('/api/admin/comments/<int:comment_id>/unapprove', methods=['POST'])
@require_admin
def admin_unapprove_comment(comment_id):
    """Unapprove a comment to hide from the blog."""
    comment = Comment.query.get(comment_id)
    if not comment:
        return jsonify({'error': 'Comment not found'}), 404

    comment.approved = False
    db.session.commit()

    return jsonify({
        'success': True,
        'message': f'Comment {comment_id} unapproved'
    })


# =============================================================
# Delete a comment
# =============================================================

@app.route('/api/admin/comments/<int:comment_id>', methods=['DELETE'])
@require_admin
def admin_delete_comment(comment_id):
    """Delete a comment permanently."""
    comment = Comment.query.get(comment_id)
    if not comment:
        return jsonify({'error': 'Comment not found'}), 404

    db.session.delete(comment)
    db.session.commit()

    return jsonify({
        'success': True,
        'message': f'Comment {comment_id} deleted'
    })
# -------------------------
# Demo cleanup
# -------------------------
def cleanup_output_folders(app):
    for folder in [OUTPUT_2D, OUTPUT_3D]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
    log("Cleaned up previous output folders")
