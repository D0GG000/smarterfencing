#!/usr/bin/env python3
"""
Run the REAL smarterfencing Flask app locally (same /demo UI + queue + pipeline).

Differences from production:
  - No Cloudflare R2: browser PUTs video to this server
  - Auto-login as a local user (skip Google/email gate)
  - Workspace under local_workspace/ (not /workspace)

Usage:
  ./start_local_webapp.sh
  # or: python run_local_webapp.py

Then open: http://127.0.0.1:5000/demo
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get("WORKSPACE_ROOT", str(APP_DIR / "local_workspace"))).resolve()
PORT = int(os.environ.get("LOCAL_WEBAPP_PORT", "5000"))
LOCAL_USER_ID = os.environ.get("LOCAL_WEBAPP_USER_ID", "local-dev-user")


def _configure_env() -> None:
    os.environ.setdefault("LOCAL_WEBAPP", "1")
    os.environ.setdefault("WORKSPACE_ROOT", str(WORKSPACE))
    os.environ.setdefault("UPLOAD_DIR", str(WORKSPACE / "uploads"))
    os.environ.setdefault("OUTPUT_2D", str(WORKSPACE / "unlabeled"))
    os.environ.setdefault("OUTPUT_3D", str(WORKSPACE / "3d_outputs"))
    os.environ.setdefault("WORKSPACE_TMP", str(WORKSPACE / "tmp"))
    os.environ.setdefault("WORKSPACE_BLOG_DIR", str(WORKSPACE / "blog"))
    os.environ.setdefault(
        "DATABASE_URL",
        f"sqlite:////{(WORKSPACE / 'blog' / 'local_webapp.db').as_posix().lstrip('/')}",
    )
    os.environ.setdefault("SECRET_KEY", "local-webapp-dev-secret")
    os.environ.setdefault("PREFERRED_URL_SCHEME", "http")
    os.environ.setdefault("SESSION_COOKIE_SECURE", "0")
    # Satisfy app.py import-time R2 requirement; unused when video is local-cached.
    os.environ.setdefault("R2_ACCOUNT_ID", "local")
    os.environ.setdefault("R2_ACCESS_KEY_ID", "local")
    os.environ.setdefault("R2_SECRET_ACCESS_KEY", "local")
    os.environ.setdefault("R2_BUCKET", "local")
    os.environ.setdefault("R2_PUBLIC_URL", "")
    os.environ.setdefault("FLASK_ENV", "development")
    # Local coaching LLM via Ollama (OpenAI-compatible). Override to use cloud OpenAI.
    os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
    os.environ.setdefault("OPENAI_MODEL", "llama3.2:3b")
    os.environ.setdefault("OPENAI_API_KEY", "ollama")
    # Attack / touch models (same defaults as demo.register_demo)
    os.environ.setdefault(
        "ATTACK_MODEL_PATH",
        str(APP_DIR / "best_attack_3d_proximity_winrobust.pth"),
    )
    os.environ.setdefault(
        "MODEL_PATH",
        str(APP_DIR / "best_touch_v346_coco17_bs10_multivid_val.pth"),
    )


def main() -> int:
    _configure_env()
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    for sub in ("uploads", "unlabeled", "3d_outputs", "tmp", "blog"):
        (WORKSPACE / sub).mkdir(parents=True, exist_ok=True)

    os.chdir(APP_DIR)
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))

    # Import after env is set (app.py reads R2_ACCOUNT_ID at import).
    import app as app_module
    from flask import request, jsonify, session, g
    from workspace_paths import tmp_path, UPLOAD_DIR, ensure_workspace_dirs
    from job_queue_models import SiteUser, UserJob, db

    flask_app = app_module.app
    ensure_workspace_dirs()

    # Allow browser on Windows talking to WSL / localhost.
    try:
        from flask_cors import CORS

        CORS(
            flask_app,
            resources={
                r"/api/*": {"origins": "*"},
                r"/upload*": {"origins": "*"},
                r"/demo": {"origins": "*"},
            },
            supports_credentials=True,
        )
    except Exception:
        pass

    flask_app.config["PREFERRED_URL_SCHEME"] = "http"
    flask_app.config["SESSION_COOKIE_SECURE"] = False
    flask_app.config["LOCAL_WEBAPP"] = True

    def _ensure_local_user():
        user = SiteUser.query.get(LOCAL_USER_ID)
        if not user:
            user = SiteUser(id=LOCAL_USER_ID, email="local@localhost")
            db.session.add(user)
            db.session.commit()
        elif not user.email:
            user.email = "local@localhost"
            db.session.commit()
        return user

    def _local_auto_login():
        if session.get("user_id") != LOCAL_USER_ID:
            session["user_id"] = LOCAL_USER_ID
            session.permanent = True
        g.user_id = LOCAL_USER_ID
        # Ensure DB row exists (lazy; avoid commit on static assets if possible)
        if request.endpoint and not str(request.endpoint).startswith("static"):
            try:
                _ensure_local_user()
            except Exception:
                pass

    # Run BEFORE create_app's user-tracking hook so session is set first.
    flask_app.before_request_funcs.setdefault(None, [])
    flask_app.before_request_funcs[None].insert(0, _local_auto_login)
    def local_uploads_init():
        """Same contract as /api/uploads/init but upload_url points at this server."""
        app_module.log("LOCAL uploads/init")
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
        _ensure_local_user()
        user_job = UserJob(
            user_id=LOCAL_USER_ID,
            job_id=job_id,
            r2_object_key=object_key,
            filename=filename,
        )
        db.session.add(user_job)
        db.session.commit()

        # Absolute URL so XHR PUT works even if relative paths confuse some browsers.
        upload_url = f"{request.host_url.rstrip('/')}/api/uploads/local/{job_id}"
        app_module.log(
            f"LOCAL job {job_id} upload_url={upload_url} content_type={content_type}"
        )
        return jsonify(
            {
                "job_id": job_id,
                "object_key": object_key,
                "upload_url": upload_url,
            }
        )

    # Replace production R2 init with local.
    flask_app.view_functions["uploads_init"] = local_uploads_init

    @flask_app.put("/api/uploads/local/<job_id>")
    def local_upload_put(job_id):
        job = UserJob.query.filter_by(job_id=job_id, user_id=LOCAL_USER_ID).first()
        if not job:
            return jsonify({"error": "Job not found"}), 404

        dest = tmp_path(f"{job_id}.mp4")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        data = request.get_data()
        if not data or len(data) < 1024:
            return jsonify({"error": "Empty upload"}), 400
        with open(dest, "wb") as f:
            f.write(data)
        # Also keep a copy under uploads for debugging
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        mirror = os.path.join(UPLOAD_DIR, f"{job_id}.mp4")
        with open(mirror, "wb") as f:
            f.write(data)
        app_module.log(f"LOCAL saved video {dest} ({len(data)} bytes)")
        return ("", 200)

    @flask_app.get("/api/uploads/local/<job_id>")
    def local_upload_options_get(job_id):
        # Some stacks probe; allow GET to confirm route exists.
        return jsonify({"ok": True, "job_id": job_id})

    print("=" * 72, flush=True)
    print("LOCAL smarterfencing webapp", flush=True)
    print(f"  workspace: {WORKSPACE}", flush=True)
    print(f"  open:      http://127.0.0.1:{PORT}/demo", flush=True)
    print(f"  user:      {LOCAL_USER_ID} (auto-login)", flush=True)
    print("  Same pipeline as production: detect-frame → queue → run_full_pipeline", flush=True)
    print("=" * 72, flush=True)

    # threaded=True so queue worker + request handlers coexist
    flask_app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
