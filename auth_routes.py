"""Session auth: Google OAuth (Authlib) and email/password."""

import os
import uuid
from datetime import datetime

from authlib.integrations.flask_client import OAuth
from flask import (
    Blueprint,
    current_app,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from sqlalchemy import func
from werkzeug.security import check_password_hash, generate_password_hash

from models import db
from job_queue_models import SiteUser, UserJob

auth_bp = Blueprint("auth", __name__)
oauth = OAuth()

COOKIE_NAME = "sf_user_id"


def _migrate_anonymous_jobs(new_user_id: str) -> None:
    """Attach jobs from legacy anonymous cookie user to the logged-in account."""
    anon_id = request.cookies.get(COOKIE_NAME)
    if not anon_id or anon_id == new_user_id:
        return
    anon = SiteUser.query.get(anon_id)
    if not anon:
        return
    if anon.google_sub or anon.password_hash:
        return
    updated = UserJob.query.filter_by(user_id=anon_id).update({"user_id": new_user_id})
    if updated:
        db.session.commit()


def register_auth(app):
    app.config.setdefault("GOOGLE_CLIENT_ID", os.environ.get("GOOGLE_CLIENT_ID", ""))
    app.config.setdefault("GOOGLE_CLIENT_SECRET", os.environ.get("GOOGLE_CLIENT_SECRET", ""))
    app.config.setdefault(
        "OAUTH_REDIRECT_URI",
        os.environ.get("OAUTH_REDIRECT_URI", "").strip(),
    )

    oauth.init_app(app)

    if app.config.get("GOOGLE_CLIENT_ID") and app.config.get("GOOGLE_CLIENT_SECRET"):
        oauth.register(
            name="google",
            client_id=app.config["GOOGLE_CLIENT_ID"],
            client_secret=app.config["GOOGLE_CLIENT_SECRET"],
            server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
            client_kwargs={"scope": "openid email profile"},
        )

    app.register_blueprint(auth_bp)


def _google_client_ready():
    return bool(
        current_app.config.get("GOOGLE_CLIENT_ID")
        and current_app.config.get("GOOGLE_CLIENT_SECRET")
    )


@auth_bp.get("/auth/google")
def google_login():
    if not _google_client_ready():
        return redirect(url_for("demo.demo") + "?auth_error=google_not_configured")
    redirect_uri = current_app.config.get("OAUTH_REDIRECT_URI") or url_for(
        "auth.google_callback", _external=True
    )
    return oauth.google.authorize_redirect(redirect_uri)


@auth_bp.get("/auth/google/callback")
def google_callback():
    if not _google_client_ready():
        return redirect("/demo?auth_error=google_not_configured")
    try:
        token = oauth.google.authorize_access_token()
        userinfo = token.get("userinfo")
        if not userinfo:
            resp = oauth.google.get("https://openidconnect.googleapis.com/v1/userinfo")
            userinfo = resp.json()
        sub = userinfo.get("sub")
        if not sub:
            return redirect("/demo?auth_error=no_sub")

        email = (userinfo.get("email") or "").strip() or None
        name = (userinfo.get("name") or "").strip() or None

        user = SiteUser.query.filter_by(google_sub=sub).first()
        if not user:
            user = SiteUser(
                id=str(uuid.uuid4()),
                google_sub=sub,
                email=email,
                display_name=name,
            )
            db.session.add(user)
        else:
            user.email = email or user.email
            user.display_name = name or user.display_name
            user.last_seen = datetime.utcnow()
        db.session.commit()

        _migrate_anonymous_jobs(user.id)
        session["user_id"] = user.id
        session.permanent = True
        return redirect("/demo")
    except Exception as e:
        current_app.logger.exception("Google OAuth failed: %s", e)
        return redirect("/demo?auth_error=oauth_failed")


@auth_bp.post("/api/auth/logout")
def api_logout():
    session.clear()
    return jsonify({"success": True})


@auth_bp.get("/auth/logout")
def logout_redirect():
    session.clear()
    return redirect("/")


@auth_bp.get("/api/auth/me")
def api_me():
    uid = session.get("user_id")
    if not uid:
        return jsonify({"logged_in": False, "user": None})
    user = SiteUser.query.get(uid)
    if not user:
        session.clear()
        return jsonify({"logged_in": False, "user": None})
    return jsonify(
        {
            "logged_in": True,
            "user": {
                **user.to_dict(),
                "id": user.id,
                "username": user.username,
                "has_username": bool(user.username),
            },
        }
    )


@auth_bp.patch("/api/auth/profile")
def api_update_profile():
    uid = session.get("user_id")
    if not uid:
        return jsonify({"success": False, "error": "Authentication required"}), 401
    user = SiteUser.query.get(uid)
    if not user:
        session.clear()
        return jsonify({"success": False, "error": "Authentication required"}), 401

    from touch_share_util import normalize_username

    data = request.get_json(silent=True) or {}

    if "display_name" in data:
        name = (data.get("display_name") or "").strip()
        user.display_name = name[:255] if name else None

    if "username" in data:
        raw = data.get("username")
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return jsonify({"success": False, "error": "Username cannot be cleared once set"}), 400
        uname = normalize_username(raw)
        if not uname:
            return jsonify(
                {
                    "success": False,
                    "error": "Username must be 3–32 characters: lowercase letters, numbers, underscore; must start with a letter.",
                }
            ), 400
        existing = SiteUser.query.filter(
            func.lower(SiteUser.username) == uname, SiteUser.id != user.id
        ).first()
        if existing:
            return jsonify({"success": False, "error": "Username already taken"}), 400
        if user.username and user.username != uname:
            return jsonify({"success": False, "error": "Username cannot be changed after it is set"}), 400
        user.username = uname

    user.last_seen = datetime.utcnow()
    db.session.commit()
    return jsonify(
        {
            "success": True,
            "user": {
                "id": user.id,
                "email": user.email,
                "display_name": user.display_name,
                "username": user.username,
                "has_username": bool(user.username),
            },
        }
    )


@auth_bp.post("/api/auth/register")
def api_register():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not email or "@" not in email:
        return jsonify({"success": False, "error": "Valid email required"}), 400
    if len(password) < 8:
        return jsonify({"success": False, "error": "Password must be at least 8 characters"}), 400

    existing = SiteUser.query.filter(func.lower(SiteUser.email) == email).first()
    if existing:
        return jsonify({"success": False, "error": "Email already registered"}), 400

    user = SiteUser(
        id=str(uuid.uuid4()),
        email=email,
        password_hash=generate_password_hash(password),
    )
    db.session.add(user)
    db.session.commit()
    _migrate_anonymous_jobs(user.id)
    session["user_id"] = user.id
    session.permanent = True
    return jsonify({"success": True, "user": {"id": user.id, "email": user.email}})


@auth_bp.post("/api/auth/login")
def api_login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not email or not password:
        return jsonify({"success": False, "error": "Email and password required"}), 400

    user = SiteUser.query.filter(func.lower(SiteUser.email) == email).first()
    if not user or not user.password_hash:
        return jsonify({"success": False, "error": "Invalid email or password"}), 401
    if not check_password_hash(user.password_hash, password):
        return jsonify({"success": False, "error": "Invalid email or password"}), 401

    user.last_seen = datetime.utcnow()
    db.session.commit()
    _migrate_anonymous_jobs(user.id)
    session["user_id"] = user.id
    session.permanent = True
    return jsonify({"success": True, "user": {"id": user.id, "email": user.email}})


@auth_bp.get("/reset-password")
def reset_password_page():
    token = (request.args.get("token") or "").strip()
    return render_template("reset_password.html", token=token)


@auth_bp.post("/api/auth/forgot-password")
def api_forgot_password():
    from auth_password_reset import RESET_MAX_AGE_SEC, send_password_reset_email, make_reset_token

    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    generic = {
        "success": True,
        "message": (
            "If an account exists for that email, we sent a password reset link. "
            f"Check your inbox (link expires in {RESET_MAX_AGE_SEC // 60} minutes)."
        ),
    }
    if not email or "@" not in email:
        return jsonify({"success": False, "error": "Valid email required"}), 400

    user = SiteUser.query.filter(func.lower(SiteUser.email) == email).first()
    if not user or not user.email:
        return jsonify(generic)

    token = make_reset_token(user.id)
    ok, err = send_password_reset_email(user.email, token)
    if not ok:
        current_app.logger.warning("Password reset email failed for %s: %s", email, err)
        return jsonify({
            "success": False,
            "error": "Could not send reset email. Try again later or contact support.",
        }), 503

    return jsonify(generic)


@auth_bp.post("/api/auth/reset-password")
def api_reset_password():
    from auth_password_reset import load_reset_user_id

    data = request.get_json(silent=True) or {}
    token = (data.get("token") or "").strip()
    password = data.get("password") or ""
    if not token:
        return jsonify({"success": False, "error": "Reset token required"}), 400
    if len(password) < 8:
        return jsonify({"success": False, "error": "Password must be at least 8 characters"}), 400

    user_id = load_reset_user_id(token)
    if not user_id:
        return jsonify({
            "success": False,
            "error": "This reset link is invalid or has expired. Request a new one.",
        }), 400

    user = SiteUser.query.get(user_id)
    if not user:
        return jsonify({"success": False, "error": "Account not found"}), 404

    user.password_hash = generate_password_hash(password)
    user.last_seen = datetime.utcnow()
    db.session.commit()

    _migrate_anonymous_jobs(user.id)
    session["user_id"] = user.id
    session.permanent = True
    return jsonify({
        "success": True,
        "message": "Password updated. You are now signed in.",
        "user": {"id": user.id, "email": user.email},
    })
