"""In-app sharing: send touches or full analyses to other users with threaded feedback."""

import uuid

from flask import Blueprint, jsonify, render_template, request, session

from job_queue_models import (
    SiteUser,
    TouchShare,
    TouchShareMessage,
    TouchShareRecipient,
    UserJob,
)
from models import db
from touch_share_util import (
    build_analysis_share_payload,
    build_touch_only_payload,
    create_touch_share,
    mark_share_read,
    message_to_dict,
    normalize_message_body,
    share_kind,
    share_list_item,
    unread_received_count,
    user_is_share_participant,
)

touch_share_bp = Blueprint("touch_share", __name__)


def _require_user():
    uid = session.get("user_id")
    if not uid:
        return None, (jsonify({"success": False, "error": "Authentication required"}), 401)
    user = SiteUser.query.get(uid)
    if not user:
        session.clear()
        return None, (jsonify({"success": False, "error": "Authentication required"}), 401)
    return user, None


def _require_username(user):
    if not user.username:
        return jsonify(
            {
                "success": False,
                "error": "Set a username in your profile before sharing.",
                "needs_username": True,
            }
        ), 400
    return None


@touch_share_bp.route("/inbox")
@touch_share_bp.route("/inbox/<share_id>")
def inbox_page(share_id=None):
    return render_template("inbox.html", share_id=share_id or "")


@touch_share_bp.route("/profile")
def profile_page():
    return render_template("profile.html")


@touch_share_bp.get("/api/touch-shares/unread-count")
def api_unread_count():
    user, err = _require_user()
    if err:
        return err
    return jsonify({"success": True, "unread": unread_received_count(user.id)})


@touch_share_bp.get("/api/touch-shares/received")
def api_received():
    user, err = _require_user()
    if err:
        return err
    recs = (
        TouchShareRecipient.query.filter_by(user_id=user.id)
        .order_by(TouchShareRecipient.joined_at.desc())
        .all()
    )
    items = []
    for rec in recs:
        share = TouchShare.query.get(rec.share_id)
        if share and share.status == "active":
            items.append(share_list_item(share, user.id, "received"))
    items.sort(key=lambda x: x.get("last_message", {}).get("created_at") or x.get("created_at") or "", reverse=True)
    return jsonify({"success": True, "shares": items})


@touch_share_bp.get("/api/touch-shares/sent")
def api_sent():
    user, err = _require_user()
    if err:
        return err
    shares = (
        TouchShare.query.filter_by(from_user_id=user.id, status="active")
        .order_by(TouchShare.created_at.desc())
        .all()
    )
    items = [share_list_item(s, user.id, "sent") for s in shares]
    items.sort(key=lambda x: x.get("last_message", {}).get("created_at") or x.get("created_at") or "", reverse=True)
    return jsonify({"success": True, "shares": items})


@touch_share_bp.post("/api/touch-shares")
def api_create_share():
    user, err = _require_user()
    if err:
        return err
    uname_err = _require_username(user)
    if uname_err:
        return uname_err

    data = request.get_json(silent=True) or {}
    job_id = (data.get("job_id") or "").strip()
    touch_id = (data.get("touch_id") or "").strip()
    kind = (data.get("kind") or ("analysis" if not touch_id else "touch")).strip().lower()
    comment = normalize_message_body(data.get("comment") or "")
    raw_recipients = (
        data.get("recipients")
        or data.get("recipient_emails")
        or data.get("recipient_usernames")
        or []
    )

    if kind not in ("touch", "analysis"):
        return jsonify({"success": False, "error": "kind must be 'touch' or 'analysis'"}), 400
    if not job_id:
        return jsonify({"success": False, "error": "job_id required"}), 400
    if kind == "touch" and not touch_id:
        return jsonify({"success": False, "error": "job_id and touch_id required"}), 400
    if not comment:
        return jsonify({"success": False, "error": "Comment is required"}), 400

    if isinstance(raw_recipients, str):
        raw_recipients = [
            r.strip() for r in raw_recipients.replace(";", ",").split(",") if r.strip()
        ]
    elif not isinstance(raw_recipients, list):
        raw_recipients = []

    job = UserJob.query.filter_by(job_id=job_id, user_id=user.id).first()
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404
    if job.status != "complete":
        return jsonify({"success": False, "error": "Job is not complete"}), 400

    if kind == "touch":
        payload = build_touch_only_payload(job, touch_id)
        if not payload:
            return jsonify({"success": False, "error": "Touch not found in this analysis"}), 404
    else:
        payload = build_analysis_share_payload(job)
        if not payload:
            return jsonify({"success": False, "error": "Analysis is not available to share"}), 404

    try:
        share = create_touch_share(
            user, job, touch_id if kind == "touch" else None, raw_recipients, comment, kind=kind
        )
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    return jsonify(
        {
            "success": True,
            "share_id": share.id,
            "share_url": f"/inbox/{share.id}",
        }
    )


@touch_share_bp.get("/api/touch-shares/<share_id>")
def api_get_share(share_id):
    user, err = _require_user()
    if err:
        return err

    share = TouchShare.query.get(share_id)
    if not share or share.status != "active":
        return jsonify({"success": False, "error": "Not found"}), 404
    if not user_is_share_participant(share, user.id):
        return jsonify({"success": False, "error": "Not found"}), 404

    job = UserJob.query.filter_by(job_id=share.job_id).first()
    if not job:
        return jsonify({"success": False, "error": "Analysis no longer available"}), 404

    kind = share_kind(share)
    if kind == "analysis":
        touch_payload = build_analysis_share_payload(job)
        if not touch_payload:
            return jsonify({"success": False, "error": "Analysis data unavailable"}), 404
    else:
        touch_payload = build_touch_only_payload(job, share.touch_id)
        if not touch_payload:
            return jsonify({"success": False, "error": "Touch data unavailable"}), 404

    messages = []
    for msg in share.messages.order_by(TouchShareMessage.created_at.asc()):
        d = message_to_dict(msg)
        d["is_mine"] = msg.author_user_id == user.id
        messages.append(d)

    mark_share_read(share, user.id)

    direction = "sent" if share.from_user_id == user.id else "received"
    return jsonify(
        {
            "success": True,
            "share": share_list_item(share, user.id, direction),
            "touch": touch_payload,
            "messages": messages,
        }
    )


@touch_share_bp.post("/api/touch-shares/<share_id>/messages")
def api_post_message(share_id):
    user, err = _require_user()
    if err:
        return err

    share = TouchShare.query.get(share_id)
    if not share or share.status != "active":
        return jsonify({"success": False, "error": "Not found"}), 404
    if not user_is_share_participant(share, user.id):
        return jsonify({"success": False, "error": "Not found"}), 404

    body = normalize_message_body((request.get_json(silent=True) or {}).get("body") or "")
    if not body:
        return jsonify({"success": False, "error": "Message cannot be empty"}), 400

    msg = TouchShareMessage(
        id=str(uuid.uuid4()),
        share_id=share.id,
        author_user_id=user.id,
        body=body,
    )
    db.session.add(msg)
    db.session.commit()

    d = message_to_dict(msg)
    d["is_mine"] = True
    return jsonify({"success": True, "message": d})


@touch_share_bp.patch("/api/touch-shares/<share_id>/read")
def api_mark_read(share_id):
    user, err = _require_user()
    if err:
        return err

    share = TouchShare.query.get(share_id)
    if not share or not user_is_share_participant(share, user.id):
        return jsonify({"success": False, "error": "Not found"}), 404

    mark_share_read(share, user.id)
    return jsonify({"success": True})


def register_touch_share_routes(app):
    app.register_blueprint(touch_share_bp)
