"""Helpers for in-app touch sharing between users."""

import json
import re
import uuid
from datetime import datetime

from sqlalchemy import func

from job_queue_models import (
    SiteUser,
    TouchShare,
    TouchShareMessage,
    TouchShareRecipient,
    UserJob,
)
from models import db
from results_merge import merge_results_payload
from r2_urls import video_playback_url
from touch_ref_util import assign_touch_refs, touch_display_summary

USERNAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,31}$")
RESERVED_USERNAMES = frozenset(
    {
        "admin",
        "api",
        "auth",
        "blog",
        "coach",
        "demo",
        "help",
        "inbox",
        "login",
        "logout",
        "profile",
        "register",
        "result",
        "root",
        "support",
        "system",
        "user",
        "www",
    }
)
MESSAGE_MAX_LEN = 4000


def normalize_username(value):
    if not value or not isinstance(value, str):
        return None
    v = value.strip().lower()
    if not USERNAME_RE.match(v):
        return None
    if v in RESERVED_USERNAMES:
        return None
    return v


def normalize_message_body(value):
    if not isinstance(value, str):
        return None
    body = value.strip()
    if not body:
        return None
    return body[:MESSAGE_MAX_LEN]


def touch_summary_from_id(touch_id, fps=30):
    if not touch_id:
        return "Touch"
    fm = re.search(r"frame(\d+)", touch_id)
    frame = int(fm.group(1)) if fm else 0
    fencer = "Fencer 1" if "fencer1" in touch_id else "Fencer 2" if "fencer2" in touch_id else ""
    t = frame / (fps or 30)
    m = int(t // 60)
    s = int(t % 60)
    ms = int((t % 1) * 100)
    time_str = f"{m}:{s:02d}.{ms:02d}"
    parts = [p for p in (fencer, f"@{time_str}") if p]
    return " · ".join(parts) if parts else touch_id


def _merged_job_results(job):
    results = json.loads(job.results_json) if job.results_json else {}
    return merge_results_payload(
        results,
        job.prediction_corrections_json,
        job.touch_deletions_json,
        getattr(job, "macro_corrections_json", None),
        job.selections_json,
    )


def build_touch_only_payload(job, touch_id):
    """Return touch-scoped playback data for share recipients (not full bout list)."""
    merged = _merged_job_results(job)
    predictions = merged.get("predictions") or []
    assign_touch_refs(predictions, job.job_id)
    pred = next((p for p in predictions if p.get("touch") == touch_id), None)
    if not pred:
        return None

    three_d_all = merged.get("3d_results") or {}
    touch_3d = {}
    if isinstance(three_d_all, dict) and touch_id in three_d_all:
        touch_3d = {touch_id: three_d_all[touch_id]}

    fps = merged.get("fps", 30)
    return {
        "kind": "touch",
        "prediction": pred,
        "predictions": [pred],
        "3d_results": touch_3d,
        "fps": fps,
        "video_url": video_playback_url(job.r2_object_key),
        "job_id": job.job_id,
        "filename": job.filename,
        "touch_id": touch_id,
        "touch_ref": pred.get("touch_ref"),
        "touch_summary": touch_display_summary(pred, job.job_id, fps),
    }


def build_analysis_share_payload(job):
    """Return full-bout playback data for analysis share recipients."""
    if not job or job.status != "complete":
        return None
    merged = _merged_job_results(job)
    fps = merged.get("fps", 30)
    payload = {
        "kind": "analysis",
        "fps": fps,
        "video_url": video_playback_url(job.r2_object_key),
        "job_id": job.job_id,
        "filename": job.filename,
        "touch_id": None,
        "touch_ref": None,
        "touch_summary": "Full bout analysis",
        "results_url": None,
    }
    try:
        from queue_routes import _ensure_share_token

        token = _ensure_share_token(job)
        payload["results_url"] = f"/result?share={token}"
    except Exception:
        payload["results_url"] = None
    return payload


def share_kind(share):
    return getattr(share, "kind", None) or "touch"


def user_is_share_participant(share, user_id):
    if not share or not user_id:
        return False
    if share.from_user_id == user_id:
        return True
    return (
        TouchShareRecipient.query.filter_by(share_id=share.id, user_id=user_id).first()
        is not None
    )


def resolve_recipient_by_email(email):
    if not email or not isinstance(email, str):
        return None
    addr = email.strip().lower()
    if "@" not in addr:
        return None
    return SiteUser.query.filter(func.lower(SiteUser.email) == addr).first()


def resolve_recipient_by_username(username):
    if not username or not isinstance(username, str):
        return None
    uname = normalize_username(username)
    if not uname:
        return None
    return SiteUser.query.filter(func.lower(SiteUser.username) == uname).first()


def resolve_recipient(identifier):
    """Look up a SiteUser by email or username."""
    if not identifier or not isinstance(identifier, str):
        return None
    raw = identifier.strip()
    if not raw:
        return None
    if "@" in raw:
        return resolve_recipient_by_email(raw)
    return resolve_recipient_by_username(raw)


def last_message_for_share(share_id):
    return (
        TouchShareMessage.query.filter_by(share_id=share_id)
        .order_by(TouchShareMessage.created_at.desc())
        .first()
    )


def message_to_dict(msg):
    author = msg.author
    return {
        "id": msg.id,
        "body": msg.body,
        "created_at": msg.created_at.isoformat() if msg.created_at else None,
        "author": author.public_profile() if author else None,
        "is_mine": False,
    }


def share_list_item(share, viewer_id, direction):
    kind = share_kind(share)
    job = UserJob.query.filter_by(job_id=share.job_id).first()
    merged = _merged_job_results(job) if job else {}
    fps = merged.get("fps", 30)
    preds = merged.get("predictions") or []
    assign_touch_refs(preds, share.job_id if job else None)

    touch_ref = share.touch_ref
    summary = "Full bout analysis"
    prediction = None
    if kind == "analysis":
        summary = "Full bout analysis"
        if job and job.filename:
            summary = f"Full bout analysis · {job.filename}"
    else:
        pred = None
        for p in preds:
            if p.get("touch") == share.touch_id:
                pred = p
                break
        touch_ref = touch_ref or (pred or {}).get("touch_ref")
        summary = (
            touch_display_summary(pred, share.job_id, fps)
            if pred
            else touch_summary_from_id(share.touch_id, fps)
        )
        prediction = (pred or {}).get("prediction")

    last_msg = last_message_for_share(share.id)
    unread = False
    if direction == "received":
        rec = TouchShareRecipient.query.filter_by(
            share_id=share.id, user_id=viewer_id
        ).first()
        if rec and last_msg:
            if not rec.last_read_at or last_msg.created_at > rec.last_read_at:
                unread = True
    elif direction == "sent" and last_msg and last_msg.author_user_id != viewer_id:
        unread = True

    from_user = share.from_user.public_profile() if share.from_user else None
    recipients = []
    for rec in share.recipients.all():
        if rec.user:
            recipients.append(rec.user.public_profile())

    item = {
        "id": share.id,
        "kind": kind,
        "job_id": share.job_id,
        "touch_id": share.touch_id or None,
        "touch_ref": touch_ref,
        "touch_summary": summary,
        "prediction": prediction,
        "filename": job.filename if job else None,
        "from_user": from_user,
        "recipients": recipients,
        "direction": direction,
        "created_at": share.created_at.isoformat() if share.created_at else None,
        "unread": unread,
        "message_count": share.messages.count(),
    }
    if last_msg:
        lm = message_to_dict(last_msg)
        lm["is_mine"] = last_msg.author_user_id == viewer_id
        item["last_message"] = lm
    return item


def mark_share_read(share, user_id):
    if share.from_user_id == user_id:
        return
    rec = TouchShareRecipient.query.filter_by(share_id=share.id, user_id=user_id).first()
    if rec:
        rec.last_read_at = datetime.utcnow()
        db.session.commit()


def _add_share_recipients(share, from_user, recipients):
    seen_recipients = set()
    for raw in recipients:
        user = resolve_recipient(raw)
        if not user:
            raise ValueError(f"No account found for {raw.strip()}")
        if user.id == from_user.id:
            raise ValueError("You cannot share with yourself")
        if user.id in seen_recipients:
            continue
        seen_recipients.add(user.id)
        db.session.add(
            TouchShareRecipient(
                share_id=share.id,
                user_id=user.id,
            )
        )
    if not seen_recipients:
        raise ValueError("Add at least one recipient (email or username)")
    return seen_recipients


def create_touch_share(from_user, job, touch_id, recipients, comment, kind="touch"):
    kind = (kind or "touch").strip().lower()
    if kind not in ("touch", "analysis"):
        raise ValueError("kind must be 'touch' or 'analysis'")

    if kind == "analysis":
        payload = build_analysis_share_payload(job)
        if not payload:
            raise ValueError("Analysis is not available to share")
        share = TouchShare(
            id=str(uuid.uuid4()),
            job_id=job.job_id,
            kind="analysis",
            touch_id="",
            touch_ref=None,
            from_user_id=from_user.id,
            status="active",
        )
        message_body = comment
    else:
        payload = build_touch_only_payload(job, touch_id)
        if not payload:
            raise ValueError("Touch not found in this analysis")
        touch_ref = payload.get("touch_ref")
        share = TouchShare(
            id=str(uuid.uuid4()),
            job_id=job.job_id,
            kind="touch",
            touch_id=touch_id,
            touch_ref=touch_ref,
            from_user_id=from_user.id,
            status="active",
        )
        message_body = (
            f"[{touch_ref}] {comment}"
            if touch_ref and not comment.strip().startswith(f"[{touch_ref}]")
            else comment
        )

    db.session.add(share)
    _add_share_recipients(share, from_user, recipients)
    db.session.add(
        TouchShareMessage(
            id=str(uuid.uuid4()),
            share_id=share.id,
            author_user_id=from_user.id,
            body=message_body,
        )
    )
    db.session.commit()
    return share


def unread_received_count(user_id):
    count = 0
    recs = TouchShareRecipient.query.filter_by(user_id=user_id).all()
    for rec in recs:
        last_msg = last_message_for_share(rec.share_id)
        if not last_msg:
            continue
        if not rec.last_read_at or last_msg.created_at > rec.last_read_at:
            count += 1
    return count
