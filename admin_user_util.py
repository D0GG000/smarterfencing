"""Admin helpers: inspect and purge all data tied to a site user."""

import logging
import os
import shutil
from datetime import datetime

from sqlalchemy import func

from job_queue_models import (
    SiteUser,
    TouchShare,
    TouchShareMessage,
    TouchShareRecipient,
    TrackedFencer,
    UserJob,
)
from models import Comment, db
from touch_share_util import normalize_username
from workspace_paths import OUTPUT_2D, OUTPUT_3D, WORKSPACE_TMP

logger = logging.getLogger(__name__)

LIST_LIMIT = 100


def _iso(dt):
    return dt.isoformat() if dt else None


def _user_dict(user):
    return {
        "id": user.id,
        "email": user.email,
        "username": user.username,
        "display_name": user.display_name,
        "has_google": bool(user.google_sub),
        "has_password": bool(user.password_hash),
        "created_at": _iso(user.created_at),
        "last_seen": _iso(user.last_seen),
    }


def find_users_by_query(query, limit=20):
    """Resolve user(s) by full id, id prefix, email, or username."""
    raw = (query or "").strip()
    if not raw:
        return []

    seen = set()
    users = []

    def add(user):
        if user and user.id not in seen:
            seen.add(user.id)
            users.append(user)

    if len(raw) == 36:
        add(SiteUser.query.get(raw))

    if "@" in raw:
        add(
            SiteUser.query.filter(func.lower(SiteUser.email) == raw.lower()).first()
        )
    else:
        uname = normalize_username(raw)
        if uname:
            add(
                SiteUser.query.filter(func.lower(SiteUser.username) == uname).first()
            )

    if len(raw) >= 8:
        for row in SiteUser.query.filter(SiteUser.id.startswith(raw)).limit(limit).all():
            add(row)

    if not users and "@" not in raw:
        like = f"%{raw.lower()}%"
        for row in (
            SiteUser.query.filter(func.lower(SiteUser.email).like(like))
            .order_by(SiteUser.last_seen.desc())
            .limit(limit)
            .all()
        ):
            add(row)
        for row in (
            SiteUser.query.filter(func.lower(SiteUser.username).like(like))
            .order_by(SiteUser.last_seen.desc())
            .limit(limit)
            .all()
        ):
            add(row)

    return users[:limit]


def user_association_counts(user_id):
    return {
        "jobs": UserJob.query.filter_by(user_id=user_id).count(),
        "comments": Comment.query.filter_by(user_id=user_id).count(),
        "tracked_fencers": TrackedFencer.query.filter_by(user_id=user_id).count(),
        "touch_shares_sent": TouchShare.query.filter_by(from_user_id=user_id).count(),
        "touch_shares_received": TouchShareRecipient.query.filter_by(
            user_id=user_id
        ).count(),
        "touch_messages_authored": TouchShareMessage.query.filter_by(
            author_user_id=user_id
        ).count(),
    }


def _job_row(job):
    return {
        "job_id": job.job_id,
        "filename": job.filename,
        "status": job.status,
        "r2_object_key": job.r2_object_key,
        "created_at": _iso(job.created_at),
        "completed_at": _iso(job.completed_at),
    }


def _comment_row(comment):
    post_title = comment.post.title if comment.post else None
    return {
        "id": comment.id,
        "post_title": post_title,
        "author_name": comment.author_name,
        "body": (comment.body or "")[:300],
        "approved": comment.approved,
        "created_at": _iso(comment.created_at),
    }


def _share_sent_row(share):
    return {
        "id": share.id,
        "job_id": share.job_id,
        "kind": getattr(share, "kind", None) or "touch",
        "touch_id": share.touch_id,
        "touch_ref": share.touch_ref,
        "status": share.status,
        "created_at": _iso(share.created_at),
        "recipient_count": share.recipients.count(),
        "message_count": share.messages.count(),
    }


def _share_received_row(recipient):
    share = recipient.share
    return {
        "share_id": recipient.share_id,
        "job_id": share.job_id if share else None,
        "kind": (getattr(share, "kind", None) or "touch") if share else None,
        "touch_ref": share.touch_ref if share else None,
        "from_user_id": share.from_user_id if share else None,
        "joined_at": _iso(recipient.joined_at),
    }


def _tracked_row(row):
    return {
        "id": row.id,
        "fencer_id": row.fencer_id,
        "display_name": row.display_name,
        "slug": row.slug,
        "created_at": _iso(row.created_at),
    }


def build_user_detail(user):
    uid = user.id
    jobs = (
        UserJob.query.filter_by(user_id=uid)
        .order_by(UserJob.created_at.desc())
        .limit(LIST_LIMIT)
        .all()
    )
    comments = (
        Comment.query.filter_by(user_id=uid)
        .order_by(Comment.created_at.desc())
        .limit(LIST_LIMIT)
        .all()
    )
    shares_sent = (
        TouchShare.query.filter_by(from_user_id=uid)
        .order_by(TouchShare.created_at.desc())
        .limit(LIST_LIMIT)
        .all()
    )
    share_recipients = (
        TouchShareRecipient.query.filter_by(user_id=uid)
        .order_by(TouchShareRecipient.joined_at.desc())
        .limit(LIST_LIMIT)
        .all()
    )
    tracked = (
        TrackedFencer.query.filter_by(user_id=uid)
        .order_by(TrackedFencer.created_at.desc())
        .limit(LIST_LIMIT)
        .all()
    )
    counts = user_association_counts(uid)

    return {
        "user": _user_dict(user),
        "counts": counts,
        "jobs": [_job_row(j) for j in jobs],
        "comments": [_comment_row(c) for c in comments],
        "touch_shares_sent": [_share_sent_row(s) for s in shares_sent],
        "touch_shares_received": [_share_received_row(r) for r in share_recipients],
        "tracked_fencers": [_tracked_row(t) for t in tracked],
        "lists_truncated": {
            "jobs": counts["jobs"] > len(jobs),
            "comments": counts["comments"] > len(comments),
            "touch_shares_sent": counts["touch_shares_sent"] > len(shares_sent),
            "touch_shares_received": counts["touch_shares_received"]
            > len(share_recipients),
            "tracked_fencers": counts["tracked_fencers"] > len(tracked),
        },
    }


def cleanup_job_local_files(job_id):
    for base in (OUTPUT_2D, OUTPUT_3D):
        path = os.path.join(base, job_id)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
    tmp_video = os.path.join(WORKSPACE_TMP, f"{job_id}.mp4")
    if os.path.isfile(tmp_video):
        try:
            os.remove(tmp_video)
        except OSError as e:
            logger.warning("Could not remove tmp video %s: %s", tmp_video, e)


def user_has_active_processing_job(user_id):
    from job_queue_worker import get_current_job_id

    job = UserJob.query.filter_by(user_id=user_id, status="processing").first()
    if not job:
        return None
    if get_current_job_id() == job.job_id:
        return job
    return None


def delete_user_and_all_data(user_id, enqueue_r2_cleanup=None):
    """
    Remove a user and all associated rows.

    enqueue_r2_cleanup(job_id, r2_object_key) is called per job for async storage cleanup.
    Returns a summary dict.
    """
    user = SiteUser.query.get(user_id)
    if not user:
        raise ValueError("User not found")

    active = user_has_active_processing_job(user_id)
    if active:
        raise ValueError(
            f"Cannot delete user while job {active.job_id} is processing"
        )

    counts_before = user_association_counts(user_id)
    jobs = UserJob.query.filter_by(user_id=user_id).all()
    job_payloads = [(j.job_id, j.r2_object_key) for j in jobs]

    shares_sent = TouchShare.query.filter_by(from_user_id=user_id).all()
    share_ids_sent = [s.id for s in shares_sent]

    deleted = {
        "user_id": user_id,
        "jobs": len(jobs),
        "touch_shares_sent": len(share_ids_sent),
        "touch_shares_received": 0,
        "touch_messages_authored": 0,
        "comments": 0,
        "tracked_fencers": 0,
    }

    try:
        deleted["touch_messages_authored"] = (
            TouchShareMessage.query.filter_by(author_user_id=user_id).delete(
                synchronize_session=False
            )
        )
        deleted["touch_shares_received"] = (
            TouchShareRecipient.query.filter_by(user_id=user_id).delete(
                synchronize_session=False
            )
        )

        # Bulk share delete does not run ORM cascades — remove children first.
        for share_id in share_ids_sent:
            TouchShareMessage.query.filter_by(share_id=share_id).delete(
                synchronize_session=False
            )
            TouchShareRecipient.query.filter_by(share_id=share_id).delete(
                synchronize_session=False
            )
            TouchShare.query.filter_by(id=share_id).delete(synchronize_session=False)

        deleted["comments"] = Comment.query.filter_by(user_id=user_id).delete(
            synchronize_session=False
        )
        deleted["tracked_fencers"] = TrackedFencer.query.filter_by(
            user_id=user_id
        ).delete(synchronize_session=False)

        for job in jobs:
            db.session.delete(job)

        db.session.delete(user)
        db.session.commit()
    except Exception:
        db.session.rollback()
        raise

    for job_id, object_key in job_payloads:
        cleanup_job_local_files(job_id)
        if enqueue_r2_cleanup:
            enqueue_r2_cleanup(job_id, object_key)

    deleted["counts_before"] = counts_before
    deleted["deleted_at"] = datetime.utcnow().isoformat()
    logger.info("Admin purged user %s: %s", user_id[:8], deleted)
    return deleted


def assign_job_to_user(job_id, new_user_id):
    """Move a job to a different site_user (admin handoff)."""
    from job_queue_worker import get_current_job_id

    job = UserJob.query.filter_by(job_id=job_id).first()
    if not job:
        raise LookupError("Job not found")

    if job.status == "processing" and get_current_job_id() == job_id:
        raise ValueError("Cannot reassign a job that is currently processing")

    new_user = SiteUser.query.get(new_user_id)
    if not new_user:
        raise LookupError("User not found")

    if job.user_id == new_user_id:
        raise ValueError("Job is already assigned to this user")

    previous_user = SiteUser.query.get(job.user_id)
    previous_user_id = job.user_id
    job.user_id = new_user_id
    # Old share links and R2 snapshots may still point at the previous owner context.
    job.share_token = None
    if job.status == "complete":
        from job_results_util import ensure_job_results_complete

        ensure_job_results_complete(job, sync_r2=True)
    db.session.commit()

    return {
        "job_id": job_id,
        "previous_user_id": previous_user_id,
        "previous_user_email": previous_user.email if previous_user else None,
        "previous_username": previous_user.username if previous_user else None,
        "user_id": new_user_id,
        "user_email": new_user.email,
        "username": new_user.username,
    }
