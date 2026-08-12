"""Public Community Hub: share analyses, clips, or highlight reels publicly.

This is intentionally separate from the private touch-share / Messages system
(touch_share_routes.py). Anyone can view posts and their touches; signed-in
users with a username can create posts and comment.
"""

import logging
import uuid

from flask import Blueprint, jsonify, render_template, request, session

from job_queue_models import (
    CommunityComment,
    CommunityPost,
    SiteUser,
    UserJob,
)
from models import db
from r2_urls import video_playback_url
from touch_share_util import (
    build_touch_only_payload,
    normalize_message_body,
    touch_summary_from_id,
)

logger = logging.getLogger(__name__)

community_bp = Blueprint("community", __name__)

POST_KINDS = ("touch", "highlight_reel", "analysis")
CAPTION_MAX_LEN = 4000
FEED_LIMIT = 100


def _current_user_id():
    return session.get("user_id")


def _require_user():
    uid = session.get("user_id")
    if not uid:
        return None, (jsonify({"success": False, "error": "Sign in to continue."}), 401)
    user = SiteUser.query.get(uid)
    if not user:
        session.clear()
        return None, (jsonify({"success": False, "error": "Sign in to continue."}), 401)
    return user, None


def _require_username(user):
    if not user.username:
        return jsonify(
            {
                "success": False,
                "error": "Set a username in your profile before posting.",
                "needs_username": True,
            }
        ), 400
    return None


def _kind_title(post):
    if post.kind == "touch":
        return touch_summary_from_id(post.touch_id) if post.touch_id else "Touch"
    if post.kind == "highlight_reel":
        return "Highlight reel"
    return "Full bout analysis"


def _comment_to_dict(comment, viewer_id=None):
    author = comment.author
    return {
        "id": comment.id,
        "body": comment.body,
        "created_at": comment.created_at.isoformat() if comment.created_at else None,
        "author": author.public_profile() if author else None,
        "is_mine": bool(viewer_id) and comment.author_user_id == viewer_id,
    }


def _post_list_item(post, viewer_id=None):
    return {
        "id": post.id,
        "kind": post.kind,
        "title": _kind_title(post),
        "caption": post.caption or "",
        "touch_ref": post.touch_ref,
        "author": post.author.public_profile() if post.author else None,
        "comment_count": post.comments.count(),
        "created_at": post.created_at.isoformat() if post.created_at else None,
        "is_mine": bool(viewer_id) and post.author_user_id == viewer_id,
        "can_delete": bool(viewer_id) and post.author_user_id == viewer_id,
    }


def _post_media(post):
    """Playback payload for the post, or None if the source job is unavailable."""
    job = UserJob.query.filter_by(job_id=post.job_id).first()
    if not job or job.status != "complete":
        return None

    media = {"kind": post.kind, "filename": job.filename}

    if post.kind == "touch" and post.touch_id:
        payload = build_touch_only_payload(job, post.touch_id)
        if not payload:
            return None
        media.update(
            {
                "video_url": payload.get("video_url"),
                "fps": payload.get("fps", 30),
                "touch_id": payload.get("touch_id"),
                "touch_ref": payload.get("touch_ref"),
                "touch_summary": payload.get("touch_summary"),
            }
        )
    elif post.kind == "highlight_reel":
        if not getattr(job, "highlight_reel_key", None):
            return None
        media.update({"video_url": video_playback_url(job.highlight_reel_key), "fps": 30})
    else:  # analysis
        media.update({"video_url": video_playback_url(job.r2_object_key), "fps": 30})
        try:
            from queue_routes import _ensure_share_token

            token = _ensure_share_token(job)
            media["results_url"] = f"/result?share={token}"
        except Exception:
            media["results_url"] = None

    return media


@community_bp.route("/community")
@community_bp.route("/community/<post_id>")
def community_page(post_id=None):
    return render_template("community.html", post_id=post_id or "")


@community_bp.get("/api/community/posts")
def api_list_posts():
    """Public feed of active posts, newest first."""
    try:
        viewer_id = _current_user_id()
        posts = (
            CommunityPost.query.filter_by(status="active")
            .order_by(CommunityPost.created_at.desc())
            .limit(FEED_LIMIT)
            .all()
        )
        items = []
        for p in posts:
            try:
                items.append(_post_list_item(p, viewer_id))
            except Exception:
                logger.exception("community: failed to serialize post %s", getattr(p, "id", "?"))
        return jsonify({"success": True, "posts": items})
    except Exception as e:
        logger.exception("community: failed to list posts")
        return jsonify(
            {"success": False, "error": "Could not load the community feed.", "detail": str(e)}
        ), 500


@community_bp.get("/api/community/posts/<post_id>")
def api_get_post(post_id):
    """Public: a single post with its media and comments."""
    try:
        post = CommunityPost.query.get(post_id)
        if not post or post.status != "active":
            return jsonify({"success": False, "error": "Post not found"}), 404

        media = _post_media(post)
        if media is None:
            return jsonify({"success": False, "error": "This analysis is no longer available"}), 404

        viewer_id = _current_user_id()
        comments = [
            _comment_to_dict(c, viewer_id)
            for c in post.comments.order_by(CommunityComment.created_at.asc())
        ]

        return jsonify(
            {
                "success": True,
                "post": _post_list_item(post, viewer_id),
                "media": media,
                "comments": comments,
                "can_delete": bool(viewer_id) and post.author_user_id == viewer_id,
            }
        )
    except Exception:
        logger.exception("community: failed to load post %s", post_id)
        return jsonify({"success": False, "error": "Could not load this post."}), 500


@community_bp.post("/api/community/posts")
def api_create_post():
    user, err = _require_user()
    if err:
        return err
    uname_err = _require_username(user)
    if uname_err:
        return uname_err

    data = request.get_json(silent=True) or {}
    job_id = (data.get("job_id") or "").strip()
    touch_id = (data.get("touch_id") or "").strip() or None
    kind = (data.get("kind") or "").strip()
    caption = normalize_message_body(data.get("caption") or "") or ""

    if not job_id:
        return jsonify({"success": False, "error": "job_id is required"}), 400

    if kind not in POST_KINDS:
        kind = "touch" if touch_id else "analysis"
    if kind == "touch" and not touch_id:
        return jsonify({"success": False, "error": "Select a touch to post"}), 400

    job = UserJob.query.filter_by(job_id=job_id, user_id=user.id).first()
    if not job:
        return jsonify({"success": False, "error": "Analysis not found"}), 404
    if job.status != "complete":
        return jsonify({"success": False, "error": "Analysis is not complete"}), 400

    try:
        touch_ref = None
        if kind == "touch":
            payload = build_touch_only_payload(job, touch_id)
            if not payload:
                return jsonify({"success": False, "error": "Touch not found in this analysis"}), 404
            touch_ref = payload.get("touch_ref")
        elif kind == "highlight_reel" and not getattr(job, "highlight_reel_key", None):
            return jsonify({"success": False, "error": "This analysis has no highlight reel"}), 400

        post = CommunityPost(
            id=str(uuid.uuid4()),
            author_user_id=user.id,
            job_id=job_id,
            kind=kind,
            touch_id=touch_id,
            touch_ref=touch_ref,
            caption=caption,
            status="active",
        )
        db.session.add(post)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.exception("community: failed to create post for job %s", job_id)
        return jsonify({"success": False, "error": "Could not create post.", "detail": str(e)}), 500

    return jsonify({"success": True, "post_id": post.id, "post_url": f"/community/{post.id}"})


@community_bp.post("/api/community/posts/<post_id>/comments")
def api_add_comment(post_id):
    user, err = _require_user()
    if err:
        return err
    uname_err = _require_username(user)
    if uname_err:
        return uname_err

    post = CommunityPost.query.get(post_id)
    if not post or post.status != "active":
        return jsonify({"success": False, "error": "Post not found"}), 404

    body = normalize_message_body((request.get_json(silent=True) or {}).get("body") or "")
    if not body:
        return jsonify({"success": False, "error": "Comment cannot be empty"}), 400

    try:
        comment = CommunityComment(
            id=str(uuid.uuid4()),
            post_id=post.id,
            author_user_id=user.id,
            body=body,
        )
        db.session.add(comment)
        db.session.commit()
    except Exception:
        db.session.rollback()
        logger.exception("community: failed to add comment to post %s", post_id)
        return jsonify({"success": False, "error": "Could not add comment."}), 500

    return jsonify({"success": True, "comment": _comment_to_dict(comment, user.id)})


@community_bp.delete("/api/community/posts/<post_id>")
def api_delete_post(post_id):
    """Author-only soft delete (removes from the public hub)."""
    user, err = _require_user()
    if err:
        return err
    post = CommunityPost.query.get(post_id)
    if not post or post.status == "deleted":
        return jsonify({"success": False, "error": "Post not found"}), 404
    if post.author_user_id != user.id:
        return jsonify({"success": False, "error": "Not allowed"}), 403
    try:
        post.status = "deleted"
        db.session.commit()
    except Exception:
        db.session.rollback()
        logger.exception("community: failed to delete post %s", post_id)
        return jsonify({"success": False, "error": "Could not delete post."}), 500
    return jsonify({"success": True})


@community_bp.delete("/api/community/comments/<comment_id>")
def api_delete_comment(comment_id):
    user, err = _require_user()
    if err:
        return err
    comment = CommunityComment.query.get(comment_id)
    if not comment:
        return jsonify({"success": False, "error": "Comment not found"}), 404
    if comment.author_user_id != user.id:
        return jsonify({"success": False, "error": "Not allowed"}), 403
    db.session.delete(comment)
    db.session.commit()
    return jsonify({"success": True})


def register_community_routes(app):
    app.register_blueprint(community_bp)
