"""Fencer lookup (Phase 1): name search via FencingTracker, link out, save bookmarks."""

from flask import Blueprint, jsonify, render_template, request, session

from fencer_tracker_registrations import summarize_schedule_flag
from fencer_tracker_search import search_fencers_by_name
from fencer_tracker_util import build_profile_url, parse_fencer_input, slug_to_display_name
from job_queue_models import TrackedFencer
from models import db

fencer_bp = Blueprint("fencer", __name__)


def _require_user_id():
    return session.get("user_id")


def _tracked_to_dict(row: TrackedFencer, *, include_schedule: bool = False) -> dict:
    payload = {
        "id": row.id,
        "fencer_id": row.fencer_id,
        "slug": row.slug,
        "display_name": row.display_name,
        "user_note": row.user_note or "",
        "profile_url": row.profile_url,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }
    if include_schedule:
        # Live FencingTracker lookup for this response only — not saved anywhere.
        schedule = summarize_schedule_flag(row.fencer_id, row.slug)
        payload.update(
            {
                "schedule_status": schedule["schedule_status"],
                "schedule_summary": schedule["schedule_summary"],
            }
        )
    return payload


def _fencer_from_request(data: dict):
    fencer_id = str(data.get("fencer_id") or "").strip()
    if fencer_id.isdigit():
        slug = (data.get("slug") or "").strip() or None
        display_name = (data.get("display_name") or "").strip()
        if not display_name and slug:
            display_name = slug_to_display_name(slug)
        if not display_name:
            display_name = f"Fencer {fencer_id}"
        return {
            "fencer_id": fencer_id,
            "slug": slug,
            "display_name": display_name,
            "profile_url": build_profile_url(fencer_id, slug),
        }

    parsed = parse_fencer_input(data.get("input") or "")
    if parsed:
        return parsed
    if fencer_id:
        return parse_fencer_input(fencer_id)
    return None


@fencer_bp.get("/fencers")
def fencer_lookup_page():
    return render_template("fencer_lookup.html")


@fencer_bp.post("/api/fencers/search")
def api_search_fencers():
    data = request.get_json(silent=True) or {}
    first_name = (data.get("first_name") or "").strip()
    last_name = (data.get("last_name") or "").strip()

    fencers, err = search_fencers_by_name(first_name, last_name)
    if err:
        return jsonify({"success": False, "error": err}), 400

    if not fencers:
        return jsonify(
            {
                "success": True,
                "fencers": [],
                "message": "No fencers found. Check spelling or try a different name.",
            }
        )

    return jsonify({"success": True, "fencers": fencers})


@fencer_bp.get("/api/fencers/tracked")
def api_list_tracked():
    uid = _require_user_id()
    if not uid:
        return jsonify({"success": False, "error": "Authentication required"}), 401

    rows = (
        TrackedFencer.query.filter_by(user_id=uid)
        .order_by(TrackedFencer.created_at.desc())
        .all()
    )
    return jsonify(
        {
            "success": True,
            "opponents": [_tracked_to_dict(r, include_schedule=True) for r in rows],
        }
    )


@fencer_bp.post("/api/fencers/tracked")
def api_add_tracked():
    uid = _require_user_id()
    if not uid:
        return jsonify({"success": False, "error": "Authentication required"}), 401

    data = request.get_json(silent=True) or {}
    parsed = _fencer_from_request(data)
    if not parsed:
        return jsonify({"success": False, "error": "Select a fencer from search results first"}), 400

    display_name = (data.get("display_name") or parsed["display_name"] or "").strip()
    if not display_name:
        display_name = f"Fencer {parsed['fencer_id']}"

    user_note = (data.get("user_note") or "").strip() or None
    slug = data.get("slug") if data.get("slug") is not None else parsed.get("slug")

    existing = TrackedFencer.query.filter_by(
        user_id=uid, fencer_id=parsed["fencer_id"]
    ).first()
    if existing:
        if display_name:
            existing.display_name = display_name[:255]
        if slug is not None:
            existing.slug = (slug or "")[:255] or None
        if user_note is not None:
            existing.user_note = user_note
        db.session.commit()
        return jsonify({"success": True, "opponent": _tracked_to_dict(existing), "updated": True})

    row = TrackedFencer(
        user_id=uid,
        fencer_id=parsed["fencer_id"],
        slug=(slug or "")[:255] or None,
        display_name=display_name[:255],
        user_note=user_note,
    )
    db.session.add(row)
    db.session.commit()
    return jsonify({"success": True, "opponent": _tracked_to_dict(row), "updated": False})


@fencer_bp.patch("/api/fencers/tracked/<int:tracked_id>")
def api_update_tracked(tracked_id: int):
    uid = _require_user_id()
    if not uid:
        return jsonify({"success": False, "error": "Authentication required"}), 401

    row = TrackedFencer.query.filter_by(id=tracked_id, user_id=uid).first()
    if not row:
        return jsonify({"success": False, "error": "Not found"}), 404

    data = request.get_json(silent=True) or {}
    if "display_name" in data:
        name = (data.get("display_name") or "").strip()
        if name:
            row.display_name = name[:255]
    if "user_note" in data:
        note = (data.get("user_note") or "").strip()
        row.user_note = note or None

    db.session.commit()
    return jsonify({"success": True, "opponent": _tracked_to_dict(row)})


@fencer_bp.delete("/api/fencers/tracked/<int:tracked_id>")
def api_delete_tracked(tracked_id: int):
    uid = _require_user_id()
    if not uid:
        return jsonify({"success": False, "error": "Authentication required"}), 401

    row = TrackedFencer.query.filter_by(id=tracked_id, user_id=uid).first()
    if not row:
        return jsonify({"success": False, "error": "Not found"}), 404

    db.session.delete(row)
    db.session.commit()
    return jsonify({"success": True})


def register_fencer_routes(app):
    app.register_blueprint(fencer_bp)
