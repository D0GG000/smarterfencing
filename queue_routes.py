# queue_routes.py
"""
API endpoints for job queue management.
"""

import json
import secrets
from datetime import datetime
from flask import Blueprint, request, jsonify, session, current_app, url_for

from models import db
from job_queue_models import UserJob, get_queue_stats
from job_queue_worker import add_to_queue, get_job_status, QueueBusyError
from results_merge import (
    ATTACK_LABELS,
    TOUCH_LABELS,
    USER_FENCER_VALUES,
    merge_results_payload,
    normalize_correction_attack,
    normalize_correction_prediction,
    normalize_macro_corrections,
    normalize_touch_note,
    parse_user_fencer,
)
from llm_bout_analysis import (
    llm_analysis_is_stale,
    parse_stored_llm_analysis,
    run_bout_llm_analysis,
)
from r2_urls import video_playback_url

queue_bp = Blueprint("queue", __name__)


def _require_user_id():
    uid = session.get("user_id")
    if not uid:
        return None
    return uid


def _validate_selections_payload(selections, require_fencers=True):
    if not isinstance(selections, dict):
        return "selections payload is required"

    def _valid_box(d):
        if not isinstance(d, dict):
            return False
        req = ("x1", "y1", "x2", "y2")
        for k in req:
            if k not in d:
                return False
            try:
                float(d[k])
            except (TypeError, ValueError):
                return False
        return True

    # Clipper jobs build a highlight reel from scoreboard lights only — no fencer
    # boxes required.
    light_mode = str(selections.get("light_mode") or "static").strip().lower()
    if light_mode not in ("static", "tracking"):
        light_mode = "static"

    if light_mode == "tracking":
        if not _valid_box(selections.get("score_apparatus")):
            return "Invalid or missing selections.score_apparatus (tracking mode)"
    else:
        required_boxes = ["fencer1_light", "fencer2_light"]
        if require_fencers:
            required_boxes = ["fencer1", "fencer2"] + required_boxes
        for key in required_boxes:
            if not _valid_box(selections.get(key)):
                return f"Invalid or missing selections.{key} box"

    if require_fencers:
        if light_mode == "tracking":
            for key in ("fencer1", "fencer2"):
                if not _valid_box(selections.get(key)):
                    return f"Invalid or missing selections.{key} box"
        uf = selections.get("user_fencer")
        if not (
            isinstance(uf, str) and uf.strip().lower() in USER_FENCER_VALUES
        ):
            return "Invalid or missing selections.user_fencer (fencer1 or fencer2)"

    for key in ("video_width", "video_height"):
        try:
            if float(selections.get(key, 0)) <= 0:
                raise ValueError()
        except (TypeError, ValueError):
            return f"Invalid or missing selections.{key}"
    return None


@queue_bp.route("/api/queue/submit", methods=["POST"])
def submit_to_queue():
    """
    Submit a job to the processing queue.
    Replaces the old /api/run-pipeline endpoint.
    """
    uid = _require_user_id()
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
    sel_error = _validate_selections_payload(selections, require_fencers=(job_type != "clipper"))
    if sel_error:
        return jsonify({"success": False, "error": sel_error}), 400

    job = UserJob.query.filter_by(job_id=job_id, user_id=uid).first()
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404

    job.job_type = job_type
    db.session.commit()

    if job.status in ["queued", "processing"]:
        position = UserJob.get_queue_position(job_id)
        return jsonify(
            {
                "success": True,
                "message": "Job already in queue",
                "status": job.status,
                "queue_position": position,
                "estimated_wait_minutes": UserJob.get_estimated_wait_time(job_id)
                if position
                else None,
            }
        )

    if job.status == "complete":
        return jsonify(
            {"success": True, "message": "Job already completed", "status": "complete"}
        )

    try:
        queue_info = add_to_queue(job_id, selections)

        return jsonify(
            {
                "success": True,
                "message": "Job added to queue",
                "status": "queued",
                "queue_position": queue_info["position"],
                "estimated_wait_minutes": queue_info["estimated_wait_minutes"],
            }
        )

    except QueueBusyError as e:
        return jsonify(
            {
                "success": False,
                "error": str(e),
                "code": "queue_busy",
                "queue_length": e.queue_length,
                "queue_max_length": e.max_length,
            }
        ), 503
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@queue_bp.route("/api/queue/status/<job_id>", methods=["GET"])
def queue_status(job_id):
    """
    Get the status of a specific job.
    """
    uid = _require_user_id()
    if not uid:
        return jsonify({"success": False, "error": "Authentication required"}), 401

    status = get_job_status(job_id, user_id=uid)

    if not status:
        return jsonify({"success": False, "error": "Job not found"}), 404

    return jsonify({"success": True, **status})


@queue_bp.route("/api/queue/stats", methods=["GET"])
def queue_stats():
    """
    Get overall queue statistics.
    """
    stats = get_queue_stats()

    return jsonify({"success": True, **stats})


def _job_list_item(job):
    """Lightweight past-results row — never parse results_json / LLM payloads."""
    d = job.to_dict()
    d["touch_count"] = None
    d["has_share_link"] = bool(getattr(job, "share_token", None))
    return d


@queue_bp.route("/api/queue/my-jobs", methods=["GET"])
def my_jobs():
    """
    List jobs for the current user (metadata only — paginated).
    Query: limit (default 10, max 50), offset (default 0).
    """
    from sqlalchemy.orm import defer

    uid = _require_user_id()
    if not uid:
        return jsonify({"success": False, "error": "Authentication required"}), 401

    try:
        limit = int(request.args.get("limit", 10))
    except (TypeError, ValueError):
        limit = 10
    try:
        offset = int(request.args.get("offset", 0))
    except (TypeError, ValueError):
        offset = 0
    limit = max(1, min(limit, 50))
    offset = max(0, offset)

    # Defer huge TEXT columns so listing past results does not load/parse bout JSON.
    base = (
        UserJob.query.filter_by(user_id=uid)
        .options(
            defer(UserJob.results_json),
            defer(UserJob.llm_analysis_json),
            defer(UserJob.selections_json),
            defer(UserJob.prediction_corrections_json),
            defer(UserJob.macro_corrections_json),
            defer(UserJob.touch_deletions_json),
        )
        .order_by(UserJob.created_at.desc())
    )
    # Fetch one extra row to know if more pages exist (avoids a separate COUNT).
    rows = base.offset(offset).limit(limit + 1).all()
    has_more = len(rows) > limit
    jobs = rows[:limit]

    return jsonify(
        {
            "success": True,
            "jobs": [_job_list_item(job) for job in jobs],
            "limit": limit,
            "offset": offset,
            "has_more": has_more,
            "next_offset": offset + len(jobs) if has_more else None,
        }
    )


def _build_results_response(job):
    """Merged results dict plus video_url for clients."""
    from touch_ref_util import assign_touch_refs, predictions_missing_touch_ref
    from job_results_util import parse_results_json, normalize_results_payload

    results = parse_results_json(job)
    if job.status == "complete" and not results.get("predictions"):
        from job_results_util import ensure_job_results_complete

        results = ensure_job_results_complete(job, sync_r2=True)
        db.session.commit()
    else:
        results = normalize_results_payload(results, job.job_id)

    merged = merge_results_payload(
        results,
        job.prediction_corrections_json,
        job.touch_deletions_json,
        getattr(job, "macro_corrections_json", None),
        job.selections_json,
    )
    preds = merged.get("predictions") or []
    needs_persist = predictions_missing_touch_ref(preds)
    assign_touch_refs(preds, job.job_id)
    if needs_persist and job.results_json:
        stored = json.loads(job.results_json)
        ref_by_touch = {
            p.get("touch"): p.get("touch_ref")
            for p in preds
            if p.get("touch") and p.get("touch_ref")
        }
        for sp in stored.get("predictions") or []:
            t = sp.get("touch")
            if t in ref_by_touch:
                sp["touch_ref"] = ref_by_touch[t]
        job.results_json = json.dumps(stored)
        db.session.commit()

    video_url = video_playback_url(job.r2_object_key)
    highlight_reel_url = (
        video_playback_url(job.highlight_reel_key)
        if getattr(job, "highlight_reel_key", None)
        else None
    )
    llm_stored = parse_stored_llm_analysis(getattr(job, "llm_analysis_json", None))
    llm_stale = False
    if merged.get("user_fencer") and (getattr(job, "job_type", "analysis") or "analysis") == "analysis":
        llm_stale = llm_analysis_is_stale(llm_stored, merged)

    return {
        "predictions": preds,
        "3d_results": merged.get("3d_results"),
        "fps": merged.get("fps", 30),
        "spatial_touch_summary": merged.get("spatial_touch_summary"),
        "arm_attempts": merged.get("arm_attempts"),
        "video_url": video_url,
        "highlight_reel_url": highlight_reel_url,
        "job_type": getattr(job, "job_type", "analysis") or "analysis",
        "job_id": job.job_id,
        "deleted_touch_ids": merged.get("deleted_touch_ids") or [],
        "user_fencer": merged.get("user_fencer"),
        "macro_corrections": merged.get("macro_corrections"),
        "llm_analysis": llm_stored,
        "llm_analysis_stale": llm_stale,
    }


def _ensure_share_token(job):
    """Assign a unique share_token to job if missing; commit."""
    if getattr(job, "share_token", None):
        return job.share_token
    for _ in range(12):
        token = secrets.token_urlsafe(32)
        existing = UserJob.query.filter_by(share_token=token).first()
        if existing:
            continue
        job.share_token = token
        db.session.commit()
        return token
    raise RuntimeError("Could not allocate share token")


@queue_bp.route("/api/queue/jobs/<job_id>/share", methods=["POST"])
def create_or_get_share_link(job_id):
    """
    Owner only: ensure a read-only share token exists and return the public results URL.
    """
    uid = _require_user_id()
    if not uid:
        return jsonify({"success": False, "error": "Authentication required"}), 401

    job = UserJob.query.filter_by(job_id=job_id, user_id=uid).first()
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404
    if job.status != "complete":
        return jsonify({"success": False, "error": "Job is not complete"}), 400

    try:
        token = _ensure_share_token(job)
    except RuntimeError as e:
        return jsonify({"success": False, "error": str(e)}), 500

    base = request.host_url.rstrip("/")
    path = url_for("demo.result")
    share_url = f"{base}{path}?share={token}"
    return jsonify({"success": True, "share_url": share_url})


@queue_bp.route("/api/queue/shared-results/<token>", methods=["GET"])
def get_shared_results(token):
    """
    Public read-only: load merged results for a completed job by share_token.

    Query: lite=1 omits 3d_results and other bulky fields (homepage tour).
    """
    if not token or not isinstance(token, str) or len(token) > 128:
        return jsonify({"success": False, "error": "Invalid link"}), 400

    job = UserJob.query.filter_by(share_token=token.strip()).first()
    if not job or job.status != "complete":
        return jsonify({"success": False, "error": "Not found"}), 404

    try:
        payload = _build_results_response(job)
        lite = str(request.args.get("lite") or "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if lite:
            payload = _lite_shared_results(payload)
        return jsonify(
            {
                "success": True,
                "job_id": job.job_id,
                "results": payload,
                "read_only": True,
                "lite": lite,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def _lite_shared_results(payload: dict) -> dict:
    """Strip heavy fields for homepage / marketing embeds."""
    preds_in = payload.get("predictions") or []
    preds = []
    for p in preds_in:
        if not isinstance(p, dict):
            continue
        preds.append(
            {
                "touch": p.get("touch"),
                "touch_ref": p.get("touch_ref"),
                "prediction": p.get("prediction"),
                "attack_prediction": p.get("attack_prediction"),
            }
        )

    arm_in = payload.get("arm_attempts")
    arm = None
    if isinstance(arm_in, dict):
        pre = arm_in.get("pre_touch_aggressor")
        pre_lite = None
        if isinstance(pre, dict):
            pre_lite = {
                k: pre.get(k)
                for k in (
                    "main_footwork_aggressor",
                    "fencer1_pre_touch_aggression",
                    "fencer2_pre_touch_aggression",
                    "even",
                    "unclear",
                    "touches_scored",
                )
                if k in pre
            }
        arm = {
            "fencer1_total": arm_in.get("fencer1_total"),
            "fencer2_total": arm_in.get("fencer2_total"),
            "pre_touch_aggressor": pre_lite,
        }
        # Homepage tour draws the forward/back overlay on the bout video.
        fb = arm_in.get("forward_back")
        if isinstance(fb, dict) and isinstance(fb.get("frames"), list):
            arm["forward_back"] = fb

    llm = payload.get("llm_analysis")
    # Keep stored coaching analysis as-is (already compact vs 3d).

    return {
        "predictions": preds,
        "fps": payload.get("fps", 30),
        "video_url": payload.get("video_url"),
        "highlight_reel_url": payload.get("highlight_reel_url"),
        "job_type": payload.get("job_type"),
        "job_id": payload.get("job_id"),
        "deleted_touch_ids": payload.get("deleted_touch_ids") or [],
        "user_fencer": payload.get("user_fencer"),
        "arm_attempts": arm,
        "llm_analysis": llm,
        "llm_analysis_stale": bool(payload.get("llm_analysis_stale")),
    }


@queue_bp.route("/api/queue/results/<job_id>", methods=["GET"])
def get_results(job_id):
    """
    Get the results of a completed job (owner only).
    """
    uid = _require_user_id()
    if not uid:
        return jsonify({"success": False, "error": "Authentication required"}), 401

    job = UserJob.query.filter_by(job_id=job_id, user_id=uid).first()
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404

    if job.status != "complete":
        return jsonify(
            {
                "success": False,
                "error": f"Job not complete. Current status: {job.status}",
                "status": job.status,
            }
        ), 400

    try:
        payload = _build_results_response(job)
        return jsonify({"success": True, "job_id": job_id, "results": payload})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@queue_bp.route("/api/queue/jobs/<job_id>/prediction-corrections", methods=["PATCH"])
def patch_prediction_corrections(job_id):
    uid = _require_user_id()
    if not uid:
        return jsonify({"success": False, "error": "Authentication required"}), 401

    job = UserJob.query.filter_by(job_id=job_id, user_id=uid).first()
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404

    if job.status != "complete":
        return jsonify({"success": False, "error": "Job is not complete"}), 400

    data = request.get_json(silent=True) or {}
    touch = data.get("touch")
    has_prediction = "prediction" in data
    has_note = "note" in data
    has_attack = "attack_prediction" in data
    pred = (
        normalize_correction_prediction(data.get("prediction"))
        if has_prediction
        else None
    )
    attack = (
        normalize_correction_attack(data.get("attack_prediction"))
        if has_attack
        else None
    )

    if not touch or not isinstance(touch, str):
        return jsonify({"success": False, "error": "touch is required"}), 400
    if not has_prediction and not has_note and not has_attack:
        return jsonify(
            {
                "success": False,
                "error": "Provide prediction, attack_prediction, and/or note to save",
            }
        ), 400
    if has_prediction and not pred:
        return jsonify(
            {
                "success": False,
                "error": f"prediction must be one of: {', '.join(sorted(TOUCH_LABELS))}",
            }
        ), 400
    if has_attack and not attack:
        return jsonify(
            {
                "success": False,
                "error": f"attack_prediction must be one of: {', '.join(sorted(ATTACK_LABELS))}",
            }
        ), 400

    raw_predictions = []
    if job.results_json:
        try:
            raw_predictions = json.loads(job.results_json).get("predictions") or []
        except (json.JSONDecodeError, TypeError):
            raw_predictions = []
    if not any(p.get("touch") == touch for p in raw_predictions):
        return jsonify({"success": False, "error": "Unknown touch id for this job"}), 400

    deleted_set = set()
    if job.touch_deletions_json:
        try:
            dlist = json.loads(job.touch_deletions_json)
            if isinstance(dlist, list):
                deleted_set = {str(x) for x in dlist if isinstance(x, str)}
        except (json.JSONDecodeError, TypeError):
            pass
    if touch in deleted_set:
        return jsonify(
            {
                "success": False,
                "error": "This touch was removed from results. Restore it before editing.",
            }
        ), 400

    corrections = {}
    if job.prediction_corrections_json:
        try:
            corrections = json.loads(job.prediction_corrections_json)
        except (json.JSONDecodeError, TypeError):
            corrections = {}

    prev = corrections.get(touch)
    entry = {}
    if isinstance(prev, dict):
        if prev.get("prediction") in TOUCH_LABELS:
            entry["prediction"] = prev["prediction"]
        if prev.get("attack_prediction") in ATTACK_LABELS:
            entry["attack_prediction"] = prev["attack_prediction"]
        pn = prev.get("note")
        if isinstance(pn, str) and pn.strip():
            entry["note"] = pn.strip()

    if has_prediction:
        entry["prediction"] = pred
    if has_attack:
        entry["attack_prediction"] = attack
    if has_note:
        n = normalize_touch_note(data.get("note"))
        if n:
            entry["note"] = n
        else:
            entry.pop("note", None)

    keep = (
        (entry.get("prediction") in TOUCH_LABELS)
        or (entry.get("attack_prediction") in ATTACK_LABELS)
        or bool(isinstance(entry.get("note"), str) and entry.get("note", "").strip())
    )
    if keep:
        entry["updated_at"] = datetime.utcnow().isoformat() + "Z"
        corrections[touch] = entry
    else:
        corrections.pop(touch, None)

    job.prediction_corrections_json = json.dumps(corrections)
    db.session.commit()

    payload = _build_results_response(job)
    return jsonify({"success": True, "results": payload})


def _raw_predictions_list(job):
    if not job.results_json:
        return []
    try:
        return json.loads(job.results_json).get("predictions") or []
    except (json.JSONDecodeError, TypeError):
        return []


@queue_bp.route("/api/queue/jobs/<job_id>/touch-deletions", methods=["PATCH"])
def patch_touch_deletions(job_id):
    uid = _require_user_id()
    if not uid:
        return jsonify({"success": False, "error": "Authentication required"}), 401

    job = UserJob.query.filter_by(job_id=job_id, user_id=uid).first()
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404

    if job.status != "complete":
        return jsonify({"success": False, "error": "Job is not complete"}), 400

    data = request.get_json(silent=True) or {}
    touch = data.get("touch")
    if not touch or not isinstance(touch, str):
        return jsonify({"success": False, "error": "touch is required"}), 400
    if "deleted" not in data or not isinstance(data.get("deleted"), bool):
        return jsonify({"success": False, "error": "deleted (boolean) is required"}), 400

    raw_predictions = _raw_predictions_list(job)
    if not any(p.get("touch") == touch for p in raw_predictions):
        return jsonify({"success": False, "error": "Unknown touch id for this job"}), 400

    deleted = []
    if job.touch_deletions_json:
        try:
            parsed = json.loads(job.touch_deletions_json)
            if isinstance(parsed, list):
                deleted = [str(x) for x in parsed if isinstance(x, str)]
        except (json.JSONDecodeError, TypeError):
            deleted = []

    s = set(deleted)
    if data["deleted"]:
        s.add(touch)
        corrections = {}
        if job.prediction_corrections_json:
            try:
                corrections = json.loads(job.prediction_corrections_json)
            except (json.JSONDecodeError, TypeError):
                corrections = {}
        if isinstance(corrections, dict) and touch in corrections:
            del corrections[touch]
            job.prediction_corrections_json = (
                json.dumps(corrections) if corrections else None
            )
    else:
        s.discard(touch)

    job.touch_deletions_json = json.dumps(sorted(s)) if s else None
    db.session.commit()

    payload = _build_results_response(job)
    return jsonify({"success": True, "results": payload})


@queue_bp.route("/api/queue/jobs/<job_id>/macro-corrections", methods=["PATCH"])
def patch_macro_corrections(job_id):
    """Owner: overlay bout-level arm-attempt / pre-touch (who advanced more) rollups."""
    uid = _require_user_id()
    if not uid:
        return jsonify({"success": False, "error": "Authentication required"}), 401

    job = UserJob.query.filter_by(job_id=job_id, user_id=uid).first()
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404
    if job.status != "complete":
        return jsonify({"success": False, "error": "Job is not complete"}), 400
    if (getattr(job, "job_type", "analysis") or "analysis") != "analysis":
        return jsonify({"success": False, "error": "Macro edits are only for analysis jobs"}), 400

    data = request.get_json(silent=True) or {}
    cleaned = normalize_macro_corrections(data)
    job.macro_corrections_json = json.dumps(cleaned) if cleaned else None
    db.session.commit()

    payload = _build_results_response(job)
    return jsonify({"success": True, "results": payload})


@queue_bp.route("/api/queue/jobs/<job_id>/llm-analysis", methods=["POST"])
def post_llm_analysis(job_id):
    """Owner: run OpenAI coaching analysis from merged macrobout data."""
    uid = _require_user_id()
    if not uid:
        return jsonify({"success": False, "error": "Authentication required"}), 401

    job = UserJob.query.filter_by(job_id=job_id, user_id=uid).first()
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404
    if job.status != "complete":
        return jsonify({"success": False, "error": "Job is not complete"}), 400
    if (getattr(job, "job_type", "analysis") or "analysis") != "analysis":
        return jsonify({"success": False, "error": "LLM analysis is only for analysis jobs"}), 400

    user_fencer = parse_user_fencer(job.selections_json)
    if not user_fencer:
        return jsonify(
            {
                "success": False,
                "error": "This job has no user_fencer selection; re-run analysis with identity set.",
            }
        ), 400

    payload = _build_results_response(job)
    try:
        stored = run_bout_llm_analysis(payload, user_fencer)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"success": False, "error": str(e)}), 502

    job.llm_analysis_json = json.dumps(stored)
    db.session.commit()

    out = _build_results_response(job)
    return jsonify({"success": True, "results": out, "llm_analysis": stored})


@queue_bp.route("/api/queue/cancel/<job_id>", methods=["POST"])
def cancel_job(job_id):
    """
    Cancel a queued job (cannot cancel processing jobs).
    """
    uid = _require_user_id()
    if not uid:
        return jsonify({"success": False, "error": "Authentication required"}), 401

    job = UserJob.query.filter_by(job_id=job_id, user_id=uid).first()

    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404

    if job.status == "processing":
        return jsonify(
            {"success": False, "error": "Cannot cancel a job that is currently processing"}
        ), 400

    if job.status == "complete":
        return jsonify({"success": False, "error": "Job already completed"}), 400

    if job.status == "queued":
        job.status = "cancelled"
        job.completed_at = datetime.utcnow()
        db.session.commit()
        return jsonify({"success": True, "message": "Job cancelled"})

    return jsonify(
        {"success": False, "error": f"Cannot cancel job with status: {job.status}"}
    ), 400


def register_queue_routes(app):
    """Register queue routes with the app."""
    app.register_blueprint(queue_bp)
