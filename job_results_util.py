"""Load, normalize, and sync completed job results between SQLite and R2."""

import json
import logging
from datetime import datetime

from models import db

logger = logging.getLogger(__name__)


def parse_results_json(job):
    if not job or not job.results_json:
        return {}
    try:
        raw = json.loads(job.results_json)
    except (json.JSONDecodeError, TypeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def normalize_results_payload(raw, job_id):
    """Ensure canonical keys: predictions (list), 3d_results (dict), fps."""
    if not isinstance(raw, dict):
        raw = {}

    preds = raw.get("predictions")
    if preds is None and isinstance(raw.get("results"), list):
        preds = raw.get("results")
    if not isinstance(preds, list):
        preds = []

    three_d = raw.get("3d_results")
    if three_d is None:
        three_d = raw.get("three_d_results")
    if not isinstance(three_d, dict):
        three_d = {}

    from touch_ref_util import assign_touch_refs

    assign_touch_refs(preds, job_id)

    return {
        "predictions": preds,
        "3d_results": three_d,
        "fps": raw.get("fps") or 30,
        "spatial_touch_summary": raw.get("spatial_touch_summary"),
        "arm_attempts": raw.get("arm_attempts") if isinstance(raw.get("arm_attempts"), dict) else None,
    }


def load_results_from_r2(job_id):
    """Read results/{job_id}_results.json from R2, or None if missing."""
    from flask import current_app

    from demo import r2_client

    s3, bucket = r2_client(current_app)
    key = f"results/{job_id}_results.json"
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read()
        return json.loads(body)
    except Exception as e:
        logger.info("No R2 results for job %s (%s): %s", job_id, key, e)
        return None


def sync_results_to_r2(job, payload):
    """Upload a snapshot JSON for email/admin ?data= links."""
    if not payload or not payload.get("predictions"):
        return False

    from email_routes import upload_json_to_r2
    from r2_urls import video_playback_url

    json_data = {
        "video_url": video_playback_url(job.r2_object_key) or "",
        "job_id": job.job_id,
        "fps": payload.get("fps") or 30,
        "predictions": payload["predictions"],
        "three_d_results": payload.get("3d_results") or {},
        "generated_at": datetime.utcnow().isoformat(),
    }
    if payload.get("spatial_touch_summary"):
        json_data["spatial_touch_summary"] = payload["spatial_touch_summary"]
    if payload.get("arm_attempts"):
        json_data["arm_attempts"] = payload["arm_attempts"]

    upload_json_to_r2(json_data, job.job_id)
    return True


def ensure_job_results_complete(job, *, sync_r2=True):
    """
    Make sure job.results_json has predictions + 3d_results.

    Merges from the R2 snapshot when the DB row is missing data, persists touch
    refs, and optionally re-uploads R2 so ?video=&data= links stay valid after
    admin reassignment.
    """
    raw = parse_results_json(job)
    normalized = normalize_results_payload(raw, job.job_id)

    needs_r2 = (
        not normalized["predictions"]
        or not normalized["3d_results"]
    )
    if needs_r2:
        r2_raw = load_results_from_r2(job.job_id)
        if r2_raw:
            r2_norm = normalize_results_payload(r2_raw, job.job_id)
            if not normalized["predictions"] and r2_norm["predictions"]:
                normalized["predictions"] = r2_norm["predictions"]
            if not normalized["3d_results"] and r2_norm["3d_results"]:
                normalized["3d_results"] = r2_norm["3d_results"]
            if not raw.get("fps") and r2_norm.get("fps"):
                normalized["fps"] = r2_norm["fps"]
            if not normalized.get("spatial_touch_summary") and r2_norm.get(
                "spatial_touch_summary"
            ):
                normalized["spatial_touch_summary"] = r2_norm[
                    "spatial_touch_summary"
                ]

    stored = {
        "predictions": normalized["predictions"],
        "3d_results": normalized["3d_results"],
        "fps": normalized["fps"],
    }
    if normalized.get("spatial_touch_summary"):
        stored["spatial_touch_summary"] = normalized["spatial_touch_summary"]

    job.results_json = json.dumps(stored)
    db.session.flush()

    if sync_r2 and normalized["predictions"]:
        try:
            sync_results_to_r2(job, normalized)
        except Exception as e:
            logger.warning("Could not sync R2 results for job %s: %s", job.job_id, e)

    return normalized
