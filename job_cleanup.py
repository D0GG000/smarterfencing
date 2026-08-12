"""
Queue job cleanup on startup and optional manual runs.

While the server is running, queue jobs are kept indefinitely. On startup,
jobs older than the startup threshold (default 48 hours) are removed; the
worker then processes remaining queued jobs.
"""

import os
import time
import logging
from datetime import datetime, timedelta

from sqlalchemy import text

from models import db

logger = logging.getLogger(__name__)

CLEANED_COUNTER_KEY = "stale_queued_jobs_cleaned_total"

_STALE_JOB_SQL = """
    SELECT id
    FROM user_job
    WHERE (
        (status IN ('pending', 'queued')
         AND COALESCE(queued_at, created_at) < :cutoff)
        OR (status = 'processing'
            AND COALESCE(started_at, queued_at, created_at) < :cutoff)
    )
    ORDER BY COALESCE(queued_at, created_at) ASC
    LIMIT :limit
"""

_STALE_JOB_COUNT_SQL = """
    SELECT COUNT(*) FROM user_job
    WHERE (
        (status IN ('pending', 'queued')
         AND COALESCE(queued_at, created_at) < :cutoff)
        OR (status = 'processing'
            AND COALESCE(started_at, queued_at, created_at) < :cutoff)
    )
"""


def _stale_hours():
    return int(os.environ.get("STALE_JOB_HOURS", "48"))


def _startup_stale_hours():
    return int(os.environ.get("STALE_JOB_STARTUP_HOURS", os.environ.get("STALE_JOB_HOURS", "48")))


def _age_cutoff(hours=None, days=None):
    if days is not None:
        return datetime.utcnow() - timedelta(days=days)
    hours = _stale_hours() if hours is None else hours
    return datetime.utcnow() - timedelta(hours=hours)


def count_stale_queued_jobs(hours=None, days=None):
    """Return pending/queued/processing jobs older than the age threshold."""
    cutoff = _age_cutoff(hours=hours, days=days)
    row = db.session.execute(
        text(_STALE_JOB_COUNT_SQL),
        {"cutoff": cutoff},
    ).scalar()
    return int(row or 0)


def get_cleaned_total():
    row = db.session.execute(
        text("SELECT value FROM app_setting WHERE key = :key"),
        {"key": CLEANED_COUNTER_KEY},
    ).scalar()
    try:
        return int(row or 0)
    except (TypeError, ValueError):
        return 0


def record_cleaned(count):
    if count <= 0:
        return get_cleaned_total()

    db.session.execute(
        text(
            """
            INSERT INTO app_setting (key, value) VALUES (:key, :value)
            ON CONFLICT(key) DO UPDATE SET
                value = CAST(CAST(app_setting.value AS INTEGER) + :delta AS TEXT)
            """
        ),
        {"key": CLEANED_COUNTER_KEY, "value": str(count), "delta": count},
    )
    db.session.commit()
    return get_cleaned_total()


def reset_cleaned_total():
    db.session.execute(
        text(
            """
            INSERT INTO app_setting (key, value) VALUES (:key, '0')
            ON CONFLICT(key) DO UPDATE SET value = '0'
            """
        ),
        {"key": CLEANED_COUNTER_KEY},
    )
    db.session.commit()
    return 0


def get_stale_job_cleanup_stats():
    return {
        "stale_jobs_cleaned_total": get_cleaned_total(),
        "startup_stale_hours": _startup_stale_hours(),
    }


def _delete_job_from_r2(job, s3, bucket):
    if job.r2_object_key:
        try:
            s3.delete_object(Bucket=bucket, Key=job.r2_object_key)
        except Exception as e:
            logger.warning("Could not delete video %s: %s", job.r2_object_key, e)

    try:
        results_key = f"results/{job.job_id}_results.json"
        s3.delete_object(Bucket=bucket, Key=results_key)
    except Exception as e:
        logger.warning("Could not delete results for %s: %s", job.job_id, e)


def cleanup_stale_queued_jobs(
    hours=None,
    days=None,
    batch_size=None,
    max_batches=None,
    sleep_ms=None,
    dry_run=False,
    flask_app=None,
):
    """
    Delete stale pending/queued/processing jobs and their R2 objects in batches.

    Returns:
        int: Total jobs removed (or counted in dry_run).
    """
    from job_queue_models import UserJob

    batch_size = batch_size or int(os.environ.get("STALE_JOB_CLEANUP_BATCH_SIZE", "5"))
    max_batches = max_batches or int(os.environ.get("STALE_JOB_CLEANUP_MAX_BATCHES", "2"))
    sleep_ms = sleep_ms or int(os.environ.get("STALE_JOB_CLEANUP_SLEEP_MS", "500"))

    batch_size = max(1, min(batch_size, 50))
    max_batches = max(1, min(max_batches, 200))
    sleep_ms = max(0, min(sleep_ms, 10000))

    cutoff = _age_cutoff(hours=hours, days=days)
    total = 0
    s3 = bucket = None

    if not dry_run:
        if flask_app is None:
            raise ValueError("flask_app is required when not dry_run")
        from demo import r2_client

        s3, bucket = r2_client(flask_app)

    for batch_num in range(max_batches):
        ids = db.session.execute(
            text(_STALE_JOB_SQL),
            {"cutoff": cutoff, "limit": batch_size},
        ).scalars().all()

        if not ids:
            break

        jobs = UserJob.query.filter(UserJob.id.in_(list(ids))).all()
        if not jobs:
            break

        if dry_run:
            total += len(jobs)
            logger.info(
                "Dry run batch %s: would delete %s stale job(s)",
                batch_num + 1,
                len(jobs),
            )
            break

        for job in jobs:
            _delete_job_from_r2(job, s3, bucket)
            db.session.delete(job)
            total += 1

        db.session.commit()
        logger.info(
            "Stale job cleanup batch %s: removed %s job(s), %s total this run",
            batch_num + 1,
            len(jobs),
            total,
        )

        if batch_num + 1 < max_batches and sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)

    if total > 0 and not dry_run:
        record_cleaned(total)

    return total


def cleanup_all_stale_queue_jobs_on_startup(flask_app):
    """
    Remove every queue job older than the startup threshold (default 48 hours).

    Loops until none remain so the worker can process younger queued jobs.
    """
    hours = _startup_stale_hours()
    max_rounds = int(os.environ.get("STALE_JOB_STARTUP_MAX_ROUNDS", "500"))
    batch_size = int(os.environ.get("STALE_JOB_STARTUP_BATCH_SIZE", "20"))
    max_batches = int(os.environ.get("STALE_JOB_STARTUP_MAX_BATCHES", "50"))
    sleep_ms = int(os.environ.get("STALE_JOB_STARTUP_SLEEP_MS", "200"))
    total = 0

    for round_num in range(max_rounds):
        remaining = count_stale_queued_jobs(hours=hours)
        if remaining == 0:
            break

        removed = cleanup_stale_queued_jobs(
            hours=hours,
            batch_size=batch_size,
            max_batches=max_batches,
            sleep_ms=sleep_ms,
            flask_app=flask_app,
        )
        total += removed
        if removed == 0:
            logger.warning(
                "Startup queue cleanup stalled with %s job(s) older than %s hour(s)",
                remaining,
                hours,
            )
            break

        logger.info(
            "Startup queue cleanup round %s: removed %s job(s), %s remaining (>=%s hours old)",
            round_num + 1,
            removed,
            count_stale_queued_jobs(hours=hours),
            hours,
        )

    return total
