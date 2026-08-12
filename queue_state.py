"""Live queue state vs DB status (reconcile after pod restarts)."""

from models import db


def job_reference_time(job):
    """Timestamp used for queue ordering and display."""
    if job.status == "processing":
        return job.started_at or job.queued_at or job.created_at
    return job.queued_at or job.created_at


def _job_admin_dict(job):
    ref = job_reference_time(job)
    return {
        "job_id": job.job_id,
        "user_email": job.user.email if job.user else None,
        "filename": job.filename,
        "file_size": job.file_size,
        "object_key": job.r2_object_key,
        "status": job.status,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "queued_at": job.queued_at.isoformat() if job.queued_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "waiting_since": ref.isoformat() if ref else None,
    }


def requeue_orphaned_processing_on_startup():
    """
    After a pod restart nothing is actually processing.
    Put leftover processing rows back on the queue so the worker can run them.
    """
    from job_queue_models import UserJob

    orphans = UserJob.query.filter_by(status="processing").all()
    if not orphans:
        return 0

    for job in orphans:
        job.status = "queued"
        job.started_at = None
        job.error_message = None

    db.session.commit()
    return len(orphans)


def reconcile_orphaned_processing_on_startup():
    """Backward-compatible alias for requeue_orphaned_processing_on_startup."""
    return requeue_orphaned_processing_on_startup()


def get_admin_queue_snapshot():
    """Admin view: live worker state + full waiting queue."""
    from job_queue_models import UserJob
    from job_queue_worker import get_current_job_id

    worker_job_id = get_current_job_id()

    candidates = (
        UserJob.query.filter(UserJob.status.in_(["pending", "queued", "processing"]))
        .order_by(UserJob.queued_at.asc().nullslast(), UserJob.created_at.asc())
        .all()
    )

    live_processing = None
    orphan_processing = []
    active_queue = []

    for job in candidates:
        row = _job_admin_dict(job)

        if job.status == "processing":
            if worker_job_id and job.job_id == worker_job_id:
                live_processing = row
            else:
                orphan_processing.append(row)
            continue

        if job.status == "pending":
            row["status_note"] = "pending_upload"
        active_queue.append(row)

    return {
        "worker_busy": worker_job_id is not None,
        "worker_current_job_id": worker_job_id,
        "live_processing": live_processing,
        "orphan_processing": orphan_processing,
        "active_queue": active_queue,
    }
