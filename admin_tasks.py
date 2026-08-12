"""
Low-priority admin background work.

Runs in a daemon thread so gunicorn request handlers return quickly and
yields when the analyzer worker is busy so SQLite stays available.
"""

import logging
import os
import queue
import threading
import time

from sqlalchemy import text

from models import db

logger = logging.getLogger(__name__)

_task_queue = queue.Queue()
_worker_lock = threading.Lock()
_worker_started = False


def is_analyzer_busy():
    try:
        from job_queue_worker import get_current_job_id

        return get_current_job_id() is not None
    except Exception:
        return False


def _yield_to_analyzer():
    """Pause admin work while a video is being analyzed."""
    while is_analyzer_busy():
        time.sleep(float(os.environ.get("ADMIN_YIELD_SLEEP_SEC", "0.5")))


def start_admin_task_worker(app):
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        thread = threading.Thread(
            target=_admin_task_worker,
            args=(app,),
            name="admin-tasks",
            daemon=True,
        )
        thread.start()
        _worker_started = True


def enqueue_admin_task(app, fn, *args, **kwargs):
    start_admin_task_worker(app)
    _task_queue.put((fn, args, kwargs))


def _admin_task_worker(app):
    while True:
        fn, args, kwargs = _task_queue.get()
        try:
            with app.app_context():
                fn(*args, **kwargs)
        except Exception:
            logger.exception("Admin background task failed")
        finally:
            _task_queue.task_done()


def cleanup_empty_users_batched():
    """Delete users with no jobs and no email in small batches."""
    batch_size = int(os.environ.get("ADMIN_CLEANUP_USER_BATCH", "25"))
    sleep_ms = int(os.environ.get("ADMIN_CLEANUP_SLEEP_MS", "300"))
    total = 0

    sql = text(
        """
        SELECT u.id
        FROM site_user u
        WHERE (u.email IS NULL OR TRIM(u.email) = '')
          AND NOT EXISTS (SELECT 1 FROM user_job j WHERE j.user_id = u.id)
        LIMIT :limit
        """
    )

    from job_queue_models import SiteUser

    while True:
        _yield_to_analyzer()
        ids = db.session.execute(sql, {"limit": batch_size}).scalars().all()
        if not ids:
            break

        deleted = (
            SiteUser.query.filter(SiteUser.id.in_(list(ids)))
            .delete(synchronize_session=False)
        )
        db.session.commit()
        total += deleted

        if deleted <= 0:
            break
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)

    logger.info("Admin background cleanup: deleted %s empty user(s)", total)


def delete_job_r2_objects(job_id, r2_object_key):
    """Slow R2 deletes — run off the request thread."""
    _yield_to_analyzer()

    from demo import r2_client
    from flask import current_app

    s3, bucket = r2_client(current_app)

    if r2_object_key:
        try:
            s3.delete_object(Bucket=bucket, Key=r2_object_key)
        except Exception as e:
            logger.warning("Could not delete video %s: %s", r2_object_key, e)

    try:
        results_key = f"results/{job_id}_results.json"
        s3.delete_object(Bucket=bucket, Key=results_key)
    except Exception as e:
        logger.warning("Could not delete results for %s: %s", job_id, e)

    logger.info("Admin background: R2 objects removed for job %s", job_id)


def purge_user_completely(user_id):
    """Delete one user and all related data (DB now, R2 in background)."""
    from admin_user_util import delete_user_and_all_data

    def _enqueue(job_id, object_key):
        delete_job_r2_objects(job_id, object_key)

    delete_user_and_all_data(user_id, enqueue_r2_cleanup=_enqueue)
