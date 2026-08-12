# job_queue_worker.py
"""
Background worker that processes jobs from the queue sequentially.
Only one job runs at a time (single GPU constraint).
"""

import os
import json
import time
import logging
import threading
from datetime import datetime

from models import db

from workspace_paths import OUTPUT_2D, OUTPUT_3D, tmp_path

# Will be set by register_queue()
_app = None
_worker_thread = None
_shutdown_flag = threading.Event()
# Job currently being processed (so status API can return state only to that client)
_current_job_id = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log(msg):
    """Log with timestamp."""
    logger.info(f"[QUEUE] {msg}")


class QueueBusyError(Exception):
    """Raised when the processing queue is at capacity."""

    def __init__(self, queue_length, max_length):
        self.queue_length = queue_length
        self.max_length = max_length
        super().__init__(
            "Server is busy. The processing queue is full. Please try again later."
        )


def process_next_job():
    """
    Process the next job in the queue.
    Returns True if a job was processed, False if queue is empty.
    """
    from job_queue_models import UserJob
    from demo import pipeline_runner, clipper_pipeline_runner, r2_client, pipeline_state, log_queue

    with _app.app_context():
        # Get the oldest queued job
        job = UserJob.query.filter_by(status='queued').order_by(UserJob.queued_at.asc()).first()

        if not job:
            return False

        log(f"Processing job {job.job_id} for user {job.user_id[:8]}...")

        global _current_job_id
        _current_job_id = job.job_id
        local_video_path = tmp_path(f"{job.job_id}.mp4")

        try:
            # Update status to processing
            job.status = 'processing'
            job.started_at = datetime.utcnow()
            db.session.commit()

            # Reset pipeline state (for this job only; global state is single-job)
            pipeline_state['current_step'] = 'queued'
            pipeline_state['error'] = None
            pipeline_state['results'] = []
            pipeline_state['3d_results'] = None
            pipeline_state['3d_results_durable'] = None
            pipeline_state['spatial_touch_summary'] = None
            pipeline_state['spatial_touch_summary_durable'] = None
            pipeline_state['highlight_reel_key'] = None

            # Clear log queue
            while not log_queue.empty():
                try:
                    log_queue.get_nowait()
                except:
                    break

            # Load selections
            selections = json.loads(job.selections_json) if job.selections_json else None

            job_type = getattr(job, 'job_type', 'analysis') or 'analysis'

            # Run the pipeline (blocking)
            if job_type == 'clipper':
                clipper_pipeline_runner(
                    _app, job.r2_object_key, local_video_path, selections, job.job_id
                )
            else:
                pipeline_runner(_app, job.r2_object_key, local_video_path, selections)

            # Check if pipeline completed successfully
            if pipeline_state.get('current_step') == 'complete':
                job.status = 'complete'
                job.completed_at = datetime.utcnow()

                from touch_ref_util import assign_touch_refs

                if job_type == 'clipper':
                    # Touch detection + highlight reel only (no 3D/prediction)
                    predictions = pipeline_state.get('results', [])
                    assign_touch_refs(predictions, job.job_id)
                    results = {
                        'predictions': predictions,
                        '3d_results': None,
                        'fps': pipeline_state.get('fps', 30),
                        'spatial_touch_summary': None,
                    }
                    job.highlight_reel_key = pipeline_state.get('highlight_reel_key')
                    job.results_json = json.dumps(results, allow_nan=False)
                    log(f"Clipper job {job.job_id} completed successfully!")
                else:
                    # Prefer durable copies: UI status polling pops "3d_results" for
                    # one-shot delivery and can race the worker save.
                    three_d = (
                        pipeline_state.get('3d_results_durable')
                        or pipeline_state.get('3d_results')
                    )
                    spatial = (
                        pipeline_state.get('spatial_touch_summary_durable')
                        or pipeline_state.get('spatial_touch_summary')
                    )
                    results = {
                        'predictions': pipeline_state.get('results', []),
                        '3d_results': three_d,
                        'fps': pipeline_state.get('fps', 30),
                        'spatial_touch_summary': spatial,
                        'arm_attempts': (
                            pipeline_state.get('arm_attempts_durable')
                            or pipeline_state.get('arm_attempts')
                        ),
                    }
                    if not three_d:
                        log(
                            f"WARNING: job {job.job_id} completed with empty 3d_results "
                            "(status poll may have raced, or lifting produced nothing)"
                        )
                    assign_touch_refs(results['predictions'], job.job_id)
                    # allow_nan=False: NaN in analysis made browser JSON.parse fail,
                    # so THREE_D_CACHE stayed empty and the viewer showed one origin dot.
                    job.results_json = json.dumps(results, allow_nan=False)

                    log(f"Job {job.job_id} completed successfully!")

                    # Best-effort OpenAI coaching when the user identified themselves.
                    try:
                        from results_merge import merge_results_payload, parse_user_fencer
                        from llm_bout_analysis import run_bout_llm_analysis

                        user_fencer = parse_user_fencer(job.selections_json)
                        if user_fencer and not getattr(job, "llm_analysis_json", None):
                            merged = merge_results_payload(
                                results,
                                job.prediction_corrections_json,
                                job.touch_deletions_json,
                                getattr(job, "macro_corrections_json", None),
                                job.selections_json,
                            )
                            stored_llm = run_bout_llm_analysis(merged, user_fencer)
                            job.llm_analysis_json = json.dumps(stored_llm)
                            log(f"LLM analysis stored for job {job.job_id}")
                    except Exception as e:
                        log(f"LLM analysis skipped for job {job.job_id}: {e}")

                    # Send email notification if user has email
                    try:
                        send_completion_email(job)
                    except Exception as e:
                        log(f"Failed to send email for job {job.job_id}: {e}")

            elif pipeline_state.get('current_step') == 'error':
                job.status = 'error'
                job.error_message = pipeline_state.get('error', 'Unknown error')
                job.completed_at = datetime.utcnow()
                log(f"Job {job.job_id} failed: {job.error_message}")

            else:
                # Unexpected state
                job.status = 'error'
                job.error_message = f"Unexpected state: {pipeline_state.get('current_step')}"
                job.completed_at = datetime.utcnow()

            db.session.commit()

            # Cleanup: remove local video and this job's output dirs (don't wipe shared dirs)
            try:
                if os.path.exists(local_video_path):
                    os.remove(local_video_path)
            except Exception:
                pass
            try:
                output_2d = _app.config.get("OUTPUT_2D", OUTPUT_2D)
                output_3d = _app.config.get("OUTPUT_3D", OUTPUT_3D)
                for d in [os.path.join(output_2d, job.job_id), os.path.join(output_3d, job.job_id)]:
                    if os.path.isdir(d):
                        import shutil
                        shutil.rmtree(d, ignore_errors=True)
            except Exception as e:
                log(f"Cleanup output dirs for {job.job_id}: {e}")

            _current_job_id = None
            return True

        except Exception as e:
            log(f"Job {job.job_id} error: {e}")
            import traceback
            traceback.print_exc()

            job.status = 'error'
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            db.session.commit()

            try:
                if os.path.exists(local_video_path):
                    os.remove(local_video_path)
            except Exception:
                pass
            try:
                import shutil
                output_2d = _app.config.get("OUTPUT_2D", OUTPUT_2D)
                output_3d = _app.config.get("OUTPUT_3D", OUTPUT_3D)
                for d in [os.path.join(output_2d, job.job_id), os.path.join(output_3d, job.job_id)]:
                    if os.path.isdir(d):
                        shutil.rmtree(d, ignore_errors=True)
            except Exception as cleanup_err:
                log(f"Cleanup output dirs for {job.job_id}: {cleanup_err}")

            _current_job_id = None
            return True


def get_current_job_id():
    """Return the job_id currently being processed, or None. Used so /api/status only returns state to that job's client."""
    return _current_job_id


def send_completion_email(job):
    """Send email notification when job completes."""
    from job_queue_models import SiteUser

    user = SiteUser.query.get(job.user_id)
    if not user or not user.email:
        log(f"No email for user {job.user_id[:8]}, skipping notification")
        return

    if job.email_sent:
        log(f"Email already sent for job {job.job_id}")
        return

    # Only send email for completed jobs
    if job.status != 'complete':
        return

    try:
        # Import email functionality
        from email_routes import send_results_email_internal
        from r2_urls import video_playback_url

        # Load results
        results = json.loads(job.results_json) if job.results_json else {}

        video_url = video_playback_url(job.r2_object_key)
        if not video_url and job.r2_object_key:
            log(
                "No video URL for email: set R2_PUBLIC_URL to your R2 portal public host "
                "(e.g. https://pub-xxxxxxxx.r2.dev)"
            )

        # Send email
        success, message = send_results_email_internal(
            app=_app,
            email=user.email,
            video_url=video_url,
            job_id=job.job_id,
            predictions=results.get('predictions', []),
            three_d_results=results.get('3d_results'),
            fps=results.get('fps', 30)
        )

        if success:
            job.email_sent = True
            db.session.commit()
            log(f"Email sent to {user.email} for job {job.job_id}")
        else:
            log(f"Failed to send completion email for {job.job_id}: {message}")

    except Exception as e:
        log(f"Failed to send completion email: {e}")


def _maybe_run_idle_user_cleanup(last_cleanup_at):
    """
    Delete a few anonymous users when the queue is idle.
    Returns updated last_cleanup_at timestamp.
    """
    if os.environ.get("USER_CLEANUP_ENABLED", "1") != "1":
        return last_cleanup_at

    interval = int(os.environ.get("USER_CLEANUP_INTERVAL_SEC", "3600"))
    if time.time() - last_cleanup_at < interval:
        return last_cleanup_at

    if _current_job_id is not None:
        return last_cleanup_at

    try:
        from user_cleanup import cleanup_anonymous_users

        with _app.app_context():
            deleted = cleanup_anonymous_users()
        if deleted:
            log(f"Idle user cleanup: deleted {deleted} anonymous user(s)")
    except Exception as e:
        log(f"Idle user cleanup error: {e}")

    return time.time()


def worker_loop():
    """Main worker loop - runs continuously."""
    log("Worker started")
    last_user_cleanup_at = 0.0

    while not _shutdown_flag.is_set():
        try:
            # Process next job
            processed = process_next_job()

            if not processed:
                # Queue empty - sleep before checking again
                time.sleep(2)
                last_user_cleanup_at = _maybe_run_idle_user_cleanup(last_user_cleanup_at)
            else:
                # Small delay between jobs
                time.sleep(1)

        except Exception as e:
            log(f"Worker error: {e}")
            time.sleep(5)  # Longer sleep on error

    log("Worker stopped")


def start_worker():
    """Start the background worker thread."""
    global _worker_thread

    if _worker_thread and _worker_thread.is_alive():
        log("Worker already running")
        return

    _shutdown_flag.clear()
    _worker_thread = threading.Thread(target=worker_loop, daemon=True)
    _worker_thread.start()
    log("Worker thread started")


def stop_worker():
    """Stop the background worker thread."""
    global _worker_thread

    if not _worker_thread:
        return

    _shutdown_flag.set()
    _worker_thread.join(timeout=10)
    _worker_thread = None
    log("Worker thread stopped")


def add_to_queue(job_id, selections):
    """
    Add a job to the queue.
    Returns queue position and estimated wait time.
    """
    from job_queue_models import UserJob, get_queue_length, is_queue_full, queue_max_length
    from demo import r2_client

    with _app.app_context():
        job = UserJob.query.filter_by(job_id=job_id).first()
        if not job:
            raise ValueError(f"Job {job_id} not found")

        if job.status in ("queued", "processing"):
            position = UserJob.get_queue_position(job_id)
            wait_time = UserJob.get_estimated_wait_time(job_id)
            return {
                "position": position,
                "estimated_wait_minutes": wait_time,
            }

        if is_queue_full():
            raise QueueBusyError(get_queue_length(), queue_max_length())

        # Get file size from R2 if not already set
        if not job.file_size and job.r2_object_key:
            try:
                s3, bucket = r2_client(_app)
                response = s3.head_object(Bucket=bucket, Key=job.r2_object_key)
                job.file_size = response.get('ContentLength', 0)
                log(f"File size for {job_id}: {job.file_size / (1024*1024):.1f} MB")
            except Exception as e:
                log(f"Could not get file size for {job_id}: {e}")

        # Store selections
        job.selections_json = json.dumps(selections) if selections else None

        # Update status
        job.status = 'queued'
        job.queued_at = datetime.utcnow()

        db.session.commit()

        # Calculate position
        position = UserJob.get_queue_position(job_id)
        wait_time = UserJob.get_estimated_wait_time(job_id)

        log(f"Job {job_id} added to queue at position {position}")

        return {
            'position': position,
            'estimated_wait_minutes': wait_time
        }


def get_job_status(job_id, user_id=None):
    """Get current status of a job. If user_id is set, hide jobs not owned by that user."""
    from job_queue_models import UserJob
    from results_merge import merge_results_payload

    with _app.app_context():
        job = UserJob.query.filter_by(job_id=job_id).first()
        if not job:
            return None
        if user_id is not None and job.user_id != user_id:
            return None

        result = job.to_dict()

        # Add queue position if queued
        if job.status in ['queued', 'processing']:
            result['queue_position'] = UserJob.get_queue_position(job_id)
            if result['queue_position']:
                result['estimated_wait_minutes'] = UserJob.get_estimated_wait_time(job_id)

        # Add results if complete (merge user touch corrections)
        if job.status == 'complete' and job.results_json:
            raw = json.loads(job.results_json)
            merged = merge_results_payload(
                raw,
                job.prediction_corrections_json,
                job.touch_deletions_json,
                getattr(job, "macro_corrections_json", None),
                job.selections_json,
            )
            merged['job_type'] = getattr(job, 'job_type', 'analysis') or 'analysis'
            if getattr(job, 'highlight_reel_key', None):
                from r2_urls import video_playback_url
                merged['highlight_reel_url'] = video_playback_url(job.highlight_reel_key)
            try:
                from llm_bout_analysis import (
                    parse_stored_llm_analysis,
                    llm_analysis_is_stale,
                )
                llm_stored = parse_stored_llm_analysis(
                    getattr(job, "llm_analysis_json", None)
                )
                merged["llm_analysis"] = llm_stored
                merged["llm_analysis_stale"] = (
                    llm_analysis_is_stale(llm_stored, merged)
                    if merged.get("user_fencer")
                    and merged.get("job_type") == "analysis"
                    else False
                )
            except Exception:
                merged["llm_analysis"] = None
                merged["llm_analysis_stale"] = False
            result['results'] = merged

        return result


def register_queue(app):
    """
    Initialize the queue system.
    Call this from app.py after creating the app.
    """
    global _app
    _app = app

    with app.app_context():
        if os.environ.get("STALE_JOB_CLEANUP_ON_STARTUP", "1") == "1":
            try:
                from job_cleanup import cleanup_all_stale_queue_jobs_on_startup

                removed = cleanup_all_stale_queue_jobs_on_startup(flask_app=app)
                if removed:
                    log(
                        f"Startup queue cleanup: removed {removed} job(s) "
                        f"older than {os.environ.get('STALE_JOB_STARTUP_HOURS', os.environ.get('STALE_JOB_HOURS', '48'))} hours"
                    )
            except Exception as e:
                log(f"Startup queue cleanup error: {e}")

        from queue_state import requeue_orphaned_processing_on_startup

        requeued = requeue_orphaned_processing_on_startup()
        if requeued:
            log(f"Re-queued {requeued} interrupted processing job(s) after startup")

        if os.environ.get("MODEL_WARMUP_ON_STARTUP", "1") == "1":
            try:
                from fencing_inference import ensure_pose_stack

                ensure_pose_stack(log)
                log("Analyzer models warmed up (RTMDet + ViTPose-H)")
            except Exception as e:
                log(f"Model warmup error: {e}")

    # Start the worker
    start_worker()

    log("Queue system initialized")
