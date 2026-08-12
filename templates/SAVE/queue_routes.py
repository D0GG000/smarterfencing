# queue_routes.py
"""
API endpoints for job queue management.
"""

import json
from datetime import datetime
from flask import Blueprint, request, jsonify, g

from models import db
from job_queue_models import UserJob, SiteUser, get_queue_stats
from job_queue_worker import add_to_queue, get_job_status

queue_bp = Blueprint('queue', __name__)


@queue_bp.route('/api/queue/submit', methods=['POST'])
def submit_to_queue():
    """
    Submit a job to the processing queue.
    Replaces the old /api/run-pipeline endpoint.
    """
    data = request.get_json(silent=True) or {}

    job_id = data.get('job_id')
    object_key = data.get('video')
    selections = data.get('selections')

    if not job_id:
        return jsonify({'success': False, 'error': 'No job_id specified'}), 400
    if not object_key:
        return jsonify({'success': False, 'error': 'No video specified'}), 400

    # Check if job exists
    job = UserJob.query.filter_by(job_id=job_id).first()
    if not job:
        return jsonify({'success': False, 'error': 'Job not found'}), 404

    # Check if job is already in queue or processing
    if job.status in ['queued', 'processing']:
        position = UserJob.get_queue_position(job_id)
        return jsonify({
            'success': True,
            'message': 'Job already in queue',
            'status': job.status,
            'queue_position': position,
            'estimated_wait_minutes': UserJob.get_estimated_wait_time(position) if position else None
        })

    # Check if job is already complete
    if job.status == 'complete':
        return jsonify({
            'success': True,
            'message': 'Job already completed',
            'status': 'complete'
        })

    try:
        # Add to queue
        queue_info = add_to_queue(job_id, selections)

        return jsonify({
            'success': True,
            'message': 'Job added to queue',
            'status': 'queued',
            'queue_position': queue_info['position'],
            'estimated_wait_minutes': queue_info['estimated_wait_minutes']
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@queue_bp.route('/api/queue/status/<job_id>', methods=['GET'])
def queue_status(job_id):
    """
    Get the status of a specific job.
    """
    status = get_job_status(job_id)

    if not status:
        return jsonify({'success': False, 'error': 'Job not found'}), 404

    return jsonify({
        'success': True,
        **status
    })


@queue_bp.route('/api/queue/stats', methods=['GET'])
def queue_stats():
    """
    Get overall queue statistics.
    """
    stats = get_queue_stats()

    return jsonify({
        'success': True,
        **stats
    })


@queue_bp.route('/api/queue/my-jobs', methods=['GET'])
def my_jobs():
    """
    Get all jobs for the current user.
    """
    user_id = g.user_id

    jobs = UserJob.query.filter_by(user_id=user_id).order_by(UserJob.created_at.desc()).all()

    return jsonify({
        'success': True,
        'jobs': [job.to_dict() for job in jobs]
    })


@queue_bp.route('/api/queue/results/<job_id>', methods=['GET'])
def get_results(job_id):
    """
    Get the results of a completed job.
    """
    job = UserJob.query.filter_by(job_id=job_id).first()

    if not job:
        return jsonify({'success': False, 'error': 'Job not found'}), 404

    if job.status != 'complete':
        return jsonify({
            'success': False,
            'error': f'Job not complete. Current status: {job.status}',
            'status': job.status
        }), 400

    try:
        results = json.loads(job.results_json) if job.results_json else {}
        return jsonify({
            'success': True,
            'job_id': job_id,
            'results': results
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@queue_bp.route('/api/queue/cancel/<job_id>', methods=['POST'])
def cancel_job(job_id):
    """
    Cancel a queued job (cannot cancel processing jobs).
    """
    user_id = g.user_id

    job = UserJob.query.filter_by(job_id=job_id, user_id=user_id).first()

    if not job:
        return jsonify({'success': False, 'error': 'Job not found'}), 404

    if job.status == 'processing':
        return jsonify({'success': False, 'error': 'Cannot cancel a job that is currently processing'}), 400

    if job.status == 'complete':
        return jsonify({'success': False, 'error': 'Job already completed'}), 400

    if job.status == 'queued':
        job.status = 'cancelled'
        job.completed_at = datetime.utcnow()
        db.session.commit()
        return jsonify({'success': True, 'message': 'Job cancelled'})

    return jsonify({'success': False, 'error': f'Cannot cancel job with status: {job.status}'}), 400


def register_queue_routes(app):
    """Register queue routes with the app."""
    app.register_blueprint(queue_bp)
