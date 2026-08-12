# job_queue_models.py
"""
Extended models for job queue system.
Adds queue status tracking to existing UserJob model.
"""

from datetime import datetime
from models import db


class SiteUser(db.Model):
    """User model for tracking visitors."""
    __tablename__ = 'site_user'

    id = db.Column(db.String(36), primary_key=True)  # UUID
    email = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)

    jobs = db.relationship('UserJob', backref='user', lazy='dynamic')

    def to_dict(self):
        return {
            'id': self.id[:8] + '...',
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_seen': self.last_seen.isoformat() if self.last_seen else None
        }


class UserJob(db.Model):
    """Job model with queue status tracking."""
    __tablename__ = 'user_job'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(36), db.ForeignKey('site_user.id'), nullable=False)
    job_id = db.Column(db.String(20), unique=True, nullable=False)  # e.g., "abc123def456"

    # File info
    r2_object_key = db.Column(db.String(255), nullable=True)  # e.g., "uploads/abc123.mp4"
    filename = db.Column(db.String(255), nullable=True)  # Original filename

    # Queue status
    status = db.Column(db.String(20), default='pending')  # pending, queued, processing, complete, error
    queue_position = db.Column(db.Integer, nullable=True)  # Position when queued

    # Pipeline data (stored as JSON string)
    selections_json = db.Column(db.Text, nullable=True)  # Browser selections
    results_json = db.Column(db.Text, nullable=True)  # Analysis results
    error_message = db.Column(db.Text, nullable=True)  # Error if failed

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    queued_at = db.Column(db.DateTime, nullable=True)
    started_at = db.Column(db.DateTime, nullable=True)
    completed_at = db.Column(db.DateTime, nullable=True)

    # Email notification
    email_sent = db.Column(db.Boolean, default=False)

    def to_dict(self):
        return {
            'job_id': self.job_id,
            'filename': self.filename,
            'status': self.status,
            'queue_position': self.queue_position,
            'r2_object_key': self.r2_object_key,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'queued_at': self.queued_at.isoformat() if self.queued_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message
        }

    @staticmethod
    def get_queue_position(job_id):
        """Get current position in queue for a job."""
        job = UserJob.query.filter_by(job_id=job_id).first()
        if not job or job.status not in ['queued', 'processing']:
            return None

        # Count jobs ahead in queue
        ahead = UserJob.query.filter(
            UserJob.status.in_(['queued', 'processing']),
            UserJob.queued_at < job.queued_at
        ).count()

        return ahead + 1  # 1-indexed position

    @staticmethod
    def get_estimated_wait_time(position):
        """Estimate wait time based on queue position."""
        # Assume ~2-3 minutes per job average
        avg_job_time = 2.5  # minutes
        return int(position * avg_job_time)


def get_queue_stats():
    """Get overall queue statistics."""
    pending = UserJob.query.filter_by(status='queued').count()
    processing = UserJob.query.filter_by(status='processing').count()

    # Get the currently processing job
    current_job = UserJob.query.filter_by(status='processing').first()

    return {
        'queued_jobs': pending,
        'processing': processing,
        'current_job_id': current_job.job_id if current_job else None,
        'is_busy': processing > 0
    }
