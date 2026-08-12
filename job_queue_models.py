# job_queue_models.py
"""
Extended models for job queue system.
Adds queue status tracking to existing UserJob model.
"""

import os
from datetime import datetime
from models import db


def queue_max_length():
    return int(os.environ.get("QUEUE_MAX_LENGTH", "50"))


def get_queue_length():
    """Jobs waiting to be processed (status=queued)."""
    return UserJob.query.filter_by(status="queued").count()


def is_queue_full():
    """True when the waiting queue exceeds the configured maximum."""
    return get_queue_length() > queue_max_length()


class SiteUser(db.Model):
    """User model for tracking visitors."""
    __tablename__ = 'site_user'

    id = db.Column(db.String(36), primary_key=True)  # UUID
    email = db.Column(db.String(255), nullable=True)
    google_sub = db.Column(db.String(255), nullable=True, unique=True)
    password_hash = db.Column(db.Text, nullable=True)
    display_name = db.Column(db.String(255), nullable=True)
    username = db.Column(db.String(32), nullable=True, unique=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)

    jobs = db.relationship('UserJob', backref='user', lazy='dynamic')
    tracked_fencers = db.relationship(
        'TrackedFencer', backref='user', lazy='dynamic', cascade='all, delete-orphan'
    )
    touch_shares_sent = db.relationship(
        'TouchShare',
        backref='from_user',
        lazy='dynamic',
        foreign_keys='TouchShare.from_user_id',
    )

    def to_dict(self):
        return {
            'id': self.id[:8] + '...',
            'email': self.email,
            'display_name': self.display_name,
            'username': self.username,
            'has_google': bool(self.google_sub),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_seen': self.last_seen.isoformat() if self.last_seen else None
        }

    def public_profile(self):
        """Fields safe to show other users (no email)."""
        label = (self.display_name or '').strip() or (self.username or 'User')
        return {
            'id': self.id,
            'username': self.username,
            'display_name': label,
        }


class TrackedFencer(db.Model):
    """User bookmark pointing at a public FencingTracker profile.

    Only the bookmark metadata below is stored locally. Tournament registration
    or live results from FencingTracker / FencingTimeLive are fetched on demand
    and are never persisted by SmarterFencing.
    """
    __tablename__ = 'tracked_fencer'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(36), db.ForeignKey('site_user.id'), nullable=False, index=True)
    fencer_id = db.Column(db.String(32), nullable=False)
    slug = db.Column(db.String(255), nullable=True)
    display_name = db.Column(db.String(255), nullable=False)
    user_note = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        db.UniqueConstraint('user_id', 'fencer_id', name='uq_tracked_fencer_user_fencer'),
    )

    @property
    def profile_url(self):
        from fencer_tracker_util import build_profile_url
        return build_profile_url(self.fencer_id, self.slug)


class UserJob(db.Model):
    """Job model with queue status tracking."""
    __tablename__ = 'user_job'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(36), db.ForeignKey('site_user.id'), nullable=False)
    job_id = db.Column(db.String(20), unique=True, nullable=False)  # e.g., "abc123def456"

    # File info
    r2_object_key = db.Column(db.String(255), nullable=True)  # e.g., "uploads/abc123.mp4"
    filename = db.Column(db.String(255), nullable=True)  # Original filename
    file_size = db.Column(db.BigInteger, nullable=True)  # File size in bytes

    # Pipeline variant: "analysis" (full 2D/3D/prediction) or "clipper" (touch detection + highlight reel only)
    job_type = db.Column(db.String(20), default='analysis', nullable=False)
    # R2 key of the stitched touch highlight reel (clipper jobs)
    highlight_reel_key = db.Column(db.String(255), nullable=True)

    # Queue status
    status = db.Column(db.String(20), default='pending')  # pending, queued, processing, complete, error
    queue_position = db.Column(db.Integer, nullable=True)  # Position when queued

    # Pipeline data (stored as JSON string)
    selections_json = db.Column(db.Text, nullable=True)  # Browser selections
    results_json = db.Column(db.Text, nullable=True)  # Analysis results
    prediction_corrections_json = db.Column(
        db.Text, nullable=True
    )  # user touch-location / attack overrides and per-touch notes
    touch_deletions_json = db.Column(db.Text, nullable=True)  # JSON list of touch ids hidden by user
    macro_corrections_json = db.Column(
        db.Text, nullable=True
    )  # user overlays for bout-level spatial / arm / footwork rollups
    llm_analysis_json = db.Column(
        db.Text, nullable=True
    )  # OpenAI archetype + practice suggestions for user_fencer
    error_message = db.Column(db.Text, nullable=True)  # Error if failed

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    queued_at = db.Column(db.DateTime, nullable=True)
    started_at = db.Column(db.DateTime, nullable=True)
    completed_at = db.Column(db.DateTime, nullable=True)

    # Email notification
    email_sent = db.Column(db.Boolean, default=False)

    # Unguessable token for read-only share links (/result?share=...)
    share_token = db.Column(db.String(64), nullable=True, unique=True, index=True)

    def to_dict(self):
        return {
            'job_id': self.job_id,
            'filename': self.filename,
            'file_size': self.file_size,
            'status': self.status,
            'job_type': self.job_type or 'analysis',
            'has_highlight_reel': bool(getattr(self, 'highlight_reel_key', None)),
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
    def get_estimated_wait_time(job_id):
        """
        Estimate wait time based on file sizes of jobs ahead in queue.
        Processing rate: approximately 5 MB per minute (based on observed performance).
        """
        job = UserJob.query.filter_by(job_id=job_id).first()
        if not job or job.status not in ['queued', 'processing']:
            return None

        # Get all jobs ahead in queue (including currently processing)
        jobs_ahead = UserJob.query.filter(
            UserJob.status.in_(['queued', 'processing']),
            UserJob.queued_at <= job.queued_at,
            UserJob.job_id != job_id
        ).all()

        # Sum file sizes of jobs ahead
        total_bytes = 0
        for j in jobs_ahead:
            if j.file_size:
                total_bytes += j.file_size
            else:
                # Fallback: assume 50MB if file size unknown
                total_bytes += 50 * 1024 * 1024

        # Processing rate: ~5 MB per minute (conservative estimate)
        # This accounts for download, processing, and upload
        mb_per_minute = 5.0
        total_mb = total_bytes / (1024 * 1024)

        estimated_minutes = total_mb / mb_per_minute

        # Minimum 1 minute if there are jobs ahead
        if jobs_ahead and estimated_minutes < 1:
            estimated_minutes = 1

        return int(round(estimated_minutes))


def get_queue_stats():
    """Get overall queue statistics."""
    pending = UserJob.query.filter_by(status='queued').count()
    processing = UserJob.query.filter_by(status='processing').count()
    max_length = queue_max_length()

    # Get the currently processing job
    current_job = UserJob.query.filter_by(status='processing').first()

    return {
        'queued_jobs': pending,
        'processing': processing,
        'current_job_id': current_job.job_id if current_job else None,
        'is_busy': processing > 0,
        'queue_max_length': max_length,
        'queue_full': pending > max_length,
    }


class TouchShare(db.Model):
    """A touch or full analysis shared from one user to one or more recipients.

    Private Messages counterpart to CommunityPost: kind is 'touch' (single touch)
    or 'analysis' (full bout). For analysis shares, touch_id is empty.
    """
    __tablename__ = 'touch_share'

    id = db.Column(db.String(36), primary_key=True)
    job_id = db.Column(db.String(20), nullable=False, index=True)
    # 'touch' (single touch) or 'analysis' (full bout)
    kind = db.Column(db.String(20), default='touch', nullable=False)
    touch_id = db.Column(db.String(512), nullable=False, default='')
    touch_ref = db.Column(db.String(32), nullable=True, index=True)
    from_user_id = db.Column(db.String(36), db.ForeignKey('site_user.id'), nullable=False, index=True)
    status = db.Column(db.String(20), default='active', nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    recipients = db.relationship(
        'TouchShareRecipient',
        backref='share',
        lazy='dynamic',
        cascade='all, delete-orphan',
    )
    messages = db.relationship(
        'TouchShareMessage',
        backref='share',
        lazy='dynamic',
        cascade='all, delete-orphan',
        order_by='TouchShareMessage.created_at',
    )


class TouchShareRecipient(db.Model):
    __tablename__ = 'touch_share_recipient'

    id = db.Column(db.Integer, primary_key=True)
    share_id = db.Column(db.String(36), db.ForeignKey('touch_share.id'), nullable=False, index=True)
    user_id = db.Column(db.String(36), db.ForeignKey('site_user.id'), nullable=False, index=True)
    last_read_at = db.Column(db.DateTime, nullable=True)
    joined_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    user = db.relationship('SiteUser', foreign_keys=[user_id])

    __table_args__ = (
        db.UniqueConstraint('share_id', 'user_id', name='uq_touch_share_recipient'),
    )


class TouchShareMessage(db.Model):
    __tablename__ = 'touch_share_message'

    id = db.Column(db.String(36), primary_key=True)
    share_id = db.Column(db.String(36), db.ForeignKey('touch_share.id'), nullable=False, index=True)
    author_user_id = db.Column(db.String(36), db.ForeignKey('site_user.id'), nullable=False)
    body = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    author = db.relationship('SiteUser', foreign_keys=[author_user_id])


class CommunityPost(db.Model):
    """A public post sharing an analysis, touch clip, or highlight reel.

    Separate from the private TouchShare/Messages system: anyone (signed in or
    not) can view these, and signed-in users can comment.
    """
    __tablename__ = 'community_post'

    id = db.Column(db.String(36), primary_key=True)
    author_user_id = db.Column(db.String(36), db.ForeignKey('site_user.id'), nullable=False, index=True)
    job_id = db.Column(db.String(20), nullable=False, index=True)
    # 'touch' (single touch clip), 'highlight_reel', or 'analysis' (full bout)
    kind = db.Column(db.String(20), default='touch', nullable=False)
    touch_id = db.Column(db.String(512), nullable=True)
    touch_ref = db.Column(db.String(32), nullable=True)
    caption = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(20), default='active', nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)

    author = db.relationship('SiteUser', foreign_keys=[author_user_id])
    comments = db.relationship(
        'CommunityComment',
        backref='post',
        lazy='dynamic',
        cascade='all, delete-orphan',
        order_by='CommunityComment.created_at',
    )


class CommunityComment(db.Model):
    __tablename__ = 'community_comment'

    id = db.Column(db.String(36), primary_key=True)
    post_id = db.Column(db.String(36), db.ForeignKey('community_post.id'), nullable=False, index=True)
    author_user_id = db.Column(db.String(36), db.ForeignKey('site_user.id'), nullable=False)
    body = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    author = db.relationship('SiteUser', foreign_keys=[author_user_id])
