# user_models.py
"""
User tracking models for cookie-based user identification and job association.
"""
from models import db
from datetime import datetime


class SiteUser(db.Model):
    """Track unique visitors via cookie-based UUID."""
    __tablename__ = 'site_users'

    id = db.Column(db.String(36), primary_key=True)  # UUID from cookie
    email = db.Column(db.String(255), nullable=True)  # Optional email
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_seen = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship to jobs
    jobs = db.relationship('UserJob', backref='user', lazy='dynamic')

    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_seen': self.last_seen.isoformat() if self.last_seen else None,
            'job_count': self.jobs.count()
        }


class UserJob(db.Model):
    """Associate uploaded video jobs with users."""
    __tablename__ = 'user_jobs'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.String(36), db.ForeignKey('site_users.id'), nullable=False)
    job_id = db.Column(db.String(36), nullable=False, index=True)  # The job_id from video upload
    r2_object_key = db.Column(db.String(255), nullable=True)  # R2 storage key
    filename = db.Column(db.String(255), nullable=True)  # Original filename
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'job_id': self.job_id,
            'r2_object_key': self.r2_object_key,
            'filename': self.filename,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
