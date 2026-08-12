from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Post(db.Model):
    __tablename__ = "posts"
    id = db.Column(db.Integer, primary_key=True)
    slug = db.Column(db.String(200), unique=True, index=True, nullable=False)
    title = db.Column(db.String(300), nullable=False)
    summary = db.Column(db.Text, nullable=True)
    body_markdown = db.Column(db.Text, nullable=False)
    body_html = db.Column(db.Text, nullable=False)

    published = db.Column(db.Boolean, default=False, nullable=False)
    published_at = db.Column(db.DateTime, nullable=True)

    source_path = db.Column(db.String(500), unique=True, nullable=False)  # file path for upserts
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

class Comment(db.Model):
    __tablename__ = "comments"
    id = db.Column(db.Integer, primary_key=True)
    post_id = db.Column(db.Integer, db.ForeignKey("posts.id"), index=True, nullable=False)

    # Link to SiteUser (from job_queue_models.py)
    user_id = db.Column(db.String(36), db.ForeignKey("site_user.id"), index=True, nullable=True)

    author_name = db.Column(db.String(120), nullable=False)
    body = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    approved = db.Column(db.Boolean, default=False, nullable=False)

    # Relationship to access user data
    user = db.relationship('SiteUser', backref=db.backref('comments', lazy='dynamic'))
    post = db.relationship('Post', backref=db.backref('comments', lazy='dynamic'))

class Like(db.Model):
    __tablename__ = "likes"
    id = db.Column(db.Integer, primary_key=True)
    post_id = db.Column(db.Integer, db.ForeignKey("posts.id"), index=True, nullable=False)
    visitor_id = db.Column(db.String(64), index=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        db.UniqueConstraint("post_id", "visitor_id", name="uq_like_post_visitor"),
    )

