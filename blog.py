# blog.py
import re
import secrets
from pathlib import Path
from datetime import datetime

import yaml
import markdown2
from dateutil import parser as date_parser
from flask import (
    Blueprint, render_template, abort,
    request, jsonify, make_response, current_app, g
)

from models import db, Post, Comment, Like

blog_bp = Blueprint("blog", __name__)

# -----------------------
# Helpers
# -----------------------

def slugify(text: str) -> str:
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:200] if s else "post"

def get_or_set_visitor_id(resp=None):
    vid = request.cookies.get("visitor_id")
    if vid:
        return vid, resp
    vid = secrets.token_hex(16)
    if resp is None:
        resp = make_response()
    resp.set_cookie(
        "visitor_id",
        vid,
        max_age=60 * 60 * 24 * 365,
        httponly=True,
        samesite="Lax"
    )
    return vid, resp

def parse_markdown_with_front_matter(path: Path):
    raw = path.read_text(encoding="utf-8")
    fm = {}
    body = raw

    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            fm = yaml.safe_load(parts[1]) or {}
            body = parts[2].lstrip()

    title = (fm.get("title") or path.stem).strip()
    slug = (fm.get("slug") or slugify(title)).strip()
    summary = (fm.get("summary") or "").strip() or None
    published = bool(fm.get("published", False))

    published_at = None
    if fm.get("date"):
        published_at = date_parser.parse(str(fm["date"]))

    html = markdown2.markdown(
        body,
        extras=["fenced-code-blocks", "tables"]
    )

    return dict(
        title=title,
        slug=slug,
        summary=summary,
        published=published,
        published_at=published_at,
        body_markdown=body,
        body_html=html,
    )

# -----------------------
# Blog pages
# -----------------------

@blog_bp.get("/blog")
def blog_index():
    posts = (
        Post.query
        .filter_by(published=True)
        .order_by(Post.published_at.desc().nullslast())
        .all()
    )
    return render_template("blog_index.html", posts=posts)

@blog_bp.get("/blog/<slug>")
def blog_post(slug):
    post = Post.query.filter_by(slug=slug, published=True).first()
    if not post:
        abort(404)

    like_count = Like.query.filter_by(post_id=post.id).count()
    comments = (
        Comment.query
        .filter_by(post_id=post.id, approved=True)
        .order_by(Comment.created_at.asc())
        .all()
    )

    visitor_id = request.cookies.get("visitor_id")
    liked = False
    if visitor_id:
        liked = Like.query.filter_by(
            post_id=post.id,
            visitor_id=visitor_id
        ).first() is not None

    return render_template(
        "blog_post.html",
        post=post,
        comments=comments,
        like_count=like_count,
        liked=liked
    )

# -----------------------
# API: Likes
# -----------------------

@blog_bp.post("/api/blog/<int:post_id>/like")
def toggle_like(post_id):
    post = Post.query.get_or_404(post_id)

    resp = make_response()
    visitor_id, resp = get_or_set_visitor_id(resp)

    existing = Like.query.filter_by(
        post_id=post.id,
        visitor_id=visitor_id
    ).first()

    if existing:
        db.session.delete(existing)
        liked = False
    else:
        db.session.add(Like(post_id=post.id, visitor_id=visitor_id))
        liked = True

    db.session.commit()
    count = Like.query.filter_by(post_id=post.id).count()

    resp.set_data(jsonify({
        "liked": liked,
        "like_count": count
    }).get_data())
    resp.mimetype = "application/json"
    return resp

# -----------------------
# API: Comments
# -----------------------

@blog_bp.post("/api/blog/<int:post_id>/comments")
def add_comment(post_id):
    post = Post.query.get_or_404(post_id)
    data = request.get_json(force=True)

    author = (data.get("author_name") or "").strip()
    body = (data.get("body") or "").strip()

    if not (1 <= len(author) <= 120):
        return jsonify(error="Invalid name"), 400
    if not (1 <= len(body) <= 2000):
        return jsonify(error="Invalid comment length"), 400

    # Get user_id from g (set by middleware in app.py)
    user_id = getattr(g, 'user_id', None)

    c = Comment(
        post_id=post.id,
        user_id=user_id,  # Associate with logged-in user
        author_name=author,
        body=body,
        approved=False
    )
    db.session.add(c)
    db.session.commit()

    return jsonify(ok=True, message="Comment submitted for approval.")

# -----------------------
# CLI: Import markdown posts
# -----------------------

def register_blog_cli(app):
    @app.cli.command("blog-import")
    def blog_import():
        content_dir = Path("content/blog")
        if not content_dir.exists():
            print("content/blog not found")
            return

        count = 0
        for md in sorted(content_dir.glob("*.md")):
            meta = parse_markdown_with_front_matter(md)

            post = Post.query.filter_by(source_path=str(md)).first()
            if not post:
                post = Post(source_path=str(md), **meta)
                db.session.add(post)
            else:
                for k, v in meta.items():
                    setattr(post, k, v)

            count += 1

        db.session.commit()
        print(f"Imported {count} post(s)")

