"""Signed password-reset tokens and reset email."""

import os
from urllib.parse import quote

from flask import current_app, request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from email_routes import send_result_email

RESET_SALT = "sf-password-reset"
RESET_MAX_AGE_SEC = int(os.environ.get("PASSWORD_RESET_MAX_AGE_SEC", "3600"))


def _serializer():
    secret = current_app.config.get("SECRET_KEY") or "dev-only-change-in-production"
    return URLSafeTimedSerializer(secret, salt=RESET_SALT)


def make_reset_token(user_id):
    return _serializer().dumps(user_id)


def load_reset_user_id(token):
    """Return user_id or None if token is invalid/expired."""
    if not token:
        return None
    try:
        return _serializer().loads(token, max_age=RESET_MAX_AGE_SEC)
    except (BadSignature, SignatureExpired):
        return None


def public_site_base_url():
    base = (os.environ.get("BASE_URL") or "").strip().rstrip("/")
    if base:
        return base
    return request.url_root.rstrip("/")


def password_reset_email_html(reset_url):
    return f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
        .container {{ max-width: 560px; margin: 0 auto; background: #16213e; border-radius: 12px; padding: 28px; }}
        h1 {{ color: #3b82f6; font-size: 1.25rem; margin: 0 0 12px; }}
        p {{ line-height: 1.5; color: #cbd5e1; }}
        .btn {{
            display: inline-block; margin: 20px 0; padding: 12px 20px;
            background: #2563eb; color: #fff !important; text-decoration: none;
            border-radius: 8px; font-weight: 600;
        }}
        .muted {{ font-size: 0.85rem; color: #94a3b8; }}
        code {{ word-break: break-all; font-size: 0.8rem; color: #93c5fd; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Reset your SmarterFencing password</h1>
        <p>We received a request to reset the password for your account. Click the button below to choose a new password.</p>
        <p><a class="btn" href="{reset_url}">Reset password</a></p>
        <p class="muted">This link expires in {RESET_MAX_AGE_SEC // 60} minutes. If you did not request a reset, you can ignore this email.</p>
        <p class="muted">If the button does not work, copy this link into your browser:<br><code>{reset_url}</code></p>
    </div>
</body>
</html>
"""


def send_password_reset_email(to_email, token):
    reset_url = f"{public_site_base_url()}/reset-password?token={quote(token)}"
    subject = "Reset your SmarterFencing password"
    html = password_reset_email_html(reset_url)
    return send_result_email(to_email, subject, html)
