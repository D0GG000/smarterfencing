# email_routes.py
"""
Email functionality for sending fencing analysis results.
Uses Gmail SMTP with App Password authentication.

Environment variables required:
- GMAIL_USER: Gmail address (e.g., smarterfencing.ai@gmail.com)
- GMAIL_APP_PASSWORD: 16-character app password from Google
- R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET: R2 credentials
- R2_PUBLIC_URL: public bucket base from Cloudflare portal (e.g. https://pub-xxxxxxxx.r2.dev)
"""

import os
import json
import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime
from urllib.parse import quote

import boto3
from botocore.config import Config

from flask import Blueprint, request, jsonify, current_app, session

from job_queue_models import UserJob
from r2_urls import get_r2_public_base_url, video_playback_url
from fencer_archetype import profiles_from_predictions
from fencing_styles_image import render_fencing_styles_png

email_bp = Blueprint("email", __name__)

# Gmail SMTP settings
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587


def get_gmail_credentials():
    """Get Gmail credentials from environment or app config."""
    gmail_user = os.environ.get("GMAIL_USER") or current_app.config.get("GMAIL_USER")
    gmail_password = os.environ.get("GMAIL_APP_PASSWORD") or current_app.config.get("GMAIL_APP_PASSWORD")
    return gmail_user, gmail_password


def get_r2_client():
    """Get R2 client using existing env variables."""
    r2_account_id = os.environ.get("R2_ACCOUNT_ID")
    r2_bucket = os.environ.get("R2_BUCKET", "smarterfencing-videos")

    if not r2_account_id:
        raise RuntimeError("Missing R2_ACCOUNT_ID")

    client = boto3.client(
        "s3",
        endpoint_url=f"https://{r2_account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )
    return client, r2_bucket, r2_account_id


def upload_json_to_r2(json_data, job_id):
    """
    Upload JSON results to R2 storage.

    Returns:
        str: Public URL of the uploaded JSON file
    """
    client, bucket, account_id = get_r2_client()

    object_key = f"results/{job_id}_results.json"
    # allow_nan=False — browsers reject NaN and then the 3D viewer never loads.
    json_str = json.dumps(json_data, indent=2, allow_nan=False)

    client.put_object(
        Bucket=bucket,
        Key=object_key,
        Body=json_str.encode('utf-8'),
        ContentType='application/json'
    )

    public_url = f"{get_r2_public_base_url()}/{object_key}"
    return public_url


def upload_bytes_to_r2(data: bytes, object_key: str, content_type: str) -> str:
    """Upload raw bytes to R2 and return the public URL."""
    client, bucket, _account_id = get_r2_client()
    client.put_object(
        Bucket=bucket,
        Key=object_key,
        Body=data,
        ContentType=content_type,
        ContentDisposition=f'attachment; filename="{os.path.basename(object_key)}"',
    )
    return f"{get_r2_public_base_url()}/{object_key}"


def build_styles_image_for_email(job_id, predictions):
    """
    Classify styles, render PNG, upload to R2 when possible.

    Returns:
        dict with keys: profiles, png_bytes (optional), image_url (optional)
    """
    profiles = profiles_from_predictions(predictions or [])
    png_bytes = render_fencing_styles_png(predictions or [], profiles=profiles)
    image_url = None
    if png_bytes and get_r2_public_base_url():
        try:
            object_key = f"results/{job_id}_fencing_styles.png"
            image_url = upload_bytes_to_r2(png_bytes, object_key, "image/png")
            logging.info("Uploaded fencing styles image to R2: %s", image_url)
        except Exception as e:
            logging.warning("Could not upload fencing styles image: %s", e)
    return {
        "profiles": profiles,
        "png_bytes": png_bytes,
        "image_url": image_url,
    }


def send_result_email(to_email, subject, html_body, image_attachment=None):
    """
    Send an email via Gmail SMTP.

    Args:
        to_email: Recipient email address
        subject: Email subject
        html_body: HTML content of the email
        image_attachment: optional dict with filename + bytes for a downloadable PNG

    Returns:
        tuple: (success: bool, message: str)
    """
    gmail_user, gmail_password = get_gmail_credentials()

    if not gmail_user or not gmail_password:
        return False, "Gmail credentials not configured. Set GMAIL_USER and GMAIL_APP_PASSWORD environment variables."

    try:
        msg = MIMEMultipart("related")
        msg['From'] = gmail_user
        msg['To'] = to_email
        msg['Subject'] = subject

        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText(html_body, 'html'))
        msg.attach(alt)

        if image_attachment and image_attachment.get("bytes"):
            filename = image_attachment.get("filename") or "fencing-styles.png"
            img = MIMEImage(image_attachment["bytes"], _subtype="png")
            img.add_header("Content-Disposition", "attachment", filename=filename)
            img.add_header("Content-ID", "<fencing_styles>")
            msg.attach(img)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(gmail_user, gmail_password)
            server.send_message(msg)

        logging.info(f"Email sent successfully to {to_email}")
        return True, "Email sent successfully"

    except smtplib.SMTPAuthenticationError as e:
        logging.error(f"SMTP Authentication failed: {e}")
        return False, "Gmail authentication failed. Check your app password."
    except smtplib.SMTPException as e:
        logging.error(f"SMTP error: {e}")
        return False, f"Failed to send email: {str(e)}"
    except Exception as e:
        logging.error(f"Email error: {e}")
        return False, f"Error sending email: {str(e)}"


def _style_card_html(fencer_num, profile):
    arch = profile.get("archetype") or {}
    targeting = profile.get("targeting") or {}
    accent = "#60a5fa" if fencer_num == 1 else "#f87171"
    badge = ""
    if targeting.get("id") and targeting.get("id") != "standard":
        badge = (
            f'<div style="display:inline-block;margin-top:10px;padding:4px 10px;'
            f'border-radius:6px;border:1px solid rgba(52,211,153,0.35);'
            f'background:rgba(16,185,129,0.12);color:#6ee7b7;font-size:12px;font-weight:600;">'
            f'{targeting.get("name", "")}</div>'
        )
    return f"""
        <div style="border-radius:12px;padding:16px 18px;margin:10px 0;
                    background:#1a222c;border:1px solid {accent};">
            <div style="font-size:11px;letter-spacing:0.12em;text-transform:uppercase;
                        color:{accent};font-weight:600;">Fencer {fencer_num}</div>
            <div style="font-size:22px;font-weight:800;color:#fff;margin:6px 0 8px;">
                {arch.get("emoji", "🤺")} {arch.get("name", "Your style")}
            </div>
            <div style="font-size:13px;line-height:1.5;color:#cbd5e1;">
                {arch.get("style", "")}
            </div>
            {badge}
        </div>
    """


def generate_result_email_html(
    video_url,
    result_page_url,
    predictions,
    job_id,
    profiles=None,
    styles_image_url=None,
    has_styles_image=False,
):
    """Generate HTML email body for analysis results."""

    # Count touches by prediction
    touch_counts = {}
    for p in predictions:
        pred = p.get('prediction', 'unknown')
        touch_counts[pred] = touch_counts.get(pred, 0) + 1

    touches_summary = ", ".join([f"{count} {loc}" for loc, count in touch_counts.items()])
    total_touches = len(predictions)

    profiles = profiles or profiles_from_predictions(predictions or [])
    show_image = bool(styles_image_url or has_styles_image)

    if show_image:
        # Image carries the style cards — keep the HTML blurb short.
        p1 = (profiles.get("fencer1") or {}).get("archetype") or {}
        p2 = (profiles.get("fencer2") or {}).get("archetype") or {}
        styles_section = f"""
        <div style="margin: 28px 0 8px;">
            <h2 style="color:#e2e8f0;font-size:18px;margin:0 0 6px;">✨ Your fencing types</h2>
            <p style="font-size:13px;color:#9ca3af;margin:0 0 14px;">
                {p1.get("emoji", "🤺")} {p1.get("name", "Fencer 1")}
                &nbsp;·&nbsp;
                {p2.get("emoji", "🤺")} {p2.get("name", "Fencer 2")}
            </p>
        </div>
        """
        if styles_image_url:
            styles_section += f"""
        <div style="text-align:center;margin:0 0 8px;">
            <img src="{styles_image_url}" alt="Your fencing styles"
                 style="max-width:100%;border-radius:12px;border:1px solid #374151;" />
            <div style="margin-top:10px;">
                <a href="{styles_image_url}"
                   style="display:inline-block;background:#0f766e;color:#ecfdf5;
                          padding:10px 18px;border-radius:8px;text-decoration:none;
                          font-weight:600;font-size:13px;">
                    📥 Download image
                </a>
            </div>
        </div>
            """
        else:
            styles_section += """
        <p style="font-size:12px;color:#9ca3af;margin:0 0 8px;">
            📎 Full style card attached as <strong>fencing-styles.png</strong>.
        </p>
            """
    else:
        styles_section = f"""
        <div style="margin: 28px 0 10px;">
            <h2 style="color:#e2e8f0;font-size:18px;margin:0 0 12px;">✨ Your fencing types</h2>
            {_style_card_html(1, profiles.get("fencer1", {}))}
            {_style_card_html(2, profiles.get("fencer2", {}))}
        </div>
        """

    return f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
        .container {{ max-width: 600px; margin: 0 auto; background: #16213e; border-radius: 12px; padding: 30px; }}
        h1 {{ color: #3b82f6; margin-bottom: 10px; }}
        .summary {{ background: #1a1a2e; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .stat {{ display: inline-block; margin-right: 20px; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #10b981; }}
        .stat-label {{ font-size: 12px; color: #9ca3af; }}
        .btn {{ display: inline-block; background: linear-gradient(135deg, #3b82f6, #8b5cf6); color: white;
                padding: 14px 28px; border-radius: 8px; text-decoration: none; font-weight: 600; margin: 10px 0; }}
        .btn:hover {{ opacity: 0.9; }}
        .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #374151; font-size: 12px; color: #6b7280; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤺 Your Fencing Analysis is Ready!</h1>
        <p>Your video analysis has been completed. Here's a summary of the results:</p>

        <div class="summary">
            <div class="stat">
                <div class="stat-value">{total_touches}</div>
                <div class="stat-label">Touches Detected</div>
            </div>
            <div class="stat">
                <div class="stat-value">{touches_summary or 'N/A'}</div>
                <div class="stat-label">Touch Locations</div>
            </div>
        </div>

        {styles_section}

        <p><strong>Job ID:</strong> {job_id}</p>
        <p><strong>Analysis Date:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>

        <div style="margin: 25px 0; text-align: center;">
            <a href="{result_page_url}" class="btn">View Full Results →</a>
        </div>

        <p style="font-size: 14px; color: #9ca3af;">
            Click the button above to view your complete analysis including the video playback,
            3D pose visualization, and detailed frame-by-frame breakdown.
        </p>

        <div class="footer">
            <p>This email was sent by SmarterFencing AI</p>
            <p>Questions? Visit <a href="https://smarterfencing.ai" style="color: #3b82f6;">smarterfencing.ai</a></p>
        </div>
    </div>
</body>
</html>
"""


@email_bp.route('/api/send-results-email', methods=['POST'])
def send_results_email():
    """
    Send analysis results via email.
    Uploads JSON to R2 and sends email with link that auto-loads data.

    Expected JSON body:
    {
        "email": "user@example.com",
        "video_url": "https://...",
        "job_id": "abc123",
        "predictions": [...],
        "three_d_results": {...},
        "fps": 30
    }
    """
    uid = session.get("user_id")
    if not uid:
        return jsonify({"success": False, "error": "Authentication required"}), 401

    data = request.get_json(silent=True) or {}

    # Validate required fields
    email = data.get('email', '').strip()
    if not email or '@' not in email:
        return jsonify({'success': False, 'error': 'Valid email address required'}), 400

    video_url = data.get('video_url')
    if not video_url:
        return jsonify({'success': False, 'error': 'Video URL required'}), 400

    job_id = data.get('job_id', 'unknown')
    if job_id and job_id != 'unknown':
        job = UserJob.query.filter_by(job_id=job_id, user_id=uid).first()
        if not job:
            return jsonify({"success": False, "error": "Job not found"}), 404

    predictions = data.get('predictions', [])
    three_d_results = data.get('three_d_results', {})
    fps = data.get('fps', 30)

    try:
        # Upload JSON results to R2
        json_data = {
            'video_url': video_url,
            'job_id': job_id,
            'fps': fps,
            'predictions': predictions,
            'three_d_results': three_d_results,
            'generated_at': datetime.now().isoformat()
        }

        json_url = upload_json_to_r2(json_data, job_id)
        logging.info(f"Uploaded results JSON to R2: {json_url}")

        # job_id first: signed-in owner loads merged DB results; video+data fallback for snapshot / logged-out
        base_url = request.host_url.rstrip('/')
        result_page_url = (
            f"{base_url}/result?job_id={quote(job_id)}"
            f"&video={quote(video_url)}&data={quote(json_url)}"
        )

        styles = build_styles_image_for_email(job_id, predictions)
        html_body = generate_result_email_html(
            video_url,
            result_page_url,
            predictions,
            job_id,
            profiles=styles["profiles"],
            styles_image_url=styles.get("image_url"),
            has_styles_image=bool(styles.get("png_bytes")),
        )

        image_attachment = None
        if styles.get("png_bytes"):
            image_attachment = {
                "filename": "fencing-styles.png",
                "bytes": styles["png_bytes"],
            }

        success, message = send_result_email(
            to_email=email,
            subject=f"🤺 Your Fencing Analysis Results - {job_id}",
            html_body=html_body,
            image_attachment=image_attachment,
        )

        if success:
            return jsonify({'success': True, 'message': message, 'result_url': result_page_url})
        else:
            return jsonify({'success': False, 'error': message}), 500

    except Exception as e:
        logging.error(f"Error in send_results_email: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def send_results_email_internal(app, email, video_url, job_id, predictions, three_d_results=None, fps=30):
    """
    Internal function to send results email (called by queue worker).

    Args:
        app: Flask app instance
        email: Recipient email
        video_url: Video playback URL (presigned or public)
        job_id: Job identifier
        predictions: List of predictions
        three_d_results: Optional 3D results dict
        fps: Video FPS

    Returns:
        tuple: (success: bool, message: str)
    """
    with app.app_context():
        try:
            # Upload JSON results to R2
            json_data = {
                'video_url': video_url,
                'job_id': job_id,
                'fps': fps,
                'predictions': predictions,
                'three_d_results': three_d_results or {},
                'generated_at': datetime.now().isoformat()
            }

            json_url = upload_json_to_r2(json_data, job_id)
            logging.info(f"Uploaded results JSON to R2: {json_url}")

            # Build result page URL
            # Use configured base URL or fallback
            base_url = os.environ.get('BASE_URL', 'https://smarterfencing.ai')
            result_page_url = (
                f"{base_url}/result?job_id={quote(job_id or '')}"
                f"&video={quote(video_url or '')}&data={quote(json_url or '')}"
            )

            styles = build_styles_image_for_email(job_id, predictions)
            html_body = generate_result_email_html(
                video_url,
                result_page_url,
                predictions,
                job_id,
                profiles=styles["profiles"],
                styles_image_url=styles.get("image_url"),
                has_styles_image=bool(styles.get("png_bytes")),
            )

            image_attachment = None
            if styles.get("png_bytes"):
                image_attachment = {
                    "filename": "fencing-styles.png",
                    "bytes": styles["png_bytes"],
                }

            success, message = send_result_email(
                to_email=email,
                subject=f"🤺 Your Fencing Analysis Results - {job_id}",
                html_body=html_body,
                image_attachment=image_attachment,
            )

            return success, message

        except Exception as e:
            logging.error(f"Error in send_results_email_internal: {e}")
            return False, str(e)


def register_email(app):
    """Register email blueprint with the Flask app."""
    app.register_blueprint(email_bp)
