"""
R2 public and presigned URL helpers (shared by queue, email, admin).
"""

from __future__ import annotations

import os
from typing import Optional

import boto3
from botocore.config import Config

# Presigned GET for email links (7 days — S3/R2 typical maximum).
EMAIL_VIDEO_URL_EXPIRES = 7 * 24 * 3600
# Presigned GET for in-app API responses.
API_VIDEO_URL_EXPIRES = 6 * 3600


def get_r2_public_base_url() -> str:
    """
    Public base URL for R2 objects.

    Set R2_PUBLIC_URL to the exact host shown in the Cloudflare R2 portal
    (e.g. https://pub-d2648056ec514bdea3d1935baa03c098.r2.dev).
    Do not derive this from R2_ACCOUNT_ID — the pub-* hash is bucket-specific.
    """
    r2_public_url = (os.environ.get("R2_PUBLIC_URL") or "").strip()
    if r2_public_url:
        return r2_public_url.rstrip("/")
    return ""


def public_object_url(object_key: Optional[str]) -> Optional[str]:
    if not object_key:
        return None
    base = get_r2_public_base_url()
    if not base:
        return None
    return f"{base}/{object_key.lstrip('/')}"


def _r2_s3_client():
    account_id = os.environ.get("R2_ACCOUNT_ID")
    if not account_id:
        return None, None
    client = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )
    bucket = os.environ.get("R2_BUCKET", "smarterfencing-videos")
    return client, bucket


def presigned_get_url(object_key: Optional[str], expires_in: int = API_VIDEO_URL_EXPIRES) -> Optional[str]:
    if not object_key:
        return None
    client, bucket = _r2_s3_client()
    if not client or not bucket:
        return None
    return client.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": object_key},
        ExpiresIn=expires_in,
    )


def video_playback_url(
    object_key: Optional[str],
    *,
    expires_in: int = API_VIDEO_URL_EXPIRES,
) -> Optional[str]:
    """
    URL for <video src=...>.

    Public R2 buckets: use stable pub-*.r2.dev (or R2_PUBLIC_URL) links.
    Presigned GET is only used when a public base URL cannot be built.
    """
    if not object_key:
        return None
    public = public_object_url(object_key)
    if public:
        return public
    return presigned_get_url(object_key, expires_in=expires_in)
