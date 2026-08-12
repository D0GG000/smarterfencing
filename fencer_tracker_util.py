"""Parse FencingTracker profile URLs/IDs locally — no HTTP requests to fencingtracker.com.

SmarterFencing does not cache or store FencingTracker profile data.
"""

import re
from typing import Dict, Optional
from urllib.parse import unquote

FT_PROFILE_RE = re.compile(
    r"(?:https?://(?:www\.)?fencingtracker\.com)?/?p/(\d+)(?:/([^/?#]+))?",
    re.IGNORECASE,
)

FT_BASE = "https://fencingtracker.com"


def slug_to_display_name(slug: str) -> str:
    """Turn URL slug into a readable name (best-effort, user can edit when saving)."""
    name = unquote(slug or "").replace("-", " ").strip()
    return name or ""


def build_profile_url(fencer_id: str, slug: Optional[str] = None) -> str:
    fid = str(fencer_id).strip()
    if slug:
        slug = slug.strip().strip("/")
        return f"{FT_BASE}/p/{fid}/{slug}"
    return f"{FT_BASE}/p/{fid}"


def parse_fencer_input(raw: str) -> Optional[Dict]:
    """
    Accept a FencingTracker profile URL or numeric fencer ID.
    Returns dict with fencer_id, slug, display_name, profile_url — or None if invalid.
    """
    text = (raw or "").strip()
    if not text:
        return None

    if text.isdigit():
        fencer_id = text
        return {
            "fencer_id": fencer_id,
            "slug": None,
            "display_name": f"Fencer {fencer_id}",
            "profile_url": build_profile_url(fencer_id),
        }

    match = FT_PROFILE_RE.search(text)
    if not match:
        return None

    fencer_id = match.group(1)
    slug = match.group(2)
    if slug:
        slug = unquote(slug).strip("/") or None
    display_name = slug_to_display_name(slug) if slug else f"Fencer {fencer_id}"

    return {
        "fencer_id": fencer_id,
        "slug": slug,
        "display_name": display_name,
        "profile_url": build_profile_url(fencer_id, slug),
    }
