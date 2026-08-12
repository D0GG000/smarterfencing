"""User-initiated name search via FencingTracker's public search endpoint (same as their site navbar).

Search results are returned to the client only and are not cached or stored.
"""

import json
import logging
import urllib.error
import urllib.request
from typing import List, Optional, Tuple

from fencer_tracker_util import build_profile_url, slug_to_display_name

logger = logging.getLogger(__name__)

FT_SEARCH_URL = "https://fencingtracker.com/search"
SEARCH_TIMEOUT = 12


def build_name_query(first_name: str, last_name: str) -> str:
    first = (first_name or "").strip()
    last = (last_name or "").strip()
    if not first or not last:
        return ""
    return f"{first} {last}"


def fencer_from_search_hit(hit: dict) -> dict:
    fencer_id = str(hit.get("usfa_id") or "").strip()
    slug = (hit.get("name") or "").strip() or None
    display_name = slug_to_display_name(slug) if slug else f"Fencer {fencer_id}"
    club = (hit.get("club") or "").strip() or None
    return {
        "fencer_id": fencer_id,
        "slug": slug,
        "display_name": display_name,
        "club": club,
        "profile_url": build_profile_url(fencer_id, slug),
    }


def search_fencers_by_name(
    first_name: str, last_name: str, limit: int = 15
) -> Tuple[Optional[List[dict]], Optional[str]]:
    query = build_name_query(first_name, last_name)
    if not query:
        return None, "First and last name are required"

    payload = json.dumps({"query": query, "limit": limit}).encode("utf-8")
    req = urllib.request.Request(
        FT_SEARCH_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "SmarterFencing/1.0 (+https://smarterfencing.ai; fencer lookup)",
            "Referer": "https://fencingtracker.com/",
            "Origin": "https://fencingtracker.com",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=SEARCH_TIMEOUT) as resp:
            hits = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        logger.warning("FencingTracker search HTTP %s", e.code)
        return None, "FencingTracker search is unavailable right now. Try again later."
    except urllib.error.URLError as e:
        logger.warning("FencingTracker search network error: %s", e)
        return None, "Could not reach FencingTracker. Check your connection and try again."
    except (json.JSONDecodeError, TimeoutError, OSError) as e:
        logger.warning("FencingTracker search failed: %s", e)
        return None, "FencingTracker search failed. Try again later."

    if not isinstance(hits, list):
        return None, "Unexpected response from FencingTracker"

    fencers = [fencer_from_search_hit(h) for h in hits if h.get("usfa_id")]
    return fencers, None
