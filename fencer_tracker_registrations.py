"""Fetch upcoming registrations from public FencingTracker profile pages.

External data from FencingTracker (and FencingTimeLive, if added later) is fetched
on demand, used for the current response, and never written to disk, database,
or a cross-request server cache.
"""

from __future__ import annotations

import logging
import re
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from html import unescape
from typing import List, Optional

from fencer_tracker_util import build_profile_url

logger = logging.getLogger(__name__)

FT_BASE = "https://fencingtracker.com"
FETCH_TIMEOUT = 14
USER_AGENT = "SmarterFencing/1.0 (+https://smarterfencing.ai; opponent registrations)"
UPCOMING_WINDOW_DAYS = 7


def _fetch_url(url: str) -> Optional[str]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
    )
    try:
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning("FencingTracker fetch failed for %s: %s", url, exc)
        return None


def _strip_tags(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value or "")
    return re.sub(r"\s+", " ", unescape(text)).strip()


def _parse_registrations_table(table_html: str) -> List[dict]:
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, flags=re.S | re.I)
    registrations: List[dict] = []
    for row in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.S | re.I)
        if len(cells) < 3:
            continue
        clean = [_strip_tags(cell) for cell in cells[:3]]
        if clean[0].lower() == "date":
            continue

        date_match = re.search(r'data-ranking-value="(\d{8})"', row, flags=re.I)
        if not date_match:
            continue

        event_match = re.search(r'href="(/event/(\d+)[^"]*)"[^>]*>(.*?)</a>', cells[2], flags=re.S | re.I)
        event_name = _strip_tags(event_match.group(3)) if event_match else clean[2]
        event_id = event_match.group(2) if event_match else None
        event_path = event_match.group(1) if event_match else None

        registrations.append(
            {
                "date_ymd": int(date_match.group(1)),
                "date_label": clean[0],
                "tournament": clean[1],
                "event": event_name,
                "event_id": event_id,
                "event_url": f"{FT_BASE}{event_path}" if event_path else None,
            }
        )
    return registrations


def parse_registrations_html(html: str) -> List[dict]:
    """Extract the Registrations table from a FencingTracker profile page."""
    heading = re.search(
        r'<h2[^>]*>\s*Registrations\s*</h2>(.*?)(?=<h2[^>]*>|$)',
        html,
        flags=re.S | re.I,
    )
    if not heading:
        return []

    section = heading.group(1)
    table_match = re.search(
        r'<table[^>]*data-ranking-table[^>]*>.*?</table>',
        section,
        flags=re.S | re.I,
    )
    if not table_match:
        return []

    return _parse_registrations_table(table_match.group(0))


def fetch_registrations(fencer_id: str, slug: Optional[str] = None) -> List[dict]:
    profile_url = build_profile_url(fencer_id, slug)
    html = _fetch_url(profile_url)
    return parse_registrations_html(html or "")


def _ymd_to_date(value: int) -> date:
    text = str(value)
    return date(int(text[0:4]), int(text[4:6]), int(text[6:8]))


def _today_ymd(when: Optional[date] = None) -> int:
    current = when or datetime.now().date()
    return int(current.strftime("%Y%m%d"))


def _format_summary(reg: dict) -> str:
    parts = [reg.get("event"), reg.get("tournament")]
    if reg.get("date_label"):
        parts.append(reg["date_label"])
    return " · ".join(part for part in parts if part)


def summarize_schedule_flag(
    fencer_id: str,
    slug: Optional[str] = None,
    when: Optional[date] = None,
) -> dict:
    """
    Return a minimal schedule flag for UI highlighting.

    schedule_status:
      - "today" — registered event is today (green)
      - "upcoming" — next registered event within 7 days (orange)
      - None — no registered event in that window
    """
    registrations = fetch_registrations(fencer_id, slug)
    if not registrations:
        return {"schedule_status": None, "schedule_summary": None}

    ref_day = when or datetime.now().date()
    today_ymd = _today_ymd(ref_day)
    week_end_ymd = _today_ymd(ref_day + timedelta(days=UPCOMING_WINDOW_DAYS))

    today_events = [reg for reg in registrations if reg.get("date_ymd") == today_ymd]
    if today_events:
        reg = min(today_events, key=lambda item: item.get("date_ymd", 0))
        return {
            "schedule_status": "today",
            "schedule_summary": _format_summary(reg),
        }

    upcoming_events = [
        reg
        for reg in registrations
        if reg.get("date_ymd") is not None and today_ymd < int(reg["date_ymd"]) <= week_end_ymd
    ]
    if upcoming_events:
        reg = min(upcoming_events, key=lambda item: item.get("date_ymd", 0))
        return {
            "schedule_status": "upcoming",
            "schedule_summary": _format_summary(reg),
        }

    return {"schedule_status": None, "schedule_summary": None}
