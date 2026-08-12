"""
Generate a shareable PNG of both fencers' style archetypes for result emails.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fencer_archetype import profiles_from_predictions

log = logging.getLogger(__name__)

# RGB
BG = (15, 20, 25)
CARD_BG = (26, 34, 44)
CARD_BORDER = (55, 65, 80)
F1_ACCENT = (96, 165, 250)  # blue-400
F2_ACCENT = (248, 113, 113)  # red-400
TITLE = (255, 255, 255)
BODY = (203, 213, 225)  # slate-300
MUTED = (148, 163, 184)
BADGE_BG = (6, 48, 40)
BADGE_FG = (110, 231, 183)
BADGE_BORDER = (16, 120, 90)
HEADER_FG = (148, 163, 184)


def _try_load_font(size: int, bold: bool = False):
    try:
        from PIL import ImageFont
    except ImportError:
        return None

    candidates: List[str] = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                "C:/Windows/Fonts/arialbd.ttf",
                "C:/Windows/Fonts/segoeuib.ttf",
            ]
        )
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
        ]
    )
    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _try_emoji_font(size: int):
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for path in (
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
        "/usr/share/fonts/google-noto-emoji/NotoColorEmoji.ttf",
        "C:/Windows/Fonts/seguiemj.ttf",
        "C:/Windows/Fonts/SegoeUIEmoji.ttf",
    ):
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return None


def _text_size(draw, text: str, font) -> Tuple[int, int]:
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    if font is not None and hasattr(font, "getsize"):
        return font.getsize(text)
    return (len(text) * 7, 14)


def _wrap_text(draw, text: str, font, max_width: int) -> List[str]:
    words = (text or "").split()
    if not words:
        return []
    lines: List[str] = []
    cur = words[0]
    for w in words[1:]:
        trial = cur + " " + w
        tw, _ = _text_size(draw, trial, font)
        if tw <= max_width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def _rounded_rect(draw, xy, radius: int, fill, outline=None, width: int = 1):
    try:
        draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)
    except Exception:
        draw.rectangle(xy, fill=fill, outline=outline, width=width)


def _draw_emoji_fallback(draw, xy: Tuple[int, int], archetype_id: str, size: int = 36):
    """Simple geometric stand-ins when emoji fonts are unavailable."""
    x, y = xy
    cx, cy = x + size // 2, y + size // 2
    if archetype_id == "duelist":
        draw.line((x + 6, y + size - 6, x + size - 6, y + 6), fill=(226, 232, 240), width=3)
        draw.line((x + 6, y + 6, x + size - 6, y + size - 6), fill=(226, 232, 240), width=3)
    elif archetype_id == "sniper":
        r = size // 2 - 2
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(248, 113, 113), width=2)
        draw.ellipse((cx - r // 2, cy - r // 2, cx + r // 2, cy + r // 2), outline=(248, 113, 113), width=2)
        draw.line((cx - r, cy, cx + r, cy), fill=(248, 113, 113), width=1)
        draw.line((cx, cy - r, cx, cy + r), fill=(248, 113, 113), width=1)
    elif archetype_id == "bolt":
        pts = [
            (cx - 4, y + 4),
            (cx + 10, y + 4),
            (cx + 2, cy + 2),
            (cx + 12, cy + 2),
            (cx - 8, y + size - 4),
            (cx - 2, cy + 6),
            (cx - 12, cy + 6),
        ]
        draw.polygon(pts, fill=(250, 204, 21))
    elif archetype_id == "tactician":
        draw.ellipse((cx - 8, y + 6, cx + 8, y + 22), outline=(167, 243, 208), width=2)
        draw.rectangle((cx - 3, y + 20, cx + 3, y + size - 6), fill=(167, 243, 208))
        draw.rectangle((cx - 10, y + size - 8, cx + 10, y + size - 4), fill=(167, 243, 208))
    elif archetype_id == "hybrid":
        draw.line((x + 6, cy, x + size - 6, cy), fill=(165, 180, 252), width=3)
        draw.ellipse((x + 4, cy - 8, x + 16, cy + 8), outline=(165, 180, 252), width=2)
        draw.ellipse((x + size - 16, cy - 8, x + size - 4, cy + 8), outline=(165, 180, 252), width=2)
    else:
        # Unknown id — neutral geometric mark (not a specific archetype)
        draw.ellipse((cx - 10, cy - 10, cx + 10, cy + 10), outline=(148, 163, 184), width=2)


def _draw_card(draw, box, fencer_num: int, profile: Dict[str, Any], fonts: Dict[str, Any]):
    x0, y0, x1, y1 = box
    accent = F1_ACCENT if fencer_num == 1 else F2_ACCENT
    _rounded_rect(draw, box, radius=18, fill=CARD_BG, outline=accent, width=2)

    pad = 22
    ax = x0 + pad
    ay = y0 + pad

    label = f"FENCER {fencer_num}"
    draw.text((ax, ay), label, fill=accent, font=fonts["eyebrow"])
    ay += 22

    arch = profile.get("archetype") or {}
    arch_id = arch.get("id") or ""
    emoji = arch.get("emoji") or "🤺"
    name = arch.get("name") or "Your style"

    emoji_font = fonts.get("emoji")
    emoji_drawn = False
    if emoji_font is not None:
        try:
            draw.text((ax, ay - 2), emoji, font=emoji_font, embedded_color=True)
            emoji_drawn = True
        except TypeError:
            try:
                draw.text((ax, ay - 2), emoji, font=emoji_font)
                emoji_drawn = True
            except Exception:
                emoji_drawn = False
        except Exception:
            emoji_drawn = False

    if not emoji_drawn:
        _draw_emoji_fallback(draw, (ax, ay), arch_id, size=40)
    name_x = ax + 48
    draw.text((name_x, ay), name, fill=TITLE, font=fonts["title"])
    ay += 48

    blurb = profile.get("styleBlurb") or arch.get("style") or ""
    for line in _wrap_text(draw, blurb, fonts["body"], max_width=(x1 - x0) - pad * 2):
        draw.text((ax, ay), line, fill=BODY, font=fonts["body"])
        ay += 22

    targeting = profile.get("targeting") or {}
    badge = targeting.get("name") or ""
    if badge and targeting.get("id") != "standard":
        ay += 10
        bw, bh = _text_size(draw, badge, fonts["badge"])
        bx0, by0 = ax, ay
        bx1, by1 = ax + bw + 20, ay + bh + 14
        _rounded_rect(draw, (bx0, by0, bx1, by1), radius=8, fill=BADGE_BG, outline=BADGE_BORDER, width=1)
        draw.text((bx0 + 10, by0 + 6), badge, fill=BADGE_FG, font=fonts["badge"])


def render_fencing_styles_png(
    predictions: Sequence[Dict[str, Any]],
    profiles: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[bytes]:
    """
    Render a PNG of both fencers' styles. Returns PNG bytes, or None on failure.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        log.warning("Pillow not available; skipping fencing styles image")
        return _render_with_cv2(predictions, profiles)

    try:
        profiles = profiles or profiles_from_predictions(predictions)
        p1 = profiles["fencer1"]
        p2 = profiles["fencer2"]

        width = 720
        margin = 28
        header_h = 72
        card_h = 210
        gap = 16
        height = margin + header_h + card_h * 2 + gap + margin

        img = Image.new("RGB", (width, height), BG)
        draw = ImageDraw.Draw(img)

        fonts = {
            "eyebrow": _try_load_font(14, bold=True),
            "title": _try_load_font(30, bold=True),
            "body": _try_load_font(16, bold=False),
            "badge": _try_load_font(13, bold=True),
            "header": _try_load_font(18, bold=True),
            "emoji": _try_emoji_font(34),
        }

        # Prefer a clear ASCII header + separate emoji glyph when color font works
        draw.text((margin, margin + 8), "YOUR FENCING STYLES", fill=HEADER_FG, font=fonts["header"])
        fun = "🤺✨"
        emoji_font = fonts.get("emoji")
        if emoji_font is not None:
            try:
                draw.text((margin + 250, margin), fun, font=emoji_font, embedded_color=True)
            except TypeError:
                try:
                    draw.text((margin + 250, margin + 2), fun, font=emoji_font)
                except Exception:
                    draw.text((margin + 250, margin + 4), ":)", fill=BADGE_FG, font=fonts["header"])
            except Exception:
                draw.text((margin + 250, margin + 4), ":)", fill=BADGE_FG, font=fonts["header"])
        else:
            # Drawn sparkles
            for ox, oy in ((margin + 250, margin + 18), (margin + 270, margin + 10), (margin + 290, margin + 20)):
                draw.ellipse((ox - 3, oy - 3, ox + 3, oy + 3), fill=BADGE_FG)
            draw.text((margin + 310, margin + 8), "GO!", fill=BADGE_FG, font=fonts["header"])

        y = margin + header_h
        card_w = width - margin * 2
        _draw_card(draw, (margin, y, margin + card_w, y + card_h), 1, p1, fonts)
        y += card_h + gap
        _draw_card(draw, (margin, y, margin + card_w, y + card_h), 2, p2, fonts)

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
    except Exception as e:
        log.exception("Failed to render fencing styles PNG with Pillow: %s", e)
        return _render_with_cv2(predictions, profiles)


def _render_with_cv2(
    predictions: Sequence[Dict[str, Any]],
    profiles: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[bytes]:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    try:
        profiles = profiles or profiles_from_predictions(predictions)
        width, height = 720, 540
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = BG[::-1]  # BGR

        def put(text, org, color, scale=0.7, thickness=1):
            cv2.putText(
                img,
                text,
                org,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color[::-1],
                thickness,
                cv2.LINE_AA,
            )

        put("YOUR FENCING STYLES  :)", (28, 40), HEADER_FG, 0.7, 2)

        def card(y0, fencer_num, profile, accent):
            x0, x1 = 28, width - 28
            y1 = y0 + 200
            cv2.rectangle(img, (x0, y0), (x1, y1), accent[::-1], 2)
            arch = profile.get("archetype") or {}
            put(f"FENCER {fencer_num}", (x0 + 20, y0 + 32), accent, 0.55, 1)
            put(arch.get("name") or "Your style", (x0 + 20, y0 + 70), TITLE, 0.9, 2)
            blurb = (profile.get("styleBlurb") or "")[:70]
            put(blurb, (x0 + 20, y0 + 110), BODY, 0.5, 1)
            targeting = profile.get("targeting") or {}
            if targeting.get("id") != "standard" and targeting.get("name"):
                put(targeting["name"], (x0 + 20, y0 + 160), BADGE_FG, 0.55, 1)

        card(70, 1, profiles["fencer1"], F1_ACCENT)
        card(290, 2, profiles["fencer2"], F2_ACCENT)

        ok, encoded = cv2.imencode(".png", img)
        if not ok:
            return None
        return encoded.tobytes()
    except Exception as e:
        log.exception("Failed to render fencing styles PNG with OpenCV: %s", e)
        return None
