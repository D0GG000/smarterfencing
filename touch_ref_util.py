"""Human-readable touch reference IDs (e.g. ABC123-01) scoped per analysis job."""

import re

_FRAME_RE = re.compile(r"frame(\d+)", re.I)
_REF_RE = re.compile(r"^[A-Z0-9]{4,8}-\d{2,3}$")


def frame_num_from_touch(touch_key):
    if not touch_key:
        return 0
    m = _FRAME_RE.search(touch_key)
    return int(m.group(1)) if m else 0


def job_touch_prefix(job_id):
    """Short uppercase prefix from job id (12-char hex)."""
    raw = (job_id or "unknown").strip().upper()
    return raw[:6] if len(raw) >= 6 else raw.ljust(6, "0")[:6]


def format_touch_ref(prefix, seq):
    return f"{prefix}-{int(seq):02d}"


def assign_touch_refs(predictions, job_id):
    """
    Ensure each prediction has a stable touch_ref. Existing refs are kept;
    missing refs get the next free sequence number (ordered by frame).
    Mutates prediction dicts in place and returns predictions.
    """
    if not predictions:
        return predictions

    prefix = job_touch_prefix(job_id)
    used_nums = set()

    for p in predictions:
        ref = p.get("touch_ref")
        if not ref or not isinstance(ref, str):
            continue
        ref = ref.strip().upper()
        p["touch_ref"] = ref
        m = re.match(rf"^{re.escape(prefix)}-(\d+)$", ref)
        if m:
            used_nums.add(int(m.group(1)))

    missing = [
        p
        for p in predictions
        if p.get("touch") and not p.get("touch_ref")
    ]
    missing.sort(key=lambda p: frame_num_from_touch(p.get("touch")))

    next_num = 1
    for p in missing:
        while next_num in used_nums:
            next_num += 1
        p["touch_ref"] = format_touch_ref(prefix, next_num)
        used_nums.add(next_num)
        next_num += 1

    return predictions


def predictions_missing_touch_ref(predictions):
    if not predictions:
        return False
    return any(p.get("touch") and not p.get("touch_ref") for p in predictions)


def find_prediction_by_touch_ref(predictions, touch_ref):
    if not touch_ref or not predictions:
        return None
    key = touch_ref.strip().upper()
    for p in predictions:
        if (p.get("touch_ref") or "").strip().upper() == key:
            return p
    return None


def find_prediction_by_touch_key(predictions, touch_key):
    if not touch_key:
        return None
    for p in predictions or []:
        if p.get("touch") == touch_key:
            return p
    return None


def touch_display_summary(pred, job_id=None, fps=30):
    """One-line label: ref · fencer · time."""
    if not pred:
        return "Touch"
    ref = pred.get("touch_ref")
    touch_key = pred.get("touch") or ""
    fencer = (
        "Fencer 1"
        if "fencer1" in touch_key
        else "Fencer 2"
        if "fencer2" in touch_key
        else ""
    )
    frame = frame_num_from_touch(touch_key)
    t = frame / (fps or 30)
    m = int(t // 60)
    s = int(t % 60)
    ms = int((t % 1) * 100)
    time_str = f"{m}:{s:02d}.{ms:02d}"
    parts = []
    if ref:
        parts.append(ref)
    if fencer:
        parts.append(fencer)
    parts.append(f"@{time_str}")
    return " · ".join(parts)
