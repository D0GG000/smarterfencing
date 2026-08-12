"""OpenAI/Ollama coaching analysis from merged bout (macrobout) data.

Archetype is decided by deterministic bout thresholds; the LLM only writes practice drills.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fencer_archetype import (
    ARCHETYPE_META,
    attack_action_pcts,
    classify_user_from_macrobout,
    target_zone_pcts,
)

ARCHETYPE_PAGE_COPY = {
    "duelist": (
        "You like to take control of the bout. Rather than waiting for opportunities, "
        "you create them. Your fencing is built around initiative, commitment, and "
        "forcing your opponent to react."
    ),
    "sniper": (
        "You are a patient hunter. Instead of forcing exchanges, you wait for the "
        "smallest mistake and punish it with precision. Your strength comes from "
        "timing, distance, and knowing exactly when to strike."
    ),
    "bolt": (
        "Speed is your advantage. You rely on sudden acceleration and explosive "
        "attacks to catch opponents before they can react."
    ),
    "tactician": (
        "You solve the puzzle of the bout. Instead of relying on one favorite weapon, "
        "you adjust your approach based on what your opponent gives you."
    ),
    "hybrid": (
        "You cannot be defined by one approach. You blend different styles together, "
        "switching between attack, defense, and timing-based actions depending on "
        "what the bout demands."
    ),
}

VALID_ARCHETYPE_IDS = frozenset(ARCHETYPE_META.keys())


def _fencer_num(touch_id: str) -> Optional[int]:
    tid = (touch_id or "").lower()
    if "fencer1" in tid:
        return 1
    if "fencer2" in tid:
        return 2
    return None


def _touches_for_fencer(
    predictions: Sequence[Dict[str, Any]], fencer_key: str
) -> List[Dict[str, Any]]:
    want = 1 if fencer_key == "fencer1" else 2
    out = []
    for p in predictions or []:
        n = _fencer_num(str(p.get("touch") or ""))
        if n == want:
            out.append(p)
    return out


def _fencer_slice(predictions: Sequence[Dict[str, Any]], fencer_key: str) -> Dict[str, Any]:
    mine = _touches_for_fencer(predictions, fencer_key)
    notes = []
    for t in mine:
        note = t.get("user_note")
        if isinstance(note, str) and note.strip():
            notes.append(
                {
                    "touch": t.get("touch"),
                    "prediction": t.get("prediction"),
                    "attack_prediction": t.get("attack_prediction"),
                    "note": note.strip()[:500],
                }
            )
    return {
        "scoring_touches": len(mine),
        "attack_mix": attack_action_pcts(mine),
        "target_mix": target_zone_pcts(mine),
        "touch_notes": notes,
    }


def build_macrobout_payload(
    merged_results: Dict[str, Any], user_fencer: str
) -> Dict[str, Any]:
    """Compact bout summary for BOTH fencers (LLM input)."""
    preds = merged_results.get("predictions") or []
    opp = "fencer2" if user_fencer == "fencer1" else "fencer1"

    arm = merged_results.get("arm_attempts")
    if not isinstance(arm, dict):
        arm = {}
    pre = arm.get("pre_touch_aggressor")
    if not isinstance(pre, dict):
        pre = {}

    f1_score = sum(1 for p in preds if _fencer_num(str(p.get("touch") or "")) == 1)
    f2_score = sum(1 for p in preds if _fencer_num(str(p.get("touch") or "")) == 2)

    # Footwork initiative is pre_touch_aggressor ("who advanced more"), not strip thirds.
    return {
        "user_fencer": user_fencer,
        "opponent_fencer": opp,
        "score": {"fencer1": f1_score, "fencer2": f2_score},
        "fencer1": _fencer_slice(preds, "fencer1"),
        "fencer2": _fencer_slice(preds, "fencer2"),
        "arm_attempts": {
            "fencer1_total": arm.get("fencer1_total"),
            "fencer2_total": arm.get("fencer2_total"),
        },
        "pre_touch_aggressor": {
            k: pre.get(k)
            for k in (
                "main_footwork_aggressor",
                "fencer1_pre_touch_aggression",
                "fencer2_pre_touch_aggression",
                "even",
                "unclear",
                "touches_scored",
            )
            if k in pre
        },
    }


def macrobout_input_hash(macrobout: Dict[str, Any]) -> str:
    blob = json.dumps(macrobout, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _system_prompt() -> str:
    return (
        "You are an epee fencing coach writing directly to the athlete. "
        "The athlete's archetype_id is already decided by bout thresholds — do not change it. "
        "Write a fresh natural-language rationale for why that archetype fits THIS bout, "
        "plus targeting_notes and exactly 3 ranked practice drills (never more, never fewer). "
        "CRITICAL: Never copy meta-instructions, field names, or rule text into the athlete "
        "prose. Do not write phrases like 'do not frame', 'without noting the win', "
        "'advance_meaning', 'result_meaning', or 'threshold_facts'. "
        "FACT RULES (follow silently): "
        "(1) Use advance_summary / who_advanced_more_before_touches — if opponent advanced "
        "more, say the athlete was usually being pushed; never claim they advanced more. "
        "(2) Pre-touch advance counts cover ALL touches, not only the athlete's scores. "
        "(3) Arm attempt fields are COUNTS, not percentages. "
        "(4) Use result_summary — if the athlete won, acknowledge the win naturally; "
        "do not treat their touch count as a failed finish rate. "
        "(5) Never write fencer1, fencer2, Fencer 1, or Fencer 2 — say 'you' / 'your opponent'. "
        "Reply with a single valid JSON object only (no markdown, no trailing commas)."
    )


def _coach_view_macrobout(macrobout: Dict[str, Any], user_fencer: str) -> Dict[str, Any]:
    """Remap macrobout to you/opponent so the model cannot swap fencer1/fencer2."""
    opp = "fencer2" if user_fencer == "fencer1" else "fencer1"
    you = macrobout.get(user_fencer) or {}
    opponent = macrobout.get(opp) or {}
    score = macrobout.get("score") or {}
    arm = macrobout.get("arm_attempts") or {}
    pre = macrobout.get("pre_touch_aggressor") or {}
    if user_fencer == "fencer1":
        your_score = int(score.get("fencer1") or you.get("scoring_touches") or 0)
        opp_score = int(score.get("fencer2") or opponent.get("scoring_touches") or 0)
        your_arm = arm.get("fencer1_total")
        opp_arm = arm.get("fencer2_total")
        your_pre = pre.get("fencer1_pre_touch_aggression")
        opp_pre = pre.get("fencer2_pre_touch_aggression")
    else:
        your_score = int(score.get("fencer2") or you.get("scoring_touches") or 0)
        opp_score = int(score.get("fencer1") or opponent.get("scoring_touches") or 0)
        your_arm = arm.get("fencer2_total")
        opp_arm = arm.get("fencer1_total")
        your_pre = pre.get("fencer2_pre_touch_aggression")
        opp_pre = pre.get("fencer1_pre_touch_aggression")

    your_pre_i = int(your_pre or 0)
    opp_pre_i = int(opp_pre or 0)
    if your_pre_i > opp_pre_i:
        who = "you"
    elif opp_pre_i > your_pre_i:
        who = "opponent"
    else:
        who = "even"

    return {
        "perspective": "All fields below are already from YOUR point of view.",
        "score": {"you": your_score, "opponent": opp_score},
        "bout_result": (
            "you_won"
            if your_score > opp_score
            else ("opponent_won" if opp_score > your_score else "tied")
        ),
        "you": you,
        "opponent": opponent,
        "arm_attempt_counts": {"you": your_arm, "opponent": opp_arm},
        "pre_touch_advance_counts": {
            "you": your_pre_i,
            "opponent": opp_pre_i,
            "who_advanced_more": who,
            "definition": (
                "Count of touches (either fencer's) where that athlete advanced more "
                "in the moments before the light. Higher opponent count means you were "
                "usually being pushed."
            ),
        },
    }


def _user_prompt(macrobout: Dict[str, Any], decided: Dict[str, Any]) -> str:
    user_fencer = str(macrobout.get("user_fencer") or "fencer1")
    payload = {
        "decided_archetype": {
            "id": decided.get("archetype_id"),
            "name": decided.get("archetype_name"),
            "style": decided.get("archetype_style"),
        },
        "threshold_facts": decided.get("threshold_facts") or {},
        "bout": _coach_view_macrobout(macrobout, user_fencer),
        "instructions": {
            "rationale": (
                "REQUIRED — 2-4 natural coaching sentences to the athlete. Paraphrase "
                "advance_summary and result_summary; never quote instruction text. "
                "If who_advanced_more is opponent, describe absorbing pressure while still "
                "scoring. If you won, say so briefly. Cite concrete numbers. No meta commentary."
            ),
            "targeting_notes": (
                "REQUIRED — 1 short sentence about your target habits in this bout"
            ),
            "practice_suggestions": (
                "REQUIRED — exactly 3 drills, ranked best-first (#1 most important). "
                "Array of {title, why (one short sentence), detail (how), focus}. "
                "Speak as 'you'. Fit the decided archetype and this bout; if the opponent "
                "advanced more, prioritize dealing with pressure / countering while "
                "retreating. Each drill must be distinct. Do not return 2 or 4+ drills."
            ),
        },
    }
    return json.dumps(payload, separators=(",", ":"), default=str)


def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _repair_json_text(text: str) -> str:
    """Best-effort fixes for common small-model JSON mistakes."""
    out = text.strip()
    out = out.replace("\u201c", '"').replace("\u201d", '"')
    out = out.replace("\u2018", "'").replace("\u2019", "'")
    # Trailing commas before } or ]
    out = re.sub(r",\s*([}\]])", r"\1", out)
    # Missing commas between properties / array elements across newlines.
    out = re.sub(
        r'("(?:\\.|[^"\\])*"|true|false|null|-?\d+(?:\.\d+)?|[}\]])\s*\n(\s*")',
        r"\1,\n\2",
        out,
    )
    return out


def _parse_llm_json(raw: str) -> Dict[str, Any]:
    text = _strip_code_fences(raw)
    candidates: List[str] = []
    if text:
        candidates.append(text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        extracted = text[start : end + 1]
        if extracted not in candidates:
            candidates.append(extracted)

    errors: List[str] = []
    for cand in candidates:
        for attempt in (cand, _repair_json_text(cand)):
            try:
                data = json.loads(attempt)
            except json.JSONDecodeError as e:
                errors.append(str(e))
                continue
            if isinstance(data, dict):
                return data
            errors.append("root was not an object")
    detail = errors[-1] if errors else "empty response"
    raise ValueError(f"LLM response was not valid JSON ({detail})")


def _fix_you_caps(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"^you\b", "You", text)
    text = re.sub(r"([.!?]\s+)you\b", lambda m: m.group(1) + "You", text)
    text = re.sub(r"^they\b", "They", text)
    text = re.sub(r"([.!?]\s+)they\b", lambda m: m.group(1) + "They", text)
    text = re.sub(r"^your opponent\b", "Your opponent", text)
    text = re.sub(
        r"([.!?]\s+)your opponent\b",
        lambda m: m.group(1) + "Your opponent",
        text,
    )
    return text


def _rewrite_fencer_labels(text: str, label_for_f1: str, label_for_f2: str) -> str:
    """Replace fencer1/fencer2 wording with second-person coaching voice."""
    if not text or not isinstance(text, str):
        return ""
    out = text
    # Longer / spaced forms first.
    pairs = (
        (r"\b[Ff]encer\s*1\b", label_for_f1),
        (r"\b[Ff]encer\s*2\b", label_for_f2),
        (r"\bfencer1\b", label_for_f1),
        (r"\bfencer2\b", label_for_f2),
        (r"\bFencer1\b", label_for_f1),
        (r"\bFencer2\b", label_for_f2),
    )
    for pattern, repl in pairs:
        out = re.sub(pattern, repl, out)
    return _fix_you_caps(out)


_META_SENTENCE_RE = re.compile(
    r"(?i)\s*(?:however,?\s*)?(?:since you won,?\s*)?"
    r"do not frame your touch count as a poor finish rate"
    r"(?:\s+without noting the win)?\.?"
)
_META_PHRASE_RE = re.compile(
    r"(?i)\b(?:advance_meaning|result_meaning|advance_summary|result_summary|"
    r"threshold_facts|who_advanced_more_before_touches|bout_result)\b"
)


def _scrub_meta_instructions(text: str) -> str:
    """Remove leaked prompt/rule sentences small models sometimes paste into prose."""
    if not text:
        return ""
    out = _META_SENTENCE_RE.sub("", text)
    out = _META_PHRASE_RE.sub("", out)
    # Drop sentences that are pure instruction echo.
    kept = []
    for part in re.split(r"(?<=[.!?])\s+", out.strip()):
        low = part.lower().strip()
        if not low:
            continue
        if low.startswith("do not ") or "without noting the win" in low:
            continue
        if "do not frame" in low:
            continue
        kept.append(part.strip())
    out = " ".join(kept).strip()
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([,.!?])", r"\1", out)
    return out.strip()


def _voice_labels(user_fencer: str, for_subject: str) -> Tuple[str, str]:
    """
    Return (label_for_f1, label_for_f2) for a block about for_subject
    ('fencer1' or 'fencer2').
    """
    if for_subject == user_fencer:
        # Speak to the user.
        if user_fencer == "fencer1":
            return ("you", "your opponent")
        return ("your opponent", "you")
    # Opponent's style card, still read by the user.
    if for_subject == "fencer1":
        return ("they", "you")
    return ("you", "they")


def _normalize_suggestions(suggestions_in: Any, user_fencer: str) -> List[Dict[str, str]]:
    suggestions: List[Dict[str, str]] = []
    if not isinstance(suggestions_in, list):
        return suggestions
    l1, l2 = _voice_labels(user_fencer, user_fencer)
    for s in suggestions_in[:3]:
        if isinstance(s, str) and s.strip():
            suggestions.append(
                {
                    "title": _rewrite_fencer_labels(s.strip(), l1, l2)[:120],
                    "why": "",
                    "detail": "",
                    "focus": "",
                }
            )
        elif isinstance(s, dict):
            title = str(s.get("title") or s.get("name") or "").strip()
            if not title:
                continue
            why = str(s.get("why") or s.get("reason") or "").strip()
            suggestions.append(
                {
                    "title": _scrub_meta_instructions(
                        _rewrite_fencer_labels(title, l1, l2)
                    )[:120],
                    "why": _scrub_meta_instructions(
                        _rewrite_fencer_labels(why, l1, l2)
                    )[:300],
                    "detail": _scrub_meta_instructions(
                        _rewrite_fencer_labels(
                            str(s.get("detail") or s.get("description") or ""), l1, l2
                        )
                    )[:800],
                    "focus": _scrub_meta_instructions(
                        _rewrite_fencer_labels(str(s.get("focus") or ""), l1, l2)
                    )[:120],
                }
            )
    return suggestions


def _normalize_suggestions_response(
    data: Dict[str, Any], user_fencer: str, decided: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge rule-based archetype with LLM rationale + practice suggestions."""
    empty_opp = {
        "archetype_id": "",
        "archetype_name": "",
        "archetype_style": "",
        "rationale": "",
        "targeting_notes": "",
    }
    l1, l2 = _voice_labels(user_fencer, user_fencer)
    rationale = _scrub_meta_instructions(
        _rewrite_fencer_labels(str(data.get("rationale") or "").strip(), l1, l2)
    )[:2000]
    targeting_notes = _scrub_meta_instructions(
        _rewrite_fencer_labels(str(data.get("targeting_notes") or "").strip(), l1, l2)
    )[:1000]
    if not rationale:
        raise ValueError("LLM returned empty rationale")

    suggestions = _normalize_suggestions(data.get("practice_suggestions"), user_fencer)
    if len(suggestions) != 3:
        raise ValueError(
            "LLM must return exactly 3 practice_suggestions (got %d)" % len(suggestions)
        )

    user_block = {
        "archetype_id": decided["archetype_id"],
        "archetype_name": decided["archetype_name"],
        "archetype_style": decided["archetype_style"],
        "rationale": rationale,
        "targeting_notes": targeting_notes,
    }

    if user_fencer == "fencer1":
        f1, f2 = user_block, empty_opp
    else:
        f1, f2 = empty_opp, user_block

    return {
        "fencer1": f1,
        "fencer2": f2,
        "practice_suggestions": suggestions,
        "archetype_id": user_block["archetype_id"],
        "archetype_name": user_block["archetype_name"],
        "archetype_style": user_block["archetype_style"],
        "rationale": user_block["rationale"],
        "targeting_notes": user_block["targeting_notes"],
        "archetype_source": "rules",
        "rationale_source": "llm",
    }


def _llm_endpoint_config() -> Dict[str, Any]:
    base = (os.environ.get("OPENAI_BASE_URL") or "").strip().rstrip("/")
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    local = bool(base) and (
        "localhost" in base or "127.0.0.1" in base or "0.0.0.0" in base
    )

    if not base:
        base = "https://api.openai.com/v1"
        local = False

    if not api_key:
        if local:
            api_key = "ollama"
        else:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured "
                "(or set OPENAI_BASE_URL to a local server like Ollama)"
            )

    default_model = "llama3.2:3b" if local else "gpt-4o-mini"
    model = (os.environ.get("OPENAI_MODEL") or default_model).strip()
    use_json_object = (
        os.environ.get("OPENAI_JSON_OBJECT") or ("0" if local else "1")
    ).strip() == "1"
    timeout = float(os.environ.get("OPENAI_TIMEOUT_SEC") or (180 if local else 90))
    return {
        "base": base,
        "api_key": api_key,
        "model": model,
        "local": local,
        "use_json_object": use_json_object,
        "timeout": timeout,
    }


def call_openai_bout_analysis(
    macrobout: Dict[str, Any], user_fencer: str, decided: Dict[str, Any]
) -> Dict[str, Any]:
    """Call the LLM for practice suggestions only; retry until valid JSON."""
    cfg = _llm_endpoint_config()
    max_attempts = max(1, int(os.environ.get("OPENAI_MAX_ATTEMPTS") or 8))
    last_err: Optional[BaseException] = None

    for attempt in range(1, max_attempts + 1):
        try:
            return _call_openai_suggestions_once(
                cfg,
                macrobout,
                user_fencer,
                decided,
                attempt=attempt,
                last_err=last_err,
            )
        except (ValueError, RuntimeError, json.JSONDecodeError, KeyError, TypeError) as e:
            last_err = e
            if attempt >= max_attempts:
                raise RuntimeError(
                    f"LLM failed after {max_attempts} attempts: {e}"
                ) from e
            time.sleep(min(0.4 * attempt, 2.0))

    raise RuntimeError("LLM failed with no attempts")  # unreachable


def _call_openai_suggestions_once(
    cfg: Dict[str, Any],
    macrobout: Dict[str, Any],
    user_fencer: str,
    decided: Dict[str, Any],
    *,
    attempt: int,
    last_err: Optional[BaseException],
) -> Dict[str, Any]:
    repair_note = ""
    if last_err is not None:
        repair_note = (
            "\n\nYour previous reply was invalid ("
            + str(last_err)[:180]
            + "). Reply again with ONLY one compact valid JSON object, "
            "no markdown, no trailing commas, double-quoted keys/strings, "
            "keys rationale, targeting_notes, practice_suggestions "
            "(practice_suggestions must be an array of exactly 3 drills)."
        )

    user_content = (
        "The athlete's archetype is already decided ("
        + str(decided.get("archetype_id"))
        + "). Do NOT choose a new archetype. "
        "Use threshold_facts (advance_summary + result_summary) and bout. "
        "If who_advanced_more_before_touches is 'opponent', the athlete was being pushed — "
        "do not claim they advanced more. "
        "Write athlete-facing prose only — never paste rules or 'do not …' instructions. "
        "Never say fencer1/fencer2 in prose — use 'you' / 'your opponent'. "
        "Reply with ONLY one JSON object (no markdown) with keys: "
        "rationale, targeting_notes, "
        "practice_suggestions (array of exactly 3 objects: {title, why, detail, focus}, "
        "ranked #1 most important). Do not return fewer or more than 3 drills.\n\n"
        + _user_prompt(macrobout, decided)
        + repair_note
    )
    temperature = 0.15 + min(0.35, 0.08 * (attempt - 1))
    body: Dict[str, Any] = {
        "model": cfg["model"],
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": user_content},
        ],
    }
    if cfg["use_json_object"]:
        body["response_format"] = {"type": "json_object"}

    url = f"{cfg['base']}/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg['api_key']}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg["timeout"]) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")[:500]
        if cfg["use_json_object"] and e.code in (400, 404, 422):
            body.pop("response_format", None)
            req2 = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {cfg['api_key']}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req2, timeout=cfg["timeout"]) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
            except Exception as e2:
                raise RuntimeError(
                    f"LLM HTTP {e.code}: {err_body}; retry failed: {e2}"
                ) from e2
        else:
            raise RuntimeError(f"LLM HTTP {e.code}: {err_body}") from e
    except urllib.error.URLError as e:
        hint = (
            " Is Ollama running? Try: ollama serve && ollama pull llama3.2:3b"
            if cfg["local"]
            else ""
        )
        raise RuntimeError(f"LLM request failed: {e}.{hint}") from e

    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError("Unexpected LLM response shape") from e

    parsed = _parse_llm_json(content)
    return _normalize_suggestions_response(parsed, user_fencer, decided)


def run_bout_llm_analysis(
    merged_results: Dict[str, Any], user_fencer: str
) -> Dict[str, Any]:
    if user_fencer not in ("fencer1", "fencer2"):
        raise ValueError("user_fencer must be fencer1 or fencer2")
    macrobout = build_macrobout_payload(merged_results, user_fencer)
    decided = classify_user_from_macrobout(macrobout, user_fencer)

    try:
        analysis = call_openai_bout_analysis(macrobout, user_fencer, decided)
    except Exception as e:
        # Rule-based archetype still returns; drills empty after retries exhausted.
        empty_opp = {
            "archetype_id": "",
            "archetype_name": "",
            "archetype_style": "",
            "rationale": "",
            "targeting_notes": "",
        }
        user_block = {
            "archetype_id": decided["archetype_id"],
            "archetype_name": decided["archetype_name"],
            "archetype_style": decided["archetype_style"],
            "rationale": decided.get("rationale") or "",
            "targeting_notes": decided.get("targeting_notes") or "",
        }
        if user_fencer == "fencer1":
            f1, f2 = user_block, empty_opp
        else:
            f1, f2 = empty_opp, user_block
        analysis = {
            "fencer1": f1,
            "fencer2": f2,
            "practice_suggestions": [],
            "archetype_id": user_block["archetype_id"],
            "archetype_name": user_block["archetype_name"],
            "archetype_style": user_block["archetype_style"],
            "rationale": user_block["rationale"],
            "targeting_notes": user_block["targeting_notes"],
            "archetype_source": "rules",
            "suggestions_error": str(e)[:300],
        }

    input_hash = macrobout_input_hash(macrobout)
    return {
        "user_fencer": user_fencer,
        "input_hash": input_hash,
        "analyzed_at": datetime.utcnow().isoformat() + "Z",
        "macrobout": macrobout,
        "analysis": analysis,
    }


def parse_stored_llm_analysis(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def llm_analysis_is_stale(
    stored: Optional[Dict[str, Any]], merged_results: Dict[str, Any]
) -> bool:
    if not stored or not isinstance(stored, dict):
        return True
    user_fencer = stored.get("user_fencer") or merged_results.get("user_fencer")
    if user_fencer not in ("fencer1", "fencer2"):
        return True
    analysis = stored.get("analysis") or {}
    if not isinstance(analysis, dict):
        return True
    # Prefer top-level user archetype; fall back to user slot from dual responses.
    has_user = bool(analysis.get("archetype_id"))
    if not has_user:
        block = analysis.get(user_fencer)
        has_user = isinstance(block, dict) and bool(block.get("archetype_id"))
    if not has_user:
        return True
    current = build_macrobout_payload(merged_results, user_fencer)
    return stored.get("input_hash") != macrobout_input_hash(current)
