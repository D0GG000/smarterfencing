"""
Fencer style archetypes + targeting signatures from bout touch predictions.
Mirrors app/static/fencer-archetype.js (keep rules in sync).
MVP: model attack label "other" is treated as Counter.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

ACTION_KEYS = ("lunge", "counter", "fleche", "other")
TARGET_KEYS = ("chest", "abdomen", "arm", "leg")

ARCHETYPE_META = {
    "duelist": {
        "id": "duelist",
        "name": "The Duelist",
        "style": "Attack-focused fencer who creates opportunities through direct attacks.",
        "emoji": "⚔️",
    },
    "sniper": {
        "id": "sniper",
        "name": "The Sniper",
        "style": "Precision fencer who waits for openings and scores through counters.",
        "emoji": "🎯",
    },
    "bolt": {
        "id": "bolt",
        "name": "The Bolt",
        "style": "Explosive fencer who relies on fast, decisive attacks.",
        "emoji": "⚡",
    },
    "tactician": {
        "id": "tactician",
        "name": "The Tactician",
        "style": "Balanced fencer who scores through a varied mix of actions.",
        "emoji": "♟️",
    },
    "hybrid": {
        "id": "hybrid",
        "name": "The Hybrid",
        "style": "A fencer who combines two major scoring approaches rather than relying on one.",
        "emoji": "⚖️",
    },
}

TARGETING_META = {
    "arm_hunter": {"id": "arm_hunter", "name": "Arm Hunter"},
    "body_hunter": {"id": "body_hunter", "name": "Body Hunter"},
    "leg_hunter": {"id": "leg_hunter", "name": "Leg Hunter"},
    "standard": {"id": "standard", "name": "Standard Target Profile"},
}


def round_pct(n: int, total: int) -> int:
    if not total:
        return 0
    return int(round((n / total) * 100))


def normalize_attack_label(label: Any) -> Optional[str]:
    if not label:
        return None
    s = str(label).lower()
    if s == "other":
        return "counter"
    if s in ("counter", "lunge", "fleche"):
        return s
    return None


def attack_action_pcts(touches: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {"lunge": 0, "counter": 0, "fleche": 0, "other": 0}
    classified = 0
    for t in touches or []:
        mapped = normalize_attack_label(t.get("attack_prediction"))
        if not mapped:
            continue
        classified += 1
        counts[mapped] += 1
    pcts = {k: round_pct(counts[k], classified) for k in ACTION_KEYS}
    return {"counts": counts, "pcts": pcts, "classifiedCount": classified}


def target_zone_pcts(touches: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {"chest": 0, "abdomen": 0, "arm": 0, "leg": 0}
    classified = 0
    for t in touches or []:
        p = t.get("prediction")
        if not p:
            continue
        p = str(p).lower()
        if p not in counts:
            continue
        counts[p] += 1
        classified += 1
    pcts = {k: round_pct(counts[k], classified) for k in TARGET_KEYS}
    return {"counts": counts, "pcts": pcts, "classifiedCount": classified}


def _top_two_actions(pcts: Dict[str, int]) -> List[Dict[str, Any]]:
    ranked = sorted(
        [{"key": k, "pct": pcts.get(k, 0)} for k in ACTION_KEYS],
        key=lambda x: x["pct"],
        reverse=True,
    )
    return ranked[:2]


def _any_action_exceeds(pcts: Dict[str, int], limit: int) -> bool:
    return any((pcts.get(k) or 0) > limit for k in ACTION_KEYS)


def _fallback_from_top_action(pcts: Dict[str, int]) -> Dict[str, str]:
    """When no rule matches, pick from the leading action — never a dump archetype."""
    ranked = sorted(
        (("lunge", "duelist"), ("counter", "sniper"), ("fleche", "bolt")),
        key=lambda pair: pcts.get(pair[0], 0),
        reverse=True,
    )
    best_key, best_id = ranked[0]
    if (pcts.get(best_key) or 0) > 0:
        return ARCHETYPE_META[best_id]
    return ARCHETYPE_META["hybrid"]


def _pct(pcts: Dict[str, int], key: str) -> int:
    return int(pcts.get(key) or 0)


def classify_archetype(action_info: Dict[str, Any]) -> Dict[str, str]:
    """Attack-mix only classification (legacy / single-fencer)."""
    return classify_archetype_comparative(action_info, None, None)


def classify_archetype_comparative(
    user_action: Dict[str, Any],
    opp_action: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Deterministic archetype from user attack mix plus optional bout context:
    opponent attack mix, arm-attempt totals, pre-touch advance wins.
    """
    pcts = user_action.get("pcts") or {}
    opp_pcts = (opp_action or {}).get("pcts") or {}
    ctx = context or {}

    user_arm = int(ctx.get("user_arm") or 0)
    opp_arm = int(ctx.get("opp_arm") or 0)
    user_pre = int(ctx.get("user_pre") or 0)
    opp_pre = int(ctx.get("opp_pre") or 0)
    advances_more = user_pre > opp_pre
    retreats_more = opp_pre > user_pre
    presses_blade = user_arm > opp_arm

    lunge = _pct(pcts, "lunge")
    counter = _pct(pcts, "counter")
    fleche = _pct(pcts, "fleche")
    other = _pct(pcts, "other")
    opp_lunge = _pct(opp_pcts, "lunge")
    opp_counter = _pct(opp_pcts, "counter")

    # Primary absolute thresholds (stable, bout-local).
    if lunge >= 57:
        return ARCHETYPE_META["duelist"]
    if counter >= 40:
        return ARCHETYPE_META["sniper"]
    if fleche >= 23:
        return ARCHETYPE_META["bolt"]

    # Comparative thresholds: same signals the LLM used to see.
    if lunge >= 45 and advances_more and (presses_blade or lunge >= opp_lunge):
        return ARCHETYPE_META["duelist"]
    if counter >= 30 and (retreats_more or counter > opp_counter):
        return ARCHETYPE_META["sniper"]
    if fleche >= 15 and fleche >= lunge and fleche >= counter:
        return ARCHETYPE_META["bolt"]
    if lunge >= 40 and advances_more and presses_blade:
        return ARCHETYPE_META["duelist"]

    tactician = (
        20 <= lunge <= 55
        and 20 <= counter <= 50
        and 10 <= fleche <= 35
        and other < 20
        and not _any_action_exceeds(pcts, 55)
    )
    if tactician:
        return ARCHETYPE_META["tactician"]

    top = _top_two_actions(pcts)
    if (
        len(top) == 2
        and abs(top[0]["pct"] - top[1]["pct"]) <= 10
        and top[0]["pct"] + top[1]["pct"] >= 65
        and top[0]["key"] != "other"
    ):
        return ARCHETYPE_META["hybrid"]

    return _fallback_from_top_action(pcts)


def classify_targeting(target_info: Dict[str, Any]) -> Dict[str, str]:
    pcts = target_info["pcts"]
    if target_info["classifiedCount"] <= 0:
        return TARGETING_META["standard"]
    if (pcts.get("arm") or 0) >= 28:
        return TARGETING_META["arm_hunter"]
    if (pcts.get("abdomen") or 0) >= 18:
        return TARGETING_META["body_hunter"]
    if (pcts.get("leg") or 0) >= 13:
        return TARGETING_META["leg_hunter"]
    return TARGETING_META["standard"]


def _threshold_facts(
    archetype_id: str,
    user_action: Dict[str, Any],
    opp_action: Dict[str, Any],
    targeting: Dict[str, str],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Machine-readable facts for the LLM to turn into natural prose (not shown as-is)."""
    pcts = user_action.get("pcts") or {}
    opp_pcts = (opp_action or {}).get("pcts") or {}
    ctx = context or {}
    user_pre = int(ctx.get("user_pre") or 0)
    opp_pre = int(ctx.get("opp_pre") or 0)
    user_arm = int(ctx.get("user_arm") or 0)
    opp_arm = int(ctx.get("opp_arm") or 0)
    user_score = int(ctx.get("user_score") or 0)
    opp_score = int(ctx.get("opp_score") or 0)
    if user_pre > opp_pre:
        who_advanced = "you"
        advance_summary = (
            f"You advanced more often before touches ({user_pre} vs opponent {opp_pre})."
        )
    elif opp_pre > user_pre:
        who_advanced = "opponent"
        advance_summary = (
            f"Your opponent advanced more often before touches "
            f"({opp_pre} vs you {user_pre}); you were usually being pushed."
        )
    else:
        who_advanced = "even"
        advance_summary = (
            f"Pre-touch advances were even (you {user_pre}, opponent {opp_pre})."
        )

    if user_score > opp_score:
        bout_result = "you_won"
        result_summary = f"You won the bout {user_score}-{opp_score}."
    elif opp_score > user_score:
        bout_result = "opponent_won"
        result_summary = f"Your opponent won the bout {opp_score}-{user_score}."
    else:
        bout_result = "tied"
        result_summary = f"Scoring touches were tied {user_score}-{opp_score}."

    return {
        "chosen_archetype_id": archetype_id,
        "your_attack_pcts": {
            "lunge": _pct(pcts, "lunge"),
            "counter": _pct(pcts, "counter"),
            "fleche": _pct(pcts, "fleche"),
            "other": _pct(pcts, "other"),
        },
        "opponent_attack_pcts": {
            "lunge": _pct(opp_pcts, "lunge"),
            "counter": _pct(opp_pcts, "counter"),
            "fleche": _pct(opp_pcts, "fleche"),
            "other": _pct(opp_pcts, "other"),
        },
        # Counts (not rates / percentages).
        "your_arm_attempt_count": user_arm,
        "opponent_arm_attempt_count": opp_arm,
        "your_pre_touch_advance_count": user_pre,
        "opponent_pre_touch_advance_count": opp_pre,
        "who_advanced_more_before_touches": who_advanced,
        "advance_summary": advance_summary,
        "your_scoring_touches": user_score,
        "opponent_scoring_touches": opp_score,
        "bout_result": bout_result,
        "result_summary": result_summary,
        "targeting_id": targeting.get("id"),
        "targeting_name": targeting.get("name"),
        "your_classified_scoring_touches": int(user_action.get("classifiedCount") or 0),
        # Keep legacy keys briefly for older prompt caches / debugging.
        "user_arm_attempts": user_arm,
        "opponent_arm_attempts": opp_arm,
        "user_pre_touch_advance_wins": user_pre,
        "opponent_pre_touch_advance_wins": opp_pre,
        "who_advanced_more": who_advanced,
        "classified_touches": int(user_action.get("classifiedCount") or 0),
    }


def classify_user_from_macrobout(
    macrobout: Dict[str, Any], user_fencer: str
) -> Dict[str, Any]:
    """Build the user's rule-based archetype block from macrobout payload."""
    opp = "fencer2" if user_fencer == "fencer1" else "fencer1"
    user = macrobout.get(user_fencer) or {}
    opponent = macrobout.get(opp) or {}
    user_action = user.get("attack_mix") or {
        "pcts": {},
        "counts": {},
        "classifiedCount": 0,
    }
    opp_action = opponent.get("attack_mix") or {
        "pcts": {},
        "counts": {},
        "classifiedCount": 0,
    }
    user_targets = user.get("target_mix") or {
        "pcts": {},
        "counts": {},
        "classifiedCount": 0,
    }

    arm = macrobout.get("arm_attempts") or {}
    pre = macrobout.get("pre_touch_aggressor") or {}
    score = macrobout.get("score") or {}
    if user_fencer == "fencer1":
        user_score = int(score.get("fencer1") or 0)
        opp_score = int(score.get("fencer2") or 0)
    else:
        user_score = int(score.get("fencer2") or 0)
        opp_score = int(score.get("fencer1") or 0)
    # Prefer per-fencer scoring_touches from slices when present.
    user_score = int((user.get("scoring_touches") if user.get("scoring_touches") is not None else user_score) or 0)
    opp_score = int(
        (opponent.get("scoring_touches") if opponent.get("scoring_touches") is not None else opp_score) or 0
    )
    context = {
        "user_arm": arm.get("fencer1_total" if user_fencer == "fencer1" else "fencer2_total")
        or 0,
        "opp_arm": arm.get("fencer2_total" if user_fencer == "fencer1" else "fencer1_total")
        or 0,
        "user_pre": pre.get(
            "fencer1_pre_touch_aggression"
            if user_fencer == "fencer1"
            else "fencer2_pre_touch_aggression"
        )
        or 0,
        "opp_pre": pre.get(
            "fencer2_pre_touch_aggression"
            if user_fencer == "fencer1"
            else "fencer1_pre_touch_aggression"
        )
        or 0,
        "user_score": user_score,
        "opp_score": opp_score,
    }

    archetype = classify_archetype_comparative(user_action, opp_action, context)
    targeting = classify_targeting(user_targets)
    facts = _threshold_facts(
        archetype["id"], user_action, opp_action, targeting, context
    )
    return {
        "archetype_id": archetype["id"],
        "archetype_name": archetype["name"],
        "archetype_style": archetype["style"],
        # Filled by LLM — leave empty so UI does not show canned copy.
        "rationale": "",
        "targeting_notes": "",
        "targeting_id": targeting["id"],
        "targeting_name": targeting["name"],
        "threshold_facts": facts,
    }


def classify_fencer_profile(touches: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    action_info = attack_action_pcts(touches)
    target_info = target_zone_pcts(touches)
    archetype = classify_archetype(action_info)
    targeting = classify_targeting(target_info)
    return {
        "archetype": archetype,
        "targeting": targeting,
        "styleBlurb": archetype["style"],
        "actionPcts": action_info["pcts"],
        "actionCounts": action_info["counts"],
        "targetPcts": target_info["pcts"],
        "targetCounts": target_info["counts"],
        "classifiedCount": action_info["classifiedCount"],
        "targetClassifiedCount": target_info["classifiedCount"],
    }


def split_predictions_by_fencer(
    predictions: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    f1: List[Dict[str, Any]] = []
    f2: List[Dict[str, Any]] = []
    for p in predictions or []:
        touch = str(p.get("touch") or "")
        if "fencer1" in touch:
            f1.append(p)
        else:
            f2.append(p)
    return f1, f2


def profiles_from_predictions(
    predictions: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    f1, f2 = split_predictions_by_fencer(predictions)
    return {
        "fencer1": classify_fencer_profile(f1),
        "fencer2": classify_fencer_profile(f2),
    }
