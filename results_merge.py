"""Merge model touch predictions with user-saved corrections."""

import json

TOUCH_LABELS = frozenset({"chest", "abdomen", "arm", "leg"})
ATTACK_LABELS = frozenset({"lunge", "fleche", "other"})
TOUCH_NOTE_MAX_LEN = 4000
USER_FENCER_VALUES = frozenset({"fencer1", "fencer2"})

# Bout-level fields users can override (overlay onto pipeline output).
# Strip thirds / pressing / main_aggressor were replaced by pre_touch_aggressor.
ARM_MACRO_INT_KEYS = ("fencer1_total", "fencer2_total")
PRE_TOUCH_MACRO_INT_KEYS = (
    "fencer1_pre_touch_aggression",
    "fencer2_pre_touch_aggression",
    "even",
    "unclear",
    "touches_scored",
)
PRE_TOUCH_MACRO_STR_KEYS = ("main_footwork_aggressor",)
FOOTWORK_AGGRESSOR_VALUES = frozenset({"fencer1", "fencer2", "even"})


def normalize_touch_note(value):
    """Return stripped note text for storage, or empty string if cleared/invalid."""
    if value is None:
        return ""
    if not isinstance(value, str):
        return ""
    return value.strip()[:TOUCH_NOTE_MAX_LEN]


def normalize_correction_prediction(value):
    if not value or not isinstance(value, str):
        return None
    v = value.strip().lower()
    return v if v in TOUCH_LABELS else None


def normalize_correction_attack(value):
    if not value or not isinstance(value, str):
        return None
    v = value.strip().lower()
    return v if v in ATTACK_LABELS else None


def parse_user_fencer(selections_json_str):
    """Return 'fencer1' | 'fencer2' | None from selections_json."""
    if not selections_json_str:
        return None
    try:
        sel = json.loads(selections_json_str)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(sel, dict):
        return None
    uf = sel.get("user_fencer")
    if isinstance(uf, str) and uf.strip().lower() in USER_FENCER_VALUES:
        return uf.strip().lower()
    return None


def _parse_json_obj(raw):
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _as_nonneg_int(value):
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n >= 0 else None


def normalize_macro_corrections(data):
    """
    Validate and return a cleaned macro-corrections dict for storage.
    Unknown keys are ignored. Empty result means clear all overlays.
    """
    if not isinstance(data, dict):
        return {}

    out = {}
    arm_in = data.get("arm_attempts")
    if isinstance(arm_in, dict):
        arm = {}
        for k in ARM_MACRO_INT_KEYS:
            if k in arm_in:
                n = _as_nonneg_int(arm_in.get(k))
                if n is not None:
                    arm[k] = n
        pre_in = arm_in.get("pre_touch_aggressor")
        if isinstance(pre_in, dict):
            pre = {}
            for k in PRE_TOUCH_MACRO_INT_KEYS:
                if k in pre_in:
                    n = _as_nonneg_int(pre_in.get(k))
                    if n is not None:
                        pre[k] = n
            if "main_footwork_aggressor" in pre_in:
                v = pre_in.get("main_footwork_aggressor")
                if (
                    isinstance(v, str)
                    and v.strip().lower() in FOOTWORK_AGGRESSOR_VALUES
                ):
                    pre["main_footwork_aggressor"] = v.strip().lower()
            if pre:
                arm["pre_touch_aggressor"] = pre
        if arm:
            out["arm_attempts"] = arm

    return out


def _apply_macro_overlays(out, macro):
    """Apply validated macro overlays onto a results dict copy."""
    if not macro:
        return out

    arm_ov = macro.get("arm_attempts")
    if isinstance(arm_ov, dict) and arm_ov:
        base_arm = out.get("arm_attempts")
        arm = dict(base_arm) if isinstance(base_arm, dict) else {}
        for k in ARM_MACRO_INT_KEYS:
            if k in arm_ov:
                arm[k] = arm_ov[k]
        pre_ov = arm_ov.get("pre_touch_aggressor")
        if isinstance(pre_ov, dict) and pre_ov:
            base_pre = arm.get("pre_touch_aggressor")
            pre = dict(base_pre) if isinstance(base_pre, dict) else {}
            for k, v in pre_ov.items():
                # Never overwrite per-touch maps from macro editor.
                if k == "by_touch":
                    continue
                pre[k] = v
            pre["macro_edited"] = True
            arm["pre_touch_aggressor"] = pre
        arm["macro_edited"] = True
        out["arm_attempts"] = arm

    return out


def merge_results_payload(
    results_dict,
    corrections_json_str,
    touch_deletions_json_str=None,
    macro_corrections_json_str=None,
    selections_json_str=None,
):
    """
    Return a copy of results_dict with predictions merged from prediction_corrections_json.
    Touches listed in touch_deletions_json_str are omitted from predictions.
    Each prediction may gain model_prediction, prediction_edited,
    model_attack_prediction, attack_prediction_edited.
    Macro corrections overlay bout-level arm-attempt / pre-touch (who advanced more) rollups.
    """
    if not results_dict:
        results_dict = {}

    corrections = _parse_json_obj(corrections_json_str)

    deleted_touches = set()
    if touch_deletions_json_str:
        try:
            raw_del = json.loads(touch_deletions_json_str)
            if isinstance(raw_del, list):
                deleted_touches = {str(x) for x in raw_del if isinstance(x, str)}
        except (json.JSONDecodeError, TypeError):
            deleted_touches = set()

    predictions = results_dict.get("predictions") or []
    merged_preds = []
    for p in predictions:
        item = dict(p)
        touch = item.get("touch")
        if touch and touch in deleted_touches:
            continue
        edited = False
        attack_edited = False
        if touch and touch in corrections:
            cor = corrections[touch] or {}
            user_pred = cor.get("prediction")
            if user_pred in TOUCH_LABELS:
                item["model_prediction"] = item.get("prediction")
                item["prediction"] = user_pred
                edited = True
            user_attack = cor.get("attack_prediction")
            if user_attack in ATTACK_LABELS:
                item["model_attack_prediction"] = item.get("attack_prediction")
                item["attack_prediction"] = user_attack
                attack_edited = True
            note_text = cor.get("note")
            if isinstance(note_text, str) and note_text.strip():
                item["user_note"] = note_text.strip()
        item["prediction_edited"] = edited
        item["attack_prediction_edited"] = attack_edited
        merged_preds.append(item)

    out = dict(results_dict)
    out["predictions"] = merged_preds
    out["deleted_touch_ids"] = sorted(deleted_touches)

    macro = normalize_macro_corrections(_parse_json_obj(macro_corrections_json_str))
    out = _apply_macro_overlays(out, macro)
    out["macro_corrections"] = macro or None

    user_fencer = parse_user_fencer(selections_json_str)
    out["user_fencer"] = user_fencer

    return out
