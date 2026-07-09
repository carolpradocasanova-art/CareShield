"""Shared symptom extraction and report-linking helpers."""

import re

# Specific subtypes are listed before broad fallbacks within each clinical family.
SYMPTOM_PATTERN_DEFINITIONS = (
    ("confusion", r"confus", "Confusion"),
    ("fall", r"\bfell\b|\bfall\b|\bfallen\b", "Falls"),
    ("fever", r"fever|temperature", "Fever"),
    (
        "swelling_angioedema",
        r"(?:facial|face|lip|lips|tongue|throat|eyelid|angioedema).{0,40}(?:swell|swollen|swelling)"
        r"|(?:swell|swollen|swelling).{0,40}(?:facial|face|lip|lips|tongue|throat|eyelid)"
        r"|\bangioedema\b",
        "Facial or lip swelling (possible allergic)",
    ),
    (
        "swelling_peripheral",
        r"(?:ankle|ankles|leg|legs|foot|feet|peripheral|pitting).{0,40}(?:swell|swollen|swelling|edema|oedema|puffiness)"
        r"|(?:swell|swollen|swelling|edema|oedema|puffiness).{0,40}(?:ankle|ankles|leg|legs|foot|feet)"
        r"|fluid retention|peripheral edema|peripheral oedema",
        "Leg or ankle swelling",
    ),
    ("blood_pressure", r"blood pressure|\bbp\b", "Blood pressure"),
    ("pain_chest", r"chest pain|chest ache|chest discomfort|chest tightness|tight chest|tightness in (?:his|her|their|the)?\s*chest|angina", "Chest pain"),
    ("pain_abdominal", r"abdominal pain|stomach pain|belly pain|tummy pain|abdominal ache", "Abdominal pain"),
    (
        "pain_joint",
        r"(?:knee|hip|shoulder|elbow|ankle|wrist|joint).{0,30}(?:pain|ache|stiff)"
        r"|(?:pain|ache|stiff).{0,30}(?:knee|hip|shoulder|elbow|ankle|wrist|joint)",
        "Joint pain",
    ),
    ("pain_head", r"headache|head ache|migraine", "Headache"),
    ("pain_general", r"\bpain\b|ache| hurting", "Pain"),
    ("nausea", r"nause|vomit", "Nausea or vomiting"),
    ("breathing", r"breath|breathing|wheez|shortness of breath", "Breathing difficulty"),
    ("rash_allergic", r"hives|urticaria|allergic reaction|allergic rash", "Allergic rash or hives"),
    ("rash_skin", r"\brash\b|skin reaction|dermatitis|eruption", "Rash"),
    ("dizziness", r"dizz|light.?headed|vertigo", "Dizziness"),
    ("fatigue", r"fatigue|tired|letharg|weakness|exhausted", "Fatigue"),
)

_SYMPTOM_LINK_STOPWORDS = frozenset({
    "this", "that", "with", "have", "been", "patient", "report", "about", "worse",
    "worsening", "getting", "today", "yesterday", "morning", "evening", "night",
    "said", "says", "they", "their", "them", "very", "more", "much", "some",
})

_SYMPTOM_FAMILY_EXCLUSIONS = {
    "pain_general": ("pain_chest", "pain_abdominal", "pain_joint", "pain_head"),
    "rash_skin": ("rash_allergic",),
    "breathing": ("pain_chest",),
}

_REASSURING_BREATHING_CLAUSE_RE = re.compile(
    r"(?:can still|still can|able to)\s+breathe|"
    r"breath(?:ing)?\s+(?:is\s+)?(?:fine|ok|okay|normal|alright|good)|"
    r"no\s+(?:trouble|difficulty|problem|issues?)\s+(?:with\s+)?breath|"
    r"breathing\s+(?:normally|well)",
    re.I,
)

# Legacy stored incident keys from before subtype split — never used for cross-family matching.
_LEGACY_GENERIC_SYMPTOM_KEYS = frozenset({"swelling", "rash", "pain"})

SESSION_RECURRENCE_PATTERN_DEFINITIONS = (
    ("confusion", r"confus", "recurrent confusion"),
    ("fall", r"\bfell\b|\bfall\b|\bfallen\b", "recurrent falls"),
    ("fever", r"fever|temperature", "recurrent fever"),
    (
        "swelling_angioedema",
        r"(?:facial|face|lip|lips|tongue|throat|eyelid|angioedema).{0,40}(?:swell|swollen|swelling)"
        r"|(?:swell|swollen|swelling).{0,40}(?:facial|face|lip|lips|tongue|throat|eyelid)"
        r"|\bangioedema\b",
        "recurrent facial or lip swelling",
    ),
    (
        "swelling_peripheral",
        r"(?:ankle|ankles|leg|legs|foot|feet|peripheral|pitting).{0,40}(?:swell|swollen|swelling|edema|oedema|puffiness)"
        r"|(?:swell|swollen|swelling|edema|oedema|puffiness).{0,40}(?:ankle|ankles|leg|legs|foot|feet)"
        r"|fluid retention|peripheral edema|peripheral oedema",
        "recurrent peripheral swelling",
    ),
    ("blood_pressure", r"blood pressure|\bbp\b", "recurrent blood pressure concerns"),
    ("pain_chest", r"chest pain|chest ache|chest discomfort|chest tightness|tight chest|tightness in (?:his|her|their|the)?\s*chest|angina", "recurrent chest pain"),
    ("pain_abdominal", r"abdominal pain|stomach pain|belly pain|tummy pain|abdominal ache", "recurrent abdominal pain"),
    (
        "pain_joint",
        r"(?:knee|hip|shoulder|elbow|ankle|wrist|joint).{0,30}(?:pain|ache|stiff)"
        r"|(?:pain|ache|stiff).{0,30}(?:knee|hip|shoulder|elbow|ankle|wrist|joint)",
        "recurrent joint pain",
    ),
    ("pain_head", r"headache|head ache|migraine", "recurrent headache"),
    ("rash_allergic", r"hives|urticaria|allergic reaction|allergic rash", "recurrent allergic rash or hives"),
    ("rash_skin", r"\brash\b|skin reaction|dermatitis|eruption", "recurrent rash"),
)


def _text_for_symptom_label_extraction(text: str) -> str:
    """Drop reassuring breathing clauses so they do not override the primary complaint."""
    clauses = re.split(r"[,;]\s*", str(text or ""))
    kept = [
        clause.strip()
        for clause in clauses
        if clause.strip() and not _REASSURING_BREATHING_CLAUSE_RE.search(clause)
    ]
    if kept:
        return ", ".join(kept)
    return str(text or "")


def _refine_extracted_symptom_keys(keys: list[str]) -> list[str]:
    refined = list(keys)
    key_set = set(refined)
    for broad_key, specific_keys in _SYMPTOM_FAMILY_EXCLUSIONS.items():
        if broad_key in key_set and key_set.intersection(specific_keys):
            refined = [key for key in refined if key != broad_key]
            key_set.discard(broad_key)
    return refined


def extract_symptoms_from_text(text: str) -> list[str]:
    if not text:
        return []
    found = []
    for key, pattern, _label in SYMPTOM_PATTERN_DEFINITIONS:
        if re.search(pattern, text, re.I):
            found.append(key)
    return _refine_extracted_symptom_keys(found)


def resolve_incident_symptom_keys(text: str, stored_symptoms: list | None = None) -> set[str]:
    """Prefer text-derived subtype keys so legacy stored 'swelling' does not over-link."""
    from_text = set(extract_symptoms_from_text(text))
    if not stored_symptoms:
        return from_text
    stored = {str(item).strip() for item in stored_symptoms if str(item).strip()}
    if stored & _LEGACY_GENERIC_SYMPTOM_KEYS:
        return from_text or (stored - _LEGACY_GENERIC_SYMPTOM_KEYS)
    return from_text | stored


def incident_symptom_relevance_score(current_text: str, incident: dict) -> int:
    incident_text = str(incident.get("text") or "").strip()
    if not incident_text:
        return 0
    current_keys = resolve_incident_symptom_keys(current_text)
    incident_keys = resolve_incident_symptom_keys(
        incident_text,
        incident.get("symptoms"),
    )
    overlap = current_keys & incident_keys
    if not overlap:
        return 0
    score = 10 * len(overlap)
    current_tokens = set(re.findall(r"[a-z]{4,}", current_text.lower()))
    incident_tokens = set(re.findall(r"[a-z]{4,}", incident_text.lower()))
    shared = (current_tokens & incident_tokens) - _SYMPTOM_LINK_STOPWORDS
    score += len(shared)
    return score


def _patterns_for_session_triggers(session_triggers: list[str]) -> list[str]:
    patterns = []
    for trigger in session_triggers or []:
        trigger_text = str(trigger or "").strip().lower()
        if not trigger_text:
            continue
        for _key, pattern, label in SESSION_RECURRENCE_PATTERN_DEFINITIONS:
            label_lower = label.lower()
            if label_lower in trigger_text or f"{label_lower} in this session" in trigger_text:
                patterns.append(pattern)
    return patterns


def _priors_matching_session_triggers(
    prior_incidents: list,
    session_triggers: list[str],
    *,
    limit: int = 2,
) -> list[dict]:
    patterns = _patterns_for_session_triggers(session_triggers)
    if not patterns:
        return []
    matched = []
    for incident in prior_incidents:
        text = str(incident.get("text") or "").strip()
        if not text:
            continue
        if any(re.search(pattern, text, re.I) for pattern in patterns):
            matched.append(incident)
    return matched[-limit:]


def select_linked_prior_incidents(
    current_text: str,
    prior_incidents: list,
    *,
    limit: int = 2,
    session_triggers: list[str] | None = None,
) -> list[dict]:
    """Prior reports that genuinely share symptom/subtype evidence with the current report."""
    if not current_text or not prior_incidents:
        return []
    related = find_symptom_related_prior_incidents(current_text, prior_incidents, limit=limit)
    if related:
        return related
    if session_triggers:
        return _priors_matching_session_triggers(
            prior_incidents,
            session_triggers,
            limit=limit,
        )
    return []


def count_linked_session_reports(
    current_text: str,
    prior_incidents: list,
    *,
    session_triggers: list[str] | None = None,
) -> int:
    """Current report plus prior reports with genuine symptom/subtype overlap."""
    linked = select_linked_prior_incidents(
        current_text,
        prior_incidents,
        session_triggers=session_triggers,
    )
    return 1 + len(linked) if linked else 1


def detect_session_symptom_recurrence(current_text: str, prior_text: str) -> list[str]:
    """Subtype-aware session recurrence labels (no broad swelling/pain/rash collapse)."""
    if not current_text or not prior_text:
        return []
    prior_lower = prior_text.lower()
    current_lower = current_text.lower()
    triggers = []
    for _key, pattern, label in SESSION_RECURRENCE_PATTERN_DEFINITIONS:
        if re.search(pattern, prior_lower, re.I) and re.search(pattern, current_lower, re.I):
            triggers.append(f"{label} in this session")
    return triggers


def find_symptom_related_prior_incidents(
    current_text: str,
    prior_incidents: list,
    *,
    limit: int = 2,
) -> list[dict]:
    if not current_text or not prior_incidents:
        return []
    scored = []
    for incident in prior_incidents:
        if not incident.get("text"):
            continue
        score = incident_symptom_relevance_score(current_text, incident)
        if score > 0:
            scored.append((score, incident))
    if scored:
        scored.sort(
            key=lambda item: (
                -item[0],
                str(item[1].get("timestamp") or item[1].get("timestamp_display") or ""),
            )
        )
        return [incident for _, incident in scored[:limit]]
    return []


def detect_session_escalation_triggers(current_text: str, prior_incidents: list) -> list[str]:
    if not current_text or not prior_incidents:
        return []

    triggers = []
    current_lower = current_text.lower()

    def _incident_text(item: dict) -> str:
        return str(item.get("text") or item.get("summary") or "")

    has_confusion_prior = any(re.search(r"confus", _incident_text(item), re.I) for item in prior_incidents)
    has_fall_prior = any(
        re.search(r"\bfell\b|\bfall\b|\bfallen\b", _incident_text(item), re.I)
        for item in prior_incidents
    )
    has_confusion_current = bool(re.search(r"confus", current_lower))
    has_fall_current = bool(re.search(r"\bfell\b|\bfall\b|\bfallen\b", current_lower))

    if has_confusion_prior and has_fall_current:
        triggers.append("confusion followed by fall in this session")
    elif has_fall_prior and has_confusion_current:
        triggers.append("fall and confusion reported in this session")

    prior_text = " ".join(_incident_text(item) for item in prior_incidents).lower()
    subtype_triggers = detect_session_symptom_recurrence(current_text, prior_text)
    if subtype_triggers:
        triggers.append(subtype_triggers[0])

    if re.search(r"(?:worsening|getting worse|worse than|not improving)", current_lower) and prior_text.strip():
        triggers.append("worsening pattern across session reports")

    return triggers


def count_linked_reports(
    prior_incidents: list,
    user_text: str,
    session_triggers: list | None = None,
) -> int:
    if not prior_incidents:
        return 1
    linked = select_linked_prior_incidents(
        user_text,
        prior_incidents,
        session_triggers=session_triggers,
    )
    if not linked:
        return 1
    return count_linked_session_reports(
        user_text,
        prior_incidents,
        session_triggers=session_triggers,
    )
