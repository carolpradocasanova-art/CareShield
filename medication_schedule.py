"""Medication timing parsing and schedule display helpers (importable without Streamlit)."""

from __future__ import annotations

import re
from datetime import datetime

VAGUE_TIMING_PHRASES = (
    "as before",
    "as directed",
    "as prescribed",
    "same as before",
    "same as usual",
    "as usual",
    "continue as before",
    "continue as directed",
    "per usual",
)

HOUR_COUNT_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    6: "six",
    8: "eight",
    12: "twelve",
}

MEAL_TIME_HOURS = {
    "before breakfast": [(8, 0)],
    "with breakfast": [(8, 0)],
    "after breakfast": [(9, 0)],
    "before lunch": [(12, 0)],
    "with lunch": [(12, 0)],
    "after lunch": [(13, 0)],
    "before dinner": [(18, 0)],
    "with dinner": [(19, 0)],
    "after dinner": [(20, 0)],
    "at bedtime": [(21, 0)],
    "bedtime": [(21, 0)],
}

FREQUENCY_SCHEDULE_SLOTS = {
    "once_daily": [(8, 0)],
    "twice_daily": [(8, 0), (20, 0)],
    "three_times_daily": [(8, 0), (14, 0), (20, 0)],
    "at_night": [(21, 0)],
}


def extract_pills_per_dose(med: dict) -> int | None:
    raw = med.get("pills_per_dose")
    if raw is not None:
        try:
            count = int(raw)
            if 1 <= count <= 10:
                return count
        except (TypeError, ValueError):
            pass
    text = " ".join(
        str(med.get(key, "") or "")
        for key in ("dosage", "timing", "time", "name", "dose", "instructions")
    )
    patterns = [
        r"(?:take|give|administer|swallow)\s+(\d+)\s*(?:pill|tablet|tab|cap|capsule)s?\b",
        r"(\d+)\s*(?:pill|tablet|tab|cap|capsule)s?\s*(?:per|each|at|once|every|daily|with)",
        r"(\d+)\s*(?:pill|tablet|tab|cap|capsule)s?\b",
        r"\bx(\d+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            count = int(match.group(1))
            if 1 <= count <= 10:
                return count
    return None


def is_prn_timing(timing: str) -> bool:
    text = (timing or "").strip().lower()
    return "as needed" in text or bool(re.search(r"\bprn\b", text))


def _is_vague_timing(text: str) -> bool:
    lower = (text or "").strip().lower()
    if not lower:
        return True
    return any(vague in lower for vague in VAGUE_TIMING_PHRASES)


def _every_hours_phrase(hours: int) -> str:
    word = HOUR_COUNT_WORDS.get(hours)
    if word:
        unit = "hour" if hours == 1 else "hours"
        return f"every {word} {unit}"
    return f"every {hours} hours"


def _plan_schedule_fields(plan_item: dict | None) -> list[str]:
    plan = plan_item or {}
    values = []
    for key in ("time", "timing", "schedule", "instructions", "dosage"):
        value = str(plan.get(key) or "").strip()
        if value and not _is_vague_timing(value):
            values.append(value)
    return values


def canonical_plan_timing(plan_item: dict | None) -> str:
    """Single schedule string — same source Stored Medications uses (timing before time)."""
    plan = plan_item or {}
    for key in ("timing", "time", "schedule"):
        value = strip_embedded_pill_count_from_timing(str(plan.get(key) or "").strip())
        if value and not _is_vague_timing(value):
            return value
    return ""


def normalize_medication_schedule_fields(med: dict) -> dict:
    """Keep one canonical schedule on a medication; prevents stale time/timing drift after updates."""
    payload = dict(med)
    canonical = canonical_plan_timing(payload)
    if canonical:
        payload["timing"] = canonical
        payload["time"] = canonical
    else:
        payload.pop("timing", None)
        payload.pop("time", None)
    payload.pop("schedule", None)
    return payload


def _unique_clock_times(times: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for raw in times:
        token = raw.strip()
        if not token:
            continue
        key = re.sub(r"\s+", " ", token.lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(token)
    return unique


def strip_embedded_pill_count_from_timing(timing: str) -> str:
    """Remove pill-count fragments already baked into parsed dosage_instructions timing."""
    text = str(timing or "").strip()
    if not text:
        return ""
    parts = [part.strip() for part in re.split(r"\s*·\s*", text) if part.strip()]
    kept = [
        part for part in parts
        if not re.fullmatch(r"\d+\s*pill\(s\)\s*per dose", part, re.I)
    ]
    return " · ".join(kept)


def format_medication_frequency(plan_item: dict | None = None, timing: str = "") -> str:
    plan = plan_item or {}
    explicit = ""
    if timing and not _is_vague_timing(timing):
        explicit = strip_embedded_pill_count_from_timing(timing.strip())
    combined = explicit or canonical_plan_timing(plan)
    if not combined:
        return ""

    lower = combined.lower()

    q_match = re.search(r"\bq\s*(\d+)\s*h\b", lower)
    if q_match:
        return _every_hours_phrase(int(q_match.group(1)))

    every_match = re.search(r"every\s+(\d+)\s*hours?", lower)
    if every_match:
        return _every_hours_phrase(int(every_match.group(1)))

    hourly_match = re.search(r"(\d+)\s*[- ]?hourly", lower)
    if hourly_match:
        return _every_hours_phrase(int(hourly_match.group(1)))

    if re.search(r"\b(bid|bd|twice daily|twice a day|two times daily|2 times daily)\b", lower):
        return "twice daily"
    if re.search(r"\b(tid|tds|three times daily|three times a day|3 times daily)\b", lower):
        return "three times daily"
    if re.search(r"\b(qid|four times daily|four times a day|4 times daily)\b", lower):
        return "four times daily"
    if re.search(r"\b(at night|at bedtime|bedtime|nightly)\b", lower):
        return "at night"
    if re.search(r"\b(once daily|once a day|one time daily|od\b)\b", lower):
        return "once daily"
    if "as needed" in lower or re.search(r"\bprn\b", lower):
        return "as needed"
    if "breakfast" in lower or "morning" in lower:
        return "once daily in the morning"
    if "lunch" in lower or "midday" in lower or "noon" in lower:
        return "once daily at midday"
    if "dinner" in lower or "evening meal" in lower:
        return "once daily with dinner"

    clock_times = re.findall(
        r"\d{1,2}\s*:\s*\d{2}\s*(?:am|pm)?|\d{1,2}\s*(?:am|pm)\b",
        combined,
        re.I,
    )
    cleaned_times = _unique_clock_times(clock_times)
    if len(cleaned_times) == 1:
        return f"once daily at {cleaned_times[0]}"
    if len(cleaned_times) > 1:
        joined = ", ".join(cleaned_times[:-1]) + f" and {cleaned_times[-1]}"
        return f"{len(cleaned_times)} times daily at {joined}"

    if not _is_vague_timing(combined):
        if re.search(r"\d", combined):
            return f"as scheduled ({combined})"
        return combined
    return ""


def format_plan_schedule_summary(plan_item: dict | None) -> str:
    """Single schedule line: frequency + pill count, without duplicate per-dose text."""
    plan = plan_item or {}
    timing = canonical_plan_timing(plan)
    pills = plan.get("pills_per_dose") or extract_pills_per_dose(plan) or 1
    pill_phrase = f"{pills} pill(s) per dose"
    frequency = format_medication_frequency(plan, timing)
    if frequency:
        return f"{frequency} · {pill_phrase}"
    if timing:
        return f"{timing} · {pill_phrase}"
    return pill_phrase


def format_timing_phrase(timing: str, plan_item: dict | None = None) -> str:
    phrase = format_medication_frequency(plan_item, timing)
    if phrase:
        return phrase
    text = (timing or "").strip()
    if text and not _is_vague_timing(text):
        return text
    return ""


def finish_dose_instruction(sentence: str, timing: str, plan_item: dict | None = None) -> str:
    phrase = format_medication_frequency(plan_item, timing)
    base = sentence.rstrip(".")
    if phrase:
        return f"{base}, {phrase}."
    return f"{base}."


def resolve_schedule_frequency(timing: str) -> str | None:
    """Map timing text to a canonical frequency key for FREQUENCY_SCHEDULE_SLOTS."""
    text = strip_embedded_pill_count_from_timing(timing).strip().lower()
    if not text or is_prn_timing(text):
        return None
    if re.search(r"\b(bid|bd|twice daily|twice a day|two times daily|2 times daily)\b", text):
        return "twice_daily"
    if (
        re.search(r"\b(tid|tds|three times daily|three times a day|3 times daily)\b", text)
        or "three times" in text
    ):
        return "three_times_daily"
    if re.search(r"\b(at night|at bedtime|bedtime|nightly)\b", text):
        return "at_night"
    if re.search(r"\b(once daily|once a day|one time daily|od\b)\b", text):
        return "once_daily"
    return None


def _parse_clock_time(hour: int, minute: int, meridiem: str = "") -> tuple[int, int]:
    mer = (meridiem or "").lower().strip()
    if mer == "pm" and hour != 12:
        hour += 12
    elif mer == "am" and hour == 12:
        hour = 0
    return hour % 24, minute % 60


def _extract_explicit_clock_times(text: str) -> list[tuple[int, int]]:
    """Parse HH:MM (optional am/pm) or bare 8am-style tokens from timing text."""
    times: list[tuple[int, int]] = []
    for match in re.finditer(r"(\d{1,2})\s*:\s*(\d{2})\s*(am|pm)?", text, re.I):
        times.append(_parse_clock_time(int(match.group(1)), int(match.group(2)), match.group(3) or ""))
    for match in re.finditer(r"(\d{1,2})\s*(am|pm)\b", text, re.I):
        # Skip minute fragments already consumed by HH:MM (e.g. "45 am" inside "11:45 am").
        if match.start() >= 1 and text[match.start() - 1] == ":":
            continue
        hour = int(match.group(1))
        if not 1 <= hour <= 12:
            continue
        times.append(_parse_clock_time(hour, 0, match.group(2)))
    return sorted(set(times))


def parse_schedule_times(timing: str) -> list[tuple[int, int]]:
    text = strip_embedded_pill_count_from_timing(timing).strip().lower()
    if not text:
        return [(8, 0)]
    if is_prn_timing(text):
        return []

    explicit = _extract_explicit_clock_times(text)
    if explicit:
        return explicit

    every_match = re.search(r"every\s+(\d+)\s*hours?", text)
    if every_match:
        interval = int(every_match.group(1))
        if interval <= 0:
            return [(8, 0)]
        times = []
        hour = 8
        seen = set()
        while len(seen) < 24:
            slot = (hour % 24, 0)
            if slot in seen:
                break
            seen.add(slot)
            times.append(slot)
            hour += interval
        return times or [(8, 0)]

    freq_key = resolve_schedule_frequency(text)
    if freq_key:
        return list(FREQUENCY_SCHEDULE_SLOTS[freq_key])

    for phrase, slots in MEAL_TIME_HOURS.items():
        if phrase in text:
            return list(slots)

    if " and " in text:
        merged = []
        for part in re.split(r"\s+and\s+", text):
            merged.extend(parse_schedule_times(part))
        if merged:
            return sorted(set(merged))

    return [(8, 0)]


def schedule_time_slots_for_medication(med: dict) -> list[tuple[int, int]]:
    """Return unique (hour, minute) slots for one medication's current canonical schedule."""
    normalized = normalize_medication_schedule_fields(med)
    timing = canonical_plan_timing(normalized)
    if not timing or is_prn_timing(timing):
        return []
    return sorted(set(parse_schedule_times(timing)))


def build_dose_events(plan_items: list) -> list:
    """
    Build scheduled dose slots from the current medication plan.

    Always clear-then-regenerate from each medication's canonical schedule — never append
    or reuse prior slots. When a schedule changes (document re-upload, manual edit, etc.),
    old time slots must not persist as phantom dose events for any medication.
    """
    events = []
    for med in plan_items or []:
        name = str(med.get("name") or "").strip()
        if not name:
            continue
        for hour, minute in schedule_time_slots_for_medication(med):
            events.append({
                "medication_name": name,
                "hour": hour,
                "minute": minute,
                "time_label": f"{hour:02d}:{minute:02d}",
                "display_time": f"{hour:02d}:{minute:02d}",
            })
    return sorted(events, key=lambda item: (item["hour"], item["minute"], item["medication_name"]))


def dose_minutes_until(dose: dict, now: datetime, tz_obj=None) -> float:
    """Minutes until the dose's scheduled wall-clock time today in the user's timezone."""
    tz = tz_obj or now.tzinfo
    if now.tzinfo is None and tz is not None:
        now = now.replace(tzinfo=tz)
    scheduled = now.replace(
        hour=int(dose["hour"]),
        minute=int(dose["minute"]),
        second=0,
        microsecond=0,
    )
    return (scheduled - now).total_seconds() / 60


def compute_dose_ui_state(
    dose: dict,
    now: datetime,
    *,
    existing_log: dict | None = None,
    tz_obj=None,
) -> str:
    """
    Shared dose status for MedCam cards and schedule helpers.

    Window: not_yet (>30 min before), actionable/Due now (-30 to +60 min), missed (>60 min after).
    """
    if existing_log:
        return existing_log.get("status", "taken")

    minutes_until = dose_minutes_until(dose, now, tz_obj)
    if minutes_until > 30:
        return "not_yet"
    if minutes_until >= -60:
        return "actionable"
    return "missed"
