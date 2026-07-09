"""Handover period filtering and SBAR event collection (Streamlit-free for tests)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


HANDOVER_PERIOD_OPTIONS = (
    ("today", "Today"),
    ("this_week", "This week"),
    ("last_week", "Last week"),
    ("last_2_weeks", "Last 2 weeks"),
    ("this_month", "This month"),
)

HANDOVER_INSUFFICIENT_DATA_MSG = (
    "There isn't enough information yet to generate a handover for this period. "
    "Log at least one update in Report & Ask (a statement about how the patient is doing, "
    "not only a question), run a MedCam check, or mark doses taken/missed — then try again."
)

REPORT_HISTORY_DEFAULT_VISIBLE = 3
REPORT_HISTORY_EXPAND_BATCH = 3

SEVERITY_RANK = {
    "ok": 0,
    "monitor": 1,
    "contact_doctor": 2,
    "emergency": 3,
}


def normalize_handover_severity(severity: str) -> str:
    value = str(severity or "monitor").strip().lower().replace(" ", "_")
    if value == "urgent":
        return "emergency"
    if value in ("contact_doctor", "contactdoctor", "doctor"):
        return "contact_doctor"
    if value in ("ok", "monitor", "emergency"):
        return value
    return "monitor"


def handover_severity_label(severity: str) -> str:
    labels = {
        "emergency": "EMERGENCY",
        "contact_doctor": "CONTACT DOCTOR",
        "monitor": "MONITOR",
        "ok": "OK",
    }
    return labels.get(normalize_handover_severity(severity), "MONITOR")


def timeline_event_is_photo(event: dict) -> bool:
    source = str(event.get("source") or "")
    return (
        source in ("symptom_photo", "pill_photo")
        or bool(event.get("has_photo"))
        or bool(str(event.get("image_b64") or "").strip())
    )


def compute_peak_severity(events: list) -> str:
    peak = "ok"
    for event in events or []:
        severity = normalize_handover_severity(event.get("severity", "monitor"))
        if SEVERITY_RANK.get(severity, 0) > SEVERITY_RANK.get(peak, 0):
            peak = severity
    return peak


def build_peak_severity_note(symptom_events: list) -> str:
    peak = compute_peak_severity(symptom_events)
    if peak == "emergency":
        return (
            "PEAK SEVERITY REACHED IN THIS PERIOD: EMERGENCY — Call 999/112 was recommended "
            "for at least one report. State this explicitly in Situation and Assessment; "
            "do not soften to generic concern language."
        )
    if peak == "contact_doctor":
        return (
            "PEAK SEVERITY REACHED IN THIS PERIOD: CONTACT DOCTOR — GP/consultant contact "
            "within 24 hours was recommended for at least one report."
        )
    return ""


def prepare_sbar_symptom_entries(symptom_events: list) -> list[dict]:
    """One SBAR row per logged symptom report — no summary-level deduplication."""
    entries = []
    for event in sorted(symptom_events or [], key=lambda item: str(item.get("timestamp") or "")):
        summary = str(event.get("text") or event.get("summary") or "").strip()
        if not summary:
            continue
        entries.append({
            "created_at": event.get("timestamp") or event.get("created_at"),
            "summary": summary,
            "severity": normalize_handover_severity(event.get("severity", "monitor")),
            "caregiver_name": event.get("caregiver") or event.get("caregiver_name", ""),
            "source": event.get("source", "report"),
            "has_photo": timeline_event_is_photo(event),
            "photo_finding": str(event.get("photo_finding") or "").strip(),
        })
    return entries


def handover_events_signature(events: list) -> list[dict]:
    """Compact fingerprint for PDF cache invalidation when timeline data changes."""
    signature = []
    for event in sorted(events or [], key=lambda item: str(item.get("timestamp") or "")):
        signature.append({
            "timestamp": str(event.get("timestamp") or "")[:19],
            "text": str(event.get("text") or "")[:80],
            "severity": normalize_handover_severity(event.get("severity", "monitor")),
            "source": str(event.get("source") or ""),
            "has_photo": timeline_event_is_photo(event),
        })
    return signature


def get_handover_period_label(period_key: str) -> str:
    return dict(HANDOVER_PERIOD_OPTIONS).get(period_key, "This week")


def get_handover_period_bounds(period_key: str, tz_obj) -> tuple[datetime, datetime]:
    now = datetime.now(tz_obj)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if period_key == "today":
        return today_start, now
    if period_key == "this_week":
        return today_start - timedelta(days=6), now
    if period_key == "last_week":
        return (
            today_start - timedelta(days=14),
            today_start - timedelta(days=7) - timedelta(microseconds=1),
        )
    if period_key == "last_2_weeks":
        return today_start - timedelta(days=13), now
    if period_key == "this_month":
        return today_start - timedelta(days=29), now
    return today_start - timedelta(days=6), now


def parse_handover_datetime(value, tz_obj) -> datetime | None:
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    normalized = raw.replace("Z", "+00:00")
    if " " in normalized and "T" not in normalized:
        normalized = normalized.replace(" ", "T", 1)
    try:
        parsed = datetime.fromisoformat(normalized)
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(tz_obj)


def event_in_handover_period(event: dict, period_key: str, tz_obj) -> bool:
    timestamp = event.get("timestamp") or event.get("created_at") or event.get("logged_at")
    parsed = parse_handover_datetime(timestamp, tz_obj)
    if parsed is None:
        return False
    start, end = get_handover_period_bounds(period_key, tz_obj)
    return start <= parsed <= end


def filter_events_by_handover_period(events: list, period_key: str, tz_obj) -> list:
    return [event for event in events if event_in_handover_period(event, period_key, tz_obj)]


def shift_log_row_to_sbar_event(row: dict) -> dict | None:
    summary = str(row.get("summary") or "").strip()
    if not summary:
        return None
    return {
        "created_at": row.get("created_at"),
        "summary": summary,
        "severity": row.get("severity", "monitor"),
        "caregiver_name": row.get("caregiver_name", ""),
        "source": row.get("source", "report"),
    }


def timeline_event_to_sbar_event(event: dict) -> dict | None:
    summary = str(event.get("text") or event.get("summary") or "").strip()
    if not summary:
        return None
    return {
        "created_at": event.get("timestamp") or event.get("created_at"),
        "summary": summary,
        "severity": event.get("severity", "monitor"),
        "caregiver_name": event.get("caregiver") or event.get("caregiver_name", ""),
        "source": event.get("source", "report"),
    }


def merge_sbar_events(*event_groups: list[dict]) -> list[dict]:
    """Deduplicate SBAR-ready events, keeping the highest severity when keys collide."""
    merged_by_key: dict[tuple[str, str, str], dict] = {}
    order: list[tuple[str, str, str]] = []
    for group in event_groups:
        for raw in group or []:
            if not isinstance(raw, dict):
                continue
            if "text" in raw or raw.get("caregiver"):
                event = timeline_event_to_sbar_event(raw)
            else:
                event = shift_log_row_to_sbar_event(raw)
            if not event:
                continue
            event["severity"] = normalize_handover_severity(event.get("severity", "monitor"))
            dedupe_key = (
                str(event.get("created_at") or "")[:19],
                str(event.get("summary") or "")[:120],
                str(event.get("source") or ""),
            )
            existing = merged_by_key.get(dedupe_key)
            if existing:
                if SEVERITY_RANK.get(event["severity"], 0) > SEVERITY_RANK.get(existing["severity"], 0):
                    merged_by_key[dedupe_key] = event
                continue
            merged_by_key[dedupe_key] = event
            order.append(dedupe_key)
    return [merged_by_key[key] for key in order]


SYMPTOM_TIMELINE_SOURCES = frozenset({
    "voice_report",
    "care_question",
    "document_upload",
    "symptom_photo",
    "pill_photo",
    "my_results_handover",
    "report",
})

ADHERENCE_TIMELINE_SOURCES = frozenset({
    "medication_log",
    "medication_check",
})


def partition_timeline_events_for_sbar(events: list) -> tuple[list[dict], list[dict]]:
    """Split combined timeline rows into symptom reports vs medication adherence."""
    symptom_events: list[dict] = []
    adherence_events: list[dict] = []
    for event in events or []:
        source = str(event.get("source") or "")
        if source in ADHERENCE_TIMELINE_SOURCES or event.get("adherence_status"):
            adherence_events.append(event)
        elif source in SYMPTOM_TIMELINE_SOURCES or source not in ADHERENCE_TIMELINE_SOURCES:
            symptom_events.append(event)
    return symptom_events, adherence_events


def build_sbar_handover_user_payload(symptom_events: list, adherence_events: list) -> str:
    """Present symptom reports and dose activity in separate sections for the AI."""
    symptom_entries = prepare_sbar_symptom_entries(symptom_events)
    symptom_lines = []
    photo_lines = []
    for event in symptom_entries:
        caregiver = event.get("caregiver_name") or "Caregiver"
        severity_label = handover_severity_label(event.get("severity"))
        line = (
            f"- [{severity_label}] [{event.get('source', 'report')}] "
            f"{caregiver}: {event.get('summary')}"
        )
        if event.get("has_photo"):
            finding = event.get("photo_finding") or "reviewed"
            line += f" (symptom photo submitted — finding: {finding})"
            photo_lines.append(line)
        symptom_lines.append(line)

    adherence_lines = []
    for event in merge_sbar_events(adherence_events):
        caregiver = event.get("caregiver_name") or event.get("caregiver") or "Caregiver"
        severity_label = handover_severity_label(event.get("severity"))
        adherence_lines.append(
            f"- [{severity_label}] [{event.get('source', 'dose')}] "
            f"{caregiver}: {event.get('summary')}"
        )

    sections = []
    peak_note = build_peak_severity_note(symptom_events)
    if peak_note:
        sections.append(peak_note)

    if symptom_lines:
        sections.append(
            f"SYMPTOM REPORTS & CARE UPDATES ({len(symptom_lines)} entries in this period — "
            "include ALL of them in Situation/Assessment; preserve severity labels):\n"
            + "\n".join(symptom_lines)
        )
    else:
        sections.append(
            "SYMPTOM REPORTS & CARE UPDATES: None logged in this period."
        )

    if photo_lines:
        sections.append(
            f"SYMPTOM PHOTOS LOGGED ({len(photo_lines)} — describe each in Assessment):\n"
            + "\n".join(photo_lines)
        )

    if adherence_lines:
        sections.append(
            f"MEDICATION ADHERENCE ({len(adherence_lines)} entries in this period):\n"
            + "\n".join(adherence_lines)
        )
    else:
        sections.append("MEDICATION ADHERENCE: No dose activity logged in this period.")

    return "\n\n".join(sections)


def collect_sbar_events_from_timelines(
    symptom_events: list,
    adherence_events: list,
    *,
    period_key: str,
    tz_obj,
) -> list[dict]:
    """Build the SBAR payload from the same timeline sources shown in the Handover tab."""
    in_period = filter_events_by_handover_period(
        list(symptom_events or []) + list(adherence_events or []),
        period_key,
        tz_obj,
    )
    return merge_sbar_events(in_period)


def count_chat_user_reports(messages: list) -> int:
    return sum(
        1
        for message in messages or []
        if message.get("role") == "user" and not message.get("welcome")
    )


def slice_messages_for_report_history(messages: list, visible_report_count: int) -> tuple[list, int, int]:
    """Return (visible_messages, hidden_report_count, total_report_count) for chat display."""
    messages = list(messages or [])
    user_positions = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user" and not message.get("welcome")
    ]
    total_reports = len(user_positions)
    if total_reports == 0:
        return messages, 0, 0
    shown_reports = min(max(int(visible_report_count or 0), 0), total_reports)
    if shown_reports >= total_reports:
        return messages, 0, total_reports

    start_index = user_positions[-shown_reports]
    prefix = []
    if messages and messages[0].get("welcome") and start_index > 0:
        prefix = [messages[0]]
    visible_messages = prefix + messages[start_index:]
    return visible_messages, total_reports - shown_reports, total_reports


def build_reported_by_entries_from_events(events: list) -> list:
    """Map timeline events to caregiver note cards for handover 'Reported by' sections."""
    entries = []
    for event in sorted(events, key=lambda item: str(item.get("timestamp") or "")):
        note = str(event.get("text") or "").strip()
        if not note:
            continue
        entry = {
            "caregiver": event.get("caregiver") or "Caregiver",
            "note": note,
        }
        image_b64 = str(event.get("image_b64") or "").strip()
        if image_b64:
            entry["image_b64"] = image_b64
        if event.get("has_photo"):
            entry["has_photo"] = True
        entries.append(entry)
    return entries


def enrich_handover_result_with_period_entries(
    sbar_result: dict,
    symptom_events: list,
    adherence_events: list,
) -> dict:
    """Ensure the generated handover lists every report in the selected period."""
    merged = dict(sbar_result or {})
    period_entries = build_reported_by_entries_from_events(
        sorted(
            list(symptom_events or []) + list(adherence_events or []),
            key=lambda item: str(item.get("timestamp") or ""),
        )
    )
    merged["reported_by"] = period_entries
    merged["period_report_count"] = len(period_entries)
    merged["peak_severity"] = compute_peak_severity(symptom_events)
    merged["photo_report_count"] = sum(
        1 for event in (symptom_events or []) if timeline_event_is_photo(event)
    )
    return merged
