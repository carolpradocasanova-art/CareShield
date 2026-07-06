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
    """Deduplicate and return SBAR-ready events with non-empty summaries."""
    merged: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
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
            dedupe_key = (
                str(event.get("created_at") or "")[:19],
                str(event.get("summary") or "")[:120],
                str(event.get("source") or ""),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            merged.append(event)
    return merged


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
    symptom_lines = []
    for event in merge_sbar_events(symptom_events):
        caregiver = event.get("caregiver_name") or event.get("caregiver") or "Caregiver"
        symptom_lines.append(
            f"- [{event.get('source', 'report')}] {caregiver}: {event.get('summary')}"
        )

    adherence_lines = []
    for event in merge_sbar_events(adherence_events):
        caregiver = event.get("caregiver_name") or event.get("caregiver") or "Caregiver"
        adherence_lines.append(
            f"- [{event.get('source', 'dose')}] {caregiver}: {event.get('summary')}"
        )

    sections = []
    if symptom_lines:
        sections.append(
            f"SYMPTOM REPORTS & CARE UPDATES ({len(symptom_lines)} entries in this period — include ALL of them in Situation/Assessment):\n"
            + "\n".join(symptom_lines)
        )
    else:
        sections.append(
            "SYMPTOM REPORTS & CARE UPDATES: None logged in this period."
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
