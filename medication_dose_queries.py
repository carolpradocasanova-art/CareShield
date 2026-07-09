"""Medication next-dose replies for Report & Ask — no Streamlit dependencies."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone


def format_friendly_dose_time(hour: int, minute: int) -> str:
    suffix = "am" if hour < 12 else "pm"
    display_hour = hour % 12 or 12
    if minute:
        return f"{display_hour}:{minute:02d}{suffix}"
    return f"{display_hour}{suffix}"


def dose_log_for_today(dose: dict, today_logs: list, date_iso: str, tz_obj):
    for log in today_logs:
        reported = parse_log_local_date(log, tz_obj)
        if reported != date_iso:
            continue
        if log.get("medication_name") == dose["medication_name"] and log.get("scheduled_time") == dose["time_label"]:
            return log
    return None


def parse_log_local_date(log: dict, tz_obj) -> str:
    raw = log.get("logged_at") or ""
    if not raw:
        return ""
    try:
        moment = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(tz_obj).date().isoformat()
    except (ValueError, TypeError):
        return ""


def dose_ui_state(dose: dict, now: datetime, today_logs: list, tz_obj) -> str:
    from medication_schedule import compute_dose_ui_state

    date_iso = now.date().isoformat()
    existing = dose_log_for_today(dose, today_logs, date_iso, tz_obj)
    return compute_dose_ui_state(dose, now, existing_log=existing, tz_obj=tz_obj)


def find_plan_item(med_name: str, plan_items: list) -> dict:
    for item in plan_items:
        if item.get("name") == med_name:
            return item
    name_lower = (med_name or "").lower()
    for item in plan_items:
        item_lower = str(item.get("name") or "").lower()
        if name_lower in item_lower or item_lower in name_lower:
            return item
    return {}


def extract_medication_name_from_question(text: str, plan_items: list) -> str | None:
    lower = (text or "").strip().lower()
    if not lower or not plan_items:
        return None

    hits: list[str] = []
    for item in plan_items:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        name_lower = name.lower()
        if name_lower in lower:
            hits.append(name)
            continue
        stem = re.split(r"[\s(\/-]+", name_lower)[0]
        if len(stem) >= 5 and stem in lower:
            hits.append(name)

    if not hits:
        return None
    if len(hits) == 1:
        return hits[0]
    return max(hits, key=len)


def _dose_next_occurrence(dose: dict, now: datetime) -> datetime:
    scheduled = now.replace(hour=dose["hour"], minute=dose["minute"], second=0, microsecond=0)
    if scheduled > now:
        return scheduled
    return scheduled + timedelta(days=1)


def _format_relative_when(minutes: float) -> str:
    mins = max(int(minutes), 1)
    if mins < 60:
        return f"in about {mins} minute{'s' if mins != 1 else ''}"
    hours = mins // 60
    rem = mins % 60
    when = f"in about {hours} hour{'s' if hours != 1 else ''}"
    if rem:
        when += f" and {rem} minutes"
    return when


def _format_med_name_list(names: list[str], max_show: int = 4) -> str:
    shown = names[:max_show]
    text = ", ".join(f"**{name}**" for name in shown)
    if len(names) > max_show:
        text += f", and **{len(names) - max_show} more**"
    return text


def _format_next_pill_when(occurrence: datetime, now: datetime, minutes: float) -> str:
    friendly = format_friendly_dose_time(occurrence.hour, occurrence.minute)
    relative = _format_relative_when(minutes)
    if occurrence.date() > now.date():
        return f"**{friendly}** tomorrow ({relative})"
    return f"**{friendly}** ({relative})"


def summarize_dose_schedule_notes(
    missed: list,
    *,
    today_logs: list | None = None,
    tz_obj=None,
    now: datetime | None = None,
    medication_name: str | None = None,
) -> str:
    if not missed:
        return ""

    if medication_name:
        missed = [dose for dose in missed if dose["medication_name"] == medication_name]
    if not missed:
        return ""

    tz_obj = tz_obj or timezone.utc
    now = now or datetime.now(tz_obj)
    date_iso = now.date().isoformat()
    logs = today_logs or []

    confirmed_missed = []
    unlogged_past_due = []
    for dose in missed:
        log = dose_log_for_today(dose, logs, date_iso, tz_obj)
        if log and str(log.get("status") or "").lower() == "missed":
            confirmed_missed.append(dose)
        elif not log:
            unlogged_past_due.append(dose)

    if confirmed_missed:
        if len(confirmed_missed) == 1:
            dose = confirmed_missed[0]
            friendly = format_friendly_dose_time(dose["hour"], dose["minute"])
            return (
                f"\n\n**{dose['medication_name']}** was marked **missed** in MedCam for **{friendly}** today. "
                "Check the care plan or GP before giving a late dose."
            )
        slot_count = len({(dose["hour"], dose["minute"]) for dose in confirmed_missed})
        return (
            f"\n\nMedCam shows **{len(confirmed_missed)} missed dose(s)** today "
            f"across **{slot_count} time slot(s)**. "
            "Check the care plan or GP before giving any late doses."
        )

    if not unlogged_past_due:
        return ""

    if not logs:
        if len(unlogged_past_due) == 1:
            dose = unlogged_past_due[0]
            friendly = format_friendly_dose_time(dose["hour"], dose["minute"])
            return (
                f"\n\n**{dose['medication_name']}** was scheduled for **{friendly}** today — "
                "nothing has been logged in **MedCam** yet."
            )
        if medication_name:
            friendly_times = ", ".join(
                format_friendly_dose_time(dose["hour"], dose["minute"])
                for dose in sorted(unlogged_past_due, key=lambda item: (item["hour"], item["minute"]))
            )
            return (
                f"\n\nScheduled dose(s) at **{friendly_times}** haven't been logged in **MedCam** yet today."
            )
        return (
            "\n\nSome earlier scheduled doses haven't been logged in **MedCam** yet today — "
            "that doesn't mean they were missed."
        )

    if len(unlogged_past_due) == 1:
        dose = unlogged_past_due[0]
        friendly = format_friendly_dose_time(dose["hour"], dose["minute"])
        return (
            f"\n\n**{dose['medication_name']}** at **{friendly}** hasn't been logged in MedCam yet today."
        )

    slot_count = len({(dose["hour"], dose["minute"]) for dose in unlogged_past_due})
    return (
        f"\n\n**{len(unlogged_past_due)} scheduled dose(s)** across **{slot_count} time slot(s)** "
        "haven't been logged in MedCam yet today."
    )


def build_next_medication_reply_core(
    plan_items: list,
    *,
    user_text: str = "",
    today_logs: list | None = None,
    now: datetime,
    tz_obj,
    unknown_med_message: str | None = None,
    empty_plan_message: str | None = None,
) -> str:
    from medication_schedule import build_dose_events

    if not plan_items:
        return empty_plan_message or (
            "I don't see a medication plan on file yet. Upload a discharge document in the "
            "**Documents** tab so I can tell you when the next dose is due."
        )

    named_med = extract_medication_name_from_question(user_text, plan_items)
    named_plan = find_plan_item(named_med, plan_items) if named_med else None
    canonical_name = (named_plan.get("name") or named_med) if named_plan else None
    if named_med and not named_plan:
        return unknown_med_message or (
            f"I don't see **{named_med}** on this patient's medication plan. "
            "Open **Documents** to confirm their medicines."
        )

    dose_events = build_dose_events(plan_items)
    if canonical_name:
        dose_events = [dose for dose in dose_events if dose["medication_name"] == canonical_name]
    if not dose_events:
        return empty_plan_message or "No scheduled dose times found in the active plan."

    logs = list(today_logs or [])
    actionable: list[dict] = []
    missed_today: list[dict] = []
    upcoming: list[dict] = []

    for dose in dose_events:
        state = dose_ui_state(dose, now, logs, tz_obj)
        if state == "taken":
            continue
        entry = {**dose, "state": state}
        if state == "actionable":
            actionable.append(entry)
        elif state == "missed":
            missed_today.append(entry)
        occurrence = _dose_next_occurrence(dose, now)
        minutes_until = (occurrence - now).total_seconds() / 60
        upcoming.append({**entry, "occurrence": occurrence, "minutes_until": minutes_until})

    missed_note = summarize_dose_schedule_notes(
        missed_today,
        today_logs=logs,
        tz_obj=tz_obj,
        now=now,
        medication_name=canonical_name,
    )
    subject = canonical_name or "pill"

    if actionable:
        slot = min(actionable, key=lambda item: (item["hour"], item["minute"], item["medication_name"]))
        friendly = format_friendly_dose_time(slot["hour"], slot["minute"])
        same_time = [
            dose for dose in actionable
            if dose["hour"] == slot["hour"] and dose["minute"] == slot["minute"]
        ]
        names = sorted({dose["medication_name"] for dose in same_time})
        if len(names) == 1:
            primary = (
                f"The next dose of **{names[0]}** is due **now** (scheduled for **{friendly}**). "
                "Use **MedCam** to verify and log it when given."
            )
        else:
            primary = (
                f"The next doses due **now** at **{friendly}** are: "
                f"{_format_med_name_list(names)}. "
                "Use **MedCam** to verify and log them when given."
            )
        return primary + missed_note

    if not upcoming:
        first = dose_events[0]
        friendly = format_friendly_dose_time(first["hour"], first["minute"])
        return (
            f"All scheduled **{subject}** doses for today look complete. "
            f"The next dose on the plan is at **{friendly}** on the next scheduled day.\n\n"
            "Open **MedCam** to log doses when you give them."
        )

    next_slot = min(upcoming, key=lambda item: item["minutes_until"])
    same_time = [dose for dose in upcoming if dose["occurrence"] == next_slot["occurrence"]]
    names = sorted({dose["medication_name"] for dose in same_time})
    when_part = _format_next_pill_when(next_slot["occurrence"], now, next_slot["minutes_until"])

    if len(names) == 1:
        primary = f"The next dose of **{names[0]}** is at {when_part}."
    else:
        primary = f"The next doses at {when_part} are: {_format_med_name_list(names)}."

    primary += " Use **MedCam** to verify and log when given."
    return primary + missed_note
