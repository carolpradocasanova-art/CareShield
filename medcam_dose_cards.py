"""MedCam dose card HTML — single source of truth for status badge rendering."""

from __future__ import annotations

import html
import re

from medication_schedule import format_plan_schedule_summary

DEFAULT_PRN_MAX_PER_DAY = 4

SCHEDULED_DOSE_STATUS_LABELS = {
    "not_yet": "Not due yet",
    "actionable": "Due now",
    "taken": "Taken",
    "missed": "Missed",
}

PRN_DOSE_STATUS_LABELS = {
    "prn_available": "Available",
    "prn_wait": "Wait",
    "prn_max": "Max reached for today",
}

SCHEDULED_DOSE_STATUS_CLASSES = {
    "not_yet": "cs-medcam-dose-status--muted",
    "actionable": "cs-medcam-dose-status--due",
    "taken": "cs-medcam-dose-status--taken",
    "missed": "cs-medcam-dose-status--missed",
}

PRN_DOSE_STATUS_CLASSES = {
    "prn_available": "cs-medcam-dose-status--available",
    "prn_wait": "cs-medcam-dose-status--muted",
    "prn_max": "cs-medcam-dose-status--missed",
}


def scheduled_dose_status_label(state: str) -> str:
    return SCHEDULED_DOSE_STATUS_LABELS.get(state, SCHEDULED_DOSE_STATUS_LABELS["not_yet"])


def prn_dose_status_label(status: dict) -> str:
    state = status.get("status", "prn_available")
    label = PRN_DOSE_STATUS_LABELS.get(state, PRN_DOSE_STATUS_LABELS["prn_available"])
    if state == "prn_wait" and status.get("wait_label"):
        return str(status["wait_label"])
    return label


def count_status_label_occurrences(card_html: str, label: str) -> int:
    """Count visible status-label text inside the dose card status element."""
    pattern = re.compile(
        r'<div class="cs-medcam-dose-status[^"]*">'
        + re.escape(html.escape(label))
        + r"</div>",
        re.I,
    )
    return len(pattern.findall(card_html))


def build_medcam_scheduled_dose_card_html(dose: dict, state: str, plan_item: dict | None = None) -> str:
    plan = plan_item or {}
    name = html.escape(str(dose["medication_name"]).upper())
    dosage = str(plan.get("dosage") or "").strip()
    meta_line = html.escape(f"{dosage} {dose['display_time']}".strip())
    schedule_line = html.escape(
        format_plan_schedule_summary(plan).replace(" · ", " • ")
    )
    status_label = scheduled_dose_status_label(state)
    status_class = SCHEDULED_DOSE_STATUS_CLASSES.get(state, SCHEDULED_DOSE_STATUS_CLASSES["not_yet"])
    return f"""
    <div class="cs-medcam-dose-card">
      <div class="cs-medcam-dose-name">{name}</div>
      <div class="cs-medcam-dose-meta">{meta_line}</div>
      <div class="cs-medcam-dose-schedule">{schedule_line}</div>
      <div class="cs-medcam-dose-status {status_class}">{html.escape(status_label)}</div>
    </div>
    """


def build_medcam_prn_dose_card_html(med: dict, status: dict) -> str:
    state = status.get("status", "prn_available")
    name = html.escape(f"{str(med['name']).upper()} · PRN")
    dosage = str(med.get("dosage") or "").strip()
    doses = status.get("doses_today", 0)
    max_d = status.get("max_per_day", DEFAULT_PRN_MAX_PER_DAY)
    meta_line = html.escape(f"{dosage} {doses} / {max_d} today".strip())
    schedule_line = html.escape(
        format_plan_schedule_summary(med).replace(" · ", " • ")
    )
    status_label = prn_dose_status_label(status)
    status_class = PRN_DOSE_STATUS_CLASSES.get(state, PRN_DOSE_STATUS_CLASSES["prn_available"])
    return f"""
    <div class="cs-medcam-dose-card cs-medcam-dose-card--prn">
      <div class="cs-medcam-dose-name">{name}</div>
      <div class="cs-medcam-dose-meta">{meta_line}</div>
      <div class="cs-medcam-dose-schedule">{schedule_line}</div>
      <div class="cs-medcam-dose-status {status_class}">{html.escape(status_label)}</div>
    </div>
    """
