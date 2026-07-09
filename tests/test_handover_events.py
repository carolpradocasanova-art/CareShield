"""Tests for handover SBAR event collection."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from handover_events import (
    collect_sbar_events_from_timelines,
    event_in_handover_period,
    merge_sbar_events,
    parse_handover_datetime,
    build_sbar_handover_user_payload,
)


def test_parse_handover_datetime_handles_space_separated_supabase_timestamp():
    tz = ZoneInfo("UTC")
    parsed = parse_handover_datetime("2026-07-04 21:13:06+00:00", tz)
    assert parsed is not None
    assert parsed.year == 2026


def test_collect_sbar_events_merges_symptom_and_adherence_timelines():
    tz = ZoneInfo("UTC")
    now = datetime.now(timezone.utc).isoformat()
    symptom = [{
        "timestamp": now,
        "text": "Patient was confused after lunch",
        "severity": "monitor",
        "caregiver": "Alex",
        "source": "voice_report",
    }]
    adherence = [{
        "timestamp": now,
        "text": "Warfarin — 8am dose logged as taken",
        "caregiver": "Alex",
        "source": "medication_log",
    }]
    events = collect_sbar_events_from_timelines(
        symptom,
        adherence,
        period_key="this_week",
        tz_obj=tz,
    )
    assert len(events) == 2
    summaries = {event["summary"] for event in events}
    assert "Patient was confused after lunch" in summaries
    assert "Warfarin — 8am dose logged as taken" in summaries


def test_collect_sbar_events_deduplicates_same_entry():
    tz = ZoneInfo("UTC")
    now = datetime.now(timezone.utc).isoformat()
    row = {
        "timestamp": now,
        "text": "MedCam medication check passed",
        "source": "medication_check",
    }
    events = collect_sbar_events_from_timelines(
        [row],
        [row],
        period_key="this_week",
        tz_obj=tz,
    )
    assert len(events) == 1


def test_event_in_handover_period_excludes_missing_timestamp():
    tz = ZoneInfo("UTC")
    assert event_in_handover_period({"text": "orphan"}, "this_week", tz) is False


def test_build_sbar_handover_user_payload_separates_symptoms_and_adherence():
    symptom = [{
        "timestamp": "2026-07-04T12:00:00+00:00",
        "text": "Patient complained of chest pain",
        "source": "voice_report",
        "caregiver": "Carlos",
    }]
    adherence = [{
        "timestamp": "2026-07-04T12:00:00+00:00",
        "text": "Warfarin — 8am dose missed",
        "source": "medication_log",
        "caregiver": "Carlos",
    }]
    payload = build_sbar_handover_user_payload(symptom, adherence)
    assert "SYMPTOM REPORTS & CARE UPDATES" in payload
    assert "chest pain" in payload
    assert "MEDICATION ADHERENCE" in payload
    assert "Warfarin" in payload


def test_build_sbar_handover_user_payload_notes_missing_symptoms():
    payload = build_sbar_handover_user_payload([], [{
        "timestamp": "2026-07-04T12:00:00+00:00",
        "text": "Hydralazine dose missed",
        "source": "medication_log",
    }])
    assert "None logged in this period" in payload
    assert "Hydralazine dose missed" in payload


def test_build_sbar_handover_user_payload_includes_photos_and_peak_severity():
    symptom = [
        {
            "timestamp": "2026-07-09T10:00:00+00:00",
            "text": "Bruising on left leg",
            "source": "symptom_photo",
            "severity": "contact_doctor",
            "caregiver": "Carolina",
            "has_photo": True,
            "photo_finding": "concern",
        },
        {
            "timestamp": "2026-07-09T11:00:00+00:00",
            "text": "Susan had a fall and bumped her head",
            "source": "voice_report",
            "severity": "contact_doctor",
            "caregiver": "Carolina",
        },
        {
            "timestamp": "2026-07-09T11:30:00+00:00",
            "text": "Susan had a fall and bumped her head — worsening",
            "source": "voice_report",
            "severity": "emergency",
            "caregiver": "Carolina",
        },
    ]
    payload = build_sbar_handover_user_payload(symptom, [])
    assert "SYMPTOM PHOTOS LOGGED" in payload
    assert "Bruising on left leg" in payload
    assert "PEAK SEVERITY REACHED IN THIS PERIOD: EMERGENCY" in payload
    assert "[EMERGENCY]" in payload
    assert payload.count("Susan had a fall") == 2


def test_merge_sbar_events_keeps_highest_severity_for_duplicate_key():
    first = {
        "timestamp": "2026-07-09T11:00:00+00:00",
        "text": "Fall with head bump",
        "severity": "contact_doctor",
        "source": "voice_report",
    }
    second = {
        "timestamp": "2026-07-09T11:00:00+00:00",
        "text": "Fall with head bump",
        "severity": "emergency",
        "source": "voice_report",
    }
    merged = merge_sbar_events([first, second])
    assert len(merged) == 1
    assert merged[0]["severity"] == "emergency"

