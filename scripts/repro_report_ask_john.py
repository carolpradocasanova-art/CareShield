#!/usr/bin/env python3
"""Reproduce Report & Ask context for a brand-new patient with one dizziness report."""

from __future__ import annotations

from ai_helpers import (
    enforce_report_ask_session_evidence,
)
from symptom_linking import count_linked_reports, detect_session_escalation_triggers


def simulate_new_patient_report_ask(user_text: str) -> dict:
    prior_incidents: list = []
    session_context = "PRIOR SESSION REPORTS: None yet this session."
    timeline_context = "RECENT SYMPTOM & CARE TIMELINE: No prior reports on file yet."
    session_triggers = detect_session_escalation_triggers(user_text, prior_incidents)
    context_report_count = count_linked_reports(prior_incidents, user_text, session_triggers)
    fake_ai_reply = (
        "John may be experiencing orthostatic dizziness. Check blood pressure when sitting "
        "and standing, ensure good hydration, and mention this to his GP — Lisinopril and "
        "Furosemide can both contribute to lightheadedness on standing."
    )
    reply, session_triggers, context_report_count = enforce_report_ask_session_evidence(
        fake_ai_reply,
        prior_incidents=prior_incidents,
        session_triggers=session_triggers,
        context_report_count=context_report_count,
    )
    return {
        "session_prior_count": len(prior_incidents),
        "session_context": session_context,
        "timeline_context": timeline_context,
        "session_triggers": session_triggers,
        "context_report_count": context_report_count,
        "reply_preview": reply[:240],
    }


def main() -> None:
    user_text = "John felt dizzy and lightheaded when he stood up this morning."
    outcome = simulate_new_patient_report_ask(user_text)
    print("=== Report & Ask repro (new patient, single dizziness message) ===")
    for key, value in outcome.items():
        print(f"{key}: {value!r}")

    assert outcome["session_prior_count"] == 0
    assert outcome["context_report_count"] == 1
    assert outcome["session_triggers"] == []
    assert "None yet this session" in outcome["session_context"]
    assert "No prior reports on file yet" in outcome["timeline_context"]
    assert "linked reports" not in outcome["reply_preview"].lower()
    assert "Connected to earlier reports" not in outcome["reply_preview"]
    print("\nPASS: no fabricated linked reports for brand-new patient session.")


if __name__ == "__main__":
    main()
