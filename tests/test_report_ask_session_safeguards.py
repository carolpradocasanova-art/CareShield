"""Safeguards against fabricated linked-report citations in Report & Ask."""

import unittest

from ai_helpers import (
    care_report_is_suspect_cross_profile_import,
    enforce_report_ask_session_evidence,
    strip_ungrounded_linked_report_citations,
)
from symptom_linking import (
    count_linked_reports,
    detect_session_escalation_triggers,
)


class SuspectCrossProfileImportTests(unittest.TestCase):
    def test_legacy_backfill_rows_are_dropped(self):
        row = {"source": "legacy_backfill", "reported_at": "2026-07-04T22:02:00+00:00"}
        self.assertTrue(care_report_is_suspect_cross_profile_import(row))

    def test_report_before_patient_creation_is_dropped(self):
        row = {"reported_at": "2026-07-04T22:02:00+00:00", "created_at": "2026-07-08T10:00:00+00:00"}
        patient = {"created_at": "2026-07-08T09:00:00+00:00"}
        self.assertTrue(care_report_is_suspect_cross_profile_import(row, patient))

    def test_same_day_report_for_patient_is_kept(self):
        row = {
            "source": "voice_report",
            "reported_at": "2026-07-08T10:30:00+00:00",
            "created_at": "2026-07-08T10:31:00+00:00",
        }
        patient = {"created_at": "2026-07-08T09:00:00+00:00"}
        self.assertFalse(care_report_is_suspect_cross_profile_import(row, patient))

    def test_voice_report_saved_later_than_event_is_kept(self):
        row = {
            "source": "voice_report",
            "reported_at": "2026-07-04T10:00:00+00:00",
            "created_at": "2026-07-08T10:00:00+00:00",
        }
        self.assertFalse(care_report_is_suspect_cross_profile_import(row))

    def test_legacy_backfill_saved_long_after_event_is_dropped(self):
        row = {
            "source": "legacy_backfill",
            "reported_at": "2026-07-04T10:00:00+00:00",
            "created_at": "2026-07-08T10:00:00+00:00",
        }
        self.assertTrue(care_report_is_suspect_cross_profile_import(row))


    def test_shift_log_utc_prefix_rows_are_dropped(self):
        row = {
            "source": "voice_report",
            "summary": "[2026-07-04 21:02 UTC] The patient experienced a fall this morning",
            "reported_at": "2026-07-08T10:00:00+00:00",
        }
        self.assertTrue(care_report_is_suspect_cross_profile_import(row))

    def test_enforce_strips_when_priors_are_stale_imports(self):
        polluted = [{
            "text": "[2026-07-04 21:02 UTC] The patient experienced a fall this morning",
            "timestamp": "2026-07-04T21:02:00+00:00",
        }]
        reply = (
            "Dizziness noted.\n\n**Connected to earlier reports:** Sat 04 Jul — fall. "
            "Taken together (confusion and fall both reported this session), these updates suggest a higher level of concern."
        )
        cleaned, triggers, count = enforce_report_ask_session_evidence(
            reply,
            prior_incidents=polluted,
            session_triggers=["confusion and fall both reported this session"],
            context_report_count=3,
            patient_id="16",
        )
        self.assertEqual(triggers, [])
        self.assertEqual(count, 1)
        self.assertNotIn("Connected to earlier reports", cleaned)


class SessionEvidenceEnforcementTests(unittest.TestCase):
    def test_strip_hallucinated_connection_note(self):
        reply = (
            "John felt dizzy when standing.\n\n"
            "**Connected to earlier reports:** Sat 04 Jul — fall in hallway. "
            "Taken together (confusion and fall both reported this session), these updates suggest a higher level of concern."
        )
        cleaned = strip_ungrounded_linked_report_citations(reply)
        self.assertNotIn("Connected to earlier reports", cleaned)
        self.assertNotIn("confusion and fall", cleaned)
        self.assertIn("dizzy", cleaned)

    def test_enforce_clears_triggers_and_count_without_priors(self):
        reply = "Based on 3 linked reports, call 999 now."
        cleaned, triggers, count = enforce_report_ask_session_evidence(
            reply,
            prior_incidents=[],
            session_triggers=["confusion and fall both reported this session"],
            context_report_count=3,
        )
        self.assertEqual(triggers, [])
        self.assertEqual(count, 1)
        self.assertNotIn("linked reports", cleaned.lower())

    def test_enforce_preserves_evidence_when_priors_exist(self):
        priors = [{"text": "mild ankle swelling", "timestamp_display": "Mon 10:00 AM"}]
        reply, triggers, count = enforce_report_ask_session_evidence(
            "Worsening swelling today.",
            prior_incidents=priors,
            session_triggers=["recurrent peripheral swelling in this session"],
            context_report_count=2,
            patient_id="10",
        )
        self.assertEqual(triggers, ["recurrent peripheral swelling in this session"])
        self.assertEqual(count, 2)
        self.assertEqual(reply, "Worsening swelling today.")


class NewPatientDizzinessEscalationTests(unittest.TestCase):
    def test_no_session_priors_means_single_report_count(self):
        self.assertEqual(
            count_linked_reports([], "John felt dizzy and lightheaded when he stood up."),
            1,
        )

    def test_stale_cross_patient_priors_do_not_trigger_confusion_fall_combo_for_dizziness(self):
        polluted_priors = [
            {"text": "confusion and disorientation after lunch", "timestamp": "1"},
            {"text": "fell in the hallway", "timestamp": "2"},
        ]
        user_text = "John felt dizzy and lightheaded when he stood up this morning."
        triggers = detect_session_escalation_triggers(user_text, polluted_priors)
        self.assertNotIn("confusion and fall both reported this session", triggers)

    def test_brand_new_session_has_no_escalation_triggers(self):
        user_text = "John felt dizzy and lightheaded when he stood up this morning."
        triggers = detect_session_escalation_triggers(user_text, [])
        self.assertEqual(triggers, [])


class SuspectCareReportPurgeTests(unittest.TestCase):
    def test_purge_removes_legacy_backfill_from_local_disk(self):
        import shutil
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        import patient_care_storage as storage
        from ai_helpers import purge_suspect_cross_profile_care_reports, save_patient_care_report

        tmpdir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(tmpdir, ignore_errors=True))
        with patch.object(storage, "_DATA_ROOT", Path(tmpdir)):
            save_patient_care_report(
                "20",
                report_text="Imported fall from another profile",
                summary="Imported fall from another profile",
                severity="emergency",
                source="legacy_backfill",
                reported_at="2026-07-04T22:02:00+00:00",
            )
            save_patient_care_report(
                "20",
                report_text="Real report today",
                summary="Real report today",
                severity="monitor",
                source="voice_report",
                reported_at="2026-07-08T10:00:00+00:00",
            )
            removed = purge_suspect_cross_profile_care_reports("20")
            self.assertEqual(removed, 1)
            rows = storage.fetch_local_care_reports("20")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source"], "voice_report")


if __name__ == "__main__":
    unittest.main()
