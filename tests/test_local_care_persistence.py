"""Tests for local patient care persistence."""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import patient_care_storage as storage
from ai_helpers import (
    fetch_patient_care_reports,
    fetch_patient_chat_thread,
    merge_chat_messages_for_storage,
    save_patient_care_report,
    save_patient_chat_thread,
    save_shift_log,
)


class LocalCarePersistenceTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._root = Path(self._tmpdir)
        patcher = patch.object(storage, "_DATA_ROOT", self._root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(lambda: shutil.rmtree(self._tmpdir, ignore_errors=True))

    def test_care_reports_survive_save_and_reload(self):
        saved = save_patient_care_report(
            "10",
            report_text="She was dizzy this morning",
            summary="Patient dizzy this morning",
            severity="monitor",
            source="voice_report",
            reported_at="2026-07-04T10:00:00+00:00",
            caregiver_name="Carol",
        )
        self.assertIsNotNone(saved)
        rows = fetch_patient_care_reports("10", limit=50)
        self.assertEqual(len(rows), 1)
        self.assertIn("dizzy", rows[0]["summary"].lower())

    def test_chat_thread_persists_locally(self):
        messages = [
            {"role": "user", "content": "Fever overnight", "timestamp": "2026-07-04T08:00:00+00:00"},
            {"role": "assistant", "content": "Monitor closely.", "severity": "monitor"},
        ]
        self.assertTrue(save_patient_chat_thread("11", messages))
        loaded = fetch_patient_chat_thread("11")
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["content"], "Fever overnight")

    def test_chat_thread_appends_without_dropping_prior_history(self):
        prior = [
            {"role": "user", "content": "Older report", "timestamp": "2026-07-01T08:00:00+00:00"},
            {"role": "assistant", "content": "Older reply", "timestamp": "2026-07-01T08:01:00+00:00"},
        ]
        self.assertTrue(save_patient_chat_thread("12", prior))
        session_only = [
            {"role": "welcome", "content": "Welcome"},
            {"role": "user", "content": "New dizziness today", "timestamp": "2026-07-04T09:00:00+00:00"},
            {"role": "assistant", "content": "Monitor and call GP if worse.", "timestamp": "2026-07-04T09:01:00+00:00"},
        ]
        self.assertTrue(save_patient_chat_thread("12", session_only))
        loaded = fetch_patient_chat_thread("12")
        self.assertEqual(len(loaded), 4)
        self.assertEqual(loaded[0]["content"], "Older report")
        self.assertEqual(loaded[-1]["content"], "Monitor and call GP if worse.")

    def test_merge_chat_messages_for_storage_dedupes(self):
        existing = [{"role": "user", "content": "Same", "timestamp": "t1"}]
        session = [
            {"role": "user", "content": "Same", "timestamp": "t1"},
            {"role": "user", "content": "New", "timestamp": "t2"},
        ]
        merged = merge_chat_messages_for_storage(existing, session)
        self.assertEqual(len(merged), 2)

    def test_save_shift_log_falls_back_without_patient_id_column(self):
        with patch("ai_helpers.supabase") as mock_supabase:
            table = mock_supabase.table.return_value
            table.insert.return_value.execute.side_effect = [
                Exception("no patient_id column"),
                None,
            ]
            ok = save_shift_log(
                caregiver_name="Carol",
                source="voice_report",
                summary="Shortness of breath after walking",
                severity="monitor",
                patient_id="10",
            )
            self.assertTrue(ok)
            final_payload = table.insert.call_args_list[-1][0][0]
            self.assertIn("[[patient:10]]", final_payload["summary"])
            self.assertNotIn("patient_id", final_payload)


if __name__ == "__main__":
    unittest.main()
