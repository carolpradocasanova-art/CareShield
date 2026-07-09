"""Ensure symptom and legacy logs never bleed across patient profiles."""

import unittest
from unittest.mock import patch

from ai_helpers import (
    backfill_legacy_shift_logs_to_local_care,
    shift_log_belongs_to_patient,
)


class ShiftLogPatientScopeTests(unittest.TestCase):
    def test_explicit_patient_id_must_match(self):
        row = {"patient_id": 12, "summary": "Fever reported", "source": "voice_report"}
        self.assertTrue(shift_log_belongs_to_patient(row, "12"))
        self.assertFalse(shift_log_belongs_to_patient(row, "99"))

    def test_marker_scopes_row_to_one_patient(self):
        row = {"summary": "[[patient:12]] Fever reported", "source": "voice_report"}
        self.assertTrue(shift_log_belongs_to_patient(row, "12"))
        self.assertFalse(shift_log_belongs_to_patient(row, "99"))

    def test_unmarked_rows_rejected_when_multiple_patients_exist(self):
        row = {"summary": "Fever reported", "source": "voice_report"}
        patients = [{"id": "12", "name": "Harold"}, {"id": "15", "name": "John"}]
        with patch("ai_helpers.list_account_patients", return_value=patients):
            self.assertFalse(shift_log_belongs_to_patient(row, "15"))

    def test_unmarked_rows_allowed_only_for_single_patient_account(self):
        row = {"summary": "Fever reported", "source": "voice_report"}
        patients = [{"id": "12", "name": "Harold"}]
        with patch("ai_helpers.list_account_patients", return_value=patients):
            self.assertTrue(shift_log_belongs_to_patient(row, "12"))
            self.assertFalse(shift_log_belongs_to_patient(row, "99"))


class LegacyBackfillIsolationTests(unittest.TestCase):
    @patch("ai_helpers.mark_legacy_backfill_completed")
    @patch("ai_helpers.save_local_care_report")
    @patch("ai_helpers.supabase")
    @patch("ai_helpers.legacy_backfill_completed", return_value=False)
    @patch("ai_helpers.list_account_patients")
    def test_new_patient_does_not_inherit_unmarked_legacy_logs(
        self,
        mock_list_patients,
        _mock_backfill_done,
        mock_supabase,
        mock_save_local,
        _mock_mark_done,
    ):
        mock_list_patients.return_value = [
            {"id": "12", "name": "Harold"},
            {"id": "15", "name": "John"},
        ]
        mock_supabase.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value.data = [
            {
                "summary": "Harold woke up with fever",
                "source": "voice_report",
                "severity": "monitor",
                "created_at": "2026-06-22T10:00:00+00:00",
            },
            {
                "patient_id": 12,
                "summary": "Chest pain after lunch",
                "source": "voice_report",
                "severity": "contact_doctor",
                "created_at": "2026-06-22T11:00:00+00:00",
            },
        ]

        imported = backfill_legacy_shift_logs_to_local_care("15", limit=50)

        self.assertEqual(imported, 0)
        mock_save_local.assert_not_called()


if __name__ == "__main__":
    unittest.main()
