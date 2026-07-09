"""Next-dose replies must honour the named medication and avoid fabricated missed-dose claims."""

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from ai_helpers import get_medication_logs
from medication_dose_queries import (
    build_next_medication_reply_core,
    extract_medication_name_from_question,
)

JOHN_PLAN = [
    {"name": "Furosemide", "timing": "once daily in the morning"},
    {"name": "Lisinopril", "timing": "once daily"},
    {"name": "Metformin", "timing": "twice daily with meals"},
]


class MedicationNameExtractionTests(unittest.TestCase):
    def test_extracts_furosemide_from_question(self):
        question = "When does John need his next Furosemide dose?"
        self.assertEqual(
            extract_medication_name_from_question(question, JOHN_PLAN),
            "Furosemide",
        )

    def test_extracts_metformin_stem(self):
        question = "When is the next metformin dose due?"
        self.assertEqual(
            extract_medication_name_from_question(question, JOHN_PLAN),
            "Metformin",
        )

    def test_generic_next_dose_question_has_no_named_filter(self):
        self.assertIsNone(
            extract_medication_name_from_question("When is the next dose due?", JOHN_PLAN)
        )


class NamedNextDoseReplyTests(unittest.TestCase):
    def setUp(self):
        self.tz = ZoneInfo("UTC")
        self.evening = datetime(2026, 7, 8, 20, 0, tzinfo=self.tz)

    def test_furosemide_question_answers_about_furosemide_not_metformin(self):
        reply = build_next_medication_reply_core(
            JOHN_PLAN,
            user_text="When does John need his next Furosemide dose?",
            today_logs=[],
            now=self.evening,
            tz_obj=self.tz,
        )
        lower = reply.lower()
        self.assertIn("furosemide", lower)
        self.assertNotIn("metformin", lower)
        self.assertNotIn("may have missed", lower)
        self.assertNotIn("missed 3 doses", lower)

    def test_metformin_question_at_evening_targets_metformin(self):
        reply = build_next_medication_reply_core(
            JOHN_PLAN,
            user_text="When does John need his next Metformin dose?",
            today_logs=[],
            now=self.evening,
            tz_obj=self.tz,
        )
        self.assertIn("Metformin", reply)
        self.assertNotIn("Furosemide", reply)

    def test_lisinopril_question_targets_lisinopril(self):
        reply = build_next_medication_reply_core(
            JOHN_PLAN,
            user_text="When should I give the next Lisinopril dose?",
            today_logs=[],
            now=self.evening,
            tz_obj=self.tz,
        )
        self.assertIn("Lisinopril", reply)
        self.assertNotIn("Metformin", reply)

    def test_no_logs_uses_unlogged_wording_not_missed_counts(self):
        reply = build_next_medication_reply_core(
            JOHN_PLAN,
            user_text="When does John need his next Furosemide dose?",
            today_logs=[],
            now=self.evening,
            tz_obj=self.tz,
        )
        self.assertNotIn("may have missed", reply.lower())
        self.assertNotIn("missed 3 doses", reply.lower())

    def test_confirmed_missed_log_uses_medcam_wording(self):
        reply = build_next_medication_reply_core(
            [{"name": "Furosemide", "timing": "once daily in the morning"}],
            user_text="Next Furosemide dose?",
            today_logs=[{
                "medication_name": "Furosemide",
                "scheduled_time": "08:00",
                "status": "missed",
                "logged_at": "2026-07-08T08:30:00+00:00",
            }],
            now=self.evening,
            tz_obj=self.tz,
        )
        self.assertIn("marked **missed** in MedCam", reply)


class MedicationLogScopeTests(unittest.TestCase):
    def test_missing_patient_id_returns_no_logs(self):
        self.assertEqual(get_medication_logs(None), [])


if __name__ == "__main__":
    unittest.main()
