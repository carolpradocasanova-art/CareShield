"""Tests that routine care questions do not receive urgency severity tags."""

import unittest

from ai_helpers import (
    cap_informational_question_severity,
    is_pure_informational_care_question,
    reports_health_symptom_topic,
)


class InformationalQuestionSeverityTests(unittest.TestCase):
    def test_furosemide_working_question_is_informational(self):
        question = "How's his Furosemide working?"
        self.assertTrue(is_pure_informational_care_question(question))
        self.assertFalse(reports_health_symptom_topic(question))

    def test_medication_interaction_question_is_informational(self):
        question = "Can John take his Metformin and Lisinopril at the same time?"
        self.assertTrue(is_pure_informational_care_question(question))

    def test_cap_drops_ai_contact_doctor_for_informational_question(self):
        question = "How's his Furosemide working?"
        self.assertEqual(
            cap_informational_question_severity(question, "contact_doctor"),
            "ok",
        )

    def test_cap_drops_monitor_for_informational_question(self):
        question = "What is Furosemide for?"
        self.assertEqual(
            cap_informational_question_severity(question, "monitor"),
            "ok",
        )

    def test_symptom_question_still_allows_severity(self):
        question = "Could Furosemide be causing his ankle swelling?"
        self.assertFalse(is_pure_informational_care_question(question))
        self.assertEqual(
            cap_informational_question_severity(question, "contact_doctor"),
            "contact_doctor",
        )

    def test_not_working_question_expresses_concern(self):
        question = "Why isn't his Furosemide working anymore?"
        self.assertFalse(is_pure_informational_care_question(question))

    def test_voice_report_not_treated_as_informational_question(self):
        report = "He seems more breathless after lunch"
        self.assertFalse(is_pure_informational_care_question(report))


if __name__ == "__main__":
    unittest.main()
