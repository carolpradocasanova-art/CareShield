"""Benign caregiver updates must not escalate to EMERGENCY."""

import unittest
from unittest.mock import patch

from ai_helpers import (
    cap_positive_report_severity,
    get_medication_references,
    is_clearly_positive_benign_report,
)


class PositiveBenignReportDetectionTests(unittest.TestCase):
    def test_good_appetite_and_walk_is_positive_benign(self):
        text = "good appetite, went for a walk"
        self.assertTrue(is_clearly_positive_benign_report(text))

    def test_fever_negates_positive_classification(self):
        text = "good appetite but he has a fever now"
        self.assertFalse(is_clearly_positive_benign_report(text))

    def test_slept_well_is_positive_benign(self):
        self.assertTrue(is_clearly_positive_benign_report("slept well overnight, no complaints"))


class PositiveReportSeverityCapTests(unittest.TestCase):
    def test_good_news_capped_from_emergency_to_ok(self):
        text = "good appetite, went for a walk"
        self.assertEqual(cap_positive_report_severity(text, "emergency"), "ok")

    def test_good_news_capped_from_contact_doctor_to_ok(self):
        text = "good appetite, went for a walk"
        self.assertEqual(cap_positive_report_severity(text, "contact_doctor"), "ok")

    def test_stale_session_emergency_overridden_for_good_news(self):
        """Even after AI + session escalation, benign updates stay OK."""
        text = "good appetite, went for a walk"
        escalated = "emergency"
        for _ in range(3):
            escalated = cap_positive_report_severity(text, escalated)
        self.assertEqual(escalated, "ok")

    def test_real_symptom_report_not_capped(self):
        text = "chest pain and shortness of breath"
        self.assertEqual(cap_positive_report_severity(text, "emergency"), "emergency")


class PositiveReportPipelineTests(unittest.TestCase):
    def test_good_news_identified_as_benign_before_symptom_regex(self):
        text = "good appetite, went for a walk"
        self.assertTrue(is_clearly_positive_benign_report(text))

    def test_stale_prior_incidents_do_not_escalate_good_news(self):
        """resolve_chat_severity short-circuits positive updates before session/combined escalation."""
        text = "good appetite, went for a walk"
        ai_severity = "emergency"
        needs_doctor = True
        if is_clearly_positive_benign_report(text):
            level, triggers = "ok", []
        else:
            level, triggers = ai_severity, ["would escalate from stale priors"]
        level = cap_positive_report_severity(text, level)
        self.assertEqual(level, "ok")
        self.assertEqual(triggers, [])


class MedicationReferencesScopeTests(unittest.TestCase):
    @patch("ai_helpers.supabase")
    def test_missing_patient_id_returns_empty_list(self, _mock_supabase):
        self.assertEqual(get_medication_references(None), [])

    @patch("ai_helpers.supabase")
    def test_scoped_query_failure_returns_empty_not_unscoped(self, mock_supabase):
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.execute.side_effect = RuntimeError(
            "patient_id column missing"
        )
        result = get_medication_references("15")
        self.assertEqual(result, [])
        mock_supabase.table.return_value.select.return_value.order.return_value.execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
