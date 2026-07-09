"""Tests for medication recent-start detection across phrasing and document formats."""

import unittest
from unittest.mock import patch

from ai_helpers import (
    _line_indicates_med_recent_start,
    _enrich_medications_from_clinical_notes,
    _plan_text_suggests_recent_start_for_med,
    build_medication_symptom_alerts,
    medication_has_explicit_recent_start,
    medication_recent_start_signals,
    medications_to_symptom_context_rows,
)


class MedicationRecentStartDetectionTests(unittest.TestCase):
    def test_comma_after_dose_still_detects_start(self):
        line = "Lisinopril 10 mg daily, started this admission"
        self.assertTrue(_line_indicates_med_recent_start(line, "Lisinopril"))

    def test_realistic_phrasing_matrix(self):
        cases = [
            ("Lisinopril 5 mg · once daily · started this admission", "Lisinopril", True),
            ("Lisinopril 5mg (started this admission)", "Lisinopril", True),
            ("Lisinopril — new medication this admission", "Lisinopril", True),
            ("Lisinopril 10 mg, started 10 days ago", "Lisinopril", True),
            ("Lisinopril dose changed on 12/03/2026", "Lisinopril", True),
            ("Apixaban 5 mg twice daily", "Apixaban", False),
            ("Metformin 500 mg · with meals", "Metformin", False),
            ("Atorvastatin 40mg, Metformin 500mg, Lisinopril 10mg started this admission", "Atorvastatin", False),
            ("Atorvastatin 40mg, Metformin 500mg, Lisinopril 10mg started this admission", "Lisinopril", True),
            ("| Lisinopril | 5mg | Started this admission |", "Lisinopril", True),
        ]
        for line, med_name, expected in cases:
            with self.subTest(line=line, med=med_name):
                self.assertEqual(_line_indicates_med_recent_start(line, med_name), expected)

    def test_dosage_only_field_with_comma_start_note(self):
        med = {
            "name": "Lisinopril",
            "dosage_instructions": "10 mg daily, started this admission",
        }
        self.assertTrue(medication_has_explicit_recent_start(med))
        self.assertTrue(medication_recent_start_signals(med))

    def test_multiline_plan_name_and_annotation_on_next_line(self):
        plan = "Current medications:\nLisinopril 5 mg\nstarted this admission"
        self.assertTrue(_plan_text_suggests_recent_start_for_med(plan, "Lisinopril"))
        med = {"name": "Lisinopril", "dosage_instructions": "5 mg daily"}
        med["notes"] = "started this admission"
        med["is_recent_start"] = True
        self.assertTrue(medication_has_explicit_recent_start(med))

    def test_only_lisinopril_gets_recent_wording_in_alerts(self):
        meds = [
            {"name": "Atorvastatin", "dosage_instructions": "40 mg nightly"},
            {"name": "Metformin", "dosage_instructions": "500 mg with meals"},
            {"name": "Apixaban", "dosage_instructions": "5 mg twice daily"},
            {"name": "Lisinopril", "dosage_instructions": "10 mg daily, started this admission"},
        ]
        alerts = build_medication_symptom_alerts("Peter feels dizzy", meds)
        by_name = {item["condition_name"]: item["education_message"] for item in alerts}
        self.assertIn("Lisinopril", by_name)
        self.assertIn("recently started or changed", by_name["Lisinopril"].lower())
        for stable_name in ("Atorvastatin", "Metformin", "Apixaban"):
            if stable_name in by_name:
                self.assertNotIn("recently started or changed", by_name[stable_name].lower())

    def test_document_text_enriches_medication_without_plan(self):
        """Uploaded document notes should enrich meds even when no patient_plan row exists."""
        med = {
            "name": "Lisinopril",
            "dosage_instructions": "10mg · once daily · 1 pill(s) per dose",
        }
        document_text = (
            "Current medications:\n"
            "Lisinopril 10 mg Oral Once daily Started this admission for blood pressure\n"
            "Note: Lisinopril was newly started during this admission."
        )
        with patch("ai_helpers.get_latest_patient_plan", return_value=None), patch(
            "ai_helpers.fetch_patient_document_texts",
            return_value=[document_text],
        ):
            enriched = _enrich_medications_from_clinical_notes([med], patient_id=14)
        self.assertTrue(enriched[0].get("is_recent_start"))
        self.assertTrue(medication_has_explicit_recent_start(enriched[0]))
        with patch("ai_helpers.get_latest_patient_plan", return_value=None), patch(
            "ai_helpers.fetch_patient_document_texts",
            return_value=[document_text],
        ):
            rows = medications_to_symptom_context_rows([med], patient_id=14)
        lisi = next(item for item in rows if item["name"] == "Lisinopril")
        self.assertTrue(lisi["is_recent_start"])
        alert = build_medication_symptom_alerts("Peter feels dizzy", rows)[0]["education_message"].lower()
        self.assertIn("recently started or changed", alert)


if __name__ == "__main__":
    unittest.main()
