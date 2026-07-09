"""Tests for Report & Ask symptom context reasoning helpers."""

import unittest

from ai_helpers import (
    build_allergy_symptom_alerts,
    build_medication_symptom_alerts,
    cap_ace_angioedema_report_severity,
    enrich_symptom_condition_analysis,
    extract_allergy_mentions_from_text,
    medication_recent_start_signals,
    medications_to_symptom_context_rows,
    normalize_symptom_condition_analysis,
    summarize_reported_symptom,
    _line_indicates_med_recent_start,
)


class SymptomContextAnalysisTests(unittest.TestCase):
    def test_recent_medication_notes_are_detected(self):
        med = {
            "name": "Ramipril",
            "dosage_instructions": "5 mg · once daily · started this admission",
        }
        signals = medication_recent_start_signals(med)
        self.assertTrue(signals)
        rows = medications_to_symptom_context_rows([med])
        self.assertTrue(rows[0]["is_recent_start"])

    def test_shared_plan_line_only_flags_named_recent_med(self):
        line = "Medications: Atorvastatin 40mg, Metformin 500mg, Lisinopril 10mg started this admission"
        self.assertFalse(_line_indicates_med_recent_start(line, "Atorvastatin"))
        self.assertFalse(_line_indicates_med_recent_start(line, "Metformin"))
        self.assertTrue(_line_indicates_med_recent_start(line, "Lisinopril"))

    def test_stable_medications_do_not_get_recent_boilerplate(self):
        meds = [
            {"name": "Atorvastatin", "dosage_instructions": "40 mg · nightly"},
            {"name": "Lisinopril", "dosage_instructions": "10 mg · started this admission"},
        ]
        alerts = build_medication_symptom_alerts("feeling very tired today", meds)
        by_name = {alert["condition_name"]: alert["education_message"] for alert in alerts}
        self.assertIn("Lisinopril", by_name)
        self.assertIn("recently started or changed", by_name["Lisinopril"])
        if "Atorvastatin" in by_name:
            self.assertNotIn("recently started or changed", by_name["Atorvastatin"])

    def test_extract_allergy_mentions_splits_lists(self):
        text = "Allergies: Penicillin, Sulfa drugs"
        found = extract_allergy_mentions_from_text(text)
        self.assertIn("Penicillin", found)
        self.assertTrue(any("sulfa" in item.lower() for item in found))

    def test_hives_alert_references_documented_allergies(self):
        alerts = build_allergy_symptom_alerts(
            "new hives on both arms",
            ["Penicillin", "Sulfa drugs"],
        )
        self.assertEqual(len(alerts), 2)
        combined = " ".join(alert["education_message"] for alert in alerts).lower()
        self.assertIn("penicillin", combined)
        self.assertIn("sulfa", combined)
        self.assertIn("hives", combined)

    def test_dizziness_flags_recent_ace_inhibitor(self):
        meds = [{
            "name": "Ramipril",
            "dosage_instructions": "5 mg · once daily · started this admission",
            "is_recent_start": True,
        }]
        alerts = build_medication_symptom_alerts("Peter feels dizzy when he stands up", meds)
        self.assertEqual(len(alerts), 1)
        self.assertIn("Ramipril", alerts[0]["education_message"])
        message = alerts[0]["education_message"].lower()
        self.assertTrue("dizzy" in message or "dizziness" in message, message)

    def test_furosemide_education_differs_by_symptom(self):
        meds = [{"name": "Furosemide", "dosage_instructions": "40 mg · morning"}]
        swelling = build_medication_symptom_alerts("worsening leg swelling", meds)[0]["education_message"]
        fatigue = build_medication_symptom_alerts("unusual fatigue today", meds)[0]["education_message"]
        self.assertNotEqual(swelling, fatigue)
        self.assertIn("swelling", swelling.lower())
        self.assertIn("fatigue", fatigue.lower())

    def test_allergy_alerts_surface_for_rash(self):
        alerts = build_allergy_symptom_alerts(
            "new rash on his arms",
            ["Allergy to Penicillin", "Sulfa drugs"],
        )
        self.assertEqual(len(alerts), 2)
        self.assertIn("Penicillin", alerts[0]["education_message"])
        self.assertIn("rash", alerts[0]["education_message"].lower())

    def test_allergy_alerts_skip_unrelated_symptoms(self):
        alerts = build_allergy_symptom_alerts("mild knee stiffness", ["Allergy to Penicillin"])
        self.assertEqual(alerts, [])

    def test_summarize_reported_symptom_uses_report_text(self):
        text = "worsening leg and ankle swelling since yesterday evening"
        self.assertIn("swelling", summarize_reported_symptom(text).lower())

    def test_swallowing_on_ace_inhibitor_flags_angioedema(self):
        meds = [{
            "name": "Lisinopril",
            "dosage_instructions": "10 mg · once daily · started this admission",
        }]
        alerts = build_medication_symptom_alerts(
            "He had trouble swallowing his pill at lunch",
            meds,
        )
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["severity_impact"], "contact_doctor")
        message = alerts[0]["education_message"].lower()
        self.assertIn("angioedema", message)
        self.assertIn("lip", message)
        self.assertIn("999", message)
        self.assertNotIn("common time to notice new side effects", message)

    def test_swallowing_on_stable_ace_inhibitor_still_flags_angioedema(self):
        meds = [{"name": "Lisinopril", "dosage_instructions": "10 mg · once daily"}]
        alerts = build_medication_symptom_alerts("difficulty swallowing food", meds)
        self.assertEqual(alerts[0]["severity_impact"], "contact_doctor")
        self.assertIn("angioedema", alerts[0]["education_message"].lower())

    def test_ace_angioedema_red_flags_escalate_to_emergency(self):
        meds = [{"name": "Ramipril", "dosage_instructions": "5 mg · nightly"}]
        alerts = build_medication_symptom_alerts(
            "lip swelling and trouble breathing",
            meds,
        )
        self.assertEqual(alerts[0]["severity_impact"], "emergency")
        self.assertIn("999", alerts[0]["education_message"])

    def test_enrich_escalates_swallowing_on_ace_inhibitor(self):
        meds = [{"name": "Lisinopril", "dosage_instructions": "10 mg · once daily"}]
        enriched = enrich_symptom_condition_analysis(
            "trouble swallowing his medication at lunch",
            normalize_symptom_condition_analysis({}),
            [],
            meds,
        )
        self.assertEqual(enriched["recommended_severity"], "contact_doctor")
        med_alerts = enriched.get("medication_symptom_alerts") or []
        self.assertTrue(any("angioedema" in a["education_message"].lower() for a in med_alerts))

    def test_cap_ace_angioedema_raises_monitor_to_contact_doctor(self):
        meds = [{"name": "Lisinopril", "dosage_instructions": "10 mg · once daily"}]
        severity = cap_ace_angioedema_report_severity(
            "trouble swallowing at lunch",
            "monitor",
            meds,
        )
        self.assertEqual(severity, "contact_doctor")

    def test_cap_ace_angioedema_raises_to_emergency_with_airway_flags(self):
        meds = [{"name": "Lisinopril", "dosage_instructions": "10 mg · once daily"}]
        severity = cap_ace_angioedema_report_severity(
            "facial swelling and hoarse voice",
            "monitor",
            meds,
        )
        self.assertEqual(severity, "emergency")

    def test_cap_ace_angioedema_skips_non_ace_patients(self):
        meds = [{"name": "Metformin", "dosage_instructions": "500 mg · twice daily"}]
        severity = cap_ace_angioedema_report_severity(
            "trouble swallowing at lunch",
            "monitor",
            meds,
        )
        self.assertEqual(severity, "monitor")

    def test_chest_tightness_not_mislabeled_as_breathing_difficulty(self):
        label = summarize_reported_symptom(
            "chest tightness, can still breathe fine",
            "John",
        )
        self.assertIn("chest tightness", label.lower())
        self.assertNotIn("breathing difficulty", label.lower())

    def test_enrich_uses_current_report_symptom_not_prior_breathing(self):
        meds = [{"name": "Lisinopril", "dosage_instructions": "10 mg · once daily"}]
        enriched = enrich_symptom_condition_analysis(
            "chest tightness, can still breathe fine",
            normalize_symptom_condition_analysis({
                "symptom_identified": "breathing difficulty",
                "condition_risks": [],
            }),
            [],
            meds,
            active_patient_name="John",
        )
        self.assertIn("chest tightness", enriched["symptom_identified"].lower())
        self.assertNotIn("breathing difficulty", enriched["symptom_identified"].lower())


class ConnectedReportLinkTests(unittest.TestCase):
    def test_leg_swelling_links_to_earlier_swelling_not_confusion(self):
        from symptom_linking import find_symptom_related_prior_incidents

        prior = [
            {"text": "Peter seemed confused after lunch", "symptoms": ["confusion"], "timestamp": "1"},
            {"text": "mild ankle swelling noticed", "symptoms": ["swelling"], "timestamp": "2"},
            {"text": "rash on forearm", "symptoms": ["rash"], "timestamp": "3"},
        ]
        related = find_symptom_related_prior_incidents("worsening leg swelling today", prior, limit=2)
        self.assertGreaterEqual(len(related), 1)
        self.assertIn("swelling", related[0]["text"].lower())
        self.assertNotIn("confus", related[0]["text"].lower())


if __name__ == "__main__":
    unittest.main()
