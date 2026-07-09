"""Tests that caregiver-facing text always uses the active patient profile name."""

import re
import unittest

from ai_helpers import (
    build_medication_symptom_alerts,
    enforce_active_patient_name_in_text,
    extract_message_patient_name_candidates,
    format_medcam_shift_log_for_timeline,
    retailor_education_messages_to_symptom,
    summarize_reported_symptom,
)


class ActivePatientNameEnforcementTests(unittest.TestCase):
    def test_extracts_mistyped_name_from_caregiver_message(self):
        message = "peter's lips look mildly swollen after lunch"
        candidates = extract_message_patient_name_candidates(message)
        self.assertIn("peter", [item.lower() for item in candidates])

    def test_wrong_message_name_replaced_with_active_profile(self):
        reply = (
            "It's important to keep an eye on Peter's mildly swollen lips. "
            "Because Peter has essential hypertension, contact the GP if swelling worsens."
        )
        message = "peter's lips look mildly swollen after lunch"
        result = enforce_active_patient_name_in_text(reply, "Frank", message)
        self.assertIn("Frank", result)
        self.assertNotRegex(result, r"\bPeter\b", re.I)

    def test_active_profile_name_preserved_when_caregiver_typed_it_correctly(self):
        reply = "Frank's lips look mildly swollen — monitor closely today."
        message = "Frank's lips look mildly swollen after lunch"
        result = enforce_active_patient_name_in_text(reply, "Frank", message)
        self.assertIn("Frank", result)
        self.assertNotIn("Peter", result)

    def test_summarize_reported_symptom_uses_active_profile_not_message_name(self):
        label = summarize_reported_symptom(
            "peter's lips look mildly swollen after lunch",
            "Frank",
        )
        self.assertIn("Frank", label)
        self.assertNotRegex(label, r"\bPeter\b", re.I)
        self.assertIn("swelling", label.lower())

    def test_summarize_reported_symptom_normalizes_shouty_caps_and_pronouns(self):
        label = summarize_reported_symptom("HE WOKE UP WITH FEVER", "Harold")
        self.assertEqual(label, "Harold's fever")
        tailored = retailor_education_messages_to_symptom(
            "HE WOKE UP WITH FEVER",
            [{
                "condition_name": "Type 2 diabetes mellitus",
                "is_relevant": True,
                "education_message": (
                    "How Type 2 diabetes mellitus Impacts This Symptom: "
                    "Fever can be more concerning in patients with diabetes."
                ),
            }],
            "Harold",
        )
        self.assertIn("For Harold's fever:", tailored[0]["education_message"])
        self.assertNotIn("hE WOKE UP WITH FEVER", tailored[0]["education_message"])

    def test_medication_alert_uses_active_profile_name(self):
        meds = [{
            "name": "Lisinopril",
            "dosage_instructions": "10 mg daily",
            "is_recent_start": True,
        }]
        alerts = build_medication_symptom_alerts(
            "peter's lips look mildly swollen after lunch",
            meds,
            active_patient_name="Frank",
        )
        self.assertEqual(len(alerts), 1)
        message = alerts[0]["education_message"]
        self.assertIn("Frank", message)
        self.assertNotRegex(message, r"\bPeter\b", re.I)


if __name__ == "__main__":
    unittest.main()
