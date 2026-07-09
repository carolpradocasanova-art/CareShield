"""Tests for allergy-aware severity classification."""

import unittest

from ai_helpers import (
    apply_allergy_severity_policy,
    build_allergy_symptom_alerts,
    cap_allergy_report_severity,
    classify_allergic_reaction_severity,
    normalize_symptom_condition_analysis,
    reported_symptom_has_anaphylaxis_red_flags,
)


class AllergySeverityTests(unittest.TestCase):
    allergies = ["Penicillin", "Sulfa drugs"]

    def test_mild_hives_without_red_flags_is_contact_doctor(self):
        symptom = "mild hives on both forearms, no breathing problems"
        self.assertFalse(reported_symptom_has_anaphylaxis_red_flags(symptom))
        self.assertEqual(classify_allergic_reaction_severity(symptom), "contact_doctor")
        self.assertEqual(cap_allergy_report_severity(symptom, "emergency"), "contact_doctor")

    def test_breathing_difficulty_is_emergency(self):
        symptom = "hives and he can't breathe properly"
        self.assertTrue(reported_symptom_has_anaphylaxis_red_flags(symptom))
        self.assertEqual(classify_allergic_reaction_severity(symptom), "emergency")
        self.assertEqual(cap_allergy_report_severity(symptom, "emergency"), "emergency")

    def test_allergy_alerts_use_contact_doctor_for_localized_rash(self):
        alerts = build_allergy_symptom_alerts("localized rash on one arm", self.allergies)
        self.assertTrue(alerts)
        for alert in alerts:
            self.assertEqual(alert["severity_impact"], "contact_doctor")

    def test_allergy_alerts_use_emergency_when_red_flags_present(self):
        alerts = build_allergy_symptom_alerts(
            "hives with facial swelling and wheezing",
            self.allergies,
        )
        self.assertTrue(alerts)
        for alert in alerts:
            self.assertEqual(alert["severity_impact"], "emergency")

    def test_apply_policy_downgrades_ai_emergency_for_penicillin_hives(self):
        raw = normalize_symptom_condition_analysis({
            "is_elevated_risk": True,
            "recommended_severity": "emergency",
            "needs_doctor": True,
            "condition_risks": [{
                "condition_name": "Penicillin",
                "is_relevant": True,
                "severity_impact": "emergency",
                "education_message": "How Penicillin Impacts This Symptom: possible reaction",
            }],
        })
        adjusted = apply_allergy_severity_policy("new hives on arms", raw, self.allergies)
        self.assertEqual(adjusted["recommended_severity"], "contact_doctor")
        self.assertEqual(adjusted["condition_risks"][0]["severity_impact"], "contact_doctor")

    def test_apply_policy_keeps_emergency_for_sulfa_with_anaphylaxis_words(self):
        raw = normalize_symptom_condition_analysis({
            "is_elevated_risk": True,
            "recommended_severity": "emergency",
            "needs_doctor": True,
            "condition_risks": [{
                "condition_name": "Sulfa drugs",
                "is_relevant": True,
                "severity_impact": "emergency",
                "education_message": "How Sulfa drugs Impacts This Symptom: possible reaction",
            }],
        })
        symptom = "widespread hives and throat tightness after new antibiotic"
        adjusted = apply_allergy_severity_policy(symptom, raw, self.allergies)
        self.assertEqual(adjusted["recommended_severity"], "emergency")

    def test_second_allergy_mild_itching_variant(self):
        symptom = "mild itching after lunch, no swelling"
        self.assertEqual(classify_allergic_reaction_severity(symptom), "contact_doctor")
        alerts = build_allergy_symptom_alerts(symptom, ["Latex"])
        self.assertEqual(alerts[0]["severity_impact"], "contact_doctor")

    def test_second_allergy_lip_swelling_variant(self):
        symptom = "lip swelling and hives after medication"
        self.assertEqual(classify_allergic_reaction_severity(symptom), "emergency")


if __name__ == "__main__":
    unittest.main()
