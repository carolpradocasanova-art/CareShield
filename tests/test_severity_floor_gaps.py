"""Explicit pass/fail tests for the four SEVERITY_SPEC gap closures."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ai_helpers import (
    apply_report_severity_floor_caps,
    build_medication_symptom_alerts,
    cap_anticoagulant_head_trauma_report_severity,
    cap_beta_blocker_bradycardia_report_severity,
    cap_cyanosis_report_severity,
    cap_hypoglycemia_report_severity,
    enrich_symptom_condition_analysis,
    normalize_symptom_condition_analysis,
    reported_symptom_has_cyanosis,
    reported_symptom_suggests_anticoagulant_head_trauma,
)


JOHN_DIABETES_CONDITIONS = [{"name": "Type 2 diabetes mellitus", "badge": "chronic"}]
JOHN_DIABETES_MEDS = [{"name": "Metformin", "dosage_instructions": "500 mg · twice daily"}]
JOHN_BETA_BLOCKER_MEDS = [{"name": "Bisoprolol", "dosage_instructions": "2.5 mg · once daily"}]
SUSAN_ANTICOAGULANT_MEDS = [{"name": "Warfarin", "dosage_instructions": "3 mg · once daily"}]


class SeverityFloorGapTests(unittest.TestCase):
    def test_01_cyanosis_keyword_detection(self):
        text = "John's lips look a bit blue and he's breathing fast"
        self.assertTrue(reported_symptom_has_cyanosis(text))

    def test_02_cyanosis_forces_emergency_from_monitor(self):
        text = "John's lips look a bit blue and he's breathing fast"
        result = cap_cyanosis_report_severity(text, "monitor")
        self.assertEqual(result, "emergency")

    def test_03_cyanosis_forces_emergency_from_ok(self):
        text = "turning blue around the mouth"
        result = cap_cyanosis_report_severity(text, "ok")
        self.assertEqual(result, "emergency")

    def test_04_beta_blocker_slow_pulse_contact_doctor(self):
        text = "his heart rate feels really slow, maybe 45"
        result = cap_beta_blocker_bradycardia_report_severity(text, "monitor", JOHN_BETA_BLOCKER_MEDS)
        self.assertEqual(result, "contact_doctor")

    def test_05_beta_blocker_slow_pulse_without_beta_blocker_no_floor(self):
        text = "his heart rate feels really slow, maybe 45"
        result = cap_beta_blocker_bradycardia_report_severity(text, "monitor", JOHN_DIABETES_MEDS)
        self.assertEqual(result, "monitor")

    def test_06_beta_blocker_slow_pulse_plus_faint_emergency(self):
        text = "slow pulse around 45 and he nearly fainted"
        result = cap_beta_blocker_bradycardia_report_severity(text, "monitor", JOHN_BETA_BLOCKER_MEDS)
        self.assertEqual(result, "emergency")

    def test_07_beta_blocker_med_rule_surfaces_bisoprolol(self):
        alerts = build_medication_symptom_alerts(
            "pulse below 50 this morning",
            JOHN_BETA_BLOCKER_MEDS,
        )
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["severity_impact"], "contact_doctor")
        self.assertIn("Bisoprolol", alerts[0]["education_message"])

    def test_08_hypoglycemia_shaky_confused_contact_doctor(self):
        text = "she's sweaty, shaky and a bit confused before lunch"
        result = cap_hypoglycemia_report_severity(
            text,
            "monitor",
            JOHN_DIABETES_CONDITIONS,
            JOHN_DIABETES_MEDS,
        )
        self.assertEqual(result, "contact_doctor")

    def test_09_hypoglycemia_unresponsive_emergency(self):
        text = "he's sweaty and shaky and now we can't wake him"
        result = cap_hypoglycemia_report_severity(
            text,
            "monitor",
            JOHN_DIABETES_CONDITIONS,
            JOHN_DIABETES_MEDS,
        )
        self.assertEqual(result, "emergency")

    def test_10_hypoglycemia_without_diabetes_context_no_floor(self):
        text = "she's sweaty, shaky and a bit confused before lunch"
        result = cap_hypoglycemia_report_severity(text, "monitor", [], [])
        self.assertEqual(result, "monitor")

    def test_11_existing_diabetes_confusion_fallback_fires(self):
        enriched = enrich_symptom_condition_analysis(
            "woke up confused",
            normalize_symptom_condition_analysis({}),
            JOHN_DIABETES_CONDITIONS,
            JOHN_DIABETES_MEDS,
            active_patient_name="John",
        )
        self.assertEqual(enriched["recommended_severity"], "contact_doctor")
        risks = enriched.get("relevant_condition_risks") or enriched.get("condition_risks") or []
        diabetes_hits = [
            item for item in risks
            if item.get("is_relevant")
            and "diabetes" in str(item.get("condition_name") or "").lower()
        ]
        self.assertTrue(diabetes_hits)

    def test_12_apply_report_severity_floor_caps_combined(self):
        text = "John's lips look a bit blue and he's breathing fast"
        result = apply_report_severity_floor_caps(
            text,
            "monitor",
            medications=JOHN_BETA_BLOCKER_MEDS,
            conditions=JOHN_DIABETES_CONDITIONS,
        )
        self.assertEqual(result, "emergency")

    def test_13_anticoagulant_fall_head_trauma_detected(self):
        text = "Susan had a fall this morning and bumped her head on the coffee table"
        self.assertTrue(reported_symptom_suggests_anticoagulant_head_trauma(text))

    def test_14_anticoagulant_fall_head_forces_emergency_from_contact_doctor(self):
        text = "Susan had a fall this morning and bumped her head on the coffee table"
        result = cap_anticoagulant_head_trauma_report_severity(
            text,
            "contact_doctor",
            SUSAN_ANTICOAGULANT_MEDS,
        )
        self.assertEqual(result, "emergency")

    def test_15_anticoagulant_fall_head_without_anticoagulant_no_floor(self):
        text = "Susan had a fall this morning and bumped her head on the coffee table"
        result = cap_anticoagulant_head_trauma_report_severity(text, "contact_doctor", JOHN_DIABETES_MEDS)
        self.assertEqual(result, "contact_doctor")

    def test_16_anticoagulant_med_rule_surfaces_warfarin(self):
        text = "Susan had a fall this morning and bumped her head on the coffee table"
        alerts = build_medication_symptom_alerts(text, SUSAN_ANTICOAGULANT_MEDS)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["severity_impact"], "emergency")
        self.assertIn("Warfarin", alerts[0]["education_message"])


def _run_individual_report():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SeverityFloorGapTests)
    results = []
    for test in suite:
        case = SeverityFloorGapTests(test._testMethodName)  # type: ignore[attr-defined]
        try:
            getattr(case, test._testMethodName)()
            results.append((test._testMethodName, "PASS"))
        except Exception as exc:
            results.append((test._testMethodName, f"FAIL — {exc}"))
    print("\nSeverity floor gap test report")
    print("=" * 60)
    for name, status in results:
        print(f"{status:6}  {name}")
    print("=" * 60)
    failed = sum(1 for _, status in results if status.startswith("FAIL"))
    print(f"Total: {len(results)}  Passed: {len(results) - failed}  Failed: {failed}")
    return failed


if __name__ == "__main__":
    failed = _run_individual_report()
    raise SystemExit(1 if failed else 0)
