import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ai_helpers import (
    MY_RESULTS_EXPLAIN_PROMPT,
    MY_RESULTS_EXTRACT_PROMPT,
    MY_RESULTS_DATE_NOT_SPECIFIED,
    build_my_results_explain_payload,
    count_my_results_review_items,
    enrich_my_results_record,
    my_results_abnormal_test_names,
    my_results_explain_is_complete,
    my_results_has_key_findings,
    my_results_limitation_is_lab_table_artifact,
    my_results_no_abnormal_note_label,
    my_results_has_abnormal_values,
    my_results_has_actionable_content,
    my_results_use_grouped_explanations,
    normalize_my_results_extract,
    normalize_my_results_explain,
    resolve_my_results_test_name,
    sanitize_my_results_plain_text,
)


CARDIOLOGY_LETTER_EXTRACT = {
    "documentType": "Cardiology consultation letter",
    "documentCategory": "clinic_letter",
    "date": "12 March 2026",
    "source": "Riverside Cardiology Clinic",
    "readability": "clear",
    "newDiagnoses": [
        {
            "name": "Atrial fibrillation",
            "detail": "New onset, not previously documented",
            "isNew": True,
        }
    ],
    "medicationChanges": [
        {
            "medication": "Furosemide",
            "changeType": "start",
            "detail": "20 mg daily for fluid management",
        },
        {
            "medication": "Apixaban",
            "changeType": "start",
            "detail": "5 mg twice daily for stroke prevention with AF",
        },
    ],
    "caregiverInstructions": [
        {
            "instruction": "Seek urgent care for chest pain, severe breathlessness, or fainting.",
            "category": "red_flag",
        },
    ],
    "followUps": [
        {
            "description": "Follow-up clinic visit",
            "dateKind": "relative",
            "relativePhrase": "in 3 weeks",
            "date": "",
            "prep": "",
        },
        {
            "description": "Echocardiogram",
            "dateKind": "explicit",
            "date": "26 March 2026",
            "prep": "No special prep required",
        },
    ],
    "results": [],
    "backgroundConditions": [
        {"name": "Hypertension", "detail": "Long-standing, unchanged"},
    ],
}

CARDIOLOGY_LETTER_WITH_ECG_RESULT = {
    **CARDIOLOGY_LETTER_EXTRACT,
    "results": [
        {
            "name": "ECG",
            "value": "atrial fibrillation with ventricular rate approximately 90 bpm",
            "unit": "",
            "status": "high",
            "referenceRange": "",
        }
    ],
}

LAB_PANEL_EXTRACT = {
    "documentType": "Comprehensive metabolic panel",
    "documentCategory": "lab_panel",
    "date": "1 Feb 2026",
    "source": "City Lab",
    "readability": "clear",
    "results": [
        {"name": "Sodium", "value": "138", "unit": "mmol/L", "status": "normal", "referenceRange": "136–145"},
        {"name": "Potassium", "value": "5.8", "unit": "mmol/L", "status": "high", "referenceRange": "3.5–5.0"},
        {"name": "Creatinine", "value": "1.0", "unit": "mg/dL", "status": "normal", "referenceRange": "0.7–1.2"},
    ],
}

LAB_PANEL_WITH_HEART_RATE = {
    **LAB_PANEL_EXTRACT,
    "results": LAB_PANEL_EXTRACT["results"]
    + [
        {"name": "Heart rate", "value": "118", "unit": "bpm", "status": "high", "referenceRange": "60-100"},
        {"name": "Heart rate", "value": "90", "unit": "bpm", "status": "high", "referenceRange": "60-100"},
    ],
}

NORMAL_VISIT_EXTRACT = {
    "documentType": "Annual wellness visit summary",
    "documentCategory": "clinic_letter",
    "readability": "clear",
    "results": [],
    "newDiagnoses": [],
    "medicationChanges": [],
}

UNREADABLE_EXTRACT = {
    "documentType": "Unknown",
    "readability": "unreadable",
    "results": [],
}


class MyResultsLogicTests(unittest.TestCase):
    def test_cardiology_letter_has_actionable_content_without_labs(self):
        normalized = normalize_my_results_extract(CARDIOLOGY_LETTER_EXTRACT)
        self.assertTrue(my_results_has_actionable_content(normalized))
        self.assertFalse(normalized["hasLabValues"])

    def test_relative_follow_up_resolves_from_document_date(self):
        normalized = normalize_my_results_extract(CARDIOLOGY_LETTER_EXTRACT)
        clinic_follow = normalized["followUps"][0]
        self.assertEqual(clinic_follow["dateKind"], "relative")
        self.assertIn("3 weeks from visit", clinic_follow["dateDisplay"])
        self.assertIn("2 Apr 2026", clinic_follow["dateDisplay"])
        self.assertNotIn("Unknown", clinic_follow["dateDisplay"])

    def test_explicit_follow_up_keeps_calendar_date(self):
        normalized = normalize_my_results_extract(CARDIOLOGY_LETTER_EXTRACT)
        echo_follow = normalized["followUps"][1]
        self.assertEqual(echo_follow["dateDisplay"], "26 Mar 2026")

    def test_unspecified_follow_up_uses_plain_label(self):
        normalized = normalize_my_results_extract(
            {
                **CARDIOLOGY_LETTER_EXTRACT,
                "followUps": [{"description": "Routine review", "dateKind": "unspecified"}],
            }
        )
        self.assertEqual(normalized["followUps"][0]["dateDisplay"], MY_RESULTS_DATE_NOT_SPECIFIED)

    def test_sanitize_strips_html_from_text_fields(self):
        cleaned = sanitize_my_results_plain_text(
            '<div class="cs-mr-empty-section">No numeric lab values were listed in this document.</div>'
        )
        self.assertEqual(cleaned, "No numeric lab values were listed in this document.")
        explain = normalize_my_results_explain(
            {
                "explanation": "<strong>Hello</strong> world",
                "questions": ["<em>Why?</em>"],
                "urgentCareInstructions": None,
            },
            extract={"results": []},
        )
        self.assertEqual(explain["explanation"], "Hello world")
        self.assertEqual(explain["questions"][0]["text"], "Why?")

    def test_lab_panel_still_actionable(self):
        normalized = normalize_my_results_extract(LAB_PANEL_EXTRACT)
        self.assertTrue(normalized["hasLabValues"])
        self.assertTrue(my_results_has_actionable_content(normalized))

    def test_ecg_rhythm_finding_not_in_numeric_table(self):
        normalized = normalize_my_results_extract(CARDIOLOGY_LETTER_WITH_ECG_RESULT)
        self.assertFalse(normalized["hasLabValues"])
        self.assertEqual(normalized["results"], [])
        self.assertFalse(
            any(row.get("status") in ("high", "low") for row in normalized["results"])
        )
        self.assertTrue(
            any(
                dx.get("name") == "Atrial fibrillation"
                for dx in normalized.get("newDiagnoses") or []
            )
        )

    def test_potassium_out_of_range_stays_high(self):
        normalized = normalize_my_results_extract(LAB_PANEL_EXTRACT)
        potassium = next(row for row in normalized["results"] if row["name"] == "Potassium")
        self.assertEqual(potassium["status"], "high")
        self.assertEqual(potassium["value"], "5.8")

    def test_heart_rate_flags_use_reference_range(self):
        normalized = normalize_my_results_extract(LAB_PANEL_WITH_HEART_RATE)
        rates = {
            row["value"]: row["status"]
            for row in normalized["results"]
            if row["name"] == "Heart rate"
        }
        self.assertEqual(rates["118"], "high")
        self.assertEqual(rates["90"], "normal")

    def test_enrich_legacy_record_adds_flags(self):
        legacy = {
            "documentType": "Clinic letter",
            "date": "12 March 2026",
            "followUps": [{"description": "Clinic visit", "date": "in 3 weeks"}],
            "results": [],
            "explanation": "Summary text",
            "questions": ["Question one"],
        }
        enriched = enrich_my_results_record(legacy)
        self.assertFalse(enriched["hasLabValues"])
        self.assertIn("approx.", enriched["followUps"][0]["dateDisplay"])

    def test_prompts_require_plain_text_and_date_kinds(self):
        self.assertIn("dateKind", MY_RESULTS_EXTRACT_PROMPT)
        self.assertIn("PLAIN TEXT ONLY", MY_RESULTS_EXPLAIN_PROMPT)
        self.assertIn("allTestNames", MY_RESULTS_EXPLAIN_PROMPT)
        self.assertIn("resultGroups", MY_RESULTS_EXPLAIN_PROMPT)

    def test_explain_payload_includes_exact_test_name_lists(self):
        extract = normalize_my_results_extract(LAB_PANEL_EXTRACT)
        payload = json.loads(
            build_my_results_explain_payload(extract, patient_name="Eleanor", known_conditions=["Diabetes"])
        )
        self.assertIn("Potassium", payload["allTestNames"])
        self.assertIn("Potassium", payload["abnormalTestNames"])
        self.assertEqual(payload["patientName"], "Eleanor")

    def test_resolve_test_name_fuzzy_and_exact(self):
        names = ["Creatinine", "eGFR", "Potassium"]
        self.assertEqual(resolve_my_results_test_name("creatinine", names), "Creatinine")
        self.assertEqual(resolve_my_results_test_name("Potasium", names), "Potassium")
        self.assertIsNone(resolve_my_results_test_name("WBC", names))

    def test_normalize_result_groups_drop_unknown_test_names_and_sort_urgency(self):
        extract = normalize_my_results_extract(LAB_PANEL_EXTRACT)
        explain = normalize_my_results_explain(
            {
                "explanation": "Lab panel overview.",
                "trendCallouts": [],
                "resultGroups": [
                    {
                        "category": "Electrolytes",
                        "urgency": "discuss_at_visit",
                        "groupSummary": "Potassium is high.",
                        "testNames": ["Potassium", "Made Up Test"],
                        "valueExplanations": [
                            {
                                "testName": "Potassium",
                                "whatItMeasures": "Potassium helps nerves and muscles work.",
                                "whatThisResultSuggests": "A high level may need review.",
                            },
                            {
                                "testName": "Fake Test",
                                "whatItMeasures": "Nope",
                                "whatThisResultSuggests": "Nope",
                            },
                        ],
                    },
                    {
                        "category": "Kidney function",
                        "urgency": "discuss_soon",
                        "groupSummary": "Kidney markers changed.",
                        "testNames": ["Creatinine"],
                        "valueExplanations": [],
                    },
                ],
                "questions": [
                    {
                        "text": "Should we recheck potassium soon?",
                        "relatedCategory": "Electrolytes",
                        "relatedTests": ["Potassium", "Fake Test"],
                    }
                ],
            },
            extract=extract,
        )
        self.assertEqual(len(explain["resultGroups"]), 2)
        self.assertEqual(explain["resultGroups"][0]["urgency"], "discuss_soon")
        self.assertEqual(explain["resultGroups"][0]["category"], "Kidney function")
        potassium_group = explain["resultGroups"][1]
        self.assertEqual(potassium_group["testNames"], ["Potassium"])
        self.assertEqual(len(potassium_group["valueExplanations"]), 1)
        self.assertEqual(explain["questions"][0]["relatedTests"], ["Potassium"])
        self.assertTrue(explain["useGroupedExplanations"])

    def test_no_abnormal_values_note_clears_groups(self):
        extract = normalize_my_results_extract(LAB_PANEL_EXTRACT)
        for row in extract["results"]:
            row["status"] = "normal"
        explain = normalize_my_results_explain(
            {
                "explanation": "Most values look routine.",
                "noAbnormalValuesNote": "No abnormal values were flagged in this panel.",
                "resultGroups": [{"category": "Should drop", "testNames": ["Potassium"], "urgency": "discuss_soon"}],
                "trendCallouts": [{"title": "Should drop", "summary": "Nope"}],
                "questions": [{"text": "Any follow-up needed?", "relatedCategory": "", "relatedTests": []}],
            },
            extract=extract,
        )
        self.assertEqual(explain["resultGroups"], [])
        self.assertEqual(explain["trendCallouts"], [])
        self.assertFalse(explain["useGroupedExplanations"])
        self.assertFalse(my_results_has_abnormal_values(extract))
        self.assertTrue(my_results_explain_is_complete(explain, extract))

    def test_use_grouped_explanations_false_when_groups_empty(self):
        extract = normalize_my_results_extract(LAB_PANEL_EXTRACT)
        for row in extract["results"]:
            row["status"] = "normal"
        explain = normalize_my_results_explain(
            {
                "explanation": "Several results need discussion.",
                "resultGroups": [],
                "questions": [{"text": "What next?", "relatedCategory": "", "relatedTests": []}],
            },
            extract=extract,
        )
        self.assertFalse(my_results_use_grouped_explanations(explain, extract))

    def test_resolve_test_name_partial_substring(self):
        names = ["BUN (Blood Urea Nitrogen)", "Creatinine"]
        self.assertEqual(resolve_my_results_test_name("BUN", names), "BUN (Blood Urea Nitrogen)")

    def test_coverage_fallback_adds_missing_abnormal_values(self):
        extract = normalize_my_results_extract(LAB_PANEL_EXTRACT)
        explain = normalize_my_results_explain(
            {
                "explanation": "Some results need review.",
                "resultGroups": [
                    {
                        "category": "Kidney function",
                        "urgency": "discuss_soon",
                        "groupSummary": "Kidney markers only.",
                        "testNames": ["Creatinine"],
                        "valueExplanations": [
                            {
                                "testName": "Creatinine",
                                "whatItMeasures": "Creatinine reflects kidney filtering.",
                                "whatThisResultSuggests": "Normal here.",
                            }
                        ],
                    }
                ],
                "questions": [{"text": "Any kidney follow-up?", "relatedCategory": "Kidney function", "relatedTests": ["Creatinine"]}],
            },
            extract=extract,
            generate_missing_explanations=False,
        )
        covered = set()
        for group in explain["resultGroups"]:
            covered.update(group.get("testNames") or [])
            for item in group.get("valueExplanations") or []:
                covered.add(item["testName"])
        self.assertIn("Potassium", covered)
        fallback = explain["resultGroups"][-1]
        self.assertEqual(fallback["category"], "Other flagged values")
        potassium_expl = next(
            item for item in fallback["valueExplanations"] if item["testName"] == "Potassium"
        )
        self.assertIn("5.8", potassium_expl["whatThisResultSuggests"])

    def test_coverage_check_noop_for_clinic_letter_without_labs(self):
        extract = normalize_my_results_extract(CARDIOLOGY_LETTER_EXTRACT)
        explain = normalize_my_results_explain(
            {
                "explanation": "New AFib diagnosis with medication changes.",
                "resultGroups": [],
                "questions": [{"text": "When is the echo?", "relatedCategory": "", "relatedTests": []}],
            },
            extract=extract,
        )
        self.assertEqual(explain["resultGroups"], [])
        self.assertEqual(my_results_abnormal_test_names(extract), [])

    def test_trend_callouts_backfilled_when_prior_comparisons_present(self):
        extract = normalize_my_results_extract({
            **LAB_PANEL_EXTRACT,
            "labComment": (
                "Elevated creatinine and reduced eGFR compared to prior panel "
                "(03/2026: Cr 1.1, eGFR 58)."
            ),
            "priorComparisons": [{
                "category": "Kidney function changed since last test",
                "summary": "Creatinine is higher and eGFR is lower than the prior panel.",
                "tests": ["Creatinine", "eGFR"],
                "priorValues": "Cr 1.1, eGFR 58",
            }],
        })
        explain = normalize_my_results_explain(
            {
                "explanation": "Several results need discussion.",
                "trendCallouts": [],
                "resultGroups": [
                    {
                        "category": "Electrolytes",
                        "urgency": "discuss_at_visit",
                        "groupSummary": "Potassium is high.",
                        "testNames": ["Potassium"],
                        "valueExplanations": [{
                            "testName": "Potassium",
                            "whatItMeasures": "Potassium helps nerves and muscles work.",
                            "whatThisResultSuggests": "A high level may need review.",
                        }],
                    }
                ],
                "questions": [{"text": "Recheck potassium?", "relatedCategory": "Electrolytes", "relatedTests": ["Potassium"]}],
            },
            extract=extract,
            generate_missing_explanations=False,
        )
        self.assertEqual(len(explain["trendCallouts"]), 1)
        self.assertIn("Kidney function", explain["trendCallouts"][0]["title"])
        self.assertIn("Cr 1.1", explain["trendCallouts"][0]["priorValues"])

    def test_requires_trend_callouts_from_lab_comment_only(self):
        extract = normalize_my_results_extract({
            **LAB_PANEL_EXTRACT,
            "labComment": "Compared to prior panel: Cr 1.1, eGFR 58.",
        })
        from ai_helpers import my_results_requires_trend_callouts
        self.assertTrue(my_results_requires_trend_callouts(extract))
        self.assertTrue(extract["priorComparisons"])

    def test_has_key_findings_for_clinic_letter(self):
        extract = normalize_my_results_extract(CARDIOLOGY_LETTER_EXTRACT)
        self.assertTrue(my_results_has_key_findings(extract))

    def test_has_key_findings_false_for_normal_visit(self):
        extract = normalize_my_results_extract(NORMAL_VISIT_EXTRACT)
        self.assertFalse(my_results_has_key_findings(extract))

    def test_limitation_lab_artifact_suppressed_without_results(self):
        extract = normalize_my_results_extract(CARDIOLOGY_LETTER_EXTRACT)
        self.assertTrue(
            my_results_limitation_is_lab_table_artifact("reference range not provided", extract)
        )
        self.assertFalse(
            my_results_limitation_is_lab_table_artifact("Conflicting follow-up dates noted", extract)
        )

    def test_no_abnormal_note_label_neutral_when_key_findings_present(self):
        self.assertEqual(
            my_results_no_abnormal_note_label({
                **CARDIOLOGY_LETTER_EXTRACT,
                "results": [],
                "hasLabValues": False,
            }),
            "No abnormal lab values",
        )

    def test_no_abnormal_note_label_good_news_when_nothing_significant(self):
        self.assertEqual(
            my_results_no_abnormal_note_label({
                **NORMAL_VISIT_EXTRACT,
                "results": [],
                "hasLabValues": False,
            }),
            "Good news",
        )


if __name__ == "__main__":
    unittest.main()
