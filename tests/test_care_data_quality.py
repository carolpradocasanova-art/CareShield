"""Tests for production vs internal QA care data separation."""

import unittest

from care_data_quality import (
    care_row_is_internal_test,
    filter_production_care_rows,
    is_designated_test_patient,
    is_internal_test_care_entry,
    should_block_test_entry_for_patient,
)


class CareDataQualityTests(unittest.TestCase):
    def test_detects_persistence_probe_text(self):
        self.assertTrue(is_internal_test_care_entry(summary="Persistence probe symptom"))
        self.assertTrue(is_internal_test_care_entry(summary="Shift log persistence probe"))

    def test_allows_real_symptom_text(self):
        self.assertFalse(is_internal_test_care_entry(summary="She was dizzy after lunch"))

    def test_blocks_test_entry_on_production_patient(self):
        self.assertTrue(should_block_test_entry_for_patient(
            "10",
            summary="Persistence probe symptom",
            caregiver_name="Probe",
            source="voice_report",
        ))

    def test_allows_test_entry_on_designated_test_patient(self):
        patient = {"id": "99", "display_name": "[TEST] QA Patient"}
        self.assertTrue(is_designated_test_patient("99", patient))
        self.assertFalse(should_block_test_entry_for_patient(
            "99",
            summary="Persistence probe symptom",
            caregiver_name="Probe",
            patient=patient,
        ))

    def test_filter_production_care_rows_removes_probe_rows(self):
        rows = [
            {"summary": "Persistence probe symptom", "caregiver_name": "Probe", "source": "voice_report"},
            {"summary": "Fall in the bathroom", "caregiver_name": "Carol", "source": "voice_report"},
        ]
        filtered = filter_production_care_rows(rows, "10")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["summary"], "Fall in the bathroom")

    def test_care_row_is_internal_test_matches_incident_shape(self):
        self.assertTrue(care_row_is_internal_test({
            "text": "probe chat",
            "summary": "probe chat",
            "source": "voice_report",
        }))


if __name__ == "__main__":
    unittest.main()
