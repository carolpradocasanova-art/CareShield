"""Tests for symptom shift-log source classification."""

import unittest

from ai_helpers import (
    ADHERENCE_SHIFT_LOG_SOURCES,
    INTERNAL_SHIFT_LOG_SOURCES,
    shift_log_is_adherence_event,
    shift_log_is_symptom_event,
)


class SymptomShiftLogClassificationTests(unittest.TestCase):
    def test_voice_report_is_symptom_not_adherence(self):
        row = {"source": "voice_report", "summary": "Patient reported dizziness"}
        self.assertTrue(shift_log_is_symptom_event(row))
        self.assertFalse(shift_log_is_adherence_event(row))

    def test_medication_check_is_adherence_not_symptom(self):
        row = {"source": "medication_check", "summary": "MedCam check"}
        self.assertFalse(shift_log_is_symptom_event(row))
        self.assertTrue(shift_log_is_adherence_event(row))

    def test_excluded_sources_are_disjoint_from_symptom_events(self):
        for source in INTERNAL_SHIFT_LOG_SOURCES | ADHERENCE_SHIFT_LOG_SOURCES:
            row = {"source": source, "summary": "example"}
            self.assertFalse(shift_log_is_symptom_event(row))


if __name__ == "__main__":
    unittest.main()
