"""Tests for caregiver-friendly MedCam adherence timeline text."""

import unittest

from ai_helpers import format_medcam_shift_log_for_timeline


class MedcamTimelineFormatTests(unittest.TestCase):
    def test_strips_patient_marker_and_technical_codes(self):
        raw = (
            "[[patient:12]] Paracetamol: x2 (high confidence, schedule=prn_wait). "
            "Verdict: Review before giving. Checked at 3:08 PM."
        )
        text = format_medcam_shift_log_for_timeline(raw)
        self.assertNotIn("[[patient:", text)
        self.assertNotIn("schedule=", text)
        self.assertNotIn("confidence", text.lower())
        self.assertIn("MedCam check", text)
        self.assertIn("Review before giving", text)
        self.assertIn("Paracetamol (2 pills)", text)

    def test_leaves_already_friendly_text_unchanged(self):
        friendly = "Paracetamol dose logged as taken"
        self.assertEqual(format_medcam_shift_log_for_timeline(friendly), friendly)


if __name__ == "__main__":
    unittest.main()
