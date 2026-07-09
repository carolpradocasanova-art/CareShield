"""Tests for grounding caregiver questions against stored patient data."""

import unittest
from unittest.mock import patch

from ai_helpers import (
    build_patient_claim_grounding_prompt_block,
    enforce_patient_record_grounding_in_reply,
    extract_unverified_patient_claims,
)


class PatientRecordGroundingTests(unittest.TestCase):
    def test_knee_surgery_question_flags_unverified_claim(self):
        with patch("ai_helpers.collect_patient_record_text_corpus", return_value="type 2 diabetes hypertension"):
            claims = extract_unverified_patient_claims(
                "What should I watch for after his knee surgery?",
                patient_id=1,
            )
        self.assertEqual(claims, ["knee surgery"])

    def test_documented_knee_surgery_is_not_flagged(self):
        corpus = "discharge summary: right knee replacement surgery completed last month"
        with patch("ai_helpers.collect_patient_record_text_corpus", return_value=corpus):
            claims = extract_unverified_patient_claims(
                "What should I watch for after his knee surgery?",
                patient_id=1,
            )
        self.assertEqual(claims, [])

    def test_grounding_prompt_requires_explicit_not_on_file(self):
        with patch("ai_helpers.collect_patient_record_text_corpus", return_value="hypertension"):
            block = build_patient_claim_grounding_prompt_block(
                "What should I watch for after his knee surgery?",
                patient_id=1,
            )
        self.assertIn("NO record", block)
        self.assertIn("knee surgery", block.lower())
        self.assertIn("Do NOT answer as if", block)

    def test_enforce_prepends_not_on_file_disclaimer(self):
        with patch("ai_helpers.collect_patient_record_text_corpus", return_value="hypertension"):
            with patch("ai_helpers.get_patient_display_name", return_value="Harold"):
                reply = enforce_patient_record_grounding_in_reply(
                    "Watch for wound redness and fever after surgery.",
                    "What should I watch for after his knee surgery?",
                    patient_id=1,
                )
        self.assertIn("any record", reply.lower())
        self.assertIn("Harold", reply)
        self.assertIn("Watch for wound redness", reply)

    def test_enforce_skips_when_reply_already_disclaims(self):
        with patch("ai_helpers.collect_patient_record_text_corpus", return_value="hypertension"):
            original = (
                "I don't have any record of knee surgery on file for this patient. "
                "Here is general guidance."
            )
            reply = enforce_patient_record_grounding_in_reply(
                original,
                "What should I watch for after his knee surgery?",
                patient_id=1,
            )
        self.assertEqual(reply, original)


if __name__ == "__main__":
    unittest.main()
