"""Tests for condition onset year normalization and display."""

import unittest

from ai_helpers import (
    _condition_fields_to_notes,
    _condition_notes_to_fields,
    normalize_condition_since,
)


class ConditionSinceTests(unittest.TestCase):
    def test_missing_onset_becomes_none(self):
        self.assertIsNone(normalize_condition_since(None))
        self.assertIsNone(normalize_condition_since(""))
        self.assertIsNone(normalize_condition_since("Unknown"))

    def test_valid_onset_is_preserved(self):
        self.assertEqual(normalize_condition_since("2019"), "2019")
        self.assertEqual(normalize_condition_since("March 2026"), "March 2026")

    def test_notes_round_trip_omits_missing_onset(self):
        notes = _condition_fields_to_notes({"name": "Migraine", "since": None, "badge": "chronic"})
        self.assertNotIn("since", notes)
        fields = _condition_notes_to_fields(notes)
        self.assertIsNone(fields["since"])

    def test_notes_round_trip_keeps_valid_onset(self):
        notes = _condition_fields_to_notes({"name": "Migraine", "since": "2019", "badge": "chronic"})
        fields = _condition_notes_to_fields(notes)
        self.assertEqual(fields["since"], "2019")


if __name__ == "__main__":
    unittest.main()
