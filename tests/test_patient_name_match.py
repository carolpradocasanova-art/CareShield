"""Tests for document patient name matching."""

import unittest

from ai_helpers import patient_names_match


class PatientNameMatchTests(unittest.TestCase):
    def test_single_first_name_matches_full_document_name(self):
        self.assertTrue(patient_names_match("Peter", "Peter Whitfield"))

    def test_full_active_name_matches_document_first_name_only(self):
        self.assertTrue(patient_names_match("Peter Whitfield", "Peter"))

    def test_different_people_do_not_match(self):
        self.assertFalse(patient_names_match("Peter", "Paul Whitfield"))

    def test_matching_full_names(self):
        self.assertTrue(patient_names_match("Peter Whitfield", "Peter Whitfield"))

    def test_nickname_prefix_still_matches(self):
        self.assertTrue(patient_names_match("Barth", "Bartholomew Nkemelu"))


if __name__ == "__main__":
    unittest.main()
