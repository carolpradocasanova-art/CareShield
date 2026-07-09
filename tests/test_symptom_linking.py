"""Tests for subtype-aware symptom linking and session recurrence."""

import unittest

from symptom_linking import (
    count_linked_session_reports,
    detect_session_symptom_recurrence,
    extract_symptoms_from_text,
    find_symptom_related_prior_incidents,
    incident_symptom_relevance_score,
    resolve_incident_symptom_keys,
    select_linked_prior_incidents,
)


class SymptomSubtypeExtractionTests(unittest.TestCase):
    def test_peripheral_swelling_subtype(self):
        keys = extract_symptoms_from_text("mild ankle and leg swelling noticed")
        self.assertIn("swelling_peripheral", keys)
        self.assertNotIn("swelling_angioedema", keys)

    def test_angioedema_subtype(self):
        keys = extract_symptoms_from_text("facial and lip swelling after new medication")
        self.assertIn("swelling_angioedema", keys)
        self.assertNotIn("swelling_peripheral", keys)

    def test_allergic_rash_distinct_from_general_rash(self):
        hives = extract_symptoms_from_text("new hives on both arms")
        rash = extract_symptoms_from_text("rash on forearm")
        self.assertIn("rash_allergic", hives)
        self.assertNotIn("rash_skin", hives)
        self.assertIn("rash_skin", rash)
        self.assertNotIn("rash_allergic", rash)

    def test_pain_subtypes_differ_by_location(self):
        chest = extract_symptoms_from_text("chest pain when walking")
        knee = extract_symptoms_from_text("knee pain after exercise")
        self.assertIn("pain_chest", chest)
        self.assertNotIn("pain_joint", chest)
        self.assertIn("pain_joint", knee)
        self.assertNotIn("pain_chest", knee)

    def test_specific_pain_suppresses_general_pain_key(self):
        keys = extract_symptoms_from_text("sharp chest pain this morning")
        self.assertIn("pain_chest", keys)
        self.assertNotIn("pain_general", keys)


class SymptomSubtypeLinkingTests(unittest.TestCase):
    def test_angioedema_does_not_link_to_leg_swelling(self):
        prior = [
            {
                "text": "mild ankle and leg swelling noticed, likely heart failure related",
                "symptoms": ["swelling"],
                "timestamp": "1",
            },
        ]
        current = "facial and lip swelling after new medication, possible allergic reaction"
        related = find_symptom_related_prior_incidents(current, prior, limit=2)
        self.assertEqual(related, [])

    def test_leg_swelling_still_links_to_earlier_leg_swelling(self):
        prior = [
            {"text": "mild ankle swelling noticed", "symptoms": ["swelling"], "timestamp": "1"},
        ]
        current = "worsening leg swelling today"
        related = find_symptom_related_prior_incidents(current, prior, limit=2)
        self.assertEqual(len(related), 1)
        self.assertIn("ankle", related[0]["text"].lower())

    def test_chest_pain_does_not_link_to_knee_pain(self):
        prior = [{"text": "chest pain when walking upstairs", "symptoms": ["pain"], "timestamp": "1"}]
        current = "knee pain after gardening"
        score = incident_symptom_relevance_score(current, prior[0])
        self.assertEqual(score, 0)

    def test_hives_do_not_link_to_unrelated_forearm_rash(self):
        prior = [{"text": "rash on forearm", "symptoms": ["rash"], "timestamp": "1"}]
        current = "new hives on both arms after medication"
        score = incident_symptom_relevance_score(current, prior[0])
        self.assertEqual(score, 0)

    def test_legacy_stored_swelling_key_re_extracts_subtype(self):
        keys = resolve_incident_symptom_keys(
            "mild ankle swelling noticed",
            ["swelling"],
        )
        self.assertIn("swelling_peripheral", keys)
        self.assertNotIn("swelling", keys)

    def test_peter_name_overlap_does_not_link_unrelated_report(self):
        prior = {"text": "Peter had a good appetite today and went for a short walk in the garden"}
        current = "Peter has facial and lip swelling today"
        self.assertEqual(incident_symptom_relevance_score(current, prior), 0)
        self.assertEqual(find_symptom_related_prior_incidents(current, [prior]), [])


class SessionReportEvidenceTests(unittest.TestCase):
    def test_mixed_session_only_cites_relevant_swelling_report(self):
        prior = [
            {
                "text": "Peter had a good appetite today and went for a short walk in the garden",
                "symptoms": [],
                "timestamp": "1",
            },
            {
                "text": "mild lip swelling noticed after lunch",
                "symptoms": ["swelling_angioedema"],
                "timestamp": "2",
            },
        ]
        current = "Peter has facial and lip swelling today, possible allergic reaction"
        linked = select_linked_prior_incidents(
            current,
            prior,
            session_triggers=["recurrent facial or lip swelling in this session"],
        )
        self.assertEqual(len(linked), 1)
        self.assertIn("lip swelling", linked[0]["text"].lower())
        self.assertNotIn("appetite", linked[0]["text"].lower())

    def test_linked_count_matches_linked_list(self):
        prior = [
            {"text": "Peter had a good appetite today and went for a short walk in the garden", "timestamp": "1"},
            {"text": "mild lip swelling noticed after lunch", "timestamp": "2"},
            {"text": "ankle swelling from yesterday", "timestamp": "3"},
            {"text": "another routine update about breakfast", "timestamp": "4"},
        ]
        current = "facial and lip swelling worsening"
        triggers = ["recurrent facial or lip swelling in this session"]
        linked = select_linked_prior_incidents(current, prior, session_triggers=triggers)
        count = count_linked_session_reports(current, prior, session_triggers=triggers)
        self.assertEqual(count, 1 + len(linked))
        self.assertEqual(len(linked), 1)

    def test_no_linked_evidence_when_only_unrelated_priors_exist(self):
        prior = [
            {"text": "Peter had a good appetite today and went for a short walk in the garden", "timestamp": "1"},
        ]
        current = "facial and lip swelling worsening"
        triggers = ["recurrent facial or lip swelling in this session"]
        linked = select_linked_prior_incidents(current, prior, session_triggers=triggers)
        count = count_linked_session_reports(current, prior, session_triggers=triggers)
        self.assertEqual(linked, [])
        self.assertEqual(count, 1)


class SessionSymptomRecurrenceTests(unittest.TestCase):
    def test_peripheral_swelling_recurrence(self):
        prior = "mild ankle swelling noticed yesterday"
        current = "worsening leg swelling today"
        triggers = detect_session_symptom_recurrence(current, prior)
        self.assertEqual(triggers, ["recurrent peripheral swelling in this session"])

    def test_angioedema_does_not_trigger_peripheral_recurrence(self):
        prior = "mild ankle swelling noticed yesterday"
        current = "facial and lip swelling after medication"
        triggers = detect_session_symptom_recurrence(current, prior)
        self.assertEqual(triggers, [])

    def test_angioedema_recurrence_is_subtype_specific(self):
        prior = "lip swelling after lunch"
        current = "facial swelling worsening"
        triggers = detect_session_symptom_recurrence(current, prior)
        self.assertEqual(triggers, ["recurrent facial or lip swelling in this session"])


if __name__ == "__main__":
    unittest.main()
