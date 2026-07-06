import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from medcam_dose_cards import (
    PRN_DOSE_STATUS_LABELS,
    SCHEDULED_DOSE_STATUS_LABELS,
    build_medcam_prn_dose_card_html,
    build_medcam_scheduled_dose_card_html,
    count_status_label_occurrences,
)

SAMPLE_MEDS = {
    "warfarin": {
        "name": "Warfarin",
        "dosage": "5mg",
        "timing": "once daily",
        "pills_per_dose": 1,
    },
    "hydralazine": {
        "name": "Hydralazine",
        "dosage": "25mg",
        "timing": "three times daily",
        "pills_per_dose": 1,
    },
    "hydroxyzine": {
        "name": "Hydroxyzine",
        "dosage": "25mg",
        "timing": "at night",
        "pills_per_dose": 1,
    },
}


def _dose_event(med_key: str, display_time: str = "08:00") -> dict:
    med = SAMPLE_MEDS[med_key]
    return {
        "medication_name": med["name"],
        "display_time": display_time,
        "time_label": display_time.replace(":", ""),
    }


class MedcamDoseCardTests(unittest.TestCase):
    def test_scheduled_status_renders_once_per_state(self):
        for state, label in SCHEDULED_DOSE_STATUS_LABELS.items():
            for med_key in SAMPLE_MEDS:
                with self.subTest(state=state, medication=med_key):
                    card = build_medcam_scheduled_dose_card_html(
                        _dose_event(med_key),
                        state,
                        SAMPLE_MEDS[med_key],
                    )
                    self.assertEqual(count_status_label_occurrences(card, label), 1)
                    self.assertEqual(card.count(label), 1)

    def test_missed_status_not_duplicated_for_hydralazine_and_warfarin(self):
        for med_key in ("hydralazine", "warfarin"):
            with self.subTest(medication=med_key):
                card = build_medcam_scheduled_dose_card_html(
                    _dose_event(med_key),
                    "missed",
                    SAMPLE_MEDS[med_key],
                )
                self.assertEqual(count_status_label_occurrences(card, "Missed"), 1)
                self.assertEqual(card.count("Missed"), 1)

    def test_prn_card_renders_status_once(self):
        med = SAMPLE_MEDS["hydroxyzine"]
        for state, label in PRN_DOSE_STATUS_LABELS.items():
            with self.subTest(state=state):
                status = {
                    "status": state,
                    "doses_today": 1,
                    "max_per_day": 4,
                }
                if state == "prn_wait":
                    status["wait_label"] = "Wait 2h 15m"
                    label = "Wait 2h 15m"
                card = build_medcam_prn_dose_card_html(med, status)
                self.assertEqual(count_status_label_occurrences(card, label), 1)
                self.assertEqual(card.count(label), 1)
                self.assertIn("PRN", card)


if __name__ == "__main__":
    unittest.main()
