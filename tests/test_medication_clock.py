import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from medication_clock import (
    CLOCK_DEGREES_PER_HOUR,
    clock_hour_label_angle_deg,
    dose_angle_deg,
    format_clock_slot_tooltip,
    group_doses_by_schedule_time,
    winning_clock_slot_status,
)

MISSED_FILL = "#FF453A"
TAKEN_FILL = "#34C759"


class MedicationClockTests(unittest.TestCase):
    def test_winning_status_precedence_missed_over_taken(self):
        self.assertEqual(
            winning_clock_slot_status(["taken", "missed"]),
            "missed",
        )

    def test_hydralazine_missed_warfarin_taken_at_08_00(self):
        """Hydralazine 08:00 Missed + Warfarin 08:00 Taken → slot shows Missed (red), not Taken."""
        doses = [
            {
                "medication_name": "Hydralazine",
                "hour": 8,
                "minute": 0,
                "display_time": "08:00",
                "time_label": "08:00",
            },
            {
                "medication_name": "Warfarin",
                "hour": 8,
                "minute": 0,
                "display_time": "08:00",
                "time_label": "08:00",
            },
        ]
        groups = group_doses_by_schedule_time(doses)
        self.assertEqual(len(groups), 1)
        (_hour, _minute), slot_doses = groups[0]
        self.assertEqual(len(slot_doses), 2)

        states = ["missed", "taken"]
        winning = winning_clock_slot_status(states)
        self.assertEqual(winning, "missed")

        palette = {
            "missed": {"fill": MISSED_FILL},
            "taken": {"fill": TAKEN_FILL},
        }
        wedge_fill = palette[winning]["fill"]
        self.assertEqual(wedge_fill, MISSED_FILL)
        self.assertNotEqual(wedge_fill, TAKEN_FILL)

        tooltip = format_clock_slot_tooltip([
            {
                "medication_name": "Hydralazine",
                "display_time": "08:00",
                "status_label": "Missed",
            },
            {
                "medication_name": "Warfarin",
                "display_time": "08:00",
                "status_label": "Taken",
            },
        ])
        self.assertEqual(
            tooltip,
            "Hydralazine · 08:00 · Missed, Warfarin · 08:00 · Taken",
        )

    def test_winning_status_precedence_order(self):
        cases = [
            (["taken", "not_yet"], "not_yet"),
            (["taken", "actionable"], "actionable"),
            (["not_yet", "actionable", "missed"], "missed"),
            (["taken"], "taken"),
        ]
        for states, expected in cases:
            with self.subTest(states=states):
                self.assertEqual(winning_clock_slot_status(states), expected)

    def test_group_doses_keeps_separate_time_slots(self):
        doses = [
            {"medication_name": "Warfarin", "hour": 8, "minute": 0},
            {"medication_name": "Hydroxyzine", "hour": 21, "minute": 0},
        ]
        groups = group_doses_by_schedule_time(doses)
        self.assertEqual(len(groups), 2)

    def test_dose_angle_uses_24_hour_dial(self):
        self.assertEqual(CLOCK_DEGREES_PER_HOUR, 15.0)
        self.assertEqual(dose_angle_deg(8, 0), 120.0)
        self.assertEqual(dose_angle_deg(14, 0), 210.0)
        self.assertEqual(dose_angle_deg(20, 0), 300.0)
        self.assertEqual(dose_angle_deg(21, 0), 315.0)
        self.assertEqual(dose_angle_deg(8, 30), 127.5)

    def test_dose_angle_aligns_with_hour_labels(self):
        """Wedge center must match the printed hour label position on the 24-hour face."""
        for hour in (8, 14, 20, 21):
            with self.subTest(hour=hour):
                self.assertEqual(
                    dose_angle_deg(hour, 0),
                    clock_hour_label_angle_deg(hour),
                )

    def test_distinct_schedule_times_have_distinct_angles(self):
        angles = [dose_angle_deg(h, 0) for h in (8, 14, 20, 21)]
        self.assertEqual(len(set(angles)), 4)


if __name__ == "__main__":
    unittest.main()
