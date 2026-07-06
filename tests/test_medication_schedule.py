import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datetime import datetime
from zoneinfo import ZoneInfo

from medication_schedule import (
    build_dose_events,
    canonical_plan_timing,
    compute_dose_ui_state,
    dose_minutes_until,
    format_medication_frequency,
    format_plan_schedule_summary,
    normalize_medication_schedule_fields,
    parse_schedule_times,
    resolve_schedule_frequency,
    schedule_time_slots_for_medication,
    strip_embedded_pill_count_from_timing,
)


class MedicationScheduleTests(unittest.TestCase):
    def test_parse_schedule_times_once_daily(self):
        self.assertEqual(parse_schedule_times("once daily"), [(8, 0)])

    def test_parse_schedule_times_twice_daily(self):
        self.assertEqual(parse_schedule_times("twice daily"), [(8, 0), (20, 0)])

    def test_parse_schedule_times_three_times_daily(self):
        self.assertEqual(parse_schedule_times("three times daily"), [(8, 0), (14, 0), (20, 0)])

    def test_parse_schedule_times_at_night(self):
        self.assertEqual(parse_schedule_times("at night"), [(21, 0)])

    def test_parse_schedule_times_at_bedtime(self):
        self.assertEqual(parse_schedule_times("at bedtime"), [(21, 0)])

    def test_parse_schedule_times_strips_embedded_pill_count(self):
        self.assertEqual(parse_schedule_times("at night · 1 pill(s) per dose"), [(21, 0)])

    def test_resolve_schedule_frequency_at_night_before_once_daily(self):
        self.assertEqual(resolve_schedule_frequency("once daily at bedtime"), "at_night")

    def test_format_medication_frequency_at_night(self):
        self.assertEqual(format_medication_frequency(timing="at night"), "at night")

    def test_strip_embedded_pill_count_from_timing(self):
        self.assertEqual(
            strip_embedded_pill_count_from_timing("once daily · 1 pill(s) per dose"),
            "once daily",
        )

    def test_format_plan_schedule_summary_no_duplicate_pill_count(self):
        summary = format_plan_schedule_summary({
            "time": "once daily · 1 pill(s) per dose",
            "pills_per_dose": 1,
        })
        self.assertEqual(summary, "once daily · 1 pill(s) per dose")
        self.assertEqual(summary.count("per dose"), 1)

    def test_format_plan_schedule_summary_at_night(self):
        summary = format_plan_schedule_summary({
            "timing": "at night",
            "pills_per_dose": 1,
        })
        self.assertEqual(summary, "at night · 1 pill(s) per dose")

    def test_format_medication_frequency_ignores_duplicate_time_and_timing(self):
        """Stale time + updated timing after re-upload must not double clock times."""
        plan = {
            "time": "08:00 / 14:00 / 20:00",
            "timing": "three times daily at 08:00, 14:00 and 20:00",
        }
        self.assertEqual(
            format_medication_frequency(plan),
            "three times daily",
        )

    def test_format_medication_frequency_warfarin_once_daily(self):
        plan = {
            "time": "11:45 AM",
            "timing": "once daily at 11:45 AM",
            "dosage": "0.5 mg",
        }
        result = format_medication_frequency(plan)
        self.assertNotIn("2 times", result)
        self.assertNotIn("and 11:45 AM", result)
        self.assertEqual(result, "once daily")

    def test_normalize_medication_schedule_fields_prefers_timing(self):
        merged = normalize_medication_schedule_fields({
            "name": "Warfarin",
            "time": "08:00",
            "timing": "11:45 AM",
            "dosage": "0.5 mg",
        })
        self.assertEqual(merged["timing"], "11:45 AM")
        self.assertEqual(merged["time"], "11:45 AM")
        self.assertNotIn("schedule", merged)

    def test_document_reupload_merge_does_not_double_frequency(self):
        """Simulate merge after re-upload: old time kept, new timing overwrites."""
        existing = {
            "name": "Hydralazine",
            "dosage": "25 mg",
            "time": "08:00 / 14:00 / 20:00",
            "timing": "08:00 / 14:00 / 20:00",
        }
        incoming = {
            "name": "Hydralazine",
            "dosage": "25 mg",
            "timing": "three times daily at 08:00, 14:00 and 20:00",
            "pills_per_dose": 1,
        }
        merged = normalize_medication_schedule_fields({**existing, **incoming})
        frequency = format_medication_frequency(merged)
        self.assertEqual(frequency, "three times daily")
        self.assertNotIn("6 times", frequency)

    def test_parse_schedule_times_ignores_minutes_as_bare_am_pm(self):
        """Regression: '45 am' inside '11:45 AM' must not become a phantom 21:00 slot."""
        self.assertEqual(parse_schedule_times("once daily at 11:45 AM"), [(11, 45)])

    def test_build_dose_events_uses_canonical_timing_not_stale_time(self):
        """Stale time field must not create phantom slots after schedule update."""
        warfarin = {
            "name": "Warfarin",
            "dosage": "0.5 mg",
            "time": "21:00",
            "timing": "once daily at 11:45 AM",
        }
        slots = schedule_time_slots_for_medication(warfarin)
        self.assertEqual(slots, [(11, 45)])

        events = build_dose_events([warfarin])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["hour"], 11)
        self.assertEqual(events[0]["minute"], 45)
        self.assertEqual(events[0]["medication_name"], "Warfarin")

    def test_build_dose_events_full_test_panel_five_cards(self):
        plan = [
            {
                "name": "Hydralazine",
                "dosage": "25 mg",
                "timing": "three times daily at 08:00, 14:00 and 20:00",
            },
            {
                "name": "Warfarin",
                "dosage": "0.5 mg",
                "timing": "once daily at 11:45 AM",
            },
            {
                "name": "Hydroxyzine",
                "dosage": "25 mg",
                "timing": "at night",
            },
        ]
        events = build_dose_events(plan)
        self.assertEqual(len(events), 5)

        by_med: dict[str, list[tuple[int, int]]] = {}
        for event in events:
            by_med.setdefault(event["medication_name"], []).append(
                (event["hour"], event["minute"])
            )

        self.assertEqual(by_med["Hydralazine"], [(8, 0), (14, 0), (20, 0)])
        self.assertEqual(by_med["Warfarin"], [(11, 45)])
        self.assertEqual(by_med["Hydroxyzine"], [(21, 0)])

    def test_schedule_change_replaces_old_slots_for_any_medication(self):
        hydroxyzine_before = {
            "name": "Hydroxyzine",
            "dosage": "25 mg",
            "time": "21:00",
            "timing": "at night",
        }
        self.assertEqual(schedule_time_slots_for_medication(hydroxyzine_before), [(21, 0)])

        hydroxyzine_after = {
            "name": "Hydroxyzine",
            "dosage": "25 mg",
            "timing": "once daily at 10:00 PM",
            "time": "21:00",
        }
        slots = schedule_time_slots_for_medication(hydroxyzine_after)
        self.assertEqual(slots, [(22, 0)])
        self.assertNotIn((21, 0), slots)

        events = build_dose_events([hydroxyzine_after])
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0]["hour"], events[0]["minute"]), (22, 0))

    def test_phantom_slot_not_cross_merged_from_other_medications(self):
        """Each medication's slots come only from its own plan row, not neighbors."""
        plan = [
            {"name": "Warfarin", "time": "21:00", "timing": "once daily at 11:45 AM"},
            {"name": "Hydroxyzine", "timing": "at night"},
        ]
        events = build_dose_events(plan)
        warfarin_slots = [
            (e["hour"], e["minute"]) for e in events if e["medication_name"] == "Warfarin"
        ]
        self.assertEqual(warfarin_slots, [(11, 45)])

    def test_dose_ui_state_actionable_six_minutes_after_scheduled(self):
        tz = ZoneInfo("Europe/Madrid")
        now = datetime(2026, 7, 4, 11, 51, tzinfo=tz)
        dose = {"hour": 11, "minute": 45, "medication_name": "Warfarin", "time_label": "11:45"}
        minutes_until = dose_minutes_until(dose, now, tz)
        self.assertAlmostEqual(minutes_until, -6.0)
        self.assertEqual(compute_dose_ui_state(dose, now, tz_obj=tz), "actionable")

    def test_dose_ui_state_not_yet_when_now_uses_wrong_timezone(self):
        """UTC now against local wall-clock schedule produces false not_yet — regression guard."""
        dose = {"hour": 11, "minute": 45, "medication_name": "Warfarin", "time_label": "11:45"}
        now_utc = datetime(2026, 7, 4, 9, 51, tzinfo=ZoneInfo("UTC"))
        now_local = datetime(2026, 7, 4, 11, 51, tzinfo=ZoneInfo("Europe/Madrid"))
        self.assertEqual(compute_dose_ui_state(dose, now_utc, tz_obj=ZoneInfo("UTC")), "not_yet")
        self.assertEqual(
            compute_dose_ui_state(dose, now_local, tz_obj=ZoneInfo("Europe/Madrid")),
            "actionable",
        )

    def test_schedule_change_dose_flips_to_due_now_minutes_after(self):
        """Any medication whose time was just edited should use the new slot for status."""
        tz = ZoneInfo("Europe/Madrid")
        now = datetime(2026, 7, 4, 11, 56, tzinfo=tz)
        hydroxyzine = {
            "name": "Hydroxyzine",
            "timing": "once daily at 11:50 AM",
        }
        events = build_dose_events([hydroxyzine])
        self.assertEqual(len(events), 1)
        dose = events[0]
        self.assertEqual((dose["hour"], dose["minute"]), (11, 50))
        self.assertEqual(compute_dose_ui_state(dose, now, tz_obj=tz), "actionable")


if __name__ == "__main__":
    unittest.main()
