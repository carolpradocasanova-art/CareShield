"""Medication schedule clock helpers (grouping and status precedence for overlapping wedges)."""

from __future__ import annotations

from itertools import groupby

CLOCK_SLOT_STATUS_PRECEDENCE = ("missed", "actionable", "not_yet", "taken")

# Printed 24-hour face: 24 at top (midnight), then 1–23 clockwise.
CLOCK_FACE_HOUR_LABELS = tuple(range(1, 25))

# Dial uses 360° / 24h = 15° per hour; 0° at top.
CLOCK_DEGREES_PER_HOUR = 360 / 24


def dose_angle_deg(hour: int, minute: int) -> float:
    """Map a 24-hour schedule time to dial degrees (0° = midnight at top)."""
    return ((hour % 24) + minute / 60) * CLOCK_DEGREES_PER_HOUR


def clock_hour_label_angle_deg(hour_label: int) -> float:
    """Map a printed face label (1–24) to dial degrees; 24 sits at the top."""
    return (hour_label % 24) * CLOCK_DEGREES_PER_HOUR


def clock_label_font_size(hour_label: int) -> float:
    """Slightly smaller type for two-digit hours so 1–24 fits on one ring."""
    return 4.0 if hour_label >= 10 else 4.4


def clock_label_radius_offset(hour_label: int) -> float:
    """Nudge inner labels on crowded quadrants for legibility."""
    if hour_label in {11, 12, 13, 23, 24, 1, 2}:
        return 8.0
    return 9.0


def winning_clock_slot_status(states: list[str]) -> str:
    """Return highest-priority status for a time slot (missed > actionable > not_yet > taken)."""
    state_set = set(states)
    for status in CLOCK_SLOT_STATUS_PRECEDENCE:
        if status in state_set:
            return status
    return "not_yet"


def group_doses_by_schedule_time(doses: list[dict]) -> list[tuple[tuple[int, int], list[dict]]]:
    """Group dose events by (hour, minute), preserving medication sort within each slot."""
    sorted_doses = sorted(
        doses,
        key=lambda item: (item["hour"], item["minute"], item["medication_name"]),
    )
    groups: list[tuple[tuple[int, int], list[dict]]] = []
    for time_key, group in groupby(
        sorted_doses,
        key=lambda item: (item["hour"], item["minute"]),
    ):
        groups.append((time_key, list(group)))
    return groups


def format_clock_slot_tooltip(entries: list[dict]) -> str:
    """Build hover text listing each medication and its individual status at a time slot."""
    return ", ".join(
        f"{entry['medication_name']} · {entry['display_time']} · {entry['status_label']}"
        for entry in entries
    )
