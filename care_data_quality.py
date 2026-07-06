"""Separate internal QA/test care entries from production patient data."""

from __future__ import annotations

import os
import re
from typing import Iterable

_TEST_PATIENT_NAME_RE = re.compile(r"^\[(TEST|QA)\]\s*", re.I)

_INTERNAL_TEST_TEXT_PATTERNS = (
    re.compile(r"\bpersistence\s+probe\b", re.I),
    re.compile(r"\bshift\s+log\s+persistence\s+probe\b", re.I),
    re.compile(r"\bprobe\s+chat\b", re.I),
    re.compile(r"\bprobe\s+symptom\b", re.I),
    re.compile(r"^probe\b", re.I),
    re.compile(r"\bqa\s+probe\b", re.I),
    re.compile(r"\binternal\s+test\b", re.I),
    re.compile(r"\btest\s+probe\b", re.I),
)

_INTERNAL_TEST_CAREGIVER_NAMES = frozenset({"probe", "test", "qa", "tester"})
_INTERNAL_TEST_SOURCES = frozenset({"test_probe"})


def configured_test_patient_ids() -> set[str]:
    raw = os.getenv("CARESHIELD_TEST_PATIENT_IDS", "")
    return {part.strip() for part in raw.split(",") if part.strip()}


def patient_row_is_designated_test(patient: dict | None) -> bool:
    if not patient:
        return False
    patient_id = str(patient.get("id") or "").strip()
    if patient_id and patient_id in configured_test_patient_ids():
        return True
    display_name = str(patient.get("display_name") or patient.get("name") or "").strip()
    return bool(_TEST_PATIENT_NAME_RE.match(display_name))


def is_designated_test_patient(patient_id, patient: dict | None = None) -> bool:
    patient_id = str(patient_id or "").strip()
    if patient_id and patient_id in configured_test_patient_ids():
        return True
    if patient_row_is_designated_test(patient):
        return True
    return False


def _combined_entry_text(
    *,
    text: str = "",
    summary: str = "",
    report_text: str = "",
) -> str:
    return " ".join(
        part.strip()
        for part in (text, summary, report_text)
        if part and str(part).strip()
    ).strip()


def is_internal_test_care_entry(
    *,
    text: str = "",
    summary: str = "",
    report_text: str = "",
    caregiver_name: str = "",
    source: str = "",
) -> bool:
    source_key = str(source or "").strip().lower()
    if source_key in _INTERNAL_TEST_SOURCES:
        return True

    combined = _combined_entry_text(text=text, summary=summary, report_text=report_text)
    if combined and any(pattern.search(combined) for pattern in _INTERNAL_TEST_TEXT_PATTERNS):
        return True

    caregiver_key = str(caregiver_name or "").strip().lower()
    if caregiver_key in _INTERNAL_TEST_CAREGIVER_NAMES and combined:
        if re.search(r"\b(probe|test|qa)\b", combined, re.I):
            return True

    return False


def care_row_is_internal_test(row: dict) -> bool:
    if not isinstance(row, dict):
        return False
    return is_internal_test_care_entry(
        text=str(row.get("text") or row.get("report_text") or ""),
        summary=str(row.get("summary") or ""),
        report_text=str(row.get("report_text") or ""),
        caregiver_name=str(row.get("caregiver_name") or row.get("caregiver") or ""),
        source=str(row.get("source") or ""),
    )


def shift_log_row_is_internal_test(row: dict) -> bool:
    if not isinstance(row, dict):
        return False
    return is_internal_test_care_entry(
        summary=str(row.get("summary") or ""),
        caregiver_name=str(row.get("caregiver_name") or ""),
        source=str(row.get("source") or ""),
    )


def filter_production_care_rows(rows: Iterable[dict], patient_id, patient: dict | None = None) -> list:
    if is_designated_test_patient(patient_id, patient):
        return [row for row in rows or [] if isinstance(row, dict)]
    return [row for row in rows or [] if isinstance(row, dict) and not care_row_is_internal_test(row)]


def filter_production_shift_log_rows(rows: Iterable[dict], patient_id, patient: dict | None = None) -> list:
    if is_designated_test_patient(patient_id, patient):
        return [row for row in rows or [] if isinstance(row, dict)]
    return [
        row for row in rows or []
        if isinstance(row, dict) and not shift_log_row_is_internal_test(row)
    ]


def should_block_test_entry_for_patient(
    patient_id,
    *,
    text: str = "",
    summary: str = "",
    report_text: str = "",
    caregiver_name: str = "",
    source: str = "",
    patient: dict | None = None,
) -> bool:
    """Return True when a QA/test entry must not be stored for this patient."""
    if is_designated_test_patient(patient_id, patient):
        return False
    return is_internal_test_care_entry(
        text=text,
        summary=summary,
        report_text=report_text,
        caregiver_name=caregiver_name,
        source=source,
    )
