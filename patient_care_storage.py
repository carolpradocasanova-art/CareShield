"""Durable per-patient Report & Ask storage on local disk.

Used when Supabase tables (patient_care_reports, patient_chat_threads) or
shift_logs.patient_id are missing from the live schema.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("careshield")

_DATA_ROOT = Path(__file__).resolve().parent / ".careshield_data"
_LOCK = threading.Lock()


def care_data_root() -> Path:
    return _DATA_ROOT


def _patient_dir(patient_id: str) -> Path:
    patient_dir = _DATA_ROOT / "patients" / str(patient_id).strip()
    patient_dir.mkdir(parents=True, exist_ok=True)
    return patient_dir


def _reports_path(patient_id: str) -> Path:
    return _patient_dir(patient_id) / "care_reports.json"


def _chat_path(patient_id: str) -> Path:
    return _patient_dir(patient_id) / "chat_thread.json"


def _photos_dir(patient_id: str) -> Path:
    photos_dir = _patient_dir(patient_id) / "photos"
    photos_dir.mkdir(parents=True, exist_ok=True)
    return photos_dir


def _read_json_list(path: Path) -> list:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return []


def _write_json_list(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")


def _normalize_reported_at_iso(reported_at=None) -> str:
    if reported_at:
        try:
            parsed = datetime.fromisoformat(str(reported_at).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).isoformat()
        except (ValueError, TypeError):
            pass
    return datetime.now(timezone.utc).isoformat()


def _report_dedupe_key(row: dict) -> tuple:
    return (
        str(row.get("reported_at") or row.get("created_at") or ""),
        str(row.get("summary") or row.get("report_text") or "")[:200],
        str(row.get("source") or ""),
    )


def save_local_care_report(
    patient_id: str,
    *,
    report_text: str,
    summary: str,
    severity: str,
    source: str,
    reported_at=None,
    caregiver_name: str = "",
    caregiver_id=None,
    photo_finding: str = "",
    photo_type: str = "",
    image_b64: str = "",
) -> dict | None:
    patient_id = str(patient_id).strip()
    report_text = str(report_text or "").strip()
    summary = str(summary or report_text).strip()
    if not patient_id or not summary:
        return None

    reported_iso = _normalize_reported_at_iso(reported_at)
    now_iso = datetime.now(timezone.utc).isoformat()
    row = {
        "id": f"local-{uuid.uuid4().hex[:12]}",
        "patient_id": patient_id,
        "caregiver_id": caregiver_id,
        "caregiver_name": caregiver_name or "Caregiver",
        "source": source or "voice_report",
        "report_text": report_text or summary,
        "summary": summary,
        "severity": severity or "monitor",
        "reported_at": reported_iso,
        "created_at": now_iso,
        "photo_finding": photo_finding or None,
        "photo_type": photo_type or None,
        "has_photo": bool(image_b64),
    }

    path = _reports_path(patient_id)
    with _LOCK:
        rows = _read_json_list(path)
        key = _report_dedupe_key(row)
        if any(_report_dedupe_key(existing) == key for existing in rows):
            for existing in rows:
                if _report_dedupe_key(existing) == key:
                    return existing
        rows.append(row)
        rows.sort(key=lambda item: str(item.get("reported_at") or item.get("created_at") or ""))
        _write_json_list(path, rows)

    if image_b64:
        try:
            photo_path = _photos_dir(patient_id) / f"{row['id']}.json"
            photo_path.write_text(
                json.dumps({"image_b64": image_b64}, default=str),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Could not save local care photo for %s: %s", patient_id, exc)

    return row


def fetch_local_care_reports(patient_id: str, limit: int = 500) -> list:
    patient_id = str(patient_id).strip()
    if not patient_id:
        return []
    rows = _read_json_list(_reports_path(patient_id))
    rows.sort(key=lambda item: str(item.get("reported_at") or item.get("created_at") or ""))
    if limit and len(rows) > limit:
        return rows[-limit:]
    return rows


def purge_local_internal_test_entries(patient_id: str, *, is_test_row) -> int:
    """Remove rows matched by is_test_row from local care reports and chat."""
    patient_id = str(patient_id).strip()
    if not patient_id:
        return 0

    removed = 0
    reports_path = _reports_path(patient_id)
    with _LOCK:
        rows = _read_json_list(reports_path)
        kept = []
        for row in rows:
            if is_test_row(row):
                removed += 1
                report_id = row.get("id")
                if report_id:
                    photo_path = _photos_dir(patient_id) / f"{report_id}.json"
                    try:
                        photo_path.unlink(missing_ok=True)
                    except OSError:
                        pass
            else:
                kept.append(row)
        if removed:
            _write_json_list(reports_path, kept)

        chat_path = _chat_path(patient_id)
        if chat_path.exists():
            try:
                payload = json.loads(chat_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            messages = payload.get("messages") if isinstance(payload, dict) else []
            if isinstance(messages, list):
                filtered_messages = []
                for message in messages:
                    if not isinstance(message, dict):
                        continue
                    if is_test_row({
                        "text": message.get("content", ""),
                        "summary": message.get("content", ""),
                        "source": "chat_message",
                    }):
                        removed += 1
                    else:
                        filtered_messages.append(message)
                if len(filtered_messages) != len(messages):
                    payload["messages"] = filtered_messages
                    chat_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return removed


def fetch_local_care_report_photo(report_id, patient_id: str | None = None) -> str:
    if report_id is None:
        return ""
    report_id = str(report_id)
    if patient_id:
        photo_path = _photos_dir(str(patient_id)) / f"{report_id}.json"
        if photo_path.exists():
            try:
                data = json.loads(photo_path.read_text(encoding="utf-8"))
                return str(data.get("image_b64") or "")
            except (OSError, json.JSONDecodeError):
                pass
    for photo_path in (_DATA_ROOT / "patients").glob(f"*/photos/{report_id}.json"):
        try:
            data = json.loads(photo_path.read_text(encoding="utf-8"))
            return str(data.get("image_b64") or "")
        except (OSError, json.JSONDecodeError):
            continue
    return ""


def save_local_chat_thread(patient_id: str, messages: list) -> bool:
    patient_id = str(patient_id).strip()
    if not patient_id:
        return False
    payload = {
        "patient_id": patient_id,
        "messages": messages or [],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with _LOCK:
            _chat_path(patient_id).write_text(
                json.dumps(payload, indent=2, default=str),
                encoding="utf-8",
            )
        return True
    except OSError as exc:
        logger.warning("save_local_chat_thread failed for %s: %s", patient_id, exc)
        return False


def fetch_local_chat_thread(patient_id: str) -> list:
    patient_id = str(patient_id).strip()
    if not patient_id:
        return []
    path = _chat_path(patient_id)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        messages = data.get("messages") if isinstance(data, dict) else None
        return messages if isinstance(messages, list) else []
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("fetch_local_chat_thread failed for %s: %s", patient_id, exc)
        return []


def _legacy_backfill_marker(patient_id: str) -> Path:
    return _patient_dir(patient_id) / ".legacy_shift_logs_imported"


def legacy_backfill_completed(patient_id: str) -> bool:
    return _legacy_backfill_marker(patient_id).exists()


def mark_legacy_backfill_completed(patient_id: str) -> None:
    _legacy_backfill_marker(patient_id).touch()


def any_patient_has_local_reports() -> bool:
    patients_root = _DATA_ROOT / "patients"
    if not patients_root.exists():
        return False
    for path in patients_root.glob("*/care_reports.json"):
        if _read_json_list(path):
            return True
    return False


def any_other_patient_has_local_reports(patient_id: str) -> bool:
    patient_id = str(patient_id).strip()
    patients_root = _DATA_ROOT / "patients"
    if not patients_root.exists():
        return False
    for path in patients_root.glob("*/care_reports.json"):
        if path.parent.name == patient_id:
            continue
        if _read_json_list(path):
            return True
    return False
