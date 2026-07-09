import hashlib
import math
import logging
import os
import streamlit as st
import base64
import csv
import html
import json
import re
import textwrap
from io import BytesIO, StringIO
from symptom_linking import (
    SYMPTOM_PATTERN_DEFINITIONS,
    count_linked_reports,
    count_linked_session_reports,
    detect_session_escalation_triggers,
    detect_session_symptom_recurrence,
    extract_symptoms_from_text,
    find_symptom_related_prior_incidents,
    incident_symptom_relevance_score,
    resolve_incident_symptom_keys,
    select_linked_prior_incidents,
)
from ai_helpers import (
    ask_ai,
    ask_ai_chat,
    save_shift_log,
    supabase,
    generate_sbar_pdf,
    generate_handover_pdf,
    extract_text_from_pdf,
    extract_text_from_pdf_with_meta,
    pdf_extraction_error_response,
    validate_document_patient_profile,
    account_is_multi_patient,
    analyze_symptom_against_conditions,
    extract_relevant_condition_risks,
    escalate_severity,
    build_condition_analysis_prompt_block,
    build_patient_report_timeline_context,
    get_patient_allergy_notes,
    get_patient_medications_for_symptom_context,
    build_stored_chat_context_for_ai,
    build_allergy_symptom_prompt_block,
    build_patient_claim_grounding_prompt_block,
    cap_allergy_report_severity,
    chat_thread_has_user_content,
    chat_thread_belongs_to_caregiver,
    apply_report_severity_floor_caps,
    cap_positive_report_severity,
    cap_informational_question_severity,
    is_clearly_positive_benign_report,
    is_pure_informational_care_question,
    is_care_question_text,
    reports_health_symptom_topic,
    enforce_report_ask_session_evidence,
    enforce_active_patient_name_in_text,
    enforce_patient_record_grounding_in_reply,
    extract_unverified_patient_claims,
    fetch_recent_document_excerpts,
    save_patient_plan,
    get_latest_patient_plan,
    log_medication_taken,
    log_medication_missed,
    log_medication_prn_taken,
    get_medication_logs,
    resolve_patient_id,
    list_account_patients,
    get_or_create_default_patient,
    get_patient_display_name,
    get_patient_conditions,
    get_patient_medications_display,
    replace_patient_conditions,
    save_patient_document_bundle,
    create_patient,
    get_patient_by_id,
    update_patient,
    delete_patient,
    fetch_shift_logs,
    get_medication_references,
    save_patient_test_document,
    save_my_result_record,
    fetch_my_result_records,
    generate_my_results_summary_pdf,
    MY_RESULTS_ASSISTANT_SYSTEM,
    MY_RESULTS_EXTRACT_PROMPT,
    MY_RESULTS_EXPLAIN_PROMPT,
    MY_RESULTS_NO_RESULTS_MESSAGE,
    normalize_my_results_extract,
    normalize_my_results_explain,
    my_results_explain_is_complete,
    my_results_question_text,
    my_results_use_grouped_explanations,
    enrich_my_results_record,
    my_results_has_key_findings,
    my_results_limitation_is_lab_table_artifact,
    my_results_no_abnormal_note_label,
    sanitize_my_results_plain_text,
    MY_RESULTS_DATE_NOT_SPECIFIED,
    my_results_has_actionable_content,
    count_my_results_review_items,
    build_my_results_explain_payload,
    build_my_results_record_from_offline_text,
    build_openai_user_content,
    ai_failure_is_recoverable_offline,
    resolve_ai_failure_reason,
    shift_log_is_internal_storage,
    fetch_symptom_shift_logs,
    fetch_medication_check_shift_logs,
    save_patient_care_report,
    fetch_patient_care_reports,
    fetch_patient_care_report_photo,
    save_patient_chat_thread,
    strip_shift_log_patient_marker,
    format_medcam_shift_log_for_timeline,
    shift_log_belongs_to_patient,
    backfill_legacy_shift_logs_to_local_care,
    purge_internal_test_patient_artifacts,
    purge_suspect_cross_profile_care_reports,
    purge_suspect_cross_profile_chat_messages,
    session_incident_is_valid_for_patient,
    normalize_condition_since,
)
from care_data_quality import (
    care_row_is_internal_test,
    is_designated_test_patient,
    should_block_test_entry_for_patient,
)
from medication_clock import (
    CLOCK_FACE_HOUR_LABELS,
    clock_hour_label_angle_deg,
    clock_label_font_size,
    clock_label_radius_offset,
    dose_angle_deg,
    format_clock_slot_tooltip,
    group_doses_by_schedule_time,
    winning_clock_slot_status,
)
from medication_schedule import (
    build_dose_events,
    compute_dose_ui_state,
    dose_minutes_until,
    extract_pills_per_dose,
    finish_dose_instruction,
    format_medication_frequency,
    format_plan_schedule_summary,
    format_timing_phrase,
    is_prn_timing,
    normalize_medication_schedule_fields,
    parse_schedule_times,
    resolve_schedule_frequency,
    strip_embedded_pill_count_from_timing,
)
from medcam_dose_cards import (
    build_medcam_prn_dose_card_html,
    build_medcam_scheduled_dose_card_html,
)
from medcam_pill_rows import (
    build_medcam_reference_thumb_html,
    build_medcam_row_html_from_display,
)
from medication_dose_queries import (
    build_next_medication_reply_core,
    extract_medication_name_from_question,
)
from handover_events import (
    HANDOVER_INSUFFICIENT_DATA_MSG,
    HANDOVER_PERIOD_OPTIONS,
    build_reported_by_entries_from_events,
    build_sbar_handover_user_payload,
    collect_sbar_events_from_timelines,
    enrich_handover_result_with_period_entries,
    event_in_handover_period,
    filter_events_by_handover_period,
    get_handover_period_bounds,
    get_handover_period_label,
    handover_events_signature,
    handover_severity_label,
    parse_handover_datetime,
    partition_timeline_events_for_sbar,
    REPORT_HISTORY_DEFAULT_VISIBLE,
    REPORT_HISTORY_EXPAND_BATCH,
    count_chat_user_reports,
    slice_messages_for_report_history,
)
from pathlib import Path
from streamlit_javascript import st_javascript
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import streamlit.components.v1 as components

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.DEBUG if os.getenv("CARESHIELD_DEBUG") else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

ADD_NEW_PROFILE_OPTION = "+ Add new profile"
ADD_NEW_PATIENT_OPTION = "+ Add new patient"
CARE_DATA_SCHEMA_VERSION = 3

DEFAULT_CAREGIVER_PROFILES = [
    {"id": "carlos", "name": "Carlos", "role": "son"},
    {"id": "maria", "name": "María", "role": "daughter"},
    {"id": "night_nurse", "name": "Night nurse", "role": ""},
]

LEGACY_CAREGIVER_LABELS = {
    "carlos": "Carlos (son)",
    "maria": "María (daughter)",
    "night_nurse": "Night nurse",
}

LEGACY_LABEL_TO_ID = {label: profile_id for profile_id, label in LEGACY_CAREGIVER_LABELS.items()}
DOT_COLORS = ["blue", "blue", "green", "green", "purple", "orange", "teal", "coral"]

_APP_DIR = Path(__file__).resolve().parent
_CARESHIELD_ICON_PATH = _APP_DIR / "assets" / "careshield-icon.png"
_careshield_icon_b64_cache: tuple[int, str] | None = None
_my_results_logger = logging.getLogger("careshield.my_results")
_documents_logger = logging.getLogger("careshield.documents")
_report_ask_logger = logging.getLogger("careshield.report_ask")


def care_hydrate_key(patient_id=None) -> str:
    return f"care_hydrated_v{CARE_DATA_SCHEMA_VERSION}_{resolve_patient_id(patient_id)}"


def get_careshield_icon_b64() -> str:
    global _careshield_icon_b64_cache
    mtime = int(_CARESHIELD_ICON_PATH.stat().st_mtime)
    if _careshield_icon_b64_cache is None or _careshield_icon_b64_cache[0] != mtime:
        _careshield_icon_b64_cache = (
            mtime,
            base64.b64encode(_CARESHIELD_ICON_PATH.read_bytes()).decode("utf-8"),
        )
    return _careshield_icon_b64_cache[1]


def normalize_medications_raw(raw):
    if raw is None:
        return []
    if isinstance(raw, list):
        items = []
        for entry in raw:
            if isinstance(entry, dict):
                items.append(entry)
            elif isinstance(entry, str) and entry.strip():
                try:
                    parsed = json.loads(entry.strip())
                    items.extend(normalize_medications_raw(parsed))
                except json.JSONDecodeError:
                    items.append({"name": entry.strip(), "dosage": "", "timing": ""})
        return items
    if isinstance(raw, dict):
        nested = raw.get("medications") or raw.get("meds") or raw.get("items")
        if nested is not None:
            return normalize_medications_raw(nested)
        return [raw]
    text = str(raw).strip()
    if not text:
        return []
    current = text
    for _ in range(3):
        try:
            parsed = json.loads(current)
            if isinstance(parsed, str):
                current = parsed.strip()
                continue
            return normalize_medications_raw(parsed)
        except json.JSONDecodeError:
            break
    return []


def parse_medications_for_display(medications_text):
    if medications_text is None:
        return []
    if isinstance(medications_text, (list, dict)):
        plan = load_medication_plan(medications_text)
        if plan["active"]:
            return [_medication_item_from_dict(med, i) for i, med in enumerate(plan["active"])][:8]

    text = str(medications_text).strip()
    if not text:
        return []

    parsed_object = _parse_medications_field_to_object(text)
    if isinstance(parsed_object, (list, dict)):
        plan = load_medication_plan(parsed_object)
        if plan["active"]:
            return [_medication_item_from_dict(med, i) for i, med in enumerate(plan["active"])][:8]

    parsed = _parse_medications_json(text)
    if parsed:
        return parsed

    chunks = re.split(r"[\n;•]+", text)
    items = []
    for i, chunk in enumerate(chunks):
        line = chunk.strip(" -–—,")
        if not line or len(line) < 3:
            continue
        time_match = re.search(r"(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)", line)
        if time_match:
            time = time_match.group(1).upper()
            name = line.replace(time_match.group(0), "").strip(" -–—,")
        else:
            time = ""
            name = line
        items.append({
            "name": name,
            "time": time,
            "color": DOT_COLORS[i % len(DOT_COLORS)],
        })
    return items[:8]


def _medication_item_from_dict(med: dict, index: int) -> dict:
    name = med.get("name") or med.get("medication") or med.get("drug") or "Medication"
    dosage = med.get("dosage") or med.get("dose") or ""
    timing = med.get("timing") or med.get("time") or med.get("schedule") or ""
    display_name = f"{name} {dosage}".strip() if dosage and str(dosage) not in str(name) else str(name)
    item = {
        "name": display_name,
        "dosage": str(dosage) if dosage else "",
        "time": str(timing),
        "color": DOT_COLORS[index % len(DOT_COLORS)],
    }
    pills = extract_pills_per_dose(med)
    if pills is None:
        pills = extract_pills_per_dose(item)
    if pills is not None:
        item["pills_per_dose"] = pills
    return item


def _parse_medications_json(text: str):
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []

    if isinstance(data, dict):
        nested = data.get("medications") or data.get("meds") or data.get("items")
        if nested is not None:
            return _parse_medications_json(json.dumps(nested))
        return [_medication_item_from_dict(data, 0)]

    if not isinstance(data, list):
        return []

    items = []
    for i, med in enumerate(data):
        if isinstance(med, dict):
            items.append(_medication_item_from_dict(med, i))
        elif isinstance(med, str) and med.strip():
            items.append({"name": med.strip(), "time": "", "color": DOT_COLORS[i % len(DOT_COLORS)]})
    return items[:8]


def normalize_condition_status(status) -> str:
    value = str(status or "").strip().lower()
    if value in ("chronic", "recovery", "acute"):
        return value
    if any(word in value for word in ("recover", "post-op", "surgery", "fracture")):
        return "recovery"
    if "acute" in value:
        return "acute"
    return "chronic"


def normalize_conditions_raw(raw) -> list:
    if raw is None:
        return []
    if isinstance(raw, list):
        items = []
        for entry in raw:
            if isinstance(entry, dict):
                name = str(entry.get("name") or entry.get("condition") or "").strip()
                if not name:
                    continue
                onset = normalize_condition_since(
                    entry.get("onset_date")
                    or entry.get("onset")
                    or entry.get("since")
                    or entry.get("date")
                )
                items.append({
                    "name": name,
                    "since": onset,
                    "badge": normalize_condition_status(
                        entry.get("status") or entry.get("type") or entry.get("badge")
                    ),
                })
            elif isinstance(entry, str) and entry.strip():
                items.append({
                    "name": entry.strip(),
                    "since": "From document",
                    "badge": "chronic",
                })
        return items
    if isinstance(raw, str) and raw.strip():
        chunks = re.split(r"[\n;•]+", raw)
        items = []
        for chunk in chunks:
            line = chunk.strip(" -–—,")
            if not line or len(line) < 3:
                continue
            badge = (
                "recovery"
                if any(w in line.lower() for w in ("post-op", "recovery", "surgery", "fracture"))
                else "chronic"
            )
            items.append({"name": line, "since": "From document", "badge": badge})
        return items
    return []


def condition_match_key(name: str) -> str:
    return med_slug(str(name or "").strip().lower())


def merge_conditions(existing: list, incoming: list) -> list:
    merged = [dict(item) for item in existing]
    known = {condition_match_key(item.get("name", "")) for item in merged}
    for item in incoming:
        key = condition_match_key(item.get("name", ""))
        if not key or key in known:
            continue
        merged.append(dict(item))
        known.add(key)
    return merged


def medication_match_key(med: dict) -> str:
    name = str(med.get("name") or med.get("medication") or med.get("drug") or "").strip().lower()
    name = re.sub(r"\s*\d+(?:\.\d+)?\s*(?:mg|g|mcg|µg)\b", "", name, flags=re.I)
    return med_slug(name)


def medication_is_discontinued(med: dict) -> bool:
    status = str(med.get("status") or med.get("action") or "").lower()
    if any(word in status for word in ("stop", "discontinu", "cease", "withhold")):
        return True
    notes = str(med.get("notes") or med.get("instructions") or "").lower()
    if any(word in notes for word in ("stop", "discontinu", "cease")):
        return True
    return False


def infer_medication_action(med: dict) -> str:
    action = str(med.get("action") or med.get("status") or "").strip().lower()
    if any(word in action for word in ("stop", "discontinu", "cease", "withhold")):
        return "discontinue"
    if any(word in action for word in ("change", "increase", "decrease", "adjust", "dose", "reduce")):
        return "dose_change"
    if any(word in action for word in ("start", "new", "add", "begin", "commence")):
        return "start"
    if medication_is_discontinued(med):
        return "discontinue"
    return "continue"


def medication_confidence(med: dict, default: str = "high") -> str:
    value = str(med.get("confidence") or default).strip().lower()
    if value in ("high", "medium", "low"):
        return value
    return default


def _normalize_med_entry_list(raw) -> list:
    if not raw:
        return []
    items = []
    entries = raw if isinstance(raw, list) else [raw]
    for entry in entries:
        if isinstance(entry, dict):
            items.append(dict(entry))
        elif isinstance(entry, str) and entry.strip():
            try:
                parsed = json.loads(entry.strip())
                items.extend(_normalize_med_entry_list(parsed if isinstance(parsed, list) else [parsed]))
            except json.JSONDecodeError:
                items.append({"name": entry.strip(), "dosage": "", "timing": ""})
    return items


def _normalize_discontinued_records(raw) -> list:
    if not raw:
        return []
    records = []
    entries = raw if isinstance(raw, list) else [raw]
    for entry in entries:
        if isinstance(entry, dict):
            name = str(entry.get("name") or entry.get("medication") or entry.get("drug") or "").strip()
            if not name:
                continue
            records.append({
                "name": name,
                "dosage": str(entry.get("dosage") or entry.get("dose") or ""),
                "timing": str(entry.get("timing") or entry.get("time") or entry.get("schedule") or ""),
                "discontinued_at": str(entry.get("discontinued_at") or entry.get("stopped_at") or ""),
                "reason": str(entry.get("reason") or entry.get("notes") or "Stopped per uploaded document"),
                "source_document": str(entry.get("source_document") or entry.get("document") or ""),
            })
        elif isinstance(entry, str) and entry.strip():
            records.append({
                "name": entry.strip(),
                "dosage": "",
                "timing": "",
                "discontinued_at": "",
                "reason": "Stopped per uploaded document",
                "source_document": "",
            })
    return records


def _normalize_review_flag_records(raw) -> list:
    if not raw:
        return []
    flags = []
    entries = raw if isinstance(raw, list) else [raw]
    for entry in entries:
        if isinstance(entry, dict):
            name = str(
                entry.get("medication_name")
                or entry.get("name")
                or entry.get("medication")
                or ""
            ).strip()
            reason = str(entry.get("reason") or entry.get("issue") or entry.get("notes") or "").strip()
            if not name and not reason:
                continue
            flags.append({
                "medication_name": name,
                "reason": reason or "Needs manual review",
                "suggested_action": str(entry.get("suggested_action") or entry.get("action") or "review"),
                "confidence": medication_confidence(entry, "low"),
                "flagged_at": str(entry.get("flagged_at") or entry.get("created_at") or ""),
                "source_document": str(entry.get("source_document") or entry.get("document") or ""),
            })
        elif isinstance(entry, str) and entry.strip():
            flags.append({
                "medication_name": entry.strip(),
                "reason": "Needs manual review",
                "suggested_action": "review",
                "confidence": "low",
                "flagged_at": "",
                "source_document": "",
            })
    return flags


def _parse_medications_field_to_object(raw):
    if raw is None:
        return []
    if isinstance(raw, (list, dict)):
        return raw
    text = str(raw).strip()
    if not text:
        return []
    current = text
    for _ in range(3):
        try:
            parsed = json.loads(current)
            if isinstance(parsed, str):
                current = parsed.strip()
                continue
            return parsed
        except json.JSONDecodeError:
            break
    return text


def load_medication_plan(medications_field) -> dict:
    default = {"active": [], "discontinued": [], "review_flags": []}
    parsed = _parse_medications_field_to_object(medications_field)
    if parsed is None or parsed == "" or parsed == []:
        return dict(default)
    if isinstance(parsed, list):
        return {**default, "active": _normalize_med_entry_list(parsed)}
    if isinstance(parsed, dict):
        if "active" in parsed or "discontinued" in parsed or "review_flags" in parsed:
            return {
                "active": _normalize_med_entry_list(parsed.get("active", [])),
                "discontinued": _normalize_discontinued_records(parsed.get("discontinued", [])),
                "review_flags": _normalize_review_flag_records(parsed.get("review_flags", [])),
            }
        nested = parsed.get("medications") or parsed.get("meds") or parsed.get("items")
        if nested is not None:
            return load_medication_plan(nested)
        return {**default, "active": _normalize_med_entry_list([parsed])}
    return dict(default)


def serialize_medication_plan(envelope: dict) -> str:
    return json.dumps({
        "active": envelope.get("active") or [],
        "discontinued": envelope.get("discontinued") or [],
        "review_flags": envelope.get("review_flags") or [],
    })


def normalize_medications_raw(raw):
    if raw is None:
        return []
    if isinstance(raw, list):
        return _normalize_med_entry_list(raw)
    if isinstance(raw, dict):
        if "active" in raw or "discontinued" in raw or "review_flags" in raw:
            return _normalize_med_entry_list(raw.get("active") or [])
        nested = raw.get("medications") or raw.get("meds") or raw.get("items")
        if nested is not None:
            return normalize_medications_raw(nested)
        return _normalize_med_entry_list([raw])
    return load_medication_plan(raw)["active"]


def find_active_medication_matches(name: str, active: list) -> list[dict]:
    if not str(name or "").strip():
        return []
    target_key = medication_match_key({"name": name})
    if not target_key:
        return []
    exact = [med for med in active if medication_match_key(med) == target_key]
    if exact:
        return exact
    partial = []
    for med in active:
        med_key = medication_match_key(med)
        if not med_key:
            continue
        if target_key in med_key or med_key in target_key:
            partial.append(med)
    return partial


def _med_payload_for_storage(med: dict) -> dict:
    payload = dict(med)
    for key in ("action", "confidence", "status", "notes", "issue", "suggested_action"):
        payload.pop(key, None)
    return normalize_medication_schedule_fields(payload)


def _append_review_flag(
    review_flags: list,
    medication_name: str,
    reason: str,
    suggested_action: str,
    source_document: str,
    flagged_at: str,
):
    review_flags.append({
        "medication_name": medication_name,
        "reason": reason,
        "suggested_action": suggested_action,
        "confidence": "low",
        "flagged_at": flagged_at,
        "source_document": source_document,
    })


def _normalize_discontinue_request(entry) -> dict:
    if isinstance(entry, str):
        return {
            "name": entry.strip(),
            "reason": "Stopped per uploaded document",
            "confidence": "high",
        }
    return {
        "name": str(entry.get("name") or entry.get("medication") or entry.get("drug") or "").strip(),
        "reason": str(entry.get("reason") or entry.get("notes") or "Stopped per uploaded document"),
        "confidence": medication_confidence(entry, "high"),
    }


def partition_incoming_medications(incoming: list) -> tuple[list, list]:
    active_incoming = []
    discontinue_requests = []
    for med in incoming or []:
        if not isinstance(med, dict):
            continue
        action = infer_medication_action(med)
        if action == "discontinue":
            discontinue_requests.append({
                "name": med.get("name") or med.get("medication") or med.get("drug") or "",
                "reason": med.get("reason") or med.get("notes") or "Stopped per uploaded document",
                "confidence": medication_confidence(med, "high"),
            })
        else:
            active_incoming.append(med)
    return active_incoming, discontinue_requests


def apply_document_medication_changes(
    existing_envelope: dict,
    incoming_meds: list,
    discontinued_entries: list | None = None,
    review_items: list | None = None,
    document_name: str = "",
    processed_at: str = "",
) -> dict:
    active = [dict(med) for med in (existing_envelope or {}).get("active", [])]
    discontinued = [dict(med) for med in (existing_envelope or {}).get("discontinued", [])]
    review_flags = [dict(flag) for flag in (existing_envelope or {}).get("review_flags", [])]

    active_incoming, discontinue_from_meds = partition_incoming_medications(incoming_meds or [])
    discontinue_requests = [
        _normalize_discontinue_request(entry)
        for entry in (discontinued_entries or [])
    ] + discontinue_from_meds

    for med in active_incoming:
        name = str(med.get("name") or med.get("medication") or med.get("drug") or "").strip()
        if not name:
            continue

        confidence = medication_confidence(med, "high")
        action = infer_medication_action(med)
        matches = find_active_medication_matches(name, active)

        if confidence != "high":
            _append_review_flag(
                review_flags,
                name,
                f"The document mentions {name}, but it is unclear whether to {action.replace('_', ' ')} it.",
                action,
                document_name,
                processed_at,
            )
            continue

        if action == "dose_change" and len(matches) != 1:
            _append_review_flag(
                review_flags,
                name,
                "The document may change this dose, but CareShield could not confidently match it to one stored medication.",
                "dose_change",
                document_name,
                processed_at,
            )
            continue

        if action in ("continue", "start", "dose_change") and len(matches) > 1:
            _append_review_flag(
                review_flags,
                name,
                "Multiple stored medications match this name. Please review manually before updating.",
                action,
                document_name,
                processed_at,
            )
            continue

        payload = _med_payload_for_storage(med)
        if len(matches) == 1:
            idx = active.index(matches[0])
            active[idx] = normalize_medication_schedule_fields({**active[idx], **payload})
        else:
            active.append(payload)

    seen_discontinue = set()
    for request in discontinue_requests:
        name = str(request.get("name") or "").strip()
        if not name:
            continue
        dedupe_key = (medication_match_key({"name": name}), request.get("reason", ""))
        if dedupe_key in seen_discontinue:
            continue
        seen_discontinue.add(dedupe_key)

        confidence = medication_confidence(request, "high")
        matches = find_active_medication_matches(name, active)

        if confidence != "high" or len(matches) != 1:
            reason = (
                f"The document may say to stop {name}, but CareShield is not confident enough to remove it automatically."
            )
            if len(matches) == 0:
                reason = (
                    f"The document says to stop {name}, but it is not on the active medication list — please review manually."
                )
            elif len(matches) > 1:
                reason = (
                    f"The document says to stop {name}, but multiple stored medications match — please review manually."
                )
            _append_review_flag(
                review_flags,
                name,
                reason,
                "discontinue",
                document_name,
                processed_at,
            )
            continue

        removed = active.pop(active.index(matches[0]))
        discontinued.append({
            **_med_payload_for_storage(removed),
            "discontinued_at": processed_at,
            "reason": request.get("reason") or "Stopped per uploaded document",
            "source_document": document_name,
        })

    for item in _normalize_review_flag_records(review_items or []):
        if not item.get("flagged_at"):
            item["flagged_at"] = processed_at
        if not item.get("source_document"):
            item["source_document"] = document_name
        review_flags.append(item)

    return {
        "active": active,
        "discontinued": discontinued,
        "review_flags": review_flags,
    }


def merge_medications(existing: list, incoming: list, discontinued_keys: list | None = None) -> list:
    """Legacy helper — prefer apply_document_medication_changes for document uploads."""
    envelope = apply_document_medication_changes(
        {"active": existing, "discontinued": [], "review_flags": []},
        incoming,
        [{"name": key, "confidence": "high"} for key in (discontinued_keys or [])],
    )
    return envelope["active"]


def parse_conditions_for_display(conditions_raw):
    return normalize_conditions_raw(conditions_raw)


def condition_badge_meta(badge: str) -> tuple[str, str]:
    if badge == "recovery":
        return "cs-badge-recovery", "Recovery"
    if badge == "acute":
        return "cs-badge-acute", "Acute"
    return "cs-badge-chronic", "Chronic"


def md_html(html_str: str):
    st.markdown(textwrap.dedent(html_str).strip(), unsafe_allow_html=True)


def render_homepage_boot_ready() -> None:
    """Deprecated — homepage transition now uses careshield_enter_from_homepage."""
    return


def render_careshield_enter_from_homepage() -> None:
    """Skip splash and show the main app immediately after the homepage CTA."""
    components.html(
        """
        <script>
        (function () {
          const doc = window.parent.document;
          const root = doc.documentElement;
          root.classList.add("cs-app-ready", "cs-enter-from-home");
          try {
            sessionStorage.setItem("cs-app-ready", "1");
          } catch (e) {}
          const splash = doc.getElementById("cs-boot-splash");
          if (splash) splash.style.display = "none";
        })();
        </script>
        """,
        height=0,
    )


def render_careshield_boot_instant_ready() -> None:
    components.html(
        """
        <script>
        (function () {
          const doc = window.parent.document;
          if (sessionStorage.getItem("cs-app-ready") === "1") {
            doc.documentElement.classList.add("cs-app-ready");
            const splash = doc.getElementById("cs-boot-splash");
            if (splash) splash.style.display = "none";
          }
        })();
        </script>
        """,
        height=0,
    )


def render_careshield_tab_restore() -> None:
    """Re-activate the Streamlit tab matching careshield_active_tab after reruns."""
    tab_id = st.session_state.get("careshield_active_tab", "documents")
    tab_index = CARESHIELD_TAB_IDS.index(tab_id) if tab_id in CARESHIELD_TAB_IDS else 0
    if tab_index == 0:
        return
    components.html(
        f"""
        <script>
        (function () {{
          const doc = window.parent.document;
          function activateTab() {{
            const tabs = doc.querySelectorAll('[data-testid="stTabs"] [data-baseweb="tab"]');
            const target = tabs[{tab_index}];
            if (!target) return;
            if (target.getAttribute("aria-selected") === "true") return;
            target.click();
          }}
          activateTab();
          window.setTimeout(activateTab, 40);
          window.setTimeout(activateTab, 120);
        }})();
        </script>
        """,
        height=0,
    )


def render_careshield_boot_reveal() -> None:
    if st.session_state.get("careshield_skip_boot_reveal"):
        return
    components.html(
        """
        <script>
        (function () {
          const doc = window.parent.document;
          const root = doc.documentElement;
          let revealed = false;
          function reveal() {
            if (revealed) return;
            revealed = true;
            root.classList.add("cs-app-ready");
            try {
              sessionStorage.setItem("cs-app-ready", "1");
            } catch (e) {}
            const splash = doc.getElementById("cs-boot-splash");
            if (splash) {
              splash.setAttribute("aria-busy", "false");
              splash.style.opacity = "0";
              window.setTimeout(function () {
                splash.style.display = "none";
              }, 200);
            }
          }
          function finishBoot() {
            if (sessionStorage.getItem("cs-app-ready") === "1") {
              reveal();
              return;
            }
            window.setTimeout(reveal, 30);
          }
          if (doc.documentElement.classList.contains("cs-enter-from-home")) {
            reveal();
            return;
          }
          if (doc.fonts && doc.fonts.ready) {
            doc.fonts.ready.then(finishBoot).catch(finishBoot);
          } else {
            finishBoot();
          }
          window.setTimeout(reveal, 1800);
        })();
        </script>
        """,
        height=0,
    )


def build_loading_banner_html(message: str) -> str:
    return f"""
    <div class="cs-loading-banner" role="status" aria-live="polite">
      <span class="cs-loading-spinner" aria-hidden="true"></span>
      <span class="cs-loading-text">{html.escape(message)}</span>
    </div>
    """


def build_upload_confirmation_html(
    filename: str,
    *,
    size_bytes: int | None = None,
    kind: str = "Document",
) -> str:
    size_label = ""
    if size_bytes:
        if size_bytes >= 1024 * 1024:
            size_label = f"{size_bytes / (1024 * 1024):.1f} MB"
        elif size_bytes >= 1024:
            size_label = f"{size_bytes / 1024:.0f} KB"
        else:
            size_label = f"{size_bytes} B"
    meta_bits = [html.escape(kind)]
    if size_label:
        meta_bits.append(html.escape(size_label))
    meta_html = f'<div class="cs-upload-confirm-meta">{" · ".join(meta_bits)}</div>'
    return f"""
    <div class="cs-upload-confirm">
      <div class="cs-upload-confirm-label">Selected file</div>
      <div class="cs-upload-confirm-name">{html.escape(filename)}</div>
      {meta_html}
      <div class="cs-upload-confirm-hint">Please confirm this is the correct file before continuing.</div>
    </div>
    """


SYMPTOM_TIMELINE_HEADING = "Symptom Timeline"
SYMPTOM_TIMELINE_DESCRIPTION = (
    "View all saved symptom reports and health updates for this patient in chronological order."
)
MEDICATION_ADHERENCE_TIMELINE_HEADING = "Medication Adherence Timeline"
MEDICATION_ADHERENCE_TIMELINE_DESCRIPTION = (
    "View the history of medications taken, missed, or recorded for this patient."
)

CARESHIELD_TAB_IDS = (
    "documents",
    "report",
    "pill",
    "medcam",
    "handover",
    "my_results",
)


def set_careshield_active_tab(tab_id: str) -> None:
    if tab_id in CARESHIELD_TAB_IDS:
        st.session_state.careshield_active_tab = tab_id


def render_section_intro(title: str, description: str) -> None:
    """Shared tab header — matches Report & Ask title spacing (no bordered box)."""
    md_html(f"""
    <div class="cs-report-intro">
      <h3 class="cs-report-title">{html.escape(title)}</h3>
      <p class="cs-report-desc">{html.escape(description)}</p>
    </div>
    """)


def build_tab_story_html(title: str, problem: str, solution: str) -> str:
    return f"""
    <div class="cs-report-intro">
      <h3 class="cs-report-title">{html.escape(title)}</h3>
    </div>
    <div class="cs-report-story-grid">
      <div class="cs-report-story-card cs-report-story-card--problem">
        <div class="cs-report-story-badge">The problem</div>
        <p class="cs-report-story-text">{html.escape(problem)}</p>
      </div>
      <div class="cs-report-story-card cs-report-story-card--solution">
        <div class="cs-report-story-badge cs-report-story-badge--solution">Our solution</div>
        <p class="cs-report-story-text">{html.escape(solution)}</p>
      </div>
    </div>
    """


def render_tab_story_section(title: str, problem: str, solution: str) -> None:
    md_html(build_tab_story_html(title, problem, solution))


TAB_PROBLEM_SOLUTION = {
    "documents": (
        "Without knowing a patient's actual medications and conditions, an AI assistant can only respond generically. "
        "It can't connect today's swollen legs to an existing heart condition, or know that new drowsiness might be "
        "a side effect of a blood pressure pill started last week, rather than something alarming on its own.",
        "CareShield's Documents feature reads hospital paperwork the moment it's uploaded, extracting medications, "
        "doses, and chronic conditions automatically. This becomes the medical context the AI uses everywhere else, "
        "so when a caregiver reports a symptom in Report & Ask, CareShield can reason about it against the patient's "
        "real history instead of guessing in the dark.",
    ),
    "report_ask": (
        "Family caregivers are often expected to track symptoms, remember when changes happened, and share accurate "
        "information with doctors—all while managing the stress and demands of caring for a loved one. As a result, "
        "important details can be forgotten, changes in health may go unnoticed, and doctors may not always have the "
        "complete picture.",
        "CareShield's Report & Ask feature helps caregivers record health updates as they happen, identify possible "
        "patterns in symptoms over time, and generate clear handover reports for healthcare professionals. By turning "
        "everyday observations into organized timelines, CareShield AI helps caregivers have more informed and productive "
        "conversations with clinicians.",
    ),
    "pill_registration": (
        "Generic pills often look alike, and a tired caregiver giving medication late at night can easily mix up two "
        "white tablets or misjudge a dose. Verbal instructions from a pharmacist are easy to forget by the time the "
        "bottle is actually opened at home.",
        "CareShield's Pill registration lets caregivers photograph each medication once, teaching the system exactly "
        "what the patient's real pills look like. Strength and dosage are calculated automatically from the discharge "
        "document, so there's no guesswork at the moment it matters.",
    ),
    "medcam": (
        "Even with a clear schedule, it's hard to know in the moment whether the right pills, in the right amount, "
        "are about to be given. Missed or doubled doses often go unnoticed until a doctor asks about adherence weeks later.",
        "CareShield's MedCam checks a photo of the pills in hand against the patient's registered medications and "
        "active schedule, confirming the right pill and the right count before it's given. Missed or mismatched doses "
        "are flagged immediately, not discovered after the fact.",
    ),
    "handover": (
        "At doctor visits, caregivers are asked to summarize weeks of symptoms, incidents, and medication changes from "
        "memory, under time pressure, often while emotionally exhausted. Crucial details get left out, and patterns "
        "connecting separate events go unspoken.",
        "CareShield's Handover automatically compiles every logged update into a clinical-grade SBAR report, complete "
        "with severity scoring, adherence tracking, and a connected symptom timeline. What used to be a rushed verbal "
        "recap becomes a structured document ready for the appointment.",
    ),
    "my_results": (
        "Lab results and clinic letters arrive full of medical terminology and reference ranges that mean little to a "
        "non-clinical caregiver. Without context, it's hard to know what's actually concerning and what questions are "
        "worth asking the doctor.",
        "CareShield's My results explains uploaded test results in plain English, flags which values are outside the "
        "normal range, and suggests specific questions to bring to the next appointment, turning a confusing report "
        "into something a caregiver can actually act on.",
    ),
}

def render_profile_manager_row(
    label: str,
    is_active: bool,
    *,
    edit_key: str,
    delete_key: str,
) -> tuple[bool, bool]:
    """Profile manager row with horizontal Edit / Delete buttons."""
    prefix = "●" if is_active else "○"
    name_col, edit_col, delete_col = st.columns(
        [2.4, 0.85, 0.85],
        gap="small",
        vertical_alignment="center",
    )
    with name_col:
        st.markdown(f"{prefix} {html.escape(label)}")
    with edit_col:
        edit_clicked = st.button("Edit", key=edit_key, use_container_width=False)
    with delete_col:
        delete_clicked = st.button("Delete", key=delete_key, use_container_width=False)
    return edit_clicked, delete_clicked


def cs_icon(name: str, size: int = 14) -> str:
    icons = {
        "warn": (
            '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            '<path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
            '<line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>'
            "</svg>"
        ),
        "clipboard": (
            '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            '<path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/>'
            '<rect x="8" y="2" width="8" height="4" rx="1" ry="1"/>'
            "</svg>"
        ),
        "pill": (
            '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            '<path d="m10.5 20.5 10-10a4.95 4.95 0 1 0-7-7l-10 10a4.95 4.95 0 1 0 7 7Z"/>'
            '<path d="m8.5 8.5 7 7"/>'
            "</svg>"
        ),
        "document": (
            '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
            '<polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/>'
            '<line x1="16" y1="17" x2="8" y2="17"/>'
            "</svg>"
        ),
        "check": (
            '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            '<polyline points="20 6 9 17 4 12"/>'
            "</svg>"
        ),
    }
    svg = icons.get(name, "").format(s=size)
    return f'<span class="cs-inline-icon cs-icon-{name}">{svg}</span>'


def build_careshield_logo_html(include_tagline: bool = True) -> str:
    tagline_html = (
        '<div class="cs-logo-sub">Family care, watched closely</div>'
        if include_tagline
        else ""
    )
    icon_b64 = get_careshield_icon_b64()
    return f"""
    <div class="cs-logo">
      <div class="cs-logo-icon" aria-hidden="true">
        <img class="cs-logo-brand-img" src="data:image/png;base64,{icon_b64}" alt="">
      </div>
      <div class="cs-logo-text">
        <div class="cs-logo-name">
          <span class="cs-logo-care">Care</span><span class="cs-logo-shield-word">Shield</span>
        </div>
        {tagline_html}
      </div>
    </div>
    """


HANDOVER_TIMELINE_DEFAULT_VISIBLE = REPORT_HISTORY_DEFAULT_VISIBLE
HANDOVER_TIMELINE_EXPAND_BATCH = REPORT_HISTORY_EXPAND_BATCH
CHAT_REPORT_HISTORY_DEFAULT_VISIBLE = REPORT_HISTORY_DEFAULT_VISIBLE
CHAT_REPORT_HISTORY_EXPAND_BATCH = REPORT_HISTORY_EXPAND_BATCH

def filter_session_incidents_by_period(period_key: str, tz_obj, patient_id=None) -> list:
    return filter_events_by_handover_period(
        get_session_incidents(patient_id),
        period_key,
        tz_obj,
    )


def get_session_photo_reviews_for_handover(period_key: str, tz_obj) -> list:
    return get_handover_photo_reviews_for_period(
        st.session_state.get("selected_patient_id"),
        period_key,
        tz_obj,
    )


def resolve_timeline_event_image_b64(event: dict, patient_id=None) -> str:
    image_b64 = str(event.get("image_b64") or "").strip()
    if image_b64:
        return image_b64
    if event.get("has_photo") and event.get("report_id") is not None:
        return fetch_patient_care_report_photo(
            event["report_id"],
            patient_id=patient_id,
        ) or ""
    return ""


def timeline_event_has_photo_review(event: dict, patient_id=None) -> bool:
    source = str(event.get("source") or "")
    if source in ("symptom_photo", "pill_photo"):
        return bool(resolve_timeline_event_image_b64(event, patient_id))
    return bool(event.get("has_photo") and resolve_timeline_event_image_b64(event, patient_id))


def get_handover_photo_reviews_for_period(patient_id, period_key: str, tz_obj) -> list:
    """All caregiver photo submissions in the selected handover period."""
    resolved_patient_id = resolve_patient_id(patient_id)
    symptom_events = filter_events_by_handover_period(
        load_symptom_timeline_events(patient_id=resolved_patient_id),
        period_key,
        tz_obj,
    )
    reviews = []
    for event in symptom_events:
        if not timeline_event_has_photo_review(event, resolved_patient_id):
            continue
        image_b64 = resolve_timeline_event_image_b64(event, resolved_patient_id)
        if not image_b64:
            continue
        reviews.append({
            **event,
            "image_b64": image_b64,
            "summary": event.get("text") or "",
        })
    return reviews


def attach_timeline_event_photos(events: list, patient_id=None) -> list:
    """Load durable symptom photos onto timeline rows for handover display/PDF."""
    resolved_id = resolve_patient_id(patient_id)
    enriched = []
    for event in events or []:
        row = dict(event)
        if not row.get("image_b64"):
            image_b64 = resolve_timeline_event_image_b64(row, resolved_id)
            if image_b64:
                row["image_b64"] = image_b64
        enriched.append(row)
    return enriched


def filter_shift_logs_by_period(logs: list, period_key: str, tz_obj) -> list:
    filtered = []
    for row in logs or []:
        if event_in_handover_period({"timestamp": row.get("created_at")}, period_key, tz_obj):
            filtered.append(row)
    return filtered


def collect_handover_events_for_sbar(patient_id, period_key: str, tz_obj) -> tuple[list, list]:
    """Return (symptom_events, adherence_events) for the selected handover period."""
    resolved_id = resolve_patient_id(patient_id)
    for cache_key in (
        f"shift_logs_{resolved_id}_250",
        f"shift_logs_{resolved_id}_50",
        f"adherence_timeline_{resolved_id}",
        f"symptom_timeline_db_{patient_id}",
        f"care_reports_timeline_{resolved_id}",
    ):
        st.session_state.pop(cache_key, None)

    all_symptom = filter_events_by_handover_period(
        load_symptom_timeline_events(force_refresh=True, patient_id=patient_id),
        period_key,
        tz_obj,
    )
    all_adherence = filter_events_by_handover_period(
        load_medication_adherence_timeline_events(patient_id, force_refresh=True),
        period_key,
        tz_obj,
    )
    return all_symptom, all_adherence


def render_handover_period_selector() -> str:
    if "handover_period" not in st.session_state:
        st.session_state.handover_period = "this_week"

    md_html("""
    <div class="cs-handover-card">
      <div class="cs-handover-title">Generate handover for period</div>
    </div>
    """)
    period_cols = st.columns(len(HANDOVER_PERIOD_OPTIONS))
    for col, (period_key, label) in zip(period_cols, HANDOVER_PERIOD_OPTIONS):
        with col:
            is_active = st.session_state.handover_period == period_key
            if st.button(
                label,
                key=f"handover_period_{period_key}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                if st.session_state.handover_period != period_key:
                    st.session_state.handover_period = period_key
                    st.session_state.pop("last_sbar_result", None)
                    st.session_state.handover_symptom_timeline_visible = HANDOVER_TIMELINE_DEFAULT_VISIBLE
                    st.session_state.handover_adherence_timeline_visible = HANDOVER_TIMELINE_DEFAULT_VISIBLE
                    st.session_state.handover_reported_by_visible = HANDOVER_TIMELINE_DEFAULT_VISIBLE
                st.rerun()
    return st.session_state.handover_period


def build_sbar_handover_system_prompt(period_label: str) -> str:
    return f"""You are an experienced nurse writing a brief SBAR handover for a family caregiver or GP.
Summarize ONLY the events in the user message — they are already filtered to: {period_label}.

The user message has two labelled sections:
1. SYMPTOM REPORTS & CARE UPDATES — from Report & Ask, documents, and photos. These MUST shape Situation, Background, and Assessment when present. Do not ignore them in favour of dose logs.
2. MEDICATION ADHERENCE — dose taken/missed logs and MedCam checks. Include these in Background and Recommendation, but do not let them replace symptom reports in Situation when both exist.

WRITING STYLE (critical):
- Write the way a calm, clear nurse would speak out loud during handover — not like a spreadsheet or audit log.
- The whole handover should be skimmable in under 30 seconds by a stressed family member or a doctor between patients.
- Use relative, natural time language: "this morning", "around midday", "over the course of the afternoon", "twice in the evening", "earlier today", "throughout the day".
- Do NOT list exact clock times (no "17:19", "18:12", etc.) anywhere in your response.
- Group repeated occurrences of the same symptom or finding into one sentence instead of listing each instance.
  Good: "Repeated chest pain and breathlessness throughout the afternoon and evening."
  Bad: "Chest pain at 17:19, 17:49, and 18:10."
- Keep specific numbers only when clinically meaningful as a summary — e.g. "blood pressure stayed elevated around 158/95 across several readings this afternoon" — but never enumerate every reading with its own time.
- Connect related incidents (e.g. confusion earlier and a fall later) and note if urgency increased over time.
- When the same concern is reported more than once, reflect the HIGHEST severity reached (e.g. if first logged as CONTACT DOCTOR and later EMERGENCY, state EMERGENCY explicitly and mention 999/112 if that was recommended).
- Every line in SYMPTOM REPORTS includes a severity label in square brackets — preserve those levels; never soften EMERGENCY to vague concern language.
- SYMPTOM PHOTOS LOGGED must be mentioned in Assessment (bruise, rash, swelling, etc.) — these are clinically important visual findings.
- Do not drop or soften anything clinically important — preserve severity, red flags, and medication issues — just phrase them in human language.

Respond with ONLY a JSON object:
1. "situation": one short sentence on what matters most right now
2. "background": relevant context for the period, grouped and time-framed naturally (2–4 sentences max)
3. "assessment": clinical interpretation of severity and risk, including linked incidents (2–3 sentences max)
4. "recommendation": the single most important next action (one clear sentence)
5. "watch_for": one short alert about what to monitor (one sentence)
6. "reported_by": list of objects with "caregiver" and "note" — each note is one plain-language summary of what that person reported (no exact timestamps; use natural time if needed)
"""


def caregiver_first_name(caregiver_label: str) -> str:
    return caregiver_label.split("(")[0].strip()


def build_welcome_message(caregiver_label: str) -> str:
    name = caregiver_first_name(caregiver_label)
    return f"""<p class="cs-welcome-lead">Hi {html.escape(name)}! How can I help you today?</p>
<p class="cs-welcome-sub">You can share an update about your loved one or ask any question about their care.</p>
<p class="cs-welcome-sub">You can also upload a photo of a symptom (such as a bruise, rash, or swelling).</p>
<p class="cs-welcome-note">Please note: only symptom photos are supported at this time.</p>
<div class="cs-welcome-toggles">
  <details class="cs-welcome-details">
    <summary>How CareShield AI Protects Your Timeline?</summary>
    <div class="cs-welcome-details-body">
      <p><strong>Automatic Timestamps:</strong> Everything you report is saved with the date and time, so you don&rsquo;t have to remember every detail yourself.</p>
      <p><strong>Pattern Recognition:</strong> CareShield AI looks for patterns across updates&mdash;like connecting early confusion with a later fall&mdash;to track important changes over time.</p>
      <p><strong>Instant Export:</strong> Download a complete, doctor-ready handover report at any time from the Handover tab.</p>
    </div>
  </details>
  <details class="cs-welcome-details">
    <summary>Examples of things you can report</summary>
    <div class="cs-welcome-details-body">
      <ul>
        <li>&ldquo;Dad woke up confused this morning and refused breakfast.&rdquo;</li>
        <li>&ldquo;He had trouble swallowing his pill at lunch.&rdquo;</li>
        <li>&ldquo;His blood pressure was higher than usual today.&rdquo;</li>
      </ul>
    </div>
  </details>
  <details class="cs-welcome-details">
    <summary>Examples of things you can ask</summary>
    <div class="cs-welcome-details-body">
      <ul>
        <li>&ldquo;When does Dad need his next medication?&rdquo;</li>
        <li>&ldquo;Is it normal that he&rsquo;s sleepier on this medication?&rdquo;</li>
        <li>&ldquo;What should I watch for after his knee surgery?&rdquo;</li>
      </ul>
    </div>
  </details>
</div>"""


def build_report_ask_story_html() -> str:
    problem, solution = TAB_PROBLEM_SOLUTION["report_ask"]
    return build_tab_story_html("Report and Ask", problem, solution)


def build_responsible_ai_footer_html() -> str:
    return """
    <div class="cs-report-safety cs-report-safety--footer">
      <div class="cs-report-safety-label">Responsible AI</div>
      <div class="cs-report-responsible-list">
        <div class="cs-report-responsible-item">
          <div class="cs-report-responsible-icon">🔒</div>
          <div>
            <div class="cs-report-responsible-title">Privacy first</div>
            <p class="cs-report-responsible-text">Care information is securely stored and only accessible to authorized users.</p>
          </div>
        </div>
        <div class="cs-report-responsible-item">
          <div class="cs-report-responsible-icon">🩺</div>
          <div>
            <div class="cs-report-responsible-title">Decision support, not diagnosis</div>
            <p class="cs-report-responsible-text">Provides insights and organization tools, never a replacement for professional medical advice.</p>
          </div>
        </div>
        <div class="cs-report-responsible-item">
          <div class="cs-report-responsible-icon">✓</div>
          <div>
            <div class="cs-report-responsible-title">Human oversight</div>
            <p class="cs-report-responsible-text">You remain in control of all information and decide what to share with clinicians.</p>
          </div>
        </div>
      </div>
      <div class="cs-report-disclaimer cs-report-disclaimer--bottom">
        <strong>Important:</strong> CareShield AI is not a doctor and may make mistakes.
        Use this tool to stay organised, but always contact the patient&rsquo;s doctor for
        medical decisions, emergencies, or anything that feels urgent.
      </div>
    </div>
    """


def render_responsible_ai_footer() -> None:
    md_html(build_responsible_ai_footer_html())


def build_report_ask_safety_html() -> str:
    return build_responsible_ai_footer_html()


def refresh_chat_welcome_message(caregiver_label: str) -> None:
    """Keep the welcome bubble in sync when its HTML changes."""
    messages = st.session_state.get("messages") or []
    if messages and messages[0].get("welcome"):
        messages[0]["content"] = build_welcome_message(caregiver_label)


def reset_report_ask_for_caregiver_switch(
    patient_id: str | None,
    caregiver_label: str,
    caregiver_id: str,
    *,
    previous_caregiver_id: str | None = None,
) -> None:
    """Start a fresh visible chat when the logged-in caregiver profile changes."""
    resolved_patient = resolve_patient_id(patient_id) if patient_id else None
    if resolved_patient and chat_thread_has_user_content(st.session_state.get("messages")):
        persist_patient_chat_thread(resolved_patient)

    st.session_state.messages = build_initial_chat_messages(caregiver_label)
    st.session_state.chat_caregiver = caregiver_label
    st.session_state.chat_caregiver_id = caregiver_id
    st.session_state.report_ask_bound_caregiver_id = caregiver_id
    st.session_state.pop("pending_chat_response", None)
    st.session_state.reset_chat_draft = True

    if resolved_patient:
        st.session_state[get_session_incidents_key(resolved_patient)] = []

    if previous_caregiver_id and previous_caregiver_id != caregiver_id:
        _report_ask_logger.info(
            "Caregiver switch: patient=%s previous_caregiver=%s new_caregiver=%s — chat reset",
            resolved_patient,
            previous_caregiver_id,
            caregiver_id,
        )


def build_initial_chat_messages(caregiver_label: str):
    return [{"role": "assistant", "content": build_welcome_message(caregiver_label), "welcome": True}]


def report_ask_needs_caregiver_reset(
    selected_caregiver_id: str,
    messages: list | None = None,
) -> bool:
    """Detect when Report & Ask visible chat belongs to a different caregiver profile."""
    bound_caregiver_id = st.session_state.get("report_ask_bound_caregiver_id")
    if bound_caregiver_id != selected_caregiver_id:
        return True
    session_chat_caregiver_id = st.session_state.get("chat_caregiver_id")
    if session_chat_caregiver_id and session_chat_caregiver_id != selected_caregiver_id:
        return True
    if not chat_thread_belongs_to_caregiver(messages or st.session_state.get("messages"), selected_caregiver_id):
        return True
    return False


def format_message_content(msg: dict) -> str:
    if msg.get("welcome"):
        return msg["content"]
    text = msg["content"]
    if msg["role"] == "user":
        return html.escape(text).replace("\n", "<br>")
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    return escaped.replace("\n", "<br>")


def normalize_chat_severity(severity: str) -> str:
    value = str(severity or "monitor").strip().lower().replace(" ", "_")
    if value == "urgent":
        return "emergency"
    if value in ("contact_doctor", "contactdoctor", "doctor"):
        return "contact_doctor"
    if value in ("ok", "monitor", "emergency"):
        return value
    return "monitor"


def is_emergency_severity(severity: str) -> bool:
    return normalize_chat_severity(severity) == "emergency"


def is_contact_doctor_severity(severity: str) -> bool:
    return normalize_chat_severity(severity) == "contact_doctor"


CONTACT_DOCTOR_GUIDANCE = (
    "Call the GP or consultant within 24 hours — do not wait longer if symptoms worsen."
)

CHAT_URGENCY_RULES = """URGENCY CLASSIFICATION — use exactly one of these four levels in "severity" (or "risk_level" for photos):
- "ok": stable, routine observation only.
- "monitor": mild or new symptoms worth watching; no GP contact required unless they worsen.
- "contact_doctor": the carer should call the GP or consultant within 24 hours. ALWAYS use this (never "monitor") when the report includes ANY of:
  • Elevated blood pressure: systolic above 150 OR diastolic above 90
  • Persistent confusion lasting more than a few hours
  • Wound that looks infected, is leaking, has pus, or unusual discharge
  • Significant or increasing swelling after surgery
  • Fever above 38°C (100.4°F)
  • Any symptom that is worsening over time
  Cross-reference PATIENT STORED CONDITIONS: if a symptom is more dangerous because of ANY stored chronic condition (e.g. fever with diabetes, COPD, or kidney disease), ALWAYS classify as "contact_doctor" or "emergency" — never "monitor".
- "emergency": life-threatening — call 999/112 immediately (chest pain, stroke signs, severe breathing difficulty, anaphylaxis with breathing difficulty or facial/throat swelling, etc.).

ALLERGY / REACTION SEVERITY:
- A documented allergy on file does NOT by itself mean emergency.
- Mild or localized rash, hives, or itching (no breathing difficulty, no facial/throat swelling, no fainting) → "contact_doctor", never "emergency".
- Use "emergency" for allergic reactions only when the caregiver's report describes red flags: difficulty breathing, lip/tongue/throat/facial swelling, wheeze, fainting, collapse, or rapidly spreading whole-body reaction.

DYNAMIC CONDITION CROSS-CHECK:
- CareShield evaluates every new symptom against the patient's full stored condition list using medical knowledge — not a hardcoded rule table.
- When a chronic condition makes the symptom more dangerous, increase urgency and explain why that condition changes the risk.

Set "needs_doctor": true when severity is "contact_doctor"."""

CHAT_INFORMATIONAL_QUESTION_RULES = """INFORMATIONAL QUESTIONS — no urgency tag unless the caregiver reports a concern:
- When the caregiver asks a general medication or care question WITHOUT reporting a new or worsening symptom, use severity "ok" and needs_doctor false.
- Examples that stay "ok": "How is his Furosemide working?", "Can he take Metformin and Lisinopril together?", "What is Furosemide for?"
- Only use monitor/contact_doctor/emergency when the caregiver's own words report or ask about a symptom, side effect, or concern (e.g. swelling, dizziness, not working, getting worse).
- Do NOT raise severity because your answer mentions possible side effects or stored conditions — classify based on what the caregiver actually said."""

CHAT_SESSION_RULES = """SESSION MEMORY — this is an ongoing conversation, not a single isolated message:
- Review PRIOR SESSION REPORTS below and connect related symptoms across messages from THIS browser session only.
- If PRIOR SESSION REPORTS says "None yet this session", do NOT reference earlier reports, linked reports, prior incidents, or counts of linked reports — reason only from the current message plus stored medications/conditions/timeline.
- Never invent or guess timestamps, falls, diagnoses, or prior incidents that are not explicitly listed in the context blocks below.
- If an earlier report mentions confusion and a new report mentions a fall (or the reverse), explicitly link them, increase urgency, and explain why the combination matters.
- Recurrent or worsening incidents across the session should increase urgency — never treat a new report in isolation when prior session reports exist.
- Briefly remind the carer that every incident is timestamped, stored for the session, and available to download in the Handover tab.
- When you spot a connection, name the earlier report and its time (e.g. "This follows your 10:15 report about confusion")."""

CHAT_SYMPTOM_PERSONALIZATION_RULES = """PERSONALIZATION — mandatory on every symptom report and care question:
- The patient's conditions, medications, allergies, documents, and timeline are on file above. NEVER give a generic answer when this data exists.
- Name the patient's specific diagnoses and medicines when they help explain the reported symptom.
- For rash, hives, swelling, breathing difficulty, or skin reactions: always check PATIENT ALLERGIES and ask about any recently introduced medicine or substance.
- For dizziness, fatigue, or other possible side effects: flag recently started or changed medicines from the medication list and notes.
- Explain how the symptom may or may not relate to their known conditions, medications, or allergies whenever appropriate.
- If PRE-COMPUTED SYMPTOM–PATIENT CROSS-CHECK is provided, weave those links into empathetic_advice in warm, plain language.
- Examples: swelling + heart failure on file → mention heart failure; dizziness + recently started ACE inhibitor → mention the medicine; rash + penicillin allergy on file → mention the allergy."""

CHAT_SINGLE_PATIENT_RULES = """SINGLE-PATIENT FAMILY ACCOUNT — critical:
- This CareShield account is for ONE patient only. They are always registered and on file for this family.
- ACTIVE PATIENT NAME is shown in the patient context block below. ALWAYS refer to the patient using that exact name in every response.
- NEVER use a different personal name from the caregiver's message, even if they mistype, mention another family member, or use the wrong name by accident.
- The caregiver may say "my son", "my dad", "Mum", or "the patient" — they always mean the same patient whose medications and conditions are listed below.
- NEVER refuse medication, dosing, or schedule questions by claiming the patient is not registered, not in the system, or that you cannot advise on their medications.
- When asked what pills to give and when, answer directly from PATIENT STORED MEDICATIONS with specific drug names and times.
- If one detail is missing from the plan, say what IS on file and suggest checking the Documents tab — do not refuse the whole question.
- If the caregiver asks about a surgery, procedure, or diagnosis that is NOT in stored conditions, documents, or timeline, say clearly that CareShield has no record of it — never answer as if confirmed."""

SESSION_HANDOVER_NOTE = (
    "\n\n**Session log:** This update is timestamped and saved. "
    "Download the full handover in the **Handover** tab. "
    "CareShield links related symptoms across your session — repeated or connected incidents are reviewed together."
)

CHAT_UNSUPPORTED_PHOTO_DEFAULT = (
    'This chat only accepts photos for symptom review (such as a bruise, rash, swelling, or wound). '
    "Please upload a symptom photo — not food, selfies, documents, or other images. "
    "For full dose verification, use the **MedCam** tab."
)


def has_stored_condition_keywords(*keywords: str) -> bool:
    for condition in get_stored_conditions():
        name = str(condition.get("name", "")).lower()
        if any(keyword in name for keyword in keywords):
            return True
    return False


def parse_blood_pressure(text: str) -> tuple[int | None, int | None]:
    lower = text.lower()
    patterns = [
        r"(?:blood pressure|bp)\s*(?:of|is|was|at|reading)?\s*(\d{2,3})\s*/\s*(\d{2,3})",
        r"(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mmhg|mm hg)?",
        r"(\d{2,3})\s*over\s*(\d{2,3})",
    ]
    for pattern in patterns:
        match = re.search(pattern, lower)
        if match:
            systolic = int(match.group(1))
            diastolic = int(match.group(2))
            if 70 <= systolic <= 250 and 40 <= diastolic <= 150:
                return systolic, diastolic
    single = re.search(
        r"(?:systolic|blood pressure|bp)\s*(?:of|is|was|at)?\s*(\d{2,3})",
        lower,
    )
    if single:
        systolic = int(single.group(1))
        if 70 <= systolic <= 250:
            return systolic, None
    return None, None


def detect_elevated_blood_pressure(text: str) -> bool:
    systolic, diastolic = parse_blood_pressure(text)
    if systolic is not None and systolic > 150:
        return True
    if diastolic is not None and diastolic > 90:
        return True
    return bool(re.search(r"\b(?:high|elevated|raised)\s+blood pressure\b", text, re.I))


def detect_contact_doctor_triggers(text: str) -> list[str]:
    if not text or not text.strip():
        return []

    triggers = []
    lower = text.lower()

    if detect_elevated_blood_pressure(text):
        triggers.append("elevated blood pressure")
    elif has_stored_condition_keywords("hypertension", "high blood pressure") and re.search(
        r"\b(?:bp|blood pressure|high bp|high blood pressure)\b", lower
    ):
        triggers.append("elevated blood pressure with stored Hypertension")

    if re.search(r"confus", lower) and re.search(
        r"(?:persist|several|few|many|couple|hours|all day|since (?:this )?morning|not (?:improving|better))",
        lower,
    ):
        triggers.append("persistent confusion")

    if re.search(
        r"(?:infect|leaking|pus|discharge|seeping|oozing|redness.*wound|wound.*(?:red|hot|swollen))",
        lower,
    ):
        triggers.append("infected or leaking wound")

    if re.search(r"(?:swell|swollen|swelling)", lower) and re.search(
        r"(?:surgery|surgical|post[- ]?op|operat|incision|knee|hip|wound)",
        lower,
    ):
        triggers.append("post-surgical swelling")
    elif re.search(
        r"(?:increasing|worsening|getting worse|more swollen).{0,40}swell|swell.{0,40}(?:increasing|worsening|getting worse)",
        lower,
    ):
        triggers.append("increasing swelling")

    for match in re.finditer(
        r"(?:fever|temp(?:erature)?)\s*(?:of|at|is|was|:)?\s*(\d{2}(?:\.\d)?)\s*(?:°?\s*c|celsius)?",
        lower,
    ):
        try:
            if float(match.group(1)) > 38:
                triggers.append("fever above 38°C")
                break
        except ValueError:
            continue
    if "fever above 38°C" not in triggers and re.search(
        r"(?:fever|temperature).{0,30}(?:high|elevated|38|39|40)", lower
    ):
        if re.search(r"3[89]|4[0-9]", lower):
            triggers.append("fever above 38°C")

    if re.search(
        r"(?:worsening|getting worse|deteriorat|worse than|more severe|not improving|declining)",
        lower,
    ):
        triggers.append("worsening symptoms")

    return triggers


def apply_contact_doctor_escalation(
    severity: str,
    user_text: str,
    *,
    needs_doctor: bool = False,
) -> str:
    level = normalize_chat_severity(severity)
    if level == "emergency":
        return level
    if needs_doctor or detect_contact_doctor_triggers(user_text):
        return "contact_doctor"
    return level


def format_chat_timestamp(dt=None, tz_name=None) -> str:
    tz_name = tz_name or (user_timezone if "user_timezone" in globals() else None)
    try:
        if tz_name:
            tz_obj = ZoneInfo(tz_name)
            if dt is None:
                moment = datetime.now(tz_obj)
            elif isinstance(dt, str):
                moment = datetime.fromisoformat(dt.replace("Z", "+00:00"))
                if moment.tzinfo is None:
                    moment = moment.replace(tzinfo=timezone.utc)
                moment = moment.astimezone(tz_obj)
            else:
                moment = dt.astimezone(tz_obj) if dt.tzinfo else dt.replace(tzinfo=timezone.utc).astimezone(tz_obj)
            return moment.strftime("%a %d %b, %I:%M %p").lstrip("0")
    except Exception:
        pass
    if isinstance(dt, str):
        try:
            moment = datetime.fromisoformat(dt.replace("Z", "+00:00"))
            return moment.strftime("%a %d %b, %I:%M %p").lstrip("0")
        except (ValueError, TypeError):
            return dt
    return datetime.now().strftime("%a %d %b, %I:%M %p").lstrip("0")


def render_history_show_more_controls(
    *,
    hidden_count: int,
    total: int,
    shown_count: int,
    session_key: str,
    show_more_key: str,
    batch: int = HANDOVER_TIMELINE_EXPAND_BATCH,
    default_visible: int = HANDOVER_TIMELINE_DEFAULT_VISIBLE,
) -> None:
    """Display-only pagination controls for report history lists."""
    if total == 0:
        return
    if hidden_count == 0:
        if total > default_visible:
            st.caption("All reports shown.")
        return
    next_batch = min(batch, hidden_count)
    if st.button(f"Show {next_batch} more", key=show_more_key, use_container_width=True):
        st.session_state[session_key] = shown_count + next_batch
        st.rerun()


def get_session_incidents_key(patient_id=None) -> str:
    resolved = resolve_patient_id(
        patient_id if patient_id is not None else st.session_state.get("selected_patient_id")
    )
    return f"session_incidents_{resolved}"


def get_session_incidents(patient_id=None) -> list:
    key = get_session_incidents_key(patient_id)
    incidents = list(st.session_state.get(key, []))
    resolved_id = resolve_patient_id(
        patient_id if patient_id is not None else st.session_state.get("selected_patient_id")
    )
    if is_designated_test_patient(resolved_id, get_patient_by_id(resolved_id)):
        return incidents
    filtered = [
        incident for incident in incidents
        if not care_row_is_internal_test(incident)
        and session_incident_is_valid_for_patient(incident, resolved_id)
    ]
    if len(filtered) != len(incidents):
        st.session_state[key] = filtered
    return filtered


def clear_session_patient_state(previous_patient_id: str, caregiver_label: str | None = None) -> None:
    """Drop in-memory state for the previous patient when switching."""
    prev_key = get_session_incidents_key(previous_patient_id)
    st.session_state.pop(prev_key, None)
    invalidate_patient_activity_cache(previous_patient_id)
    st.session_state.pop(f"stored_conditions_{previous_patient_id}", None)
    st.session_state.pop(f"stored_medications_{previous_patient_id}", None)
    st.session_state.pop(f"med_plan_meta_{previous_patient_id}", None)
    st.session_state.pop(my_results_cache_key(previous_patient_id), None)
    st.session_state.pop(my_results_processed_key(previous_patient_id), None)
    st.session_state.pop(my_results_error_key(previous_patient_id), None)
    st.session_state.pop(my_results_error_debug_key(previous_patient_id), None)
    st.session_state.pop(my_results_feedback_key(previous_patient_id), None)
    st.session_state.pop("my_results_error", None)
    st.session_state.pop("my_results_error_debug", None)
    st.session_state.pop("my_results_pending_upload", None)
    st.session_state.pop("last_sbar_result", None)
    st.session_state.pop("pending_chat_response", None)
    st.session_state.pop("pill_modal", None)
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith("care_hydrated_"):
            st.session_state.pop(key, None)
    st.session_state.pop("messages", None)
    if caregiver_label:
        st.session_state.reset_chat_draft = True


def format_patient_label(patient: dict) -> str:
    label = patient.get("display_name") or patient.get("name") or "Patient"
    return str(label).strip() or "Patient"


def patient_avatar_initial(patient: dict) -> str:
    if patient.get("initial"):
        return str(patient["initial"]).strip()[0].upper()
    source = str(patient.get("display_name") or patient.get("name") or "Patient").strip()
    return source[0].upper() if source else "P"


def build_patient_avatars_html(patients: list[dict], selected_id: str) -> str:
    avatar_items = []
    for patient in patients:
        initial = html.escape(patient_avatar_initial(patient))
        active_class = " active" if str(patient.get("id")) == str(selected_id) else ""
        avatar_items.append(f'<div class="cs-avatar{active_class}">{initial}</div>')
    return "".join(avatar_items)


@st.dialog("Add patient")
def add_patient_dialog() -> None:
    st.markdown("Add someone you care for. Their medications, documents, and handover notes stay separate.")
    st.text_input(
        "Patient name",
        placeholder="Dad, Mum, Pedro...",
        key="add_patient_name",
    )
    save_col, cancel_col = st.columns(2)
    with save_col:
        if st.button("Save patient", type="primary", use_container_width=True, key="add_patient_save"):
            name = str(st.session_state.get("add_patient_name", "")).strip()
            patient, error = create_patient(name)
            if patient is None:
                st.error(error or "Could not save patient.")
            else:
                previous_id = st.session_state.get("selected_patient_id")
                caregiver_label = get_caregiver_profile_label(
                    st.session_state.get("selected_caregiver_id", "")
                )
                if previous_id:
                    clear_session_patient_state(str(previous_id), caregiver_label)
                new_patient_id = str(patient["id"])
                st.session_state.selected_patient_id = new_patient_id
                invalidate_patient_activity_cache(new_patient_id)
                hydrate_patient_care_session(new_patient_id, caregiver_label)
                st.session_state.pop("open_add_patient_dialog", None)
                st.session_state.pop("patient_profile_picker", None)
                st.session_state.pop("add_patient_name", None)
                st.rerun()
    with cancel_col:
        if st.button("Cancel", use_container_width=True, key="add_patient_cancel"):
            st.session_state.pop("open_add_patient_dialog", None)
            st.session_state.pop("add_patient_name", None)
            st.rerun()


def init_selected_patient() -> str:
    """Ensure session has a selected patient id before tabs render."""
    if st.session_state.get("selected_patient_id"):
        return str(st.session_state.selected_patient_id)
    patient_id = get_or_create_default_patient()
    if patient_id:
        st.session_state.selected_patient_id = patient_id
        return patient_id
    return ""


def render_patient_selector() -> str:
    """Patient picker — mirrors the caregiver profile switcher, backed by Supabase."""
    init_selected_patient()
    patients = list_account_patients()
    if not patients:
        st.warning("Could not load patients. Check your Supabase connection.")
        return str(st.session_state.get("selected_patient_id") or "")

    selected_id = str(st.session_state.selected_patient_id)
    labels = [format_patient_label(patient) for patient in patients]
    ids = [str(patient["id"]) for patient in patients]

    if selected_id not in ids:
        selected_id = ids[0]
        st.session_state.selected_patient_id = selected_id

    current_label = labels[ids.index(selected_id)]
    md_html(f"""
    <div class="cs-user-area cs-user-area-right">
      <div class="cs-user-label">Currently caring for</div>
      <div class="cs-avatars cs-avatars-right">{build_patient_avatars_html(patients, selected_id)}</div>
    </div>
    """)

    picker_col, manage_col = st.columns([5, 1])
    with picker_col:
        picked = st.selectbox(
            "Currently caring for",
            options=labels + [ADD_NEW_PATIENT_OPTION],
            index=labels.index(current_label) if current_label in labels else 0,
            label_visibility="collapsed",
            key="patient_profile_picker",
        )
    with manage_col:
        with st.popover("", help="Manage patients", icon=":material/edit:"):
            render_patient_profile_manager()

    if picked == ADD_NEW_PATIENT_OPTION:
        st.session_state.open_add_patient_dialog = True
        st.session_state.pop("patient_profile_picker", None)
        st.rerun()
    elif picked in labels:
        picked_id = ids[labels.index(picked)]
        if picked_id != selected_id:
            caregiver_label = get_caregiver_profile_label(st.session_state.get("selected_caregiver_id", ""))
            clear_session_patient_state(selected_id, caregiver_label)
            st.session_state.selected_patient_id = picked_id
            hydrate_patient_care_session(picked_id, caregiver_label)
            st.session_state.pop("patient_profile_picker", None)
            st.rerun()

    if st.session_state.pop("open_add_patient_dialog", False):
        add_patient_dialog()

    if st.session_state.get("show_edit_patient_dialog") and st.session_state.get("editing_patient_id"):
        edit_patient_dialog(st.session_state.editing_patient_id)

    render_responsible_ai_safety_button()

    return str(st.session_state.selected_patient_id)


@st.dialog("Edit patient")
def edit_patient_dialog(patient_id: str) -> None:
    patient = get_patient_by_id(patient_id)
    if not patient:
        st.error("Patient not found.")
        return
    st.markdown("Update how this patient appears across CareShield.")
    name = st.text_input(
        "Patient name",
        value=str(patient.get("display_name") or patient.get("name") or ""),
        key=f"edit_patient_name_{patient_id}",
    )
    save_col, cancel_col = st.columns(2)
    with save_col:
        if st.button("Save changes", type="primary", use_container_width=True, key=f"edit_patient_save_{patient_id}"):
            success, error = update_patient(patient_id, name)
            if success:
                st.session_state.pop("show_edit_patient_dialog", None)
                st.session_state.pop("editing_patient_id", None)
                st.session_state.pop("patient_profile_picker", None)
                st.rerun()
            else:
                st.error(error or "Could not update patient.")
    with cancel_col:
        if st.button("Cancel", use_container_width=True, key=f"edit_patient_cancel_{patient_id}"):
            st.session_state.pop("show_edit_patient_dialog", None)
            st.session_state.pop("editing_patient_id", None)
            st.rerun()


def render_patient_profile_manager() -> None:
    patients = list_account_patients()
    selected_id = str(st.session_state.get("selected_patient_id", ""))
    for patient in patients:
        label = format_patient_label(patient)
        patient_id = str(patient["id"])
        edit_clicked, delete_clicked = render_profile_manager_row(
            label,
            patient_id == selected_id,
            edit_key=f"patient_edit_{patient_id}",
            delete_key=f"patient_delete_{patient_id}",
        )
        if edit_clicked:
            st.session_state.editing_patient_id = patient_id
            st.session_state.show_edit_patient_dialog = True
            st.rerun()
        if delete_clicked:
            if len(patients) <= 1:
                st.error("You need at least one patient profile.")
            else:
                success, error = delete_patient(patient_id)
                if not success:
                    st.error(error or "Could not delete patient.")
                else:
                    clear_session_patient_state(patient_id)
                    if selected_id == patient_id:
                        remaining = [
                            row for row in list_account_patients()
                            if str(row["id"]) != patient_id
                        ]
                        if remaining:
                            st.session_state.selected_patient_id = str(remaining[0]["id"])
                        else:
                            st.session_state.pop("selected_patient_id", None)
                    st.session_state.pop("patient_profile_picker", None)
                    st.rerun()
    st.markdown("---")
    if st.button(ADD_NEW_PATIENT_OPTION, key="patient_add_from_manager", use_container_width=True):
        st.session_state.open_add_patient_dialog = True
        st.rerun()


def get_current_patient_id() -> str:
    return resolve_patient_id(st.session_state.get("selected_patient_id"))


def get_stored_conditions(patient_id=None):
    patient_id = resolve_patient_id(patient_id or st.session_state.get("selected_patient_id"))
    if not patient_id:
        return []
    cache_key = f"stored_conditions_{patient_id}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = get_patient_conditions(patient_id)
    return st.session_state[cache_key]


def get_stored_medications_display(patient_id=None):
    patient_id = resolve_patient_id(patient_id or st.session_state.get("selected_patient_id"))
    if not patient_id:
        return []
    cache_key = f"stored_medications_{patient_id}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = get_patient_medications_display(patient_id)
    return st.session_state[cache_key]


def warm_patient_profile_cache(patient_id=None) -> None:
    """Prefetch profile data so the Documents tab can render without waiting on other tabs."""
    patient_id = resolve_patient_id(patient_id or st.session_state.get("selected_patient_id"))
    if not patient_id:
        return
    warmed_key = f"profile_cache_warmed_{patient_id}"
    if st.session_state.get(warmed_key):
        return
    get_stored_conditions(patient_id)
    get_stored_medications_display(patient_id)
    get_patient_plan_meta(patient_id)
    st.session_state[warmed_key] = True


def invalidate_patient_activity_cache(patient_id=None) -> None:
    """Drop cached shift logs, dose logs, timelines, and PDFs after new activity is saved."""
    patient_id = resolve_patient_id(patient_id or st.session_state.get("selected_patient_id"))
    if not patient_id:
        return
    prefixes = (
        f"shift_logs_{patient_id}_",
        f"medication_logs_{patient_id}",
        f"symptom_timeline_{patient_id}",
        f"care_reports_timeline_{patient_id}",
        f"symptom_timeline_db_{patient_id}",
        f"adherence_timeline_{patient_id}",
        f"patient_plans_{patient_id}",
        f"medication_refs_{patient_id}",
        f"medcam_audit_{patient_id}",
        f"my_results_pdf_{patient_id}_",
        "handover_pdf_",
    )
    for key in list(st.session_state.keys()):
        if not isinstance(key, str):
            continue
        if key == f"profile_cache_warmed_{patient_id}":
            st.session_state.pop(key, None)
            continue
        if any(key.startswith(prefix) for prefix in prefixes):
            st.session_state.pop(key, None)


def cached_shift_logs(patient_id=None, limit: int = 250) -> list:
    patient_id = resolve_patient_id(patient_id or st.session_state.get("selected_patient_id"))
    cache_key = f"shift_logs_{patient_id}_{limit}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = fetch_shift_logs(patient_id, limit=limit)
    return st.session_state[cache_key]


def cached_medication_logs(patient_id=None) -> list:
    patient_id = resolve_patient_id(patient_id or st.session_state.get("selected_patient_id"))
    cache_key = f"medication_logs_{patient_id}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = get_medication_logs(patient_id)
    return st.session_state[cache_key]


def cached_patient_plans(patient_id=None) -> list:
    from ai_helpers import get_patient_plans

    patient_id = resolve_patient_id(patient_id or st.session_state.get("selected_patient_id"))
    cache_key = f"patient_plans_{patient_id}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = get_patient_plans(patient_id)
    return st.session_state[cache_key]


def cached_medication_references(patient_id=None) -> list:
    patient_id = resolve_patient_id(patient_id or st.session_state.get("selected_patient_id"))
    cache_key = f"medication_refs_{patient_id}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = get_medication_references(patient_id)
    return st.session_state[cache_key]


def get_patient_plan_meta(patient_id=None) -> dict:
    """Discontinued meds and manual-review flags scoped to one patient."""
    patient_id = resolve_patient_id(patient_id or st.session_state.get("selected_patient_id"))
    if not patient_id:
        return {"discontinued": [], "review_flags": []}
    cache_key = f"med_plan_meta_{patient_id}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    latest_plan = get_latest_patient_plan(patient_id)
    if latest_plan:
        envelope = load_medication_plan(latest_plan.get("medications"))
        meta = {
            "discontinued": envelope.get("discontinued", []),
            "review_flags": envelope.get("review_flags", []),
        }
        st.session_state[cache_key] = meta
        return meta
    return {"discontinued": [], "review_flags": []}


def render_no_patient_medications_guidance() -> None:
    md_html("""
    <div class="cs-empty-plan-guide">
      <p class="cs-empty-plan-guide-lead">
        You haven&rsquo;t uploaded any document about this patient&rsquo;s medications yet.
      </p>
      <p>
        First go to the <strong>Documents</strong> section and upload their health documents
        so we can understand their conditions and medications.
      </p>
      <p>
        Then go to <strong>Pill Registration</strong> to register the pills, and
        <strong>MedCam</strong> to verify them before the patient takes them.
      </p>
    </div>
    """)


def set_stored_conditions(conditions: list, patient_id=None) -> None:
    patient_id = resolve_patient_id(patient_id or st.session_state.get("selected_patient_id"))
    cache_key = f"stored_conditions_{patient_id}"
    st.session_state[cache_key] = conditions
    replace_patient_conditions(patient_id, conditions)


def record_session_incident(
    *,
    text: str,
    severity: str,
    timestamp: str,
    timestamp_display: str,
    caregiver: str,
    caregiver_id: str | None = None,
    source: str,
    summary: str = "",
    image_b64: str = "",
    photo_finding: str = "",
    photo_type: str = "",
) -> None:
    resolved_id = caregiver_id or resolve_caregiver_id(caregiver)
    patient_id = resolve_patient_id(st.session_state.get("selected_patient_id"))
    summary_for_log = str(summary or text or "").strip()
    caregiver_label = resolve_caregiver_label(resolved_id) or caregiver
    if should_block_test_entry_for_patient(
        patient_id,
        text=text,
        summary=summary_for_log,
        caregiver_name=caregiver_label,
        source=source,
        patient=get_patient_by_id(patient_id),
    ):
        st.warning(
            "That looks like internal QA/test data. Create or select a patient named "
            '"[TEST] …" (or set CARESHIELD_TEST_PATIENT_IDS) before running persistence checks.'
        )
        return

    incidents = st.session_state.setdefault(get_session_incidents_key(patient_id), [])
    entry = {
        "text": text,
        "severity": normalize_chat_severity(severity),
        "timestamp": timestamp,
        "timestamp_display": timestamp_display,
        "caregiver": resolve_caregiver_label(resolved_id),
        "caregiver_id": resolved_id,
        "source": source,
        "summary": summary or text,
        "symptoms": extract_symptoms_from_text(text),
    }
    if image_b64:
        entry["image_b64"] = image_b64
    if photo_finding:
        entry["photo_finding"] = photo_finding
    if photo_type:
        entry["photo_type"] = photo_type
    incidents.append(entry)

    summary_for_log = str(entry.get("summary") or text or "").strip()
    if summary_for_log and patient_id:
        saved_report = save_patient_care_report(
            patient_id,
            report_text=text,
            summary=summary_for_log,
            severity=entry["severity"],
            source=source,
            reported_at=timestamp,
            caregiver_name=resolve_caregiver_label(resolved_id) or caregiver,
            caregiver_id=resolved_id,
            photo_finding=photo_finding,
            photo_type=photo_type,
            image_b64=image_b64,
        )
        if saved_report and saved_report.get("id") is not None:
            entry["report_id"] = saved_report["id"]
        if save_shift_log(
            caregiver_name=resolve_caregiver_label(resolved_id) or caregiver,
            source=source,
            summary=summary_for_log,
            severity=entry["severity"],
            reported_at=timestamp,
            caregiver_id=resolved_id,
            patient_id=patient_id,
        ):
            invalidate_patient_activity_cache(patient_id)
        st.session_state.pop(f"symptom_timeline_db_{patient_id}", None)
        st.session_state.pop(f"care_reports_timeline_{patient_id}", None)


def care_report_row_to_session_incident(row: dict) -> dict:
    text = str(row.get("report_text") or row.get("summary") or "").strip()
    image_b64 = ""
    if row.get("has_photo") and row.get("id") is not None:
        image_b64 = fetch_patient_care_report_photo(
            row["id"],
            patient_id=row.get("patient_id"),
        )
    return {
        "text": text,
        "severity": normalize_chat_severity(row.get("severity", "monitor")),
        "timestamp": row.get("reported_at") or row.get("created_at") or "",
        "timestamp_display": format_chat_timestamp(row.get("reported_at") or row.get("created_at")),
        "caregiver": row.get("caregiver_name") or resolve_caregiver_label(row.get("caregiver_id")),
        "caregiver_id": row.get("caregiver_id"),
        "source": row.get("source", "voice_report"),
        "summary": str(row.get("summary") or text),
        "symptoms": extract_symptoms_from_text(text),
        "image_b64": image_b64,
        "photo_finding": row.get("photo_finding") or "",
        "photo_type": row.get("photo_type") or "",
    }


def care_report_row_to_timeline_event(row: dict) -> dict:
    text = str(row.get("report_text") or row.get("summary") or "").strip()
    reported_at = row.get("reported_at") or row.get("created_at") or ""
    return {
        "timestamp": reported_at,
        "timestamp_display": format_chat_timestamp(reported_at),
        "text": text,
        "severity": normalize_chat_severity(row.get("severity", "monitor")),
        "caregiver": row.get("caregiver_name") or resolve_caregiver_label(row.get("caregiver_id")),
        "source": row.get("source", "voice_report"),
        "symptoms": extract_symptoms_from_text(text),
        "report_id": row.get("id"),
        "has_photo": bool(row.get("has_photo")),
        "photo_finding": row.get("photo_finding") or "",
        "photo_type": row.get("photo_type") or "",
    }


def persist_patient_chat_thread(patient_id=None) -> None:
    patient_id = resolve_patient_id(patient_id or st.session_state.get("selected_patient_id"))
    if patient_id and st.session_state.get("messages"):
        save_patient_chat_thread(patient_id, st.session_state.messages)


def hydrate_patient_care_session(patient_id, caregiver_label: str) -> None:
    """Reload durable Report & Ask history and incidents for the active patient."""
    patient_id = resolve_patient_id(patient_id)
    if not patient_id:
        return

    purge_internal_test_patient_artifacts(patient_id)
    removed_suspect = purge_suspect_cross_profile_care_reports(patient_id)
    removed_chat = purge_suspect_cross_profile_chat_messages(patient_id)
    if removed_suspect or removed_chat:
        _report_ask_logger.info(
            "Purged polluted profile data patient=%s care_reports=%d chat_messages=%d",
            patient_id,
            removed_suspect,
            removed_chat,
        )
    st.session_state.pop(f"care_reports_timeline_{patient_id}", None)
    st.session_state.pop(f"symptom_timeline_db_{patient_id}", None)

    multi_patient_account = account_is_multi_patient(patient_id)

    reports = fetch_patient_care_reports(patient_id, limit=500)
    if not reports and not multi_patient_account:
        backfill_legacy_shift_logs_to_local_care(patient_id, limit=500)
        reports = fetch_patient_care_reports(patient_id, limit=500)
    if not reports and not multi_patient_account:
        for row in fetch_symptom_shift_logs(patient_id, limit=500):
            if not shift_log_belongs_to_patient(row, patient_id):
                continue
            summary = strip_shift_log_patient_marker(str(row.get("summary") or "").strip())
            if not summary:
                continue
            saved = save_patient_care_report(
                patient_id,
                report_text=summary,
                summary=summary,
                severity=row.get("severity", "monitor"),
                source="legacy_backfill",
                reported_at=row.get("created_at"),
                caregiver_name=row.get("caregiver_name", "Caregiver"),
                caregiver_id=row.get("caregiver_id"),
            )
            if saved:
                reports.append(saved)

    st.session_state[get_session_incidents_key(patient_id)] = []

    # Fresh visible chat on each page load; durable history stays in care_reports + chat_thread.
    st.session_state.messages = build_initial_chat_messages(caregiver_label)
    st.session_state.chat_caregiver = caregiver_label
    st.session_state.chat_caregiver_id = resolve_caregiver_id(
        st.session_state.get("selected_caregiver_id") or caregiver_label
    )
    st.session_state.report_ask_bound_caregiver_id = st.session_state.chat_caregiver_id

    st.session_state.pop(f"care_reports_timeline_{patient_id}", None)
    st.session_state.pop(f"symptom_timeline_db_{patient_id}", None)
    st.session_state[care_hydrate_key(patient_id)] = True
    cleanup_orphaned_session_incidents(patient_id)


def cleanup_orphaned_session_incidents(patient_id=None) -> None:
    """Drop stray in-memory incident buckets — never merge across profiles."""
    patient_id = resolve_patient_id(patient_id or st.session_state.get("selected_patient_id"))
    if not patient_id:
        return
    target_key = get_session_incidents_key(patient_id)

    for key in list(st.session_state.keys()):
        if not isinstance(key, str) or not key.startswith("session_incidents_"):
            continue
        if key == target_key:
            continue
        st.session_state.pop(key, None)


def migrate_legacy_session_incidents(patient_id=None) -> None:
    """Backward-compatible alias — safe cleanup only, no cross-patient merge when multi-patient."""
    cleanup_orphaned_session_incidents(patient_id)


def load_symptom_events_from_chat_messages(patient_id=None) -> list:
    """Rebuild symptom rows from Report & Ask chat history when incidents were not linked."""
    patient_id = resolve_patient_id(patient_id or st.session_state.get("selected_patient_id"))
    caregiver_id = (
        st.session_state.get("chat_caregiver_id")
        or st.session_state.get("selected_caregiver_id")
    )
    caregiver = (
        resolve_caregiver_label(caregiver_id)
        or st.session_state.get("chat_caregiver")
        or "Caregiver"
    )
    events = []
    messages = st.session_state.get("messages", [])
    for index, message in enumerate(messages):
        if message.get("role") != "user":
            continue
        text = str(message.get("content") or "").strip()
        if not text or text == "Photo for review":
            continue
        timestamp = message.get("timestamp") or ""
        if not timestamp:
            continue
        message_caregiver_id = message.get("caregiver_id") or caregiver_id
        message_caregiver = (
            resolve_caregiver_label(message_caregiver_id)
            if message_caregiver_id
            else caregiver
        )
        severity = "monitor"
        if index + 1 < len(messages) and messages[index + 1].get("role") == "assistant":
            severity = messages[index + 1].get("severity", "monitor")
        source = "care_question" if is_question(text) else "voice_report"
        events.append({
            "timestamp": timestamp,
            "timestamp_display": message.get("timestamp_display") or format_chat_timestamp(timestamp),
            "text": text,
            "severity": normalize_chat_severity(severity),
            "caregiver": message_caregiver,
            "caregiver_id": message_caregiver_id,
            "source": source,
            "symptoms": extract_symptoms_from_text(text),
        })
    return events


def detect_recurring_symptom_patterns(incidents: list | None = None, min_count: int = 3) -> list[dict]:
    incidents = incidents or get_session_incidents()
    counts: dict[str, int] = {}
    labels = {key: label for key, _pattern, label in SYMPTOM_PATTERN_DEFINITIONS}
    for incident in incidents:
        for key in incident.get("symptoms") or extract_symptoms_from_text(incident.get("text", "")):
            counts[key] = counts.get(key, 0) + 1
    return [
        {"key": key, "label": labels.get(key, key.replace("_", " ").title()), "count": count}
        for key, count in counts.items()
        if count >= min_count
    ]


def build_condition_risk_alerts_html(condition_risks: list | None) -> str:
    risks = condition_risks or []
    if not risks:
        return ""
    blocks = []
    for risk in risks:
        message = str(risk.get("education_message") or "").strip()
        if not message:
            continue
        match = re.match(r"How (.+?) Impacts This Symptom:\s*(.+)", message, re.S | re.I)
        if not match:
            match = re.match(r"Why (.+?) Changes the Game:\s*(.+)", message, re.S | re.I)
        if match:
            condition = html.escape(match.group(1).strip())
            body = html.escape(match.group(2).strip())
            blocks.append(
                f'<div class="cs-condition-risk-alert">'
                f'<div class="cs-condition-risk-title">How {condition} Impacts This Symptom</div>'
                f'<div class="cs-condition-risk-body">{body}</div>'
                f"</div>"
            )
        else:
            blocks.append(
                f'<div class="cs-condition-risk-alert">'
                f'<div class="cs-condition-risk-body">{html.escape(message)}</div>'
                f"</div>"
            )
    return "".join(blocks)


def merge_symptom_condition_analysis(
    severity: str,
    reply: str,
    analysis: dict | None,
    *,
    needs_doctor: bool = False,
) -> tuple[str, str, list, bool]:
    """Apply stored-condition cross-check to severity and structured caregiver alerts."""
    if not analysis:
        return severity, reply, [], needs_doctor

    relevant = extract_relevant_condition_risks(analysis)
    if analysis.get("is_elevated_risk"):
        severity = escalate_severity(severity, analysis.get("recommended_severity", severity))
        if analysis.get("needs_doctor"):
            needs_doctor = True

    return severity, reply, relevant, needs_doctor


def build_session_reports_context(patient_id=None) -> str:
    incidents = get_session_incidents(patient_id)
    if not incidents:
        return "PRIOR SESSION REPORTS: None yet this session."
    lines = ["PRIOR SESSION REPORTS (timestamped — connect related symptoms across these):"]
    for incident in incidents[-15:]:
        lines.append(
            f"- [{incident['timestamp_display']}] {incident['text']} "
            f"(severity: {incident.get('severity', 'monitor')})"
        )
    return "\n".join(lines)


def build_session_connection_note(
    prior_incidents: list,
    session_triggers: list[str],
    current_text: str = "",
    patient_id=None,
) -> str:
    if not session_triggers or not prior_incidents or not current_text:
        return ""
    valid_priors = [
        item for item in prior_incidents
        if session_incident_is_valid_for_patient(item, patient_id)
    ]
    if not valid_priors:
        return ""
    recent = select_linked_prior_incidents(
        current_text,
        valid_priors,
        session_triggers=session_triggers,
    )
    if not recent:
        return ""
    links = "; ".join(
        f"{item.get('timestamp_display') or item.get('timestamp') or 'Earlier'} — {item['text'][:80]}"
        for item in recent
    )
    trigger_text = ", ".join(session_triggers)
    return (
        f"\n\n**Connected to earlier reports:** {links}. "
        f"Taken together ({trigger_text}), these updates suggest a higher level of concern."
    )


def resolve_chat_severity(
    severity: str,
    user_text: str,
    *,
    needs_doctor: bool = False,
    prior_incidents: list | None = None,
) -> tuple[str, list[str]]:
    if is_clearly_positive_benign_report(user_text):
        return "ok", []
    if is_pure_informational_care_question(user_text):
        return "ok", []

    prior = prior_incidents or []

    level = apply_contact_doctor_escalation(severity, user_text, needs_doctor=needs_doctor)
    if level != "emergency":
        if detect_contact_doctor_triggers(user_text) and level in ("ok", "monitor"):
            level = "contact_doctor"
        session_triggers = detect_session_escalation_triggers(user_text, prior)
        if session_triggers and level in ("ok", "monitor"):
            level = "contact_doctor"
        recurring = detect_recurring_symptom_patterns(
            prior + [{"text": user_text, "symptoms": extract_symptoms_from_text(user_text)}]
        )
        if recurring and level in ("ok", "monitor"):
            level = "contact_doctor"
            if not session_triggers:
                session_triggers = [
                    f"recurring {item['label'].lower()} ({item['count']} reports)"
                    for item in recurring
                ]
        has_confusion = bool(re.search(r"confus", user_text, re.I)) or any(
            re.search(r"confus", str(item.get("text") or item.get("summary") or ""), re.I)
            for item in prior
        )
        has_fall = bool(re.search(r"\bfell\b|\bfall\b|\bfallen\b", user_text, re.I)) or any(
            re.search(r"\bfell\b|\bfall\b|\bfallen\b", str(item.get("text") or item.get("summary") or ""), re.I)
            for item in prior
        )
        has_head_injury = bool(
            re.search(r"hit head|head injury|unconscious|passed out|won'?t wake", user_text, re.I)
        ) or any(
            re.search(r"hit head|head injury|unconscious|passed out|won'?t wake", str(item.get("text") or ""), re.I)
            for item in prior
        )
        current_contributes = bool(
            re.search(
                r"confus|\bfell\b|\bfall\b|\bfallen\b|hit head|head injury|unconscious|passed out|won'?t wake",
                user_text,
                re.I,
            )
        )
        if has_confusion and has_fall and has_head_injury and current_contributes:
            level = "emergency"
    else:
        session_triggers = detect_session_escalation_triggers(user_text, prior)

    return level, session_triggers


def severity_badge_html(severity: str) -> str:
    level = normalize_chat_severity(severity)
    labels = {
        "emergency": "EMERGENCY",
        "contact_doctor": "CONTACT DOCTOR",
        "monitor": "MONITOR",
        "ok": "OK",
    }
    label = labels.get(level, level.upper())
    return f'<div class="cs-msg-severity-banner cs-tag-{level}">{label}</div>'


def severity_header_html(severity: str, context_report_count: int | None = None) -> str:
    parts = [severity_badge_html(severity)]
    if context_report_count is not None:
        if context_report_count <= 1:
            confidence = "Based on 1 report"
        else:
            confidence = f"Based on {context_report_count} linked reports"
        parts.append(f'<div class="cs-confidence-indicator">{html.escape(confidence)}</div>')
    return "".join(parts)


def emergency_call_bar_html() -> str:
    return """
    <div class="cs-emergency-call-bar">
      <p class="cs-emergency-call-label">Need help right now? Tap to dial — no need to leave CareShield.</p>
      <div class="cs-emergency-call-actions">
        <a class="cs-emergency-call-btn" href="tel:999">Call 999</a>
        <a class="cs-emergency-call-btn cs-emergency-call-btn-alt" href="tel:112">Call 112</a>
      </div>
    </div>
    """


TIMELINE_SEVERITY_STYLES = {
    "ok": ("OK", "#D4EDDA", "#2D6A4F"),
    "monitor": ("MONITOR", "#FEF3C7", "#92400E"),
    "contact_doctor": ("CONTACT DOCTOR", "#FFEDD5", "#C2410C"),
    "emergency": ("EMERGENCY", "#FEE2E2", "#B91C1C"),
}

TIMELINE_SOURCE_LABELS = {
    "voice_report": "Report",
    "symptom_photo": "Symptom photo",
    "pill_photo": "Pill ID",
    "medication_check": "MedCam",
    "medication_log": "Dose log",
    "report": "Report",
}

ADHERENCE_TIMELINE_STYLES = {
    "taken": ("TAKEN", "#D4EDDA", "#2D6A4F"),
    "missed": ("MISSED", "#FEE2E2", "#B91C1C"),
    "check": ("MED CHECK", "#E8F0FE", "#1D4ED8"),
}


def _timeline_sort_key(event: dict) -> str:
    return str(event.get("timestamp") or "")


def _severity_rank(severity: str) -> int:
    from handover_events import SEVERITY_RANK, normalize_handover_severity
    return SEVERITY_RANK.get(normalize_handover_severity(severity), 0)


def _merge_timeline_duplicate(existing: dict, incoming: dict) -> dict:
    merged = dict(existing)
    for key in (
        "report_id",
        "image_b64",
        "photo_finding",
        "photo_type",
        "timestamp_display",
        "caregiver",
        "symptoms",
    ):
        if incoming.get(key):
            merged[key] = incoming[key]
    if _severity_rank(incoming.get("severity")) > _severity_rank(merged.get("severity")):
        merged["severity"] = incoming.get("severity")
    merged["has_photo"] = bool(
        merged.get("has_photo")
        or incoming.get("has_photo")
        or merged.get("image_b64")
        or incoming.get("image_b64")
        or str(merged.get("source") or "") in ("symptom_photo", "pill_photo")
        or str(incoming.get("source") or "") in ("symptom_photo", "pill_photo")
    )
    return merged


def _dedupe_timeline_events(events: list) -> list:
    merged_by_key: dict[tuple, dict] = {}
    order: list[tuple] = []
    for event in sorted(events, key=_timeline_sort_key):
        if not (event.get("text") or "").strip():
            continue
        dedupe_key = (
            str(event.get("timestamp") or "")[:19],
            (event.get("text") or "")[:100],
            event.get("source", ""),
        )
        existing = merged_by_key.get(dedupe_key)
        if existing:
            merged_by_key[dedupe_key] = _merge_timeline_duplicate(existing, event)
            continue
        merged_by_key[dedupe_key] = dict(event)
        order.append(dedupe_key)
    return [merged_by_key[key] for key in order]


def load_symptom_timeline_events(force_refresh: bool = False, patient_id=None) -> list:
    patient_id = resolve_patient_id(patient_id or st.session_state.get("selected_patient_id"))
    migrate_legacy_session_incidents(patient_id)
    cache_key = f"care_reports_timeline_{patient_id}"

    if force_refresh or cache_key not in st.session_state:
        persisted_events = []
        try:
            for row in fetch_patient_care_reports(patient_id, limit=500):
                persisted_events.append(care_report_row_to_timeline_event(row))
        except Exception:
            persisted_events = []

        st.session_state[cache_key] = persisted_events

    session_events = []
    for incident in get_session_incidents(patient_id):
        source = incident.get("source", "voice_report")
        session_events.append({
            "timestamp": incident.get("timestamp") or "",
            "timestamp_display": incident.get("timestamp_display") or format_chat_timestamp(incident.get("timestamp")),
            "text": incident.get("text") or incident.get("summary") or "",
            "severity": incident.get("severity", "monitor"),
            "caregiver": resolve_caregiver_label(incident.get("caregiver_id")) or incident.get("caregiver", ""),
            "source": source,
            "symptoms": incident.get("symptoms") or extract_symptoms_from_text(incident.get("text", "")),
            "image_b64": incident.get("image_b64") or "",
            "has_photo": bool(incident.get("image_b64")) or source in ("symptom_photo", "pill_photo"),
            "photo_finding": incident.get("photo_finding") or "",
            "photo_type": incident.get("photo_type") or "",
            "report_id": incident.get("report_id"),
        })

    return _dedupe_timeline_events(
        st.session_state.get(cache_key, []) + session_events
    )


def load_medication_adherence_timeline_events(patient_id, force_refresh: bool = False) -> list:
    cache_key = f"adherence_timeline_{resolve_patient_id(patient_id)}"
    if not force_refresh and cache_key in st.session_state:
        return st.session_state[cache_key]

    events = []
    resolved_patient_id = resolve_patient_id(patient_id)

    for log in cached_medication_logs(resolved_patient_id):
        status = str(log.get("status") or "").lower()
        med = log.get("medication_name") or "Medication"
        slot = log.get("scheduled_time") or "scheduled dose"
        logged_at = log.get("logged_at") or ""
        caregiver = resolve_caregiver_label(log.get("caregiver_id")) or log.get("caregiver_id") or "Caregiver"
        if status == "taken":
            text = f"{med} — {slot} dose logged as taken"
            adherence_status = "taken"
        elif status == "missed":
            text = f"{med} — {slot} dose missed"
            adherence_status = "missed"
        else:
            text = f"{med} — {slot} ({status or 'logged'})"
            adherence_status = "check"
        events.append({
            "timestamp": logged_at,
            "timestamp_display": format_chat_timestamp(logged_at),
            "text": text,
            "caregiver": caregiver,
            "source": "medication_log",
            "adherence_status": adherence_status,
        })

    try:
        resolved_patient_id = resolve_patient_id(patient_id or st.session_state.get("selected_patient_id"))
        for row in fetch_medication_check_shift_logs(resolved_patient_id, limit=100):
            created_at = row.get("created_at") or ""
            raw_summary = row.get("summary") or "MedCam medication check"
            events.append({
                "timestamp": created_at,
                "timestamp_display": format_chat_timestamp(created_at),
                "text": format_medcam_shift_log_for_timeline(raw_summary),
                "severity": row.get("severity", "monitor"),
                "caregiver": resolve_caregiver_label(row.get("caregiver_id")) or row.get("caregiver_name", ""),
                "source": "medication_check",
                "adherence_status": "check",
            })
    except Exception:
        pass

    events = _dedupe_timeline_events(events)
    st.session_state[cache_key] = events
    return events


def load_handover_timeline_events(patient_id=None) -> list:
    """Combined timeline — prefer separate symptom/adherence loaders in the UI."""
    combined = load_symptom_timeline_events(patient_id=patient_id) + load_medication_adherence_timeline_events(patient_id)
    return _dedupe_timeline_events(combined)


def build_connected_report_links(incidents: list | None = None) -> list[dict]:
    incidents = incidents or get_session_incidents()
    links = []
    for index, incident in enumerate(incidents):
        prior = incidents[:index]
        if not prior:
            continue
        text = incident.get("text", "")
        triggers = detect_session_escalation_triggers(text, prior)
        if not triggers:
            recurring = detect_recurring_symptom_patterns(prior + [incident])
            if recurring:
                triggers = [f"recurring {item['label'].lower()}" for item in recurring]
        if not triggers:
            continue
        related = select_linked_prior_incidents(text, prior, session_triggers=triggers)
        if not related:
            continue
        links.append({
            "time": incident.get("timestamp_display", ""),
            "report": text,
            "severity": incident.get("severity", "monitor"),
            "connected_to": related,
            "reason": ", ".join(triggers),
        })
    return links


def _timeline_event_styles(event: dict, timeline_kind: str) -> tuple[str, str, str, str]:
    source_label = TIMELINE_SOURCE_LABELS.get(event.get("source", ""), "Report")
    if timeline_kind == "adherence" and event.get("adherence_status"):
        label, bg, fg = ADHERENCE_TIMELINE_STYLES.get(
            event["adherence_status"],
            ADHERENCE_TIMELINE_STYLES["check"],
        )
        return label, bg, fg, source_label
    level = normalize_chat_severity(event.get("severity", "monitor"))
    label, bg, fg = TIMELINE_SEVERITY_STYLES.get(level, TIMELINE_SEVERITY_STYLES["monitor"])
    return label, bg, fg, source_label


def build_handover_timeline_html(
    events: list,
    *,
    heading: str,
    description: str = "",
    empty_text: str,
    timeline_class: str = "cs-symptom-timeline",
    show_hidden_indicator: bool = False,
    timeline_kind: str = "symptom",
    accent: bool = True,
) -> str:
    accent_class = " cs-timeline-accent-card" if accent else ""
    heading_html = f'<div class="cs-timeline-heading">{html.escape(heading)}</div>'
    description_html = ""
    if description:
        description_html = f'<p class="cs-timeline-description">{html.escape(description)}</p>'

    if not events and not show_hidden_indicator:
        return f"""
        <div class="{timeline_class}{accent_class} {timeline_class}-empty">
          {heading_html}
          {description_html}
          <p class="cs-timeline-empty-text">{empty_text}</p>
        </div>
        """

    items = [f'<div class="{timeline_class}{accent_class}">', heading_html, description_html]
    items.append('<div class="cs-timeline-track">')
    if show_hidden_indicator:
        items.append('<div class="cs-timeline-more-indicator">···</div>')
    for event in events:
        label, bg, fg, source_label = _timeline_event_styles(event, timeline_kind)
        caregiver = html.escape(str(event.get("caregiver") or "Caregiver"))
        time_text = html.escape(str(event.get("timestamp_display") or "Unknown time"))
        text = html.escape(str(event.get("text") or ""))
        level_class = event.get("adherence_status") or normalize_chat_severity(event.get("severity", "monitor"))
        items.append(
            f'<div class="cs-timeline-item cs-timeline-{level_class}">'
            f'<div class="cs-timeline-dot" style="background:{bg};border-color:{fg};"></div>'
            f'<div class="cs-timeline-card" style="border-left-color:{fg};">'
            f'<div class="cs-timeline-meta">'
            f'<span class="cs-timeline-time">{time_text}</span>'
            f'<span class="cs-timeline-badge" style="background:{bg};color:{fg};">{label}</span>'
            f'<span class="cs-timeline-source">{html.escape(source_label)}</span>'
            f'</div>'
            f'<div class="cs-timeline-text">{text}</div>'
            f'<div class="cs-timeline-caregiver">{caregiver}</div>'
            f'</div></div>'
        )
    items.append("</div></div>")
    return "".join(items)


def build_symptom_timeline_html(events: list, show_hidden_indicator: bool = False) -> str:
    return build_handover_timeline_html(
        events,
        heading=SYMPTOM_TIMELINE_HEADING,
        description=SYMPTOM_TIMELINE_DESCRIPTION,
        empty_text=(
            "No reports logged yet. Incidents from Report &amp; Ask appear here in chronological order "
            "with severity colour coding."
        ),
        timeline_class="cs-symptom-timeline",
        show_hidden_indicator=show_hidden_indicator,
        timeline_kind="symptom",
    )


def render_paginated_handover_timeline(
    events: list,
    visible_session_key: str,
    *,
    heading: str,
    description: str = "",
    empty_text: str,
    timeline_class: str,
    timeline_kind: str,
    button_key: str,
) -> None:
    if visible_session_key not in st.session_state:
        st.session_state[visible_session_key] = HANDOVER_TIMELINE_DEFAULT_VISIBLE

    total = len(events)
    if total == 0:
        md_html(build_handover_timeline_html(
            [],
            heading=heading,
            description=description,
            empty_text=empty_text,
            timeline_class=timeline_class,
            timeline_kind=timeline_kind,
        ))
        return

    shown_count = min(st.session_state[visible_session_key], total)
    start_idx = max(0, total - shown_count)
    visible_events = events[start_idx:]
    hidden_count = start_idx

    md_html(build_handover_timeline_html(
        visible_events,
        heading=heading,
        description=description,
        empty_text=empty_text,
        timeline_class=timeline_class,
        show_hidden_indicator=hidden_count > 0,
        timeline_kind=timeline_kind,
    ))
    if hidden_count > 0:
        md_html('<div class="cs-timeline-more-indicator">···</div>')

    render_history_show_more_controls(
        hidden_count=hidden_count,
        total=total,
        shown_count=shown_count,
        session_key=visible_session_key,
        show_more_key=button_key,
    )


def render_handover_visible_timelines(patient_id, period_key: str, tz_obj) -> None:
    all_symptom_events = load_symptom_timeline_events(patient_id=patient_id)
    symptom_events = filter_events_by_handover_period(all_symptom_events, period_key, tz_obj)

    md_html('<div class="cs-handover-timeline-divider"></div>')
    render_paginated_handover_timeline(
        symptom_events,
        "handover_symptom_timeline_visible",
        heading=SYMPTOM_TIMELINE_HEADING,
        description=SYMPTOM_TIMELINE_DESCRIPTION,
        empty_text=(
            "No symptom reports or care updates logged for this period yet. "
            "Use Report & Ask to record observations."
        ),
        timeline_class="cs-handover-timeline--symptom",
        timeline_kind="symptom",
        button_key="handover_symptom_show_more",
    )

    all_adherence_events = load_medication_adherence_timeline_events(patient_id)
    adherence_events = filter_events_by_handover_period(all_adherence_events, period_key, tz_obj)
    md_html('<div class="cs-handover-timeline-divider"></div>')
    render_paginated_handover_timeline(
        adherence_events,
        "handover_adherence_timeline_visible",
        heading=MEDICATION_ADHERENCE_TIMELINE_HEADING,
        description=MEDICATION_ADHERENCE_TIMELINE_DESCRIPTION,
        empty_text="No medication doses logged for this period yet.",
        timeline_class="cs-handover-timeline--adherence",
        timeline_kind="adherence",
        button_key="handover_adherence_show_more",
    )


def render_handover_dashboard(patient_id, period_key: str, tz_obj) -> None:
    all_symptom_events = load_symptom_timeline_events(patient_id=patient_id)
    symptom_events = filter_events_by_handover_period(all_symptom_events, period_key, tz_obj)
    render_handover_charts(patient_id, symptom_events, period_key, tz_obj)


def render_handover_timelines(patient_id, period_key: str, tz_obj) -> None:
    render_handover_dashboard(patient_id, period_key, tz_obj)
    render_handover_visible_timelines(patient_id, period_key, tz_obj)


def render_handover_timeline(_timeline_events: list | None = None) -> None:
    tz_obj, _ = get_schedule_tz()
    period_key = st.session_state.get("handover_period", "this_week")
    render_handover_timelines(st.session_state.get("selected_patient_id"), period_key, tz_obj)


def compute_adherence_stats(patient_id, period_key: str, tz_obj) -> dict:
    now = datetime.now(tz_obj)
    logs = cached_medication_logs(resolve_patient_id(patient_id))
    taken = 0
    missed = 0
    by_date: dict[str, dict[str, int]] = {}

    for log in logs:
        if not event_in_handover_period(log, period_key, tz_obj):
            continue
        status = str(log.get("status") or "").lower()
        if status not in ("taken", "missed"):
            continue
        date_iso = parse_log_local_date(log, tz_obj)
        if not date_iso:
            continue
        if status == "taken":
            taken += 1
        else:
            missed += 1
        day = by_date.setdefault(date_iso, {"taken": 0, "missed": 0})
        day[status] += 1

    total = taken + missed
    rate = round(100 * taken / total) if total else None
    period_start, period_end = get_handover_period_bounds(period_key, tz_obj)
    daily = []
    cursor = period_start.date()
    end_date = min(period_end.date(), now.date())
    while cursor <= end_date:
        day_iso = cursor.isoformat()
        counts = by_date.get(day_iso, {"taken": 0, "missed": 0})
        daily.append({
            "label": cursor.strftime("%a"),
            "taken": counts["taken"],
            "missed": counts["missed"],
            "total": counts["taken"] + counts["missed"],
        })
        cursor += timedelta(days=1)
    if len(daily) > 14:
        daily = daily[-14:]
    return {
        "taken": taken,
        "missed": missed,
        "total": total,
        "rate": rate,
        "daily": daily,
    }


def compute_symptom_severity_counts(events: list) -> dict:
    counts = {"emergency": 0, "contact_doctor": 0, "monitor": 0, "ok": 0}
    for event in events:
        level = normalize_chat_severity(event.get("severity", "monitor"))
        if level in counts:
            counts[level] += 1
        else:
            counts["monitor"] += 1
    return counts


def build_adherence_donut_html(pct: float, ring_color: str) -> str:
    pct_safe = min(max(float(pct), 0.0), 100.0)
    return (
        f'<div class="cs-adherence-ring-css" style="background: conic-gradient('
        f'{ring_color} 0% {pct_safe:g}%, #E8E4DA {pct_safe:g}% 100%);">'
        f'<div class="cs-adherence-ring-hole">'
        f'<span class="cs-adherence-ring-pct">{pct_safe:g}%</span>'
        f'<span class="cs-adherence-ring-sub">ADHERENCE</span>'
        f'</div></div>'
    )


def build_adherence_chart_html(stats: dict, daily_title: str = "Daily doses in period") -> str:
    taken = stats.get("taken", 0)
    missed = stats.get("missed", 0)
    total = stats.get("total", 0)
    rate = stats.get("rate")
    daily = stats.get("daily") or []

    pct = min(max(rate or 0, 0), 100) if total else 0
    ring_color = "#34C759" if pct >= 80 else "#EAB308" if pct >= 60 else "#FF453A"
    if total == 0:
        ring_color = "#D1D5DB"

    day_bars = []
    max_day_total = max((day.get("total") or 0 for day in daily), default=1) or 1
    for day in daily:
        taken_h = (day.get("taken") / max_day_total * 100) if max_day_total else 0
        missed_h = (day.get("missed") / max_day_total * 100) if max_day_total else 0
        day_bars.append(
            '<div class="cs-adherence-day" title="'
            + f'{day["taken"]} taken, {day["missed"]} missed">'
            + '<div class="cs-adherence-day-stack">'
            + f'<div class="cs-adherence-day-seg cs-adherence-day-seg--missed" style="height:{missed_h:.1f}%"></div>'
            + f'<div class="cs-adherence-day-seg cs-adherence-day-seg--taken" style="height:{taken_h:.1f}%"></div>'
            + '</div>'
            + f'<span class="cs-adherence-day-label">{html.escape(day["label"])}</span>'
            + '</div>'
        )

    summary = f"{taken} taken · {missed} missed" if total else "No doses logged yet"
    summary_sub = (
        f"{total} dose(s) logged in total"
        if total
        else "Taken and missed doses from MedCam will appear here."
    )

    parts = [
        '<div class="cs-handover-chart-card">',
        '<div class="cs-handover-chart-title">Medication adherence</div>',
        '<div class="cs-adherence-chart-body">',
        '<div class="cs-adherence-ring-wrap">',
        build_adherence_donut_html(pct, ring_color),
        '</div>',
        '<div class="cs-adherence-chart-copy">',
        f'<div class="cs-adherence-summary">{html.escape(summary)}</div>',
        f'<div class="cs-adherence-summary-sub">{html.escape(summary_sub)}</div>',
        '<div class="cs-adherence-legend">',
        '<span class="cs-adherence-legend-item"><span class="cs-adherence-swatch cs-adherence-swatch--taken"></span>Taken</span>',
        '<span class="cs-adherence-legend-item"><span class="cs-adherence-swatch cs-adherence-swatch--missed"></span>Missed</span>',
        '</div></div></div>',
        f'<div class="cs-adherence-daily-title">{html.escape(daily_title)}</div>',
        f'<div class="cs-adherence-daily">{"".join(day_bars)}</div>',
        '</div>',
    ]
    return "".join(parts)


def build_symptom_severity_chart_html(counts: dict) -> str:
    rows = [
        ("Urgent", counts.get("emergency", 0), "#B91C1C", "#FEE2E2"),
        ("Contact doctor", counts.get("contact_doctor", 0), "#C2410C", "#FFEDD5"),
        ("Monitor", counts.get("monitor", 0), "#92400E", "#FEF3C7"),
        ("Routine", counts.get("ok", 0), "#166534", "#DCFCE7"),
    ]
    total = sum(value for _, value, _, _ in rows)
    if total == 0:
        return (
            '<div class="cs-handover-chart-card">'
            '<div class="cs-handover-chart-title">Symptom &amp; incident severity</div>'
            '<p class="cs-handover-chart-empty">No symptom reports logged yet. Report &amp; Ask updates will appear here.</p>'
            '</div>'
        )

    max_count = max(value for _, value, _, _ in rows) or 1
    bar_rows = []
    for label, value, fg, bg in rows:
        width = max(8.0, 100 * value / max_count) if value else 0
        bar_rows.append(
            '<div class="cs-severity-bar-row">'
            f'<span class="cs-severity-bar-label">{html.escape(label)}</span>'
            '<div class="cs-severity-bar-track">'
            f'<div class="cs-severity-bar-fill" style="width:{width:.1f}%;background:{bg};border:1px solid {fg};">'
            f'<span class="cs-severity-bar-fill-text" style="color:{fg};">{value}</span>'
            '</div></div>'
            f'<span class="cs-severity-bar-count">{value}</span>'
            '</div>'
        )

    return (
        '<div class="cs-handover-chart-card">'
        '<div class="cs-handover-chart-title">Symptom &amp; incident severity</div>'
        f'<div class="cs-severity-chart-sub">{total} logged report{"" if total == 1 else "s"} from Report &amp; Ask in this period</div>'
        f'<div class="cs-severity-bars">{"".join(bar_rows)}</div>'
        '</div>'
    )


def compute_recurring_care_topics(events: list) -> list[dict]:
    """Count repeated symptoms/topics logged in the selected handover period."""
    counts: dict[str, int] = {}
    labels = {key: label for key, _pattern, label in SYMPTOM_PATTERN_DEFINITIONS}
    for event in events or []:
        text = str(event.get("text") or "")
        keys = event.get("symptoms") or extract_symptoms_from_text(text)
        for key in keys:
            counts[key] = counts.get(key, 0) + 1
    results = [
        {
            "label": labels.get(key, key.replace("_", " ").title()),
            "count": count,
        }
        for key, count in counts.items()
        if count > 0
    ]
    results.sort(key=lambda item: (-item["count"], item["label"]))
    return results[:8]


def build_recurring_topics_chart_html(topics: list, period_key: str) -> str:
    period_label = get_handover_period_label(period_key).lower()
    if not topics:
        return (
            '<div class="cs-handover-chart-card cs-handover-chart-card--wide">'
            '<div class="cs-handover-chart-title">Most repeated concerns</div>'
            '<p class="cs-handover-chart-empty">No repeated symptoms or incidents logged yet for '
            f'{html.escape(period_label)}. Report &amp; Ask updates will appear here.</p>'
            '</div>'
        )

    max_count = max(item["count"] for item in topics) or 1
    bar_rows = []
    for item in topics:
        value = item["count"]
        width = max(8.0, 100 * value / max_count)
        label = html.escape(item["label"])
        times_word = "time" if value == 1 else "times"
        bar_rows.append(
            '<div class="cs-severity-bar-row cs-recurring-bar-row">'
            f'<span class="cs-severity-bar-label">{label}</span>'
            '<div class="cs-severity-bar-track">'
            f'<div class="cs-severity-bar-fill cs-recurring-bar-fill" style="width:{width:.1f}%;">'
            f'<span class="cs-severity-bar-fill-text">{value}</span>'
            '</div></div>'
            f'<span class="cs-severity-bar-count cs-recurring-bar-count">{value} {times_word}</span>'
            '</div>'
        )

    return (
        '<div class="cs-handover-chart-card cs-handover-chart-card--wide">'
        '<div class="cs-handover-chart-title">Most repeated concerns</div>'
        f'<div class="cs-severity-chart-sub">Symptoms and incidents logged more than once — {html.escape(period_label)}</div>'
        f'<div class="cs-severity-bars">{"".join(bar_rows)}</div>'
        '</div>'
    )


def render_handover_charts(
    patient_id,
    symptom_events: list,
    period_key: str,
    tz_obj,
) -> None:
    adherence_stats = compute_adherence_stats(patient_id, period_key, tz_obj)
    severity_counts = compute_symptom_severity_counts(symptom_events)
    recurring_topics = compute_recurring_care_topics(symptom_events)
    daily_title = f"DOSES — {get_handover_period_label(period_key).upper()}"
    chart_left, chart_right = st.columns(2)
    with chart_left:
        md_html(build_adherence_chart_html(adherence_stats, daily_title=daily_title))
    with chart_right:
        md_html(build_symptom_severity_chart_html(severity_counts))
    md_html(build_recurring_topics_chart_html(recurring_topics, period_key))


def _why_home_icon(name: str, size: int = 22) -> str:
    icons = {
        "care-log": (
            '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
            '<polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/>'
            '<line x1="16" y1="17" x2="8" y2="17"/>'
            "</svg>"
        ),
        "stethoscope": (
            '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            '<path d="M4.8 2.3A2 2 0 0 0 3 4v3a5 5 0 0 0 5 5 5 5 0 0 0 5-5V4a2 2 0 0 0-2-2"/>'
            '<path d="M8 15v1a6 6 0 0 0 12 0v-3"/>'
            '<circle cx="20" cy="10" r="2"/>'
            "</svg>"
        ),
        "med-plan": (
            '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            '<path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/>'
            '<rect x="8" y="2" width="8" height="4" rx="1" ry="1"/>'
            '<line x1="9" y1="12" x2="15" y2="12"/><line x1="9" y1="16" x2="13" y2="16"/>'
            "</svg>"
        ),
        "handover": (
            '<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            '<path d="m17 2 4 4-4 4"/><path d="M3 11v-1a4 4 0 0 1 4-4h14"/>'
            '<path d="m7 22-4-4 4-4"/><path d="M21 13v1a4 4 0 0 1-4 4H3"/>'
            "</svg>"
        ),
    }
    svg = icons.get(name, "").format(s=size)
    return f'<span class="cs-why-feature-icon">{svg}</span>'


def build_how_to_use_steps_html() -> str:
    steps = [
        {
            "num": 1,
            "title": "Documents",
            "text": (
                "Upload a discharge letter or prescription. "
                "Meds and conditions are extracted automatically."
            ),
            "tone": "documents",
        },
        {
            "num": 2,
            "title": "Report and ask",
            "text": (
                "Carer logs symptoms or asks care questions in chat. "
                "Entries are timestamped and saved for handover."
            ),
            "tone": "report",
        },
        {
            "num": 3,
            "title": "Pill registration",
            "text": (
                "Photograph each medication once so the app learns it. "
                "Dose counts come from the discharge document."
            ),
            "tone": "pill",
        },
        {
            "num": 4,
            "title": "MedCam",
            "text": (
                "Photograph pills in hand before giving a dose. "
                "MedCam checks against the schedule and flags mismatches."
            ),
            "tone": "medcam",
        },
        {
            "num": 5,
            "title": "Handover",
            "text": (
                "All logged symptoms, reports, and adherence data compile into "
                "a clinical-grade SBAR report for the doctor."
            ),
            "tone": "handover",
        },
        {
            "num": 6,
            "title": "My results",
            "text": (
                "Upload a test result or letter and CareShield explains it in plain English, "
                "with questions to ask the doctor."
            ),
            "tone": "results",
        },
    ]
    cards = []
    for step in steps:
        cards.append(
            f'<div class="cs-how-step cs-how-step--{step["tone"]}">'
            f'<div class="cs-how-step-num">{step["num"]}</div>'
            f'<div class="cs-how-step-title">{html.escape(step["title"])}</div>'
            f'<div class="cs-how-step-text">{html.escape(step["text"])}</div>'
            f"</div>"
        )
    return (
        '<div class="cs-how-section">'
        '<div class="cs-how-section-label">How to use CareShield</div>'
        '<div class="cs-how-nav-row">'
        '<button type="button" class="cs-how-nav-btn" data-cs-how-scroll="-1" aria-label="Previous steps">&#8249;</button>'
        '<div class="cs-how-track">'
        '<div class="cs-how-line" aria-hidden="true"></div>'
        f'<div class="cs-how-steps" id="cs-how-steps">{"".join(cards)}</div>'
        '</div>'
        '<button type="button" class="cs-how-nav-btn" data-cs-how-scroll="1" aria-label="Next steps">&#8250;</button>'
        '</div>'
        '</div>'
    )


def render_how_to_use_bar() -> None:
    md_html(f'<div class="cs-how-bar">{build_how_to_use_steps_html()}</div>')
    components.html(
        """
        <script>
        (function () {
          const doc = window.parent.document;
          const steps = doc.getElementById("cs-how-steps");
          if (!steps) return;
          const scrollAmount = () => {
            const card = steps.querySelector(".cs-how-step");
            return (card ? card.offsetWidth : 168) + 10;
          };
          doc.querySelectorAll("[data-cs-how-scroll]").forEach((btn) => {
            btn.onclick = (event) => {
              event.preventDefault();
              const dir = Number(btn.getAttribute("data-cs-how-scroll") || "1");
              steps.scrollBy({ left: dir * scrollAmount(), behavior: "smooth" });
            };
          });
        })();
        </script>
        """,
        height=0,
    )


def render_first_time_guide_button() -> None:
    """Legacy hook — profile help now lives in the header profile switchers."""
    return


PROFILE_MANAGEMENT_HELP = """
**Important:** Right now this is single-login, multi-profile — like a shared family account. In production we'd add per-caregiver authentication/permissions if this handled real medical data.

**Who's logged in?** (left)

- Pick your name from the dropdown if you're already set up.
- Choose **+ Add new profile** if you're a new carer on this account.
- Tap the **pencil icon** to edit or remove carer profiles.

**Currently caring for** (right)

- Select the patient you are looking after today.
- Choose **+ Add new patient** if this person is new to CareShield.
- Tap the **pencil icon** to update patient details.

You can switch carer or patient at any time — reports stay linked to the right people.
"""


RESPONSIBLE_AI_SAFETY_CONTENT = """
CareShield sits between a frightened or exhausted family member and decisions about someone they love. I didn't take that lightly while building it.

**Privacy, honestly.** My data model keeps each patient's medications, conditions, symptom reports, and photos scoped to their own profile, built around how care actually works — usually several family members looking after one person, not one account per patient. Right now, in this hackathon build, I've kept profile selection open and database policies loose on purpose, so I could spend my limited time getting the clinical reasoning right instead of also building full authentication. That was a conscious tradeoff, not something I missed. Row Level Security is already switched on at the database level, and the next real step is caregiver login and care-circle permissions — the schema is already built to support that without starting over. Documents and photos are only used to pull out structured medical data, nothing else. And anything sent to OpenAI for AI responses isn't used to train their models.

**I built for the blank screen, not just the perfect demo.** A brand-new patient has no documents, no symptoms logged, no doses recorded yet — and I wanted the app to be honest about that instead of pretending otherwise. Handover tells you plainly there isn't enough logged yet to generate a report. Documents says "no medications on file" instead of just showing nothing. Pill Registration and MedCam both notice when there's nothing to check against and walk you toward uploading a document first. A system that quietly guesses when it has no data is worse than one that just says so.

**I tried to catch mistakes before they become real ones.** Upload a document for the wrong patient, and CareShield refuses it rather than silently mixing someone else's medical history into the wrong profile. If MedCam isn't confident about a pill in a photo, it says so and tells you to check it yourself instead of guessing. And if a scheduled dose is missing from the photo, it doesn't just log it as skipped — it tells you exactly what's at stake: "Amlodipine was due earlier today but was not in this photo. If you still need to give it, check the care plan for late doses — do not double up without checking with the doctor or pharmacist first." A wrong guess here isn't a bug, it's a real risk to a real person, so I tried to make sure uncertainty always gets handed back to a human.

**This is support, not a diagnosis.** CareShield never tells you what to do medically. Every response comes with a clear disclaimer, and MedCam says "review before giving," never a pass or fail — because a photo of pills in someone's hand is never 100% certain. It's here to organize and notice patterns. The doctor still makes the calls.

**I'd rather admit uncertainty than fake confidence.** Photos get blurry, angles are wrong, lighting is bad — a caregiver doing this at 11pm, exhausted, will sometimes miss a pill in frame. So instead of guessing, MedCam tells you exactly what it saw and didn't: "Only 1 of 2 pills detected, check you have the full dose." If it's not sure, it says "could not verify" — because a false "all good" is more dangerous than admitting it doesn't know.

**The important stuff doesn't get buried.** Every symptom report gets a severity label — Monitor, Contact doctor, or Urgent — so a caregiver skimming a long history in a stressful moment sees what actually needs attention first, instead of having to reread everything.

**You're still the one in charge.** You decide what gets reported, whether to act on a MedCam warning, and what goes into the Handover report. CareShield never calls a doctor for you, never assumes a dose was given, never makes a decision on your behalf. It organizes. You act.
"""


@st.dialog("Responsible AI and Safety")
def responsible_ai_safety_dialog() -> None:
    st.markdown(RESPONSIBLE_AI_SAFETY_CONTENT)
    if st.button("Close", use_container_width=True, key="close_responsible_ai_dialog"):
        st.rerun()


def render_profile_management_help_button() -> None:
    md_html('<div class="cs-header-help-anchor cs-header-help-anchor--profiles"></div>')
    with st.popover("Help managing profiles", use_container_width=True):
        st.markdown(PROFILE_MANAGEMENT_HELP)


def render_responsible_ai_safety_button() -> None:
    md_html('<div class="cs-header-help-anchor cs-header-help-anchor--safety"></div>')
    if st.button(
        "Responsible AI & Safety",
        key="responsible_ai_safety_btn",
        use_container_width=True,
    ):
        st.session_state.show_responsible_ai_dialog = True
        st.rerun()


def render_homepage_back_button() -> None:
    md_html('<div class="cs-home-back-anchor"></div>')
    if st.button("Back to homepage", key="back_to_homepage", type="secondary"):
        st.session_state.why_this_exists_seen = False
        st.rerun()


def build_why_feature_grid_html() -> str:
    feature_cards = [
        (
            "care-log",
            "Shared care log",
            "Everyone on the care team adds updates. Nothing stays with just one person.",
        ),
        (
            "stethoscope",
            "Doctor-ready timeline",
            "A clear record of symptoms and changes, ready before every appointment.",
        ),
        (
            "med-plan",
            "Medication plans",
            "Always up to date, always visible to whoever is on duty.",
        ),
        (
            "handover",
            "Seamless handovers",
            "The next carer walks in knowing everything the last one did.",
        ),
    ]
    cards = []
    for icon, title, text in feature_cards:
        cards.append(
            '<div class="cs-why-feature-card">'
            + _why_home_icon(icon)
            + f'<div class="cs-why-feature-title">{html.escape(title)}</div>'
            + f'<div class="cs-why-feature-text">{html.escape(text)}</div>'
            + "</div>"
        )
    return '<div class="cs-why-feature-grid">' + "".join(cards) + "</div>"


def render_why_this_exists_screen() -> None:
    md_html(f"""
    <div class="cs-why-screen">
      <div class="cs-why-logo">
        {build_careshield_logo_html(include_tagline=False)}
      </div>

      <p class="cs-why-kicker">For families caring at home</p>

      <h1 class="cs-why-headline">
        <span class="cs-why-headline-dark">Your family sees everything.</span>
        <span class="cs-why-headline-accent">Your doctor hears almost none of it.</span>
      </h1>

      <p class="cs-why-lead">
        Between shifts, visits, and phone calls, critical changes slip through the gaps —
        and your next appointment starts from scratch.
      </p>

      <div class="cs-why-stat-grid">
        <div class="cs-why-stat-card">
          <div class="cs-why-stat-value">7%</div>
          <div class="cs-why-stat-text">
            of carers are actually <strong>identified</strong> by their GP — even when
            they&rsquo;re in the room.
          </div>
          <div class="cs-why-stat-source">NHS England / Macmillan Briefing on Carers</div>
        </div>
        <div class="cs-why-stat-card">
          <div class="cs-why-stat-value">51%</div>
          <div class="cs-why-stat-text">
            of unpaid carers say they <strong>need more NHS support</strong> — and
            aren&rsquo;t getting it.
          </div>
          <div class="cs-why-stat-source">Carers UK, State of Caring 2023</div>
        </div>
      </div>

      <div class="cs-why-section">
        <div class="cs-why-section-label">What happens between appointments</div>
        <p class="cs-why-section-copy">
          A change shows up on Tuesday. Three different people notice something feels off.
          By Friday&rsquo;s appointment, <strong>nobody has written it down</strong> — and the
          doctor gets a rough &ldquo;she&rsquo;s been a bit worse lately.&rdquo;
        </p>
      </div>

      <p class="cs-why-solution">
        <strong>CareShield gives your whole care team
        <span class="cs-why-solution-accent">one shared record</span>
        — so nothing falls through.</strong>
      </p>

      <p class="cs-why-support">
        Log changes as they happen, upload documents, hand over to the next person in seconds.
        Walk into every appointment with a clear timeline your doctor can actually use.
      </p>

      <div class="cs-why-cta-box">
        <div class="cs-why-cta-title">Walk in prepared. Walk out with answers.</div>
        <p class="cs-why-cta-copy">
          Show your doctor <strong>exactly what changed, when, and how often</strong> — not a
          rough memory. Clearer information means faster decisions and care that actually fits.
        </p>
      </div>
    </div>
    """)
    if st.button(
        "Continue to CareShield",
        key="why_continue",
        use_container_width=True,
        type="primary",
    ):
        st.session_state.why_this_exists_seen = True
        st.session_state.careshield_enter_from_homepage = True
        st.rerun()


def build_chat_history_for_api(messages: list, limit: int = 20) -> list:
    history = []
    for msg in messages[-limit:]:
        if msg.get("welcome"):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role not in ("user", "assistant") or not content:
            continue
        text = re.sub(r"<[^>]+>", " ", str(content))
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        stamp = msg.get("timestamp_display") or ""
        prefix = f"[{stamp}] " if stamp else ""
        history.append({"role": role, "content": prefix + text})
    while history and history[0]["role"] != "user":
        history.pop(0)
    merged = []
    for entry in history:
        if merged and merged[-1]["role"] == entry["role"]:
            merged[-1]["content"] += f"\n{entry['content']}"
        else:
            merged.append(dict(entry))
    return merged


def coerce_ai_text(value, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [coerce_ai_text(item) for item in value]
        return "\n".join(part for part in parts if part) or default
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def humanize_caregiver_label(text: str) -> str:
    """Turn internal codes (snake_case, etc.) into plain language for caregivers."""
    cleaned = coerce_ai_text(text).strip()
    if not cleaned:
        return ""
    if re.search(r"[A-Za-z] [A-Za-z]", cleaned) and not re.search(r"[_-]", cleaned):
        return cleaned
    if re.search(r"[_-]", cleaned):
        parts = [part for part in re.split(r"[_-]+", cleaned) if part]
        return " ".join(word.capitalize() for word in parts)
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", cleaned)
    return spaced.strip().capitalize() if spaced else cleaned


def format_caregiver_clinical_tags(tags) -> str:
    labels = []
    for tag in tags or []:
        label = humanize_caregiver_label(tag)
        if label:
            labels.append(label)
    return " · ".join(labels)


def reports_health_symptom_topic(text: str) -> bool:
    from ai_helpers import reports_health_symptom_topic as _reports_health_symptom_topic
    return _reports_health_symptom_topic(text)


def is_question(text: str) -> bool:
    from ai_helpers import is_care_question_text
    return is_care_question_text(text)


def append_care_guidance(reply: str, severity: str, needs_doctor: bool = False) -> str:
    level = normalize_chat_severity(severity)
    if needs_doctor and level not in ("emergency", "contact_doctor"):
        level = "contact_doctor"
    if is_emergency_severity(level):
        return reply + "\n\nCall 999 (or 112) immediately — do not wait."
    if is_contact_doctor_severity(level):
        return reply + f"\n\n{CONTACT_DOCTOR_GUIDANCE}"
    return reply


def build_med_rows_html(items, demo_class=""):
    rows = []
    for item in items:
        name = html.escape(str(item.get("name", "")))
        dosage = str(item.get("dosage", "") or "")
        timing = html.escape(str(item.get("time") or item.get("timing") or ""))
        time_html = f'<span class="cs-med-time">{timing}</span>' if timing else ""
        dosage_html = ""
        if dosage and dosage not in str(item.get("name", "")):
            dosage_html = f'<div class="cs-plan-dosage">{html.escape(dosage)}</div>'
        rows.append(
            f'<div class="cs-med-row{demo_class}">'
            f'<div class="cs-med-left">'
            f'<span class="cs-med-dot cs-med-dot-{item["color"]}"></span>'
            f'<div><span class="cs-med-name">{name}</span>{dosage_html}</div>'
            f'</div>{time_html}</div>'
        )
    return "".join(rows)


def format_medications_for_prompt(items):
    if not items:
        return "No medication plan uploaded yet."
    lines = []
    for med in items:
        line = str(med.get("name", ""))
        dosage = str(med.get("dosage", "") or "")
        timing = str(med.get("time") or med.get("timing") or "")
        pills = med.get("pills_per_dose") or extract_pills_per_dose(med)
        if dosage and dosage not in line:
            line += f" ({dosage})"
        if pills:
            line += f", take {pills} pill(s) per dose"
        if timing:
            line += f" at {timing}"
        lines.append(line)
    return "; ".join(lines)


def is_medication_schedule_question(text: str) -> bool:
    if not text or not text.strip():
        return False
    lower = text.lower()
    if re.search(r"\bhow(?:'s| is| are| was| were)\b.+\bworking\b", lower):
        return True
    keywords = (
        "pill", "pills", "medication", "medications", "medicine", "medicines",
        "dose", "doses", "tablet", "tablets", "give", "take", "schedule",
        "what time", "when should", "when do", "which pill", "what should i give",
    )
    return any(keyword in lower for keyword in keywords)


def looks_like_medication_refusal(reply) -> bool:
    lower = coerce_ai_text(reply).lower()
    refusal_markers = (
        "not registered",
        "unregistered",
        "cannot provide medication",
        "can't provide medication",
        "only provide medication advice for registered",
        "don't have access to",
        "do not have access to",
        "unable to provide medication",
        "cannot advise on medication",
        "can't advise on medication",
        "consult a pharmacist for his specific medication schedule",
        "consult a pharmacist for their specific medication schedule",
    )
    return any(marker in lower for marker in refusal_markers)


def build_medication_schedule_reply(plan_items: list | None = None) -> str:
    items = plan_items or get_active_plan_items()
    if not items:
        return (
            "I don't see a medication plan on file yet. Upload a discharge document in the "
            "**Documents** tab so I can list what to give and when."
        )
    lines = ["Here's **your patient's medication schedule** from their care plan:\n"]
    for med in items:
        name = med.get("name", "Medication")
        timing = med.get("time") or med.get("timing") or "time not specified"
        dosage = med.get("dosage") or ""
        pills = med.get("pills_per_dose") or extract_pills_per_dose(med)
        detail_parts = []
        if dosage and str(dosage) not in str(name):
            detail_parts.append(str(dosage))
        if pills:
            detail_parts.append(f"{pills} pill(s) per dose")
        detail = f" ({', '.join(detail_parts)})" if detail_parts else ""
        lines.append(f"- **{name}**{detail} — **{timing}**")
    lines.append(
        "\nUse this schedule unless their doctor has told you otherwise. "
        "Open **MedCam** when you want to verify pills before giving a dose."
    )
    return "\n".join(lines)


def build_medication_schedule_prompt_block(plan_items: list | None = None) -> str:
    items = plan_items or get_active_plan_items()
    return f"""AUTHORITATIVE MEDICATION SCHEDULE FOR THIS PATIENT (always on file for this account):
{format_medications_for_prompt(items)}

When asked what to give, when to give it, or which pills — answer directly from this schedule.
NEVER say the patient is unregistered or that you cannot provide medication guidance."""


def is_next_medication_question(text: str) -> bool:
    lower = (text or "").strip().lower()
    if not is_medication_schedule_question(text):
        return False
    return any(
        phrase in lower
        for phrase in (
            "next pill",
            "next dose",
            "next medication",
            "next med",
            "next medicine",
            "when should i give",
            "when do i give",
            "when does",
            "when should",
            "what's due",
            "what is due",
            "due now",
            "due next",
        )
    )


def _dose_next_occurrence(dose: dict, now: datetime) -> datetime:
    scheduled = now.replace(hour=dose["hour"], minute=dose["minute"], second=0, microsecond=0)
    if scheduled > now:
        return scheduled
    return scheduled + timedelta(days=1)


def _format_relative_when(minutes: float) -> str:
    mins = max(int(minutes), 1)
    if mins < 60:
        return f"in about {mins} minute{'s' if mins != 1 else ''}"
    hours = mins // 60
    rem = mins % 60
    when = f"in about {hours} hour{'s' if hours != 1 else ''}"
    if rem:
        when += f" and {rem} minutes"
    return when


def _format_med_name_list(names: list[str], max_show: int = 4) -> str:
    shown = names[:max_show]
    text = ", ".join(f"**{name}**" for name in shown)
    if len(names) > max_show:
        text += f", and **{len(names) - max_show} more**"
    return text


def build_next_medication_reply(
    plan_items: list | None = None,
    patient_id=None,
    user_text: str = "",
    today_logs: list | None = None,
    now: datetime | None = None,
) -> str:
    items = plan_items or get_active_plan_items()
    tz_obj, _ = get_schedule_tz()
    now = now or datetime.now(tz_obj)
    today = now.date().isoformat()
    resolved_patient_id = resolve_patient_id(
        patient_id if patient_id is not None else st.session_state.get("selected_patient_id")
    )

    if today_logs is None:
        today_logs = [
            log for log in cached_medication_logs(resolved_patient_id)
            if parse_log_local_date(log, tz_obj) == today
        ]

    named_med = extract_medication_name_from_question(user_text, items)
    patient_label = get_patient_display_name(resolved_patient_id)
    unknown_med = None
    if named_med:
        unknown_med = (
            f"I don't see **{named_med}** on {patient_label}'s medication plan. "
            "Open **Documents** to confirm their medicines, or ask about a drug on the plan."
        )

    reply = build_next_medication_reply_core(
        items,
        user_text=user_text,
        today_logs=today_logs,
        now=now,
        tz_obj=tz_obj,
        unknown_med_message=unknown_med,
    )
    if reply.startswith("No scheduled dose times"):
        return build_medication_schedule_reply(items)
    return reply


def find_plan_item(med_name: str, plan_items: list) -> dict:
    for item in plan_items:
        if item["name"] == med_name:
            return item
    name_lower = med_name.lower()
    for item in plan_items:
        item_lower = item["name"].lower()
        if name_lower in item_lower or item_lower in name_lower:
            return item
    return {}


def get_required_pills_per_dose(plan_item: dict | None) -> int:
    """Pills required for a single scheduled/PRN dose (defaults to 1 when not explicit)."""
    if not plan_item:
        return 1
    pills = plan_item.get("pills_per_dose") or extract_pills_per_dose(plan_item)
    if pills is not None:
        try:
            count = int(pills)
            if 1 <= count <= 10:
                return count
        except (TypeError, ValueError):
            pass
    return 1


def get_plan_dose_label(plan_item: dict) -> str:
    if not plan_item:
        return "Dose quantity from discharge document"
    pills = plan_item.get("pills_per_dose") or extract_pills_per_dose(plan_item)
    timing = plan_item.get("time") or plan_item.get("timing") or ""
    if pills:
        label = f"Take {pills} pill{'s' if pills != 1 else ''}"
        if timing:
            label += f" at {timing}"
        return f"{label} · from discharge plan"
    if timing:
        return f"Scheduled at {timing} · quantity read from discharge document"
    return "Quantity per dose read from discharge document"


def parse_schedule_badge(timing: str) -> str:
    text = (timing or "").strip().lower()
    if not text:
        return "—"
    every_match = re.search(r"every\s+(\d+)\s*hours?", text)
    if every_match:
        return f"{every_match.group(1)}h"
    if "twice" in text or "two times" in text:
        return "2×/d"
    if "three times" in text or "tid" in text:
        return "3×/d"
    if "once daily" in text or "once a day" in text:
        return "1×/d"
    if "as needed" in text or "prn" in text:
        return "PRN"
    if "bedtime" in text or "at bedtime" in text or "at night" in text:
        return "9pm"
    if "breakfast" in text:
        return "8am"
    if "lunch" in text:
        return "12pm"
    if "dinner" in text:
        return "6pm"
    clock_match = re.search(r"(\d{1,2})\s*:\s*(\d{2})\s*(am|pm)?", text, re.I)
    if clock_match:
        hour = int(clock_match.group(1))
        minute = clock_match.group(2)
        meridiem = (clock_match.group(3) or "").lower()
        if meridiem == "pm" and hour != 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        elif not meridiem and hour >= 24:
            hour = hour % 24
        suffix = "am" if hour < 12 else "pm"
        display_hour = hour % 12 or 12
        if minute == "00":
            return f"{display_hour}{suffix}"
        return f"{display_hour}:{minute}{suffix}"
    mer_match = re.search(r"(\d{1,2})\s*(am|pm)\b", text, re.I)
    if mer_match:
        return f"{mer_match.group(1)}{mer_match.group(2).lower()}"
    return text[:10]


def build_schedule_badge_html(plan_item: dict, include_pill_count: bool = True) -> str:
    timing = (plan_item or {}).get("time") or (plan_item or {}).get("timing") or ""
    badge = html.escape(parse_schedule_badge(timing))
    count_html = ""
    if include_pill_count:
        pills = (plan_item or {}).get("pills_per_dose") or extract_pills_per_dose(plan_item or {})
        if pills:
            count_html = f'<span class="cs-sched-badge-count">{pills}×</span>'
    return (
        f'<span class="cs-sched-badge">{count_html}'
        f'{cs_icon("pill", 12)}'
        f'<span class="cs-sched-badge-time">{badge}</span></span>'
    )


def format_ref_strength_short(meta: dict) -> str:
    strength = meta.get("pill_strength")
    unit = meta.get("strength_unit") or "mg"
    if strength:
        return f"{strength:g}{unit}"
    return ""


def render_medication_checkin_banner(plan_items: list, patient_id) -> None:
    nudges = get_actionable_dose_nudges(plan_items, patient_id)
    banner_html = build_missed_dose_nudge_html(nudges, sticky=True)
    if banner_html:
        md_html(banner_html)


def build_plan_dose_prompt_block(
    plan_items: list,
    latest_plan: dict | None = None,
    med_refs: list | None = None,
) -> str:
    lines = ["Medication dose rules from the discharge plan:"]
    for med in plan_items:
        pills = get_effective_pills_per_dose(med["name"], med, med_refs)
        pill_text = (
            f"{pills} pill(s) per dose"
            if pills is not None
            else "quantity not explicit — infer from document text"
        )
        timing = med.get("time") or med.get("timing") or "as prescribed"
        dosage = med.get("dosage") or ""
        line = f"- {med['name']}: {pill_text}, scheduled {timing}"
        if dosage:
            line += f", strength note: {dosage}"
        lines.append(line)
    if latest_plan and latest_plan.get("raw_text"):
        raw = str(latest_plan["raw_text"]).strip()
        if raw:
            lines.append("\nFull discharge document excerpt:")
            lines.append(raw[:3500])
    return "\n".join(lines)


def med_slug(name: str) -> str:
    return re.sub(r"[^\w]+", "_", name.strip()).strip("_")[:48] or "med"


def guess_med_strength(name: str, dosage: str = ""):
    text = f"{name} {dosage}"
    match = re.search(r"(\d+(?:\.\d+)?)\s*(mg|g|mcg|µg)", text, re.I)
    if match:
        return float(match.group(1)), match.group(2).lower().replace("µg", "mcg")
    return None, "mg"


INJECTABLE_KEYWORDS = (
    "enoxaparin", "clexane", "heparin", "insulin", "liraglutide", "ozempic",
    "victoza", "adalimumab", "humira", "injection", "injectable", "prefilled syringe",
    "syringe", "subcutaneous", "sub-cut", "intramuscular", " i.m.", "pen device",
)

STRENGTH_TO_MG = {"mg": 1.0, "g": 1000.0, "mcg": 0.001}


def normalize_strength_unit(unit: str) -> str:
    normalized = (unit or "mg").lower().replace("µg", "mcg")
    return normalized if normalized in STRENGTH_TO_MG else "mg"


def strength_in_mg(amount: float | None, unit: str) -> float | None:
    if amount is None:
        return None
    factor = STRENGTH_TO_MG.get(normalize_strength_unit(unit))
    if factor is None:
        return None
    return float(amount) * factor


def format_strength_amount(amount: float, unit: str) -> str:
    return f"{amount:g}{normalize_strength_unit(unit)}"


def format_strength_spoken(amount: float, unit: str) -> str:
    return f"{amount:g} {normalize_strength_unit(unit)}"


def format_prescribed_dose_label(total_mg: float, preferred_unit: str) -> str:
    unit = normalize_strength_unit(preferred_unit)
    if unit == "g":
        return f"{total_mg / 1000:g} g"
    if unit == "mcg":
        return f"{total_mg * 1000:g} mcg"
    return f"{total_mg:g} mg"


def is_injectable_medication(med_name: str, plan_item: dict | None = None) -> bool:
    plan = plan_item or {}
    text = " ".join(
        str(value or "")
        for value in (
            med_name,
            plan.get("name"),
            plan.get("dosage"),
            plan.get("timing"),
            plan.get("time"),
        )
    ).lower()
    return any(keyword in text for keyword in INJECTABLE_KEYWORDS)


def get_prescribed_dose_mg(plan_item: dict) -> tuple[float | None, str]:
    if not plan_item:
        return None, "mg"
    per_pill, unit = guess_med_strength(
        str(plan_item.get("name") or ""),
        str(plan_item.get("dosage") or ""),
    )
    pills = plan_item.get("pills_per_dose") or extract_pills_per_dose(plan_item) or 1
    if per_pill is None:
        return None, unit
    return strength_in_mg(per_pill, unit) * pills, normalize_strength_unit(unit)


def classify_dose_quantity(qty: float) -> str:
    """whole = clean integer multiple; half = 0.5 or 1.5 etc.; blocked = anything else."""
    if abs(qty - round(qty)) < 1e-6:
        return "whole"
    doubled = qty * 2
    if abs(doubled - round(doubled)) < 1e-6:
        return "half"
    return "blocked"


def format_quantity_multiplier(qty: float) -> str:
    if abs(qty - 0.5) < 1e-6:
        return "½"
    if abs(qty - round(qty)) < 1e-6:
        return str(int(round(qty)))
    return f"{qty:g}"


def format_quantity_spoken(qty: float) -> str:
    if abs(qty - round(qty)) < 1e-6:
        count = int(round(qty))
        return "1 pill" if count == 1 else f"{count} pills"
    if abs(qty - 0.5) < 1e-6:
        return "half a pill"
    if abs(qty - 1.5) < 1e-6:
        return "one and a half pills"
    return f"{qty:g} pills"


def format_dose_per_admin_badge(qty: float) -> str:
    multiplier = format_quantity_multiplier(qty)
    if multiplier == "½":
        return "½×/dose"
    return f"{multiplier}×/dose"


def format_dose_equation(qty: float, reg_amount: float, reg_unit: str, prescribed_mg: float, plan_unit: str) -> str:
    qty_label = format_quantity_multiplier(qty)
    reg_label = format_strength_amount(reg_amount, reg_unit)
    presc_label = format_prescribed_dose_label(prescribed_mg, plan_unit)
    tablet_word = "tablet" if qty_label in ("1", "½") else "tablets"
    return f"{qty_label} × {reg_label} {tablet_word} = {presc_label} prescribed"


SPLIT_TABLET_NOTE = (
    "Check the tablet can be split — avoid splitting if coated, capsule, or modified-release."
)
BLOCKED_DOSE_WARNING = (
    "This strength cannot be divided into a standard dose. Contact your pharmacist."
)
INJECTABLE_DOSE_WARNING = (
    "Strength differs from prescription. "
    "Do not adjust injectable doses without medical advice."
)


def calculate_registered_dose_fit(
    plan_item: dict | None,
    meta: dict | None,
    med_name: str = "",
) -> dict | None:
    meta = meta or {}
    plan = plan_item or {}
    med_name = med_name or str(plan.get("name") or "")

    reg_strength = meta.get("pill_strength")
    if reg_strength is None:
        return None

    reg_unit = normalize_strength_unit(meta.get("strength_unit") or "mg")
    reg_amount = float(reg_strength)
    reg_mg = strength_in_mg(reg_amount, reg_unit)
    prescribed_mg, plan_unit = get_prescribed_dose_mg(plan)

    if is_injectable_medication(med_name, plan):
        if prescribed_mg is None or reg_mg is None or abs(prescribed_mg - reg_mg) <= 1e-6:
            return None
        return {
            "status": "injectable",
            "badge": "",
            "equation": "",
            "note": "",
            "warnings": [INJECTABLE_DOSE_WARNING],
        }

    if prescribed_mg is None or reg_mg is None or reg_mg <= 0:
        return None

    qty = prescribed_mg / reg_mg
    dose_class = classify_dose_quantity(qty)

    if dose_class == "blocked":
        return {
            "status": "blocked",
            "quantity": qty,
            "badge": "",
            "equation": "",
            "note": "",
            "warnings": [BLOCKED_DOSE_WARNING],
        }

    equation = format_dose_equation(qty, reg_amount, reg_unit, prescribed_mg, plan_unit)
    result = {
        "status": dose_class,
        "quantity": qty,
        "badge": format_dose_per_admin_badge(qty),
        "equation": equation,
        "note": "",
        "warnings": [],
    }
    if dose_class == "half":
        result["note"] = SPLIT_TABLET_NOTE
    return result


def pills_per_dose_from_registered_fit(calc: dict | None) -> int | None:
    """Physical pill count for one dose from registration strength fit (Pill Registration logic)."""
    if not calc:
        return None
    status = calc.get("status")
    if status in ("blocked", "injectable"):
        return None
    qty = calc.get("quantity")
    if qty is None:
        return None
    try:
        qty_f = float(qty)
    except (TypeError, ValueError):
        return None
    if status == "whole":
        count = int(round(qty_f))
    elif status == "half":
        count = max(1, int(math.ceil(qty_f)))
    else:
        return None
    if 1 <= count <= 10:
        return count
    return None


def find_medication_reference(med_name: str, med_refs: list | None) -> dict | None:
    """Best registration row for a plan medication name (handles duplicates / legacy rows)."""
    if not med_name or not med_refs:
        return None

    exact = [ref for ref in med_refs if ref.get("medication_name") == med_name]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return _pick_preferred_medication_reference(exact)

    med_lower = med_name.strip().lower()
    fuzzy = []
    for ref in med_refs:
        ref_name = str(ref.get("medication_name") or "").strip()
        ref_lower = ref_name.lower()
        if ref_lower == med_lower or ref_lower in med_lower or med_lower in ref_lower:
            fuzzy.append(ref)
    if not fuzzy:
        return None
    if len(fuzzy) == 1:
        return fuzzy[0]
    return _pick_preferred_medication_reference(fuzzy)


def filter_med_refs_for_plan(med_refs: list | None, plan_items: list) -> list:
    """Keep only registration rows that match medications in the active discharge plan."""
    plan_refs = []
    seen_ids = set()
    for med in plan_items:
        ref = find_medication_reference(med["name"], med_refs)
        ref_id = ref.get("id") if ref else None
        if ref and ref_id not in seen_ids:
            plan_refs.append(ref)
            seen_ids.add(ref_id)
    return plan_refs


def get_plan_registration_status(plan_items: list, med_refs: list | None) -> dict:
    """Registration counts scoped to the active discharge plan only."""
    refs_by_name = {}
    unregistered = []
    for med in plan_items:
        ref = find_medication_reference(med["name"], med_refs)
        if ref:
            refs_by_name[med["name"]] = ref
        else:
            unregistered.append(med)
    total = len(plan_items)
    registered_count = total - len(unregistered)
    return {
        "registered_count": registered_count,
        "total": total,
        "all_registered": registered_count == total and total > 0,
        "unregistered": unregistered,
        "refs_by_name": refs_by_name,
        "plan_refs": list(refs_by_name.values()),
    }


def _pick_preferred_medication_reference(candidates: list) -> dict:
    """Prefer JSON strength metadata and the most recently saved reference row."""
    with_strength = [
        ref for ref in candidates
        if parse_ref_meta(ref).get("pill_strength") is not None
    ]
    pool = with_strength or candidates
    return max(
        pool,
        key=lambda ref: (
            str(ref.get("updated_at") or ref.get("created_at") or ""),
            ref.get("id") or 0,
        ),
    )


def load_live_medication_references(patient_id=None) -> list:
    """Uncached registration rows from Supabase — always call at MedCam check time."""
    return cached_medication_references(patient_id)


def build_registration_meta(
    *,
    pill_strength: float,
    strength_unit: str,
    brand: str,
    plan_item: dict | None,
    med_name: str,
    back_image_b64: str | None = None,
) -> dict:
    meta = {
        "pill_strength": float(pill_strength),
        "strength_unit": strength_unit,
        "brand": (brand or "").strip(),
    }
    if back_image_b64:
        meta["back_image_b64"] = back_image_b64
    pills = pills_per_dose_from_registered_fit(
        calculate_registered_dose_fit(plan_item, meta, med_name),
    )
    if pills is not None:
        meta["pills_per_dose"] = pills
    return meta


def get_registered_pills_per_dose(
    med_name: str,
    plan_item: dict | None,
    med_refs: list | None,
) -> int | None:
    """Pills per dose from live registration strength + current discharge plan."""
    ref = find_medication_reference(med_name, med_refs)
    if not ref:
        return None

    meta = parse_ref_meta(ref)
    reg_strength = meta.get("pill_strength")
    if reg_strength is None:
        guessed_strength, guessed_unit = guess_med_strength(
            med_name,
            str((plan_item or {}).get("dosage") or ""),
        )
        if guessed_strength is not None:
            meta = {
                **meta,
                "pill_strength": guessed_strength,
                "strength_unit": guessed_unit,
            }
        else:
            stored = meta.get("pills_per_dose")
            if stored is not None:
                try:
                    count = int(stored)
                    if 1 <= count <= 10:
                        return count
                except (TypeError, ValueError):
                    pass
            return None

    calc = calculate_registered_dose_fit(plan_item, meta, med_name)
    computed = pills_per_dose_from_registered_fit(calc)
    if computed is not None:
        return computed

    stored = meta.get("pills_per_dose")
    if stored is not None:
        try:
            count = int(stored)
            if 1 <= count <= 10:
                return count
        except (TypeError, ValueError):
            pass
    return None


def get_effective_pills_per_dose(
    med_name: str,
    plan_item: dict | None,
    med_refs: list | None = None,
    *,
    patient_id=None,
) -> int | None:
    """
    Pills required for one MedCam dose check.
    Registered medications: always derived from live registration + plan (never discharge text).
    Unregistered medications: discharge plan text only.
    """
    if patient_id is not None:
        med_refs = load_live_medication_references(patient_id)
    elif med_refs is None:
        med_refs = []

    ref = find_medication_reference(med_name, med_refs)
    if ref:
        registered = get_registered_pills_per_dose(med_name, plan_item, med_refs)
        if registered is not None:
            return registered
        return None

    return get_required_pills_per_dose(plan_item)


NON_SPLITTABLE_KEYWORDS = (
    "capsule", "capsules", " cap ", " caps ", "enteric", "enteric-coated", "enteric coated",
    "coated tablet", "film-coated", "film coated", "gastro-resistant", "gastro resistant",
    "modified-release", "modified release", "slow-release", "slow release",
    "extended-release", "extended release", " prolonged", " retard", " sr ", " mr ", " xl ",
    "ec tablet", "gr tablet", "drc",
)

NON_SPLITTABLE_FORM_NOTE = (
    "This medicine is a capsule or coated/modified-release tablet and must not be split. "
    "Ask your pharmacist for a strength that matches the prescription."
)


def is_non_splittable_form(med_name: str, plan_item: dict | None = None) -> bool:
    plan = plan_item or {}
    text = " ".join(
        str(value or "")
        for value in (med_name, plan.get("name"), plan.get("dosage"), plan.get("timing"), plan.get("time"))
    ).lower()
    return any(keyword in text for keyword in NON_SPLITTABLE_KEYWORDS)


def build_blocked_strength_message(
    qty: float,
    reg_amount: float,
    reg_unit: str,
    prescribed_mg: float,
    plan_unit: str,
) -> str:
    reg_label = format_strength_spoken(reg_amount, reg_unit)
    presc_label = format_prescribed_dose_label(prescribed_mg, plan_unit)
    reg_mg = strength_in_mg(reg_amount, reg_unit) or 0
    pills_floor = int(qty) if qty >= 1 else 0
    if pills_floor >= 1 and reg_mg:
        achieved_label = format_prescribed_dose_label(pills_floor * reg_mg, plan_unit)
        pill_word = "pill" if pills_floor == 1 else "pills"
        return (
            f"The patient needs {presc_label} per dose, but each pill is {reg_label}. "
            f"Even taking {pills_floor} {pill_word} ({achieved_label}) would not reach {presc_label}. "
            f"Ask your pharmacist for a pill with a different strength."
        )
    return (
        f"The patient needs {presc_label} per dose, but each pill is {reg_label}. "
        f"You would need {qty:g} pills per dose, which is not possible. "
        f"Ask your pharmacist for a pill with a different strength."
    )


def build_pill_reg_dose_instruction_text(
    plan_item: dict | None,
    meta: dict | None,
    med_name: str = "",
    is_registered: bool = False,
) -> dict:
    plan = plan_item or {}
    meta = meta or {}
    med_name = med_name or str(plan.get("name") or "")
    timing = (plan.get("time") or plan.get("timing") or "").strip()
    non_splittable = is_non_splittable_form(med_name, plan)

    if not is_registered:
        per_pill, unit = guess_med_strength(str(plan.get("name") or ""), str(plan.get("dosage") or ""))
        pills = plan.get("pills_per_dose") or extract_pills_per_dose(plan) or 1
        frequency = format_medication_frequency(plan, timing)
        if per_pill is not None:
            strength_label = format_strength_spoken(per_pill, unit)
            pill_phrase = "1 pill" if pills == 1 else f"{pills} pills"
            instruction = finish_dose_instruction(
                f"Take {pill_phrase} of {strength_label}",
                timing,
                plan,
            )
        elif frequency:
            instruction = (
                f"Take as prescribed, {frequency} — register this pill to confirm strength."
            )
        else:
            instruction = "Register this pill to see dose instructions from your discharge plan."
        return {"instruction": instruction, "notes": [], "warnings": []}

    reg_strength = meta.get("pill_strength")
    reg_unit = normalize_strength_unit(meta.get("strength_unit") or "mg")
    if reg_strength is None:
        guessed_strength, guessed_unit = guess_med_strength(med_name, str(plan.get("dosage") or ""))
        if guessed_strength is not None:
            reg_strength = guessed_strength
            reg_unit = normalize_strength_unit(guessed_unit)
    if reg_strength is None:
        return {
            "instruction": finish_dose_instruction(
                "Registered — open Edit pill to confirm strength and see dose instructions",
                timing,
                plan,
            ),
            "notes": [],
            "warnings": [],
        }

    reg_amount = float(reg_strength)
    reg_label = format_strength_spoken(reg_amount, reg_unit)
    meta = {**meta, "pill_strength": reg_amount, "strength_unit": reg_unit}
    prescribed_mg, plan_unit = get_prescribed_dose_mg(plan)
    presc_label = format_prescribed_dose_label(prescribed_mg, plan_unit) if prescribed_mg else ""

    calc = calculate_registered_dose_fit(plan_item, meta, med_name)
    if calc and calc.get("status") == "injectable":
        return {
            "instruction": "",
            "notes": [],
            "warnings": [
                "Strength differs from prescription. "
                "Do not adjust injectable doses without medical advice."
            ],
        }

    if calc and calc.get("status") == "blocked":
        qty = calc.get("quantity", 0)
        warning = build_blocked_strength_message(qty, reg_amount, reg_unit, prescribed_mg, plan_unit)
        if non_splittable:
            warning = f"{warning} Capsules and coated or modified-release tablets cannot be split."
        return {"instruction": "", "notes": [], "warnings": [warning]}

    if calc is None:
        instruction = finish_dose_instruction(f"Take 1 pill of {reg_label}", timing, plan)
        if presc_label:
            instruction = finish_dose_instruction(
                f"Take 1 pill of {reg_label} to meet the {presc_label} strength requirement",
                timing,
                plan,
            )
        return {"instruction": instruction, "notes": [], "warnings": []}

    qty = calc.get("quantity", 1)
    status = calc.get("status")
    notes = []
    warnings = []

    if status == "half":
        if non_splittable:
            warnings.append(
                f"The prescription requires {presc_label} per dose, but your registered pill is "
                f"{reg_label}. This medicine is a capsule or coated/modified-release tablet and "
                f"cannot be split. Ask your pharmacist for a strength that matches the prescription."
            )
            return {"instruction": "", "notes": [], "warnings": warnings}
        quantity_phrase = format_quantity_spoken(qty)
        instruction = finish_dose_instruction(
            f"Take {quantity_phrase} of {reg_label} to meet the {presc_label} strength requirement",
            timing,
            plan,
        )
        notes.append(
            "Check the tablet can be split — avoid splitting if coated, capsule, or modified-release."
        )
        return {"instruction": instruction, "notes": notes, "warnings": warnings}

    count = int(round(qty))
    if count == 1:
        instruction = finish_dose_instruction(f"Take 1 pill of {reg_label}", timing, plan)
    else:
        instruction = finish_dose_instruction(
            f"Take {count} pills of {reg_label} to meet the {presc_label} strength requirement",
            timing,
            plan,
        )
    return {"instruction": instruction, "notes": notes, "warnings": warnings}


def build_pill_reg_dose_instruction_html(
    plan_item: dict | None,
    meta: dict | None,
    med_name: str = "",
    is_registered: bool = False,
) -> str:
    copy = build_pill_reg_dose_instruction_text(plan_item, meta, med_name, is_registered)
    parts = []
    if copy.get("instruction"):
        parts.append(
            f'<p class="cs-pill-reg-instruction">{html.escape(copy["instruction"])}</p>'
        )
    for note in copy.get("notes", []):
        parts.append(f'<p class="cs-pill-reg-note">{html.escape(note)}</p>')
    for warning in copy.get("warnings", []):
        parts.append(
            f'<p class="cs-pill-reg-alert">{cs_icon("warn", 14)}{html.escape(warning)}</p>'
        )
    return "".join(parts)


def build_dose_fit_html(plan_item: dict | None, meta: dict | None, med_name: str = "") -> tuple[str, str]:
    calc = calculate_registered_dose_fit(plan_item, meta, med_name)
    if not calc:
        return "", ""
    badge_html = ""
    if calc.get("badge"):
        badge_html = f'<span class="cs-dose-fit-badge">{html.escape(calc["badge"])}</span>'
    detail_parts = []
    if calc.get("equation"):
        detail_parts.append(
            f'<div class="cs-dose-fit-equation">{html.escape(calc["equation"])}</div>'
        )
    if calc.get("note"):
        detail_parts.append(f'<div class="cs-dose-fit-note">{html.escape(calc["note"])}</div>')
    for warning in calc.get("warnings", []):
        detail_parts.append(f'<div class="cs-dose-fit-warning">{html.escape(warning)}</div>')
    detail_html = f'<div class="cs-dose-fit">{"".join(detail_parts)}</div>' if detail_parts else ""
    return badge_html, detail_html


def parse_ref_meta(ref: dict) -> dict:
    raw = ref.get("description") or ""
    try:
        meta = json.loads(raw)
        if isinstance(meta, dict):
            return meta
    except (json.JSONDecodeError, TypeError):
        pass
    if raw and raw not in ("Registered via MedCam setup", ""):
        return {"brand": raw}
    return {}


def format_ref_meta_label(meta: dict, plan_item: dict | None = None) -> str:
    strength = meta.get("pill_strength")
    unit = meta.get("strength_unit") or "mg"
    brand = (meta.get("brand") or "").strip()
    parts = []
    if strength:
        parts.append(f"{strength:g}{unit} per pill")
    if brand:
        parts.append(f"brand: {brand}")
    if plan_item is not None:
        dose_label = get_plan_dose_label(plan_item)
        if parts:
            return " · ".join(parts) + f" · {dose_label}"
        return dose_label
    return " · ".join(parts) if parts else ""


def build_med_ref_registry_prompt(med_refs: list, plan_items: list) -> str:
    blocks = []
    for med in plan_items:
        ref = find_medication_reference(med["name"], med_refs)
        if not ref:
            continue
        meta = parse_ref_meta(ref)
        plan = med
        default_strength, default_unit = guess_med_strength(
            med["name"],
            str(plan.get("dosage") or ""),
        )
        strength = meta.get("pill_strength") or default_strength
        unit = meta.get("strength_unit") or default_unit
        brand = (meta.get("brand") or "").strip()
        pills = get_effective_pills_per_dose(med["name"], plan, med_refs)
        timing = plan.get("time") or plan.get("timing") or ""
        line = f"- {med['name']}: registered reference photo"
        if strength:
            line += f"; each pill is {strength:g}{unit}"
        if pills is not None:
            line += f"; give {pills} pill(s) per dose"
        else:
            line += "; pill count per dose could not be derived from registration"
        if brand:
            line += f"; current brand: {brand}"
        if timing:
            line += f"; scheduled at {timing}"
        blocks.append(line)
    return "\n".join(blocks)


REGISTERED_PILL_ACCENTS = ["#5B8DEF", "#4CAF7D", "#9B7EDE", "#E8945A", "#3BA99C", "#E07070"]


def build_tab_how_to_use_html(instruction: str) -> str:
    safe_instruction = html.escape(instruction)
    return f"""
    <div class="cs-tab-howto">
      <div class="cs-tab-howto-title">How to use it?</div>
      <div class="cs-tab-howto-pill">{safe_instruction}</div>
    </div>
    """


PILL_REGISTRATION_PHOTO_INSTRUCTION = (
    "Upload 2 photos of the pill: the front and the back. Use a white or neutral background, "
    "natural light or good artificial light, and place the pill in the center with no surrounding "
    "objects. If it has an engraved code or number, make sure it is clearly visible — this is "
    "what helps most to identify it correctly."
)

PILL_REGISTRATION_EXAMPLE_IMAGE = Path(__file__).resolve().parent / "assets" / "pill-registration-example.png"


def build_pill_registration_photo_instruction_html() -> str:
    return html.escape(PILL_REGISTRATION_PHOTO_INSTRUCTION)


def render_pill_registration_example_image() -> None:
    if not PILL_REGISTRATION_EXAMPLE_IMAGE.is_file():
        return
    st.image(str(PILL_REGISTRATION_EXAMPLE_IMAGE), use_container_width=True)


def render_pill_reg_photo_guide_row() -> None:
    md_html('<div class="cs-pill-reg-photo-guide-shell">')
    guide_left, guide_right = st.columns(2, gap="medium")
    with guide_left:
        md_html(
            f'<div class="cs-pill-reg-photo-guide-card">'
            f'<p class="cs-pill-reg-photo-instruction">{build_pill_registration_photo_instruction_html()}</p>'
            f"</div>"
        )
    with guide_right:
        md_html('<div class="cs-pill-reg-example-wrap">')
        render_pill_registration_example_image()
        md_html("</div>")
    md_html("</div>")


def render_pill_reg_summary_with_photo_guide(
    plan_items: list,
    registered_count: int,
    total: int,
) -> None:
    md_html(build_pill_reg_summary_panel_html(plan_items, registered_count, total))
    render_pill_reg_photo_guide_row()


def build_pill_reg_upload_panel_shell_open_html() -> str:
    return '<div class="cs-pill-reg-upload-shell">'


def build_pill_reg_upload_panel_shell_close_html() -> str:
    return "</div>"


def build_pill_reg_upload_panel_header_html(label: str) -> str:
    return f'<div class="cs-pill-reg-upload-panel-label">{html.escape(label)}</div>'


def render_pill_registration_upload_panels(
    key_prefix: str,
    *,
    is_update: bool = False,
    existing_ref: dict | None = None,
    initial_meta: dict | None = None,
):
    md_html(build_pill_reg_upload_panel_shell_open_html())
    upload_front, upload_back = st.columns(2, gap="small")
    with upload_front:
        md_html(build_pill_reg_upload_panel_header_html("FRONT VIEW (SIDE A)"))
        ref_photo_front = st.file_uploader(
            "Front view (Side A)",
            type=["jpg", "jpeg", "png"],
            key=f"{key_prefix}_photo_front",
            label_visibility="collapsed",
        )
        if ref_photo_front:
            st.image(ref_photo_front, use_container_width=True, caption="Front view preview")
        elif is_update and existing_ref and existing_ref.get("image_b64"):
            st.image(
                f"data:image/jpeg;base64,{existing_ref['image_b64']}",
                use_container_width=True,
                caption="Current front photo on file",
            )
    with upload_back:
        md_html(build_pill_reg_upload_panel_header_html("REAR VIEW (SIDE B)"))
        ref_photo_back = st.file_uploader(
            "Back view (Side B)",
            type=["jpg", "jpeg", "png"],
            key=f"{key_prefix}_photo_back",
            label_visibility="collapsed",
        )
        if ref_photo_back:
            st.image(ref_photo_back, use_container_width=True, caption="Back view preview")
        elif is_update and initial_meta and initial_meta.get("back_image_b64"):
            st.image(
                f"data:image/jpeg;base64,{initial_meta['back_image_b64']}",
                use_container_width=True,
                caption="Current back photo on file",
            )
    md_html(build_pill_reg_upload_panel_shell_close_html())
    return ref_photo_front, ref_photo_back


def encode_uploaded_photo(uploaded_file) -> str | None:
    if uploaded_file is None:
        return None
    return base64.b64encode(uploaded_file.getvalue()).decode("utf-8")


def build_pill_reg_how_to_use_html() -> str:
    return build_tab_how_to_use_html(
        "Tap a card to register a medication: upload front and back photos of the pill, then confirm "
        "the strength (mg) and brand. Clear photos on a plain background help MedCam identify pills correctly."
    )


def build_medcam_how_to_use_html() -> str:
    return build_tab_how_to_use_html(
        "Click 'Upload' to submit a real-time photo of the medication right before administering it to the "
        "patient. CareShield AI will instantly verify the pill against schedules and registered data to provide "
        "a final safety check and automatically log adherence for future handovers."
    )


def build_documents_how_to_use_html() -> str:
    return build_tab_how_to_use_html(
        'Click the "Upload Document" button to submit medical paperwork like doctor\'s letters, care plans, '
        "or hospital discharge summaries. This allows the system to securely scan and save the patient's "
        "conditions and medications, creating a complete health picture so CareShield AI can better understand "
        "and support their daily care."
    )


def build_report_ask_how_to_use_html() -> str:
    return build_tab_how_to_use_html(
        "Simply type a message in the chatbox to ask a question or share a health update—like a new symptom, "
        "a behavioral change, or an image of a rash. Using advanced text and image recognition, the system "
        "automatically logs the entire timeline, assesses the severity of potential health risks to urgently "
        "prompt an emergency call if needed, and provides safe next steps tailored entirely to the patient's "
        "existing conditions and medications."
    )


def build_handover_how_to_use_html() -> str:
    return build_tab_how_to_use_html(
        "Select the tracking period that matches your appointment, then generate the SBAR handover report. "
        "Review the charts below for adherence and symptom patterns, then download the PDF to share with "
        "the clinician."
    )


def build_my_results_how_to_use_html() -> str:
    return build_tab_how_to_use_html(
        "Upload any medical document from an appointment or test — blood test printouts, scan reports, or "
        "clinic letters. CareShield explains what matters most — new diagnoses, medication changes, follow-ups, "
        "red-flag instructions, and flagged lab values — in plain language, then suggests questions to ask the doctor."
    )


def build_pill_reg_active_plan_html(plan_items: list) -> str:
    rows = []
    for item in plan_items:
        name = html.escape(str(item.get("name") or ""))
        dosage = str(item.get("dosage") or "").strip()
        if dosage:
            label = f"{name} ({html.escape(dosage)})"
        else:
            label = name
        timing_part = html.escape(format_plan_schedule_summary(item))
        rows.append(f'<li><strong>{label}:</strong> {timing_part}</li>')
    return f"""
    <div class="cs-pill-reg-active-plan">
      <div class="cs-active-plan-label">Active plan</div>
      <ul class="cs-pill-reg-plan-list">{"".join(rows)}</ul>
    </div>
    """


def build_pill_reg_summary_panel_html(
    plan_items: list,
    registered_count: int,
    total: int,
) -> str:
    return f"""
    <div class="cs-pill-reg-summary">
      {build_registration_progress_ring_html(registered_count, total)}
      {build_pill_reg_active_plan_html(plan_items)}
    </div>
    """


def build_pill_reg_success_banner_html() -> str:
    return (
        '<div class="cs-pill-reg-success-banner">'
        "All medications registered. Switch to the MedCam tab to verify doses."
        "</div>"
    )


def build_registration_progress_ring_html(registered_count: int, total: int) -> str:
    if total <= 0:
        return ""
    pct = min(max(registered_count / total, 0.0), 1.0)
    radius = 52
    circumference = 2 * math.pi * radius
    dash = circumference * pct
    gap = max(circumference - dash, 0.001)
    return f"""
    <div class="cs-pill-reg-progress">
      <svg class="cs-reg-progress-ring" viewBox="0 0 128 128" aria-hidden="true">
        <circle cx="64" cy="64" r="{radius}" fill="none" stroke="#E8E4DA" stroke-width="11"></circle>
        <circle cx="64" cy="64" r="{radius}" fill="none" stroke="#4CAF7D" stroke-width="11"
          stroke-dasharray="{dash:.2f} {gap:.2f}" stroke-linecap="round"
          transform="rotate(-90 64 64)"></circle>
        <text x="64" y="56" text-anchor="middle" class="cs-reg-progress-num">{registered_count}/{total}</text>
        <text x="64" y="74" text-anchor="middle" class="cs-reg-progress-label">REGISTERED</text>
      </svg>
    </div>
    """


def build_pill_grid_card_html(
    med: dict,
    is_registered: bool,
    meta: dict | None = None,
    plan_item: dict | None = None,
    ref: dict | None = None,
) -> str:
    name = html.escape(med["name"])
    plan = plan_item or med
    state_class = "cs-pill-grid-card--done" if is_registered else "cs-pill-grid-card--pending"
    dose_instruction_html = build_pill_reg_dose_instruction_html(
        plan, meta, med.get("name"), is_registered=is_registered,
    )
    if is_registered and not dose_instruction_html.strip():
        safe_meta = meta or {}
        if safe_meta.get("pill_strength"):
            strength_text = format_strength_spoken(
                float(safe_meta["pill_strength"]),
                safe_meta.get("strength_unit") or "mg",
            )
            fallback = finish_dose_instruction(
                f"Take 1 pill of {strength_text}",
                plan.get("time") or plan.get("timing") or "",
                plan,
            )
        else:
            fallback = finish_dose_instruction(
                "Take as prescribed",
                plan.get("time") or plan.get("timing") or "",
                plan,
            )
        dose_instruction_html = f'<p class="cs-pill-reg-instruction">{html.escape(fallback)}</p>'

    if is_registered:
        status = f'<span class="cs-pill-grid-status cs-pill-grid-status--done">{cs_icon("check", 13)} Registered</span>'
        if ref and ref.get("image_b64"):
            visual = (
                f'<img class="cs-pill-grid-thumb" '
                f'src="data:image/jpeg;base64,{ref["image_b64"]}" '
                f'alt="{name} reference photo">'
            )
        else:
            visual = f'<div class="cs-pill-grid-thumb cs-pill-grid-thumb--fallback">{cs_icon("pill", 28)}</div>'
    else:
        status = '<span class="cs-pill-grid-status">Tap to register</span>'
        visual = (
            '<div class="cs-pill-grid-thumb cs-pill-grid-thumb--empty">'
            '<span class="cs-pill-grid-empty-icon">+</span>'
            '<span class="cs-pill-grid-empty-label">Add photos</span>'
            '</div>'
        )

    return f"""
    <div class="cs-pill-grid-card {state_class}">
      <div class="cs-pill-grid-layout">
        <div class="cs-pill-grid-visual">{visual}</div>
        <div class="cs-pill-grid-content">
          <div class="cs-pill-grid-top">{status}</div>
          <div class="cs-pill-grid-name">{name}</div>
          {dose_instruction_html}
        </div>
      </div>
    </div>
    """


def build_registered_pill_card_html(ref: dict, meta: dict, plan_item: dict, index: int) -> str:
    accent = REGISTERED_PILL_ACCENTS[index % len(REGISTERED_PILL_ACCENTS)]
    dose_instruction_html = build_pill_reg_dose_instruction_html(
        plan_item, meta, ref.get("medication_name"), is_registered=True,
    )
    meta_label = html.escape(format_ref_meta_label(meta, None))
    thumb_html = ""
    if ref.get("image_b64"):
        thumb_html = (
            f'<img class="cs-registered-pill-thumb" '
            f'src="data:image/jpeg;base64,{ref["image_b64"]}" '
            f'alt="{html.escape(ref["medication_name"])}">'
        )
    return f"""
    <div class="cs-registered-pill-card" style="border-left-color: {accent};">
      <div class="cs-registered-pill-layout">
        {thumb_html or f'<div class="cs-registered-pill-thumb cs-registered-pill-thumb--fallback">{cs_icon("pill", 30)}</div>'}
        <div class="cs-registered-pill-body">
          <div class="cs-registered-pill-name">{html.escape(ref['medication_name'])}</div>
          <div class="cs-registered-pill-meta">{meta_label}</div>
          {dose_instruction_html}
        </div>
      </div>
    </div>
    """


def render_pill_registration_form(
    med: dict,
    key_prefix: str,
    is_update: bool = False,
    initial_meta: dict | None = None,
    existing_ref: dict | None = None,
):
    default_strength, default_unit = guess_med_strength(
        med["name"],
        str(med.get("dosage") or ""),
    )
    if initial_meta:
        if initial_meta.get("pill_strength") is not None:
            default_strength = float(initial_meta["pill_strength"])
        if initial_meta.get("strength_unit") in ("mg", "g", "mcg"):
            default_unit = initial_meta["strength_unit"]
    initial_brand = (initial_meta or {}).get("brand") or ""
    title = "Update reference" if is_update else "Register this medication"
    md_html(f"""
    <div class="cs-enroll-card">
      <div class="cs-enroll-card-title">{html.escape(med['name'])}</div>
      <p class="cs-enroll-card-hint">
        {'Upload new front and back photos if the brand or packaging changed.' if is_update else
         'Confirm the strength per pill from your discharge document after uploading both sides.'}
      </p>
    </div>
    """)
    ref_photo_front, ref_photo_back = render_pill_registration_upload_panels(
        key_prefix,
        is_update=is_update,
        existing_ref=existing_ref,
        initial_meta=initial_meta,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        strength = st.number_input(
            "Strength per pill",
            min_value=0.0,
            value=float(default_strength or 500.0),
            step=0.5,
            format="%.1f",
            key=f"{key_prefix}_strength",
            help="Amount of active ingredient in each pill, e.g. 500 for Paracetamol 500mg.",
        )
    with col_b:
        unit = st.selectbox(
            "Unit",
            ["mg", "g", "mcg"],
            index=["mg", "g", "mcg"].index(default_unit if default_unit in ("mg", "g", "mcg") else "mg"),
            key=f"{key_prefix}_unit",
        )
    brand = st.text_input(
        "Brand or pharmacy (optional)",
        value=initial_brand,
        placeholder="e.g. Calpol, generic white round tablet",
        key=f"{key_prefix}_brand",
    )
    btn_label = "Save updated reference" if is_update else f"Save {med['name']}"
    save_clicked = st.button(btn_label, key=f"{key_prefix}_save", use_container_width=True, type="primary")
    return save_clicked, ref_photo_front, ref_photo_back, strength, unit, brand


@st.dialog("Pill registration", width="large")
def pill_registration_dialog():
    modal = st.session_state.get("pill_modal") or {}
    med_name = modal.get("med_name")
    mode = modal.get("mode", "register")
    if not med_name:
        st.warning("No medication selected.")
        if st.button("Close", use_container_width=True):
            st.session_state.pop("pill_modal", None)
            st.rerun()
        return

    from ai_helpers import (
        upsert_medication_reference,
        update_medication_reference,
        delete_medication_reference,
        get_medication_references,
        get_patient_plans,
    )

    latest_plan = None
    plans = cached_patient_plans(st.session_state.selected_patient_id)
    if plans:
        latest_plan = plans[0]
    plan_items = get_active_plan_items(st.session_state.selected_patient_id)

    med = next((item for item in plan_items if item["name"] == med_name), {"name": med_name})

    med_refs = cached_medication_references(st.session_state.selected_patient_id)
    ref = find_medication_reference(med_name, med_refs)
    slug = med_slug(med_name)
    is_update = mode == "edit" and ref is not None
    initial_meta = parse_ref_meta(ref) if ref else None

    save_clicked, ref_photo_front, ref_photo_back, strength, unit, brand = render_pill_registration_form(
        med,
        f"modal_{slug}",
        is_update=is_update,
        initial_meta=initial_meta,
        existing_ref=ref,
    )

    action_cols = st.columns(2)
    with action_cols[0]:
        if st.button("Close", use_container_width=True, key=f"modal_close_{slug}"):
            st.session_state.pop("pill_modal", None)
            st.rerun()
    with action_cols[1]:
        remove_clicked = False
        if is_update and ref:
            remove_clicked = st.button(
                "Remove registration",
                use_container_width=True,
                key=f"modal_remove_{slug}",
            )

    if remove_clicked and ref:
        removed_patient_id = st.session_state.selected_patient_id
        delete_medication_reference(ref["id"], removed_patient_id)
        st.session_state.pop("pill_modal", None)
        st.session_state.pop(f"medication_refs_{resolve_patient_id(removed_patient_id)}", None)
        invalidate_patient_activity_cache(removed_patient_id)
        st.success(f"Removed {med_name}.")
        st.rerun()

    if save_clicked:
        reg_meta = build_registration_meta(
            pill_strength=strength,
            strength_unit=unit,
            brand=brand,
            plan_item=med,
            med_name=med_name,
        )
        pills_per_dose = reg_meta.get("pills_per_dose")
        patient_id = st.session_state.selected_patient_id
        front_b64 = encode_uploaded_photo(ref_photo_front)
        back_b64 = encode_uploaded_photo(ref_photo_back)
        if is_update and ref:
            if not front_b64:
                front_b64 = ref.get("image_b64")
            if not back_b64:
                back_b64 = (initial_meta or {}).get("back_image_b64")
        if not front_b64 or not back_b64:
            st.warning("Please upload both a front and a back photo before saving.")
        elif is_update and ref:
            with st.spinner("Updating reference..."):
                saved = update_medication_reference(
                    ref_id=ref["id"],
                    image_b64=front_b64,
                    pill_strength=strength,
                    strength_unit=unit,
                    brand=brand,
                    pills_per_dose=pills_per_dose,
                    back_image_b64=back_b64,
                    patient_id=patient_id,
                )
            if not saved:
                st.error(f"Could not update {med_name}. Please try again.")
            else:
                st.session_state.pop("pill_modal", None)
                st.session_state.pop(f"medication_refs_{resolve_patient_id(patient_id)}", None)
                st.session_state.pop(f"medcam_last_{resolve_patient_id(patient_id)}", None)
                invalidate_patient_activity_cache(patient_id)
                st.success(f"Updated {med_name}.")
                st.rerun()
        else:
            with st.spinner("Saving reference..."):
                saved = upsert_medication_reference(
                    medication_name=med_name,
                    image_b64=front_b64,
                    pill_strength=strength,
                    strength_unit=unit,
                    brand=brand,
                    pills_per_dose=pills_per_dose,
                    patient_id=patient_id,
                    back_image_b64=back_b64,
                )
            if not saved:
                st.error(f"Could not save {med_name}. Please try again.")
            else:
                st.session_state.pop("pill_modal", None)
                st.session_state.pop(f"medication_refs_{resolve_patient_id(patient_id)}", None)
                st.session_state.pop(f"medcam_last_{resolve_patient_id(patient_id)}", None)
                invalidate_patient_activity_cache(patient_id)
                st.success(f"{med_name} registered.")
                st.rerun()


def build_plan_rows_html(items):
    rows = []
    for item in items:
        name = html.escape(str(item.get("name", "")))
        dosage = str(item.get("dosage", "") or "")
        timing = html.escape(str(item.get("time") or item.get("timing") or ""))
        dosage_html = ""
        if dosage and dosage not in str(item.get("name", "")):
            dosage_html = f'<div class="cs-plan-dosage">{html.escape(dosage)}</div>'
        rows.append(
            f'<div class="cs-plan-row">'
            f'<div class="cs-plan-left"><span class="cs-plan-med">{name}</span>{dosage_html}</div>'
            f'<span class="cs-plan-time">{timing}</span>'
            f'</div>'
        )
    return "".join(rows)


def finalize_chat_patient_facing_text(
    text: str,
    *,
    patient_id=None,
    user_text: str = "",
) -> str:
    return enforce_active_patient_name_in_text(
        coerce_ai_text(text),
        get_patient_display_name(patient_id),
        user_text,
    )


def finalize_condition_risk_alerts(
    alerts: list | None,
    *,
    patient_id=None,
    user_text: str = "",
) -> list:
    from ai_helpers import retailor_education_messages_to_symptom

    active_name = get_patient_display_name(patient_id)
    tailored = retailor_education_messages_to_symptom(
        user_text,
        alerts or [],
        active_name,
    )
    finalized = []
    for item in tailored:
        row = dict(item)
        row["education_message"] = enforce_active_patient_name_in_text(
            str(row.get("education_message") or ""),
            active_name,
            user_text,
        )
        finalized.append(row)
    return finalized


def build_patient_context_block(patient_id=None):
    patient_id = resolve_patient_id(patient_id or st.session_state.get("selected_patient_id"))
    active_patient_name = get_patient_display_name(patient_id)
    raw_meds = get_patient_medications_display(patient_id)

    med_lines = []
    for index, med in enumerate(raw_meds):
        display = _medication_item_from_dict(med, index)
        details = [display["name"]]
        if display.get("dosage") and str(display["dosage"]) not in str(display["name"]):
            details.append(f"strength {display['dosage']}")
        timing = display.get("time") or med.get("timing") or med.get("schedule")
        if timing:
            details.append(f"schedule {timing}")
        pills = display.get("pills_per_dose") or extract_pills_per_dose(med)
        if pills:
            details.append(f"{pills} pill(s) per dose")
        med_lines.append("- " + "; ".join(details))

    conditions = get_stored_conditions(patient_id)
    cond_lines = []
    for condition in conditions:
        _, status_label = condition_badge_meta(condition.get("badge", "chronic"))
        onset = normalize_condition_since(condition.get("since"))
        onset_part = f"; onset {onset}" if onset else ""
        cond_lines.append(
            f"- {condition['name']} ({status_label}){onset_part}"
        )

    allergy_lines = [f"- {item}" for item in get_patient_allergy_notes(patient_id)]
    plan_meta = get_patient_plan_meta(patient_id)
    discontinued_lines = []
    for item in plan_meta.get("discontinued") or []:
        name = str(item.get("name") or "").strip()
        if name:
            discontinued_lines.append(f"- {name} (discontinued)")

    document_lines = []
    for doc in fetch_recent_document_excerpts(patient_id):
        file_name = str(doc.get("file_name") or "Uploaded document").strip()
        excerpt = str(doc.get("excerpt") or "").strip()
        if excerpt:
            document_lines.append(f"- {file_name}: {excerpt}")

    meds_section = "\n".join(med_lines) if med_lines else "- None on file"
    cond_section = "\n".join(cond_lines) if cond_lines else "- None on file"
    allergy_section = "\n".join(allergy_lines) if allergy_lines else "- None documented in uploaded records"
    discontinued_section = "\n".join(discontinued_lines) if discontinued_lines else "- None on file"
    document_section = "\n".join(document_lines) if document_lines else "- No uploaded document excerpts on file"

    return f"""THIS IS A SINGLE-PATIENT FAMILY ACCOUNT — the patient below is always on file for this CareShield home.

ACTIVE PATIENT NAME (always use this exact name in responses — ignore any other name the caregiver types): {active_patient_name}

PATIENT STORED MEDICATIONS — always use this list for dosing, timing, and drug-specific advice:
{meds_section}

PATIENT STORED CONDITIONS — always use this list for diagnosis, risk, and care context:
{cond_section}

PATIENT ALLERGIES & ADVERSE REACTIONS (from uploaded documents):
{allergy_section}

DISCONTINUED MEDICATIONS (do not suggest restarting):
{discontinued_section}

UPLOADED MEDICAL DOCUMENT EXCERPTS (from hospital letters, discharge summaries, etc.):
{document_section}"""


def render_chat_messages_html(messages, caregiver_label: str, show_thinking: bool = False) -> str:
    parts = ['<div class="cs-chat-messages" id="cs-chat-scroll">']
    user_initial = caregiver_first_name(caregiver_label)[0].upper()

    for msg in messages:
        role_class = "user" if msg["role"] == "user" else "assistant"
        content = format_message_content(msg)
        if msg.get("has_image"):
            content += '<div class="cs-msg-photo-tag">Photo attached</div>'
        tag_html = ""
        if msg.get("severity"):
            tag_html = severity_header_html(
                msg["severity"],
                msg.get("context_report_count"),
            )
        condition_risk_html = build_condition_risk_alerts_html(msg.get("condition_risk_alerts"))
        emergency_call_html = ""
        if msg.get("severity") and is_emergency_severity(msg["severity"]):
            emergency_call_html = emergency_call_bar_html()
        time_html = ""
        if msg.get("timestamp_display"):
            time_html = f'<span class="cs-msg-time">{html.escape(msg["timestamp_display"])}</span>'

        if role_class == "assistant":
            rag_badge = (
                f'<span class="cs-msg-rag-badge">{cs_icon("document", 12)} Based on clinical guidelines</span>'
                if msg.get("used_rag")
                else ""
            )
            disclaimer = ""
            if not msg.get("welcome"):
                disclaimer = (
                    f'<div class="cs-msg-disclaimer">{cs_icon("warn", 14)}'
                    f'This is not medical advice. Always consult a healthcare professional.</div>'
                )
            parts.append(
                f'<div class="cs-msg {role_class}">'
                f'<div class="cs-msg-avatar">CS</div>'
                f'<div class="cs-msg-body">'
                f'<div class="cs-msg-meta"><span class="cs-msg-sender">CareShield</span>'
                f'<span class="cs-msg-badge">App</span>{time_html}{rag_badge}</div>'
                f'<div class="cs-msg-bubble">{tag_html}{content}{condition_risk_html}{emergency_call_html}{disclaimer}</div>'
                f'</div></div>'
            )
        else:
            parts.append(
                f'<div class="cs-msg {role_class}">'
                f'<div class="cs-msg-body">'
                f'<div class="cs-msg-meta cs-msg-meta-right">'
                f'{time_html}'
                f'<span class="cs-msg-sender">{html.escape(caregiver_first_name(caregiver_label))}</span>'
                f'</div>'
                f'<div class="cs-msg-bubble">{content}</div>'
                f'</div>'
                f'<div class="cs-msg-avatar cs-msg-avatar-user">{user_initial}</div>'
                f'</div>'
            )

    if show_thinking:
        parts.append("""
        <div class="cs-msg assistant cs-msg-thinking">
          <div class="cs-msg-avatar">CS</div>
          <div class="cs-msg-body">
            <div class="cs-msg-meta"><span class="cs-msg-sender">CareShield</span><span class="cs-msg-badge">App</span></div>
            <div class="cs-msg-bubble cs-thinking-bubble">
              <span class="cs-thinking-spinner"></span>
              CareShield Agent evaluating context...
            </div>
          </div>
        </div>
        """)

    parts.append('</div>')
    return "".join(parts)


def render_chat_auto_scroll() -> None:
    if not st.session_state.pop("chat_scroll_to_bottom", False):
        return
    components.html(
        """
        <script>
        (function () {
          function scrollChat() {
            const doc = window.parent.document;
            const el = doc.getElementById("cs-chat-scroll");
            if (!el) return;
            el.scrollTop = el.scrollHeight;
            const last = el.querySelector(".cs-msg:last-child");
            if (last) {
              last.scrollIntoView({ behavior: "smooth", block: "end" });
            }
          }
          [0, 80, 200, 450, 900].forEach(function (delay) {
            window.setTimeout(scrollChat, delay);
          });
        })();
        </script>
        """,
        height=0,
    )


NORMAL_PHOTO_REASSURANCE = (
    "Everything looks normal in this photo — I don't see anything concerning "
    "in the area shown. Keep monitoring as usual."
)


def ensure_normal_photo_reassurance(reply: str) -> str:
    reply = str(reply or "").strip()
    lower = reply.lower()
    reassuring_phrases = (
        "normal",
        "no concern",
        "looks okay",
        "looks ok",
        "nothing concerning",
        "all clear",
        "everything seems okay",
        "everything looks okay",
    )
    if any(phrase in lower for phrase in reassuring_phrases):
        return reply
    return f"{NORMAL_PHOTO_REASSURANCE}\n\n{reply}".strip()


def build_chat_photo_system_prompt() -> str:
    return f"""You are CareShield in the Report & Ask chat. The caregiver uploaded a photo.

{CHAT_SINGLE_PATIENT_RULES}

STEP 1 — Classify the uploaded photo (last image) as exactly one of:
- "pill": medicine tablets or capsules (pills in hand, on a surface, blister pack focused on pills, etc.)
- "symptom": a patient symptom or clinical sign (bruise, rash, wound, swelling, skin change, injury, etc.)
- "unsupported": anything else (selfies, food, pets, scenery, documents, random objects, unreadable blur, etc.)

STEP 2 — Respond based on type:

IF "unsupported":
- Set rejection_message explaining this chat ONLY accepts pill-identification photos or symptom-review photos — nothing else. Mention the MedCam tab for dose verification.
- Do NOT analyse the image. severity "ok", needs_doctor false.

IF "pill":
- You MUST identify pills using the registered reference photos when provided — NEVER refuse or say you cannot identify pills.
- Compare the uploaded photo to each reference image.
- If matched: name the medication, describe visible pills, mention registered strength if relevant.
- If pills are visible but no reference matches: say so and suggest Pill Registration or MedCam.
- If no references were provided: tell them to register pills in Pill Registration first.
- severity "ok" when confidently matched, otherwise "monitor".

IF "symptom":
- Analyse the clinical sign using the patient profile below.
- If the photo shows healthy skin or tissue with NO abnormal bruise, rash, swelling, wound, discharge, or skin change in the area shown: set clinical_finding to "normal", risk_level to "ok", needs_doctor false, and clearly reassure the caregiver that everything looks okay in what you can see.
- If there IS a concerning sign, describe it and set clinical_finding to "concern".
- Apply urgency rules for any concerning findings.

{CHAT_URGENCY_RULES}

{CHAT_SESSION_RULES}

Respond with ONLY a JSON object:
1. "photo_type": "pill" | "symptom" | "unsupported"
2. "analysis": warm, clear message for the caregiver
3. "rejection_message": string when unsupported, otherwise null
4. "risk_level": one of "ok", "monitor", "contact_doctor", or "emergency"
5. "needs_doctor": boolean
6. "comfort_note": optional short reassuring sentence
7. "matched_medication": medication name or null (pill type only)
8. "pills_detected": brief description of visible pills or null
9. "clinical_finding": "normal" | "concern" | null (symptom photos only)
"""


def build_chat_photo_ai_content(
    image_b64: str,
    med_refs: list,
    plan_items: list,
    *,
    context: str,
    session_context: str,
    user_text: str,
    current_time_str: str,
    medical_context: str,
) -> list:
    user_content = []
    plan_refs = filter_med_refs_for_plan(med_refs, plan_items)
    for ref in plan_refs:
        meta = parse_ref_meta(ref)
        plan_match = find_plan_item(ref["medication_name"], plan_items)
        meta_label = format_ref_meta_label(meta, plan_match)
        user_content.append({
            "type": "text",
            "text": f"Registered reference — {ref['medication_name']} ({meta_label}):",
        })
        if ref.get("image_b64"):
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{ref['image_b64']}"},
            })
        back_b64 = meta.get("back_image_b64")
        if back_b64:
            user_content.append({
                "type": "text",
                "text": f"Registered reference back view — {ref['medication_name']}:",
            })
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{back_b64}"},
            })

    registry_prompt = (
        build_med_ref_registry_prompt(med_refs, plan_items)
        if plan_refs
        else "No pill reference photos registered yet."
    )
    caregiver_note = user_text.strip() or "Photo uploaded for review"
    user_content.append({
        "type": "text",
        "text": (
            f"Patient profile:\n{context}\n\n"
            f"{session_context}\n\n"
            f"{('RELEVANT MEDICAL KNOWLEDGE:\\n' + medical_context + '\\n\\n') if medical_context else ''}"
            f"Current date/time: {current_time_str}\n\n"
            f"Registered pill references:\n{registry_prompt}\n\n"
            f"Caregiver message with uploaded photo: {caregiver_note}\n\n"
            "Classify the CAREGIVER UPLOADED PHOTO (the last image below) and respond per instructions.\n"
            "UPLOADED PHOTO FROM CAREGIVER:"
        ),
    })
    user_content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
    })
    return user_content


def process_chat_photo_response(
    *,
    caregiver_label: str,
    caregiver_id: str,
    image_b64: str,
    user_text: str,
    reported_at: str,
    timestamp_display: str,
    current_time_str: str,
    context: str,
    session_context: str,
    prior_incidents: list,
    medical_context: str,
) -> bool:
    from ai_helpers import ask_ai, get_medication_references

    patient_id = st.session_state.get("selected_patient_id")
    _report_ask_logger.info(
        "Report & Ask photo context patient=%s session_prior_count=%d user_preview=%r",
        patient_id,
        len(prior_incidents),
        (user_text or "")[:120],
    )

    med_refs = cached_medication_references(st.session_state.selected_patient_id)
    plan_items = get_active_plan_items()
    user_content = build_chat_photo_ai_content(
        image_b64,
        med_refs,
        plan_items,
        context=context,
        session_context=session_context,
        user_text=user_text,
        current_time_str=current_time_str,
        medical_context=medical_context,
    )
    result = ask_ai(build_chat_photo_system_prompt(), user_content)

    if result.get("error"):
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.get(
                "message",
                "We couldn't process this right now. Please try again.",
            ),
        })
        st.session_state.chat_scroll_to_bottom = True
        return False

    photo_type = str(result.get("photo_type") or "symptom").strip().lower()

    if photo_type == "unsupported":
        reply = (
            result.get("rejection_message")
            or result.get("analysis")
            or CHAT_UNSUPPORTED_PHOTO_DEFAULT
        )
        st.session_state.messages.append({
            "role": "assistant",
            "content": reply,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "timestamp_display": format_chat_timestamp(),
        })
        st.session_state.chat_scroll_to_bottom = True
        return True

    reply = str(result.get("analysis") or "").strip()
    matched = result.get("matched_medication")
    pills_detected = result.get("pills_detected")
    photo_finding = ""
    session_triggers: list[str] = []
    condition_risk_alerts: list = []

    if photo_type == "pill":
        if matched and str(matched).lower() not in reply.lower():
            reply = f"**{matched}** — {reply}" if reply else f"This looks like **{matched}**."
        if pills_detected and str(pills_detected).lower() not in reply.lower():
            reply = f"{reply}\n\nVisible pills: {pills_detected}".strip()
        if result.get("comfort_note"):
            reply += f"\n\n{result['comfort_note']}"
        severity = "ok" if matched else "monitor"
        if matched:
            severity = normalize_chat_severity(result.get("risk_level", "ok"))
        photo_finding = "identified" if matched else "unidentified"
        reply += SESSION_HANDOVER_NOTE
        source = "pill_photo"
        incident_text = user_text or f"Pill photo — {matched or 'identification requested'}"
        doctor_note = f"Pill photo: {matched or 'No registered match'} — {pills_detected or incident_text}"
        context_report_count = 0
    else:
        if result.get("comfort_note"):
            reply += f"\n\n{result['comfort_note']}"
        clinical_finding = str(result.get("clinical_finding") or "").strip().lower()
        if clinical_finding == "normal":
            severity = "ok"
            session_triggers = []
            reply = ensure_normal_photo_reassurance(reply)
            photo_finding = "normal"
        else:
            symptom_context = user_text or reply
            symptom_context_medications = get_patient_medications_for_symptom_context(
                st.session_state.get("selected_patient_id")
            )
            condition_analysis = analyze_symptom_against_conditions(
                symptom_context,
                patient_id=st.session_state.get("selected_patient_id"),
                medications=symptom_context_medications,
                allergies=get_patient_allergy_notes(st.session_state.get("selected_patient_id")),
            )
            severity, session_triggers = resolve_chat_severity(
                result.get("risk_level", "monitor"),
                symptom_context,
                needs_doctor=result.get("needs_doctor"),
                prior_incidents=prior_incidents,
            )
            severity, reply, condition_risk_alerts, needs_doctor_flag = merge_symptom_condition_analysis(
                severity,
                reply,
                condition_analysis,
                needs_doctor=bool(result.get("needs_doctor")),
            )
            severity, session_triggers = resolve_chat_severity(
                severity,
                symptom_context,
                needs_doctor=needs_doctor_flag,
                prior_incidents=prior_incidents,
            )
            severity = cap_allergy_report_severity(symptom_context, severity)
            severity = apply_report_severity_floor_caps(
                symptom_context,
                severity,
                medications=symptom_context_medications,
                conditions=get_patient_conditions(st.session_state.get("selected_patient_id")),
            )
            severity = cap_positive_report_severity(symptom_context, severity)
            photo_finding = clinical_finding or "concern"
        reply += build_session_connection_note(
            prior_incidents,
            session_triggers,
            current_text=(user_text or reply) if photo_finding != "normal" else "",
            patient_id=st.session_state.get("selected_patient_id"),
        )
        if severity != "ok":
            reply = append_care_guidance(reply, severity)
        reply += SESSION_HANDOVER_NOTE
        if not reply.strip():
            reply = "I couldn't analyse that photo just now. Please try again."
        source = "symptom_photo"
        incident_text = user_text or "Symptom photo for review"
        if photo_finding == "normal":
            doctor_note = f"Symptom photo review: normal appearance — {user_text or 'Photo submitted'}"
        else:
            doctor_note = f"Symptom photo review: {user_text or 'Photo submitted'}"
        context_report_count = count_linked_reports(
            prior_incidents,
            user_text or reply,
            session_triggers,
        )
        reply, session_triggers, context_report_count = enforce_report_ask_session_evidence(
            reply,
            prior_incidents=prior_incidents,
            session_triggers=session_triggers,
            context_report_count=context_report_count,
            patient_id=st.session_state.get("selected_patient_id"),
        )

    patient_id = st.session_state.get("selected_patient_id")
    reply = finalize_chat_patient_facing_text(
        reply,
        patient_id=patient_id,
        user_text=user_text or incident_text,
    )
    condition_risk_alerts = finalize_condition_risk_alerts(
        condition_risk_alerts,
        patient_id=patient_id,
        user_text=user_text or incident_text,
    )

    record_session_incident(
        text=incident_text,
        severity=severity,
        timestamp=reported_at,
        timestamp_display=timestamp_display,
        caregiver=caregiver_label,
        caregiver_id=caregiver_id,
        source=source,
        summary=doctor_note,
        image_b64=image_b64,
        photo_finding=photo_finding,
        photo_type=photo_type,
    )
    message = {
        "role": "assistant",
        "content": reply,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "timestamp_display": format_chat_timestamp(),
    }
    if severity != "ok" or photo_finding == "normal":
        message["severity"] = severity
        if severity != "ok":
            message["context_report_count"] = context_report_count
    if condition_risk_alerts:
        message["condition_risk_alerts"] = condition_risk_alerts
    st.session_state.messages.append(message)
    st.session_state.chat_scroll_to_bottom = True
    persist_patient_chat_thread(st.session_state.get("selected_patient_id"))
    return True


def process_pending_chat_response(caregiver_label: str, current_time_str: str) -> bool:
    pending = st.session_state.pending_chat_response
    caregiver_id = resolve_caregiver_id(caregiver_label)
    patient_id = get_current_patient_id()
    prior_incidents = get_session_incidents(patient_id)
    context = build_patient_context_block(patient_id)
    session_context = build_session_reports_context(patient_id)
    timeline_context = build_patient_report_timeline_context(patient_id, prior_incidents)
    stored_chat_context = build_stored_chat_context_for_ai(patient_id)
    stored_chat_block = f"\n{stored_chat_context}\n" if stored_chat_context else ""
    user_text = pending.get("text", "").strip()
    _report_ask_logger.info(
        "Report & Ask context patient=%s session_prior_count=%d stored_reports=%d stored_chat_chars=%d user_preview=%r",
        patient_id,
        len(prior_incidents),
        len(fetch_patient_care_reports(patient_id, limit=500)),
        len(stored_chat_context or ""),
        user_text[:120],
    )
    if prior_incidents:
        _report_ask_logger.debug(
            "Session priors for patient=%s: %s",
            patient_id,
            [
                {
                    "time": item.get("timestamp_display"),
                    "text": str(item.get("text") or "")[:120],
                    "severity": item.get("severity"),
                }
                for item in prior_incidents[-5:]
            ],
        )
    _report_ask_logger.debug(
        "Timeline context preview for patient=%s: %s",
        patient_id,
        timeline_context[:500],
    )
    _report_ask_logger.debug(
        "Session context for patient=%s: %s",
        patient_id,
        session_context[:500],
    )
    reported_at = pending.get("reported_at") or datetime.now(timezone.utc).isoformat()
    timestamp_display = pending.get("timestamp_display") or format_chat_timestamp(reported_at)
    from ai_helpers import get_relevant_medical_context
    medical_context = ""
    if user_text and not extract_unverified_patient_claims(user_text, patient_id):
        medical_context = get_relevant_medical_context(user_text)
    image_b64 = pending.get("image_b64")
    chat_history = build_chat_history_for_api(st.session_state.messages)
    condition_analysis = None
    personalization_block = ""
    allergy_notes = get_patient_allergy_notes(patient_id)
    symptom_context_medications = get_patient_medications_for_symptom_context(patient_id)
    symptom_context_conditions = get_patient_conditions(patient_id)
    allergy_prompt_block = build_allergy_symptom_prompt_block(user_text, allergy_notes)

    claim_grounding_block = ""
    if user_text and is_question(user_text):
        claim_grounding_block = build_patient_claim_grounding_prompt_block(user_text, patient_id)
        if claim_grounding_block:
            claim_grounding_block = f"\n\n{claim_grounding_block}"

    if user_text and reports_health_symptom_topic(user_text):
        condition_analysis = analyze_symptom_against_conditions(
            user_text,
            patient_id=patient_id,
            medications=symptom_context_medications,
            allergies=allergy_notes,
        )
        personalization_block = build_condition_analysis_prompt_block(condition_analysis)
        if allergy_prompt_block and allergy_prompt_block not in personalization_block:
            personalization_block = f"{personalization_block}\n\n{allergy_prompt_block}".strip()
    elif allergy_prompt_block:
        personalization_block = allergy_prompt_block

    if image_b64:
        return process_chat_photo_response(
            caregiver_label=caregiver_label,
            caregiver_id=caregiver_id,
            image_b64=image_b64,
            user_text=user_text,
            reported_at=reported_at,
            timestamp_display=timestamp_display,
            current_time_str=current_time_str,
            context=context,
            session_context=session_context,
            prior_incidents=prior_incidents,
            medical_context=medical_context,
        )

    if is_question(user_text):
        plan_items = get_active_plan_items()
        if is_next_medication_question(user_text):
            reply = build_next_medication_reply(
                plan_items,
                patient_id=patient_id,
                user_text=user_text,
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": reply,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "timestamp_display": format_chat_timestamp(),
            })
            st.session_state.chat_scroll_to_bottom = True
            record_session_incident(
                text=user_text,
                severity="ok",
                timestamp=reported_at,
                timestamp_display=timestamp_display,
                caregiver=caregiver_label,
                caregiver_id=caregiver_id,
                source="care_question",
                summary=user_text,
            )
            persist_patient_chat_thread(patient_id)
            return True

        med_schedule_block = ""
        if is_medication_schedule_question(user_text):
            med_schedule_block = "\n\n" + build_medication_schedule_prompt_block(plan_items)
        system_prompt = f"""You are CareShield, a medical assistant AI answering a family caregiver's question.
You MUST use the patient's full stored medications, conditions, allergies, documents, and timeline below to give specific, contextual answers — not generic ones.
Reference their actual drugs, schedules, diagnoses, and recent reports when relevant.

{CHAT_SINGLE_PATIENT_RULES}
{CHAT_SYMPTOM_PERSONALIZATION_RULES}
{CHAT_INFORMATIONAL_QUESTION_RULES}
{context}
{med_schedule_block}

{timeline_context}

{session_context}
{stored_chat_block}{personalization_block}{claim_grounding_block}
{("RELEVANT MEDICAL KNOWLEDGE:\n" + medical_context) if medical_context else ""}
Current date/time: {current_time_str}
Answer clearly and reassuringly. For medication timing questions, list each drug with its scheduled time from the plan.
Respond with ONLY a JSON object with:
1. "answer": a clear, direct answer
2. "severity": one of "ok", "monitor", "contact_doctor", or "emergency"
3. "needs_doctor": true when severity is "contact_doctor", otherwise false

{CHAT_URGENCY_RULES}

{CHAT_SESSION_RULES}
"""
        result = ask_ai_chat(system_prompt, user_text, chat_history=chat_history)
    else:
        system_prompt = f"""You are CareShield, a medical assistant AI supporting family caregivers.
Interpret every report using the patient's full stored medications, conditions, allergies, documents, and timeline below — not generic assumptions.
Flag risks that relate to their specific drugs and diagnoses. When a symptom is more dangerous because of a stored chronic condition, increase urgency accordingly.

{CHAT_SINGLE_PATIENT_RULES}
{CHAT_SYMPTOM_PERSONALIZATION_RULES}
{context}

{timeline_context}

{session_context}
{stored_chat_block}{personalization_block}
{("RELEVANT MEDICAL KNOWLEDGE:\n" + medical_context) if medical_context else ""}
Current date/time: {current_time_str}
Respond with ONLY a JSON object with:
1. "empathetic_advice": a short, warm, practical tip for the caregiver that names relevant conditions or medicines when they apply
2. "clinical_tags": up to 3 short plain-English phrases for the caregiver (e.g. "stiff joints", "changes in walking") — everyday words only; never codes, snake_case, underscores, or jargon labels
3. "doctor_note": the same information rewritten in neutral, clinical language
4. "severity": one of "ok", "monitor", "contact_doctor", or "emergency"
5. "needs_doctor": true when severity is "contact_doctor", otherwise false

{CHAT_URGENCY_RULES}

{CHAT_SESSION_RULES}
"""
        result = ask_ai_chat(system_prompt, user_text, chat_history=chat_history)

    if result.get("error"):
        if user_text:
            record_session_incident(
                text=user_text,
                severity="monitor",
                timestamp=reported_at,
                timestamp_display=timestamp_display,
                caregiver=caregiver_label,
                caregiver_id=caregiver_id,
                source="care_question" if is_question(user_text) else "voice_report",
                summary=user_text,
            )
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.get(
                "message",
                "We couldn't process this right now. Please try again.",
            ),
        })
        st.session_state.chat_scroll_to_bottom = True
        persist_patient_chat_thread(patient_id)
        return False

    if is_question(user_text):
        reply = coerce_ai_text(result.get("answer") or result.get("analysis"))
        reply = enforce_patient_record_grounding_in_reply(reply, user_text, patient_id)
        plan_items = get_active_plan_items()
        if is_medication_schedule_question(user_text) and looks_like_medication_refusal(reply):
            reply = build_medication_schedule_reply(plan_items)
            severity = "ok"
            session_triggers = []
        else:
            severity, session_triggers = resolve_chat_severity(
                result.get("severity", "monitor"),
                user_text,
                needs_doctor=result.get("needs_doctor"),
                prior_incidents=prior_incidents,
            )
        condition_risk_alerts: list = []
        if condition_analysis:
            severity, reply, condition_risk_alerts, needs_doctor_flag = merge_symptom_condition_analysis(
                severity,
                reply,
                condition_analysis,
                needs_doctor=bool(result.get("needs_doctor")),
            )
            severity, session_triggers = resolve_chat_severity(
                severity,
                user_text,
                needs_doctor=needs_doctor_flag,
                prior_incidents=prior_incidents,
            )
        severity = cap_allergy_report_severity(user_text, severity)
        severity = apply_report_severity_floor_caps(
            user_text,
            severity,
            medications=symptom_context_medications,
            conditions=symptom_context_conditions,
        )
        severity = cap_positive_report_severity(user_text, severity)
        severity = cap_informational_question_severity(user_text, severity)
        reply += build_session_connection_note(
            prior_incidents,
            session_triggers,
            current_text=user_text,
            patient_id=patient_id,
        )
        reply = append_care_guidance(reply, severity)
        if not reply.strip():
            reply = "I couldn't generate an answer just now. Please try again."
        context_report_count = count_linked_reports(prior_incidents, user_text, session_triggers)
        reply, session_triggers, context_report_count = enforce_report_ask_session_evidence(
            reply,
            prior_incidents=prior_incidents,
            session_triggers=session_triggers,
            context_report_count=context_report_count,
            patient_id=patient_id,
        )
        reply = finalize_chat_patient_facing_text(reply, patient_id=patient_id, user_text=user_text)
        condition_risk_alerts = finalize_condition_risk_alerts(
            condition_risk_alerts,
            patient_id=patient_id,
            user_text=user_text,
        )
        message = {
            "role": "assistant",
            "content": reply,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "timestamp_display": format_chat_timestamp(),
        }
        if severity != "ok":
            message["severity"] = severity
            message["context_report_count"] = context_report_count
        if condition_risk_alerts:
            message["condition_risk_alerts"] = condition_risk_alerts
        st.session_state.messages.append(message)
        st.session_state.chat_scroll_to_bottom = True
        record_session_incident(
            text=user_text,
            severity=severity,
            timestamp=reported_at,
            timestamp_display=timestamp_display,
            caregiver=caregiver_label,
            caregiver_id=caregiver_id,
            source="care_question",
            summary=f"Question: {user_text[:300]}",
        )
    else:
        tags_str = format_caregiver_clinical_tags(result.get("clinical_tags"))
        reply = coerce_ai_text(result.get("empathetic_advice") or result.get("answer"))
        if tags_str:
            reply += f"\n\n**Things to keep an eye on:** {tags_str}"
        severity, session_triggers = resolve_chat_severity(
            result.get("severity", "monitor"),
            user_text,
            needs_doctor=result.get("needs_doctor"),
            prior_incidents=prior_incidents,
        )
        condition_risk_alerts: list = []
        if condition_analysis:
            severity, reply, condition_risk_alerts, needs_doctor_flag = merge_symptom_condition_analysis(
                severity,
                reply,
                condition_analysis,
                needs_doctor=bool(result.get("needs_doctor")),
            )
            severity, session_triggers = resolve_chat_severity(
                severity,
                user_text,
                needs_doctor=needs_doctor_flag,
                prior_incidents=prior_incidents,
            )
        severity = cap_allergy_report_severity(user_text, severity)
        severity = apply_report_severity_floor_caps(
            user_text,
            severity,
            medications=symptom_context_medications,
            conditions=symptom_context_conditions,
        )
        severity = cap_positive_report_severity(user_text, severity)
        reply += build_session_connection_note(
            prior_incidents,
            session_triggers,
            current_text=user_text,
            patient_id=patient_id,
        )
        reply = append_care_guidance(reply, severity)
        reply += SESSION_HANDOVER_NOTE
        if not reply.strip():
            reply = "I couldn't generate a response just now. Please try again."
        context_report_count = count_linked_reports(prior_incidents, user_text, session_triggers)
        reply, session_triggers, context_report_count = enforce_report_ask_session_evidence(
            reply,
            prior_incidents=prior_incidents,
            session_triggers=session_triggers,
            context_report_count=context_report_count,
            patient_id=patient_id,
        )
        reply = finalize_chat_patient_facing_text(reply, patient_id=patient_id, user_text=user_text)
        condition_risk_alerts = finalize_condition_risk_alerts(
            condition_risk_alerts,
            patient_id=patient_id,
            user_text=user_text,
        )
        doctor_note = result.get("doctor_note", user_text)
        record_session_incident(
            text=user_text,
            severity=severity,
            timestamp=reported_at,
            timestamp_display=timestamp_display,
            caregiver=caregiver_label,
            caregiver_id=caregiver_id,
            source="voice_report",
            summary=doctor_note,
        )
        assistant_message = {
            "role": "assistant",
            "content": reply,
            "severity": severity,
            "context_report_count": context_report_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "timestamp_display": format_chat_timestamp(),
        }
        if condition_risk_alerts:
            assistant_message["condition_risk_alerts"] = condition_risk_alerts
        st.session_state.messages.append(assistant_message)
        st.session_state.chat_scroll_to_bottom = True

    last_msg = (st.session_state.messages or [])[-1] if st.session_state.get("messages") else {}
    if last_msg.get("role") == "assistant":
        _report_ask_logger.info(
            "Report & Ask outcome patient=%s severity=%s linked_count=%s",
            patient_id,
            last_msg.get("severity", "ok"),
            last_msg.get("context_report_count", 1),
        )

    persist_patient_chat_thread(patient_id)
    return True

def build_condition_since_html(since) -> str:
    label = normalize_condition_since(since)
    if not label:
        return ""
    return f'<div class="cs-condition-date">{html.escape(label)}</div>'


def build_condition_rows_html(conditions):
    rows = []
    for item in conditions:
        badge_class, badge_label = condition_badge_meta(item["badge"])
        rows.append(
            f'<div class="cs-condition-row">'
            f'<div><div class="cs-condition-name">{item["name"]}</div>'
            f'{build_condition_since_html(item.get("since"))}</div>'
            f'<span class="cs-badge {badge_class}">{badge_label}</span></div>'
        )
    return "".join(rows)


def build_stored_conditions_medcam_html(conditions):
    return f"""
    <div class="cs-conditions-card">
      <div class="cs-conditions-header">
        <span class="cs-conditions-icon">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
          </svg>
        </span>
        <span class="cs-conditions-title">Stored conditions</span>
      </div>
      {build_condition_rows_html(conditions)}
    </div>
    """


def format_medcam_log_time():
    if user_timezone:
        tz_obj = ZoneInfo(user_timezone)
        return datetime.now(tz_obj).strftime("%I:%M %p").lstrip("0")
    return datetime.now().strftime("%I:%M %p").lstrip("0")


# ── PRN (as-needed) scheduling ───────────────────────────────────────────────
PRN_SCHEDULED_TIME_LABEL = "PRN"
DEFAULT_PRN_MAX_PER_DAY = 4
DEFAULT_PRN_MIN_INTERVAL_HOURS = 4.0
PARACETAMOL_MIN_INTERVAL_HOURS = 4.0

# MedCam identification threshold — pills need high or medium confidence to count as identified.
MEDCAM_IDENTIFIED_CONFIDENCE = frozenset({"high", "medium"})
MEDCAM_CONFIDENCE_THRESHOLD_LABEL = "high or medium confidence"


def parse_prn_max_per_day(timing: str) -> int:
    text = (timing or "").strip().lower()
    patterns = [
        r"max(?:imum)?\s*(\d+)\s*(?:x|times?)",
        r"up to (\d+)\s*(?:x|times?|doses?)",
        r"(\d+)\s*times?\s*(?:a|per)\s*day",
        r"(\d+)\s*x\s*/?\s*day",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            count = int(match.group(1))
            if 1 <= count <= 12:
                return count
    return DEFAULT_PRN_MAX_PER_DAY


def get_prn_min_interval_hours(med_name: str, timing: str = "") -> float:
    combined = f"{med_name} {timing}".lower()
    if "paracetamol" in combined or "acetaminophen" in combined:
        return PARACETAMOL_MIN_INTERVAL_HOURS
    every_match = re.search(r"every\s+(\d+(?:\.\d+)?)\s*hours?", (timing or "").lower())
    if every_match:
        return max(float(every_match.group(1)), 1.0)
    return DEFAULT_PRN_MIN_INTERVAL_HOURS


def partition_plan_items(plan_items: list) -> tuple[list, list]:
    scheduled, prn = [], []
    for med in plan_items or []:
        timing = med.get("time") or med.get("timing") or ""
        if is_prn_timing(timing):
            prn.append(med)
        else:
            scheduled.append(med)
    return scheduled, prn


def get_prn_dose_logs_today(medication_name: str, today_logs: list) -> list:
    return [
        log for log in today_logs
        if log.get("medication_name") == medication_name
        and log.get("scheduled_time") == PRN_SCHEDULED_TIME_LABEL
        and log.get("status") == "taken"
    ]


def parse_log_datetime(log: dict, tz_obj) -> datetime | None:
    raw = log.get("logged_at") or ""
    if not raw:
        return None
    try:
        moment = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(tz_obj)
    except (ValueError, TypeError):
        return None


def format_wait_remaining(total_minutes: float) -> str:
    minutes = max(0, int(math.ceil(total_minutes)))
    hours, rem = divmod(minutes, 60)
    if hours and rem:
        return f"Wait {hours}h {rem}m"
    if hours:
        return f"Wait {hours}h"
    return f"Wait {rem}m"


def get_prn_medication_status(
    med_name: str,
    plan_item: dict,
    now: datetime,
    today_logs: list,
    tz_obj,
) -> dict:
    timing = plan_item.get("time") or plan_item.get("timing") or ""
    max_per_day = parse_prn_max_per_day(timing)
    min_interval_h = get_prn_min_interval_hours(med_name, timing)
    prn_logs = get_prn_dose_logs_today(med_name, today_logs)
    doses_today = len(prn_logs)

    base = {
        "doses_today": doses_today,
        "max_per_day": max_per_day,
        "min_interval_hours": min_interval_h,
        "scheduled_display": "as needed",
    }

    if doses_today >= max_per_day:
        return {
            **base,
            "status": "prn_max",
            "message": f"Max reached for today ({doses_today} of {max_per_day} doses used)",
            "wait_label": "Max reached for today",
        }

    last_dt = None
    for log in prn_logs:
        moment = parse_log_datetime(log, tz_obj)
        if moment and (last_dt is None or moment > last_dt):
            last_dt = moment

    if last_dt:
        elapsed_h = (now - last_dt).total_seconds() / 3600
        if elapsed_h < min_interval_h:
            remaining_m = (min_interval_h - elapsed_h) * 60
            wait_label = format_wait_remaining(remaining_m)
            return {
                **base,
                "status": "prn_wait",
                "message": f"{wait_label} — minimum {min_interval_h:g}h between doses",
                "wait_label": wait_label,
                "wait_minutes": remaining_m,
            }

    return {
        **base,
        "status": "prn_available",
        "message": f"Available ({doses_today} of {max_per_day} doses used today)",
        "wait_label": "Available",
    }


def format_schedule_time(hour: int, minute: int) -> str:
    return f"{hour:02d}:{minute:02d}"


def dose_log_key(medication_name: str, time_label: str, date_iso: str) -> str:
    return f"{medication_name}|{time_label}|{date_iso}"


def init_caregiver_profiles() -> None:
    if "caregiver_profiles" in st.session_state:
        return

    from patient_care_storage import load_saved_caregiver_profiles

    saved = load_saved_caregiver_profiles()
    if saved and saved.get("profiles"):
        st.session_state.caregiver_profiles = [dict(profile) for profile in saved["profiles"]]
        active_profiles = [
            profile for profile in st.session_state.caregiver_profiles
            if not profile.get("is_deleted")
        ]
        selected_id = str(saved.get("selected_caregiver_id") or "").strip()
        if selected_id and any(profile.get("id") == selected_id for profile in active_profiles):
            st.session_state.selected_caregiver_id = selected_id
        elif active_profiles:
            st.session_state.selected_caregiver_id = active_profiles[0]["id"]
        else:
            st.session_state.selected_caregiver_id = st.session_state.caregiver_profiles[0]["id"]
        return

    st.session_state.caregiver_profiles = [dict(profile) for profile in DEFAULT_CAREGIVER_PROFILES]
    st.session_state.selected_caregiver_id = DEFAULT_CAREGIVER_PROFILES[0]["id"]


def persist_caregiver_profiles_to_disk() -> None:
    from patient_care_storage import save_caregiver_profiles

    init_caregiver_profiles()
    save_caregiver_profiles(
        st.session_state.caregiver_profiles,
        st.session_state.selected_caregiver_id,
    )


def get_all_caregiver_profiles() -> list[dict]:
    init_caregiver_profiles()
    return list(st.session_state.caregiver_profiles)


def get_caregiver_profiles() -> list[dict]:
    return [
        profile for profile in get_all_caregiver_profiles()
        if not profile.get("is_deleted")
    ]


def format_caregiver_label(profile: dict) -> str:
    name = str(profile.get("name") or "").strip()
    role = str(profile.get("role") or "").strip()
    if name and role:
        return f"{name} ({role})"
    return name or role or "Caregiver"


def profile_avatar_initial(profile: dict) -> str:
    source = str(profile.get("name") or profile.get("role") or "?").strip()
    return source[0].upper() if source else "?"


def get_caregiver_profile_by_id(profile_id: str, include_deleted: bool = False) -> dict | None:
    for profile in get_all_caregiver_profiles():
        if profile.get("id") != profile_id:
            continue
        if include_deleted or not profile.get("is_deleted"):
            return profile
    return None


def get_caregiver_profile_by_label(label: str) -> dict | None:
    for profile in get_caregiver_profiles():
        if format_caregiver_label(profile) == label:
            return profile
    return None


def get_caregiver_profile_label(profile_id: str) -> str:
    profile = get_caregiver_profile_by_id(profile_id, include_deleted=True)
    if profile:
        return format_caregiver_label(profile)
    return LEGACY_CAREGIVER_LABELS.get(profile_id, profile_id or "Caregiver")


def build_caregiver_label_maps() -> tuple[dict[str, str], dict[str, str]]:
    label_to_id: dict[str, str] = {}
    id_to_label: dict[str, str] = {}
    for profile in get_caregiver_profiles():
        label = format_caregiver_label(profile)
        label_to_id[label] = profile["id"]
        id_to_label[profile["id"]] = label
    return label_to_id, id_to_label


def slugify_caregiver_id(name: str, role: str, existing_ids: set[str]) -> str:
    base_source = name.strip() or role.strip() or "carer"
    base = re.sub(r"[^a-z0-9]+", "_", base_source.lower()).strip("_") or "carer"
    candidate = base
    suffix = 2
    while candidate in existing_ids:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def add_caregiver_profile(name: str, role: str) -> dict | None:
    clean_name = str(name or "").strip()
    clean_role = str(role or "").strip()
    if not clean_name and not clean_role:
        return None
    profiles = get_all_caregiver_profiles()
    existing_ids = {profile["id"] for profile in profiles}
    profile = {
        "id": slugify_caregiver_id(clean_name, clean_role, existing_ids),
        "name": clean_name,
        "role": clean_role,
    }
    profiles.append(profile)
    st.session_state.caregiver_profiles = profiles
    previous_caregiver_id = st.session_state.get("selected_caregiver_id")
    reset_report_ask_for_caregiver_switch(
        str(st.session_state.get("selected_patient_id") or ""),
        format_caregiver_label(profile),
        profile["id"],
        previous_caregiver_id=previous_caregiver_id,
    )
    st.session_state.selected_caregiver_id = profile["id"]
    st.session_state.pop("caregiver_profile_picker", None)
    persist_caregiver_profiles_to_disk()
    return profile


def update_caregiver_profile(profile_id: str, name: str, role: str) -> bool:
    clean_name = str(name or "").strip()
    clean_role = str(role or "").strip()
    if not clean_name and not clean_role:
        return False
    profiles = get_all_caregiver_profiles()
    for profile in profiles:
        if profile.get("id") == profile_id:
            profile["name"] = clean_name
            profile["role"] = clean_role
            profile["is_deleted"] = False
            st.session_state.caregiver_profiles = profiles
            if st.session_state.get("chat_caregiver_id") == profile_id:
                st.session_state.chat_caregiver = format_caregiver_label(profile)
            persist_caregiver_profiles_to_disk()
            return True
    return False


def delete_caregiver_profile(profile_id: str) -> tuple[bool, str | None]:
    """Soft-delete: hide from selector but keep profile for attribution lookups."""
    active_profiles = get_caregiver_profiles()
    if len(active_profiles) <= 1:
        return False, "You need at least one logged-in profile."
    profiles = get_all_caregiver_profiles()
    target = None
    for profile in profiles:
        if profile.get("id") == profile_id:
            target = profile
            break
    if target is None:
        return False, "Profile not found."
    target["is_deleted"] = True
    st.session_state.caregiver_profiles = profiles
    remaining_active = get_caregiver_profiles()
    if st.session_state.selected_caregiver_id == profile_id:
        st.session_state.selected_caregiver_id = remaining_active[0]["id"]
        reset_report_ask_for_caregiver_switch(
            str(st.session_state.get("selected_patient_id") or ""),
            format_caregiver_label(remaining_active[0]),
            remaining_active[0]["id"],
            previous_caregiver_id=profile_id,
        )
    st.session_state.pop("caregiver_profile_picker", None)
    persist_caregiver_profiles_to_disk()
    return True, None


def resolve_caregiver_id(caregiver_label_or_id: str) -> str:
    if not caregiver_label_or_id:
        return caregiver_label_or_id
    profile = get_caregiver_profile_by_id(caregiver_label_or_id, include_deleted=True)
    if profile:
        return profile["id"]
    label_to_id, _ = build_caregiver_label_maps()
    if caregiver_label_or_id in label_to_id:
        return label_to_id[caregiver_label_or_id]
    return LEGACY_LABEL_TO_ID.get(caregiver_label_or_id, caregiver_label_or_id)


def resolve_caregiver_label(caregiver_id_or_label: str) -> str:
    if not caregiver_id_or_label:
        return "Caregiver"
    profile = get_caregiver_profile_by_id(caregiver_id_or_label, include_deleted=True)
    if profile:
        return format_caregiver_label(profile)
    _, id_to_label = build_caregiver_label_maps()
    if caregiver_id_or_label in id_to_label:
        return id_to_label[caregiver_id_or_label]
    if caregiver_id_or_label in LEGACY_CAREGIVER_LABELS:
        return LEGACY_CAREGIVER_LABELS[caregiver_id_or_label]
    profile = get_caregiver_profile_by_label(caregiver_id_or_label)
    if profile:
        return format_caregiver_label(profile)
    return caregiver_id_or_label


def incident_caregiver_id(incident: dict) -> str:
    if incident.get("caregiver_id"):
        return incident["caregiver_id"]
    return resolve_caregiver_id(incident.get("caregiver", ""))


@st.dialog("Add profile")
def add_caregiver_profile_dialog() -> None:
    st.markdown("Create a carer profile so entries are tagged correctly.")
    name = st.text_input("Name", placeholder="Pedro")
    role = st.text_input("Relationship or role", placeholder="grandson, Night nurse, Carer...")
    save_col, cancel_col = st.columns(2)
    with save_col:
        if st.button("Save profile", type="primary", use_container_width=True, key="add_profile_save"):
            profile = add_caregiver_profile(name, role)
            if profile is None:
                st.error("Please enter at least a name or a role.")
            else:
                st.session_state.pop("open_add_profile_dialog", None)
                st.rerun()
    with cancel_col:
        if st.button("Cancel", use_container_width=True, key="add_profile_cancel"):
            st.session_state.pop("open_add_profile_dialog", None)
            st.rerun()


@st.dialog("Edit profile")
def edit_caregiver_profile_dialog(profile_id: str) -> None:
    profile = get_caregiver_profile_by_id(profile_id)
    if not profile:
        st.error("Profile not found.")
        return
    st.markdown("Update how this carer appears across CareShield.")
    name = st.text_input("Name", value=str(profile.get("name") or ""), key=f"edit_profile_name_{profile_id}")
    role = st.text_input(
        "Relationship or role",
        value=str(profile.get("role") or ""),
        key=f"edit_profile_role_{profile_id}",
    )
    save_col, cancel_col = st.columns(2)
    with save_col:
        if st.button("Save changes", type="primary", use_container_width=True, key=f"edit_profile_save_{profile_id}"):
            if update_caregiver_profile(profile_id, name, role):
                st.session_state.pop("show_edit_profile_dialog", None)
                st.session_state.pop("editing_profile_id", None)
                st.rerun()
            else:
                st.error("Please enter at least a name or a role.")
    with cancel_col:
        if st.button("Cancel", use_container_width=True, key=f"edit_profile_cancel_{profile_id}"):
            st.session_state.pop("show_edit_profile_dialog", None)
            st.session_state.pop("editing_profile_id", None)
            st.rerun()


def render_caregiver_profile_manager() -> None:
    profiles = get_caregiver_profiles()
    selected_id = st.session_state.selected_caregiver_id
    for profile in profiles:
        label = format_caregiver_label(profile)
        profile_id = profile["id"]
        edit_clicked, delete_clicked = render_profile_manager_row(
            label,
            profile_id == selected_id,
            edit_key=f"profile_edit_{profile_id}",
            delete_key=f"profile_delete_{profile_id}",
        )
        if edit_clicked:
            st.session_state.editing_profile_id = profile_id
            st.session_state.show_edit_profile_dialog = True
            st.rerun()
        if delete_clicked:
            success, error = delete_caregiver_profile(profile_id)
            if not success:
                st.error(error or "Could not delete profile.")
            else:
                st.rerun()
    st.markdown("---")
    if st.button(ADD_NEW_PROFILE_OPTION, key="profile_add_from_manager", use_container_width=True):
        st.session_state.open_add_profile_dialog = True
        st.rerun()


def build_caregiver_avatars_html(profiles: list[dict], selected_id: str) -> str:
    avatar_items = []
    for profile in profiles:
        initial = html.escape(profile_avatar_initial(profile))
        active_class = " active" if profile.get("id") == selected_id else ""
        avatar_items.append(f'<div class="cs-avatar{active_class}">{initial}</div>')
    return "".join(avatar_items)


def render_caregiver_profile_switcher() -> tuple[str, str]:
    init_caregiver_profiles()

    profiles = get_caregiver_profiles()
    selected_id = st.session_state.selected_caregiver_id
    labels = [format_caregiver_label(profile) for profile in profiles]
    current_label = get_caregiver_profile_label(selected_id)
    if current_label not in labels and labels:
        previous_caregiver_id = selected_id
        current_label = labels[0]
        st.session_state.selected_caregiver_id = profiles[0]["id"]
        selected_id = profiles[0]["id"]
        reset_report_ask_for_caregiver_switch(
            str(st.session_state.get("selected_patient_id") or ""),
            format_caregiver_label(profiles[0]),
            profiles[0]["id"],
            previous_caregiver_id=previous_caregiver_id,
        )

    md_html(f"""
    <div class="cs-user-area cs-user-area-left">
      <div class="cs-user-label">Who's logged in?</div>
      <div class="cs-avatars cs-avatars-left">{build_caregiver_avatars_html(profiles, selected_id)}</div>
    </div>
    """)

    picker_col, manage_col = st.columns([5, 1])
    options = labels + [ADD_NEW_PROFILE_OPTION]
    with picker_col:
        picked = st.selectbox(
            "Who's logged in?",
            options=options,
            index=labels.index(current_label) if current_label in labels else 0,
            label_visibility="collapsed",
            key="caregiver_profile_picker",
        )
    with manage_col:
        with st.popover("", help="Manage profiles", icon=":material/edit:"):
            render_caregiver_profile_manager()

    if picked == ADD_NEW_PROFILE_OPTION:
        st.session_state.open_add_profile_dialog = True
        st.session_state.pop("caregiver_profile_picker", None)
        st.rerun()
    elif picked in labels:
        profile = get_caregiver_profile_by_label(picked)
        if profile and profile["id"] != selected_id:
            reset_report_ask_for_caregiver_switch(
                str(st.session_state.get("selected_patient_id") or ""),
                format_caregiver_label(profile),
                profile["id"],
                previous_caregiver_id=selected_id,
            )
            st.session_state.selected_caregiver_id = profile["id"]
            persist_caregiver_profiles_to_disk()
            st.session_state.pop("caregiver_profile_picker", None)
            st.rerun()

    if st.session_state.pop("open_add_profile_dialog", False):
        add_caregiver_profile_dialog()
    if st.session_state.get("show_edit_profile_dialog") and st.session_state.get("editing_profile_id"):
        edit_caregiver_profile_dialog(st.session_state.editing_profile_id)

    render_profile_management_help_button()

    profile_id = st.session_state.selected_caregiver_id
    return profile_id, get_caregiver_profile_label(profile_id)


def parse_log_local_date(log: dict, tz_obj) -> str:
    raw = log.get("logged_at") or ""
    if not raw:
        return ""
    try:
        logged_at = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if logged_at.tzinfo is None:
            logged_at = logged_at.replace(tzinfo=timezone.utc)
        return logged_at.astimezone(tz_obj).date().isoformat()
    except (ValueError, TypeError):
        return ""


def dose_log_for_today(dose: dict, today_logs: list, date_iso: str, tz_obj):
    for log in today_logs:
        if parse_log_local_date(log, tz_obj) != date_iso:
            continue
        if log.get("medication_name") == dose["medication_name"] and log.get("scheduled_time") == dose["time_label"]:
            return log
    return None


def dose_ui_state(dose: dict, now: datetime, today_logs: list, tz_obj) -> str:
    date_iso = now.date().isoformat()
    existing = dose_log_for_today(dose, today_logs, date_iso, tz_obj)
    return compute_dose_ui_state(dose, now, existing_log=existing, tz_obj=tz_obj)


def format_friendly_dose_time(hour: int, minute: int) -> str:
    suffix = "am" if hour < 12 else "pm"
    display_hour = hour % 12 or 12
    if minute:
        return f"{display_hour}:{minute:02d}{suffix}"
    return f"{display_hour}{suffix}"


def dose_events_for_medication(med_name: str, plan_items: list) -> list:
    plan = find_plan_item(med_name, plan_items)
    if not plan:
        return []
    target = plan.get("name") or med_name
    return [
        dose for dose in build_dose_events(plan_items)
        if dose["medication_name"] == target
    ]


def get_medication_timing_verdict(
    med_name: str,
    plan_items: list,
    now: datetime,
    today_logs: list,
    tz_obj,
) -> dict:
    plan = find_plan_item(med_name, plan_items)
    timing = (plan.get("time") or plan.get("timing") or "") if plan else ""

    if plan and is_prn_timing(timing):
        return get_prn_medication_status(med_name, plan, now, today_logs, tz_obj)

    events = dose_events_for_medication(med_name, plan_items)
    if not events:
        return {
            "status": "unknown",
            "scheduled_display": timing or "unknown",
            "message": "schedule not clear — check the care plan",
        }

    annotated = [
        (dose, dose_ui_state(dose, now, today_logs, tz_obj))
        for dose in events
    ]

    if any(state == "actionable" for _, state in annotated):
        actionable = [dose for dose, state in annotated if state == "actionable"]
        times = ", ".join(format_friendly_dose_time(d["hour"], d["minute"]) for d in actionable)
        return {
            "status": "due_now",
            "scheduled_display": times,
            "message": f"due now ({times})",
        }

    if all(state == "taken" for _, state in annotated):
        times = ", ".join(format_friendly_dose_time(d["hour"], d["minute"]) for d, _ in annotated)
        return {
            "status": "taken",
            "scheduled_display": times,
            "message": "already logged as taken today",
        }

    future = [dose for dose, state in annotated if state == "not_yet"]
    missed = [dose for dose, state in annotated if state == "missed"]

    if future and not missed:
        next_dose = min(future, key=lambda item: (item["hour"], item["minute"]))
        friendly = format_friendly_dose_time(next_dose["hour"], next_dose["minute"])
        return {
            "status": "not_due_yet",
            "scheduled_display": friendly,
            "message": f"not due until {friendly}",
        }

    if missed:
        latest = max(missed, key=lambda item: (item["hour"], item["minute"]))
        friendly = format_friendly_dose_time(latest["hour"], latest["minute"])
        return {
            "status": "past_due",
            "scheduled_display": friendly,
            "message": f"was due at {friendly}, not now",
        }

    return {
        "status": "unknown",
        "scheduled_display": timing or "unknown",
        "message": "check the schedule before giving",
    }


def parse_detected_pill_count(pill: dict) -> int | None:
    """Read how many pills the vision step reported for one group (None if not provided)."""
    if not isinstance(pill, dict):
        return None
    for key in ("count", "quantity", "pill_count", "num_pills", "detected_count"):
        raw = pill.get(key)
        if raw is None or raw == "":
            continue
        try:
            return max(int(raw), 0)
        except (TypeError, ValueError):
            continue
    return None


def evaluate_dose_quantity_match(detected: int | None, required: int) -> tuple[str, str]:
    """Compare detected pill count vs plan-required count for one medication group."""
    if detected is None:
        return (
            "unknown",
            f"Could not verify pill count in photo — plan requires {required} "
            f"pill{'s' if required != 1 else ''} per dose; count manually",
        )
    if detected == required:
        return (
            "ok",
            f"Count OK — {detected} pill{'s' if detected != 1 else ''} detected, "
            f"{required} required for this dose",
        )
    if detected > required:
        return (
            "high",
            f"{detected} pill(s) detected, only {required} needed for this dose — "
            "please confirm before giving",
        )
    return (
        "low",
        f"Only {detected} of {required} pill(s) detected — check you have the full dose",
    )


def normalize_medcam_ai_pill_groups(pills_in_photo: list) -> list:
    """
    Merge multiple AI groups that identify the same medication, summing detected counts.
    Applies uniformly to every medication — shape/colour do not affect grouping logic.
    """
    normalized: list[dict] = []
    index_by_med: dict[str, int] = {}
    for pill in pills_in_photo or []:
        if not isinstance(pill, dict):
            continue
        entry = dict(pill)
        raw_name = (entry.get("medication_name") or "").strip()
        confidence = str(entry.get("match_confidence") or "none").lower()
        med_key = (
            raw_name if raw_name and confidence in MEDCAM_IDENTIFIED_CONFIDENCE else None
        )
        detected = parse_detected_pill_count(entry)
        if detected is not None:
            entry["count"] = detected

        if med_key and med_key in index_by_med:
            existing = normalized[index_by_med[med_key]]
            existing_detected = parse_detected_pill_count(existing)
            if detected is not None:
                total = (existing_detected or 0) + detected
                existing["count"] = total
            continue

        if med_key:
            index_by_med[med_key] = len(normalized)
        normalized.append(entry)
    return normalized


def build_medcam_verification_system_prompt() -> str:
    return """You are a medical assistant AI helping a caregiver verify medication safety before giving pills to a patient.

You will receive:
1. Registered pill reference photos (what each pill looks like) and strength per pill (mg)
2. The patient's discharge plan and full discharge document text
3. The current local date/time and per-medication schedule context computed by the app
4. A verification photo showing pills in the caregiver's hand

Your job:
- Identify EVERY distinct pill or pill group visible in the verification photo
- Match each group to at most ONE registered medication using the reference photos
- Count EVERY individual physical pill visible for that group — count pills, not doses
- If you cannot confidently match a pill group, say so — do NOT guess a medication name

CRITICAL counting rules (apply to every medication equally):
- "count" must be the number of separate pills you can see in the photo for that group
- Do NOT set count to the prescribed dose from the plan — count what is visible
- If two identical pills are visible, count must be 2 even when the plan says 1 per dose
- If only one pill is visible, count must be 1 even when the plan says 2 per dose
- Count each pill regardless of shape, colour, size, or brand markings

Respond with ONLY a JSON object with:
1. "pills_in_photo": array of objects, one per distinct pill group you see:
   - "medication_name": matched registered name ONLY if match_confidence is high or medium; otherwise null
   - "description": brief visual description (shape, colour, markings)
   - "count": integer — number of individual pills visible in this group (required for every entry)
   - "match_confidence": "high", "medium", "low", or "none"
   - "matched_reference": true if you matched a registered reference photo, else false
   - "closest_match_name": optional — best registered medication guess when medication_name is null but something looked similar (omit if no plausible candidate)
2. "unidentified_description": optional brief note about pills you could not match (or empty string)
3. "severity_hint": "ok", "monitor", or "urgent" — your best guess before schedule rules are applied
4. "caregiver_message": leave as empty string "" — the app will compose the final message

Identification threshold: only set medication_name when confidence is high or medium. For low/none, leave medication_name null and use closest_match_name if there was a weak candidate.

Do NOT give one global pass/fail for the whole photo. Always enumerate each pill group separately."""


def enrich_medcam_pill_results(
    ai_result: dict,
    plan_items: list,
    now: datetime,
    today_logs: list,
    tz_obj,
    med_refs: list | None = None,
    patient_id=None,
) -> list[dict]:
    if patient_id is not None:
        med_refs = load_live_medication_references(patient_id)
    elif med_refs is None:
        med_refs = []

    enriched = []
    pill_groups = normalize_medcam_ai_pill_groups(ai_result.get("pills_in_photo") or [])
    for pill in pill_groups:
        raw_name = (pill.get("medication_name") or "").strip() or None
        confidence = str(pill.get("match_confidence") or "none").lower()
        closest_match = (pill.get("closest_match_name") or "").strip() or None
        med_name = raw_name if raw_name and confidence in MEDCAM_IDENTIFIED_CONFIDENCE else None
        if not med_name and confidence == "low" and raw_name:
            closest_match = closest_match or raw_name

        detected_count = parse_detected_pill_count(pill)
        plan = find_plan_item(med_name, plan_items) if med_name else {}
        required_count = (
            get_effective_pills_per_dose(med_name, plan, med_refs) if med_name else None
        )

        quantity_status = "unknown"
        quantity_message = ""
        if med_name and confidence in MEDCAM_IDENTIFIED_CONFIDENCE and required_count is not None:
            quantity_status, quantity_message = evaluate_dose_quantity_match(
                detected_count,
                required_count,
            )

        schedule = (
            get_medication_timing_verdict(med_name, plan_items, now, today_logs, tz_obj)
            if med_name and confidence in MEDCAM_IDENTIFIED_CONFIDENCE
            else {
                "status": "unknown",
                "scheduled_display": "",
                "message": "could not verify schedule — identify the pill first",
            }
        )

        enriched.append({
            **pill,
            "medication_name": med_name,
            "closest_match_name": closest_match,
            "match_confidence": confidence,
            "count": detected_count,
            "expected_count": required_count,
            "quantity_status": quantity_status,
            "quantity_message": quantity_message,
            "schedule": schedule,
        })
    return enriched


def get_medcam_pill_row_display(pill: dict) -> dict:
    med = (pill.get("medication_name") or "").strip() or None
    confidence = str(pill.get("match_confidence") or "none").lower()
    identified = bool(med and confidence in MEDCAM_IDENTIFIED_CONFIDENCE)
    closest = (pill.get("closest_match_name") or "").strip() or None

    if identified:
        label = f"Identified — {med}"
    else:
        label = "Not identified"

    sched_status = (pill.get("schedule") or {}).get("status")
    sched = pill.get("schedule") or {}
    qty_status = pill.get("quantity_status")

    if not identified:
        verdict = "Check manually"
        role = "warn" if closest else "neutral"
    elif qty_status == "high":
        verdict = "Too many pills"
        role = "danger"
    elif qty_status == "low":
        verdict = "Too few pills"
        role = "danger"
    elif qty_status == "unknown":
        verdict = "Verify count"
        role = "warn"
    elif sched_status == "prn_available":
        verdict = "Available"
        role = "info"
    elif sched_status == "prn_wait":
        verdict = sched.get("wait_label") or "Wait"
        role = "warn"
    elif sched_status == "prn_max":
        verdict = "Max reached for today"
        role = "danger"
    elif sched_status == "due_now":
        verdict = "Give now"
        role = "success"
    elif sched_status == "past_due":
        verdict = "Missed dose"
        role = "danger"
    elif sched_status == "not_due_yet":
        verdict = "Not due yet"
        role = "warn"
    elif sched_status == "taken":
        verdict = "Already taken"
        role = "warn"
    else:
        verdict = "Review schedule"
        role = "neutral"

    return {
        "label": label,
        "verdict": verdict,
        "role": role,
        "detail": build_medcam_pill_row_detail(pill, identified=identified, closest_match=closest),
    }


def build_medcam_pill_row_detail(
    pill: dict,
    *,
    identified: bool,
    closest_match: str | None = None,
) -> str:
    med = (pill.get("medication_name") or "").strip() or None
    confidence = str(pill.get("match_confidence") or "none").lower()
    description = (pill.get("description") or "unidentified pill").strip()
    closest = (closest_match or pill.get("closest_match_name") or "").strip() or None

    if not identified:
        if closest:
            return (
                f"Closest match: {closest} (not confident enough to confirm — requires "
                f"{MEDCAM_CONFIDENCE_THRESHOLD_LABEL}). Please verify manually against the pill pack."
            )
        if confidence == "low":
            return (
                f"We are not confident this matches a registered medication ({description}). "
                "Compare against the pill pack or re-take the photo with one pill type clearly visible."
            )
        return (
            f"We could not match the {description} to any registered reference photo. "
            "Check manually against the pill pack or re-take the photo."
        )

    sched = pill.get("schedule") or {}
    sched_status = sched.get("status")
    scheduled = sched.get("scheduled_display") or ""
    parts = []

    if sched_status == "prn_available":
        parts.append(
            f"{med} is as-needed (PRN). {sched.get('message', '')} "
            "Give only if symptoms require it and the care plan allows."
        )
    elif sched_status == "prn_wait":
        parts.append(
            f"{med} is as-needed (PRN). {sched.get('message', '')} "
            "Wait until the minimum interval has passed before giving another dose."
        )
    elif sched_status == "prn_max":
        parts.append(
            f"{med} is as-needed (PRN). {sched.get('message', '')} "
            "Do not give another dose today unless a doctor has advised otherwise."
        )
    elif sched_status == "due_now":
        parts.append(
            f"{med} is due now"
            + (f" (scheduled {scheduled})" if scheduled else "")
            + ". Confirm the count before giving."
        )
    elif sched_status == "not_due_yet":
        parts.append(
            f"{med} {sched.get('message', 'is not due yet')}. Do not give until the scheduled time."
        )
    elif sched_status == "past_due":
        parts.append(
            f"{med} {sched.get('message', 'was due earlier')}. "
            "Check the care plan for what to do about a missed dose, or call the doctor or "
            "pharmacist if unsure — don't double up on the next dose without checking first."
        )
    elif sched_status == "taken":
        parts.append(
            f"{med} was already logged as taken today. "
            "Do not give again unless the doctor advised a repeat dose."
        )
    else:
        parts.append(f"{med}: {sched.get('message', 'check the schedule before giving')}.")

    if pill.get("quantity_message"):
        qty_msg = str(pill["quantity_message"]).strip()
        if qty_msg:
            parts.append(qty_msg if qty_msg.endswith(".") else f"{qty_msg}.")

    if confidence == "medium":
        parts.append(
            f"Match confidence is medium (threshold for auto-identify: {MEDCAM_CONFIDENCE_THRESHOLD_LABEL}). "
            "Double-check against the pill pack if unsure."
        )

    return " ".join(parts)


def build_medcam_absence_warning_row(warning: dict) -> dict:
    med = warning["medication_name"]
    sched_status = warning["schedule_status"]
    if sched_status == "past_due":
        verdict = "Missed but absent"
        detail = (
            f"{med} was due earlier today but was not in this photo. "
            "If you still need to give it, check the care plan for late doses — do not double up "
            "without checking with the doctor or pharmacist first."
        )
    else:
        verdict = "Due but absent"
        detail = (
            f"{med} is due now but was not in this photo — did you mean to include it? "
            "Verify you have all medications for this dose before giving."
        )
    return {
        "label": f"Missing from photo — {med}",
        "verdict": verdict,
        "role": "warning",
        "detail": detail,
    }


def build_plan_absence_warnings(
    enriched_pills: list,
    plan_items: list,
    now: datetime,
    today_logs: list,
    tz_obj,
) -> list[dict]:
    identified_meds = {
        pill["medication_name"]
        for pill in enriched_pills
        if pill.get("medication_name") and pill.get("match_confidence") in MEDCAM_IDENTIFIED_CONFIDENCE
    }
    warnings = []
    for med in plan_items:
        timing = med.get("time") or med.get("timing") or ""
        if is_prn_timing(timing):
            continue
        name = med["name"]
        if name in identified_meds:
            continue
        verdict = get_medication_timing_verdict(name, plan_items, now, today_logs, tz_obj)
        status = verdict.get("status")
        if status not in ("due_now", "past_due"):
            continue
        warnings.append({
            "medication_name": name,
            "schedule_status": status,
            "schedule_message": verdict.get("message", ""),
        })
    return warnings


def build_medcam_pill_row_html(
    pill: dict,
    index: int,
    med_refs: list | None = None,
) -> str:
    row = get_medcam_pill_row_display(pill)
    med = (pill.get("medication_name") or "").strip()
    confidence = str(pill.get("match_confidence") or "none").lower()
    identified = bool(med and confidence in MEDCAM_IDENTIFIED_CONFIDENCE)
    thumb_html = ""
    if identified:
        thumb_html = build_medcam_reference_thumb_html(
            med,
            med_refs,
            find_reference=find_medication_reference,
        )
    return build_medcam_row_html_from_display(row, thumb_html=thumb_html)


def save_medcam_audit_record(
    patient_id,
    caregiver_id,
    caregiver_name: str,
    record: dict,
) -> None:
    patient_id = resolve_patient_id(patient_id)
    cache_key = f"medcam_audit_{patient_id}"
    cached = list(st.session_state.get(cache_key, []))
    cached.insert(0, record)
    st.session_state[cache_key] = cached[:40]
    try:
        supabase.table("shift_logs").insert({
            "caregiver_name": caregiver_name,
            "caregiver_id": caregiver_id,
            "source": "medcam_audit",
            "summary": json.dumps(record, default=str),
            "severity": record.get("severity", "monitor"),
            "patient_id": patient_id,
        }).execute()
    except Exception:
        try:
            supabase.table("shift_logs").insert({
                "caregiver_name": caregiver_name,
                "source": "medcam_audit",
                "summary": json.dumps(record, default=str),
                "severity": record.get("severity", "monitor"),
                "patient_id": patient_id,
            }).execute()
        except Exception:
            pass


def get_medcam_audit_records(patient_id=None, limit: int = 15) -> list[dict]:
    patient_id = resolve_patient_id(patient_id)
    cache_key = f"medcam_audit_{patient_id}"
    cached = st.session_state.get(cache_key)
    if isinstance(cached, list) and cached:
        return cached[:limit]
    records = []
    for row in cached_shift_logs(patient_id, limit=limit * 4):
        if row.get("source") != "medcam_audit":
            continue
        try:
            parsed = json.loads(row.get("summary") or "")
            if isinstance(parsed, dict):
                records.append(parsed)
        except (json.JSONDecodeError, TypeError):
            continue
        if len(records) >= limit:
            break
    st.session_state[cache_key] = records
    return records[:limit]


def build_medcam_audit_history_html(records: list) -> str:
    if not records:
        return (
            '<p class="cs-medcam-history-empty">No past checks yet. Run Check medication to '
            "create an audit entry.</p>"
        )
    items = []
    for record in records[:10]:
        when = html.escape(str(record.get("checked_at_display") or record.get("checked_at") or ""))
        title = html.escape(str(record.get("verdict_title") or "MedCam check"))
        pill_count = len(record.get("pills_identified") or [])
        warn_count = len(record.get("absence_warnings") or [])
        summary = html.escape(
            f"{pill_count} pill group(s) in photo"
            + (f", {warn_count} plan warning(s)" if warn_count else "")
        )
        items.append(
            f'<div class="cs-medcam-history-item">'
            f'<div class="cs-medcam-history-when">{when}</div>'
            f'<div class="cs-medcam-history-item-title">{title}</div>'
            f'<div class="cs-medcam-history-sub">{summary}</div>'
            f"</div>"
        )
    return "".join(items)


def build_medcam_verdict_panel_html(
    verdict: dict,
    enriched_pills: list,
    log_time: str,
    ai_result: dict | None = None,
    absence_warnings: list | None = None,
    med_refs: list | None = None,
) -> str:
    title = html.escape(str(verdict.get("title", "Review")))
    time_label = html.escape(str(log_time))
    result_class = html.escape(str(verdict.get("result_class", "warn")))

    if enriched_pills:
        rows_html = "".join(
            build_medcam_pill_row_html(pill, index, med_refs)
            for index, pill in enumerate(enriched_pills)
        )
    else:
        unidentified = ""
        if ai_result:
            unidentified = (ai_result.get("unidentified_description") or "").strip()
        detail = (
            f"We could not identify pills in this photo ({unidentified}). "
            "Please check manually or re-take the photo with pills clearly visible."
            if unidentified
            else "We could not identify any pills in this photo. "
            "Please check manually or re-take the photo with pills clearly visible."
        )
        rows_html = build_medcam_row_html_from_display({
            "label": "Not identified",
            "verdict": "Check manually",
            "role": "neutral",
            "detail": detail,
        })

    for idx, warning in enumerate(absence_warnings or []):
        row = build_medcam_absence_warning_row(warning)
        rows_html += build_medcam_row_html_from_display(row)

    return f"""
    <div class="cs-medcam-verdict-panel cs-medcam-verdict-panel--{result_class}">
      <div class="cs-medcam-verdict-header">
        <div class="cs-medcam-verdict-title">{title}</div>
        <div class="cs-medcam-verdict-time">Checked at {time_label}</div>
      </div>
      <div class="cs-medcam-pill-rows">{rows_html}</div>
    </div>
    """


def compose_medcam_verdict(enriched_pills: list, ai_result: dict) -> dict:
    lines = []

    matched = [
        pill for pill in enriched_pills
        if pill.get("medication_name") and pill.get("match_confidence") in MEDCAM_IDENTIFIED_CONFIDENCE
    ]
    uncertain = [
        pill for pill in enriched_pills
        if not pill.get("medication_name") or pill.get("match_confidence") not in MEDCAM_IDENTIFIED_CONFIDENCE
    ]

    total_groups = len(enriched_pills)
    if total_groups == 0:
        unidentified = (ai_result.get("unidentified_description") or "").strip()
        if unidentified:
            lines.append(f"We could not identify pills in this photo ({unidentified}).")
        else:
            lines.append("We could not identify any pills in this photo.")
        lines.append("Please check manually or re-take the photo with pills clearly visible.")
        return {
            "title": "Could not verify",
            "message": " ".join(lines),
            "severity": "urgent",
            "result_class": "fail",
        }

    if total_groups == 1 and matched:
        inventory = f"You have 1 pill group in your hand: {matched[0]['medication_name']}."
    elif matched:
        names = [pill["medication_name"] for pill in matched]
        extra = len(uncertain)
        if extra:
            inventory = (
                f"You have {total_groups} pill groups in your hand: "
                f"{_format_med_name_list(names)}, and {extra} we could not identify confidently."
            )
        else:
            inventory = (
                f"You have {len(matched)} medications in your hand: {_format_med_name_list(names)}."
            )
    elif uncertain:
        inventory = (
            f"We see {len(uncertain)} pill group(s) but could not confidently match "
            "them to your registered medications."
        )
    else:
        inventory = f"We see {total_groups} pill group(s) in your hand."
    lines.append(inventory)

    for pill in matched:
        med = pill["medication_name"]
        sched = pill.get("schedule") or {}
        sched_status = sched.get("status")
        detail_parts = [f"**{med}**"]

        if sched_status == "due_now":
            detail_parts.append("is due now — you can give this dose after confirming the count.")
        elif sched_status == "prn_available":
            detail_parts.append("is available as-needed (PRN) — give only if symptoms require it.")
        elif sched_status == "prn_wait":
            detail_parts.append(f"is as-needed (PRN) — {sched.get('message', 'wait before next dose')}.")
        elif sched_status == "prn_max":
            detail_parts.append(f"is as-needed (PRN) — {sched.get('message', 'max reached for today')}.")
        elif sched_status == "not_due_yet":
            detail_parts.append(
                f"is not due yet ({sched.get('message', 'check the schedule')}) — do not give now."
            )
        elif sched_status == "past_due":
            detail_parts.append(
                f"{sched.get('message', 'was due earlier')} — do not give now. "
                "Check the care plan for what to do about a missed dose."
            )
        elif sched_status == "taken":
            detail_parts.append("was already logged as taken today — do not give again unless the doctor said to.")
        else:
            detail_parts.append(f"{sched.get('message', 'check the schedule before giving')}.")

        if pill.get("quantity_message"):
            detail_parts.append(pill["quantity_message"] + ".")

        lines.append(" ".join(detail_parts))

    for pill in uncertain:
        desc = (pill.get("description") or "unidentified pill").strip()
        conf = pill.get("match_confidence") or "none"
        if conf == "low":
            lines.append(
                f"We are not confident about the {desc} — please check manually or re-take the photo."
            )
        else:
            lines.append(
                f"We could not confidently match the {desc} to a registered medication — "
                "please check manually or re-take the photo."
            )

    unidentified = (ai_result.get("unidentified_description") or "").strip()
    if unidentified and not uncertain:
        lines.append(f"Note: {unidentified}")

    has_quantity_issue = any(
        pill.get("quantity_status") != "ok" for pill in matched
    )
    has_schedule_block = any(
        pill.get("schedule", {}).get("status") not in ("due_now", "prn_available")
        for pill in matched
    )
    if not matched:
        severity = "urgent"
    elif uncertain or has_quantity_issue or has_schedule_block:
        severity = "monitor"
    else:
        severity = "ok"

    if severity == "ok" and matched:
        title = "Ready to give"
    elif matched and severity == "monitor":
        title = "Review before giving"
    elif matched:
        title = "Verified"
    else:
        title = "Could not verify"

    result_class = "success" if severity == "ok" else "warn" if severity == "monitor" else "fail"
    return {
        "title": title,
        "message": " ".join(lines),
        "severity": severity,
        "result_class": result_class,
    }


def build_medcam_shift_log_summary(enriched_pills: list, verdict: dict, log_time: str) -> str:
    parts = []
    for pill in enriched_pills:
        med = pill.get("medication_name") or pill.get("description") or "unknown pill"
        conf = pill.get("match_confidence") or "none"
        count = pill.get("count", "?")
        sched = (pill.get("schedule") or {}).get("status", "unknown")
        parts.append(f"{med}: x{count} ({conf} confidence, schedule={sched})")
    summary = "; ".join(parts) if parts else verdict.get("message", "MedCam check")
    return f"{summary}. Verdict: {verdict.get('title', 'Review')}. Checked at {log_time}."


def run_medcam_verification(
    *,
    base64_image: str,
    med_refs: list,
    plan_items: list,
    latest_plan: dict | None,
    discharge_plan: str,
    patient_id,
    caregiver_name: str,
    caregiver_id,
) -> dict:
    tz_obj, tz_name = get_schedule_tz()
    now = datetime.now(tz_obj)
    today = now.date().isoformat()
    today_logs = [
        log for log in cached_medication_logs(patient_id)
        if parse_log_local_date(log, tz_obj) == today
    ]

    med_refs = filter_med_refs_for_plan(load_live_medication_references(patient_id), plan_items)

    plan_dose_prompt = build_plan_dose_prompt_block(plan_items, latest_plan, med_refs)
    registry_prompt = build_med_ref_registry_prompt(med_refs, plan_items)
    schedule_lines = []
    for med in plan_items:
        verdict = get_medication_timing_verdict(
            med["name"], plan_items, now, today_logs, tz_obj,
        )
        schedule_lines.append(
            f"- {med['name']}: {verdict['message']} "
            f"(scheduled: {med.get('time') or med.get('timing') or 'see plan'})"
        )
    schedule_context = "\n".join(schedule_lines)

    ref_images_content = []
    for ref in med_refs:
        meta = parse_ref_meta(ref)
        plan_match = find_plan_item(ref["medication_name"], plan_items)
        meta_label = format_ref_meta_label(meta, plan_match)
        ref_images_content.append({
            "type": "text",
            "text": f"Reference front view for {ref['medication_name']} ({meta_label}):",
        })
        ref_images_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{ref['image_b64']}"},
        })
        back_b64 = meta.get("back_image_b64")
        if back_b64:
            ref_images_content.append({
                "type": "text",
                "text": f"Reference back view for {ref['medication_name']}:",
            })
            ref_images_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{back_b64}"},
            })

    user_content = ref_images_content + [
        {
            "type": "text",
            "text": (
                f"Registered pill references:\n{registry_prompt}\n\n"
                f"{plan_dose_prompt}\n\n"
                f"Current local time: {now.strftime('%A %d %b %Y, %H:%M')} ({tz_name})\n"
                f"Per-medication schedule status now:\n{schedule_context}\n\n"
                f"Structured discharge plan summary:\n{discharge_plan}\n\n"
                "Identify every pill group in this verification photo. "
                "For each group, set count to the number of individual pills you can see — "
                "not the prescribed dose from the plan:"
            ),
        },
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
    ]

    ai_result = ask_ai(build_medcam_verification_system_prompt(), user_content)
    if ai_result.get("error"):
        return {"error": True, "message": ai_result.get("message", "Verification failed.")}

    enriched = enrich_medcam_pill_results(
        ai_result,
        plan_items,
        now,
        today_logs,
        tz_obj,
        patient_id=patient_id,
    )
    absence_warnings = build_plan_absence_warnings(
        enriched, plan_items, now, today_logs, tz_obj,
    )
    verdict = compose_medcam_verdict(enriched, ai_result)
    if absence_warnings:
        verdict = {
            **verdict,
            "severity": "monitor" if verdict.get("severity") != "urgent" else verdict["severity"],
            "result_class": "warn" if verdict.get("result_class") != "fail" else verdict["result_class"],
            "title": "Review before giving" if verdict.get("title") == "Ready to give" else verdict.get("title"),
        }
    log_time = format_medcam_log_time()
    photo_reference = hashlib.sha256(base64_image.encode("utf-8")).hexdigest()[:16]
    audit_record = {
        "checked_at": now.isoformat(),
        "checked_at_display": log_time,
        "photo_reference": photo_reference,
        "pills_identified": [
            {
                "medication_name": pill.get("medication_name"),
                "closest_match_name": pill.get("closest_match_name"),
                "description": pill.get("description"),
                "confidence": pill.get("match_confidence"),
                "count": pill.get("count"),
                "schedule_status": (pill.get("schedule") or {}).get("status"),
                "quantity_status": pill.get("quantity_status"),
            }
            for pill in enriched
        ],
        "plan_crosscheck": [
            {
                "medication_name": med["name"],
                "schedule_status": get_medication_timing_verdict(
                    med["name"], plan_items, now, today_logs, tz_obj,
                ).get("status"),
                "is_prn": is_prn_timing(med.get("time") or med.get("timing") or ""),
            }
            for med in plan_items
        ],
        "absence_warnings": absence_warnings,
        "verdict_title": verdict.get("title"),
        "severity": verdict.get("severity"),
    }
    return {
        "error": False,
        "ai_result": ai_result,
        "enriched_pills": enriched,
        "absence_warnings": absence_warnings,
        "verdict": verdict,
        "log_time": log_time,
        "audit_record": audit_record,
        "shift_log_summary": build_medcam_shift_log_summary(enriched, verdict, log_time),
    }


MEDCAM_PENDING_UPLOAD_KEY = "medcam_pending_upload_sig"


def apply_medcam_verification_result(
    result: dict,
    *,
    patient_id,
    caregiver_name: str,
    caregiver_id,
) -> None:
    """Persist a completed MedCam verification result (success or error message)."""
    resolved_id = resolve_patient_id(patient_id)
    if result.get("error"):
        st.session_state[f"medcam_error_{resolved_id}"] = result.get(
            "message", "Verification failed.",
        )
        return

    st.session_state.pop(f"medcam_error_{resolved_id}", None)
    verdict = result["verdict"]
    save_medcam_audit_record(
        patient_id,
        caregiver_id,
        caregiver_name,
        result["audit_record"],
    )
    st.session_state[f"medcam_last_{resolved_id}"] = result
    save_shift_log(
        caregiver_name=caregiver_name,
        source="medication_check",
        summary=result["shift_log_summary"],
        severity=verdict.get("severity", "monitor"),
        caregiver_id=caregiver_id,
        patient_id=patient_id,
    )
    invalidate_patient_activity_cache(patient_id)


def process_medcam_pending_verification(
    *,
    uploaded_image,
    upload_sig: str,
    plan_refs,
    plan_items,
    latest_plan,
    discharge_plan,
    patient_id,
    caregiver_name: str,
    caregiver_id,
) -> bool:
    """Run a queued MedCam check on the next rerun so loading clears after completion."""
    pending_sig = st.session_state.get(MEDCAM_PENDING_UPLOAD_KEY)
    if not pending_sig:
        return False

    set_careshield_active_tab("medcam")
    if not uploaded_image or upload_sig != pending_sig:
        st.session_state.pop(MEDCAM_PENDING_UPLOAD_KEY, None)
        st.warning("Upload expired — please choose the photo again and tap Check medication.")
        return True

    md_html(build_loading_banner_html("Checking each pill against your registered medications..."))
    try:
        base64_image = base64.b64encode(uploaded_image.read()).decode("utf-8")
        result = run_medcam_verification(
            base64_image=base64_image,
            med_refs=plan_refs,
            plan_items=plan_items,
            latest_plan=latest_plan,
            discharge_plan=discharge_plan,
            patient_id=patient_id,
            caregiver_name=caregiver_name,
            caregiver_id=caregiver_id,
        )
        apply_medcam_verification_result(
            result,
            patient_id=patient_id,
            caregiver_name=caregiver_name,
            caregiver_id=caregiver_id,
        )
    except Exception as exc:
        resolved_id = resolve_patient_id(patient_id)
        st.session_state[f"medcam_error_{resolved_id}"] = (
            f"Something went wrong during verification: {exc}"
        )
    finally:
        st.session_state.pop(MEDCAM_PENDING_UPLOAD_KEY, None)
    st.rerun()
    return True


def get_schedule_tz():
    tz_name = user_timezone or "UTC"
    try:
        return ZoneInfo(tz_name), tz_name
    except Exception:
        return timezone.utc, "UTC"


def get_active_plan_items(patient_id=None):
    patient_id = resolve_patient_id(patient_id or st.session_state.get("selected_patient_id"))
    if not patient_id:
        return []
    return get_patient_medications_display(patient_id)


def get_actionable_dose_nudges(plan_items: list, patient_id) -> list[dict]:
    tz_obj, _ = get_schedule_tz()
    now = datetime.now(tz_obj)
    today = now.date().isoformat()
    resolved_patient_id = resolve_patient_id(patient_id)
    dose_events = build_dose_events(plan_items)
    all_logs = cached_medication_logs(resolved_patient_id)
    today_logs = [
        log for log in all_logs
        if parse_log_local_date(log, tz_obj) == today
    ]
    nudges = []
    for dose in dose_events:
        if dose_ui_state(dose, now, today_logs, tz_obj) != "actionable":
            continue
        nudges.append({
            "medication_name": dose["medication_name"],
            "friendly_time": format_friendly_dose_time(dose["hour"], dose["minute"]),
            "display_time": dose["display_time"],
        })
    return sorted(nudges, key=lambda item: (item["display_time"], item["medication_name"]))


def build_missed_dose_nudge_html(nudges: list[dict], sticky: bool = False) -> str:
    if not nudges:
        return ""
    items = []
    for nudge in nudges[:3]:
        med_name = html.escape(_short_med_name(nudge["medication_name"], 42))
        friendly_time = html.escape(nudge["friendly_time"])
        items.append(
            f'<div class="cs-dose-nudge-item">Have you given the {friendly_time} {med_name}?</div>'
        )
    extra = ""
    if len(nudges) > 3:
        extra = f'<div class="cs-dose-nudge-more">+ {len(nudges) - 3} more dose(s) due now</div>'
    sticky_class = " cs-dose-nudge-sticky" if sticky else ""
    return (
        f'<div class="cs-dose-nudge{sticky_class}">'
        '<div class="cs-dose-nudge-title">Medication check-in</div>'
        + "".join(items)
        + extra
        + '<div class="cs-dose-nudge-hint">Log doses in MedCam when given.</div>'
        + "</div>"
    )


def apply_auto_missed_doses(
    dose_events: list,
    now: datetime,
    today_logs: list,
    tz_obj,
    patient_id,
    caregiver_id,
) -> list:
    """Log overdue doses once and return an updated in-memory log list (no rerun)."""
    date_iso = now.date().isoformat()
    updated = list(today_logs)
    for dose in dose_events:
        if dose_log_for_today(dose, updated, date_iso, tz_obj):
            continue
        if dose_minutes_until(dose, now, tz_obj) >= -60:
            continue
        session_key = f"auto_missed:{date_iso}:{dose_log_key(dose['medication_name'], dose['time_label'], date_iso)}"
        if st.session_state.get(session_key):
            if not dose_log_for_today(dose, updated, date_iso, tz_obj):
                updated.append({
                    "medication_name": dose["medication_name"],
                    "scheduled_time": dose["time_label"],
                    "status": "missed",
                    "logged_at": datetime.now(timezone.utc).isoformat(),
                    "caregiver_id": caregiver_id,
                    "patient_id": patient_id,
                })
            continue
        try:
            log_medication_missed(
                patient_id,
                dose["medication_name"],
                dose["time_label"],
                caregiver_id,
            )
            st.session_state[session_key] = True
            updated.append({
                "medication_name": dose["medication_name"],
                "scheduled_time": dose["time_label"],
                "status": "missed",
                "logged_at": datetime.now(timezone.utc).isoformat(),
                "caregiver_id": caregiver_id,
                "patient_id": patient_id,
            })
        except Exception:
            st.session_state[session_key] = True
            updated.append({
                "medication_name": dose["medication_name"],
                "scheduled_time": dose["time_label"],
                "status": "missed",
                "logged_at": datetime.now(timezone.utc).isoformat(),
                "caregiver_id": caregiver_id,
                "patient_id": patient_id,
            })
    return updated


def build_adherence_report_csv(logs: list, tz_obj) -> str:
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Date", "Medication", "Scheduled Time", "Status", "Logged At", "Caregiver"])
    for log in logs:
        raw = log.get("logged_at") or ""
        date_str = ""
        logged_at_str = ""
        if raw:
            try:
                logged_at = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                if logged_at.tzinfo is None:
                    logged_at = logged_at.replace(tzinfo=timezone.utc)
                local_dt = logged_at.astimezone(tz_obj)
                date_str = local_dt.date().isoformat()
                logged_at_str = local_dt.strftime("%Y-%m-%d %I:%M %p").lstrip("0")
            except (ValueError, TypeError):
                logged_at_str = raw
        caregiver = resolve_caregiver_label(log.get("caregiver_id")) or log.get("caregiver_id") or ""
        writer.writerow([
            date_str,
            log.get("medication_name", ""),
            log.get("scheduled_time", ""),
            log.get("status", ""),
            logged_at_str,
            caregiver,
        ])
    return buffer.getvalue()


DOSE_STATUS_COLORS = {
    "taken": "#166534",
    "missed": "#B91C1C",
    "upcoming": "#C2410C",
    "later": "#1D4ED8",
}

APPLE_CLOCK_RING = "#E5E5EA"
APPLE_CLOCK_RING_OUTER = "#D1D1D6"
APPLE_CLOCK_FACE = "#FFFFFF"
APPLE_CLOCK_HAND = "#2C2C2E"
APPLE_CLOCK_LABEL = "#3C3C43"
APPLE_CLOCK_LABEL_MUTED = "#8E8E93"
APPLE_CLOCK_TICK = "#E5E5EA"

GRAPHIC_CLOCK_INK = "#2C2C2E"
GRAPHIC_CLOCK_SQUIRCLE_BG = "#1C1C1E"
GRAPHIC_CLOCK_FACE_FILL = "#F2F2F7"
GRAPHIC_CLOCK_FACE_RADIUS = 38
GRAPHIC_CLOCK_SQUIRCLE_RADIUS = 22
GRAPHIC_CLOCK_NUMERAL = "#1D1D1F"
GRAPHIC_CLOCK_HAND = "#2C2C2E"
GRAPHIC_CLOCK_SLICE_HALF_WIDTH = 7.5

DOSE_CARD_THEMES = {
    "not_yet": ("cs-sbar-situation", "Not due yet"),
    "actionable": ("cs-sbar-assessment", "Due now"),
    "taken": ("cs-dose-card-taken", "Taken"),
    "missed": ("cs-sbar-recommendation", "Missed"),
}

PRN_DOSE_CARD_THEMES = {
    "prn_available": ("cs-dose-card-prn-available", "Available"),
    "prn_wait": ("cs-dose-card-prn-wait", "Wait"),
    "prn_max": ("cs-dose-card-prn-max", "Max reached for today"),
}

CLOCK_SLICE_COLORS = {
    "not_yet": {"fill": "#D8D8DC", "line": "#C7C7CC", "label": "#AEAEB2"},
    "actionable": {"fill": "#FFD60A", "line": "#E6C200", "label": APPLE_CLOCK_LABEL},
    "taken": {"fill": "#34C759", "line": "#2EB04C", "label": APPLE_CLOCK_LABEL},
    "missed": {"fill": "#FF453A", "line": "#E03E34", "label": APPLE_CLOCK_LABEL},
}

CHECKLIST_STATUS_COLORS = {
    "not_yet": "#9CA3AF",
    "actionable": "#EAB308",
    "taken": "#166534",
    "missed": "#B91C1C",
}


def checklist_status_square_html(state: str) -> str:
    color = CHECKLIST_STATUS_COLORS.get(state, CHECKLIST_STATUS_COLORS["not_yet"])
    return f'<span class="cs-dose-status-square" style="background:{color};"></span>'


def build_dose_card_html(dose: dict, state: str, plan_item: dict | None = None) -> str:
    theme_class, status_label = DOSE_CARD_THEMES.get(state, DOSE_CARD_THEMES["not_yet"])
    schedule_badge = build_schedule_badge_html(plan_item or {})
    return f"""
    <div class="cs-sbar-card {theme_class} cs-dose-card">
      <div class="cs-sbar-label">{html.escape(dose['medication_name'])}</div>
      <div class="cs-sbar-body">
        <div class="cs-dose-card-badges">{schedule_badge}<span class="cs-dose-card-clock">{html.escape(dose['display_time'])}</span></div>
        <div class="cs-dose-card-status">{html.escape(status_label)}</div>
      </div>
    </div>
    """


def build_prn_dose_card_html(med: dict, status: dict) -> str:
    state = status.get("status", "prn_available")
    theme_class, status_label = PRN_DOSE_CARD_THEMES.get(
        state, PRN_DOSE_CARD_THEMES["prn_available"],
    )
    if state == "prn_wait" and status.get("wait_label"):
        status_label = status["wait_label"]
    doses = status.get("doses_today", 0)
    max_d = status.get("max_per_day", DEFAULT_PRN_MAX_PER_DAY)
    return f"""
    <div class="cs-sbar-card {theme_class} cs-dose-card cs-dose-card--prn">
      <div class="cs-sbar-label">{html.escape(med['name'])}</div>
      <div class="cs-sbar-body">
        <div class="cs-dose-card-badges"><span class="cs-dose-card-clock">PRN · {doses}/{max_d} today</span></div>
        <div class="cs-dose-card-status">{html.escape(status_label)}</div>
      </div>
    </div>
    """


def build_prn_clock_chips_html(
    prn_meds: list,
    today_logs: list,
    tz_obj,
    now: datetime,
) -> str:
    if not prn_meds:
        return ""
    chips = []
    for med in prn_meds:
        status = get_prn_medication_status(med["name"], med, now, today_logs, tz_obj)
        short_name = html.escape(_short_med_name(med["name"], 28))
        chips.append(
            f'<span class="cs-prn-chip cs-prn-chip--{html.escape(status["status"].replace("prn_", ""))}">'
            f"{short_name}: {status['doses_today']} of {status['max_per_day']} doses used today"
            f"</span>"
        )
    return f'<div class="cs-prn-chip-row">{"".join(chips)}</div>'


def classify_dose(dose: dict, now: datetime, today_logs: list, tz_obj) -> tuple[str, str, str]:
    state = dose_ui_state(dose, now, today_logs, tz_obj)
    if state == "taken":
        return "taken", DOSE_STATUS_COLORS["taken"], "Taken"
    if state == "missed":
        return "missed", DOSE_STATUS_COLORS["missed"], "Missed"
    if state == "actionable":
        return "upcoming", DOSE_STATUS_COLORS["upcoming"], "Due soon"
    return "later", DOSE_STATUS_COLORS["later"], "Later today"


def clock_dose_visual(
    dose: dict,
    now: datetime,
    today_logs: list,
    tz_obj,
) -> dict:
    state = dose_ui_state(dose, now, today_logs, tz_obj)
    status_labels = {
        "taken": "Taken",
        "missed": "Missed",
        "actionable": "Due now",
        "not_yet": "Not yet",
    }
    palette = CLOCK_SLICE_COLORS.get(state, CLOCK_SLICE_COLORS["not_yet"])
    return {
        "state": state,
        "fill": palette["fill"],
        "line_color": palette["line"],
        "line_width": 0.75,
        "label_color": palette["label"],
        "status_label": status_labels.get(state, "Not yet"),
    }


def clock_xy_from_angle(angle_deg: float, radius: float) -> tuple[float, float]:
    fraction = angle_deg / 360
    theta = fraction * 2 * math.pi
    return radius * math.sin(theta), radius * math.cos(theta)


def _short_med_name(name: str, max_len: int = 24) -> str:
    text = str(name).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _svg_polar(cx: float, cy: float, radius: float, angle_deg: float) -> tuple[float, float]:
    radians = math.radians(angle_deg)
    return cx + radius * math.sin(radians), cy - radius * math.cos(radians)


def _svg_sector_path(cx: float, cy: float, radius: float, start_deg: float, end_deg: float) -> str:
    x1, y1 = _svg_polar(cx, cy, radius, start_deg)
    x2, y2 = _svg_polar(cx, cy, radius, end_deg)
    span = (end_deg - start_deg) % 360
    if span == 0:
        span = 360
    large_arc = 1 if span > 180 else 0
    return (
        f"M {cx:.3f} {cy:.3f} L {x1:.3f} {y1:.3f} "
        f"A {radius:.3f} {radius:.3f} 0 {large_arc} 1 {x2:.3f} {y2:.3f} Z"
    )


def build_graphic_medication_clock_html(
    dose_events: list,
    now: datetime,
    today_logs: list,
    tz_obj,
    tz_name: str,
) -> str:
    cx = cy = 50.0
    face_r = GRAPHIC_CLOCK_FACE_RADIUS
    initial_angle = (now.hour + now.minute / 60) / 24 * 360
    tz_json = json.dumps(tz_name)
    squircle_r = GRAPHIC_CLOCK_SQUIRCLE_RADIUS

    parts = [
        '<div class="cs-graphic-clock-card" style="'
        "display:flex;flex-direction:column;align-items:center;justify-content:center;"
        "width:100%;padding:8px 4px 4px;"
        '">',
        '<svg viewBox="0 0 100 100" role="img" aria-label="24-hour medication schedule clock" '
        'style="width:min(320px,100%);aspect-ratio:1;height:auto;overflow:visible;">',
        "<defs>",
        '<filter id="clock-face-shadow" x="-30%" y="-30%" width="160%" height="160%">',
        '<feDropShadow dx="0" dy="1.6" stdDeviation="2.4" flood-color="#000000" flood-opacity="0.22"/>',
        "</filter>",
        '<filter id="clock-hand-shadow" x="-30%" y="-30%" width="160%" height="160%">',
        '<feDropShadow dx="0" dy="0.6" stdDeviation="0.5" flood-color="#000000" flood-opacity="0.18"/>',
        "</filter>",
        f'<clipPath id="clock-face-clip"><circle cx="{cx}" cy="{cy}" r="{face_r}"/></clipPath>',
        "</defs>",
        f'<rect x="3" y="3" width="94" height="94" rx="{squircle_r}" ry="{squircle_r}" '
        f'fill="{GRAPHIC_CLOCK_SQUIRCLE_BG}"/>',
        f'<circle cx="{cx}" cy="{cy}" r="{face_r}" fill="{GRAPHIC_CLOCK_FACE_FILL}" '
        f'filter="url(#clock-face-shadow)"/>',
    ]

    sorted_doses = sorted(
        dose_events,
        key=lambda item: (item["hour"], item["minute"], item["medication_name"]),
    )
    half_w = GRAPHIC_CLOCK_SLICE_HALF_WIDTH
    parts.append(f'<g clip-path="url(#clock-face-clip)">')
    for (_hour, _minute), slot_doses in group_doses_by_schedule_time(sorted_doses):
        slot_entries = []
        for dose in slot_doses:
            visual = clock_dose_visual(dose, now, today_logs, tz_obj)
            slot_entries.append({
                "dose": dose,
                "visual": visual,
                "medication_name": dose["medication_name"],
                "display_time": dose["display_time"],
                "status_label": visual["status_label"],
                "state": visual["state"],
            })
        winning_state = winning_clock_slot_status([entry["state"] for entry in slot_entries])
        winning_palette = CLOCK_SLICE_COLORS.get(winning_state, CLOCK_SLICE_COLORS["not_yet"])
        anchor = slot_entries[0]["dose"]
        angle = dose_angle_deg(anchor["hour"], anchor["minute"])
        slice_path = _svg_sector_path(cx, cy, face_r, angle - half_w, angle + half_w)
        title = html.escape(format_clock_slot_tooltip(slot_entries))
        parts.append(
            f'<path d="{slice_path}" fill="{winning_palette["fill"]}" '
            f'stroke="{winning_palette["line"]}" stroke-width="0.75">'
            f"<title>{title}</title></path>"
        )
    parts.append("</g>")

    for hour in CLOCK_FACE_HOUR_LABELS:
        angle = clock_hour_label_angle_deg(hour)
        label_r = face_r - clock_label_radius_offset(hour)
        nx, ny = _svg_polar(cx, cy, label_r, angle)
        font_size = clock_label_font_size(hour)
        parts.append(
            f'<text x="{nx:.3f}" y="{ny:.3f}" text-anchor="middle" dominant-baseline="middle" '
            f'font-family="-apple-system, SF Pro Display, Helvetica Neue, Arial, sans-serif" '
            f'font-size="{font_size}" font-weight="600" fill="{GRAPHIC_CLOCK_NUMERAL}">{hour}</text>'
        )

    hand_len = face_r - 9
    parts.extend([
        f'<g id="clock-hand" transform="rotate({initial_angle:.4f} {cx} {cy})" '
        'style="transition:transform 0.9s cubic-bezier(0.4, 0, 0.2, 1);">',
        f'<line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy - hand_len}" stroke="{GRAPHIC_CLOCK_HAND}" '
        'stroke-width="2.8" stroke-linecap="round" filter="url(#clock-hand-shadow)"/>',
        "</g>",
        f'<circle cx="{cx}" cy="{cy}" r="2.4" fill="{GRAPHIC_CLOCK_HAND}"/>',
        f'<circle cx="{cx}" cy="{cy}" r="1.1" fill="{GRAPHIC_CLOCK_FACE_FILL}"/>',
        "</svg>",
        f"""<script>
(function () {{
  const tz = {tz_json};
  const cx = {cx};
  const cy = {cy};
  function clockHandAngle() {{
    const parts = new Intl.DateTimeFormat("en-GB", {{
      timeZone: tz,
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    }}).formatToParts(new Date());
    let hour = 0;
    let minute = 0;
    for (const part of parts) {{
      if (part.type === "hour") hour = parseInt(part.value, 10);
      if (part.type === "minute") minute = parseInt(part.value, 10);
    }}
    return ((hour + minute / 60) / 24) * 360;
  }}
  function updateClockHand() {{
    const hand = document.getElementById("clock-hand");
    if (!hand) return;
    hand.setAttribute("transform", "rotate(" + clockHandAngle() + " " + cx + " " + cy + ")");
  }}
  updateClockHand();
  setInterval(updateClockHand, 60000);
}})();
</script>""",
        "</div>",
    ])
    return "".join(parts)


def build_med_schedule_context(plan_items: list, caregiver_label: str, patient_id) -> dict:
    tz_name = user_timezone or "UTC"
    try:
        tz_obj = ZoneInfo(tz_name)
    except Exception:
        tz_obj = timezone.utc
    now = datetime.now(tz_obj)
    today = now.date().isoformat()
    resolved_patient_id = resolve_patient_id(patient_id)
    caregiver_id = resolve_caregiver_id(caregiver_label)
    _scheduled_meds, prn_meds = partition_plan_items(plan_items)
    dose_events = build_dose_events(plan_items)
    all_logs = cached_medication_logs(resolved_patient_id)
    today_logs = [
        log for log in all_logs
        if parse_log_local_date(log, tz_obj) == today
    ]
    if dose_events:
        today_logs = apply_auto_missed_doses(
            dose_events, now, today_logs, tz_obj, resolved_patient_id, caregiver_id
        )
    prn_chips_html = build_prn_clock_chips_html(prn_meds, today_logs, tz_obj, now)
    clock_height = 360
    return {
        "tz_name": tz_name,
        "tz_obj": tz_obj,
        "now": now,
        "dose_events": dose_events,
        "prn_meds": prn_meds,
        "today_logs": today_logs,
        "resolved_patient_id": resolved_patient_id,
        "caregiver_id": caregiver_id,
        "prn_chips_html": prn_chips_html,
        "clock_height": clock_height,
    }


def render_medcam_dose_cards_column(plan_items: list, caregiver_label: str, patient_id) -> None:
    ctx = build_med_schedule_context(plan_items, caregiver_label, patient_id)
    dose_events = ctx["dose_events"]
    prn_meds = ctx["prn_meds"]
    now = ctx["now"]
    today_logs = ctx["today_logs"]
    tz_obj = ctx["tz_obj"]
    resolved_patient_id = ctx["resolved_patient_id"]
    caregiver_id = ctx["caregiver_id"]

    if not dose_events and not prn_meds:
        st.info("No scheduled dose times found in the active plan.")
        return

    for dose in dose_events:
        state = dose_ui_state(dose, now, today_logs, tz_obj)
        plan_match = find_plan_item(dose["medication_name"], plan_items)
        md_html(build_medcam_scheduled_dose_card_html(dose, state, plan_match))
        if state == "actionable":
            if st.button(
                "Mark as Taken",
                key=f"take_{med_slug(dose['medication_name'])}_{dose['time_label']}",
                use_container_width=True,
            ):
                log_medication_taken(
                    resolved_patient_id,
                    dose["medication_name"],
                    dose["time_label"],
                    caregiver_id,
                )
                invalidate_patient_activity_cache(resolved_patient_id)
                st.rerun()
        elif state == "not_yet":
            st.button(
                f"Not yet — due at {dose['display_time']}",
                key=f"wait_{med_slug(dose['medication_name'])}_{dose['time_label']}",
                use_container_width=True,
                disabled=True,
            )

    for med in prn_meds:
        prn_status = get_prn_medication_status(med["name"], med, now, today_logs, tz_obj)
        md_html(build_medcam_prn_dose_card_html(med, prn_status))
        slug = med_slug(med["name"])
        if prn_status["status"] == "prn_available":
            if st.button(
                "Log PRN dose taken",
                key=f"prn_take_{slug}",
                use_container_width=True,
            ):
                log_medication_prn_taken(
                    resolved_patient_id,
                    med["name"],
                    caregiver_id,
                )
                invalidate_patient_activity_cache(resolved_patient_id)
                st.rerun()
        elif prn_status["status"] == "prn_wait":
            st.button(
                prn_status.get("wait_label", "Wait"),
                key=f"prn_wait_{slug}",
                use_container_width=True,
                disabled=True,
            )
        else:
            st.button(
                "Max reached for today",
                key=f"prn_max_{slug}",
                use_container_width=True,
                disabled=True,
            )


def render_medcam_clock_column(plan_items: list, caregiver_label: str, patient_id) -> None:
    ctx = build_med_schedule_context(plan_items, caregiver_label, patient_id)
    dose_events = ctx["dose_events"]
    prn_meds = ctx["prn_meds"]
    if not dose_events and not prn_meds:
        return

    md_html('<div class="cs-medcam-clock-shell"></div>')
    if dose_events:
        components.html(
            build_graphic_medication_clock_html(
                dose_events,
                ctx["now"],
                ctx["today_logs"],
                ctx["tz_obj"],
                ctx["tz_name"],
            ),
            height=ctx["clock_height"],
            scrolling=False,
        )

    md_html("""
    <div class="cs-clock-legend">
      <span class="cs-clock-legend-item"><span class="cs-dose-status-square" style="background:#9CA3AF;"></span> Not due yet</span>
      <span class="cs-clock-legend-item"><span class="cs-dose-status-square" style="background:#EAB308;"></span> Due now</span>
      <span class="cs-clock-legend-item"><span class="cs-dose-status-square" style="background:#166534;"></span> Taken</span>
      <span class="cs-clock-legend-item"><span class="cs-dose-status-square" style="background:#B91C1C;"></span> Missed</span>
      <span class="cs-clock-legend-item"><span class="cs-dose-status-square" style="background:var(--cs-blue);"></span> PRN (as needed)</span>
    </div>
    """)


def render_med_schedule_section(plan_items: list, caregiver_label: str, patient_id):
    left, right = st.columns([1, 1], gap="large")
    with left:
        render_medcam_dose_cards_column(plan_items, caregiver_label, patient_id)
    with right:
        render_medcam_clock_column(plan_items, caregiver_label, patient_id)



DOCUMENTS_EXTRACT_SYSTEM_PROMPT = """You are a medical assistant AI reading a hospital discharge document.
Extract the patient's medication plan and current diagnoses.
Respond with ONLY a JSON object with:
1. "medications": a JSON array of medication objects for medicines the patient should START, CONTINUE, or have a DOSE CHANGE for. Each object must include:
   - "name": medication name
   - "dosage": strength per pill (e.g. 500mg, 20mg)
   - "timing": when to take it (e.g. 8:00 AM, twice daily)
   - "pills_per_dose": integer number of pills/tablets/capsules to take at each scheduled time
   - "action": exactly one of "start", "continue", or "dose_change"
   - "confidence": exactly one of "high" or "low" — use "high" ONLY when the document clearly states the action; use "low" if wording is ambiguous
   Do NOT put medications that should be stopped in this array.
2. "discontinued_medications": a JSON array of medications to STOP or DISCONTINUE. Each entry must be either:
   - a string medication name, OR
   - an object with "name", optional "reason", and "confidence" ("high" only if the document clearly says stop/discontinue/cease; otherwise "low")
   Include ONLY medicines this document explicitly says to stop. Use an empty array if none.
3. "medication_review_items": a JSON array for ambiguous medication instructions you are NOT sure about. Each object must include:
   - "medication_name": the medication mentioned
   - "reason": why it is unclear (e.g. cannot tell if it should be stopped or continued)
   - "suggested_action": one of "discontinue", "dose_change", "start", "continue", "review"
   - "confidence": always "low"
   Use this when the document is vague, the name may not match exactly, or you cannot tell whether to add, update, or stop a medicine.
4. "conditions": a JSON array of condition objects — one object per diagnosis or medical condition. NEVER combine multiple conditions into a single entry. If the document lists several conditions together in one phrase, split them into separate objects. Each object must include:
   - "name": the single condition name only (e.g. "Hypertension", not "Hypertension and diabetes")
   - "status": exactly one of "Chronic", "Recovery", or "Acute"
   - "onset_date": when it started or was diagnosed (e.g. "2019", "March 2026", "Since 2021"); use null if not stated

Rules:
- If a medicine is being stopped, put it ONLY in discontinued_medications, never in medications.
- If a dose changes for an existing medicine, put it in medications with action "dose_change".
- If you are not confident, use confidence "low" and add a medication_review_items entry rather than guessing.
"""


def process_discharge_document_upload(uploaded_pdf, patient_id=None) -> dict:
    file_name = getattr(uploaded_pdf, "name", "document.pdf")
    active_patient_name = get_patient_display_name(patient_id)
    _documents_logger.info(
        "Document upload started: name=%s size=%s active_patient=%s",
        file_name,
        getattr(uploaded_pdf, "size", "?"),
        active_patient_name,
    )

    uploaded_pdf.seek(0)
    pdf_meta = extract_text_from_pdf_with_meta(uploaded_pdf)
    _documents_logger.info(
        "PDF extraction meta: bytes=%s pages=%s chars=%s code=%s",
        pdf_meta.get("byte_count"),
        pdf_meta.get("page_count"),
        pdf_meta.get("char_count"),
        pdf_meta.get("error_code"),
    )

    pdf_error = pdf_extraction_error_response(pdf_meta)
    if pdf_error:
        _documents_logger.warning(
            "Document upload rejected: stage=%s message=%s details=%s",
            pdf_error.get("stage"),
            pdf_error.get("message"),
            pdf_error.get("details"),
        )
        return pdf_error

    raw_text = pdf_meta.get("text") or ""

    patient_mismatch = validate_document_patient_profile(raw_text, active_patient_name)
    if patient_mismatch:
        _documents_logger.warning(
            "Document upload blocked: stage=%s active=%s document=%s",
            patient_mismatch.get("stage"),
            active_patient_name,
            (patient_mismatch.get("details") or {}).get("document_patient_name"),
        )
        return patient_mismatch

    try:
        result = ask_ai(DOCUMENTS_EXTRACT_SYSTEM_PROMPT, raw_text)
    except Exception as exc:
        _documents_logger.exception("Document AI extraction failed for %s", file_name)
        return {
            "error": True,
            "stage": "ai_request_failed",
            "message": "Something went wrong while reading this document. Please try again.",
            "details": f"{type(exc).__name__}: {exc}",
        }

    if result.get("error"):
        return {
            "error": True,
            "stage": "ai_parse_error",
            "message": result.get(
                "message",
                "We couldn't interpret this document. Please try again.",
            ),
            "details": result,
        }

    return {"error": False, "raw_text": raw_text, "result": result}


def my_results_cache_key(patient_id=None) -> str:
    return f"my_results_latest_{resolve_patient_id(patient_id)}"


def my_results_processed_key(patient_id=None) -> str:
    return f"my_results_processed_{resolve_patient_id(patient_id)}"


def my_results_error_key(patient_id=None) -> str:
    return f"my_results_error_{resolve_patient_id(patient_id)}"


def my_results_error_debug_key(patient_id=None) -> str:
    return f"my_results_error_debug_{resolve_patient_id(patient_id)}"


def my_results_feedback_key(patient_id=None) -> str:
    return f"my_results_feedback_{resolve_patient_id(patient_id)}"


def serialize_my_results_upload(uploaded_file) -> dict:
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    mime_type = (getattr(uploaded_file, "type", None) or "").lower()
    return {
        "name": uploaded_file.name,
        "bytes": file_bytes,
        "mime": mime_type,
        "size": len(file_bytes),
        "sig": f"{uploaded_file.name}:{len(file_bytes)}",
    }


def restore_my_results_upload(stored: dict):
    buffer = BytesIO(stored.get("bytes") or b"")
    buffer.name = stored.get("name") or "upload"
    buffer.type = stored.get("mime") or ""
    buffer.size = int(stored.get("size") or len(stored.get("bytes") or b""))
    return buffer


def apply_my_results_processing_result(
    result: dict,
    *,
    patient_id,
    upload_sig: str,
    cache_key: str,
    processed_key: str,
    error_key: str,
    error_debug_key: str,
) -> None:
    st.session_state[processed_key] = upload_sig
    feedback_key = my_results_feedback_key(patient_id)

    if result.get("error"):
        if result.get("stage") or result.get("details"):
            _my_results_logger.warning(
                "My Results upload failed: stage=%s message=%s details=%s",
                result.get("stage"),
                result.get("message"),
                result.get("details"),
            )
        message = result.get("message") or MY_RESULTS_NO_RESULTS_MESSAGE
        st.session_state[error_key] = message
        st.session_state[error_debug_key] = {
            "stage": result.get("stage"),
            "details": result.get("details"),
        }
        st.session_state[feedback_key] = {
            "type": "error",
            "message": message,
        }
        return

    invalidate_patient_activity_cache(patient_id)
    st.session_state[cache_key] = result
    st.session_state.pop(error_key, None)
    st.session_state.pop(error_debug_key, None)
    st.session_state.pop("my_results_error", None)
    st.session_state.pop("my_results_error_debug", None)
    st.session_state[feedback_key] = {
        "type": "success",
        "message": "Your document has been analysed. Scroll down to read the explanation.",
    }
    st.session_state.my_results_uploader_key = (
        int(st.session_state.get("my_results_uploader_key") or 0) + 1
    )


def load_my_results_record(patient_id=None) -> dict | None:
    patient_id = resolve_patient_id(patient_id)
    if not patient_id:
        return None
    cache_key = my_results_cache_key(patient_id)
    if cache_key in st.session_state:
        cached = st.session_state.get(cache_key)
        if not cached:
            st.session_state.pop(cache_key, None)
        elif str(cached.get("patient_id") or patient_id) != str(patient_id):
            st.session_state.pop(cache_key, None)
        else:
            return enrich_my_results_record(cached)
    records = fetch_my_result_records(patient_id, limit=1)
    record = enrich_my_results_record(records[0]) if records else None
    if record and str(record.get("patient_id") or patient_id) != str(patient_id):
        record = None
    if record:
        st.session_state[cache_key] = record
    elif cache_key in st.session_state:
        st.session_state.pop(cache_key, None)
    return record


def get_my_results_pdf_bytes(record: dict, patient_name: str, patient_id=None) -> bytes:
    patient_id = resolve_patient_id(patient_id)
    signature = hashlib.md5(
        json.dumps(record, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    cache_key = f"my_results_pdf_{patient_id}_{signature}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = generate_my_results_summary_pdf(record, patient_name)
    return st.session_state[cache_key]


def get_handover_pdf_bytes(
    patient_id,
    period_key: str,
    tz_obj,
    result: dict,
) -> bytes:
    patient_id = resolve_patient_id(patient_id)
    symptom_events_for_pdf = filter_events_by_handover_period(
        load_symptom_timeline_events(patient_id=patient_id),
        period_key,
        tz_obj,
    )
    adherence_events_for_pdf = filter_events_by_handover_period(
        load_medication_adherence_timeline_events(patient_id),
        period_key,
        tz_obj,
    )
    photo_reviews_for_pdf = get_handover_photo_reviews_for_period(patient_id, period_key, tz_obj)
    signature = hashlib.md5(
        json.dumps(
            {
                "patient_id": patient_id,
                "period": period_key,
                "result": result,
                "symptom_events": handover_events_signature(symptom_events_for_pdf),
                "adherence_events": handover_events_signature(adherence_events_for_pdf),
                "photo_count": len(photo_reviews_for_pdf),
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:16]
    cache_key = f"handover_pdf_{signature}"
    if cache_key not in st.session_state:
        connected_links_for_pdf = build_connected_report_links(
            filter_session_incidents_by_period(period_key, tz_obj, patient_id=patient_id)
        )
        st.session_state[cache_key] = generate_handover_pdf(
            symptom_events_for_pdf,
            connected_links_for_pdf,
            result,
            photo_reviews=photo_reviews_for_pdf,
            adherence_events=adherence_events_for_pdf,
            patient_label=get_patient_display_name(patient_id),
            period_label=get_handover_period_label(period_key),
        )
    return st.session_state[cache_key]


def format_my_results_upload_date(record: dict) -> str:
    raw = record.get("uploaded_at") or record.get("date") or ""
    if not raw or raw == "Unknown":
        return datetime.now().strftime("%d %b")
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return parsed.strftime("%d %b")
    except (ValueError, TypeError):
        return str(raw)


def count_my_results_to_review(record: dict | list | None) -> int:
    if isinstance(record, dict):
        return count_my_results_review_items(record)
    return count_my_results_review_items({"results": record or []})


def build_my_results_flag_html(status: str) -> str:
    value = str(status or "normal").strip().lower()
    if value == "low":
        return '<span class="cs-mr-flag cs-mr-flag--low">Low</span>'
    if value in ("high", "abnormal"):
        return '<span class="cs-mr-flag cs-mr-flag--high">High</span>'
    if value == "not_provided":
        return '<span class="cs-mr-flag cs-mr-flag--na">No range</span>'
    return '<span class="cs-mr-flag cs-mr-flag--normal">Normal</span>'


def build_my_results_results_rows_html(results: list) -> str:
    rows = []
    for row in results or []:
        name = html.escape(str(row.get("name") or "—"))
        value = html.escape(str(row.get("value") or "—"))
        unit = str(row.get("unit") or "").strip()
        if unit:
            value = f"{value} {html.escape(unit)}"
        ref = str(row.get("referenceRange") or "").strip()
        ref_html = (
            f'<div class="cs-mr-result-ref">Ref: {html.escape(ref)}</div>'
            if ref else ""
        )
        rows.append(
            f'<div class="cs-mr-result-row">'
            f'<div class="cs-mr-result-name">{name}{ref_html}</div>'
            f'<div class="cs-mr-result-value">{value}</div>'
            f'<div class="cs-mr-result-flag">{build_my_results_flag_html(row.get("status"))}</div>'
            f"</div>"
        )
    if not rows:
        return ""
    return "".join(rows)


def build_my_results_results_section_html(record: dict) -> str:
    results = record.get("results") or []
    if not record.get("hasLabValues", bool(results)):
        return (
            '<div class="cs-mr-results-block">'
            '<p class="cs-mr-empty-section">No numeric lab values were listed in this document.</p>'
            "</div>"
        )
    return (
        '<div class="cs-mr-results-block">'
        '<div class="cs-mr-results-label">Lab &amp; numeric values</div>'
        f"{build_my_results_results_rows_html(results)}"
        "</div>"
    )


def build_my_results_key_findings_html(record: dict) -> str:
    items = []
    for dx in record.get("newDiagnoses") or []:
        if not isinstance(dx, dict) or not dx.get("name"):
            continue
        detail = str(dx.get("detail") or "").strip()
        body = f": {html.escape(detail)}" if detail else ""
        items.append(
            f'<li><span class="cs-mr-key-label">New diagnosis</span> '
            f'<strong>{html.escape(str(dx["name"]))}</strong>{body}</li>'
        )
    for med in record.get("medicationChanges") or []:
        if not isinstance(med, dict) or not med.get("medication"):
            continue
        change = html.escape(str(med.get("changeType") or "change").replace("_", " "))
        detail = str(med.get("detail") or "").strip()
        body = f" — {html.escape(detail)}" if detail else ""
        items.append(
            f'<li><span class="cs-mr-key-label">Medication {change}</span> '
            f'<strong>{html.escape(str(med["medication"]))}</strong>{body}</li>'
        )
    for follow in record.get("followUps") or []:
        if not isinstance(follow, dict) or not follow.get("description"):
            continue
        date_label = html.escape(
            str(follow.get("dateDisplay") or MY_RESULTS_DATE_NOT_SPECIFIED)
        )
        prep = str(follow.get("prep") or "").strip()
        prep_html = f' <span class="cs-mr-key-meta">Prep: {html.escape(prep)}</span>' if prep else ""
        items.append(
            f'<li><span class="cs-mr-key-label">Follow-up</span> '
            f'{html.escape(str(follow["description"]))} '
            f'<span class="cs-mr-key-meta">({date_label})</span>{prep_html}</li>'
        )
    for img in record.get("imagingFindings") or []:
        if not isinstance(img, dict):
            continue
        study = str(img.get("study") or "Imaging").strip()
        finding = str(img.get("finding") or "").strip()
        if not finding:
            continue
        items.append(
            f'<li><span class="cs-mr-key-label">{html.escape(study)}</span> '
            f'{html.escape(finding)}</li>'
        )
    if not items:
        return ""
    return (
        '<div class="cs-mr-key-findings">'
        '<div class="cs-mr-key-findings-title">Key findings</div>'
        f'<ul class="cs-mr-key-findings-list">{"".join(items)}</ul>'
        "</div>"
    )


def build_my_results_urgent_care_html(record: dict) -> str:
    urgent = sanitize_my_results_plain_text(record.get("urgentCareInstructions"))
    red_flags = [
        str(item.get("instruction") or "").strip()
        for item in (record.get("caregiverInstructions") or [])
        if isinstance(item, dict) and item.get("category") == "red_flag"
    ]
    red_flags = [item for item in red_flags if item]
    if not urgent and not red_flags:
        return ""
    body_parts = []
    if urgent:
        body_parts.append(f"<p>{html.escape(urgent)}</p>")
    if red_flags and not urgent:
        body_parts.append(
            "<ul>"
            + "".join(f"<li>{html.escape(item)}</li>" for item in red_flags)
            + "</ul>"
        )
    return (
        '<div class="cs-mr-urgent-card">'
        '<div class="cs-mr-urgent-title">When to seek urgent care</div>'
        f'{"".join(body_parts)}'
        "</div>"
    )


def build_my_results_urgency_badge_html(urgency: str) -> str:
    level = str(urgency or "discuss_at_visit").strip().lower()
    if level == "discuss_soon":
        return (
            '<span class="cs-mr-urgency cs-mr-urgency--soon">'
            '<span class="cs-mr-urgency-dot" aria-hidden="true"></span>'
            "Discuss soon</span>"
        )
    return (
        '<span class="cs-mr-urgency cs-mr-urgency--visit">'
        '<span class="cs-mr-urgency-dot" aria-hidden="true"></span>'
        "Discuss at next visit</span>"
    )


def _my_results_result_lookup(record: dict) -> dict[str, dict]:
    lookup = {}
    for row in record.get("results") or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if name:
            lookup[name.lower()] = row
    return lookup


def _my_results_format_result_value(row: dict | None) -> str:
    if not row:
        return ""
    value = str(row.get("value") or "—").strip()
    unit = str(row.get("unit") or "").strip()
    if unit:
        value = f"{value} {unit}"
    ref = str(row.get("referenceRange") or "").strip()
    parts = [html.escape(value)]
    if ref:
        parts.append(f'<span class="cs-mr-group-test-ref">Ref: {html.escape(ref)}</span>')
    parts.append(build_my_results_flag_html(row.get("status")))
    return " · ".join(parts)


def build_my_results_trend_callouts_html(record: dict) -> str:
    callouts = record.get("trendCallouts") or []
    if not callouts:
        return ""
    blocks = []
    for item in callouts:
        if not isinstance(item, dict):
            continue
        title = html.escape(str(item.get("title") or "Change since last test").strip())
        summary = html.escape(sanitize_my_results_plain_text(item.get("summary")))
        if not summary:
            continue
        prior = sanitize_my_results_plain_text(item.get("priorValues"))
        prior_html = ""
        if prior:
            prior_html = (
                f'<div class="cs-mr-trend-prior">'
                f'<span class="cs-mr-trend-prior-label">Previous values:</span> '
                f"{html.escape(prior)}</div>"
            )
        blocks.append(
            f'<div class="cs-mr-trend-card">'
            f'<div class="cs-mr-trend-badge">Change from last test</div>'
            f'<div class="cs-mr-trend-title">{title}</div>'
            f'<p class="cs-mr-trend-summary">{summary}</p>'
            f"{prior_html}"
            f"</div>"
        )
    if not blocks:
        return ""
    return (
        '<div class="cs-mr-trend-section">'
        '<div class="cs-mr-section-label">Trends to know about</div>'
        + "".join(blocks)
        + "</div>"
    )


def build_my_results_grouped_explanations_html(record: dict) -> str:
    groups = record.get("resultGroups") or []
    if not groups:
        return ""

    result_lookup = _my_results_result_lookup(record)
    group_blocks = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        category = html.escape(str(group.get("category") or "Results to discuss").strip())
        group_summary = html.escape(sanitize_my_results_plain_text(group.get("groupSummary")))
        urgency_html = build_my_results_urgency_badge_html(group.get("urgency"))
        test_names = [
            str(name).strip()
            for name in (group.get("testNames") or [])
            if str(name).strip()
        ]

        value_blocks = []
        explained = {
            str(item.get("testName") or "").strip().lower()
            for item in (group.get("valueExplanations") or [])
            if isinstance(item, dict) and item.get("testName")
        }
        for item in group.get("valueExplanations") or []:
            if not isinstance(item, dict):
                continue
            test_name = str(item.get("testName") or "").strip()
            if not test_name:
                continue
            row = result_lookup.get(test_name.lower())
            value_line = _my_results_format_result_value(row)
            measures = html.escape(sanitize_my_results_plain_text(item.get("whatItMeasures")))
            suggests = html.escape(sanitize_my_results_plain_text(item.get("whatThisResultSuggests")))
            define_terms = sanitize_my_results_plain_text(item.get("defineTerms"))
            body_parts = []
            if value_line:
                body_parts.append(f'<div class="cs-mr-group-test-value">{value_line}</div>')
            if measures:
                body_parts.append(f'<p class="cs-mr-group-test-measures"><strong>What it measures:</strong> {measures}</p>')
            if suggests:
                body_parts.append(f'<p class="cs-mr-group-test-suggests"><strong>What this result suggests:</strong> {suggests}</p>')
            if define_terms:
                body_parts.append(f'<p class="cs-mr-group-test-define"><strong>In plain words:</strong> {html.escape(define_terms)}</p>')
            if not body_parts:
                continue
            value_blocks.append(
                f'<div class="cs-mr-group-test">'
                f'<div class="cs-mr-group-test-name">{html.escape(test_name)}</div>'
                f'{"".join(body_parts)}'
                f"</div>"
            )

        for test_name in test_names:
            if test_name.lower() in explained:
                continue
            row = result_lookup.get(test_name.lower())
            value_line = _my_results_format_result_value(row)
            if not value_line:
                continue
            value_blocks.append(
                f'<div class="cs-mr-group-test cs-mr-group-test--compact">'
                f'<div class="cs-mr-group-test-name">{html.escape(test_name)}</div>'
                f'<div class="cs-mr-group-test-value">{value_line}</div>'
                f"</div>"
            )

        chips = "".join(
            f'<span class="cs-mr-group-chip">{html.escape(name)}</span>'
            for name in test_names
        )
        summary_html = f'<p class="cs-mr-group-summary">{group_summary}</p>' if group_summary else ""
        chips_html = f'<div class="cs-mr-group-chips">{chips}</div>' if chips else ""
        group_blocks.append(
            f'<div class="cs-mr-group-card">'
            f'<div class="cs-mr-group-header">'
            f'<div class="cs-mr-group-title">{category}</div>'
            f"{urgency_html}"
            f"</div>"
            f"{summary_html}"
            f"{chips_html}"
            f'{"".join(value_blocks)}'
            f"</div>"
        )

    if not group_blocks:
        return ""
    return (
        '<div class="cs-mr-groups-section">'
        '<div class="cs-mr-meaning-title">What this means</div>'
        + "".join(group_blocks)
        + "</div>"
    )


def build_my_results_explanation_body_html(record: dict) -> str:
    use_grouped = record.get("useGroupedExplanations")
    if use_grouped is None:
        use_grouped = my_results_use_grouped_explanations(record, record)

    explanation = sanitize_my_results_plain_text(record.get("explanation"))
    trend_html = build_my_results_trend_callouts_html(record)
    parts = []

    if explanation:
        label = "At a glance" if use_grouped else "What this means"
        parts.append(
            f'<div class="cs-mr-meaning-card cs-mr-meaning-card--overview">'
            f'<div class="cs-mr-meaning-title">{html.escape(label)}</div>'
            f'<p class="cs-mr-meaning-text">{html.escape(explanation)}</p>'
            f"</div>"
        )

    if trend_html:
        parts.append(trend_html)

    if use_grouped:
        grouped_html = build_my_results_grouped_explanations_html(record)
        if grouped_html:
            parts.append(grouped_html)
    elif not explanation and trend_html:
        parts.append(
            '<div class="cs-mr-meaning-card">'
            '<div class="cs-mr-meaning-title">What this means</div>'
            '<p class="cs-mr-meaning-text">Review the flagged values below and discuss them with the doctor.</p>'
            "</div>"
        )

    return "".join(parts)


def build_my_results_no_abnormal_note_html(record: dict) -> str:
    note = sanitize_my_results_plain_text(record.get("noAbnormalValuesNote"))
    if not note:
        return ""
    label = my_results_no_abnormal_note_label(record)
    label_class = (
        "cs-mr-no-abnormal-label cs-mr-no-abnormal-label--neutral"
        if my_results_has_key_findings(record)
        else "cs-mr-no-abnormal-label"
    )
    return (
        '<div class="cs-mr-no-abnormal-note">'
        f'<div class="{label_class}">{html.escape(label)}</div>'
        f"<p>{html.escape(note)}</p>"
        "</div>"
    )


def build_my_results_questions_html(record: dict) -> str:
    question_items = []
    for idx, question in enumerate(record.get("questions") or [], start=1):
        text = my_results_question_text(question)
        if not text:
            continue
        meta_html = ""
        if isinstance(question, dict):
            category = sanitize_my_results_plain_text(question.get("relatedCategory"))
            tests = [
                str(name).strip()
                for name in (question.get("relatedTests") or [])
                if str(name).strip()
            ]
            meta_bits = []
            if category:
                meta_bits.append(html.escape(category))
            if tests:
                meta_bits.append(html.escape(", ".join(tests)))
            if meta_bits:
                meta_html = (
                    f'<div class="cs-mr-question-meta">{" · ".join(meta_bits)}</div>'
                )
        question_items.append(
            f'<li>'
            f'<span class="cs-mr-question-num">{idx}</span>'
            f'<div class="cs-mr-question-body">'
            f'<span class="cs-mr-question-text">{html.escape(text)}</span>'
            f"{meta_html}"
            f"</div></li>"
        )
    if not question_items:
        return ""
    return (
        '<div class="cs-mr-questions-card">'
        '<div class="cs-mr-questions-header">'
        '<div class="cs-mr-questions-title">Questions to ask the doctor</div>'
        "</div>"
        f'<ol class="cs-mr-questions-list">{"".join(question_items)}</ol>'
        "</div>"
    )


def build_my_results_limitations_html(record: dict) -> str:
    notes = []
    for key, label in (
        ("languageNote", "Language"),
        ("patientIdentityNote", "Patient identity"),
    ):
        value = str(record.get(key) or "").strip()
        if value and value.lower() not in ("null", "none"):
            notes.append(f"<strong>{label}:</strong> {html.escape(value)}")
    for item in record.get("limitations") or []:
        text = str(item).strip()
        if text and not my_results_limitation_is_lab_table_artifact(text, record):
            notes.append(html.escape(text))
    if not notes:
        return ""
    return (
        '<div class="cs-mr-limitations">'
        + "".join(f"<p>{part}</p>" for part in notes)
        + "</div>"
    )


def build_my_results_card_html(record: dict) -> str:
    title = html.escape(str(record.get("documentType") or record.get("file_name") or "Medical document"))
    uploaded = format_my_results_upload_date(record)
    source = html.escape(str(record.get("source") or "Unknown source"))
    review_count = count_my_results_to_review(record)
    badge_html = ""
    if review_count:
        label = "1 to review" if review_count == 1 else f"{review_count} to review"
        badge_html = f'<span class="cs-mr-review-badge">{html.escape(label)}</span>'

    key_findings_html = build_my_results_key_findings_html(record)
    urgent_html = build_my_results_urgent_care_html(record)
    no_abnormal_html = build_my_results_no_abnormal_note_html(record)
    limitations_html = build_my_results_limitations_html(record)
    explanation_html = build_my_results_explanation_body_html(record)
    results_section_html = build_my_results_results_section_html(record)
    questions_html = build_my_results_questions_html(record)

    return f"""
    <div class="cs-mr-latest-label">Latest result</div>
    <div class="cs-mr-card">
      <div class="cs-mr-card-header">
        <div class="cs-mr-card-title-wrap">
          <div>
            <div class="cs-mr-card-title">{title}</div>
            <div class="cs-mr-card-meta">Uploaded {uploaded} · {source}</div>
          </div>
        </div>
        {badge_html}
      </div>
      {key_findings_html}
      {urgent_html}
      {no_abnormal_html}
      {limitations_html}
      {explanation_html}
      {results_section_html}
      {questions_html}
    </div>
    """


def append_my_results_to_handover(
    patient_id,
    record: dict,
    caregiver_name: str,
    caregiver_id=None,
) -> None:
    questions = list(record.get("questions") or [])
    if not questions:
        return
    title = record.get("documentType") or record.get("file_name") or "Test results"
    entries = [
        {"question": my_results_question_text(question), "from": title}
        for question in questions
        if my_results_question_text(question)
    ]

    pending_key = f"pending_handover_questions_{resolve_patient_id(patient_id)}"
    pending = list(st.session_state.get(pending_key, []))
    pending.extend(entries)
    st.session_state[pending_key] = pending

    if st.session_state.get("last_sbar_result"):
        existing = list(st.session_state.last_sbar_result.get("doctor_questions") or [])
        existing.extend(entries)
        st.session_state.last_sbar_result["doctor_questions"] = existing

    summary_bits = [
        f"{idx}. {my_results_question_text(question)}"
        for idx, question in enumerate(questions[:4], start=1)
        if my_results_question_text(question)
    ]
    summary = f"Questions from {title}: " + " | ".join(summary_bits)
    save_shift_log(
        caregiver_name=caregiver_name,
        source="my_results_handover",
        summary=summary,
        severity="monitor",
        caregiver_id=caregiver_id,
        patient_id=patient_id,
    )


def merge_pending_handover_questions(patient_id, sbar_result: dict) -> dict:
    pending_key = f"pending_handover_questions_{resolve_patient_id(patient_id)}"
    pending = list(st.session_state.get(pending_key, []))
    if not pending:
        return sbar_result
    merged = dict(sbar_result or {})
    existing = list(merged.get("doctor_questions") or [])
    existing.extend(pending)
    merged["doctor_questions"] = existing
    st.session_state[pending_key] = []
    return merged


def _finalize_my_results_record(
    record: dict,
    *,
    patient_id,
    file_name: str,
    raw_text: str,
    caregiver_name: str,
    caregiver_id=None,
) -> dict:
    doc_meta = {}
    try:
        doc_meta = save_patient_test_document(
            patient_id,
            file_name=file_name,
            raw_text=raw_text,
            caregiver_id=caregiver_id,
        )
    except Exception:
        _my_results_logger.exception("Failed to save My Results document copy")

    record = {
        **record,
        "file_name": file_name,
        "document_id": doc_meta.get("document_id"),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "patient_id": str(resolve_patient_id(patient_id) or ""),
    }
    record = enrich_my_results_record(record) or record
    try:
        save_my_result_record(
            patient_id,
            record,
            caregiver_name,
            caregiver_id=caregiver_id,
        )
    except Exception:
        _my_results_logger.exception("Failed to persist My Results analysis to shift log")
    return record


def _try_my_results_offline_fallback(
    raw_text: str,
    *,
    file_name: str,
    patient_id,
    patient_name: str,
    conditions: list,
    caregiver_name: str,
    caregiver_id=None,
) -> dict | None:
    if not str(raw_text or "").strip():
        return None
    condition_names = [
        str(item.get("name") or "").strip()
        for item in (conditions or [])
        if str(item.get("name") or "").strip()
    ]
    offline_record = build_my_results_record_from_offline_text(
        raw_text,
        file_name=file_name,
        patient_name=patient_name,
        known_conditions=condition_names,
    )
    if not offline_record:
        return None
    _my_results_logger.warning(
        "My Results using offline text fallback for %s after AI service failure",
        file_name,
    )
    return _finalize_my_results_record(
        offline_record,
        patient_id=patient_id,
        file_name=file_name,
        raw_text=raw_text,
        caregiver_name=caregiver_name,
        caregiver_id=caregiver_id,
    )


def process_my_results_upload(
    uploaded_file,
    patient_id,
    patient_name: str,
    conditions: list,
    caregiver_name: str,
    caregiver_id=None,
) -> dict:
    file_name = uploaded_file.name
    extension = file_name.lower().rsplit(".", 1)[-1]
    mime_type = (getattr(uploaded_file, "type", None) or "").lower()
    is_pdf = extension == "pdf" or mime_type == "application/pdf"
    raw_text = ""
    extract_content = None

    _my_results_logger.info(
        "Upload started: name=%s ext=%s mime=%s size=%s",
        file_name,
        extension,
        mime_type or "unknown",
        getattr(uploaded_file, "size", "?"),
    )

    if is_pdf:
        uploaded_file.seek(0)
        pdf_meta = extract_text_from_pdf_with_meta(uploaded_file)
        _my_results_logger.info(
            "PDF extraction meta: bytes=%s pages=%s chars=%s code=%s",
            pdf_meta.get("byte_count"),
            pdf_meta.get("page_count"),
            pdf_meta.get("char_count"),
            pdf_meta.get("error_code"),
        )

        pdf_error = pdf_extraction_error_response(pdf_meta)
        if pdf_error:
            return pdf_error

        raw_text = pdf_meta.get("text") or ""
        extract_content = f"Document text:\n\n{raw_text}"
    else:
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        if not file_bytes:
            return {
                "error": True,
                "stage": "image_empty_buffer",
                "message": "The uploaded image appears empty. Please try uploading the file again.",
                "details": {"byte_count": 0, "mime_type": mime_type},
            }
        media_type = mime_type if mime_type.startswith("image/") else "image/jpeg"
        if extension == "png":
            media_type = "image/png"
        elif extension == "webp":
            media_type = "image/webp"
        image_b64 = base64.b64encode(file_bytes).decode("utf-8")
        raw_text = f"[Image upload: {file_name}]"
        extract_content = build_openai_user_content(
            "Extract clinically significant information from this medical document image. "
            "Use narrative extraction for letters and mixed documents — not only lab tables.",
            image_b64,
            media_type,
        )
        _my_results_logger.info(
            "Image upload path: bytes=%s mime=%s",
            len(file_bytes),
            media_type,
        )

    extract_raw = ask_ai(
        f"{MY_RESULTS_ASSISTANT_SYSTEM}\n\n{MY_RESULTS_EXTRACT_PROMPT}",
        extract_content,
    )
    if extract_raw.get("error"):
        _my_results_logger.warning(
            "Extract AI returned error: stage=ai_extract details=%s",
            extract_raw.get("details") or extract_raw.get("message"),
        )
        failure_reason = resolve_ai_failure_reason(extract_raw)
        if ai_failure_is_recoverable_offline(failure_reason) and raw_text:
            offline_record = _try_my_results_offline_fallback(
                raw_text,
                file_name=file_name,
                patient_id=patient_id,
                patient_name=patient_name,
                conditions=conditions,
                caregiver_name=caregiver_name,
                caregiver_id=caregiver_id,
            )
            if offline_record:
                return offline_record
        if extract_raw.get("reason") == "unreadable" or extract_raw.get("readability") == "unreadable":
            return {
                "error": True,
                "stage": "ai_extract_unreadable",
                "message": (
                    "We couldn't read this document clearly. "
                    "Please upload a sharper photo or an unprotected PDF and try again."
                ),
                "details": extract_raw.get("details") or extract_raw.get("message"),
            }
        return {
            **extract_raw,
            "stage": "ai_extract",
        }

    extract = normalize_my_results_extract(extract_raw)
    if extract.get("readability") == "unreadable":
        return {
            "error": True,
            "stage": "ai_extract_unreadable",
            "message": (
                "We couldn't read this document clearly. "
                "Please upload a sharper photo or an unprotected PDF and try again."
            ),
            "details": extract.get("limitations"),
        }

    if not my_results_has_actionable_content(extract):
        _my_results_logger.info(
            "Extract AI found no actionable content: documentType=%s file=%s",
            extract.get("documentType"),
            file_name,
        )
        return {
            "error": True,
            "stage": "no_actionable_content",
            "message": MY_RESULTS_NO_RESULTS_MESSAGE,
            "details": {
                "documentType": extract.get("documentType"),
                "source": extract.get("source"),
                "limitations": extract.get("limitations"),
            },
        }

    condition_names = [
        str(item.get("name") or "").strip()
        for item in (conditions or [])
        if str(item.get("name") or "").strip()
    ]
    explain_payload = build_my_results_explain_payload(
        extract,
        patient_name=patient_name,
        known_conditions=condition_names,
    )

    explain_raw = ask_ai(
        f"{MY_RESULTS_ASSISTANT_SYSTEM}\n\n{MY_RESULTS_EXPLAIN_PROMPT}",
        explain_payload,
    )
    if explain_raw.get("error"):
        _my_results_logger.warning(
            "Explain AI returned error: details=%s",
            explain_raw.get("details") or explain_raw.get("message"),
        )
        failure_reason = resolve_ai_failure_reason(explain_raw)
        if ai_failure_is_recoverable_offline(failure_reason) and raw_text:
            offline_record = _try_my_results_offline_fallback(
                raw_text,
                file_name=file_name,
                patient_id=patient_id,
                patient_name=patient_name,
                conditions=conditions,
                caregiver_name=caregiver_name,
                caregiver_id=caregiver_id,
            )
            if offline_record:
                return offline_record
        return {**explain_raw, "stage": "ai_explain"}

    explain = normalize_my_results_explain(
        explain_raw,
        extract=extract,
        patient_name=patient_name,
        known_conditions=condition_names,
        generate_missing_explanations=True,
    )
    if not my_results_explain_is_complete(explain, extract):
        _my_results_logger.warning(
            "Explain AI missing explanation or questions: keys=%s",
            list(explain_raw.keys()),
        )
        return {
            "error": True,
            "stage": "ai_explain_incomplete",
            "message": "We couldn't generate an explanation right now. Please try again.",
        }

    return _finalize_my_results_record(
        {**extract, **explain},
        patient_id=patient_id,
        file_name=file_name,
        raw_text=raw_text,
        caregiver_name=caregiver_name,
        caregiver_id=caregiver_id,
    )


def build_sbar_results_html(result):
    doctor_questions_html = ""
    doctor_questions = result.get("doctor_questions") or []
    if doctor_questions:
        question_items = []
        for entry in doctor_questions:
            if isinstance(entry, dict):
                question = entry.get("question") or entry.get("text") or ""
                source = entry.get("from") or entry.get("title") or ""
            else:
                question = str(entry)
                source = ""
            if not question:
                continue
            source_html = (
                f'<span class="cs-mr-handover-q-source">{html.escape(source)}</span>'
                if source else ""
            )
            question_items.append(
                f'<li>{source_html}{html.escape(question)}</li>'
            )
        if question_items:
            doctor_questions_html = f"""
            <div class="cs-mr-handover-questions">
              <div class="cs-mr-handover-questions-label">Questions from test results</div>
              <ul class="cs-mr-handover-questions-list">
                {"".join(question_items)}
              </ul>
            </div>
            """

    peak_severity = str(result.get("peak_severity") or "").strip().lower()
    peak_severity_html = ""
    if peak_severity in ("emergency", "contact_doctor"):
        peak_label = handover_severity_label(peak_severity)
        peak_detail = (
            "Call 999/112 was recommended during this period."
            if peak_severity == "emergency"
            else "GP/consultant contact within 24 hours was recommended during this period."
        )
        peak_class = "cs-sbar-peak-emergency" if peak_severity == "emergency" else "cs-sbar-peak-contact"
        peak_severity_html = (
            f'<div class="cs-sbar-peak {peak_class}">'
            f'<div class="cs-sbar-peak-label">Peak severity this period</div>'
            f'<div class="cs-sbar-peak-value">{html.escape(peak_label)}</div>'
            f'<div class="cs-sbar-peak-detail">{html.escape(peak_detail)}</div>'
            f"</div>"
        )
    photo_count = int(result.get("photo_report_count") or 0)
    photo_note_html = ""
    if photo_count > 0:
        photo_note_html = (
            f'<div class="cs-sbar-photo-note">'
            f"{photo_count} symptom photo{'s' if photo_count != 1 else ''} logged this period "
            f"— included in the downloadable PDF."
            f"</div>"
        )

    return f"""
    {peak_severity_html}
    {photo_note_html}
    <div class="cs-sbar-grid">
      <div class="cs-sbar-card cs-sbar-situation">
        <div class="cs-sbar-label">Situation</div>
        <div class="cs-sbar-body">{html.escape(str(result.get('situation', '')))}</div>
      </div>
      <div class="cs-sbar-card cs-sbar-background">
        <div class="cs-sbar-label">Background</div>
        <div class="cs-sbar-body">{html.escape(str(result.get('background', '')))}</div>
      </div>
      <div class="cs-sbar-card cs-sbar-assessment">
        <div class="cs-sbar-label">Assessment</div>
        <div class="cs-sbar-body">{html.escape(str(result.get('assessment', '')))}</div>
      </div>
      <div class="cs-sbar-card cs-sbar-recommendation">
        <div class="cs-sbar-label">Recommendation</div>
        <div class="cs-sbar-body">{html.escape(str(result.get('recommendation', '')))}</div>
      </div>
    </div>
    <div class="cs-sbar-watch">
      <div class="cs-sbar-watch-label">Watch for</div>
      <div class="cs-sbar-watch-text">{html.escape(str(result.get('watch_for', '')))}</div>
    </div>
    {doctor_questions_html}
    """


def build_reported_by_card_html(caregiver: str, note: str, image_b64: str = "") -> str:
    name = html.escape(caregiver)
    body = html.escape(note)
    image_html = ""
    if image_b64:
        image_html = (
            f'<img src="data:image/jpeg;base64,{image_b64}" '
            'style="max-width:100%;max-height:220px;border-radius:8px;margin-top:10px;" '
            'alt="Symptom photo" />'
        )
    return (
        '<div style="background-color: #E8F5E9; padding: 15px; border-radius: 10px; '
        'margin-bottom: 10px; border-left: 5px solid #2E7D32;">'
        f'<strong style="color: #1B5E20; font-size: 16px;">{name}:</strong>'
        f'<p style="color: #2E7D32; margin: 5px 0 0 0;">{body}</p>'
        f'{image_html}'
        '</div>'
    )


def render_handover_symptom_photos(patient_id, period_key: str, tz_obj) -> None:
    """Show every symptom photo logged in the selected handover period."""
    reviews = get_handover_photo_reviews_for_period(patient_id, period_key, tz_obj)
    if not reviews:
        return
    md_html('<div class="cs-reported-by-heading">Symptom photos in this period</div>')
    row_size = 3
    for row_start in range(0, len(reviews), row_size):
        cols = st.columns(row_size)
        for col_idx, review in enumerate(reviews[row_start:row_start + row_size]):
            image_b64 = review.get("image_b64")
            if not image_b64:
                continue
            caption = (
                f"{review.get('timestamp_display') or 'Unknown time'} · "
                f"{review.get('caregiver') or 'Caregiver'}"
            )
            with cols[col_idx]:
                st.image(
                    f"data:image/jpeg;base64,{image_b64}",
                    caption=caption,
                    use_container_width=True,
                )


def render_paginated_reported_by(
    reported_by: list,
    *,
    heading: str = "Things reported",
    session_key: str = "handover_reported_by_visible",
    show_more_key: str = "handover_reported_by_show_more",
    show_less_key: str = "handover_reported_by_show_less",
) -> None:
    if not reported_by:
        return

    if session_key not in st.session_state:
        st.session_state[session_key] = HANDOVER_TIMELINE_DEFAULT_VISIBLE

    total = len(reported_by)
    shown_count = min(st.session_state[session_key], total)
    start_idx = max(0, total - shown_count)
    visible_entries = reported_by[start_idx:]
    hidden_count = start_idx

    md_html(f'<div class="cs-reported-by-heading">{html.escape(heading)}</div>')
    for entry in visible_entries:
        caregiver = str(entry.get("caregiver", "Caregiver"))
        note = str(entry.get("note", ""))
        image_b64 = str(entry.get("image_b64") or "")
        st.markdown(
            build_reported_by_card_html(caregiver, note, image_b64=image_b64),
            unsafe_allow_html=True,
        )

    if hidden_count > 0:
        md_html('<div class="cs-timeline-more-indicator">···</div>')

    render_history_show_more_controls(
        hidden_count=hidden_count,
        total=total,
        shown_count=shown_count,
        session_key=session_key,
        show_more_key=show_more_key,
    )


def format_plan_timestamp(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "Date unknown"
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt.strftime("%d %b %Y")
    except ValueError:
        return text[:10]


def build_discontinued_meds_html(records) -> str:
    if not records:
        return ""
    blocks = []
    for item in records:
        name = html.escape(str(item.get("name", "")))
        dosage = str(item.get("dosage") or "")
        timing = html.escape(str(item.get("timing") or ""))
        reason = html.escape(str(item.get("reason") or "Stopped per uploaded document"))
        stopped = html.escape(format_plan_timestamp(item.get("discontinued_at", "")))
        source = html.escape(str(item.get("source_document") or ""))
        details = []
        if dosage:
            details.append(html.escape(dosage))
        if timing:
            details.append(timing)
        detail_html = f'<div class="cs-doc-item-detail">{" · ".join(details)}</div>' if details else ""
        source_html = f'<div class="cs-doc-item-detail">From: {source}</div>' if source else ""
        blocks.append(
            f'<div class="cs-doc-item-block cs-doc-item-block--discontinued">'
            f'<div class="cs-doc-item-name">{name}</div>'
            f'{detail_html}'
            f'<div class="cs-doc-item-detail">Stopped {stopped} — {reason}</div>'
            f'{source_html}'
            f'</div>'
        )
    return f"""
    <div class="cs-sbar-card cs-sbar-assessment cs-doc-discontinued-card">
      <div class="cs-sbar-label">Discontinued medications</div>
      <div class="cs-sbar-body cs-doc-sbar-body">{"".join(blocks)}</div>
    </div>
    """


def build_medication_review_flags_html(flags) -> str:
    if not flags:
        return ""
    blocks = []
    for item in flags:
        name = html.escape(str(item.get("medication_name") or "Medication"))
        reason = html.escape(str(item.get("reason") or "Needs manual review"))
        action = html.escape(str(item.get("suggested_action") or "review").replace("_", " "))
        flagged = html.escape(format_plan_timestamp(item.get("flagged_at", "")))
        source = html.escape(str(item.get("source_document") or ""))
        source_html = f'<div class="cs-doc-item-detail">From: {source}</div>' if source else ""
        blocks.append(
            f'<div class="cs-doc-review-flag">'
            f'<div class="cs-doc-review-flag-title">{name}</div>'
            f'<div class="cs-doc-item-detail">Suggested action: {action} · Flagged {flagged}</div>'
            f'<div class="cs-doc-review-flag-body">{reason}</div>'
            f'{source_html}'
            f'</div>'
        )
    return f"""
    <div class="cs-sbar-card cs-sbar-recommendation cs-doc-review-card">
      <div class="cs-sbar-label">Needs manual review</div>
      <div class="cs-sbar-body cs-doc-sbar-body">
        <div class="cs-doc-review-intro">
          CareShield left these unchanged because the document was ambiguous. Please check them yourself.
        </div>
        {"".join(blocks)}
      </div>
    </div>
    """


def build_stored_meds_sbar_html(items, is_demo=False):
    display_items = items or []
    blocks = []
    for item in display_items:
        name = html.escape(str(item.get("name", "")))
        dosage = str(item.get("dosage", "") or "")
        timing = html.escape(str(item.get("time") or item.get("timing") or ""))
        details = []
        if dosage and dosage not in str(item.get("name", "")):
            details.append(html.escape(dosage))
        if timing:
            details.append(timing)
        detail_html = ""
        if details:
            detail_html = f'<div class="cs-doc-item-detail">{" · ".join(details)}</div>'
        blocks.append(
            f'<div class="cs-doc-item-block">'
            f'<div class="cs-doc-item-name">{name}</div>'
            f'{detail_html}'
            f'</div>'
        )
    if not blocks:
        blocks.append(
            '<div class="cs-doc-item-detail">No medications on file yet. Upload a discharge document to add them.</div>'
        )
    demo_note = ""
    if is_demo:
        demo_note = '<div class="cs-doc-item-detail cs-doc-demo-note">Sample plan shown until a document is processed.</div>'
    return f"""
    <div class="cs-sbar-card cs-sbar-situation">
      <div class="cs-sbar-label">Stored medications</div>
      <div class="cs-sbar-body cs-doc-sbar-body">{"".join(blocks)}{demo_note}</div>
    </div>
    """


def build_stored_conditions_sbar_html(conditions):
    blocks = []
    for item in conditions or []:
        badge_class, badge_label = condition_badge_meta(item["badge"])
        since_html = ""
        since_label = normalize_condition_since(item.get("since"))
        if since_label:
            since_html = f'<div class="cs-doc-item-detail">{html.escape(since_label)}</div>'
        blocks.append(
            f'<div class="cs-doc-item-block">'
            f'<div class="cs-doc-condition-top">'
            f'<span class="cs-doc-item-name">{html.escape(item["name"])}</span>'
            f'<span class="cs-badge {badge_class}">{badge_label}</span>'
            f'</div>'
            f'{since_html}'
            f'</div>'
        )
    if not blocks:
        blocks.append(
            '<div class="cs-doc-item-detail">No conditions on file yet. Upload a discharge document to add them.</div>'
        )
    return f"""
    <div class="cs-sbar-card cs-sbar-background">
      <div class="cs-sbar-label">Stored conditions</div>
      <div class="cs-sbar-body cs-doc-sbar-body">{"".join(blocks)}</div>
    </div>
    """


def render_documents_stored_overview(
    medications,
    conditions,
    is_demo_meds=False,
    discontinued=None,
    review_flags=None,
):
    discontinued_html = build_discontinued_meds_html(discontinued or [])
    review_html = build_medication_review_flags_html(review_flags or [])
    md_html(f"""
    <div class="cs-sbar-grid cs-doc-stored-grid">
      {build_stored_meds_sbar_html(medications, is_demo=is_demo_meds)}
      {build_stored_conditions_sbar_html(conditions)}
    </div>
    {discontinued_html}
    {review_html}
    """)

st.set_page_config(page_title="CareShield", page_icon="🛡️", layout="centered")

# ── GLOBAL CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Lora:wght@600;700&display=swap');

/* Boot splash — prevent messy first paint on cold load only */
html:not(.cs-app-ready):not(.cs-enter-from-home) [data-testid="stAppViewContainer"]:has(.cs-main-app-marker) .main {
    opacity: 0 !important;
    visibility: hidden !important;
}
html.cs-app-ready [data-testid="stAppViewContainer"]:has(.cs-main-app-marker) .main,
html.cs-enter-from-home [data-testid="stAppViewContainer"]:has(.cs-main-app-marker) .main {
    opacity: 1 !important;
    visibility: visible !important;
    transition: opacity 0.18s ease;
}
#cs-boot-splash {
    position: fixed;
    inset: 0;
    z-index: 999999;
    background: #F9F8F3;
    display: none;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    gap: 14px;
    opacity: 1;
    transition: opacity 0.2s ease;
}
html:not(.cs-app-ready):not(.cs-enter-from-home) [data-testid="stAppViewContainer"]:has(.cs-main-app-marker) #cs-boot-splash {
    display: flex;
}
html.cs-enter-from-home #cs-boot-splash {
    display: none !important;
}
.cs-boot-spinner {
    width: 34px;
    height: 34px;
    border: 3px solid #E8E4DA;
    border-top-color: #5B21B6;
    border-radius: 50%;
    animation: cs-boot-spin 0.8s linear infinite;
}
.cs-boot-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    font-weight: 600;
    color: #7A7568;
    letter-spacing: 0.02em;
}
@keyframes cs-boot-spin {
    to { transform: rotate(360deg); }
}

:root {
    --cs-bg: #F9F8F3;
    --cs-bg-soft: #F3F1EA;
    --cs-surface: #FFFFFF;
    --cs-border: #E8E4DA;
    --cs-border-soft: #F0EBE2;
    --cs-text: #1A2B23;
    --cs-text-muted: #7A7568;
    --cs-text-light: #A09880;
    --cs-green-soft: #E8F5EC;
    --cs-green-badge: #D4EDDA;
    --cs-green-text: #2D6A4F;
    --cs-red-soft: #FDECEA;
    --cs-red-text: #9B2C2C;
    --cs-yellow-soft: #FEF3C7;
    --cs-yellow-text: #92400E;
    --cs-orange-soft: #FFEDD5;
    --cs-orange-text: #C2410C;
    --cs-terracotta: #C45C3E;
    --cs-terracotta-hover: #A84D34;
    --cs-blue: #5B8DEF;
    --cs-blue-soft: #E8F0FE;
    --cs-brand-care: #1B3A6B;
    --cs-brand-shield: #F5B87A;
    --cs-home-navy: #2D3F6B;
    --cs-home-icon-bg: #E8ECF5;
    --cs-home-amber: #E8B97A;
    --cs-home-grey: #7A7469;
    --cs-home-navy-hover: #243456;
    --cs-accent: #5B21B6;
    --cs-accent-hover: #4C1D95;
    --cs-accent-soft: #F5F3FF;
    --cs-accent-border: #C4B5FD;
    --cs-accent-shadow: rgba(91, 33, 182, 0.22);
}

/* Reset & base */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: var(--cs-bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    color: var(--cs-text);
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stCaptionContainer"] p {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
}
[data-testid="stMarkdownContainer"] h3 {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    letter-spacing: -0.4px !important;
    line-height: 1.2 !important;
    color: var(--cs-text) !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* Main container */
.main .block-container {
    max-width: 780px;
    padding: 0 20px 56px 20px !important;
    margin: 0 auto;
}

/* ── HEADER ── */
.cs-header-row {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 24px;
    padding: 28px 0 24px 0;
    border-bottom: 1px solid var(--cs-border);
    margin-bottom: 28px;
}
.cs-logo {
    display: flex;
    align-items: center;
    gap: 14px;
}
.cs-logo-icon {
    width: 44px; height: 44px;
    background: transparent;
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    overflow: hidden;
}
.cs-logo-brand-img {
    width: 44px;
    height: 44px;
    display: block;
    border-radius: 12px;
    object-fit: cover;
}
.cs-logo-text { line-height: 1.15; }
.cs-logo-name {
    font-size: 32px; font-weight: 800;
    letter-spacing: -0.8px;
    font-family: 'DM Sans', sans-serif;
}
.cs-logo-care {
    color: var(--cs-brand-care);
}
.cs-logo-shield-word {
    color: var(--cs-brand-shield);
}
.cs-logo-sub {
    font-size: 10px; font-weight: 600;
    color: var(--cs-text-muted); letter-spacing: 1.4px; text-transform: uppercase;
    margin-top: 4px;
}
.cs-user-area {
    min-width: 0;
}
.cs-user-area-left {
    text-align: left;
}
.cs-user-area-right {
    text-align: right;
}
.cs-avatars {
    display: flex;
    gap: 6px;
    margin-bottom: 10px;
}
.cs-avatars-left {
    justify-content: flex-start;
}
.cs-avatars-right {
    justify-content: flex-end;
}
.cs-user-label {
    font-size: 11px; font-weight: 600;
    color: var(--cs-text-muted);
    letter-spacing: 0.3px;
    margin-bottom: 8px;
}
.cs-header-help-anchor {
    display: block;
    height: 0;
    overflow: hidden;
}
div[data-testid="column"]:has(.cs-header-help-anchor--profiles) [data-testid="stPopover"] > button,
div[data-testid="column"]:has(.cs-header-help-anchor--profiles) [data-testid="stPopover"] button {
    width: 100% !important;
    background: var(--cs-accent) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--cs-accent) !important;
    border-radius: 12px !important;
    min-height: 42px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    margin-top: 8px !important;
    box-shadow: 0 2px 10px var(--cs-accent-shadow) !important;
}
div[data-testid="column"]:has(.cs-header-help-anchor--profiles) [data-testid="stPopover"] > button:hover,
div[data-testid="column"]:has(.cs-header-help-anchor--profiles) [data-testid="stPopover"] button:hover {
    background: var(--cs-accent-hover) !important;
    border-color: var(--cs-accent-hover) !important;
    color: #FFFFFF !important;
}
div[data-testid="column"]:has(.cs-header-help-anchor--safety) [data-testid="stButton"] button {
    width: 100% !important;
    background: var(--cs-accent) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--cs-accent) !important;
    border-radius: 12px !important;
    min-height: 42px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    margin-top: 8px !important;
    box-shadow: 0 2px 10px var(--cs-accent-shadow) !important;
}
div[data-testid="column"]:has(.cs-header-help-anchor--safety) [data-testid="stButton"] button:hover {
    background: var(--cs-accent-hover) !important;
    border-color: var(--cs-accent-hover) !important;
    color: #FFFFFF !important;
}
div[data-testid="column"]:has(.cs-home-back-anchor) [data-testid="stButton"] button {
    background: var(--cs-accent) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--cs-accent) !important;
    border-radius: 10px !important;
    min-height: 34px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    margin-top: 10px !important;
    padding: 0.35rem 0.75rem !important;
    box-shadow: 0 2px 8px var(--cs-accent-shadow) !important;
}
div[data-testid="column"]:has(.cs-home-back-anchor) [data-testid="stButton"] button:hover {
    background: var(--cs-accent-hover) !important;
    border-color: var(--cs-accent-hover) !important;
    color: #FFFFFF !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-medcam-clock-shell) {
    display: flex;
    flex-direction: column;
    align-items: center;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-medcam-clock-shell) iframe {
    margin-left: auto !important;
    margin-right: auto !important;
}
.cs-avatar {
    width: 32px; height: 32px; border-radius: 50%;
    background: var(--cs-bg-soft);
    border: 2px solid var(--cs-border);
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
}
.cs-avatar.active {
    border-color: var(--cs-text);
    background: var(--cs-green-soft);
}
.cs-profile-picker-row {
    margin-top: 2px;
}
[data-testid="stPopover"] button {
    min-width: 42px !important;
    padding: 0.45rem 0.65rem !important;
    border-radius: 12px !important;
    font-size: 16px !important;
    white-space: nowrap !important;
}
[data-testid="stPopover"] > button {
    min-height: 2.35rem !important;
}
[data-testid="stPopoverBody"] {
    min-width: 340px !important;
}
.cs-handover-chart-card--wide {
    margin-top: 16px;
}
.cs-recurring-bar-fill {
    background: #E8F0FE !important;
    border: 1px solid #5B8DEF !important;
    color: #1E3A8A !important;
}
.cs-recurring-bar-row {
    grid-template-columns: 100px minmax(0, 1fr) minmax(4.5rem, auto) !important;
}
.cs-recurring-bar-count {
    white-space: nowrap !important;
    word-break: keep-all !important;
    min-width: 4.5rem !important;
    text-align: right !important;
}
[data-testid="stPopoverBody"] [data-testid="stHorizontalBlock"] [data-testid="column"] {
    min-width: 0 !important;
}
[data-testid="stPopoverBody"] [data-testid="stHorizontalBlock"]:has([data-testid="stButton"]) {
    gap: 0.4rem !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
}
[data-testid="stPopoverBody"] [data-testid="stHorizontalBlock"]:has([data-testid="stButton"]) [data-testid="column"] {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 4.25rem !important;
    max-width: none !important;
}
[data-testid="stPopoverBody"] [data-testid="stButton"] {
    width: auto !important;
    min-width: 4.25rem !important;
}
[data-testid="stPopoverBody"] [data-testid="stButton"] > button {
    writing-mode: horizontal-tb !important;
    white-space: nowrap !important;
    width: auto !important;
    min-width: 4.25rem !important;
    max-width: none !important;
    height: 2.1rem !important;
    min-height: 2.1rem !important;
    padding: 0 0.65rem !important;
    font-size: 12px !important;
    line-height: 1 !important;
    word-break: normal !important;
    overflow: visible !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}
[data-testid="stPopoverBody"] [data-testid="stButton"] > button p,
[data-testid="stPopoverBody"] [data-testid="stButton"] > button span,
[data-testid="stPopoverBody"] [data-testid="stButton"] > button div {
    white-space: nowrap !important;
    word-break: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    writing-mode: horizontal-tb !important;
    display: inline !important;
    line-height: 1 !important;
}

/* ── CARD ── */
.cs-card {
    background: var(--cs-surface);
    border-radius: 18px;
    padding: 26px;
    margin-bottom: 18px;
    border: 1px solid var(--cs-border);
    box-shadow: 0 2px 8px rgba(26, 43, 35, 0.04);
}
.cs-card-title {
    font-size: 16px; font-weight: 700;
    color: var(--cs-text); margin: 0 0 18px 0;
    display: flex; align-items: center; gap: 8px;
    letter-spacing: -0.2px;
    font-family: 'DM Sans', sans-serif;
}
.cs-active-plan {
    background: var(--cs-green-soft);
    border: 1px solid #C6E7D0;
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 16px;
}
.cs-active-plan-label {
    font-size: 10px; font-weight: 700;
    color: var(--cs-green-text);
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

/* ── REPORT & ASK INTRO (shared section header pattern for all tabs) ── */
.cs-report-intro {
    font-family: 'DM Sans', sans-serif;
    margin-bottom: 16px;
}
.cs-report-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: var(--cs-text);
    letter-spacing: -0.4px;
    line-height: 1.2;
    margin: 0 0 8px 0;
}
.cs-report-desc {
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    line-height: 1.6;
    color: var(--cs-text-muted);
    margin: 0 0 12px 0;
    max-width: 680px;
}
.cs-report-disclaimer {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    line-height: 1.55;
    color: #7F1D1D;
    background: #FEF2F2;
    border: 1px solid #FCA5A5;
    border-radius: 12px;
    padding: 12px 14px;
    margin: 0;
    max-width: 680px;
}
.cs-report-disclaimer strong {
    color: #991B1B;
}
.cs-report-story-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
    margin-bottom: 20px;
}
.cs-report-story-card {
    border-radius: 16px;
    padding: 18px 18px 16px 18px;
    border: 1px solid transparent;
}
.cs-report-story-card--problem {
    background: linear-gradient(145deg, #FFF5F0 0%, #FFE8E0 100%);
    border-color: #FECACA;
}
.cs-report-story-card--solution {
    background: linear-gradient(145deg, #F0F7FF 0%, #E8ECF5 100%);
    border-color: #C9D3EA;
}
.cs-report-story-badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.1px;
    text-transform: uppercase;
    color: #9B3D2E;
    background: rgba(255, 255, 255, 0.75);
    border-radius: 999px;
    padding: 4px 10px;
    margin-bottom: 12px;
}
.cs-report-story-badge--solution {
    color: #2D3F6B;
}
.cs-report-story-text {
    font-size: 14px;
    line-height: 1.65;
    color: var(--cs-text);
    margin: 0;
}
.cs-report-safety {
    margin-top: 24px;
    padding-top: 8px;
}
.cs-report-safety-label {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--cs-home-grey);
    margin-bottom: 14px;
}
.cs-report-safety-label--spaced {
    margin-top: 28px;
}
.cs-report-safety-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
}
.cs-report-safety-card {
    background: #FFFFFF;
    border: 1px solid var(--cs-border);
    border-radius: 14px;
    padding: 16px 14px;
    box-shadow: 0 1px 8px rgba(26, 43, 35, 0.04);
}
.cs-report-safety-icon {
    width: 40px;
    height: 40px;
    border-radius: 10px;
    background: var(--cs-home-icon-bg);
    color: var(--cs-home-navy);
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 12px;
}
.cs-report-safety-card-title {
    font-size: 15px;
    font-weight: 800;
    color: var(--cs-text);
    margin-bottom: 6px;
}
.cs-report-safety-card-text {
    font-size: 13px;
    line-height: 1.55;
    color: var(--cs-text-muted);
    margin: 0;
}
.cs-report-responsible-list {
    display: flex;
    flex-direction: column;
    gap: 14px;
}
.cs-report-responsible-item {
    display: flex;
    gap: 12px;
    align-items: flex-start;
}
.cs-report-responsible-icon {
    width: 28px;
    height: 28px;
    border-radius: 8px;
    background: var(--cs-bg-soft);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
    color: var(--cs-home-grey);
}
.cs-report-responsible-title {
    font-size: 14px;
    font-weight: 800;
    color: var(--cs-text);
    margin-bottom: 4px;
}
.cs-report-responsible-text {
    font-size: 13px;
    line-height: 1.55;
    color: var(--cs-text-muted);
    margin: 0;
}
.cs-report-disclaimer--bottom {
    margin-top: 24px;
    max-width: none;
}
@media (max-width: 720px) {
    .cs-report-story-grid,
    .cs-report-safety-grid {
        grid-template-columns: 1fr;
    }
}
.cs-chat-upload-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    font-weight: 600;
    color: var(--cs-text);
    margin: 12px 14px 6px 14px;
}
.cs-inline-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    vertical-align: middle;
    flex-shrink: 0;
    color: currentColor;
}
.cs-inline-icon svg {
    display: block;
}
.cs-icon-warn { color: #B45309; }
.cs-icon-check { color: #2D6A4F; }
.cs-icon-pill { color: #1D4ED8; }
.cs-icon-document { color: #5B21B6; }
.cs-pill-grid-thumb--fallback,
.cs-registered-pill-thumb--fallback {
    display: flex;
    align-items: center;
    justify-content: center;
    color: #64748B;
}
.cs-pill-grid-status--done {
    display: inline-flex;
    align-items: center;
    gap: 5px;
}
.cs-msg-rag-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
}
.cs-msg-disclaimer {
    display: flex;
    align-items: flex-start;
    gap: 6px;
    margin-top: 10px;
    font-size: 12px;
    color: var(--cs-text-muted);
}

/* ── CHAT (messages + composer in one bordered container) ── */
.cs-chat-messages {
    min-height: 380px;
    max-height: 520px;
    overflow-y: auto;
    padding: 20px 22px;
    display: flex;
    flex-direction: column;
    gap: 18px;
    width: 100%;
    box-sizing: border-box;
    background: linear-gradient(145deg, #eef2ff 0%, #fce7f3 35%, #fff7ed 70%, #ecfdf5 100%);
    border-radius: 16px 16px 0 0;
    margin: 0;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-chat-messages) [data-testid="stMarkdownContainer"],
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-chat-messages) [data-testid="stMarkdownContainer"] > div {
    width: 100% !important;
    max-width: 100% !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-chat-messages) {
    border-radius: 20px !important;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(91, 141, 239, 0.08);
    border-color: var(--cs-border) !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-chat-messages) [data-testid="stTextInput"] {
    margin: 0 14px 10px 14px !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-chat-messages) [data-testid="stTextInput"] input {
    background: #ffffff !important;
    color: #1A2B23 !important;
    border: 2.5px solid #1A2B23 !important;
    border-radius: 18px !important;
    height: 58px !important;
    min-height: 58px !important;
    padding: 0 18px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    box-shadow: none !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-chat-messages) [data-testid="stTextInput"] input:focus {
    border-color: #1A2B23 !important;
    box-shadow: 0 0 0 3px rgba(26, 43, 35, 0.1) !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-chat-messages) [data-testid="stFileUploader"] {
    margin: 0 14px 14px 14px !important;
    padding: 0 !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-chat-messages) [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"],
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-chat-messages) [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] > div,
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-chat-messages) [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    border: 1.5px dashed #DADCE0 !important;
    border-radius: 14px !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-chat-messages) [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"]:hover {
    background: #FFFFFF !important;
    border-color: #B0B4BA !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-chat-messages) [data-testid="stButton"] button {
    background: var(--cs-accent) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    height: 46px !important;
    min-height: 46px !important;
    font-weight: 700 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    box-shadow: 0 2px 10px var(--cs-accent-shadow) !important;
    margin: 0 14px 10px 14px !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-chat-messages) [data-testid="stButton"] button:hover {
    background: var(--cs-accent-hover) !important;
    color: #FFFFFF !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-chat-messages) [data-testid="stHorizontalBlock"] {
    padding: 0;
    background: #ffffff;
    border-top: none;
}

.cs-msg {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    animation: cs-fade-in 0.25s ease;
}
@keyframes cs-fade-in {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

.cs-msg.user {
    flex-direction: row;
    justify-content: flex-end;
}
.cs-msg-body {
    max-width: 82%;
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.cs-msg.user .cs-msg-body {
    align-items: flex-end;
}

.cs-msg-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 2px;
}
.cs-msg-meta-right {
    justify-content: flex-end;
}
.cs-msg-sender {
    font-size: 13px;
    font-weight: 700;
    color: var(--cs-text);
}
.cs-msg-badge {
    font-size: 10px;
    font-weight: 600;
    color: var(--cs-text-muted);
    background: #f1f3f4;
    border-radius: 6px;
    padding: 2px 8px;
}
.cs-msg-time {
    font-size: 11px;
    font-weight: 500;
    color: var(--cs-text-muted);
}
.cs-msg-meta-right .cs-msg-time {
    margin-right: 4px;
}
.cs-msg-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: -0.3px;
}
.cs-msg.assistant .cs-msg-avatar {
    background: #e8f0fe;
    color: #1a73e8;
    border: 1.5px solid #c6dafc;
}
.cs-msg-avatar-user {
    background: #d2e3fc;
    color: #174ea6;
    border: 1.5px solid #aecbfa;
}

.cs-msg-bubble {
    padding: 12px 16px;
    border-radius: 20px;
    font-size: 14px;
    line-height: 1.55;
    color: #202124;
    word-break: break-word;
}
.cs-msg.assistant .cs-msg-bubble {
    background: #f1f3f4;
    border: none;
    border-radius: 4px 20px 20px 20px;
}
.cs-msg.user .cs-msg-bubble {
    background: #d3e3fd;
    color: #202124;
    border-radius: 20px 4px 20px 20px;
}

.cs-welcome-lead {
    font-size: 15px;
    font-weight: 600;
    margin: 0 0 6px 0;
    color: #202124;
}
.cs-welcome-sub {
    font-size: 13px;
    color: #5f6368;
    margin: 0 0 4px 0;
}
.cs-welcome-note {
    font-size: 12px;
    line-height: 1.55;
    color: var(--cs-home-grey);
    margin: 6px 0 0 0;
    font-style: italic;
}
.cs-welcome-toggles {
    margin-top: 14px;
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.cs-welcome-details {
    border: 1px solid var(--cs-border);
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.65);
    overflow: hidden;
}
.cs-welcome-details summary {
    list-style: none;
    cursor: pointer;
    padding: 10px 12px;
    font-size: 13px;
    font-weight: 700;
    color: var(--cs-home-navy);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    user-select: none;
}
.cs-welcome-details summary::-webkit-details-marker {
    display: none;
}
.cs-welcome-details summary::after {
    content: "›";
    font-size: 18px;
    font-weight: 400;
    color: var(--cs-home-grey);
    line-height: 1;
    transition: transform 0.15s ease;
}
.cs-welcome-details[open] summary::after {
    transform: rotate(90deg);
}
.cs-welcome-details-body {
    padding: 0 12px 12px 12px;
    font-size: 13px;
    line-height: 1.6;
    color: var(--cs-text-muted);
    border-top: 1px solid var(--cs-border-soft);
}
.cs-welcome-details-body p {
    margin: 10px 0 0 0;
}
.cs-welcome-details-body ul {
    margin: 10px 0 0 0;
    padding-left: 18px;
}
.cs-welcome-details-body li {
    margin-bottom: 6px;
}
.cs-welcome-examples {
    margin-top: 12px;
}
.cs-welcome-examples p {
    font-size: 13px;
    font-weight: 600;
    margin: 12px 0 6px 0;
    color: var(--cs-text-muted);
}
.cs-welcome-examples p:first-child {
    margin-top: 4px;
}
.cs-welcome-examples ul {
    margin: 0 0 4px 0;
    padding-left: 18px;
    font-size: 13px;
    color: var(--cs-text-muted);
    line-height: 1.45;
}
.cs-welcome-examples li {
    margin-bottom: 4px;
}
.cs-msg-tag {
    display: inline-block;
    padding: 2px 12px;
    border-radius: 100px;
    font-size: 11px;
    font-weight: 700;
    margin-top: 8px;
    letter-spacing: 0.3px;
    text-transform: uppercase;
}
.cs-msg-severity-banner {
    margin: -12px -16px 12px -16px;
    padding: 10px 16px;
    border-radius: 4px 20px 0 0;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.6px;
    text-align: center;
    text-transform: uppercase;
}
.cs-confidence-indicator {
    margin: -4px 0 10px 0;
    padding: 0 2px 2px 2px;
    font-size: 11px;
    font-weight: 600;
    color: #5f6368;
    text-align: center;
}
.cs-condition-risk-alert {
    margin-top: 12px;
    padding: 12px 14px;
    border-radius: 12px;
    background: linear-gradient(145deg, #FEFCE8 0%, #FEF9C3 100%);
    border-left: 4px solid #CA8A04;
    border: 1.5px solid #FDE047;
    border-left-width: 4px;
}
.cs-condition-risk-title {
    font-size: 13px;
    font-weight: 800;
    color: #854D0E;
    margin-bottom: 6px;
    line-height: 1.4;
}
.cs-condition-risk-body {
    font-size: 13px;
    line-height: 1.55;
    color: #713F12;
}
.cs-dose-nudge {
    margin: 0 14px 12px 14px;
    padding: 14px 16px;
    border-radius: 14px;
    background: #EFF6FF;
    border: 1.5px solid #BFDBFE;
}
.cs-dose-nudge-sticky {
    position: sticky;
    top: 0;
    z-index: 100;
    margin: -8px -1rem 14px -1rem;
    padding: 14px 18px;
    border-radius: 0 0 16px 16px;
    box-shadow: 0 4px 18px rgba(29, 78, 216, 0.12);
}
.cs-dose-nudge-title {
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.4px;
    text-transform: uppercase;
    color: #1D4ED8;
    margin-bottom: 8px;
}
.cs-dose-nudge-item {
    font-size: 15px;
    font-weight: 700;
    line-height: 1.45;
    color: #1E3A8A;
    margin-bottom: 6px;
}
.cs-dose-nudge-more,
.cs-dose-nudge-hint {
    font-size: 12px;
    line-height: 1.45;
    color: #475569;
    margin-top: 4px;
}
.cs-tag-ok { background: var(--cs-green-badge); color: var(--cs-green-text); }
.cs-tag-monitor { background: var(--cs-yellow-soft); color: var(--cs-yellow-text); }
.cs-tag-contact_doctor { background: var(--cs-orange-soft); color: var(--cs-orange-text); }
.cs-tag-emergency,
.cs-tag-urgent {
    background: #DC2626;
    color: #FFFFFF;
}

.cs-emergency-call-bar {
    margin-top: 14px;
    padding: 14px;
    border-radius: 14px;
    background: #FEE2E2;
    border: 1.5px solid #FCA5A5;
}
.cs-emergency-call-label {
    margin: 0 0 12px 0;
    font-size: 13px;
    font-weight: 700;
    color: #991B1B;
    line-height: 1.45;
}
.cs-emergency-call-actions {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}
.cs-emergency-call-btn {
    flex: 1 1 120px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 52px;
    padding: 12px 18px;
    border-radius: 14px;
    background: #DC2626;
    color: #FFFFFF !important;
    text-decoration: none !important;
    font-size: 16px;
    font-weight: 800;
    letter-spacing: 0.2px;
    box-shadow: 0 4px 14px rgba(220, 38, 38, 0.28);
}
.cs-emergency-call-btn-alt {
    background: #FFFFFF;
    color: #991B1B !important;
    border: 2px solid #DC2626;
    box-shadow: none;
}

.cs-msg-photo-tag {
    font-size: 12px;
    color: #5f6368;
    margin-top: 8px;
    font-weight: 500;
}
.cs-msg-thinking .cs-thinking-bubble {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #5f6368;
    font-style: italic;
}
.cs-thinking-spinner {
    width: 16px;
    height: 16px;
    border: 2px solid #dadce0;
    border-top-color: #1a73e8;
    border-radius: 50%;
    flex-shrink: 0;
    animation: cs-spin 0.9s linear infinite;
}
@keyframes cs-spin { to { transform: rotate(360deg); } }

.cs-chat-placeholder {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    padding: 30px 20px;
    color: var(--cs-text-light);
    text-align: center;
}
.cs-chat-placeholder .cs-placeholder-icon {
    font-size: 32px;
    margin-bottom: 12px;
    opacity: 0.5;
}
.cs-chat-placeholder .cs-placeholder-text {
    font-size: 14px;
    font-weight: 400;
    line-height: 1.5;
    max-width: 320px;
}

.cs-chat-input-area {
    padding: 14px 16px 16px 16px;
    border-top: 1px solid var(--cs-border-soft);
    background: #ffffff;
    border-radius: 0 0 20px 20px;
    margin-top: -2px;
    border: 1px solid var(--cs-border);
    border-top: 1px solid var(--cs-border-soft);
    box-shadow: 0 4px 24px rgba(91, 141, 239, 0.06);
}
.cs-chat-panel [data-testid="stTextArea"] textarea {
    background: #ffffff !important;
    border: 1px solid #dadce0 !important;
    border-radius: 24px !important;
    min-height: 48px !important;
    padding: 12px 18px !important;
}
.cs-chat-panel [data-testid="stButton"] button {
    border-radius: 24px !important;
}
.cs-chat-panel [data-testid="stButton"] button[kind="secondary"],
.cs-chat-panel div[data-testid="column"]:last-child [data-testid="stButton"] button {
    background: #ffffff !important;
    color: #5f6368 !important;
    border: 1px solid #dadce0 !important;
}

/* ── MED TABLE ── */
.cs-med-table {
    width: 100%; border-collapse: collapse;
    font-size: 14px;
}
.cs-med-table th {
    text-align: left; padding: 8px 12px;
    font-size: 11px; font-weight: 600;
    color: var(--cs-text-muted); letter-spacing: 0.5px;
    text-transform: uppercase;
    border-bottom: 1px solid var(--cs-border);
}
.cs-med-table td {
    padding: 10px 12px;
    border-bottom: 1px solid var(--cs-bg);
    color: var(--cs-text);
}
.cs-med-table tr:last-child td { border-bottom: none; }
.cs-med-badge {
    display: inline-block;
    background: var(--cs-green-badge); color: var(--cs-green-text);
    border-radius: 100px; padding: 2px 10px;
    font-size: 12px; font-weight: 500;
}

/* ── CONDITIONS ── */
.cs-condition {
    display: flex; justify-content: space-between;
    align-items: center; padding: 10px 0;
    border-bottom: 1px solid #F5F0E8;
}
.cs-condition:last-child { border-bottom: none; }
.cs-condition-name { font-size: 14px; font-weight: 600; color: var(--cs-text); }
.cs-condition-date { font-size: 12px; color: var(--cs-text-muted); }
.cs-badge {
    font-size: 10px; font-weight: 700;
    padding: 4px 12px; border-radius: 100px;
    text-transform: uppercase; letter-spacing: 0.6px;
}
.cs-badge-chronic { background: var(--cs-red-soft); color: var(--cs-red-text); }
.cs-badge-recovery { background: var(--cs-yellow-soft); color: var(--cs-yellow-text); }
.cs-badge-acute { background: #FEE2E2; color: #B91C1C; }

/* ── SBAR / HANDOVER ── */
.cs-handover-card {
    padding: 4px 2px 8px 2px;
}
.cs-handover-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: var(--cs-text);
    letter-spacing: -0.4px;
    line-height: 1.2;
    margin-bottom: 8px;
}
.cs-handover-desc {
    font-size: 14px;
    line-height: 1.6;
    color: var(--cs-text-muted);
    margin: 0 0 20px 0;
    max-width: 680px;
}
.cs-handover-onboard {
    background: #FFFFFF;
    border: 1px solid var(--cs-border);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
    font-family: 'DM Sans', sans-serif;
}
.cs-handover-onboard h3 {
    font-family: 'DM Sans', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: var(--cs-text);
    margin: 0 0 8px 0;
    letter-spacing: -0.3px;
}
.cs-handover-onboard em {
    font-size: 15px;
    line-height: 1.65;
    color: var(--cs-text-muted);
    font-style: italic;
}

/* ── WHY THIS EXISTS / HOME ── */
.cs-why-screen {
    max-width: 760px;
    margin: 36px auto 24px auto;
    padding: 0 8px;
    font-family: 'DM Sans', sans-serif;
}
.cs-why-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 36px;
}
.cs-why-logo .cs-logo-care {
    color: var(--cs-home-navy);
}
.cs-why-logo .cs-logo-shield-word {
    color: var(--cs-home-amber);
}
.cs-why-logo .cs-logo-sub {
    color: var(--cs-home-grey);
}
.cs-why-kicker {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: var(--cs-home-grey);
    margin: 0 0 18px 0;
}
.cs-why-headline {
    margin: 0 0 18px 0;
    font-family: 'DM Sans', sans-serif;
    font-size: clamp(34px, 6vw, 46px);
    font-weight: 800;
    line-height: 1.08;
    letter-spacing: -1px;
}
.cs-why-headline-dark {
    display: block;
    color: var(--cs-home-navy);
}
.cs-why-headline-accent {
    display: block;
    color: var(--cs-home-amber);
}
.cs-why-lead {
    margin: 0 0 28px 0;
    font-size: 17px;
    line-height: 1.65;
    color: var(--cs-home-navy);
    max-width: 640px;
}
.cs-why-stat-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 16px;
    margin-bottom: 32px;
}
.cs-why-stat-card {
    background: var(--cs-home-navy);
    border: none;
    border-radius: 16px;
    padding: 22px 20px 18px 20px;
    box-shadow: none;
}
.cs-why-stat-value {
    font-size: 42px;
    font-weight: 800;
    color: var(--cs-home-amber);
    line-height: 1;
    margin-bottom: 12px;
}
.cs-why-stat-text {
    font-size: 15px;
    line-height: 1.55;
    color: #FFFFFF;
    margin-bottom: 18px;
}
.cs-why-stat-text strong {
    color: #FFFFFF;
    font-weight: 800;
}
.cs-why-stat-source {
    font-size: 11px;
    line-height: 1.45;
    color: rgba(255, 255, 255, 0.72);
}
.cs-why-section {
    border-left: 4px solid var(--cs-home-amber);
    padding-left: 16px;
    margin-bottom: 24px;
}
.cs-why-section-label {
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--cs-home-amber);
    margin-bottom: 12px;
}
.cs-why-section-copy {
    margin: 0;
    font-size: 16px;
    line-height: 1.65;
    color: var(--cs-home-navy);
}
.cs-why-section-copy strong {
    font-weight: 800;
}
.cs-why-solution {
    margin: 0 0 14px 0;
    font-size: 17px;
    line-height: 1.6;
    color: var(--cs-home-navy);
}
.cs-why-solution strong {
    font-weight: 800;
}
.cs-why-solution-accent {
    color: var(--cs-home-amber);
}
.cs-why-support {
    margin: 0 0 28px 0;
    font-size: 16px;
    line-height: 1.65;
    color: var(--cs-home-navy);
}
.cs-why-cta-box {
    border: 1.5px solid var(--cs-home-icon-bg);
    border-radius: 14px;
    background: #FFFFFF;
    padding: 18px 20px;
    margin-bottom: 28px;
}
.cs-why-cta-title {
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.1px;
    text-transform: uppercase;
    color: var(--cs-home-navy);
    margin-bottom: 10px;
}
.cs-why-cta-copy {
    margin: 0;
    font-size: 15px;
    line-height: 1.65;
    color: var(--cs-home-navy);
}
.cs-why-cta-copy strong {
    font-weight: 800;
}
.cs-how-section {
    margin-top: 28px;
    margin-bottom: 8px;
}
.cs-how-bar {
    margin: 0 0 24px 0;
}
.cs-how-bar .cs-how-section {
    margin-top: 0;
    margin-bottom: 0;
}
.cs-how-nav-row {
    display: flex;
    align-items: center;
    gap: 8px;
}
.cs-how-nav-btn {
    flex: 0 0 34px;
    width: 34px;
    height: 34px;
    border-radius: 50%;
    border: 1px solid var(--cs-border);
    background: #FFFFFF;
    color: var(--cs-home-navy);
    font-size: 22px;
    line-height: 1;
    font-weight: 400;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0 0 2px 0;
    box-shadow: 0 1px 4px rgba(45, 63, 107, 0.08);
    transition: background 0.15s ease, border-color 0.15s ease;
}
.cs-how-nav-btn:hover {
    background: var(--cs-home-icon-bg);
    border-color: #C9D3EA;
}
.cs-how-track {
    flex: 1;
    min-width: 0;
    position: relative;
    padding: 6px 0 4px 0;
}
.cs-first-time-guide-row {
    margin: -8px 0 16px 0;
}
.cs-report-safety--footer {
    margin-top: 28px;
    padding-top: 4px;
}
.cs-how-section-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--cs-home-grey);
    margin-bottom: 16px;
    text-align: center;
}
.cs-how-line {
    position: absolute;
    top: 50%;
    left: 28px;
    right: 28px;
    height: 2px;
    background: #E8E4DA;
    transform: translateY(-50%);
    z-index: 0;
    pointer-events: none;
}
.cs-how-steps {
    display: flex;
    gap: 10px;
    overflow-x: auto;
    padding: 4px 2px 8px 2px;
    position: relative;
    z-index: 1;
    scroll-snap-type: x mandatory;
    -webkit-overflow-scrolling: touch;
}
.cs-how-step {
    flex: 0 0 168px;
    min-width: 168px;
    border-radius: 14px;
    padding: 14px 12px 16px 12px;
    text-align: center;
    scroll-snap-align: start;
    box-shadow: 0 1px 6px rgba(26, 43, 35, 0.04);
}
.cs-how-step-num {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid rgba(255, 255, 255, 0.95);
    font-size: 14px;
    font-weight: 800;
    line-height: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 10px auto;
}
.cs-how-step-title {
    font-size: 14px;
    font-weight: 800;
    line-height: 1.25;
    margin-bottom: 8px;
}
.cs-how-step-text {
    font-size: 12px;
    line-height: 1.55;
    font-weight: 500;
}
.cs-how-step--documents {
    background: #E8F5EC;
    color: #2D6A4F;
}
.cs-how-step--documents .cs-how-step-num {
    color: #2D6A4F;
}
.cs-how-step--report {
    background: #FFE8E0;
    color: #9B3D2E;
}
.cs-how-step--report .cs-how-step-num {
    color: #9B3D2E;
}
.cs-how-step--pill {
    background: #EDE9FE;
    color: #5B21B6;
}
.cs-how-step--pill .cs-how-step-num {
    color: #5B21B6;
}
.cs-how-step--medcam {
    background: #E8ECF5;
    color: #2D3F6B;
}
.cs-how-step--medcam .cs-how-step-num {
    color: #2D3F6B;
}
.cs-how-step--handover {
    background: #FFEDD5;
    color: #C2410C;
}
.cs-how-step--handover .cs-how-step-num {
    color: #C2410C;
}
.cs-how-step--results {
    background: #EAF4E8;
    color: #3B6D11;
}
.cs-how-step--results .cs-how-step-num {
    color: #3B6D11;
}
.cs-why-feature-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 16px;
    margin-bottom: 8px;
}
.cs-why-feature-card {
    background: #FFFFFF;
    border: 1px solid var(--cs-home-icon-bg);
    border-radius: 14px;
    padding: 18px 18px 16px 18px;
}
.cs-why-feature-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: var(--cs-home-navy);
    margin-bottom: 12px;
}
.cs-why-feature-title {
    font-size: 16px;
    font-weight: 800;
    color: var(--cs-home-navy);
    margin-bottom: 8px;
    line-height: 1.3;
}
.cs-why-feature-text {
    font-size: 14px;
    line-height: 1.6;
    color: var(--cs-home-grey);
}
@media (max-width: 680px) {
    .cs-why-stat-grid,
    .cs-why-feature-grid {
        grid-template-columns: 1fr;
    }
    .cs-how-line {
        display: none;
    }
    .cs-how-steps {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        overflow-x: visible;
    }
    .cs-how-step {
        flex: none;
        min-width: 0;
        width: 100%;
    }
}
@media (max-width: 480px) {
    .cs-how-steps {
        grid-template-columns: 1fr;
    }
}
.main .block-container:has(.cs-why-screen) [data-testid="stButton"] button {
    background: var(--cs-home-navy) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--cs-home-navy) !important;
    border-radius: 14px !important;
    min-height: 52px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    box-shadow: 0 2px 10px rgba(45, 63, 107, 0.18) !important;
}
.main .block-container:has(.cs-why-screen) [data-testid="stButton"] button:hover,
.main .block-container:has(.cs-why-screen) [data-testid="stButton"] button:focus:not(:active) {
    background: var(--cs-home-navy-hover) !important;
    border-color: var(--cs-home-navy-hover) !important;
    color: #FFFFFF !important;
}
.main .block-container:has(.cs-why-screen) [data-testid="stButton"] button:focus-visible {
    outline: 3px solid rgba(232, 185, 122, 0.55) !important;
    outline-offset: 3px !important;
}

/* ── SYMPTOM TIMELINE ── */
.cs-symptom-timeline {
    margin-bottom: 0;
}
.cs-med-adherence-timeline {
    margin-bottom: 0;
}
.cs-med-adherence-timeline-empty,
.cs-symptom-timeline-empty {
    background: #FFFFFF;
    border: 1px solid var(--cs-border);
    border-radius: 18px;
    padding: 22px 24px;
}
.cs-handover-charts-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 16px;
    margin-bottom: 8px;
}
@media (max-width: 720px) {
    .cs-handover-charts-grid {
        grid-template-columns: 1fr;
    }
}
.cs-handover-chart-card {
    background: #FFFFFF;
    border: 1px solid var(--cs-border);
    border-radius: 18px;
    padding: 18px 18px 16px 18px;
    min-height: 280px;
}
.cs-handover-chart-title {
    font-size: 17px;
    font-weight: 700;
    color: var(--cs-text);
    margin-bottom: 14px;
    letter-spacing: -0.2px;
}
.cs-handover-chart-empty {
    margin: 0;
    font-size: 13px;
    line-height: 1.55;
    color: var(--cs-text-muted);
}
.cs-adherence-chart-body {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 16px;
}
.cs-adherence-ring-wrap {
    flex-shrink: 0;
}
.cs-adherence-ring-css {
    width: 112px;
    height: 112px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
}
.cs-adherence-ring-hole {
    width: 78px;
    height: 78px;
    border-radius: 50%;
    background: #FFFFFF;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    box-shadow: inset 0 0 0 1px rgba(26, 43, 35, 0.06);
}
.cs-adherence-ring-pct {
    font-size: 22px;
    font-weight: 800;
    color: #1A2B23;
    line-height: 1;
}
.cs-adherence-ring-sub {
    font-size: 9px;
    font-weight: 700;
    color: #7A7568;
    letter-spacing: 0.3px;
    text-transform: uppercase;
    margin-top: 4px;
}
.cs-adherence-summary {
    font-size: 16px;
    font-weight: 700;
    color: var(--cs-text);
    line-height: 1.3;
}
.cs-adherence-summary-sub {
    font-size: 12px;
    color: var(--cs-text-muted);
    margin-top: 4px;
}
.cs-adherence-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 10px;
}
.cs-adherence-legend-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 600;
    color: var(--cs-text-muted);
}
.cs-adherence-swatch {
    width: 10px;
    height: 10px;
    border-radius: 3px;
    display: inline-block;
}
.cs-adherence-swatch--taken { background: #34C759; }
.cs-adherence-swatch--missed { background: #FF453A; }
.cs-adherence-daily-title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.35px;
    text-transform: uppercase;
    color: var(--cs-text-muted);
    margin-bottom: 8px;
}
.cs-adherence-daily {
    display: grid;
    grid-template-columns: repeat(7, minmax(0, 1fr));
    gap: 6px;
    align-items: end;
}
.cs-adherence-day {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
}
.cs-adherence-day-stack {
    width: 100%;
    max-width: 28px;
    height: 72px;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    background: #F3F1EA;
    border-radius: 8px;
    overflow: hidden;
}
.cs-adherence-day-seg--taken { background: #34C759; }
.cs-adherence-day-seg--missed { background: #FF453A; }
.cs-adherence-day-label {
    font-size: 10px;
    font-weight: 700;
    color: var(--cs-text-muted);
}
.cs-severity-chart-sub {
    font-size: 12px;
    color: var(--cs-text-muted);
    margin: -6px 0 16px 0;
}
.cs-severity-bars {
    display: flex;
    flex-direction: column;
    gap: 14px;
}
.cs-severity-bar-row {
    display: grid;
    grid-template-columns: 92px 1fr 28px;
    gap: 10px;
    align-items: center;
}
.cs-severity-bar-label {
    font-size: 12px;
    font-weight: 700;
    color: var(--cs-text);
}
.cs-severity-bar-track {
    height: 28px;
    background: #F3F1EA;
    border-radius: 999px;
    overflow: hidden;
}
.cs-severity-bar-fill {
    height: 100%;
    min-width: 0;
    border-radius: 999px;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding: 0 10px;
    box-sizing: border-box;
}
.cs-severity-bar-fill-text {
    font-size: 12px;
    font-weight: 800;
}
.cs-severity-bar-count {
    font-size: 14px;
    font-weight: 800;
    color: var(--cs-text);
    text-align: right;
}
.cs-handover-timeline-divider {
    height: 1px;
    background: var(--cs-border);
    margin: 28px 0 24px 0;
}
.cs-handover-dashboard-shell {
    min-height: 4px;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-handover-dashboard-shell) {
    border-radius: 24px !important;
    overflow: hidden;
    box-shadow: 0 2px 16px rgba(26, 43, 35, 0.06);
    border-color: var(--cs-border) !important;
    margin-bottom: 8px;
}
.cs-timeline-heading {
    font-size: 20px;
    font-weight: 700;
    color: var(--cs-text);
    margin-bottom: 16px;
    letter-spacing: -0.3px;
}
.cs-timeline-accent-card {
    background: linear-gradient(180deg, #FBFAFF 0%, #FFFFFF 100%);
    border: 1.5px solid var(--cs-accent-border);
    border-radius: 18px;
    padding: 22px 24px;
    margin-bottom: 8px;
    box-shadow: 0 2px 16px rgba(91, 33, 182, 0.08);
}
.cs-timeline-accent-card .cs-timeline-heading {
    color: var(--cs-accent);
    margin-bottom: 8px;
}
.cs-timeline-description {
    margin: 0 0 18px 0;
    font-size: 14px;
    line-height: 1.55;
    color: var(--cs-text-muted);
}
.cs-loading-banner {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 18px;
    margin: 0 0 16px 0;
    background: var(--cs-accent-soft);
    border: 1px solid var(--cs-accent-border);
    border-radius: 14px;
    color: var(--cs-accent);
    font-size: 14px;
    font-weight: 600;
}
.cs-loading-text {
    line-height: 1.45;
}
.cs-loading-spinner {
    width: 18px;
    height: 18px;
    border: 2px solid var(--cs-accent-border);
    border-top-color: var(--cs-accent);
    border-radius: 50%;
    animation: cs-spin 0.8s linear infinite;
    flex-shrink: 0;
}
@keyframes cs-spin {
    to { transform: rotate(360deg); }
}
.cs-upload-confirm {
    margin: 12px 0 4px 0;
    padding: 16px 18px;
    background: #FFFFFF;
    border: 1px solid var(--cs-border);
    border-radius: 14px;
}
.cs-upload-confirm-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    color: var(--cs-text-muted);
    margin-bottom: 6px;
}
.cs-upload-confirm-name {
    font-size: 15px;
    font-weight: 700;
    color: var(--cs-text);
    word-break: break-word;
}
.cs-upload-confirm-meta {
    margin-top: 4px;
    font-size: 13px;
    color: var(--cs-text-muted);
}
.cs-upload-confirm-hint {
    margin-top: 10px;
    font-size: 13px;
    line-height: 1.5;
    color: var(--cs-text-muted);
}
.cs-timeline-empty-text {
    margin: 0;
    font-size: 14px;
    line-height: 1.6;
    color: var(--cs-text-muted);
}
.cs-timeline-track {
    position: relative;
    padding-left: 28px;
}
.cs-timeline-track::before {
    content: "";
    position: absolute;
    left: 9px;
    top: 8px;
    bottom: 8px;
    width: 2px;
    background: linear-gradient(180deg, #DADCE0 0%, #E8E4DA 100%);
    border-radius: 2px;
}
.cs-timeline-item {
    position: relative;
    margin-bottom: 16px;
}
.cs-timeline-item:last-child {
    margin-bottom: 0;
}
.cs-timeline-more-indicator {
    position: relative;
    margin: 0 0 14px 0;
    padding: 2px 0 2px 0;
    text-align: center;
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 4px;
    color: var(--cs-text-light);
    line-height: 1;
}
.cs-timeline-dot {
    position: absolute;
    left: -28px;
    top: 18px;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    border: 3px solid #92400E;
    background: #FEF3C7;
    box-shadow: 0 0 0 3px #FFFFFF;
    z-index: 1;
}
.cs-timeline-card {
    background: #FFFFFF;
    border: 1px solid var(--cs-border);
    border-left: 4px solid #92400E;
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: 0 1px 8px rgba(26, 43, 35, 0.04);
}
.cs-timeline-meta {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
}
.cs-timeline-time {
    font-size: 12px;
    font-weight: 700;
    color: var(--cs-text);
}
.cs-timeline-badge {
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.5px;
    padding: 3px 10px;
    border-radius: 100px;
    text-transform: uppercase;
}
.cs-timeline-source {
    font-size: 11px;
    font-weight: 600;
    color: var(--cs-text-muted);
    background: var(--cs-bg-soft);
    padding: 2px 8px;
    border-radius: 100px;
}
.cs-timeline-text {
    font-size: 14px;
    line-height: 1.55;
    color: var(--cs-text);
    margin-bottom: 6px;
}
.cs-timeline-caregiver {
    font-size: 12px;
    color: var(--cs-text-muted);
}

.cs-sbar-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin: 24px 0 18px 0;
    font-family: 'DM Sans', sans-serif;
}
.cs-sbar-card {
    border-radius: 16px;
    padding: 22px 24px;
    font-family: 'DM Sans', sans-serif;
}
.cs-sbar-label {
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.1px;
    margin-bottom: 12px;
    font-family: 'DM Sans', sans-serif;
}
.cs-sbar-body {
    font-size: 16px;
    line-height: 1.6;
    color: var(--cs-text);
    font-family: 'DM Sans', sans-serif;
}
.cs-sbar-situation { background: #DBEAFE; }
.cs-sbar-situation .cs-sbar-label { color: #1E40AF; }
.cs-sbar-background { background: #EDE9FE; }
.cs-sbar-background .cs-sbar-label { color: #5B21B6; }
.cs-sbar-assessment { background: #FEF9C3; }
.cs-sbar-assessment .cs-sbar-label { color: #92400E; }
.cs-sbar-recommendation { background: #FFE4E6; }
.cs-sbar-recommendation .cs-sbar-label { color: #9F1239; }
.cs-dose-card-taken { background: #D1FAE5; }
.cs-dose-card-taken .cs-sbar-label { color: #065F46; }
.cs-dose-card {
    min-height: 132px;
    margin-bottom: 4px;
}
.cs-dose-card-time {
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 6px;
}
.cs-dose-card-status {
    font-size: 14px;
    color: var(--cs-text-muted);
}
.cs-sbar-watch {
    background: #FFF8ED;
    border: 1px solid #F3DEB0;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 24px;
    font-family: 'DM Sans', sans-serif;
}
.cs-sbar-watch-label {
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.1px;
    color: #92400E;
    margin-bottom: 10px;
    font-family: 'DM Sans', sans-serif;
}
.cs-sbar-watch-text {
    font-size: 16px;
    line-height: 1.6;
    color: var(--cs-text);
    font-family: 'DM Sans', sans-serif;
}
.cs-sbar-peak {
    border-radius: 16px;
    padding: 16px 20px;
    margin-bottom: 16px;
    font-family: 'DM Sans', sans-serif;
}
.cs-sbar-peak-emergency {
    background: #FEE2E2;
    border: 1px solid #FCA5A5;
}
.cs-sbar-peak-contact {
    background: #FFEDD5;
    border: 1px solid #FDBA74;
}
.cs-sbar-peak-label {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #7F1D1D;
    margin-bottom: 6px;
}
.cs-sbar-peak-value {
    font-size: 20px;
    font-weight: 800;
    color: #991B1B;
    margin-bottom: 4px;
}
.cs-sbar-peak-contact .cs-sbar-peak-value { color: #9A3412; }
.cs-sbar-peak-detail {
    font-size: 14px;
    color: #7F1D1D;
}
.cs-sbar-photo-note {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 16px;
    font-size: 14px;
    color: #166534;
    font-family: 'DM Sans', sans-serif;
}
.cs-reported-by-heading {
    font-family: 'DM Sans', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: var(--cs-text);
    margin: 8px 0 14px 0;
    letter-spacing: -0.2px;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-handover-card) div[data-testid="column"] [data-testid="stButton"] button {
    background: var(--cs-text) !important;
    color: var(--cs-bg) !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-handover-card) div[data-testid="column"] [data-testid="stButton"] button[kind="secondary"] {
    background: var(--cs-surface) !important;
    color: var(--cs-text) !important;
    border: 1px solid var(--cs-border) !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-handover-generate-card) {
    border-radius: 24px !important;
    overflow: hidden;
    box-shadow: 0 2px 16px rgba(26, 43, 35, 0.06);
    border-color: var(--cs-border) !important;
    margin-bottom: 8px;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-handover-generate-card) [data-testid="stButton"] button {
    background: var(--cs-accent) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    width: 100% !important;
    min-height: 52px !important;
    padding: 14px 24px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    letter-spacing: -0.2px !important;
    box-shadow: 0 4px 14px var(--cs-accent-shadow) !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-handover-generate-card) [data-testid="stButton"] button:hover {
    background: var(--cs-accent-hover) !important;
    color: #FFFFFF !important;
    box-shadow: 0 6px 18px rgba(91, 33, 182, 0.28) !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-handover-card) [data-testid="stDownloadButton"] button {
    background: var(--cs-text) !important;
    color: var(--cs-bg) !important;
    border: none !important;
    border-radius: 100px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    padding: 12px 28px !important;
    box-shadow: 0 2px 10px rgba(26, 43, 35, 0.12);
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-handover-card) {
    border-radius: 24px !important;
    overflow: hidden;
    box-shadow: 0 2px 16px rgba(26, 43, 35, 0.06);
    border-color: var(--cs-border) !important;
    margin-bottom: 8px;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-handover-card) .cs-handover-card {
    margin-bottom: 0;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-handover-card) .cs-period-row {
    margin-bottom: 8px;
}

/* ── BUTTONS ── */
.cs-btn-primary {
    background: var(--cs-text); color: var(--cs-bg);
    border: none; border-radius: 100px;
    padding: 11px 24px; font-size: 14px; font-weight: 600;
    cursor: pointer; font-family: 'DM Sans', sans-serif;
    transition: opacity 0.15s;
}
.cs-btn-primary:hover { opacity: 0.85; }
.cs-btn-secondary {
    background: transparent; color: var(--cs-text);
    border: 1px solid var(--cs-border); border-radius: 100px;
    padding: 10px 22px; font-size: 14px; font-weight: 500;
    cursor: pointer; font-family: 'DM Sans', sans-serif;
}
.cs-btn-terracotta {
    display: inline-block;
    background: var(--cs-terracotta); color: #FFFFFF;
    border: none; border-radius: 100px;
    padding: 12px 28px; font-size: 14px; font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    margin-top: 14px;
    text-align: center;
}
.cs-alert-card {
    background: var(--cs-red-soft);
    border: 1px solid #F5C6C0;
    border-radius: 16px;
    padding: 20px 22px;
    margin-top: 14px;
}
.cs-alert-card-title {
    font-size: 14px; font-weight: 700;
    color: var(--cs-red-text);
    margin-bottom: 8px;
}
.cs-alert-card-body {
    font-size: 13px; color: #7F1D1D;
    line-height: 1.5;
}

/* ── PERIOD SELECTOR ── */
.cs-period-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 18px; }
.cs-period-btn {
    padding: 9px 18px; border-radius: 100px;
    font-size: 13px; font-weight: 500;
    border: 1px solid var(--cs-border);
    background: var(--cs-surface); color: var(--cs-text-muted); cursor: pointer;
    font-family: 'DM Sans', sans-serif;
}
.cs-period-btn.active {
    background: var(--cs-text); color: var(--cs-bg); border-color: var(--cs-text);
}

/* ── ALERT BANNER ── */
.cs-alert {
    background: #FFF7ED; border: 1px solid #FED7AA;
    border-radius: 14px; padding: 14px 18px;
    font-size: 14px; color: #9A3412;
    margin-bottom: 16px; display: flex; gap: 10px;
}

/* Override Streamlit native widgets to match theme */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: var(--cs-bg-soft) !important;
    border: 1px solid var(--cs-border) !important;
    border-radius: 14px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    color: var(--cs-text) !important;
}
[data-testid="stButton"] button {
    background: var(--cs-text) !important;
    color: var(--cs-bg) !important;
    border: none !important;
    border-radius: 100px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 0.5rem 1.25rem !important;
}
[data-testid="stButton"] button:hover {
    background: #0F1A15 !important;
    color: var(--cs-bg) !important;
}
[data-testid="stButton"] button[kind="secondary"] {
    background: var(--cs-surface) !important;
    color: var(--cs-text) !important;
    border: 1px solid var(--cs-border) !important;
}
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] {
    background: #FFFFFF !important;
    border: 1.5px dashed #DADCE0 !important;
    border-radius: 16px !important;
    color: var(--cs-text-muted) !important;
}
[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] > div,
[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] small {
    background: transparent !important;
    color: var(--cs-text-muted) !important;
}
[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button {
    background: #FFFFFF !important;
    color: #1A2B23 !important;
    border: 1.5px solid #1A2B23 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}
[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button svg,
[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button span,
[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button p {
    color: #1A2B23 !important;
    fill: #1A2B23 !important;
    stroke: #1A2B23 !important;
}
[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"]:hover {
    border-color: #B0B4BA !important;
    background: #FFFFFF !important;
}
[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button:hover {
    background: #FFFFFF !important;
    color: #1A2B23 !important;
    border-color: #1A2B23 !important;
}
/* Force readable upload controls over Streamlit dark defaults */
[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"],
[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] > div,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] {
    background-color: #FFFFFF !important;
    background: #FFFFFF !important;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 6px;
    border-bottom: 1px solid var(--cs-border-soft);
    padding-bottom: 4px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: var(--cs-bg-soft) !important;
    border-radius: 100px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    color: var(--cs-text-muted) !important;
    padding: 8px 18px !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: var(--cs-text) !important;
    color: var(--cs-bg) !important;
}
[data-testid="stSelectbox"] > div {
    border-radius: 100px !important;
    background: var(--cs-bg-soft) !important;
    border-color: var(--cs-border) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
}
[data-testid="stSelectbox"] label { display: none !important; }
[data-testid="stCaptionContainer"] p,
.stCaption { color: var(--cs-text-light) !important; font-style: italic; }
[data-testid="stDownloadButton"] button {
    background: var(--cs-text) !important;
    color: var(--cs-bg) !important;
    border-radius: 100px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
}
div[data-testid="stSuccess"] {
    background: #F0FDF4 !important;
    border: 1px solid #86EFAC !important;
    border-radius: 12px !important;
    color: #166534 !important;
}
div[data-testid="stWarning"] {
    background: #FFFBEB !important;
    border: 1px solid #FCD34D !important;
    border-radius: 12px !important;
    color: #92400E !important;
}
div[data-testid="stError"] {
    background: #FEF2F2 !important;
    border: 1px solid #FCA5A5 !important;
    border-radius: 12px !important;
    color: #991B1B !important;
}
div[data-testid="stInfo"] {
    background: #EFF6FF !important;
    border: 1px solid #BFDBFE !important;
    border-radius: 12px !important;
    color: #1E40AF !important;
}

/* ── DOCUMENTS TAB ── */
.cs-doc-intro {
    font-family: 'DM Sans', sans-serif;
}
.cs-doc-header {
    margin-bottom: 8px;
}
.cs-doc-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: var(--cs-text);
    letter-spacing: -0.4px;
    line-height: 1.2;
    margin: 0;
}
.cs-doc-desc {
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    line-height: 1.6;
    color: var(--cs-text-muted);
    margin: 0 0 4px 0;
    max-width: 680px;
}
.cs-doc-stored-grid {
    margin: 8px 0 0 0;
}
.cs-doc-sbar-body {
    display: flex;
    flex-direction: column;
    gap: 0;
}
.cs-doc-item-block {
    padding: 14px 0;
    border-bottom: 1px solid rgba(26, 43, 35, 0.08);
}
.cs-doc-item-block:first-child {
    padding-top: 0;
}
.cs-doc-item-block:last-child {
    padding-bottom: 0;
    border-bottom: none;
}
.cs-doc-item-name {
    font-family: 'DM Sans', sans-serif;
    font-size: 16px;
    font-weight: 600;
    color: var(--cs-text);
    line-height: 1.45;
}
.cs-doc-item-detail {
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    line-height: 1.55;
    color: var(--cs-text-muted);
    margin-top: 4px;
}
.cs-doc-demo-note {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid rgba(26, 43, 35, 0.08);
}
.cs-doc-discontinued-card,
.cs-doc-review-card {
    margin-top: 16px;
}
.cs-doc-item-block--discontinued .cs-doc-item-name {
    color: #6B7280;
    text-decoration: line-through;
    text-decoration-color: rgba(107, 114, 128, 0.55);
}
.cs-doc-review-intro {
    font-size: 13px;
    line-height: 1.5;
    color: var(--cs-text-muted);
    margin-bottom: 12px;
}
.cs-doc-review-flag {
    background: #FFFBEB;
    border: 1px solid #FCD34D;
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 10px;
}
.cs-doc-review-flag:last-child {
    margin-bottom: 0;
}
.cs-doc-review-flag-title {
    font-size: 15px;
    font-weight: 700;
    color: #92400E;
}
.cs-doc-review-flag-body {
    font-size: 13px;
    line-height: 1.5;
    color: #78350F;
    margin-top: 4px;
}
.cs-doc-condition-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    flex-wrap: wrap;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-doc-upload-panel) {
    border-radius: 24px !important;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(26, 43, 35, 0.06);
    border-color: var(--cs-border) !important;
    margin-bottom: 20px;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-doc-upload-panel) [data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-doc-upload-panel) [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] {
    background: #FFFFFF !important;
    border: 1.5px dashed #DADCE0 !important;
    border-radius: 18px !important;
    padding: 36px 24px !important;
    min-height: 140px;
    align-items: center;
    justify-content: center;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-doc-upload-panel) [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"]:hover {
    border-color: #B0B4BA !important;
    background: #FFFFFF !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-doc-upload-panel) [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button {
    background: #FFFFFF !important;
    color: #1A2B23 !important;
    border: 1.5px solid #1A2B23 !important;
    border-radius: 10px !important;
    box-shadow: none !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-doc-upload-panel) [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button svg,
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-doc-upload-panel) [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button span,
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-doc-upload-panel) [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button p {
    color: #1A2B23 !important;
    fill: #1A2B23 !important;
    stroke: #1A2B23 !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-doc-upload-panel) [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button:hover {
    background: #FFFFFF !important;
    color: #1A2B23 !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-doc-upload-panel) [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] small,
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-doc-upload-panel) [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] span {
    color: var(--cs-text-muted) !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-doc-upload-panel) [data-testid="stFileUploader"] label {
    display: none !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-doc-upload-panel) [data-testid="stButton"] button {
    background: var(--cs-accent) !important;
    color: #FFFFFF !important;
    border: none !important;
    width: 100%;
    padding: 14px 24px !important;
    font-size: 15px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 10px var(--cs-accent-shadow) !important;
    margin-top: 12px;
    border-radius: 100px !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-doc-upload-panel) [data-testid="stButton"] button:hover {
    background: var(--cs-accent-hover) !important;
    color: #FFFFFF !important;
}
.cs-plan-empty {
    font-size: 14px;
    color: var(--cs-text-muted);
    margin: 0;
    line-height: 1.5;
}

/* ── STORED OVERVIEW CARDS ── */
.cs-stored-card {
    background: var(--cs-surface);
    border-radius: 24px;
    padding: 24px 22px;
    border: 1px solid var(--cs-border);
    box-shadow: 0 2px 12px rgba(26, 43, 35, 0.05);
    height: 100%;
}
.cs-stored-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 18px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--cs-border-soft);
}
.cs-stored-icon {
    width: 28px; height: 28px;
    border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 700;
    flex-shrink: 0;
    line-height: 1;
}
.cs-stored-icon-meds {
    background: #E8F0FE;
    color: #2563EB;
    border: 1.5px solid #BFDBFE;
}
.cs-stored-icon-heart {
    background: #FDECEA;
    color: #C45C3E;
    border: 1.5px solid #F5C6C0;
    font-size: 15px;
}
.cs-stored-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: var(--cs-text);
    letter-spacing: -0.2px;
}
.cs-med-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 13px 0;
    border-bottom: 1px solid var(--cs-border-soft);
    gap: 12px;
}
.cs-med-row:last-child { border-bottom: none; }
.cs-med-left {
    display: flex;
    align-items: center;
    gap: 12px;
    min-width: 0;
}
.cs-med-dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    flex-shrink: 0;
}
.cs-med-dot-blue { background: #5B8DEF; box-shadow: 0 0 0 3px rgba(91, 141, 239, 0.18); }
.cs-med-dot-green { background: #4CAF7D; box-shadow: 0 0 0 3px rgba(76, 175, 125, 0.18); }
.cs-med-dot-purple { background: #9B7EDE; box-shadow: 0 0 0 3px rgba(155, 126, 222, 0.18); }
.cs-med-dot-orange { background: #E8945A; box-shadow: 0 0 0 3px rgba(232, 148, 90, 0.18); }
.cs-med-dot-teal { background: #3BA99C; box-shadow: 0 0 0 3px rgba(59, 169, 156, 0.18); }
.cs-med-dot-coral { background: #E07070; box-shadow: 0 0 0 3px rgba(224, 112, 112, 0.18); }
.cs-med-name {
    font-size: 14px;
    font-weight: 600;
    color: var(--cs-text);
    line-height: 1.3;
}
.cs-med-time {
    font-size: 13px;
    font-weight: 500;
    color: var(--cs-text-muted);
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
}
.cs-stored-empty .cs-med-name { color: var(--cs-text-muted); }
.cs-condition-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 13px 0;
    border-bottom: 1px solid var(--cs-border-soft);
    gap: 12px;
}
.cs-condition-row:last-child { border-bottom: none; }

/* ── MEDCAM TAB ── */
.cs-medcam-dose-card {
    background: #FFFFFF;
    border: 1px solid var(--cs-border);
    border-radius: 16px;
    padding: 16px 18px;
    margin-bottom: 10px;
    box-shadow: 0 1px 8px rgba(26, 43, 35, 0.04);
}
.cs-medcam-dose-name {
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 0.5px;
    color: #1D4ED8;
    margin-bottom: 6px;
}
.cs-medcam-dose-meta {
    font-size: 15px;
    font-weight: 700;
    color: var(--cs-text);
    margin-bottom: 4px;
}
.cs-medcam-dose-schedule {
    font-size: 13px;
    line-height: 1.5;
    color: var(--cs-text-muted);
    margin-bottom: 8px;
}
.cs-medcam-dose-status {
    font-size: 13px;
    font-weight: 600;
}
.cs-medcam-dose-status--muted {
    color: #7A7568;
}
.cs-medcam-dose-status--due {
    color: #B45309;
}
.cs-medcam-dose-status--taken {
    color: #166534;
}
.cs-medcam-dose-status--missed {
    color: #B91C1C;
}
.cs-medcam-dose-status--available {
    color: #1D4ED8;
}
.cs-medcam-upload-panel {
    margin-bottom: 4px;
}
.cs-medcam-upload-label {
    font-size: 14px;
    font-weight: 600;
    color: var(--cs-text);
    margin: 0 0 12px 0;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-medcam-upload-panel) {
    background: #FAFAF7 !important;
    border-color: var(--cs-border) !important;
    border-radius: 18px !important;
    padding-top: 16px !important;
    margin-bottom: 16px !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-medcam-upload-panel) [data-testid="stButton"] button {
    background: #FFFFFF !important;
    color: #1A2B23 !important;
    border: 1.5px solid #1A2B23 !important;
    border-radius: 999px !important;
    min-height: 46px !important;
    font-weight: 700 !important;
}
[data-testid="stColumn"]:has(.cs-medcam-dose-card) [data-testid="stButton"] button {
    background: #FFFFFF !important;
    color: #1A2B23 !important;
    border: 1.5px solid #D1D5DB !important;
    border-radius: 12px !important;
    min-height: 42px !important;
    font-weight: 700 !important;
}
[data-testid="stColumn"]:has(.cs-medcam-dose-card) [data-testid="stButton"] button:disabled {
    opacity: 1 !important;
    color: #57534E !important;
}
.cs-medcam-card {
    padding: 4px 2px 8px 2px;
}
.cs-medcam-header {
    margin-bottom: 8px;
}
.cs-medcam-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: var(--cs-text);
    letter-spacing: -0.4px;
    line-height: 1.2;
}
.cs-medcam-desc {
    font-size: 14px;
    line-height: 1.6;
    color: var(--cs-text-muted);
    margin: 0 0 22px 0;
    max-width: 620px;
}
.cs-medcam-result {
    border-radius: 16px;
    padding: 18px 20px;
    margin: 16px 0 4px 0;
    font-size: 14px;
    line-height: 1.55;
}
.cs-medcam-result strong {
    display: block;
    font-size: 15px;
    margin-bottom: 6px;
}
.cs-medcam-result-success {
    background: #ECFDF3;
    border: 1px solid #86EFAC;
    color: #14532D;
}
.cs-medcam-result-success strong {
    color: #166534;
}
.cs-medcam-result-fail {
    background: #FEF2F2;
    border: 1px solid #FCA5A5;
    color: #7F1D1D;
}
.cs-medcam-result-fail strong {
    color: #991B1B;
}
.cs-medcam-result-warn {
    background: #FFFBEB;
    border: 1px solid #FCD34D;
    color: #78350F;
}
.cs-medcam-result-warn strong {
    color: #92400E;
}
.cs-medcam-verdict-panel {
    border-radius: 16px;
    padding: 18px 20px;
    margin: 16px 0 4px 0;
    border: 1px solid var(--cs-border);
    background: var(--cs-surface);
}
.cs-medcam-verdict-panel--success {
    border-color: var(--cs-green-badge);
    background: var(--cs-green-soft);
}
.cs-medcam-verdict-panel--warn {
    border-color: var(--cs-yellow-soft);
    background: var(--cs-yellow-soft);
}
.cs-medcam-verdict-panel--fail {
    border-color: var(--cs-red-soft);
    background: var(--cs-red-soft);
}
.cs-medcam-verdict-header {
    margin-bottom: 14px;
}
.cs-medcam-verdict-title {
    font-size: 15px;
    font-weight: 700;
    color: var(--cs-text);
    margin-bottom: 4px;
}
.cs-medcam-verdict-panel--success .cs-medcam-verdict-title {
    color: var(--cs-green-text);
}
.cs-medcam-verdict-panel--warn .cs-medcam-verdict-title {
    color: var(--cs-yellow-text);
}
.cs-medcam-verdict-panel--fail .cs-medcam-verdict-title {
    color: var(--cs-red-text);
}
.cs-medcam-verdict-time {
    font-size: 12px;
    color: var(--cs-text-muted);
}
.cs-medcam-pill-rows {
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.cs-medcam-pill-row {
    --row-border: var(--cs-text-muted);
    --row-bg: var(--cs-bg-soft);
    --row-accent: var(--cs-text-muted);
    border-left: 3px solid var(--row-border);
    border-radius: 0 10px 10px 0;
    background: var(--row-bg);
    overflow: hidden;
}
.cs-medcam-pill-row--success {
    --row-border: var(--cs-green-text);
    --row-bg: var(--cs-green-soft);
    --row-accent: var(--cs-green-text);
}
.cs-medcam-pill-row--danger {
    --row-border: var(--cs-red-text);
    --row-bg: var(--cs-red-soft);
    --row-accent: var(--cs-red-text);
}
.cs-medcam-pill-row--warn {
    --row-border: var(--cs-yellow-text);
    --row-bg: var(--cs-yellow-soft);
    --row-accent: var(--cs-yellow-text);
}
.cs-medcam-pill-row--neutral {
    --row-border: var(--cs-text-muted);
    --row-bg: var(--cs-bg-soft);
    --row-accent: var(--cs-text-muted);
}
.cs-medcam-pill-row--info {
    --row-border: var(--cs-blue);
    --row-bg: var(--cs-blue-soft);
    --row-accent: var(--cs-blue);
}
.cs-medcam-pill-row--warning {
    --row-border: var(--cs-yellow-text);
    --row-bg: var(--cs-yellow-soft);
    --row-accent: var(--cs-yellow-text);
}
.cs-medcam-history {
    margin-top: 18px;
    padding-top: 16px;
    border-top: 1px solid var(--cs-border);
}
.cs-medcam-history-heading {
    font-size: 14px;
    font-weight: 700;
    color: var(--cs-text);
    margin-bottom: 10px;
}
.cs-medcam-history-item {
    padding: 10px 12px;
    border-radius: 10px;
    background: var(--cs-bg-soft);
    margin-bottom: 8px;
    border-left: 3px solid var(--cs-border);
}
.cs-medcam-history-when {
    font-size: 11px;
    color: var(--cs-text-muted);
    margin-bottom: 2px;
}
.cs-medcam-history-item-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--cs-text);
}
.cs-medcam-history-sub {
    font-size: 12px;
    color: var(--cs-text-muted);
}
.cs-medcam-history-empty {
    font-size: 13px;
    color: var(--cs-text-muted);
    margin: 0;
}
.cs-prn-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    padding: 10px 8px 4px;
    max-width: 320px;
    margin: 0 auto;
}
.cs-prn-chip {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    line-height: 1.35;
    padding: 6px 10px;
    border-radius: 999px;
    background: var(--cs-blue-soft);
    color: var(--cs-blue);
    border: 1px solid var(--cs-blue-soft);
}
.cs-prn-chip--wait {
    background: var(--cs-yellow-soft);
    color: var(--cs-yellow-text);
}
.cs-prn-chip--max {
    background: var(--cs-red-soft);
    color: var(--cs-red-text);
}
.cs-dose-card-prn-available {
    background: var(--cs-blue-soft);
}
.cs-dose-card-prn-available .cs-sbar-label,
.cs-dose-card-prn-available .cs-dose-card-status {
    color: var(--cs-blue);
}
.cs-dose-card-prn-wait {
    background: var(--cs-yellow-soft);
}
.cs-dose-card-prn-wait .cs-sbar-label,
.cs-dose-card-prn-wait .cs-dose-card-status {
    color: var(--cs-yellow-text);
}
.cs-dose-card-prn-max {
    background: var(--cs-red-soft);
}
.cs-dose-card-prn-max .cs-sbar-label,
.cs-dose-card-prn-max .cs-dose-card-status {
    color: var(--cs-red-text);
}
.cs-medcam-pill-row-summary {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 12px 14px 12px 16px;
    cursor: pointer;
    list-style: none;
}
.cs-medcam-pill-row-summary-inner {
    display: flex;
    align-items: center;
    gap: 12px;
    min-width: 0;
    flex: 1;
}
.cs-medcam-pill-row-ref-thumb {
    width: 52px;
    height: 52px;
    border-radius: 10px;
    object-fit: cover;
    flex-shrink: 0;
    border: 2px solid #FFFFFF;
    box-shadow: 0 2px 6px rgba(26, 43, 35, 0.1);
}
@media (max-width: 480px) {
    .cs-medcam-pill-row-ref-thumb {
        width: 44px;
        height: 44px;
        border-radius: 8px;
    }
    .cs-medcam-pill-row-summary {
        padding: 10px 12px 10px 14px;
        gap: 8px;
    }
    .cs-medcam-pill-row-summary-inner {
        gap: 10px;
    }
}
.cs-medcam-pill-row-summary::-webkit-details-marker {
    display: none;
}
.cs-medcam-pill-row-main {
    min-width: 0;
    flex: 1;
}
.cs-medcam-pill-row-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.45px;
    text-transform: uppercase;
    color: var(--row-accent);
    margin-bottom: 4px;
}
.cs-medcam-pill-row-verdict {
    font-size: 17px;
    font-weight: 700;
    line-height: 1.25;
    color: var(--row-accent);
}
.cs-medcam-pill-row-chevron {
    flex-shrink: 0;
    color: var(--row-accent);
    transition: transform 0.2s ease;
}
.cs-medcam-pill-row[open] .cs-medcam-pill-row-chevron {
    transform: rotate(180deg);
}
.cs-medcam-pill-row-detail {
    padding: 12px 16px 14px 16px;
    font-size: 13px;
    line-height: 1.55;
    color: var(--cs-text-muted);
    border-top: 1px solid rgba(0, 0, 0, 0.05);
    margin-left: 3px;
}
.cs-enroll-progress {
    font-size: 13px;
    font-weight: 600;
    color: var(--cs-text-muted);
    margin: 0 0 16px 0;
}
.cs-pill-reg-shell {
    margin-bottom: 8px;
}
.cs-pill-reg-progress {
    margin: 0;
    flex-shrink: 0;
    display: flex;
    align-items: center;
}
.cs-pill-reg-howto,
.cs-tab-howto {
    text-align: center;
    margin: 4px 0 22px 0;
}
.cs-pill-reg-howto-title,
.cs-tab-howto-title {
    font-size: 18px;
    font-weight: 700;
    color: #D97706;
    margin-bottom: 12px;
}
.cs-pill-reg-howto-pill,
.cs-tab-howto-pill {
    background: #FEF9C3;
    border: 1px solid #FDE047;
    border-radius: 999px;
    padding: 14px 26px;
    font-size: 14px;
    line-height: 1.55;
    color: #1A2B23;
    max-width: 920px;
    margin: 0 auto;
}
.cs-pill-reg-summary {
    display: flex;
    align-items: stretch;
    gap: 20px;
    margin: 0 0 18px 0;
}
.cs-pill-reg-active-plan {
    flex: 1;
    min-width: 0;
    background: #E3E9E1;
    border: 1px solid #C8D5C4;
    border-radius: 16px;
    padding: 18px 20px;
}
.cs-pill-reg-active-plan .cs-active-plan-label {
    color: #3D5A45;
    margin-bottom: 12px;
}
.cs-pill-reg-plan-list {
    list-style: none;
    margin: 0;
    padding: 0;
}
.cs-pill-reg-plan-list li {
    font-size: 14px;
    line-height: 1.5;
    color: var(--cs-text);
    padding: 7px 0;
}
.cs-pill-reg-plan-list li + li {
    border-top: 1px solid rgba(61, 90, 69, 0.12);
}
.cs-pill-reg-photo-guide-card {
    height: 100%;
    min-height: 220px;
    display: flex;
    align-items: center;
}
.cs-pill-reg-success-banner {
    background: #DCFCE7;
    border: 1px solid #86EFAC;
    border-radius: 12px;
    padding: 14px 18px;
    font-size: 14px;
    font-weight: 600;
    color: #166534;
    text-align: center;
    margin-top: 20px;
}
@media (max-width: 640px) {
    .cs-pill-reg-summary {
        flex-direction: column;
        align-items: center;
    }
    .cs-pill-reg-active-plan {
        width: 100%;
    }
}
.cs-pill-reg-progress .cs-reg-progress-ring {
    width: 128px;
    height: 128px;
    display: block;
}
.cs-empty-plan-guide {
    background: var(--cs-surface);
    border: 1px solid var(--cs-border);
    border-radius: 14px;
    padding: 20px 22px;
    margin: 8px 0 16px 0;
    color: var(--cs-text-muted);
    line-height: 1.55;
}
.cs-empty-plan-guide-lead {
    color: var(--cs-text);
    font-weight: 600;
    margin: 0 0 10px 0;
}
.cs-empty-plan-guide p {
    margin: 0 0 10px 0;
}
.cs-empty-plan-guide p:last-child {
    margin-bottom: 0;
}
.cs-pill-reg-header {
    display: flex;
    align-items: center;
    gap: 22px;
    margin-bottom: 22px;
}
.cs-reg-progress-ring {
    width: 128px;
    height: 128px;
    flex-shrink: 0;
}
.cs-reg-progress-num {
    font-size: 22px;
    font-weight: 800;
    fill: #1A2B23;
    font-family: 'DM Sans', sans-serif;
}
.cs-reg-progress-label {
    font-size: 10px;
    font-weight: 700;
    fill: #7A7568;
    font-family: 'DM Sans', sans-serif;
    letter-spacing: 0.4px;
    text-transform: uppercase;
}
.cs-pill-reg-header-copy {
    min-width: 0;
}
.cs-pill-reg-header-copy .cs-medcam-title {
    margin-bottom: 6px;
}
.cs-pill-reg-header-copy .cs-medcam-desc {
    margin: 0;
}
.cs-pill-grid-section-title {
    font-size: 18px;
    font-weight: 700;
    color: var(--cs-text);
    margin: 0 0 14px 0;
    letter-spacing: -0.3px;
}
.cs-pill-grid-card {
    border-radius: 16px;
    padding: 14px 14px 12px 14px;
    min-height: 118px;
    margin-bottom: 8px;
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.cs-pill-grid-card--pending {
    border: 2px dashed #B0B4BA;
    background: #FAFAFA;
}
.cs-pill-grid-card--done {
    border: 2px solid #4ADE80;
    background: #FFFFFF;
}
.cs-pill-grid-top {
    display: flex;
    justify-content: flex-end;
    min-height: 22px;
}
.cs-pill-grid-status {
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.4px;
    text-transform: uppercase;
    color: #7A7568;
}
.cs-pill-grid-status--done {
    color: #166534;
}
.cs-pill-grid-name {
    font-size: 15px;
    font-weight: 700;
    color: var(--cs-text);
    line-height: 1.3;
    margin-bottom: 4px;
}
.cs-pill-reg-instruction {
    font-size: 13px;
    font-weight: 600;
    line-height: 1.5;
    color: #1A2B23;
    margin: 4px 0 0 0;
}
.cs-pill-reg-note {
    font-size: 11px;
    font-weight: 500;
    line-height: 1.45;
    color: #57534E;
    background: #F5F5F4;
    border: 1px solid #E7E5E4;
    border-radius: 10px;
    padding: 6px 8px;
    margin: 6px 0 0 0;
}
.cs-pill-reg-alert {
    display: flex;
    align-items: flex-start;
    gap: 6px;
    font-size: 11px;
    font-weight: 600;
    line-height: 1.45;
    color: #B45309;
    background: #FFFBEB;
    border: 1px solid #FCD34D;
    border-radius: 10px;
    padding: 6px 8px;
    margin: 6px 0 0 0;
}
.cs-pill-grid-layout {
    display: flex;
    align-items: center;
    gap: 12px;
}
.cs-pill-grid-visual {
    flex-shrink: 0;
}
.cs-pill-grid-content {
    min-width: 0;
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.cs-pill-grid-thumb {
    width: 64px;
    height: 64px;
    border-radius: 14px;
    object-fit: cover;
    border: 2px solid #FFFFFF;
    box-shadow: 0 2px 8px rgba(26, 43, 35, 0.12);
}
.cs-pill-grid-thumb--empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 2px;
    background: #FFFFFF;
    border: 2px dashed #B0B4BA;
    box-shadow: none;
    color: #7A7568;
}
.cs-pill-grid-thumb--fallback {
    display: flex;
    align-items: center;
    justify-content: center;
    background: #DCFCE7;
    border: 2px solid #4ADE80;
    font-size: 28px;
}
.cs-pill-grid-empty-icon {
    font-size: 20px;
    font-weight: 700;
    line-height: 1;
    color: #9CA3AF;
}
.cs-pill-grid-empty-label {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.3px;
    text-transform: uppercase;
}
.cs-pill-grid-badges {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
    margin-top: 2px;
}
.cs-pill-grid-strength {
    font-size: 11px;
    font-weight: 700;
    color: #166534;
    background: #DCFCE7;
    border-radius: 999px;
    padding: 2px 8px;
}
.cs-sched-badge {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 999px;
    padding: 3px 9px 3px 6px;
    font-size: 12px;
    font-weight: 800;
    color: #1D4ED8;
    line-height: 1;
    white-space: nowrap;
}
.cs-sched-badge-icon {
    font-size: 11px;
    line-height: 1;
}
.cs-sched-badge-time {
    letter-spacing: -0.2px;
}
.cs-sched-badge-count {
    font-size: 10px;
    font-weight: 800;
    color: #1E40AF;
    background: #DBEAFE;
    border-radius: 999px;
    padding: 2px 5px;
    margin-right: 2px;
}
.cs-dose-fit {
    margin-top: 6px;
}
.cs-dose-fit-badge {
    display: inline-block;
    background: #DCFCE7;
    color: #166534;
    border: 1px solid #86EFAC;
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 12px;
    font-weight: 800;
    margin-bottom: 4px;
}
.cs-dose-fit-equation {
    font-size: 12px;
    font-weight: 600;
    color: #1A2B23;
    line-height: 1.45;
    margin-top: 2px;
}
.cs-dose-fit-note {
    font-size: 11px;
    font-weight: 500;
    line-height: 1.45;
    color: #57534E;
    background: #F5F5F4;
    border: 1px solid #E7E5E4;
    border-radius: 10px;
    padding: 6px 8px;
    margin-top: 6px;
}
.cs-dose-fit-warning {
    font-size: 11px;
    font-weight: 600;
    line-height: 1.45;
    color: #B45309;
    background: #FFFBEB;
    border: 1px solid #FCD34D;
    border-radius: 10px;
    padding: 6px 8px;
    margin-top: 6px;
}
.cs-dose-card-badges {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}
.cs-dose-card-clock {
    font-size: 13px;
    font-weight: 700;
    color: var(--cs-text-muted);
}
.cs-pill-grid-meta {
    font-size: 12px;
    color: var(--cs-text-muted);
}
.cs-pill-grid-hint {
    font-size: 12px;
    color: var(--cs-text-light);
    margin-top: auto;
    padding-top: 4px;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-pill-grid-card) {
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    background: transparent !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-pill-grid-card) [data-testid="stButton"] {
    margin-top: -2px;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-pill-grid-card) [data-testid="stButton"] button {
    background: #FFFFFF !important;
    color: #1A2B23 !important;
    border: 1.5px solid #D1D5DB !important;
    border-radius: 12px !important;
    min-height: 40px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 13px !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-pill-grid-card--done) [data-testid="stButton"] button {
    border-color: #D1D5DB !important;
    color: #1A2B23 !important;
}
.cs-enroll-card {
    background: #FAFAF7;
    border: 1px solid var(--cs-border);
    border-radius: 16px;
    padding: 16px 18px;
    margin: 0 0 14px 0;
}
.cs-enroll-card-title {
    font-size: 16px;
    font-weight: 700;
    color: var(--cs-text);
    margin-bottom: 6px;
}
.cs-enroll-card-hint,
.cs-enroll-upload-label {
    font-size: 13px;
    line-height: 1.5;
    color: var(--cs-text-muted);
    margin: 0 0 10px 0;
}
.cs-pill-reg-photo-guide {
    margin: 0 0 24px 0;
}
.cs-pill-reg-photo-guide-shell {
    margin: 0 0 24px 0;
}
.cs-pill-reg-photo-guide-shell [data-testid="stHorizontalBlock"] {
    align-items: stretch;
}
.cs-pill-reg-photo-guide-shell [data-testid="column"] {
    display: flex;
    flex-direction: column;
}
.cs-pill-reg-photo-guide-card {
    flex: 1;
    display: flex;
    align-items: center;
    height: 100%;
}
.cs-pill-reg-photo-instruction {
    margin: 0;
    width: 100%;
    padding: 18px 20px;
    background: #F5F3FF;
    border: 1px solid #C4B5FD;
    border-radius: 16px;
    font-size: 14px;
    line-height: 1.65;
    color: #312E81;
}
.cs-pill-reg-example-wrap {
    flex: 1;
    margin: 0;
    height: 100%;
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid #D1D5DB;
    box-shadow: 0 2px 12px rgba(26, 43, 35, 0.06);
    background: #E5E7EB;
    display: flex;
    align-items: center;
}
.cs-pill-reg-example-wrap [data-testid="stImage"] {
    margin: 0;
    width: 100%;
}
.cs-pill-reg-example-wrap img {
    display: block;
    width: 100%;
    height: auto;
}
@media (max-width: 768px) {
    .cs-pill-reg-photo-guide-shell [data-testid="stHorizontalBlock"] {
        flex-direction: column;
    }
    .cs-pill-reg-photo-guide-card,
    .cs-pill-reg-example-wrap {
        min-height: 0;
    }
}
.cs-pill-reg-upload-shell {
    margin: 0 0 18px 0;
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid #D1D5DB;
    box-shadow: 0 2px 12px rgba(26, 43, 35, 0.06);
}
.cs-pill-reg-upload-shell [data-testid="column"] {
    background: #E5E7EB;
    padding: 16px 14px 14px;
    min-height: 260px;
}
.cs-pill-reg-upload-shell [data-testid="column"]:first-child {
    border-right: 2px solid #FFFFFF;
}
.cs-pill-reg-upload-panel-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    text-align: center;
    color: #4B5563;
    margin: 0 0 14px 0;
}
.cs-pill-reg-upload-shell [data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
.cs-pill-reg-upload-shell [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] {
    background: rgba(255, 255, 255, 0.72) !important;
    border: 1.5px dashed #9CA3AF !important;
    border-radius: 12px !important;
    min-height: 150px !important;
    align-items: center;
    justify-content: center;
}
.cs-pill-reg-upload-shell [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"]:hover {
    border-color: #6B7280 !important;
    background: rgba(255, 255, 255, 0.92) !important;
}
.cs-pill-reg-upload-shell [data-testid="stImage"] {
    margin-top: 10px;
}
.cs-pill-reg-example-card {
    border: 1px solid var(--cs-border);
    border-radius: 16px;
    overflow: hidden;
    background: #FFFFFF;
    box-shadow: 0 2px 12px rgba(26, 43, 35, 0.06);
}
.cs-pill-reg-example-img {
    display: block;
    width: 100%;
    height: auto;
}
.cs-pill-reg-upload-slot {
    margin-bottom: 8px;
}
.cs-pill-reg-upload-slot-label {
    font-size: 14px;
    font-weight: 700;
    color: var(--cs-text);
    margin-bottom: 4px;
}
.cs-pill-reg-upload-slot-hint {
    font-size: 12px;
    line-height: 1.45;
    color: var(--cs-text-muted);
    margin-bottom: 8px;
}
.cs-ref-meta {
    font-size: 13px;
    color: var(--cs-text-muted);
    margin-top: 2px;
}
.cs-ref-update-note {
    font-size: 12px;
    color: var(--cs-text-light);
    margin: 8px 0 0 0;
}
.cs-registered-panel {
    background: linear-gradient(145deg, #EEF2FF 0%, #FCE7F3 45%, #ECFDF5 100%);
    border: 1px solid #C7D2FE;
    border-radius: 20px;
    padding: 18px 20px 8px 20px;
    margin: 18px 0 12px 0;
}
.cs-registered-panel-title {
    font-size: 18px;
    font-weight: 700;
    color: var(--cs-text);
    margin-bottom: 6px;
}
.cs-registered-panel-desc {
    font-size: 13px;
    line-height: 1.5;
    color: var(--cs-text-muted);
    margin: 0 0 8px 0;
}
.cs-registered-pill-card {
    background: #FFFFFF;
    border: 1px solid var(--cs-border);
    border-left: 5px solid #5B8DEF;
    border-radius: 16px;
    padding: 14px 16px;
    margin: 0 0 12px 0;
    box-shadow: 0 2px 10px rgba(26, 43, 35, 0.05);
}
.cs-registered-pill-layout {
    display: flex;
    align-items: center;
    gap: 12px;
}
.cs-registered-pill-body {
    min-width: 0;
    flex: 1;
}
.cs-registered-pill-thumb {
    width: 52px;
    height: 52px;
    border-radius: 12px;
    object-fit: cover;
    flex-shrink: 0;
    border: 2px solid #FFFFFF;
    box-shadow: 0 2px 8px rgba(26, 43, 35, 0.1);
}
.cs-registered-pill-thumb--fallback {
    display: flex;
    align-items: center;
    justify-content: center;
    background: #EEF2FF;
    font-size: 24px;
}
.cs-registered-pill-name {
    font-size: 15px;
    font-weight: 700;
    color: var(--cs-text);
    margin-bottom: 4px;
}
.cs-registered-pill-meta {
    font-size: 13px;
    color: var(--cs-text-muted);
    margin-bottom: 6px;
    line-height: 1.45;
}
.cs-registered-pill-badges {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
}
.cs-plan-dose-badge {
    display: inline-block;
    background: #E8F5EC;
    color: #2D6A4F;
    border: 1px solid #A7DCC2;
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
}
.cs-schedule-section {
    background: linear-gradient(145deg, #F0F7FA 0%, #EEF2FF 100%);
    border: 1px solid #B8D4E3;
    border-radius: 18px;
    padding: 16px 18px;
    margin: 18px 0 8px 0;
}
.cs-schedule-title {
    font-size: 17px;
    font-weight: 700;
    color: var(--cs-text);
    margin-bottom: 4px;
}
.cs-schedule-desc {
    font-size: 13px;
    line-height: 1.45;
    color: var(--cs-text-muted);
    margin: 0;
}
.cs-schedule-checklist-title {
    font-size: 15px;
    font-weight: 700;
    color: var(--cs-text);
    margin: 8px 0 10px 0;
}
.cs-clock-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 16px 24px;
    justify-content: center;
    margin: 4px 0 18px 0;
    font-size: 13px;
    color: var(--cs-text);
}
.cs-clock-legend-item {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-weight: 600;
}
.cs-clock-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    border: 2px solid #FFFFFF;
    box-shadow: 0 0 0 1px #CBD5E1;
    flex-shrink: 0;
}
.cs-dose-status-square {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 2px;
    margin-right: 8px;
    vertical-align: -1px;
    flex-shrink: 0;
    box-shadow: 0 0 0 1px rgba(15, 23, 42, 0.08);
}
.cs-conditions-card {
    background: linear-gradient(145deg, #F0F7FA 0%, #E8F2F8 100%);
    border: 1px solid #B8D4E3;
    border-radius: 24px;
    padding: 26px 24px;
    margin-top: 20px;
    box-shadow: 0 2px 12px rgba(47, 111, 142, 0.08);
}
.cs-conditions-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 18px;
    padding-bottom: 14px;
    border-bottom: 1px solid rgba(47, 111, 142, 0.15);
}
.cs-conditions-icon {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: #D6EAF5;
    color: #2F6F8E;
    border: 1px solid #A8CCE0;
    flex-shrink: 0;
}
.cs-conditions-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #1E4D63;
    letter-spacing: -0.4px;
    line-height: 1.2;
}
.cs-conditions-card .cs-condition-name {
    color: #1E4D63;
}
.cs-conditions-card .cs-condition-date {
    color: #5A8FA8;
}
.cs-conditions-card .cs-condition-row {
    border-bottom-color: rgba(47, 111, 142, 0.12);
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-medcam-card) {
    border-radius: 24px !important;
    overflow: hidden;
    box-shadow: 0 2px 16px rgba(26, 43, 35, 0.06);
    border-color: var(--cs-border) !important;
    margin-bottom: 8px;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-medcam-card) .cs-medcam-card {
    border: none;
    box-shadow: none;
    margin-bottom: 0;
    padding-bottom: 8px;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-medcam-card) [data-testid="stButton"] button {
    margin-bottom: 4px;
}
.cs-medcam-active-plan {
    background: #E3E9E1;
    border: 1px solid #C8D5C4;
    border-radius: 16px;
    padding: 18px 20px;
    margin-bottom: 20px;
}
.cs-medcam-active-plan .cs-active-plan-label {
    color: #3D5A45;
    margin-bottom: 12px;
}
.cs-plan-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 0;
    gap: 12px;
}
.cs-plan-row + .cs-plan-row {
    border-top: 1px solid rgba(61, 90, 69, 0.12);
}
.cs-plan-med {
    font-size: 14px;
    font-weight: 500;
    color: var(--cs-text);
}
.cs-plan-time {
    font-size: 13px;
    font-weight: 500;
    color: var(--cs-text-muted);
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
}
.cs-medcam-upload-label {
    font-size: 14px;
    line-height: 1.6;
    font-family: 'DM Sans', sans-serif;
    color: var(--cs-text-muted);
    margin: 0 0 10px 0;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-medcam-card) [data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-medcam-card) [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] {
    background: #FFFFFF !important;
    border: 1.5px dashed #DADCE0 !important;
    border-radius: 16px !important;
    padding: 28px 20px !important;
    min-height: 100px;
    color: var(--cs-text-muted) !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-medcam-card) [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] small,
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-medcam-card) [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] span {
    color: var(--cs-text-muted) !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-medcam-card) [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button {
    background: #FFFFFF !important;
    color: #1A2B23 !important;
    border: 1.5px solid #1A2B23 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-medcam-card) [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button svg,
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-medcam-card) [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button span,
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-medcam-card) [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button p {
    color: #1A2B23 !important;
    fill: #1A2B23 !important;
    stroke: #1A2B23 !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-medcam-card) [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button:hover {
    background: #FFFFFF !important;
    color: #1A2B23 !important;
    border-color: #1A2B23 !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-medcam-card) [data-testid="stFileUploader"] label {
    display: none !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-medcam-card) [data-testid="stButton"] button {
    background: var(--cs-text) !important;
    color: var(--cs-bg) !important;
    border: none !important;
    border-radius: 14px !important;
    box-shadow: none !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-medcam-card) [data-testid="stButton"] button:hover {
    background: #0F1A15 !important;
    color: var(--cs-bg) !important;
}
.cs-plan-left {
    min-width: 0;
}
.cs-plan-dosage {
    font-size: 12px;
    color: var(--cs-text-muted);
    margin-top: 2px;
}

/* ── UPLOAD READABILITY (final override) ── */
[data-testid="stAppViewContainer"],
[data-testid="stFileUploader"] {
    color-scheme: light !important;
}
[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] {
    background-color: #FFFFFF !important;
    background: #FFFFFF !important;
    border: 1.5px dashed #DADCE0 !important;
    border-radius: 14px !important;
    color: #7A7568 !important;
}
[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] > div {
    background: transparent !important;
    color: #7A7568 !important;
}
[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] small,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] {
    color: #7A7568 !important;
    -webkit-text-fill-color: #7A7568 !important;
    opacity: 1 !important;
}
[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button {
    background-color: #FFFFFF !important;
    background: #FFFFFF !important;
    color: #1A2B23 !important;
    -webkit-text-fill-color: #1A2B23 !important;
    border: 2px solid #1A2B23 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}
[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button:hover {
    background-color: #FFFFFF !important;
    background: #FFFFFF !important;
    color: #1A2B23 !important;
    border-color: #1A2B23 !important;
}
[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button span,
[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button p,
[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button svg,
[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button svg path {
    color: #1A2B23 !important;
    fill: #1A2B23 !important;
    stroke: #1A2B23 !important;
    -webkit-text-fill-color: #1A2B23 !important;
}

/* ── MY RESULTS ── */
.cs-mr-intro .cs-report-title {
    font-size: 28px !important;
    letter-spacing: -0.6px !important;
}
.cs-mr-disclaimer {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    background: #FFF9E8;
    border: 1px solid var(--cs-home-amber);
    border-radius: 14px;
    padding: 14px 16px;
    margin: 18px 0 22px 0;
    color: var(--cs-text);
    font-size: 14px;
    line-height: 1.55;
}
.cs-mr-disclaimer-icon {
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: var(--cs-home-amber);
    color: #FFFFFF;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 1px;
}
.cs-mr-upload-shell {
    margin-bottom: 8px;
}
.cs-mr-upload-heading {
    font-size: 16px;
    font-weight: 700;
    color: var(--cs-text);
    margin-bottom: 4px;
}
.cs-mr-upload-sub {
    font-size: 14px;
    color: var(--cs-home-grey);
    margin-bottom: 14px;
}
.cs-mr-empty {
    text-align: center;
    padding: 28px 18px 8px 18px;
    color: var(--cs-home-grey);
    font-size: 15px;
    line-height: 1.6;
}
.cs-mr-latest-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--cs-home-grey);
    margin: 24px 0 12px 0;
}
.cs-mr-card {
    background: var(--cs-surface);
    border: 1px solid var(--cs-border);
    border-radius: 18px;
    padding: 18px 18px 16px 18px;
    box-shadow: 0 2px 14px rgba(45, 63, 107, 0.05);
    margin-bottom: 18px;
}
.cs-mr-card-header {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: flex-start;
    margin-bottom: 16px;
}
.cs-mr-card-title-wrap {
    display: flex;
    gap: 10px;
    align-items: flex-start;
}
.cs-mr-card-icon {
    font-size: 18px;
    line-height: 1;
    margin-top: 2px;
}
.cs-mr-card-title {
    font-size: 17px;
    font-weight: 700;
    color: var(--cs-text);
    margin-bottom: 4px;
}
.cs-mr-card-meta {
    font-size: 13px;
    color: var(--cs-home-grey);
}
.cs-mr-review-badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 10px;
    border-radius: 999px;
    background: #E8F0FE;
    color: #2D3F6B;
    font-size: 12px;
    font-weight: 700;
    white-space: nowrap;
}
.cs-mr-results-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.1px;
    text-transform: uppercase;
    color: var(--cs-home-grey);
    margin-bottom: 8px;
}
.cs-mr-result-row {
    display: grid;
    grid-template-columns: 1.4fr 1fr auto;
    gap: 10px;
    align-items: center;
    padding: 10px 0;
    border-top: 1px solid var(--cs-border-soft);
}
.cs-mr-result-row:first-of-type {
    border-top: none;
    padding-top: 0;
}
.cs-mr-result-name {
    font-size: 14px;
    font-weight: 600;
    color: var(--cs-text);
}
.cs-mr-result-value {
    font-size: 14px;
    color: var(--cs-text);
}
.cs-mr-result-flag {
    justify-self: end;
}
.cs-mr-flag {
    display: inline-flex;
    align-items: center;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    white-space: nowrap;
}
.cs-mr-flag--normal {
    background: #EAF4E8;
    color: #3B6D11;
}
.cs-mr-flag--low {
    background: #FFF0E0;
    color: #854F0B;
}
.cs-mr-flag--high {
    background: #FDEAEA;
    color: #A32D2D;
}
.cs-mr-flag--na {
    background: #F3F4F6;
    color: #4B5563;
}
.cs-mr-result-ref {
    font-size: 11px;
    font-weight: 500;
    color: var(--cs-home-grey);
    margin-top: 2px;
}
.cs-mr-empty-section {
    font-size: 13px;
    color: var(--cs-home-grey);
    padding: 8px 0 4px 0;
}
.cs-mr-key-findings {
    margin-bottom: 14px;
    padding: 14px 16px;
    border-radius: 14px;
    background: #F8FAFF;
    border: 1px solid #D8E2F8;
}
.cs-mr-key-findings-title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.1px;
    text-transform: uppercase;
    color: var(--cs-home-navy);
    margin-bottom: 8px;
}
.cs-mr-key-findings-list {
    margin: 0;
    padding-left: 18px;
    color: var(--cs-text);
    font-size: 14px;
    line-height: 1.55;
}
.cs-mr-key-findings-list li {
    margin-bottom: 6px;
}
.cs-mr-key-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--cs-home-grey);
    margin-right: 6px;
}
.cs-mr-key-meta {
    color: var(--cs-home-grey);
    font-size: 13px;
}
.cs-mr-urgent-card {
    margin-bottom: 14px;
    padding: 14px 16px;
    border-radius: 14px;
    background: linear-gradient(145deg, #FFF1F2 0%, #FFE4E6 100%);
    border: 1.5px solid #FDA4AF;
}
.cs-mr-urgent-title {
    font-size: 14px;
    font-weight: 800;
    color: #BE123C;
    margin-bottom: 8px;
}
.cs-mr-urgent-card p,
.cs-mr-urgent-card li {
    margin: 0 0 6px 0;
    font-size: 14px;
    line-height: 1.6;
    color: #881337;
}
.cs-mr-urgent-card ul {
    margin: 0;
    padding-left: 18px;
}
.cs-mr-limitations {
    margin-bottom: 12px;
    padding: 10px 12px;
    border-radius: 12px;
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    font-size: 13px;
    line-height: 1.55;
    color: #92400E;
}
.cs-mr-limitations p {
    margin: 0 0 6px 0;
}
.cs-mr-meaning-card {
    margin-top: 16px;
    padding: 16px;
    border: 1px solid var(--cs-border);
    border-radius: 14px;
    background: #FCFBFA;
}
.cs-mr-meaning-title {
    font-size: 15px;
    font-weight: 700;
    color: var(--cs-home-navy);
    margin-bottom: 8px;
}
.cs-mr-meaning-text {
    margin: 0;
    font-size: 14px;
    line-height: 1.65;
    color: var(--cs-text);
}
.cs-mr-meaning-card--overview {
    background: #F7F9FD;
    border-color: #C9D3EA;
}
.cs-mr-section-label {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--cs-home-navy);
    margin-bottom: 10px;
}
.cs-mr-trend-section {
    margin-top: 14px;
}
.cs-mr-trend-card {
    border: 1px solid #D8C4F0;
    border-left: 4px solid #7B4BB9;
    border-radius: 12px;
    background: #FBF7FF;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.cs-mr-trend-badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #5E3A87;
    margin-bottom: 6px;
}
.cs-mr-trend-title {
    font-size: 15px;
    font-weight: 700;
    color: var(--cs-home-navy);
    margin-bottom: 6px;
}
.cs-mr-trend-summary,
.cs-mr-trend-prior {
    margin: 0;
    font-size: 14px;
    line-height: 1.6;
    color: var(--cs-text);
}
.cs-mr-trend-prior {
    margin-top: 8px;
    font-size: 13px;
    color: var(--cs-home-grey);
}
.cs-mr-trend-prior-label {
    font-weight: 700;
    color: #5E3A87;
}
.cs-mr-groups-section {
    margin-top: 14px;
}
.cs-mr-groups-section > .cs-mr-meaning-title {
    margin-bottom: 12px;
}
.cs-mr-group-card {
    border: 1px solid var(--cs-border);
    border-radius: 14px;
    background: #FCFBFA;
    padding: 14px 16px;
    margin-bottom: 12px;
}
.cs-mr-group-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 8px;
}
.cs-mr-group-title {
    font-size: 16px;
    font-weight: 700;
    color: var(--cs-home-navy);
}
.cs-mr-urgency {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 999px;
    white-space: nowrap;
}
.cs-mr-urgency-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.cs-mr-urgency--soon {
    background: #FFF1E8;
    color: #9A4A12;
}
.cs-mr-urgency--soon .cs-mr-urgency-dot {
    background: #E07A2D;
}
.cs-mr-urgency--visit {
    background: #EEF6F1;
    color: #2F6B4F;
}
.cs-mr-urgency--visit .cs-mr-urgency-dot {
    background: #4F9D7A;
}
.cs-mr-group-summary {
    margin: 0 0 10px 0;
    font-size: 14px;
    line-height: 1.65;
    color: var(--cs-text);
}
.cs-mr-group-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 10px;
}
.cs-mr-group-chip {
    font-size: 12px;
    font-weight: 600;
    color: var(--cs-home-navy);
    background: #EEF2FA;
    border-radius: 999px;
    padding: 4px 10px;
}
.cs-mr-group-test {
    border-top: 1px solid var(--cs-border-soft);
    padding-top: 10px;
    margin-top: 10px;
}
.cs-mr-group-test:first-of-type {
    border-top: none;
    padding-top: 0;
    margin-top: 0;
}
.cs-mr-group-test-name {
    font-size: 14px;
    font-weight: 700;
    color: var(--cs-home-navy);
    margin-bottom: 4px;
}
.cs-mr-group-test-value {
    font-size: 13px;
    color: var(--cs-text);
    margin-bottom: 6px;
}
.cs-mr-group-test-ref {
    color: var(--cs-home-grey);
}
.cs-mr-group-test-measures,
.cs-mr-group-test-suggests,
.cs-mr-group-test-define {
    margin: 0 0 6px 0;
    font-size: 13px;
    line-height: 1.6;
    color: var(--cs-text);
}
.cs-mr-no-abnormal-note {
    margin-top: 14px;
    padding: 14px 16px;
    border-radius: 12px;
    border: 1px solid #B9D8C8;
    background: #F2FAF6;
}
.cs-mr-no-abnormal-label {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #2F6B4F;
    margin-bottom: 6px;
}
.cs-mr-no-abnormal-note p {
    margin: 0;
    font-size: 14px;
    line-height: 1.6;
    color: var(--cs-text);
}
.cs-mr-question-body {
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-width: 0;
}
.cs-mr-question-meta {
    font-size: 12px;
    font-weight: 600;
    color: var(--cs-home-grey);
}
.cs-mr-questions-card {
    margin-top: 14px;
    border: 1px solid #C9D3EA;
    border-radius: 14px;
    overflow: hidden;
}
.cs-mr-questions-header {
    background: var(--cs-home-navy);
    padding: 12px 14px;
}
.cs-mr-questions-title {
    font-size: 14px;
    font-weight: 700;
    color: #FFFFFF;
}
.cs-mr-questions-list {
    list-style: none;
    margin: 0;
    padding: 0;
    background: var(--cs-surface);
}
.cs-mr-questions-list li {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    padding: 14px 14px;
    border-top: 1px solid var(--cs-border-soft);
    font-size: 14px;
    line-height: 1.55;
    color: var(--cs-text);
}
.cs-mr-questions-list li:first-child {
    border-top: none;
}
.cs-mr-question-num {
    font-weight: 700;
    color: var(--cs-home-navy);
    min-width: 16px;
}
.cs-mr-actions-note {
    font-size: 13px;
    color: var(--cs-home-grey);
    margin: 8px 0 0 0;
    text-align: center;
}
.cs-mr-handover-questions {
    margin-top: 18px;
    padding: 16px 18px;
    border-radius: 14px;
    border: 1px solid #C9D3EA;
    background: #F7F9FD;
}
.cs-mr-handover-questions-label {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--cs-home-navy);
    margin-bottom: 10px;
}
.cs-mr-handover-questions-list {
    margin: 0;
    padding-left: 18px;
    color: var(--cs-text);
    font-size: 14px;
    line-height: 1.6;
}
.cs-mr-handover-q-source {
    display: block;
    font-size: 11px;
    font-weight: 700;
    color: var(--cs-home-grey);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 2px;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-mr-upload-shell) {
    border-radius: 18px !important;
    border-style: dashed !important;
    border-color: #CFC7BC !important;
    background: #FCFBFA !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-mr-upload-shell) [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-mr-upload-shell) [data-testid="stButton"] button {
    background: var(--cs-accent) !important;
    color: #FFFFFF !important;
    border: none !important;
    width: 100%;
    padding: 14px 24px !important;
    font-size: 15px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 10px var(--cs-accent-shadow) !important;
    margin-top: 12px;
    border-radius: 12px !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-mr-upload-shell) [data-testid="stButton"] button:hover {
    background: var(--cs-accent-hover) !important;
    color: #FFFFFF !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-mr-actions) [data-testid="stButton"] button[kind="secondary"] {
    background: var(--cs-surface) !important;
    color: var(--cs-text) !important;
    border: 1px solid var(--cs-border) !important;
    border-radius: 12px !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-mr-actions) [data-testid="stDownloadButton"] button,
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-mr-actions) [data-testid="stButton"] button[kind="primary"] {
    background: var(--cs-accent) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 10px var(--cs-accent-shadow) !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-mr-actions) [data-testid="stDownloadButton"] button:hover,
[data-testid="stVerticalBlockBorderWrapper"]:has(.cs-mr-actions) [data-testid="stButton"] button[kind="primary"]:hover {
    background: var(--cs-accent-hover) !important;
    color: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)

if "why_this_exists_seen" not in st.session_state:
    st.session_state.why_this_exists_seen = False

if not st.session_state.why_this_exists_seen:
    render_why_this_exists_screen()
    st.stop()

enter_from_homepage = st.session_state.pop("careshield_enter_from_homepage", False)

if enter_from_homepage:
    st.session_state.careshield_skip_boot_reveal = True
    md_html("""
    <style id="cs-enter-from-home-style">
    [data-testid="stAppViewContainer"] .main {
        opacity: 1 !important;
        visibility: visible !important;
    }
    #cs-boot-splash { display: none !important; }
    </style>
    """)
    render_careshield_enter_from_homepage()

if not enter_from_homepage and not st.session_state.get("careshield_skip_boot_reveal"):
    md_html("""
    <div id="cs-boot-splash" aria-live="polite" aria-busy="true">
      <div class="cs-boot-spinner" aria-hidden="true"></div>
      <div class="cs-boot-label">Loading CareShield…</div>
    </div>
    """)

# Legacy widget/session key from an older Responsible AI button — clear to avoid crashes.
st.session_state.pop("open_responsible_ai_dialog", None)

md_html('<div class="cs-main-app-marker" aria-hidden="true"></div>')
if not enter_from_homepage:
    render_careshield_boot_instant_ready()

# ── TIMEZONE (detect before tabs so dose status uses local wall-clock time) ───
if "careshield_user_timezone" not in st.session_state:
    st.session_state.careshield_user_timezone = "UTC"

if not st.session_state.get("careshield_timezone_checked"):
    previous_tz = st.session_state.careshield_user_timezone
    detected_tz = st_javascript("""await (async () => {
        return Intl.DateTimeFormat().resolvedOptions().timeZone;
    })().then(returnValue => returnValue)""")
    if detected_tz:
        st.session_state.careshield_user_timezone = detected_tz
        st.session_state.careshield_timezone_checked = True
        if detected_tz != previous_tz:
            st.rerun()

user_timezone = st.session_state.get("careshield_user_timezone") or "UTC"
try:
    tz = ZoneInfo(user_timezone)
    current_time = datetime.now(tz)
    current_time_str = current_time.strftime("%A, %B %d, %Y at %I:%M %p (%Z)")
except Exception:
    current_time_str = "Unknown (timezone not detected yet)"

# ── HEADER ──────────────────────────────────────────────────────────────────
header_logo, header_caregiver, header_patient = st.columns([1.3, 1, 1])
with header_logo:
    md_html(build_careshield_logo_html())
    render_homepage_back_button()
with header_caregiver:
    selected_caregiver_id, selected_caregiver = render_caregiver_profile_switcher()
with header_patient:
    selected_patient_id = render_patient_selector()

if st.session_state.pop("show_responsible_ai_dialog", False):
    responsible_ai_safety_dialog()

st.markdown('<div style="border-bottom:1px solid #E8E4DA;margin-bottom:20px;"></div>', unsafe_allow_html=True)

try:
    render_how_to_use_bar()

    warm_patient_profile_cache(selected_patient_id)

    # ── TABS ────────────────────────────────────────────────────────────────────
    tab_docs, tab_report, tab_pill, tab_medcam, tab_handover, tab_results = st.tabs(
        ["Documents", "Report & Ask", "Pill Registration", "MedCam", "Handover", "My Results"]
    )
except Exception as main_ui_exc:
    logging.getLogger("careshield").exception("CareShield failed to render main navigation")
    st.error("CareShield could not load the main screen. Please refresh the page.")
    if os.getenv("CARESHIELD_DEBUG"):
        st.exception(main_ui_exc)
    st.stop()

# ════════════════════════════════════════════════════════════════
# TAB — DOCUMENTS
# ════════════════════════════════════════════════════════════════
with tab_docs:
    if "pdf_uploader_key" not in st.session_state:
        st.session_state.pdf_uploader_key = 0

    if st.session_state.get("doc_process_flash"):
        st.success(st.session_state.pop("doc_process_flash"))

    render_tab_story_section("Documents", *TAB_PROBLEM_SOLUTION["documents"])
    md_html(build_documents_how_to_use_html())

    with st.container(border=True):
        md_html('<div class="cs-doc-upload-panel"></div>')

        active_patient_name = get_patient_display_name(get_current_patient_id())
        st.warning(
            "Please upload a document that belongs to and matches the active profile "
            f"({active_patient_name}). Documents for another patient will not be saved."
        )

        if st.session_state.get("doc_patient_mismatch_warning"):
            st.warning(st.session_state["doc_patient_mismatch_warning"])

        uploaded_pdf = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            key=f"pdf_uploader_{st.session_state.pdf_uploader_key}",
            label_visibility="collapsed",
        )

        if uploaded_pdf:
            md_html(build_upload_confirmation_html(
                uploaded_pdf.name,
                size_bytes=uploaded_pdf.size,
                kind="PDF",
            ))
            upload_sig = f"{uploaded_pdf.name}:{uploaded_pdf.size}"
            if st.session_state.get("doc_upload_sig") != upload_sig:
                st.session_state.doc_upload_sig = upload_sig
                set_careshield_active_tab("documents")
        else:
            st.session_state.pop("doc_upload_sig", None)

        if st.button(
            "Analyse document",
            key="process_pdf_button",
            use_container_width=True,
            type="primary",
        ):
            if uploaded_pdf is None:
                st.warning("Please upload a PDF first.")
            else:
                set_careshield_active_tab("documents")
                md_html(build_loading_banner_html("Analysing document and extracting medication plan..."))
                patient_id = get_current_patient_id()
                try:
                    upload_result = process_discharge_document_upload(
                        uploaded_pdf,
                        patient_id=patient_id,
                    )
                except Exception as exc:
                    _documents_logger.exception("Unexpected document upload failure")
                    upload_result = {
                        "error": True,
                        "stage": "unexpected_exception",
                        "message": "Something went wrong while processing this document. Please try again.",
                        "details": f"{type(exc).__name__}: {exc}",
                    }

                if upload_result.get("stage") == "patient_name_mismatch":
                    st.session_state["doc_patient_mismatch_warning"] = upload_result.get("message", "")
                    st.warning(upload_result.get("message", "This document has not been saved."))
                    debug_info = upload_result.get("details")
                    if debug_info and os.getenv("CARESHIELD_DEBUG"):
                        st.caption(f"Debug: stage={upload_result.get('stage')} · details={debug_info}")
                elif upload_result.get("error"):
                    st.session_state.pop("doc_patient_mismatch_warning", None)
                    st.error(upload_result.get("message", "Something went wrong. Please try again."))
                    debug_info = upload_result.get("details")
                    if debug_info and os.getenv("CARESHIELD_DEBUG"):
                        st.caption(f"Debug: stage={upload_result.get('stage')} · details={debug_info}")
                else:
                    st.session_state.pop("doc_patient_mismatch_warning", None)
                    result = upload_result["result"]
                    raw_text = upload_result["raw_text"]
                    processed_at = datetime.now(timezone.utc).isoformat()
                    existing_envelope = {
                        "active": get_patient_medications_display(patient_id),
                        "discontinued": [],
                        "review_flags": [],
                    }
                    latest_plan = get_latest_patient_plan(patient_id)
                    if latest_plan:
                        plan_envelope = load_medication_plan(latest_plan.get("medications"))
                        existing_envelope["discontinued"] = plan_envelope.get("discontinued", [])
                        existing_envelope["review_flags"] = plan_envelope.get("review_flags", [])
                    updated_envelope = apply_document_medication_changes(
                        existing_envelope,
                        result.get("medications", []),
                        result.get("discontinued_medications", []),
                        result.get("medication_review_items", []),
                        document_name=uploaded_pdf.name,
                        processed_at=processed_at,
                    )

                    incoming_conditions = normalize_conditions_raw(result.get("conditions"))
                    merged_conditions = merge_conditions(
                        get_patient_conditions(patient_id),
                        incoming_conditions,
                    )

                    try:
                        save_patient_document_bundle(
                            patient_id,
                            file_name=uploaded_pdf.name,
                            raw_text=raw_text,
                            active_medications=updated_envelope["active"],
                            conditions=merged_conditions,
                            caregiver_id=st.session_state.get("selected_caregiver_id"),
                        )

                        try:
                            save_patient_plan(
                                raw_text,
                                serialize_medication_plan(updated_envelope),
                                patient_id=patient_id,
                            )
                        except Exception:
                            save_patient_plan(raw_text, serialize_medication_plan(updated_envelope))

                        st.session_state.pop(f"stored_conditions_{patient_id}", None)
                        st.session_state.pop(f"stored_medications_{patient_id}", None)
                        st.session_state[f"stored_conditions_{patient_id}"] = merged_conditions
                        st.session_state[f"med_plan_meta_{patient_id}"] = {
                            "discontinued": updated_envelope["discontinued"],
                            "review_flags": updated_envelope["review_flags"],
                        }

                        summary_bits = [
                            f"{len(updated_envelope['active'])} active medication(s) on file",
                        ]
                        added_discontinued = len(updated_envelope["discontinued"]) - len(existing_envelope.get("discontinued", []))
                        if added_discontinued > 0:
                            summary_bits.append(f"{added_discontinued} moved to discontinued")
                        new_flags = len(updated_envelope["review_flags"]) - len(existing_envelope.get("review_flags", []))
                        if new_flags > 0:
                            summary_bits.append(f"{new_flags} flagged for manual review")
                        st.session_state.doc_process_flash = (
                            "Document processed. " + "; ".join(summary_bits) + "."
                        )
                        try:
                            save_shift_log(
                                caregiver_name=get_caregiver_profile_label(
                                    st.session_state.get("selected_caregiver_id", ""),
                                ),
                                source="document_upload",
                                summary=(
                                    f"Hospital document uploaded: {uploaded_pdf.name}. "
                                    + "; ".join(summary_bits)
                                ),
                                severity="ok",
                                caregiver_id=st.session_state.get("selected_caregiver_id"),
                                patient_id=patient_id,
                            )
                        except Exception:
                            pass
                        invalidate_patient_activity_cache(patient_id)
                        st.session_state.pop(f"profile_cache_warmed_{patient_id}", None)
                        st.session_state.pdf_uploader_key += 1
                        st.rerun()
                    except Exception as exc:
                        _documents_logger.exception("Document save failed")
                        st.error("The document was read but could not be saved. Please try again.")
                        if os.getenv("CARESHIELD_DEBUG"):
                            st.caption(f"Debug: {type(exc).__name__}: {exc}")

    patient_id = get_current_patient_id()
    plan_meta = get_patient_plan_meta(patient_id)
    med_items = get_stored_medications_display(patient_id)
    conditions = get_stored_conditions(patient_id)

    render_documents_stored_overview(
        med_items,
        conditions,
        is_demo_meds=False,
        discontinued=plan_meta.get("discontinued"),
        review_flags=plan_meta.get("review_flags"),
    )

    render_responsible_ai_footer()

# ════════════════════════════════════════════════════════════════
# TAB — REPORT & ASK
# ════════════════════════════════════════════════════════════════
with tab_report:
    patient_id = get_current_patient_id()
    hydrate_key = care_hydrate_key(patient_id)
    if patient_id and not st.session_state.get(hydrate_key):
        hydrate_patient_care_session(patient_id, selected_caregiver)

    if "messages" not in st.session_state:
        st.session_state.messages = build_initial_chat_messages(selected_caregiver)
        st.session_state.report_ask_bound_caregiver_id = selected_caregiver_id
    if "chat_caregiver" not in st.session_state:
        st.session_state.chat_caregiver = selected_caregiver
    if "chat_caregiver_id" not in st.session_state:
        st.session_state.chat_caregiver_id = selected_caregiver_id
    if "chat_draft" not in st.session_state:
        st.session_state.chat_draft = ""
    if "chat_photo_uploader_key" not in st.session_state:
        st.session_state.chat_photo_uploader_key = 0
    if "chat_report_history_visible" not in st.session_state:
        st.session_state.chat_report_history_visible = CHAT_REPORT_HISTORY_DEFAULT_VISIBLE

    bound_caregiver_id = st.session_state.get("report_ask_bound_caregiver_id")
    if report_ask_needs_caregiver_reset(selected_caregiver_id, st.session_state.get("messages")):
        reset_report_ask_for_caregiver_switch(
            patient_id,
            selected_caregiver,
            selected_caregiver_id,
            previous_caregiver_id=bound_caregiver_id,
        )
    else:
        refresh_chat_welcome_message(selected_caregiver)

    if st.session_state.pop("reset_chat_draft", False):
        st.session_state.chat_draft = ""

    if st.session_state.get("pending_chat_response"):
        set_careshield_active_tab("report")
        md_html(build_loading_banner_html("Generating response..."))
        try:
            process_pending_chat_response(selected_caregiver, current_time_str)
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Something went wrong: {e}. Please try again.",
            })
            st.session_state.chat_scroll_to_bottom = True
            persist_patient_chat_thread(patient_id)
        finally:
            st.session_state.pop("pending_chat_response", None)
        st.rerun()

    md_html(build_report_ask_story_html())
    md_html(build_report_ask_how_to_use_html())

    with st.container(border=True):
        chat_plan_items = get_active_plan_items()
        visible_report_count = st.session_state.get(
            "chat_report_history_visible",
            CHAT_REPORT_HISTORY_DEFAULT_VISIBLE,
        )
        visible_messages, hidden_report_count, total_report_count = slice_messages_for_report_history(
            st.session_state.messages,
            visible_report_count,
        )
        st.markdown(
            render_chat_messages_html(
                visible_messages,
                selected_caregiver,
            ),
            unsafe_allow_html=True,
        )
        render_history_show_more_controls(
            hidden_count=hidden_report_count,
            total=total_report_count,
            shown_count=min(visible_report_count, total_report_count),
            session_key="chat_report_history_visible",
            show_more_key="chat_report_history_show_more",
            batch=CHAT_REPORT_HISTORY_EXPAND_BATCH,
            default_visible=CHAT_REPORT_HISTORY_DEFAULT_VISIBLE,
        )
        render_chat_auto_scroll()
        st.text_input(
            "Type a message",
            placeholder="Type a message",
            label_visibility="collapsed",
            key="chat_draft",
        )
        submit_clicked = st.button("Submit", use_container_width=True, key="chat_send", type="primary")

        md_html(
            '<p class="cs-chat-upload-label">Upload photo (optional — symptom only, such as a bruise, rash, or swelling)</p>'
        )
        chat_photo = st.file_uploader(
            "Upload photo",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key=f"chat_photo_upload_{st.session_state.chat_photo_uploader_key}",
        )

    if submit_clicked:
        text = st.session_state.chat_draft.strip()
        image_b64 = None
        if chat_photo is not None:
            image_b64 = base64.b64encode(chat_photo.read()).decode("utf-8")

        if not text and not image_b64:
            st.warning("Type a message or upload a photo first.")
        else:
            reported_at = datetime.now(timezone.utc).isoformat()
            timestamp_display = format_chat_timestamp(reported_at)
            st.session_state.messages.append({
                "role": "user",
                "content": text or "Photo for review",
                "has_image": bool(image_b64),
                "timestamp": reported_at,
                "timestamp_display": timestamp_display,
                "caregiver_id": selected_caregiver_id,
            })
            st.session_state.pending_chat_response = {
                "text": text,
                "image_b64": image_b64,
                "reported_at": reported_at,
                "timestamp_display": timestamp_display,
            }
            set_careshield_active_tab("report")
            st.session_state.reset_chat_draft = True
            if image_b64:
                st.session_state.chat_photo_uploader_key += 1
            st.session_state.chat_scroll_to_bottom = True
            persist_patient_chat_thread(patient_id)
            st.rerun()

    render_responsible_ai_footer()

# ════════════════════════════════════════════════════════════════
# TAB — PILL REGISTRATION
# ════════════════════════════════════════════════════════════════
with tab_pill:
    from ai_helpers import get_medication_references, get_patient_plans

    render_tab_story_section("Pill registration", *TAB_PROBLEM_SOLUTION["pill_registration"])

    latest_plan = None
    plans = cached_patient_plans(st.session_state.selected_patient_id)
    if plans:
        latest_plan = plans[0]
    plan_items = get_active_plan_items(st.session_state.selected_patient_id)

    if not plan_items:
        render_no_patient_medications_guidance()
    else:
        med_refs = cached_medication_references(st.session_state.selected_patient_id)
        reg_status = get_plan_registration_status(plan_items, med_refs)
        refs_by_name = reg_status["refs_by_name"]
        all_registered = reg_status["all_registered"]
        registered_count = reg_status["registered_count"]

        md_html(build_pill_reg_how_to_use_html())
        md_html('<div class="cs-pill-grid-section-title">Your medications</div>')
        render_pill_reg_summary_with_photo_guide(plan_items, registered_count, len(plan_items))

        for row_start in range(0, len(plan_items), 2):
            cols = st.columns(2, gap="medium")
            row_meds = plan_items[row_start:row_start + 2]
            for col_idx, med in enumerate(row_meds):
                is_registered = med["name"] in refs_by_name
                ref = refs_by_name.get(med["name"])
                meta = parse_ref_meta(ref) if ref else {}
                plan_match = find_plan_item(med["name"], plan_items) or med
                slug = med_slug(med["name"])
                with cols[col_idx]:
                    with st.container(border=True):
                        md_html(build_pill_grid_card_html(med, is_registered, meta, plan_match, ref))
                        btn_label = "Edit pill" if is_registered else "Register pill"
                        if st.button(btn_label, key=f"pill_open_{slug}", use_container_width=True):
                            st.session_state.pill_modal = {
                                "med_name": med["name"],
                                "mode": "edit" if is_registered else "register",
                            }
                            st.rerun()

        if st.session_state.get("pill_modal"):
            pill_registration_dialog()

        if all_registered:
            md_html(build_pill_reg_success_banner_html())
        elif registered_count:
            st.info(f"{registered_count} of {len(plan_items)} done. Tap any remaining card to finish.")
        else:
            st.info("Tap a medication card above to register your first pill.")

    render_responsible_ai_footer()

# ════════════════════════════════════════════════════════════════
# TAB — MEDCAM
# ════════════════════════════════════════════════════════════════
with tab_medcam:
    from ai_helpers import (
        get_medication_references,
        get_patient_plans,
    )

    render_tab_story_section("MedCam", *TAB_PROBLEM_SOLUTION["medcam"])

    latest_plan = None
    plans = cached_patient_plans(st.session_state.selected_patient_id)
    if plans:
        latest_plan = plans[0]
    plan_items = get_active_plan_items(st.session_state.selected_patient_id)

    if not plan_items:
        render_no_patient_medications_guidance()
    else:
        discharge_plan = format_medications_for_prompt(plan_items)

        med_refs = cached_medication_references(st.session_state.selected_patient_id)
        reg_status = get_plan_registration_status(plan_items, med_refs)
        all_registered = reg_status["all_registered"]
        plan_refs = reg_status["plan_refs"]

        md_html(build_medcam_how_to_use_html())

        if not all_registered:
            if reg_status["registered_count"]:
                st.info(
                    f"{reg_status['registered_count']} of {reg_status['total']} medications registered. "
                    "Finish in the Pill Registration tab to unlock MedCam verification."
                )
            else:
                st.info("Register all medications in the Pill Registration tab to unlock MedCam verification.")

        if all_registered:
            uploaded_image = None
            upload_sig = ""
            med_left, med_right = st.columns([1, 1], gap="large")
            with med_left:
                render_medcam_dose_cards_column(
                    plan_items,
                    selected_caregiver,
                    st.session_state.selected_patient_id,
                )
            with med_right:
                with st.container(border=True):
                    md_html('<div class="cs-medcam-upload-panel"></div>')
                    md_html('<p class="cs-medcam-upload-label">Take or upload a photo of the pills to verify</p>')
                    uploaded_image = st.file_uploader(
                        "Take or upload a photo of the pills",
                        type=["jpg", "jpeg", "png"],
                        key="medcam_uploader",
                        label_visibility="collapsed",
                    )
                    if uploaded_image:
                        st.image(uploaded_image, caption="Pills to verify", width=280)
                        upload_sig = f"{uploaded_image.name}:{uploaded_image.size}"
                        if st.session_state.get("medcam_upload_sig") != upload_sig:
                            st.session_state.medcam_upload_sig = upload_sig
                            set_careshield_active_tab("medcam")
                    else:
                        st.session_state.pop("medcam_upload_sig", None)
                    check_clicked = st.button("Check medication", key="medcam_button", use_container_width=True)
                render_medcam_clock_column(
                    plan_items,
                    selected_caregiver,
                    st.session_state.selected_patient_id,
                )

            if process_medcam_pending_verification(
                uploaded_image=uploaded_image,
                upload_sig=upload_sig,
                plan_refs=plan_refs,
                plan_items=plan_items,
                latest_plan=latest_plan,
                discharge_plan=discharge_plan,
                patient_id=st.session_state.selected_patient_id,
                caregiver_name=selected_caregiver,
                caregiver_id=selected_caregiver_id,
            ):
                st.stop()

            if check_clicked:
                if uploaded_image is None:
                    st.warning("Please upload a photo first.")
                else:
                    set_careshield_active_tab("medcam")
                    st.session_state[MEDCAM_PENDING_UPLOAD_KEY] = upload_sig
                    st.rerun()

            resolved_patient_id = resolve_patient_id(st.session_state.selected_patient_id)
            medcam_error = st.session_state.get(f"medcam_error_{resolved_patient_id}")
            if medcam_error:
                st.error(medcam_error)

            last_result = st.session_state.get(f"medcam_last_{resolved_patient_id}")
            if last_result and not last_result.get("error"):
                md_html(build_medcam_verdict_panel_html(
                    last_result["verdict"],
                    last_result.get("enriched_pills") or [],
                    last_result["log_time"],
                    ai_result=last_result.get("ai_result"),
                    absence_warnings=last_result.get("absence_warnings"),
                    med_refs=load_live_medication_references(resolved_patient_id),
                ))

            audit_records = get_medcam_audit_records(st.session_state.selected_patient_id)
            md_html(
                f'<div class="cs-medcam-history">'
                f'<div class="cs-medcam-history-heading">Past checks</div>'
                f"{build_medcam_audit_history_html(audit_records)}"
                f"</div>"
            )

    render_responsible_ai_footer()


# ════════════════════════════════════════════════════════════════
# TAB — HANDOVER
# ════════════════════════════════════════════════════════════════
with tab_handover:
    handover_patient_id = get_current_patient_id()
    if handover_patient_id and not st.session_state.get(care_hydrate_key(handover_patient_id)):
        hydrate_patient_care_session(handover_patient_id, selected_caregiver)

    if "last_sbar_result" not in st.session_state:
        st.session_state.last_sbar_result = None
    if "handover_symptom_timeline_visible" not in st.session_state:
        st.session_state.handover_symptom_timeline_visible = HANDOVER_TIMELINE_DEFAULT_VISIBLE
    if "handover_adherence_timeline_visible" not in st.session_state:
        st.session_state.handover_adherence_timeline_visible = HANDOVER_TIMELINE_DEFAULT_VISIBLE
    if "handover_reported_by_visible" not in st.session_state:
        st.session_state.handover_reported_by_visible = HANDOVER_TIMELINE_DEFAULT_VISIBLE
    if "handover_period" not in st.session_state:
        st.session_state.handover_period = "this_week"

    tz_obj, _ = get_schedule_tz()

    render_tab_story_section("Handover", *TAB_PROBLEM_SOLUTION["handover"])
    md_html(build_handover_how_to_use_html())

    with st.container(border=True):
        period_key = render_handover_period_selector()
        st.caption(f"Showing entries for **{get_handover_period_label(period_key)}**.")
        md_html('<div class="cs-handover-generate-card"></div>')
        if st.button(
            "Generate SBAR handover",
            key="handover_button",
            use_container_width=True,
            type="primary",
        ):
            set_careshield_active_tab("handover")
            patient_id = get_current_patient_id()
            invalidate_patient_activity_cache(patient_id)
            symptom_events, adherence_events = collect_handover_events_for_sbar(
                patient_id,
                period_key,
                tz_obj,
            )

            if not symptom_events and not adherence_events:
                st.warning(HANDOVER_INSUFFICIENT_DATA_MSG)
            else:
                period_label = get_handover_period_label(period_key)
                md_html(build_loading_banner_html("Generating handover card..."))
                system_prompt = build_sbar_handover_system_prompt(period_label)
                sbar_payload = build_sbar_handover_user_payload(
                    symptom_events,
                    adherence_events,
                )
                result = ask_ai(system_prompt, sbar_payload)

                if result.get("error"):
                    st.error(result["message"])
                else:
                    symptom_events_for_report = attach_timeline_event_photos(
                        symptom_events,
                        patient_id,
                    )
                    enriched = enrich_handover_result_with_period_entries(
                        merge_pending_handover_questions(
                            patient_id,
                            result,
                        ),
                        symptom_events_for_report,
                        adherence_events,
                    )
                    st.session_state.last_sbar_result = enriched
                    st.rerun()

        if st.session_state.last_sbar_result:
            result = st.session_state.last_sbar_result
            md_html(build_sbar_results_html(result))
            sbar_reported = result.get("reported_by") or []
            if sbar_reported:
                render_paginated_reported_by(
                    sbar_reported,
                    heading="Reported by",
                    session_key="handover_reported_by_visible",
                    show_more_key="handover_reported_by_show_more",
                    show_less_key="handover_reported_by_show_less",
                )
            render_handover_symptom_photos(
                st.session_state.get("selected_patient_id"),
                period_key,
                tz_obj,
            )

            pdf_bytes = get_handover_pdf_bytes(
                st.session_state.get("selected_patient_id"),
                period_key,
                tz_obj,
                result,
            )
            _dl_left, _dl_center, _dl_right = st.columns([1, 2, 1])
            with _dl_center:
                st.download_button(
                    label="Download full handover with SBAR (PDF)",
                    data=pdf_bytes,
                    file_name="careshield_gp_handover.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="handover_sbar_download",
                )

    period_key = st.session_state.handover_period
    with st.container(border=True):
        md_html('<div class="cs-handover-dashboard-shell"></div>')
        render_handover_dashboard(
            st.session_state.get("selected_patient_id"),
            period_key,
            tz_obj,
        )
        render_handover_visible_timelines(
            st.session_state.get("selected_patient_id"),
            period_key,
            tz_obj,
        )

    render_responsible_ai_footer()

# ════════════════════════════════════════════════════════════════
# TAB — MY RESULTS
# ════════════════════════════════════════════════════════════════
with tab_results:
    if "my_results_uploader_key" not in st.session_state:
        st.session_state.my_results_uploader_key = 0

    patient_id = get_current_patient_id()
    patient_name = get_patient_display_name(patient_id)
    conditions = get_patient_conditions(patient_id)
    cache_key = my_results_cache_key(patient_id)
    processed_key = my_results_processed_key(patient_id)
    error_key = my_results_error_key(patient_id)
    error_debug_key = my_results_error_debug_key(patient_id)
    feedback_key = my_results_feedback_key(patient_id)

    pending_upload = st.session_state.get("my_results_pending_upload")
    if pending_upload:
        set_careshield_active_tab("my_results")
        stage = str(pending_upload.get("stage") or "extract")
        loading_message = (
            "Preparing your plain-English explanation..."
            if stage == "explain"
            else "Analysing document and preparing your explanation..."
        )
        md_html(build_loading_banner_html(loading_message))
        upload_sig = pending_upload.get("sig") or ""
        restored_upload = restore_my_results_upload(pending_upload)
        try:
            result = process_my_results_upload(
                restored_upload,
                patient_id,
                patient_name,
                conditions,
                selected_caregiver,
                caregiver_id=selected_caregiver_id,
            )
        except Exception as exc:
            _my_results_logger.exception("Unexpected My Results upload failure")
            result = {
                "error": True,
                "stage": "unexpected_exception",
                "message": "Something went wrong while reading this file. Please try again.",
                "details": f"{type(exc).__name__}: {exc}",
            }
        finally:
            st.session_state.pop("my_results_pending_upload", None)

        apply_my_results_processing_result(
            result,
            patient_id=patient_id,
            upload_sig=upload_sig,
            cache_key=cache_key,
            processed_key=processed_key,
            error_key=error_key,
            error_debug_key=error_debug_key,
        )
        st.rerun()

    render_tab_story_section("My results", *TAB_PROBLEM_SOLUTION["my_results"])
    md_html(build_my_results_how_to_use_html())

    with st.container(border=True):
        md_html("""
        <div class="cs-mr-upload-shell">
          <div class="cs-mr-upload-heading">Upload a result or letter</div>
          <div class="cs-mr-upload-sub">PDF, photo, or image — blood tests, scans, clinic letters</div>
        </div>
        """)
        uploaded_result = st.file_uploader(
            "Upload a result or letter",
            type=["pdf", "jpg", "jpeg", "png", "webp"],
            key=f"my_results_uploader_{st.session_state.my_results_uploader_key}",
            label_visibility="collapsed",
        )

        if uploaded_result is not None:
            file_kind = "PDF" if uploaded_result.name.lower().endswith(".pdf") else "Image"
            md_html(build_upload_confirmation_html(
                uploaded_result.name,
                size_bytes=uploaded_result.size,
                kind=file_kind,
            ))
            upload_sig = f"{uploaded_result.name}:{uploaded_result.size}"
            if st.session_state.get("my_results_upload_sig") != upload_sig:
                st.session_state.my_results_upload_sig = upload_sig
                set_careshield_active_tab("my_results")
        else:
            st.session_state.pop("my_results_upload_sig", None)

        if st.button(
            "Analyse document",
            key="my_results_analyse_btn",
            use_container_width=True,
            type="primary",
        ):
            if uploaded_result is None:
                st.warning("Please select a file first.")
            else:
                set_careshield_active_tab("my_results")
                st.session_state.pop(error_key, None)
                st.session_state.pop(error_debug_key, None)
                st.session_state.pop(feedback_key, None)
                st.session_state.pop("my_results_error", None)
                st.session_state.pop("my_results_error_debug", None)
                st.session_state.my_results_pending_upload = serialize_my_results_upload(uploaded_result)
                st.rerun()

    feedback = st.session_state.get(feedback_key)
    if feedback:
        if feedback.get("type") == "success":
            st.success(str(feedback.get("message") or "Analysis complete."))
        else:
            st.error(str(feedback.get("message") or MY_RESULTS_NO_RESULTS_MESSAGE))

    if st.session_state.get(error_key):
        st.error(st.session_state[error_key])
        debug_info = st.session_state.get(error_debug_key)
        if debug_info and os.getenv("CARESHIELD_DEBUG"):
            st.caption(f"Debug: stage={debug_info.get('stage')} · details={debug_info.get('details')}")

    record = load_my_results_record(patient_id)
    if not record:
        md_html(
            '<div class="cs-mr-empty">'
            "Drop in a result and we'll help you understand it before the appointment."
            "</div>"
        )
    else:
        md_html(build_my_results_card_html(record))

        pdf_bytes = get_my_results_pdf_bytes(record, patient_name, patient_id)
        safe_name = re.sub(r"[^\w\-]+", "_", record.get("file_name") or "my_results").strip("_")
        with st.container(border=True):
            md_html('<div class="cs-mr-actions"></div>')
            action_left, action_right = st.columns(2)
            with action_left:
                if st.button(
                    "Add to handover",
                    key="my_results_add_handover",
                    use_container_width=True,
                    type="secondary",
                ):
                    append_my_results_to_handover(
                        patient_id,
                        record,
                        selected_caregiver,
                        caregiver_id=selected_caregiver_id,
                    )
                    st.success("Questions added to your handover.")
            with action_right:
                st.download_button(
                    label="Download summary",
                    data=pdf_bytes,
                    file_name=f"careshield_{safe_name or 'my_results'}_summary.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary",
                    key="my_results_download_summary",
                )
            md_html(
                '<p class="cs-mr-actions-note">Add the doctor questions to your Handover tab, '
                "or download a printable summary to take to the appointment.</p>"
            )

    render_responsible_ai_footer()

render_careshield_tab_restore()
render_careshield_boot_reveal()