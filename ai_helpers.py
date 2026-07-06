import os
import json
import re
import logging
from difflib import get_close_matches
from datetime import date, datetime, timedelta, timezone
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=90.0)

try:
    import anthropic
    anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
except Exception:
    anthropic_client = None

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

from care_data_quality import (
    care_row_is_internal_test,
    filter_production_care_rows,
    filter_production_shift_log_rows,
    is_designated_test_patient,
    is_internal_test_care_entry,
    shift_log_row_is_internal_test,
    should_block_test_entry_for_patient,
)
from patient_care_storage import (
    any_other_patient_has_local_reports,
    fetch_local_care_report_photo,
    fetch_local_care_reports,
    fetch_local_chat_thread,
    legacy_backfill_completed,
    mark_legacy_backfill_completed,
    purge_local_internal_test_entries,
    save_local_care_report,
    save_local_chat_thread,
)

logger = logging.getLogger("careshield")
my_results_logger = logging.getLogger("careshield.my_results")

from io import BytesIO
from pypdf import PdfReader
from pypdf.errors import (
    EmptyFileError,
    FileNotDecryptedError,
    ParseError,
    PdfReadError,
    PdfStreamError,
    WrongPasswordError,
)

PDF_MAGIC = b"%PDF"
MIN_PDF_TEXT_CHARS = 1

PDF_ERROR_MESSAGES = {
    "blank_pdf": (
        "No medical information or text found in this document. Please upload a valid document."
    ),
    "corrupted_pdf": (
        "The uploaded file appears to be corrupted or damaged. Please check the file and try again."
    ),
    "password_protected": (
        "This PDF is password-protected. Please upload an unprotected version of the document."
    ),
    "file_type_mismatch": (
        "Invalid file format. The file extension does not match the actual file content."
    ),
    "empty_buffer": (
        "The uploaded file appears to be corrupted or damaged. Please check the file and try again."
    ),
}


def _set_pdf_meta_error(meta: dict, code: str, technical_error: str | None = None) -> dict:
    meta["error_code"] = code
    meta["error"] = technical_error or code
    meta["user_message"] = PDF_ERROR_MESSAGES.get(code, PDF_ERROR_MESSAGES["corrupted_pdf"])
    return meta


def _read_upload_bytes(uploaded_file) -> bytes:
    uploaded_file.seek(0)
    if hasattr(uploaded_file, "getvalue"):
        return uploaded_file.getvalue() or b""
    return uploaded_file.read() or b""


def _bytes_look_like_pdf(pdf_bytes: bytes) -> bool:
    if not pdf_bytes:
        return False
    return pdf_bytes.lstrip()[:4] == PDF_MAGIC


def _classify_pdf_exception(exc: Exception) -> str:
    if isinstance(exc, (WrongPasswordError, FileNotDecryptedError)):
        return "password_protected"
    message = str(exc).lower()
    if "password" in message or "encrypted" in message or "decrypt" in message:
        return "password_protected"
    return "corrupted_pdf"


def pdf_extraction_error_response(meta: dict) -> dict | None:
    """Return a UI-ready error payload when PDF extraction failed, else None."""
    code = meta.get("error_code")
    if not code:
        return None
    return {
        "error": True,
        "stage": code,
        "message": meta.get("user_message") or PDF_ERROR_MESSAGES.get(code, PDF_ERROR_MESSAGES["corrupted_pdf"]),
        "details": {
            "technical_error": meta.get("error"),
            "byte_count": meta.get("byte_count"),
            "page_count": meta.get("page_count"),
            "char_count": meta.get("char_count"),
            "mime_type": meta.get("mime_type"),
        },
    }


def extract_text_from_pdf_with_meta(uploaded_file) -> dict:
    """
    Extract embedded text from a PDF (not OCR).
    Returns dict with keys: text, byte_count, page_count, char_count, mime_type,
    error, error_code, user_message.
    """
    mime_type = getattr(uploaded_file, "type", None) or ""
    file_name = getattr(uploaded_file, "name", "?")
    meta = {
        "text": "",
        "byte_count": 0,
        "page_count": 0,
        "char_count": 0,
        "mime_type": mime_type,
        "error": None,
        "error_code": None,
        "user_message": None,
    }
    try:
        pdf_bytes = _read_upload_bytes(uploaded_file)
        meta["byte_count"] = len(pdf_bytes)
        if not pdf_bytes:
            logger.warning(
                "PDF extraction: empty buffer (mime=%s, name=%s)",
                mime_type,
                file_name,
            )
            return _set_pdf_meta_error(meta, "empty_buffer", "empty_buffer")

        if not _bytes_look_like_pdf(pdf_bytes):
            logger.warning(
                "PDF extraction: file type mismatch (mime=%s, name=%s, head=%r)",
                mime_type,
                file_name,
                pdf_bytes[:16],
            )
            return _set_pdf_meta_error(meta, "file_type_mismatch", "file_type_mismatch")

        try:
            reader = PdfReader(BytesIO(pdf_bytes))
        except (PdfReadError, PdfStreamError, EmptyFileError, ParseError) as exc:
            code = _classify_pdf_exception(exc)
            logger.warning(
                "PDF extraction: parse failed (%s) name=%s: %s",
                code,
                file_name,
                exc,
            )
            return _set_pdf_meta_error(meta, code, f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            code = _classify_pdf_exception(exc)
            logger.exception(
                "PDF extraction: unexpected parse failure name=%s: %s",
                file_name,
                exc,
            )
            return _set_pdf_meta_error(meta, code, f"{type(exc).__name__}: {exc}")

        if reader.is_encrypted:
            decrypt_ok = 0
            try:
                decrypt_ok = reader.decrypt("")
            except Exception as decrypt_exc:
                logger.warning(
                    "PDF extraction: decrypt failed name=%s: %s",
                    file_name,
                    decrypt_exc,
                )
            if not decrypt_ok or reader.is_encrypted:
                logger.warning("PDF extraction: password-protected name=%s", file_name)
                return _set_pdf_meta_error(meta, "password_protected", "password_protected")

        meta["page_count"] = len(reader.pages)
        parts = []
        for page_index, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception as page_exc:
                logger.warning(
                    "PDF extraction: page %s/%s failed name=%s: %s",
                    page_index,
                    meta["page_count"],
                    file_name,
                    page_exc,
                )
                return _set_pdf_meta_error(
                    meta,
                    _classify_pdf_exception(page_exc),
                    f"{type(page_exc).__name__}: {page_exc}",
                )
            if not page_text.strip():
                logger.info(
                    "PDF extraction: page %s/%s returned no text (name=%s)",
                    page_index,
                    meta["page_count"],
                    file_name,
                )
            parts.append(page_text)

        text = "".join(parts)
        meta["text"] = text
        meta["char_count"] = len(text.strip())

        if meta["char_count"] < MIN_PDF_TEXT_CHARS:
            logger.warning(
                "PDF extraction: blank PDF name=%s bytes=%s pages=%s",
                file_name,
                meta["byte_count"],
                meta["page_count"],
            )
            return _set_pdf_meta_error(meta, "blank_pdf", "blank_pdf")

        logger.info(
            "PDF extraction ok: name=%s bytes=%s pages=%s chars=%s mime=%s",
            file_name,
            meta["byte_count"],
            meta["page_count"],
            meta["char_count"],
            mime_type or "unknown",
        )
        return meta
    except Exception as exc:
        code = _classify_pdf_exception(exc)
        logger.exception(
            "PDF extraction failed for %s: %s",
            file_name,
            exc,
        )
        return _set_pdf_meta_error(meta, code, f"{type(exc).__name__}: {exc}")


def extract_text_from_pdf(uploaded_file):
    """
    Extracts all text from an uploaded PDF file (Streamlit file object).
    Uses embedded text extraction only — does not OCR or convert to images.
    """
    return extract_text_from_pdf_with_meta(uploaded_file)["text"]


DOCUMENTS_PATIENT_NAME_EXTRACT_PROMPT = """Read the hospital or clinic document and identify the patient it belongs to.
Respond with ONLY a JSON object:
{
  "patient_name": "full patient name as written in the document, or null if unclear",
  "confidence": "high" or "low"
}
Rules:
- Use "high" only when the document clearly names one patient (letterhead, discharge header, "Patient:", "Re:", etc.).
- Use "low" when the name is missing, ambiguous, or multiple patients are mentioned.
- Return the full name exactly as written (e.g. "Bartholomew Nkemelu", "Eleanor Whitfield").
"""


def person_name_parts(name: str) -> list[str]:
    cleaned = re.sub(r"[^a-zA-Z'\- ]", " ", str(name or ""))
    return [part for part in cleaned.lower().split() if part]


def patient_names_match(active_name: str, document_name: str) -> bool:
    """Return True when the document name plausibly refers to the active profile."""
    active_parts = person_name_parts(active_name)
    document_parts = person_name_parts(document_name)
    if not active_parts or not document_parts:
        return True

    if active_parts[-1] != document_parts[-1]:
        return False

    active_first = active_parts[0]
    document_first = document_parts[0]
    if active_first == document_first:
        return True

    shorter = min(len(active_first), len(document_first))
    longer = max(len(active_first), len(document_first))
    if shorter >= 1 and longer >= 2:
        if active_first.startswith(document_first) or document_first.startswith(active_first):
            return True
    return False


def build_patient_name_mismatch_message(active_name: str, document_name: str) -> str:
    active_label = str(active_name or "the active patient").strip()
    document_label = str(document_name or "another patient").strip()
    return (
        f"Warning: The uploaded document appears to belong to another patient "
        f"({document_label}) and does not match the active profile ({active_label}). "
        "This document has not been saved."
    )


def extract_document_patient_name(raw_text: str) -> dict:
    excerpt = (raw_text or "").strip()
    if len(excerpt) > 12000:
        excerpt = excerpt[:12000]
    result = ask_ai(
        DOCUMENTS_PATIENT_NAME_EXTRACT_PROMPT,
        f"Document text:\n\n{excerpt}",
    )
    if result.get("error"):
        return {
            "patient_name": None,
            "confidence": "low",
            "error": result.get("message"),
        }
    patient_name = result.get("patient_name")
    if isinstance(patient_name, str):
        patient_name = patient_name.strip() or None
    else:
        patient_name = None
    confidence = str(result.get("confidence") or "low").strip().lower()
    if confidence not in ("high", "low"):
        confidence = "low"
    return {
        "patient_name": patient_name,
        "confidence": confidence,
    }


def validate_document_patient_profile(raw_text: str, active_patient_name: str) -> dict | None:
    """
    Compare the patient named in the document with the active profile.
    Returns a UI-ready warning payload on mismatch, else None.
    """
    active_name = str(active_patient_name or "").strip()
    if not active_name:
        return None

    extracted = extract_document_patient_name(raw_text)
    document_name = extracted.get("patient_name")
    confidence = extracted.get("confidence", "low")

    if not document_name or confidence != "high":
        logger.info(
            "Document patient validation skipped: active=%s document=%s confidence=%s",
            active_name,
            document_name,
            confidence,
        )
        return None

    if patient_names_match(active_name, document_name):
        logger.info(
            "Document patient validation passed: active=%s document=%s",
            active_name,
            document_name,
        )
        return None

    logger.warning(
        "Document patient validation failed: active=%s document=%s",
        active_name,
        document_name,
    )
    return {
        "error": True,
        "warning": True,
        "stage": "patient_name_mismatch",
        "message": build_patient_name_mismatch_message(active_name, document_name),
        "details": {
            "active_patient_name": active_name,
            "document_patient_name": document_name,
            "confidence": confidence,
        },
    }

# ── Patient / document schema (Supabase tables: patients, medications, conditions, documents) ──

MEDICATION_DISPLAY_COLORS = ["blue", "blue", "green", "green", "purple", "orange", "teal", "coral"]


def resolve_patient_id(patient_id=None):
    if patient_id is not None and str(patient_id).strip():
        return str(patient_id)
    return None


_SHIFT_LOG_PATIENT_MARKER_RE = re.compile(r"\[\[patient:([^\]]+)\]\]")


def shift_log_patient_marker(patient_id) -> str:
    return f"[[patient:{resolve_patient_id(patient_id)}]]"


def summary_matches_shift_log_patient(summary: str, patient_id) -> bool:
    resolved_id = resolve_patient_id(patient_id)
    if not resolved_id:
        return False
    return shift_log_patient_marker(resolved_id) in str(summary or "")


def strip_shift_log_patient_marker(summary: str) -> str:
    text = str(summary or "").strip()
    return _SHIFT_LOG_PATIENT_MARKER_RE.sub("", text).strip()


def _merge_care_report_rows(*groups: list) -> list:
    merged = []
    seen = set()
    for group in groups:
        for row in group or []:
            key = (
                str(row.get("reported_at") or row.get("created_at") or ""),
                str(row.get("summary") or row.get("report_text") or "")[:200],
                str(row.get("source") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
    merged.sort(key=lambda item: str(item.get("reported_at") or item.get("created_at") or ""))
    return merged


def _patient_id_int(patient_id) -> int | None:
    if patient_id is None or str(patient_id).strip() == "":
        return None
    return int(patient_id)


def list_account_patients(account_id=None) -> list:
    """All patients for this family account (account_id kept for API compatibility)."""
    del account_id
    try:
        response = (
            supabase.table("patients")
            .select("*")
            .order("created_at")
            .execute()
        )
        return response.data or []
    except Exception:
        return []


def get_patient_by_id(patient_id) -> dict | None:
    patient_id_int = _patient_id_int(patient_id)
    if patient_id_int is None:
        return None
    try:
        response = (
            supabase.table("patients")
            .select("*")
            .eq("id", patient_id_int)
            .limit(1)
            .execute()
        )
        if response.data:
            return response.data[0]
    except Exception:
        pass
    return None


def get_or_create_default_patient(account_id=None, name: str = "Patient") -> str | None:
    """Return the first patient id, creating a default row if the table is empty."""
    del account_id
    patients = list_account_patients()
    if patients:
        return str(patients[0]["id"])
    patient, _error = create_patient(name)
    return str(patient["id"]) if patient else None


def get_patient_display_name(patient_id=None) -> str:
    patient = get_patient_by_id(patient_id)
    if patient and patient.get("display_name"):
        return str(patient["display_name"])
    return "Patient"


def create_patient(name: str, account_id=None) -> tuple[dict | None, str | None]:
    """Insert a row into public.patients (initial + display_name)."""
    del account_id
    clean_name = str(name or "").strip()
    if not clean_name:
        return None, "Please enter a patient name."

    payload = {
        "initial": clean_name[0].upper(),
        "display_name": clean_name,
    }
    try:
        response = supabase.table("patients").insert(payload).select("*").execute()
        if response.data:
            return response.data[0], None
    except Exception as exc:
        return None, f"Could not save patient: {exc}"

    try:
        response = (
            supabase.table("patients")
            .select("*")
            .eq("display_name", clean_name)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if response.data:
            return response.data[0], None
    except Exception as exc:
        return None, f"Could not save patient: {exc}"

    return None, "Could not save patient. Check your Supabase connection."


def update_patient(patient_id, display_name: str) -> tuple[bool, str | None]:
    patient_id_int = _patient_id_int(patient_id)
    clean_name = str(display_name or "").strip()
    if patient_id_int is None:
        return False, "Invalid patient."
    if not clean_name:
        return False, "Please enter a patient name."
    try:
        supabase.table("patients").update({
            "display_name": clean_name,
            "initial": clean_name[0].upper(),
        }).eq("id", patient_id_int).execute()
        return True, None
    except Exception as exc:
        return False, f"Could not update patient: {exc}"


def delete_patient(patient_id) -> tuple[bool, str | None]:
    patient_id_int = _patient_id_int(patient_id)
    if patient_id_int is None:
        return False, "Invalid patient."
    try:
        supabase.table("patients").delete().eq("id", patient_id_int).execute()
        return True, None
    except Exception as exc:
        return False, f"Could not delete patient: {exc}"


def format_medication_dosage_instructions(med: dict) -> str:
    parts = []
    dosage = str(med.get("dosage") or "").strip()
    timing = str(med.get("timing") or med.get("time") or "").strip()
    pills = med.get("pills_per_dose")
    if dosage:
        parts.append(dosage)
    if timing:
        parts.append(timing)
    if pills:
        parts.append(f"{pills} pill(s) per dose")
    return " · ".join(parts)


def parse_dosage_instructions(instructions: str) -> dict:
    text = str(instructions or "").strip()
    if not text:
        return {"dosage": "", "time": "", "timing": ""}
    parts = [part.strip() for part in text.split("·")]
    dosage = ""
    timing = text
    if parts:
        if re.search(r"\d\s*mg|mcg|g\b", parts[0], re.I):
            dosage = parts[0]
            timing = " · ".join(parts[1:]) if len(parts) > 1 else ""
        else:
            timing = text
    return {"dosage": dosage, "time": timing, "timing": timing}


def medications_to_display_rows(rows: list) -> list:
    from medication_schedule import normalize_medication_schedule_fields

    items = []
    for index, row in enumerate(rows or []):
        parsed = parse_dosage_instructions(row.get("dosage_instructions"))
        items.append(normalize_medication_schedule_fields({
            "name": row.get("name", ""),
            "dosage": parsed["dosage"],
            "time": parsed["time"],
            "timing": parsed["timing"],
            "color": MEDICATION_DISPLAY_COLORS[index % len(MEDICATION_DISPLAY_COLORS)],
        }))
    return items


def get_patient_medications(patient_id=None) -> list:
    patient_id_int = _patient_id_int(patient_id)
    if patient_id_int is None:
        return []
    try:
        response = (
            supabase.table("medications")
            .select("*")
            .eq("patient_id", patient_id_int)
            .order("created_at")
            .execute()
        )
        return response.data or []
    except Exception:
        return []


def get_patient_medications_display(patient_id=None) -> list:
    return medications_to_display_rows(get_patient_medications(patient_id))


def _condition_notes_to_fields(notes: str) -> dict:
    text = str(notes or "").strip()
    if not text:
        return {"since": "Unknown", "badge": "chronic"}
    if text.startswith("{"):
        try:
            data = json.loads(text)
            return {
                "since": data.get("since") or "Unknown",
                "badge": data.get("badge") or "chronic",
            }
        except json.JSONDecodeError:
            pass
    if "|" in text:
        since, badge = text.split("|", 1)
        return {"since": since.strip() or "Unknown", "badge": badge.strip() or "chronic"}
    return {"since": text, "badge": "chronic"}


def _condition_fields_to_notes(condition: dict) -> str:
    return json.dumps({
        "since": condition.get("since") or "Unknown",
        "badge": condition.get("badge") or "chronic",
    })


def get_patient_conditions(patient_id=None) -> list:
    patient_id_int = _patient_id_int(patient_id)
    if patient_id_int is None:
        return []
    try:
        response = (
            supabase.table("conditions")
            .select("*")
            .eq("patient_id", patient_id_int)
            .order("created_at")
            .execute()
        )
        return [
            {
                "name": row.get("name", ""),
                "since": _condition_notes_to_fields(row.get("notes")).get("since", "Unknown"),
                "badge": _condition_notes_to_fields(row.get("notes")).get("badge", "chronic"),
            }
            for row in (response.data or [])
            if row.get("name")
        ]
    except Exception:
        return []


CHAT_SEVERITY_RANK = {
    "ok": 0,
    "monitor": 1,
    "contact_doctor": 2,
    "emergency": 3,
    "urgent": 3,
}


def normalize_severity_level(severity: str) -> str:
    value = str(severity or "monitor").strip().lower().replace(" ", "_")
    if value == "urgent":
        return "emergency"
    if value in ("contact_doctor", "contactdoctor", "doctor"):
        return "contact_doctor"
    if value in CHAT_SEVERITY_RANK:
        return value
    return "monitor"


def escalate_severity(current: str, proposed: str) -> str:
    current_level = normalize_severity_level(current)
    proposed_level = normalize_severity_level(proposed)
    if CHAT_SEVERITY_RANK.get(proposed_level, 1) > CHAT_SEVERITY_RANK.get(current_level, 1):
        return proposed_level
    return current_level


SYMPTOM_CONDITION_CROSSCHECK_PROMPT = """You are a clinical safety assistant for family caregivers using CareShield.

You receive a caregiver's new symptom report and the patient's stored chronic conditions from their medical record.
Use your medical knowledge to decide whether this symptom is MORE dangerous for THIS patient because of ANY stored condition.
Do not rely on a fixed rule list — reason dynamically about vulnerability, complications, and red-flag combinations for every condition provided.

Examples (illustrative only — apply the same reasoning to any condition on file):
- Fever + Type 1 Diabetes → infection/DKA risk; often needs urgent clinical review
- Fever + COPD → respiratory infection and breathing-compromise risk
- Fever + Chronic Kidney Disease → infection plus fluid/electrolyte and kidney-stress risk

Respond with ONLY a JSON object:
{
  "symptom_identified": "brief symptom label (e.g. fever, chest pain, confusion)",
  "is_elevated_risk": boolean — true when ANY stored condition makes this symptom meaningfully more dangerous than in an otherwise healthy person,
  "recommended_severity": one of "ok", "monitor", "contact_doctor", "emergency",
  "needs_doctor": boolean — true when recommended_severity is "contact_doctor" or "emergency",
  "condition_risks": [
    {
      "condition_name": "exact name from PATIENT CONDITIONS",
      "is_relevant": boolean,
      "severity_impact": "none" | "monitor" | "contact_doctor" | "emergency",
      "education_message": "How [Condition Name] Impacts This Symptom: [clear, calm explanation]" or null when not relevant
    }
  ]
}

Rules for education_message (critical):
- Include one object per stored condition; set is_relevant=true when that condition helps explain or changes the risk of this symptom — even if the symptom is mild.
- When is_relevant=true, education_message MUST begin exactly with: "How [Condition Name] Impacts This Symptom: " using the condition_name from the list.
- Explain how the reported symptom affects that specific condition and what dangerous risks the caregiver should watch for — educational, not panicked, no diagnosis.
- ALWAYS mark is_relevant=true for obvious pairs (examples: joint stiffness + osteoarthritis; swelling + heart failure; unusual bruising when anticoagulants are on file; fever + diabetes/COPD/kidney disease).
- If no conditions are on file OR none relate to the symptom: is_elevated_risk=false and all is_relevant=false.
"""


def build_patient_conditions_payload(patient_id=None) -> dict:
    """Structured condition list from Supabase for symptom cross-check prompts."""
    patient_id = resolve_patient_id(patient_id)
    conditions = get_patient_conditions(patient_id)
    return {
        "patient_id": patient_id,
        "patient_name": get_patient_display_name(patient_id),
        "conditions": [
            {
                "name": str(item.get("name") or "").strip(),
                "status": str(item.get("badge") or "chronic").strip(),
                "since": str(item.get("since") or "Unknown").strip(),
            }
            for item in conditions
            if str(item.get("name") or "").strip()
        ],
    }


def ensure_condition_education_format(condition_name: str, message: str) -> str:
    name = str(condition_name or "").strip()
    text = str(message or "").strip()
    prefix = f"How {name} Impacts This Symptom: "
    if text.startswith(prefix):
        return text
    if re.match(r"^How .+ Impacts This Symptom:", text, re.I):
        return text
    body = text
    for header_pattern in (
        r"impacts this symptom:\s*",
        r"changes the game:\s*",
    ):
        split_match = re.split(header_pattern, text, maxsplit=1, flags=re.I)
        if len(split_match) > 1:
            body = split_match[1].strip()
            break
    return prefix + body


def normalize_symptom_condition_analysis(raw: dict) -> dict:
    condition_risks = []
    for item in raw.get("condition_risks") or []:
        if not isinstance(item, dict):
            continue
        condition_name = str(item.get("condition_name") or "").strip()
        is_relevant = bool(item.get("is_relevant"))
        education_message = item.get("education_message")
        if is_relevant and education_message:
            education_message = ensure_condition_education_format(
                condition_name,
                str(education_message),
            )
        else:
            education_message = None
        condition_risks.append({
            "condition_name": condition_name,
            "is_relevant": is_relevant,
            "severity_impact": normalize_severity_level(item.get("severity_impact") or "none"),
            "education_message": education_message,
        })

    relevant = [item for item in condition_risks if item.get("is_relevant")]
    recommended = normalize_severity_level(raw.get("recommended_severity") or "monitor")
    if relevant:
        for item in relevant:
            impact = item.get("severity_impact") or "none"
            if impact != "none":
                recommended = escalate_severity(recommended, impact)

    is_elevated = bool(raw.get("is_elevated_risk")) or bool(relevant)
    if is_elevated and recommended in ("ok", "monitor"):
        recommended = "contact_doctor"

    needs_doctor = bool(raw.get("needs_doctor")) or recommended in ("contact_doctor", "emergency")
    return {
        "symptom_identified": str(raw.get("symptom_identified") or "").strip(),
        "is_elevated_risk": is_elevated,
        "recommended_severity": recommended,
        "needs_doctor": needs_doctor,
        "condition_risks": condition_risks,
        "relevant_condition_risks": [item for item in condition_risks if item.get("is_relevant") and item.get("education_message")],
    }


def build_symptom_condition_analysis_input(
    symptom_text: str,
    conditions_payload: dict,
    medications: list | None = None,
) -> str:
    return json.dumps(
        {
            "reported_symptom_or_update": str(symptom_text or "").strip(),
            "patient": {
                "name": conditions_payload.get("patient_name"),
                "conditions": conditions_payload.get("conditions") or [],
            },
            "current_medications": medications or [],
        },
        ensure_ascii=False,
        indent=2,
    )


def analyze_symptom_against_conditions(
    symptom_text: str,
    patient_id=None,
    medications: list | None = None,
) -> dict:
    """
    Cross-check a reported symptom against the patient's stored chronic conditions.
    Returns structured severity guidance and caregiver education messages.
    """
    symptom_text = str(symptom_text or "").strip()
    if not symptom_text:
        return normalize_symptom_condition_analysis({})

    payload = build_patient_conditions_payload(patient_id)
    if not payload.get("conditions"):
        return normalize_symptom_condition_analysis({
            "is_elevated_risk": False,
            "recommended_severity": "monitor",
            "needs_doctor": False,
            "condition_risks": [],
            "skipped": True,
        })

    user_input = build_symptom_condition_analysis_input(symptom_text, payload, medications)
    result = ask_ai(SYMPTOM_CONDITION_CROSSCHECK_PROMPT, user_input)
    if result.get("error"):
        logger.warning("Symptom-condition cross-check failed: %s", result.get("message"))
        return normalize_symptom_condition_analysis({
            "is_elevated_risk": False,
            "recommended_severity": "monitor",
            "needs_doctor": False,
            "condition_risks": [],
            "error": result.get("message"),
        })

    normalized = normalize_symptom_condition_analysis(result)
    normalized = enrich_symptom_condition_analysis(
        symptom_text,
        normalized,
        payload.get("conditions") or [],
        medications or [],
    )
    logger.info(
        "Symptom-condition cross-check: symptom=%s elevated=%s severity=%s relevant=%s",
        normalized.get("symptom_identified") or symptom_text[:80],
        normalized.get("is_elevated_risk"),
        normalized.get("recommended_severity"),
        len(normalized.get("relevant_condition_risks") or []),
    )
    return normalized


FALLBACK_CONDITION_SYMPTOM_RULES = (
    {
        "symptom_re": r"stiff|stiffness|joint|knee|hip|ache|aching|pain",
        "condition_re": r"osteoarthritis|arthritis|rheumat",
        "education": (
            "Joint stiffness or discomfort is common with {name}. Monitor for worsening pain, swelling, "
            "warmth, redness, fever, or reduced mobility — these may need a GP review."
        ),
        "severity_impact": "monitor",
    },
    {
        "symptom_re": r"stiff|stiffness|joint|knee|hip|ache|aching|pain",
        "condition_re": r"diabetes",
        "education": (
            "With diabetes, joint pain or stiffness warrants attention because of the risk of joint or nerve "
            "complications. Monitor for increased pain, swelling, warmth, redness, or fever, as these may "
            "indicate a more serious issue requiring medical attention."
        ),
        "severity_impact": "contact_doctor",
    },
    {
        "symptom_re": r"swell|swollen|swelling|edema|puffiness|fluid",
        "condition_re": r"heart failure|cardiac|kidney|renal|ckd|cirrhosis|liver",
        "education": (
            "New or worsening swelling can be important with {name}. Track ankle or leg swelling, weight gain, "
            "and breathlessness, and contact the GP if these worsen."
        ),
        "severity_impact": "contact_doctor",
    },
    {
        "symptom_re": r"fever|temperature|hot|burning up",
        "condition_re": r"diabetes|copd|kidney|renal|immunocomprom|transplant",
        "education": (
            "Fever can be more serious when someone has {name}. Monitor temperature, hydration, and overall "
            "condition closely, and contact the GP if fever persists or they seem unwell."
        ),
        "severity_impact": "contact_doctor",
    },
    {
        "symptom_re": r"confus|disorient|drowsy|sleepy|letharg",
        "condition_re": r"diabetes|dementia|kidney|renal|stroke|parkinson",
        "education": (
            "Changes in alertness can be significant with {name}. Watch for sudden worsening, fever, "
            "dehydration, or inability to wake them — contact the GP promptly if concerned."
        ),
        "severity_impact": "contact_doctor",
    },
)

MEDICATION_SYMPTOM_RULES = (
    {
        "symptom_re": r"bruise|bruising|bleed|bleeding|blood|nosebleed|hematoma",
        "med_re": r"warfarin|apixaban|rivaroxaban|edoxaban|dabigatran|heparin|enoxaparin|clopidogrel|aspirin|ticagrelor|prasugrel|anticoagul|blood thinner",
        "education": (
            "Unusual bruising or bleeding can be linked to blood-thinning medicines like {name}. "
            "Contact the GP promptly if bruising is new, widespread, or follows a minor injury."
        ),
        "severity_impact": "contact_doctor",
    },
    {
        "symptom_re": r"swell|swollen|swelling|edema|puffiness|fluid",
        "med_re": r"furosemide|bumetanide|torsemide|spironolactone|hydrochlorothiazide|diuretic",
        "education": (
            "Swelling while taking {name} may mean fluid balance needs review. "
            "Track ankle swelling, weight, and breathlessness, and contact the GP if swelling worsens."
        ),
        "severity_impact": "contact_doctor",
    },
)


def _condition_name_matches_pattern(condition_name: str, pattern: str) -> bool:
    return bool(re.search(pattern, str(condition_name or ""), re.I))


def _build_fallback_education_message(condition_name: str, body_template: str) -> str:
    name = str(condition_name or "").strip()
    body = body_template.format(name=name)
    return ensure_condition_education_format(name, body)


def apply_condition_relevance_fallback(
    symptom_text: str,
    analysis: dict,
    conditions: list,
) -> dict:
    """Add rule-based condition links when the AI cross-check missed obvious matches."""
    symptom_lower = str(symptom_text or "").lower()
    if not symptom_lower or not conditions:
        return analysis

    existing = {
        str(item.get("condition_name") or "").strip().lower()
        for item in (analysis.get("condition_risks") or [])
        if item.get("is_relevant")
    }
    merged_risks = [dict(item) for item in (analysis.get("condition_risks") or [])]
    known_names = {
        str(item.get("condition_name") or "").strip().lower()
        for item in merged_risks
    }

    for condition in conditions:
        condition_name = str(condition.get("name") or "").strip()
        if not condition_name:
            continue
        key = condition_name.lower()
        if key in existing:
            continue
        for rule in FALLBACK_CONDITION_SYMPTOM_RULES:
            if not re.search(rule["symptom_re"], symptom_lower, re.I):
                continue
            if not _condition_name_matches_pattern(condition_name, rule["condition_re"]):
                continue
            merged_risks.append({
                "condition_name": condition_name,
                "is_relevant": True,
                "severity_impact": rule["severity_impact"],
                "education_message": _build_fallback_education_message(
                    condition_name,
                    rule["education"],
                ),
            })
            known_names.add(key)
            break

    if not merged_risks:
        return analysis

    payload = dict(analysis)
    payload["condition_risks"] = merged_risks
    return normalize_symptom_condition_analysis(payload)


def build_medication_symptom_alerts(symptom_text: str, medications: list | None) -> list[dict]:
    """Rule-based links between reported symptoms and current medications."""
    symptom_lower = str(symptom_text or "").lower()
    if not symptom_lower or not medications:
        return []

    alerts = []
    seen = set()
    for med in medications:
        med_name = str(med.get("name") or med.get("medication_name") or "").strip()
        if not med_name:
            continue
        med_blob = " ".join(
            str(med.get(key) or "")
            for key in ("name", "medication_name", "dosage", "dosage_instructions", "timing", "schedule")
        )
        for rule in MEDICATION_SYMPTOM_RULES:
            if not re.search(rule["symptom_re"], symptom_lower, re.I):
                continue
            if not re.search(rule["med_re"], med_blob, re.I):
                continue
            key = med_name.lower()
            if key in seen:
                continue
            seen.add(key)
            alerts.append({
                "condition_name": med_name,
                "is_relevant": True,
                "severity_impact": rule["severity_impact"],
                "education_message": f"How {med_name} Relates to This Symptom: {rule['education'].format(name=med_name)}",
            })
            break
    return alerts


def enrich_symptom_condition_analysis(
    symptom_text: str,
    analysis: dict,
    conditions: list,
    medications: list | None = None,
) -> dict:
    enriched = apply_condition_relevance_fallback(symptom_text, analysis, conditions)
    med_alerts = build_medication_symptom_alerts(symptom_text, medications)
    if not med_alerts:
        return enriched

    merged_risks = [dict(item) for item in (enriched.get("condition_risks") or [])]
    existing_names = {
        str(item.get("condition_name") or "").strip().lower()
        for item in merged_risks
        if item.get("is_relevant")
    }
    for alert in med_alerts:
        key = str(alert.get("condition_name") or "").strip().lower()
        if key in existing_names:
            continue
        merged_risks.append(alert)

    payload = dict(enriched)
    payload["condition_risks"] = merged_risks
    normalized = normalize_symptom_condition_analysis(payload)
    normalized["medication_symptom_alerts"] = med_alerts
    return normalized


def extract_allergy_mentions_from_text(raw_text: str, limit: int = 8) -> list[str]:
    text = str(raw_text or "")
    if not text.strip():
        return []
    found = []
    seen = set()
    patterns = (
        r"(?:allerg(?:y|ies)|adverse reaction(?:s)?)\s*(?:to|:)?\s*([A-Za-z0-9][A-Za-z0-9\s\-/,]{1,60})",
        r"(?:NKDA|no known drug allergies)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            snippet = re.sub(r"\s+", " ", match.group(0)).strip(" .,;")
            key = snippet.lower()
            if key in seen:
                continue
            seen.add(key)
            found.append(snippet)
            if len(found) >= limit:
                return found
    return found


def fetch_recent_document_excerpts(patient_id=None, max_docs: int = 2, max_chars: int = 700) -> list[dict]:
    patient_id_int = _patient_id_int(resolve_patient_id(patient_id))
    if patient_id_int is None:
        return []
    try:
        response = (
            supabase.table("documents")
            .select("file_name, raw_text, created_at")
            .eq("patient_id", patient_id_int)
            .order("created_at", desc=True)
            .limit(max_docs)
            .execute()
        )
        excerpts = []
        for row in response.data or []:
            raw_text = str(row.get("raw_text") or "").strip()
            if not raw_text:
                continue
            excerpts.append({
                "file_name": str(row.get("file_name") or "Uploaded document"),
                "excerpt": raw_text[:max_chars],
            })
        return excerpts
    except Exception:
        return []


def get_patient_allergy_notes(patient_id=None) -> list[str]:
    allergies = []
    seen = set()
    latest_plan = get_latest_patient_plan(patient_id)
    if latest_plan:
        for item in extract_allergy_mentions_from_text(latest_plan.get("raw_text") or ""):
            key = item.lower()
            if key not in seen:
                seen.add(key)
                allergies.append(item)
    for doc in fetch_recent_document_excerpts(patient_id):
        for item in extract_allergy_mentions_from_text(doc.get("excerpt") or ""):
            key = item.lower()
            if key not in seen:
                seen.add(key)
                allergies.append(item)
    return allergies[:8]


def build_patient_report_timeline_context(
    patient_id=None,
    session_incidents: list | None = None,
    limit: int = 10,
) -> str:
    """Recent symptom reports from this session and stored shift logs."""
    lines = []
    for incident in (session_incidents or [])[-limit:]:
        text = str(incident.get("text") or incident.get("summary") or "").strip()
        if not text:
            continue
        stamp = str(incident.get("timestamp_display") or "").strip()
        severity = str(incident.get("severity") or "monitor")
        lines.append(f"- [{stamp or 'This session'}] {text} (severity: {severity})")

    remaining = max(0, limit - len(lines))
    if remaining:
        for row in fetch_symptom_shift_logs(patient_id, limit=limit * 2):
            summary = str(row.get("summary") or "").strip()
            if not summary:
                continue
            created_at = row.get("created_at") or ""
            try:
                stamp = datetime.fromisoformat(str(created_at).replace("Z", "+00:00")).strftime("%d %b %Y, %H:%M")
            except (ValueError, TypeError):
                stamp = str(created_at)[:16]
            severity = str(row.get("severity") or "monitor")
            lines.append(f"- [{stamp}] {summary} (severity: {severity})")
            if len(lines) >= limit:
                break

    if not lines:
        return "RECENT SYMPTOM & CARE TIMELINE: No prior reports on file yet."
    return "RECENT SYMPTOM & CARE TIMELINE (use to spot patterns and recurring symptoms):\n" + "\n".join(lines[:limit])


def build_condition_analysis_prompt_block(analysis: dict | None) -> str:
    if not analysis:
        return ""
    lines = ["PRE-COMPUTED SYMPTOM–PATIENT CROSS-CHECK (you MUST use this in your reply):"]
    symptom = str(analysis.get("symptom_identified") or "").strip()
    if symptom:
        lines.append(f"- Symptom identified: {symptom}")
    relevant = extract_relevant_condition_risks(analysis)
    if relevant:
        lines.append("- Relevant links for this patient:")
        for item in relevant:
            name = str(item.get("condition_name") or "").strip()
            message = str(item.get("education_message") or "").strip()
            if name and message:
                lines.append(f"  • {name}: {message}")
    else:
        lines.append("- No stored condition or medication links were flagged for this symptom.")
    recommended = str(analysis.get("recommended_severity") or "").strip()
    if recommended:
        lines.append(f"- Recommended severity from cross-check: {recommended}")
    lines.append(
        "- Your empathetic_advice MUST reference the patient's actual conditions and medications above "
        "when they relate to this report. Never give a generic answer when patient-specific links exist."
    )
    return "\n".join(lines)


def extract_relevant_condition_risks(analysis: dict | None) -> list:
    if not analysis:
        return []
    risks = analysis.get("relevant_condition_risks")
    if risks is not None:
        return list(risks)
    return [
        item
        for item in (analysis.get("condition_risks") or [])
        if item.get("is_relevant") and item.get("education_message")
    ]


def replace_patient_conditions(patient_id, conditions: list) -> None:
    patient_id_int = _patient_id_int(patient_id)
    if patient_id_int is None:
        return
    try:
        supabase.table("conditions").delete().eq("patient_id", patient_id_int).execute()
        payload = []
        for condition in conditions or []:
            name = str(condition.get("name") or "").strip()
            if not name:
                continue
            payload.append({
                "patient_id": patient_id_int,
                "name": name,
                "notes": _condition_fields_to_notes(condition),
            })
        if payload:
            supabase.table("conditions").insert(payload).execute()
    except Exception:
        pass


def replace_patient_medications(
    patient_id,
    active_medications: list,
    source_document_id=None,
) -> None:
    patient_id_int = _patient_id_int(patient_id)
    if patient_id_int is None:
        return
    try:
        supabase.table("medications").delete().eq("patient_id", patient_id_int).execute()
        payload = []
        for med in active_medications or []:
            name = str(med.get("name") or med.get("medication") or "").strip()
            if not name:
                continue
            row = {
                "patient_id": patient_id_int,
                "name": name,
                "dosage_instructions": format_medication_dosage_instructions(med),
            }
            if source_document_id is not None:
                row["source_document_id"] = int(source_document_id)
            payload.append(row)
        if payload:
            supabase.table("medications").insert(payload).execute()
    except Exception:
        pass


def save_patient_document_bundle(
    patient_id,
    *,
    file_name: str,
    raw_text: str,
    active_medications: list,
    conditions: list,
    caregiver_id=None,
) -> dict:
    """
    Insert into documents, then replace medications + conditions for this patient.
    Returns {"document_id": ...} on success.
    """
    patient_id_int = _patient_id_int(patient_id)
    if patient_id_int is None:
        return {}

    doc_payload = {
        "patient_id": patient_id_int,
        "file_name": file_name,
        "raw_text": raw_text,
    }
    if caregiver_id is not None and str(caregiver_id).isdigit():
        doc_payload["uploaded_by_caregiver_id"] = int(caregiver_id)

    document_id = None
    try:
        response = supabase.table("documents").insert(doc_payload).select("*").execute()
        if response.data:
            document_id = response.data[0].get("id")
    except Exception:
        pass

    replace_patient_medications(patient_id_int, active_medications, source_document_id=document_id)
    replace_patient_conditions(patient_id_int, conditions)
    return {"document_id": document_id}


def using_session_patient_store() -> bool:
    """Legacy hook — patients are always stored in Supabase now."""
    return False


def save_shift_log(
    caregiver_name,
    source,
    summary,
    severity,
    reported_at=None,
    caregiver_id=None,
    patient_id=None,
) -> bool:
    """
    Saves one event to the shift_logs table in Supabase, scoped to patient_id.
    Returns True when the row was stored.
    """
    summary = str(summary or "").strip()
    if not summary:
        logger.warning("save_shift_log skipped: empty summary for source=%s", source)
        return False

    resolved_patient_id = resolve_patient_id(patient_id)
    if not resolved_patient_id:
        logger.warning("save_shift_log skipped: missing patient_id for source=%s", source)
        return False

    patient_row = get_patient_by_id(resolved_patient_id)
    if should_block_test_entry_for_patient(
        resolved_patient_id,
        summary=summary,
        caregiver_name=caregiver_name,
        source=source,
        patient=patient_row,
    ):
        logger.warning(
            "Blocked internal test shift_log for production patient=%s source=%s",
            resolved_patient_id,
            source,
        )
        return False

    if reported_at:
        try:
            ts = datetime.fromisoformat(reported_at.replace("Z", "+00:00"))
            stamp = ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        except (ValueError, TypeError):
            stamp = reported_at
        summary = f"[{stamp}] {summary}"

    payload = {
        "caregiver_name": caregiver_name,
        "source": source,
        "summary": summary,
        "severity": severity,
        "patient_id": resolved_patient_id,
    }
    if caregiver_id:
        payload["caregiver_id"] = caregiver_id

    attempts = [dict(payload)]
    if caregiver_id:
        without_caregiver = dict(payload)
        without_caregiver.pop("caregiver_id", None)
        attempts.append(without_caregiver)

    legacy_summary = f"{shift_log_patient_marker(resolved_patient_id)} {summary}"
    legacy_payload = {
        "caregiver_name": caregiver_name,
        "source": source,
        "summary": legacy_summary,
        "severity": severity,
    }
    if caregiver_id:
        legacy_with_caregiver = dict(legacy_payload)
        legacy_with_caregiver["caregiver_id"] = caregiver_id
        attempts.append(legacy_with_caregiver)
    attempts.append(legacy_payload)

    for attempt in attempts:
        try:
            supabase.table("shift_logs").insert(attempt).execute()
            return True
        except Exception as exc:
            logger.debug(
                "save_shift_log insert failed for patient=%s source=%s: %s",
                resolved_patient_id,
                source,
                exc,
            )
    return False


MAX_STORED_CHAT_MESSAGES = 400
MAX_STORED_PHOTO_B64_CHARS = 350_000


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


def save_patient_care_report(
    patient_id,
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
    """Persist one Report & Ask incident keyed by patient profile."""
    patient_id = resolve_patient_id(patient_id)
    report_text = str(report_text or "").strip()
    summary = str(summary or report_text).strip()
    if not patient_id or not summary:
        return None

    patient_row = get_patient_by_id(patient_id)
    if should_block_test_entry_for_patient(
        patient_id,
        report_text=report_text,
        summary=summary,
        caregiver_name=caregiver_name,
        source=source,
        patient=patient_row,
    ):
        logger.warning(
            "Blocked internal test care entry for production patient=%s source=%s",
            patient_id,
            source,
        )
        return None

    local_saved = save_local_care_report(
        patient_id,
        report_text=report_text,
        summary=summary,
        severity=severity,
        source=source,
        reported_at=reported_at,
        caregiver_name=caregiver_name,
        caregiver_id=caregiver_id,
        photo_finding=photo_finding,
        photo_type=photo_type,
        image_b64=image_b64,
    )

    payload = {
        "patient_id": patient_id,
        "caregiver_id": caregiver_id,
        "caregiver_name": caregiver_name or "Caregiver",
        "source": source or "voice_report",
        "report_text": report_text or summary,
        "summary": summary,
        "severity": severity or "monitor",
        "reported_at": _normalize_reported_at_iso(reported_at),
        "photo_finding": photo_finding or None,
        "photo_type": photo_type or None,
        "has_photo": bool(image_b64),
    }
    try:
        response = (
            supabase.table("patient_care_reports")
            .insert(payload)
            .select("*")
            .execute()
        )
        if response.data:
            saved = response.data[0]
            if image_b64 and len(image_b64) <= MAX_STORED_PHOTO_B64_CHARS:
                try:
                    supabase.table("patient_care_report_photos").insert({
                        "report_id": saved["id"],
                        "image_b64": image_b64,
                    }).execute()
                except Exception as exc:
                    logger.debug("patient_care_report_photos insert failed: %s", exc)
            return saved
    except Exception as exc:
        logger.debug("save_patient_care_report supabase failed for patient=%s: %s", patient_id, exc)

    return local_saved


def fetch_patient_care_reports(patient_id=None, limit: int = 500) -> list:
    """Load durable care reports for one patient, oldest-first for timelines."""
    patient_id = resolve_patient_id(patient_id)
    if not patient_id:
        return []

    local_rows = fetch_local_care_reports(patient_id, limit=limit)
    remote_rows = []
    try:
        response = (
            supabase.table("patient_care_reports")
            .select("*")
            .eq("patient_id", patient_id)
            .order("reported_at", desc=False)
            .limit(limit)
            .execute()
        )
        remote_rows = response.data or []
    except Exception as exc:
        logger.debug("fetch_patient_care_reports supabase failed for %s: %s", patient_id, exc)

    merged = _merge_care_report_rows(local_rows, remote_rows)
    patient_row = get_patient_by_id(patient_id)
    merged = filter_production_care_rows(merged, patient_id, patient_row)
    if limit and len(merged) > limit:
        return merged[-limit:]
    return merged


def fetch_patient_care_report_photo(report_id, patient_id=None) -> str:
    if report_id is None:
        return ""
    local_photo = fetch_local_care_report_photo(report_id, patient_id=patient_id)
    if local_photo:
        return local_photo
    try:
        response = (
            supabase.table("patient_care_report_photos")
            .select("image_b64")
            .eq("report_id", report_id)
            .limit(1)
            .execute()
        )
        if response.data:
            return str(response.data[0].get("image_b64") or "")
    except Exception:
        pass
    return ""


def sanitize_chat_message_for_storage(message: dict) -> dict | None:
    if not isinstance(message, dict):
        return None
    if message.get("welcome"):
        return None
    role = str(message.get("role") or "").strip()
    if role not in ("user", "assistant"):
        return None
    stored = {
        "role": role,
        "content": str(message.get("content") or ""),
        "timestamp": message.get("timestamp") or "",
        "timestamp_display": message.get("timestamp_display") or "",
        "severity": message.get("severity"),
        "has_image": bool(message.get("has_image")),
        "context_report_count": message.get("context_report_count"),
        "condition_risk_alerts": message.get("condition_risk_alerts"),
    }
    return {key: value for key, value in stored.items() if value not in (None, "", [])}


def save_patient_chat_thread(patient_id, messages: list) -> bool:
    """Persist Report & Ask chat transcript for one patient profile."""
    patient_id = resolve_patient_id(patient_id)
    if not patient_id:
        return False
    storable = []
    for message in messages or []:
        cleaned = sanitize_chat_message_for_storage(message)
        if cleaned:
            storable.append(cleaned)
    storable = storable[-MAX_STORED_CHAT_MESSAGES:]
    local_saved = save_local_chat_thread(patient_id, storable)
    payload = {
        "patient_id": patient_id,
        "messages": storable,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        supabase.table("patient_chat_threads").upsert(payload).execute()
        return True
    except Exception as exc:
        logger.debug("save_patient_chat_thread supabase failed for patient=%s: %s", patient_id, exc)
        return local_saved


def fetch_patient_chat_thread(patient_id=None) -> list:
    patient_id = resolve_patient_id(patient_id)
    if not patient_id:
        return []

    local_messages = fetch_local_chat_thread(patient_id)
    patient_row = get_patient_by_id(patient_id)
    if local_messages:
        if is_designated_test_patient(patient_id, patient_row):
            return local_messages
        return [
            message for message in local_messages
            if isinstance(message, dict)
            and not is_internal_test_care_entry(
                text=str(message.get("content") or ""),
                summary=str(message.get("content") or ""),
                source="chat_message",
            )
        ]

    try:
        response = (
            supabase.table("patient_chat_threads")
            .select("messages")
            .eq("patient_id", patient_id)
            .limit(1)
            .execute()
        )
        if response.data:
            messages = response.data[0].get("messages") or []
            if isinstance(messages, list):
                return messages
    except Exception as exc:
        logger.debug("fetch_patient_chat_thread supabase failed for %s: %s", patient_id, exc)
    return []


def _production_shift_logs(rows, patient_id=None) -> list:
    patient_row = get_patient_by_id(patient_id) if patient_id else None
    return filter_production_shift_log_rows(rows or [], patient_id, patient_row)


def fetch_shift_logs(patient_id=None, limit: int = 250) -> list:
    """
    Load shift_logs for one patient when patient_id column exists.
    Falls back to summary markers or legacy unscoped rows for single-patient accounts.
    """
    patient_id = resolve_patient_id(patient_id)
    if patient_id:
        try:
            response = (
                supabase.table("shift_logs")
                .select("*")
                .eq("patient_id", patient_id)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            rows = response.data or []
            if rows:
                return _production_shift_logs(rows[:limit], patient_id)
        except Exception as exc:
            logger.debug("shift_logs query for patient %s failed: %s", patient_id, exc)

        try:
            response = (
                supabase.table("shift_logs")
                .select("*")
                .order("created_at", desc=True)
                .limit(max(limit * 4, 500))
                .execute()
            )
            all_rows = response.data or []
            marked_rows = [
                row for row in all_rows
                if summary_matches_shift_log_patient(row.get("summary"), patient_id)
            ]
            if marked_rows:
                return _production_shift_logs(marked_rows[:limit], patient_id)
            patients = list_account_patients()
            if len(patients) == 1 and str(patients[0].get("id")) == str(patient_id):
                legacy_rows = [
                    row for row in all_rows
                    if "[[patient:" not in str(row.get("summary") or "")
                ]
                if legacy_rows:
                    return _production_shift_logs(legacy_rows[:limit], patient_id)
        except Exception as exc:
            logger.debug("shift_logs marker fallback failed for %s: %s", patient_id, exc)

        try:
            patients = list_account_patients()
            if len(patients) == 1 and str(patients[0].get("id")) == str(patient_id):
                response = (
                    supabase.table("shift_logs")
                    .select("*")
                    .is_("patient_id", "null")
                    .order("created_at", desc=True)
                    .limit(limit)
                    .execute()
                )
                legacy_rows = response.data or []
                if legacy_rows:
                    return _production_shift_logs(legacy_rows, patient_id)
        except Exception as exc:
            logger.debug("shift_logs legacy null-patient query failed: %s", exc)
        return []

    try:
        response = (
            supabase.table("shift_logs")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return _production_shift_logs(response.data or [], None)
    except Exception:
        return []


def clean_json_response(raw_text):
    """
    GPT-4o sometimes wraps JSON in ```json ... ``` markdown blocks
    even when told not to. This strips that wrapper if present,
    then parses the result into a real Python dictionary.
    """
    text = raw_text.strip()

    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    return json.loads(text)


def ask_ai(system_prompt, user_content):
    """
    Sends a request to GPT-4o and returns a clean Python dictionary.
    user_content can be a string (text) or a list (text + image).
    If something fails, returns a safe fallback dictionary instead of crashing.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        )
        raw_text = response.choices[0].message.content
        if not raw_text or not str(raw_text).strip():
            return {
                "error": True,
                "message": "We couldn't process this right now. Please try again.",
                "details": "empty_model_response",
            }
        return clean_json_response(raw_text)

    except json.JSONDecodeError as e:
        logger.warning("ask_ai JSON parse failed: %s", e)
        return {
            "error": True,
            "message": "We couldn't read the AI response. Please try again.",
            "details": str(e),
        }
    except Exception as e:
        logger.exception("ask_ai request failed")
        return {
            "error": True,
            "message": "We couldn't process this right now. Please try again, or contact a healthcare provider if this is urgent.",
            "details": str(e)
        }


def _build_openai_user_content(user_text="", image_b64=None, media_type="image/jpeg"):
    if image_b64:
        parts = []
        if user_text:
            parts.append({"type": "text", "text": user_text})
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
        })
        return parts
    return user_text or "Please analyze the attached information."


def _build_claude_user_content(user_text="", image_b64=None, media_type="image/jpeg"):
    parts = []
    if user_text:
        parts.append({"type": "text", "text": user_text})
    if image_b64:
        parts.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": image_b64},
        })
    if not parts:
        parts.append({"type": "text", "text": "Please analyze the attached image."})
    return parts


def ask_ai_chat(system_prompt, user_text="", image_b64=None, media_type="image/jpeg", chat_history=None):
    """
    Chat agent call — prefers Claude 3.5 Sonnet for multimodal chat, falls back to GPT-4o.
    Returns a parsed JSON dictionary when possible, otherwise a plain-text answer dict.
    """
    history = []
    for msg in chat_history or []:
        role = msg.get("role")
        content = msg.get("content")
        if role in ("user", "assistant") and content:
            history.append({"role": role, "content": str(content)})

    if history and history[-1]["role"] == "user" and user_text:
        if history[-1]["content"].strip() == user_text.strip():
            history = history[:-1]

    try:
        if os.getenv("ANTHROPIC_API_KEY") and anthropic_client:
            try:
                messages = [
                    *history,
                    {
                        "role": "user",
                        "content": _build_claude_user_content(user_text, image_b64, media_type),
                    },
                ]
                response = anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1200,
                    system=system_prompt,
                    messages=messages,
                    timeout=30.0,
                )
                raw_text = response.content[0].text
            except Exception:
                messages = [{"role": "system", "content": system_prompt}, *history]
                messages.append({
                    "role": "user",
                    "content": _build_openai_user_content(user_text, image_b64, media_type),
                })
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    timeout=30.0,
                )
                raw_text = response.choices[0].message.content
        else:
            messages = [{"role": "system", "content": system_prompt}, *history]
            messages.append({
                "role": "user",
                "content": _build_openai_user_content(user_text, image_b64, media_type),
            })
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                timeout=30.0,
            )
            raw_text = response.choices[0].message.content

        try:
            return clean_json_response(raw_text)
        except json.JSONDecodeError:
            return {"answer": raw_text.strip(), "needs_doctor": False}

    except Exception as e:
        return {
            "error": True,
            "message": "We couldn't process this right now. Please try again, or contact a healthcare provider if this is urgent.",
            "details": str(e),
        }
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from io import BytesIO
import base64


SEVERITY_PDF_META = {
    "ok": ("OK", colors.HexColor("#D4EDDA"), colors.HexColor("#2D6A4F")),
    "monitor": ("MONITOR", colors.HexColor("#FEF3C7"), colors.HexColor("#92400E")),
    "contact_doctor": ("CONTACT DOCTOR", colors.HexColor("#FFEDD5"), colors.HexColor("#C2410C")),
    "emergency": ("EMERGENCY", colors.HexColor("#FEE2E2"), colors.HexColor("#B91C1C")),
    "urgent": ("EMERGENCY", colors.HexColor("#FEE2E2"), colors.HexColor("#B91C1C")),
}


def _normalize_pdf_severity(value):
    level = str(value or "monitor").strip().lower().replace(" ", "_")
    if level == "urgent":
        return "emergency"
    if level in SEVERITY_PDF_META:
        return level
    return "monitor"


def _pdf_escape(text):
    return (
        str(text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _pdf_image_from_b64(image_b64, max_width=5.0 * inch, max_height=3.5 * inch):
    if not image_b64:
        return None
    try:
        raw = base64.b64decode(image_b64)
        bio = BytesIO(raw)
        img = Image(bio)
        iw, ih = img.drawWidth, img.drawHeight
        if iw > max_width:
            ratio = max_width / iw
            iw, ih = max_width, ih * ratio
        if ih > max_height:
            ratio = max_height / ih
            iw, ih = iw * ratio, max_height
        img.drawWidth = iw
        img.drawHeight = ih
        return img
    except Exception:
        return None


def _photo_review_finding_label(review):
    finding = str(review.get("photo_finding") or "").strip().lower()
    source = str(review.get("source") or "").strip().lower()
    if finding == "normal":
        return "Normal — no concerning signs seen"
    if finding == "concern":
        return "Review recommended"
    if source == "pill_photo":
        if finding == "identified":
            return "Pill identified"
        return "Pill review"
    return "Reviewed"


def _photo_review_type_label(review):
    source = str(review.get("source") or "").strip().lower()
    if source == "pill_photo":
        return "Pill identification"
    return "Symptom review"


def generate_handover_pdf(
    timeline_events=None,
    connected_links=None,
    sbar_data=None,
    patient_label="Patient",
    photo_reviews=None,
):
    """
    GP handover sheet with severity-coded timeline, timestamps,
    connected-report links, optional compact SBAR summary, and photo reviews.
    """
    timeline_events = timeline_events or []
    connected_links = connected_links or []
    photo_reviews = photo_reviews or []
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.45 * inch,
        rightMargin=0.45 * inch,
        topMargin=0.4 * inch,
        bottomMargin=0.4 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "HandoverTitle",
        parent=styles["Title"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1A2B23"),
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "HandoverSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#7A7568"),
        alignment=TA_CENTER,
        spaceAfter=10,
    )
    section_style = ParagraphStyle(
        "HandoverSection",
        parent=styles["Heading2"],
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#1A2B23"),
        spaceBefore=6,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "HandoverBody",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1A2B23"),
    )
    small_style = ParagraphStyle(
        "HandoverSmall",
        parent=styles["Normal"],
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#5F6368"),
    )
    story = []

    generated_at = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
    page_note = "Printable summary"
    if photo_reviews:
        page_note = "Printable summary · Photo reviews on page 2"
    story.append(Paragraph("CareShield — GP Handover Sheet", title_style))
    story.append(Paragraph(
        f"{_pdf_escape(patient_label)} · Generated {generated_at} · {page_note}",
        subtitle_style,
    ))

    legend_rows = []
    for key in ("emergency", "contact_doctor", "monitor", "ok"):
        label, bg, fg = SEVERITY_PDF_META[key]
        legend_rows.append([
            Paragraph(f'<font color="{fg.hexval()}">●</font>', body_style),
            Paragraph(f"<b>{label}</b>", body_style),
        ])
    legend_table = Table(legend_rows, colWidths=[0.18 * inch, 0.95 * inch], hAlign="LEFT")
    legend_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
    ]))
    legend_wrap = Table([[legend_table]], colWidths=[6.6 * inch])
    legend_wrap.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F9F8F3")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E8E4DA")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(legend_wrap)
    story.append(Spacer(1, 8))

    if sbar_data:
        story.append(Paragraph("SBAR Summary", section_style))
        sbar_pairs = [
            ("Situation", sbar_data.get("situation", "")),
            ("Background", sbar_data.get("background", "")),
            ("Assessment", sbar_data.get("assessment", "")),
            ("Recommendation", sbar_data.get("recommendation", "")),
            ("Watch for", sbar_data.get("watch_for", "")),
        ]
        for label, content in sbar_pairs:
            if content:
                story.append(Paragraph(
                    f"<b>{_pdf_escape(label)}:</b> {_pdf_escape(content)}",
                    body_style,
                ))
        doctor_questions = sbar_data.get("doctor_questions") or []
        if doctor_questions:
            story.append(Spacer(1, 4))
            story.append(Paragraph("<b>Questions from test results</b>", body_style))
            for entry in doctor_questions:
                if isinstance(entry, dict):
                    question = entry.get("question") or entry.get("text") or ""
                    source = entry.get("from") or entry.get("title") or ""
                else:
                    question = str(entry)
                    source = ""
                if not question:
                    continue
                prefix = f"({_pdf_escape(source)}) " if source else ""
                story.append(Paragraph(f"• {prefix}{_pdf_escape(question)}", body_style))
        story.append(Spacer(1, 6))

    story.append(Paragraph("Symptom timeline", section_style))
    if timeline_events:
        table_data = [[
            Paragraph("<b>Time</b>", body_style),
            Paragraph("<b>Severity</b>", body_style),
            Paragraph("<b>Report</b>", body_style),
            Paragraph("<b>Carer</b>", body_style),
        ]]
        for event in timeline_events[-8:]:
            level = _normalize_pdf_severity(event.get("severity"))
            label, bg, fg = SEVERITY_PDF_META[level]
            time_text = _pdf_escape(event.get("timestamp_display") or "—")
            report_text = _pdf_escape(event.get("text") or "")[:220]
            carer_text = _pdf_escape(event.get("caregiver") or "—")
            severity_cell = Paragraph(
                f'<font color="{fg.hexval()}">●</font> <b>{label}</b>',
                body_style,
            )
            table_data.append([
                Paragraph(time_text, body_style),
                severity_cell,
                Paragraph(report_text, body_style),
                Paragraph(carer_text, small_style),
            ])
        timeline_table = Table(
            table_data,
            colWidths=[1.05 * inch, 1.05 * inch, 3.55 * inch, 0.95 * inch],
            repeatRows=1,
        )
        row_styles = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A2B23")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E8E4DA")),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        for row_idx, event in enumerate(timeline_events[-8:], start=1):
            level = _normalize_pdf_severity(event.get("severity"))
            _, bg, _ = SEVERITY_PDF_META[level]
            row_styles.append(("BACKGROUND", (1, row_idx), (1, row_idx), bg))
        timeline_table.setStyle(TableStyle(row_styles))
        story.append(timeline_table)
        if len(timeline_events) > 8:
            story.append(Paragraph(
                f"Showing latest 8 of {len(timeline_events)} logged reports.",
                small_style,
            ))
    else:
        story.append(Paragraph("No reports logged yet this session.", body_style))

    if connected_links:
        story.append(Spacer(1, 6))
        story.append(Paragraph("Connected reports", section_style))
        for link in connected_links[:4]:
            related_lines = []
            for prior in link.get("connected_to") or []:
                stamp = _pdf_escape(prior.get("timestamp_display") or prior.get("time") or "")
                text = _pdf_escape(prior.get("text") or prior.get("report") or "")[:90]
                related_lines.append(f"↳ {stamp} — {text}")
            related_html = "<br/>".join(related_lines) if related_lines else "↳ Earlier session reports"
            story.append(Paragraph(
                (
                    f"<b>{_pdf_escape(link.get('time', ''))}</b> · "
                    f"{_pdf_escape(link.get('report', ''))[:120]}<br/>"
                    f"<i>Linked because:</i> {_pdf_escape(link.get('reason', ''))}<br/>"
                    f"{related_html}"
                ),
                body_style,
            ))
            story.append(Spacer(1, 3))

    if photo_reviews:
        story.append(PageBreak())
        story.append(Paragraph("Photo reviews", section_style))
        story.append(Paragraph(
            "Caregiver photos submitted for symptom review or pill identification during this period.",
            small_style,
        ))
        story.append(Spacer(1, 4))
        for review in photo_reviews:
            level = _normalize_pdf_severity(review.get("severity"))
            label, bg, fg = SEVERITY_PDF_META[level]
            stamp = _pdf_escape(review.get("timestamp_display") or "—")
            review_type = _pdf_escape(_photo_review_type_label(review))
            finding = _pdf_escape(_photo_review_finding_label(review))
            carer = _pdf_escape(review.get("caregiver") or "—")
            note = _pdf_escape(review.get("text") or review.get("summary") or "")
            story.append(Paragraph(
                (
                    f"<b>{stamp}</b> · {review_type}<br/>"
                    f'<font color="{fg.hexval()}">●</font> <b>{label}</b> · {finding}<br/>'
                    f"<i>Reported by:</i> {carer}<br/>"
                    f"<i>Caregiver note:</i> {note[:220]}"
                ),
                body_style,
            ))
            img = _pdf_image_from_b64(review.get("image_b64"))
            if img:
                story.append(Spacer(1, 4))
                story.append(img)
            else:
                story.append(Paragraph("Photo unavailable for this entry.", small_style))
            story.append(Spacer(1, 10))

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "CareShield family handover · Not a clinical diagnosis · "
        "Share with GP or consultant at next contact",
        small_style,
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def generate_sbar_pdf(sbar_data):
    """Backward-compatible wrapper around the full handover PDF."""
    return generate_handover_pdf([], [], sbar_data)


def save_patient_plan(raw_text, medications, patient_id=None):
    """
    Saves the extracted hospital document text and structured medication summary.
    Scoped to patient_id so multiple patients do not share one plan history.
    """
    payload = {
        "raw_text": raw_text,
        "medications": medications,
        "patient_id": resolve_patient_id(patient_id),
    }
    try:
        supabase.table("patient_plan").insert(payload).execute()
    except Exception:
        payload.pop("patient_id", None)
        supabase.table("patient_plan").insert(payload).execute()


def get_latest_patient_plan(patient_id=None):
    """
    Retrieves the most recently uploaded patient plan for one patient, if any.
    Never falls back to another patient's plan.
    """
    patient_id = resolve_patient_id(patient_id)
    if not patient_id:
        return None
    try:
        response = (
            supabase.table("patient_plan")
            .select("*")
            .eq("patient_id", patient_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if response.data:
            return response.data[0]
    except Exception:
        pass
    return None


def get_patient_plans(patient_id=None):
    """
    Returns patient plans for one patient, most recent first.
    Never falls back to another patient's plans.
    """
    patient_id = resolve_patient_id(patient_id)
    if not patient_id:
        return []
    try:
        response = (
            supabase.table("patient_plan")
            .select("*")
            .eq("patient_id", patient_id)
            .order("created_at", desc=True)
            .execute()
        )
        return response.data or []
    except Exception:
        return []


def get_relevant_medical_context(query: str, match_count: int = 3) -> str:
    """
    Searches medical_knowledge for chunks relevant to the query
    and returns them as a formatted string to inject into the system prompt.
    """
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query,
        )
        query_embedding = response.data[0].embedding
        result = supabase.rpc("match_medical_knowledge", {
            "query_embedding": query_embedding,
            "match_count": match_count,
        }).execute()
        if not result.data:
            return ""
        chunks = []
        for row in result.data:
            if row.get("similarity", 0) > 0.3:
                chunks.append(f"### {row['title']}\n{row['content']}")
        if not chunks:
            return ""
        return "\n\n---\n\n".join(chunks)
    except Exception:
        return ""

def save_medication_reference(
    medication_name: str,
    image_b64: str,
    description: str = "",
    pill_strength: float | None = None,
    strength_unit: str = "mg",
    brand: str = "",
    pills_per_dose: int | None = None,
    patient_id=None,
    back_image_b64: str | None = None,
):
    """Saves a reference photo and pill strength metadata for one patient's medication."""
    if not description:
        payload_meta = {
            "pill_strength": pill_strength,
            "strength_unit": strength_unit,
            "brand": brand.strip(),
        }
        if pills_per_dose is not None:
            try:
                count = int(pills_per_dose)
                if 1 <= count <= 10:
                    payload_meta["pills_per_dose"] = count
            except (TypeError, ValueError):
                pass
        if back_image_b64:
            payload_meta["back_image_b64"] = back_image_b64
        description = json.dumps(payload_meta)
    payload = {
        "medication_name": medication_name,
        "image_b64": image_b64,
        "description": description,
        "patient_id": resolve_patient_id(patient_id),
    }
    try:
        supabase.table("medication_references").insert(payload).execute()
    except Exception:
        payload.pop("patient_id", None)
        supabase.table("medication_references").insert(payload).execute()


def upsert_medication_reference(
    medication_name: str,
    image_b64: str,
    pill_strength: float | None = None,
    strength_unit: str = "mg",
    brand: str = "",
    pills_per_dose: int | None = None,
    patient_id=None,
    back_image_b64: str | None = None,
):
    """Replace any existing reference for this medication on this patient, then save."""
    patient_id = resolve_patient_id(patient_id)
    try:
        existing = (
            supabase.table("medication_references")
            .select("id")
            .eq("patient_id", patient_id)
            .eq("medication_name", medication_name)
            .execute()
        )
    except Exception:
        existing = supabase.table("medication_references").select("id").eq("medication_name", medication_name).execute()
    for row in existing.data or []:
        supabase.table("medication_references").delete().eq("id", row["id"]).execute()
    save_medication_reference(
        medication_name=medication_name,
        image_b64=image_b64,
        pill_strength=pill_strength,
        strength_unit=strength_unit,
        brand=brand,
        pills_per_dose=pills_per_dose,
        patient_id=patient_id,
        back_image_b64=back_image_b64,
    )


def update_medication_reference(
    ref_id: int,
    image_b64: str | None = None,
    pill_strength: float | None = None,
    strength_unit: str = "mg",
    brand: str | None = None,
    pills_per_dose: int | None = None,
    back_image_b64: str | None = None,
):
    """Update pill strength metadata and optionally the reference photo."""
    response = supabase.table("medication_references").select("*").eq("id", ref_id).limit(1).execute()
    if not response.data:
        return
    ref = response.data[0]
    try:
        meta = json.loads(ref.get("description") or "{}")
        if not isinstance(meta, dict):
            meta = {}
    except (json.JSONDecodeError, TypeError):
        meta = {}
    if pill_strength is not None:
        meta["pill_strength"] = pill_strength
    if strength_unit:
        meta["strength_unit"] = strength_unit
    if brand is not None:
        meta["brand"] = brand.strip()
    if pills_per_dose is not None:
        try:
            count = int(pills_per_dose)
            if 1 <= count <= 10:
                meta["pills_per_dose"] = count
        except (TypeError, ValueError):
            pass
    if back_image_b64:
        meta["back_image_b64"] = back_image_b64
    updates = {"description": json.dumps(meta)}
    if image_b64:
        updates["image_b64"] = image_b64
    supabase.table("medication_references").update(updates).eq("id", ref_id).execute()

def get_medication_references(patient_id=None) -> list:
    """Returns saved medication reference photos for one patient."""
    patient_id = resolve_patient_id(patient_id)
    try:
        response = (
            supabase.table("medication_references")
            .select("*")
            .eq("patient_id", patient_id)
            .order("created_at")
            .execute()
        )
        return response.data or []
    except Exception:
        try:
            response = supabase.table("medication_references").select("*").order("created_at").execute()
            return response.data or []
        except Exception:
            return []


def delete_medication_reference(ref_id: int, patient_id=None):
    """Deletes a medication reference by id, scoped to patient when possible."""
    patient_id = resolve_patient_id(patient_id)
    try:
        supabase.table("medication_references").delete().eq("id", ref_id).eq("patient_id", patient_id).execute()
    except Exception:
        supabase.table("medication_references").delete().eq("id", ref_id).execute()


def log_medication_prn_taken(
    patient_id,
    medication_name: str,
    caregiver_id=None,
    notes: str = "",
):
    """Log one as-needed (PRN) dose taken today."""
    supabase.table("medication_logs").insert({
        "patient_id": resolve_patient_id(patient_id),
        "medication_name": medication_name,
        "scheduled_time": "PRN",
        "status": "taken",
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "caregiver_id": caregiver_id,
        "notes": notes or "",
    }).execute()


def get_medication_logs(patient_id=None):
    patient_id = resolve_patient_id(patient_id)
    try:
        response = (
            supabase.table("medication_logs")
            .select("*")
            .eq("patient_id", patient_id)
            .order("logged_at", desc=True)
            .execute()
        )
        return response.data or []
    except Exception:
        return []


def log_medication_taken(
    patient_id,
    medication_name: str,
    scheduled_time: str,
    caregiver_id=None,
):
    supabase.table("medication_logs").insert({
        "patient_id": resolve_patient_id(patient_id),
        "medication_name": medication_name,
        "scheduled_time": scheduled_time,
        "status": "taken",
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "caregiver_id": caregiver_id,
        "notes": "",
    }).execute()


def log_medication_missed(
    patient_id,
    medication_name: str,
    scheduled_time: str,
    caregiver_id=None,
):
    supabase.table("medication_logs").insert({
        "patient_id": resolve_patient_id(patient_id),
        "medication_name": medication_name,
        "scheduled_time": scheduled_time,
        "status": "missed",
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "caregiver_id": caregiver_id,
        "notes": "",
    }).execute()


MY_RESULTS_ASSISTANT_SYSTEM = (
    "You are a medical document assistant helping non-clinical family caregivers understand medical documents. "
    "You explain clearly and warmly in plain English, without diagnosing or alarming language. "
    "You always recommend discussing findings with the patient's doctor."
)

MY_RESULTS_EXTRACT_PROMPT = """Read the uploaded medical document (lab printout, imaging report, clinic letter, discharge summary, referral, ER note, medication notice, vaccination record, therapy note, portal after-visit summary, or photo/scan of any of these).

Do NOT assume a table of test / result / reference range exists. Use a narrative fallback for prose letters and mixed documents.

Respond with ONLY a JSON object:
{
  "documentType": "short human title e.g. Cardiology consultation letter",
  "documentCategory": "lab_panel|imaging|pathology|clinic_letter|discharge_summary|referral|er_summary|medication_notice|vaccination|therapy_note|mental_health|dental|other",
  "date": "date from document or Unknown",
  "source": "clinic/hospital/lab name or Unknown",
  "readability": "clear|partial|unreadable",
  "languageNote": "null, or note if non-English / translation needed",
  "patientIdentityNote": "null, or note if multiple patients or ambiguous identity",
  "limitations": ["optional list of caveats e.g. reference range not provided, conflicting info"],
  "newDiagnoses": [
    {"name": "diagnosis", "detail": "what was said", "isNew": true}
  ],
  "medicationChanges": [
    {"medication": "drug name", "changeType": "start|stop|dose_change|switch", "detail": "why/how changed"}
  ],
  "caregiverInstructions": [
    {"instruction": "plain text", "category": "red_flag|monitor|general"}
  ],
  "followUps": [
    {"description": "test/appointment", "dateKind": "explicit|relative|unspecified", "date": "calendar date if explicitly stated, else empty", "relativePhrase": "exact relative wording e.g. in 3 weeks, next month, or empty", "prep": "prep if any or empty"}
  ],
  "imagingFindings": [
    {"study": "e.g. chest X-ray", "finding": "summary", "status": "normal|abnormal|unclear"}
  ],
  "results": [
    {
      "name": "test name",
      "value": "numeric or text value",
      "unit": "unit if shown",
      "status": "normal|low|high|abnormal|not_provided",
      "referenceRange": "range if shown, else empty string"
    }
  ],
  "backgroundConditions": [
    {"name": "condition", "detail": "one brief line — only if mentioned in passing"}
  ],
  "labComment": "verbatim or near-verbatim interpretive lab comment / notes from the report (e.g. 'compared to prior panel', 'clinical correlation suggested'), or empty string",
  "priorComparisons": [
    {
      "category": "plain label e.g. Kidney function",
      "summary": "plain English: what changed compared to the prior result",
      "tests": ["Creatinine", "eGFR"],
      "currentValues": "optional shorthand for current values e.g. Cr 1.4, eGFR 42",
      "priorValues": "prior values exactly as stated e.g. Cr 1.1, eGFR 58",
      "priorDate": "date of prior panel if stated, else empty"
    }
  ]
}

EXTRACTION PRIORITY (most important first — do not let lower-priority items crowd out higher):
1. NEW or CHANGED diagnoses — language like "newly noted", "new onset", "first identified", "not previously documented", new AFib, new infection, etc.
2. NEW, STOPPED, or CHANGED medications — "start", "began", "discontinue", "increase dose", "switch from X to Y"
3. Explicit caregiver/patient instructions and red flags — "watch for", "seek care if", "call if", "monitor", weight/symptom thresholds, chest pain/breathlessness triggers
4. Scheduled follow-ups, procedures, repeat tests — with dates and purpose
5. Abnormal quantitative values — flagged high/low vs reference range OR described as elevated/low/abnormal in prose without a numeric range
6. Background/chronic conditions mentioned only in passing — at most one brief line each; NEVER treat these as the main story if items 1–4 exist

Detection examples:
- New diagnosis: "newly diagnosed atrial fibrillation", "first noted on this visit"
- Med change: "started apixaban", "increase furosemide to 40mg", "stop aspirin"
- Red flags: "seek urgent care if chest pain", "call if weight gain >3 lb in a day"
- Prose abnormality: "creatinine remains elevated" counts even without a table range
- Do NOT fabricate reference ranges; use status "not_provided" and note in limitations if range missing
- NUMERIC FLAGGING RULES for "results" array:
  • ONLY include true quantitative lab/vital measurements with a numeric value (e.g. potassium 5.8 mmol/L, glucose 142 mg/dL)
  • ONLY use status "high" or "low" when a reference range is shown in the document OR the value is clearly outside a standard range for that named test
  • Do NOT put rhythm/ECG/imaging/pathology impressions in "results" — route to imagingFindings (e.g. "atrial fibrillation", "murmur", "opacity", "fracture")
  • A ventricular rate mentioned alongside an arrhythmia is context, NOT a high/low lab value — if rate is 60–100 bpm it is NOT "high" even when the rhythm is abnormal
  • If a number has no valid reference range and is not a standalone quantitative test, omit it from "results" entirely
- DATE HANDLING for followUps (and any other dated items):
  • explicit: document states a calendar date → set dateKind "explicit" and put the date in "date"
  • relative: wording like "in 3 weeks", "next month", "in one week" → set dateKind "relative" and copy exact wording into relativePhrase (and/or date if that's where it appeared)
  • unspecified: no date or timeframe at all → set dateKind "unspecified" and leave date/relativePhrase empty — NEVER use "Unknown"
- Output PLAIN TEXT ONLY in all string fields — never HTML tags or markup

Edge cases:
- readability "unreadable" if scan/photo is illegible or text is mostly gibberish — return minimal arrays and set error guidance in limitations
- Non-English: set languageNote; still extract if you can, otherwise explain limitation
- Multiple patients in one document: set patientIdentityNote; do not blend data
- Pediatric: do not apply adult reference ranges; note in limitations if age-specific ranges unclear
- Mental health / sensitive topics: extract factually with calm tone markers in detail fields
- No abnormal values AND no new diagnoses/changes: return empty arrays where appropriate — that is valid
- Conflicting information: add to limitations rather than guessing
- LAB COMMENT / PRIOR PANEL: if the report includes an interpretive lab comment or compares to a prior panel (e.g. "compared to prior panel", "since last test", prior Cr/eGFR values), copy it into labComment and populate priorComparisons with structured before/after details — do NOT drop this text

If unreadable, return:
{"error": true, "message": "The document text could not be read.", "reason": "unreadable", "readability": "unreadable", ...empty arrays...}

If readable, NEVER return error just because there are no numeric lab values — clinic letters and imaging reports are valid."""

MY_RESULTS_EXPLAIN_PROMPT = """You receive structured extraction from a medical document for a family caregiver.

The payload includes:
- patientName, knownConditions
- allTestNames: exact test name strings from extraction — you MUST use these verbatim in testNames and valueExplanations.testName
- abnormalTestNames: subset that are high/low/abnormal — group and explain these
- hasKeyFindings: true when newDiagnoses, medicationChanges, followUps, or imagingFindings exist — affects noAbnormalValuesNote tone
- labComment: interpretive lab notes from the report (may include prior-panel comparisons)
- priorComparisons: structured before/after comparisons when the document states them
- requiresTrendCallouts: when true, trendCallouts is REQUIRED (see TREND RULES)

Write grouped plain-English explanations and doctor questions. Respond with ONLY a JSON object:
{
  "explanation": "1-2 sentence overview ONLY — document type + single biggest takeaway. Do NOT repeat group details here.",
  "trendCallouts": [
    {
      "title": "short label e.g. Kidney function changed since last test",
      "summary": "plain English: what changed; define any clinical terms on first use",
      "priorValues": "prior values exactly as stated in document e.g. Cr 1.1, eGFR 58"
    }
  ],
  "resultGroups": [
    {
      "category": "plain-language heading e.g. Kidney function",
      "urgency": "discuss_soon|discuss_at_visit",
      "groupSummary": "2-4 sentences: what these tests measure together and what the pattern suggests for this patient",
      "relatedTo": "optional — category name of another resultGroup when clinically linked (e.g. Electrolytes relatedTo Kidney function when high potassium matters because kidney function is reduced), or null",
      "testNames": ["Creatinine", "eGFR"],
      "valueExplanations": [
        {
          "testName": "Creatinine",
          "whatItMeasures": "one sentence, 8th-grade reading level",
          "whatThisResultSuggests": "1-2 sentences for THIS abnormal value only",
          "defineTerms": "optional plain definition if a clinical term appears — else null"
        }
      ]
    }
  ],
  "questions": [
    {
      "text": "specific question tied to a category and values",
      "relatedCategory": "Kidney function",
      "relatedTests": ["Creatinine", "eGFR"]
    }
  ],
  "urgentCareInstructions": "null OR explicit red-flag / seek-care instructions in caregiver-friendly language",
  "noAbnormalValuesNote": "null OR one brief sentence ONLY about the absence of flagged numeric lab values — see NO-ABNORMAL LAB NOTE rules"
}

TEST NAME RULES (critical):
- testNames and valueExplanations.testName MUST copy strings EXACTLY from allTestNames / abnormalTestNames in the payload
- NEVER invent, abbreviate, or rename tests (e.g. use "Creatinine" not "Cr" unless that exact string is in allTestNames)
- relatedTests in questions must also use exact strings from allTestNames

GROUPING RULES:
- Create resultGroups only for abnormal values (from abnormalTestNames) and/or clearly abnormal imaging findings
- EVERY name in abnormalTestNames MUST appear in testNames of exactly one resultGroup — do not skip any flagged value
- Group related tests under plain-language category headings (e.g. Kidney function, Blood sugar, Blood counts, Cholesterol, Electrolytes)
- A single category MAY include tests pointing in different clinical directions (e.g. Blood counts with low hemoglobin/hematocrit suggesting anemia AND high WBC suggesting possible infection) — each test still gets its own accurate valueExplanations; groupSummary should acknowledge both patterns when they coexist
- Potassium, sodium, and similar electrolytes belong in an "Electrolytes" group (or Kidney function when clearly related) — never omit an abnormal electrolyte
- When kidney function is reduced and potassium is high, set relatedTo on the Electrolytes group to "Kidney function" and mention the link in groupSummary — only when both are in the extract
- Each valueExplanations entry is for ONE abnormal test; include whatItMeasures + whatThisResultSuggests
- urgency "discuss_soon" = worth calling/booking before the next routine visit; "discuss_at_visit" = mention at the next scheduled appointment
- ORDER resultGroups with discuss_soon groups first, then discuss_at_visit

TREND RULES:
- When requiresTrendCallouts is true, you MUST include at least one trendCallout for every priorComparisons entry — use labComment and priorComparisons as your source
- Do NOT treat trends as optional when priorComparisons or labComment contains explicit before/after language ("compared to prior panel", "since last test", stated prior values)
- trendCallouts render above groups — a change over time usually matters more than a single number
- Do not bury trend information only inside groupSummary when requiresTrendCallouts is true

QUESTION RULES:
- At least one question per major abnormal group; reference specific values and goals when the document provides them
- BAD: "What steps can we take to better manage these?"
- GOOD: "Given Eleanor's LDL is 148 (goal under 100), should we consider starting or adjusting a statin?"
- relatedCategory and relatedTests must match the groups you created

NO-ABNORMAL PATH:
- If abnormalTestNames is empty AND there are no abnormal imaging findings AND nothing new to action, set resultGroups and trendCallouts to [] and populate noAbnormalValuesNote instead
- Still provide a brief explanation and 2-3 general follow-up questions if new diagnoses, med changes, or follow-ups exist

NO-ABNORMAL LAB NOTE (noAbnormalValuesNote):
- This note is ONLY about numeric lab values — not whether the overall visit was good or bad
- Check hasKeyFindings in the payload (new diagnoses, medication changes, follow-ups, or imaging findings)
- When hasKeyFindings is true: use neutral factual wording only (e.g. "This document did not include numeric lab results to flag."). Do NOT use good-news, reassuring, or "focus on managing..." framing that could overshadow important diagnoses or medication changes already captured above
- When hasKeyFindings is false AND there is nothing else to action: a warm reassuring sentence is fine (e.g. no abnormal labs and nothing new requiring follow-up)

PRIORITY (same as extraction):
1. NEW diagnoses and medication changes first in overview and questions
2. Then follow-ups / red flags
3. Then abnormal labs/imaging groups
4. Background/chronic conditions: brief context only

TONE: 8th-grade reading level, warm, not alarmist. Define abbreviations on first use. PLAIN TEXT ONLY — no HTML.

If you cannot produce a valid response, return:
{"error": true, "message": "We couldn't generate an explanation right now. Please try again."}"""

MY_RESULTS_UNCOVERED_EXPLAIN_PROMPT = """You receive lab results that were not placed in the main grouped explanation.
Write specific per-value plain-English explanations for a family caregiver.

The payload includes:
- patientName, knownConditions
- uncoveredTests: exact test names that need explanations
- results: the full result rows for those tests only (value, unit, status, referenceRange)

Respond with ONLY a JSON object:
{
  "valueExplanations": [
    {
      "testName": "exact name from uncoveredTests",
      "whatItMeasures": "one sentence, 8th-grade reading level",
      "whatThisResultSuggests": "1-2 sentences specific to THIS patient's value and status — e.g. high WBC may suggest infection/inflammation, low hemoglobin may suggest anemia",
      "defineTerms": "optional plain definition, else null"
    }
  ]
}

RULES:
- testName MUST match uncoveredTests exactly
- whatItMeasures and whatThisResultSuggests are REQUIRED for every test — no generic placeholders
- Use the actual value, unit, and reference range from the results rows
- PLAIN TEXT ONLY — no HTML

If you cannot produce a valid response, return:
{"error": true, "message": "We couldn't generate explanations right now."}"""

MY_RESULTS_PRIOR_COMPARISON_RE = re.compile(
    r"compared to|prior panel|since last|previous (?:test|result|panel|value)|"
    r"changed since|vs\.?\s+prior|from prior|earlier (?:test|result|panel)",
    re.I,
)

MY_RESULTS_NO_RESULTS_MESSAGE = (
    "We couldn't extract anything useful from this document. "
    "Try uploading a clearer photo or PDF, or a document with visit findings, lab values, or instructions from the clinic."
)

MY_RESULTS_DATE_NOT_SPECIFIED = "date not specified"

MY_RESULTS_DOCUMENT_DATE_FORMATS = (
    "%d %B %Y",
    "%d %b %Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%d.%m.%Y",
)

MY_RESULTS_WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "eight": 8,
    "twelve": 12,
    "a": 1,
    "an": 1,
}


def sanitize_my_results_plain_text(value) -> str:
    """Strip HTML/markup from model text fields before display or storage."""
    text = str(value or "").strip()
    if not text or text.lower() in ("null", "none"):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_my_results_anchor_date(date_text: str) -> date | None:
    cleaned = sanitize_my_results_plain_text(date_text)
    if not cleaned or cleaned.lower() in ("unknown", "date not specified"):
        return None
    for fmt in MY_RESULTS_DOCUMENT_DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    iso_match = re.match(r"^(\d{4}-\d{2}-\d{2})", cleaned)
    if iso_match:
        try:
            return datetime.strptime(iso_match.group(1), "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def _word_to_int(token: str) -> int | None:
    token = str(token or "").strip().lower()
    if token.isdigit():
        return int(token)
    return MY_RESULTS_WORD_NUMBERS.get(token)


def detect_relative_offset_phrase(text: str) -> tuple[str, timedelta] | None:
    cleaned = sanitize_my_results_plain_text(text)
    if not cleaned:
        return None
    lower = cleaned.lower()

    patterns = [
        (r"\b(?:in|within)\s+(\d+|one|two|three|four|five|six|eight|twelve|a|an)\s+weeks?\b", "week"),
        (r"\b(?:in|within)\s+(\d+|one|two|three|four|five|six|eight|twelve|a|an)\s+months?\b", "month"),
        (r"\b(?:in|within)\s+(\d+|one|two|three|four|five|six|eight|twelve|a|an)\s+days?\b", "day"),
        (r"\bnext\s+week\b", "next_week"),
        (r"\bnext\s+month\b", "next_month"),
        (r"\bin\s+a\s+fortnight\b", "fortnight"),
        (r"\bin\s+one\s+fortnight\b", "fortnight"),
    ]
    for pattern, unit in patterns:
        match = re.search(pattern, lower)
        if not match:
            continue
        if unit == "next_week":
            return ("next week", timedelta(weeks=1))
        if unit == "next_month":
            return ("next month", timedelta(days=30))
        if unit == "fortnight":
            return ("in 2 weeks", timedelta(weeks=2))
        amount = _word_to_int(match.group(1))
        if not amount:
            continue
        if unit == "week":
            phrase = f"in {amount} week{'s' if amount != 1 else ''}"
            return (phrase, timedelta(weeks=amount))
        if unit == "month":
            phrase = f"in {amount} month{'s' if amount != 1 else ''}"
            return (phrase, timedelta(days=amount * 30))
        phrase = f"in {amount} day{'s' if amount != 1 else ''}"
        return (phrase, timedelta(days=amount))
    return None


def format_my_results_display_date(value: date) -> str:
    text = value.strftime("%d %b %Y")
    return text[1:] if text.startswith("0") else text


def infer_my_results_date_kind(entry: dict) -> str:
    kind = str(entry.get("dateKind") or "").strip().lower()
    if kind in ("explicit", "relative", "unspecified"):
        return kind
    date_text = sanitize_my_results_plain_text(entry.get("date"))
    relative_text = sanitize_my_results_plain_text(entry.get("relativePhrase"))
    description = sanitize_my_results_plain_text(entry.get("description"))
    if relative_text or detect_relative_offset_phrase(date_text) or detect_relative_offset_phrase(description):
        return "relative"
    if date_text and parse_my_results_anchor_date(date_text):
        return "explicit"
    if date_text:
        return "relative"
    return "unspecified"


def resolve_my_results_follow_up_entry(entry: dict, anchor: date | None) -> dict:
    resolved = dict(entry or {})
    date_kind = infer_my_results_date_kind(resolved)
    date_text = sanitize_my_results_plain_text(resolved.get("date"))
    relative_text = sanitize_my_results_plain_text(resolved.get("relativePhrase"))
    description = sanitize_my_results_plain_text(resolved.get("description"))

    if date_kind == "explicit":
        explicit_date = parse_my_results_anchor_date(date_text)
        if explicit_date:
            resolved["dateKind"] = "explicit"
            resolved["dateDisplay"] = format_my_results_display_date(explicit_date)
            return resolved
        date_kind = "relative" if (relative_text or detect_relative_offset_phrase(date_text)) else "unspecified"

    if date_kind == "relative":
        relative = detect_relative_offset_phrase(relative_text) or detect_relative_offset_phrase(date_text) or detect_relative_offset_phrase(description)
        if relative and anchor:
            phrase, offset = relative
            approx = anchor + offset
            display_phrase = phrase[3:] if phrase.startswith("in ") else phrase
            resolved["dateKind"] = "relative"
            resolved["dateDisplay"] = f"~{display_phrase} from visit (approx. {format_my_results_display_date(approx)})"
            return resolved
        if relative:
            phrase, _offset = relative
            display_phrase = phrase[3:] if phrase.startswith("in ") else phrase
            resolved["dateKind"] = "relative"
            resolved["dateDisplay"] = f"~{display_phrase} from visit (exact date not confirmed in letter)"
            return resolved

    resolved["dateKind"] = "unspecified"
    resolved["dateDisplay"] = MY_RESULTS_DATE_NOT_SPECIFIED
    return resolved


MY_RESULTS_QUALITATIVE_NAME_RE = re.compile(
    r"\b("
    r"ecg|ekg|electrocardiogram|"
    r"rhythm|arrhythmia|arrhythm|afib|a\.?fib|atrial fibrillation|"
    r"murmur|heart sounds?|"
    r"x-?ray|radiograph|ct scan|mri|ultrasound|sonograph|echo(?:cardiogram)?|"
    r"impression|pathology|histology|biopsy|cytology|"
    r"fracture|opacity|infiltrate|consolidation|mass|nodule|lesion|"
    r"malignant|benign|carcinoma|"
    r"ventricular rate"
    r")\b",
    re.I,
)

MY_RESULTS_QUALITATIVE_VALUE_RE = re.compile(
    r"\b("
    r"atrial fibrillation|a\.?fib|afib|"
    r"arrhythmia|arrhythm|sinus rhythm|irregular|"
    r"murmur|tachycardia|bradycardia|"
    r"fracture|opacity|infiltrate|consolidation|mass|nodule|"
    r"malignant|benign|normal sinus"
    r")\b",
    re.I,
)

MY_RESULTS_STANDARD_RANGES = {
    "heart rate": (60.0, 100.0),
    "pulse": (60.0, 100.0),
    "pulse rate": (60.0, 100.0),
    "resting heart rate": (60.0, 100.0),
    "respiratory rate": (12.0, 20.0),
    "resp rate": (12.0, 20.0),
    "oxygen saturation": (95.0, 100.0),
    "spo2": (95.0, 100.0),
    "o2 saturation": (95.0, 100.0),
}


def _canonical_my_results_test_name(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", str(name or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_my_results_numeric_value(value: str, unit: str = "") -> float | None:
    text = sanitize_my_results_plain_text(f"{value} {unit}")
    if not text or text in ("—", "-"):
        return None
    match = re.search(r"(-?\d+(?:\.\d+)?)", text.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_my_results_reference_range(reference_range: str) -> tuple[float, float] | None:
    text = sanitize_my_results_plain_text(reference_range)
    if not text:
        return None
    text = text.replace("–", "-").replace("—", "-").replace(" to ", "-")
    between = re.search(
        r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)",
        text,
    )
    if between:
        low = float(between.group(1))
        high = float(between.group(2))
        if low > high:
            low, high = high, low
        return low, high
    upper = re.search(r"(?:<=|<|≤)\s*(-?\d+(?:\.\d+)?)", text)
    if upper:
        return float("-inf"), float(upper.group(1))
    lower = re.search(r"(?:>=|>|≥)\s*(-?\d+(?:\.\d+)?)", text)
    if lower:
        return float(lower.group(1)), float("inf")
    return None


def get_my_results_reference_range(name: str, reference_range: str) -> tuple[float, float] | None:
    explicit = parse_my_results_reference_range(reference_range)
    if explicit:
        return explicit
    canonical = _canonical_my_results_test_name(name)
    return MY_RESULTS_STANDARD_RANGES.get(canonical)


def is_qualitative_my_results_entry(name: str, value: str, unit: str = "") -> bool:
    combined = f"{name} {value} {unit}".strip()
    if not combined:
        return False
    if MY_RESULTS_QUALITATIVE_NAME_RE.search(combined):
        return True
    if MY_RESULTS_QUALITATIVE_VALUE_RE.search(value):
        return True
    numeric = extract_my_results_numeric_value(value, unit)
    if numeric is None:
        return True
    prose = re.sub(r"[\d.,\s/\-–—%]", "", value)
    if len(prose) >= 4:
        return True
    return False


def derive_my_results_numeric_status(
    *,
    name: str,
    value: str,
    unit: str,
    status: str,
    reference_range: str,
) -> str | None:
    """Return corrected status, or None when the row should leave the numeric table."""
    status = str(status or "normal").strip().lower()
    numeric = extract_my_results_numeric_value(value, unit)
    if numeric is None:
        return None

    ref = get_my_results_reference_range(name, reference_range)
    if not ref:
        return None

    low, high = ref
    if numeric < low:
        return "low"
    if numeric > high:
        return "high"
    return "normal"


def my_results_finding_already_captured(normalized: dict, study: str, finding: str) -> bool:
    study_l = study.lower().strip()
    finding_l = finding.lower().strip()
    for dx in normalized.get("newDiagnoses") or []:
        dx_name = str(dx.get("name") or "").lower()
        if dx_name and (dx_name in finding_l or finding_l in dx_name):
            return True
    for img in normalized.get("imagingFindings") or []:
        existing_study = str(img.get("study") or "").lower()
        existing_finding = str(img.get("finding") or "").lower()
        if finding_l and finding_l == existing_finding:
            return True
        if study_l and finding_l and study_l in existing_study and finding_l in existing_finding:
            return True
    return False


def normalize_my_results_lab_entry(item: dict, normalized: dict) -> dict | None:
    """Validate one lab row; reroute qualitative findings and drop unflaggable numbers."""
    name = str(item.get("name") or "").strip()
    value = str(item.get("value") or "—").strip()
    unit = str(item.get("unit") or "").strip()
    status = str(item.get("status") or "normal").strip().lower()
    reference_range = str(item.get("referenceRange") or "").strip()

    if not name:
        return None

    if is_qualitative_my_results_entry(name, value, unit):
        finding_text = value if value not in ("—", "-", "") else name
        study = name or "Finding"
        abnormal = status in ("high", "low", "abnormal") or bool(
            MY_RESULTS_QUALITATIVE_VALUE_RE.search(finding_text)
        )
        if not my_results_finding_already_captured(normalized, study, finding_text):
            normalized["imagingFindings"].append({
                "study": study,
                "finding": finding_text,
                "status": "abnormal" if abnormal else "unclear",
            })
        return None

    corrected_status = derive_my_results_numeric_status(
        name=name,
        value=value,
        unit=unit,
        status=status,
        reference_range=reference_range,
    )
    if corrected_status is None:
        return None

    ref = get_my_results_reference_range(name, reference_range)
    display_range = reference_range
    if ref and not display_range:
        low, high = ref
        if low > float("-inf") and high < float("inf"):
            low_label = str(int(low)) if float(low).is_integer() else str(low)
            high_label = str(int(high)) if float(high).is_integer() else str(high)
            display_range = f"{low_label}–{high_label}"

    return {
        "name": name,
        "value": value,
        "unit": unit,
        "status": corrected_status,
        "referenceRange": display_range,
    }


def finalize_my_results_extract(normalized: dict) -> dict:
    anchor = parse_my_results_anchor_date(normalized.get("date", ""))
    normalized["documentDate"] = anchor.isoformat() if anchor else None
    normalized["results"] = [
        row
        for row in (normalized.get("results") or [])
        if isinstance(row, dict) and row.get("name")
    ]
    normalized["hasLabValues"] = bool(normalized["results"])
    normalized["followUps"] = [
        resolve_my_results_follow_up_entry(item, anchor)
        for item in (normalized.get("followUps") or [])
    ]

    for key in ("documentType", "source", "languageNote", "patientIdentityNote", "labComment"):
        if normalized.get(key) is not None:
            normalized[key] = sanitize_my_results_plain_text(normalized.get(key)) or normalized.get(key)

    normalized["priorComparisons"] = _normalize_my_results_prior_comparisons(
        normalized.get("priorComparisons"),
        lab_comment=normalized.get("labComment") or "",
    )

    normalized["limitations"] = [
        sanitize_my_results_plain_text(item)
        for item in (normalized.get("limitations") or [])
        if sanitize_my_results_plain_text(item)
    ]

    for collection in ("newDiagnoses", "medicationChanges", "caregiverInstructions", "backgroundConditions", "imagingFindings"):
        cleaned_items = []
        for item in normalized.get(collection) or []:
            if not isinstance(item, dict):
                continue
            cleaned = {
                key: sanitize_my_results_plain_text(value) if isinstance(value, str) else value
                for key, value in item.items()
            }
            cleaned_items.append(cleaned)
        normalized[collection] = cleaned_items

    return normalized


def enrich_my_results_record(record: dict | None) -> dict | None:
    """Re-normalize stored records so legacy payloads pick up date display and flags."""
    if not record:
        return None
    merged = dict(record)
    normalized = normalize_my_results_extract(merged)
    normalized.update(normalize_my_results_explain(
        merged,
        extract=normalized,
        generate_missing_explanations=False,
    ))
    for passthrough in ("file_name", "document_id", "uploaded_at"):
        if merged.get(passthrough) not in (None, ""):
            normalized[passthrough] = merged[passthrough]
    return normalized


def _my_results_list(value) -> list:
    return list(value) if isinstance(value, list) else []


def normalize_my_results_extract(raw: dict | None) -> dict:
    raw = dict(raw or {})
    normalized = {
        "documentType": str(raw.get("documentType") or "Medical document").strip(),
        "documentCategory": str(raw.get("documentCategory") or "other").strip().lower(),
        "date": str(raw.get("date") or "Unknown").strip(),
        "source": str(raw.get("source") or "Unknown").strip(),
        "readability": str(raw.get("readability") or "clear").strip().lower(),
        "languageNote": raw.get("languageNote"),
        "patientIdentityNote": raw.get("patientIdentityNote"),
        "limitations": [
            str(item).strip()
            for item in _my_results_list(raw.get("limitations"))
            if str(item).strip()
        ],
        "newDiagnoses": [],
        "medicationChanges": [],
        "caregiverInstructions": [],
        "followUps": [],
        "imagingFindings": [],
        "results": [],
        "backgroundConditions": [],
        "labComment": "",
        "priorComparisons": [],
    }

    for item in _my_results_list(raw.get("newDiagnoses")):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        normalized["newDiagnoses"].append({
            "name": name,
            "detail": str(item.get("detail") or "").strip(),
            "isNew": bool(item.get("isNew", True)),
        })

    for item in _my_results_list(raw.get("medicationChanges")):
        if not isinstance(item, dict):
            continue
        medication = str(item.get("medication") or item.get("name") or "").strip()
        if not medication:
            continue
        change_type = str(item.get("changeType") or "start").strip().lower()
        normalized["medicationChanges"].append({
            "medication": medication,
            "changeType": change_type,
            "detail": str(item.get("detail") or "").strip(),
        })

    for item in _my_results_list(raw.get("caregiverInstructions")):
        if not isinstance(item, dict):
            continue
        instruction = str(item.get("instruction") or "").strip()
        if not instruction:
            continue
        category = str(item.get("category") or "general").strip().lower()
        if category not in ("red_flag", "monitor", "general"):
            category = "general"
        normalized["caregiverInstructions"].append({
            "instruction": instruction,
            "category": category,
        })

    for item in _my_results_list(raw.get("followUps")):
        if not isinstance(item, dict):
            continue
        description = str(item.get("description") or "").strip()
        if not description:
            continue
        normalized["followUps"].append({
            "description": description,
            "dateKind": str(item.get("dateKind") or "").strip().lower(),
            "date": sanitize_my_results_plain_text(item.get("date")),
            "relativePhrase": sanitize_my_results_plain_text(item.get("relativePhrase")),
            "prep": sanitize_my_results_plain_text(item.get("prep")),
        })

    for item in _my_results_list(raw.get("imagingFindings")):
        if not isinstance(item, dict):
            continue
        study = str(item.get("study") or "").strip()
        finding = str(item.get("finding") or "").strip()
        if not study and not finding:
            continue
        status = str(item.get("status") or "unclear").strip().lower()
        normalized["imagingFindings"].append({
            "study": study or "Imaging",
            "finding": finding,
            "status": status,
        })

    for item in _my_results_list(raw.get("results")):
        if not isinstance(item, dict):
            continue
        lab_entry = normalize_my_results_lab_entry(item, normalized)
        if lab_entry:
            normalized["results"].append(lab_entry)

    for item in _my_results_list(raw.get("backgroundConditions")):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        normalized["backgroundConditions"].append({
            "name": name,
            "detail": sanitize_my_results_plain_text(item.get("detail")),
        })

    normalized["labComment"] = sanitize_my_results_plain_text(raw.get("labComment"))
    normalized["priorComparisons"] = _my_results_list(raw.get("priorComparisons"))

    return finalize_my_results_extract(normalized)


def my_results_has_actionable_content(extract: dict | None) -> bool:
    extract = normalize_my_results_extract(extract)
    if extract.get("readability") == "unreadable":
        return False
    priority_keys = (
        "newDiagnoses",
        "medicationChanges",
        "caregiverInstructions",
        "followUps",
        "imagingFindings",
        "results",
    )
    return any(extract.get(key) for key in priority_keys)


def count_my_results_review_items(extract: dict | None) -> int:
    extract = normalize_my_results_extract(extract)
    count = sum(
        1
        for row in extract.get("results") or []
        if str(row.get("status") or "normal").strip().lower() not in ("normal", "not_provided")
    )
    count += len(extract.get("newDiagnoses") or [])
    count += len(extract.get("medicationChanges") or [])
    count += sum(
        1
        for item in extract.get("caregiverInstructions") or []
        if item.get("category") == "red_flag"
    )
    count += sum(
        1
        for item in extract.get("imagingFindings") or []
        if str(item.get("status") or "").strip().lower() == "abnormal"
    )
    return count


def my_results_abnormal_statuses() -> frozenset[str]:
    return frozenset({"high", "low", "abnormal"})


def my_results_extract_test_names(extract: dict | None) -> list[str]:
    names = []
    seen = set()
    for row in (extract or {}).get("results") or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        names.append(name)
    return names


def my_results_abnormal_test_names(extract: dict | None) -> list[str]:
    abnormal = my_results_abnormal_statuses()
    names = []
    seen = set()
    for row in (extract or {}).get("results") or []:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "").strip().lower()
        if status not in abnormal:
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        names.append(name)
    return names


def my_results_has_abnormal_imaging(extract: dict | None) -> bool:
    for item in (extract or {}).get("imagingFindings") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("status") or "").strip().lower() == "abnormal":
            return True
    return False


def my_results_has_key_findings(extract: dict | None) -> bool:
    """True when diagnoses, med changes, follow-ups, or imaging findings carry the real story."""
    extract = extract or {}
    if extract.get("newDiagnoses"):
        return True
    if extract.get("medicationChanges"):
        return True
    for follow in extract.get("followUps") or []:
        if isinstance(follow, dict) and str(follow.get("description") or "").strip():
            return True
    for img in extract.get("imagingFindings") or []:
        if isinstance(img, dict) and str(img.get("finding") or "").strip():
            return True
    return False


def my_results_no_abnormal_note_label(record: dict | None) -> str:
    """Section heading for noAbnormalValuesNote — reassuring only when nothing else is significant."""
    if my_results_has_key_findings(record):
        return "No abnormal lab values"
    return "Good news"


MY_RESULTS_LAB_TABLE_LIMITATION_RE = re.compile(
    r"reference range|ref(?:erence)?\.?\s*range|lab table|numeric (?:lab|result|value)",
    re.I,
)


def my_results_limitation_is_lab_table_artifact(text: str, extract: dict | None = None) -> bool:
    """Limitations that only apply when a numeric results table exists."""
    cleaned = str(text or "").strip()
    if not cleaned:
        return False
    has_lab_values = bool((extract or {}).get("hasLabValues") or (extract or {}).get("results"))
    if has_lab_values:
        return False
    return bool(MY_RESULTS_LAB_TABLE_LIMITATION_RE.search(cleaned))


def my_results_has_abnormal_values(extract: dict | None) -> bool:
    return bool(my_results_abnormal_test_names(extract)) or my_results_has_abnormal_imaging(extract)


def my_results_lab_comment_has_prior_comparison(lab_comment: str) -> bool:
    return bool(MY_RESULTS_PRIOR_COMPARISON_RE.search(str(lab_comment or "")))


def _normalize_my_results_prior_comparisons(raw_items, *, lab_comment: str = "") -> list[dict]:
    comparisons = []
    for item in _my_results_list(raw_items):
        if not isinstance(item, dict):
            continue
        category = sanitize_my_results_plain_text(item.get("category"))
        summary = sanitize_my_results_plain_text(item.get("summary"))
        prior_values = sanitize_my_results_plain_text(item.get("priorValues"))
        current_values = sanitize_my_results_plain_text(item.get("currentValues"))
        prior_date = sanitize_my_results_plain_text(item.get("priorDate"))
        tests = [
            str(name).strip()
            for name in _my_results_list(item.get("tests"))
            if str(name).strip()
        ]
        if not category and not summary and not prior_values and not tests:
            continue
        comparisons.append({
            "category": category or "Results changed since last test",
            "summary": summary,
            "tests": tests,
            "currentValues": current_values or None,
            "priorValues": prior_values or None,
            "priorDate": prior_date or None,
        })

    if comparisons:
        return comparisons

    comment = sanitize_my_results_plain_text(lab_comment)
    if not my_results_lab_comment_has_prior_comparison(comment):
        return []

    prior_values = ""
    prior_match = re.search(
        r"\(([^)]*(?:cr|egfr|prior|/\d{4})[^)]*)\)",
        comment,
        re.I,
    )
    if prior_match:
        prior_values = prior_match.group(1).strip()

    category = "Kidney function changed since last test"
    if not re.search(r"creatinine|egfr|kidney|renal", comment, re.I):
        category = "Results changed since last test"

    return [{
        "category": category,
        "summary": comment,
        "tests": [],
        "currentValues": None,
        "priorValues": prior_values or None,
        "priorDate": None,
    }]


def my_results_requires_trend_callouts(extract: dict | None) -> bool:
    extract = extract or {}
    if extract.get("priorComparisons"):
        return True
    return my_results_lab_comment_has_prior_comparison(extract.get("labComment") or "")


def _build_trend_callouts_from_extract(extract: dict | None) -> list[dict]:
    callouts = []
    for item in (extract or {}).get("priorComparisons") or []:
        if not isinstance(item, dict):
            continue
        title = sanitize_my_results_plain_text(item.get("category")) or "Change since last test"
        summary = sanitize_my_results_plain_text(item.get("summary"))
        prior_values = sanitize_my_results_plain_text(item.get("priorValues"))
        if not summary and not prior_values:
            continue
        if not summary:
            summary = (
                f"Prior values were {prior_values}."
                if prior_values
                else "The document notes a change compared to a prior test."
            )
        callouts.append({
            "title": title,
            "summary": summary,
            "priorValues": prior_values or None,
        })
    return callouts


def _ensure_my_results_trend_callouts(
    trend_callouts: list[dict],
    extract: dict | None,
) -> list[dict]:
    if trend_callouts:
        return trend_callouts
    if not my_results_requires_trend_callouts(extract):
        return []
    built = _build_trend_callouts_from_extract(extract)
    if built:
        my_results_logger.info(
            "my_results explain: backfilled %s trendCallout(s) from extract priorComparisons/labComment",
            len(built),
        )
    return built


def _generate_my_results_uncovered_explanations(
    uncovered_names: list[str],
    extract: dict | None,
    *,
    patient_name: str = "",
    known_conditions: list | None = None,
) -> list[dict]:
    extract_names = my_results_extract_test_names(extract)
    rows = []
    for name in uncovered_names:
        row = _my_results_result_row_by_name(extract, name)
        if row:
            rows.append(row)
    if not rows:
        return []

    payload = json.dumps({
        "patientName": patient_name or "the patient",
        "knownConditions": [
            str(item).strip()
            for item in (known_conditions or [])
            if str(item).strip()
        ] or ["None recorded"],
        "uncoveredTests": uncovered_names,
        "results": rows,
    }, ensure_ascii=False)

    try:
        raw = ask_ai(
            f"{MY_RESULTS_ASSISTANT_SYSTEM}\n\n{MY_RESULTS_UNCOVERED_EXPLAIN_PROMPT}",
            payload,
        )
    except Exception:
        my_results_logger.exception("my_results explain: uncovered explanation LLM call failed")
        raw = {"error": True}

    if raw.get("error"):
        my_results_logger.warning(
            "my_results explain: uncovered explanation LLM returned error for %s",
            uncovered_names,
        )
        return [
            _build_my_results_fallback_value_explanation(row)
            for row in rows
        ]

    explanations = []
    explained = set()
    for item in _my_results_list(raw.get("valueExplanations")):
        cleaned = _sanitize_my_results_value_explanation(item, extract_names)
        if not cleaned:
            continue
        key = cleaned["testName"].lower()
        if key in explained:
            continue
        explained.add(key)
        explanations.append(cleaned)

    for name in uncovered_names:
        if name.lower() in explained:
            continue
        row = _my_results_result_row_by_name(extract, name)
        if row:
            my_results_logger.warning(
                "my_results explain: LLM missed uncovered test %r — using minimal fallback",
                name,
            )
            explanations.append(_build_my_results_fallback_value_explanation(row))

    return explanations


def resolve_my_results_test_name(name: str, extract_names: list[str]) -> str | None:
    """Map an LLM-provided test name to an exact extract.results name, or None."""
    candidate = str(name or "").strip()
    if not candidate or not extract_names:
        return None
    lookup = {str(item).strip().lower(): str(item).strip() for item in extract_names if str(item).strip()}
    exact = lookup.get(candidate.lower())
    if exact:
        return exact

    candidate_lower = candidate.lower()
    partial_matches = []
    for extract_name in extract_names:
        extract_clean = str(extract_name).strip()
        if not extract_clean:
            continue
        extract_lower = extract_clean.lower()
        if candidate_lower == extract_lower:
            return extract_clean
        if candidate_lower in extract_lower or extract_lower in candidate_lower:
            partial_matches.append(extract_clean)
    if len(partial_matches) == 1:
        return partial_matches[0]
    if len(partial_matches) > 1:
        partial_matches.sort(key=len, reverse=True)
        best = get_close_matches(candidate, partial_matches, n=1, cutoff=0.75)
        if best:
            return best[0]
        return partial_matches[0]

    fuzzy = get_close_matches(candidate, extract_names, n=1, cutoff=0.82)
    return fuzzy[0] if fuzzy else None


def _sanitize_my_results_value_explanation(item: dict, extract_names: list[str]) -> dict | None:
    if not isinstance(item, dict):
        return None
    raw_name = item.get("testName")
    resolved_name = resolve_my_results_test_name(raw_name, extract_names)
    if not resolved_name:
        if str(raw_name or "").strip():
            my_results_logger.warning(
                "my_results explain: dropped valueExplanation — no extract match for testName=%r",
                raw_name,
            )
        return None
    what_it_measures = sanitize_my_results_plain_text(item.get("whatItMeasures"))
    what_suggests = sanitize_my_results_plain_text(item.get("whatThisResultSuggests"))
    if not what_it_measures and not what_suggests:
        my_results_logger.warning(
            "my_results explain: dropped valueExplanation — empty text for testName=%r",
            resolved_name,
        )
        return None
    define_terms = sanitize_my_results_plain_text(item.get("defineTerms"))
    return {
        "testName": resolved_name,
        "whatItMeasures": what_it_measures,
        "whatThisResultSuggests": what_suggests,
        "defineTerms": define_terms or None,
    }


def _normalize_my_results_result_groups(raw_groups, extract: dict | None) -> list[dict]:
    extract_names = my_results_extract_test_names(extract)
    if not extract_names:
        return []

    urgency_rank = {"discuss_soon": 0, "discuss_at_visit": 1}
    normalized_groups = []

    for group in _my_results_list(raw_groups):
        if not isinstance(group, dict):
            continue
        category = sanitize_my_results_plain_text(group.get("category"))
        group_summary = sanitize_my_results_plain_text(group.get("groupSummary"))
        urgency = str(group.get("urgency") or "discuss_at_visit").strip().lower()
        if urgency not in urgency_rank:
            urgency = "discuss_at_visit"

        resolved_test_names = []
        seen_names = set()
        for raw_name in _my_results_list(group.get("testNames")):
            resolved = resolve_my_results_test_name(raw_name, extract_names)
            if not resolved:
                if str(raw_name or "").strip():
                    my_results_logger.warning(
                        "my_results explain: dropped testName=%r in group %r — no extract match",
                        raw_name,
                        category or "(unnamed)",
                    )
                continue
            key = resolved.lower()
            if key in seen_names:
                continue
            seen_names.add(key)
            resolved_test_names.append(resolved)

        value_explanations = []
        explained_names = set()
        for item in _my_results_list(group.get("valueExplanations")):
            cleaned = _sanitize_my_results_value_explanation(item, extract_names)
            if not cleaned:
                continue
            key = cleaned["testName"].lower()
            if key in explained_names:
                continue
            explained_names.add(key)
            value_explanations.append(cleaned)
            if key not in seen_names:
                seen_names.add(key)
                resolved_test_names.append(cleaned["testName"])

        if not category and not group_summary and not resolved_test_names and not value_explanations:
            continue
        if not resolved_test_names and not value_explanations:
            continue

        related_to = sanitize_my_results_plain_text(group.get("relatedTo"))

        normalized_groups.append({
            "category": category or "Results to discuss",
            "urgency": urgency,
            "groupSummary": group_summary,
            "relatedTo": related_to or None,
            "testNames": resolved_test_names,
            "valueExplanations": value_explanations,
        })

    normalized_groups.sort(key=lambda item: urgency_rank.get(item.get("urgency"), 1))
    return normalized_groups


def _my_results_covered_test_names(result_groups: list[dict] | None) -> set[str]:
    covered = set()
    for group in result_groups or []:
        if not isinstance(group, dict):
            continue
        for name in group.get("testNames") or []:
            cleaned = str(name or "").strip()
            if cleaned:
                covered.add(cleaned.lower())
        for item in group.get("valueExplanations") or []:
            if not isinstance(item, dict):
                continue
            cleaned = str(item.get("testName") or "").strip()
            if cleaned:
                covered.add(cleaned.lower())
    return covered


def _my_results_result_row_by_name(extract: dict | None, name: str) -> dict | None:
    target = str(name or "").strip().lower()
    if not target:
        return None
    for row in (extract or {}).get("results") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("name") or "").strip().lower() == target:
            return row
    return None


def _my_results_status_phrase(status: str) -> str:
    normalized = str(status or "").strip().lower()
    return {
        "high": "higher than the usual range",
        "low": "lower than the usual range",
        "abnormal": "outside the usual range",
    }.get(normalized, "flagged as needing review")


def _build_my_results_fallback_value_explanation(row: dict) -> dict:
    name = str(row.get("name") or "").strip() or "This test"
    value = str(row.get("value") or "").strip()
    unit = str(row.get("unit") or "").strip()
    ref = str(row.get("referenceRange") or "").strip()
    status = str(row.get("status") or "").strip().lower()
    value_text = f"{value} {unit}".strip() if value else "the reported value"
    ref_text = f" The usual range is {ref}." if ref else ""
    return {
        "testName": name,
        "whatItMeasures": f"{name} is a lab test listed on this report.",
        "whatThisResultSuggests": (
            f"The result is {value_text}, which is {_my_results_status_phrase(status)}.{ref_text} "
            "Ask the doctor what this means for day-to-day care."
        ),
        "defineTerms": None,
    }


def _ensure_my_results_abnormal_coverage(
    result_groups: list[dict],
    extract: dict | None,
    *,
    patient_name: str = "",
    known_conditions: list | None = None,
    generate_missing_explanations: bool = True,
) -> list[dict]:
    """Guarantee every abnormal extract result appears in at least one resultGroup."""
    abnormal_names = my_results_abnormal_test_names(extract)
    if not abnormal_names:
        return result_groups

    covered = _my_results_covered_test_names(result_groups)
    uncovered = [name for name in abnormal_names if name.lower() not in covered]
    if not uncovered:
        return result_groups

    my_results_logger.warning(
        "my_results explain: uncovered abnormal value(s) after normalize — adding fallback group: %s",
        uncovered,
    )

    if generate_missing_explanations:
        fallback_explanations = _generate_my_results_uncovered_explanations(
            uncovered,
            extract,
            patient_name=patient_name,
            known_conditions=known_conditions,
        )
    else:
        fallback_explanations = []
        for name in uncovered:
            row = _my_results_result_row_by_name(extract, name)
            if row:
                fallback_explanations.append(_build_my_results_fallback_value_explanation(row))

    if not fallback_explanations:
        return result_groups

    return list(result_groups) + [{
        "category": "Other flagged values",
        "urgency": "discuss_at_visit",
        "groupSummary": (
            "These results were flagged as outside the usual range. "
            "They should still be discussed with the doctor."
        ),
        "testNames": uncovered,
        "valueExplanations": fallback_explanations,
    }]


def _normalize_my_results_trend_callouts(raw_callouts) -> list[dict]:
    callouts = []
    for item in _my_results_list(raw_callouts):
        if not isinstance(item, dict):
            continue
        title = sanitize_my_results_plain_text(item.get("title"))
        summary = sanitize_my_results_plain_text(item.get("summary"))
        if not title and not summary:
            continue
        prior_values = sanitize_my_results_plain_text(item.get("priorValues"))
        callouts.append({
            "title": title or "Change since last test",
            "summary": summary,
            "priorValues": prior_values or None,
        })
    return callouts


def _normalize_my_results_questions(raw_questions, extract: dict | None) -> list[dict]:
    extract_names = my_results_extract_test_names(extract)
    questions = []
    for item in _my_results_list(raw_questions):
        if isinstance(item, str):
            text = sanitize_my_results_plain_text(item)
            if text:
                questions.append({
                    "text": text,
                    "relatedCategory": "",
                    "relatedTests": [],
                })
            continue
        if not isinstance(item, dict):
            continue
        text = sanitize_my_results_plain_text(item.get("text"))
        if not text:
            continue
        related_category = sanitize_my_results_plain_text(item.get("relatedCategory"))
        related_tests = []
        seen = set()
        for raw_name in _my_results_list(item.get("relatedTests")):
            resolved = resolve_my_results_test_name(raw_name, extract_names)
            if not resolved:
                continue
            key = resolved.lower()
            if key in seen:
                continue
            seen.add(key)
            related_tests.append(resolved)
        questions.append({
            "text": text,
            "relatedCategory": related_category,
            "relatedTests": related_tests,
        })
    return questions


def my_results_explain_is_complete(explain: dict | None, extract: dict | None = None) -> bool:
    explain = explain or {}
    explanation = sanitize_my_results_plain_text(explain.get("explanation"))
    questions = explain.get("questions") or []
    note = sanitize_my_results_plain_text(explain.get("noAbnormalValuesNote"))

    if note and not my_results_has_abnormal_values(extract):
        return bool(explanation)

    if explain.get("resultGroups"):
        return bool(explanation) and bool(questions)

    return bool(explanation) and bool(questions)


def my_results_use_grouped_explanations(explain: dict | None, extract: dict | None) -> bool:
    """True when validated grouped content should render instead of legacy flat paragraph."""
    explain = explain or {}
    if explain.get("noAbnormalValuesNote") and not my_results_has_abnormal_values(extract):
        return False
    return bool(explain.get("resultGroups"))


def my_results_question_text(item) -> str:
    if isinstance(item, dict):
        return sanitize_my_results_plain_text(item.get("text"))
    return sanitize_my_results_plain_text(item)


def build_my_results_explain_payload(
    extract: dict,
    *,
    patient_name: str,
    known_conditions: list | None = None,
) -> str:
    payload = normalize_my_results_extract(extract)
    all_test_names = my_results_extract_test_names(payload)
    abnormal_test_names = my_results_abnormal_test_names(payload)
    payload["patientName"] = patient_name
    payload["knownConditions"] = [
        str(item).strip()
        for item in (known_conditions or [])
        if str(item).strip()
    ] or ["None recorded"]
    payload["allTestNames"] = all_test_names
    payload["abnormalTestNames"] = abnormal_test_names
    payload["hasKeyFindings"] = my_results_has_key_findings(payload)
    payload["labComment"] = payload.get("labComment") or ""
    payload["priorComparisons"] = payload.get("priorComparisons") or []
    payload["requiresTrendCallouts"] = my_results_requires_trend_callouts(payload)
    return json.dumps(payload, ensure_ascii=False)


def normalize_my_results_explain(
    raw: dict | None,
    extract: dict | None = None,
    *,
    patient_name: str = "",
    known_conditions: list | None = None,
    generate_missing_explanations: bool = True,
) -> dict:
    raw = dict(raw or {})
    note_text = sanitize_my_results_plain_text(raw.get("noAbnormalValuesNote"))
    has_abnormal_values = my_results_has_abnormal_values(extract)

    if note_text and not has_abnormal_values:
        questions = _normalize_my_results_questions(raw.get("questions"), extract)
        urgent_text = sanitize_my_results_plain_text(raw.get("urgentCareInstructions"))
        return {
            "explanation": sanitize_my_results_plain_text(raw.get("explanation")),
            "trendCallouts": [],
            "resultGroups": [],
            "urgentCareInstructions": urgent_text or None,
            "questions": questions,
            "noAbnormalValuesNote": note_text,
            "useGroupedExplanations": False,
        }

    trend_callouts = _ensure_my_results_trend_callouts(
        _normalize_my_results_trend_callouts(raw.get("trendCallouts")),
        extract,
    )
    result_groups = _normalize_my_results_result_groups(raw.get("resultGroups"), extract)
    result_groups = _ensure_my_results_abnormal_coverage(
        result_groups,
        extract,
        patient_name=patient_name,
        known_conditions=known_conditions,
        generate_missing_explanations=generate_missing_explanations,
    )
    questions = _normalize_my_results_questions(raw.get("questions"), extract)
    urgent_text = sanitize_my_results_plain_text(raw.get("urgentCareInstructions"))

    return {
        "explanation": sanitize_my_results_plain_text(raw.get("explanation")),
        "trendCallouts": trend_callouts,
        "resultGroups": result_groups,
        "urgentCareInstructions": urgent_text or None,
        "questions": questions,
        "noAbnormalValuesNote": note_text or None,
        "useGroupedExplanations": bool(result_groups),
    }


MY_RESULTS_LOG_PREFIX = "CS_MY_RESULTS:"

# Stored in shift_logs for persistence but not shown as raw timeline/SBAR events.
INTERNAL_SHIFT_LOG_SOURCES = frozenset({"my_results", "medcam_audit"})
ADHERENCE_SHIFT_LOG_SOURCES = frozenset({"medication_check"})


def shift_log_is_internal_storage(row: dict) -> bool:
    return str(row.get("source") or "") in INTERNAL_SHIFT_LOG_SOURCES


def shift_log_is_adherence_event(row: dict) -> bool:
    return str(row.get("source") or "") in ADHERENCE_SHIFT_LOG_SOURCES


def shift_log_is_symptom_event(row: dict) -> bool:
    source = str(row.get("source") or "")
    if not source or shift_log_is_internal_storage(row) or shift_log_is_adherence_event(row):
        return False
    return True


def backfill_legacy_shift_logs_to_local_care(patient_id=None, limit: int = 500) -> int:
    """Import legacy unscoped shift_logs into local care reports for one patient."""
    patient_id = resolve_patient_id(patient_id)
    if not patient_id or legacy_backfill_completed(patient_id):
        return 0

    patients = list_account_patients()
    allow_unmarked = len(patients) == 1 or not any_other_patient_has_local_reports(patient_id)
    try:
        response = (
            supabase.table("shift_logs")
            .select("*")
            .order("created_at", desc=False)
            .limit(limit)
            .execute()
        )
        rows = response.data or []
    except Exception as exc:
        logger.debug("backfill_legacy_shift_logs query failed: %s", exc)
        return 0

    imported = 0
    for row in rows:
        if not shift_log_is_symptom_event(row):
            continue
        summary = str(row.get("summary") or "").strip()
        if not summary:
            continue
        if summary_matches_shift_log_patient(summary, patient_id):
            target_patient = patient_id
        elif allow_unmarked and "[[patient:" not in summary:
            target_patient = patient_id
        else:
            continue
        clean_summary = strip_shift_log_patient_marker(summary)
        if shift_log_row_is_internal_test({
            "summary": clean_summary,
            "caregiver_name": row.get("caregiver_name"),
            "source": row.get("source"),
        }):
            continue
        saved = save_local_care_report(
            target_patient,
            report_text=clean_summary,
            summary=clean_summary,
            severity=row.get("severity", "monitor"),
            source=row.get("source", "voice_report"),
            reported_at=row.get("created_at"),
            caregiver_name=row.get("caregiver_name", "Caregiver"),
            caregiver_id=row.get("caregiver_id"),
        )
        if saved:
            imported += 1
    if imported or rows:
        mark_legacy_backfill_completed(patient_id)
    return imported


def fetch_symptom_shift_logs(patient_id=None, limit: int = 250) -> list:
    """Care reports from Report & Ask, documents, and photos — not MedCam dose checks."""
    patient_id = resolve_patient_id(patient_id)
    excluded_sources = sorted(INTERNAL_SHIFT_LOG_SOURCES | ADHERENCE_SHIFT_LOG_SOURCES)
    try:
        query = (
            supabase.table("shift_logs")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
        )
        if patient_id:
            query = query.eq("patient_id", patient_id)
        if excluded_sources:
            query = query.not_.in_("source", excluded_sources)
        response = query.execute()
        rows = response.data or []
        if rows:
            return rows
    except Exception as exc:
        logger.debug("symptom shift_logs filtered query failed: %s", exc)

    rows = fetch_shift_logs(patient_id, limit=max(limit * 4, 500))
    symptom_rows = [row for row in rows if shift_log_is_symptom_event(row)]
    return symptom_rows[:limit]


def _shift_log_targets_patient_for_cleanup(row: dict, patient_id: str) -> bool:
    summary = str(row.get("summary") or "")
    if summary_matches_shift_log_patient(summary, patient_id):
        return True
    return "[[patient:" not in summary


def purge_internal_test_patient_artifacts(patient_id=None) -> dict:
    """Permanently remove QA/test entries from a production patient profile."""
    patient_id = resolve_patient_id(patient_id)
    if not patient_id:
        return {"patient_id": None, "removed_local": 0, "removed_shift_logs": 0, "skipped": True}

    patient_row = get_patient_by_id(patient_id)
    if is_designated_test_patient(patient_id, patient_row):
        return {
            "patient_id": patient_id,
            "removed_local": 0,
            "removed_shift_logs": 0,
            "skipped": True,
        }

    removed_local = purge_local_internal_test_entries(patient_id, is_test_row=care_row_is_internal_test)
    removed_shift_logs = 0
    try:
        response = (
            supabase.table("shift_logs")
            .select("id, summary, caregiver_name, source")
            .order("created_at", desc=True)
            .limit(500)
            .execute()
        )
        for row in response.data or []:
            if not shift_log_row_is_internal_test(row):
                continue
            if not _shift_log_targets_patient_for_cleanup(row, patient_id):
                continue
            try:
                supabase.table("shift_logs").delete().eq("id", row["id"]).execute()
                removed_shift_logs += 1
            except Exception as exc:
                logger.debug("Could not delete test shift_log id=%s: %s", row.get("id"), exc)
    except Exception as exc:
        logger.debug("purge_internal_test_patient_artifacts shift_logs query failed: %s", exc)

    return {
        "patient_id": patient_id,
        "removed_local": removed_local,
        "removed_shift_logs": removed_shift_logs,
        "skipped": False,
    }


def fetch_medication_check_shift_logs(patient_id=None, limit: int = 100) -> list:
    rows = fetch_shift_logs(patient_id, limit=max(limit * 4, 250))
    medcam_rows = [row for row in rows if shift_log_is_adherence_event(row)]
    return medcam_rows[:limit]


def build_openai_user_content(user_text="", image_b64=None, media_type="image/jpeg"):
    """Public helper for GPT-4o multimodal user messages."""
    return _build_openai_user_content(user_text, image_b64, media_type)


def save_patient_test_document(
    patient_id,
    *,
    file_name: str,
    raw_text: str,
    caregiver_id=None,
) -> dict:
    """Store a test-result document without changing medications or conditions."""
    patient_id_int = _patient_id_int(patient_id)
    if patient_id_int is None:
        return {}

    doc_payload = {
        "patient_id": patient_id_int,
        "file_name": file_name,
        "raw_text": raw_text,
    }
    if caregiver_id is not None and str(caregiver_id).isdigit():
        doc_payload["uploaded_by_caregiver_id"] = int(caregiver_id)

    try:
        response = supabase.table("documents").insert(doc_payload).select("*").execute()
        if response.data:
            return {"document_id": response.data[0].get("id")}
    except Exception:
        pass
    return {}


def encode_my_result_payload(record: dict) -> str:
    return MY_RESULTS_LOG_PREFIX + json.dumps(record, ensure_ascii=False)


def parse_my_result_payload(summary: str) -> dict | None:
    text = str(summary or "")
    if not text.startswith(MY_RESULTS_LOG_PREFIX):
        return None
    try:
        return json.loads(text[len(MY_RESULTS_LOG_PREFIX):])
    except json.JSONDecodeError:
        return None


def save_my_result_record(
    patient_id,
    record: dict,
    caregiver_name: str,
    caregiver_id=None,
) -> None:
    """Persist structured My Results analysis on the patient record."""
    title = record.get("documentType") or record.get("file_name") or "Test results"
    summary = encode_my_result_payload(record)
    save_shift_log(
        caregiver_name=caregiver_name,
        source="my_results",
        summary=summary,
        severity="monitor",
        caregiver_id=caregiver_id,
        patient_id=patient_id,
    )


def fetch_my_result_records(patient_id=None, limit: int = 20) -> list:
    """Load saved My Results analyses for one patient, newest first."""
    patient_id = resolve_patient_id(patient_id)
    if not patient_id:
        return []
    logs = fetch_shift_logs(patient_id, limit=250)
    records = []
    for row in logs:
        if row.get("source") != "my_results":
            continue
        row_patient_id = row.get("patient_id")
        if row_patient_id is not None and str(row_patient_id) != str(patient_id):
            continue
        if row_patient_id is None:
            continue
        parsed = parse_my_result_payload(row.get("summary") or "")
        if not parsed:
            continue
        parsed["patient_id"] = str(patient_id)
        if not parsed.get("uploaded_at") and row.get("created_at"):
            parsed["uploaded_at"] = row["created_at"]
        records.append(parsed)
        if len(records) >= limit:
            break
    return records


def generate_my_results_summary_pdf(record: dict, patient_label: str = "Patient") -> bytes:
    """Downloadable plain-English summary of one My Results analysis."""
    record = record or {}
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "MyResultsTitle",
        parent=styles["Title"],
        fontSize=17,
        leading=21,
        textColor=colors.HexColor("#2D3F6B"),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "MyResultsSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#7A7469"),
        spaceAfter=10,
    )
    section_style = ParagraphStyle(
        "MyResultsSection",
        parent=styles["Heading2"],
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#2D3F6B"),
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "MyResultsBody",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#1A2B23"),
    )
    story = [
        Paragraph("CareShield — My Results Summary", title_style),
        Paragraph(
            f"{_pdf_escape(patient_label)} · "
            f"{_pdf_escape(record.get('documentType') or record.get('file_name') or 'Test results')}",
            subtitle_style,
        ),
        Paragraph(
            "<i>This explains your results — always discuss them with your doctor.</i>",
            body_style,
        ),
        Spacer(1, 8),
    ]

    results = record.get("results") or []
    key_findings = []
    for item in record.get("newDiagnoses") or []:
        if isinstance(item, dict) and item.get("name"):
            key_findings.append(f"New diagnosis: {item['name']} — {item.get('detail', '')}".strip(" —"))
    for item in record.get("medicationChanges") or []:
        if isinstance(item, dict) and item.get("medication"):
            key_findings.append(
                f"Medication {item.get('changeType', 'change')}: {item['medication']} — {item.get('detail', '')}".strip(" —")
            )
    for item in record.get("followUps") or []:
        if isinstance(item, dict) and item.get("description"):
            date_label = item.get("dateDisplay") or MY_RESULTS_DATE_NOT_SPECIFIED
            key_findings.append(
                f"Follow-up: {item['description']} ({date_label})"
            )
    if key_findings:
        story.append(Paragraph("Key findings", section_style))
        for line in key_findings[:8]:
            story.append(Paragraph(f"• {_pdf_escape(line)}", body_style))
        story.append(Spacer(1, 6))

    urgent = record.get("urgentCareInstructions")
    if urgent:
        story.append(Paragraph("When to seek urgent care", section_style))
        story.append(Paragraph(_pdf_escape(str(urgent)), body_style))
        story.append(Spacer(1, 6))

    explanation = record.get("explanation") or ""
    use_grouped = bool(record.get("useGroupedExplanations") and record.get("resultGroups"))
    if explanation:
        overview_title = "At a glance" if use_grouped else "What this means"
        story.append(Paragraph(overview_title, section_style))
        story.append(Paragraph(_pdf_escape(explanation), body_style))
        story.append(Spacer(1, 6))

    for item in record.get("trendCallouts") or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "Change since last test").strip()
        summary = str(item.get("summary") or "").strip()
        if not summary:
            continue
        story.append(Paragraph(_pdf_escape(title), section_style))
        story.append(Paragraph(_pdf_escape(summary), body_style))
        prior = str(item.get("priorValues") or "").strip()
        if prior:
            story.append(Paragraph(f"<i>Previous values: {_pdf_escape(prior)}</i>", body_style))
        story.append(Spacer(1, 4))

    use_grouped = record.get("useGroupedExplanations")
    if use_grouped and record.get("resultGroups"):
        story.append(Paragraph("What this means", section_style))
        for group in record.get("resultGroups") or []:
            if not isinstance(group, dict):
                continue
            category = str(group.get("category") or "Results to discuss").strip()
            urgency = str(group.get("urgency") or "discuss_at_visit").replace("_", " ")
            story.append(Paragraph(f"<b>{_pdf_escape(category)}</b> ({_pdf_escape(urgency)})", body_style))
            summary = str(group.get("groupSummary") or "").strip()
            if summary:
                story.append(Paragraph(_pdf_escape(summary), body_style))
            for item in group.get("valueExplanations") or []:
                if not isinstance(item, dict):
                    continue
                test_name = str(item.get("testName") or "").strip()
                if not test_name:
                    continue
                lines = [f"<b>{_pdf_escape(test_name)}</b>"]
                if item.get("whatItMeasures"):
                    lines.append(f"What it measures: {_pdf_escape(str(item['whatItMeasures']))}")
                if item.get("whatThisResultSuggests"):
                    lines.append(f"What this suggests: {_pdf_escape(str(item['whatThisResultSuggests']))}")
                story.append(Paragraph("<br/>".join(lines), body_style))
            story.append(Spacer(1, 4))
        story.append(Spacer(1, 4))

    note = record.get("noAbnormalValuesNote")
    if note:
        story.append(Paragraph("Summary", section_style))
        story.append(Paragraph(_pdf_escape(str(note)), body_style))
        story.append(Spacer(1, 6))

    if record.get("hasLabValues", bool(results)):
        story.append(Paragraph("Results", section_style))
        table_data = [[
            Paragraph("<b>Test</b>", body_style),
            Paragraph("<b>Value</b>", body_style),
            Paragraph("<b>Status</b>", body_style),
        ]]
        for row in results:
            value = str(row.get("value") or "—")
            unit = str(row.get("unit") or "").strip()
            if unit:
                value = f"{value} {unit}"
            status = str(row.get("status") or "normal").strip().capitalize()
            table_data.append([
                Paragraph(_pdf_escape(row.get("name") or "—"), body_style),
                Paragraph(_pdf_escape(value), body_style),
                Paragraph(_pdf_escape(status), body_style),
            ])
        results_table = Table(table_data, colWidths=[2.4 * inch, 2.0 * inch, 1.4 * inch])
        results_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2D3F6B")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E8E4DA")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(results_table)
        story.append(Spacer(1, 8))

    questions = record.get("questions") or []
    if questions:
        story.append(Paragraph("Questions to ask the doctor", section_style))
        for idx, question in enumerate(questions, start=1):
            text = my_results_question_text(question)
            if not text:
                continue
            meta = ""
            if isinstance(question, dict):
                bits = []
                if question.get("relatedCategory"):
                    bits.append(str(question["relatedCategory"]))
                related_tests = question.get("relatedTests") or []
                if related_tests:
                    bits.append(", ".join(str(name) for name in related_tests))
                if bits:
                    meta = f" ({_pdf_escape(' · '.join(bits))})"
            story.append(Paragraph(
                f"<b>{idx}.</b> {_pdf_escape(text)}{meta}",
                body_style,
            ))
            story.append(Spacer(1, 3))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
