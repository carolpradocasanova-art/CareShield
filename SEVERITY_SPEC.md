# Report & Ask Severity Specification

This document describes how CareShield decides message severity (`ok` / `monitor` / `contact_doctor` / `emergency`) for Report & Ask. It is the reference for hackathon writeups and future severity work.

**UI rule:** severity banners appear only when final severity is not `ok`.

---

## 1. Pipeline overview

```
User input
  → Route (photo / next-dose / question / voice report)
  → [Optional] Symptom–condition cross-check LLM + rule enrichment
  → Main chat LLM (severity in JSON)
  → resolve_chat_severity() — keyword triggers, session linking
  → merge_symptom_condition_analysis() — cross-check recommended_severity
  → resolve_chat_severity() — second pass
  → cap_allergy_report_severity() — down-only
  → apply_report_severity_floor_caps() — up-only floors
  → cap_positive_report_severity() — down-only
  → cap_informational_question_severity() — down-only (questions only)
  → append_care_guidance() — footer text
```

Key files: `app.py` (`process_pending_chat_response`, `resolve_chat_severity`), `ai_helpers.py` (cross-check, rules, caps), `symptom_linking.py` (session recurrence).

---

## 2. Structural severity floors (up-only)

Applied via `apply_report_severity_floor_caps()` after AI/session resolution. Each floor can only **raise** severity.

| Floor | Patient context required | Minimum severity | Trigger examples |
|-------|-------------------------|------------------|------------------|
| **Cyanosis** | None (universal) | `emergency` | blue/bluish/purple/grey lips, cyanosis, turning blue around mouth |
| **ACE inhibitor angioedema** | ACE inhibitor on file | `contact_doctor` or `emergency` | swallowing difficulty; lip/tongue swelling + breathing/hoarseness |
| **Beta-blocker bradycardia** | Beta-blocker on file (e.g. Bisoprolol) | `contact_doctor` or `emergency` | slow pulse / HR ~40s / below 50; + faint/dizziness → emergency |
| **Hypoglycaemia** | Diabetes diagnosis or diabetes medication on file | `contact_doctor` or `emergency` | shaky, sweaty, clammy, confused; unresponsive / can't wake → emergency |
| **Anticoagulant + head trauma** | Anticoagulant on file (e.g. Warfarin) | `emergency` | fall/fell + head impact / bumped head / head injury |

Implementation: `cap_cyanosis_report_severity`, `cap_ace_angioedema_report_severity`, `cap_beta_blocker_bradycardia_report_severity`, `cap_hypoglycemia_report_severity`, `cap_anticoagulant_head_trauma_report_severity` in `ai_helpers.py`.

Cross-check enrichment applies parallel policies in `enrich_symptom_condition_analysis()` via `MEDICATION_SYMPTOM_RULES`, `FALLBACK_CONDITION_SYMPTOM_RULES`, and `apply_*_severity_policy()` helpers.

---

## 3. Structural caps (down-only)

| Cap | Effect |
|-----|--------|
| `cap_allergy_report_severity` | `emergency` → `contact_doctor` when allergic symptoms lack anaphylaxis red flags in user text |
| `cap_positive_report_severity` | Any elevated → `ok` for clearly positive benign reports |
| `cap_informational_question_severity` | Any elevated → `ok` for pure informational care questions |
| `resolve_chat_severity` early exit | Positive benign reports and informational questions → `ok` before session escalation |

---

## 4. Keyword contact-doctor triggers (user text)

`detect_contact_doctor_triggers()` in `app.py`: elevated BP, persistent confusion, infected wound, post-surgical/increasing swelling, fever >38°C, worsening symptoms.

---

## 5. Session linking

`resolve_chat_severity()` raises `ok`/`monitor` → `contact_doctor` when:

- Confusion + fall combo across session
- Same symptom subtype recurs in current + prior session text
- Same symptom key appears ≥3 times in session
- Confusion + fall + head injury across session → `emergency` (current message must contribute)

---

## 6. LLM judgment layers

1. **Main chat** — `CHAT_URGENCY_RULES` in system prompt
2. **Symptom cross-check** — `SYMPTOM_CONDITION_CROSSCHECK_PROMPT` when `reports_health_symptom_topic()` is true

Non-question messages always run cross-check (except positive benign). Questions run cross-check only if symptom keywords appear in the question.

---

## 7. My Results / document red flags — architectural note

Cardiology letters and clinic documents may extract urgent-care instructions (e.g. “seek urgent attention if heart rate below 50 bpm”) via My Results / document parsing. **These extracted red-flag lines are not currently wired into Report & Ask severity rules** — they are stored and displayed, not fed into `apply_report_severity_floor_caps()`.

The **generic Bisoprolol + bradycardia rule** covers John's cardiology-letter scenario structurally without patient-specific document integration. Full document-to-severity wiring remains a separate architectural gap.

---

## 8. Test coverage for gap closures

Run: `python3 -m unittest tests.test_severity_floor_gaps -v`

| Test | Scenario | Expected |
|------|----------|----------|
| test_01 | Blue lips + fast breathing | Cyanosis detected |
| test_02–03 | Cyanosis from monitor/ok | `emergency` |
| test_04 | Bisoprolol + HR ~45 | `contact_doctor` |
| test_06 | Bisoprolol + slow pulse + faint | `emergency` |
| test_08 | Diabetes + sweaty/shaky/confused | `contact_doctor` |
| test_09 | Diabetes + can't wake | `emergency` |
| test_11 | “woke up confused” + Type 2 diabetes | Existing fallback → `contact_doctor` |
| test_14 | Warfarin + fall + bumped head | `emergency` (overrides `contact_doctor`) |

---

## 9. Structurally covered (recent additions)

Moved out of “gaps” as of this spec:

- ✅ **Cyanosis / bluish lips** — unconditional `emergency` floor
- ✅ **Beta-blocker bradycardia** — `MEDICATION_SYMPTOM_RULES` + floor (Bisoprolol + slow HR)
- ✅ **Hypoglycaemia** — `FALLBACK_CONDITION_SYMPTOM_RULES` + diabetes medication rules + floor
- ✅ **Diabetes + confusion** — pre-existing `FALLBACK_CONDITION_SYMPTOM_RULES` entry confirmed by test_11
- ✅ **Anticoagulant + head trauma after fall** — `MEDICATION_SYMPTOM_RULES` + emergency floor (Warfarin + fall + head bump)

---

## 10. Remaining gaps (not yet structurally covered)

### Symptom / clinical presentations
- Stroke-specific FAST pattern as a unit
- Seizure
- Severe chest pain as standalone keyword trigger (chest tightness in labeling only)
- Hypoglycaemia symptoms **without** diabetes context on file
- Bluish lips already covered — removed from this list

### Medication-specific risks
- Metformin lactic acidosis pattern
- Insulin/sulfonylurea hypoglycaemia without diabetes on file
- Opioid respiratory depression
- Potassium-sparing diuretic + hyperkalaemia
- NSAID + anticoagulant GI bleed beyond partial bleed rules
- Anticoagulant + head trauma after fall covered — removed from this list

### Message types & architecture
- Voice reports that are conversational/informational but not phrased as questions
- Monitor vs contact_doctor boundary (mostly LLM discretion)
- **Patient-specific document red flags → severity** (My Results / clinic letter thresholds)
- Medical RAG chunks influencing AI severity with no structural override

### Session / data
- Pre-session stored timeline has no direct severity function (only via LLM reading context)
