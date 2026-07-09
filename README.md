# CareShield

**AI-powered care support for family caregivers.**

CareShield helps family caregivers track a loved one's health, catch warning signs early, and walk into doctor's appointments prepared — instead of relying on memory, guesswork, and generic advice.

---

## The Problem

Family caregivers are often expected to track symptoms, remember when changes happened, manage medications correctly, and summarize weeks of health history for doctors — all while under emotional and physical strain. Details get forgotten, patterns go unnoticed, and medical paperwork full of jargon is hard to act on. Meanwhile, generic health advice can't connect a new symptom to a patient's actual conditions and medications.

## The Solution

CareShield builds a real medical profile for each patient — their conditions, medications, and history — directly from uploaded documents, and uses that context everywhere: when a caregiver reports a symptom, when medication is administered, and when it's time to prepare for an appointment. Instead of generic responses, CareShield reasons against the patient's actual medical picture.

---

## Features

### Documents

**Problem:** Without knowing a patient's actual medications and conditions, an AI assistant can only respond generically — it can't connect today's swollen legs to an existing heart condition, or recognize that new drowsiness might be a side effect of a pill started last week.

**Solution:** Documents reads hospital paperwork the moment it's uploaded, automatically extracting medications, doses, and chronic conditions. This becomes the medical context CareShield's AI uses everywhere else, so symptom reports are reasoned about against the patient's real history — not guessed at.

**How to use it:** Click **Upload Document** to submit medical paperwork — doctor's letters, care plans, hospital discharge summaries. The system securely scans and saves conditions and medications, building a complete health picture for CareShield AI.

> Upload documents that belong to and match the active patient profile. Documents for another patient will not be saved.

---

### Report & Ask

**Problem:** Caregivers are expected to track symptoms, remember when changes happened, and relay accurate information to doctors — all while managing the stress of caregiving. Important details get forgotten, changes go unnoticed, and doctors don't always get the full picture.

**Solution:** Report & Ask lets caregivers record health updates as they happen, surfaces possible patterns in symptoms over time, and generates clear handover reports for healthcare professionals — turning everyday observations into an organized timeline.

**How to use it:** Type a message in the chatbox to ask a question or share a health update — a new symptom, a behavioral change, or an image of a rash. Using text and image recognition, CareShield logs the timeline, assesses severity to prompt an emergency call if needed, and provides next steps tailored to the patient's existing conditions and medications.

---

### Pill Registration

**Problem:** Generic pills often look alike, and a tired caregiver giving medication late at night can easily mix up two white tablets or misjudge a dose. Verbal pharmacist instructions are easy to forget by the time the bottle is opened at home.

**Solution:** Pill Registration lets caregivers photograph each medication once, teaching the system what the patient's real pills look like. Strength and dosage are calculated automatically from the discharge document — no guesswork when it matters.

**How to use it:** Tap a card to register a medication — upload front and back photos of the pill, then confirm strength (mg) and brand. Clear photos on a plain background help MedCam identify pills correctly.

---

### MedCam

**Problem:** Even with a clear schedule, it's hard to know in the moment whether the right pills, in the right amount, are about to be given. Missed or doubled doses often go unnoticed until a doctor asks about adherence weeks later.

**Solution:** MedCam checks a photo of the pills in hand against the patient's registered medications and active schedule, confirming the right pill and count before it's given. Missed or mismatched doses are flagged immediately.

**How to use it:** Click **Upload** to submit a real-time photo of the medication before administering it. CareShield AI verifies the pill against schedules and registered data, provides a final safety check, and automatically logs adherence for future handovers.

---

### Handover

**Problem:** At doctor visits, caregivers are asked to summarize weeks of symptoms, incidents, and medication changes from memory, under time pressure, often while emotionally exhausted. Crucial details get left out, and patterns connecting separate events go unspoken.

**Solution:** Handover automatically compiles every logged update into a clinical-grade SBAR report, complete with severity scoring, adherence tracking, and a connected symptom timeline — turning a rushed verbal recap into a structured document ready for the appointment.

**How to use it:** Select the tracking period that matches your appointment, then generate the SBAR handover report. Review the adherence and symptom-pattern charts, then download the PDF to share with the clinician.

---

### My Results

**Problem:** Lab results and clinic letters arrive full of medical terminology and reference ranges that mean little to a non-clinical caregiver. Without context, it's hard to know what's actually concerning or worth asking about.

**Solution:** My Results explains uploaded test results in plain English, flags values outside the normal range, and suggests specific questions to bring to the next appointment — turning a confusing report into something a caregiver can act on.

**How to use it:** Upload any medical document from an appointment or test — blood test printouts, scan reports, clinic letters. CareShield explains what matters most — new diagnoses, medication changes, follow-ups, red-flag instructions, and flagged lab values — in plain language, then suggests questions to ask the doctor.

---

## How It All Fits Together

1. **Documents** builds the patient's medical profile from uploaded paperwork.
2. **Pill Registration** and **MedCam** use that profile to keep medication administration safe and accurate.
3. **Report & Ask** uses that profile to reason about new symptoms against real history, not generic advice.
4. **My Results** uses that profile to make lab results and letters understandable.
5. **Handover** pulls everything together — symptoms, adherence, patterns — into a report ready for the doctor.

---

## Responsible AI and Safety

CareShield is built with the understanding that it sits between a frightened or exhausted family member and decisions about someone they love. Every part of the product reflects that responsibility.

**Privacy by design (in progress).** CareShield's data model is structured around per-patient profiles, with medications, conditions, symptom reports, and uploaded photos scoped to a patient rather than pooled globally — the architecture assumes multiple caregivers sharing one patient's record, not one account per patient. In this hackathon build, profile selection is intentionally left open and database policies are permissive, so we could iterate quickly on the clinical reasoning without also building full authentication— I made that tradeoff deliberately, not by oversight. Row Level Security is already enabled at the database level; the next concrete step is caregiver authentication and care-circle-scoped policies (a caregiver can only query rows for patients they've been added to), which the schema is already structured to support without a redesign. Documents and photos are processed to extract structured data, not stored or reused for anything else. Text sent to OpenAI's API for AI responses is not used to train OpenAI's models, per OpenAI's API data usage policy.

**Designed for the empty state, not just the happy path.** A brand-new patient profile has no documents, no symptom reports, and no logged doses — and CareShield treats that as an expected state, not an error. Handover explicitly tells the caregiver there isn't enough information yet to generate a report, rather than returning a blank or broken page. Documents shows "No medications on file yet" instead of silently omitting the section. Pill Registration and MedCam both detect the missing prerequisite and guide the caregiver to upload a discharge document first, rather than letting them register or verify a pill against data that doesn't exist. This matters clinically: a system that guesses in the absence of data is more dangerous than one that says it doesn't know yet.

**Catching mistakes before they happen, not after.** If a caregiver uploads a document for the wrong patient, Documents rejects it outright rather than silently attaching someone else's medical history to the active profile. If MedCam can't confidently identify a pill in a photo, it says so directly and tells the caregiver to check manually, instead of guessing a match. And when a scheduled dose is missing from a photo, MedCam doesn't just flag it as skipped — it names the medication and the risk that matters most in that moment: *"Amlodipine was due earlier today but was not in this photo. If you still need to give it, check the care plan for late doses — do not double up without checking with the doctor or pharmacist first."* A wrong guess here isn't a UX bug, it's a caregiver potentially over- or under-medicating a patient, so every uncertain outcome is designed to route back to the human rather than resolve itself silently.

**Decision support, not diagnosis.** CareShield never tells a caregiver what to do medically. Every AI response in Report & Ask is paired with a visible disclaimer, and the MedCam verification flow explicitly states "Review before giving" rather than issuing a pass or fail, because a photo of pills in a hand is never 100% certain. The product is designed to organize information and surface patterns, leaving every clinical judgment to the patient's actual doctor.

**Failure modes we designed for, not around.** A caregiver photographing pills will sometimes miss one in frame, photograph the wrong angle, or use poor lighting. Rather than guessing, MedCam reports exactly what it could and couldn't detect, for example "Only 1 of 2 pills detected, check you have the full dose," and flags medications missing from the photo as "missed but absent" instead of silently assuming they were taken. When confidence is low, the system says so directly: "Could not verify," rather than producing a false positive that gives a caregiver unearned confidence.

**Severity is surfaced, not buried.** Symptom reports are automatically scored as Monitor, Contact doctor, or Urgent, so a caregiver scanning a long history during a stressful moment sees what needs attention first, instead of having to read every entry to find the one that mattered.

**Human oversight stays with the human.** Caregivers choose what gets reported, decide whether to act on a MedCam warning, and control what gets shared in the Handover report. CareShield never auto-contacts a doctor, auto-administers a reminder as confirmation of a dose given, or makes any decision on the caregiver's behalf. It compiles and organizes; the person remains the one who acts.

---

CareShield is a caregiving support tool and does not provide medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for any health concerns. In an emergency, contact your local emergency services immediately.

---

## Status

This project is under active development.