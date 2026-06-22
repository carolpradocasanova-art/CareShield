# CareShield 🛡️

An AI copilot for family caregivers of high-dependency patients (Parkinson's, post-stroke, dementia).

## The Problem

Hospitals like Burjeel Holdings save lives in the operating room — but when the patient goes home, the family caregiver is left alone, untrained, and overwhelmed. Medication errors, missed symptoms, and chaotic shift handovers between caregivers put patients at risk.

## How It Works — Three Layers

### 1. Voice & Support `main.py`) — "The Shadow Chart"

The caregiver describes the patient's condition in plain speech (simulated as text for now). The AI extracts clinical signals and responds with empathetic, practical advice — while quietly generating a clinical note a doctor could use.

### 2. Vision & Safety `medcam.py`) — "MedCam"

The caregiver photographs pills before administering them. The AI compares what it sees against the patient's discharge plan and flags mismatches before a medication error happens.

### 3. Handover `handover.py`) — "The Handover Blueprint"

When a shift ends, the AI condenses everything that happened — symptoms, medication checks, who reported what — into a short, scannable card for the next caregiver.

## Project Structure

## Tech Stack

- **AI**: OpenAI GPT-4o (text + vision)

- **Backend**: Python

- **Coming soon**: Supabase (database), Streamlit (frontend)

## Status

- ✅ Layer 1 (Voice) — working

- ✅ Layer 2 (Vision) — working

- ✅ Layer 3 (Handover) — working

- ✅ Shared, reusable AI logic `ai_helpers.py`)

- ⬜ Database (Supabase)

- ⬜ Frontend (Streamlit)

- ⬜ Multi-caregiver profile switcher

## Responsible AI Notes

- The AI's medication-matching judgment is a **decision support tool, not a replacement for clinical verification** — flagged mismatches should always be confirmed by checking the original prescription.

- Patient data is currently simulated; a production version would need encryption and access controls before storing real medical information.

- The system is designed to fail toward caution: when uncertain, severity defaults toward "monitor" or "urgent" rather than "ok."