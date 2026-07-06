from dotenv import load_dotenv
load_dotenv()

from ai_helpers import ask_ai, save_shift_log
import re

# --- Safety layer: red flags that do NOT depend on the LLM ---
# If these patterns appear, severity is ALWAYS forced to "urgent",
# without waiting on the model's judgment.
RED_FLAG_PATTERNS = [
    r"confus(ed|ion)",
    r"trouble (swallowing|breathing)",
    r"chest pain",
    r"face droop|facial asymmetry|one side of (his|her|the) face",
    r"sudden(ly)?",
    r"can'?t (move|speak)|unable to (move|speak)",
    r"fell|fall|fainted|passed out",
    r"severe (headache|pain)",
]

def detect_red_flags(text: str) -> list[str]:
    text_lower = text.lower()
    return [p for p in RED_FLAG_PATTERNS if re.search(p, text_lower)]


SYSTEM_PROMPT = """You are an assistant that helps family caregivers document a patient's condition for review by a healthcare professional. You do NOT diagnose and do NOT replace clinical judgment.

You will receive a caregiver's description of a patient's condition.
Respond with ONLY a JSON object (no extra text, no markdown), with these fields:

1. "empathetic_advice": a short, warm, practical tip for the caregiver (never clinical treatment advice — only situational guidance, e.g. "keep him seated and monitor closely")
2. "clinical_observations": up to 3 strings describing objective observations in neutral language (do NOT use diagnostic language like "possible stroke"; describe only what was reported, e.g. "difficulty swallowing reported", "new-onset confusion")
3. "doctor_note": the same information rewritten in neutral clinical language, third person, suitable for a doctor's report. Reported facts only — no diagnostic interpretation.
4. "severity": one of "ok", "monitor", "urgent". Use "urgent" for any combination of sudden onset, confusion, difficulty swallowing/breathing, chest pain, falls, or facial asymmetry.
5. "recommend_professional_contact": boolean. true if severity is "monitor" or "urgent".

Never generate a specific diagnosis (disease name). Limit yourself to describing and referring onward.
"""

def process_shift_report(caregiver_name: str, caregiver_message: str):
    # 1. Check for red flags BEFORE calling the model
    flags_found = detect_red_flags(caregiver_message)

    # 2. Call the model
    result = ask_ai(SYSTEM_PROMPT, caregiver_message)

    # 3. If the model didn't flag urgency but regex did, force "urgent"
    #    (never the other way around: if the model says urgent, that's respected)
    if flags_found and result.get("severity") != "urgent":
        result["severity"] = "urgent"
        result["doctor_note"] += f" [Automatic alert: warning patterns detected: {', '.join(flags_found)}]"

    # 4. Save to database
    save_shift_log(
        caregiver_name=caregiver_name,
        source="voice_report",
        summary=result["doctor_note"],
        severity=result["severity"],
    )

    # 5. If urgent, this should trigger an immediate notification
    #    (push, SMS, call) — not just sit in the DB for later review
    if result["severity"] == "urgent":
        # trigger_urgent_alert(caregiver_name, result)  # your notification logic here
        print("⚠️ URGENT ALERT — notify family/professional immediately")

    return result


# --- Usage ---
caregiver_message = "Dad woke up really confused this morning, refused breakfast, and had trouble swallowing his red pill. He also said his hip hurt when he got up."

result = process_shift_report("Carlos", caregiver_message)
print("AI result:", result)
print("Saved to Supabase!")