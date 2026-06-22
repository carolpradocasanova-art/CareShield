from ai_helpers import ask_ai, save_shift_log

caregiver_message = "Dad woke up really confused this morning, refused breakfast, and had trouble swallowing his red pill. He also said his hip hurt when he got up."

system_prompt = """You are a medical assistant AI supporting family caregivers.
You will receive a caregiver's spoken description of a patient's condition.
Respond with ONLY a JSON object (no extra text, no markdown), with these fields:

1. "empathetic_advice": a short, warm, practical tip for the caregiver
2. "clinical_tags": a list of up to 3 short strings naming possible clinical risks (e.g. "Possible dysphagia")
3. "doctor_note": the same information rewritten in neutral, clinical language, suitable for a doctor's report
4. "severity": one of "ok", "monitor", or "urgent"
"""

result = ask_ai(system_prompt, caregiver_message)

print("AI result:", result)

# Save this event to the database
save_shift_log(
    caregiver_name="Carlos",
    source="voice_report",
    summary=result["doctor_note"],
    severity=result["severity"]
)

print("Saved to Supabase!")
