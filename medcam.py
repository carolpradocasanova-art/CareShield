import base64
from ai_helpers import ask_ai, save_shift_log

image_path = "test_pills.jpg"
with open(image_path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

discharge_plan = "Patient should take 1 round white pill (Paracetamol 500mg) every morning at 8:00 AM."

system_prompt = """You are a medical assistant AI helping a caregiver verify medication before giving it to a patient.
You will receive a photo of pills in the caregiver's hand, plus the patient's discharge plan.
Compare what you see in the photo to the discharge plan.

Respond with ONLY a JSON object (no extra text, no markdown), with these fields:

1. "pills_detected": a short description of what you see in the photo (shape, color, count)
2. "matches_plan": true or false
3. "caregiver_message": a short, clear message for the caregiver
4. "severity": "ok" if it matches, "urgent" if it does not match
"""

user_content = [
    {"type": "text", "text": f"Discharge plan: {discharge_plan}"},
    {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
    }
]

result = ask_ai(system_prompt, user_content)

print("AI result:", result)

save_shift_log(
    caregiver_name=selected_caregiver,
    source="medication_check",
    summary=f"Pills detected: {result['pills_detected']}. Matches plan: {result['matches_plan']}.",
    severity=result["severity"]
)

print("Saved to Supabase!")