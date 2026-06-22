from ai_helpers import ask_ai, supabase

# Read the most recent shift events from the database (instead of hardcoded data)
response = supabase.table("shift_logs").select("*").order("created_at", desc=True).limit(10).execute()
shift_events = response.data

if not shift_events:
    print("No shift events found yet. Run main.py or medcam.py first to generate some data.")
else:
    system_prompt = """You are a medical assistant AI generating a shift handover card for family caregivers.
You will receive a list of events from the current caregiver's shift, each with who reported it, the source, summary, and severity.

Respond with ONLY a JSON object (no extra text, no markdown), with these fields:

1. "patient_status": one short sentence describing the patient's current overall state
2. "next_critical_action": the single most important thing the next caregiver needs to do
3. "watch_for": one short alert about what to monitor
4. "reported_by": a list of objects, each with "caregiver" (name) and "note" (what they reported)
"""

    result = ask_ai(system_prompt, str(shift_events))

    print("Handover card:", result)
    