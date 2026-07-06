
import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client
 
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
 
CHUNKS = [
    # ── FEVER ───────────────────────────────────────────────────────────────
    {
        "source": "CareShield Medical Knowledge Base",
        "title": "Fever in elderly patients — assessment and response",
        "content": """In older adults, fever is defined as a temperature ≥38°C (100.4°F).
Elderly patients can have serious infections with low-grade or no fever, so any
temperature change should be taken seriously.
 
CALL THE DOCTOR IMMEDIATELY IF:
- Temperature ≥39.5°C (103°F)
- Fever with confusion, disorientation or unusual drowsiness
- Fever with difficulty breathing
- Fever with pain when urinating (possible urinary tract infection, common in elderly)
- Fever lasting more than 48 hours without a clear cause
- Fever in a patient who is immunosuppressed or has diabetes
 
HOME MANAGEMENT (mild fever 38–38.9°C):
- Paracetamol 500–1000mg every 6 hours (do not exceed 3g/day in elderly)
- Ibuprofen ONLY if the doctor has authorised it (renal risk in older adults)
- Hydration: offer water, broths or juices every 30 minutes
- Light clothing, cool environment (do not over-wrap)
- Check temperature every 2 hours and record it
- Do NOT use aspirin in patients over 65 without medical indication
 
ADDITIONAL RED FLAGS:
- Neck stiffness with fever → emergency (possible meningitis)
- Rash with fever → call the doctor
- Chest pain with fever → emergency"""
    },
    {
        "source": "CareShield Medical Knowledge Base",
        "title": "Urinary tract infection in elderly — atypical symptoms",
        "content": """Urinary tract infections (UTIs) are the most common bacterial infection
in older adults and frequently present without classic symptoms.
 
TYPICAL SYMPTOMS (may be absent in elderly):
- Burning sensation when urinating
- Urgent and frequent need to urinate
- Cloudy or foul-smelling urine
 
ATYPICAL SYMPTOMS IN ELDERLY (warning signs):
- Sudden confusion or delirium (especially in patients over 75)
- Unexplained falls
- Agitation or behaviour changes
- Low-grade fever or no fever at all
- General weakness without obvious cause
 
ACTION: If an elderly person presents sudden confusion with no other obvious cause,
suspect a UTI and contact the doctor that same day. Untreated UTIs can rapidly
progress to sepsis in older adults.
 
PREVENTION:
- Adequate hydration (at least 6–8 glasses of water per day)
- Correct intimate hygiene (front to back)
- Do not hold urine
- Change pads or incontinence products frequently if used"""
    },
 
    # ── MEDICATIONS ─────────────────────────────────────────────────────────
    {
        "source": "CareShield Medical Knowledge Base",
        "title": "Safe medication management in elderly patients",
        "content": """Older adults metabolise medications more slowly, increasing the risk of
side effects and toxic accumulation.
 
KEY PRINCIPLES:
- Never double a dose if one is missed — take the next dose at the normal time
- Do not stop chronic medication (antihypertensives, anticoagulants, antidiabetics)
  without consulting the doctor
- Keep an updated medication list visible at home
- Use a weekly pill organiser to avoid missed or double doses
 
COMMON DANGEROUS INTERACTIONS:
- Anticoagulants (warfarin, acenocoumarol) + NSAIDs (ibuprofen, naproxen) → bleeding risk
- Antihypertensives + diuretics + heat → risk of hypotension and falls
- Metformin + iodinated contrast → stop 48h before contrast imaging procedures
- Benzodiazepines (diazepam, lorazepam) → significantly increase fall risk in elderly
 
SYMPTOMS REQUIRING URGENT MEDICATION REVIEW:
- New confusion after starting or changing a medication
- Dizziness or orthostatic hypotension (on standing)
- Persistent nausea
- Unusual bleeding (gums, dark stools, bruising)
- New skin rash"""
    },
    {
        "source": "CareShield Medical Knowledge Base",
        "title": "Paracetamol and ibuprofen — safe use in elderly patients",
        "content": """PARACETAMOL (acetaminophen):
- First-line choice for pain and fever in elderly patients
- Dose: 500–1000mg every 6–8 hours
- Maximum daily dose: 3000mg in patients over 65 (not 4g as in younger adults)
- CAUTION: reduce dose if liver disease or regular alcohol use
- Can be taken with or without food
 
IBUPROFEN AND OTHER NSAIDs:
- AVOID in patients over 65 unless explicitly authorised by the doctor
- Risks: kidney damage, gastric ulcer, fluid retention, raised blood pressure
- If the doctor authorises: always take with food, lowest possible dose,
  maximum 3–5 days
 
ASPIRIN:
- Do NOT use as a painkiller in elderly patients
- Only if prescribed by the doctor at low dose as antiplatelet therapy (75–100mg)
 
SIGNS OF PARACETAMOL OVERDOSE:
- Nausea, vomiting, abdominal pain in the first few hours
- Yellowing of skin or eyes (jaundice) at 24–72h
- → Emergency: go to A&E immediately"""
    },
 
    # ── RED FLAGS ────────────────────────────────────────────────────────────
    {
        "source": "CareShield Medical Knowledge Base",
        "title": "Red flags — when to call emergency services (999 / 112)",
        "content": """CALL EMERGENCY SERVICES IMMEDIATELY if the patient shows:
 
NEUROLOGICAL:
- Sudden weakness or numbness in face, arm or leg (especially one side)
- Sudden difficulty speaking or understanding speech
- Sudden loss of vision in one or both eyes
- Very intense headache of sudden onset ("worst headache of my life")
- Loss of consciousness or unresponsive
 
CARDIORESPIRATORY:
- Chest pain lasting more than 2 minutes
- Difficulty breathing at rest
- Blue lips or fingernails (cyanosis)
- Very fast, very slow or irregular pulse with dizziness
 
DIGESTIVE:
- Vomiting blood or coffee-ground material
- Black, tarry stools (melaena) → digestive bleeding
- Very sudden, severe abdominal pain
 
OTHER:
- Fall with head trauma, loss of consciousness or subsequent confusion
- Suspected fracture (do not move the patient)
- Severe allergic reaction: swelling of lips/throat, difficulty breathing
 
GENERAL RULE: When in doubt, call. A unnecessary call is better than
delaying a medical emergency."""
    },
    {
        "source": "CareShield Medical Knowledge Base",
        "title": "Signs of rapid deterioration in elderly — act today",
        "content": """These signs are not an immediate emergency but require same-day medical contact:
 
- New confusion or worsening of existing confusion
- Refusal to eat or drink for more than 12 hours
- Pain not controlled with usual analgesics
- Fever >38.5°C that does not come down with paracetamol
- Fall without apparent trauma but with pain on movement
- New swelling in legs or ankles
- Very dark urine or no urination in more than 8 hours
- Surgical wound with redness, warmth, pus or separated edges
- Sudden change in activity level (from active to bedridden)
 
DELIRIUM IN ELDERLY PATIENTS:
Delirium (acute confusion, disorientation, agitation) is a medical emergency
in older adults even if it appears to be "just confusion".
Common causes: infection, dehydration, medication change,
urinary retention, severe constipation.
→ Contact the doctor that same day."""
    },
 
    # ── POST-OPERATIVE CARE ──────────────────────────────────────────────────
    {
        "source": "CareShield Medical Knowledge Base",
        "title": "Post-operative home care for elderly patients",
        "content": """FIRST 48 HOURS AFTER DISCHARGE:
- Monitor temperature every 6 hours (post-op fever may indicate infection)
- Check wound: some redness and mild swelling is normal in the first few days
- Pain: follow the analgesic schedule from discharge exactly, do not skip doses
- Mobilisation: follow the physiotherapist/surgeon instructions precisely
- Hydration and nutrition: offer small, frequent meals
 
SIGNS OF POST-OPERATIVE COMPLICATION — call surgeon or go to A&E:
- Fever >38.5°C after day 2 of discharge
- Wound: spreading redness, excessive warmth, pus, bad odour, open edges
- Pain that increases rather than improves after 48h
- Leg swelling with pain and warmth (possible deep vein thrombosis)
- Difficulty breathing or chest pain (possible pulmonary embolism)
 
WOUND CARE:
- Do not wet the wound until medically indicated
- Change dressing as per discharge instructions
- Do not apply creams, alcohol or home remedies without medical indication
- If staples or sutures: do not remove at home
 
SAFE MOBILISATION AFTER SURGERY:
- Knee/hip arthroplasty: do not cross legs, do not bend without assistance
- Abdominal surgery: do not lift >2kg for the first 4 weeks
- Always wear closed, non-slip footwear"""
    },
    {
        "source": "CareShield Medical Knowledge Base",
        "title": "Recovery after knee replacement — home care",
        "content": """Total knee replacement (arthroplasty) requires specific home care
to avoid complications.
 
FIRST 2 WEEKS:
- Elevate the operated leg when seated or in bed (pillow under the heel)
- Ice on the knee for 20 minutes, 3–4 times a day to reduce swelling
- Physiotherapy exercises as prescribed — do not skip any session
- Use crutches or walker as instructed — do not bear weight without authorisation
 
WARNING SIGNS:
- Swollen, warm, painful calf → possible DVT → go to A&E
- Wound weeping or opening
- Fever >38°C
- Knee that "gives way" or sharp pain when trying to move
 
TYPICAL MEDICATION AFTER SURGERY:
- Anticoagulant (low molecular weight heparin or tablets):
  Do NOT miss any dose — DVT risk
- Analgesia: follow discharge schedule exactly
- Omeprazole or similar: protects the stomach if taking NSAIDs or steroids
 
RECOVERY MILESTONES:
- Week 1–2: bend the knee to 90°
- Week 4–6: walking indoors without crutches
- 3 months: normal low-intensity activity"""
    },
 
    # ── FALLS AND MOBILITY ───────────────────────────────────────────────────
    {
        "source": "CareShield Medical Knowledge Base",
        "title": "Fall prevention in elderly patients",
        "content": """Falls are the leading cause of serious injury in older adults.
1 in 3 adults over 65 falls at least once a year.
 
RISK FACTORS TO MANAGE:
- Medications: antihypertensives, diuretics, sleeping pills, antidepressants increase risk
- Vision: annual eye check-up
- Footwear: closed toe, non-slip sole, no heel, well-fitted
- Environment: loose rugs, cables on the floor, poor lighting
 
HOME ADAPTATIONS:
- Bathroom: grab rails next to the toilet and in the shower, non-slip mat,
  shower seat if needed
- Bedroom: bed at appropriate height, night light, phone within reach
- Stairs: handrails on both sides, good lighting
- Remove loose rugs and obstacles in walking areas
 
WHAT TO DO AFTER A FALL:
1. Do not move the patient if there is pain in the neck, back or suspected fracture
2. Call 999/112 if: loss of consciousness, confusion, serious wound,
   inability to move a limb
3. If able to move: help get up slowly, check for injuries
4. Record the fall: time, location, activity, whether there was dizziness beforehand
5. All falls should be reported to the doctor to review medication and risk factors"""
    },
    {
        "source": "CareShield Medical Knowledge Base",
        "title": "What to do if an elderly patient falls and cannot get up",
        "content": """IF THE PATIENT IS ON THE FLOOR AND CANNOT GET UP:
 
STEP 1 — Assess:
- Are they conscious and responding?
- Is there intense pain in the hip, thigh or back? → Do not move, call 999/112
- Is there a head wound or did they lose consciousness? → Call 999/112
 
STEP 2 — If no serious injury is suspected:
- Stay calm and reassure the patient
- Place a pillow under their head
- Cover with a blanket (cold worsens shock)
- Call for help — do not try to lift an elderly person alone
 
STEP 3 — Safe assisted standing (if the patient can cooperate):
1. Move a sturdy chair next to the patient
2. Ask them to roll to one side and push up onto hands and knees
3. Crawl towards the chair
4. Place hands on the seat and raise the stronger leg first
5. Slowly turn and sit down
 
STEP 4 — After the fall:
- Observe for the next 24 hours: confusion, vomiting, increasing headache
  → A&E (possible delayed intracranial haemorrhage)
- Report to the doctor even if the fall appears minor"""
    },
 
    # ── DIABETES ─────────────────────────────────────────────────────────────
    {
        "source": "CareShield Medical Knowledge Base",
        "title": "Type 2 diabetes in elderly — management and red flags",
        "content": """Blood glucose control in elderly patients should be less strict than in
younger adults to avoid hypoglycaemia, which is more dangerous and less
symptomatic in older people.
 
BLOOD GLUCOSE TARGETS IN ELDERLY (indicative — confirm with doctor):
- Fasting glucose: 90–150 mg/dL
- Postprandial glucose (2h after eating): <200 mg/dL
- HbA1c: 7.5–8.5% (less strict than in younger patients)
 
HYPOGLYCAEMIA (glucose <70 mg/dL) — URGENT:
Symptoms: trembling, sweating, confusion, pallor, sudden hunger
In elderly patients it may present ONLY as confusion or drowsiness
 
Immediate treatment:
1. If conscious: 15g of fast-acting sugar (3 sugar sachets, 150ml fruit juice,
   3 glucose sweets)
2. Wait 15 minutes and recheck blood glucose
3. If no improvement or unconscious → call 999/112
 
HYPERGLYCAEMIA (glucose >300 mg/dL):
- Symptoms: extreme thirst, very frequent urination, fatigue, blurred vision
- Contact doctor that day
- If vomiting, difficulty breathing or confusion → A&E
 
DIABETIC FOOT CARE:
- Check feet daily (use a mirror if unable to reach)
- Any wound, blister or redness → doctor that day
- Do not cut corns or ingrown toenails at home
- Always wear footwear, never barefoot"""
    },
 
    # ── HYPERTENSION ─────────────────────────────────────────────────────────
    {
        "source": "CareShield Medical Knowledge Base",
        "title": "Hypertension in elderly — management and hypertensive crisis",
        "content": """Hypertension is very common in older adults. Blood pressure targets
in elderly patients are less strict: <150/90 mmHg in patients over 65.
 
CORRECT BLOOD PRESSURE MEASUREMENT:
- Rest for 5 minutes beforehand
- Seated, back supported, legs uncrossed
- Arm at heart level
- Do not talk during the measurement
- Take 2 readings 1 minute apart and average them
- Same time every day (ideally morning before medication)
 
REFERENCE VALUES:
- Normal: <140/90
- Elevated but manageable at home: 140–160/90–100 → record and inform doctor
- Hypertensive crisis: >180/120 → call doctor urgently
 
HYPERTENSIVE CRISIS WITH SYMPTOMS → EMERGENCY (999/112):
- BP >180/120 + severe headache
- BP >180/120 + chest pain
- BP >180/120 + blurred or double vision
- BP >180/120 + difficulty speaking or moving limbs
 
DO NOT DO AT HOME:
- Do not double the antihypertensive dose without medical instruction
- Do not stop treatment because BP readings look good
  (the medication IS the reason BP is controlled)
- Do not use sublingual nifedipine (dangerous — causes sudden BP drop)
 
ORTHOSTATIC HYPOTENSION (dizziness on standing):
Common in elderly patients on antihypertensives. Prevention:
- Rise in 2 stages: sit first, wait 30 seconds, then stand
- Adequate hydration
- Compression stockings if recommended by the doctor"""
    },
]
 
def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding
 
def seed():
    print(f"Seeding {len(CHUNKS)} chunks into medical_knowledge...\n")
    for i, chunk in enumerate(CHUNKS):
        print(f"[{i+1}/{len(CHUNKS)}] Embedding: {chunk['title']}")
        text_to_embed = f"{chunk['title']}\n\n{chunk['content']}"
        embedding = get_embedding(text_to_embed)
        supabase.table("medical_knowledge").insert({
            "source": chunk["source"],
            "source_url": chunk.get("source_url"),
            "title": chunk["title"],
            "content": chunk["content"],
            "embedding": embedding,
        }).execute()
        time.sleep(0.3)
    print(f"\n✓ Done. {len(CHUNKS)} chunks inserted.")
 
if __name__ == "__main__":
    seed()
 