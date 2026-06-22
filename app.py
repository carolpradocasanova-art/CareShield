import streamlit as st
import base64
from ai_helpers import ask_ai, save_shift_log, supabase, generate_sbar_pdf, extract_text_from_pdf, save_patient_plan, get_latest_patient_plan
from streamlit_javascript import st_javascript
from datetime import datetime
import pytz

st.set_page_config(page_title="CareShield", page_icon="🛡️", layout="centered")

st.title("🛡️ CareShield")

user_timezone = st_javascript("""await (async () => {
    return Intl.DateTimeFormat().resolvedOptions().timeZone;
})().then(returnValue => returnValue)""")

if user_timezone:
    tz = pytz.timezone(user_timezone)
    current_time = datetime.now(tz)
    current_time_str = current_time.strftime("%A, %B %d, %Y at %I:%M %p (%Z)")
else:
    current_time_str = "Unknown (timezone not detected yet)"

caregiver_options = ["Carlos (son)", "María (daughter)", "Night nurse"]
selected_caregiver = st.selectbox("Who's logged in?", caregiver_options)

tab1, tab2, tab3, tab4 = st.tabs(["💬 Ask & Report", "💊 MedCam", "🔄 Handover", "📄 Hospital Documents"])

# ---------- TAB 1: ASK & REPORT ----------
with tab1:
    st.subheader("Ask & Report")

    mode = st.radio(
        "What would you like to do?",
        ["Report something", "Ask a question"],
        key="ask_report_mode"
    )

    if mode == "Report something":
        st.caption("💡 Examples: 'Dad woke up very confused and refused breakfast' · 'Mum fell getting out of bed' · 'He's been more tired than usual since starting the new pill'")
        caregiver_message = st.text_area(
            "Describe how the patient is doing:",
            placeholder="e.g. Dad woke up confused this morning, refused breakfast, and had trouble swallowing his pill...",
            key="voice_input"
        )

        if st.button("Get advice", key="voice_button"):
            if caregiver_message.strip() == "":
                st.warning("Please describe the patient's condition first.")
            else:
                with st.spinner("Analyzing..."):
                    system_prompt = """You are a medical assistant AI supporting family caregivers.
You will receive a caregiver's spoken description of a patient's condition.
Respond with ONLY a JSON object (no extra text, no markdown), with these fields:

1. "empathetic_advice": a short, warm, practical tip for the caregiver
2. "clinical_tags": a list of up to 3 short strings naming possible clinical risks
3. "doctor_note": the same information rewritten in neutral, clinical language
4. "severity": one of "ok", "monitor", or "urgent"
"""
                    result = ask_ai(system_prompt, caregiver_message)

                if result.get("error"):
                    st.error(result["message"])
                else:
                    save_shift_log(
                        caregiver_name=selected_caregiver,
                        source="voice_report",
                        summary=result["doctor_note"],
                        severity=result["severity"]
                    )
                    st.success(result["empathetic_advice"])
                    st.write("**Clinical tags:**")
                    for tag in result["clinical_tags"]:
                        st.write(f"- {tag}")
                    st.write(f"**Severity:** {result['severity']}")

    else:  # Ask a question
        st.caption("💡 Examples: 'When does Dad need his next pill?' · 'Is it normal he's been more confused since starting Clopidogrel?' · 'Can he take Paracetamol and Omeprazole together?'")
        caregiver_question = st.text_input(
            "What would you like to know?",
            placeholder="e.g. When does Dad need his next pill? Is it normal he's sleepier on this medication?",
            key="question_input"
        )

        if st.button("Ask", key="question_button"):
            if caregiver_question.strip() == "":
                st.warning("Please type a question first.")
            else:
                with st.spinner("Looking into it..."):
                    latest_plan = get_latest_patient_plan()
                    plan_context = latest_plan["medications"] if latest_plan else "No medication plan has been uploaded yet."

                    system_prompt = f"""You are a medical assistant AI answering a family caregiver's question about their patient.
The patient's current medication plan is: {plan_context}
The current date and time, in the caregiver's local timezone, is: {current_time_str}

Answer the caregiver's question clearly and reassuringly when appropriate, but always recommend contacting a doctor for anything urgent or outside routine care.

Respond with ONLY a JSON object (no extra text, no markdown), with these fields:

1. "answer": a clear, direct answer to the caregiver's question
2. "needs_doctor": true if this question suggests something a doctor should be consulted about, false otherwise
"""
                    result = ask_ai(system_prompt, caregiver_question)

                if result.get("error"):
                    st.error(result["message"])
                else:
                    st.success(result["answer"])
                    if result["needs_doctor"]:
                        st.warning("⚠️ Consider reaching out to the patient's doctor about this.")


# ---------- TAB 2: MEDCAM ----------
with tab2:
    st.subheader("MedCam — Medication Check")

    latest_plan = get_latest_patient_plan()
    if latest_plan:
        discharge_plan = latest_plan["medications"]
        st.info(f"📋 Using saved medication plan: {discharge_plan}")
    else:
        discharge_plan = "No medication plan uploaded yet. Please upload a hospital document first."
        st.warning("⚠️ No medication plan found. Upload one in Hospital Documents first.")

    uploaded_image = st.file_uploader("Take or upload a photo of the pills", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Pills to verify", width=300)

    if current_time_str != "Unknown (timezone not detected yet)":
        st.caption(f"🕐 Current time: {current_time_str}")

    if st.button("Check medication", key="medcam_button"):
        if uploaded_image is None:
            st.warning("Please upload a photo first.")
        else:
            with st.spinner("Checking pills against discharge plan..."):
                base64_image = base64.b64encode(uploaded_image.read()).decode("utf-8")

                system_prompt = """You are a medical assistant AI helping a caregiver verify medication before giving it to a patient.
You will receive a photo of pills in the caregiver's hand, plus the patient's discharge plan.
Compare what you see in the photo to the discharge plan.

Respond with ONLY a JSON object (no extra text, no markdown), with these fields:

1. "pills_detected": a short description of what you see in the photo
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

            if result.get("error"):
                st.error(result["message"])
            else:
                save_shift_log(
                    caregiver_name=selected_caregiver,
                    source="medication_check",
                    summary=f"Pills detected: {result['pills_detected']}. Matches plan: {result['matches_plan']}.",
                    severity=result["severity"]
                )
                if result["matches_plan"]:
                    st.success(result["caregiver_message"])
                else:
                    st.error(result["caregiver_message"])


# ---------- TAB 3: HANDOVER ----------
with tab3:
    st.subheader("Shift Handover Card")

    if st.button("Generate handover card", key="handover_button"):
        with st.spinner("Reading shift history and generating card..."):
            response = supabase.table("shift_logs").select("*").order("created_at", desc=True).limit(10).execute()
            shift_events = response.data

            if not shift_events:
                st.warning("No shift events found yet. Use Ask & Report or MedCam first.")
            else:
                system_prompt = """You are a medical assistant AI generating a professional shift handover report for family caregivers, using the SBAR framework (Situation, Background, Assessment, Recommendation) — the same structure used in real hospital shift changes.

You will receive a list of events from the current caregiver's shift, each with the caregiver's name, source, summary, and severity.
Refer to caregivers by name when relevant.

Respond with ONLY a JSON object (no extra text, no markdown), with these fields:

1. "situation": one short sentence on what's happening right now with the patient
2. "background": relevant context from the shift (what led to the current state)
3. "assessment": the system's clinical interpretation of severity and risk
4. "recommendation": the single most important action the next caregiver needs to take
5. "watch_for": one short alert about what to monitor going forward
6. "reported_by": a list of objects, each with "caregiver" (name) and "note" (what they reported)
"""
                result = ask_ai(system_prompt, str(shift_events))

                if result.get("error"):
                    st.error(result["message"])
                else:
                    st.markdown("### 📋 SBAR Handover Report")

                    card_style = """
                        padding: 20px;
                        border-radius: 12px;
                        height: 180px;
                        overflow-y: auto;
                        margin-bottom: 16px;
                        font-size: 15px;
                        line-height: 1.5;
                    """

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("🔵 **Situation**")
                        st.markdown(f'<div style="{card_style} background-color: #dbeafe; color: #1e3a5f;">{result["situation"]}</div>', unsafe_allow_html=True)
                        st.markdown("🟡 **Assessment**")
                        st.markdown(f'<div style="{card_style} background-color: #fef9c3; color: #713f12;">{result["assessment"]}</div>', unsafe_allow_html=True)

                    with col2:
                        st.markdown("🟣 **Background**")
                        st.markdown(f'<div style="{card_style} background-color: #ede9fe; color: #3b0764;">{result["background"]}</div>', unsafe_allow_html=True)
                        st.markdown("🔴 **Recommendation**")
                        st.markdown(f'<div style="{card_style} background-color: #ffe4e6; color: #881337;">{result["recommendation"]}</div>', unsafe_allow_html=True)

                    st.markdown(f"**👁️ Watch for:** {result['watch_for']}")

                    st.write("**Reported by:**")
                    for entry in result["reported_by"]:
                        st.write(f"- **{entry['caregiver']}**: {entry['note']}")

                    pdf_bytes = generate_sbar_pdf(result)
                    st.download_button(
                        label="📄 Download SBAR Report (PDF)",
                        data=pdf_bytes,
                        file_name="careshield_handover_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )


# ---------- TAB 4: HOSPITAL DOCUMENTS ----------
with tab4:
    st.subheader("Upload Hospital Documents")
    st.write("Upload a discharge plan or prescription PDF, and CareShield will automatically learn the patient's medication schedule.")

    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_uploader")

    if st.button("Process document", key="process_pdf_button"):
        if uploaded_pdf is None:
            st.warning("Please upload a PDF first.")
        else:
            with st.spinner("Reading document and extracting medication plan..."):
                raw_text = extract_text_from_pdf(uploaded_pdf)

                system_prompt = """You are a medical assistant AI reading a hospital discharge document.
Extract the patient's medication plan and current diagnoses from the text.

Respond with ONLY a JSON object (no extra text, no markdown), with these fields:

1. "medications": a clear, structured summary of what medications the patient should take, including dosage and timing
2. "conditions": a short summary of the patient's current diagnoses or medical conditions
"""
                result = ask_ai(system_prompt, raw_text)

            if result.get("error"):
                st.error(result["message"])
            else:
                save_patient_plan(raw_text, result["medications"])
                st.success("Document processed! Medication plan saved.")

    st.divider()
    st.write("**Patient Overview:**")
    latest_plan = get_latest_patient_plan()
    if latest_plan:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("💊 **Current Medications**")
            st.info(latest_plan["medications"])
        with col2:
            st.markdown("🩺 **Current Conditions**")
            st.info("Upload a document to extract conditions.")
    else:
        st.write("No plan uploaded yet.")