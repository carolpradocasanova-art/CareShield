content = open("app.py").read()

old = '''# TAB 2 — MEDCAM
# ═══════════════════════════_patient_plan()
    if latest_plan:
        plan_items = parse_medications_for_display(latest_plan["medications"])
        discharge_plan = format_medications_for_prompt(plan_items)
    else:
        plan_items = DEMO_MEDICATIONS
        discharge_plan = format_medications_for_prompt(plan_items)
    med_refs = get_medication_references()
    registered_names = {r["medication_name"] for r in med_refs}
    unregistered = [m for m in plan_items if m["name"] not in registered_names]
    all_registered = len(unregistered) == 0 and len(plan_items) > 0
    with st.container(border=True):
        md_html(f"""
        <div class="cs-medcam-card">
          <div class="cs-medcam-header">
            <div class="cs-medcam-title">MedCam — Medication Check</div>
          </div>
          <p class="cs-medcam-desc">
            MedCam uses advanced vision AI to verify your prescription safety.
            Simply take a photo of the pills in your hand to confirm your dosage.
          </p>
          <div class="cs-medcam-active-plan">
            <div class="cs-active-plan-label">Active plan</div>
            {active_plan_html}
          </div>
          <p class="cs-medcam-upload-label">Take or upload a photo of the pills</p>
        </div>
        """)
        uploaded_image = st.file_uploader(
            "Take or upload a photo of the pills",
            type=["jpg", "jpeg", "png"],
            key="medcam_uploader",
            label_visibility="collapsed",
        )
        if uploaded_image:
            st.image(uploaded_image, caption="Pills to verify", width=280)
        check_clicked = st.button("Check medication", key="medcam_button", use_container_width=True)
    if not latest_plan:
        st.warning("No medication plan found. Upload one in Documents first.")
    if check_clicked:
        if uploaded_image is None:
            st.warning("Please upload a photo first.")
        else:
            with st.spinner("Analyzing medication identity and dosage alignment..."):
                base64_image = base64.b64encode(uploaded_image.read()).decode("utf-8")
                system_prompt = """You are a medical assistant AI helping verify medication before giving it to a patient.
Compare what you see in the photo to the discharge plan.
Respond with ONLY a JSON object with:
1. "pills_detected": a short description of what you see
2. "matches_plan": true or false
3. "caregiver_message": a short, clear message for the caregiver
4. "severity": "ok" if matches, "urgent" if not
"""
                user_content = [
                    {"type": "text", "text": f"Discharge plan: {discharge_plan}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ]
                result = ask_ai(system_prompt, user_content)
            if result.get("error"):
                st.error(result["message"])
            elif result.get("matches_plan"):
                log_time = format_medcam_log_time()
                save_shift_log(
                    caregiver_name=selected_caregiver,
                    source="medication_check",
                    summary=f"Pills detected: {result['pills_detected']}. Verified at {log_time}.",
                    severity=result.get("severity", "ok"),
                )
                md_html(f"""
                <div class="cs-medcam-result cs-medcam-result-success">
                  <strong>✓ Verified</strong>
                  The pills in your hand match your scheduled prescription.
                  We have logged that you took your medication today at {html.escape(log_time)}.
                </div>
                """)
            else:
                save_shift_log(
                    caregiver_name=selected_caregiver,
                    source="medication_check",
                    summary=f"Pills detected: {result.get('pills_detected', 'unknown')}. Match failed.",
                    severity=result.get("severity", "urgent"),
                )
                md_html("""
                <div class="cs-medcam-result cs-medcam-result-fail">
                  <strong>⚠ Warning</strong>
                  Unrecognized medication or incorrect dosage. The pills detected do not match
                  your active plan. Please verify before taking.
                </div>
                """)
    md_html(build_stored_conditions_medcam_html(get_stored_conditions()))'''

new = '''# TAB 2 — MEDCAM
# ════════════════════════════════════════════════════════════════
with tab2:
    from ai_helpers import get_medication_references, save_medication_reference, delete_medication_reference
    latest_plan = get_latest_patient_plan()
    if latest_plan:
        plan_items = parse_medications_for_display(latest_plan["medications"])
        discharge_plan = format_medications_for_prompt(plan_items)
    else:
        plan_items = DEMO_MEDICATIONS
        discharge_plan = format_medications_for_prompt(plan_items)
    med_refs = get_medication_references()
    registered_names = {r["medication_name"] for r in med_refs}
    unregistered = [m for m in plan_items if m["name"] not in registered_names]
    all_registered = len(unregistered) == 0 and len(plan_items) > 0

    # ── ENROLLMENT SECTION ──────────────────────────────────────
    if not all_registered:
        with st.container(border=True):
            md_html("""
            <div class="cs-medcam-card">
              <div class="cs-medcam-title">MedCam — First-time Setup</div>
              <p class="cs-medcam-desc">
                Before using MedCam, you need to register each medication once.
                This lets MedCam recognise the exact pills your patient takes —
                since the same medication can look different depending on the brand or pharmacy.<br><br>
                <strong>You only need to do this once per medication.</strong>
                If the brand changes or a new medication is added, register it again below.
              </p>
            </div>
            """)
            for med in unregistered:
                st.markdown(f"**Register: {med['name']}**")
                ref_photo = st.file_uploader(
                    f"Photo of {med['name']}",
                    type=["jpg", "jpeg", "png"],
                    key=f"ref_upload_{med['name']}",
                    label_visibility="collapsed",
                )
                if ref_photo:
                    st.image(ref_photo, width=180, caption=f"Reference photo for {med['name']}")
                    if st.button(f"Save reference for {med['name']}", key=f"ref_save_{med['name']}"):
                        ref_b64 = base64.b64encode(ref_photo.read()).decode("utf-8")
                        with st.spinner("Saving reference..."):
                            save_medication_reference(
                                medication_name=med["name"],
                                image_b64=ref_b64,
                                description="Registered via MedCam setup",
                            )
                        st.success(f"✓ {med['name']} registered. Reload the page to continue.")
                st.markdown("---")
        if med_refs:
            st.info(f"{len(med_refs)} of {len(plan_items)} medications registered. Register the remaining ones above to unlock MedCam.")
        else:
            st.info("Register all your medications above to unlock MedCam verification.")

    # ── REGISTERED MEDICATIONS MANAGEMENT ───────────────────────
    if med_refs:
        with st.expander("Registered medications — click to view or remove"):
            for ref in med_refs:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{ref['medication_name']}**")
                with col2:
                    if st.button("Remove", key=f"del_ref_{ref['id']}"):
                        delete_medication_reference(ref["id"])
                        st.rerun()

    # ── MEDCAM VERIFICATION (only if all registered) ─────────────
    if all_registered:
        active_plan_html = build_plan_rows_html(plan_items)
        with st.container(border=True):
            md_html(f"""
            <div class="cs-medcam-card">
              <div class="cs-medcam-header">
                <div class="cs-medcam-title">MedCam — Medication Check</div>
              </div>
              <p class="cs-medcam-desc">
                Take a photo of the pills in your hand. MedCam will compare them
                against your registered reference photos to verify they are correct.
              </p>
              <div class="cs-medcam-active-plan">
                <div class="cs-active-plan-label">Active plan</div>
                {active_plan_html}
              </div>
              <p class="cs-medcam-upload-label">Take or upload a photo of the pills to verify</p>
            </div>
            """)
            uploaded_image = st.file_uploader(
                "Take or upload a photo of the pills",
                type=["jpg", "jpeg", "png"],
                key="medcam_uploader",
                label_visibility="collapsed",
            )
            if uploaded_image:
                st.image(uploaded_image, caption="Pills to verify", width=280)
            check_clicked = st.button("Check medication", key="medcam_button", use_container_width=True)
        if not latest_plan:
            st.warning("No medication plan found. Upload one in Documents first.")
        if check_clicked:
            if uploaded_image is None:
                st.warning("Please upload a photo first.")
            else:
                with st.spinner("Comparing against your registered medication references..."):
                    base64_image = base64.b64encode(uploaded_image.read()).decode("utf-8")
                    ref_images_content = []
                    for r in med_refs:
                        ref_images_content.append({"type": "text", "text": f"Reference photo for {r['medication_name']}:"})
                        ref_images_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{r['image_b64']}"}})
                    system_prompt = """You are a medical assistant AI helping a caregiver verify medication safety.
You will receive reference photos of the patient's registered medications, followed by a photo of the pills the caregiver is about to give.
Compare the pill in the verification photo against the reference photos.
Focus on: shape, colour, size, markings, and texture.
Respond with ONLY a JSON object with:
1. "pills_detected": brief description of what you see in the verification photo
2. "matches_plan": true if the pill visually matches one of the references, false if not
3. "matched_medication": name of the medication it matches, or null
4. "caregiver_message": short clear message for the caregiver
5. "severity": "ok" if matches, "urgent" if not
"""
                    user_content = ref_images_content + [
                        {"type": "text", "text": f"Discharge plan:\n{discharge_plan}\n\nNow verify this photo:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ]
                    result = ask_ai(system_prompt, user_content)
                if result.get("error"):
                    st.error(result["message"])
                elif result.get("matches_plan"):
                    log_time = format_medcam_log_time()
                    matched = result.get("matched_medication", "medication")
                    save_shift_log(
                        caregiver_name=selected_caregiver,
                        source="medication_check",
                        summary=f"Pills detected: {result['pills_detected']}. Matched: {matched}. Verified at {log_time}.",
                        severity=result.get("severity", "ok"),
                    )
                    md_html(f"""
                    <div class="cs-medcam-result cs-medcam-result-success">
                      <strong>✓ Verified — {html.escape(str(matched))}</strong>
                      The pill in your hand matches the registered reference for {html.escape(str(matched))}.
                      Logged at {html.escape(log_time)}.
                    </div>
                    """)
                else:
                    save_shift_log(
                        caregiver_name=selected_caregiver,
                        source="medication_check",
                        summary=f"Pills detected: {result.get('pills_detected', 'unknown')}. Match failed.",
                        severity=result.get("severity", "urgent"),
                    )
                    md_html("""
                    <div class="cs-medcam-result cs-medcam-result-fail">
                      <strong>⚠ Warning</strong>
                      The pill in your hand does not match any registered reference.
                      Please double-check before giving this medication.
                    </div>
                    """)
    md_html(build_stored_conditions_medcam_html(get_stored_conditions()))'''

if old in content:
    content = content.replace(old, new)
    open("app.py", "w").write(content)
    print("Done")
else:
    print("NOT FOUND")
