import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)


def save_shift_log(caregiver_name, source, summary, severity):
    """
    Saves one event to the shift_logs table in Supabase.
    """
    supabase.table("shift_logs").insert({
        "caregiver_name": caregiver_name,
        "source": source,
        "summary": summary,
        "severity": severity
    }).execute()


def clean_json_response(raw_text):
    """
    GPT-4o sometimes wraps JSON in ```json ... ``` markdown blocks
    even when told not to. This strips that wrapper if present,
    then parses the result into a real Python dictionary.
    """
    text = raw_text.strip()

    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    return json.loads(text)


def ask_ai(system_prompt, user_content):
    """
    Sends a request to GPT-4o and returns a clean Python dictionary.
    user_content can be a string (text) or a list (text + image).
    If something fails, returns a safe fallback dictionary instead of crashing.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        )
        raw_text = response.choices[0].message.content
        return clean_json_response(raw_text)

    except Exception as e:
        return {
            "error": True,
            "message": "We couldn't process this right now. Please try again, or contact a healthcare provider if this is urgent.",
            "details": str(e)
        }
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from io import BytesIO


def generate_sbar_pdf(sbar_data):
    """
    Builds a PDF in memory (no file saved to disk) containing the SBAR report.
    Returns the raw PDF bytes, ready to be offered as a download in Streamlit.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("CareShield — SBAR Handover Report", styles["Title"]))
    story.append(Spacer(1, 16))

    sections = [
        ("Situation", sbar_data["situation"]),
        ("Background", sbar_data["background"]),
        ("Assessment", sbar_data["assessment"]),
        ("Recommendation", sbar_data["recommendation"]),
        ("Watch For", sbar_data["watch_for"]),
    ]

    for label, content in sections:
        story.append(Paragraph(f"<b>{label}</b>", styles["Heading2"]))
        story.append(Paragraph(content, styles["Normal"]))
        story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Reported By</b>", styles["Heading2"]))
    for entry in sbar_data["reported_by"]:
        story.append(Paragraph(f"<b>{entry['caregiver']}:</b> {entry['note']}", styles["Normal"]))
        story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
    from pypdf import PdfReader


def extract_text_from_pdf(uploaded_file):
    """
    Extracts all text from an uploaded PDF file (Streamlit file object).
    """
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def save_patient_plan(raw_text, medications):
    """
    Saves the extracted hospital document text and structured medication summary.
    """
    supabase.table("patient_plan").insert({
        "raw_text": raw_text,
        "medications": medications
    }).execute()


def get_latest_patient_plan():
    """
    Retrieves the most recently uploaded patient plan, if any.
    """
    response = supabase.table("patient_plan").select("*").order("created_at", desc=True).limit(1).execute()
    if response.data:
        return response.data[0]
    return None

from pypdf import PdfReader


def extract_text_from_pdf(uploaded_file):
    """
    Extracts all text from an uploaded PDF file (Streamlit file object).
    """
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def save_patient_plan(raw_text, medications):
    """
    Saves the extracted hospital document text and structured medication summary.
    """
    supabase.table("patient_plan").insert({
        "raw_text": raw_text,
        "medications": medications
    }).execute()


def get_latest_patient_plan():
    """
    Retrieves the most recently uploaded patient plan, if any.
    """
    response = supabase.table("patient_plan").select("*").order("created_at", desc=True).limit(1).execute()
    if response.data:
        return response.data[0]
    return None