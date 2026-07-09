import base64
import os
import sys
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ai_helpers import generate_handover_pdf
from handover_events import enrich_handover_result_with_period_entries


def _sample_event(index: int, text: str) -> dict:
    stamp = datetime(2026, 7, 9, 8 + index, 0, tzinfo=timezone.utc).isoformat()
    return {
        "timestamp": stamp,
        "timestamp_display": f"Thu 9 Jul, {8 + index:02d}:00 AM",
        "text": text,
        "severity": "monitor",
        "caregiver": "Carolina",
        "source": "voice_report",
    }


class HandoverPdfTests(unittest.TestCase):
    def test_pdf_generates_for_all_period_reports(self):
        short_events = [_sample_event(i, f"Report {i}") for i in range(8)]
        long_events = [_sample_event(i, f"Report {i}") for i in range(14)]
        short_pdf = generate_handover_pdf(short_events, [], None, period_label="Today")
        long_pdf = generate_handover_pdf(long_events, [], None, period_label="Today")
        self.assertGreater(len(long_pdf), len(short_pdf))
        self.assertNotIn(b"Showing latest 8 of", long_pdf)

    def test_pdf_adds_photo_review_pages(self):
        tiny_png = base64.b64encode(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
            b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
            b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        ).decode("ascii")
        events = [{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "timestamp_display": "Thu 9 Jul, 10:00 AM",
            "text": "Rash on left arm",
            "severity": "monitor",
            "caregiver": "María",
            "source": "symptom_photo",
            "image_b64": tiny_png,
            "photo_finding": "concern",
        }]
        without_photos = generate_handover_pdf(events, [], None, period_label="Today")
        with_photos = generate_handover_pdf(
            events,
            [],
            None,
            photo_reviews=events,
            period_label="Today",
        )
        self.assertGreater(len(with_photos), len(without_photos))
        self.assertIn(b"/Count 2", with_photos)


class HandoverReportEnrichmentTests(unittest.TestCase):
    def test_enrich_handover_result_lists_every_period_entry(self):
        symptom_events = [_sample_event(i, f"Symptom report {i}") for i in range(5)]
        adherence_events = [{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "text": "Warfarin dose missed",
            "caregiver": "María",
            "source": "medication_log",
        }]
        enriched = enrich_handover_result_with_period_entries(
            {"situation": "Test"},
            symptom_events,
            adherence_events,
        )
        self.assertEqual(enriched["period_report_count"], 6)
        self.assertEqual(len(enriched["reported_by"]), 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
