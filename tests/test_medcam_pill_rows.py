import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from medcam_pill_rows import (
    build_medcam_reference_thumb_html,
    build_medcam_row_html_from_display,
)


def _fake_find_reference(med_name: str, med_refs: list):
    for ref in med_refs:
        if str(ref.get("medication_name") or "").lower() == med_name.lower():
            return ref
    return None


class MedcamPillRowTests(unittest.TestCase):
    def test_identified_row_includes_reference_thumbnail(self):
        thumb = build_medcam_reference_thumb_html(
            "Warfarin",
            [{"medication_name": "Warfarin", "image_b64": "abc123"}],
            find_reference=_fake_find_reference,
        )
        self.assertIn("cs-medcam-pill-row-ref-thumb", thumb)
        self.assertIn("abc123", thumb)
        row_html = build_medcam_row_html_from_display(
            {
                "label": "Identified — Warfarin",
                "verdict": "Give now",
                "role": "success",
                "detail": "Due now.",
            },
            thumb_html=thumb,
        )
        self.assertIn("cs-medcam-pill-row-summary-inner", row_html)
        self.assertIn("Warfarin reference photo", row_html)

    def test_not_identified_row_omits_thumbnail(self):
        thumb = build_medcam_reference_thumb_html(
            None,
            [{"medication_name": "Warfarin", "image_b64": "abc123"}],
            find_reference=_fake_find_reference,
        )
        self.assertEqual(thumb, "")
        row_html = build_medcam_row_html_from_display(
            {
                "label": "Not identified",
                "verdict": "Check manually",
                "role": "neutral",
                "detail": "Could not match.",
            },
        )
        self.assertNotIn("cs-medcam-pill-row-ref-thumb", row_html)
        self.assertNotIn("<img", row_html)
        self.assertIn('<summary class="cs-medcam-pill-row-summary">', row_html)
        self.assertIn("cs-medcam-pill-row-chevron", row_html)
        self.assertNotIn("/div>", row_html.replace("</div>", ""))
        self.assertEqual(row_html.count("<summary"), 1)
        self.assertEqual(row_html.count("</summary>"), 1)

    def test_missing_image_b64_does_not_render_broken_img(self):
        thumb = build_medcam_reference_thumb_html(
            "Hydralazine",
            [{"medication_name": "Hydralazine", "image_b64": ""}],
            find_reference=_fake_find_reference,
        )
        self.assertEqual(thumb, "")


if __name__ == "__main__":
    unittest.main()
