"""Tests for durable patient care storage helpers."""

import unittest

from ai_helpers import sanitize_chat_message_for_storage


class PatientCareStorageTests(unittest.TestCase):
    def test_sanitize_chat_message_strips_welcome_and_image_payload(self):
        self.assertIsNone(sanitize_chat_message_for_storage({
            "role": "assistant",
            "content": "Hi",
            "welcome": True,
        }))
        stored = sanitize_chat_message_for_storage({
            "role": "user",
            "content": "She was dizzy today",
            "timestamp": "2026-07-04T12:00:00+00:00",
            "has_image": True,
            "image_b64": "huge-payload",
        })
        self.assertEqual(stored["content"], "She was dizzy today")
        self.assertTrue(stored["has_image"])
        self.assertNotIn("image_b64", stored)

    def test_sanitize_keeps_assistant_severity(self):
        stored = sanitize_chat_message_for_storage({
            "role": "assistant",
            "content": "Watch her closely.",
            "severity": "monitor",
            "timestamp": "2026-07-04T12:01:00+00:00",
        })
        self.assertEqual(stored["severity"], "monitor")


if __name__ == "__main__":
    unittest.main()
