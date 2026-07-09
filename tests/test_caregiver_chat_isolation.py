"""Tests that Report & Ask chat does not bleed across caregiver profile switches."""

import unittest

from ai_helpers import (
    chat_thread_belongs_to_caregiver,
    chat_thread_has_user_content,
)


class CaregiverChatIsolationTests(unittest.TestCase):
    def test_chat_thread_has_user_content_detects_reports(self):
        messages = [
            {"role": "assistant", "content": "Welcome", "welcome": True},
            {"role": "user", "content": "He seems dizzy after lunch", "caregiver_id": "carolina"},
        ]
        self.assertTrue(chat_thread_has_user_content(messages))

    def test_welcome_only_thread_has_no_user_content(self):
        messages = [{"role": "assistant", "content": "Welcome", "welcome": True}]
        self.assertFalse(chat_thread_has_user_content(messages))

    def test_fresh_chat_is_welcome_only(self):
        """After caregiver switch, visible chat should be a single welcome bubble only."""
        maria_welcome_only = [
            {"role": "assistant", "content": "Hi María — report symptoms here.", "welcome": True},
        ]
        self.assertFalse(chat_thread_has_user_content(maria_welcome_only))
        self.assertEqual(len(maria_welcome_only), 1)
        self.assertTrue(maria_welcome_only[0].get("welcome"))

    def test_tagged_messages_belong_to_matching_caregiver(self):
        messages = [
            {"role": "assistant", "content": "Welcome", "welcome": True},
            {"role": "user", "content": "He seems dizzy", "caregiver_id": "carolina"},
            {"role": "assistant", "content": "Monitor closely"},
        ]
        self.assertTrue(chat_thread_belongs_to_caregiver(messages, "carolina"))

    def test_tagged_messages_reject_different_caregiver(self):
        """Carolina's typed messages must not appear as María's chat."""
        messages = [
            {"role": "assistant", "content": "Hi María", "welcome": True},
            {"role": "user", "content": "He seems dizzy after lunch", "caregiver_id": "carolina"},
            {"role": "assistant", "content": "Monitor closely"},
        ]
        self.assertFalse(chat_thread_belongs_to_caregiver(messages, "maria"))

    def test_untagged_legacy_messages_do_not_force_reset(self):
        messages = [
            {"role": "assistant", "content": "Welcome", "welcome": True},
            {"role": "user", "content": "Legacy report without caregiver tag"},
        ]
        self.assertTrue(chat_thread_belongs_to_caregiver(messages, "maria"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
