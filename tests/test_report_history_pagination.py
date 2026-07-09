import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from handover_events import (
    REPORT_HISTORY_DEFAULT_VISIBLE,
    REPORT_HISTORY_EXPAND_BATCH,
    slice_messages_for_report_history,
)


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


class ReportHistoryPaginationTests(unittest.TestCase):
    def test_slice_messages_shows_latest_three_reports(self):
        messages = [
            {"role": "assistant", "content": "Welcome", "welcome": True},
            _user("Report 1"),
            _assistant("Reply 1"),
            _user("Report 2"),
            _assistant("Reply 2"),
            _user("Report 3"),
            _assistant("Reply 3"),
            _user("Report 4"),
            _assistant("Reply 4"),
            _user("Report 5"),
            _assistant("Reply 5"),
        ]
        visible, hidden, total = slice_messages_for_report_history(messages, 3)
        self.assertEqual(total, 5)
        self.assertEqual(hidden, 2)
        self.assertEqual(visible[0].get("welcome"), True)
        self.assertEqual(visible[1]["content"], "Report 3")
        self.assertEqual(visible[-1]["content"], "Reply 5")

    def test_slice_messages_reveals_all_when_count_high_enough(self):
        messages = [_user(f"Report {index}") for index in range(8)]
        visible, hidden, total = slice_messages_for_report_history(messages, 8)
        self.assertEqual(total, 8)
        self.assertEqual(hidden, 0)
        self.assertEqual(len(visible), 8)

    def test_report_history_defaults_to_three(self):
        self.assertEqual(REPORT_HISTORY_DEFAULT_VISIBLE, 3)
        self.assertEqual(REPORT_HISTORY_EXPAND_BATCH, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
