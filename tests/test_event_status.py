"""Tests for cautious pending-event status hints."""
import unittest
from datetime import datetime

from model.event_status import pending_event_notice


class PendingEventNoticeTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 13, 18, 30)

    def test_past_date_pending_is_flagged_not_settled(self):
        notice = pending_event_notice("pending", "2026-10-12", "20:00", self.now)
        self.assertIn("результат не подтверждён", notice)

    def test_today_past_kickoff_is_flagged(self):
        notice = pending_event_notice("pending", "2026-10-13", "17:00", self.now)
        self.assertIn("ожидает расчёта", notice)

    def test_today_future_kickoff_is_not_flagged(self):
        self.assertIsNone(pending_event_notice("pending", "2026-10-13", "20:00", self.now))

    def test_today_without_time_is_not_assumed_overdue(self):
        self.assertIsNone(pending_event_notice("pending", "2026-10-13", "", self.now))

    def test_settled_bet_has_no_notice(self):
        self.assertIsNone(pending_event_notice("won", "2026-10-12", "20:00", self.now))

    def test_invalid_date_is_ignored(self):
        self.assertIsNone(pending_event_notice("pending", "unknown", "20:00", self.now))


if __name__ == "__main__":
    unittest.main()
