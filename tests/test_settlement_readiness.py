import unittest
from datetime import datetime, timedelta

from betting.settlement import settlement_not_before, settlement_ready


class SettlementReadinessTests(unittest.TestCase):
    def test_does_not_query_before_actual_kickoff_plus_full_time_grace(self):
        match_date = "2026-11-03"
        not_before = settlement_not_before(match_date, "20:45")
        self.assertEqual(not_before, datetime(2026, 11, 3, 23, 0))
        self.assertFalse(settlement_ready(
            match_date, "20:45", now=datetime(2026, 11, 3, 17, 11)
        ))
        self.assertTrue(settlement_ready(
            match_date, "20:45", now=datetime(2026, 11, 3, 23, 0)
        ))

    def test_missing_kickoff_time_is_deferred_until_next_day(self):
        not_before = settlement_not_before("2026-11-03", None)
        self.assertEqual(not_before, datetime(2026, 11, 4, 2, 15))
        self.assertFalse(settlement_ready(
            "2026-11-03", None, now=datetime(2026, 11, 3, 23, 59)
        ))
        self.assertTrue(settlement_ready(
            "2026-11-03", None, now=datetime(2026, 11, 4, 2, 15)
        ))

    def test_iso_datetime_and_date_formats_are_supported(self):
        self.assertEqual(
            settlement_not_before("2026-11-03T18:30:00", None),
            datetime(2026, 11, 3, 20, 45),
        )
        self.assertEqual(
            settlement_not_before("03.11.2026", "18:30"),
            datetime(2026, 11, 3, 20, 45),
        )
        self.assertIsNone(settlement_not_before("not-a-date", "18:30"))

    def test_custom_settlement_grace_is_applied(self):
        self.assertEqual(
            settlement_not_before(
                "2026-11-03", "20:45", grace=timedelta(hours=3)
            ),
            datetime(2026, 11, 3, 23, 45),
        )


if __name__ == "__main__":
    unittest.main()
