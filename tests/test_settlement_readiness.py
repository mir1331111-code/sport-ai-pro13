import unittest
from datetime import datetime, timedelta, timezone

from betting.settlement import settlement_not_before, settlement_ready


class SettlementReadinessTests(unittest.TestCase):
    def test_uses_authoritative_utc_kickoff_without_local_timezone_guess(self):
        not_before = settlement_not_before(
            "2026-11-03", "20:45",
            kickoff_at="2026-11-03T20:45:00Z",
        )
        self.assertEqual(
            not_before,
            datetime(2026, 11, 3, 23, 0, tzinfo=timezone.utc),
        )
        self.assertFalse(settlement_ready(
            "2026-11-03", "20:45",
            kickoff_at="2026-11-03T20:45:00Z",
            now=datetime(2026, 11, 3, 20, 59, tzinfo=timezone.utc),
        ))
        self.assertTrue(settlement_ready(
            "2026-11-03", "20:45",
            kickoff_at="2026-11-03T20:45:00Z",
            now=datetime(2026, 11, 3, 23, 0, tzinfo=timezone.utc),
        ))

    def test_naive_provider_clock_uses_timezone_safety_bound(self):
        not_before = settlement_not_before("2026-11-03", "20:45")
        self.assertEqual(
            not_before,
            datetime(2026, 11, 4, 11, 0, tzinfo=timezone.utc),
        )
        self.assertFalse(settlement_ready(
            "2026-11-03", "20:45", now=datetime(2026, 11, 4, 10, 59)
        ))
        self.assertTrue(settlement_ready(
            "2026-11-03", "20:45", now=datetime(2026, 11, 4, 11, 0)
        ))

    def test_missing_kickoff_time_uses_latest_possible_local_kickoff(self):
        not_before = settlement_not_before("2026-11-03", None)
        self.assertEqual(
            not_before,
            datetime(2026, 11, 4, 14, 14, tzinfo=timezone.utc),
        )
        self.assertFalse(settlement_ready(
            "2026-11-03", None, now=datetime(2026, 11, 4, 14, 13)
        ))
        self.assertTrue(settlement_ready(
            "2026-11-03", None, now=datetime(2026, 11, 4, 14, 14)
        ))

    def test_offset_timestamp_is_normalized_to_utc(self):
        not_before = settlement_not_before(
            "2026-11-03", kickoff_at="2026-11-03T20:30:00+02:00"
        )
        self.assertEqual(
            not_before,
            datetime(2026, 11, 3, 20, 45, tzinfo=timezone.utc),
        )

    def test_naive_iso_datetime_is_treated_as_provider_local_time(self):
        self.assertEqual(
            settlement_not_before("2026-11-03T18:30:00"),
            datetime(2026, 11, 4, 8, 45, tzinfo=timezone.utc),
        )

    def test_date_formats_and_invalid_dates_are_supported(self):
        self.assertEqual(
            settlement_not_before("03.11.2026", "18:30"),
            datetime(2026, 11, 4, 8, 45, tzinfo=timezone.utc),
        )
        self.assertIsNone(settlement_not_before("not-a-date", "18:30"))

    def test_custom_settlement_grace_is_applied(self):
        self.assertEqual(
            settlement_not_before(
                "2026-11-03", "20:45", grace=timedelta(hours=3)
            ),
            datetime(2026, 11, 4, 11, 45, tzinfo=timezone.utc),
        )


if __name__ == "__main__":
    unittest.main()
