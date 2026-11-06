import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from data.fixture_time import fixture_kickoff_local


class FixtureKickoffTimeTests(unittest.TestCase):
    def test_utc_timestamp_is_converted_to_app_timezone(self):
        kickoff = fixture_kickoff_local(
            {"kickoff_utc": "2026-11-06T18:00:00Z"},
            timezone_name="Europe/Amsterdam",
        )
        self.assertEqual(
            kickoff,
            datetime(2026, 11, 6, 19, 0, tzinfo=ZoneInfo("Europe/Amsterdam")),
        )

    def test_same_day_started_fixture_is_detectable_by_full_timestamp(self):
        kickoff = fixture_kickoff_local(
            {"Date": "2026-11-06", "Time": "19:00",
             "kickoff_utc": "2026-11-06T18:00:00Z"},
            timezone_name="Europe/Amsterdam",
        )
        now = datetime(2026, 11, 6, 23, 20, tzinfo=ZoneInfo("Europe/Amsterdam"))
        self.assertIsNotNone(kickoff)
        self.assertLessEqual(kickoff, now)

    def test_local_date_and_clock_are_used_when_utc_timestamp_is_absent(self):
        kickoff = fixture_kickoff_local(
            {"Date": "2026-11-07", "Time": "00:30"},
            timezone_name="Europe/Amsterdam",
        )
        self.assertEqual(
            kickoff,
            datetime(2026, 11, 7, 0, 30, tzinfo=ZoneInfo("Europe/Amsterdam")),
        )

    def test_naive_kickoff_utc_is_treated_as_utc(self):
        kickoff = fixture_kickoff_local(
            {"kickoff_utc": "2026-11-06T22:00:00"},
            timezone_name="Europe/Amsterdam",
        )
        self.assertEqual(
            kickoff,
            datetime(2026, 11, 6, 23, 0, tzinfo=ZoneInfo("Europe/Amsterdam")),
        )

    def test_same_day_without_any_clock_time_is_not_guessed(self):
        self.assertIsNone(
            fixture_kickoff_local(
                {"Date": "2026-11-06"},
                timezone_name="Europe/Amsterdam",
            )
        )


if __name__ == "__main__":
    unittest.main()
