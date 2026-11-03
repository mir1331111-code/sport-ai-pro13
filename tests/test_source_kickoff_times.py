import unittest
from datetime import datetime, timezone

from data.sources import _tsdb_kickoff_utc


class SourceKickoffTimeTests(unittest.TestCase):
    def test_accepts_iso_timestamp_with_explicit_timezone(self):
        self.assertEqual(
            _tsdb_kickoff_utc("2026-11-03T20:45:00Z"),
            "2026-11-03T20:45:00Z",
        )
        self.assertEqual(
            _tsdb_kickoff_utc("2026-11-03T22:45:00+02:00"),
            "2026-11-03T20:45:00Z",
        )

    def test_accepts_unix_seconds_and_milliseconds(self):
        expected = datetime(2026, 11, 3, 20, 45, tzinfo=timezone.utc)
        seconds = expected.timestamp()
        self.assertEqual(
            _tsdb_kickoff_utc(str(int(seconds))),
            "2026-11-03T20:45:00Z",
        )
        self.assertEqual(
            _tsdb_kickoff_utc(str(int(seconds * 1000))),
            "2026-11-03T20:45:00Z",
        )

    def test_rejects_naive_or_malformed_timestamps_without_raising(self):
        self.assertEqual(_tsdb_kickoff_utc("2026-11-03 20:45:00"), "")
        self.assertEqual(_tsdb_kickoff_utc("not-a-timestamp"), "")
        self.assertEqual(_tsdb_kickoff_utc(None), "")


if __name__ == "__main__":
    unittest.main()
