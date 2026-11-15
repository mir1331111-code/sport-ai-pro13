import unittest
from unittest.mock import patch

from data import scores


class ExactScoreFixtureMatchingTests(unittest.TestCase):
    def test_espn_ignores_same_teams_on_adjacent_calendar_day(self):
        adjacent = [{
            "home": "Home FC", "away": "Away FC",
            "hs": 3, "as": 0, "status": "FT",
            "date": "2026-11-16", "fixture_id": "espn-next-day",
        }]
        with patch.object(
            scores, "_espn_day",
            side_effect=lambda day: adjacent if day == "2026-11-16" else [],
        ):
            result = scores._find_espn("Home", "Away", "2026-11-15")
        self.assertIsNone(result)

    def test_espn_accepts_exact_local_day_from_adjacent_utc_query(self):
        local_fixture = [{
            "home": "Home FC", "away": "Away FC",
            "hs": 1, "as": 1, "status": "FT",
            "date": "2026-11-15", "fixture_id": "espn-local-midnight",
        }]
        with patch.object(
            scores, "_espn_day",
            side_effect=lambda day: local_fixture if day == "2026-11-14" else [],
        ):
            result = scores._find_espn("Home", "Away", "2026-11-15")
        self.assertEqual(result["home"], 1)
        self.assertEqual(result["away"], 1)
        self.assertEqual(result["fixture_id"], "espn-local-midnight")

    def test_espn_rejects_multiple_distinct_same_day_fixtures(self):
        same_day = [
            {
                "home": "Home FC", "away": "Away FC",
                "hs": 1, "as": 0, "status": "FT",
                "date": "2026-11-15", "fixture_id": "espn-cup",
            },
            {
                "home": "Home", "away": "Away",
                "hs": 2, "as": 0, "status": "FT",
                "date": "2026-11-15", "fixture_id": "espn-league",
            },
        ]
        with patch.object(
            scores, "_espn_day",
            side_effect=lambda day: same_day if day == "2026-11-15" else [],
        ):
            result = scores._find_espn("Home FC", "Away FC", "2026-11-15")
        self.assertIsNone(result)

    def test_fdorg_requires_exact_local_event_date(self):
        rows = [{
            "home": "Home FC", "away": "Away FC",
            "hs": 2, "as": 1, "status": "FT",
            "date": "2026-11-16", "fixture_id": "fdorg-next-day",
        }]
        with patch.object(scores, "_fdorg_range", return_value=rows):
            result = scores._find_fdorg("Home", "Away", "2026-11-15", token="token")
        self.assertIsNone(result)

    def test_fdorg_rejects_ambiguous_same_day_fixture_ids(self):
        rows = [
            {
                "home": "Home FC", "away": "Away FC",
                "hs": 2, "as": 1, "status": "FT",
                "date": "2026-11-15", "fixture_id": "fdorg-cup",
            },
            {
                "home": "Home", "away": "Away",
                "hs": 1, "as": 0, "status": "FT",
                "date": "2026-11-15", "fixture_id": "fdorg-league",
            },
        ]
        with patch.object(scores, "_fdorg_range", return_value=rows):
            result = scores._find_fdorg("Home FC", "Away FC", "2026-11-15", token="token")
        self.assertIsNone(result)

    def test_openligadb_requires_exact_local_event_date(self):
        rows = [{
            "home": "Home FC", "away": "Away FC",
            "hs": 1, "as": 0, "status": "FT",
            "date": "2026-11-16", "fixture_id": "olb-next-day",
        }]
        with (
            patch.object(scores, "_olb_season", return_value=2026),
            patch.object(scores, "_olb_league", return_value=rows),
        ):
            result = scores._find_olb("Home", "Away", "2026-11-15")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
