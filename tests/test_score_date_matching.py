import unittest
from unittest.mock import patch

from data import scores


class ExactScoreFixtureMatchingTests(unittest.TestCase):
    def test_espn_marks_extra_time_and_penalty_final_status_ids_as_aet(self):
        from types import SimpleNamespace

        def response(status_type):
            return SimpleNamespace(
                status_code=200,
                json=lambda: {"events": [{
                    "id": "espn-cup-final",
                    "date": "2026-11-15T19:00:00Z",
                    "league": {"name": "Test Cup"},
                    "competitions": [{
                        "status": {"type": {
                            "state": "post",
                            "completed": True,
                            **status_type,
                        }},
                        "competitors": [
                            {"team": {"displayName": "Home FC"},
                             "homeAway": "home", "score": "2"},
                            {"team": {"displayName": "Away FC"},
                             "homeAway": "away", "score": "1"},
                        ],
                    }],
                }]},
            )

        for status_type in (
            {"id": "45", "name": "STATUS_FINAL", "detail": "Final"},
            {"id": 47, "name": "STATUS_FINAL", "shortDetail": "Final"},
            {"id": "3", "name": "STATUS_FINAL", "detail": "After Penalties"},
        ):
            with self.subTest(status_type=status_type), \
                 patch.object(scores, "cache_get", return_value=None), \
                 patch.object(scores, "cache_put"), \
                 patch.object(scores._sess, "get", return_value=response(status_type)):
                rows = scores._espn_day("2026-11-15")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "AET")

    def test_espn_keeps_regular_full_time_as_ft(self):
        from types import SimpleNamespace

        response = SimpleNamespace(
            status_code=200,
            json=lambda: {"events": [{
                "id": "espn-league-final",
                "date": "2026-11-15T19:00:00Z",
                "league": {"name": "Test League"},
                "competitions": [{
                    "status": {"type": {
                        "id": "3", "name": "STATUS_FINAL",
                        "state": "post", "completed": True, "detail": "Final",
                    }},
                    "competitors": [
                        {"team": {"displayName": "Home FC"},
                         "homeAway": "home", "score": "1"},
                        {"team": {"displayName": "Away FC"},
                         "homeAway": "away", "score": "0"},
                    ],
                }],
            }]},
        )
        with (
            patch.object(scores, "cache_get", return_value=None),
            patch.object(scores, "cache_put"),
            patch.object(scores._sess, "get", return_value=response),
        ):
            rows = scores._espn_day("2026-11-15")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "FT")

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
