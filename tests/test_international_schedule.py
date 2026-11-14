import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from data import sstats
from data.fixture_merge import merge_fixture_sources
from data.sources import _tsdb_league_meta


class _Response:
    status_code = 200

    def __init__(self, data):
        self._data = data

    def json(self):
        return {"status": "success", "data": self._data}


class InternationalScheduleTests(unittest.TestCase):
    def test_sstats_maps_national_and_womens_tournaments_separately(self):
        self.assertEqual(sstats._sstats_div_code("UEFA Nations League"), "NT_UNL")
        self.assertEqual(sstats._sstats_div_code("International Friendly"), "NT_FR")
        self.assertEqual(
            sstats._sstats_div_code("UEFA Women's Champions League"),
            "W_C1",
        )
        self.assertEqual(sstats._sstats_div_code("Women's Super League"), "W_E0")
        self.assertEqual(sstats._sstats_div_code("Men's U21 International Friendly"), "NT_U21_FR")

    def test_unmapped_sstats_competitions_do_not_fall_into_shared_g_bucket(self):
        first = sstats._sstats_div_code("Unknown Invitational A")
        second = sstats._sstats_div_code("Unknown Invitational B")
        self.assertNotEqual(first, "G")
        self.assertNotEqual(second, "G")
        self.assertNotEqual(first, second)

    def test_tsdb_classifies_senior_youth_and_womens_national_competitions(self):
        self.assertEqual(_tsdb_league_meta("UEFA Nations League")[0], "NT_UNL")
        self.assertEqual(_tsdb_league_meta("International Friendlies")[0], "NT_FR")
        self.assertEqual(_tsdb_league_meta("England U19 International Friendly")[0], "NT_U19")
        women = _tsdb_league_meta("UEFA Women's Champions League")
        self.assertEqual(women[0], "W_C1")
        self.assertTrue(women[2])
        women_u21 = _tsdb_league_meta("Women's U21 International Friendly")
        self.assertTrue(women_u21[2])
        self.assertTrue(women_u21[3])
        self.assertNotEqual(women_u21[0], "NT_U21")

    @patch.object(sstats, "_sstats_request")
    @patch.object(sstats, "_sstats_timezone")
    @patch.object(sstats, "cache_put")
    @patch.object(sstats, "cache_get", return_value=None)
    def test_offset_aware_evening_kickoff_is_not_lost_by_utc_local_mismatch(
        self, _cache_get, _cache_put, timezone_mock, request_mock
    ):
        local_zone = timezone(timedelta(hours=3))
        now = datetime(2026, 11, 14, 20, 0, tzinfo=local_zone)
        timezone_mock.return_value = (now, 3)
        request_mock.return_value = _Response([{
            "id": 991,
            "date": "2026-11-14T18:00:00+00:00",
            "homeTeam": {"name": "Norway"},
            "awayTeam": {"name": "Wales"},
            "season": {"league": {"name": "UEFA Nations League"}},
            "odds": [],
        }])
        rows = sstats.sstats_upcoming_matches(3, "test-key")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Date"], "2026-11-14 21:00:00")
        self.assertEqual(rows[0]["Time"], "21:00")
        self.assertEqual(rows[0]["Div"], "NT_UNL")
        self.assertEqual(rows[0]["kickoff_utc"], "2026-11-14T18:00:00Z")

    def test_sstats_remains_authoritative_when_supplemented_by_other_schedules(self):
        sstats_rows = [{
            "Date": "2026-11-14",
            "HomeTeam": "Norway",
            "AwayTeam": "Wales",
            "Div": "NT_UNL",
            "League": "UEFA Nations League",
            "fixture_id": "sstats:991",
            "sstats_game_id": 991,
            "market_odds": {"П1": 2.10, "X": 3.30, "П2": 3.60},
        }]
        tsdb_rows = [{
            "Date": "2026-11-14",
            "HomeTeam": "Norway",
            "AwayTeam": "Wales",
            "Div": "NT_UNL",
            "League": "UEFA Nations League",
            "fixture_id": "tsdb:991",
            "home_badge": "https://example.invalid/norway.png",
        }, {
            "Date": "2026-11-14",
            "HomeTeam": "Uganda",
            "AwayTeam": "Botswana",
            "Div": "NT_AFCON",
            "League": "AFCON Qualifiers",
            "fixture_id": "tsdb:992",
        }]
        merged = merge_fixture_sources(sstats_rows, tsdb_rows)
        self.assertEqual(len(merged), 2)
        norway = next(row for row in merged if row["HomeTeam"] == "Norway")
        self.assertEqual(norway["fixture_id"], "sstats:991")
        self.assertEqual(norway["sstats_game_id"], 991)
        self.assertEqual(norway["market_odds"]["П1"], 2.10)
        self.assertEqual(norway["home_badge"], "https://example.invalid/norway.png")
        uganda = next(row for row in merged if row["HomeTeam"] == "Uganda")
        self.assertNotIn("market_odds", uganda)


if __name__ == "__main__":
    unittest.main()
