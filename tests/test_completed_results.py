import unittest
from unittest.mock import patch

from data.scores import completed_matches_for_day, _source_local_day


class CompletedMatchArchiveTests(unittest.TestCase):
    def test_source_utc_date_converts_to_local_day(self):
        with patch.dict("os.environ", {"APP_TIMEZONE": "Europe/Warsaw"}):
            self.assertEqual(
                _source_local_day("2026-11-06T23:30:00Z"), "2026-11-07"
            )

    def test_merges_provider_duplicates_and_calculates_daily_summary(self):
        with (
            patch("data.scores._espn_day", return_value=[
                {"home": "Bayern Munich", "away": "Borussia Dortmund",
                 "hs": 2, "as": 1, "status": "FT", "date": "2026-11-06",
                 "league": "Bundesliga"}
            ]),
            patch("data.scores._fdorg_range", return_value=[
                {"home": "Bayern", "away": "Borussia Dortmund",
                 "hs": 2, "as": 1, "status": "FT", "date": "2026-11-06",
                 "league": "Bundesliga"}
            ]),
            patch("data.scores._olb_season", return_value=2026),
            patch("data.scores._olb_league", return_value=[]),
        ):
            result = completed_matches_for_day(
                "2026-11-06", token="test-token"
            )

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["goals"], 3)
        self.assertEqual(result["home_wins"], 1)
        self.assertEqual(result["draws"], 0)
        self.assertEqual(result["away_wins"], 0)
        self.assertEqual(result["matches"][0]["sources"], ["ESPN", "football-data.org"])

    def test_keeps_conflicting_scores_visible_instead_of_silently_overwriting(self):
        with (
            patch("data.scores._espn_day", return_value=[
                {"home": "Bayern Munich", "away": "Borussia Dortmund",
                 "hs": 2, "as": 1, "status": "FT", "date": "2026-11-06"}
            ]),
            patch("data.scores._fdorg_range", return_value=[
                {"home": "Bayern", "away": "Borussia Dortmund",
                 "hs": 1, "as": 1, "status": "FT", "date": "2026-11-06"}
            ]),
            patch("data.scores._olb_season", return_value=2026),
            patch("data.scores._olb_league", return_value=[]),
        ):
            result = completed_matches_for_day("2026-11-06", token="test-token")

        self.assertEqual(result["total"], 1)
        self.assertEqual(result["score_conflicts"], 1)
        self.assertEqual(result["matches"][0]["score"], "2:1")
        self.assertEqual(
            result["matches"][0]["alternate_scores"],
            [{"score": "1:1", "source": "football-data.org"}],
        )

    def test_filters_out_results_from_other_dates(self):
        with (
            patch("data.scores._espn_day", return_value=[
                {"home": "Home FC", "away": "Away FC", "hs": 1, "as": 0,
                 "status": "FT", "date": "2026-11-05"}
            ]),
            patch("data.scores._fdorg_range", return_value=[]),
            patch("data.scores._olb_season", return_value=2026),
            patch("data.scores._olb_league", return_value=[]),
        ):
            result = completed_matches_for_day("2026-11-06", token="test-token")
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["goals"], 0)

    def test_legacy_espn_cache_rows_without_date_are_recovered(self):
        with (
            patch("data.scores._espn_day", return_value=[
                {"home": "Legacy Home", "away": "Legacy Away",
                 "hs": 2, "as": 0, "status": "FT"}
            ]),
            patch("data.scores._fdorg_range", return_value=[]),
            patch("data.scores._olb_season", return_value=2026),
            patch("data.scores._olb_league", return_value=[]),
        ):
            result = completed_matches_for_day("2026-11-06")
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["matches"][0]["score"], "2:0")
        self.assertEqual(result["source_counts"]["ESPN"], 1)

    def test_provider_utc_timestamp_is_grouped_by_local_calendar_day(self):
        with (
            patch.dict("os.environ", {"APP_TIMEZONE": "Europe/Warsaw"}),
            patch("data.scores._espn_day", return_value=[]),
            patch("data.scores._fdorg_range", return_value=[
                {"home": "Late Home", "away": "Late Away",
                 "hs": 0, "as": 1, "status": "FT",
                 "date": "2026-11-06T23:30:00Z"}
            ]),
            patch("data.scores._olb_season", return_value=2026),
            patch("data.scores._olb_league", return_value=[]),
        ):
            result = completed_matches_for_day("2026-11-07", token="test-token")
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["matches"][0]["date"], "2026-11-07")

    def test_espn_adjacent_day_rows_are_included_by_local_event_date(self):
        previous_day = [
            {"home": "Midnight Home", "away": "Midnight Away",
             "hs": 1, "as": 0, "status": "FT", "date": "2026-11-07T00:30:00+02:00"}
        ]
        with (
            patch.dict("os.environ", {"APP_TIMEZONE": "Europe/Warsaw"}),
            patch("data.scores._espn_day", side_effect=lambda d: previous_day if d == "2026-11-06" else []),
            patch("data.scores._fdorg_range", return_value=[]),
            patch("data.scores._olb_season", return_value=2026),
            patch("data.scores._olb_league", return_value=[]),
        ):
            result = completed_matches_for_day("2026-11-06")
        self.assertEqual(result["total"], 0)

    def test_rejects_invalid_calendar_dates(self):
        with self.assertRaises(ValueError):
            completed_matches_for_day("2026-02-30")

    def test_missing_fdorg_token_does_not_block_other_sources(self):
        with (
            patch("data.scores._get_token", return_value=""),
            patch("data.scores._espn_day", return_value=[
                {"home": "Home FC", "away": "Away FC", "hs": 1, "as": 0,
                 "status": "FT", "date": "2026-11-06"}
            ]),
            patch("data.scores._fdorg_range") as fdorg,
            patch("data.scores._olb_season", return_value=2026),
            patch("data.scores._olb_league", return_value=[]),
        ):
            result = completed_matches_for_day("2026-11-06")
        fdorg.assert_not_called()
        self.assertEqual(result["total"], 1)
        self.assertEqual(
            result["source_errors"]["football-data.org"],
            "API token not configured",
        )


if __name__ == "__main__":
    unittest.main()
