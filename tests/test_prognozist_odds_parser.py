"""Regression tests for the free Prognozist/Marathon odds parser."""
import unittest
import os
from datetime import datetime
from unittest.mock import patch

from data.sources import _prognozist_extract_row_odds


class PrognozistOddsParserTests(unittest.TestCase):
    def test_extracts_explicitly_labelled_1x2_prices(self):
        row = ["Levante — Athletic Club", "П1 2,60 X 3,10 П2 2,90"]
        self.assertEqual(
            _prognozist_extract_row_odds(row),
            {"П1": 2.60, "X": 3.10, "П2": 2.90},
        )

    def test_parses_compact_match_centre_markup_shape(self):
        # Current public match-centre rows often concatenate labels and prices.
        row = ["21:00СкороМанчестер Сити U21Сток Сити U21П11.14X6.35П211.50—"]
        self.assertEqual(
            _prognozist_extract_row_odds(row),
            {"П1": 1.14, "X": 6.35, "П2": 11.50},
        )

    def test_extracts_prices_from_separate_table_cells(self):
        row = ["Levante", "Athletic Club", "2.60", "3.10", "2.90"]
        self.assertEqual(
            _prognozist_extract_row_odds(row),
            {"П1": 2.60, "X": 3.10, "П2": 2.90},
        )

    def test_american_odds_conversion(self):
        from data.sources import _american_to_decimal
        self.assertAlmostEqual(_american_to_decimal(-110), 1.9090909, places=5)
        self.assertAlmostEqual(_american_to_decimal(150), 2.5, places=5)
        self.assertIsNone(_american_to_decimal(0))

    def test_rundown_sport_mapping_is_limited_to_supported_leagues(self):
        from data.sources import THERUNDOWN_SPORT_BY_DIV
        self.assertEqual(THERUNDOWN_SPORT_BY_DIV["E0"], 11)
        self.assertEqual(THERUNDOWN_SPORT_BY_DIV["SP1"], 14)
        self.assertNotIn("R1", THERUNDOWN_SPORT_BY_DIV)

    def test_prognozist_source_is_disabled_without_explicit_permission(self):
        from data.sources import _prognozist_allowed, _sess
        with patch.dict(os.environ, {"PROGNOZIST_PERMISSION": ""}):
            with patch.object(_sess, "get") as mocked_get:
                self.assertFalse(_prognozist_allowed("https://prognozist.ru/mc/football/"))
                mocked_get.assert_not_called()

    def test_marathon_lookup_rejects_wrong_date(self):
        from data import marathon
        event = {"home": "Arsenal", "away": "Chelsea", "date_str": "23.10.2026 20:00",
                 "odds": {"П1": 2.1, "X": 3.2, "П2": 3.5}}
        with patch.object(marathon, "fetch_league_events", return_value=[event]):
            self.assertIsNone(marathon.marathon_odds(
                "Arsenal", "Chelsea", "2026-10-24", "E0"
            ))

    def test_marathon_lookup_selects_correct_date_for_repeated_fixture(self):
        from data import marathon
        old = {"home": "Arsenal", "away": "Chelsea", "date_str": "23.10.2026 20:00",
               "odds": {"П1": 2.1, "X": 3.2, "П2": 3.5}}
        current = {"home": "Arsenal", "away": "Chelsea", "date_str": "24.10.2026 20:00",
                   "odds": {"П1": 2.0, "X": 3.3, "П2": 3.7}}
        with patch.object(marathon, "fetch_league_events", return_value=[old, current]):
            result = marathon.marathon_odds("Arsenal", "Chelsea", "2026-10-24", "E0")
        self.assertEqual(result["П1"], 2.0)

    def test_marathon_lookup_rejects_ambiguous_same_date_lines(self):
        from data import marathon
        events = [
            {"home": "Arsenal", "away": "Chelsea", "date_str": "24.10.2026 18:00",
             "odds": {"П1": 2.0, "X": 3.3, "П2": 3.7}},
            {"home": "Arsenal", "away": "Chelsea", "date_str": "24.10.2026 20:00",
             "odds": {"П1": 2.2, "X": 3.1, "П2": 3.4}},
        ]
        with patch.object(marathon, "fetch_league_events", return_value=events):
            self.assertIsNone(marathon.marathon_odds(
                "Arsenal", "Chelsea", "2026-10-24", "E0"
            ))

    def test_marathon_fallback_is_disabled_without_permission(self):
        from data.marathon import _permitted_by_robots, _sess
        with patch.dict(os.environ, {"PROGNOZIST_PERMISSION": ""}):
            with patch.object(_sess, "get") as mocked_get:
                self.assertFalse(_permitted_by_robots(
                    "https://prognozist.ru/line/marathon/event_line.php?category=21520"
                ))
                mocked_get.assert_not_called()

    def test_marathon_cached_line_does_not_bypass_disabled_permission(self):
        from data import marathon
        cached_events = [{"home": "Arsenal", "away": "Chelsea",
                          "odds": {"П1": 2.1, "X": 3.2, "П2": 3.5}}]
        with patch.dict(os.environ, {"PROGNOZIST_PERMISSION": ""}):
            with (
                patch.object(marathon, "cache_get", return_value=cached_events) as cached,
                patch.object(marathon, "_permitted_by_robots", return_value=False) as permission,
                patch.object(marathon._sess, "get") as mocked_get,
            ):
                self.assertEqual(marathon.fetch_league_events("E0"), [])
                permission.assert_called_once()
                cached.assert_not_called()
                mocked_get.assert_not_called()

    def test_source_status_reports_safe_default(self):
        from data.sources import prognozist_source_status
        with patch.dict(os.environ, {"PROGNOZIST_PERMISSION": ""}):
            self.assertEqual(prognozist_source_status(), "disabled_permission")

    def test_match_lookup_uses_line_only_when_permission_enabled(self):
        from data import sources
        odds = {"П1": 2.1, "X": 3.2, "П2": 3.5}
        with patch.dict(os.environ, {"PROGNOZIST_PERMISSION": ""}):
            with patch.object(sources, "cache_get", return_value=odds) as cached:
                self.assertIsNone(sources.prognozist_marathon_odds("Arsenal", "Chelsea"))
                cached.assert_not_called()

    def test_match_lookup_honors_streamlit_secret_permission(self):
        from data import sources
        odds = {"П1": 2.1, "X": 3.2, "П2": 3.5}
        with patch.dict(os.environ, {"PROGNOZIST_PERMISSION": ""}):
            with (
                patch.object(sources, "prognozist_permission_enabled", return_value=True),
                patch.object(sources, "cache_get", return_value=None),
                patch.object(sources, "_prognozist_line_index", return_value=[
                    {"text": "Arsenal Chelsea П1 2.10 X 3.20 П2 3.50", "odds": odds}
                ]),
                patch.object(sources, "cache_put"),
            ):
                self.assertEqual(
                    sources.prognozist_marathon_odds("Arsenal", "Chelsea"),
                    odds,
                )

    def test_cached_prognozist_odds_do_not_bypass_robots_block(self):
        from data import sources
        with patch.dict(os.environ, {"PROGNOZIST_PERMISSION": "1"}):
            with (
                patch.object(sources, "_prognozist_allowed", return_value=False),
                patch.object(sources, "cache_get", return_value={"П1": 2.1, "X": 3.2, "П2": 3.5}) as cached,
            ):
                self.assertIsNone(sources.prognozist_marathon_odds("Arsenal", "Chelsea"))
                cached.assert_not_called()

    def test_cached_prognozist_line_index_does_not_bypass_robots_block(self):
        from data import sources
        with patch.dict(os.environ, {"PROGNOZIST_PERMISSION": "1"}):
            with (
                patch.object(sources, "_prognozist_allowed", return_value=False),
                patch.object(sources, "cache_get", return_value=[
                    {"text": "Arsenal Chelsea", "odds": {"П1": 2.1, "X": 3.2, "П2": 3.5}}
                ]) as cached,
            ):
                self.assertEqual(sources._prognozist_line_index(), [])
                cached.assert_not_called()

    def test_match_lookup_returns_unique_real_line(self):
        from data import sources
        odds = {"П1": 2.1, "X": 3.2, "П2": 3.5}
        with patch.dict(os.environ, {"PROGNOZIST_PERMISSION": "1"}):
            with (
                patch.object(sources, "cache_get", return_value=None),
                patch.object(sources, "_prognozist_line_index", return_value=[
                    {"text": "Arsenal Chelsea П1 2.10 X 3.20 П2 3.50", "odds": odds}
                ]),
                patch.object(sources, "cache_put"),
            ):
                self.assertEqual(
                    sources.prognozist_marathon_odds("Arsenal", "Chelsea"),
                    odds,
                )

    def test_match_lookup_rejects_line_for_different_calendar_date(self):
        from data import sources
        odds = {"П1": 2.1, "X": 3.2, "П2": 3.5}
        with patch.dict(os.environ, {"PROGNOZIST_PERMISSION": "1"}):
            with (
                patch.object(sources, "cache_get", return_value=None),
                patch.object(sources, "_prognozist_line_index", return_value=[
                    {"text": "23.10.2026 Arsenal Chelsea", "odds": odds}
                ]),
                patch.object(sources, "cache_put"),
            ):
                self.assertIsNone(
                    sources.prognozist_marathon_odds(
                        "Arsenal", "Chelsea", kickoff=datetime(2026, 10, 24)
                    )
                )

    def test_match_lookup_rejects_ambiguous_lines(self):
        from data import sources
        odds_a = {"П1": 2.1, "X": 3.2, "П2": 3.5}
        odds_b = {"П1": 2.2, "X": 3.1, "П2": 3.4}
        with patch.dict(os.environ, {"PROGNOZIST_PERMISSION": "1"}):
            with (
                patch.object(sources, "cache_get", return_value=None),
                patch.object(sources, "_prognozist_line_index", return_value=[
                    {"text": "Arsenal Chelsea", "odds": odds_a},
                    {"text": "Arsenal Chelsea", "odds": odds_b},
                ]),
                patch.object(sources, "cache_put"),
            ):
                self.assertIsNone(
                    sources.prognozist_marathon_odds("Arsenal", "Chelsea")
                )

    def test_rejects_header_without_prices(self):
        self.assertIsNone(_prognozist_extract_row_odds(["П1", "X", "П2"]))


if __name__ == "__main__":
    unittest.main()
