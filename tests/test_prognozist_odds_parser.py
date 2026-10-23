"""Regression tests for the free Prognozist/Marathon odds parser."""
import unittest
import os
from unittest.mock import patch

from data.sources import _prognozist_extract_row_odds


class PrognozistOddsParserTests(unittest.TestCase):
    def test_extracts_explicitly_labelled_1x2_prices(self):
        row = ["Levante — Athletic Club", "П1 2,60 X 3,10 П2 2,90"]
        self.assertEqual(
            _prognozist_extract_row_odds(row),
            {"П1": 2.60, "X": 3.10, "П2": 2.90},
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

    def test_match_lookup_uses_line_only_when_permission_enabled(self):
        from data import sources
        odds = {"П1": 2.1, "X": 3.2, "П2": 3.5}
        with patch.dict(os.environ, {"PROGNOZIST_PERMISSION": ""}):
            with patch.object(sources, "cache_get", return_value=odds) as cached:
                self.assertIsNone(sources.prognozist_marathon_odds("Arsenal", "Chelsea"))
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
