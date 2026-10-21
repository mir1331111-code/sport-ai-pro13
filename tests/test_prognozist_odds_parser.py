"""Regression tests for the free Prognozist/Marathon odds parser."""
import unittest

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

    def test_rejects_header_without_prices(self):
        self.assertIsNone(_prognozist_extract_row_odds(["П1", "X", "П2"]))


if __name__ == "__main__":
    unittest.main()
