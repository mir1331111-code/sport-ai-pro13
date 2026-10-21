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

    def test_rejects_header_without_prices(self):
        self.assertIsNone(_prognozist_extract_row_odds(["П1", "X", "П2"]))


if __name__ == "__main__":
    unittest.main()
