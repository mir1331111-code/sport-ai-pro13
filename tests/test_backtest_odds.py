import unittest

from ui.tabs.backtest import _pick_historical_odds


class HistoricalOddsTests(unittest.TestCase):
    def test_prefers_complete_pinnacle_triplet(self):
        row = {
            "PSH": "2.1", "PSD": "3.4", "PSA": "3.8",
            "B365H": "2.2", "B365D": "3.5", "B365A": "3.9",
        }
        self.assertEqual(
            _pick_historical_odds(row),
            {"П1": 2.1, "X": 3.4, "П2": 3.8},
        )

    def test_falls_back_to_complete_bet365_triplet(self):
        row = {
            "PSH": "2.1", "PSD": "", "PSA": "3.8",
            "B365H": "2.2", "B365D": "3.5", "B365A": "3.9",
        }
        self.assertEqual(
            _pick_historical_odds(row),
            {"П1": 2.2, "X": 3.5, "П2": 3.9},
        )

    def test_does_not_mix_incomplete_bookmakers(self):
        row = {
            "PSH": "2.1", "PSD": "", "PSA": "3.8",
            "B365H": "", "B365D": "3.5", "B365A": "3.9",
        }
        self.assertEqual(
            _pick_historical_odds(row),
            {"П1": None, "X": None, "П2": None},
        )

    def test_rejects_non_positive_odds(self):
        row = {
            "PSH": "0", "PSD": "3.4", "PSA": "3.8",
            "B365H": "2.2", "B365D": "-1", "B365A": "3.9",
        }
        self.assertEqual(
            _pick_historical_odds(row),
            {"П1": None, "X": None, "П2": None},
        )


if __name__ == "__main__":
    unittest.main()
