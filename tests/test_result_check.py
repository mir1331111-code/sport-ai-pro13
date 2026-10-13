"""Tests for conservative official-score settlement suggestions."""
import unittest

from betting.result_check import suggest_settlement


class SuggestSettlementTests(unittest.TestCase):
    def test_ft_score_suggests_win(self):
        got = suggest_settlement("1X2", "П1", {"status": "FT", "home": 2, "away": 0})
        self.assertEqual(got["outcome"], "won")

    def test_ft_score_suggests_loss(self):
        got = suggest_settlement("1X2", "П1", {"status": "FT", "home": 0, "away": 1})
        self.assertEqual(got["outcome"], "lost")

    def test_total_push_is_preserved(self):
        got = suggest_settlement("OU", "ТБ 2.0", {"status": "FT", "home": 1, "away": 1})
        self.assertEqual(got["outcome"], "push")

    def test_aet_is_not_auto_suggested(self):
        self.assertIsNone(suggest_settlement("1X2", "П1", {"status": "AET", "home": 2, "away": 1}))

    def test_invalid_score_is_rejected(self):
        self.assertIsNone(suggest_settlement("1X2", "П1", {"status": "FT", "home": -1, "away": 0}))

    def test_unknown_market_is_not_guessed(self):
        self.assertIsNone(suggest_settlement("mystery", "?", {"status": "FT", "home": 1, "away": 0}))


if __name__ == "__main__":
    unittest.main()
