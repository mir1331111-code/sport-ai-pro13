import unittest
from unittest.mock import patch

from data.sources import highlightly_odds_fixture


class HighlightlyMarketTests(unittest.TestCase):
    @patch("data.sources.cache_get", return_value=[{
        "match": {"homeTeam": {"name": "Team A"}, "awayTeam": {"name": "Team B"}},
        "bookmaker": {"name": "TestBook"},
        "markets": [
            {"name": "Full Time Result", "odds": [
                {"label": "Home", "value": 2.1}, {"label": "Draw", "value": 3.2},
                {"label": "Away", "value": 3.8}]},
            {"name": "Total Goals", "odds": [
                {"label": "Over", "point": 2.5, "value": 1.9},
                {"label": "Under", "point": 2.5, "value": 1.95}]},
            {"name": "Total Corners", "odds": [
                {"label": "Over", "point": 9.5, "value": 1.88},
                {"label": "Under", "point": 9.5, "value": 1.92}]},
            {"name": "Total Cards", "odds": [
                {"label": "Over", "point": 4.5, "value": 1.85},
                {"label": "Under", "point": 4.5, "value": 1.98}]},
            {"name": "Both Teams To Score", "odds": [
                {"label": "Yes", "value": 1.7}, {"label": "No", "value": 2.1}]},
        ],
    }])
    def test_extracts_1x2_goals_corners_cards_and_btts(self, _cache):
        result = highlightly_odds_fixture("2026-10-27", "Team A", "Team B", "test-key")
        self.assertEqual(result["П1"], 2.1)
        self.assertEqual(result["X"], 3.2)
        self.assertEqual(result["П2"], 3.8)
        self.assertEqual(result["total_market_odds"]["ТБ 2.5"], 1.9)
        self.assertEqual(result["total_market_odds"]["ТМ 2.5"], 1.95)
        self.assertEqual(result["stat_market_odds"]["Угл ТБ 9.5"], 1.88)
        self.assertEqual(result["stat_market_odds"]["Карты ТБ 4.5"], 1.85)
        self.assertEqual(result["btts_market_odds"]["BTTS Да"], 1.7)
        self.assertTrue(result["source"].startswith("highlightly"))

    @patch("data.sources.cache_get", return_value=[{
        "match": {"homeTeam": {"name": "Other"}, "awayTeam": {"name": "Team B"}},
        "bookmaker": {}, "markets": [],
    }])
    def test_rejects_different_fixture(self, _cache):
        self.assertIsNone(highlightly_odds_fixture(
            "2026-10-27", "Team A", "Team B", "test-key"
        ))


if __name__ == "__main__":
    unittest.main()
