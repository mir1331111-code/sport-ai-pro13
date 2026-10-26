import unittest
from unittest.mock import patch

from data.sources import highlightly_odds_fixture


class FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class HighlightlyMarketTests(unittest.TestCase):
    def _request(self, url, **kwargs):
        if url.endswith("/matches"):
            return FakeResponse({"data": [{
                "id": 123,
                "homeTeam": {"name": "Team A"},
                "awayTeam": {"name": "Team B"},
            }]})
        return FakeResponse({"plan": {"tier": "PRO"}, "data": [{
            "matchId": 123,
            "odds": [
                {"market": "Full Time Result", "bookmakerName": "TestBook", "values": [
                    {"value": "Home", "odd": 2.1}, {"value": "Draw", "odd": 3.2},
                    {"value": "Away", "odd": 3.8}]},
                {"market": "Totals 2.5", "values": [
                    {"value": "Over", "odd": 1.9}, {"value": "Under", "odd": 1.95}]},
                {"market": "Total Corners 9.5", "values": [
                    {"value": "Over", "odd": 1.88}, {"value": "Under", "odd": 1.92}]},
                {"market": "Total Cards 4.5", "values": [
                    {"value": "Over", "odd": 1.85}, {"value": "Under", "odd": 1.98}]},
                {"market": "Both Teams To Score", "values": [
                    {"value": "Yes", "odd": 1.7}, {"value": "No", "odd": 2.1}]},
            ],
        }]})

    @patch("data.sources.cache_put")
    @patch("data.sources.cache_get", return_value=None)
    @patch("data.sources._sess.get")
    def test_extracts_documented_highlightly_schema(self, mock_get, _cache_get, _cache_put):
        mock_get.side_effect = self._request
        result = highlightly_odds_fixture("2026-10-27", "Team A", "Team B", "test-key")
        self.assertEqual(result["П1"], 2.1)
        self.assertEqual(result["X"], 3.2)
        self.assertEqual(result["П2"], 3.8)
        self.assertEqual(result["total_market_odds"]["ТБ 2.5"], 1.9)
        self.assertEqual(result["total_market_odds"]["ТМ 2.5"], 1.95)
        self.assertEqual(result["stat_market_odds"]["Угл ТБ 9.5"], 1.88)
        self.assertEqual(result["stat_market_odds"]["Карты ТБ 4.5"], 1.85)
        self.assertEqual(result["btts_market_odds"]["BTTS Да"], 1.7)
        self.assertEqual(result["source"], "highlightly")
        self.assertEqual(mock_get.call_count, 2)

    @patch("data.sources.cache_put")
    @patch("data.sources.cache_get", return_value=None)
    @patch("data.sources._sess.get")
    def test_rejects_different_fixture(self, mock_get, _cache_get, _cache_put):
        mock_get.return_value = FakeResponse({"data": [{
            "id": 123, "homeTeam": {"name": "Other"}, "awayTeam": {"name": "Team B"},
        }]})
        self.assertIsNone(highlightly_odds_fixture("2026-10-27", "Team A", "Team B", "test-key"))
        self.assertEqual(mock_get.call_count, 1)


if __name__ == "__main__":
    unittest.main()
