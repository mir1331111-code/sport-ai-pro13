import unittest
from unittest.mock import patch

from data import sources


class _Response:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class HighlightlyOddsTests(unittest.TestCase):
    def test_parses_documented_football_markets_and_best_prices(self):
        entries = [{
            "matchId": 123,
            "odds": [
                {
                    "type": "prematch",
                    "market": "Full Time Result",
                    "values": [
                        {"value": "Home", "odd": 1.82},
                        {"value": "Draw", "odd": 3.60},
                        {"value": "Away", "odd": 4.70},
                    ],
                },
                {
                    "type": "prematch",
                    "market": "Full Time Result",
                    "values": [{"value": "Home", "odd": 1.91}],
                },
                {
                    "type": "prematch",
                    "market": "Total Goals 2.5",
                    "values": [
                        {"value": "Over", "odd": 1.88},
                        {"value": "Under", "odd": 1.96},
                    ],
                },
                {
                    "type": "prematch",
                    "market": "Both Teams to Score",
                    "values": [
                        {"value": "Yes", "odd": 1.74},
                        {"value": "No", "odd": 2.08},
                    ],
                },
                {
                    "type": "prematch",
                    "market": "Total Corners 9.5",
                    "values": [
                        {"value": "Over", "odd": 1.90},
                        {"value": "Under", "odd": 1.92},
                    ],
                },
                {
                    "type": "prematch",
                    "market": "Total Cards 4.5",
                    "values": [
                        {"value": "Over", "odd": 1.85},
                        {"value": "Under", "odd": 1.95},
                    ],
                },
                {
                    "type": "prematch",
                    "market": "Asian Handicap -1.25/+1.25",
                    "values": [
                        {"value": "Home", "odd": 1.95},
                        {"value": "Away", "odd": 1.87},
                    ],
                },
                {
                    "type": "live",
                    "market": "Full Time Result",
                    "values": [{"value": "Home", "odd": 9.99}],
                },
            ],
        }]

        result = sources._parse_highlightly_odds(entries, {"123"})

        self.assertEqual(result["П1"], 1.91)
        self.assertEqual(result["X"], 3.60)
        self.assertEqual(result["П2"], 4.70)
        self.assertEqual(result["total_market_odds"]["ТБ 2.5"], 1.88)
        self.assertEqual(result["total_market_odds"]["ТМ 2.5"], 1.96)
        self.assertEqual(result["btts_market_odds"]["BTTS Да"], 1.74)
        self.assertEqual(result["btts_market_odds"]["BTTS Нет"], 2.08)
        self.assertEqual(result["stat_market_odds"]["Угл ТБ 9.5"], 1.90)
        self.assertEqual(result["stat_market_odds"]["Карты ТМ 4.5"], 1.95)
        self.assertEqual(result["asian_handicap_market_odds"]["Ф1 -1.25"], 1.95)
        self.assertEqual(result["asian_handicap_market_odds"]["Ф2 +1.25"], 1.87)

    def test_total_market_line_parser_handles_total_goals_cards_and_corners(self):
        self.assertEqual(sources._market_number("Total Goals 2.5"), 2.5)
        self.assertEqual(sources._market_number("Total Cards 4"), 4.0)
        self.assertEqual(sources._market_number("Total Corners 9.5"), 9.5)

    @patch.object(
        sources._sess,
        "get",
        side_effect=[
            _Response(403, text="Football route not enabled"),
            _Response(200, {"data": [], "pagination": {"totalCount": 0}}),
        ],
    )
    def test_falls_back_from_football_host_to_all_sports_host(self, get_mock):
        response, base = sources._highlightly_api_get(
            "matches", {"date": "2026-10-30"}, "test-secret"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(base, "https://sports.highlightly.net/football")
        self.assertIn("https://soccer.highlightly.net/matches", get_mock.call_args_list[0].args[0])
        self.assertIn(
            "https://sports.highlightly.net/football/matches",
            get_mock.call_args_list[1].args[0],
        )
        self.assertEqual(
            get_mock.call_args_list[0].kwargs["headers"]["x-rapidapi-key"],
            "test-secret",
        )


if __name__ == "__main__":
    unittest.main()
