import unittest
from unittest.mock import Mock, patch

from data import sources


class _Response:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class SStatsTests(unittest.TestCase):
    def test_unwrap_api_response(self):
        self.assertEqual(
            sources._sstats_unwrap({"status": "success", "count": 1, "data": [{"id": 7}]}),
            [{"id": 7}],
        )
        with self.assertRaisesRegex(ValueError, "invalid api key"):
            sources._sstats_unwrap({"status": "error", "message": "invalid api key"})

    def test_parse_compact_and_bookmaker_market_structures(self):
        compact = [
            {
                "marketId": 1,
                "marketName": "Match Result",
                "odds": [
                    {"name": "1", "value": 1.82},
                    {"name": "X", "value": 3.55},
                    {"name": "2", "value": 4.70},
                ],
            },
            {
                "marketId": 2,
                "marketName": "Total Goals",
                "odds": [
                    {"name": "Over 2.5", "value": 1.91},
                    {"name": "Under 2.5", "value": 1.95},
                    {"name": "Over 3.5", "value": 2.35},
                    {"name": "Under 3.5", "value": 1.55},
                ],
            },
            {
                "marketId": 3,
                "marketName": "Both Teams to Score",
                "odds": [
                    {"name": "Yes", "value": 1.80},
                    {"name": "No", "value": 2.00},
                ],
            },
            {
                "marketId": 4,
                "marketName": "Total Corners",
                "odds": [
                    {"name": "Over 9.5", "value": 1.88},
                    {"name": "Under 9.5", "value": 1.92},
                ],
            },
            {
                "marketId": 5,
                "marketName": "Total Cards",
                "odds": [
                    {"name": "Over 4.5", "value": 1.85},
                    {"name": "Under 4.5", "value": 1.95},
                ],
            },
        ]
        detailed = [{
            "bookmakerId": 10,
            "bookmakerName": "Example bookmaker",
            "odds": compact,
        }]

        compact_result = sources._sstats_parse_markets(compact)
        detailed_result = sources._sstats_parse_markets(detailed)

        for result in (compact_result, detailed_result):
            self.assertEqual(result["П1"], 1.82)
            self.assertEqual(result["X"], 3.55)
            self.assertEqual(result["П2"], 4.70)
            self.assertEqual(result["total_market_odds"]["ТБ 2.5"], 1.91)
            self.assertEqual(result["total_market_odds"]["ТМ 2.5"], 1.95)
            self.assertEqual(result["total_market_odds"]["ТБ 3.5"], 2.35)
            self.assertEqual(result["btts_market_odds"]["BTTS Да"], 1.80)
            self.assertEqual(result["btts_market_odds"]["BTTS Нет"], 2.00)
            self.assertEqual(result["stat_market_odds"]["Угл ТБ 9.5"], 1.88)
            self.assertEqual(result["stat_market_odds"]["Карты ТМ 4.5"], 1.95)
            self.assertEqual(result["source"], "sstats")
        self.assertEqual(detailed_result["bookmaker_by_pick"]["П1"], "Example bookmaker")
        self.assertEqual(detailed_result["bookmaker_by_pick"]["ТБ 2.5"], "Example bookmaker")

    @patch.object(sources, "_sstats_request")
    @patch.object(sources, "_sstats_timezone")
    @patch.object(sources, "cache_put")
    @patch.object(sources, "cache_get", return_value=None)
    def test_upcoming_fixtures_normalized_from_official_game_shape(
        self, _cache_get, _cache_put, timezone_mock, request_mock
    ):
        from datetime import datetime, timedelta
        now = datetime(2026, 10, 30, 12, 0)
        timezone_mock.return_value = (now, 3)
        request_mock.return_value = _Response(200, {
            "status": "success",
            "count": 1,
            "data": [{
                "id": 5678,
                "date": (now + timedelta(days=1)).isoformat(),
                "status": 2,
                "homeTeam": {"id": 100, "name": "Arsenal"},
                "awayTeam": {"id": 200, "name": "Chelsea"},
                "season": {"league": {"name": "Premier League"}},
                "odds": [
                    {
                        "marketId": 1,
                        "marketName": "Match Result",
                        "odds": [
                            {"name": "1", "value": 1.80},
                            {"name": "X", "value": 3.60},
                            {"name": "2", "value": 4.80},
                        ],
                    },
                    {
                        "marketId": 2,
                        "marketName": "Total Goals",
                        "odds": [
                            {"name": "Over 2.5", "value": 1.92},
                            {"name": "Under 2.5", "value": 1.90},
                        ],
                    },
                ],
            }],
        })

        logs = []
        rows = sources.sstats_upcoming_matches(3, "do-not-log-this-key", logs)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["HomeTeam"], "Arsenal")
        self.assertEqual(rows[0]["AwayTeam"], "Chelsea")
        self.assertEqual(rows[0]["Div"], "E0")
        self.assertEqual(rows[0]["sstats_game_id"], 5678)
        self.assertEqual(rows[0]["market_odds"]["П1"], 1.80)
        self.assertEqual(rows[0]["total_market_odds"]["ТБ 2.5"], 1.92)
        params = request_mock.call_args.kwargs["params"]
        self.assertEqual(params["Upcoming"], "true")
        self.assertNotIn("do-not-log-this-key", "\\n".join(logs))
        self.assertEqual(request_mock.call_args.args[0], "Games/list")

    @patch.object(sources, "_sstats_request")
    @patch.object(sources, "cache_put")
    @patch.object(sources, "cache_get", return_value=None)
    def test_full_odds_uses_bookmaker_data_and_caches(self, _cache_get, _cache_put, request_mock):
        request_mock.return_value = _Response(200, {
            "status": "success",
            "count": 1,
            "data": [{
                "bookmakerId": 1,
                "bookmakerName": "Bookmaker A",
                "odds": [{
                    "marketId": 1,
                    "marketName": "Match Result",
                    "odds": [
                        {"name": "1", "value": 1.87},
                        {"name": "X", "value": 3.5},
                        {"name": "2", "value": 4.2},
                    ],
                }],
            }],
        })
        result = sources.sstats_odds_fixture(9876, "private-key", [])
        self.assertEqual(result["П1"], 1.87)
        self.assertEqual(result["bookmakers"], ["Bookmaker A"])
        self.assertEqual(request_mock.call_args.args[0], "Odds/9876")
        self.assertEqual(request_mock.call_args.kwargs["params"]["opening"], "false")

    @patch.object(sources, "_sstats_request")
    def test_account_check_confirms_key_without_exposing_it(self, request_mock):
        request_mock.side_effect = [
            _Response(200, {"status": "success", "data": {"userName": "test-user"}}),
            _Response(200, {"status": "success", "count": 0, "data": []}),
        ]
        result = sources.sstats_account_check("never-echo-this-key")
        self.assertTrue(result["ok"])
        self.assertTrue(result["games_ok"])
        self.assertEqual(result["games_count"], 0)
        self.assertNotIn("never-echo-this-key", repr(result))


if __name__ == "__main__":
    unittest.main()
