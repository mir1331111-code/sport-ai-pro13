import os
import unittest
from unittest.mock import Mock, patch

from data import sstats


class _Response:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class SStatsTests(unittest.TestCase):

    def test_sstats_timezone_uses_deployment_environment_override(self):
        with patch.dict(os.environ, {"APP_TIMEZONE": "Asia/Yekaterinburg"}):
            now, offset = sstats._sstats_timezone()
        self.assertEqual(getattr(now.tzinfo, "key", None), "Asia/Yekaterinburg")
        self.assertEqual(offset, 5)

    def test_division_mapping_handles_second_bundesliga_before_first_division(self):
        self.assertEqual(sstats._sstats_div_code("2. Bundesliga"), "D2")
        self.assertEqual(sstats._sstats_div_code("2 Bundesliga"), "D2")
        self.assertEqual(sstats._sstats_div_code("Bundesliga"), "D1")

    def test_ls_event_keys_preserve_string_id_and_match_adjacent_utc_local_dates(self):
        event = {
            "id": 12345,
            "eventId": "flashscore-string-id",
            "date": "2026-11-12T00:30:00+01:00",
            "teams": {
                "home": {"name": "Home FC"},
                "away": {"name": "Away FC"},
            },
        }
        keys, event_id = sstats._sstats_ls_event_keys(event, timezone_offset=0)
        self.assertEqual(event_id, "flashscore-string-id")
        self.assertIn(
            ("2026-11-11", sstats._norm_name("Home FC"), sstats._norm_name("Away FC")),
            keys,
        )
        self.assertIn(
            ("2026-11-12", sstats._norm_name("Home FC"), sstats._norm_name("Away FC")),
            keys,
        )

    def test_public_sstats_functions_remain_reexported_from_sources(self):
        from data import sources
        self.assertIs(sources.sstats_upcoming_matches, sstats.sstats_upcoming_matches)
        self.assertIs(
            sources.sstats_historical_matches_for_fixtures,
            sstats.sstats_historical_matches_for_fixtures,
        )
        self.assertIs(sources.sstats_match_statistics, sstats.sstats_match_statistics)
        self.assertIs(sources.sstats_match_result, sstats.sstats_match_result)

    def test_sstats_league_info_uses_agreeing_home_and_away_team_countries(self):
        name, league_id, country = sstats._sstats_league_info({
            "league": {"name": "Premier League"},
            "homeTeam": {
                "id": 10, "name": "Home FC",
                "country": {"name": "Mauritania"},
            },
            "awayTeam": {
                "id": 11, "name": "Away FC",
                "countryName": "Mauritania",
            },
        })
        self.assertEqual(name, "Premier League")
        self.assertEqual(league_id, "")
        self.assertEqual(country, "Mauritania")

    def test_sstats_league_info_does_not_guess_country_when_team_countries_conflict(self):
        _name, _league_id, country = sstats._sstats_league_info({
            "league": {"name": "Cup"},
            "homeTeam": {"name": "Home FC", "countryName": "Mauritania"},
            "awayTeam": {"name": "Away FC", "countryName": "Senegal"},
        })
        self.assertEqual(country, "")

    def test_sstats_league_info_handles_top_level_and_nested_league_objects(self):
        name, league_id, country = sstats._sstats_league_info({
            "league": {
                "LeagueID": 741,
                "Name": "Pro League",
                "Country": {"Name": "United Arab Emirates"},
            },
            "season": {"name": "2026/27"},
        })
        self.assertEqual(name, "Pro League")
        self.assertEqual(league_id, "741")
        self.assertEqual(country, "United Arab Emirates")

        name, league_id, country = sstats._sstats_league_info({
            "season": {
                "league": {"id": 742, "name": "Iraqi League"},
                "country": "Iraq",
            }
        })
        self.assertEqual((name, league_id, country), ("Iraqi League", "742", "Iraq"))

    def test_league_catalog_resolves_exact_name_and_country(self):
        fixtures = [{
            "League": "Pro League",
            "league_country": "UAE",
            "Div": "S_OLD_LABEL",
            "sstats_league_id": "",
        }]
        catalog = [
            {"id": "741", "name": "Pro League", "country": "UAE"},
            {"id": "742", "name": "Pro League", "country": "Iraq"},
        ]
        report = sstats._sstats_resolve_fixture_league_ids(fixtures, catalog)
        self.assertEqual(report["resolved"], 1)
        self.assertEqual(report["ambiguous"], 0)
        self.assertEqual(fixtures[0]["sstats_league_id"], "741")
        self.assertEqual(
            fixtures[0]["Div"],
            "S_" + sstats.hashlib.sha1(b"sstats-league:741").hexdigest()[:10].upper(),
        )

    def test_league_catalog_does_not_guess_when_same_name_has_multiple_countries(self):
        fixtures = [{
            "League": "Pro League",
            "Div": "S_OLD_LABEL",
            "sstats_league_id": "",
        }]
        catalog = [
            {"id": "741", "name": "Pro League", "country": "UAE"},
            {"id": "742", "name": "Pro League", "country": "Iraq"},
        ]
        report = sstats._sstats_resolve_fixture_league_ids(fixtures, catalog)
        self.assertEqual(report["resolved"], 0)
        self.assertEqual(report["ambiguous"], 1)
        self.assertEqual(fixtures[0]["sstats_league_id"], "")

    @patch.object(sstats, "_sstats_request")
    @patch.object(sstats, "cache_put")
    @patch.object(sstats, "cache_get", return_value=None)
    def test_league_catalog_parser_reads_sstats_leagues_endpoint(
        self, _cache_get, cache_put, request_mock
    ):
        request_mock.return_value = _Response(200, {
            "status": "success",
            "data": [
                {"Id": 741, "Name": "Pro League",
                 "Country": {"Name": "UAE"}},
                {"LeagueId": 742, "LeagueName": "Iraqi League",
                 "CountryName": "Iraq"},
            ],
        })
        catalog = sstats._sstats_league_catalog("private-key", logs=[])
        self.assertEqual(
            catalog,
            [
                {"id": "741", "name": "Pro League", "country": "UAE"},
                {"id": "742", "name": "Iraqi League", "country": "Iraq"},
            ],
        )
        self.assertEqual(request_mock.call_args.args[0], "Leagues")
        self.assertTrue(cache_put.called)

    def test_unwrap_api_response(self):
        self.assertEqual(
            sstats._sstats_unwrap({"status": "success", "count": 1, "data": [{"id": 7}]}),
            [{"id": 7}],
        )
        with self.assertRaisesRegex(ValueError, "invalid api key"):
            sstats._sstats_unwrap({"status": "error", "message": "invalid api key"})

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

        compact_result = sstats._sstats_parse_markets(compact)
        detailed_result = sstats._sstats_parse_markets(detailed)

        # Compact /Games/list has no bookmaker identity: preserve the raw
        # observations for diagnostics, but do not publish an unverified line.
        self.assertNotIn("П1", compact_result)
        self.assertNotIn("X", compact_result)
        self.assertNotIn("П2", compact_result)
        self.assertEqual(
            compact_result["one_x_two_diagnostics"]["status"],
            "unverified_bookmaker",
        )
        self.assertEqual(detailed_result["П1"], 1.82)
        self.assertEqual(detailed_result["X"], 3.55)
        self.assertEqual(detailed_result["П2"], 4.70)
        self.assertEqual(detailed_result["one_x_two_diagnostics"]["status"], "confirmed")

        for result in (compact_result, detailed_result):
            self.assertEqual(result["btts_market_odds"]["BTTS Да"], 1.80)
            self.assertEqual(result["btts_market_odds"]["BTTS Нет"], 2.00)
            self.assertEqual(result["stat_market_odds"]["Угл ТБ 9.5"], 1.88)
            self.assertEqual(result["stat_market_odds"]["Карты ТМ 4.5"], 1.95)
            self.assertEqual(result["source"], "sstats")
        # Anonymous compact totals stay hidden. A named one-book quote can be
        # shown as a raw price but is not corroborated by a second bookmaker.
        self.assertEqual(compact_result["total_market_odds"], {})
        self.assertEqual(detailed_result["total_market_odds"]["ТБ 2.5"], 1.91)
        self.assertEqual(detailed_result["total_market_book_count"]["ТБ 2.5"], 1)
        self.assertFalse(detailed_result["total_market_confirmed"]["ТБ 2.5"])
        self.assertEqual(detailed_result["bookmaker_by_pick"]["П1"], "Example bookmaker")

    @patch.object(sstats, "_sstats_request")
    @patch.object(sstats, "_sstats_timezone")
    @patch.object(sstats, "cache_put")
    @patch.object(sstats, "cache_get", return_value=None)
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
                "season": {"league": {"id": 991, "name": "Premier League"}},
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
        rows = sstats.sstats_upcoming_matches(3, "do-not-log-this-key", logs)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["HomeTeam"], "Arsenal")
        self.assertEqual(rows[0]["AwayTeam"], "Chelsea")
        self.assertEqual(rows[0]["Div"], "S_" + sstats.hashlib.sha1(
            b"sstats-league:991"
        ).hexdigest()[:10].upper())
        self.assertEqual(rows[0]["sstats_game_id"], 5678)
        self.assertEqual(rows[0]["sstats_league_id"], "991")
        self.assertEqual(rows[0]["market_odds"], {})
        # Games/list compact odds have no bookmaker provenance, so they stay
        # diagnostic-only until detailed Odds/{gameId} confirms a named book.
        self.assertEqual(
            rows[0]["one_x_two_diagnostics"]["status"],
            "unverified_bookmaker",
        )
        self.assertEqual(rows[0]["total_market_odds"], {})
        params = request_mock.call_args.kwargs["params"]
        self.assertEqual(params["Upcoming"], "true")
        self.assertNotIn("do-not-log-this-key", "\\n".join(logs))
        self.assertEqual(request_mock.call_args.args[0], "Games/list")

    @patch.object(sstats, "_sstats_request")
    @patch.object(sstats, "_sstats_timezone")
    @patch.object(sstats, "cache_put")
    @patch.object(sstats, "cache_get", return_value=None)
    @patch.object(sstats, "_sstats_league_catalog")
    def test_historical_loader_resolves_league_id_from_catalog_before_query(
        self, catalog_mock, _cache_get, _cache_put, timezone_mock, request_mock
    ):
        from datetime import datetime
        timezone_mock.return_value = (datetime(2026, 11, 19, 12, 0), 3)
        catalog_mock.return_value = [
            {"id": "741", "name": "Pro League", "country": "UAE"},
        ]
        request_mock.return_value = _Response(200, {
            "status": "success",
            "data": [{
                "id": 12, "date": "2026-06-01T18:00:00",
                "status": 8,
                "homeTeam": {"name": "Al Ain"},
                "awayTeam": {"name": "Al Dhafra"},
                "homeResult": 2, "awayResult": 0,
            }],
        })
        fixture = {
            "League": "Pro League",
            "league_country": "UAE",
            "Div": "S_OLD_LABEL",
            "sstats_league_id": "",
        }
        logs = []
        rows = sstats.sstats_historical_matches_for_fixtures(
            [fixture], "private-key", logs=logs,
        )
        self.assertEqual(fixture["sstats_league_id"], "741")
        self.assertEqual(
            fixture["Div"],
            "S_" + sstats.hashlib.sha1(b"sstats-league:741").hexdigest()[:10].upper(),
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Div"], fixture["Div"])
        self.assertEqual(request_mock.call_args.kwargs["params"]["LeagueId"], "741")
        self.assertIn("разрешено по точному каталогу 1", "\n".join(logs))

    @patch.object(sstats, "_sstats_request")
    @patch.object(sstats, "_sstats_timezone")
    @patch.object(sstats, "cache_put")
    @patch.object(sstats, "cache_get", return_value=None)
    def test_historical_league_results_are_loaded_as_strict_ft_rows(
        self, _cache_get, cache_put, timezone_mock, request_mock
    ):
        from datetime import datetime
        timezone_mock.return_value = (datetime(2026, 11, 19, 12, 0), 3)
        request_mock.return_value = _Response(200, {
            "status": "success",
            "data": [
                {
                    "id": 1, "date": "2026-05-01T18:00:00",
                    "status": 8,
                    "homeTeam": {"name": "Home FC"},
                    "awayTeam": {"name": "Away FC"},
                    "homeResult": 2, "awayResult": 1,
                },
                {
                    "id": 2, "date": "2026-05-08T18:00:00",
                    "status": 7,
                    "homeTeam": {"name": "Home FC"},
                    "awayTeam": {"name": "Away FC"},
                    "homeResult": 9, "awayResult": 0,
                },
                {
                    "id": 3, "date": "2026-05-15T18:00:00",
                    "status": 8,
                    "homeTeam": {"name": "Other FC"},
                    "awayTeam": {"name": "Away FC"},
                    "homeResult": None, "awayResult": 0,
                },
                {
                    "id": 4, "date": "2027-05-15T18:00:00",
                    "status": 8,
                    "homeTeam": {"name": "Future FC"},
                    "awayTeam": {"name": "Away FC"},
                    "homeResult": 1, "awayResult": 0,
                },
            ],
        })

        logs = []
        rows = sstats.sstats_historical_matches_for_fixtures(
            [
                {
                    "sstats_league_id": "991",
                    "Div": "S_TEST991",
                    "League": "Local Premier League",
                    "market_odds": {"П1": 2.1, "X": 3.0, "П2": 3.4},
                },
                {
                    "sstats_league_id": "991",
                    "Div": "S_TEST991",
                    "League": "Local Premier League",
                },
            ],
            "private-api-key",
            years_back=5,
            max_leagues=10,
            logs=logs,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Div"], "S_TEST991")
        self.assertEqual(rows[0]["FTHG"], 2)
        self.assertEqual(rows[0]["FTAG"], 1)
        self.assertEqual(rows[0]["source"], "sstats_history")
        self.assertEqual(request_mock.call_count, 1)
        self.assertEqual(request_mock.call_args.args[0], "Games/list")
        params = request_mock.call_args.kwargs["params"]
        self.assertEqual(params["LeagueId"], "991")
        self.assertEqual(params["Ended"], "true")
        self.assertEqual(params["To"], "2026-11-19")
        self.assertEqual(params["From"], "2021-11-20")
        self.assertTrue(cache_put.called)
        self.assertIn("1 FT-матчей", "\n".join(logs))

    @patch.object(sstats, "_sstats_request")
    @patch.object(sstats, "_sstats_timezone")
    @patch.object(sstats, "cache_put")
    @patch.object(sstats, "cache_get", return_value=None)
    def test_historical_loader_accepts_missing_status_only_when_ended_filter_has_valid_score(
        self, _cache_get, _cache_put, timezone_mock, request_mock
    ):
        from datetime import datetime
        timezone_mock.return_value = (datetime(2026, 11, 19, 12, 0), 3)
        request_mock.return_value = _Response(200, {
            "status": "success",
            "data": [
                {
                    "Date": "2026-05-01T18:00:00",
                    "HomeTeamName": "Home FC", "AwayTeamName": "Away FC",
                    "HomeResult": 2, "AwayResult": 1,
                },
                {
                    "DATE": "2026-05-08T18:00:00", "Status": "FT",
                    "HomeTeam": {"Name": "Home FC"},
                    "AwayTeam": {"Name": "Away FC"},
                    "HomeGoals": 1, "AwayGoals": 1,
                },
                {
                    "date": "2026-05-15T18:00:00", "Status": 7,
                    "homeTeamName": "Home FC", "awayTeamName": "Away FC",
                    "homeResult": 9, "awayResult": 0,
                },
                {
                    "date": "2026-05-22T18:00:00", "homeTeamName": "Home FC",
                    "awayTeamName": "Away FC", "homeResult": None, "awayResult": 0,
                },
            ],
        })
        logs = []
        rows = sstats.sstats_historical_matches_for_fixtures(
            [{
                "sstats_league_id": "991",
                "Div": "S_TEST991",
                "League": "Local Premier League",
            }],
            "private-key", logs=logs,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual([(row["FTHG"], row["FTAG"]) for row in rows], [(2, 1), (1, 1)])
        joined = "\n".join(logs)
        self.assertIn("без поля Status (Ended=true)=2", joined)
        self.assertIn("отклонено по Status=1", joined)
        self.assertIn("без валидного счёта=1", joined)

    def test_historical_loader_does_not_fetch_without_sstats_league_ids(self):
        with patch.object(sstats, "_sstats_league_catalog", return_value=[]):
            logs = []
            rows = sstats.sstats_historical_matches_for_fixtures(
                [{"Div": "S_NO_ID", "League": "Unknown"}],
                "private-key", logs=logs,
            )
        self.assertEqual(rows, [])
        self.assertIn("нет ни одного безопасно определённого LeagueId", "\n".join(logs))

    @patch.object(sstats, "_sstats_request")
    @patch.object(sstats, "_sstats_timezone", return_value=(None, 2))
    @patch.object(sstats, "cache_put")
    @patch.object(sstats, "cache_get", return_value=None)
    def test_legacy_match_id_resolution_requires_exact_unique_match(
        self, _cache_get, _cache_put, _timezone, request_mock
    ):
        request_mock.return_value = _Response(200, {
            "status": "success",
            "data": [{
                "id": 9876, "date": "2026-11-02T18:00:00",
                "homeTeam": {"name": "Arsenal"},
                "awayTeam": {"name": "Chelsea"},
            }],
        })
        result = sstats.sstats_find_match_id(
            "2026-11-02", "Arsenal", "Chelsea", "private-key"
        )
        self.assertEqual(result, 9876)
        self.assertEqual(request_mock.call_args.args[0], "Games/list")
        params = request_mock.call_args.kwargs["params"]
        self.assertEqual(params["Date"], "2026-11-02")
        self.assertTrue(params["Ended"])

    @patch.object(sstats, "_sstats_request")
    @patch.object(sstats, "_sstats_timezone", return_value=(None, 2))
    @patch.object(sstats, "cache_put")
    @patch.object(sstats, "cache_get", return_value=None)
    def test_legacy_match_id_resolution_rejects_ambiguous_matches(
        self, _cache_get, _cache_put, _timezone, request_mock
    ):
        game = {
            "date": "2026-11-02T18:00:00",
            "homeTeam": {"name": "Arsenal"},
            "awayTeam": {"name": "Chelsea"},
        }
        request_mock.return_value = _Response(200, {
            "status": "success",
            "data": [dict(game, id=1), dict(game, id=2)],
        })
        self.assertIsNone(sstats.sstats_find_match_id(
            "2026-11-02", "Arsenal", "Chelsea", "private-key"
        ))

    def test_sstats_completed_result_requires_exact_finished_fixture(self):
        payload = {
            "status": "success",
            "data": [{
                "id": 9876,
                "date": "2026-11-02T18:00:00",
                "homeTeam": {"name": "Arsenal"},
                "awayTeam": {"name": "Chelsea"},
                "status": 8,
                "homeResult": 2,
                "awayResult": 1,
            }],
        }
        with (
            patch.object(sstats, "cache_get", return_value=None),
            patch.object(sstats, "cache_put") as cache_write,
            patch.object(sstats, "_sstats_timezone", return_value=(None, 2)),
            patch.object(sstats, "_sstats_request", return_value=_Response(200, payload)) as request_mock,
        ):
            result = sstats.sstats_match_result(
                "2026-11-02", "Arsenal FC", "Chelsea", "private-key",
            )

        self.assertEqual(result["home"], 2)
        self.assertEqual(result["away"], 1)
        self.assertEqual(result["status"], "FT")
        self.assertTrue(result["regulation_time"])
        self.assertEqual(result["source"], "sstats")
        self.assertEqual(result["fixture_id"], "9876")
        self.assertEqual(request_mock.call_args.args[0], "Games/list")
        self.assertEqual(request_mock.call_args.kwargs["params"]["Date"], "2026-11-02")
        self.assertEqual(request_mock.call_args.kwargs["params"]["Ended"], "true")
        self.assertNotIn("private-key", repr(result))
        self.assertEqual(cache_write.call_count, 2)
        self.assertTrue(cache_write.call_args_list[0].args[0].startswith("sstats_ft_day_v1_"))
        self.assertTrue(cache_write.call_args_list[1].args[0].startswith("sstats_ft_result_v1_"))

    def test_completed_games_daily_payload_is_reused_across_fixture_lookups(self):
        payload = {
            "status": "success",
            "data": [
                {
                    "id": 9876,
                    "date": "2026-11-02T18:00:00",
                    "homeTeam": {"name": "Arsenal"},
                    "awayTeam": {"name": "Chelsea"},
                    "status": 8, "homeResult": 2, "awayResult": 1,
                },
                {
                    "id": 9877,
                    "date": "2026-11-02T20:00:00",
                    "homeTeam": {"name": "Liverpool"},
                    "awayTeam": {"name": "Manchester City"},
                    "status": 8, "homeResult": 1, "awayResult": 1,
                },
            ],
        }
        cache = {}

        def cache_get(key, _ttl):
            return cache.get(key)

        def cache_put(key, value):
            cache[key] = value

        with (
            patch.object(sstats, "cache_get", side_effect=cache_get),
            patch.object(sstats, "cache_put", side_effect=cache_put),
            patch.object(sstats, "_sstats_timezone", return_value=(None, 2)),
            patch.object(
                sstats, "_sstats_request", return_value=_Response(200, payload),
            ) as request_mock,
        ):
            first = sstats.sstats_match_result(
                "2026-11-02", "Arsenal", "Chelsea", "private-key",
            )
            second = sstats.sstats_match_result(
                "2026-11-02", "Liverpool", "Manchester City", "private-key",
            )

        self.assertEqual((first["home"], first["away"]), (2, 1))
        self.assertEqual((second["home"], second["away"]), (1, 1))
        self.assertEqual(request_mock.call_count, 1)
        self.assertTrue(any(key.startswith("sstats_ft_day_v1_") for key in cache))

    def test_sstats_completed_result_rejects_non_ft_and_ambiguous_fixture(self):
        base = {
            "id": 9876,
            "date": "2026-11-02T18:00:00",
            "homeTeam": {"name": "Arsenal"},
            "awayTeam": {"name": "Chelsea"},
            "status": 6,
            "homeResult": 2,
            "awayResult": 1,
        }
        with (
            patch.object(sstats, "cache_get", return_value=None),
            patch.object(sstats, "cache_put"),
            patch.object(sstats, "_sstats_timezone", return_value=(None, 2)),
            patch.object(
                sstats, "_sstats_request",
                return_value=_Response(200, {"status": "success", "data": [base]}),
            ),
        ):
            self.assertIsNone(sstats.sstats_match_result(
                "2026-11-02", "Arsenal", "Chelsea", "private-key",
            ))

        finished = dict(base, status=8)
        with (
            patch.object(sstats, "cache_get", return_value=None),
            patch.object(sstats, "cache_put"),
            patch.object(sstats, "_sstats_timezone", return_value=(None, 2)),
            patch.object(
                sstats, "_sstats_request",
                return_value=_Response(200, {
                    "status": "success", "data": [finished, dict(finished, id=9877)],
                }),
            ),
        ):
            self.assertIsNone(sstats.sstats_match_result(
                "2026-11-02", "Arsenal", "Chelsea", "private-key",
            ))

    @patch.object(sstats, "_sstats_request")
    @patch.object(sstats, "cache_put")
    @patch.object(sstats, "cache_get", return_value=None)
    def test_final_match_statistics_returns_corners_and_all_cards(
        self, _cache_get, cache_write, request_mock
    ):
        request_mock.return_value = _Response(200, {
            "status": "success",
            "data": {
                "game": {"id": 5678, "status": 8, "statusName": "Finished"},
                "statistics": {
                    "cornerKicksHome": 7, "cornerKicksAway": 4,
                    "yellowCardsHome": 2, "yellowCardsAway": 1,
                    "redCardsHome": 0, "redCardsAway": 1,
                },
            },
        })
        result = sstats.sstats_match_statistics(5678, "private-key")
        self.assertEqual(result["status"], 8)
        self.assertTrue(result["finished"])
        self.assertEqual(result["corners"], 11)
        self.assertEqual(result["cards"], 4)
        self.assertEqual(result["source"], "sstats")
        self.assertEqual(request_mock.call_args.args[0], "Games/5678")
        self.assertNotIn("private-key", repr(result))
        cache_write.assert_called_once()

    @patch.object(sstats, "_sstats_request")
    @patch.object(sstats, "cache_put")
    @patch.object(sstats, "cache_get", return_value=None)
    def test_live_or_extra_time_match_is_not_marked_final(
        self, _cache_get, cache_write, request_mock
    ):
        request_mock.return_value = _Response(200, {
            "status": "success",
            "data": {
                "game": {"id": 5678, "status": 6, "statusName": "Extra time"},
                "statistics": {"cornerKicksHome": 5, "cornerKicksAway": 4},
            },
        })
        result = sstats.sstats_match_statistics(5678, "private-key")
        self.assertFalse(result["finished"])
        self.assertIsNone(result["corners"])
        cache_write.assert_not_called()

    @patch.object(sstats, "_sstats_request")
    @patch.object(sstats, "cache_put")
    @patch.object(sstats, "cache_get", return_value=None)
    def test_missing_stat_fields_are_not_treated_as_zero(
        self, _cache_get, cache_write, request_mock
    ):
        request_mock.return_value = _Response(200, {
            "status": "success",
            "data": {
                "game": {"id": 5678, "status": 8},
                "statistics": {
                    "cornerKicksHome": 5, "cornerKicksAway": None,
                    "yellowCardsHome": 1, "yellowCardsAway": 0,
                    "redCardsHome": 0, "redCardsAway": None,
                },
            },
        })
        result = sstats.sstats_match_statistics(5678, "private-key")
        self.assertIsNone(result["corners"])
        self.assertIsNone(result["cards"])
        cache_write.assert_called_once()

    @patch.object(sstats, "_sstats_request")
    @patch.object(sstats, "cache_put")
    @patch.object(sstats, "cache_get", return_value=None)
    def test_context_enrichment_uses_sstats_ids_for_injuries_lineups_and_referee(
        self, _cache_get, _cache_put, request_mock
    ):
        from datetime import datetime, timezone
        from data.sstats import sstats_context_enrich

        players_home = [{"playerName": f"Home Player {i}", "isStarting": True} for i in range(11)]
        players_away = [{"playerName": f"Away Player {i}", "isStarting": True} for i in range(11)]
        def response(data):
            return _Response(200, {"status": "success", "data": data})
        request_mock.side_effect = [
            response([{
                "id": "flash-home-away",
                "date": "2026-11-11T14:00:00Z",
                "homeTeam": {"name": "Home FC"},
                "awayTeam": {"name": "Away FC"},
            }]),
            response([{
                "player": {"id": 9, "name": "Injured Player", "type": "Missing Fixture"},
                "team": {"id": 1, "name": "Home FC"},
                "reason": "Injury",
            }]),
            response({"referee": {"name": "Referee SStats"}}),
            response({
                "homeTeam": {"name": "Home FC", "players": players_home},
                "awayTeam": {"name": "Away FC", "players": players_away},
            }),
        ]
        rows = [{
            "Date": "2026-11-11T14:00:00",
            "kickoff_utc": "2026-11-11T14:00:00Z",
            "HomeTeam": "Home FC",
            "AwayTeam": "Away FC",
            "sstats_game_id": 777,
        }]
        mapping, logs = sstats_context_enrich(
            rows, "private-key",
            date_from="2026-11-11", date_to="2026-11-11",
            now=datetime(2026, 11, 11, 10, tzinfo=timezone.utc),
        )
        context = mapping[("2026-11-11", sstats._norm_name("Home FC"), sstats._norm_name("Away FC"))]
        self.assertEqual(context["provider"], "sstats")
        self.assertTrue(context["injuries_is_confirmed"])
        self.assertEqual(context["injuries"][0]["player_name"], "Injured Player")
        self.assertTrue(context["referee"]["is_confirmed"])
        self.assertEqual(context["referee"]["name"], "Referee SStats")
        self.assertTrue(context["lineup"]["is_confirmed"])
        self.assertEqual(len(context["lineup"]["home"]["starting_xi"]), 11)
        self.assertEqual(len(context["lineup"]["away"]["starting_xi"]), 11)
        self.assertEqual(
            [call.args[0] for call in request_mock.call_args_list],
            ["Ls/List", "Games/injuries", "Ls/GameInfo", "Ls/GameLineups"],
        )
        self.assertEqual(
            request_mock.call_args_list[0].kwargs["params"],
            {"Upcoming": "true", "Limit": 1000},
        )
        self.assertTrue(any("linked fixtures=1/1" in line for line in logs))
        self.assertNotIn("private-key", repr(mapping))
        self.assertTrue(any("confirmed XI=1" in line for line in logs))

    def test_injury_feed_requires_a_structured_list_response(self):
        from data.sstats import _sstats_context_injuries_confirmed
        self.assertTrue(_sstats_context_injuries_confirmed([]))
        self.assertTrue(_sstats_context_injuries_confirmed({"injuries": []}))
        self.assertTrue(_sstats_context_injuries_confirmed({"home": [], "away": []}))
        self.assertFalse(_sstats_context_injuries_confirmed({}))
        self.assertFalse(_sstats_context_injuries_confirmed({"status": "success"}))
        self.assertFalse(_sstats_context_injuries_confirmed(None))

    def test_context_lineup_parser_does_not_confirm_incomplete_or_substitute_lists(self):
        from data.sstats import _sstats_context_side_lineup
        partial = {"homeTeam": {"name": "Home FC", "players": [
            {"playerName": f"Player {i}", "isStarting": True} for i in range(10)
        ]}}
        names = _sstats_context_side_lineup(
            {"status": "success", "data": partial}, "home", "Home FC"
        )
        self.assertEqual(len(names), 10)

        # A generic squad with 11 names is not a starting XI unless the feed
        # explicitly marks those records as starters.
        unmarked = {"homeTeam": {"name": "Home FC", "players": [
            {"playerName": f"Squad Player {i}"} for i in range(11)
        ]}}
        self.assertEqual(
            _sstats_context_side_lineup(
                {"status": "success", "data": unmarked}, "home", "Home FC"
            ),
            [],
        )

        substitutes = {"homeTeam": {"name": "Home FC", "players": [
            {"playerName": f"Substitute {i}", "isStarting": False} for i in range(11)
        ]}}
        self.assertEqual(
            _sstats_context_side_lineup(
                {"status": "success", "data": substitutes}, "home", "Home FC"
            ),
            [],
        )

    @patch.object(sstats, "_sstats_request")
    @patch.object(sstats, "cache_put")
    @patch.object(sstats, "cache_get", return_value=None)
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
        result = sstats.sstats_odds_fixture(9876, "private-key", [])
        self.assertEqual(result["П1"], 1.87)
        self.assertEqual(result["bookmakers"], ["Bookmaker A"])
        self.assertEqual(request_mock.call_args.args[0], "Odds/9876")
        self.assertEqual(request_mock.call_args.kwargs["params"]["opening"], "false")

    @patch.object(sstats, "cache_put")
    @patch.object(sstats, "cache_get", return_value=None)
    @patch.object(sstats, "_sstats_timezone")
    @patch.object(sstats, "_sstats_request")
    def test_account_check_tests_multiple_markets_and_confirms_totals(
        self, request_mock, timezone_mock, _cache_get, _cache_put
    ):
        from datetime import datetime, timedelta
        now = datetime(2026, 11, 2, 12, 0)
        timezone_mock.return_value = (now, 2)
        game_date = (now + timedelta(days=1)).isoformat()
        request_mock.side_effect = [
            _Response(200, {"status": "success", "data": {"userName": "test-user"}}),
            _Response(200, {"status": "success", "data": [{
                "id": 777, "date": game_date,
                "homeTeam": {"name": "Home"}, "awayTeam": {"name": "Away"},
            }]}),
            _Response(200, {"status": "success", "data": [
                {"bookmakerName": "Book A", "odds": [{
                    "marketName": "Total Goals",
                    "odds": [
                        {"name": "Over 2.5", "value": "1.80"},
                        {"name": "Under 2.5", "value": "1.95"},
                    ],
                }]},
                {"bookmakerName": "Book B", "odds": [{
                    "marketName": "Total Goals",
                    "odds": [
                        {"name": "Over 2.5", "value": "1.85"},
                        {"name": "Under 2.5", "value": "1.90"},
                    ],
                }]},
            ]}),
        ]

        result = sstats.sstats_account_check("private-test-key")

        self.assertTrue(result["ok"])
        self.assertTrue(result["odds_checked"])
        self.assertEqual(result["checked_matches"], 1)
        self.assertEqual(result["totals_fixtures"], 1)
        self.assertEqual(result["total_line_count"], 1)
        self.assertIn("подтверждённые тоталы: 1 линий", result["message"])
        self.assertNotIn("private-test-key", repr(result))

    @patch.object(sstats, "cache_put")
    @patch.object(sstats, "cache_get", return_value=None)
    @patch.object(sstats, "_sstats_timezone")
    @patch.object(sstats, "_sstats_request")
    def test_account_check_reports_single_book_total_as_unconfirmed(
        self, request_mock, timezone_mock, _cache_get, _cache_put
    ):
        from datetime import datetime, timedelta
        now = datetime(2026, 11, 2, 12, 0)
        timezone_mock.return_value = (now, 2)
        request_mock.side_effect = [
            _Response(200, {"status": "success", "data": {"userName": "test-user"}}),
            _Response(200, {"status": "success", "data": [{
                "id": 778, "date": (now + timedelta(days=1)).isoformat(),
                "homeTeam": {"name": "Home"}, "awayTeam": {"name": "Away"},
            }]}),
            _Response(200, {"status": "success", "data": [{
                "bookmakerName": "Book A",
                "odds": [{
                    "marketName": "Total Goals",
                    "odds": [
                        {"name": "Over 2.5", "value": "1.85"},
                        {"name": "Under 2.5", "value": "1.95"},
                    ],
                }],
            }]}),
        ]

        result = sstats.sstats_account_check("private-test-key")

        self.assertTrue(result["odds_checked"])
        self.assertEqual(result["total_quote_line_count"], 1)
        self.assertEqual(result["total_line_count"], 0)
        self.assertEqual(result["unconfirmed_total_line_count"], 1)
        self.assertIn("только от одного букмекера", result["message"])
        self.assertNotIn("private-test-key", repr(result))

    @patch.object(sstats, "_sstats_request")
    def test_account_check_confirms_key_without_exposing_it(self, request_mock):
        request_mock.side_effect = [
            _Response(200, {"status": "success", "data": {"userName": "test-user"}}),
            _Response(200, {"status": "success", "count": 0, "data": []}),
        ]
        result = sstats.sstats_account_check("never-echo-this-key")
        self.assertTrue(result["ok"])
        self.assertTrue(result["games_ok"])
        self.assertEqual(result["games_count"], 0)
        self.assertNotIn("never-echo-this-key", repr(result))


if __name__ == "__main__":
    unittest.main()
