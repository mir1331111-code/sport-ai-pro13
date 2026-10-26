"""Regression tests for the free Prognozist/Marathon odds parser."""
import unittest
import os
from datetime import datetime
from unittest.mock import patch

from data.sources import _prognozist_extract_row_odds


class PrognozistOddsParserTests(unittest.TestCase):
    def test_extracts_explicitly_labelled_1x2_prices(self):
        row = ["Levante — Athletic Club", "П1 2,60 X 3,10 П2 2,90"]
        self.assertEqual(
            _prognozist_extract_row_odds(row),
            {"П1": 2.60, "X": 3.10, "П2": 2.90},
        )

    def test_parses_compact_match_centre_markup_shape(self):
        # Current public match-centre rows often concatenate labels and prices.
        row = ["21:00СкороМанчестер Сити U21Сток Сити U21П11.14X6.35П211.50—"]
        self.assertEqual(
            _prognozist_extract_row_odds(row),
            {"П1": 1.14, "X": 6.35, "П2": 11.50},
        )

    def test_extracts_prices_from_separate_table_cells(self):
        row = ["Levante", "Athletic Club", "2.60", "3.10", "2.90"]
        self.assertEqual(
            _prognozist_extract_row_odds(row),
            {"П1": 2.60, "X": 3.10, "П2": 2.90},
        )

    def test_match_detail_href_stays_attached_to_row_odds(self):
        from data.sources import _prognozist_rows

        html = (
            '<table><tr>'
            '<td><a href="/mc/football/zenit-dinamo-123/">'
            '18:30 Скоро Зенит Динамо Махачкала'
            '</a></td>'
            '<td>П1</td><td>1.26</td><td>X</td><td>6.05</td>'
            '<td>П2</td><td>12.25</td>'
            '</tr></table>'
        )
        rows = _prognozist_rows(html)
        row = next(row for row in rows if row and str(row[0]).startswith("/mc/football/"))
        self.assertEqual(row[0], "/mc/football/zenit-dinamo-123/")
        self.assertEqual(
            _prognozist_extract_row_odds(row),
            {"П1": 1.26, "X": 6.05, "П2": 12.25},
        )

    def test_prognozist_fixture_requires_clear_teams_date_and_full_line(self):
        from data.sources import _prognozist_fixture_from_item
        item = {
            "cells": ["25.10.2026", "19:30", "Атлетико Мадрид", "Севилья", "1.95", "3.40", "4.10"],
            "odds": {"П1": 1.95, "X": 3.40, "П2": 4.10},
        }
        fixture = _prognozist_fixture_from_item(item, today=datetime(2026, 10, 25).date())
        self.assertIsNotNone(fixture)
        self.assertEqual(fixture["Date"], "2026-10-25")
        self.assertEqual(fixture["Time"], "19:30")
        self.assertEqual(fixture["HomeTeam"], "Атлетико Мадрид")
        self.assertEqual(fixture["AwayTeam"], "Севилья")
        self.assertEqual(fixture["prognozist_odds"]["X"], 3.40)

    def test_prognozist_fixture_does_not_guess_compact_team_split(self):
        from data.sources import _prognozist_fixture_from_item
        item = {
            "cells": ["21:00СкороМанчестер Сити U21Сток Сити U21П11.14X6.35П211.50"],
            "odds": {"П1": 1.14, "X": 6.35, "П2": 11.50},
        }
        self.assertIsNone(_prognozist_fixture_from_item(item, today=datetime(2026, 10, 25).date()))

    def test_parses_real_match_centre_heading_and_russian_date(self):
        from data.sources import _prognozist_parse_detail_heading
        fixture = _prognozist_parse_detail_heading(
            "Зенит — Динамо Махачкала прогноз на 25 октября 2026 в 18:30 МСК",
            {"П1": 1.26, "X": 6.05, "П2": 12.25},
            today=datetime(2026, 10, 25).date(),
        )
        self.assertIsNotNone(fixture)
        self.assertEqual(fixture["HomeTeam"], "Зенит")
        self.assertEqual(fixture["AwayTeam"], "Динамо Махачкала")
        self.assertEqual(fixture["Date"], "2026-10-25")
        self.assertEqual(fixture["Time"], "18:30")
        self.assertEqual(fixture["prognozist_odds"]["П2"], 12.25)

    def test_compact_match_centre_row_uses_detail_page_not_guessed_split(self):
        from data import sources
        item = {
            "cells": ["/mc/football/zenit-sankt-peterburg-dinamo-mahachkala-31127120/",
                      "18:30СкороЗенитДинамо МахачкалаП11.26X6.05П212.25—"],
            "odds": {"П1": 1.26, "X": 6.05, "П2": 12.25},
        }
        expected = {"HomeTeam": "Зенит", "AwayTeam": "Динамо Махачкала",
                    "Date": "2026-10-25", "Time": "18:30",
                    "prognozist_odds": {"П1": 1.26, "X": 6.05, "П2": 12.25}}
        with patch.object(sources, "_prognozist_fetch_detail_fixture", return_value=expected) as fetch:
            result = sources._prognozist_fixture_from_item(item, today=datetime(2026, 10, 25).date())
        self.assertEqual(result, expected)
        fetch.assert_called_once()

    def test_extracts_btts_and_main_goal_total_prices(self):
        from data.sources import _prognozist_extract_extra_markets
        html = """
        <h2>Коэффициенты Marathon</h2>
        <h2>Тотал голов</h2><div>Больше 2.5 1.91 Меньше 2.5 1.89</div>
        <h2>Обе забьют?</h2><div>Да 1.72 Нет 2.05</div>
        <h2>1 тайм Тотал</h2><div>Больше 2.5 8.00 Меньше 2.5 1.05</div>
        """
        result = _prognozist_extract_extra_markets(html)
        self.assertEqual(result["btts_market_odds"], {"BTTS Да": 1.72, "BTTS Нет": 2.05})
        self.assertEqual(result["total_market_odds"], {"ТБ 2.5": 1.91, "ТМ 2.5": 1.89})

    def test_parses_compact_marathon_line_total_formats(self):
        from data.sources import _prognozist_extract_extra_markets
        html = """
        <h2>Коэффициенты Marathon</h2>
        <div>Тотал бол. (3.5) 2.375; Тотал мен. (3.5) 1.60</div>
        <div>Дополнительный тотал: (2.5) Бол. 1.181; Мен. 3.95</div>
        <div>Тотал голов, 1-й тайм: Тотал бол. (1.5) 2.10; Тотал мен. (1.5) 1.70</div>
        """
        result = _prognozist_extract_extra_markets(html)
        self.assertEqual(result["total_market_odds"]["ТБ 3.5"], 2.375)
        self.assertEqual(result["total_market_odds"]["ТМ 3.5"], 1.60)
        self.assertEqual(result["total_market_odds"]["ТБ 2.5"], 1.181)
        self.assertEqual(result["total_market_odds"]["ТМ 2.5"], 3.95)
        self.assertNotIn("ТБ 1.5", result["total_market_odds"])

    def test_extracts_total_corners_and_cards_as_distinct_markets(self):
        from data.sources import _prognozist_extract_extra_markets
        html = """
        <h2>Коэффициенты Marathon</h2>
        <h2>Тотал угловых</h2><div>Больше 9.5 1.87 Меньше 9.5 1.93</div>
        <h2>Тотал желтых карточек</h2><div>Больше 4.5 1.90 Меньше 4.5 1.90</div>
        """
        result = _prognozist_extract_extra_markets(html)
        self.assertEqual(result["stat_market_odds"]["Угл ТБ 9.5"], 1.87)
        self.assertEqual(result["stat_market_odds"]["Угл ТМ 9.5"], 1.93)
        self.assertEqual(result["stat_market_odds"]["Карты ТБ 4.5"], 1.90)
        self.assertEqual(result["stat_market_odds"]["Карты ТМ 4.5"], 1.90)

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

    def test_marathon_team_match_does_not_confuse_manchester_clubs(self):
        from data.marathon import _find_team_match
        self.assertFalse(_find_team_match("Manchester United", "Manchester City"))
        self.assertTrue(_find_team_match("Manchester United", "Man Utd"))
        self.assertFalse(_find_team_match("Arsenal", "Arsenal U21"))
        self.assertTrue(_find_team_match("Levante", "Levante UD"))

    def test_marathon_lookup_rejects_wrong_date(self):
        from data import marathon
        event = {"home": "Arsenal", "away": "Chelsea", "date_str": "23.10.2026 20:00",
                 "odds": {"П1": 2.1, "X": 3.2, "П2": 3.5}}
        with patch.object(marathon, "fetch_league_events", return_value=[event]):
            self.assertIsNone(marathon.marathon_odds(
                "Arsenal", "Chelsea", "2026-10-24", "E0"
            ))

    def test_marathon_lookup_selects_correct_date_for_repeated_fixture(self):
        from data import marathon
        old = {"home": "Arsenal", "away": "Chelsea", "date_str": "23.10.2026 20:00",
               "odds": {"П1": 2.1, "X": 3.2, "П2": 3.5}}
        current = {"home": "Arsenal", "away": "Chelsea", "date_str": "24.10.2026 20:00",
                   "odds": {"П1": 2.0, "X": 3.3, "П2": 3.7}}
        with patch.object(marathon, "fetch_league_events", return_value=[old, current]):
            result = marathon.marathon_odds("Arsenal", "Chelsea", "2026-10-24", "E0")
        self.assertEqual(result["П1"], 2.0)

    def test_marathon_date_parser_infers_year_from_fixture(self):
        from data.marathon import _event_calendar_date
        self.assertEqual(
            _event_calendar_date("24.10 20:00", "2026-10-24"),
            "2026-10-24",
        )
        self.assertEqual(
            _event_calendar_date("23.10 20:00", "2026-10-24"),
            "2026-10-23",
        )

    def test_marathon_lookup_rejects_ambiguous_same_date_lines(self):
        from data import marathon
        events = [
            {"home": "Arsenal", "away": "Chelsea", "date_str": "24.10.2026 18:00",
             "odds": {"П1": 2.0, "X": 3.3, "П2": 3.7}},
            {"home": "Arsenal", "away": "Chelsea", "date_str": "24.10.2026 20:00",
             "odds": {"П1": 2.2, "X": 3.1, "П2": 3.4}},
        ]
        with patch.object(marathon, "fetch_league_events", return_value=events):
            self.assertIsNone(marathon.marathon_odds(
                "Arsenal", "Chelsea", "2026-10-24", "E0"
            ))

    def test_marathon_fallback_is_disabled_without_permission(self):
        from data.marathon import _permitted_by_robots, _sess
        with patch.dict(os.environ, {"PROGNOZIST_PERMISSION": ""}):
            with patch.object(_sess, "get") as mocked_get:
                self.assertFalse(_permitted_by_robots(
                    "https://prognozist.ru/line/marathon/event_line.php?category=21520"
                ))
                mocked_get.assert_not_called()

    def test_marathon_cached_line_does_not_bypass_disabled_permission(self):
        from data import marathon
        cached_events = [{"home": "Arsenal", "away": "Chelsea",
                          "odds": {"П1": 2.1, "X": 3.2, "П2": 3.5}}]
        with patch.dict(os.environ, {"PROGNOZIST_PERMISSION": ""}):
            with (
                patch.object(marathon, "cache_get", return_value=cached_events) as cached,
                patch.object(marathon, "_permitted_by_robots", return_value=False) as permission,
                patch.object(marathon._sess, "get") as mocked_get,
            ):
                self.assertEqual(marathon.fetch_league_events("E0"), [])
                permission.assert_called_once()
                cached.assert_not_called()
                mocked_get.assert_not_called()

    def test_source_status_reports_safe_default(self):
        from data.sources import prognozist_source_status
        with patch.dict(os.environ, {"PROGNOZIST_PERMISSION": ""}):
            self.assertEqual(prognozist_source_status(), "disabled_permission")

    def test_odds_api_does_not_match_senior_fixture_to_u21_event(self):
        from data import sources
        events = [{
            "home_team": "Manchester City U21",
            "away_team": "Arsenal",
            "bookmakers": [{
                "markets": [{"key": "h2h", "outcomes": [
                    {"name": "Manchester City U21", "price": 1.4},
                    {"name": "Draw", "price": 4.8},
                    {"name": "Arsenal", "price": 6.0},
                ]}]
            }],
        }]
        with (
            patch.object(sources, "cache_get", side_effect=[None, events]),
            patch.object(sources.usage, "odds_remaining", return_value=10),
        ):
            self.assertIsNone(sources.odds_api_fixture(
                "soccer_epl", "Manchester City", "Arsenal", "test-key", []
            ))

    def test_prognozist_match_respects_home_away_order(self):
        from data.sources import _prognozist_match
        self.assertTrue(_prognozist_match(
            ["Manchester United", "Manchester City", "П1 2.10 X 3.20 П2 3.50"],
            "Manchester United", "Manchester City",
        ))
        self.assertFalse(_prognozist_match(
            ["Manchester City", "Manchester United", "П1 2.10 X 3.20 П2 3.50"],
            "Manchester United", "Manchester City",
        ))

    def test_prognozist_match_does_not_confuse_city_and_united(self):
        from data.sources import _prognozist_match
        self.assertFalse(_prognozist_match(
            ["Manchester City", "Arsenal", "П1 2.10 X 3.20 П2 3.50"],
            "Manchester United", "Arsenal",
        ))

    def test_match_lookup_uses_line_only_when_permission_enabled(self):
        from data import sources
        odds = {"П1": 2.1, "X": 3.2, "П2": 3.5}
        with patch.dict(os.environ, {"PROGNOZIST_PERMISSION": ""}):
            with patch.object(sources, "cache_get", return_value=odds) as cached:
                self.assertIsNone(sources.prognozist_marathon_odds("Arsenal", "Chelsea"))
                cached.assert_not_called()

    def test_match_lookup_honors_streamlit_secret_permission(self):
        from data import sources
        odds = {"П1": 2.1, "X": 3.2, "П2": 3.5}
        with patch.dict(os.environ, {"PROGNOZIST_PERMISSION": ""}):
            with (
                patch.object(sources, "prognozist_permission_enabled", return_value=True),
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

    def test_cached_prognozist_odds_do_not_bypass_robots_block(self):
        from data import sources
        with patch.dict(os.environ, {"PROGNOZIST_PERMISSION": "1"}):
            with (
                patch.object(sources, "_prognozist_allowed", return_value=False),
                patch.object(sources, "cache_get", return_value={"П1": 2.1, "X": 3.2, "П2": 3.5}) as cached,
            ):
                self.assertIsNone(sources.prognozist_marathon_odds("Arsenal", "Chelsea"))
                cached.assert_not_called()

    def test_cached_prognozist_line_index_does_not_bypass_robots_block(self):
        from data import sources
        with patch.dict(os.environ, {"PROGNOZIST_PERMISSION": "1"}):
            with (
                patch.object(sources, "_prognozist_allowed", return_value=False),
                patch.object(sources, "cache_get", return_value=[
                    {"text": "Arsenal Chelsea", "odds": {"П1": 2.1, "X": 3.2, "П2": 3.5}}
                ]) as cached,
            ):
                self.assertEqual(sources._prognozist_line_index(), [])
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

    def test_match_lookup_rejects_line_for_different_calendar_date(self):
        from data import sources
        odds = {"П1": 2.1, "X": 3.2, "П2": 3.5}
        with patch.dict(os.environ, {"PROGNOZIST_PERMISSION": "1"}):
            with (
                patch.object(sources, "cache_get", return_value=None),
                patch.object(sources, "_prognozist_line_index", return_value=[
                    {"text": "23.10.2026 Arsenal Chelsea", "odds": odds}
                ]),
                patch.object(sources, "cache_put"),
            ):
                self.assertIsNone(
                    sources.prognozist_marathon_odds(
                        "Arsenal", "Chelsea", kickoff=datetime(2026, 10, 24)
                    )
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



    def test_prognozist_matches_russian_match_centre_names_to_english_fixtures(self):
        from data.sources import _prognozist_match
        row = ["17:00СкороЛиверпульБрайтонП12.12X3.98П23.24—"]
        self.assertTrue(_prognozist_match(row, "Liverpool", "Brighton & Hove Albion"))
        self.assertFalse(_prognozist_match(row, "Brighton & Hove Albion", "Liverpool"))

    def test_prognozist_lookup_returns_public_russian_page_line(self):
        from data import sources
        odds = {"П1": 2.12, "X": 3.98, "П2": 3.24}
        with patch.dict(os.environ, {"PROGNOZIST_PERMISSION": "1"}):
            with (
                patch.object(sources, "_prognozist_allowed", return_value=True),
                patch.object(sources, "cache_get", return_value=None),
                patch.object(sources, "_prognozist_line_index", return_value=[
                    {"text": "17:00СкороЛиверпульБрайтонП12.12X3.98П23.24—", "odds": odds}
                ]),
                patch.object(sources, "cache_put"),
            ):
                result = sources.prognozist_marathon_odds(
                    "Liverpool", "Brighton & Hove Albion",
                    kickoff=datetime(2026, 10, 25, 17, 0),
                )
        self.assertEqual(result, odds)

    def test_odds_api_retries_h2h_when_endpoint_rejects_market_combo(self):
        from types import SimpleNamespace
        from data import sources
        events = [{
            "home_team": "Liverpool",
            "away_team": "Brighton & Hove Albion",
            "bookmakers": [{
                "markets": [{
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Liverpool", "price": 2.12},
                        {"name": "Draw", "price": 3.98},
                        {"name": "Brighton & Hove Albion", "price": 3.24},
                    ],
                }]
            }],
        }]
        bad = SimpleNamespace(
            status_code=422,
            text='{"error_code":"INVALID_MARKET","message":"Markets not supported by this endpoint: totals"}',
        )
        good = SimpleNamespace(status_code=200, text="[]", json=lambda: events)
        logs = []
        with (
            patch.object(sources, "cache_get", return_value=None),
            patch.object(sources, "cache_put"),
            patch.object(sources.usage, "odds_remaining", return_value=10),
            patch.object(sources.usage, "odds_increment"),
            patch.object(sources._sess, "get", side_effect=[bad, good]) as mocked_get,
        ):
            result = sources.odds_api_fixture(
                "soccer_epl", "Liverpool", "Brighton & Hove Albion", "test-key", logs
            )
        self.assertEqual(result, {"П1": 2.12, "X": 3.98, "П2": 3.24})
        self.assertEqual(mocked_get.call_count, 2)
        self.assertEqual(mocked_get.call_args_list[0].kwargs["params"]["markets"], "h2h,totals")
        self.assertEqual(mocked_get.call_args_list[1].kwargs["params"]["markets"], "h2h")
        self.assertNotIn("btts", mocked_get.call_args_list[0].kwargs["params"]["markets"])
        self.assertTrue(any("повторяю запрос только с h2h" in line for line in logs))



    def test_unlabelled_decimal_odds_cells_are_not_mistaken_for_dates(self):
        from data import sources
        odds = {"П1": 2.60, "X": 3.10, "П2": 2.90}
        item = {
            "text": "Liverpool Brighton 2.60 3.10 2.90",
            "cells": ["Liverpool", "Brighton", "2.60", "3.10", "2.90"],
            "odds": odds,
        }
        with patch.dict(os.environ, {"PROGNOZIST_PERMISSION": "1"}):
            with (
                patch.object(sources, "_prognozist_allowed", return_value=True),
                patch.object(sources, "cache_get", return_value=None),
                patch.object(sources, "_prognozist_line_index", return_value=[item]),
                patch.object(sources, "cache_put"),
            ):
                result = sources.prognozist_marathon_odds(
                    "Liverpool", "Brighton", kickoff=datetime(2026, 10, 25, 17, 0)
                )
        self.assertEqual(result, odds)



    def test_odds_api_retries_h2h_when_featured_markets_are_rejected(self):
        from types import SimpleNamespace
        from data import sources

        events = [{
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "bookmakers": [{
                "markets": [{
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Arsenal", "price": 2.1},
                        {"name": "Draw", "price": 3.2},
                        {"name": "Chelsea", "price": 3.5},
                    ],
                }]
            }],
        }]
        rejected = SimpleNamespace(
            status_code=422,
            text='{"error_code":"INVALID_MARKET","message":"unsupported market"}',
        )
        accepted = SimpleNamespace(status_code=200, text="", json=lambda: events)
        logs = []
        with (
            patch.object(sources, "cache_get", side_effect=[None, None]),
            patch.object(sources, "cache_put"),
            patch.object(sources.usage, "odds_remaining", return_value=10),
            patch.object(sources.usage, "odds_increment"),
            patch.object(sources._sess, "get", side_effect=[rejected, accepted]) as mocked_get,
        ):
            result = sources.odds_api_fixture(
                "soccer_epl", "Arsenal", "Chelsea", "test-key", logs
            )

        self.assertEqual(result, {"П1": 2.1, "X": 3.2, "П2": 3.5})
        self.assertEqual(mocked_get.call_count, 2)
        first_params = mocked_get.call_args_list[0].kwargs["params"]
        second_params = mocked_get.call_args_list[1].kwargs["params"]
        self.assertEqual(first_params["markets"], "h2h,totals")
        self.assertNotIn("btts", first_params["markets"].lower())
        self.assertEqual(second_params["markets"], "h2h")
        self.assertTrue(any("повторяю запрос только с h2h" in line for line in logs))

if __name__ == "__main__":
    unittest.main()

    def test_prognozist_matches_returns_empty_line_without_name_error(self):
        from data import sources
        with (
            patch.object(sources, "prognozist_permission_enabled", return_value=True),
            patch.object(sources, "_prognozist_line_index", return_value=[]),
        ):
            self.assertEqual(sources.prognozist_matches(days=1, logs=[]), [])
