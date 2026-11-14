import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llm.deep_analysis import (
    analyze_top_matches,
    build_match_dossier,
    format_top_match_opinion,
    merge_context_snapshots,
    select_top_matches,
    top_match_shortlist_diagnostics,
)


def make_card(
    *,
    match="Home FC vs Away FC",
    date="2026-11-20",
    p1=0.72,
    px=0.18,
    p2=0.10,
    home_sample=7,
    away_sample=6,
    odds=None,
    context=None,
):
    return {
        "fixture_id": f"sstats:{match}:{date}",
        "sstats_game_id": 789,
        "home_team": "Home FC",
        "away_team": "Away FC",
        "match": match,
        "match_ru": "Home FC — Away FC",
        "league": "Test League",
        "date_iso": date,
        "time": "19:00",
        "odds_source": "sstats",
        "market_odds": odds or {"П1": 2.10, "X": 3.40, "П2": 3.60},
        "p1": p1,
        "px": px,
        "p2": p2,
        "games": min(home_sample, away_sample),
        "fh": "ВВПВВ",
        "fa": "ППВНП",
        "xg_used": True,
        "independent_features": {
            "home_scored_at_home_avg": 1.8,
            "home_conceded_at_home_avg": 0.7,
            "away_scored_away_avg": 1.1,
            "away_conceded_away_avg": 1.4,
            "home_home_sample": home_sample,
            "away_away_sample": away_sample,
            "markets": {
                "corners": {
                    "home_for_avg": 5.4,
                    "home_against_avg": 3.8,
                    "away_for_avg": 4.1,
                    "away_against_avg": 4.7,
                    "history_matches": 240,
                    "home_team_games": 8,
                    "away_team_games": 7,
                },
                "cards": {
                    "home_for_avg": 2.0,
                    "home_against_avg": 1.8,
                    "away_for_avg": 2.1,
                    "away_against_avg": 2.0,
                    "history_matches": 220,
                    "home_team_games": 8,
                    "away_team_games": 7,
                },
            },
        },
        "historical_stats": {
            "cards": {
                "referee": {"available": True, "n": 18, "mean": 4.2, "variance": 1.4}
            }
        },
        "api_context": context or {
            "provider": "sstats",
            "sstats_game_id": "789",
            "sstats_event_id": "event-789",
            "kickoff_at": "2026-11-20T19:00:00Z",
            "timestamp": "2026-11-20T12:00:00Z",
            "lineup": {
                "home": {"starting_xi": [f"Home Player {i}" for i in range(11)], "confirmed": True},
                "away": {"starting_xi": [f"Away Player {i}" for i in range(11)], "confirmed": True},
                "is_confirmed": True,
            },
            "referee": {
                "name": "Referee Example",
                "is_confirmed": True,
            },
            "injuries_is_confirmed": True,
            "injuries": [{
                "player_name": "Injured Example",
                "team_name": "Home FC",
                "type": "Missing Fixture",
                "reason": "Injury",
            }],
        },
    }


class TopMatchSelectionTests(unittest.TestCase):
    def test_independent_shortlist_does_not_require_model_favourite_or_odds(self):
        low_probability = make_card(
            match="Low FC vs Mid FC", p1=0.45, px=0.30, p2=0.25,
            odds={"П1": 1.5},
        )
        low_probability.pop("sstats_game_id")
        poor_history = make_card(
            match="No History FC vs Away FC", home_sample=3,
        )
        selected = select_top_matches([low_probability, poor_history])
        self.assertEqual(len(selected), 1)
        self.assertIs(selected[0]["card"], low_probability)
        self.assertEqual(selected[0]["home_sample_games"], 7)
        self.assertEqual(selected[0]["away_sample_games"], 6)

    def test_shortlist_prioritizes_evidence_not_primary_model_probability(self):
        thin_high_probability = make_card(
            match="Thin FC vs Thin Away FC", p1=0.95, px=0.03, p2=0.02,
            home_sample=5, away_sample=5,
        )
        rich_lower_probability = make_card(
            match="Rich FC vs Rich Away FC", p1=0.50, px=0.28, p2=0.22,
            home_sample=12, away_sample=11,
        )
        selected = select_top_matches([thin_high_probability, rich_lower_probability])
        self.assertEqual(selected[0]["card"], rich_lower_probability)

    def test_returns_at_most_seven_unique_matches_and_does_not_fill_slots(self):
        cards = [
            make_card(
                match=f"Home {i} FC vs Away {i} FC",
                date=f"2026-11-{20 + i:02d}",
                p1=0.75 - i * 0.01,
                px=0.15 + i * 0.005,
                p2=0.10 + i * 0.005,
            )
            for i in range(9)
        ]
        duplicate = dict(cards[0])
        duplicate["p1"] = 0.01
        duplicate["px"] = 0.01
        duplicate["p2"] = 0.98
        result = select_top_matches(cards + [duplicate], limit=7)
        self.assertEqual(len(result), 7)
        self.assertEqual(len({
            (item["card"]["date_iso"], item["card"]["match"])
            for item in result
        }), 7)
        self.assertEqual(len(result), 7)
        self.assertTrue(all("evidence_score" in item for item in result))

    def test_generic_overall_games_count_does_not_override_venue_samples(self):
        card = make_card(home_sample=6, away_sample=7)
        card["games"] = 2
        selected = select_top_matches([card])
        self.assertEqual(len(selected), 1)
        self.assertIs(selected[0]["card"], card)

    def test_non_sstats_fixture_can_enter_independent_shortlist(self):
        card = make_card()
        card.pop("sstats_game_id")
        self.assertEqual(len(select_top_matches([card])), 1)

    def test_missing_prices_do_not_exclude_independent_analysis(self):
        card = make_card()
        card["market_odds"] = {}
        card["total_market_odds"] = {}
        card["btts_market_odds"] = {}
        card["stat_market_odds"] = {}
        card["asian_handicap_market_odds"] = {}
        self.assertEqual(len(select_top_matches([card])), 1)

    def test_diagnostics_explain_history_and_market_coverage(self):
        cards = [
            make_card(),
            make_card(match="No History FC vs Away FC", home_sample=2),
            make_card(
                match="No Corners FC vs Away FC",
                home_sample=8, away_sample=8,
                odds={},
            ),
        ]
        cards[2]["independent_features"]["markets"]["corners"] = {}
        result = top_match_shortlist_diagnostics(cards)
        self.assertEqual(result["fixtures"], 3)
        self.assertEqual(result["enough_venue_history"], 2)
        self.assertEqual(result["with_corners_history"], 1)
        self.assertEqual(result["eligible"], 2)


class MatchDossierTests(unittest.TestCase):
    def test_dossier_contains_sstats_facts_but_never_model_pick_or_bookmaker_prices(self):
        card = make_card()
        card["independent_features"]["xg_history"] = {
            "provider": {
                "home_xg_for_avg": 1.42, "home_xga_avg": 0.92,
                "away_xg_for_avg": 1.11, "away_xga_avg": 1.31,
                "home_xg_samples": 8, "home_xga_samples": 8,
                "away_xg_samples": 7, "away_xga_samples": 7,
            },
            "proxy": {"home_xg_for_avg": 1.30, "home_xga_avg": 1.05,
                      "away_xg_for_avg": 1.05, "away_xga_avg": 1.22,
                      "home_xg_samples": 9, "home_xga_samples": 9,
                      "away_xg_samples": 8, "away_xga_samples": 8},
        }
        dossier = build_match_dossier(card)
        self.assertEqual(dossier["xg"]["provider_history"]["home_xg_for_avg"], 1.42)
        self.assertEqual(dossier["xg"]["proxy_history"]["home_xg_for_avg"], 1.3)
        self.assertEqual(dossier["sstats_context"]["referee"]["name"], "Referee Example")
        self.assertTrue(dossier["sstats_context"]["lineup"]["home_confirmed"])
        self.assertEqual(len(dossier["sstats_context"]["lineup"]["home_players"]), 11)
        self.assertEqual(
            dossier["sstats_context"]["injuries_listed"][0]["player"],
            "Injured Example",
        )
        self.assertEqual(dossier["history"]["goals"]["home_home_sample"], 7)
        self.assertNotIn("market_odds", repr(dossier))
        self.assertNotIn('"p1"', dossier)
        self.assertNotIn("model_pick_code", dossier)

    def test_quality_is_capped_when_lineups_or_referee_are_unconfirmed(self):
        context = {
            "provider": "sstats",
            "lineup": {
                "home": {"starting_xi": [], "confirmed": False},
                "away": {"starting_xi": [], "confirmed": False},
            },
            "referee": {"name": None, "is_confirmed": False},
            "injuries_is_confirmed": False,
            "injuries": [],
        }
        dossier = build_match_dossier(make_card(context=context))
        self.assertEqual(dossier["quality_cap"], "medium")
        self.assertFalse(dossier["source_status"]["starting_xi_both_teams_confirmed"])

    def test_context_merge_preserves_confirmed_old_data_when_refresh_is_empty(self):
        from llm.deep_analysis import merge_context_snapshots
        old = {
            "provider": "sstats",
            "referee": {"name": "Referee A", "is_confirmed": True},
            "lineup": {
                "home": {"starting_xi": ["H1"], "confirmed": True},
                "away": {"starting_xi": ["A1"], "confirmed": True},
                "is_confirmed": True,
            },
            "injuries_is_confirmed": True,
            "injuries": [{"player_name": "Player A"}],
        }
        fresh = {
            "provider": "sstats",
            "referee": {"name": None, "is_confirmed": False},
            "lineup": {
                "home": {"starting_xi": [], "confirmed": False},
                "away": {"starting_xi": [], "confirmed": False},
                "is_confirmed": False,
            },
            "injuries_is_confirmed": False,
            "injuries": [],
        }
        merged = merge_context_snapshots(old, fresh)
        self.assertTrue(merged["referee"]["is_confirmed"])
        self.assertEqual(merged["referee"]["name"], "Referee A")
        self.assertTrue(merged["lineup"]["is_confirmed"])
        self.assertTrue(merged["injuries_is_confirmed"])
        self.assertEqual(merged["injuries"][0]["player_name"], "Player A")


class BatchGroqAnalysisTests(unittest.TestCase):
    def test_one_batch_request_returns_validated_analysis_and_caps_quality(self):
        from llm import deep_analysis

        card = make_card(home_sample=9, away_sample=10)
        dossier = build_match_dossier(card)
        response_content = json.dumps({
            "analyses": [{
                "fixture_id": dossier["fixture_id"],
                "independent_pick": "П1",
                "probabilities": {"П1": 52, "X": 28, "П2": 20},
                "goal_markets": {
                    "expected_home_goals": 1.7, "expected_away_goals": 1.0,
                    "over_1_5": 78, "over_2_5": 56, "over_3_5": 34,
                    "over_4_5": 17, "btts_yes": 49,
                },
                "corner_markets": {
                    "expected_home_corners": 5.4, "expected_away_corners": 4.2,
                    "over_8_5": 62, "over_9_5": 51, "over_10_5": 40,
                    "over_11_5": 29,
                },
                "card_markets": {
                    "expected_home_cards": 2.0, "expected_away_cards": 2.1,
                    "over_2_5": 77, "over_3_5": 62, "over_4_5": 43,
                    "over_5_5": 25,
                },
                "lineup_assessment": "Оба стартовых XI подтверждены.",
                "injury_assessment": "В фиде SStats перечислен один отсутствующий игрок.",
                "referee_assessment": "Назначение арбитра подтверждено.",
                "form_assessment": "Домашняя и выездная выборки достаточны для базового сравнения.",
                "match_assessment": "У хозяев умеренное преимущество, но исход не выглядит безрисковым.",
                "risks": "Нужна повторная проверка перед началом матча.",
                "data_quality": "high",
            }]
        }, ensure_ascii=False)
        response = SimpleNamespace(
            status_code=200,
            text=response_content,
            json=lambda: {"choices": [{
                "message": {"content": response_content},
                "finish_reason": "stop",
            }]},
        )
        with (
            patch.object(deep_analysis, "cache_get", return_value=None),
            patch.object(deep_analysis, "cache_put") as cache_write,
            patch.object(deep_analysis.usage, "llm_remaining", return_value=10),
            patch.object(deep_analysis.usage, "llm_increment") as increment,
            patch.object(deep_analysis._sess, "post", return_value=response) as post,
        ):
            result = deep_analysis.analyze_top_matches({
                "api_key": "private-unit-test-key",
                "provider": "Groq (бесплатно, быстро)",
                "model": "openai/gpt-oss-20b",
                "candidates": [dossier],
            })

        self.assertIn(dossier["fixture_id"], result)
        analysis = result[dossier["fixture_id"]]
        self.assertEqual(analysis["independent_pick"], "П1")
        self.assertEqual(sum(analysis["probabilities"].values()), 100.0)
        self.assertEqual(analysis["data_quality"], "high")
        self.assertEqual(analysis["goal_markets"]["over_2_5"], 56.0)
        self.assertEqual(analysis["corner_markets"]["expected_home_corners"], 5.4)
        self.assertEqual(analysis["card_markets"]["over_4_5"], 43.0)
        request = post.call_args.kwargs["json"]
        self.assertEqual(request["reasoning_effort"], "low")
        self.assertEqual(request["reasoning_format"], "hidden")
        self.assertEqual(request["response_format"]["type"], "json_schema")
        prompt = request["messages"][0]["content"]
        self.assertIn("Referee Example", prompt)
        self.assertIn("Injured Example", prompt)
        self.assertNotIn("private-unit-test-key", prompt)
        self.assertNotIn("market_odds", prompt)
        self.assertNotIn("model_pick_code", prompt)
        increment.assert_called_once_with(1)
        cache_write.assert_called_once()

    def test_drops_unknown_fixture_ids_and_rejects_bad_probability_distributions(self):
        from llm import deep_analysis

        dossier = build_match_dossier(make_card())
        response_content = json.dumps({
            "analyses": [
                {
                    "fixture_id": dossier["fixture_id"],
                    "independent_pick": "П1",
                    "probabilities": {"П1": 100, "X": 100, "П2": 100},
                    "goal_markets": null, "corner_markets": null, "card_markets": null,
                    "lineup_assessment": "x", "injury_assessment": "x",
                    "referee_assessment": "x", "form_assessment": "x",
                    "match_assessment": "x", "risks": "x", "data_quality": "high",
                },
                {
                    "fixture_id": "unknown-fixture",
                    "independent_pick": "П1",
                    "probabilities": {"П1": 50, "X": 30, "П2": 20},
                    "goal_markets": null, "corner_markets": null, "card_markets": null,
                    "lineup_assessment": "x", "injury_assessment": "x",
                    "referee_assessment": "x", "form_assessment": "x",
                    "match_assessment": "x", "risks": "x", "data_quality": "low",
                },
            ]
        }, ensure_ascii=False)
        response = SimpleNamespace(
            status_code=200, text=response_content,
            json=lambda: {"choices": [{"message": {"content": response_content}}]},
        )
        with (
            patch.object(deep_analysis, "cache_get", return_value=None),
            patch.object(deep_analysis, "cache_put"),
            patch.object(deep_analysis.usage, "llm_remaining", return_value=10),
            patch.object(deep_analysis.usage, "llm_increment"),
            patch.object(deep_analysis._sess, "post", return_value=response),
        ):
            result = deep_analysis.analyze_top_matches({
                "api_key": "test",
                "provider": "Groq (бесплатно, быстро)",
                "model": "openai/gpt-oss-20b",
                "candidates": [dossier],
            })
        self.assertEqual(result, {})

    def test_xai_grok_uses_current_model_and_strict_multimarket_schema(self):
        from llm import deep_analysis

        dossier = build_match_dossier(make_card())
        response_content = json.dumps({
            "analyses": [{
                "fixture_id": dossier["fixture_id"],
                "independent_pick": "X",
                "probabilities": {"П1": 30, "X": 45, "П2": 25},
                "goal_markets": None,
                "corner_markets": None,
                "card_markets": None,
                "lineup_assessment": "Составы не подтверждены.",
                "injury_assessment": "Нет подтверждённых записей.",
                "referee_assessment": "Назначение не подтверждено.",
                "form_assessment": "Исторические выборки доступны.",
                "match_assessment": "Матч выглядит равным.",
                "risks": "Нет подтверждённых составов и судьи.",
                "data_quality": "medium",
            }]
        }, ensure_ascii=False)
        response = SimpleNamespace(
            status_code=200, text=response_content,
            json=lambda: {"choices": [{"message": {"content": response_content}}]},
        )
        with (
            patch.object(deep_analysis, "cache_get", return_value=None),
            patch.object(deep_analysis, "cache_put"),
            patch.object(deep_analysis.usage, "llm_remaining", return_value=10),
            patch.object(deep_analysis.usage, "llm_increment"),
            patch.object(deep_analysis._sess, "post", return_value=response) as post,
        ):
            result = deep_analysis.analyze_top_matches({
                "api_key": "xai-unit-test-key",
                "provider": "Grok (x.ai)",
                "model": "grok-4.7",
                "candidates": [dossier],
            })
        self.assertIn(dossier["fixture_id"], result)
        request = post.call_args.kwargs["json"]
        self.assertEqual(post.call_args.args[0], "https://api.x.ai/v1/chat/completions")
        self.assertEqual(request["model"], "grok-4.7")
        self.assertEqual(request["response_format"]["type"], "json_schema")
        self.assertEqual(request["response_format"]["json_schema"]["name"],
                         "top_match_independent_multimarket_analysis")

    def test_non_monotone_market_probabilities_are_discarded_without_losing_1x2(self):
        from llm.deep_analysis import _normalize_analysis

        dossier = build_match_dossier(make_card())
        result = _normalize_analysis({
            "fixture_id": dossier["fixture_id"],
            "independent_pick": "П1",
            "probabilities": {"П1": 50, "X": 30, "П2": 20},
            "goal_markets": {
                "expected_home_goals": 1.5, "expected_away_goals": 1.0,
                "over_1_5": 50, "over_2_5": 60, "over_3_5": 30,
                "over_4_5": 10, "btts_yes": 45,
            },
            "corner_markets": None, "card_markets": None,
            "lineup_assessment": "x", "injury_assessment": "x",
            "referee_assessment": "x", "form_assessment": "x",
            "match_assessment": "x", "risks": "x", "data_quality": "medium",
        }, {dossier["fixture_id"]: dossier})
        self.assertIsNotNone(result)
        self.assertIsNone(result["goal_markets"])
        self.assertEqual(result["independent_pick"], "П1")

    def test_formatted_opinion_remains_compatible_with_forecast_ledger(self):
        from model.independent_forecasts import parse_llm_probabilities
        analysis = {
            "independent_pick": "X",
            "probabilities": {"П1": 30.0, "X": 45.0, "П2": 25.0},
            "data_quality": "medium",
            "match_assessment": "Матч близкий.",
            "risks": "Составы не подтверждены.",
            "goal_markets": {
                "expected_home_goals": 1.4, "expected_away_goals": 1.1,
                "over_1_5": 78, "over_2_5": 56, "over_3_5": 34,
                "over_4_5": 17, "btts_yes": 52,
            },
        }
        formatted = format_top_match_opinion(analysis, model_pick_code="П1")
        self.assertIn("РАСХОЖДЕНИЕ", formatted)
        self.assertIn("ТБ 2.5 56%", formatted)
        parsed = parse_llm_probabilities(formatted)
        self.assertEqual(parsed, {"H": 0.30, "D": 0.45, "A": 0.25})


if __name__ == "__main__":
    unittest.main()
