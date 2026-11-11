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
    def test_selects_only_strong_historically_supported_and_priced_matches(self):
        strong = make_card()
        low_probability = make_card(match="Low FC vs Mid FC", p1=0.62, px=0.24, p2=0.14)
        poor_history = make_card(match="No History FC vs Away FC", home_sample=3)
        no_full_price = make_card(
            match="No Price FC vs Away FC",
            odds={"П1": 1.80, "X": 3.20},
        )
        result = select_top_matches([
            strong, low_probability, poor_history, no_full_price,
        ])
        self.assertEqual(len(result), 1)
        self.assertIs(result[0]["card"], strong)
        self.assertEqual(result[0]["pick"], "П1")
        self.assertGreaterEqual(result[0]["probability"], 0.65)

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
        duplicate["p1"] = 0.80
        duplicate["px"] = 0.12
        duplicate["p2"] = 0.08
        result = select_top_matches(cards + [duplicate], limit=7)
        self.assertEqual(len(result), 7)
        self.assertEqual(len({
            (item["card"]["date_iso"], item["card"]["match"])
            for item in result
        }), 7)
        self.assertIs(result[0]["card"], duplicate)

    def test_generic_overall_games_count_does_not_override_venue_samples(self):
        card = make_card(home_sample=6, away_sample=7)
        card["games"] = 2
        selected = select_top_matches([card])
        self.assertEqual(len(selected), 1)
        self.assertIs(selected[0]["card"], card)

    def test_non_sstats_fixture_cannot_enter_sstats_top_shortlist(self):
        card = make_card()
        card.pop("sstats_game_id")
        self.assertEqual(select_top_matches([card]), [])

    def test_diagnostics_show_why_candidates_are_excluded(self):
        cards = [
            make_card(),
            make_card(match="Low FC vs Away FC", p1=0.60, px=0.25, p2=0.15),
            make_card(match="No History FC vs Away FC", home_sample=2),
            make_card(match="No Odds FC vs Away FC", odds={"П1": 2.1}),
        ]
        result = top_match_shortlist_diagnostics(cards)
        self.assertEqual(result["fixtures"], 4)
        self.assertEqual(result["strong_probability"], 3)
        self.assertEqual(result["enough_venue_history"], 2)
        self.assertEqual(result["complete_1x2_prices"], 1)
        self.assertEqual(result["eligible"], 1)


class MatchDossierTests(unittest.TestCase):
    def test_dossier_contains_sstats_facts_but_never_model_pick_or_bookmaker_prices(self):
        card = make_card()
        dossier = build_match_dossier(card)
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
                    "lineup_assessment": "x", "injury_assessment": "x",
                    "referee_assessment": "x", "form_assessment": "x",
                    "match_assessment": "x", "risks": "x", "data_quality": "high",
                },
                {
                    "fixture_id": "unknown-fixture",
                    "independent_pick": "П1",
                    "probabilities": {"П1": 50, "X": 30, "П2": 20},
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

    def test_formatted_opinion_remains_compatible_with_forecast_ledger(self):
        from model.independent_forecasts import parse_llm_probabilities
        analysis = {
            "independent_pick": "X",
            "probabilities": {"П1": 30.0, "X": 45.0, "П2": 25.0},
            "data_quality": "medium",
            "match_assessment": "Матч близкий.",
            "risks": "Составы не подтверждены.",
        }
        formatted = format_top_match_opinion(analysis, model_pick_code="П1")
        self.assertIn("РАСХОЖДЕНИЕ", formatted)
        parsed = parse_llm_probabilities(formatted)
        self.assertEqual(parsed, {"H": 0.30, "D": 0.45, "A": 0.25})


if __name__ == "__main__":
    unittest.main()
