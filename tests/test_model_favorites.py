import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from betting.ranking import (
    model_favorite, rank_best_fixtures, rank_model_favorites,
    rank_model_favorites_with_minimum,
)


class ModelFavoriteTests(unittest.TestCase):
    def test_selects_home_win_as_most_likely_1x2_outcome(self):
        result = model_favorite({
            "home": "Alpha FC", "away": "Beta FC",
            "p1": 0.68, "px": 0.20, "p2": 0.12,
        })
        self.assertEqual(result["pick"], "П1")
        self.assertEqual(result["label"], "Победа Alpha FC")
        self.assertAlmostEqual(result["probability"], 0.68)
        self.assertEqual(result["strength"], "Фаворит")

    def test_selects_draw_when_draw_is_most_likely(self):
        result = model_favorite({
            "match_ru": "Alpha — Beta",
            "p1": 0.29, "px": 0.43, "p2": 0.28,
        })
        self.assertEqual(result["pick"], "X")
        self.assertEqual(result["label"], "Ничья")

    def test_selects_away_win_as_most_likely_outcome(self):
        result = model_favorite({
            "match": "Alpha vs Beta",
            "p1": 0.18, "px": 0.22, "p2": 0.60,
        })
        self.assertEqual(result["pick"], "П2")
        self.assertEqual(result["label"], "Победа Beta")
        self.assertEqual(result["strength"], "Фаворит")

    def test_rejects_missing_or_malformed_probabilities(self):
        self.assertIsNone(model_favorite({"p1": 0.6, "px": 0.4}))
        self.assertIsNone(model_favorite({"p1": 60, "px": 20, "p2": 20}))
        self.assertIsNone(model_favorite({"p1": float("nan"), "px": 0.2, "p2": 0.8}))
        self.assertIsNone(model_favorite({"p1": 0.0, "px": 0.0, "p2": 0.0}))
        self.assertIsNone(model_favorite(None))

    def test_ranking_uses_probability_before_ev_or_input_order(self):
        lower = {"match": "Lower vs Side", "p1": 0.57, "px": 0.23, "p2": 0.20,
                 "verdict": {"ev": 0.50}}
        higher = {"match": "Higher vs Side", "p1": 0.72, "px": 0.16, "p2": 0.12,
                  "verdict": {"ev": -0.20}}
        ranked = rank_model_favorites([lower, higher])
        self.assertEqual([row["card"]["match"] for row in ranked],
                         ["Higher vs Side", "Lower vs Side"])

    def test_ties_are_stable_and_use_fixture_date_then_match(self):
        later = {"match": "Zed vs Away", "date_iso": "2026-11-05",
                 "p1": 0.6, "px": 0.2, "p2": 0.2}
        earlier = {"match": "Alpha vs Away", "date_iso": "2026-11-04",
                   "p1": 0.6, "px": 0.2, "p2": 0.2}
        ranked = rank_model_favorites([later, earlier])
        self.assertEqual(ranked[0]["card"]["match"], "Alpha vs Away")

    def test_order_is_independent_of_source_position_and_prioritizes_probability(self):
        low = {
            "match": "Low vs Other", "date_iso": "2026-11-06",
            "p1": 0.56, "px": 0.24, "p2": 0.20, "games": 12,
            "verdict": {"ev": 0.80},
        }
        high = {
            "match": "High vs Other", "date_iso": "2026-11-06",
            "p1": 0.71, "px": 0.17, "p2": 0.12, "games": 12,
            "verdict": {"ev": -0.50},
        }
        forward = rank_model_favorites([low, high])
        reverse = rank_model_favorites([high, low])
        self.assertEqual(
            [row["card"]["match"] for row in forward],
            ["High vs Other", "Low vs Other"],
        )
        self.assertEqual(
            [row["card"]["match"] for row in reverse],
            ["High vs Other", "Low vs Other"],
        )

    def test_probability_always_outranks_larger_sample(self):
        higher_probability = {
            "match": "Higher Probability vs Other", "p1": 0.66, "px": 0.20, "p2": 0.14,
            "games": 1,
        }
        lower_probability = {
            "match": "Lower Probability vs Other", "p1": 0.65, "px": 0.20, "p2": 0.15,
            "games": 12,
        }
        ranked = rank_model_favorites([lower_probability, higher_probability])
        self.assertEqual(ranked[0]["card"]["match"], "Higher Probability vs Other")

    def test_data_support_breaks_equal_probability_ranks(self):
        weak_sample = {
            "match": "Weak Sample vs Other", "p1": 0.66, "px": 0.20, "p2": 0.14,
            "games": 1,
        }
        strong_sample = {
            "match": "Strong Sample vs Other", "p1": 0.66, "px": 0.20, "p2": 0.14,
            "games": 12,
        }
        ranked = rank_model_favorites([weak_sample, strong_sample])
        self.assertEqual(ranked[0]["card"]["match"], "Strong Sample vs Other")
        self.assertEqual(ranked[0]["sample_games"], 12)

    def test_clearer_favorite_breaks_equal_probability_tie(self):
        narrow = {"match": "Narrow vs Other", "p1": 0.62, "px": 0.35, "p2": 0.03, "games": 12}
        clear = {"match": "Clear vs Other", "p1": 0.62, "px": 0.20, "p2": 0.18, "games": 12}
        ranked = rank_model_favorites([narrow, clear])
        self.assertEqual(ranked[0]["card"]["match"], "Clear vs Other")
        self.assertGreater(ranked[0]["probability_margin"], ranked[1]["probability_margin"])

    def test_duplicate_fixture_keeps_best_forecast_not_first_provider_row(self):
        first = {
            "match": "Alpha FC vs Beta FC", "date_iso": "2026-11-06",
            "league": "Soccer", "p1": 0.55, "px": 0.25, "p2": 0.20,
            "games": 5, "fixture_id": "tsdb-1",
        }
        better = {
            "match": "Alpha vs Beta", "date_iso": "2026-11-06",
            "league": "Serie A", "p1": 0.71, "px": 0.17, "p2": 0.12,
            "games": 12, "fixture_id": "sstats-2",
        }
        forward = rank_model_favorites([first, better])
        reverse = rank_model_favorites([better, first])
        self.assertEqual(len(forward), 1)
        self.assertEqual(len(reverse), 1)
        self.assertEqual(forward[0]["card"]["league"], "Serie A")
        self.assertEqual(reverse[0]["card"]["league"], "Serie A")

    def test_ranking_deduplicates_known_team_aliases_but_keeps_reserve_matches(self):
        senior_a = {
            "match": "Manchester United vs Brighton", "date_iso": "2026-11-06",
            "p1": 0.61, "px": 0.22, "p2": 0.17, "games": 12,
        }
        senior_b = {
            "match": "Man Utd vs Brighton & Hove Albion", "date_iso": "2026-11-06",
            "p1": 0.70, "px": 0.18, "p2": 0.12, "games": 12,
        }
        reserve = {
            "match": "Manchester United B vs Brighton", "date_iso": "2026-11-06",
            "p1": 0.66, "px": 0.19, "p2": 0.15, "games": 12,
        }
        ranked = rank_model_favorites([senior_a, senior_b, reserve])
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0]["card"]["match"], senior_b["match"])
        self.assertTrue(
            any(item["card"]["match"] == reserve["match"] for item in ranked)
        )

    def test_fixture_limit_is_applied_after_unique_probability_ranking(self):
        cards = [
            {"match": f"Team {i} vs Opponent", "date_iso": "2026-11-06",
             "p1": 0.50 + i * 0.02, "px": 0.30, "p2": 0.20 - i * 0.02,
             "games": 12}
            for i in range(6)
        ]
        ranked = rank_model_favorites(cards, limit=3)
        self.assertEqual(len(ranked), 3)
        self.assertEqual(ranked[0]["card"]["match"], "Team 5 vs Opponent")
        self.assertEqual(ranked[-1]["card"]["match"], "Team 3 vs Opponent")

    def test_does_not_mutate_original_fixture_cards(self):
        card = {"match": "A vs B", "p1": 0.65, "px": 0.2, "p2": 0.15}
        before = dict(card)
        rank_model_favorites([card])
        self.assertEqual(card, before)

    def test_probabilities_outside_tolerance_are_rejected(self):
        self.assertIsNone(model_favorite({"p1": 0.8, "px": 0.3, "p2": 0.2}))

    def test_minimum_seven_display_includes_lower_history_as_forecast_only(self):
        cards = [
            {
                "match": f"Team {i} vs Opponent {i}",
                "date_iso": f"2026-11-{i + 1:02d}",
                "p1": 0.50 + i * 0.01,
                "px": 0.30,
                "p2": 0.20 - i * 0.01,
                "games": 12 if i < 4 else 2,
            }
            for i in range(7)
        ]
        ranked = rank_model_favorites_with_minimum(
            cards, minimum=7, limit=10, min_sample_games=5,
        )
        self.assertEqual(len(ranked), 7)
        self.assertEqual(sum(item["history_sufficient"] for item in ranked), 4)
        self.assertEqual(sum(not item["history_sufficient"] for item in ranked), 3)
        self.assertTrue(all("card" in item for item in ranked))

    def test_minimum_seven_never_invents_missing_fixtures(self):
        cards = [
            {"match": f"Team {i} vs Opponent", "date_iso": "2026-11-01",
             "p1": 0.60, "px": 0.22, "p2": 0.18, "games": 12}
            for i in range(6)
        ]
        ranked = rank_model_favorites_with_minimum(
            cards, minimum=7, limit=10, min_sample_games=5,
        )
        self.assertEqual(len(ranked), 6)

    def test_minimum_floor_overrides_smaller_nominal_limit(self):
        cards = [
            {"match": f"Team {i} vs Opponent", "date_iso": "2026-11-01",
             "p1": 0.60, "px": 0.22, "p2": 0.18, "games": 12}
            for i in range(8)
        ]
        ranked = rank_model_favorites_with_minimum(
            cards, minimum=7, limit=5, min_sample_games=5,
        )
        self.assertEqual(len(ranked), 7)

    def test_minimum_display_keeps_probability_first_order(self):
        cards = [
            {"match": f"Team {i} vs Opponent", "date_iso": "2026-11-01",
             "p1": 0.50 + i * 0.02, "px": 0.30, "p2": 0.20 - i * 0.02,
             "games": 1 if i == 6 else 12}
            for i in range(7)
        ]
        ranked = rank_model_favorites_with_minimum(cards, minimum=7, limit=10)
        self.assertEqual(ranked[0]["card"]["match"], "Team 6 vs Opponent")
        self.assertFalse(ranked[0]["history_sufficient"])


    def test_venue_history_overrides_total_match_count(self):
        result = model_favorite({
            "home": "Home FC", "away": "Away FC",
            "p1": 0.66, "px": 0.21, "p2": 0.13,
            "games": 22,
            "independent_features": {
                "home_home_sample": 3,
                "away_away_sample": 8,
            },
        })
        self.assertEqual(result["home_sample"], 3)
        self.assertEqual(result["away_sample"], 8)
        self.assertEqual(result["sample_games"], 3)

    def _daily_card(
        self, home, away, kickoff, odds=None, *, division="E0",
        probabilities=(0.42, 0.29, 0.29), home_n=0, away_n=0, league_n=0,
        bookmaker="Book A",
    ):
        card = {
            "home": home, "away": away,
            "home_team": home, "away_team": away,
            "match": f"{home} vs {away}",
            "date": f"19.11 {kickoff}",
            "date_iso": "2026-11-19",
            "time": kickoff,
            "div": division,
            "league": division,
            "p1": probabilities[0],
            "px": probabilities[1],
            "p2": probabilities[2],
            "independent_features": {
                "home_home_sample": home_n,
                "away_away_sample": away_n,
            },
            "league_history_matches": league_n,
        }
        if odds:
            card["odds_source"] = "sstats"
            card["market_odds"] = odds
            card["bookmaker_by_pick"] = {
                "П1": bookmaker, "X": bookmaker, "П2": bookmaker,
            }
        return card

    def test_daily_ranking_prefers_no_vig_market_favourite_over_prior_only_model(self):
        tz = ZoneInfo("Europe/Paris")
        now = datetime(2026, 11, 19, 8, 0, tzinfo=tz)
        strong_market = self._daily_card(
            "FC Fredericia", "Hillerod", "19:00",
            {"П1": 1.33, "X": 5.69, "П2": 7.93},
        )
        balanced_market = self._daily_card(
            "Svikaita", "Hegelmann II", "10:00",
            {"П1": 2.42, "X": 3.60, "П2": 2.46},
        )
        diagnostics = {}
        ranked = rank_best_fixtures(
            [balanced_market, strong_market],
            now=now,
            period_end=datetime(2026, 11, 20, 0, 0, tzinfo=tz),
            limit=5,
            include_limited_history=True,
            diagnostics=diagnostics,
        )
        self.assertEqual(
            [item["card"]["match"] for item in ranked],
            ["FC Fredericia vs Hillerod", "Svikaita vs Hegelmann II"],
        )
        self.assertEqual(ranked[0]["pass_pick"], "П1")
        self.assertEqual(ranked[0]["pass_probability_source"], "market_no_vig")
        self.assertGreater(ranked[0]["pass_probability"], 0.65)
        self.assertAlmostEqual(ranked[0]["model_favourite_probability"], 0.42)
        self.assertEqual(diagnostics["market_unique_candidates"], 2)
        self.assertEqual(diagnostics["model_only_unique_candidates"], 0)

    def test_daily_ranking_can_select_draw_as_strongest_market_outcome(self):
        tz = ZoneInfo("Europe/Paris")
        now = datetime(2026, 11, 19, 8, 0, tzinfo=tz)
        draw_favourite = self._daily_card(
            "Draw FC", "Other FC", "19:00",
            {"П1": 4.20, "X": 2.05, "П2": 3.90},
        )
        ranked = rank_best_fixtures(
            [draw_favourite],
            now=now,
            period_end=datetime(2026, 11, 20, 0, 0, tzinfo=tz),
        )
        self.assertEqual(ranked[0]["pass_pick"], "X")
        self.assertGreater(
            ranked[0]["pass_probability"],
            ranked[0]["market_probabilities"]["П1"],
        )
        self.assertGreater(
            ranked[0]["pass_probability"],
            ranked[0]["market_probabilities"]["П2"],
        )

    def test_daily_ranking_can_select_away_win_as_strongest_market_outcome(self):
        tz = ZoneInfo("Europe/Paris")
        now = datetime(2026, 11, 19, 8, 0, tzinfo=tz)
        away_favourite = self._daily_card(
            "Home FC", "Away FC", "19:00",
            {"П1": 4.40, "X": 3.70, "П2": 1.72},
        )
        ranked = rank_best_fixtures(
            [away_favourite],
            now=now,
            period_end=datetime(2026, 11, 20, 0, 0, tzinfo=tz),
        )
        self.assertEqual(ranked[0]["pass_pick"], "П2")
        self.assertGreater(
            ranked[0]["pass_probability"],
            ranked[0]["market_probabilities"]["П1"],
        )
        self.assertGreater(
            ranked[0]["pass_probability"],
            ranked[0]["market_probabilities"]["X"],
        )

    def test_approved_pick_override_ranks_exact_selection_not_generic_favourite(self):
        tz = ZoneInfo("Europe/Paris")
        now = datetime(2026, 11, 19, 8, 0, tzinfo=tz)
        card = self._daily_card(
            "Away Value FC", "Other FC", "19:00",
            {"П1": 1.95, "X": 3.30, "П2": 4.10},
        )
        card["_daily_required_pick"] = "П2"
        ranked = rank_best_fixtures(
            [card],
            now=now,
            period_end=datetime(2026, 11, 20, 0, 0, tzinfo=tz),
        )
        self.assertEqual(ranked[0]["pass_pick"], "П2")
        self.assertAlmostEqual(
            ranked[0]["pass_probability"],
            ranked[0]["market_probabilities"]["П2"],
        )

    def test_daily_ranking_uses_market_probability_not_kickoff_time(self):
        tz = ZoneInfo("Europe/Paris")
        now = datetime(2026, 11, 19, 8, 0, tzinfo=tz)
        early_balanced = self._daily_card(
            "Early FC", "Other", "09:00",
            {"П1": 2.30, "X": 3.30, "П2": 2.90},
        )
        late_favourite = self._daily_card(
            "Late United", "Opponent", "20:00",
            {"П1": 1.40, "X": 4.80, "П2": 7.00},
        )
        ranked = rank_best_fixtures(
            [early_balanced, late_favourite],
            now=now,
            period_end=datetime(2026, 11, 20, 0, 0, tzinfo=tz),
        )
        self.assertEqual(ranked[0]["card"]["match"], "Late United vs Opponent")
        self.assertGreater(ranked[0]["kickoff"], ranked[1]["kickoff"])
        self.assertGreater(ranked[0]["pass_probability"], ranked[1]["pass_probability"])

    def test_model_only_event_requires_league_and_venue_specific_history(self):
        tz = ZoneInfo("Europe/Paris")
        now = datetime(2026, 11, 19, 8, 0, tzinfo=tz)
        supported = self._daily_card(
            "Supported FC", "Guest FC", "19:30", None,
            probabilities=(0.66, 0.20, 0.14),
            home_n=8, away_n=7, league_n=120,
        )
        prior_only = self._daily_card(
            "Unknown FC", "Other FC", "20:00", None,
            probabilities=(0.88, 0.08, 0.04),
            home_n=1, away_n=2, league_n=0,
        )
        ranked = rank_best_fixtures(
            [prior_only, supported],
            now=now,
            period_end=datetime(2026, 11, 20, 0, 0, tzinfo=tz),
            include_limited_history=True,
        )
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["card"]["match"], "Supported FC vs Guest FC")
        self.assertEqual(ranked[0]["pass_probability_source"], "model_supported")
        self.assertTrue(ranked[0]["model_supported"])

    def test_incoherent_best_odds_across_books_do_not_create_market_pass_probability(self):
        tz = ZoneInfo("Europe/Paris")
        now = datetime(2026, 11, 19, 8, 0, tzinfo=tz)
        card = self._daily_card(
            "Split Book FC", "Other FC", "19:00",
            {"П1": 1.40, "X": 4.80, "П2": 7.00},
        )
        card["bookmaker_by_pick"] = {
            "П1": "Book A", "X": "Book B", "П2": "Book C",
        }
        ranked = rank_best_fixtures(
            [card], now=now,
            period_end=datetime(2026, 11, 20, 0, 0, tzinfo=tz),
            include_limited_history=True,
        )
        self.assertEqual(ranked, [])

    def test_no_line_and_unsupported_prior_is_never_promoted_even_when_legacy_fallback_enabled(self):
        tz = ZoneInfo("Europe/Paris")
        now = datetime(2026, 11, 19, 8, 0, tzinfo=tz)
        card = self._daily_card(
            "No History FC", "Other FC", "19:00", None,
            probabilities=(0.87, 0.08, 0.05),
            home_n=0, away_n=0, league_n=0,
        )
        ranked = rank_best_fixtures(
            [card], now=now,
            period_end=datetime(2026, 11, 20, 0, 0, tzinfo=tz),
            include_limited_history=True,
        )
        self.assertEqual(ranked, [])

    def test_daily_ranking_keeps_all_market_events_tied_at_cutoff(self):
        tz = ZoneInfo("Europe/Paris")
        now = datetime(2026, 11, 19, 8, 0, tzinfo=tz)
        odds = {"П1": 1.80, "X": 3.50, "П2": 4.50}
        cards = [
            self._daily_card(
                f"Home {index}", f"Away {index}", f"{10 + index:02d}:00",
                odds,
            )
            for index in range(6)
        ]
        ranked = rank_best_fixtures(
            cards, now=now,
            period_end=datetime(2026, 11, 20, 0, 0, tzinfo=tz),
            limit=5,
        )
        self.assertEqual(len(ranked), 6)
        self.assertEqual(
            {item["card"]["match"] for item in ranked},
            {item["match"] for item in cards},
        )

    def test_daily_ranking_respects_local_day_and_excludes_started_events(self):
        tz = ZoneInfo("Europe/Paris")
        now = datetime(2026, 11, 19, 12, 0, tzinfo=tz)
        started = self._daily_card(
            "Started FC", "Other FC", "11:00",
            {"П1": 1.20, "X": 6.00, "П2": 12.0},
        )
        today = self._daily_card(
            "Today FC", "Other FC", "19:00",
            {"П1": 1.70, "X": 3.80, "П2": 5.20},
        )
        tomorrow = dict(today)
        tomorrow["date_iso"] = "2026-11-20"
        tomorrow["date"] = "20.11 08:00"
        tomorrow["time"] = "08:00"
        tomorrow["match"] = "Tomorrow FC vs Other FC"
        tomorrow["home_team"] = "Tomorrow FC"
        tomorrow["home"] = "Tomorrow FC"
        ranked = rank_best_fixtures(
            [started, tomorrow, today], now=now,
            period_end=datetime(2026, 11, 20, 0, 0, tzinfo=tz),
        )
        self.assertEqual([item["card"]["match"] for item in ranked], ["Today FC vs Other FC"])



if __name__ == "__main__":
    unittest.main()
