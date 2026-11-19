import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from betting.ranking import (
    favorite_display, model_favorite, rank_best_fixtures,
    rank_model_favorites, rank_model_favorites_with_minimum,
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

    def test_prior_only_favorite_is_not_presented_as_a_match_pick(self):
        ranked = rank_model_favorites_with_minimum(
            [{
                "match": "Unknown Home vs Unknown Away",
                "date_iso": "2026-11-19",
                "p1": 0.421, "px": 0.289, "p2": 0.289,
                "independent_features": {
                    "home_home_sample": 0,
                    "away_away_sample": 0,
                },
            }],
            minimum=1, limit=1, min_sample_games=5,
        )
        presentation = favorite_display(ranked[0])
        self.assertIsNone(presentation["pick"])
        self.assertIsNone(presentation["probability"])
        self.assertFalse(presentation["history_sufficient"])
        self.assertIn("PRIOR-ONLY", presentation["label"])

    def test_supported_favorite_keeps_its_p1_x_or_p2_label(self):
        for probabilities, expected in (
            ({"p1": 0.62, "px": 0.20, "p2": 0.18}, "Победа Alpha"),
            ({"p1": 0.20, "px": 0.62, "p2": 0.18}, "Ничья"),
            ({"p1": 0.18, "px": 0.20, "p2": 0.62}, "Победа Beta"),
        ):
            item = {
                **model_favorite({
                    "home": "Alpha", "away": "Beta", **probabilities,
                    "probability_calibrated": True,
                    "calibration_matches": 150,
                    "independent_features": {
                        "home_home_sample": 12,
                        "away_away_sample": 12,
                    },
                }),
                "history_sufficient": True,
                "history_floor": 10,
            }
            self.assertEqual(favorite_display(item)["label"], expected)
            self.assertIsNotNone(favorite_display(item)["pick"])

    def test_eight_match_history_is_not_enough_for_a_strong_pick(self):
        ranked = rank_model_favorites_with_minimum(
            [{
                "home": "Alpha", "away": "Beta",
                "p1": 0.929, "px": 0.040, "p2": 0.031,
                "probability_calibrated": True,
                "calibration_matches": 150,
                "independent_features": {
                    "home_home_sample": 8,
                    "away_away_sample": 8,
                },
            }],
            minimum=1, limit=1, min_sample_games=10,
        )
        presentation = favorite_display(ranked[0])
        self.assertFalse(ranked[0]["history_sufficient"])
        self.assertIsNone(presentation["pick"])
        self.assertIn("ОГРАНИЧЕННАЯ ИСТОРИЯ · 8/10", presentation["label"])

    def test_high_probability_without_league_calibration_is_not_a_strong_pick(self):
        ranked = rank_model_favorites_with_minimum(
            [{
                "home": "Alpha", "away": "Beta",
                "p1": 0.929, "px": 0.040, "p2": 0.031,
                "probability_calibrated": False,
                "calibration_matches": 72,
                "independent_features": {
                    "home_home_sample": 12,
                    "away_away_sample": 12,
                },
            }],
            minimum=1, limit=1, min_sample_games=10,
        )
        presentation = favorite_display(ranked[0])
        self.assertTrue(ranked[0]["history_sufficient"])
        self.assertFalse(ranked[0]["calibration_ready"])
        self.assertIsNone(presentation["pick"])
        self.assertIn("НЕ КАЛИБРОВАНО", presentation["label"])
        self.assertIn("72 матчей", presentation["label"])

    def test_calibrated_high_probability_with_ten_samples_is_displayable(self):
        ranked = rank_model_favorites_with_minimum(
            [{
                "home": "Alpha", "away": "Beta",
                "p1": 0.70, "px": 0.18, "p2": 0.12,
                "probability_calibrated": True,
                "calibration_matches": 150,
                "independent_features": {
                    "home_home_sample": 10,
                    "away_away_sample": 11,
                },
            }],
            minimum=1, limit=1, min_sample_games=10,
        )
        presentation = favorite_display(ranked[0])
        self.assertTrue(presentation["calibrated"])
        self.assertEqual(presentation["pick"], "П1")
        self.assertAlmostEqual(presentation["probability"], 0.70)

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

    def test_daily_shortlist_ranks_best_forecasts_across_the_whole_day(self):
        tz = ZoneInfo("Europe/Warsaw")
        now = datetime(2026, 11, 18, 8, 0, tzinfo=tz)
        next_day = datetime(2026, 11, 19, 0, 0, tzinfo=tz)
        def card(home, away, kickoff, probability, home_n=7, away_n=7, games=18):
            return {
                "home": home, "away": away,
                "match": f"{home} vs {away}",
                "date": f"18.11 {kickoff}",
                "date_iso": "2026-11-18",
                "time": kickoff,
                "p1": probability,
                "px": 0.20,
                "p2": 0.80 - probability,
                "games": games,
                "independent_features": {
                    "home_home_sample": home_n,
                    "away_away_sample": away_n,
                },
            }

        early = card("Early FC", "Side FC", "09:00", 0.61)
        late_best = card("Late United", "Opponent FC", "18:30", 0.74)
        weak_history = card("Unknown FC", "Other FC", "19:00", 0.79,
                            home_n=0, away_n=0, games=30)
        low_probability = card("Even FC", "Other Side", "20:00", 0.50)
        legacy_without_venue_samples = card(
            "Legacy FC", "Other FC", "21:00", 0.88,
        )
        legacy_without_venue_samples.pop("independent_features")
        yesterday = {
            **card("Past FC", "Other FC", "23:00", 0.90),
            "date": "17.11 23:00",
            "date_iso": "2026-11-17",
        }
        tomorrow = {
            **card("Tomorrow FC", "Other FC", "01:00", 0.95),
            "date": "19.11 01:00",
            "date_iso": "2026-11-19",
        }

        ranked = rank_best_fixtures(
            [
                early, late_best, weak_history, low_probability,
                legacy_without_venue_samples, yesterday, tomorrow,
            ],
            now=now, period_end=next_day, limit=None,
            min_sample_games=5, min_probability=0.52,
        )
        self.assertEqual(
            [item["card"]["match"] for item in ranked],
            ["Late United vs Opponent FC", "Early FC vs Side FC"],
        )
        self.assertGreater(ranked[0]["kickoff"], ranked[1]["kickoff"])

    def test_daily_shortlist_keeps_limited_history_as_provisional(self):
        tz = ZoneInfo("Europe/Warsaw")
        now = datetime(2026, 11, 19, 9, 0, tzinfo=tz)
        card = {
            "home": "Unknown Home", "away": "Unknown Away",
            "match": "Unknown Home vs Unknown Away",
            "date_iso": "2026-11-19", "time": "19:30",
            "p1": 0.44, "px": 0.30, "p2": 0.26,
            "games": 25,
            "independent_features": {
                "home_home_sample": 0,
                "away_away_sample": 1,
            },
        }
        diagnostics = {}
        ranked = rank_best_fixtures(
            [card], now=now, period_end=datetime(2026, 11, 20, 0, 0, tzinfo=tz),
            include_limited_history=True, diagnostics=diagnostics,
        )
        self.assertEqual(len(ranked), 1)
        self.assertFalse(ranked[0]["strict_eligible"])
        self.assertEqual(ranked[0]["home_sample"], 0)
        self.assertEqual(ranked[0]["away_sample"], 1)
        self.assertEqual(diagnostics["strict_unique"], 0)
        self.assertEqual(diagnostics["provisional_unique"], 1)

    def test_daily_shortlist_puts_supported_events_ahead_of_earlier_provisional_ones(self):
        tz = ZoneInfo("Europe/Warsaw")
        now = datetime(2026, 11, 19, 8, 0, tzinfo=tz)
        provisional = {
            "home": "Provisional FC", "away": "Guest FC",
            "match": "Provisional FC vs Guest FC",
            "date_iso": "2026-11-19", "time": "09:00",
            "p1": 0.85, "px": 0.10, "p2": 0.05,
            "independent_features": {
                "home_home_sample": 1, "away_away_sample": 2,
            },
        }
        supported = {
            "home": "Supported FC", "away": "Guest FC",
            "match": "Supported FC vs Guest FC",
            "date_iso": "2026-11-19", "time": "20:00",
            "p1": 0.62, "px": 0.22, "p2": 0.16,
            "independent_features": {
                "home_home_sample": 7, "away_away_sample": 8,
            },
        }
        ranked = rank_best_fixtures(
            [provisional, supported], now=now,
            period_end=datetime(2026, 11, 20, 0, 0, tzinfo=tz),
            include_limited_history=True,
        )
        self.assertEqual(ranked[0]["card"]["match"], "Supported FC vs Guest FC")
        self.assertTrue(ranked[0]["strict_eligible"])
        self.assertEqual(ranked[1]["card"]["match"], "Provisional FC vs Guest FC")
        self.assertFalse(ranked[1]["strict_eligible"])

    def test_daily_shortlist_quality_beats_earlier_kickoff(self):
        tz = ZoneInfo("Europe/Warsaw")
        now = datetime(2026, 11, 19, 8, 0, tzinfo=tz)
        weak_early = {
            "home": "Early FC", "away": "Guest FC",
            "match": "Early FC vs Guest FC",
            "date_iso": "2026-11-19", "time": "09:00",
            "p1": 0.62, "px": 0.22, "p2": 0.16,
            "independent_features": {
                "home_home_sample": 5, "away_away_sample": 5,
            },
        }
        strong_late = {
            "home": "Late FC", "away": "Guest FC",
            "match": "Late FC vs Guest FC",
            "date_iso": "2026-11-19", "time": "20:00",
            "p1": 0.62, "px": 0.22, "p2": 0.16,
            "independent_features": {
                "home_home_sample": 12, "away_away_sample": 12,
            },
            "xg_used": True,
            "xg_modelled_used": False,
            "xg_source": "provider",
            "market_odds": {"П1": 1.90, "X": 3.40, "П2": 4.20},
        }
        ranked = rank_best_fixtures(
            [weak_early, strong_late],
            now=now,
            period_end=datetime(2026, 11, 20, 0, 0, tzinfo=tz),
            limit=5,
        )
        self.assertEqual(ranked[0]["card"]["match"], "Late FC vs Guest FC")
        self.assertGreater(ranked[0]["quality_score"], ranked[1]["quality_score"])
        self.assertEqual(ranked[0]["quality_components"]["venue_history"], 1.0)
        self.assertTrue(ranked[0]["has_provider_xg"])
        self.assertTrue(ranked[0]["has_complete_market_line"])

    def test_daily_shortlist_keeps_all_events_tied_at_display_cutoff(self):
        tz = ZoneInfo("Europe/Warsaw")
        now = datetime(2026, 11, 19, 8, 0, tzinfo=tz)
        cards = [
            {
                "home": f"Home {i}", "away": f"Away {i}",
                "match": f"Home {i} vs Away {i}",
                "date_iso": "2026-11-19", "time": f"{10 + i:02d}:00",
                "p1": 0.60, "px": 0.24, "p2": 0.16,
                "independent_features": {
                    "home_home_sample": 7, "away_away_sample": 7,
                },
            }
            for i in range(6)
        ]
        ranked = rank_best_fixtures(
            cards,
            now=now,
            period_end=datetime(2026, 11, 20, 0, 0, tzinfo=tz),
            limit=5,
        )
        self.assertEqual(len(ranked), 6)
        self.assertEqual(
            {item["card"]["match"] for item in ranked},
            {card["match"] for card in cards},
        )

    def test_daily_shortlist_respects_requested_window_and_limit(self):
        tz = ZoneInfo("Europe/Warsaw")
        now = datetime(2026, 11, 18, 8, 0, tzinfo=tz)
        cards = [
            {
                "home": f"Home {i}", "away": f"Away {i}",
                "match": f"Home {i} vs Away {i}",
                "date_iso": "2026-11-18", "time": f"{10 + i:02d}:00",
                "p1": 0.60 + i * 0.02, "px": 0.20, "p2": 0.20 - i * 0.02,
                "independent_features": {
                    "home_home_sample": 6, "away_away_sample": 6,
                },
            }
            for i in range(5)
        ]
        end = datetime(2026, 11, 18, 12, 0, tzinfo=tz)
        ranked = rank_best_fixtures(
            cards, now=now, period_end=end, limit=2,
        )
        self.assertEqual(len(ranked), 2)
        self.assertTrue(all(item["kickoff"] < end for item in ranked))
        self.assertEqual(ranked[0]["card"]["match"], "Home 1 vs Away 1")


if __name__ == "__main__":
    unittest.main()
