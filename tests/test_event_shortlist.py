import unittest

from betting.event_shortlist import (
    build_best_market_events,
    build_high_probability_market_events,
    build_market_coverage,
)


def _base_card():
    return {
        "fixture_id": "fixture-1",
        "match": "Home FC vs Away FC",
        "match_ru": "Хозяева — Гости",
        "league": "Test League",
        "date_iso": "2026-11-15",
        "time": "18:00",
        "p1": 0.70,
        "px": 0.16,
        "p2": 0.14,
        "over": 0.58,
        "lam_h": 1.6,
        "lam_a": 1.2,
        "btts": 0.67,
        "games": 16,
        "market_odds": {"П1": 1.90, "X": 3.50, "П2": 4.40},
        "total_market_odds": {"ТБ 2.5": 1.95, "ТМ 2.5": 1.90},
        "btts_market_odds": {"BTTS Да": 1.88, "BTTS Нет": 1.95},
        "stat_market_odds": {
            "Угл ТБ 8.5": 1.90, "Угл ТМ 8.5": 1.90,
            "Карты ТБ 3.5": 1.90, "Карты ТМ 3.5": 1.90,
        },
        "asian_handicap_market_odds": {"Ф1 -0.5": 2.00, "Ф2 +0.5": 1.90},
        "bookmaker_by_pick": {
            "П1": "Book A", "X": "Book A", "П2": "Book A",
            "ТБ 2.5": "Book B", "ТМ 2.5": "Book B",
            "BTTS Да": "Book C", "BTTS Нет": "Book C",
            "Угл ТБ 8.5": "Book D", "Угл ТМ 8.5": "Book D",
            "Карты ТБ 3.5": "Book D", "Карты ТМ 3.5": "Book D",
        },
        "historical_stats": {
            "corners": {
                "total": 10.4, "total_variance": 10.4,
                "history_matches": 140, "home_team_games": 8, "away_team_games": 7,
            },
            "cards": {
                "total": 4.7, "total_variance": 4.7,
                "history_matches": 130, "home_team_games": 7, "away_team_games": 7,
            },
        },
        "asian_handicap_forecasts": [
            {
                "pick": "Ф1 -0.5",
                "line": -0.5,
                "odds": 2.00,
                "probabilities": {
                    "win": 0.52, "half_win": 0.0, "push": 0.0,
                    "half_loss": 0.0, "loss": 0.48,
                },
            },
            {
                "pick": "Ф2 +0.5",
                "line": 0.5,
                "odds": 1.90,
                "probabilities": {
                    "win": 0.48, "half_win": 0.0, "push": 0.0,
                    "half_loss": 0.0, "loss": 0.52,
                },
            },
        ],
        "context_quality": {"strong_value_eligible": True},
        "best_by_market": {},
    }


class UnifiedMarketEventShortlistTests(unittest.TestCase):
    def test_high_probability_shortlist_includes_all_supported_markets_at_minimum_odds(self):
        card = _base_card()
        card["over"] = 0.78
        card["btts"] = 0.76
        card["p1"] = 0.70
        card["lam_h"] = 2.0
        card["lam_a"] = 1.2
        card["historical_stats"]["corners"].update({
            "total": 13.0, "total_variance": 8.0,
            "history_matches": 140, "home_team_games": 12, "away_team_games": 11,
        })
        card["historical_stats"]["cards"].update({
            "total": 7.0, "total_variance": 5.0,
            "history_matches": 130, "home_team_games": 12, "away_team_games": 11,
        })
        card["independent_features"] = {
            "home_home_sample": 12, "away_away_sample": 11,
        }
        card.update({
            "odds_source": "SStats",
            "total_odds_source": "SStats",
            "btts_odds_source": "SStats",
            "extra_market_source": "SStats",
        })
        card["bookmaker_by_pick"] = {
            "П1": "Book A",
            "ТБ 2.5": "Book B",
            "BTTS Да": "Book C",
            "Угл ТБ 8.5": "Book D",
            "Карты ТБ 3.5": "Book D",
        }
        events = build_high_probability_market_events([card], limit=30)
        families = {item["family"] for item in events}
        self.assertEqual(families, {"winner", "goals", "btts", "corners", "cards"})
        self.assertTrue(all(item["odds"] >= 1.70 for item in events))
        self.assertTrue(all(item["probability"] >= 0.65 for item in events))
        self.assertTrue(all(not item["value_verified"] for item in events))
        self.assertTrue(all("НЕ VALUE" in item["status_label"] for item in events))

    def test_high_probability_shortlist_excludes_low_odds_missing_bookmaker_or_unverified_source(self):
        card = _base_card()
        card["independent_features"] = {
            "home_home_sample": 12, "away_away_sample": 12,
        }
        card["odds_source"] = "SStats"
        card["bookmaker_by_pick"] = {"П1": "Book A"}
        card["market_odds"]["П1"] = 1.69
        events = build_high_probability_market_events([card])
        self.assertNotIn("winner", {item["family"] for item in events})

        card["market_odds"]["П1"] = 1.80
        card["odds_source"] = "model_only"
        events = build_high_probability_market_events([card])
        self.assertNotIn("winner", {item["family"] for item in events})

    def test_shortlist_includes_each_market_family_and_asian_handicap(self):
        card = _base_card()
        events = build_best_market_events([card], limit=30)
        families = {item["family"] for item in events}

        self.assertEqual(
            families,
            {"winner", "goals", "btts", "corners", "cards", "handicap"},
        )
        self.assertEqual(len(events), 6)

    def test_asian_handicap_is_research_only_and_uses_full_settlement_ev(self):
        card = _base_card()
        events = build_best_market_events([card], limit=30)
        handicap = next(item for item in events if item["family"] == "handicap")

        self.assertEqual(handicap["pick"], "Ф1 -0.5")
        self.assertAlmostEqual(handicap["probability"], 0.52)
        self.assertAlmostEqual(handicap["model_ev"], 0.04)
        self.assertEqual(handicap["status"], "handicap_research_only")
        self.assertIsNone(handicap["fair_odds"])
        self.assertFalse(handicap["value_verified"])
        self.assertIn("ИССЛЕДОВАТЕЛЬСКАЯ", handicap["status_label"])

    def test_quarter_handicap_ev_accounts_for_half_win_push_and_half_loss(self):
        card = _base_card()
        card["asian_handicap_forecasts"] = [{
            "pick": "Ф1 -0.25",
            "line": -0.25,
            "odds": 2.10,
            "probabilities": {
                "win": 0.40,
                "half_win": 0.10,
                "push": 0.10,
                "half_loss": 0.10,
                "loss": 0.30,
            },
        }]
        events = build_best_market_events([card], limit=30)
        handicap = next(item for item in events if item["family"] == "handicap")
        # 0.40*(2.10-1) + 0.10*0.5*(2.10-1) - 0.10*0.5 - 0.30
        self.assertAlmostEqual(handicap["model_ev"], 0.145)
        self.assertIn("полпобеды", handicap["outcome_detail"])

    def test_only_explicitly_passed_recommendation_is_value_verified(self):
        card = _base_card()
        card["best_by_market"] = {
            "winner": {
                "score": 0.81,
                "best": ("1X2", "П1", 1.90, 0.08, 0.74, 10.0),
                "verdict": {
                    "value_passed": True,
                    "prob": 0.74,
                    "model_prob": 0.70,
                    "ev": 0.08,
                    "edge": 0.06,
                    "kelly_pct": 0.01,
                },
            },
            # A recommendation-like object without a passing flag is not approved.
            "goals": {
                "score": 0.95,
                "best": ("OU", "ТБ 2.5", 1.95, 0.20, 0.80, 10.0),
                "verdict": {"value_passed": False, "ev": 0.20, "edge": 0.10},
            },
        }
        events = build_best_market_events([card], limit=30)
        winner = next(item for item in events if item["family"] == "winner")
        goals = next(item for item in events if item["family"] == "goals")

        self.assertTrue(winner["value_verified"])
        self.assertEqual(winner["status"], "value_verified")
        self.assertAlmostEqual(winner["probability"], 0.74)
        self.assertAlmostEqual(winner["edge"], 0.06)
        self.assertFalse(goals["value_verified"])
        self.assertEqual(goals["status"], "forecast_with_price")
        self.assertIsNone(goals["edge"])

    def test_approved_alternative_pick_is_kept_even_when_not_the_probability_favourite(self):
        card = _base_card()
        card["p1"], card["px"], card["p2"] = 0.58, 0.18, 0.24
        card["best_by_market"] = {
            "winner": {
                "score": 0.83,
                "best": ("1X2", "П2", 4.40, 0.12, 0.30, 4.0),
                "verdict": {
                    "value_passed": True,
                    "prob": 0.30,
                    "model_prob": 0.24,
                    "ev": 0.12,
                    "edge": 0.07,
                    "kelly_pct": 0.01,
                    "label": "Победа гостей (П2)",
                },
            }
        }
        events = build_best_market_events([card], limit=30)
        winner = next(item for item in events if item["family"] == "winner")
        self.assertEqual(winner["pick"], "П2")
        self.assertTrue(winner["value_verified"])
        self.assertEqual(winner["status"], "value_verified")
        self.assertAlmostEqual(winner["probability"], 0.30)

    def test_no_bookmaker_price_never_creates_value_or_fabricated_odds(self):
        card = _base_card()
        card["market_odds"] = {}
        card["total_market_odds"] = {}
        card["btts_market_odds"] = {}
        card["stat_market_odds"] = {}
        card["asian_handicap_market_odds"] = {}
        card["asian_handicap_forecasts"] = []

        events = build_best_market_events([card], limit=30)
        self.assertTrue(events)
        self.assertTrue(all(item["odds"] is None for item in events))
        self.assertTrue(all(not item["value_verified"] for item in events))
        self.assertTrue(all(item["model_ev"] is None for item in events))

    def test_asian_handicap_event_carries_sstats_source_and_explicit_probability_basis(self):
        card = _base_card()
        events = build_best_market_events([card], limit=30)
        handicap = next(item for item in events if item["family"] == "handicap")
        self.assertEqual(handicap["source"], "SStats")
        self.assertEqual(handicap["probability_basis"], "win + half-win")
        self.assertIsNone(handicap["fair_odds"])

    def test_market_coverage_counts_unique_matches_by_family(self):
        card = _base_card()
        coverage = {item["family"]: item for item in build_market_coverage([card])}

        self.assertEqual(coverage["winner"]["forecast_matches"], 1)
        self.assertEqual(coverage["winner"]["quoted_matches"], 1)
        self.assertEqual(coverage["winner"]["value_matches"], 0)
        self.assertEqual(coverage["goals"]["forecast_matches"], 1)
        self.assertEqual(coverage["goals"]["quoted_matches"], 1)
        self.assertEqual(coverage["handicap"]["forecast_matches"], 1)
        self.assertEqual(coverage["handicap"]["quoted_matches"], 1)
        self.assertEqual(coverage["handicap"]["value_matches"], 0)

    def test_value_shortlist_is_ranked_before_unvalidated_forecasts(self):
        card = _base_card()
        card["best_by_market"]["winner"] = {
            "score": 0.70,
            "best": ("1X2", "П1", 1.90, 0.08, 0.74, 10.0),
            "verdict": {
                "value_passed": True, "prob": 0.74,
                "model_prob": 0.70, "ev": 0.08,
                "edge": 0.06, "kelly_pct": 0.01,
            },
        }
        other = _base_card()
        other["fixture_id"] = "fixture-2"
        other["p1"] = 0.88
        other["px"] = 0.06
        other["p2"] = 0.06
        other["best_by_market"] = {}
        events = build_best_market_events([other, card], limit=30)
        self.assertEqual(events[0]["fixture_id"], "fixture-1")
        self.assertTrue(events[0]["value_verified"])

    def test_limit_is_applied_after_global_sort(self):
        cards = [_base_card() for _ in range(4)]
        for index, card in enumerate(cards):
            card["fixture_id"] = f"fixture-{index}"
        events = build_best_market_events(cards, limit=5)
        self.assertEqual(len(events), 5)
        self.assertEqual(
            [item["fixture_id"] for item in events[:5]],
            [item["fixture_id"] for item in build_best_market_events(cards, limit=100)[:5]],
        )


if __name__ == "__main__":
    unittest.main()
