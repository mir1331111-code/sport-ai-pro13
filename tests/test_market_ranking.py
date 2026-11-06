import unittest

from betting.market_ranking import market_forecast_summary, rank_market_forecasts


def _card(name, *, p1, over, btts, corners, cards, games=20):
    return {
        "match": name,
        "match_ru": name.replace(" vs ", " — "),
        "date_iso": "2026-11-06",
        "p1": p1, "px": 0.20, "p2": 0.80 - p1,
        "over": over, "btts": btts,
        "lam_h": 1.4, "lam_a": 1.1, "games": games,
        "historical_stats": {
            "corners": {
                "total": corners, "total_variance": corners,
                "history_matches": 120, "home_team_games": 8, "away_team_games": 7,
            },
            "cards": {
                "total": cards, "total_variance": cards,
                "history_matches": 110, "home_team_games": 7, "away_team_games": 6,
            },
        },
    }


class IndependentMarketRankingTests(unittest.TestCase):
    def test_each_market_uses_its_own_probability_not_1x2(self):
        one_x_two_favourite = _card(
            "Team A vs Team B", p1=0.72, over=0.48, btts=0.51,
            corners=8.0, cards=3.0,
        )
        totals_favourite = _card(
            "Team C vs Team D", p1=0.44, over=0.82, btts=0.48,
            corners=10.2, cards=4.1,
        )
        corners_favourite = _card(
            "Team E vs Team F", p1=0.41, over=0.55, btts=0.54,
            corners=12.5, cards=3.1,
        )
        cards_favourite = _card(
            "Team G vs Team H", p1=0.43, over=0.52, btts=0.56,
            corners=8.2, cards=6.8,
        )
        pool = [
            one_x_two_favourite, totals_favourite,
            corners_favourite, cards_favourite,
        ]

        self.assertEqual(rank_market_forecasts(pool, "winner")[0]["card"], one_x_two_favourite)
        self.assertEqual(rank_market_forecasts(pool, "goals")[0]["card"], totals_favourite)
        self.assertEqual(rank_market_forecasts(pool, "corners")[0]["card"], corners_favourite)
        self.assertEqual(rank_market_forecasts(pool, "cards")[0]["card"], cards_favourite)

    def test_btts_ranking_uses_yes_no_probability(self):
        btts_yes = _card(
            "Yes FC vs No FC", p1=0.40, over=0.51, btts=0.86,
            corners=9.0, cards=4.0,
        )
        other = _card(
            "Other FC vs Side FC", p1=0.75, over=0.51, btts=0.58,
            corners=9.0, cards=4.0,
        )
        ranked = rank_market_forecasts([other, btts_yes], "btts")
        self.assertEqual(ranked[0]["card"], btts_yes)
        self.assertEqual(ranked[0]["pick"], "BTTS Да")
        self.assertAlmostEqual(ranked[0]["probability"], 0.86)

    def test_stat_markets_require_sufficient_market_specific_history(self):
        insufficient = _card(
            "Thin data vs Team", p1=0.90, over=0.9, btts=0.9,
            corners=12.0, cards=8.0,
        )
        insufficient["historical_stats"]["corners"]["history_matches"] = 29
        insufficient["historical_stats"]["cards"]["away_team_games"] = 3
        self.assertIsNone(market_forecast_summary(insufficient, "corners"))
        self.assertIsNone(market_forecast_summary(insufficient, "cards"))
        self.assertEqual(rank_market_forecasts([insufficient], "corners"), [])

    def test_strong_forecast_threshold_is_65_percent_and_inclusive(self):
        at_threshold = _card(
            "Threshold FC vs Team", p1=0.65, over=0.65, btts=0.65,
            corners=10.0, cards=5.0,
        )
        below_threshold = _card(
            "Below FC vs Team", p1=0.649, over=0.649, btts=0.649,
            corners=10.0, cards=5.0,
        )
        for family in ("winner", "goals", "btts"):
            with self.subTest(family=family):
                passed = market_forecast_summary(at_threshold, family)
                failed = market_forecast_summary(below_threshold, family)
                self.assertEqual(passed["strength_threshold"], 0.65)
                self.assertTrue(passed["strong_forecast"])
                self.assertFalse(failed["strong_forecast"])

    def test_balanced_ranking_uses_only_already_approved_value_metrics(self):
        high_probability = _card(
            "High probability vs Team", p1=0.78, over=0.60, btts=0.60,
            corners=10.0, cards=5.0,
        )
        unvalidated_value = _card(
            "Unvalidated price vs Team", p1=0.77, over=0.60, btts=0.60,
            corners=10.0, cards=5.0,
        )
        unvalidated_value["best_by_market"] = {
            "winner": {
                "verdict": {"value_passed": False, "ev": 0.50, "edge": 0.40},
                "best": ("1X2", "П1", 2.5, 0.5, 0.79, 1.0),
            }
        }
        validated_value = _card(
            "Validated value vs Team", p1=0.74, over=0.60, btts=0.60,
            corners=10.0, cards=5.0,
        )
        validated_value["best_by_market"] = {
            "winner": {
                "verdict": {
                    "value_passed": True, "ev": 0.10, "edge": 0.08, "prob": 0.75,
                },
                "best": ("1X2", "П1", 2.2, 0.10, 0.75, 10.0),
            }
        }
        ranked = rank_market_forecasts(
            [high_probability, unvalidated_value, validated_value],
            "winner", mode="balanced",
        )
        self.assertEqual(ranked[0]["card"], validated_value)
        self.assertEqual(ranked[1]["card"], high_probability)
        self.assertEqual(ranked[2]["card"], unvalidated_value)
        self.assertAlmostEqual(ranked[0]["ranking_score"], 0.844, places=6)

    def test_invalid_probabilities_are_not_ranked(self):
        bad = _card(
            "Bad FC vs Data FC", p1=0.95, over=1.2, btts=-0.2,
            corners=9.0, cards=4.0,
        )
        self.assertIsNone(market_forecast_summary(bad, "btts"))
        self.assertIsNone(market_forecast_summary(bad, "goals"))

    def test_zero_and_certain_probabilities_are_not_promoted(self):
        at_zero = _card(
            "Zero FC vs Team", p1=0.0, over=0.0, btts=0.0,
            corners=0.1, cards=0.1,
        )
        at_one = _card(
            "Certain FC vs Team", p1=1.0, over=1.0, btts=1.0,
            corners=50.0, cards=50.0,
        )
        for family in ("winner", "goals", "btts", "corners", "cards"):
            with self.subTest(family=family):
                self.assertIsNone(market_forecast_summary(at_zero, family))
                self.assertIsNone(market_forecast_summary(at_one, family))

    def test_limit_is_applied_after_market_specific_ranking(self):
        high = _card("High vs Low", p1=.6, over=.9, btts=.6, corners=12, cards=5)
        low = _card("Low vs High", p1=.6, over=.6, btts=.6, corners=9, cards=3)
        ranked = rank_market_forecasts([low, high], "goals", limit=1)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["card"], high)


if __name__ == "__main__":
    unittest.main()
