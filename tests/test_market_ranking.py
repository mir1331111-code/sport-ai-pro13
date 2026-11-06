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

    def test_invalid_probabilities_are_not_ranked(self):
        bad = _card(
            "Bad FC vs Data FC", p1=0.95, over=1.2, btts=-0.2,
            corners=9.0, cards=4.0,
        )
        self.assertIsNone(market_forecast_summary(bad, "btts"))
        self.assertIsNone(market_forecast_summary(bad, "goals"))

    def test_limit_is_applied_after_market_specific_ranking(self):
        high = _card("High vs Low", p1=.6, over=.9, btts=.6, corners=12, cards=5)
        low = _card("Low vs High", p1=.6, over=.6, btts=.6, corners=9, cards=3)
        ranked = rank_market_forecasts([low, high], "goals", limit=1)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["card"], high)


if __name__ == "__main__":
    unittest.main()
