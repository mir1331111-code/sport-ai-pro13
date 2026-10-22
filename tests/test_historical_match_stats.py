import unittest

from model.historical_match_stats import HistoricalMatchStats, poisson_over_probability
from betting.kelly import market_type


class HistoricalMatchStatsTests(unittest.TestCase):
    def test_trains_only_from_observed_corners_and_cards(self):
        model = HistoricalMatchStats(min_team_games=1)
        self.assertTrue(model.learn_step({
            "HomeTeam": "Alpha", "AwayTeam": "Beta",
            "HC": "6", "AC": "4", "HY": "2", "AY": "1",
            "HR": "1", "AR": "0",
        }, "E0"))
        pred = model.predict("Alpha", "Beta", "E0")
        self.assertAlmostEqual(pred["corners"]["total"], 10.0)
        self.assertAlmostEqual(pred["cards"]["total"], 4.0)
        self.assertEqual(pred["corners"]["history_matches"], 1)

    def test_missing_statistics_do_not_become_zero(self):
        model = HistoricalMatchStats()
        self.assertFalse(model.learn_step({
            "HomeTeam": "Alpha", "AwayTeam": "Beta",
            "FTHG": "2", "FTAG": "0",
        }, "E0"))
        self.assertEqual(model.predict("Alpha", "Beta", "E0"), {})

    def test_poisson_over_probability_is_bounded_and_monotonic(self):
        p_low = poisson_over_probability(8.0, 9.5)
        p_high = poisson_over_probability(11.0, 9.5)
        self.assertGreaterEqual(p_low, 0.0)
        self.assertLessEqual(p_high, 1.0)
        self.assertGreater(p_high, p_low)

    def test_stat_markets_are_not_goal_score_settled(self):
        self.assertEqual(market_type("Угл ТБ 9.5"), "STAT")
        self.assertEqual(market_type("Карты ТМ 4.5"), "STAT")
        self.assertEqual(market_type("BTTS Да"), "BTTS")


if __name__ == "__main__":
    unittest.main()
