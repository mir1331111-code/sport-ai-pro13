import unittest

from model.market_models import (
    adjust_corners_for_game_state, asian_handicap_expected_value,
    asian_handicap_probabilities, negative_binomial_over_probability,
    referee_discipline_stats, split_asian_handicap,
)


class MarketModelTests(unittest.TestCase):
    def test_quarter_lines_split_to_adjacent_half_goal_lines(self):
        self.assertEqual(split_asian_handicap(-0.25), (-0.5, 0.0))
        self.assertEqual(split_asian_handicap(0.75), (0.5, 1.0))
        self.assertEqual(split_asian_handicap(-1.0), (-1.0,))

    def test_asian_handicap_returns_push_and_split_outcomes(self):
        matrix = [
            [0.10, 0.10, 0.05],
            [0.10, 0.20, 0.10],
            [0.05, 0.10, 0.20],
        ]
        result = asian_handicap_probabilities(matrix, 0.0, side="home")
        self.assertGreater(result["push"], 0.0)
        self.assertAlmostEqual(sum(result.values()), 1.0)
        quarter = asian_handicap_probabilities(matrix, -0.25, side="home")
        self.assertGreater(quarter["half_loss"] + quarter["half_win"], 0.0)
        self.assertAlmostEqual(sum(quarter.values()), 1.0)
        self.assertIsInstance(asian_handicap_expected_value(quarter, 1.90), float)

    def test_negative_binomial_handles_overdispersion(self):
        poisson = negative_binomial_over_probability(4.0, 4.0, 4.5)
        overdispersed = negative_binomial_over_probability(4.0, 10.0, 4.5)
        self.assertIsNotNone(poisson)
        self.assertIsNotNone(overdispersed)
        self.assertNotAlmostEqual(poisson, overdispersed)

    def test_referee_requires_minimum_sample_and_returns_variance(self):
        rows = [
            {"referee_id": "r1", "HY": 2, "AY": 3, "HR": 0, "AR": 0},
            {"referee_id": "r1", "HY": 1, "AY": 2, "HR": 0, "AR": 0},
            {"referee_id": "r1", "HY": 4, "AY": 2, "HR": 1, "AR": 0},
            {"referee_id": "r1", "HY": 3, "AY": 2, "HR": 0, "AR": 0},
        ]
        self.assertFalse(referee_discipline_stats(rows, "r1")["available"])
        rows.append({"referee_id": "r1", "HY": 1, "AY": 1, "HR": 0, "AR": 0})
        result = referee_discipline_stats(rows, "r1")
        self.assertTrue(result["available"])
        self.assertGreaterEqual(result["variance"], 0.0)

    def test_corners_game_state_adjusts_only_when_historical_split_exists(self):
        skipped = adjust_corners_for_game_state(
            10, state="home_trailing", historical_state_means={"level": 9.0}
        )
        self.assertFalse(skipped["adjusted"])
        result = adjust_corners_for_game_state(
            10, state="home_trailing",
            historical_state_means={"level": 9.0, "home_trailing": 11.0},
            opponent_style_factor=1.05,
        )
        self.assertTrue(result["adjusted"])
        self.assertAlmostEqual(result["mean"], 10 * 11 / 9 * 1.05)


if __name__ == "__main__":
    unittest.main()
