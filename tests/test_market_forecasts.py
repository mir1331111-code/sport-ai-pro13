import unittest

from model.market_forecasts import count_over_probability, goal_total_probabilities, is_half_line


class MarketForecastTests(unittest.TestCase):
    def test_negative_binomial_handles_overdispersed_corner_counts(self):
        poisson = count_over_probability(10.0, 12.5, variance=10.0)
        negative_binomial = count_over_probability(10.0, 12.5, variance=24.0)
        self.assertIsNotNone(poisson)
        self.assertIsNotNone(negative_binomial)
        self.assertGreater(negative_binomial, poisson)

    def test_count_probability_is_monotone_across_lines(self):
        values = [count_over_probability(10.0, line, variance=18.0)
                  for line in (7.5, 8.5, 9.5, 10.5, 11.5)]
        self.assertTrue(all(0.0 <= value <= 1.0 for value in values))
        self.assertTrue(all(values[i] >= values[i + 1] for i in range(len(values) - 1)))

    def test_goal_market_baseline_returns_totals_and_btts(self):
        result = goal_total_probabilities(1.5, 1.0)
        self.assertAlmostEqual(result["expected_total_goals"], 2.5)
        self.assertGreater(result["over_1_5"], result["over_2_5"])
        self.assertGreater(result["over_2_5"], result["over_3_5"])
        self.assertGreater(result["btts_yes"], 0.0)
        self.assertLess(result["btts_yes"], 1.0)

    def test_only_half_lines_are_supported_for_binary_betting_ev(self):
        for line in (0.5, 1.5, 2.5, 9.5, 10.5):
            self.assertTrue(is_half_line(line))
        for line in (1.0, 2.0, 8.0, 9.25, 10.75, float("nan"), None):
            self.assertFalse(is_half_line(line))

    def test_invalid_counts_are_rejected(self):
        self.assertIsNone(count_over_probability(0, 9.5))
        self.assertIsNone(count_over_probability(float("nan"), 9.5))


if __name__ == "__main__":
    unittest.main()
