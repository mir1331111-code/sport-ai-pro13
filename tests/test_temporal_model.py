from datetime import date, timedelta
import math
import unittest

from model.engine import Engine
from model.temporal import days_since_last_match, estimate_dixon_coles_rho, weighted_mean


class TemporalModelTests(unittest.TestCase):
    def test_weighted_mean_emphasizes_recent_observations(self):
        result = weighted_mean([0.0, 10.0], ["2020-01-01", "2025-01-01"], "2025-01-02", default=5.0, half_life_days=30)
        self.assertGreater(result, 9.9)

    def test_weighted_mean_excludes_future_observations(self):
        result = weighted_mean([2.0, 100.0], ["2025-01-01", "2025-02-01"], "2025-01-15")
        self.assertAlmostEqual(result, 2.0)

    def test_weighted_mean_falls_back_when_dates_are_missing(self):
        result = weighted_mean([1.0, 3.0], [None, "2025-01-01"], "2025-01-02")
        self.assertAlmostEqual(result, 2.0)

    def test_rest_days_are_safe_for_missing_and_future_dates(self):
        self.assertEqual(days_since_last_match("2025-01-01", "2025-01-08"), 7)
        self.assertIsNone(days_since_last_match(None, "2025-01-08"))
        self.assertIsNone(days_since_last_match("2025-01-09", "2025-01-08"))

    def test_rho_uses_prior_until_minimum_history_and_stays_bounded(self):
        history = [("2025-01-01", 0, 0)] * 20
        self.assertAlmostEqual(estimate_dixon_coles_rho(history, 1.4, 1.1, prior_rho=-0.13, min_matches=80), -0.13)
        history = [((date(2023, 1, 1) + timedelta(days=i)).isoformat(), i % 3, (i // 2) % 3) for i in range(80)]
        rho = estimate_dixon_coles_rho(history, 1.45, 1.15, prior_rho=-0.13, min_matches=80, as_of="2025-01-01")
        self.assertGreaterEqual(rho, -0.25)
        self.assertLessEqual(rho, 0.25)

    def test_engine_prediction_is_normalized_and_exposes_rest_and_rho(self):
        engine = Engine()
        for i in range(24):
            stamp = (date(2024, 1, 1) + timedelta(days=7 * i)).isoformat()
            engine.add("Home", "Away", i % 3, (i + 1) % 3, row={"Div": "E0"}, match_date=stamp)
        forecast = engine.predict("Home", "Away", "E0", match_date="2025-01-01")
        self.assertTrue(math.isclose(forecast["p1"] + forecast["x"] + forecast["p2"], 1.0, abs_tol=1e-9))
        self.assertAlmostEqual(forecast["rho"], -0.13)  # fewer than 80 eligible league matches
        self.assertIsNotNone(forecast["home_rest_days"])
        self.assertIsNotNone(forecast["away_rest_days"])


if __name__ == "__main__":
    unittest.main()
