import math
import unittest

from betting.best_picks import best_value_market
from betting.market_math import blend_probability, proportional_devig


class MarketMathTests(unittest.TestCase):
    def test_proportional_devig_removes_overround_and_normalizes(self):
        odds = {"П1": 2.10, "X": 3.50, "П2": 3.80}
        fair = proportional_devig(odds, ("П1", "X", "П2"))
        self.assertIsNotNone(fair)
        self.assertAlmostEqual(sum(fair.values()), 1.0, places=12)
        self.assertLess(fair["П1"], 1.0 / odds["П1"])
        self.assertTrue(all(0.0 < p < 1.0 for p in fair.values()))

    def test_partial_or_invalid_market_line_fails_closed(self):
        self.assertIsNone(proportional_devig({"П1": 2.1, "X": 3.4}, ("П1", "X", "П2")))
        self.assertIsNone(proportional_devig({"П1": 2.1, "X": 1.0, "П2": 3.4}, ("П1", "X", "П2")))
        self.assertIsNone(proportional_devig({"П1": float("nan"), "X": 3.4, "П2": 3.4}, ("П1", "X", "П2")))

    def test_blend_moves_model_probability_toward_market(self):
        self.assertAlmostEqual(blend_probability(0.70, 0.50, 0.40), 0.58)
        self.assertAlmostEqual(blend_probability(0.30, 0.50, 0.40), 0.42)

    def test_value_screen_uses_no_vig_benchmark_for_complete_line(self):
        verdict = {
            "pick": "П1", "label": "Победа хозяев", "prob": 0.60,
            "fair_odd": 1 / 0.60, "is_action": True, "confidence": "хорошая",
            "confidence_score": 70, "confidence_label": "B", "reasons": [],
        }
        rows = [
            {"pick": "П1", "label": "Победа хозяев", "prob": 0.60, "fair_odd": 1 / 0.60},
            {"pick": "X", "label": "Ничья", "prob": 0.25, "fair_odd": 4.0},
            {"pick": "П2", "label": "Победа гостей", "prob": 0.15, "fair_odd": 1 / 0.15},
        ]
        odds = {"П1": 2.10, "X": 3.50, "П2": 3.80}
        chosen, best = best_value_market(
            verdict, rows, odds, bank=1000, kelly_frac=0.25, min_prob=0.50
        )
        self.assertIsNotNone(best)
        self.assertEqual(chosen["market_fair_prob_source"], "complete_market_proportional_devig")
        fair = proportional_devig(odds, ("П1", "X", "П2"))["П1"]
        self.assertAlmostEqual(chosen["market_fair_prob"], fair)
        self.assertAlmostEqual(chosen["prob"], fair + 0.40 * (0.60 - fair))
        self.assertAlmostEqual(chosen["ev"], chosen["prob"] * 2.10 - 1.0)
        self.assertAlmostEqual(chosen["edge"], chosen["prob"] - fair)
        self.assertTrue(math.isfinite(chosen["stake"]))


if __name__ == "__main__":
    unittest.main()
