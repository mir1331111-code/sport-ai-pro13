import unittest

from betting.verdict import refine_with_real_odds


class MarketValueGateTests(unittest.TestCase):
    def _candidate(self, prob):
        return {
            "pick": "П1",
            "prob": prob,
            "is_action": True,
            "fair_odd": 1.0 / prob,
        }

    def test_manual_real_odds_must_pass_edge_and_ev(self):
        verdict, best = refine_with_real_odds(
            self._candidate(0.51), [{"pick": "П1", "prob": 0.51}],
            {"П1": 2.0}, bank=1000, kelly_frac=0.25,
            min_edge=0.03, min_ev=0.03,
        )
        self.assertIsNone(best)
        self.assertFalse(verdict["value_passed"])
        self.assertEqual(verdict["stake"], 0.0)

    def test_qualified_pick_uses_market_blended_probability_and_small_cap(self):
        verdict, best = refine_with_real_odds(
            self._candidate(0.70), [{"pick": "П1", "prob": 0.70}],
            {"П1": 2.0}, bank=1000, kelly_frac=0.25,
            min_edge=0.03, min_ev=0.03,
        )
        self.assertIsNotNone(best)
        self.assertAlmostEqual(verdict["prob"], 0.58)
        self.assertEqual(verdict["probability_method"], "market_blend_0.4_model_delta")
        self.assertLessEqual(verdict["stake"], 20.0)


if __name__ == "__main__":
    unittest.main()
