import unittest

from betting.best_picks import best_value_market


def _verdict():
    return {
        "pick": "П1", "label": "Победа хозяев", "prob": 0.60,
        "fair_odd": 1 / 0.60, "is_action": True, "confidence": "хорошая",
        "confidence_score": 70, "confidence_label": "B", "reasons": [],
    }


def _rows():
    return [
        {"pick": "П1", "label": "Победа хозяев", "prob": 0.60, "fair_odd": 1 / 0.60},
        {"pick": "X", "label": "Ничья", "prob": 0.25, "fair_odd": 4.0},
        {"pick": "П2", "label": "Победа гостей", "prob": 0.15, "fair_odd": 1 / 0.15},
        {"pick": "ТБ 2.5", "label": "Тотал Больше 2.5", "prob": 0.60, "fair_odd": 1 / 0.60},
    ]


class BestValueMarketTests(unittest.TestCase):
    def test_selects_1x2_market_with_value_after_market_blend(self):
        # Модель 0.60, цена 2.10 (рынок 0.476): смесь 0.4 даёт ~0.526, edge ~5%, EV ~10%.
        chosen, best = best_value_market(
            _verdict(), _rows(), {"П1": 2.10, "ТБ 2.5": 1.90},
            bank=1000, kelly_frac=0.25, min_prob=0.50,
        )
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen["pick"], "П1")
        self.assertTrue(chosen["real_odds"])
        self.assertTrue(chosen["value_passed"])
        self.assertEqual(chosen["probability_method"], "market_blend_0.4_model_delta")
        self.assertEqual(best[1], "П1")
        self.assertEqual(best[2], 2.10)

    def test_confidence_uses_market_blended_probability_not_raw_model_probability(self):
        chosen, best = best_value_market(
            _verdict(), _rows(), {"П1": 2.10},
            bank=1000, kelly_frac=0.25, min_prob=0.50,
        )
        self.assertIsNotNone(best)
        self.assertAlmostEqual(chosen["confidence_score"], chosen["prob"] * 100, places=1)
        self.assertEqual(chosen["confidence_basis"],
                         "market_blended_probability_not_historical_hit_rate")
        self.assertNotEqual(chosen["confidence"], "высокая")

    def test_non_1x2_markets_never_become_automatic_candidates(self):
        chosen, best = best_value_market(
            _verdict(), _rows(), {"ТБ 2.5": 1.90},
            bank=1000, kelly_frac=0.25, min_prob=0.50,
        )
        self.assertIsNone(chosen)
        self.assertIsNone(best)

    def test_never_creates_bet_from_fair_odds_or_missing_market(self):
        chosen, best = best_value_market(
            _verdict(), _rows(), None, bank=1000, kelly_frac=0.25, min_prob=0.50,
        )
        self.assertIsNone(chosen)
        self.assertIsNone(best)

    def test_rejects_price_without_edge_and_ev_thresholds(self):
        chosen, best = best_value_market(
            _verdict(), _rows(), {"П1": 1.60},
            bank=1000, kelly_frac=0.25, min_prob=0.50,
        )
        self.assertIsNone(chosen)
        self.assertIsNone(best)

    def test_probability_threshold_is_applied_to_blended_probability(self):
        chosen, best = best_value_market(
            _verdict(), _rows(), {"П1": 2.10},
            bank=1000, kelly_frac=0.25, min_prob=0.60,
        )
        self.assertIsNone(chosen)
        self.assertIsNone(best)


if __name__ == "__main__":
    unittest.main()
