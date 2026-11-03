import unittest

from betting.best_picks import best_value_market, best_value_markets, _paired_picks
from betting.kelly import market_type


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

    def test_market_type_recognizes_all_goal_total_lines(self):
        for pick in ("ТБ 0.5", "ТМ 1.5", "ТБ 2.5", "ТМ 3.5", "ТБ 4.5", "ТМ 5.5"):
            with self.subTest(pick=pick):
                self.assertEqual(market_type(pick), "OU")

    def test_pairs_all_supported_markets_by_exact_line(self):
        self.assertEqual(_paired_picks("ТБ 2.5"), ("ТБ 2.5", "ТМ 2.5"))
        self.assertEqual(_paired_picks("BTTS Да"), ("BTTS Да", "BTTS Нет"))
        self.assertEqual(_paired_picks("Угл ТБ 9.5"), ("Угл ТБ 9.5", "Угл ТМ 9.5"))
        self.assertEqual(_paired_picks("Карты ТМ 4.5"), ("Карты ТБ 4.5", "Карты ТМ 4.5"))

    def test_selects_validated_total_market_with_same_bookmaker_pair(self):
        rows = [{"pick": "ТБ 2.5", "label": "Тотал больше 2.5", "prob": 0.72}]
        odds = {"ТБ 2.5": 1.95, "ТМ 2.5": 1.95}
        books = {"ТБ 2.5": "Book A", "ТМ 2.5": "Book A"}
        chosen, best = best_value_market(
            _verdict(), rows, odds, bank=1000, kelly_frac=0.25,
            min_prob=0.50, min_edge=0.05, min_ev=0.05,
            bookmaker_by_pick=books,
            market_validation_by_pick={"ТБ 2.5": True},
        )
        self.assertIsNotNone(chosen)
        self.assertEqual(best[0], "OU")
        self.assertEqual(best[1], "ТБ 2.5")
        self.assertEqual(best[2], 1.95)

    def test_rejects_unvalidated_or_mixed_bookmaker_two_way_markets(self):
        rows = [{"pick": "BTTS Да", "label": "Обе забьют — да", "prob": 0.75}]
        odds = {"BTTS Да": 1.95, "BTTS Нет": 1.95}
        mixed = {"BTTS Да": "Book A", "BTTS Нет": "Book B"}
        no_validation = best_value_market(
            _verdict(), rows, odds, bank=1000, kelly_frac=0.25,
            min_prob=0.50, bookmaker_by_pick={"BTTS Да": "Book A", "BTTS Нет": "Book A"},
        )
        mixed_books = best_value_market(
            _verdict(), rows, odds, bank=1000, kelly_frac=0.25,
            min_prob=0.50, bookmaker_by_pick=mixed,
            market_validation_by_pick={"BTTS Да": True},
        )
        self.assertEqual(no_validation, (None, None))
        self.assertEqual(mixed_books, (None, None))

    def test_non_1x2_markets_never_become_candidates_without_validation(self):
        chosen, best = best_value_market(
            _verdict(), _rows(), {"ТБ 2.5": 1.90},
            bank=1000, kelly_frac=0.25, min_prob=0.50,
        )
        self.assertIsNone(chosen)
        self.assertIsNone(best)

    def test_best_picks_are_selected_independently_for_each_market_family(self):
        rows = [
            {"pick": "П1", "prob": 0.90, "label": "П1"},
            {"pick": "ТБ 2.5", "prob": 0.90, "label": "ТБ 2.5"},
            {"pick": "BTTS Да", "prob": 0.90, "label": "BTTS Да"},
            {"pick": "Угл ТБ 9.5", "prob": 0.90, "label": "Угл ТБ 9.5"},
            {"pick": "Карты ТБ 4.5", "prob": 0.90, "label": "Карты ТБ 4.5"},
        ]
        odds = {
            "П1": 2.20,
            "ТБ 2.5": 2.20, "ТМ 2.5": 2.20,
            "BTTS Да": 2.20, "BTTS Нет": 2.20,
            "Угл ТБ 9.5": 2.20, "Угл ТМ 9.5": 2.20,
            "Карты ТБ 4.5": 2.20, "Карты ТМ 4.5": 2.20,
        }
        books = {
            key: "Book A" for key in odds
        }
        validation = {row["pick"]: True for row in rows if row["pick"] != "П1"}

        result = best_value_markets(
            _verdict(), rows, odds, bank=1000, kelly_frac=0.25,
            min_prob=0.50, bookmaker_by_pick=books,
            market_validation_by_pick=validation,
        )

        self.assertEqual(
            set(result),
            {"winner", "goals", "btts", "corners", "cards"},
        )
        self.assertEqual(result["winner"]["pick"], "П1")
        self.assertEqual(result["goals"]["pick"], "ТБ 2.5")
        self.assertEqual(result["btts"]["pick"], "BTTS Да")
        self.assertEqual(result["corners"]["pick"], "Угл ТБ 9.5")
        self.assertEqual(result["cards"]["pick"], "Карты ТБ 4.5")

    def test_missing_or_mixed_price_pair_rejects_only_its_own_market(self):
        rows = [
            {"pick": "П1", "prob": 0.90},
            {"pick": "BTTS Да", "prob": 0.90},
            {"pick": "Угл ТБ 9.5", "prob": 0.90},
        ]
        odds = {
            "П1": 2.20,
            "BTTS Да": 2.20, "BTTS Нет": 2.20,
            "Угл ТБ 9.5": 2.20, "Угл ТМ 9.5": 2.20,
        }
        books = {
            "BTTS Да": "Book A", "BTTS Нет": "Book B",
            "Угл ТБ 9.5": "Book A", "Угл ТМ 9.5": "Book A",
        }
        result = best_value_markets(
            _verdict(), rows, odds, bank=1000, kelly_frac=0.25,
            min_prob=0.50, bookmaker_by_pick=books,
            market_validation_by_pick={
                "BTTS Да": True, "Угл ТБ 9.5": True,
            },
        )
        self.assertEqual(set(result), {"winner", "corners"})

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
