import unittest

from betting.favorites import model_favorite, rank_model_favorites


class ModelFavoriteTests(unittest.TestCase):
    def test_selects_home_win_as_most_likely_1x2_outcome(self):
        result = model_favorite({
            "home": "Alpha FC", "away": "Beta FC",
            "p1": 0.68, "px": 0.20, "p2": 0.12,
        })
        self.assertEqual(result["pick"], "П1")
        self.assertEqual(result["label"], "Победа Alpha FC")
        self.assertAlmostEqual(result["probability"], 0.68)
        self.assertEqual(result["strength"], "Фаворит")

    def test_selects_draw_when_draw_is_most_likely(self):
        result = model_favorite({
            "match_ru": "Alpha — Beta",
            "p1": 0.29, "px": 0.43, "p2": 0.28,
        })
        self.assertEqual(result["pick"], "X")
        self.assertEqual(result["label"], "Ничья")

    def test_selects_away_win_as_most_likely_outcome(self):
        result = model_favorite({
            "match": "Alpha vs Beta",
            "p1": 0.18, "px": 0.22, "p2": 0.60,
        })
        self.assertEqual(result["pick"], "П2")
        self.assertEqual(result["label"], "Победа Beta")
        self.assertEqual(result["strength"], "Фаворит")

    def test_rejects_missing_or_malformed_probabilities(self):
        self.assertIsNone(model_favorite({"p1": 0.6, "px": 0.4}))
        self.assertIsNone(model_favorite({"p1": 60, "px": 20, "p2": 20}))
        self.assertIsNone(model_favorite({"p1": float("nan"), "px": 0.2, "p2": 0.8}))
        self.assertIsNone(model_favorite({"p1": 0.0, "px": 0.0, "p2": 0.0}))
        self.assertIsNone(model_favorite(None))

    def test_ranking_uses_probability_before_ev_or_input_order(self):
        lower = {"match": "Lower vs Side", "p1": 0.57, "px": 0.23, "p2": 0.20,
                 "verdict": {"ev": 0.50}}
        higher = {"match": "Higher vs Side", "p1": 0.72, "px": 0.16, "p2": 0.12,
                  "verdict": {"ev": -0.20}}
        ranked = rank_model_favorites([lower, higher])
        self.assertEqual([row["card"]["match"] for row in ranked],
                         ["Higher vs Side", "Lower vs Side"])

    def test_ties_are_stable_and_use_fixture_date_then_match(self):
        later = {"match": "Zed vs Away", "date_iso": "2026-11-05",
                 "p1": 0.6, "px": 0.2, "p2": 0.2}
        earlier = {"match": "Alpha vs Away", "date_iso": "2026-11-04",
                   "p1": 0.6, "px": 0.2, "p2": 0.2}
        ranked = rank_model_favorites([later, earlier])
        self.assertEqual(ranked[0]["card"]["match"], "Alpha vs Away")

    def test_does_not_mutate_original_fixture_cards(self):
        card = {"match": "A vs B", "p1": 0.65, "px": 0.2, "p2": 0.15}
        before = dict(card)
        rank_model_favorites([card])
        self.assertEqual(card, before)

    def test_probabilities_outside_tolerance_are_rejected(self):
        self.assertIsNone(model_favorite({"p1": 0.8, "px": 0.3, "p2": 0.2}))


if __name__ == "__main__":
    unittest.main()
