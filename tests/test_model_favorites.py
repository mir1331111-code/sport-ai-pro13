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

    def test_order_is_independent_of_source_position_and_prioritizes_probability(self):
        low = {
            "match": "Low vs Other", "date_iso": "2026-11-06",
            "p1": 0.56, "px": 0.24, "p2": 0.20, "games": 12,
            "verdict": {"ev": 0.80},
        }
        high = {
            "match": "High vs Other", "date_iso": "2026-11-06",
            "p1": 0.71, "px": 0.17, "p2": 0.12, "games": 12,
            "verdict": {"ev": -0.50},
        }
        forward = rank_model_favorites([low, high])
        reverse = rank_model_favorites([high, low])
        self.assertEqual(
            [row["card"]["match"] for row in forward],
            ["High vs Other", "Low vs Other"],
        )
        self.assertEqual(
            [row["card"]["match"] for row in reverse],
            ["High vs Other", "Low vs Other"],
        )

    def test_probability_always_outranks_larger_sample(self):
        higher_probability = {
            "match": "Higher Probability vs Other", "p1": 0.66, "px": 0.20, "p2": 0.14,
            "games": 1,
        }
        lower_probability = {
            "match": "Lower Probability vs Other", "p1": 0.65, "px": 0.20, "p2": 0.15,
            "games": 12,
        }
        ranked = rank_model_favorites([lower_probability, higher_probability])
        self.assertEqual(ranked[0]["card"]["match"], "Higher Probability vs Other")

    def test_data_support_breaks_equal_probability_ranks(self):
        weak_sample = {
            "match": "Weak Sample vs Other", "p1": 0.66, "px": 0.20, "p2": 0.14,
            "games": 1,
        }
        strong_sample = {
            "match": "Strong Sample vs Other", "p1": 0.66, "px": 0.20, "p2": 0.14,
            "games": 12,
        }
        ranked = rank_model_favorites([weak_sample, strong_sample])
        self.assertEqual(ranked[0]["card"]["match"], "Strong Sample vs Other")
        self.assertEqual(ranked[0]["sample_games"], 12)

    def test_clearer_favorite_breaks_equal_probability_tie(self):
        narrow = {"match": "Narrow vs Other", "p1": 0.62, "px": 0.35, "p2": 0.03, "games": 12}
        clear = {"match": "Clear vs Other", "p1": 0.62, "px": 0.20, "p2": 0.18, "games": 12}
        ranked = rank_model_favorites([narrow, clear])
        self.assertEqual(ranked[0]["card"]["match"], "Clear vs Other")
        self.assertGreater(ranked[0]["probability_margin"], ranked[1]["probability_margin"])

    def test_duplicate_fixture_keeps_best_forecast_not_first_provider_row(self):
        first = {
            "match": "Alpha FC vs Beta FC", "date_iso": "2026-11-06",
            "league": "Soccer", "p1": 0.55, "px": 0.25, "p2": 0.20,
            "games": 5, "fixture_id": "tsdb-1",
        }
        better = {
            "match": "Alpha vs Beta", "date_iso": "2026-11-06",
            "league": "Serie A", "p1": 0.71, "px": 0.17, "p2": 0.12,
            "games": 12, "fixture_id": "sstats-2",
        }
        forward = rank_model_favorites([first, better])
        reverse = rank_model_favorites([better, first])
        self.assertEqual(len(forward), 1)
        self.assertEqual(len(reverse), 1)
        self.assertEqual(forward[0]["card"]["league"], "Serie A")
        self.assertEqual(reverse[0]["card"]["league"], "Serie A")

    def test_fixture_limit_is_applied_after_unique_probability_ranking(self):
        cards = [
            {"match": f"Team {i} vs Opponent", "date_iso": "2026-11-06",
             "p1": 0.50 + i * 0.02, "px": 0.30, "p2": 0.20 - i * 0.02,
             "games": 12}
            for i in range(6)
        ]
        ranked = rank_model_favorites(cards, limit=3)
        self.assertEqual(len(ranked), 3)
        self.assertEqual(ranked[0]["card"]["match"], "Team 5 vs Opponent")
        self.assertEqual(ranked[-1]["card"]["match"], "Team 3 vs Opponent")

    def test_does_not_mutate_original_fixture_cards(self):
        card = {"match": "A vs B", "p1": 0.65, "px": 0.2, "p2": 0.15}
        before = dict(card)
        rank_model_favorites([card])
        self.assertEqual(card, before)

    def test_probabilities_outside_tolerance_are_rejected(self):
        self.assertIsNone(model_favorite({"p1": 0.8, "px": 0.3, "p2": 0.2}))


if __name__ == "__main__":
    unittest.main()
