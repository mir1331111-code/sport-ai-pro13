import unittest

from model.walk_forward import (_market_probs, _metrics, _paired_differences,
                                _binary_metrics, _total_market_probs,
                                _binary_paired_differences, totals_market_is_validated,
                                walk_forward_compare)


class WalkForwardMarketBenchmarkTests(unittest.TestCase):
    def test_market_triplet_is_normalized_to_remove_overround(self):
        probabilities, source = _market_probs({
            "B365H": "2.00", "B365D": "3.50", "B365A": "4.00"
        })
        self.assertEqual(source, "B365H.B365D.B365A")
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        self.assertGreater(probabilities["H"], probabilities["D"])
        self.assertGreater(probabilities["D"], probabilities["A"])

    def test_incomplete_or_invalid_triplet_is_ignored(self):
        probabilities, source = _market_probs({"B365H": "2.0", "B365D": "3.0"})
        self.assertIsNone(probabilities)
        self.assertIsNone(source)
        probabilities, source = _market_probs({
            "AvgH": "1.0", "AvgD": "3.0", "AvgA": "4.0"
        })
        self.assertIsNone(probabilities)
        self.assertIsNone(source)


    def test_historical_totals_odds_are_de_vigged(self):
        probabilities, source = _total_market_probs({
            "B365>2.5": "1.80", "B365<2.5": "2.05"
        })
        self.assertEqual(source, "B365>2.5/B365<2.5")
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        self.assertGreater(probabilities["over"], probabilities["under"])

    def test_betbrain_totals_columns_are_supported(self):
        probabilities, source = _total_market_probs({
            "BbAv>2.5": "1.85", "BbAv<2.5": "2.00"
        })
        self.assertEqual(source, "BbAv>2.5/BbAv<2.5")
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)

    def test_paired_totals_comparison_and_validation_gate(self):
        candidate = [({"over": 0.80, "under": 0.20}, "over")] * 600
        market = [({"over": 0.55, "under": 0.45}, "over")] * 600
        result = _binary_paired_differences(candidate, market, bootstrap_samples=100, seed=4)
        self.assertEqual(result["n"], 600)
        self.assertLess(result["differences"]["brier"], 0)
        self.assertLess(result["confidence_intervals_95"]["brier"]["high"], 0)
        self.assertTrue(totals_market_is_validated(result, min_n=500))
        self.assertFalse(totals_market_is_validated(result, min_n=601))

    def test_totals_validation_fails_if_confidence_interval_crosses_zero(self):
        report = {
            "n": 700,
            "differences": {"brier": -0.01, "log_loss": -0.02},
            "confidence_intervals_95": {
                "brier": {"low": -0.03, "high": 0.01},
                "log_loss": {"low": -0.04, "high": -0.01},
            },
        }
        self.assertFalse(totals_market_is_validated(report))

    def test_binary_metrics_report_brier_log_loss_and_intervals(self):
        records = [
            ({"over": 0.70, "under": 0.30}, "over"),
            ({"over": 0.25, "under": 0.75}, "under"),
            ({"over": 0.55, "under": 0.45}, "over"),
        ]
        result = _binary_metrics(records, bootstrap_samples=100, seed=7)
        self.assertEqual(result["n"], 3)
        self.assertIsNotNone(result["brier"])
        self.assertIsNotNone(result["log_loss"])
        self.assertLessEqual(result["confidence_intervals_95"]["brier"]["low"],
                             result["confidence_intervals_95"]["brier"]["high"])

    def test_walk_forward_report_includes_paired_totals_benchmark(self):
        rows = [
            {"Date": "2024-01-01", "Div": "E0", "HomeTeam": "A", "AwayTeam": "B",
             "FTHG": 2, "FTAG": 1, "B365>2.5": 1.80, "B365<2.5": 2.05},
            {"Date": "2024-01-08", "Div": "E0", "HomeTeam": "B", "AwayTeam": "C",
             "FTHG": 0, "FTAG": 0, "B365>2.5": 2.10, "B365<2.5": 1.75},
            {"Date": "2024-01-15", "Div": "E0", "HomeTeam": "A", "AwayTeam": "C",
             "FTHG": 1, "FTAG": 2, "BbAv>2.5": 1.95, "BbAv<2.5": 1.90},
        ]
        result = walk_forward_compare(rows, bootstrap_samples=0, totals_bootstrap_samples=0)
        self.assertEqual(result["goals_over25"]["n"], 3)
        self.assertEqual(result["goals_over25_market"]["n"], 3)
        self.assertEqual(result["goals_over25_candidate_minus_market"]["n"], 3)
        self.assertIn("brier", result["goals_over25_candidate_minus_market"]["differences"])

    def test_power_and_shin_methods_return_valid_probabilities(self):
        row = {"B365H": "2.00", "B365D": "3.50", "B365A": "4.00"}
        for method in ("proportional", "power", "shin"):
            probabilities, source = _market_probs(row, method)
            self.assertIsNotNone(source)
            self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=7)
            self.assertTrue(all(0.0 < p < 1.0 for p in probabilities.values()))

    def test_metrics_include_bootstrap_intervals_and_rps(self):
        records = [
            ({"H": 0.6, "D": 0.25, "A": 0.15}, "H"),
            ({"H": 0.2, "D": 0.3, "A": 0.5}, "A"),
            ({"H": 0.25, "D": 0.5, "A": 0.25}, "D"),
        ]
        result = _metrics(records, bootstrap_samples=100, seed=1)
        self.assertEqual(result["n"], 3)
        self.assertIsNotNone(result["rps_hda_order"])
        self.assertLessEqual(result["confidence_intervals_95"]["log_loss"]["low"],
                             result["confidence_intervals_95"]["log_loss"]["high"])

    def test_paired_bootstrap_compares_same_fixtures(self):
        candidate = [({"H": .6, "D": .25, "A": .15}, "H"),
                     ({"H": .2, "D": .3, "A": .5}, "A")]
        market = [({"H": .5, "D": .3, "A": .2}, "H"),
                  ({"H": .25, "D": .3, "A": .45}, "A")]
        result = _paired_differences(candidate, market, bootstrap_samples=50, seed=3)
        self.assertEqual(result["n"], 2)
        self.assertIn("log_loss", result["differences"])
        self.assertIn("low", result["confidence_intervals_95"]["brier"])

if __name__ == "__main__":
    unittest.main()
