import unittest

from model.walk_forward import _market_probs, _metrics, _paired_differences


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
