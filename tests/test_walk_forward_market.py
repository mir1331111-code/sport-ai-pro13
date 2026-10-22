import unittest

from model.walk_forward import _market_probs


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


if __name__ == "__main__":
    unittest.main()
