import unittest

from model.evaluation import expected_calibration_error, reliability_table, temporal_calibration_comparison


class ProbabilityEvaluationTests(unittest.TestCase):
    def test_reliability_bins_track_frequency(self):
        predictions = [
            (0.8, 0.1, 0.1, 0),
            (0.7, 0.2, 0.1, 0),
            (0.1, 0.8, 0.1, 1),
            (0.1, 0.7, 0.2, 2),
        ]
        table = reliability_table(predictions, n_bins=5)
        home = [row for row in table if row["Исход"] == "П1"]
        self.assertTrue(home)
        self.assertEqual(sum(row["N"] for row in table), 12)
        self.assertIsNotNone(expected_calibration_error(table))

    def test_invalid_rows_are_ignored(self):
        table = reliability_table([
            (float("nan"), 0.5, 0.5, 0),
            (0.3, 0.3, 0.4, 7),
            (0.0, 0.0, 0.0, 1),
        ])
        self.assertEqual(table, [])
        self.assertIsNone(expected_calibration_error(table))

    def test_temporal_comparison_pairs_raw_and_calibrated_scores(self):
        rows = [
            (0.7, 0.2, 0.1, 0, None, 0.5, 0.3, 0.2),
            (0.2, 0.6, 0.2, 1, None, 0.2, 0.6, 0.2),
            (0.1, 0.2, 0.7, 2, None, 0.1, 0.2, 0.7),
            (0.6, 0.2, 0.2, 0, None, 0.6, 0.2, 0.2),
        ]
        result = temporal_calibration_comparison(rows)
        self.assertEqual(len(result), 2)
        self.assertEqual([r["Матчей"] for r in result], [2, 2])
        self.assertGreater(result[0]["Raw Brier"], result[0]["Calibrated Brier"])

    def test_temporal_comparison_requires_two_valid_pairs(self):
        self.assertEqual(
            temporal_calibration_comparison([(0.5, 0.3, 0.2, 0, None, 0.5, 0.3, 0.2)]),
            [],
        )

    def test_perfect_frequency_bin_has_zero_error(self):
        table = [
            {"N": 10, "Средняя P": 0.5, "Фактическая частота": 0.5}
        ]
        self.assertEqual(expected_calibration_error(table), 0.0)


if __name__ == "__main__":
    unittest.main()
