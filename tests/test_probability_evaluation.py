import unittest

from model.evaluation import expected_calibration_error, reliability_table


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

    def test_perfect_frequency_bin_has_zero_error(self):
        table = [
            {"N": 10, "Средняя P": 0.5, "Фактическая частота": 0.5}
        ]
        self.assertEqual(expected_calibration_error(table), 0.0)


if __name__ == "__main__":
    unittest.main()
