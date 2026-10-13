import unittest

from model.evaluation import expected_calibration_error, reliability_table, temporal_calibration_comparison, paired_score_uncertainty


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

    def test_paired_bootstrap_reports_interval_and_pair_counts(self):
        rows = [
            (0.7, 0.2, 0.1, 0, (0.5, 0.3, 0.2), 0.5, 0.3, 0.2),
            (0.2, 0.6, 0.2, 1, (0.3, 0.4, 0.3), 0.2, 0.6, 0.2),
            (0.1, 0.2, 0.7, 2, (0.2, 0.3, 0.5), 0.1, 0.2, 0.7),
            (0.6, 0.2, 0.2, 0, (0.4, 0.3, 0.3), 0.6, 0.2, 0.2),
        ]
        result = paired_score_uncertainty(rows, n_bootstrap=200)
        self.assertEqual(len(result), 2)
        raw = next(x for x in result if x["Сравнение"] == "Калибровка − raw")
        market = next(x for x in result if x["Сравнение"] == "Модель − рынок")
        self.assertEqual(raw["Матчей"], 4)
        self.assertEqual(market["Матчей"], 4)
        self.assertLessEqual(raw["Brier CI 2.5%"], raw["Brier CI 97.5%"])
        self.assertLessEqual(market["Log Loss CI 2.5%"], market["Log Loss CI 97.5%"])

    def test_bootstrap_requires_reasonable_replicates(self):
        with self.assertRaises(ValueError):
            paired_score_uncertainty([], n_bootstrap=10)

    def test_perfect_frequency_bin_has_zero_error(self):
        table = [
            {"N": 10, "Средняя P": 0.5, "Фактическая частота": 0.5}
        ]
        self.assertEqual(expected_calibration_error(table), 0.0)


if __name__ == "__main__":
    unittest.main()
