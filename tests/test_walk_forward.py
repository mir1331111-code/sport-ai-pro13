import unittest
from datetime import date, timedelta

from model.walk_forward import (
    _ClusteredRecords, _metrics, _paired_differences,
    _binary_metrics, _binary_paired_differences, walk_forward_compare,
)


class WalkForwardTests(unittest.TestCase):
    def test_compares_baseline_and_candidate_chronologically(self):
        rows = []
        for index in range(24):
            stamp = (date(2024, 1, 1) + timedelta(days=index)).isoformat()
            rows.append({
                "Date": stamp,
                "Div": "E0",
                "HomeTeam": f"Home{index % 4}",
                "AwayTeam": f"Away{index % 5}",
                "FTHG": index % 3,
                "FTAG": (index + 1) % 3,
            })
        result = walk_forward_compare(rows)
        self.assertEqual(result["n"], 24)
        self.assertEqual(result["skipped"], 0)
        self.assertEqual(result["baseline"]["n"], 24)
        self.assertEqual(result["candidate"]["n"], 24)
        self.assertIsNotNone(result["delta_candidate_minus_baseline"]["brier"])
        self.assertIn("untouched chronological holdout", result["interpretation"])
        self.assertEqual(set(result["goals_totals_by_line"]), {"0.5", "1.5", "2.5", "3.5", "4.5", "5.5"})
        for line_report in result["goals_totals_by_line"].values():
            self.assertEqual(line_report["n"], 24)
            self.assertIsNotNone(line_report["metrics"]["brier"])
            self.assertIsNotNone(line_report["metrics"]["log_loss"])

    def test_metrics_resample_calendar_day_clusters_not_individual_fixtures(self):
        records = _ClusteredRecords()
        records.set_cluster("2024-01-01")
        records.append(({"H": 0.80, "D": 0.10, "A": 0.10}, "H"))
        records.append(({"H": 0.10, "D": 0.10, "A": 0.80}, "H"))
        records.set_cluster("2024-01-02")
        records.append(({"H": 0.10, "D": 0.80, "A": 0.10}, "D"))
        records.append(({"H": 0.10, "D": 0.10, "A": 0.80}, "D"))

        report = _metrics(records, bootstrap_samples=100, seed=123)
        self.assertEqual(report["n"], 4)
        self.assertEqual(report["bootstrap_unit"], "matchday")
        self.assertEqual(report["cluster_count"], 2)
        self.assertLessEqual(report["confidence_intervals_95"]["brier"]["low"],
                             report["confidence_intervals_95"]["brier"]["high"])

    def test_paired_and_binary_metrics_keep_matchday_cluster_alignment(self):
        left = _ClusteredRecords()
        right = _ClusteredRecords()
        binary_left = _ClusteredRecords()
        binary_right = _ClusteredRecords()
        for day in ("2024-02-01", "2024-02-02", "2024-02-03"):
            for index in range(2):
                left.set_cluster(day)
                right.set_cluster(day)
                binary_left.set_cluster(day)
                binary_right.set_cluster(day)
                outcome = ("H", "D", "A")[index]
                base = {"H": 0.60, "D": 0.25, "A": 0.15}
                improved = {"H": 0.70, "D": 0.20, "A": 0.10}
                left.append((improved, outcome))
                right.append((base, outcome))
                binary_left.append(({"over": 0.65, "under": 0.35}, "over"))
                binary_right.append(({"over": 0.50, "under": 0.50}, "over"))

        paired = _paired_differences(left, right, bootstrap_samples=80, seed=7)
        binary = _binary_metrics(binary_left, bootstrap_samples=80, seed=7)
        binary_paired = _binary_paired_differences(
            binary_left, binary_right, bootstrap_samples=80, seed=7,
        )
        self.assertEqual(paired["n"], 6)
        self.assertEqual(paired["cluster_count"], 3)
        self.assertEqual(binary["cluster_count"], 3)
        self.assertEqual(binary_paired["cluster_count"], 3)
        self.assertEqual(paired["bootstrap_unit"], "matchday")
        self.assertEqual(binary_paired["bootstrap_unit"], "matchday")

    def test_paired_metrics_reject_misaligned_day_clusters(self):
        left = _ClusteredRecords()
        right = _ClusteredRecords()
        left.set_cluster("2024-03-01")
        right.set_cluster("2024-03-02")
        left.append(({"H": 0.6, "D": 0.2, "A": 0.2}, "H"))
        right.append(({"H": 0.5, "D": 0.3, "A": 0.2}, "H"))
        report = _paired_differences(left, right, bootstrap_samples=10)
        self.assertEqual(report["n"], 0)
        self.assertIn("matchday", report["error"])

    def test_skips_rows_without_valid_dates_or_scores(self):
        result = walk_forward_compare([
            {"Date": "not-a-date", "HomeTeam": "A", "AwayTeam": "B", "FTHG": 1, "FTAG": 0},
            {"Date": "2024-01-01", "HomeTeam": "C", "AwayTeam": "D", "FTHG": "bad", "FTAG": 0},
        ])
        self.assertEqual(result["n"], 0)
        self.assertEqual(result["skipped"], 2)
        self.assertIsNone(result["candidate"]["log_loss"])


    def test_fractional_goal_scores_are_skipped_not_truncated(self):
        result = walk_forward_compare([{
            "Date": "2024-01-01", "Div": "E0",
            "HomeTeam": "A", "AwayTeam": "B",
            "FTHG": 1.5, "FTAG": 0,
        }])
        self.assertEqual(result["n"], 0)
        self.assertEqual(result["skipped"], 1)

    def test_market_gate_bootstrap_can_run_without_full_model_bootstrap(self):
        rows = []
        for index in range(40):
            stamp = (date(2024, 1, 1) + timedelta(days=index)).isoformat()
            rows.append({
                "Date": stamp,
                "Div": "E0",
                "HomeTeam": f"Home{index % 5}",
                "AwayTeam": f"Away{index % 7}",
                "FTHG": index % 3,
                "FTAG": (index + 1) % 3,
                "B365H": 2.10,
                "B365D": 3.40,
                "B365A": 3.80,
            })
        result = walk_forward_compare(
            rows, bootstrap_samples=0, market_bootstrap_samples=30,
            totals_bootstrap_samples=0,
        )
        paired = result["candidate_minus_market"]
        self.assertEqual(paired["n"], 40)
        self.assertIsNotNone(paired["confidence_intervals_95"]["brier"]["low"])
        self.assertIsNotNone(paired["confidence_intervals_95"]["log_loss"]["high"])
        self.assertEqual(result["candidate"]["confidence_intervals_95"]["brier"]["low"], None)
        self.assertEqual(result["market_bootstrap_samples"], 30)



if __name__ == "__main__":
    unittest.main()
