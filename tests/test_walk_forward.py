import unittest
from datetime import date, timedelta

from model.walk_forward import walk_forward_compare


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

    def test_skips_rows_without_valid_dates_or_scores(self):
        result = walk_forward_compare([
            {"Date": "not-a-date", "HomeTeam": "A", "AwayTeam": "B", "FTHG": 1, "FTAG": 0},
            {"Date": "2024-01-01", "HomeTeam": "C", "AwayTeam": "D", "FTHG": "bad", "FTAG": 0},
        ])
        self.assertEqual(result["n"], 0)
        self.assertEqual(result["skipped"], 2)
        self.assertIsNone(result["candidate"]["log_loss"])


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
