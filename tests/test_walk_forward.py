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

    def test_skips_rows_without_valid_dates_or_scores(self):
        result = walk_forward_compare([
            {"Date": "not-a-date", "HomeTeam": "A", "AwayTeam": "B", "FTHG": 1, "FTAG": 0},
            {"Date": "2024-01-01", "HomeTeam": "C", "AwayTeam": "D", "FTHG": "bad", "FTAG": 0},
        ])
        self.assertEqual(result["n"], 0)
        self.assertEqual(result["skipped"], 2)
        self.assertIsNone(result["candidate"]["log_loss"])


if __name__ == "__main__":
    unittest.main()
