import unittest
from datetime import date, timedelta

from model.walk_forward import walk_forward_compare


class WalkForwardTests(unittest.TestCase):
    def _rows(self, with_xg=True):
        rows = []
        start = date(2022, 1, 1)
        for index in range(100):
            row = {
                "Date": (start + timedelta(days=index)).isoformat(),
                "Div": "E0",
                "HomeTeam": f"Home{index % 8}",
                "AwayTeam": f"Away{index % 9}",
                "FTHG": index % 4,
                "FTAG": (index + 1) % 3,
            }
            if with_xg:
                row["HxG"] = 2.3 if index % 3 == 0 else 0.7
                row["AxG"] = 0.5 if index % 3 == 0 else 1.8
            rows.append(row)
        return rows

    def test_compares_three_arms_on_same_chronological_fixtures(self):
        result = walk_forward_compare(self._rows())
        self.assertEqual(result["n"], 100)
        self.assertEqual(result["skipped"], 0)
        self.assertEqual(result["baseline"]["n"], 100)
        self.assertEqual(result["candidate"]["n"], 100)
        self.assertEqual(result["temporal_goals_only"]["n"], 100)
        self.assertEqual(result["xg_rows_available"], 100)
        self.assertIn("brier", result["xg_delta_vs_same_temporal_model"])
        self.assertIn("untouched chronological holdout", result["interpretation"])

    def test_no_xg_rows_means_ablation_arms_are_identical(self):
        result = walk_forward_compare(self._rows(with_xg=False))
        self.assertEqual(result["xg_rows_available"], 0)
        self.assertEqual(result["xg_forecasts_used"], 0)
        self.assertEqual(result["candidate"], result["temporal_goals_only"])
        self.assertEqual(result["xg_delta_vs_same_temporal_model"], {
            "brier": 0.0, "log_loss": 0.0, "accuracy": 0.0
        })

    def test_skips_malformed_rows(self):
        result = walk_forward_compare([{"Date": "bad", "HomeTeam": "A"}])
        self.assertEqual(result["n"], 0)
        self.assertEqual(result["skipped"], 1)


if __name__ == "__main__":
    unittest.main()
