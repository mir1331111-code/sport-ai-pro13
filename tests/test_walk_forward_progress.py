import unittest
from datetime import date, timedelta

from model.historical_match_stats import walk_forward_market_validation
from model.walk_forward import walk_forward_compare


def _rows(n=8):
    start = date(2025, 1, 1)
    rows = []
    for index in range(n):
        stamp = start + timedelta(days=index)
        rows.append({
            "Date": stamp.isoformat(),
            "Div": "E0",
            "HomeTeam": f"Home {index % 4}",
            "AwayTeam": f"Away {index % 4}",
            "FTHG": index % 3,
            "FTAG": (index + 1) % 3,
            "HC": 4 + index % 3,
            "AC": 3 + index % 2,
            "HY": index % 2,
            "AY": (index + 1) % 2,
            "HR": 0,
            "AR": 0,
        })
    return rows


class WalkForwardProgressTests(unittest.TestCase):
    def test_model_validation_reports_monotonic_match_progress(self):
        progress = []
        result = walk_forward_compare(
            _rows(),
            matrix_n=6,
            bootstrap_samples=0,
            totals_bootstrap_samples=0,
            market_bootstrap_samples=0,
            progress_callback=lambda done, total: progress.append((done, total)),
        )

        self.assertEqual(result["n"], 8)
        self.assertTrue(progress)
        self.assertEqual(progress[0], (0, 8))
        self.assertEqual(progress[-1], (8, 8))
        self.assertEqual([done for done, _ in progress], sorted(done for done, _ in progress))
        self.assertTrue(all(total == 8 for _, total in progress))

    def test_corners_and_cards_validation_reports_monotonic_match_progress(self):
        progress = []
        result = walk_forward_market_validation(
            _rows(),
            min_history=2,
            min_n=500,
            bootstrap_samples=0,
            progress_callback=lambda done, total: progress.append((done, total)),
        )

        self.assertEqual(result["n_rows"], 8)
        self.assertTrue(progress)
        self.assertEqual(progress[0], (0, 8))
        self.assertEqual(progress[-1], (8, 8))
        self.assertEqual([done for done, _ in progress], sorted(done for done, _ in progress))
        self.assertTrue(all(total == 8 for _, total in progress))
        self.assertEqual(result["min_n"], 500)


if __name__ == "__main__":
    unittest.main()
