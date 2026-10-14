import unittest

from betting.performance import performance_rows as _performance_rows


class PerformanceReportTests(unittest.TestCase):
    def test_roi_uses_only_settled_wins_and_losses(self):
        bets = [
            {"status": "won", "stake": 10, "odds": 2.5, "market": "1X2"},
            {"status": "lost", "stake": 5, "odds": 3.0, "market": "1X2"},
            {"status": "pending", "stake": 20, "odds": 2.0, "market": "1X2"},
            {"status": "push", "stake": 7, "odds": 2.0, "market": "1X2"},
        ]
        rows = _performance_rows(bets, "market")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["N"], 2)
        self.assertEqual(rows[0]["Stake"], 15)
        self.assertEqual(rows[0]["PnL"], 10)
        self.assertAlmostEqual(rows[0]["ROI"], 10 / 15)
        self.assertEqual(rows[0]["Won"], 1)
        self.assertEqual(rows[0]["Lost"], 1)

    def test_odds_buckets(self):
        bets = [
            {"status": "won", "stake": 1, "odds": 1.4},
            {"status": "lost", "stake": 1, "odds": 1.8},
            {"status": "won", "stake": 1, "odds": 2.4},
            {"status": "lost", "stake": 1, "odds": 4.2},
            {"status": "won", "stake": 1, "odds": 6.0},
        ]
        rows = _performance_rows(bets, "odds")
        self.assertEqual({row["Группа"] for row in rows},
                         {"< 1.50", "1.50–1.99", "2.00–2.99",
                          "3.00–4.99", "5.00+"})


if __name__ == "__main__":
    unittest.main()
