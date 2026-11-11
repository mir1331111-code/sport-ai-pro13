import unittest

from betting.risk import portfolio_risk_report


class PortfolioRiskTests(unittest.TestCase):
    def test_groups_open_exposure_and_flags_same_match_overlap(self):
        bets = [
            {"status": "pending", "stake": 30, "odds": 2.0,
             "date_iso": "2026-11-11", "match": "A vs B", "league": "League A",
             "market": "1X2"},
            {"status": "pending", "stake": 25, "odds": 2.1,
             "date_iso": "2026-11-11", "match": "A vs B", "league": "League A",
             "market": "OU"},
            {"status": "pending", "stake": 20, "odds": 1.9,
             "date_iso": "2026-11-11", "match": "C vs D", "league": "League B",
             "market": "1X2"},
        ]
        report = portfolio_risk_report(bets, bank=1000)
        self.assertEqual(report["open_positions"], 3)
        self.assertEqual(report["total_exposure"], 75)
        self.assertEqual(len(report["overlap"]), 1)
        self.assertEqual(report["overlap"][0]["positions"], 2)
        self.assertEqual(report["overlap"][0]["match"], "2026-11-11|A vs B")

    def test_limit_alerts_match_league_day_and_total_portfolio(self):
        bets = [
            {"status": "pending", "stake": 60, "date_iso": "2026-11-11",
             "match": "A vs B", "league": "League A"},
            {"status": "pending", "stake": 50, "date_iso": "2026-11-11",
             "match": "C vs D", "league": "League A"},
            {"status": "pending", "stake": 30, "date_iso": "2026-11-12",
             "match": "E vs F", "league": "League B"},
        ]
        report = portfolio_risk_report(
            bets, bank=1000, limits={"portfolio_pct": 10},
        )
        messages = " ".join(report["alerts"])
        self.assertIn("Общая открытая экспозиция", messages)
        self.assertIn("Лига League A", messages)
        self.assertIn("Дата 2026-11-11", messages)
        self.assertIn("Матч 2026-11-11|A vs B", messages)

    def test_drawdown_is_reconstructed_from_closed_paper_results(self):
        bets = [
            {"status": "lost", "stake": 50, "odds": 2.0, "settled_at": "2026-01-01"},
            {"status": "won", "stake": 25, "odds": 2.0, "settled_at": "2026-01-02"},
            {"status": "pending", "stake": 10, "odds": 2.0},
        ]
        report = portfolio_risk_report(bets, bank=915, initial_bank=1000)
        self.assertEqual(report["settled_positions"], 2)
        self.assertAlmostEqual(report["max_drawdown"], 0.05)


if __name__ == "__main__":
    unittest.main()
