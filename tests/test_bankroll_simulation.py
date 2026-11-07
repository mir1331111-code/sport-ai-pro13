import unittest

from model.bankroll_simulation import simulate_bankroll


class BankrollMonteCarloTests(unittest.TestCase):
    def _rows(self, *, n=20):
        rows = []
        for index in range(n):
            won = index % 2 == 0
            rows.append({
                "fixture_id": f"fixture-{index}",
                "forecast_at": f"2026-10-{index+1:02d}T10:00:00+00:00",
                "kickoff_at": f"2026-10-{index+1:02d}T18:00:00+00:00",
                "model_version": "main-model-v13",
                "market": "1X2",
                "selection": "П1",
                "probability": 0.55,
                "entry_odds": 2.0,
                "unit_stake": 1.0,
                "result_status": "won" if won else "lost",
            })
        return rows

    def test_simulation_is_deterministic_and_reports_bankroll_risk(self):
        rows = self._rows()
        first = simulate_bankroll(rows, simulations=300, seed=77)
        second = simulate_bankroll(rows, simulations=300, seed=77)
        self.assertTrue(first["available"])
        self.assertEqual(first["final_bank"], second["final_bank"])
        self.assertEqual(first["probability_below_start"], second["probability_below_start"])
        self.assertEqual(first["sample_n"], 20)
        self.assertEqual(first["bets_per_path"], 20)
        self.assertGreaterEqual(first["max_drawdown"]["p95"], 0.0)
        self.assertLessEqual(first["probability_50pct_drawdown"], 1.0)

    def test_duplicate_fixture_does_not_get_double_staked(self):
        rows = self._rows(n=10)
        duplicate = dict(rows[0], selection="ТБ 2.5", market="OU",
                         probability=0.8, entry_odds=2.2)
        rows.append(duplicate)
        result = simulate_bankroll(rows, simulations=200, market=None)
        self.assertEqual(result["sample_n"], 10)
        self.assertEqual(result["counts"]["duplicate_fixture_rows"], 1)

    def test_requires_minimum_sample_and_excludes_unsettled_rows(self):
        rows = self._rows(n=12)
        rows[0]["result_status"] = None
        result = simulate_bankroll(rows, simulations=200)
        self.assertTrue(result["available"])
        self.assertEqual(result["sample_n"], 11)
        result = simulate_bankroll(rows[:8], simulations=200)
        self.assertFalse(result["available"])

    def test_kelly_mode_does_not_stake_negative_edge(self):
        rows = self._rows(n=10)
        for row in rows:
            row["probability"] = 0.3
            row["entry_odds"] = 2.0
        result = simulate_bankroll(
            rows, simulations=200, stake_mode="fractional_kelly",
        )
        self.assertEqual(result["final_bank"]["median"], 10_000.0)


if __name__ == "__main__":
    unittest.main()
