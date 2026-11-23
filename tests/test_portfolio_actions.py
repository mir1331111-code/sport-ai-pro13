import unittest

from ui.tabs.portfolio import _cancel_bet


class PortfolioCancellationTests(unittest.TestCase):
    def test_cancel_refunds_stake_and_keeps_void_audit_record(self):
        original = {
            "bank": 9900.0,
            "bets": [{
                "match": "Home FC vs Away FC",
                "market": "1X2",
                "pick": "П1",
                "stake": 100.0,
                "odds": 2.1,
                "status": "pending",
                "market_forecast_key": "market-forecast-123",
            }],
            "stats": {"won": 1, "lost": 2, "void": 0, "profit": -20.0},
        }
        updated = _cancel_bet(original, 0)

        self.assertEqual(len(updated["bets"]), 1)
        bet = updated["bets"][0]
        self.assertEqual(bet["status"], "void")
        self.assertEqual(bet["score"], "cancelled by user")
        self.assertEqual(bet["settlement_source"], "manual_cancel")
        self.assertTrue(bet["settled_at"])
        self.assertEqual(bet["market_forecast_key"], "market-forecast-123")
        self.assertEqual(updated["bank"], 10000.0)
        self.assertEqual(updated["stats"]["void"], 1)

        # The state transition is immutable: do not mutate the caller's data.
        self.assertEqual(original["bets"][0]["status"], "pending")
        self.assertEqual(original["bank"], 9900.0)

    def test_cancel_ignores_non_pending_or_missing_positions(self):
        data = {
            "bank": 500.0,
            "bets": [{"status": "won", "stake": 50.0}],
            "stats": {"void": 0},
        }
        self.assertIs(_cancel_bet(data, 0), data)
        self.assertIs(_cancel_bet(data, 5), data)


if __name__ == "__main__":
    unittest.main()
