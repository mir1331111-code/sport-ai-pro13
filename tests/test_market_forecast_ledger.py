import tempfile
import unittest
from datetime import datetime, timezone

from model.forecast_ledger import MarketForecastLedger


class MarketForecastLedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = self.tmp.name + "/audit.sqlite"
        self.ledger = MarketForecastLedger(self.path)
        self.common = dict(
            forecast_key="fx|model|OU|ТБ 2.5", fixture_id="fx",
            home_team="Home", away_team="Away", model_version="v1",
            market="OU", selection="ТБ 2.5", probability=.60,
            entry_odds=2.0, forecast_at=datetime(2026, 10, 1, 10, tzinfo=timezone.utc),
            kickoff_at=datetime(2026, 10, 1, 18, tzinfo=timezone.utc),
            league="League", odds_source="book-a",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_idempotent_and_payload_immutable(self):
        self.assertEqual(
            self.ledger.record_recommendation(**self.common),
            self.ledger.record_recommendation(**self.common),
        )
        with self.assertRaises(ValueError):
            self.ledger.record_recommendation(**dict(self.common, probability=.61))
        self.assertEqual(len(self.ledger.rows()), 1)

    def test_unit_stake_roi_clv_and_calibration_metrics(self):
        self.ledger.record_recommendation(**self.common)
        self.ledger.settle(
            self.common["forecast_key"], "won", result_score="2:1",
            closing_odds=1.8, settlement_source="verified-score",
        )
        row = self.ledger.metrics()["overall"]
        self.assertEqual(row["settled"], 1)
        self.assertAlmostEqual(row["pnl"], 1.0)
        self.assertAlmostEqual(row["roi"], 1.0)
        self.assertAlmostEqual(row["mean_clv"], 2.0 / 1.8 - 1.0)
        self.assertAlmostEqual(row["brier"], (.60 - 1.0) ** 2)

    def test_result_is_immutable(self):
        self.ledger.record_recommendation(**self.common)
        self.ledger.settle(self.common["forecast_key"], "lost")
        with self.assertRaises(ValueError):
            self.ledger.settle(self.common["forecast_key"], "won")


if __name__ == "__main__":
    unittest.main()
