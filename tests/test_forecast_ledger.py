import tempfile
import unittest
from datetime import datetime, timezone, timedelta

from model.forecast_ledger import ForecastLedger


class ForecastLedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger = ForecastLedger(self.tmp.name + "/forecast.sqlite")
        self.t0 = datetime(2026, 4, 1, 12, tzinfo=timezone.utc)

    def tearDown(self):
        self.tmp.cleanup()

    def record(self, key="fx-1", probs=None, odds=None):
        return self.ledger.record_prediction(
            forecast_key=key, fixture_id="fixture-1", home_team="Home",
            away_team="Away", model_version="test-v1",
            probabilities=probs or {"H": 0.6, "D": 0.25, "A": 0.15},
            forecast_at=self.t0, kickoff_at=self.t0 + timedelta(hours=4),
            entry_odds=odds or {"H": 2.0, "D": 3.5, "A": 4.5},
            odds_source="test-book",
        )

    def test_record_is_idempotent_and_prediction_is_immutable(self):
        first = self.record()
        second = self.record()
        self.assertEqual(first, second)
        with self.assertRaises(ValueError):
            self.record(probs={"H": 0.2, "D": 0.3, "A": 0.5})
        rows = self.ledger.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["probabilities"]["H"], 0.6)

    def test_settlement_and_metrics(self):
        self.record()
        self.ledger.settle(
            "fx-1", "H", home_score=2, away_score=0,
            settled_at=self.t0 + timedelta(hours=5),
            closing_odds={"H": 1.8, "D": 3.6, "A": 5.0},
            closing_odds_source="close-book",
        )
        metrics = self.ledger.metrics()
        self.assertEqual(metrics["n"], 1)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["log_loss"], -__import__("math").log(0.6))
        self.assertAlmostEqual(metrics["mean_clv"], ((2.0/1.8-1)+(3.5/3.6-1)+(4.5/5.0-1))/3)
        self.assertEqual(metrics["clv_n"], 3)

    def test_unknown_forecast_and_conflicting_result_fail(self):
        with self.assertRaises(KeyError):
            self.ledger.settle("missing", "H")
        self.record()
        self.ledger.settle("fx-1", "H")
        with self.assertRaises(ValueError):
            self.ledger.settle("fx-1", "A")

    def test_settlement_scores_cannot_be_rewritten(self):
        self.record()
        self.ledger.settle("fx-1", "H", home_score=2, away_score=0)
        with self.assertRaises(ValueError):
            self.ledger.settle("fx-1", "H", home_score=1, away_score=0)

    def test_invalid_probability_and_timezone_rejected(self):
        with self.assertRaises(ValueError):
            self.record(probs={"H": 0.5, "D": 0.2, "A": 0.2})
        with self.assertRaises(ValueError):
            self.ledger.record_prediction(
                forecast_key="naive", fixture_id="f", home_team="h",
                away_team="a", model_version="v",
                probabilities={"H": 0.4, "D": 0.3, "A": 0.3},
                forecast_at=datetime(2026, 1, 1),
            )

    def test_score_must_match_result(self):
        self.record()
        with self.assertRaises(ValueError):
            self.ledger.settle("fx-1", "A", home_score=2, away_score=0)

    def test_unsettled_predictions_do_not_enter_metrics(self):
        self.record()
        self.assertEqual(self.ledger.metrics()["n"], 0)
        self.assertIsNone(self.ledger.metrics()["log_loss"])


if __name__ == "__main__":
    unittest.main()
