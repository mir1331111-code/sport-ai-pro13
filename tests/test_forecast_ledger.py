import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone, timedelta

from model.forecast_ledger import ForecastLedger, SCHEMA


class ForecastLedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger = ForecastLedger(self.tmp.name + "/forecast.sqlite")
        self.t0 = datetime(2026, 4, 1, 12, tzinfo=timezone.utc)

    def tearDown(self):
        self.tmp.cleanup()

    def record(self, key="fx-1", probs=None, odds=None, xg_used=None):
        return self.ledger.record_prediction(
            forecast_key=key, fixture_id="fixture-1", home_team="Home",
            away_team="Away", model_version="test-v1",
            probabilities=probs or {"H": 0.6, "D": 0.25, "A": 0.15},
            forecast_at=self.t0, kickoff_at=self.t0 + timedelta(hours=4),
            entry_odds=odds or {"H": 2.0, "D": 3.5, "A": 4.5},
            odds_source="test-book", xg_used=xg_used,
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

    def test_reliability_bins_and_xg_share_use_only_known_flags_and_settled_results(self):
        self.record("fx-1", xg_used=True)
        self.ledger.settle("fx-1", "H")
        self.record(
            "fx-2", probs={"H": 0.4, "D": 0.35, "A": 0.25}, xg_used=False,
        )
        self.ledger.settle("fx-2", "A")

        report = self.ledger.metrics()
        self.assertEqual(report["xg_usage"]["forecasts"], 2)
        self.assertEqual(report["xg_usage"]["tracked"], 2)
        self.assertEqual(report["xg_usage"]["xg_used"], 1)
        self.assertAlmostEqual(report["xg_usage"]["share"], 0.5)
        self.assertEqual(report["calibration_bins"]["n"], 2)
        home_bins = [
            row for row in report["calibration_bins"]["rows"]
            if row["outcome"] == "H"
        ]
        self.assertEqual(sum(row["n"] for row in home_bins), 2)
        bin_60 = next(row for row in home_bins if row["bin_index"] == 3)
        self.assertAlmostEqual(bin_60["mean_probability"], 0.6)
        self.assertEqual(bin_60["observed_frequency"], 1.0)

    def test_old_ledger_schema_is_migrated_without_guessing_xg(self):
        path = self.tmp.name + "/old-schema.sqlite"
        old_schema = SCHEMA.replace(
            " xg_used INTEGER CHECK(xg_used IS NULL OR xg_used IN (0,1)),\n", ""
        )
        with sqlite3.connect(path) as connection:
            connection.executescript(old_schema)
        migrated = ForecastLedger(path)
        columns = {
            row[1] for row in sqlite3.connect(path).execute(
                "PRAGMA table_info(forecasts)"
            ).fetchall()
        }
        self.assertIn("xg_used", columns)
        migrated.record_prediction(
            forecast_key="legacy", fixture_id="legacy", home_team="Home",
            away_team="Away", model_version="main-model-old",
            probabilities={"H": 0.5, "D": 0.3, "A": 0.2},
            forecast_at=self.t0, xg_used=None,
        )
        # A legacy immutable row remains unknown on later scans. Do not infer it.
        migrated.record_prediction(
            forecast_key="legacy", fixture_id="legacy", home_team="Home",
            away_team="Away", model_version="main-model-old",
            probabilities={"H": 0.5, "D": 0.3, "A": 0.2},
            forecast_at=self.t0, xg_used=True,
        )
        self.assertIsNone(migrated.rows()[0]["xg_used"])

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
