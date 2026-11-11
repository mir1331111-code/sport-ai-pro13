import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone, timedelta

from model.forecast_ledger import ForecastLedger, MarketForecastLedger, SCHEMA


class ForecastLedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger = ForecastLedger(self.tmp.name + "/forecast.sqlite")
        self.t0 = datetime(2026, 4, 1, 12, tzinfo=timezone.utc)

    def tearDown(self):
        self.tmp.cleanup()

    def record(self, key="fx-1", probs=None, odds=None, xg_used=None, feature_snapshot=None):
        return self.ledger.record_prediction(
            forecast_key=key, fixture_id="fixture-1", home_team="Home",
            away_team="Away", model_version="test-v1",
            probabilities=probs or {"H": 0.6, "D": 0.25, "A": 0.15},
            forecast_at=self.t0, kickoff_at=self.t0 + timedelta(hours=4),
            entry_odds=odds or {"H": 2.0, "D": 3.5, "A": 4.5},
            odds_source="test-book", xg_used=xg_used,
            feature_snapshot=feature_snapshot,
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
        bin_60 = next(row for row in home_bins if row["bin_index"] == 6)
        self.assertAlmostEqual(bin_60["mean_probability"], 0.6)
        self.assertEqual(bin_60["observed_frequency"], 1.0)

    def test_market_metrics_deduplicate_old_scan_reruns_but_keep_other_families(self):
        path = self.tmp.name + "/market-forecast.sqlite"
        market_ledger = MarketForecastLedger(path)
        fixture_id = "ind:2026-04-01:duplicate"
        kickoff = self.t0 + timedelta(hours=4)
        first_time = self.t0
        later_time = self.t0 + timedelta(minutes=15)

        def add(family, stamp, selection, odds, probability):
            key = (
                f"{fixture_id}|main-model-v1|market|{family}|{stamp.isoformat()}"
            )
            market_ledger.record_recommendation(
                forecast_key=key, fixture_id=fixture_id,
                home_team="Home", away_team="Away",
                model_version="main-model-v1", market="STAT",
                selection=selection, probability=probability,
                entry_odds=odds, forecast_at=stamp, kickoff_at=kickoff,
                odds_source="test-book",
            )
            return key

        first_key = add("corners", first_time, "Corners Over 9.5", 1.9, 0.6)
        duplicate_key = add("corners", later_time, "Corners Over 9.5", 2.0, 0.62)
        other_family_key = add("cards", later_time, "Cards Over 3.5", 1.8, 0.55)
        market_ledger.settle(first_key, "won")
        market_ledger.settle(duplicate_key, "lost")
        market_ledger.settle(other_family_key, "lost")

        report = market_ledger.metrics()
        self.assertEqual(len(market_ledger.rows()), 3)  # Raw audit rows are preserved.
        self.assertEqual(report["raw_rows"], 3)
        self.assertEqual(report["duplicate_rows_excluded"], 1)
        self.assertEqual(report["overall"]["n"], 2)
        self.assertEqual(report["overall"]["settled"], 2)
        self.assertEqual(report["overall"]["wins"], 1)
        self.assertEqual(report["overall"]["losses"], 1)

    def test_old_ledger_schema_is_migrated_without_guessing_xg(self):
        path = self.tmp.name + "/old-schema.sqlite"
        old_schema = SCHEMA.replace(
            " xg_used INTEGER CHECK(xg_used IS NULL OR xg_used IN (0,1)),\n", ""
        ).replace(" feature_snapshot_json TEXT,\n", "")
        with sqlite3.connect(path) as connection:
            connection.executescript(old_schema)
        migrated = ForecastLedger(path)
        columns = {
            row[1] for row in sqlite3.connect(path).execute(
                "PRAGMA table_info(forecasts)"
            ).fetchall()
        }
        self.assertIn("xg_used", columns)
        self.assertIn("feature_snapshot_json", columns)
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

    def test_feature_snapshot_is_stored_immutably(self):
        snapshot = {
            "competition_gender": "women",
            "lineup_home_confirmed": True,
            "lineup_away_confirmed": False,
            "referee_present": True,
            "xg_used": False,
        }
        self.record(feature_snapshot=snapshot)
        row = self.ledger.rows()[0]
        self.assertEqual(row["feature_snapshot"], snapshot)
        self.assertIsNone(row["xg_used"])

        with self.assertRaises(ValueError):
            self.record(feature_snapshot={**snapshot, "referee_present": False})

    def test_old_ledger_schema_adds_feature_snapshot_without_backfill(self):
        path = self.tmp.name + "/pre-feature-schema.sqlite"
        old_schema = SCHEMA.replace(
            " feature_snapshot_json TEXT,\n", ""
        )
        with sqlite3.connect(path) as connection:
            connection.executescript(old_schema)
            connection.execute(
                "INSERT INTO forecasts "
                "(forecast_key,fixture_id,forecast_at,model_version,home_team,away_team,"
                "probabilities_json,created_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    "old", "fixture-old", self.t0.isoformat(), "old-v1",
                    "Old Home", "Old Away",
                    '{"H": 0.5, "D": 0.3, "A": 0.2}', self.t0.isoformat(),
                ),
            )
        migrated = ForecastLedger(path)
        row = migrated.rows()[0]
        self.assertIsNone(row["feature_snapshot"])

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
