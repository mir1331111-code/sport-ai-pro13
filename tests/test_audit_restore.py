import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from model.audit_export import build_audit_export
from model.audit_restore import restore_audit_export_bytes
from model.forecast_ledger import ForecastLedger, MarketForecastLedger


class AuditArchiveRestoreTests(unittest.TestCase):
    def _archive(self) -> bytes:
        payload = {
            "generated_at_utc": "2026-11-14T18:00:00+00:00",
            "forecasts": [{
                "id": 1,
                "forecast_key": "main-forecast-1",
                "fixture_id": "fixture-1",
                "kickoff_at": "2026-11-15T18:00:00+00:00",
                "forecast_at": "2026-11-14T18:00:00+00:00",
                "model_version": "main-model-test",
                "home_team": "Home",
                "away_team": "Away",
                "league": "Test League",
                "probabilities": {"H": 0.5, "D": 0.3, "A": 0.2},
                "entry_odds": {"H": 2.1, "D": 3.2, "A": 3.8},
                "odds_source": "Test Book",
                "xg_used": 0,
                "result": "H",
                "home_score": 2,
                "away_score": 1,
                "closing_odds": {"H": 2.0, "D": 3.3, "A": 4.0},
                "closing_odds_source": "Test Close",
                "settled_at": "2026-11-15T20:00:00+00:00",
                "created_at": "2026-11-14T18:00:00+00:00",
            }],
            "market_forecasts": [{
                "id": 1,
                "forecast_key": "market-forecast-1",
                "fixture_id": "fixture-1",
                "kickoff_at": "2026-11-15T18:00:00+00:00",
                "forecast_at": "2026-11-14T18:01:00+00:00",
                "model_version": "main-model-test",
                "home_team": "Home",
                "away_team": "Away",
                "league": "Test League",
                "market": "OU",
                "selection": "ТБ 2.5",
                "probability": 0.58,
                "entry_odds": 1.9,
                "odds_source": "Test Book",
                "unit_stake": 1.0,
                "result_status": "won",
                "result_score": "2:1",
                "actual_value": 3.0,
                "closing_odds": 1.85,
                "settlement_source": "test",
                "settled_at": "2026-11-15T20:00:00+00:00",
                "created_at": "2026-11-14T18:01:00+00:00",
            }],
            "paper_bets": [{
                "id": 1,
                "match": "Home vs Away",
                "match_ru": "Дом — Гости",
                "div": "E0",
                "league": "Test League",
                "market": "1X2",
                "pick": "П1",
                "odds": 2.1,
                "stake": 10.0,
                "prob": 0.5,
                "status": "won",
                "mode": "paper",
                "fixture_id": "fixture-1",
                "date_iso": "2026-11-15",
                "created_at": "2026-11-14 18:00:00",
            }],
            "bank_history": [{
                "ts": "2026-11-15 20:00:00",
                "bank": 1010.0,
                "event": "settlement",
                "pnl": 10.0,
            }],
            "decision_snapshots": [{
                "id": 1,
                "snapshot_key": "decision-1",
                "fixture_id": "fixture-1",
                "match": "Home vs Away",
                "match_ru": "Дом — Гости",
                "league": "Test League",
                "market": "1X2",
                "pick": "П1",
                "decision": "WATCH",
                "decision_reason": "value gate",
                "model_prob": 0.5,
                "fair_odd": 2.0,
                "market_odd": 2.1,
                "edge": 0.05,
                "ev": 0.05,
                "kelly_pct": 0.01,
                "confidence": 0.6,
                "value_score": 0.5,
                "odds_source": "Test Book",
                "date_iso": "2026-11-15",
                "created_at": "2026-11-14 18:00:00",
                "result_status": "won",
                "result_score": "2:1",
                "virtual_pnl": 10.0,
                "evaluated_at": "2026-11-15 20:00:00",
            }],
        }
        return build_audit_export(payload)

    def test_legacy_current_audit_zip_restores_all_audit_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "neuro.db"
            result = restore_audit_export_bytes(self._archive(), path)

            self.assertEqual(result["forecasts"], 1)
            self.assertEqual(result["market_forecasts"], 1)
            self.assertEqual(result["paper_bets"], 1)
            self.assertEqual(result["bank_history"], 1)
            self.assertEqual(result["decision_snapshots"], 1)

            forecast = ForecastLedger(path).rows()[0]
            self.assertEqual(forecast["probabilities"], {"H": 0.5, "D": 0.3, "A": 0.2})
            self.assertEqual(forecast["result"], "H")
            self.assertEqual(forecast["home_score"], 2)

            market = MarketForecastLedger(path).rows()[0]
            self.assertEqual(market["result_status"], "won")
            self.assertEqual(market["selection"], "ТБ 2.5")

            with sqlite3.connect(path) as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM bets").fetchone()[0], 1)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM bank_history").fetchone()[0], 1)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM decision_snapshots").fetchone()[0], 1)

    def test_audit_restore_refuses_to_replace_existing_local_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "neuro.db"
            with sqlite3.connect(path) as conn:
                conn.execute("CREATE TABLE keep_me (value TEXT)")
                conn.execute("INSERT INTO keep_me(value) VALUES ('live')")
            with self.assertRaises(FileExistsError):
                restore_audit_export_bytes(self._archive(), path)
            with sqlite3.connect(path) as conn:
                self.assertEqual(
                    conn.execute("SELECT value FROM keep_me").fetchone()[0],
                    "live",
                )

    def test_invalid_audit_zip_is_rejected_without_creating_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "neuro.db"
            with self.assertRaises(ValueError):
                restore_audit_export_bytes(b"not-a-zip", path)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
