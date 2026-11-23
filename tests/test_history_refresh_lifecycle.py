import threading
import unittest
from unittest.mock import patch

from model.incremental import stable_engine_fingerprint
from ui import scanner_engine


class FakeFittedEngine:
    def __init__(self):
        self.match_stats = object()
        self.trained_n = 1234
        self._incremental_manifest = [("2025-01-01", "fixture-digest")]


class HistoricalRefreshLifecycleTests(unittest.TestCase):
    def _stable_key(self, matrix_n=12, footballapi_key=""):
        years, divisions = scanner_engine._history_catalog()
        return stable_engine_fingerprint(
            matrix_n=matrix_n,
            xg_enabled=bool(footballapi_key.strip()),
            divisions=divisions,
            history_years=years,
        )

    def test_normal_scan_returns_fitted_engine_without_reading_history(self):
        engine = FakeFittedEngine()
        key = self._stable_key()
        cache = {"engines": {key: engine}, "locks": {}, "lock": threading.RLock()}
        logs = []

        with (
            patch.object(scanner_engine, "_shared_engine_cache", return_value=cache),
            patch.object(scanner_engine, "_load_seasonal_history") as load_history,
        ):
            actual = scanner_engine._load_engine(
                12, logs, lambda *args, **kwargs: None,
                refresh_history=False,
            )

        self.assertIs(actual, engine)
        load_history.assert_not_called()
        self.assertTrue(any("Исторический архив не перечитывался" in line for line in logs))

    def test_cache_presence_check_does_not_read_historical_sources(self):
        engine = FakeFittedEngine()
        key = self._stable_key()
        cache = {"engines": {key: engine}, "locks": {}, "lock": threading.RLock()}

        with (
            patch.object(scanner_engine, "_shared_engine_cache", return_value=cache),
            patch.object(scanner_engine, "_load_seasonal_history") as load_history,
        ):
            ready = scanner_engine.has_cached_engine(12)

        self.assertTrue(ready)
        load_history.assert_not_called()

    def test_seasonal_csv_cache_is_bypassed_only_on_explicit_refresh(self):
        from data import sources

        response = type("Response", (), {
            "status_code": 200,
            "content": (
                b"Date,HomeTeam,AwayTeam,FTHG,FTAG\n"
                b"01/01/2026,Home, Away,2,1\n"
            ),
        })()
        with (
            patch.object(sources, "cache_get", return_value=[{"HomeTeam": "Cached"}]) as cache_read,
            patch.object(sources, "_sess") as session,
            patch.object(sources, "cache_put") as cache_write,
        ):
            session.get.return_value = response
            rows = sources.load_seasonal("E0", "2526", force_refresh=True)

        cache_read.assert_not_called()
        session.get.assert_called_once()
        cache_write.assert_called_once()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["HomeTeam"], "Home")


if __name__ == "__main__":
    unittest.main()
