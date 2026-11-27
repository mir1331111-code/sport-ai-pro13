import tempfile
import unittest
from unittest.mock import patch

from storage import sqlite_store


class SQLiteStoreUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db_path = self.temp.name + "/bets.sqlite"
        self.db_file_patch = patch.object(sqlite_store, "DB_FILE", self.db_path)
        self.db_file_patch.start()
        self.addCleanup(self.db_file_patch.stop)
        self.assertTrue(
            sqlite_store.db_init(),
            sqlite_store.SQLITE_BOOT_ERROR,
        )
        sqlite_store.invalidate_caches()

    def test_update_bet_by_identity_persists_legacy_manual_settlement(self):
        inserted = sqlite_store.insert_bets_batch([{
            "match": "Home FC vs Away FC",
            "match_ru": "Home FC — Away FC",
            "league": "Premier League",
            "market": "1X2",
            "pick": "П1",
            "odds": 2.10,
            "stake": 10.0,
            "prob": 0.55,
            "status": "pending",
            "mode": "paper",
            "fixture_id": "fixture-123",
            "date": "23.11 18:00",
            "date_iso": "2026-11-23",
        }])
        self.assertEqual(inserted, 1)

        # This is the legacy path used when the Streamlit session copy has no
        # SQLite row id. It must not raise and must update the persisted row.
        sqlite_store.update_bet_by_identity(
            "Home FC vs Away FC",
            "1X2",
            "П1",
            "2026-11-23",
            status="won",
            score="2:1",
            settled_at="2026-11-23T19:50:00+00:00",
        )

        rows = sqlite_store.fetch_bets()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "won")
        self.assertEqual(rows[0]["score"], "2:1")
        self.assertEqual(rows[0]["settled_at"], "2026-11-23T19:50:00+00:00")

    def test_update_by_identity_is_safe_when_no_pending_position_matches(self):
        sqlite_store.update_bet_by_identity(
            "Unknown vs Match", "1X2", "X", "2026-11-23",
            status="won",
        )
        self.assertEqual(sqlite_store.fetch_bets(), [])

    def test_completed_results_archive_round_trips_and_replaces_same_day(self):
        first = {
            "date": "2026-11-26",
            "total": 1,
            "matches": [{"home": "Home FC", "away": "Away FC", "score": "2:1"}],
        }
        stamp = sqlite_store.save_completed_results(first)
        self.assertTrue(stamp)
        loaded = sqlite_store.fetch_completed_results("2026-11-26")
        self.assertEqual(loaded["matches"][0]["score"], "2:1")
        self.assertEqual(loaded["archive_updated_at"], stamp)

        updated = {
            "date": "2026-11-26",
            "total": 2,
            "matches": [
                {"home": "Home FC", "away": "Away FC", "score": "2:1"},
                {"home": "Another FC", "away": "Visitor FC", "score": "0:0"},
            ],
        }
        sqlite_store.save_completed_results(updated)
        loaded_again = sqlite_store.fetch_completed_results("2026-11-26")
        self.assertEqual(loaded_again["total"], 2)
        self.assertEqual(len(loaded_again["matches"]), 2)
        self.assertEqual(sqlite_store.completed_results_archive_days()[0]["date"], "2026-11-26")

    def test_completed_results_archive_rejects_invalid_calendar_day(self):
        with self.assertRaises(ValueError):
            sqlite_store.save_completed_results({"date": "2026-02-30"})
        with self.assertRaises(ValueError):
            sqlite_store.fetch_completed_results("not-a-date")


if __name__ == "__main__":
    unittest.main()
