import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from storage import sqlite_store


class SQLiteStoreTests(unittest.TestCase):
    def test_decision_snapshot_insert_count_and_cache_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = os.path.join(directory, "test-neuro.sqlite")
            previous_state = (
                sqlite_store.SQLITE_BOOT_OK,
                sqlite_store.SQLITE_BOOT_ERROR,
                sqlite_store._WAL_INITIALIZED_PATH,
            )
            try:
                with (
                    patch.object(sqlite_store, "DB_FILE", database_path),
                    patch.object(sqlite_store, "invalidate_caches") as invalidate,
                ):
                    sqlite_store._WAL_INITIALIZED_PATH = ""
                    self.assertTrue(sqlite_store.db_init(), sqlite_store.SQLITE_BOOT_ERROR)

                    first = {
                        "snapshot_key": "decision-1",
                        "match": "Home - Away",
                        "market": "1X2",
                        "pick": "H",
                        "decision": "TAKE",
                    }
                    second = {
                        "snapshot_key": "decision-2",
                        "match": "Home Two - Away Two",
                        "market": "1X2",
                        "pick": "D",
                        "decision": "SKIP",
                    }

                    self.assertEqual(sqlite_store.insert_decision_snapshots([first]), 1)
                    invalidate.assert_called_once()

                    invalidate.reset_mock()
                    # One row is a duplicate and is ignored; only the new row counts.
                    self.assertEqual(
                        sqlite_store.insert_decision_snapshots([first, second]), 1,
                    )
                    invalidate.assert_called_once()

                    invalidate.reset_mock()
                    self.assertEqual(sqlite_store.insert_decision_snapshots([first]), 0)
                    invalidate.assert_not_called()

                    with sqlite3.connect(database_path) as connection:
                        count = connection.execute(
                            "SELECT COUNT(*) FROM decision_snapshots"
                        ).fetchone()[0]
                    self.assertEqual(count, 2)
            finally:
                (
                    sqlite_store.SQLITE_BOOT_OK,
                    sqlite_store.SQLITE_BOOT_ERROR,
                    sqlite_store._WAL_INITIALIZED_PATH,
                ) = previous_state


    def test_update_bet_by_identity_persists_repeated_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = os.path.join(directory, "test-neuro.sqlite")
            previous_state = (
                sqlite_store.SQLITE_BOOT_OK,
                sqlite_store.SQLITE_BOOT_ERROR,
                sqlite_store._WAL_INITIALIZED_PATH,
            )
            try:
                with (
                    patch.object(sqlite_store, "DB_FILE", database_path),
                    patch.object(sqlite_store, "invalidate_caches") as invalidate,
                ):
                    sqlite_store._WAL_INITIALIZED_PATH = ""
                    self.assertTrue(sqlite_store.db_init(), sqlite_store.SQLITE_BOOT_ERROR)
                    sqlite_store.insert_bets_batch([{
                        "match": "Home - Away",
                        "market": "1X2",
                        "pick": "H",
                        "odds": 2.0,
                        "stake": 10.0,
                        "status": "pending",
                        "date_iso": "2026-11-15",
                    }])
                    invalidate.reset_mock()

                    update_args = (
                        "Home - Away", "1X2", "H", "2026-11-15",
                    )
                    sqlite_store.update_bet_by_identity(*update_args, status="won")
                    invalidate.assert_called_once()

                    # Simulate another writer returning the row to pending.
                    # The exact same function call must execute again, not come
                    # back from a cached None result.
                    with sqlite3.connect(database_path) as connection:
                        connection.execute(
                            "UPDATE bets SET status='pending' WHERE match=?",
                            ("Home - Away",),
                        )
                    invalidate.reset_mock()
                    sqlite_store.update_bet_by_identity(*update_args, status="won")
                    invalidate.assert_called_once()

                    with sqlite3.connect(database_path) as connection:
                        status = connection.execute(
                            "SELECT status FROM bets WHERE match=?",
                            ("Home - Away",),
                        ).fetchone()[0]
                    self.assertEqual(status, "won")
            finally:
                (
                    sqlite_store.SQLITE_BOOT_OK,
                    sqlite_store.SQLITE_BOOT_ERROR,
                    sqlite_store._WAL_INITIALIZED_PATH,
                ) = previous_state

    def test_log_bank_invalidates_bank_history_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = os.path.join(directory, "test-neuro.sqlite")
            previous_state = (
                sqlite_store.SQLITE_BOOT_OK,
                sqlite_store.SQLITE_BOOT_ERROR,
                sqlite_store._WAL_INITIALIZED_PATH,
            )
            try:
                with (
                    patch.object(sqlite_store, "DB_FILE", database_path),
                    patch.object(sqlite_store, "invalidate_caches") as invalidate,
                ):
                    sqlite_store._WAL_INITIALIZED_PATH = ""
                    self.assertTrue(sqlite_store.db_init(), sqlite_store.SQLITE_BOOT_ERROR)
                    sqlite_store.log_bank(1234.5, event="test")
                    invalidate.assert_called_once()
                    with sqlite3.connect(database_path) as connection:
                        count = connection.execute(
                            "SELECT COUNT(*) FROM bank_history"
                        ).fetchone()[0]
                    self.assertEqual(count, 1)
            finally:
                (
                    sqlite_store.SQLITE_BOOT_OK,
                    sqlite_store.SQLITE_BOOT_ERROR,
                    sqlite_store._WAL_INITIALIZED_PATH,
                ) = previous_state


if __name__ == "__main__":
    unittest.main()
