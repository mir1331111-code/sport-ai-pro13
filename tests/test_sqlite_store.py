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


if __name__ == "__main__":
    unittest.main()
