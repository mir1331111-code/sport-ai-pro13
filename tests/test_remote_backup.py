import gzip
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from storage import remote_backup


class RemoteSqliteBackupTests(unittest.TestCase):
    def _database(self, path: Path, value: str = "forecast") -> None:
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS sample (value TEXT NOT NULL)")
            conn.execute("INSERT INTO sample(value) VALUES (?)", (value,))

    def test_manual_sqlite_backup_round_trip_preserves_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.sqlite"
            restored = Path(tmp) / "restored.sqlite"
            self._database(source, "model-history")

            payload = remote_backup.create_sqlite_backup_bytes(source)
            result = remote_backup.restore_sqlite_backup_bytes(payload, restored)

            self.assertTrue(result["restored"])
            self.assertEqual(result["tables"]["sample"], 1)
            with sqlite3.connect(restored) as conn:
                value = conn.execute("SELECT value FROM sample").fetchone()[0]
            self.assertEqual(value, "model-history")

    def test_manual_restore_refuses_to_overwrite_nonempty_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.sqlite"
            target = Path(tmp) / "target.sqlite"
            self._database(source, "backup")
            self._database(target, "live")

            payload = remote_backup.create_sqlite_backup_bytes(source)
            with self.assertRaises(FileExistsError):
                remote_backup.restore_sqlite_backup_bytes(payload, target)

            with sqlite3.connect(target) as conn:
                value = conn.execute("SELECT value FROM sample").fetchone()[0]
            self.assertEqual(value, "live")

    def test_empty_local_database_is_recognized_as_safe_restore_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.sqlite"
            with sqlite3.connect(path) as conn:
                conn.execute("CREATE TABLE sample (value TEXT)")
            self.assertFalse(remote_backup._database_has_user_rows(path))

    def test_existing_user_rows_are_never_replaced_by_automatic_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "live.sqlite"
            self._database(path, "keep-me")
            with patch.object(
                remote_backup, "_credentials",
                return_value=("gist-id", "token", "unused"),
            ), patch.object(
                remote_backup, "_get_gist",
                side_effect=AssertionError("remote Gist must not be read"),
            ):
                result = remote_backup.restore_database_if_empty(path)
            self.assertEqual(result["state"], "local_data_present")
            with sqlite3.connect(path) as conn:
                value = conn.execute("SELECT value FROM sample").fetchone()[0]
            self.assertEqual(value, "keep-me")

    def test_remote_backup_requires_gist_key_and_encryption_key(self):
        with patch.object(
            remote_backup, "_credentials", return_value=("", "", ""),
        ):
            result = remote_backup.backup_database_to_gist()
        self.assertFalse(result["ok"])
        self.assertEqual(result["state"], "not_configured")
        self.assertIn("NEURO_BACKUP_KEY", result["message"])

    def test_new_local_snapshot_cannot_shrink_verified_remote_tables(self):
        with self.assertRaisesRegex(ValueError, "fewer rows"):
            remote_backup._assert_backup_does_not_regress(
                {"forecasts": 14, "market_forecasts": 8, "bets": 3},
                {"forecasts": 0, "market_forecasts": 0, "bets": 3},
            )

    def test_new_local_snapshot_may_add_rows_without_regressing_existing_tables(self):
        remote_backup._assert_backup_does_not_regress(
            {"forecasts": 14, "market_forecasts": 8},
            {"forecasts": 15, "market_forecasts": 8, "bank_history": 4},
        )

    def test_secret_gist_visibility_is_required(self):
        with patch("storage.remote_backup.requests.get") as get:
            response = get.return_value
            response.status_code = 200
            response.json.return_value = {"public": True, "files": {}}
            with self.assertRaises(PermissionError):
                remote_backup._get_gist("gist-id", "token")


if __name__ == "__main__":
    unittest.main()
