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

    def test_manual_restore_rejects_gzip_bomb_before_unbounded_expansion(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "target.sqlite"
            payload = gzip.compress(b"x" * 4096)
            with patch.object(remote_backup, "MAX_SQLITE_BYTES", 64):
                with self.assertRaisesRegex(ValueError, "safety limit"):
                    remote_backup.restore_sqlite_backup_bytes(payload, target)
            self.assertFalse(target.exists())

    def test_manual_restore_rejects_truncated_or_trailing_gzip_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "target.sqlite"
            valid = gzip.compress(b"not a sqlite database")
            for payload in (valid[:-4], valid + b"trailing"):
                with self.subTest(payload_length=len(payload)):
                    with self.assertRaises(ValueError):
                        remote_backup.restore_sqlite_backup_bytes(payload, target)
            self.assertFalse(target.exists())

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

    def test_remote_backup_requires_only_the_github_token(self):
        with patch.object(
            remote_backup, "_credentials", return_value=("", "", ""),
        ):
            result = remote_backup.backup_database_to_gist()
        self.assertFalse(result["ok"])
        self.assertEqual(result["state"], "not_configured")
        self.assertIn("GITHUB_TOKEN", result["message"])

    def test_single_github_token_derives_a_stable_fernet_key(self):
        fake_token = "github-pat-for-unit-test"
        with patch.object(
            remote_backup, "_secret",
            side_effect=lambda name: fake_token if name == "GITHUB_TOKEN" else "",
        ):
            first = remote_backup._credentials()
            second = remote_backup._credentials()

        self.assertEqual(first[0], "")
        self.assertEqual(first[1], fake_token)
        self.assertTrue(first[2])
        self.assertEqual(first[2], second[2])
        remote_backup._fernet(first[2])

    def test_managed_gist_discovery_selects_only_the_matching_secret_gist(self):
        managed = {
            "id": "managed-id",
            "description": remote_backup.MANAGED_GIST_DESCRIPTION,
            "public": False,
            "files": {remote_backup.MANAGED_GIST_MARKER: {}},
        }
        public_copy = {**managed, "id": "public-id", "public": True}
        unrelated = {**managed, "id": "other-id", "description": "unrelated"}
        with patch("storage.remote_backup.requests.get") as get:
            response = get.return_value
            response.status_code = 200
            response.json.return_value = [public_copy, unrelated, managed]
            result = remote_backup._find_managed_gist("token")

        self.assertEqual(result["id"], "managed-id")
        self.assertEqual(get.call_args.kwargs["params"], {"per_page": 100, "page": 1})

    def test_app_creates_a_non_public_managed_gist_automatically(self):
        with patch("storage.remote_backup.requests.post") as post:
            response = post.return_value
            response.status_code = 201
            response.json.return_value = {"id": "new-gist-id", "public": False}
            result = remote_backup._create_managed_gist("token")

        self.assertEqual(result, "new-gist-id")
        payload = post.call_args.kwargs["json"]
        self.assertFalse(payload["public"])
        self.assertIn(remote_backup.MANAGED_GIST_MARKER, payload["files"])

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


    def test_remote_restore_rejects_authenticated_gzip_bomb_with_bounded_output(self):
        from cryptography.fernet import Fernet

        key = Fernet.generate_key().decode("ascii")
        compressed = gzip.compress(b"x" * 4096)
        encrypted = Fernet(key.encode("ascii")).encrypt(compressed)
        manifest = {
            "format": "neuro-sqlite-fernet-chunks-v1",
            "parts": 1,
            # Deliberately falsified to fit the configured test limit.
            "raw_bytes": 64,
            "encrypted_bytes": len(encrypted),
            "encrypted_sha256": __import__("hashlib").sha256(encrypted).hexdigest(),
        }
        gist = {
            "files": {
                remote_backup.MANIFEST_NAME: {"content": __import__("json").dumps(manifest)},
                f"{remote_backup.PART_PREFIX}000.txt": {"content": encrypted.decode("ascii")},
            }
        }
        with patch.object(remote_backup, "MAX_SQLITE_BYTES", 64):
            with self.assertRaisesRegex(ValueError, "safety limit"):
                remote_backup._decode_remote_snapshot(gist, "test-token", key)

    def test_remote_restore_rejects_oversized_encrypted_gist_part(self):
        key = remote_backup.generate_backup_key()
        encrypted = b"x" * (remote_backup.PART_BYTES + 1)
        manifest = {
            "format": "neuro-sqlite-fernet-chunks-v1",
            "parts": 1,
            "raw_bytes": 1,
            "encrypted_bytes": len(encrypted),
            "encrypted_sha256": __import__("hashlib").sha256(encrypted).hexdigest(),
        }
        gist = {
            "files": {
                remote_backup.MANIFEST_NAME: {"content": __import__("json").dumps(manifest)},
                f"{remote_backup.PART_PREFIX}000.txt": {"content": encrypted.decode("ascii")},
            }
        }
        with self.assertRaisesRegex(ValueError, "part exceeds"):
            remote_backup._decode_remote_snapshot(gist, "test-token", key)

    def test_secret_gist_visibility_is_required(self):
        with patch("storage.remote_backup.requests.get") as get:
            response = get.return_value
            response.status_code = 200
            response.json.return_value = {"public": True, "files": {}}
            with self.assertRaises(PermissionError):
                remote_backup._get_gist("gist-id", "token")


if __name__ == "__main__":
    unittest.main()
