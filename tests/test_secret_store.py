import json
import os
import tempfile
import unittest
from pathlib import Path

from storage.secret_store import load_local_secrets, save_local_secrets


class LocalSecretStoreTests(unittest.TestCase):
    def test_round_trip_only_whitelisted_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".neuro_secrets.json"
            self.assertTrue(save_local_secrets({
                "odds_api_key": "  odds-secret  ",
                "llm_api_key": "llm-secret",
                "sstats_api_key": "sstats-secret",
                "bank": "should-not-save",
                "unknown": "also-not-save",
            }, path))
            self.assertEqual(load_local_secrets(path), {
                "odds_api_key": "odds-secret",
                "llm_api_key": "llm-secret",
                "sstats_api_key": "sstats-secret",
            })
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("bank", raw)
            self.assertNotIn("unknown", raw)
            if os.name == "posix":
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_empty_values_delete_saved_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".neuro_secrets.json"
            self.assertTrue(save_local_secrets({"odds_api_key": "secret"}, path))
            self.assertTrue(save_local_secrets({"odds_api_key": ""}, path))
            self.assertEqual(load_local_secrets(path), {})
            self.assertFalse(path.exists())

    def test_missing_or_corrupt_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".neuro_secrets.json"
            self.assertEqual(load_local_secrets(path), {})
            path.write_text("{broken", encoding="utf-8")
            self.assertEqual(load_local_secrets(path), {})


if __name__ == "__main__":
    unittest.main()
