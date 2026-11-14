import sqlite3
import tempfile
import unittest
from pathlib import Path

from storage.user_datasets import (
    export_user_datasets,
    load_user_dataset,
    save_user_dataset,
)


class UserDatasetStoreTests(unittest.TestCase):
    def test_dataset_survives_a_new_database_connection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "neuro.db")
            rows = [
                {
                    "Date": "2025-04-01",
                    "HomeTeam": "Home W",
                    "AwayTeam": "Away W",
                    "FTHG": 2,
                    "FTAG": 1,
                    "Div": "W_E0",
                }
            ]
            saved = save_user_dataset(
                "custom_historical_match_rows", rows, db_path=path,
            )
            self.assertEqual(saved["row_count"], 1)
            self.assertEqual(
                load_user_dataset("custom_historical_match_rows", db_path=path),
                rows,
            )
            exported = export_user_datasets(db_path=path)
            self.assertEqual(exported[0]["dataset_key"], "custom_historical_match_rows")
            self.assertEqual(exported[0]["rows"], rows)

    def test_context_history_and_training_history_are_stored_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "neuro.db")
            context = [{
                "fixture_id": "fixture-1",
                "kickoff_at": "2025-04-01T18:00:00+00:00",
                "observed_at": "2025-04-01T12:00:00+00:00",
            }]
            training = [{
                "Date": "2025-04-01",
                "HomeTeam": "Home",
                "AwayTeam": "Away",
                "FTHG": 1,
                "FTAG": 0,
                "Div": "NT_EURO",
            }]
            save_user_dataset("historical_context_rows", context, db_path=path)
            save_user_dataset("custom_historical_match_rows", training, db_path=path)

            self.assertEqual(
                load_user_dataset("historical_context_rows", db_path=path),
                context,
            )
            self.assertEqual(
                load_user_dataset("custom_historical_match_rows", db_path=path),
                training,
            )

    def test_invalid_dataset_name_or_row_shape_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "neuro.db")
            with self.assertRaises(ValueError):
                save_user_dataset("api_keys", [{"key": "secret"}], db_path=path)
            with self.assertRaises(ValueError):
                save_user_dataset("historical_context_rows", ["not a row"], db_path=path)

    def test_non_finite_values_are_rejected_without_overwriting_existing_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "neuro.db")
            original = [{"fixture_id": "saved"}]
            save_user_dataset("historical_context_rows", original, db_path=path)
            with self.assertRaises(ValueError):
                save_user_dataset(
                    "historical_context_rows",
                    [{"fixture_id": "bad", "probability": float("nan")}],
                    db_path=path,
                )
            self.assertEqual(
                load_user_dataset("historical_context_rows", db_path=path),
                original,
            )

    def test_sha256_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "neuro.db")
            save_user_dataset(
                "custom_historical_match_rows", [{"fixture_id": "one"}],
                db_path=path,
            )
            with sqlite3.connect(path) as conn:
                conn.execute(
                    "UPDATE user_datasets SET payload_json=? WHERE dataset_key=?",
                    ('[{"fixture_id":"tampered"}]', "custom_historical_match_rows"),
                )
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_user_dataset("custom_historical_match_rows", db_path=path)


if __name__ == "__main__":
    unittest.main()
