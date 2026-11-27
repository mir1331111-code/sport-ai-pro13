import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from storage import remote_backup, sqlite_store, usage, user_datasets


class DurableAppStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.local_path = root / "neuro_local.json"
        self.db_path = str(root / "neuro.sqlite")

        self.old_boot_ok = sqlite_store.SQLITE_BOOT_OK
        self.old_wal_path = sqlite_store._WAL_INITIALIZED_PATH
        self.db_patch = patch.object(sqlite_store, "DB_FILE", self.db_path)
        self.local_patch = patch.object(usage, "LOCAL_FILE", str(self.local_path))
        self.dataset_patch = patch.object(user_datasets, "DB_FILE", self.db_path)
        self.db_patch.start()
        self.local_patch.start()
        self.dataset_patch.start()
        self.addCleanup(self.db_patch.stop)
        self.addCleanup(self.local_patch.stop)
        self.addCleanup(self.dataset_patch.stop)
        self.addCleanup(self._restore_sqlite_globals)
        self.assertTrue(sqlite_store.db_init(), sqlite_store.SQLITE_BOOT_ERROR)
        sqlite_store.invalidate_caches()

    def _restore_sqlite_globals(self):
        sqlite_store.SQLITE_BOOT_OK = self.old_boot_ok
        sqlite_store._WAL_INITIALIZED_PATH = self.old_wal_path
        sqlite_store.invalidate_caches()

    def test_portfolio_cards_reports_bank_and_bets_survive_local_file_loss(self):
        expected = {
            "version": 18,
            "bank": 9876.5,
            "bets": [{
                "match": "Home FC vs Away FC",
                "market": "1X2",
                "pick": "П1",
                "odds": 1.95,
                "stake": 20.0,
                "status": "pending",
            }],
            "cards": [{"fixture_id": "sstats:123", "p1": 0.61}],
            "report": ["scan completed"],
            "funnel": {"trained": 1000},
            "meta": {"initial_bank": 10000.0},
            "stats": {"won": 0, "lost": 0, "profit": 0.0, "push": 0, "void": 0},
            "mode": "paper",
        }
        usage.set_local_data(expected)
        self.assertTrue(self.local_path.exists())
        self.local_path.unlink()

        recovered = usage.get_local_data()
        self.assertEqual(recovered, expected)

    def test_durable_sqlite_state_excludes_api_credentials(self):
        expected = {
            "bank": 10000.0,
            "bets": [],
            "cards": [],
            "meta": {
                "initial_bank": 10000.0,
                "sstats_api_key": "DO_NOT_PERSIST_THIS",
                "llm_api_key": "DO_NOT_PERSIST_THIS_EITHER",
            },
        }
        usage.set_local_data(expected)
        stored = sqlite_store.load_app_state()["data"]
        serialized = sqlite_store.get_meta("app_state_json")

        self.assertEqual(stored["meta"], {"initial_bank": 10000.0})
        self.assertNotIn("DO_NOT_PERSIST_THIS", serialized)
        self.assertNotIn("sstats_api_key", stored["meta"])

    def test_encrypted_sqlite_backup_restores_dashboard_and_portfolio_state(self):
        expected = {
            "bank": 8450.0,
            "bets": [{"match": "A vs B", "pick": "X", "stake": 10.0}],
            "cards": [{"fixture_id": "fixture:abc", "p1": 0.5}],
            "report": ["last scan log"],
            "meta": {"initial_bank": 10000.0},
        }
        usage.set_local_data(expected)
        history_rows = [{
            "Date": "2026-11-21", "Div": "S_ABC123",
            "HomeTeam": "Home FC", "AwayTeam": "Away FC",
            "FTHG": 2, "FTAG": 1,
        }]
        user_datasets.save_user_dataset("custom_historical_match_rows", history_rows)
        payload = remote_backup.create_sqlite_backup_bytes(self.db_path)

        restored_path = str(Path(self.temp.name) / "restored.sqlite")
        remote_backup.restore_sqlite_backup_bytes(payload, restored_path)
        with (
            patch.object(sqlite_store, "DB_FILE", restored_path),
            patch.object(user_datasets, "DB_FILE", restored_path),
        ):
            self.local_path.unlink()
            recovered = usage.get_local_data()
            restored_history = user_datasets.load_user_dataset(
                "custom_historical_match_rows", db_path=restored_path,
            )
        self.assertEqual(recovered, expected)
        self.assertEqual(restored_history, history_rows)

    def test_imported_match_history_is_merged_without_losing_existing_statistics(self):
        original = [{
            "Date": "2026-11-20", "Div": "W_E0",
            "HomeTeam": "Home FC", "AwayTeam": "Away FC",
            "FTHG": 2, "FTAG": 1, "HC": "", "AC": "",
        }]
        first = user_datasets.merge_user_dataset(
            "custom_historical_match_rows", original, db_path=self.db_path,
        )
        self.assertEqual(first["added_rows"], 1)

        imported = [
            {
                "Date": "2026-11-20", "Div": "W_E0",
                "HomeTeam": "Home FC", "AwayTeam": "Away FC",
                "FTHG": 2, "FTAG": 1, "HC": 8, "AC": 4,
                "HY": 2, "AY": 1, "HR": 0, "AR": 0,
                "HxG": 1.7, "AxG": 0.8,
            },
            {
                "Date": "2026-11-21", "Div": "W_E0",
                "HomeTeam": "New Home", "AwayTeam": "New Away",
                "FTHG": 0, "FTAG": 0, "HC": 5, "AC": 6,
            },
            {
                "Date": "2026-11-20", "Div": "W_E0",
                "HomeTeam": "Home FC", "AwayTeam": "Away FC",
                "FTHG": 9, "FTAG": 0, "HC": 12, "AC": 0,
            },
        ]
        merged = user_datasets.merge_user_dataset(
            "custom_historical_match_rows", imported, db_path=self.db_path,
        )
        rows = user_datasets.load_user_dataset(
            "custom_historical_match_rows", db_path=self.db_path,
        )
        self.assertEqual(merged["row_count"], 2)
        self.assertEqual(merged["added_rows"], 1)
        self.assertEqual(merged["enriched_rows"], 1)
        self.assertEqual(merged["conflicts"], 1)
        preserved = next(row for row in rows if row["HomeTeam"] == "Home FC")
        self.assertEqual((preserved["FTHG"], preserved["FTAG"]), (2, 1))
        self.assertEqual((preserved["HC"], preserved["AC"]), (8, 4))
        self.assertEqual((preserved["HY"], preserved["AY"]), (2, 1))
        self.assertEqual((preserved["HR"], preserved["AR"]), (0, 0))
        self.assertEqual((preserved["HxG"], preserved["AxG"]), (1.7, 0.8))

    def test_context_history_merge_keeps_distinct_observations_and_enriches_blanks(self):
        fixture = {
            "fixture_id": "fixture-123",
            "home_team": "Home FC", "away_team": "Away FC",
            "kickoff_at": "2026-11-25T18:00:00+00:00",
            "observed_at": "2026-11-25T10:00:00+00:00",
            "referee_id": "", "referee_yellow_mean": "",
        }
        user_datasets.merge_user_dataset("historical_context_rows", [fixture], db_path=self.db_path)
        enriched = {**fixture, "referee_id": "ref-9", "referee_yellow_mean": "3.8"}
        second = {
            **fixture,
            "observed_at": "2026-11-25T11:00:00+00:00",
            "referee_id": "ref-9",
        }
        result = user_datasets.merge_user_dataset(
            "historical_context_rows", [enriched, second], db_path=self.db_path,
        )
        rows = user_datasets.load_user_dataset("historical_context_rows", db_path=self.db_path)
        self.assertEqual(result["row_count"], 2)
        self.assertEqual(result["enriched_rows"], 1)
        self.assertEqual(result["added_rows"], 1)
        first = next(row for row in rows if row["observed_at"] == fixture["observed_at"])
        self.assertEqual(first["referee_id"], "ref-9")
        self.assertEqual(first["referee_yellow_mean"], "3.8")

    def test_legacy_sqlite_bets_are_recovered_when_app_state_was_not_in_old_backup(self):
        sqlite_store.insert_bets_batch([{
            "match": "Home FC vs Away FC",
            "match_ru": "Home — Away",
            "league": "Premier League",
            "market": "1X2",
            "pick": "П1",
            "odds": 2.0,
            "stake": 25.0,
            "prob": 0.6,
            "status": "won",
            "mode": "paper",
            "fixture_id": "sstats:12345",
            "date": "23.11.2026",
            "date_iso": "2026-11-23",
        }])
        sqlite_store.log_bank(10025.0, event="settlement")
        with patch.object(usage, "_gist_read", return_value={}):
            recovered = usage.get_local_data()

        self.assertEqual(len(recovered["bets"]), 1)
        self.assertEqual(recovered["bets"][0]["status"], "won")
        self.assertEqual(recovered["bets"][0]["sstats_game_id"], 12345)
        self.assertEqual(recovered["bank"], 10025.0)
        self.assertTrue(recovered["meta"]["portfolio_recovered_from_legacy_sqlite"])


if __name__ == "__main__":
    unittest.main()
