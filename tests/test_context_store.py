import tempfile
import unittest
from pathlib import Path

from storage.context_store import ContextStore, referee_entity_id


class ContextStoreTests(unittest.TestCase):
    def test_referee_entity_is_stable_and_not_upstream_numeric_id(self):
        first = referee_entity_id("  Referee A ")
        second = referee_entity_id("referee a")
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("api-football:referee:"))

    def test_snapshot_persists_referee_and_confirmation_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ContextStore(Path(directory) / "context.sqlite")
            store.record_snapshot(
                fixture_id="fixture-1",
                kickoff_at="2026-11-12T20:00:00+00:00",
                observed_at="2026-11-12T19:30:00+00:00",
                context={
                    "api_fixture_id": 42,
                    "referee": {
                        "name": "Referee A",
                        "is_confirmed": True,
                        "source_reliability": None,
                    },
                    "lineup": {
                        "is_confirmed": True,
                        "home": {"confirmed": True},
                        "away": {"confirmed": True},
                    },
                    "injuries_is_confirmed": True,
                },
            )
            self.assertEqual(len(store.referee_rows()), 1)
            snapshot = store.snapshots("fixture-1")[0]
            self.assertEqual(snapshot["lineup_confirmed"], 1)
            self.assertEqual(snapshot["injuries_confirmed"], 1)
            self.assertIsNone(snapshot["source_reliability"])
            self.assertEqual(snapshot["payload"]["api_fixture_id"], 42)

    def test_post_kickoff_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ContextStore(Path(directory) / "context.sqlite")
            with self.assertRaises(ValueError):
                store.record_snapshot(
                    fixture_id="fixture-2",
                    kickoff_at="2026-11-12T20:00:00+00:00",
                    observed_at="2026-11-12T20:00:00+00:00",
                    context={},
                )


if __name__ == "__main__":
    unittest.main()
