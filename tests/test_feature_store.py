import unittest
from datetime import datetime, timedelta, timezone
from data.feature_store import FeatureSnapshot, FeatureStore


class SnapshotTests(unittest.TestCase):
    def test_snapshot_visibility_respects_cutoff(self):
        store = FeatureStore()
        cutoff = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
        later = cutoff + timedelta(hours=1)
        store.add(FeatureSnapshot("fixture-1", "source-a", "xg", later, {"home": 1.2}))
        self.assertIsNone(store.latest("fixture-1", "xg", cutoff))
        self.assertEqual(store.latest("fixture-1", "xg", later).payload["home"], 1.2)

    def test_timezone_is_required(self):
        store = FeatureStore()
        with self.assertRaises(ValueError):
            store.add(FeatureSnapshot("fixture-1", "source-a", "xg", datetime(2026, 1, 1), {}))


if __name__ == "__main__":
    unittest.main()
