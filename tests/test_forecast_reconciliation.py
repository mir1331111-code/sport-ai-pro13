import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from model.forecast_ledger import ForecastLedger
from ui.tabs import scanner


class ForecastReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger = ForecastLedger(self.tmp.name + "/ledger.sqlite")
        self.kickoff = datetime.now(timezone.utc) - timedelta(days=2)
        self.forecast_at = self.kickoff - timedelta(hours=3)

    def tearDown(self):
        self.tmp.cleanup()

    def record(self, fixture_id, key):
        self.ledger.record_prediction(
            forecast_key=key, fixture_id=fixture_id,
            home_team="Home FC", away_team="Away FC",
            model_version="test-v1",
            probabilities={"H": 0.5, "D": 0.3, "A": 0.2},
            forecast_at=self.forecast_at, kickoff_at=self.kickoff,
        )

    def test_tsdb_settles_without_fdorg_token(self):
        self.record("tsdb:123", "tsdb:123:test-v1")
        with patch.object(scanner, "tsdb_match_result",
                          return_value={"home": 2, "away": 1, "status": "FT"}) as tsdb:
            settled = scanner._reconcile_forecast_results(self.ledger, "", [])
        self.assertEqual(settled, 1)
        self.assertEqual(self.ledger.rows()[0]["result"], "H")
        tsdb.assert_called_once_with("123", unittest.mock.ANY)

    def test_espn_settles_without_fdorg_token(self):
        self.record("espn:456", "espn:456:test-v1")
        with patch.object(scanner, "espn_match_result",
                          return_value={"home": 0, "away": 0, "status": "FT"}) as espn:
            settled = scanner._reconcile_forecast_results(self.ledger, "", [])
        self.assertEqual(settled, 1)
        self.assertEqual(self.ledger.rows()[0]["result"], "D")
        espn.assert_called_once()

    def test_fdorg_legacy_entry_requires_token_and_stays_pending_without_it(self):
        self.record("789", "fdorg:789:test-v1")
        settled = scanner._reconcile_forecast_results(self.ledger, "", [])
        self.assertEqual(settled, 0)
        self.assertIsNone(self.ledger.rows()[0]["result"])

    def test_unfinished_provider_result_does_not_settle(self):
        self.record("tsdb:321", "tsdb:321:test-v1")
        with patch.object(scanner, "tsdb_match_result",
                          return_value={"home": 2, "away": 1, "status": "LIVE"}):
            settled = scanner._reconcile_forecast_results(self.ledger, "", [])
        self.assertEqual(settled, 0)
        self.assertIsNone(self.ledger.rows()[0]["result"])


if __name__ == "__main__":
    unittest.main()
