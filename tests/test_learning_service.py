import tempfile
import unittest
from datetime import datetime, timezone, timedelta

from model.forecast_ledger import ForecastLedger
from model.error_learner import OnlineErrorLearner
from model.learning_service import ForecastLearningService


class ForecastLearningServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger = ForecastLedger(self.tmp.name + "/forecast.sqlite")
        self.learner = OnlineErrorLearner(
            self.ledger, bins=5, min_bin_samples=2, prior_strength=2,
            max_adjustment=0.12,
        )
        self.service = ForecastLearningService(
            self.ledger, self.learner, enabled=True, min_history=3
        )
        self.start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def tearDown(self):
        self.tmp.cleanup()

    def add_settled(self, i):
        forecast_at = self.start + timedelta(days=i)
        key = f"history-{i}"
        self.ledger.record_prediction(
            forecast_key=key, fixture_id=key, home_team="Home", away_team="Away",
            model_version="test-v1",
            probabilities={"H": 0.6, "D": 0.25, "A": 0.15},
            forecast_at=forecast_at, kickoff_at=forecast_at + timedelta(hours=4),
            league="L",
        )
        self.ledger.settle(
            key, "H", home_score=2, away_score=0,
            settled_at=forecast_at + timedelta(hours=3),
        )

    def forecast(self, key="target", at=None):
        at = at or self.start + timedelta(days=20)
        return self.service.forecast(
            forecast_key=key, fixture_id=key, home_team="Home", away_team="Away",
            model_version="test-v1",
            probabilities={"H": 0.6, "D": 0.25, "A": 0.15},
            forecast_at=at, kickoff_at=at + timedelta(hours=4), league="L",
        )

    def test_calibration_is_gated_by_prior_settled_history(self):
        raw = self.forecast("early", self.start + timedelta(days=1))
        self.assertFalse(raw["correction_enabled"])
        self.assertEqual(raw["probabilities"], raw["raw_probabilities"])
        for i in range(4):
            self.add_settled(i)
        learned = self.forecast()
        self.assertTrue(learned["correction_enabled"])
        self.assertAlmostEqual(sum(learned["probabilities"].values()), 1.0)
        self.assertEqual(learned["history_n"], 4)

    def test_retry_returns_original_served_probabilities(self):
        for i in range(4):
            self.add_settled(i)
        first = self.forecast("stable")
        # A later settlement must not change the already-recorded decision on retry.
        self.ledger.settle(
            "stable", "A", home_score=0, away_score=1,
            settled_at=self.start + timedelta(days=21),
        )
        retry = self.forecast("stable")
        self.assertFalse(first["idempotent_replay"])
        self.assertTrue(retry["idempotent_replay"])
        self.assertEqual(first["probabilities"], retry["probabilities"])
        self.assertEqual(first["raw_probabilities"], retry["raw_probabilities"])

    def test_rejects_forecast_at_or_after_kickoff(self):
        at = self.start + timedelta(days=2)
        with self.assertRaises(ValueError):
            self.service.forecast(
                forecast_key="late", fixture_id="late", home_team="H", away_team="A",
                model_version="v1", probabilities={"H": .5, "D": .3, "A": .2},
                forecast_at=at, kickoff_at=at, league="L",
            )

    def test_requires_source_for_verified_settlement(self):
        self.forecast("settle-me")
        at = self.start + timedelta(days=21)
        with self.assertRaises(ValueError):
            self.service.settle_verified(
                "settle-me", "H", home_score=1, away_score=0,
                settled_at=at, source=" ",
            )
        self.assertTrue(self.service.settle_verified(
            "settle-me", "H", home_score=1, away_score=0,
            settled_at=at, source="trusted-feed",
        ))


if __name__ == "__main__":
    unittest.main()
