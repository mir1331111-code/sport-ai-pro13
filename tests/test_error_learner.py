import tempfile
import unittest
from datetime import datetime, timezone, timedelta

from model.forecast_ledger import ForecastLedger
from model.error_learner import OnlineErrorLearner


class OnlineErrorLearnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger = ForecastLedger(self.tmp.name + "/forecast.sqlite")
        self.learner = OnlineErrorLearner(self.ledger, bins=5, min_bin_samples=3,
                                          prior_strength=5, max_adjustment=0.12)
        self.start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def tearDown(self):
        self.tmp.cleanup()

    def add(self, i, result, forecast_at=None, settled_at=None, league="League"):
        forecast_at = forecast_at or self.start + timedelta(days=i)
        settled_at = settled_at or forecast_at + timedelta(hours=3)
        key = f"f-{i}"
        self.ledger.record_prediction(
            forecast_key=key, fixture_id=key, home_team="H", away_team="A",
            league=league, model_version="v1",
            probabilities={"H": 0.6, "D": 0.25, "A": 0.15},
            forecast_at=forecast_at)
        self.ledger.settle(key, result, settled_at=settled_at)

    def test_no_history_means_no_correction(self):
        as_of = self.start + timedelta(days=1)
        report = self.learner.calibrate({"H": 0.6, "D": 0.25, "A": 0.15}, as_of=as_of)
        self.assertEqual(report["history_n"], 0)
        self.assertEqual(report["probabilities"], report["raw_probabilities"])

    def test_learns_from_prior_mistakes_and_keeps_probabilities_valid(self):
        for i in range(12):
            self.add(i, "H")
        as_of = self.start + timedelta(days=20)
        report = self.learner.calibrate({"H": 0.6, "D": 0.25, "A": 0.15}, as_of=as_of)
        self.assertGreater(report["probabilities"]["H"], 0.6)
        self.assertAlmostEqual(sum(report["probabilities"].values()), 1.0)
        self.assertTrue(all(0 <= p <= 1 for p in report["probabilities"].values()))

    def test_future_settlement_cannot_leak_into_calibration(self):
        future_forecast = self.start + timedelta(days=5)
        self.add(1, "H", forecast_at=future_forecast,
                 settled_at=self.start + timedelta(days=30))
        report = self.learner.calibrate({"H": 0.6, "D": 0.25, "A": 0.15},
                                        as_of=self.start + timedelta(days=20))
        self.assertEqual(report["history_n"], 0)

    def test_league_filter_and_evaluation(self):
        for i in range(6):
            self.add(i, "H", league="League")
        self.add(7, "A", league="Other")
        as_of = self.start + timedelta(days=20)
        self.assertEqual(self.learner.calibrate(
            {"H": 0.6, "D": 0.25, "A": 0.15}, as_of=as_of, league="League"
        )["history_n"], 6)
        evaluation = self.learner.evaluate(as_of=as_of, league="League")
        self.assertEqual(evaluation["n"], 6)
        self.assertIsNotNone(evaluation["raw_brier"])

    def test_model_version_prefix_keeps_ai_forecasts_out_of_main_model_learning(self):
        base = self.start
        for index, version in enumerate((
            "main-model-v14.4.16", "main-model-v14.4.16",
            "independent-llm-groq", "independent-llm-grok",
        )):
            key = f"version-{index}"
            at = base + timedelta(days=index)
            self.ledger.record_prediction(
                forecast_key=key, fixture_id=key, home_team="H", away_team="A",
                model_version=version, probabilities={"H": .6, "D": .25, "A": .15},
                forecast_at=at, kickoff_at=at + timedelta(hours=2), league="League",
            )
            self.ledger.settle(
                key, "H", home_score=1, away_score=0,
                settled_at=at + timedelta(hours=3),
            )
        main_only = OnlineErrorLearner(
            self.ledger, bins=5, min_bin_samples=1,
            model_version_prefix="main-model-",
        )
        report = main_only.calibrate(
            {"H": .6, "D": .25, "A": .15},
            as_of=base + timedelta(days=10),
        )
        self.assertEqual(report["history_n"], 2)
        self.assertEqual(report["model_version_prefix"], "main-model-")

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):
            OnlineErrorLearner(self.ledger, bins=1)
        with self.assertRaises(ValueError):
            OnlineErrorLearner(self.ledger, max_adjustment=0.9)


if __name__ == "__main__":
    unittest.main()
