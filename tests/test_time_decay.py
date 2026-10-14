"""Tests for time-decayed team features and rest-day diagnostics."""
import unittest
from datetime import datetime, timedelta

from model.engine import Engine


class TimeDecayTests(unittest.TestCase):
    def test_recent_observation_has_more_weight(self):
        engine = Engine()
        old = datetime(2026, 1, 1)
        recent = datetime(2026, 3, 1)
        engine.team_history["Team"] = [
            (old.date(), {"goals": 0.0}),
            (recent.date(), {"goals": 2.0}),
        ]
        result = engine._time_weighted_mean(
            "Team", "goals", datetime(2026, 3, 2), 1.0, half_life=30.0
        )
        self.assertGreater(result, 1.0)

    def test_future_observations_are_excluded(self):
        engine = Engine()
        today = datetime(2026, 3, 1).date()
        engine.team_history["Team"] = [
            (today - timedelta(days=10), {"goals": 1.0}),
            (today + timedelta(days=10), {"goals": 9.0}),
        ]
        result = engine._time_weighted_mean("Team", "goals", today, 0.0)
        self.assertAlmostEqual(result, 1.0)

    def test_rest_days_are_reported_from_prior_match(self):
        engine = Engine()
        engine.add("Home", "Away", 1, 0, match_date=datetime(2026, 1, 1))
        prediction = engine.predict(
            "Home", "New Opponent", match_date=datetime(2026, 1, 5)
        )
        self.assertEqual(prediction["rest_days_home"], 4)
        self.assertIsNone(prediction["rest_days_away"])

    def test_missing_dates_do_not_invent_rest_days(self):
        engine = Engine()
        engine.add("Home", "Away", 1, 0)
        prediction = engine.predict("Home", "Away")
        self.assertIsNone(prediction["rest_days_home"])
        self.assertIsNone(prediction["rest_days_away"])


if __name__ == "__main__":
    unittest.main()
