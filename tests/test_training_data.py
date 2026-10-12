"""Regression tests for safe historical training input validation."""
import unittest
from datetime import date, datetime

from model.training_data import completed_training_row


class CompletedTrainingRowTests(unittest.TestCase):
    def setUp(self):
        self.today = date(2026, 10, 12)
        self.valid = {
            "HomeTeam": "Home FC", "AwayTeam": "Away FC",
            "FTHG": 0, "FTAG": 0, "Date": "11/10/2026",
        }
        self.when = datetime(2026, 10, 11)

    def test_valid_zero_zero_is_a_real_result(self):
        result = completed_training_row(self.valid, self.when, self.today)
        self.assertEqual(result, (self.when, "Home FC", "Away FC", 0.0, 0.0))

    def test_missing_score_is_not_inferred_as_zero(self):
        row = dict(self.valid)
        row.pop("FTHG")
        self.assertIsNone(completed_training_row(row, self.when, self.today))

    def test_empty_score_is_rejected(self):
        row = dict(self.valid, FTAG="")
        self.assertIsNone(completed_training_row(row, self.when, self.today))

    def test_fractional_score_is_rejected(self):
        row = dict(self.valid, FTHG=1.5)
        self.assertIsNone(completed_training_row(row, self.when, self.today))

    def test_future_match_is_rejected(self):
        future = datetime(2026, 10, 13)
        self.assertIsNone(completed_training_row(self.valid, future, self.today))

    def test_missing_team_is_rejected(self):
        row = dict(self.valid, HomeTeam=" ")
        self.assertIsNone(completed_training_row(row, self.when, self.today))

    def test_non_finite_score_is_rejected(self):
        row = dict(self.valid, FTHG=float("nan"))
        self.assertIsNone(completed_training_row(row, self.when, self.today))


if __name__ == "__main__":
    unittest.main()
