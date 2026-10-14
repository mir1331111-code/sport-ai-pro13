"""Regression tests for probability calibration input validation."""
import unittest

import numpy as np

from model.calibration import PlattCalibrator


class CalibrationTests(unittest.TestCase):
    def _valid_data(self, n_matches=90):
        rng = np.random.default_rng(7)
        logits = rng.normal(size=(n_matches, 3)).reshape(-1).tolist()
        outcomes = np.zeros((n_matches, 3), dtype=float)
        outcomes[np.arange(n_matches), rng.integers(0, 3, size=n_matches)] = 1.0
        return logits, outcomes.reshape(-1).tolist()

    def test_fit_reports_finite_aggregate_validation_loss(self):
        logits, outcomes = self._valid_data()
        calibrator = PlattCalibrator(max_iter=30)
        calibrator.fit(logits, outcomes)
        self.assertTrue(calibrator.trained)
        self.assertIsNotNone(calibrator.val_nll)
        self.assertTrue(np.isfinite(calibrator.val_nll))

    def test_non_finite_logits_are_rejected(self):
        logits, outcomes = self._valid_data()
        logits[0] = float("nan")
        calibrator = PlattCalibrator()
        calibrator.fit(logits, outcomes)
        self.assertFalse(calibrator.trained)

    def test_non_one_hot_outcomes_are_rejected(self):
        logits, outcomes = self._valid_data()
        outcomes[0] = 0.5
        calibrator = PlattCalibrator()
        calibrator.fit(logits, outcomes)
        self.assertFalse(calibrator.trained)

    def test_incomplete_triplet_is_rejected(self):
        logits, outcomes = self._valid_data()
        logits.append(0.1)
        outcomes.append(0.0)
        calibrator = PlattCalibrator()
        calibrator.fit(logits, outcomes)
        self.assertFalse(calibrator.trained)


if __name__ == "__main__":
    unittest.main()
