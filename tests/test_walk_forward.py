"""Regression tests for frozen walk-forward predictions."""
import unittest
from datetime import datetime
from unittest.mock import patch

from model.engine import Engine


class FrozenPredictionTests(unittest.TestCase):
    def test_learn_step_uses_precomputed_prediction_without_repredicting(self):
        engine = Engine()
        prediction = {
            "p1": 0.50, "x": 0.25, "p2": 0.25,
            "p1_raw": 0.48, "x_raw": 0.27, "p2_raw": 0.25,
        }

        with patch.object(engine, "predict", side_effect=AssertionError(
            "learn_step must not recompute a frozen prediction"
        )) as predict:
            returned = engine.learn_step(
                "Home FC", "Away FC", 1, 0,
                row={"Date": "2026-01-01"},
                lg="E0",
                match_num=121,
                total=200,
                match_date=datetime(2026, 1, 1),
                prediction=prediction,
            )

        predict.assert_not_called()
        self.assertIs(returned, prediction)
        self.assertEqual(engine.trained_n, 1)
        self.assertEqual(engine.match_count, 1)
        self.assertEqual(len(engine.calib_logits), 3)
        self.assertEqual(len(engine.calib_outcomes), 3)
        self.assertEqual(engine.calib_outcomes, [1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
