import unittest

from model.context_model import (
    assess_context_quality, context_probabilities, evaluate_ablation, fit_context_blender,
)


class ContextModelTests(unittest.TestCase):
    def test_missing_xi_or_referee_blocks_strong_and_value(self):
        quality = assess_context_quality({
            "lineup": {"is_confirmed": True},
            "referee": {"is_confirmed": False, "name": None},
            "injuries_is_confirmed": True,
        })
        self.assertEqual(quality["level"], "Medium")
        self.assertFalse(quality["strong_value_eligible"])
        self.assertIn("referee", quality["missing"])

    def test_context_forecast_is_base_until_trained_blender_exists(self):
        result = context_probabilities(
            {"H": 0.55, "D": 0.25, "A": 0.20},
            {"lineup": {}, "referee": {}},
            model=None,
        )
        self.assertEqual(result["Base"], result["Context"])
        self.assertFalse(result["adjustment_applied"])
        self.assertEqual(result["delta"], {"H": 0.0, "D": 0.0, "A": 0.0})

    def test_ablation_uses_identical_fixture_intersection(self):
        base = [
            {"fixture_id": "a", "result": "H", "probabilities": {"H": .7, "D": .2, "A": .1}},
            {"fixture_id": "b", "result": "D", "probabilities": {"H": .2, "D": .6, "A": .2}},
        ]
        context = [
            {"fixture_id": "a", "result": "H", "probabilities": {"H": .65, "D": .25, "A": .1}},
            {"fixture_id": "c", "result": "A", "probabilities": {"H": .2, "D": .2, "A": .6}},
        ]
        report = evaluate_ablation(base, context)
        self.assertEqual(report["n"], 1)
        self.assertEqual(report["fixture_ids"], ["a"])
        self.assertTrue(report["same_sample"])

    def test_blender_requires_minimum_unique_confirmed_pre_match_rows(self):
        fitted = fit_context_blender([], min_samples=200)
        self.assertFalse(fitted["trained"])
        self.assertEqual(fitted["n"], 0)


if __name__ == "__main__":
    unittest.main()
