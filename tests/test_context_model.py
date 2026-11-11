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

    def test_low_source_reliability_invalidates_confirmed_lineup_or_referee(self):
        quality = assess_context_quality({
            "lineup": {
                "is_confirmed": True,
                "source_reliability": 0.45,
                "home": {"confirmed": True},
                "away": {"confirmed": True},
            },
            "referee": {
                "is_confirmed": True,
                "name": "Referee A",
                "source_reliability": 0.95,
            },
            "injuries_is_confirmed": True,
        })
        self.assertFalse(quality["lineup_confirmed"])
        self.assertFalse(quality["strong_value_eligible"])
        self.assertIn("low_source_reliability", quality["missing"])

    def test_walk_forward_ablation_uses_latest_pre_kickoff_snapshot_per_fixture(self):
        from model.context_model import walk_forward_context_ablation

        rows = []
        for index in range(6):
            base = {
                "fixture_id": f"f-{index}",
                "kickoff_at": f"2026-01-{index + 2:02d}T12:00:00+00:00",
                "observed_at": f"2026-01-{index + 2:02d}T10:00:00+00:00",
                "is_confirmed": True,
                "source_reliability": 0.9,
                "lineup_strength_home": 0.6 + index * 0.01,
                "lineup_strength_away": 0.4,
                "key_players_missing_home": 0,
                "key_players_missing_away": index % 2,
                "base_p_home": 0.55,
                "base_p_draw": 0.25,
                "base_p_away": 0.20,
                "result": ("H", "D", "A")[index % 3],
            }
            rows.append(base)
            if index == 0:
                newer = dict(base)
                newer["observed_at"] = "2026-01-02T11:00:00+00:00"
                newer["lineup_strength_home"] = 0.95
                rows.append(newer)
        report = walk_forward_context_ablation(
            rows, min_train=2, min_validation=1, epochs=2,
        )
        self.assertTrue(report["walk_forward"])
        self.assertEqual(report["n"], 6)
        self.assertEqual(len(report["fixture_ids"]), 6)
        self.assertTrue(report["same_sample"])

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
