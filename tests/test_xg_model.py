import unittest
from datetime import date, timedelta

from model.engine import Engine, _row_xg_pair


class XGModelTests(unittest.TestCase):
    def test_extracts_supported_xg_schemas_and_rejects_missing_values(self):
        self.assertEqual(_row_xg_pair({"HxG": "1.45", "AxG": "0.72"}), (1.45, 0.72))
        self.assertEqual(_row_xg_pair({"xG": "2.1", "xGA": "0.8"}), (2.1, 0.8))
        self.assertEqual(_row_xg_pair({"home_xg": 1.2, "away_xg": 1.1}), (1.2, 1.1))
        self.assertEqual(_row_xg_pair({"HxG": "nan", "AxG": 0.8}), (None, None))
        self.assertEqual(_row_xg_pair({"HxG": 9.0, "AxG": 0.8}), (None, None))
        self.assertEqual(_row_xg_pair({"FTHG": 3, "FTAG": 0}), (None, None))

    def test_xg_blends_with_goals_only_after_sufficient_league_and_team_history(self):
        engine = Engine()
        start = date(2024, 1, 1)
        for i in range(40):
            # Recent xG profile diverges from the unchanged 1-1 scoreline.
            # Last twelve home observations have low xG; away observations high.
            hxg, axg = (2.4, 0.5) if i < 20 else (0.5, 2.4)
            engine.add(
                "Home", "Away", 1, 1,
                row={"Div": "E0", "HxG": hxg, "AxG": axg},
                match_date=(start + timedelta(days=7 * i)).isoformat(),
            )
        forecast = engine.predict("Home", "Away", "E0", match_date="2025-01-01")
        self.assertTrue(forecast["xg_used"])
        self.assertGreaterEqual(forecast["xg_league_samples"], 30)
        self.assertEqual(forecast["xg_home_samples"], 12)
        self.assertEqual(forecast["xg_away_samples"], 12)
        self.assertGreater(forecast["xg_weight"], 0.0)
        self.assertLess(forecast["lams"][0], forecast["lams"][1])

    def test_modelled_proxy_is_separate_and_has_low_blend_weight(self):
        engine = Engine()
        start = date(2023, 1, 1)
        for i in range(80):
            engine.add("Home", "Away", 1, 1, row={
                "Div": "E0", "HxG": 1.7, "AxG": 0.8,
                "xg_source": "shot_aggregate_ridge_v1",
                "xg_kind": "modelled_proxy",
            }, match_date=(start + timedelta(days=7 * i)).isoformat())
        prediction = engine.predict("Home", "Away", "E0", match_date="2025-01-01")
        self.assertTrue(prediction["xg_used"])
        self.assertTrue(prediction["xg_modelled_used"])
        self.assertEqual(prediction["xg_source"], "shot_model_v1_proxy")
        self.assertGreaterEqual(prediction["xg_modelled_league_samples"], 30)
        self.assertLessEqual(prediction["xg_weight"], 0.12)
        self.assertEqual(prediction["xg_league_samples"], 0)

    def test_genuine_provider_xg_is_not_blended_with_proxy_stream(self):
        engine = Engine()
        start = date(2023, 1, 1)
        for i in range(80):
            kind = "modelled_proxy" if i < 50 else None
            source = "shot_aggregate_ridge_v1" if kind else "footballapi.org"
            row = {"Div": "E0", "HxG": 1.7, "AxG": 0.8, "xg_source": source}
            if kind:
                row["xg_kind"] = kind
            engine.add("Home", "Away", 1, 1, row=row,
                       match_date=(start + timedelta(days=7 * i)).isoformat())
        prediction = engine.predict("Home", "Away", "E0", match_date="2025-01-01")
        self.assertTrue(prediction["xg_used"])
        self.assertFalse(prediction["xg_modelled_used"])
        self.assertEqual(prediction["xg_source"], "provider")

    def test_missing_xg_keeps_legacy_goals_only_path(self):
        engine = Engine()
        for i in range(40):
            engine.add(
                "Home", "Away", 1, 1, row={"Div": "E0"},
                match_date=(date(2024, 1, 1) + timedelta(days=7 * i)).isoformat(),
            )
        forecast = engine.predict("Home", "Away", "E0", match_date="2025-01-01")
        self.assertFalse(forecast["xg_used"])
        self.assertEqual(forecast["xg_weight"], 0.0)
        self.assertEqual(forecast["xg_league_samples"], 0)


if __name__ == "__main__":
    unittest.main()
