import unittest
from datetime import datetime, timedelta

from model.engine import Engine


class CompetitionIsolationTests(unittest.TestCase):
    def test_team_form_elo_h2h_and_rest_are_not_shared_between_divisions(self):
        model = Engine(matrix_n=8, temporal_weighting=False, league_rho_fitting=False)
        other_comp = Engine(matrix_n=8, temporal_weighting=False, league_rho_fitting=False)
        start = datetime(2024, 1, 1)

        for index in range(8):
            model.add(
                "Same Team", "Other Team", 4, 0,
                row={"Div": "W_E0"},
                match_date=(start + timedelta(days=7 * index)).isoformat(),
            )

        women = model.predict("Same Team", "Other Team", "W_E0", match_date="2025-01-01")
        men's = model.predict("Same Team", "Other Team", "E0", match_date="2025-01-01")
        untouched_mens = other_comp.predict(
            "Same Team", "Other Team", "E0", match_date="2025-01-01"
        )

        self.assertEqual(women["games"], 8)
        self.assertGreater(women["h2h_n"], 0)
        self.assertEqual(men["games"], 0)
        self.assertEqual(men["h2h_n"], 0)
        self.assertEqual(men["lams"], untouched_mens["lams"])
        self.assertEqual(men["p1"], untouched_mens["p1"])
        self.assertNotEqual(women["lams"], men["lams"])
        self.assertEqual(model.form_str("Same Team", "W_E0"), "ВВВВВ")
        self.assertEqual(model.form_str("Same Team", "E0"), "—")

    def test_sparse_competition_does_not_borrow_global_goal_average(self):
        model = Engine(matrix_n=8, temporal_weighting=False, league_rho_fitting=False)
        for _ in range(25):
            model.add(
                "A", "B", 5, 0,
                row={"Div": "E0"},
                match_date="2024-01-01",
            )
        actual = model.predict("New A", "New B", "W_E0")
        untouched = Engine(matrix_n=8, temporal_weighting=False, league_rho_fitting=False).predict(
            "New A", "New B", "W_E0"
        )
        self.assertEqual(actual["lams"], untouched["lams"])
        self.assertEqual(actual["p1"], untouched["p1"])

    def test_calibration_targets_are_stored_per_competition(self):
        model = Engine()
        prediction = {"p1_raw": 0.55, "x_raw": 0.25, "p2_raw": 0.20}
        model.record_calibration("W_E0", prediction, "H")
        self.assertEqual(len(model.calib_logits["W_E0"]), 3)
        self.assertEqual(model.calib_logits["E0"], [])
        self.assertIsNot(model.calibrators["W_E0"], model.calibrators["E0"])

    def test_default_legacy_namespace_remains_compatible(self):
        model = Engine()
        for _ in range(6):
            model.add("Home FC", "Away FC", 2, 0)
        self.assertEqual(len(model.h2h[("Home FC", "Away FC")]), 6)
        self.assertEqual(model.predict("Home FC", "Away FC")["h2h_n"], 6)


if __name__ == "__main__":
    unittest.main()
