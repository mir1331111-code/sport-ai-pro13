import unittest
from datetime import datetime

from data.sources import _tsdb_league_meta
from model.engine import Engine, team_history_key


class GenderIsolationTests(unittest.TestCase):
    def test_womens_europa_competition_is_classified_before_generic_euro(self):
        self.assertEqual(
            _tsdb_league_meta("UEFA Women's Europa Cup"),
            ("W_EUROPA", "women", True, False),
        )
        self.assertEqual(
            team_history_key("Eintracht Frankfurt", "W_EUROPA"),
            "women::eintrachtfrankfurt",
        )
        self.assertEqual(
            team_history_key("Eintracht Frankfurt", "D1"),
            "eintrachtfrankfurt",
        )

    def test_same_named_mens_and_womens_teams_have_separate_histories(self):
        model = Engine(matrix_n=8, temporal_weighting=False, league_rho_fitting=False)
        model.add(
            "Eintracht Frankfurt", "PSV", 3, 0,
            {"Div": "D1"}, match_date=datetime(2026, 1, 1),
        )
        model.add(
            "Eintracht Frankfurt", "PSV", 0, 2,
            {"Div": "W_EUROPA"}, match_date=datetime(2026, 1, 2),
        )
        men_home = model.st[team_history_key("Eintracht Frankfurt", "D1")]
        women_home = model.st[team_history_key("Eintracht Frankfurt", "W_EUROPA")]
        self.assertEqual(men_home["hs"], [3])
        self.assertEqual(women_home["hs"], [0])
        self.assertIn("eintrachtfrankfurt", model.elo)
        self.assertIn("women::eintrachtfrankfurt", model.elo)
        self.assertEqual(len(model.h2h[("eintrachtfrankfurt", "psv")]), 1)
        self.assertEqual(len(model.h2h[("women::eintrachtfrankfurt", "women::psv")]), 1)

    def test_womens_prediction_does_not_borrow_mens_global_rates_or_calibration(self):
        mixed = Engine(matrix_n=8, temporal_weighting=False, league_rho_fitting=False)
        women_only = Engine(matrix_n=8, temporal_weighting=False, league_rho_fitting=False)
        mixed.add(
            "Same FC", "Other FC", 7, 0, {"Div": "D1"},
            match_date=datetime(2026, 1, 1),
        )
        women's_row = {"Div": "W_EUROPA"}
        mixed.add(
            "Same FC", "Other FC", 0, 2, women's_row,
            match_date=datetime(2026, 1, 2),
        )
        women_only.add(
            "Same FC", "Other FC", 0, 2, women's_row,
            match_date=datetime(2026, 1, 2),
        )
        left = mixed.predict(
            "Same FC", "Other FC", "W_EUROPA",
            match_date=datetime(2026, 1, 3),
        )
        right = women_only.predict(
            "Same FC", "Other FC", "W_EUROPA",
            match_date=datetime(2026, 1, 3),
        )
        for field in ("p1", "x", "p2", "over", "btts"):
            self.assertAlmostEqual(left[field], right[field], places=12)

        mixed._learn_observed_match(
            {
                "home": "Same FC", "away": "Other FC",
                "home_goals": 0, "away_goals": 1,
                "row": {"Div": "W_EUROPA"},
                "match_date": datetime(2026, 1, 4),
            },
            {"p1_raw": 0.2, "x_raw": 0.3, "p2_raw": 0.5},
        )
        self.assertEqual(len(mixed.calib_logits_by_group["women"]), 3)
        self.assertEqual(mixed.calib_logits_by_group["men"], [])


if __name__ == "__main__":
    unittest.main()
