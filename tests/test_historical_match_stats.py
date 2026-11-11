import pickle
import unittest

from model.historical_match_stats import HistoricalMatchStats, poisson_over_probability
from betting.kelly import market_type


class HistoricalMatchStatsTests(unittest.TestCase):
    def test_trains_only_from_observed_corners_and_cards(self):
        model = HistoricalMatchStats(min_team_games=1)
        self.assertTrue(model.learn_step({
            "HomeTeam": "Alpha", "AwayTeam": "Beta",
            "HC": "6", "AC": "4", "HY": "2", "AY": "1",
            "HR": "1", "AR": "0", "Referee": "Referee A",
            "HTHG": "1", "HTAG": "0", "HTHC": "2", "HTAC": "1",
        }, "E0"))
        pred = model.predict("Alpha", "Beta", "E0")
        self.assertAlmostEqual(pred["corners"]["total"], 10.0)
        self.assertAlmostEqual(pred["cards"]["total"], 4.0)
        self.assertEqual(pred["corners"]["history_matches"], 1)

    def test_prediction_exposes_team_venue_averages_and_dispersion(self):
        model = HistoricalMatchStats(min_team_games=1)
        rows = [
            {"HomeTeam": "Alpha", "AwayTeam": "Beta", "HC": 6, "AC": 4, "HY": 2, "AY": 1, "HR": 0, "AR": 0},
            {"HomeTeam": "Alpha", "AwayTeam": "Gamma", "HC": 8, "AC": 3, "HY": 3, "AY": 2, "HR": 0, "AR": 0},
            {"HomeTeam": "Delta", "AwayTeam": "Beta", "HC": 5, "AC": 7, "HY": 1, "AY": 3, "HR": 0, "AR": 0},
        ]
        for row in rows:
            model.learn_step(row, "E0")
        pred = model.predict("Alpha", "Beta", "E0")
        self.assertEqual(pred["corners"]["home_team_games"], 2)
        self.assertEqual(pred["corners"]["away_team_games"], 2)
        self.assertAlmostEqual(pred["corners"]["home_for_avg"], 7.0)
        self.assertAlmostEqual(pred["corners"]["away_for_avg"], 5.5)
        self.assertGreater(pred["corners"]["total_variance"], 0)
        self.assertIn("home_against_avg", pred["cards"])

    def test_unseen_league_returns_no_forecast(self):
        model = HistoricalMatchStats()
        # Unknown competitions have no observations; the model must not invent
        # league averages merely to populate a card.
        self.assertEqual(model.predict("Venezia", "Udinese", "G"), {})

    def test_missing_statistics_do_not_become_zero(self):
        model = HistoricalMatchStats()
        self.assertFalse(model.learn_step({
            "HomeTeam": "Alpha", "AwayTeam": "Beta",
            "FTHG": "2", "FTAG": "0",
        }, "E0"))
        self.assertEqual(model.predict("Alpha", "Beta", "E0"), {})

    def test_fractional_match_statistics_are_not_training_observations(self):
        model = HistoricalMatchStats(min_team_games=1)
        self.assertFalse(model.learn_step({
            "HomeTeam": "Alpha", "AwayTeam": "Beta",
            "HC": 5.5, "AC": 4, "HY": 2, "AY": 1,
        }, "E0"))
        self.assertFalse(model.learn_step({
            "HomeTeam": "Alpha", "AwayTeam": "Beta",
            "HY": 2, "AY": 1.5, "HR": 0, "AR": 0,
        }, "E0"))
        self.assertEqual(model.predict("Alpha", "Beta", "E0"), {})

    def test_missing_red_card_fields_do_not_become_zero_card_observations(self):
        model = HistoricalMatchStats(min_team_games=1)
        self.assertTrue(model.learn_step({
            "HomeTeam": "Alpha", "AwayTeam": "Beta",
            "HC": 6, "AC": 4, "HY": 2, "AY": 1,
            # HR/AR are deliberately absent.
        }, "E0"))
        prediction = model.predict("Alpha", "Beta", "E0")
        self.assertIn("corners", prediction)
        self.assertNotIn("cards", prediction)

        self.assertTrue(model.learn_step({
            "HomeTeam": "Gamma", "AwayTeam": "Delta",
            "HC": 5, "AC": 3, "HY": 1, "AY": 2,
            "HR": 0, "AR": 1,
        }, "E0"))
        prediction = model.predict("Gamma", "Delta", "E0")
        self.assertEqual(prediction["cards"]["history_matches"], 1)
        self.assertAlmostEqual(prediction["cards"]["total"], 4.0)

    def test_poisson_over_probability_is_bounded_and_monotonic(self):
        p_low = poisson_over_probability(8.0, 9.5)
        p_high = poisson_over_probability(11.0, 9.5)
        self.assertGreaterEqual(p_low, 0.0)
        self.assertLessEqual(p_high, 1.0)
        self.assertGreater(p_high, p_low)

    def test_stat_markets_are_not_goal_score_settled(self):
        self.assertEqual(market_type("Угл ТБ 9.5"), "STAT")
        self.assertEqual(market_type("Карты ТМ 4.5"), "STAT")
        self.assertEqual(market_type("BTTS Да"), "BTTS")

    def test_stat_market_walk_forward_validation_builds_line_reports(self):
        from datetime import date, timedelta
        from model.historical_match_stats import walk_forward_market_validation

        rows = []
        start = date(2024, 1, 1)
        for index in range(100):
            current = start + timedelta(days=index)
            rows.append({
                "Date": current.strftime("%d/%m/%Y"),
                "Div": "E0",
                "HomeTeam": f"Team{index % 10}",
                "AwayTeam": f"Team{(index + 3) % 10}",
                "HC": 5 + (index % 4),
                "AC": 3 + (index % 3),
                "HY": 1 + (index % 2),
                "AY": 1 + (index % 3 == 0),
                "HR": 0,
                "AR": 0,
            })
        report = walk_forward_market_validation(
            rows, min_history=15, min_n=10, bootstrap_samples=20,
        )
        self.assertEqual(report["n_rows"], 100)
        self.assertIn("corners", report["markets"])
        self.assertIn("9.5", report["markets"]["corners"])
        self.assertIn("cards", report["markets"])
        self.assertIn("4.5", report["markets"]["cards"])
        self.assertGreater(report["markets"]["corners"]["9.5"]["n"], 0)
        self.assertIn("validation_passed", report["markets"]["cards"]["4.5"])

        # A row without both red-card fields remains valid for corners but is
        # excluded from the total-cards calibration sample.
        incomplete_rows = [dict(row) for row in rows]
        incomplete_rows[50].pop("HR")
        incomplete_rows[50].pop("AR")
        incomplete_report = walk_forward_market_validation(
            incomplete_rows, min_history=15, min_n=10, bootstrap_samples=0,
        )
        self.assertEqual(
            incomplete_report["markets"]["corners"]["9.5"]["n"],
            report["markets"]["corners"]["9.5"]["n"],
        )
        self.assertEqual(
            incomplete_report["markets"]["cards"]["4.5"]["n"],
            report["markets"]["cards"]["4.5"]["n"] - 1,
        )

    def test_referee_rates_blend_into_cards_with_dispersion_after_five_matches(self):
        model = HistoricalMatchStats(min_team_games=1)
        rows = [
            {"HomeTeam": f"H{i}", "AwayTeam": f"A{i}", "HC": 5, "AC": 4,
             "HY": 4, "AY": 2, "HR": 0, "AR": 0,
             "Referee": "Referee A"}
            for i in range(6)
        ]
        rows.extend(
            {"HomeTeam": f"BH{i}", "AwayTeam": f"BA{i}", "HC": 5, "AC": 4,
             "HY": 1, "AY": 1, "HR": 0, "AR": 0,
             "Referee": "Referee B"}
            for i in range(6)
        )
        for row in rows:
            model.learn_step(row, "E0")
        prediction = model.predict("New Home", "New Away", "E0", referee_id="REFEREE A")
        referee = prediction["cards"]["referee"]
        self.assertTrue(referee["available"])
        self.assertEqual(referee["n"], 6)
        self.assertGreaterEqual(referee["variance"], 0.0)
        self.assertGreater(referee["blend_weight"], 0.0)
        self.assertNotEqual(
            prediction["cards"]["total"],
            prediction["cards"]["league_home_avg"] + prediction["cards"]["league_away_avg"],
        )
        self.assertGreaterEqual(prediction["cards"]["total_variance"], 0.1)

    def test_corners_expose_halftime_score_state_splits_without_leaking_them_into_base(self):
        model = HistoricalMatchStats(min_team_games=1)
        rows = [
            {"HomeTeam": "H1", "AwayTeam": "A1", "HC": 7, "AC": 4, "HTHG": 0, "HTAG": 1,
             "HTHC": 3, "HTAC": 2, "HY": 1, "AY": 1, "HR": 0, "AR": 0},
            {"HomeTeam": "H2", "AwayTeam": "A2", "HC": 8, "AC": 5, "HTHG": 0, "HTAG": 0,
             "HTHC": 4, "HTAC": 3, "HY": 1, "AY": 1, "HR": 0, "AR": 0},
            {"HomeTeam": "H3", "AwayTeam": "A3", "HC": 9, "AC": 5, "HTHG": 1, "HTAG": 0,
             "HTHC": 4, "HTAC": 2, "HY": 1, "AY": 1, "HR": 0, "AR": 0},
        ]
        for row in rows:
            model.learn_step(row, "E0")
        prediction = model.predict("New Home", "New Away", "E0")
        splits = prediction["corners"]["game_state_splits"]
        self.assertEqual(splits["home_trailing"]["n"], 1)
        self.assertEqual(splits["level"]["n"], 1)
        self.assertEqual(splits["away_trailing"]["n"], 1)
        self.assertAlmostEqual(prediction["corners"]["total"], 38.0 / 3.0)
        self.assertAlmostEqual(splits["home_trailing"]["mean"], 6.0)
        self.assertAlmostEqual(splits["level"]["mean"], 6.0)
        self.assertAlmostEqual(splits["away_trailing"]["mean"], 8.0)

    def test_history_stats_survive_pickle_round_trip_and_keep_defaultdicts(self):
        from model.engine import Engine

        model = HistoricalMatchStats(min_team_games=1)
        model.learn_step({
            "HomeTeam": "Alpha", "AwayTeam": "Beta",
            "HC": "6", "AC": "4", "HY": "2", "AY": "1",
            "HR": "1", "AR": "0",
        }, "E0")

        engine = Engine(matrix_n=5)
        engine.match_stats = model
        engine.trained_n = 1
        restored_engine = pickle.loads(pickle.dumps(engine))
        restored = restored_engine.match_stats
        pred = restored.predict("Alpha", "Beta", "E0")
        self.assertAlmostEqual(pred["corners"]["total"], 10.0)
        self.assertAlmostEqual(pred["cards"]["total"], 4.0)
        self.assertEqual(restored.trained_matches["E0"], 1)

        # New keys created after restore must retain the original defaultdict behavior.
        self.assertTrue(restored.learn_step({
            "HomeTeam": "Gamma", "AwayTeam": "Delta",
            "HC": "5", "AC": "6", "HY": "1", "AY": "2",
        }, "E1"))
        self.assertEqual(restored.trained_matches["E1"], 1)


if __name__ == "__main__":
    unittest.main()
