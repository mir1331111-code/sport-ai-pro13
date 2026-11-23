import unittest
from datetime import date, datetime

from model.engine import Engine, _team_state_key
from model.historical_match_stats import HistoricalMatchStats
from model.training import (
    prepare_completed_training_rows,
    train_engine_chronologically,
)


class ChronologicalTrainingTests(unittest.TestCase):
    @staticmethod
    def row(day, home, away, hg=1, ag=0, **extra):
        return {
            "Date": day,
            "HomeTeam": home,
            "AwayTeam": away,
            "FTHG": hg,
            "FTAG": ag,
            **extra,
        }

    def test_preparation_sorts_all_leagues_globally_and_rejects_missing_scores(self):
        history = {
            "E0": [[
                self.row("03/01/2024", "EHome", "EAway", 2, 1),
                self.row("01/01/2024", "Bad", "Score", None, 0),
            ]],
            "SP1": [[
                self.row("02/01/2024", "SHome", "SAway", 0, 0),
                self.row("01/01/2024", "NoHome", "", 0, 0),
                self.row("01/01/2024", "Valid", "Early", 1, 2),
            ]],
        }
        rows, skipped = prepare_completed_training_rows(history, ("E0", "SP1"))
        self.assertEqual(
            [(item["match_date"].date(), item["league"], item["home"]) for item in rows],
            [
                (date(2024, 1, 1), "SP1", "Valid"),
                (date(2024, 1, 2), "SP1", "SHome"),
                (date(2024, 1, 3), "E0", "EHome"),
            ],
        )
        self.assertEqual(skipped["missing_or_invalid_score"], 1)
        self.assertEqual(skipped["missing_teams"], 1)

    def test_prediction_exposes_competition_specific_calibration_state(self):
        model = Engine(matrix_n=8, temporal_weighting=False, league_rho_fitting=False)
        prediction = model.predict("A", "B", "S_TEST")
        self.assertFalse(prediction["probability_calibrated"])
        self.assertEqual(prediction["calibration_matches"], 0)

    def test_engine_joins_safe_provider_aliases_but_keeps_reserves_separate(self):
        model = Engine(matrix_n=8, temporal_weighting=False, league_rho_fitting=False)
        first_day = datetime(2024, 1, 1)
        model.add(
            "Manchester United", "Brighton & Hove Albion", 2, 0,
            {"Div": "E0"}, match_date=first_day,
        )

        self.assertEqual(
            _team_state_key("E0", "Manchester United"),
            _team_state_key("E0", "Man Utd"),
        )
        self.assertEqual(
            _team_state_key("E0", "Brighton & Hove Albion"),
            _team_state_key("E0", "Brighton"),
        )
        self.assertNotEqual(
            _team_state_key("E0", "Manchester United"),
            _team_state_key("E0", "Manchester United B"),
        )

        prediction = model.predict(
            "Man Utd", "Brighton", "E0",
            match_date=datetime(2024, 1, 8),
        )
        self.assertEqual(prediction["model_support_status"], "team_supported")
        self.assertGreater(prediction["home_team_history_matches"], 0)
        self.assertGreater(prediction["away_team_history_matches"], 0)
        self.assertGreater(prediction["h2h_n"], 0)

    def test_batch_predicts_all_same_day_fixtures_before_ingesting_results(self):
        model = Engine(matrix_n=8, temporal_weighting=False, league_rho_fitting=False)
        untouched = Engine(matrix_n=8, temporal_weighting=False, league_rho_fitting=False)
        day = datetime(2024, 1, 1)
        batch = [
            {
                "home": "A", "away": "B", "home_goals": 5, "away_goals": 0,
                "league": "E0", "match_date": day, "row": {"Div": "E0"},
            },
            {
                "home": "C", "away": "D", "home_goals": 0, "away_goals": 4,
                "league": "E0", "match_date": day, "row": {"Div": "E0"},
            },
        ]
        predictions = model.learn_batch(batch)
        reference = untouched.predict("C", "D", "E0", match_date=day)
        for key in ("p1", "x", "p2", "over", "btts"):
            self.assertAlmostEqual(predictions[1][key], reference[key], places=12)
        self.assertEqual(model.trained_n, 2)
        self.assertEqual(model.match_count, 2)

    def test_chronological_trainer_uses_one_batch_per_calendar_day(self):
        history = {
            "E0": [[
                self.row("02/01/2024", "B", "C", 0, 1,
                         HC=5, AC=4, HY=2, AY=1, HR=0, AR=0),
                self.row("01/01/2024", "A", "B", 1, 0,
                         HC=6, AC=3, HY=1, AY=1, HR=0, AR=0),
            ]],
            "SP1": [[
                self.row("01/01/2024", "C", "D", 2, 0,
                         HC=4, AC=5, HY=2, AY=1, HR=0, AR=0),
            ]],
        }
        rows, skipped = prepare_completed_training_rows(history, ("E0", "SP1"))
        engine = Engine(matrix_n=8, temporal_weighting=False, league_rho_fitting=False)
        stats = HistoricalMatchStats()
        progress = []
        summary = train_engine_chronologically(
            engine, stats, rows, on_progress=lambda done, total: progress.append((done, total)),
        )
        self.assertEqual(summary, {"trained": 3, "days": 2, "skipped": 0})
        self.assertEqual(engine.trained_n, 3)
        self.assertEqual(stats.trained_matches["E0"], 2)
        self.assertEqual(stats.trained_matches["SP1"], 1)
        self.assertEqual(progress[-1], (3, 3))
        self.assertEqual(skipped["missing_or_invalid_score"], 0)

    def test_single_match_wrapper_keeps_legacy_leakage_safe_contract(self):
        model = Engine(matrix_n=8, temporal_weighting=False, league_rho_fitting=False)
        prediction = model.learn_step(
            "A", "B", 2, 1, {"Div": "E0"}, "E0",
            match_num=0, total=1, match_date=datetime(2024, 1, 1),
        )
        self.assertEqual(set(("p1", "x", "p2", "over", "btts")) & set(prediction), {
            "p1", "x", "p2", "over", "btts",
        })
        self.assertEqual(model.trained_n, 1)
        self.assertEqual(model.match_count, 1)


    def test_incremental_trainer_keeps_absolute_progress_counters(self):
        class CaptureEngine:
            def __init__(self):
                self.batches = []
                self.trained_n = 0
            def learn_batch(self, batch):
                self.batches.append([
                    (item["match_num"], item["total"]) for item in batch
                ])

        class CaptureStats:
            def learn_step(self, row, league):
                return True

        rows, _ = prepare_completed_training_rows({
            "E0": [[
                self.row("01/01/2024", "A", "B"),
                self.row("02/01/2024", "C", "D"),
                self.row("03/01/2024", "E", "F"),
            ]],
        }, ("E0",))
        engine, stats = CaptureEngine(), CaptureStats()
        train_engine_chronologically(engine, stats, rows[:2])
        train_engine_chronologically(
            engine, stats, rows[2:], processed_offset=2, total_count=3,
        )
        self.assertEqual(
            engine.batches, [[(0, 2)], [(1, 2)], [(2, 3)]],
        )
        self.assertEqual(engine.trained_n, 3)


if __name__ == "__main__":
    unittest.main()
