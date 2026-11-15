import unittest
from datetime import date, datetime

from model.engine import Engine
from model.historical_match_stats import HistoricalMatchStats
from model.training import (
    prepare_completed_training_rows,
    plan_incremental_training,
    training_day_digests,
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

    def test_incremental_trainer_matches_full_chronological_replay(self):
        seed = [{
            "home": "Seed", "away": "Opponent", "home_goals": 1, "away_goals": 0,
            "league": "E0", "match_date": datetime(2023, 12, 31),
            "row": {"Div": "E0"},
        }]
        new_day = [
            {"home": "A", "away": "B", "home_goals": 1, "away_goals": 0,
             "league": "E0", "match_date": datetime(2024, 1, 1), "row": {"Div": "E0"}},
            {"home": "C", "away": "D", "home_goals": 0, "away_goals": 1,
             "league": "SP1", "match_date": datetime(2024, 1, 1), "row": {"Div": "SP1"}},
        ]
        incremental_engine = Engine(matrix_n=8, league_rho_fitting=False)
        incremental_stats = HistoricalMatchStats()
        train_engine_chronologically(incremental_engine, incremental_stats, seed)
        summary = train_engine_chronologically(
            incremental_engine, incremental_stats, new_day,
            start_index=1, total_rows=3,
        )

        full_engine = Engine(matrix_n=8, league_rho_fitting=False)
        full_stats = HistoricalMatchStats()
        train_engine_chronologically(full_engine, full_stats, seed + new_day)

        self.assertEqual(summary, {"trained": 3, "days": 1, "skipped": 0})
        self.assertEqual(incremental_engine.trained_n, 3)
        self.assertEqual(incremental_stats.trained_matches["E0"], 2)
        self.assertEqual(incremental_stats.trained_matches["SP1"], 1)
        incremental_prediction = incremental_engine.predict(
            "A", "C", "E0", match_date="2024-01-02",
        )
        full_prediction = full_engine.predict(
            "A", "C", "E0", match_date="2024-01-02",
        )
        for key in ("p1", "x", "p2", "over", "btts"):
            self.assertAlmostEqual(
                incremental_prediction[key], full_prediction[key], places=12,
            )

    def test_incremental_plan_accepts_only_unchanged_or_whole_new_days(self):
        from model.training import TRAINING_STATE_SCHEMA

        history = {
            "E0": [
                [self.row("01/01/2024", "A", "B", 1, 0)],
                [self.row("02/01/2024", "C", "D", 0, 1)],
            ]
        }
        prepared, _ = prepare_completed_training_rows(history, ("E0",))
        digests = training_day_digests(prepared)
        engine = Engine(matrix_n=8, temporal_weighting=False, league_rho_fitting=False)
        stats = HistoricalMatchStats()
        train_engine_chronologically(engine, stats, prepared)
        checkpoint = {
            "schema": TRAINING_STATE_SCHEMA,
            "day_digests": digests,
            "trained_rows": 2,
            "engine": engine,
        }
        self.assertEqual(plan_incremental_training(checkpoint, digests)["mode"], "unchanged")

        appended = dict(self.row("03/01/2024", "E", "F", 2, 0))
        updated, _ = prepare_completed_training_rows(
            {"E0": history["E0"] + [[appended]]}, ("E0",),
        )
        plan = plan_incremental_training(checkpoint, training_day_digests(updated))
        self.assertEqual(plan["mode"], "append")
        self.assertEqual(plan["start_index"], 2)

        same_day_addition = dict(self.row("02/01/2024", "New", "Match", 1, 1))
        changed, _ = prepare_completed_training_rows(
            {"E0": history["E0"] + [[same_day_addition]]}, ("E0",),
        )
        changed_plan = plan_incremental_training(checkpoint, training_day_digests(changed))
        self.assertEqual(changed_plan["mode"], "full")
        self.assertEqual(changed_plan["reason"], "historical_prefix_changed")

        corrected = dict(history["E0"][0][0])
        corrected["FTHG"] = 3
        corrected_history = {"E0": [[corrected], history["E0"][1][0:]]}
        corrected_rows, _ = prepare_completed_training_rows(corrected_history, ("E0",))
        self.assertEqual(
            plan_incremental_training(checkpoint, training_day_digests(corrected_rows))["mode"],
            "full",
        )

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


if __name__ == "__main__":
    unittest.main()
