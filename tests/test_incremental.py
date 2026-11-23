import unittest

from model.engine import Engine
from model.historical_match_stats import HistoricalMatchStats
from model.incremental import (
    IncrementalHistoryChanged, bring_engine_up_to_date, record_training_manifest,
)
from model.training import prepare_completed_training_rows, train_engine_chronologically


def row(day, home, away, hg=1, ag=0):
    return {
        "Date": day, "HomeTeam": home, "AwayTeam": away,
        "FTHG": hg, "FTAG": ag, "Div": "E0",
    }


class IncrementalTrainingTests(unittest.TestCase):
    @staticmethod
    def trained_engine(history):
        rows, _ = prepare_completed_training_rows(history, ("E0",))
        engine = Engine(matrix_n=8, temporal_weighting=False, league_rho_fitting=False)
        engine.match_stats = HistoricalMatchStats()
        train_engine_chronologically(engine, engine.match_stats, rows)
        record_training_manifest(engine, rows)
        return engine

    def test_append_only_history_trains_only_new_dates(self):
        initial = {"E0": [[row("01/01/2024", "A", "B"), row("02/01/2024", "C", "D")]]}
        engine = self.trained_engine(initial)
        extended = {"E0": [[
            row("01/01/2024", "A", "B"),
            row("02/01/2024", "C", "D"),
            row("03/01/2024", "E", "F"),
        ]]}
        logs = []
        self.assertEqual(bring_engine_up_to_date(engine, extended, ("E0",), logs), 1)
        self.assertEqual(engine.trained_n, 3)
        self.assertIn("+1 матчей", " ".join(logs))

    def test_corrected_old_score_forces_full_rebuild(self):
        initial = {"E0": [[row("01/01/2024", "A", "B"), row("02/01/2024", "C", "D")]]}
        engine = self.trained_engine(initial)
        corrected = {"E0": [[
            row("01/01/2024", "A", "B", hg=3, ag=0),
            row("02/01/2024", "C", "D"),
        ]]}
        with self.assertRaises(IncrementalHistoryChanged):
            bring_engine_up_to_date(engine, corrected, ("E0",), [])

    def test_late_match_on_trained_date_forces_full_rebuild(self):
        initial = {"E0": [[row("01/01/2024", "A", "B"), row("02/01/2024", "C", "D")]]}
        engine = self.trained_engine(initial)
        late = {"E0": [[
            row("01/01/2024", "A", "B"),
            row("02/01/2024", "C", "D"),
            row("02/01/2024", "E", "F"),
        ]]}
        with self.assertRaises(IncrementalHistoryChanged):
            bring_engine_up_to_date(engine, late, ("E0",), [])


if __name__ == "__main__":
    unittest.main()
