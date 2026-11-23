import unittest
from unittest.mock import patch

import ui.scanner_engine as scanner_engine
from ui.scanner_engine import _PRIORITY_DIVS, _validation_divisions


class ScannerEnginePriorityTests(unittest.TestCase):
    def test_validation_divisions_preserve_priority_order_and_training_scope(self):
        train_divs = ["E2", "D1", "E0", "OTHER", "SP1"]
        self.assertEqual(
            _validation_divisions(train_divs),
            ("E0", "SP1", "D1"),
        )

    def test_validation_divisions_handle_empty_and_missing_training_divisions(self):
        self.assertEqual(_validation_divisions([]), ())
        self.assertEqual(_validation_divisions(["E2", "D3"]), ())

    def test_seasonal_snapshot_is_reused_without_reloading_csv_archives(self):
        cache = {}
        with (
            patch.object(scanner_engine, "_PRIORITY_DIVS", ("E0",)),
            patch.object(scanner_engine, "DIV_TO_TSDB", {}),
            patch.object(scanner_engine, "_HISTORY_SNAPSHOT_CACHE", cache),
            patch.object(
                scanner_engine, "load_seasonal",
                return_value=[{"Date": "2026-01-01", "HomeTeam": "A", "AwayTeam": "B"}],
            ) as load_mock,
        ):
            first = scanner_engine._load_seasonal_history([], lambda *_args: None)
            second_logs = []
            second = scanner_engine._load_seasonal_history(
                second_logs, lambda *_args: None,
            )

        expected_calls = len(first["train_divs"]) * len(first["history_years"])
        self.assertEqual(load_mock.call_count, expected_calls)
        self.assertIs(first["history"], second["history"])
        self.assertTrue(any("памяти процесса" in line for line in second_logs))

    def test_cold_start_training_can_defer_heavy_market_benchmark_fail_closed(self):
        logs = []
        with (
            patch.object(
                scanner_engine, "_prepare_training_history",
                return_value=({}, ["E0"]),
            ),
            patch.object(
                scanner_engine, "prepare_completed_training_rows",
                return_value=([{}], {
                    "missing_date": 0, "missing_teams": 0,
                    "missing_or_invalid_score": 0, "invalid_row": 0,
                }),
            ),
            patch.object(
                scanner_engine, "train_engine_chronologically",
                return_value={"trained": 1, "days": 1},
            ),
            patch(
                "model.incremental.record_training_manifest"
            ) as manifest_mock,
            patch.object(
                scanner_engine, "_refresh_engine_market_validation"
            ) as validation_mock,
        ):
            engine = scanner_engine._train_engine(
                12, logs, lambda *_args: None,
                seasonal_snapshot={}, run_market_validation=False,
            )

        validation_mock.assert_not_called()
        manifest_mock.assert_called_once()
        self.assertEqual(engine.trained_n, 1)
        self.assertFalse(engine.winner_market_validation)
        self.assertFalse(engine.total_market_validation)
        self.assertFalse(engine.btts_market_validation)
        self.assertEqual(engine.stat_market_validation, {})
        self.assertEqual(engine.winner_market_validation_report["state"], "deferred")
        self.assertTrue(any("Value и автодопуск закрыты" in line for line in logs))

    def test_priority_validation_list_is_stable_and_unique(self):
        self.assertEqual(len(_PRIORITY_DIVS), 15)
        self.assertEqual(len(set(_PRIORITY_DIVS)), len(_PRIORITY_DIVS))
        self.assertEqual(_PRIORITY_DIVS[0], "E0")


if __name__ == "__main__":
    unittest.main()
