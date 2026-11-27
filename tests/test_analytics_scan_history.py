import unittest
from unittest.mock import patch

from ui.tabs import analytics


class ScanHistoryUiTests(unittest.TestCase):
    def test_render_scan_history_displays_saved_and_interrupted_runs(self):
        saved_runs = [
            {
                "scan_run_id": "run-complete",
                "status": "completed",
                "started_at": "2026-11-27T10:00:00+00:00",
                "date_from": "2026-11-27",
                "date_to_inclusive": "2026-11-29",
                "analysed_matches": 8,
                "saved_card_count": 8,
                "forecast_cards": 8,
                "source_rows": 12,
                "report": [],
            },
            {
                "scan_run_id": "run-interrupted",
                "status": "failed",
                "started_at": "2026-11-26T10:00:00+00:00",
                "date_from": "2026-11-26",
                "date_to_inclusive": "2026-11-27",
                "analysed_matches": 3,
                "saved_card_count": 2,
                "forecast_cards": 2,
                "source_rows": 5,
                "report": ["interrupted"],
            },
        ]
        with (
            patch.object(analytics, "ScanLedger") as ledger_factory,
            patch.object(analytics.st, "subheader"),
            patch.object(analytics.st, "dataframe") as dataframe,
            patch.object(analytics.st, "caption"),
        ):
            ledger_factory.return_value.runs.return_value = saved_runs

            analytics.render_scan_history()

        ledger_factory.assert_called_once_with(analytics.DB_FILE)
        ledger_factory.return_value.runs.assert_called_once_with(limit=10)
        rows = dataframe.call_args.args[0]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["Статус"], "🟢 Завершено")
        self.assertEqual(rows[1]["Статус"], "🔴 Прервано")
        self.assertEqual(rows[1]["Карточек сохранено"], 2)

    def test_render_scan_history_handles_an_empty_database(self):
        with (
            patch.object(analytics, "ScanLedger") as ledger_factory,
            patch.object(analytics.st, "subheader"),
            patch.object(analytics.st, "info") as info,
            patch.object(analytics.st, "dataframe") as dataframe,
        ):
            ledger_factory.return_value.runs.return_value = []

            analytics.render_scan_history()

        info.assert_called_once()
        dataframe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
