import unittest
from unittest.mock import MagicMock, patch

from ui.tabs import analytics


class LearningCenterSyncUiTests(unittest.TestCase):
    def test_manual_sync_button_sets_request_before_rerun(self):
        session_state = {}
        with (
            patch.object(analytics.st, "session_state", session_state),
            patch.object(analytics.st, "subheader"),
            patch.object(analytics.st, "button", return_value=True),
            patch.object(analytics.st, "rerun") as rerun,
        ):
            analytics.render_forecast_sync_status()

        self.assertTrue(session_state["_force_forecast_sync_requested"])
        rerun.assert_called_once()

    def test_sync_status_shows_settlement_queue_diagnostics(self):
        session_state = {
            "_forecast_ledger_sync_status": {
                "mode": "manual",
                "checked_at": "2026-11-27T11:30:00+00:00",
                "fixtures_checked": 4,
                "forecasts_settled": 3,
                "pending_forecasts": 21,
                "eligible_fixtures_remaining": 2,
                "invalid_fixture_ids": 1,
                "not_due_grace_fixtures": 5,
                "future_fixtures": 6,
                "age_limited_fixtures": 0,
                "retry_deferred_fixtures": 2,
                "not_found_fixtures": 1,
                "non_ft_fixtures": 0,
                "errors": 0,
                "result_sources": {"test-provider": 3},
            }
        }
        columns = [MagicMock() for _ in range(4)]
        with (
            patch.object(analytics.st, "session_state", session_state),
            patch.object(analytics.st, "subheader"),
            patch.object(analytics.st, "button", return_value=False),
            patch.object(analytics.st, "columns", return_value=columns),
            patch.object(analytics.st, "caption") as caption,
            patch.object(analytics.st, "warning") as warning,
        ):
            analytics.render_forecast_sync_status()

        columns[0].metric.assert_called_once_with("Матчей проверено", 4)
        columns[1].metric.assert_called_once_with("Прогнозов закрыто", 3)
        columns[2].metric.assert_called_once_with("Ожидают FT", 21)
        warning.assert_called_once()
        captions = " ".join(str(call.args[0]) for call in caption.call_args_list)
        self.assertIn("невалидный kickoff/ID: 1", captions)
        self.assertIn("test-provider: 3", captions)


class EmptyRiskPortfolioUiTests(unittest.TestCase):
    def test_empty_portfolio_is_explained_without_claiming_positions_exist(self):
        session_state = {"data": {"bets": [], "bank": 1000.0}}
        columns = [MagicMock() for _ in range(4)]
        with (
            patch.object(analytics.st, "session_state", session_state),
            patch.object(analytics.db, "fetch_bets", return_value=[]),
            patch.object(analytics.db, "get_meta", return_value=None),
            patch.object(analytics.db, "set_meta"),
            patch.object(analytics.st, "columns", return_value=columns),
            patch.object(analytics.st, "caption"),
            patch.object(analytics.st, "subheader"),
            patch.object(analytics.st, "dataframe"),
            patch.object(analytics.st, "info") as info,
        ):
            analytics.render_portfolio_risk()

        messages = " ".join(str(call.args[0]) for call in info.call_args_list)
        self.assertIn("нет зарегистрированных ставок", messages.casefold())


if __name__ == "__main__":
    unittest.main()
