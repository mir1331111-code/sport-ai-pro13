import unittest
from datetime import datetime, timedelta, timezone

from model.analytics_quality import card_quality, portfolio_risk, source_state


class AnalyticsQualityTests(unittest.TestCase):
    def test_source_states_distinguish_missing_from_unavailable(self):
        self.assertEqual(source_state(configured=True, responded=True, has_data=False), "empty")
        self.assertEqual(source_state(configured=True, responded=False), "unavailable")
        self.assertEqual(source_state(configured=False), "not_configured")
        self.assertEqual(source_state(configured=True, responded=True, has_data=True, conflict=True), "conflict")

    def test_card_quality_never_invents_missing_context(self):
        report = card_quality({"match": "A vs B", "market_odds": {}})
        states = {x["name"]: x["state"] for x in report["checks"]}
        self.assertEqual(states["Состав хозяев"], "unavailable")
        self.assertEqual(states["Судья"], "unavailable")
        self.assertEqual(states["Коэффициенты 1X2"], "empty")
        self.assertEqual(report["completeness"], 0)

    def test_card_quality_marks_old_context_stale_and_odds_conflict(self):
        observed = (datetime.now(timezone.utc) - timedelta(hours=8)).isoformat()
        report = card_quality({"api_context": {"timestamp": observed},
                               "market_odds": {"H": 2},
                               "data_conflicts": ["1x2_odds"]})
        states = {x["name"]: x["state"] for x in report["checks"]}
        self.assertEqual(states["API-Football"], "stale")
        self.assertEqual(states["Коэффициенты 1X2"], "conflict")

    def test_card_quality_completeness_is_not_win_probability(self):
        report = card_quality({"api_context": {"lineup": {"home": {"starting_xi": list(range(11)), "confirmed": True},
                                                         "away": {"starting_xi": list(range(11)), "confirmed": True}},
                                                "referee": {"name": "Ref"}, "injuries": [], "injuries_is_confirmed": True},
                               "market_odds": {"H": 2, "D": 3, "A": 4}})
        self.assertGreater(report["completeness"], 0)
        self.assertLessEqual(report["completeness"], 100)

    def test_portfolio_risk_groups_fixture_and_league(self):
        bets = [
            {"status":"pending","stake":10,"fixture_id":"f1","league":"L1","date_iso":"2026-11-11"},
            {"status":"pending","stake":5,"fixture_id":"f1","league":"L1","date_iso":"2026-11-11"},
            {"status":"pending","stake":8,"fixture_id":"f2","league":"L2","date_iso":"2026-11-12"},
        ]
        report = portfolio_risk(bets, 100)
        self.assertEqual(report["open_count"], 3)
        self.assertEqual(report["exposure"], 23)
        self.assertEqual(report["fixture_top"][0], ("f1", 15))
        self.assertEqual(report["league_top"][0], ("L1", 15))
        self.assertEqual(report["exposure_pct"], 0.23)
        self.assertTrue(report["warning"])

    def test_drawdown_uses_settled_positions(self):
        bets = [
            {"status":"won","stake":10,"odds":2,"settled_at":"2026-01-01"},
            {"status":"lost","stake":10,"odds":2,"settled_at":"2026-01-02"},
        ]
        report = portfolio_risk(bets, 100)
        self.assertEqual(report["settled_count"], 2)
        self.assertGreater(report["max_drawdown"], 0)


if __name__ == "__main__":
    unittest.main()
