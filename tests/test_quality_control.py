import unittest
from datetime import datetime, timedelta, timezone

from model.quality_control import (
    assess_fixture_quality,
    check_proposed_position,
    portfolio_risk_report,
)


class DataQualityTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 11, 11, 12, tzinfo=timezone.utc)
        stamp = self.now.isoformat()
        self.card = {
            "match": "Home FC vs Away FC",
            "match_ru": "Home — Away",
            "date_iso": "2026-11-11",
            "time": "18:00",
            "kickoff_utc": (self.now + timedelta(hours=5)).isoformat(),
            "div": "E0",
            "league": "English Premier League",
            "market_odds": {"П1": 2.0, "X": 3.4, "П2": 4.2},
            "api_context": {
                "provider": "api-football",
                "api_fixture_id": 12345,
                "timestamp": stamp,
                "referee": {"name": "Referee One", "timestamp": stamp},
                "lineup": {
                    "timestamp": stamp,
                    "home": {
                        "confirmed": True,
                        "team_name": "Home FC",
                        "starting_xi": [{"name": f"H{i}"} for i in range(11)],
                    },
                    "away": {
                        "confirmed": True,
                        "team_name": "Away FC",
                        "starting_xi": [{"name": f"A{i}"} for i in range(11)],
                    },
                },
                "injuries": [],
                "injuries_is_confirmed": True,
                "injuries_timestamp": stamp,
            },
        }

    def test_complete_fixture_scores_only_completeness_not_probability(self):
        report = assess_fixture_quality(
            self.card, api_configured=True, now=self.now,
        )
        self.assertEqual(report["score"], 100)
        self.assertEqual(report["gender"], "men")
        self.assertEqual(report["statuses"]["API-Football"], "returned")
        self.assertEqual(report["statuses"]["Состав хозяев"], "confirmed")
        self.assertEqual(report["statuses"]["Состав гостей"], "confirmed")
        self.assertEqual(report["statuses"]["Травмы"], "responded_empty")
        self.assertIn("score", report)
        self.assertIn("это не доказывает отсутствие травм", " ".join(report["notes"]))

    def test_unconfigured_and_unmatched_are_distinct(self):
        no_key = assess_fixture_quality(
            self.card, api_configured=False, now=self.now,
        )
        self.assertEqual(no_key["statuses"]["API-Football"], "returned")
        empty_card = dict(self.card)
        empty_card["api_context"] = {}
        unmatched = assess_fixture_quality(
            empty_card, api_configured=True, now=self.now,
        )
        self.assertEqual(unmatched["statuses"]["API-Football"], "not_matched")

    def test_stale_data_and_conflicting_team_names_are_reported(self):
        stale = dict(self.card)
        old_stamp = (self.now - timedelta(days=2)).isoformat()
        stale["api_context"] = {
            **self.card["api_context"],
            "timestamp": old_stamp,
            "lineup": {
                "timestamp": old_stamp,
                "home": {
                    "confirmed": True, "team_name": "Wrong Team",
                    "starting_xi": [{"name": f"H{i}"} for i in range(11)],
                },
                "away": self.card["api_context"]["lineup"]["away"],
            },
        }
        report = assess_fixture_quality(
            stale, api_configured=True, now=self.now,
        )
        self.assertEqual(report["statuses"]["Свежесть данных"], "stale")
        self.assertEqual(report["statuses"]["Состав хозяев"], "stale")
        self.assertEqual(report["statuses"]["Противоречия"], "conflict")
        self.assertLessEqual(report["score"], 40)

    def test_missing_timestamp_is_not_claimed_fresh(self):
        empty = dict(self.card)
        empty["api_context"] = {}
        report = assess_fixture_quality(
            empty, api_configured=False, now=self.now,
        )
        self.assertEqual(report["statuses"]["Свежесть данных"], "unknown")

    def test_risk_guard_blocks_match_concentration(self):
        result = check_proposed_position(
            [{"match": "Home vs Away", "league": "E0", "date_iso": "2026-11-11",
              "stake": 40, "status": "pending"}],
            bank=960,
            candidate={
                "match": "Home vs Away", "league": "E0",
                "date_iso": "2026-11-11", "stake": 20,
            },
        )
        self.assertFalse(result["allowed"])
        self.assertTrue(any("Матч" in reason for reason in result["reasons"]))

    def test_loss_streak_prevents_increasing_the_next_stake(self):
        bets = [
            {"status": "lost", "stake": 10, "settled_at": "2026-11-01T12:00:00+00:00"},
            {"status": "lost", "stake": 12, "settled_at": "2026-11-02T12:00:00+00:00"},
            {"status": "lost", "stake": 12, "settled_at": "2026-11-03T12:00:00+00:00"},
        ]
        blocked = check_proposed_position(
            bets, bank=1000,
            candidate={"match": "A vs B", "league": "E0",
                       "date_iso": "2026-11-11", "stake": 13},
        )
        allowed = check_proposed_position(
            bets, bank=1000,
            candidate={"match": "A vs B", "league": "E0",
                       "date_iso": "2026-11-11", "stake": 12},
        )
        self.assertFalse(blocked["allowed"])
        self.assertTrue(any("серии из 3 поражений" in x for x in blocked["reasons"]))
        self.assertTrue(allowed["allowed"])

    def test_portfolio_report_groups_dependent_picks_and_drawdown(self):
        bets = [
            {"status": "pending", "match": "A vs B", "date_iso": "2026-11-11",
             "league": "E0", "market": "OU", "stake": 25},
            {"status": "pending", "match": "A vs B", "date_iso": "2026-11-11",
             "league": "E0", "market": "BTTS", "stake": 15},
        ]
        report = portfolio_risk_report(
            bets, bank=960,
            bank_history=[
                {"ts": "2026-11-01T00:00:00+00:00", "bank": 1000},
                {"ts": "2026-11-02T00:00:00+00:00", "bank": 900},
            ],
        )
        self.assertEqual(report["pending_count"], 2)
        self.assertEqual(len(report["correlated_clusters"]), 1)
        self.assertAlmostEqual(report["max_drawdown"], 0.1)


if __name__ == "__main__":
    unittest.main()
