import unittest
from datetime import datetime, timezone, timedelta

from data.fixture_merge import merge_fixture_sources
from data.sources import _tsdb_league_meta
from model.data_quality import assess_fixture_quality
from model.engine import Engine
from model.market_validation import validate_market_rows
from model.portfolio_risk import build_portfolio_risk_report


class V14AnalyticsSuiteTests(unittest.TestCase):
    def test_womens_europa_cup_is_not_misclassified_as_national_euro(self):
        self.assertEqual(
            _tsdb_league_meta("UEFA Women's Europa Cup"),
            ("W_", "women", True, False),
        )
        self.assertEqual(
            _tsdb_league_meta("UEFA Women's EURO"),
            ("W_", "women_national", True, True),
        )
        self.assertEqual(
            _tsdb_league_meta("UEFA Europa League"),
            ("EL", "club", False, False),
        )

    def test_fixture_merge_detects_men_women_category_conflict(self):
        merged = merge_fixture_sources(
            [{
                "Date": "2026-11-11", "HomeTeam": "Shared FC", "AwayTeam": "Rival FC",
                "Div": "D1", "League": "Bundesliga", "category_verified": True,
                "source_provider": "ESPN",
            }],
            [{
                "Date": "2026-11-11", "HomeTeam": "Shared", "AwayTeam": "Rival",
                "Div": "W_", "League": "Women's Europa Cup", "women": True,
                "category_verified": True, "source_provider": "TheSportsDB",
            }],
        )
        self.assertEqual(len(merged), 1)
        self.assertTrue(any("category conflict" in item for item in merged[0]["source_conflicts"]))
        self.assertTrue(merged[0]["women"])

    def test_engine_keeps_men_women_history_and_calibration_separate(self):
        engine = Engine(matrix_n=8, temporal_weighting=False, league_rho_fitting=False)
        day = datetime(2026, 1, 1)
        engine.add("Shared", "Rival", 4, 0, {"Div": "D1"}, match_date=day)
        engine.add("Shared", "Rival", 0, 2, {"Div": "W_"}, match_date=day)
        self.assertEqual(engine.st["Shared"]["hs"], [4])
        self.assertEqual(engine.st["women::Shared"]["hs"], [0])
        self.assertEqual(engine.hg, [4])
        self.assertEqual(engine.namespace_hg["women"], [0])

        men = engine.predict("Shared", "Rival", "D1", match_date=day + timedelta(days=2))
        women = engine.predict("Shared", "Rival", "W_", match_date=day + timedelta(days=2))
        self.assertEqual(men["namespace"], "")
        self.assertEqual(women["namespace"], "women")
        self.assertEqual(women["home_history_n"], 1)
        self.assertEqual(women["home_venue_history_n"], 1)
        self.assertEqual(engine.form_str("Shared"), "В")
        self.assertEqual(engine.form_str("women::Shared"), "П")
        # The women's prediction must use women's fallback scoring rates, not men's.
        self.assertNotAlmostEqual(men["lams"][0], women["lams"][0], places=3)

    def test_data_quality_distinguishes_empty_injury_response_from_no_response(self):
        now = datetime(2026, 11, 11, 10, tzinfo=timezone.utc)
        stamp = now.isoformat()
        card = {
            "date_iso": "2026-11-11", "time": "18:30",
            "div": "D1", "league": "Bundesliga", "category_verified": True,
            "games": 20, "odds_source": "sstats",
            "market_odds": {"П1": 2.1, "X": 3.2, "П2": 3.8},
            "api_context": {
                "provider": "api-football", "provider_status": "CONNECTED",
                "fixture_match_status": "MATCHED", "api_fixture_id": 123,
                "timestamp": stamp,
                "referee": {"name": "Ref A", "is_confirmed": True, "status": "CONFIRMED", "timestamp": stamp},
                "lineup": {
                    "status": "CONFIRMED", "timestamp": stamp, "response_received": True,
                    "home": {"confirmed": True, "starting_xi": [{"name": f"H{i}"} for i in range(11)]},
                    "away": {"confirmed": True, "starting_xi": [{"name": f"A{i}"} for i in range(11)]},
                },
                "injuries": [], "injuries_is_confirmed": True,
                "injuries_status": "NO_DATA", "injuries_timestamp": stamp,
            },
        }
        report = assess_fixture_quality(card, now=now)
        statuses = {row["key"]: row["status"] for row in report["components"]}
        self.assertEqual(statuses["api_football"], "CONNECTED")
        self.assertEqual(statuses["home_lineup"], "CONFIRMED")
        self.assertEqual(statuses["away_lineup"], "CONFIRMED")
        self.assertEqual(statuses["referee"], "CONFIRMED")
        self.assertEqual(statuses["injuries"], "NO_DATA")
        self.assertEqual(statuses["odds"], "CONFIRMED")
        self.assertIsInstance(report["score"], int)
        self.assertLess(report["score"], 100)
        self.assertNotIn("probability", report)

    def test_data_quality_marks_stale_lineup_and_source_conflict(self):
        now = datetime(2026, 11, 11, 10, tzinfo=timezone.utc)
        old = (now - timedelta(hours=12)).isoformat()
        card = {
            "date_iso": "2026-11-11", "time": "18:00",
            "div": "W_", "league": "Women's Europa Cup", "women": True,
            "category_verified": True, "source_conflicts": ["category conflict: men vs women"],
            "api_context": {
                "provider": "api-football", "provider_status": "CONNECTED",
                "fixture_match_status": "MATCHED", "api_fixture_id": 2,
                "referee": {"status": "NO_DATA", "timestamp": old},
                "lineup": {
                    "status": "CONFIRMED", "timestamp": old,
                    "home": {"confirmed": True, "starting_xi": [{"name": "P"}] * 11},
                    "away": {"confirmed": True, "starting_xi": [{"name": "P"}] * 11},
                },
                "injuries": [], "injuries_is_confirmed": False,
            },
        }
        report = assess_fixture_quality(card, now=now)
        status_by_key = {item["key"]: item["status"] for item in report["components"]}
        self.assertTrue(report["conflict"])
        self.assertEqual(status_by_key["home_lineup"], "STALE")
        self.assertEqual(status_by_key["category"], "CONFLICT")

    def test_market_validation_needs_sample_and_paired_improvement(self):
        small = [
            {"market": "OU", "result_status": "won" if i % 2 == 0 else "lost",
             "probability": 0.9 if i % 2 == 0 else 0.1, "entry_odds": 2.0}
            for i in range(499)
        ]
        self.assertEqual(
            validate_market_rows(small)["markets"][0]["status"],
            "INSUFFICIENT_SAMPLE",
        )
        sufficient = [
            {"market": "OU", "result_status": "won" if i % 2 == 0 else "lost",
             "probability": 0.9 if i % 2 == 0 else 0.1, "entry_odds": 2.0}
            for i in range(500)
        ]
        report = validate_market_rows(sufficient)["markets"][0]
        self.assertEqual(report["status"], "VALIDATED_FOR_REVIEW")
        self.assertGreater(report["brier_improvement_ci_low_95"], 0)
        self.assertGreater(report["log_loss_improvement"], 0)

    def test_portfolio_risk_detects_match_league_day_and_losing_streak(self):
        bets = [
            {"status": "pending", "stake": 200, "fixture_id": "f1",
             "match": "A vs B", "league": "League 1", "date_iso": "2026-11-11", "market": "OU"},
            {"status": "pending", "stake": 150, "fixture_id": "f1",
             "match": "A vs B", "league": "League 1", "date_iso": "2026-11-11", "market": "BTTS"},
            {"status": "pending", "stake": 250, "fixture_id": "f2",
             "match": "C vs D", "league": "League 1", "date_iso": "2026-11-11", "market": "1X2"},
            {"status": "lost", "stake": 100, "settled_at": "2026-11-09T10:00:00Z"},
            {"status": "lost", "stake": 100, "settled_at": "2026-11-10T10:00:00Z"},
        ]
        report = build_portfolio_risk_report(
            bets, bank=10_000, initial_bank=10_000,
            bank_history=[{"bank": 9_000}, {"bank": 8_000}],
        )
        self.assertEqual(report["pending_count"], 3)
        self.assertAlmostEqual(report["total_exposure"], 600)
        self.assertEqual(report["loss_streak"], 2)
        self.assertTrue(report["same_match_overlaps"])
        self.assertTrue(any(row["limit_exceeded"] for row in report["match_rows"]))
        self.assertTrue(any(row["limit_exceeded"] for row in report["league_rows"]))
        self.assertTrue(any(row["limit_exceeded"] for row in report["day_rows"]))
        self.assertAlmostEqual(report["drawdown"]["max_drawdown"], 0.2)


if __name__ == "__main__":
    unittest.main()
