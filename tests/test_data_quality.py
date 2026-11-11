import unittest
from datetime import datetime, timedelta, timezone

from model.data_quality import (
    competition_category_known, evaluate_fixture_quality,
)


class DataQualityTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 11, 11, 12, tzinfo=timezone.utc)

    def full_card(self):
        stamp = self.now.isoformat()
        return {
            "div": "W_EUROPA",
            "league": "UEFA Women's Europa Cup",
            "women": True,
            "gender_known": True,
            "date_iso": "2026-11-11",
            "time": "17:30",
            "kickoff_utc": "2026-11-11T17:30:00Z",
            "model_sample": {"home_home": 6, "away_away": 7},
            "market_odds": {"П1": 2.2, "X": 3.2, "П2": 3.4},
            "odds_source": "SStats",
            "api_football_configured": True,
            "api_context": {
                "api_status": "matched",
                "api_fixture_id": 123,
                "timestamp": stamp,
                "lineup": {
                    "home": {"confirmed": True, "starting_xi": [{}] * 11},
                    "away": {"confirmed": True, "starting_xi": [{}] * 11},
                    "is_confirmed": True,
                    "timestamp": stamp,
                    "status": "confirmed",
                },
                "referee": {
                    "name": "Referee A", "is_confirmed": True,
                    "timestamp": stamp, "status": "confirmed",
                },
                "injuries": [],
                "injuries_is_confirmed": True,
                "injuries_status": "empty",
                "injuries_timestamp": stamp,
            },
        }

    def test_complete_fixture_distinguishes_empty_injuries_from_confirmed_data(self):
        result = evaluate_fixture_quality(
            self.full_card(), api_configured=True, now=self.now,
        )
        self.assertEqual(result["api_status"], "confirmed")
        self.assertEqual(result["lineup_status"], "confirmed")
        self.assertEqual(result["sources"]["Травмы / дисквалификации"]["status"], "empty")
        self.assertGreaterEqual(result["score"], 90)
        self.assertTrue(result["women"])

    def test_failed_api_is_not_confused_with_successful_empty_response(self):
        card = self.full_card()
        card["api_context"]["api_status"] = "unavailable"
        card["api_context"]["lineup"]["status"] = "unavailable"
        card["api_context"]["lineup"]["home"]["confirmed"] = False
        card["api_context"]["lineup"]["away"]["confirmed"] = False
        card["api_context"]["lineup"]["home"]["starting_xi"] = []
        card["api_context"]["lineup"]["away"]["starting_xi"] = []
        card["api_context"]["referee"]["status"] = "unavailable"
        card["api_context"]["injuries_is_confirmed"] = False
        card["api_context"]["injuries_status"] = "unavailable"
        result = evaluate_fixture_quality(card, api_configured=True, now=self.now)
        self.assertEqual(result["api_status"], "unavailable")
        self.assertEqual(result["lineup_status"], "unknown")
        self.assertEqual(result["sources"]["Травмы / дисквалификации"]["status"], "unavailable")

    def test_stale_lineup_and_unmatched_fixture_are_explicit(self):
        card = self.full_card()
        old = (self.now - timedelta(hours=8)).isoformat()
        card["api_context"]["api_status"] = "no_match"
        card["api_context"]["lineup"]["timestamp"] = old
        result = evaluate_fixture_quality(card, api_configured=True, now=self.now)
        self.assertEqual(result["api_status"], "unmatched")
        self.assertEqual(result["lineup_status"], "stale")

    def test_missing_api_key_is_not_a_provider_outage(self):
        result = evaluate_fixture_quality(
            self.full_card(), api_configured=False, now=self.now,
        )
        self.assertEqual(result["api_status"], "not_configured")

    def test_unknown_generic_soccer_category_is_not_accepted(self):
        self.assertFalse(competition_category_known({
            "Div": "G", "League": "Soccer", "gender_known": False,
        }))
        self.assertTrue(competition_category_known({
            "Div": "W_EUROPA", "League": "UEFA Women's Europa Cup",
            "women": True, "gender_known": True,
        }))
        self.assertTrue(competition_category_known({
            "Div": "N1", "League": "Eredivisie",
        }))


if __name__ == "__main__":
    unittest.main()
