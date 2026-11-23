import unittest

from model.context_model import assess_context_quality
from ui.tabs.scanner import _recheck_value_after_priority_context


def _confirmed_sstats_context():
    return {
        "provider": "sstats",
        "sstats_game_id": "12345",
        "sstats_event_id": "event-12345",
        "kickoff_at": "2026-11-24T18:00:00+00:00",
        "timestamp": "2026-11-24T16:00:00+00:00",
        "referee": {
            "name": "Referee One",
            "is_confirmed": True,
            "timestamp": "2026-11-24T16:00:00+00:00",
            "source_reliability": None,
        },
        "lineup": {
            "home": {"starting_xi": [f"Home Player {i}" for i in range(11)], "confirmed": True},
            "away": {"starting_xi": [f"Away Player {i}" for i in range(11)], "confirmed": True},
            "is_confirmed": True,
            "timestamp": "2026-11-24T16:00:00+00:00",
            "source_reliability": None,
        },
        "injuries": [],
        "injuries_is_confirmed": True,
        "injuries_timestamp": "2026-11-24T16:00:00+00:00",
    }


def _value_recheck_payload(*, model_history=True):
    return {
        "base_verdict": {
            "is_action": False,
            "real_odds": False,
            "prob": 0.80,
            "reasons": ["Ожидает подтверждения состава"],
        },
        "candidate_rows": [
            {"pick": "П1", "prob": 0.80, "label": "Победа хозяев"},
        ],
        "selection_odds": {"П1": 2.20, "X": 3.50, "П2": 5.00},
        "bank": 1000.0,
        "kelly_frac": 0.25,
        "min_prob": 0.50,
        "min_edge": 0.05,
        "min_ev": 0.05,
        "bookmaker_by_pick": {
            "П1": "Bet365", "X": "Bet365", "П2": "Bet365",
        },
        "market_validation_by_pick": {"П1": True, "X": True, "П2": True},
        "market_source_by_pick": {"П1": "sstats", "X": "sstats", "П2": "sstats"},
        "model_bet_history_ready": model_history,
    }


class PriorityContextValueRecheckTests(unittest.TestCase):
    def _card(self, *, model_history=True):
        old_context = {
            "provider": "sstats",
            "referee": {"name": None, "is_confirmed": False},
            "lineup": {
                "home": {"starting_xi": [], "confirmed": False},
                "away": {"starting_xi": [], "confirmed": False},
                "is_confirmed": False,
            },
            "injuries": [],
            "injuries_is_confirmed": False,
        }
        return {
            "fixture_id": "fixture-123",
            "sstats_game_id": "12345",
            "date_iso": "2026-11-24",
            "kickoff_utc": "2026-11-24T18:00:00+00:00",
            "home_team": "Home FC",
            "away_team": "Away FC",
            "p1": 0.80, "px": 0.15, "p2": 0.05,
            "api_context": old_context,
            "context_quality": assess_context_quality(old_context),
            "verdict": {"is_action": False, "real_odds": False},
            "best": None,
            "best_by_market": {},
            "_value_recheck": _value_recheck_payload(model_history=model_history),
        }

    def test_confirmed_sstats_context_rechecks_same_value_and_kelly_gates(self):
        card = self._card()
        original_probabilities = (card["p1"], card["px"], card["p2"])

        result = _recheck_value_after_priority_context(
            card, _confirmed_sstats_context(),
        )

        self.assertTrue(result["updated"])
        self.assertTrue(result["old_eligible"] is False)
        self.assertTrue(result["new_eligible"])
        self.assertTrue(result["rechecked"])
        self.assertTrue(result["approved"])
        self.assertIsNotNone(card["best"])
        self.assertEqual(card["best"][1], "П1")
        self.assertEqual(card["best"][2], 2.20)
        self.assertTrue(card["verdict"]["value_passed"])
        self.assertEqual(card["best_odds_source"], "sstats")
        self.assertTrue(card["context_quality"]["strong_value_eligible"])
        self.assertEqual((card["p1"], card["px"], card["p2"]), original_probabilities)
        self.assertNotIn("_value_recheck", card)

    def test_incomplete_official_context_does_not_unlock_value(self):
        card = self._card()
        refreshed = _confirmed_sstats_context()
        refreshed["lineup"]["away"]["starting_xi"] = ["Only one player"]
        refreshed["lineup"]["away"]["confirmed"] = False
        refreshed["lineup"]["is_confirmed"] = False

        result = _recheck_value_after_priority_context(card, refreshed)

        self.assertFalse(result["new_eligible"])
        self.assertFalse(result["rechecked"])
        self.assertIsNone(card["best"])
        self.assertFalse(card["context_quality"]["strong_value_eligible"])

    def test_confirmed_context_still_cannot_unlock_without_team_history(self):
        card = self._card(model_history=False)

        result = _recheck_value_after_priority_context(
            card, _confirmed_sstats_context(),
        )

        self.assertTrue(result["updated"])
        self.assertFalse(result["new_eligible"])
        self.assertFalse(result["rechecked"])
        self.assertIsNone(card["best"])

    def test_llm_or_browser_context_cannot_unlock_value(self):
        card = self._card()
        untrusted = _confirmed_sstats_context()
        untrusted["provider"] = "Groq GPT-OSS 120B"

        result = _recheck_value_after_priority_context(card, untrusted)

        self.assertFalse(result["updated"])
        self.assertEqual(result["reason"], "untrusted_context_provider")
        self.assertIsNone(card["best"])
        self.assertFalse(card["context_quality"]["strong_value_eligible"])


if __name__ == "__main__":
    unittest.main()
