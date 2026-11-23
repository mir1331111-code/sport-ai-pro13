import unittest

from betting.gate_report import classify_card, near_misses, value_gates


def make_card(**changes):
    card = {
        "match": "Home FC vs Away FC",
        "match_ru": "Хозяева — Гости",
        "date_iso": "2026-11-23",
        "home_team": "Home FC",
        "away_team": "Away FC",
        "p1": 0.72, "px": 0.16, "p2": 0.12,
        "model_support_status": "team_supported",
        "context_quality": {"strong_value_eligible": False, "reason": "нет XI"},
        "independent_features": {
            "home_home_sample": 12, "away_away_sample": 11,
        },
        "winner_market_validation": False,
        "winner_market_validation_report": {"n": 430},
        "odds_source": "sstats",
        "market_odds": {"П1": 1.85, "X": 3.5, "П2": 4.3},
        "market_odds_source_by_pick": {
            "П1": "sstats", "X": "sstats", "П2": "sstats",
        },
        "bookmaker_by_pick": {
            "П1": "Book A", "X": "Book A", "П2": "Book A",
        },
        "best_by_market": {},
    }
    card.update(changes)
    return card


class GateReportTests(unittest.TestCase):
    def test_market_and_value_failures_have_clear_reasons(self):
        gates = value_gates(make_card(), "winner")
        self.assertFalse(gates[0].passed)
        self.assertIn("430/500", gates[0].detail)
        self.assertTrue(gates[1].passed)
        self.assertFalse(gates[2].passed)
        self.assertIn("нет XI", gates[2].detail)
        self.assertFalse(gates[3].passed)

    def test_short_history_is_provisional_and_reports_sample_counts(self):
        card = make_card(independent_features={
            "home_home_sample": 4, "away_away_sample": 11,
        })
        status, gates = classify_card(card, "winner")
        self.assertEqual(status, "provisional")
        self.assertIn("4/10", " ".join(gate.detail for gate in gates))

    def test_market_pass_does_not_skip_context_and_ev_gates(self):
        card = make_card(
            winner_market_validation=True,
            winner_market_validation_report={"n": 640},
        )
        gates = value_gates(card, "winner")
        self.assertTrue(gates[0].passed)
        self.assertTrue(gates[1].passed)
        self.assertFalse(gates[2].passed)
        self.assertFalse(gates[3].passed)

    def test_near_misses_ranks_candidates_by_missing_gate_count(self):
        cards = [
            make_card(p1=0.69, px=0.18, p2=0.13),
            make_card(
                p1=0.75, px=0.14, p2=0.11,
                independent_features={
                    "home_home_sample": 3, "away_away_sample": 4,
                },
                model_support_status="prior_only",
            ),
        ]
        result = near_misses(cards, "winner", limit=3)
        self.assertEqual(len(result), 2)
        self.assertTrue(result[0]["failed_gates"])
        self.assertEqual(result[0]["match"], "Хозяева — Гости")


    def test_fully_qualified_value_is_marked_approved(self):
        card = make_card(
            winner_market_validation=True,
            winner_market_validation_report={"n": 640},
            context_quality={"strong_value_eligible": True},
            best_by_market={
                "winner": {
                    "best": ("1X2", "П1", 2.0, 0.12, 0.72, 1.1),
                    "verdict": {
                        "real_odds": True, "value_passed": True,
                        "is_bet": True, "edge": 0.08,
                    },
                },
            },
        )
        status, gates = classify_card(card, "winner")
        self.assertEqual(status, "approved")
        self.assertTrue(all(gate.passed for gate in gates))


if __name__ == "__main__":
    unittest.main()
