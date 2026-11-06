import unittest

from ui.tabs.portfolio import _latest_value_signals
from ui.tabs.stats import _scan_summary


def _qualified_card(match="Alpha FC vs Beta FC"):
    return {
        "match": match,
        "match_ru": match.replace(" vs ", " — "),
        "league": "Test League",
        "date": "06.11 18:00",
        "best": ("1X2", "П1", 2.10, 0.12, 0.65, 20.0),
        "verdict": {
            "real_odds": True,
            "value_passed": True,
            "edge": 0.08,
        },
        "market_odds_source_by_pick": {"П1": "SStats"},
    }


class PortfolioSignalPreviewTests(unittest.TestCase):
    def test_shows_unadded_validated_signals_separately_from_bets(self):
        card = _qualified_card()
        rows = _latest_value_signals({"cards": [card], "bets": []})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Матч"], "Alpha FC — Beta FC")
        self.assertEqual(rows[0]["Исход"], "П1")
        self.assertAlmostEqual(rows[0]["Кэф"], 2.10)
        self.assertEqual(rows[0]["Источник"], "SStats")

    def test_excludes_unpriced_or_unvalidated_forecasts(self):
        unpriced = _qualified_card("Unpriced FC vs Other FC")
        unpriced["verdict"]["real_odds"] = False
        unvalidated = _qualified_card("Unvalidated FC vs Other FC")
        unvalidated["verdict"]["value_passed"] = False
        rows = _latest_value_signals({
            "cards": [_qualified_card(), unpriced, unvalidated],
            "bets": [],
        })
        self.assertEqual(len(rows), 1)

    def test_does_not_repeat_a_signal_already_in_portfolio(self):
        card = _qualified_card()
        rows = _latest_value_signals({
            "cards": [card],
            "bets": [{"match": card["match"], "pick": "П1", "status": "pending"}],
        })
        self.assertEqual(rows, [])

    def test_rejects_malformed_odds_and_probability(self):
        bad_odds = _qualified_card("Bad Odds FC vs Other FC")
        bad_odds["best"] = ("1X2", "П1", 1.0, 0.12, 0.65, 20.0)
        bad_probability = _qualified_card("Bad Probability FC vs Other FC")
        bad_probability["best"] = ("1X2", "П1", 2.1, 0.12, 1.2, 20.0)
        self.assertEqual(_latest_value_signals({
            "cards": [bad_odds, bad_probability], "bets": [],
        }), [])


class ScanSummaryTests(unittest.TestCase):
    def test_counts_fixtures_forecasts_quotes_value_signals_and_ai(self):
        cards = [
            {
                "p1": 0.60, "px": 0.25, "p2": 0.15,
                "market_odds": {"П1": 1.9, "X": 3.2, "П2": 4.1},
                "llm_opinion": "Independent opinion",
                "best": ("1X2", "П1", 1.9, 0.08, 0.65, 10),
                "verdict": {"real_odds": True, "value_passed": True},
            },
            {
                "p1": 0.30, "px": 0.35, "p2": 0.35,
                "total_market_odds": {"ТБ 2.5": 1.8},
                "best": None, "verdict": {},
            },
            {
                "p1": 0.55, "px": 0.25, "p2": 0.20,
                "market_odds": {}, "llm_opinion": "",
                "best": ("1X2", "П1", 2.0, 0.04, 0.55, 5),
                "verdict": {"real_odds": False, "value_passed": False},
            },
        ]
        self.assertEqual(_scan_summary(cards), {
            "matches": 3,
            "forecasted": 3,
            "quoted_1x2": 1,
            "quoted_any": 2,
            "value_signals": 1,
            "ai_opinions": 1,
        })

    def test_empty_scan_returns_zero_counts(self):
        self.assertEqual(_scan_summary([]), {
            "matches": 0, "forecasted": 0, "quoted_1x2": 0,
            "quoted_any": 0, "value_signals": 0, "ai_opinions": 0,
        })


if __name__ == "__main__":
    unittest.main()
