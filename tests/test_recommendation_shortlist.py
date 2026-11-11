import unittest

from ui.tabs.scanner import (
    _is_actionable_recommendation, _qualified_bet_cards,
    _portfolio_auto_recommendation,
)


def valid_card():
    best = ("1X2", "П1", 2.24, 0.12, 0.54, 10.0)
    verdict = {
        "real_odds": True,
        "value_passed": True,
        "is_bet": True,
        "edge": 0.07,
    }
    winner = {"best": best, "verdict": verdict, "score": 0.16}
    return {
        "best": best,
        "best_by_market": {"winner": winner},
        "market_odds": {"П1": 2.24, "X": 3.40, "П2": 3.60},
        "bookmaker_by_pick": {"П1": "1xBet", "X": "1xBet", "П2": "1xBet"},
        "verdict": verdict,
        "best_odds_source": "SStats",
        "context_quality": {"strong_value_eligible": True, "level": "High"},
    }

class RecommendationShortlistTests(unittest.TestCase):
    def test_shortlist_contains_only_markets_with_full_portfolio_validation(self):
        winner = {
            "best": ("1X2", "П1", 2.20, 0.12, 0.60, 10.0),
            "verdict": {"real_odds": True, "value_passed": True, "is_bet": True, "edge": 0.08},
            "score": 0.16,
        }
        btts = {
            "best": ("BTTS", "BTTS Да", 2.10, 0.25, 0.70, 9.0),
            "verdict": {"real_odds": True, "value_passed": True, "is_bet": True, "edge": 0.20},
            "score": 0.30,
        }
        corners = {
            "best": ("STAT", "Угл ТБ 9.5", 2.10, 0.25, 0.70, 9.0),
            "verdict": {"real_odds": True, "value_passed": True, "is_bet": True, "edge": 0.20},
            "score": 0.30,
        }
        card = {
            "best_by_market": {"winner": winner, "btts": btts, "corners": corners},
            "bookmaker_by_pick": {"П1": "Book A", "X": "Book A", "П2": "Book A",
                                  "BTTS Да": "Book B", "BTTS Нет": "Book B",
                                  "Угл ТБ 9.5": "Book C", "Угл ТМ 9.5": "Book C"},
            "market_odds": {"П1": 2.20, "X": 3.50, "П2": 3.30},
            "context_quality": {"strong_value_eligible": True, "level": "High"},
        }
        result = _qualified_bet_cards([card])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["best"][1], "П1")

        # A fixture with only a statistical forecast must not be called
        # portfolio-ready, even though its own market tab may show the signal.
        stat_only = {
            "best_by_market": {"corners": corners},
            "bookmaker_by_pick": card["bookmaker_by_pick"],
        }
        self.assertEqual(_qualified_bet_cards([stat_only]), [])

    def test_total_shortlist_requires_exact_25_line_and_bookmaker_benchmark(self):
        total = {
            "best": ("OU", "ТБ 2.5", 2.10, 0.15, 0.62, 9.0),
            "verdict": {"real_odds": True, "value_passed": True, "is_bet": True, "edge": 0.10},
            "score": 0.18,
        }
        card = {
            "best_by_market": {"goals": total},
            "bookmaker_by_pick": {"ТБ 2.5": "Book A", "ТМ 2.5": "Book A"},
            "total_market_odds": {"ТБ 2.5": 2.10, "ТМ 2.5": 1.80},
            "total_market_validation": False,
            "context_quality": {"strong_value_eligible": True, "level": "High"},
        }
        self.assertEqual(_qualified_bet_cards([card]), [])
        card["total_market_validation"] = True
        self.assertEqual(_qualified_bet_cards([card])[0]["best"][1], "ТБ 2.5")


    def test_shortlist_rejects_missing_bookmaker_value_and_kelly(self):
        eligible = valid_card()
        no_bookmaker = valid_card()
        no_bookmaker["bookmaker_by_pick"] = {}
        model_only = valid_card()
        model_only["verdict"]["real_odds"] = False
        no_value = valid_card()
        no_value["verdict"]["value_passed"] = False
        no_kelly = valid_card()
        no_kelly["verdict"]["is_bet"] = False
        low_edge = valid_card()
        low_edge["verdict"]["edge"] = 0.049
        low_ev = valid_card()
        low_ev["best"] = ("1X2", "П1", 2.24, 0.049, 0.54, 10.0)
        low_ev["best_by_market"]["winner"]["best"] = low_ev["best"]

        result = _qualified_bet_cards([
            eligible, no_bookmaker, model_only, no_value,
            no_kelly, low_edge, low_ev,
        ])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["best"][1], "П1")

    def test_portfolio_autopick_allows_only_1x2_and_market_benchmarked_over25(self):
        winner = {
            "verdict": {"real_odds": True, "value_passed": True, "is_bet": True, "edge": 0.08},
            "best": ("1X2", "П1", 2.20, 0.12, 0.60, 10.0),
            "score": 0.16,
        }
        totals = {
            "verdict": {"real_odds": True, "value_passed": True, "is_bet": True, "edge": 0.10},
            "best": ("OU", "ТБ 2.5", 2.10, 0.15, 0.62, 9.0),
            "score": 0.18,
        }
        btts = {
            "verdict": {"real_odds": True, "value_passed": True, "is_bet": True, "edge": 0.20},
            "best": ("BTTS", "BTTS Да", 2.10, 0.25, 0.70, 9.0),
            "score": 0.30,
        }
        corners = {
            "verdict": {"real_odds": True, "value_passed": True, "is_bet": True, "edge": 0.20},
            "best": ("STAT", "Угл ТБ 9.5", 2.10, 0.25, 0.70, 9.0),
            "score": 0.30,
        }
        card = {
            "best_by_market": {"winner": winner, "goals": totals, "btts": btts, "corners": corners},
            "bookmaker_by_pick": {"П1": "Book A", "X": "Book A", "П2": "Book A",
                                   "ТБ 2.5": "Book A", "ТМ 2.5": "Book A"},
            "market_odds": {"П1": 2.20, "X": 3.50, "П2": 3.30},
            "total_market_odds": {"ТБ 2.5": 2.10, "ТМ 2.5": 1.80},
            "total_market_validation": False,
            "context_quality": {"strong_value_eligible": True, "level": "High"},
        }
        selected = _portfolio_auto_recommendation(card)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["best"][1], "П1")

        card["total_market_validation"] = True
        selected = _portfolio_auto_recommendation(card)
        self.assertEqual(selected["best"][1], "ТБ 2.5")

    def test_legacy_cards_without_confirmed_context_are_not_portfolio_ready(self):
        card = valid_card()
        card.pop("context_quality")
        self.assertEqual(_qualified_bet_cards([card]), [])
        self.assertIsNone(_portfolio_auto_recommendation(card))

    def test_portfolio_never_auto_adds_btts_or_statistics_markets(self):
        card = {
            "best_by_market": {
                "btts": {"best": ("BTTS", "BTTS Да", 2.1, 0.2, 0.7, 9.0),
                         "verdict": {"real_odds": True, "value_passed": True, "is_bet": True, "edge": 0.2}, "score": 0.3},
                "cards": {"best": ("STAT", "Карты ТБ 4.5", 2.1, 0.2, 0.7, 9.0),
                          "verdict": {"real_odds": True, "value_passed": True, "is_bet": True, "edge": 0.2}, "score": 0.3},
            },
            "bookmaker_by_pick": {"BTTS Да": "Book A", "BTTS Нет": "Book A",
                                   "Карты ТБ 4.5": "Book A", "Карты ТМ 4.5": "Book A"},
        }
        self.assertIsNone(_portfolio_auto_recommendation(card))

    def test_empty_or_invalid_input_produces_no_recommendation(self):
        self.assertFalse(_is_actionable_recommendation(None))
        self.assertFalse(_is_actionable_recommendation({"best": None}))
        self.assertEqual(_qualified_bet_cards(None), [])
        self.assertEqual(_qualified_bet_cards([]), [])


if __name__ == "__main__":
    unittest.main()
