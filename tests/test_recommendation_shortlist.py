import unittest

from ui.tabs.scanner import (
    _is_actionable_recommendation, _qualified_bet_cards,
    _portfolio_auto_recommendation,
)


def valid_card():
    return {
        "best": ("1X2", "П1", 2.24, 0.12, 0.54, 10.0),
        "bookmaker_by_pick": {"П1": "1xBet"},
        "verdict": {
            "real_odds": True,
            "value_passed": True,
            "is_bet": True,
            "edge": 0.07,
        },
    }


class RecommendationShortlistTests(unittest.TestCase):
    def test_independent_market_recommendations_can_appear_in_shortlist(self):
        winner = valid_card()

        total = valid_card()
        total["best"] = ("OU", "ТБ 2.5", 2.10, 0.12, 0.54, 10.0)
        total["bookmaker_by_pick"] = {"ТБ 2.5": "Book B"}

        corners = valid_card()
        corners["best"] = ("STAT", "Угл ТБ 9.5", 2.10, 0.12, 0.54, 10.0)
        corners["bookmaker_by_pick"] = {"Угл ТБ 9.5": "Book C"}

        cards = valid_card()
        cards["best"] = ("STAT", "Карты ТМ 4.5", 2.10, 0.12, 0.54, 10.0)
        cards["bookmaker_by_pick"] = {"Карты ТМ 4.5": "Book D"}

        btts = valid_card()
        btts["best"] = ("BTTS", "BTTS Да", 2.10, 0.12, 0.54, 10.0)
        btts["bookmaker_by_pick"] = {"BTTS Да": "Book E"}

        self.assertTrue(_is_actionable_recommendation(winner))
        self.assertEqual(
            _qualified_bet_cards([winner, total, corners, cards, btts]),
            [winner, total, corners, cards, btts],
        )

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

        self.assertEqual(
            _qualified_bet_cards([
                eligible, no_bookmaker, model_only, no_value,
                no_kelly, low_edge, low_ev,
            ]),
            [eligible],
        )

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
        }
        selected = _portfolio_auto_recommendation(card)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["best"][1], "П1")

        card["total_market_validation"] = True
        selected = _portfolio_auto_recommendation(card)
        self.assertEqual(selected["best"][1], "ТБ 2.5")

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
