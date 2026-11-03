import unittest

from ui.tabs.scanner import _is_actionable_recommendation, _qualified_bet_cards


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
    def test_keeps_only_fully_qualified_named_bookmaker_1x2_bets(self):
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
        non_1x2 = valid_card()
        non_1x2["best"] = ("OU", "ТБ 2.5", 1.9, 0.12, 0.54, 10.0)

        self.assertTrue(_is_actionable_recommendation(eligible))
        self.assertEqual(
            _qualified_bet_cards([
                eligible, no_bookmaker, model_only, no_value,
                no_kelly, low_edge, low_ev, non_1x2,
            ]),
            [eligible],
        )

    def test_empty_or_invalid_input_produces_no_recommendation(self):
        self.assertFalse(_is_actionable_recommendation(None))
        self.assertFalse(_is_actionable_recommendation({"best": None}))
        self.assertEqual(_qualified_bet_cards(None), [])
        self.assertEqual(_qualified_bet_cards([]), [])


if __name__ == "__main__":
    unittest.main()
