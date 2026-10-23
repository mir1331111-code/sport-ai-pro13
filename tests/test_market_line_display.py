import unittest

from ui.cards import render_verdict_card


class MarketLineDisplayTests(unittest.TestCase):
    def test_real_line_is_visible_even_when_no_bet_qualifies(self):
        card = {
            "match": "Manchester United vs Arsenal",
            "match_ru": "Manchester United — Arsenal",
            "date": "24.10",
            "div": "E0",
            "league": "Premier League",
            "games": 10,
            "fh": "W-D-W",
            "fa": "D-L-W",
            "odds_source": "prognozist",
            "market_odds": {"П1": 2.15, "X": 3.30, "П2": 3.60},
            "verdict": {
                "label": "Победа хозяев",
                "pick": "П1",
                "prob": 0.54,
                "odd": None,
                "real_odds": False,
                "confidence": "средняя",
                "conf_color": "#fbbf24",
                "reasons": [],
                "alternatives": [],
                "is_action": False,
            },
        }
        rendered = render_verdict_card(card, 0.60)
        self.assertIn("Реальная линия 1X2", rendered)
        self.assertIn("PROGNOZIST", rendered)
        self.assertIn("2.15", rendered)
        self.assertIn("3.30", rendered)
        self.assertIn("3.60", rendered)
        self.assertIn("Кэф:</b>", rendered)
        self.assertIn(">—</b>", rendered)

    def test_invalid_market_line_is_not_rendered(self):
        card = {
            "match": "Arsenal vs Chelsea",
            "date": "24.10",
            "div": "E0",
            "games": 1,
            "verdict": {
                "label": "Ничья", "prob": 0.3, "odd": None,
                "real_odds": False, "confidence": "низкая",
                "conf_color": "#f87171", "reasons": [],
                "alternatives": [], "is_action": False,
            },
            "odds_source": "prognozist",
            "market_odds": {"П1": 0, "X": 3.3, "П2": 3.6},
        }
        rendered = render_verdict_card(card, 0.6)
        self.assertNotIn("Реальная линия 1X2", rendered)


if __name__ == "__main__":
    unittest.main()
