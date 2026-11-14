import unittest

from ui.tabs.stats import _scan_summary


class ScanSummaryTests(unittest.TestCase):
    def test_counts_valid_model_and_ai_forecasts_separately_from_text_opinions(self):
        cards = [
            {
                "p1": 0.50, "px": 0.25, "p2": 0.25,
                "market_odds": {"П1": 2.0, "X": 3.2, "П2": 4.0},
                "llm_opinion": "Оценка ИИ: П1 60% · X 20% · П2 20%",
            },
            {
                "p1": 0.40, "px": 0.30, "p2": 0.30,
                "llm_opinion": "Мнение ИИ без структурированной строки вероятностей",
            },
            {
                "p1": 0.80, "px": 0.40, "p2": 0.20,
                "llm_opinion": "",
            },
        ]

        summary = _scan_summary(cards)

        self.assertEqual(summary["matches"], 3)
        self.assertEqual(summary["forecasted"], 2)
        self.assertEqual(summary["ai_forecasted"], 1)
        self.assertEqual(summary["ai_opinions"], 2)
        self.assertEqual(summary["quoted_1x2"], 1)


if __name__ == "__main__":
    unittest.main()
