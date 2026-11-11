import unittest

from ui.tabs.dashboard import context_summary, favorite, has_prices, match_name


class DashboardHelperTests(unittest.TestCase):
    def test_only_complete_sstats_prices_are_displayed_as_real_line(self):
        complete = {
            "odds_source": "sstats",
            "market_odds": {"П1": 2.1, "X": 3.3, "П2": 3.6},
        }
        partial = {
            "odds_source": "sstats",
            "market_odds": {"П1": 2.1, "X": 3.3},
        }
        invalid = {
            "odds_source": "model_only",
            "market_odds": {"П1": 2.1, "X": 3.3, "П2": 3.6},
        }
        self.assertTrue(has_prices(complete))
        self.assertFalse(has_prices(partial))
        self.assertFalse(has_prices(invalid))

    def test_context_summary_never_promotes_unknown_fields(self):
        card = {
            "api_context": {
                "lineup": {
                    "home": {"confirmed": True},
                    "away": {"confirmed": False},
                },
                "referee": {"name": None, "is_confirmed": False},
                "injuries_is_confirmed": False,
            }
        }
        summary = context_summary(card)
        self.assertFalse(summary["xi"])
        self.assertFalse(summary["referee"])
        self.assertFalse(summary["injuries"])

    def test_favorite_uses_valid_model_probability_distribution(self):
        card = {
            "p1": 0.71, "px": 0.18, "p2": 0.11,
            "home_team": "Home FC", "away_team": "Away FC",
            "games": 8,
        }
        result = favorite(card)
        self.assertIsNotNone(result)
        self.assertEqual(result["pick"], "П1")
        self.assertAlmostEqual(result["probability"], 0.71)

        malformed = dict(card, p1=71, px=18, p2=11)
        self.assertIsNone(favorite(malformed))

    def test_match_name_uses_existing_fixture_names(self):
        self.assertEqual(
            match_name({"home_team": "Home FC", "away_team": "Away FC",
                        "match_ru": "Дом — Гости"}),
            "Дом — Гости",
        )


if __name__ == "__main__":
    unittest.main()
