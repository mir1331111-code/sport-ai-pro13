import unittest

from ui.tabs.dashboard import (
    _match_table_row, _quality_panel, _reliability_svg, _sparkline_svg,
    context_summary, favorite, has_prices, match_name,
)


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

    def test_fixture_rows_render_real_badges_and_competition_flags(self):
        row = _match_table_row({
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "home_badge": "https://assets.test/arsenal.png",
            "away_badge": "https://assets.test/chelsea.png",
            "div": "E0",
            "league": "Premier League",
            "p1": 0.5, "px": 0.25, "p2": 0.25,
            "market_odds": {"П1": 2.0, "X": 3.2, "П2": 4.0},
            "odds_source": "sstats",
        })
        self.assertIn("https://assets.test/arsenal.png", row)
        self.assertIn("https://assets.test/chelsea.png", row)
        self.assertIn("neo-league-flag", row)
        self.assertIn("Premier League", row)

    def test_fixture_row_displays_daily_event_quality_score(self):
        row = _match_table_row({
            "home_team": "Home FC", "away_team": "Away FC",
            "p1": 0.62, "px": 0.22, "p2": 0.16,
            "market_odds": {"П1": 1.90, "X": 3.40, "П2": 4.20},
            "_daily_strict_eligible": True,
            "_daily_quality_score": 76.4,
        })
        self.assertIn("БАЛЛ СОРТИРОВКИ · 76/100", row)
        self.assertIn("neo-pick-note rank-score", row)

    def test_fixture_row_distinguishes_forecast_from_value_approved_bet(self):
        no_line = _match_table_row({
            "home_team": "Forecast FC", "away_team": "Other FC",
            "p1": 0.70, "px": 0.18, "p2": 0.12,
            "market_odds": {},
            "verdict": {"real_odds": False, "value_passed": False},
        })
        self.assertIn("ТОЛЬКО ПРОГНОЗ", no_line)
        self.assertNotIn("VALUE · СТАВКА ДОПУЩЕНА", no_line)

        rejected = _match_table_row({
            "home_team": "Rejected FC", "away_team": "Other FC",
            "p1": 0.70, "px": 0.18, "p2": 0.12,
            "market_odds": {"П1": 1.9, "X": 3.5, "П2": 4.5},
            "odds_source": "sstats",
            "verdict": {"real_odds": False, "value_passed": False},
        })
        self.assertIn("НЕ ДОПУЩЕНО К СТАВКЕ", rejected)

        approved = _match_table_row({
            "home_team": "Approved FC", "away_team": "Other FC",
            "p1": 0.70, "px": 0.18, "p2": 0.12,
            "market_odds": {"П1": 1.9, "X": 3.5, "П2": 4.5},
            "odds_source": "sstats",
            "verdict": {"real_odds": True, "value_passed": True},
        })
        self.assertIn("VALUE · СТАВКА ДОПУЩЕНА", approved)

    def test_fixture_row_labels_provisional_daily_forecasts(self):
        row = _match_table_row({
            "home_team": "Home FC", "away_team": "Away FC",
            "p1": 0.44, "px": 0.30, "p2": 0.26,
            "market_odds": {},
            "_daily_strict_eligible": False,
            "_daily_home_sample": 1,
            "_daily_away_sample": 2,
            "_daily_selection_note": "Ознакомительный прогноз",
        })
        self.assertIn("neo-pick-note provisional", row)
        self.assertIn("Дом/гость: 1/2", row)

    def test_fixture_rows_keep_initial_fallback_when_badges_are_missing(self):
        row = _match_table_row({
            "home_team": "Home FC",
            "away_team": "Away FC",
            "p1": 0.5, "px": 0.25, "p2": 0.25,
            "market_odds": {},
        })
        self.assertIn(">H</span>", row)
        self.assertIn(">A</span>", row)

    def test_dashboard_sparkline_is_omitted_without_a_real_series(self):
        self.assertEqual(_sparkline_svg([]), "")
        self.assertEqual(_sparkline_svg([0.2]), "")
        self.assertIn("polyline", _sparkline_svg([0.2, 0.3, 0.25]))

    def test_reliability_bars_only_render_for_supported_bins(self):
        report = {
            "rows": [
                {
                    "outcome": "H", "bin_index": 2, "n": 8,
                    "mean_probability": 0.25, "observed_frequency": 0.375,
                },
                {
                    "outcome": "D", "bin_index": 2, "n": 4,
                    "mean_probability": 0.27, "observed_frequency": 0.25,
                },
                {
                    "outcome": "A", "bin_index": 7, "n": 2,
                    "mean_probability": 0.75, "observed_frequency": 0.5,
                },
            ],
            "n": 8,
        }
        chart = _reliability_svg(report)
        self.assertIn("neo-calibration-bar", chart)
        self.assertIn("Идеальная калибровка", chart)
        self.assertEqual(chart.count("neo-calibration-bar"), 1)

    def test_data_quality_gauge_does_not_claim_full_quality_without_cards(self):
        panel = _quality_panel([])
        self.assertIn("КАЧЕСТВО", panel)
        self.assertIn("<strong>—</strong>", panel)
        self.assertNotIn("100%", panel)


if __name__ == "__main__":
    unittest.main()
