import unittest

from ui.tabs.dashboard import (
    _match_table_row, _quality_panel, _reliability_svg, _sparkline_svg,
    approved_1x2_recommendation, context_summary, favorite, has_prices, match_name,
)


class DashboardHelperTests(unittest.TestCase):
    def test_only_complete_sstats_prices_are_displayed_as_real_line(self):
        complete = {
            "odds_source": "sstats",
            "market_odds": {"П1": 2.1, "X": 3.3, "П2": 3.6},
            "bookmaker_by_pick": {"П1": "Book A", "X": "Book A", "П2": "Book A"},
        }
        partial = {
            "odds_source": "sstats",
            "market_odds": {"П1": 2.1, "X": 3.3},
        }
        invalid = {
            "odds_source": "model_only",
            "market_odds": {"П1": 2.1, "X": 3.3, "П2": 3.6},
            "bookmaker_by_pick": {"П1": "Book A", "X": "Book A", "П2": "Book A"},
        }
        mixed_bookmakers = {
            "odds_source": "sstats",
            "market_odds": {"П1": 2.1, "X": 3.3, "П2": 3.6},
            "bookmaker_by_pick": {"П1": "Book A", "X": "Book B", "П2": "Book A"},
        }
        self.assertTrue(has_prices(complete))
        self.assertFalse(has_prices(partial))
        self.assertFalse(has_prices(invalid))
        self.assertFalse(has_prices(mixed_bookmakers))

    def _approved_card(self, pick):
        odds = {"П1": 2.10, "X": 3.40, "П2": 4.10}
        quoted_odd = odds[pick]
        return {
            "div": "E0",
            "odds_source": "sstats",
            "market_odds": odds,
            "bookmaker_by_pick": {
                "П1": "Book A", "X": "Book A", "П2": "Book A",
            },
            "context_quality": {"strong_value_eligible": True},
            "winner_market_validation": True,
            "winner_market_validation_scope": "competition_independent_holdout",
            "winner_market_validation_division": "E0",
            "market_holdout_id": "holdout-e0-v1",
            "best_by_market": {
                "winner": {
                    "best": ("1X2", pick, quoted_odd, 0.12, 0.52, 1.0),
                    "verdict": {
                        "real_odds": True, "value_passed": True,
                        "is_bet": True, "edge": 0.08,
                    },
                },
            },
        }

    def test_main_dashboard_approval_accepts_draw_and_away_outcomes(self):
        for pick in ("X", "П2"):
            with self.subTest(pick=pick):
                recommendation = approved_1x2_recommendation(
                    self._approved_card(pick)
                )
                self.assertIsNotNone(recommendation)
                self.assertEqual(recommendation["best"][1], pick)

    def test_main_dashboard_rejects_stale_or_mismatched_approved_price(self):
        card = self._approved_card("П2")
        card["best_by_market"]["winner"]["best"] = (
            "1X2", "П2", 2.10, 0.12, 0.52, 1.0,
        )
        self.assertIsNone(approved_1x2_recommendation(card))

    def test_main_dashboard_hides_forecast_only_or_global_only_matches(self):
        forecast_only = self._approved_card("П1")
        forecast_only["best_by_market"] = {}
        self.assertIsNone(approved_1x2_recommendation(forecast_only))

        global_only = self._approved_card("П1")
        global_only["winner_market_validation_scope"] = "global_research_only"
        global_only["winner_market_validation_division"] = ""
        global_only["market_holdout_id"] = ""
        self.assertIsNone(approved_1x2_recommendation(global_only))

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
            "odds_source": "sstats",
            "bookmaker_by_pick": {"П1": "Book A", "X": "Book A", "П2": "Book A"},
            "context_quality": {"strong_value_eligible": True},
            "winner_market_validation": True,
            "winner_market_validation_scope": "competition_independent_holdout",
            "winner_market_validation_division": "E0",
            "market_holdout_id": "holdout-e0-v1",
            "div": "E0",
            "verdict": {"real_odds": True, "value_passed": True, "is_bet": True},
        })
        self.assertIn("VALUE · СТАВКА ДОПУЩЕНА", approved)


    def test_fixture_row_uses_market_pass_probability_and_discloses_model_gap(self):
        row = _match_table_row({
            "home_team": "Fredericia",
            "away_team": "Hillerod",
            "p1": 0.42, "px": 0.29, "p2": 0.29,
            "market_odds": {"П1": 1.33, "X": 5.69, "П2": 7.93},
            "odds_source": "sstats",
            "bookmaker_by_pick": {"П1": "Book A", "X": "Book A", "П2": "Book A"},
            "_daily_pass_pick": "П1",
            "_daily_pass_probability": 0.69,
            "_daily_pass_source": "market_no_vig",
            "_daily_model_probability_for_pass": 0.42,
            "_daily_model_market_gap": -0.27,
            "_daily_home_sample": 0,
            "_daily_away_sample": 0,
            "_daily_league_sample": 0,
        })
        self.assertIn("П1 · 69%", row)
        self.assertIn("РЫНОК БЕЗ МАРЖИ", row)
        self.assertIn("модель: 42%", row)
        self.assertIn("лига 0/30", row)
        self.assertNotIn("VALUE · СТАВКА ДОПУЩЕНА", row)

    def test_row_does_not_show_approved_bet_for_global_only_validation(self):
        row = _match_table_row({
            "home_team": "Global FC", "away_team": "Other FC",
            "div": "E0",
            "p1": 0.70, "px": 0.18, "p2": 0.12,
            "market_odds": {"П1": 1.90, "X": 3.50, "П2": 4.50},
            "odds_source": "sstats",
            "bookmaker_by_pick": {"П1": "Book A", "X": "Book A", "П2": "Book A"},
            "context_quality": {"strong_value_eligible": True},
            "winner_market_validation": True,
            "winner_market_validation_scope": "global_research_only",
            "verdict": {"real_odds": True, "value_passed": True, "is_bet": True},
        })
        self.assertIn("НЕ ДОПУЩЕНО К СТАВКЕ", row)
        self.assertNotIn("VALUE · СТАВКА ДОПУЩЕНА", row)

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
