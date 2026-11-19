import unittest
from datetime import datetime, timedelta

from data.fixture_time import app_timezone
from betting.ranking import rank_best_fixtures
from ui.tabs.dashboard import (
    _forecast_history_support, _forecast_rejection_reason,
    _grok_panel, _https_image_url, _match_table_row, _probability_health_note,
    _quality_panel, _reliability_svg, _sparkline_svg,
    context_summary, dashboard_bet_approved, favorite, has_prices, match_name,
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

    def test_model_favorite_can_select_draw_or_away_win(self):
        draw = favorite({
            "p1": 0.24, "px": 0.52, "p2": 0.24,
            "home_team": "Home FC", "away_team": "Away FC",
            "games": 8,
        })
        away = favorite({
            "p1": 0.20, "px": 0.25, "p2": 0.55,
            "home_team": "Home FC", "away_team": "Away FC",
            "games": 8,
        })
        self.assertEqual(draw["pick"], "X")
        self.assertEqual(away["pick"], "П2")

    def test_provisional_match_counts_are_visible_in_diagnostics_but_not_approved(self):
        now = datetime.now(app_timezone())
        kickoff = now + timedelta(hours=2)
        card = {
            "fixture_id": "sstats:provisional-1",
            "date_iso": kickoff.date().isoformat(),
            "time": kickoff.strftime("%H:%M"),
            "home_team": "Home FC",
            "away_team": "Away FC",
            "p1": 0.42, "px": 0.29, "p2": 0.29,
            "independent_features": {
                "home_home_sample": 0,
                "away_away_sample": 0,
            },
        }
        diagnostics = {}
        ranked = rank_best_fixtures(
            [card], now=now, limit=None, min_sample_games=5,
            min_probability=0.52, include_limited_history=True,
            diagnostics=diagnostics,
        )
        self.assertEqual(len(ranked), 1)
        self.assertFalse(ranked[0]["strict_eligible"])
        self.assertEqual(diagnostics["unique_candidates"], 1)
        self.assertEqual(diagnostics["strict_unique"], 0)
        self.assertEqual(diagnostics["provisional_unique"], 1)

    def test_dashboard_shows_only_strict_value_and_kelly_approved_bets(self):
        approved = {
            "_daily_strict_eligible": True,
            "odds_source": "sstats",
            "market_odds": {"П1": 2.2, "X": 3.1, "П2": 3.4},
            "verdict": {
                "pick": "X", "real_odds": True,
                "value_passed": True, "is_bet": True,
            },
        }
        self.assertTrue(dashboard_bet_approved(approved))
        self.assertFalse(dashboard_bet_approved(dict(
            approved, _daily_strict_eligible=False,
        )))
        self.assertFalse(dashboard_bet_approved({
            **approved,
            "verdict": {**approved["verdict"], "is_bet": False},
        }))
        self.assertFalse(dashboard_bet_approved({
            **approved,
            "market_odds": {"П1": 2.2, "X": 3.1},
        }))

    def test_repeated_probability_vector_is_flagged_as_prior_only_risk(self):
        cards = [
            {"p1": 0.42, "px": 0.29, "p2": 0.29}
            for _ in range(6)
        ]
        note = _probability_health_note(cards)
        self.assertIn("Повторяется один и тот же вектор 1X2", note)
        self.assertIn("42%", note)
        self.assertEqual(_probability_health_note(cards[:4]), "")

    def test_grok_panel_never_mislabels_groq_as_grok(self):
        panel = _grok_panel([], {"llm_provider": "Groq GPT-OSS 20B"})
        self.assertIn("Grok · независимый анализ", panel)
        self.assertIn("выбран Groq", panel)
        self.assertIn("XAI_API_KEY", panel)

    def test_grok_panel_renders_only_explicit_xai_analysis(self):
        panel = _grok_panel([{
            "home_team": "Home FC",
            "away_team": "Away FC",
            "p1": 0.55, "px": 0.25, "p2": 0.20,
            "independent_features": {
                "home_home_sample": 8,
                "away_away_sample": 8,
            },
            "deep_analysis_provider": "Grok (x.ai)",
            "deep_analysis_model": "grok-4.7",
            "deep_analysis": {
                "independent_pick": "X",
                "probabilities": {"П1": 40, "X": 35, "П2": 25},
                "match_assessment": "Независимая оценка матча.",
                "risks": "Составы не подтверждены.",
                "data_quality": "medium",
            },
        }], {"llm_provider": "Groq GPT-OSS 20B"})
        self.assertIn("40.0%", panel)
        self.assertIn("Grok: <b>X</b>", panel)
        self.assertIn("расходится с основной моделью", panel)
        self.assertIn("Независимая оценка матча.", panel)

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

    def test_badge_urls_upgrade_http_and_protocol_relative_sources(self):
        self.assertEqual(
            _https_image_url("http://assets.test/crest.png"),
            "https://assets.test/crest.png",
        )
        self.assertEqual(
            _https_image_url("//assets.test/crest.png"),
            "https://assets.test/crest.png",
        )
        self.assertEqual(_https_image_url("javascript:alert(1)"), "")

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

    def test_preview_forecast_rows_are_never_labeled_as_bets(self):
        card = {
            "home_team": "Forecast FC",
            "away_team": "Other FC",
            "p1": 0.42, "px": 0.29, "p2": 0.29,
            "market_odds": {"П1": 2.5, "X": 3.1, "П2": 2.8},
            "odds_source": "sstats",
            "verdict": {
                "real_odds": True, "value_passed": True, "is_bet": True,
            },
        }
        row = _match_table_row(card, preview_only=True)
        self.assertIn("ПРОГНОЗ · НЕ ДОПУЩЕН К СТАВКЕ", row)
        self.assertNotIn("VALUE · СТАВКА ДОПУЩЕНА", row)
        self.assertNotIn("class='neo-pick value'", row)

    def test_forecast_rejection_reason_explains_history_context_and_market(self):
        card = {
            "_daily_selection_note": (
                "Ознакомительный прогноз · дома 0/5; в гостях 0/5; "
                "нет выраженного фаворита (42% < 52%)"
            ),
            "context_quality": {
                "strong_value_eligible": False,
                "reason": "нет подтверждённого стартового XI и/или судьи",
            },
            "market_odds": {},
            "verdict": {"real_odds": False, "value_passed": False, "is_bet": False},
        }
        reason = _forecast_rejection_reason(card)
        self.assertIn("дома 0/5", reason)
        self.assertIn("52%", reason)
        self.assertIn("стартового XI", reason)
        self.assertIn("полной подтверждённой линии", reason)

    def test_prior_only_forecast_does_not_label_p1_as_a_pick(self):
        card = {
            "home_team": "Al Ain",
            "away_team": "Al-Dhafra",
            "date_iso": "2026-11-19",
            "time": "16:25",
            "p1": 0.421, "px": 0.289, "p2": 0.290,
            "independent_features": {
                "home_home_sample": 0,
                "away_away_sample": 0,
            },
            "market_odds": {"П1": 1.14, "X": 9.59, "П2": 14.70},
            "odds_source": "sstats",
            "verdict": {"real_odds": False, "value_passed": False, "is_bet": False},
            "_daily_home_sample": 0,
            "_daily_away_sample": 0,
            "_daily_selection_note": (
                "Ознакомительный прогноз · дома 0/5; в гостях 0/5; "
                "нет выраженного фаворита (42% < 52%)"
            ),
        }
        row = _match_table_row(card, preview_only=True)
        self.assertIn("PRIOR-ONLY · НЕ ВЫБИРАЕМ ИСХОД", row)
        self.assertNotIn("П1 · 42%", row)
        self.assertIn("42%", row)
        self.assertIn("29%", row)
        self.assertIn("Нет достаточной домашней/выездной истории", row)
        self.assertIn("ПРОГНОЗ · НЕ ДОПУЩЕН К СТАВКЕ", row)

    def test_evidence_backed_preview_can_show_p1_draw_or_p2(self):
        for probabilities, expected in (
            ({"p1": 0.60, "px": 0.22, "p2": 0.18}, "П1 · 60%"),
            ({"p1": 0.18, "px": 0.62, "p2": 0.20}, "X · 62%"),
            ({"p1": 0.18, "px": 0.22, "p2": 0.60}, "П2 · 60%"),
        ):
            card = {
                "home_team": "Alpha FC",
                "away_team": "Beta FC",
                **probabilities,
                "independent_features": {
                    "home_home_sample": 8,
                    "away_away_sample": 9,
                },
                "_daily_home_sample": 8,
                "_daily_away_sample": 9,
                "verdict": {"real_odds": False, "value_passed": False, "is_bet": False},
            }
            row = _match_table_row(card, preview_only=True)
            self.assertIn(expected, row)
            self.assertNotIn("PRIOR-ONLY", row)

    def test_forecast_history_support_uses_venue_specific_counts(self):
        self.assertEqual(
            _forecast_history_support({
                "_daily_home_sample": 6,
                "_daily_away_sample": 2,
                "independent_features": {
                    "home_home_sample": 10,
                    "away_away_sample": 10,
                },
            }),
            (6, 2, False),
        )
        self.assertEqual(
            _forecast_history_support({
                "independent_features": {
                    "home_home_sample": 5,
                    "away_away_sample": 5,
                },
            }),
            (5, 5, True),
        )

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
            "verdict": {"real_odds": True, "value_passed": True, "is_bet": True},
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
