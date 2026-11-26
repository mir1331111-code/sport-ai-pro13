import unittest
from datetime import datetime, timedelta

from data.fixture_time import app_timezone
from betting.ranking import rank_best_fixtures
from ui.tabs.dashboard import (
    DASHBOARD_DEFAULT_PERIOD, DASHBOARD_PERIOD_OPTIONS,
    DASHBOARD_FORECAST_VISIBLE_LIMIT,
    _forecast_rejection_reason, _grok_panel, _https_image_url,
    _ai_comparison_html, _match_table_row, _probability_health_note,
    _dashboard_real_1x2_quotes, _dashboard_latest_ai_status,
    _quality_panel, _reliability_svg, _sparkline_svg,
    dashboard_recommendation_groups, dashboard_schedule_diagnostics,
    dashboard_primary_statuses_to_render, recommendation_group_html,
    dashboard_strong_events, strong_events_html,
    all_market_candidates_html, dashboard_all_market_candidates,
    context_summary, dashboard_bet_approved, favorite, has_prices, match_name,
)
from config import MIN_FAVORITE_HISTORY, DEFAULT_SCAN_HORIZON_DAYS


class DashboardHelperTests(unittest.TestCase):
    def test_dashboard_defaults_to_the_full_scan_horizon(self):
        self.assertEqual(DASHBOARD_DEFAULT_PERIOD, DASHBOARD_PERIOD_OPTIONS[-1])
        self.assertEqual(DASHBOARD_FORECAST_VISIBLE_LIMIT, 50)

    def test_dashboard_hides_empty_primary_panels_when_only_forecast_only_items_exist(self):
        groups = {
            "approved": [],
            "forecast": [],
            "near_strong": [],
            "provisional": [{"fixture_id": "thin-history"}],
            "background": [{"fixture_id": "lower-probability"}],
        }
        self.assertEqual(dashboard_primary_statuses_to_render(groups), ())

    def test_dashboard_keeps_nonempty_primary_categories_visible(self):
        groups = {
            "approved": [],
            "forecast": [{"fixture_id": "strong-but-not-value"}],
            "near_strong": [],
            "provisional": [{"fixture_id": "thin-history"}],
            "background": [],
        }
        self.assertEqual(
            dashboard_primary_statuses_to_render(groups),
            ("forecast",),
        )

    def test_dashboard_shows_only_the_three_requested_statuses(self):
        from data.fixture_time import app_timezone

        now = datetime(2026, 11, 23, 10, 0, tzinfo=app_timezone())

        def card(name, probability, *, approved=False, goals=False):
            value = {
                "fixture_id": f"fixture:{name}",
                "date_iso": "2026-11-24",
                "time": "18:00",
                "league": "Premier League",
                "home_team": name,
                "away_team": "Away FC",
                "p1": probability,
                "px": (1.0 - probability) * 0.55,
                "p2": (1.0 - probability) * 0.45,
                "model_support_status": "team_supported",
                "context_quality": {"strong_value_eligible": True},
                "independent_features": {
                    "home_home_sample": 14,
                    "away_away_sample": 12,
                },
                "winner_market_validation": bool(approved),
                "winner_market_validation_report": {"n": 520 if approved else 0},
                "odds_source": "sstats",
                "market_odds": {"П1": 1.85, "X": 3.5, "П2": 4.5},
                "market_odds_source_by_pick": {
                    "П1": "sstats", "X": "sstats", "П2": "sstats",
                },
                "bookmaker_by_pick": {
                    "П1": "Book A", "X": "Book A", "П2": "Book A",
                },
            }
            if approved:
                value["best_by_market"] = {
                    "winner": {
                        "best": ("1X2", "П1", 1.85, 0.12, probability, 10.0),
                        "verdict": {
                            "real_odds": True, "value_passed": True, "is_bet": True,
                            "prob": probability, "ev": 0.12, "edge": 0.08,
                            "stake": 10.0,
                        },
                    },
                }
            if goals:
                value.update({"lam_h": 1.0, "lam_a": 1.0})
            return value

        approved = card("Approved FC", .68, approved=True, goals=True)
        strong = card("Strong FC", .70)
        near = card("Near FC", .61)
        background = card("Background FC", .50)
        groups = dashboard_recommendation_groups(
            [approved, strong, near, background],
            now=now,
            period_end=now + timedelta(days=3),
        )

        self.assertEqual([item["card"]["home_team"] for item in groups["approved"]],
                         ["Approved FC"])
        self.assertEqual([item["card"]["home_team"] for item in groups["forecast"]],
                         ["Strong FC"])
        self.assertEqual([item["card"]["home_team"] for item in groups["near_strong"]],
                         ["Near FC"])
        all_visible = [
            item["card"]["home_team"]
            for group in groups.values() for item in group
        ]
        self.assertEqual([item["card"]["home_team"] for item in groups["background"]],
                         ["Background FC"])
        self.assertEqual(len(all_visible), len(set(all_visible)))

    def test_recommendation_groups_keep_real_market_price_for_nonapproved_matches(self):
        from data.fixture_time import app_timezone

        now = datetime(2026, 11, 23, 10, 0, tzinfo=app_timezone())
        strong = {
            "fixture_id": "priced-strong",
            "date_iso": "2026-11-24", "time": "18:00",
            "league": "Premier League",
            "home_team": "Strong FC", "away_team": "Away FC",
            "p1": .70, "px": .18, "p2": .12,
            "model_support_status": "team_supported",
            "context_quality": {"strong_value_eligible": True},
            "independent_features": {
                "home_home_sample": 14, "away_away_sample": 12,
            },
            "winner_market_validation": False,
            "winner_market_validation_report": {"n": 120},
            "odds_source": "SStats",
            "market_odds": {"П1": 1.91, "X": 3.40, "П2": 4.20},
            "market_odds_source_by_pick": {
                "П1": "SStats", "X": "SStats", "П2": "SStats",
            },
            "bookmaker_by_pick": {
                "П1": "Bet365", "X": "Bet365", "П2": "Bet365",
            },
        }
        groups = dashboard_recommendation_groups(
            [strong], now=now, period_end=now + timedelta(days=3),
        )
        self.assertEqual(len(groups["forecast"]), 1)
        item = groups["forecast"][0]
        self.assertEqual(item["pick"], "П1")
        self.assertEqual(item["odds"], 1.91)
        self.assertEqual(item["bookmaker"], "Bet365")
        self.assertEqual(item["odds_source"], "SStats")

        rendered = recommendation_group_html("forecast", groups["forecast"])
        self.assertIn("1.91", rendered)
        self.assertIn("Bet365", rendered)
        self.assertIn("SStats", rendered)
        self.assertIn("Реальные коэффициенты 1X2", rendered)
        self.assertIn("Цена выбранного рынка", rendered)

    def test_compact_recommendation_rows_keep_all_three_real_odds_badges_and_ai_context(self):
        from data.fixture_time import app_timezone

        now = datetime(2026, 11, 23, 10, 0, tzinfo=app_timezone())
        card = {
            "fixture_id": "full-recommendation-row",
            "date_iso": "2026-11-24", "time": "18:00",
            "league": "Premier League", "div": "E0",
            "home_team": "Home FC", "away_team": "Away FC",
            "home_badge": "https://assets.example/home.png",
            "away_badge": "https://assets.example/away.png",
            "league_badge": "https://assets.example/league.png",
            "p1": .70, "px": .18, "p2": .12,
            "model_support_status": "team_supported",
            "context_quality": {"strong_value_eligible": True},
            "independent_features": {
                "home_home_sample": 14, "away_away_sample": 12,
            },
            "winner_market_validation": False,
            "winner_market_validation_report": {"n": 120},
            "odds_source": "SStats",
            "market_odds": {"П1": 1.91, "X": 3.40, "П2": 4.20},
            "market_odds_source_by_pick": {
                "П1": "SStats", "X": "SStats", "П2": "SStats",
            },
            "bookmaker_by_pick": {
                "П1": "Bet365", "X": "Pinnacle", "П2": "Betano",
            },
            "deep_analysis_provider": "Groq GPT-OSS 120B",
            "deep_analysis_model": "openai/gpt-oss-120b",
            "deep_analysis": {
                "independent_pick": "П1",
                "probabilities": {"П1": 62, "X": 24, "П2": 14},
                "lineup_assessment": "PREDICTED XI · https://club.example/lineup",
                "injury_assessment": "Один игрок под вопросом",
                "referee_assessment": "Назначение пока не опубликовано",
                "match_assessment": "Фактор домашнего поля положительный",
                "risks": "Стартовый XI ещё не подтверждён",
                "data_quality": "medium",
            },
        }
        self.assertEqual(
            [item["odds"] for item in _dashboard_real_1x2_quotes(card)],
            [1.91, 3.40, 4.20],
        )
        groups = dashboard_recommendation_groups(
            [card], now=now, period_end=now + timedelta(days=3),
        )
        self.assertEqual(len(groups["forecast"]), 1)
        rendered = recommendation_group_html("forecast", groups["forecast"])
        for value in (
            "1.91", "3.40", "4.20",
            "Bet365", "Pinnacle", "Betano",
            "assets.example/home.png", "assets.example/away.png",
            "assets.example/league.png",
            "Groq GPT-OSS 120B", "openai/gpt-oss-120b",
            "PREDICTED XI", "Один игрок под вопросом",
            "Назначение пока не опубликовано",
        ):
            self.assertIn(value, rendered)

    def test_dashboard_latest_ai_status_extracts_most_recent_provider_reason(self):
        report = [
            "🏆 TOP MATCHES · AI STATUS: completed · Groq · 3/7 analyses",
            "intermediate log",
            "🏆 TOP MATCHES · AI STATUS: rate_limited · Groq · HTTP 429",
        ]
        self.assertEqual(
            _dashboard_latest_ai_status(report),
            "rate_limited · Groq · HTTP 429",
        )
        self.assertEqual(_dashboard_latest_ai_status([]), "")
        self.assertEqual(_dashboard_latest_ai_status(None), "")

    def test_dashboard_keeps_unknown_model_support_in_provisional_section(self):
        from data.fixture_time import app_timezone

        now = datetime(2026, 11, 23, 10, 0, tzinfo=app_timezone())
        unsupported = {
            "fixture_id": "unknown-support",
            "date_iso": "2026-11-24", "time": "18:00",
            "league": "Premier League",
            "home_team": "Unknown FC", "away_team": "Away FC",
            "p1": .61, "px": .23, "p2": .16,
            "model_support_status": "unknown",
            "independent_features": {
                "home_home_sample": 12, "away_away_sample": 12,
            },
            "odds_source": "SStats",
            "market_odds": {"П1": 1.9, "X": 3.5, "П2": 4.0},
        }
        groups = dashboard_recommendation_groups(
            [unsupported], now=now, period_end=now + timedelta(days=3),
        )
        self.assertEqual(len(groups["near_strong"]), 0)
        self.assertEqual(len(groups["approved"]), 0)
        self.assertEqual(len(groups["provisional"]), 1)
        self.assertIn("поддержка модели: unknown", groups["provisional"][0]["reason"])

    def test_dashboard_schedule_diagnostics_explain_missing_started_and_outside_window(self):
        now = datetime(2026, 11, 23, 10, 0, tzinfo=app_timezone())
        cards = [
            {
                "fixture_id": "future",
                "date_iso": "2026-11-23", "time": "12:00",
                "p1": .5, "px": .3, "p2": .2,
            },
            {
                "fixture_id": "missing-time",
                "date_iso": "2026-11-23",
                "p1": .5, "px": .3, "p2": .2,
            },
            {
                "fixture_id": "started",
                "date_iso": "2026-11-23", "time": "09:00",
                "p1": .5, "px": .3, "p2": .2,
            },
            {
                "fixture_id": "outside",
                "date_iso": "2026-11-26", "time": "12:00",
                "p1": .5, "px": .3, "p2": .2,
            },
        ]
        result = dashboard_schedule_diagnostics(
            cards, now=now, period_end=now + timedelta(days=1),
        )
        self.assertEqual(result["total"], 4)
        self.assertEqual(result["in_period"], 1)
        self.assertEqual(result["missing_kickoff"], 1)
        self.assertEqual(result["already_started"], 1)
        self.assertEqual(result["outside_period"], 1)
        self.assertEqual(result["valid_1x2"], 4)

    def test_dashboard_preserves_low_history_forecasts_as_non_actionable(self):
        now = datetime(2026, 11, 23, 10, 0, tzinfo=app_timezone())
        card = {
            "fixture_id": "thin-history",
            "date_iso": "2026-11-24",
            "time": "18:00",
            "home_team": "Thin FC",
            "away_team": "Away FC",
            "p1": .62, "px": .23, "p2": .15,
            "model_support_status": "team_supported",
            "independent_features": {
                "home_home_sample": 4,
                "away_away_sample": 6,
            },
        }
        groups = dashboard_recommendation_groups(
            [card], now=now, period_end=now + timedelta(days=2),
        )
        self.assertEqual(len(groups["provisional"]), 1)
        self.assertIn("4/10", groups["provisional"][0]["reason"])
        self.assertIn("Ставка заблокирована", groups["provisional"][0]["reason"])

    def test_model_only_prices_are_not_rendered_as_real_1x2_quotes(self):
        card = {
            "market_odds": {"П1": 1.9, "X": 3.5, "П2": 4.0},
            "odds_source": "model_only",
        }
        self.assertEqual(_dashboard_real_1x2_quotes(card), [])

    def test_approved_recommendation_uses_exact_value_price_not_recomputed_price(self):
        from data.fixture_time import app_timezone

        now = datetime(2026, 11, 23, 10, 0, tzinfo=app_timezone())
        approved = {
            "fixture_id": "approved-exact-price",
            "date_iso": "2026-11-24", "time": "18:00",
            "league": "Premier League",
            "home_team": "Approved FC", "away_team": "Away FC",
            "p1": .70, "px": .18, "p2": .12,
            "model_support_status": "team_supported",
            "context_quality": {"strong_value_eligible": True},
            "independent_features": {
                "home_home_sample": 14, "away_away_sample": 12,
            },
            "winner_market_validation": True,
            "winner_market_validation_report": {"n": 520},
            "odds_source": "SStats",
            "market_odds": {"П1": 1.91, "X": 3.40, "П2": 4.20},
            "market_odds_source_by_pick": {
                "П1": "SStats", "X": "SStats", "П2": "SStats",
            },
            "bookmaker_by_pick": {"П1": "Bet365", "X": "Bet365", "П2": "Bet365"},
            "best_by_market": {
                "winner": {
                    "best": ("1X2", "П1", 1.87, 0.12, .70, 10.0),
                    "verdict": {
                        "real_odds": True, "value_passed": True, "is_bet": True,
                        "prob": .70, "ev": .12, "edge": .08, "stake": 10.0,
                    },
                },
            },
        }
        groups = dashboard_recommendation_groups(
            [approved], now=now, period_end=now + timedelta(days=3),
        )
        self.assertEqual(len(groups["approved"]), 1)
        self.assertEqual(groups["approved"][0]["odds"], 1.87)

    def test_dashboard_recommendation_filter_omits_started_and_out_of_period_matches(self):
        from data.fixture_time import app_timezone

        now = datetime(2026, 11, 23, 10, 0, tzinfo=app_timezone())
        future = {
            "fixture_id": "future",
            "date_iso": "2026-11-23",
            "time": "18:00",
            "league": "Premier League",
            "home_team": "Future FC",
            "away_team": "Away FC",
            "p1": .70, "px": .18, "p2": .12,
            "model_support_status": "team_supported",
            "independent_features": {
                "home_home_sample": 12, "away_away_sample": 12,
            },
        }
        started = {**future, "fixture_id": "started", "time": "09:00"}
        later = {**future, "fixture_id": "later", "date_iso": "2026-11-27"}
        groups = dashboard_recommendation_groups(
            [future, started, later],
            now=now,
            period_end=now + timedelta(days=1),
        )
        visible = [
            item["card"]["fixture_id"]
            for group in groups.values() for item in group
        ]
        self.assertEqual(visible, ["future"])

    def test_recommendation_group_html_displays_only_requested_status_heading(self):
        rendered = recommendation_group_html("approved", [])
        self.assertIn("Одобрено к ставке", rendered)
        self.assertIn("Сейчас нет матчей", rendered)
        self.assertNotIn("Фоновый прогноз", rendered)
        self.assertIn("Фоновый прогноз", recommendation_group_html("background", []))

    def test_scan_defaults_to_three_days(self):
        self.assertEqual(DEFAULT_SCAN_HORIZON_DAYS, 3)

    def test_all_market_dashboard_shortlist_hides_unverified_candidates(self):
        events = [{
            "date": "23.11", "time": "18:00", "league": "Premier League",
            "match": "Home — Away", "family_label": "Угловые",
            "pick": "Угл ТМ 23.5", "probability": 1.0, "odds": 1.91,
            "bookmaker": "Book A", "history_matches": 120,
            "team_samples": 12, "value_verified": False,
            "status_label": "ПРОГНОЗ + КОЭФФИЦИЕНТ · НЕ VALUE",
        }, {
            "date": "23.11", "time": "19:00", "league": "Serie A",
            "match": "Home 2 — Away 2", "family_label": "Победа / 1X2",
            "pick": "П1", "probability": 0.72, "odds": 1.85,
            "bookmaker": "Book B", "history_matches": 0,
            "team_samples": 14, "value_verified": True,
            "status_label": "ПРОВЕРЕННЫЙ VALUE",
        }]
        rendered = all_market_candidates_html(events)
        self.assertNotIn("Угловые", rendered)
        self.assertNotIn("23.5", rendered)
        self.assertNotIn("1.91", rendered)
        self.assertIn("ПРОВЕРЕННЫЙ VALUE", rendered)
        self.assertIn("1.85", rendered)
        self.assertIn("Проверенных Value-сигналов: 1", rendered)

    def test_all_market_dashboard_shortlist_has_explicit_empty_state(self):
        rendered = all_market_candidates_html([])
        self.assertIn("независимую проверку именно этого рынка", rendered)
        self.assertIn("Непроверенные вероятности", rendered)

    def test_dashboard_history_floor_matches_strong_forecast_policy(self):
        self.assertEqual(MIN_FAVORITE_HISTORY, 10)
        now = datetime.now(app_timezone())
        kickoff = now + timedelta(hours=2)
        card = {
            "fixture_id": "sstats:eight-sample",
            "date_iso": kickoff.date().isoformat(),
            "time": kickoff.strftime("%H:%M"),
            "kickoff_utc": kickoff.astimezone(__import__("datetime").timezone.utc).isoformat(),
            "home_team": "Home FC", "away_team": "Away FC",
            "p1": 0.929, "px": 0.040, "p2": 0.031,
            "probability_calibrated": True,
            "independent_features": {
                "home_home_sample": 8, "away_away_sample": 8,
            },
        }
        diagnostics = {}
        ranked = rank_best_fixtures(
            [card], now=now, limit=None,
            min_sample_games=MIN_FAVORITE_HISTORY,
            min_probability=0.52,
            include_limited_history=True,
            diagnostics=diagnostics,
        )
        self.assertEqual(len(ranked), 1)
        self.assertFalse(ranked[0]["history_ready"])
        self.assertFalse(ranked[0]["strict_eligible"])

    def test_dashboard_best_matches_render_as_a_separate_forecast_rank(self):
        from ui.tabs.dashboard import best_matches_html

        items = [{
            "card": {
                "date": "22.11 19:00", "time": "19:00",
                "league": "Premier League",
                "home_team": "Home FC", "away_team": "Away FC",
            },
            "pick": "П1", "label": "Победа Home FC",
            "probability": 0.74, "probability_margin": 0.31,
            "home_sample": 12, "away_sample": 11,
            "quality_score": 84.5,
            "calibration_ready": True, "calibration_matches": 150,
        }]
        rendered = best_matches_html(items)
        self.assertIn("Home FC", rendered)
        self.assertIn("74.0%", rendered)
        self.assertIn("12/11", rendered)
        self.assertIn("84/100", rendered)
        self.assertIn("Да · n=150", rendered)

    def test_dashboard_best_matches_empty_state_does_not_invent_candidates(self):
        from ui.tabs.dashboard import best_matches_html

        rendered = best_matches_html([])
        self.assertIn("нет матчей", rendered)
        self.assertIn("prior-only", rendered)

    def test_strong_events_require_real_validated_price_probability_and_value(self):
        card = {
            "fixture_id": "sstats:strong-1",
            "date": "20.11 18:00",
            "league": "Premier League",
            "home_team": "Home FC",
            "away_team": "Away FC",
            "context_quality": {"strong_value_eligible": True},
            "independent_features": {
                "home_home_sample": 14,
                "away_away_sample": 12,
            },
            "market_odds_source_by_pick": {"П1": "sstats"},
            "bookmaker_by_pick": {"П1": "Book A"},
            "best_by_market": {
                "winner": {
                    "best": ("1X2", "П1", 1.85, 0.12, 0.68, 10.0),
                    "verdict": {
                        "real_odds": True, "value_passed": True, "is_bet": True,
                        "prob": 0.68, "ev": 0.12, "edge": 0.08, "stake": 10.0,
                    },
                },
            },
        }
        events = dashboard_strong_events([card])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["pick"], "П1")
        self.assertEqual(events[0]["odds"], 1.85)
        self.assertIn("1.85", strong_events_html(events))

    def test_strong_events_reject_low_odds_probability_and_insufficient_history(self):
        base = {
            "fixture_id": "sstats:strong-2",
            "context_quality": {"strong_value_eligible": True},
            "independent_features": {
                "home_home_sample": 12, "away_away_sample": 12,
            },
            "market_odds_source_by_pick": {"П1": "sstats"},
            "best_by_market": {
                "winner": {
                    "best": ("1X2", "П1", 1.69, 0.12, 0.68, 10.0),
                    "verdict": {
                        "real_odds": True, "value_passed": True, "is_bet": True,
                        "prob": 0.68, "ev": 0.12, "edge": 0.08, "stake": 10.0,
                    },
                },
            },
        }
        self.assertEqual(dashboard_strong_events([base]), [])
        low_probability = {
            **base,
            "best_by_market": {
                "winner": {
                    "best": ("1X2", "П1", 1.85, 0.12, 0.64, 10.0),
                    "verdict": {
                        "real_odds": True, "value_passed": True, "is_bet": True,
                        "prob": 0.64, "ev": 0.12, "edge": 0.08, "stake": 10.0,
                    },
                },
            },
        }
        self.assertEqual(dashboard_strong_events([low_probability]), [])
        thin_history = {
            **base,
            "independent_features": {
                "home_home_sample": 9, "away_away_sample": 12,
            },
        }
        self.assertEqual(dashboard_strong_events([thin_history]), [])

    def test_strong_events_reject_unverified_quote_or_context(self):
        base = {
            "fixture_id": "sstats:strong-3",
            "context_quality": {"strong_value_eligible": True},
            "independent_features": {
                "home_home_sample": 12, "away_away_sample": 12,
            },
            "market_odds_source_by_pick": {"П1": "model_only"},
            "best_by_market": {
                "winner": {
                    "best": ("1X2", "П1", 1.85, 0.12, 0.68, 10.0),
                    "verdict": {
                        "real_odds": True, "value_passed": True, "is_bet": True,
                        "prob": 0.68, "ev": 0.12, "edge": 0.08, "stake": 10.0,
                    },
                },
            },
        }
        self.assertEqual(dashboard_strong_events([base]), [])
        no_context = {**base, "context_quality": {"strong_value_eligible": False}}
        self.assertEqual(dashboard_strong_events([no_context]), [])

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

    def test_ai_comparison_panel_has_explicit_empty_state(self):
        rendered = _ai_comparison_html({
            "ai_forecasts": 12, "main_forecasts": 20, "paired_n": 0, "models": [],
        })
        self.assertIn("Пока нет парных результатов", rendered)
        self.assertIn("подтверждённым FT", rendered)

    def test_ai_comparison_panel_reports_paired_metrics_without_promoting_ai(self):
        rendered = _ai_comparison_html({
            "ai_forecasts": 12,
            "main_forecasts": 20,
            "paired_n": 4,
            "models": [{
                "model_version": "independent-llm-groq-gpt-oss-120b",
                "n": 4,
                "main": {"brier": 0.42},
                "ai": {"brier": 0.39},
                "brier_delta": -0.03,
                "log_loss_delta": -0.08,
                "accuracy_delta": 0.25,
            }],
        })
        self.assertIn("independent-llm-groq-gpt-oss-120b", rendered)
        self.assertIn("4 парных матчей", rendered)
        self.assertIn("-0.0300", rendered)
        self.assertIn("не автоматическое смешивание вероятностей", rendered)

    def test_provider_panel_shows_ai_to_validated_value_crosscheck(self):
        panel = _grok_panel([{
            "home_team": "Home FC",
            "away_team": "Away FC",
            "p1": 0.55, "px": 0.25, "p2": 0.20,
            "deep_analysis_provider": "Groq GPT-OSS 120B",
            "deep_analysis_model": "openai/gpt-oss-120b",
            "deep_analysis": {
                "independent_pick": "X",
                "probabilities": {"П1": 30, "X": 45, "П2": 25},
                "match_assessment": "Независимый анализ.",
            },
            "ai_crosscheck": {
                "validated_value_pick": "X",
                "validated_value_odds": 3.4,
                "ai_agrees_with_value_pick": True,
            },
        }], {"llm_provider": "Groq GPT-OSS 120B"})
        self.assertIn("Проверенный Value-селектор: X @ 3.40", panel)
        self.assertIn("AI совпадает с Value-выбором", panel)
        self.assertIn("не меняет вероятность, EV, Edge или размер ставки", panel)

    def test_provider_panel_identifies_groq_120b_without_calling_it_grok(self):
        panel = _grok_panel([], {"llm_provider": "Groq GPT-OSS 120B"})
        self.assertIn("Независимый AI-анализ", panel)
        self.assertIn("Groq", panel)
        self.assertIn("openai/gpt-oss-120b", panel)
        self.assertNotIn("XAI_API_KEY", panel)

    def test_provider_panel_hides_stale_xai_analysis_when_groq_is_selected(self):
        panel = _grok_panel([{
            "home_team": "Home FC",
            "away_team": "Away FC",
            "p1": 0.55, "px": 0.25, "p2": 0.20,
            "deep_analysis_provider": "Grok (x.ai)",
            "deep_analysis_model": "grok-4.7",
            "deep_analysis": {
                "independent_pick": "X",
                "probabilities": {"П1": 40, "X": 35, "П2": 25},
                "match_assessment": "Независимая оценка матча.",
                "risks": "Составы не подтверждены.",
                "data_quality": "medium",
            },
        }], {"llm_provider": "Groq GPT-OSS 120B"})
        self.assertNotIn("40.0%", panel)
        self.assertNotIn("Независимая оценка матча.", panel)

    def test_provider_panel_renders_matching_groq_120b_analysis(self):
        panel = _grok_panel([{
            "home_team": "Home FC",
            "away_team": "Away FC",
            "p1": 0.55, "px": 0.25, "p2": 0.20,
            "deep_analysis_provider": "Groq GPT-OSS 120B",
            "deep_analysis_model": "openai/gpt-oss-120b",
            "deep_analysis": {
                "independent_pick": "X",
                "probabilities": {"П1": 40, "X": 35, "П2": 25},
                "lineup_assessment": "PREDICTED XI · источник: https://club.example/lineup",
                "injury_assessment": "Травмирован один игрок · источник: https://club.example/injuries",
                "referee_assessment": "Судья не подтверждён.",
                "match_assessment": "Независимая оценка матча.",
                "risks": "Официальные составы пока не опубликованы.",
                "data_quality": "medium",
            },
        }], {"llm_provider": "Groq GPT-OSS 120B"})
        self.assertIn("40.0%", panel)
        self.assertIn("Groq GPT-OSS 120B", panel)
        self.assertIn("AI · Groq GPT-OSS 120B", panel)
        self.assertIn("PREDICTED XI", panel)
        self.assertIn("https://club.example/lineup", panel)
        self.assertIn("Травмирован один игрок", panel)
        self.assertIn("Судья не подтверждён", panel)
        self.assertIn("расходится с основной моделью", panel)
        self.assertIn("Независимая оценка матча.", panel)

    def test_provider_panel_renders_matching_xai_analysis(self):
        panel = _grok_panel([{
            "home_team": "Home FC",
            "away_team": "Away FC",
            "p1": 0.55, "px": 0.25, "p2": 0.20,
            "deep_analysis_provider": "Grok (x.ai)",
            "deep_analysis_model": "grok-4.7",
            "deep_analysis": {
                "independent_pick": "X",
                "probabilities": {"П1": 40, "X": 35, "П2": 25},
                "match_assessment": "Независимая оценка матча.",
                "risks": "Составы не подтверждены.",
                "data_quality": "medium",
            },
        }], {"llm_provider": "Grok (x.ai)"})
        self.assertIn("40.0%", panel)
        self.assertIn("AI · Grok (x.ai)", panel)

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


    def test_dashboard_rows_expose_specific_gate_failures(self):
        row = _match_table_row({
            "home_team": "Home FC", "away_team": "Away FC",
            "p1": 0.72, "px": 0.16, "p2": 0.12,
            "model_support_status": "team_supported",
            "independent_features": {
                "home_home_sample": 12, "away_away_sample": 11,
            },
            "context_quality": {
                "strong_value_eligible": False,
                "reason": "нет подтверждённого XI",
            },
            "winner_market_validation": False,
            "winner_market_validation_report": {"n": 420},
            "market_odds": {"П1": 1.9, "X": 3.4, "П2": 4.0},
            "odds_source": "sstats",
            "market_odds_source_by_pick": {
                "П1": "sstats", "X": "sstats", "П2": "sstats",
            },
            "bookmaker_by_pick": {
                "П1": "Book A", "X": "Book A", "П2": "Book A",
            },
        }, preview_only=True)
        self.assertIn("Почему такой статус", row)
        self.assertIn("420/500", row)
        self.assertIn("нет подтверждённого XI", row)


if __name__ == "__main__":
    unittest.main()
