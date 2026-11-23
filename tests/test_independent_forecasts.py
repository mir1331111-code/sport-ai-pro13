import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
from types import SimpleNamespace

from model.forecast_ledger import ForecastLedger, MarketForecastLedger
from model.independent_forecasts import (
    compare_forecast_rows,
    historical_features,
    historical_xg_features,
    parse_llm_probabilities,
    structured_ai_probabilities,
    build_ai_crosscheck,
    record_scan_forecasts,
    settle_due_forecasts,
)


class IndependentForecastTests(unittest.TestCase):
    def test_paired_comparison_reports_error_correlation(self):
        from model.independent_forecasts import compare_forecast_rows

        rows = []
        actuals = ("H", "D", "A", "H", "A", "D")
        main_probs = (
            {"H": .70, "D": .20, "A": .10},
            {"H": .20, "D": .60, "A": .20},
            {"H": .15, "D": .20, "A": .65},
            {"H": .45, "D": .35, "A": .20},
            {"H": .25, "D": .25, "A": .50},
            {"H": .25, "D": .50, "A": .25},
        )
        ai_probs = (
            {"H": .65, "D": .25, "A": .10},
            {"H": .25, "D": .55, "A": .20},
            {"H": .20, "D": .20, "A": .60},
            {"H": .40, "D": .40, "A": .20},
            {"H": .30, "D": .25, "A": .45},
            {"H": .30, "D": .45, "A": .25},
        )
        for index, actual in enumerate(actuals):
            shared = {
                "fixture_id": f"fixture-{index}",
                "result": actual,
                "kickoff_at": f"2026-11-{index+1:02d}T18:00:00+00:00",
            }
            rows.extend([
                {**shared, "model_version": "main-model-v1",
                 "forecast_at": f"2026-11-{index+1:02d}T10:00:00+00:00",
                 "probabilities": main_probs[index]},
                {**shared, "model_version": "independent-llm-test",
                 "forecast_at": f"2026-11-{index+1:02d}T10:01:00+00:00",
                 "probabilities": ai_probs[index]},
            ])
        report = compare_forecast_rows(rows)["best_paired"]
        self.assertEqual(report["error_correlation"]["n"], len(actuals))
        self.assertIsNotNone(report["error_correlation"]["log_loss"])
        self.assertIsNotNone(report["error_correlation"]["brier"])

    def test_llm_probabilities_are_extracted_and_normalized(self):
        text = (
            "САМОСТОЯТЕЛЬНЫЙ ПРОГНОЗ ИИ: ничья (X). "
            "Оценка ИИ: П1 30% · X 45% · П2 25%. Риски: мало данных."
        )
        probs = parse_llm_probabilities(text)
        self.assertEqual(probs, {"H": 0.3, "D": 0.45, "A": 0.25})
        self.assertIsNone(parse_llm_probabilities("ИИ согласен с прогнозом"))

    def test_llm_features_use_observed_venue_goals_not_lambdas(self):
        engine = SimpleNamespace(st={
            "Home": {"hs": [1, 2, 3], "hc": [0, 1, 2]},
            "Away": {"as": [0, 1, 2, 3], "ac": [2, 2, 1, 1]},
        })
        features = historical_features(engine, "Home", "Away", limit=8)
        self.assertAlmostEqual(features["home_scored_at_home_avg"], 2.0)
        self.assertAlmostEqual(features["home_conceded_at_home_avg"], 1.0)
        self.assertAlmostEqual(features["away_scored_away_avg"], 1.5)
        self.assertAlmostEqual(features["away_conceded_away_avg"], 1.5)
        self.assertEqual(features["home_home_sample"], 3)
        self.assertEqual(features["away_away_sample"], 4)
        self.assertAlmostEqual(features["simple_baseline_lambda_home"], 1.75)
        self.assertAlmostEqual(features["simple_baseline_lambda_away"], 1.25)
        self.assertAlmostEqual(features["simple_baseline_lambda_total"], 3.0)
        self.assertGreater(features["simple_baseline_over_25"], 0.5)

    def test_history_support_counts_more_observations_than_average_window(self):
        # The display average uses the latest 8 games, while eligibility must
        # still see all pre-match venue observations needed by the 10-game gate.
        dates = [f"2026-11-{day:02d}" for day in range(3, 17)]
        engine = SimpleNamespace(st={
            ("E0", "Home FC"): {
                "hs": list(range(3, 17)),
                "hc": list(range(4, 18)),
                "hs_dates": dates,
                "hc_dates": dates,
            },
            ("E0", "Away FC"): {
                "as": list(range(5, 19)),
                "ac": list(range(6, 20)),
                "as_dates": dates,
                "ac_dates": dates,
            },
        })
        from model.temporal import weighted_mean

        cutoff = datetime(2026, 11, 15, 18, 0)
        result = historical_features(
            engine, "Home FC", "Away FC", limit=8,
            league="E0", match_date=cutoff,
        )
        # Twelve earlier observations count; matchday 15 and future day 16 do not.
        self.assertEqual(result["home_home_sample"], 12)
        self.assertEqual(result["away_away_sample"], 12)
        # Averages retain the existing eight-observation, time-weighted window.
        expected_home = weighted_mean(
            list(range(7, 15)), dates[4:12], cutoff, half_life_days=180.0,
        )
        expected_away = weighted_mean(
            list(range(9, 17)), dates[4:12], cutoff, half_life_days=180.0,
        )
        self.assertAlmostEqual(result["home_scored_at_home_avg"], expected_home)
        self.assertAlmostEqual(result["away_scored_away_avg"], expected_away)

    def test_llm_historical_features_match_normalized_names_in_same_competition(self):
        engine = SimpleNamespace(st={
            ("E0", "Home"): {
                "hs": [1, 2, 3, 2, 1], "hc": [0, 1, 1, 0, 0],
                "hs_xg": [1.1, 1.2, 1.4, 1.3, 1.0],
                "hs_xga": [0.5, 0.7, 0.8, 0.6, 0.4],
                "hs_xg_dates": ["2026-01-01"] * 5,
                "hs_xga_dates": ["2026-01-01"] * 5,
            },
            ("E0", "Away"): {
                "as": [1, 0, 1, 2, 1], "ac": [2, 1, 1, 0, 2],
                "as_xg": [1.0, 0.9, 1.1, 1.3, 1.2],
                "as_xga": [1.8, 1.4, 1.1, 0.8, 1.6],
                "as_xg_dates": ["2026-01-01"] * 5,
                "as_xga_dates": ["2026-01-01"] * 5,
            },
            ("W_E0", "Home"): {"hs": [4, 4, 4], "hc": [0, 0, 0]},
            ("W_E0", "Away"): {"as": [0, 0, 0], "ac": [4, 4, 4]},
        })
        features = historical_features(
            engine, "Home FC", "Away SC", league="E0",
        )
        self.assertEqual(features["home_home_sample"], 5)
        self.assertEqual(features["away_away_sample"], 5)
        self.assertAlmostEqual(features["home_scored_at_home_avg"], 1.8)
        xg = historical_xg_features(engine, "Home FC", "Away SC", league="E0")
        self.assertEqual(xg["provider"]["home_xg_samples"], 5)
        self.assertEqual(xg["provider"]["away_xg_samples"], 5)
        self.assertAlmostEqual(xg["provider"]["home_xg_for_avg"], 1.2)

    def test_historical_features_join_team_aliases_used_by_upcoming_fixtures(self):
        from model.engine import Engine

        engine = Engine(matrix_n=8, temporal_weighting=False, league_rho_fitting=False)
        engine.add(
            "Manchester United", "Brighton & Hove Albion", 2, 0,
            {"Div": "E0"}, match_date=datetime(2026, 11, 1),
        )
        features = historical_features(
            engine, "Man Utd", "Brighton", league="E0",
            match_date=datetime(2026, 11, 8),
        )
        self.assertEqual(features["home_home_sample"], 1)
        self.assertEqual(features["away_away_sample"], 1)
        self.assertIsNotNone(features["home_scored_at_home_avg"])
        self.assertIsNotNone(features["away_scored_away_avg"])

    def test_team_alias_normalization_is_idempotent_for_engine_canonical_keys(self):
        from data.team_names import normalize_team

        for name in ("Man Utd", "Manchester United", "man_united",
                     "Real Madrid", "real_madrid", "Brighton & Hove Albion"):
            self.assertEqual(normalize_team(normalize_team(name)), normalize_team(name))

    def test_normalized_team_index_is_reused_and_invalidated_after_training(self):
        class CountingTeams(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.items_calls = 0

            def items(self):
                self.items_calls += 1
                return super().items()

        teams = CountingTeams({
            ("E0", "Home"): {"hs": [1, 2], "hc": [0, 1]},
            ("E0", "Away"): {"as": [1, 2], "ac": [1, 0]},
            ("W_E0", "Home"): {"hs": [9, 9], "hc": [0, 0]},
            ("E0", "Spare"): {"hs": [1], "hc": [1]},
        })
        engine = SimpleNamespace(st=teams, trained_n=100)

        historical_features(engine, "Home FC", "Away FC", league="E0")
        historical_xg_features(engine, "Home FC", "Away FC", league="E0")
        self.assertEqual(teams.items_calls, 1)

        teams[("E0", "New Club FC")] = {"hs": [4, 5], "hc": [0, 0]}
        engine.trained_n += 1
        updated = historical_features(
            engine, "New Club", "Away FC", league="E0",
        )
        self.assertEqual(updated["home_scored_at_home_avg"], 4.5)
        self.assertEqual(teams.items_calls, 2)

    def test_llm_goal_features_exclude_target_day_and_weight_recent_history(self):
        from model.temporal import weighted_mean

        engine = SimpleNamespace(
            half_life_days=30.0,
            st={
                ("E0", "Home"): {
                    "hs": [1, 2, 99, 100],
                    "hs_dates": ["2026-11-01", "2026-11-13", "2026-11-14", "2026-11-16"],
                    "hc": [0.5, 1.0, 99, 100],
                    "hc_dates": ["2026-11-01", "2026-11-13", "2026-11-14", "2026-11-16"],
                },
                ("E0", "Away"): {
                    "as": [1, 1.5, 99, 100],
                    "as_dates": ["2026-11-01", "2026-11-13", "2026-11-14", "2026-11-16"],
                    "ac": [1.5, 0.5, 99, 100],
                    "ac_dates": ["2026-11-01", "2026-11-13", "2026-11-14", "2026-11-16"],
                },
            },
        )
        cutoff = datetime(2026, 11, 14, 18, 0)
        result = historical_features(
            engine, "Home", "Away", league="E0", match_date=cutoff,
        )
        expected_home = weighted_mean(
            [1.0, 2.0], ["2026-11-01", "2026-11-13"],
            cutoff, half_life_days=30.0,
        )
        self.assertAlmostEqual(result["home_scored_at_home_avg"], expected_home)
        self.assertGreater(result["home_scored_at_home_avg"], 1.5)
        self.assertEqual(result["home_home_sample"], 2)
        self.assertEqual(result["away_away_sample"], 2)
        self.assertEqual(result["as_of_date"], "2026-11-14")

    def test_llm_historical_features_do_not_cross_competitions(self):
        engine = SimpleNamespace(st={
            ("W_E0", "Same Team"): {"hs": [4, 4, 4], "hc": [0, 0, 0]},
            ("W_E0", "Opponent"): {"as": [0, 0, 0], "ac": [4, 4, 4]},
            ("E0", "Same Team"): {"hs": [1, 1], "hc": [2, 2]},
            ("E0", "Opponent"): {"as": [2, 2], "ac": [1, 1]},
        })
        women = historical_features(engine, "Same Team", "Opponent", league="W_E0")
        men = historical_features(engine, "Same Team", "Opponent", league="E0")
        self.assertEqual(women["home_scored_at_home_avg"], 4.0)
        self.assertEqual(men["home_scored_at_home_avg"], 1.0)
        self.assertNotEqual(
            women["simple_baseline_lambda_total"],
            men["simple_baseline_lambda_total"],
        )

    def test_structured_ai_probabilities_are_preferred_to_formatted_text(self):
        card = {
            "deep_analysis": {
                "probabilities": {"П1": 40, "X": 35, "П2": 25},
            },
            "llm_opinion": "legacy text with no validated percentage line",
        }
        self.assertEqual(
            structured_ai_probabilities(card), {"H": .4, "D": .35, "A": .25}
        )
        self.assertIsNone(structured_ai_probabilities({
            "deep_analysis": {"probabilities": {"П1": 120, "X": -10, "П2": 0}},
            "llm_opinion": "legacy text with no validated percentage line",
        }))

    def test_ai_crosscheck_compares_model_ai_and_value_without_changing_stakes(self):
        card = {
            "p1": .55, "px": .25, "p2": .20,
            "best_by_market": {
                "winner": {
                    "best": ("1X2", "X", 3.40, .12, .62, 1.0),
                    "verdict": {"value_passed": True},
                },
            },
        }
        analysis = {
            "probabilities": {"П1": 30, "X": 45, "П2": 25},
            "data_quality": "medium",
        }
        before = dict(card)
        check = build_ai_crosscheck(card, analysis)
        self.assertEqual(check["status"], "disagreement")
        self.assertEqual(check["main_pick"], "П1")
        self.assertEqual(check["ai_pick"], "X")
        self.assertTrue(check["ai_agrees_with_value_pick"])
        self.assertEqual(check["validated_value_odds"], 3.40)
        self.assertEqual(check["decision_role"],
                         "diagnostic_only_not_used_to_change_probability_ev_edge_or_stake")
        self.assertEqual(card, before)

    def test_scan_records_pre_kickoff_main_and_ai_once(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        now = datetime(2026, 10, 25, 10, 0, tzinfo=timezone.utc)
        card = {
            "match": "Home FC vs Away FC",
            "match_ru": "Home FC — Away FC",
            "div": "E0", "league": "Premier League",
            "date_iso": "2026-10-25", "time": "16:30",
            "p1": 0.54, "px": 0.26, "p2": 0.20,
            "raw_model_probabilities": {"H": 0.52, "D": 0.28, "A": 0.20},
            "market_odds": {"П1": 2.1, "X": 3.4, "П2": 3.8},
            "odds_source": "test-book",
            "xg_used": True,
            "llm_opinion": (
                "САМОСТОЯТЕЛЬНЫЙ ПРОГНОЗ ИИ: победа гостей (П2). "
                "Оценка ИИ: П1 30% · X 25% · П2 45%."
            ),
        }
        first = record_scan_forecasts(
            [card], provider="test provider", model="test-model",
            db_path=tmp.name + "/forecasts.sqlite", now=now,
        )
        self.assertEqual(first["main_recorded"], 1)
        self.assertEqual(first["ai_recorded"], 1)
        self.assertEqual(first["learning_recorded"], 1)
        self.assertEqual(first["learning_promoted"], 0)
        self.assertEqual(first["learning_errors"], 0)
        from model.learning_service import ForecastLearningService
        path = tmp.name + "/forecasts.sqlite"
        saved = ForecastLedger(path).rows()
        learning_rows = ForecastLearningService(ForecastLedger(path)).decisions()
        self.assertEqual(len(learning_rows), 1)
        self.assertEqual(
            learning_rows[0]["raw_probabilities"], {"H": .52, "D": .28, "A": .20}
        )
        self.assertFalse(learning_rows[0]["correction_enabled"])
        main_saved = next(row for row in saved if row["model_version"].startswith("main-model-"))
        ai_saved = next(row for row in saved if row["model_version"].startswith("independent-llm-"))
        self.assertEqual(main_saved["xg_used"], 1)
        self.assertIsNone(ai_saved["xg_used"])
        second = record_scan_forecasts(
            [card], provider="test provider", model="test-model",
            db_path=tmp.name + "/forecasts.sqlite", now=now,
        )
        self.assertEqual(second["main_recorded"], 0)
        self.assertEqual(second["ai_recorded"], 0)
        self.assertEqual(second["learning_recorded"], 0)
        rows = ForecastLedger(tmp.name + "/forecasts.sqlite").rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["fixture_id"] for row in rows}, {rows[0]["fixture_id"]})
        self.assertTrue(all(row["result"] is None for row in rows))
        self.assertTrue(all(row["kickoff_at"] for row in rows))

    def test_prior_and_league_only_probabilities_are_not_added_to_main_forecast_ledger(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = tmp.name + "/forecasts.sqlite"
        now = datetime(2026, 10, 25, 10, 0, tzinfo=timezone.utc)
        cards = []
        for index, status in enumerate(("prior_only", "league_only")):
            cards.append({
                "match": f"Unknown Home {index} vs Unknown Away {index}",
                "match_ru": f"Unknown Home {index} — Unknown Away {index}",
                "div": "S_UNKNOWN", "league": "Unknown League",
                "date_iso": "2026-10-25", "time": "16:30",
                "kickoff_utc": "2026-10-25T16:30:00Z",
                "fixture_id": f"fixture-{index}",
                "p1": 0.421, "px": 0.290, "p2": 0.289,
                "raw_model_probabilities": {"H": 0.421, "D": 0.290, "A": 0.289},
                "model_support_status": status,
            })
        result = record_scan_forecasts(cards, db_path=path, now=now)
        self.assertEqual(result["main_recorded"], 0)
        self.assertEqual(result["prior_only_skipped"], 1)
        self.assertEqual(result["unsupported_history_skipped"], 2)
        self.assertEqual(ForecastLedger(path).rows(), [])

    def test_scan_records_market_recommendation_once(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = tmp.name + "/forecasts.sqlite"
        now = datetime(2026, 10, 25, 10, 0, tzinfo=timezone.utc)
        card = {
            "match": "Home FC vs Away FC", "match_ru": "Home FC — Away FC",
            "div": "E0", "league": "Premier League", "date_iso": "2026-10-25",
            "time": "16:30", "p1": .52, "px": .28, "p2": .20,
            "market_odds": {"П1": 2.1, "X": 3.4, "П2": 3.8},
            "best": ("STAT", "Угл ТБ 9.5", 1.95, .12, .58, 1.0),
            "best_odds_source": "test-book",
        }
        first = record_scan_forecasts([card], db_path=path, now=now)
        second = record_scan_forecasts([card], db_path=path, now=now + timedelta(minutes=15))
        self.assertEqual(first["market_recorded"], 1)
        self.assertEqual(second["market_recorded"], 0)
        rows = MarketForecastLedger(path).rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["market"], "STAT")
        self.assertEqual(rows[0]["selection"], "Угл ТБ 9.5")
        self.assertEqual(rows[0]["odds_source"], "test-book")
        self.assertEqual(card["market_forecast_key"], rows[0]["forecast_key"])

    def test_scan_audits_each_market_family_independently(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = tmp.name + "/forecasts.sqlite"
        now = datetime(2026, 10, 25, 10, 0, tzinfo=timezone.utc)
        card = {
            "match": "Home FC vs Away FC", "match_ru": "Home FC — Away FC",
            "div": "E0", "league": "Premier League", "date_iso": "2026-10-25",
            "time": "16:30", "p1": .52, "px": .28, "p2": .20,
            "market_odds": {"П1": 2.1, "X": 3.4, "П2": 3.8},
            "best": ("1X2", "П1", 2.1, .20, .60, 1.0),
            "best_by_market": {
                "winner": {
                    "best": ("1X2", "П1", 2.1, .20, .60, 1.0),
                    "verdict": {"prob": .60},
                },
                "goals": {
                    "best": ("OU", "ТБ 2.5", 1.95, .15, .58, 1.0),
                    "verdict": {"prob": .58},
                },
                "corners": {
                    "best": ("STAT", "Угл ТБ 9.5", 1.90, .18, .62, 1.0),
                    "verdict": {"prob": .62},
                },
            },
            "market_odds_source_by_pick": {
                "П1": "Book A", "ТБ 2.5": "Book B", "Угл ТБ 9.5": "Book C",
            },
        }

        first = record_scan_forecasts([card], db_path=path, now=now)
        second = record_scan_forecasts([card], db_path=path, now=now + timedelta(minutes=15))
        rows = MarketForecastLedger(path).rows()

        self.assertEqual(first["market_recorded"], 3)
        self.assertEqual(second["market_recorded"], 0)
        self.assertEqual(len(rows), 3)
        self.assertEqual(
            {row["market"] for row in rows}, {"1X2", "OU", "STAT"}
        )
        self.assertEqual(
            set(card["market_forecast_keys"]), {"winner", "goals", "corners"}
        )
        self.assertEqual(
            card["market_forecast_key"], card["market_forecast_keys"]["winner"]
        )

    def test_scan_does_not_record_after_kickoff_or_without_time(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        now = datetime(2026, 10, 25, 20, 0, tzinfo=timezone.utc)
        base = {
            "match": "Home vs Away", "div": "E0", "league": "L",
            "date_iso": "2026-10-25", "p1": .5, "px": .3, "p2": .2,
        }
        after = dict(base, time="12:00")
        no_time = dict(base, date_iso="2026-10-26", time="")
        result = record_scan_forecasts(
            [after, no_time], db_path=tmp.name + "/forecasts.sqlite", now=now,
        )
        self.assertEqual(result["skipped_started"], 1)
        self.assertEqual(result["skipped_no_time"], 1)
        self.assertEqual(ForecastLedger(tmp.name + "/forecasts.sqlite").rows(), [])

    def test_comparison_only_uses_same_settled_fixture(self):
        fixture = "ind:2026-01-01:abc"
        rows = [
            {"fixture_id": fixture, "model_version": "main-model-vX",
             "forecast_key": "m", "forecast_at": "2026-01-01T10:00:00+00:00",
             "kickoff_at": "2026-01-01T12:00:00+00:00", "home_team": "H", "away_team": "A",
             "probabilities": {"H": .6, "D": .25, "A": .15}, "result": "H"},
            {"fixture_id": fixture, "model_version": "independent-llm-test",
             "forecast_key": "a", "forecast_at": "2026-01-01T10:00:01+00:00",
             "kickoff_at": "2026-01-01T12:00:00+00:00", "home_team": "H", "away_team": "A",
             "probabilities": {"H": .3, "D": .25, "A": .45}, "result": "H"},
            {"fixture_id": "ind:2026-01-02:def", "model_version": "main-model-vX",
             "forecast_key": "m2", "forecast_at": "2026-01-02T10:00:00+00:00",
             "kickoff_at": "2026-01-02T12:00:00+00:00", "home_team": "H2", "away_team": "A2",
             "probabilities": {"H": .6, "D": .25, "A": .15}, "result": "H"},
        ]
        report = compare_forecast_rows(rows)
        self.assertEqual(report["main_forecasts"], 2)
        self.assertEqual(report["ai_forecasts"], 1)
        self.assertEqual(report["paired_n"], 1)
        paired = report["best_paired"]
        self.assertEqual(paired["main"]["accuracy"], 1.0)
        self.assertEqual(paired["ai"]["accuracy"], 0.0)
        self.assertEqual(paired["accuracy_delta"], -1.0)

    def test_verified_ft_result_settles_both_forecasts(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = tmp.name + "/forecasts.sqlite"
        ledger = ForecastLedger(path)
        now = datetime(2026, 10, 25, 20, 0, tzinfo=timezone.utc)
        common = {
            "fixture_id": "ind:2026-10-25:abc",
            "home_team": "Home", "away_team": "Away",
            "forecast_at": now - timedelta(hours=5),
            "kickoff_at": now - timedelta(hours=3), "league": "L",
        }
        ledger.record_prediction(
            forecast_key="main", model_version="main-model-vX",
            probabilities={"H": .5, "D": .3, "A": .2}, **common,
        )
        ledger.record_prediction(
            forecast_key="ai", model_version="independent-llm-test",
            probabilities={"H": .4, "D": .3, "A": .3}, **common,
        )
        with patch("data.scores.find_match_score", return_value={
            "home": 2, "away": 0, "status": "FT", "source": "test",
        }):
            result = settle_due_forecasts(
                db_path=path, now=now, timezone_name="Europe/Warsaw",
            )
        self.assertEqual(result["forecasts_settled"], 2)
        self.assertEqual(result["eligible_fixtures"], 1)
        self.assertEqual(result["eligible_fixtures_remaining"], 0)
        self.assertEqual(len(ledger.rows(settled_only=True)), 2)


    def test_forecast_result_lookup_waits_for_full_time_grace(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = tmp.name + "/forecasts.sqlite"
        ledger = ForecastLedger(path)
        now = datetime(2026, 10, 25, 20, 0, tzinfo=timezone.utc)
        ledger.record_prediction(
            forecast_key="grace", fixture_id="ind:2026-10-25:grace",
            home_team="Home", away_team="Away", model_version="main-model-vX",
            probabilities={"H": .5, "D": .3, "A": .2},
            forecast_at=now - timedelta(hours=4),
            kickoff_at=now - timedelta(hours=2), league="L",
        )
        with patch("data.scores.find_match_score", return_value={
            "home": 1, "away": 0, "status": "FT", "source": "test",
        }) as lookup:
            early = settle_due_forecasts(db_path=path, now=now)
            self.assertEqual(early["fixtures_checked"], 0)
            self.assertEqual(early["pending_unique_fixtures"], 1)
            self.assertEqual(early["not_due_grace_fixtures"], 1)
            lookup.assert_not_called()

            ready_at = now + timedelta(minutes=16)
            ready = settle_due_forecasts(db_path=path, now=ready_at)
            self.assertEqual(ready["fixtures_checked"], 1)
            self.assertEqual(ready["forecasts_settled"], 1)
            lookup.assert_called_once()

    def test_missing_score_lookup_is_throttled_between_streamlit_reruns(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = tmp.name + "/forecasts.sqlite"
        ledger = ForecastLedger(path)
        now = datetime(2026, 10, 25, 20, 0, tzinfo=timezone.utc)
        ledger.record_prediction(
            forecast_key="retry", fixture_id="ind:2026-10-25:retry",
            home_team="Home", away_team="Away", model_version="main-model-vX",
            probabilities={"H": .5, "D": .3, "A": .2},
            forecast_at=now - timedelta(hours=6),
            kickoff_at=now - timedelta(hours=4), league="L",
        )
        with patch("data.scores.find_match_score", return_value=None) as lookup:
            first = settle_due_forecasts(db_path=path, now=now)
            self.assertEqual(first["fixtures_checked"], 1)
            self.assertEqual(first["eligible_fixtures"], 1)
            self.assertEqual(first["eligible_fixtures_remaining"], 1)
            lookup.assert_called_once()

        with patch("data.scores.find_match_score", return_value=None) as lookup:
            second = settle_due_forecasts(
                db_path=path, now=now + timedelta(minutes=10),
            )
            self.assertEqual(second["fixtures_checked"], 0)
            self.assertEqual(second["eligible_fixtures_remaining"], 1)
            lookup.assert_not_called()

        with patch("data.scores.find_match_score", return_value={
            "home": 0, "away": 0, "status": "FT", "source": "test",
        }) as lookup:
            later = settle_due_forecasts(
                db_path=path, now=now + timedelta(hours=7),
            )
            self.assertEqual(later["fixtures_checked"], 1)
            self.assertEqual(later["forecasts_settled"], 1)
            self.assertEqual(later["eligible_fixtures_remaining"], 0)
            lookup.assert_called_once()


    def test_settlement_passes_user_entered_football_data_token(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = tmp.name + "/forecasts.sqlite"
        now = datetime(2026, 10, 25, 20, 0, tzinfo=timezone.utc)
        ledger = ForecastLedger(path)
        ledger.record_prediction(
            forecast_key="token-test",
            fixture_id="ind:2026-10-25:token-test",
            home_team="Home FC",
            away_team="Away FC",
            model_version="main-model-vX",
            probabilities={"H": .5, "D": .3, "A": .2},
            forecast_at=now - timedelta(hours=5),
            kickoff_at=now - timedelta(hours=3),
            league="L",
        )
        with patch("data.scores.find_match_score", return_value={
            "home": 2, "away": 1, "status": "FT", "source": "football-data.org",
        }) as lookup:
            result = settle_due_forecasts(
                db_path=path, now=now, token="user-entered-fdorg-token",
            )
        self.assertEqual(result["forecasts_settled"], 1)
        self.assertEqual(result["fixtures_checked"], 1)
        lookup.assert_called_once_with(
            "Home FC", "Away FC", "2026-10-25",
            token="user-entered-fdorg-token",
        )

    def test_settlement_prefers_exact_sstats_result_before_public_fallbacks(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = tmp.name + "/forecasts.sqlite"
        ledger = ForecastLedger(path)
        now = datetime(2026, 10, 25, 20, 0, tzinfo=timezone.utc)
        ledger.record_prediction(
            forecast_key="sstats-priority",
            fixture_id="ind:2026-10-25:sstats-priority",
            home_team="Home FC",
            away_team="Away FC",
            model_version="main-model-vX",
            probabilities={"H": .5, "D": .3, "A": .2},
            forecast_at=now - timedelta(hours=5),
            kickoff_at=now - timedelta(hours=3),
            league="Regional League",
        )
        with (
            patch("data.sstats.sstats_match_result", return_value={
                "home": 2, "away": 0, "status": "FT",
                "regulation_time": True, "source": "sstats",
            }) as sstats_lookup,
            patch("data.scores.find_match_score") as fallback,
        ):
            result = settle_due_forecasts(
                db_path=path, now=now, sstats_api_key="user-sstats-token",
            )

        self.assertEqual(result["fixtures_checked"], 1)
        self.assertEqual(result["forecasts_settled"], 1)
        self.assertEqual(result["pending_forecasts"], 0)
        self.assertEqual(result["sstats_lookups"], 1)
        self.assertEqual(result["result_sources"], {"sstats": 1})
        sstats_lookup.assert_called_once_with(
            "2026-10-25", "Home FC", "Away FC", "user-sstats-token",
        )
        fallback.assert_not_called()
        saved = ledger.rows(settled_only=True)[0]
        self.assertEqual(saved["home_score"], 2)
        self.assertEqual(saved["away_score"], 0)
        self.assertEqual(saved["result"], "H")

    def test_settlement_falls_back_to_public_sources_when_sstats_has_no_result(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = tmp.name + "/forecasts.sqlite"
        ledger = ForecastLedger(path)
        now = datetime(2026, 10, 25, 20, 0, tzinfo=timezone.utc)
        ledger.record_prediction(
            forecast_key="sstats-fallback",
            fixture_id="ind:2026-10-25:sstats-fallback",
            home_team="Home FC",
            away_team="Away FC",
            model_version="main-model-vX",
            probabilities={"H": .5, "D": .3, "A": .2},
            forecast_at=now - timedelta(hours=5),
            kickoff_at=now - timedelta(hours=3),
            league="Regional League",
        )
        with (
            patch("data.sstats.sstats_match_result", return_value=None) as sstats_lookup,
            patch("data.scores.find_match_score", return_value={
                "home": 0, "away": 1, "status": "FT", "source": "espn",
            }) as fallback,
        ):
            result = settle_due_forecasts(
                db_path=path, now=now, token="football-data-token",
                sstats_api_key="user-sstats-token",
            )
        self.assertEqual(result["forecasts_settled"], 1)
        self.assertEqual(result["result_sources"], {"espn": 1})
        sstats_lookup.assert_called_once()
        fallback.assert_called_once_with(
            "Home FC", "Away FC", "2026-10-25", token="football-data-token",
        )


    def test_forecasts_older_than_fourteen_days_are_backfilled(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = tmp.name + "/forecasts.sqlite"
        ledger = ForecastLedger(path)
        now = datetime(2026, 11, 15, 20, 0, tzinfo=timezone.utc)
        kickoff = datetime(2026, 10, 25, 17, 0, tzinfo=timezone.utc)
        ledger.record_prediction(
            forecast_key="backfill-21d",
            fixture_id="ind:2026-10-25:backfill",
            home_team="Home FC",
            away_team="Away FC",
            model_version="main-model-vX",
            probabilities={"H": .5, "D": .3, "A": .2},
            forecast_at=kickoff - timedelta(hours=5),
            kickoff_at=kickoff,
            league="L",
        )
        with patch("data.scores.find_match_score", return_value={
            "home": 0, "away": 1, "status": "FT", "source": "ESPN",
        }) as lookup:
            result = settle_due_forecasts(db_path=path, now=now)
        self.assertEqual(result["fixtures_checked"], 1)
        self.assertEqual(result["forecasts_settled"], 1)
        self.assertEqual(result["pending_forecasts"], 0)
        self.assertEqual(result["age_limited_fixtures"], 0)
        self.assertEqual(len(ledger.rows(settled_only=True)), 1)
        lookup.assert_called_once()

    def test_manual_force_retries_fixture_before_six_hour_deadline(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = tmp.name + "/forecasts.sqlite"
        ledger = ForecastLedger(path)
        now = datetime(2026, 10, 25, 20, 0, tzinfo=timezone.utc)
        ledger.record_prediction(
            forecast_key="force-retry",
            fixture_id="ind:2026-10-25:force-retry",
            home_team="Home FC",
            away_team="Away FC",
            model_version="main-model-vX",
            probabilities={"H": .5, "D": .3, "A": .2},
            forecast_at=now - timedelta(hours=6),
            kickoff_at=now - timedelta(hours=4),
            league="L",
        )
        with patch("data.scores.find_match_score", return_value=None):
            first = settle_due_forecasts(db_path=path, now=now)
        self.assertEqual(first["fixtures_checked"], 1)
        with patch("data.scores.find_match_score", return_value={
            "home": 1, "away": 0, "status": "FT", "source": "ESPN",
        }) as lookup:
            forced = settle_due_forecasts(
                db_path=path, now=now + timedelta(minutes=10), force=True,
            )
        self.assertEqual(forced["fixtures_checked"], 1)
        self.assertEqual(forced["forecasts_settled"], 1)
        self.assertEqual(forced["retry_deferred_fixtures"], 0)
        lookup.assert_called_once()

    def test_force_reconciliation_respects_ten_minute_minimum_retry_cooldown(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = tmp.name + "/forecasts.sqlite"
        ledger = ForecastLedger(path)
        now = datetime(2026, 10, 25, 20, 0, tzinfo=timezone.utc)
        ledger.record_prediction(
            forecast_key="force-cooldown",
            fixture_id="ind:2026-10-25:force-cooldown",
            home_team="Home FC",
            away_team="Away FC",
            model_version="main-model-vX",
            probabilities={"H": .5, "D": .3, "A": .2},
            forecast_at=now - timedelta(hours=6),
            kickoff_at=now - timedelta(hours=4),
            league="L",
        )
        with patch("data.scores.find_match_score", return_value=None) as first_lookup:
            first = settle_due_forecasts(db_path=path, now=now)
        self.assertEqual(first["fixtures_checked"], 1)
        first_lookup.assert_called_once()

        with patch("data.scores.find_match_score") as too_soon_lookup:
            too_soon = settle_due_forecasts(
                db_path=path, now=now + timedelta(minutes=1), force=True,
            )
        self.assertEqual(too_soon["fixtures_checked"], 0)
        self.assertEqual(too_soon["retry_deferred_fixtures"], 1)
        too_soon_lookup.assert_not_called()

        with patch("data.scores.find_match_score", return_value={
            "home": 1, "away": 0, "status": "FT", "source": "ESPN",
        }) as allowed_lookup:
            allowed = settle_due_forecasts(
                db_path=path, now=now + timedelta(minutes=11), force=True,
            )
        self.assertEqual(allowed["fixtures_checked"], 1)
        self.assertEqual(allowed["forecasts_settled"], 1)
        allowed_lookup.assert_called_once()

    def test_verified_regulation_time_score_from_aet_settles_1x2(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = tmp.name + "/forecasts.sqlite"
        ledger = ForecastLedger(path)
        now = datetime(2026, 10, 25, 20, 0, tzinfo=timezone.utc)
        ledger.record_prediction(
            forecast_key="aet-regulation",
            fixture_id="ind:2026-10-25:aet-regulation",
            home_team="Cup Home",
            away_team="Cup Away",
            model_version="main-model-vX",
            probabilities={"H": .3, "D": .4, "A": .3},
            forecast_at=now - timedelta(hours=6),
            kickoff_at=now - timedelta(hours=4),
            league="Cup",
        )
        with patch("data.scores.find_match_score", return_value={
            "home": 1, "away": 1, "status": "AET",
            "regulation_time": True, "source": "football-data.org",
        }) as lookup:
            result = settle_due_forecasts(db_path=path, now=now)
        self.assertEqual(result["forecasts_settled"], 1)
        self.assertEqual(result["pending_forecasts"], 0)
        self.assertEqual(ledger.rows(settled_only=True)[0]["result"], "D")
        lookup.assert_called_once()

    def test_aet_without_verified_regulation_score_stays_pending(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = tmp.name + "/forecasts.sqlite"
        ledger = ForecastLedger(path)
        now = datetime(2026, 10, 25, 20, 0, tzinfo=timezone.utc)
        ledger.record_prediction(
            forecast_key="aet-no-regulation",
            fixture_id="ind:2026-10-25:aet-no-regulation",
            home_team="Cup Home",
            away_team="Cup Away",
            model_version="main-model-vX",
            probabilities={"H": .3, "D": .4, "A": .3},
            forecast_at=now - timedelta(hours=6),
            kickoff_at=now - timedelta(hours=4),
            league="Cup",
        )
        with patch("data.scores.find_match_score", return_value={
            "home": 2, "away": 1, "status": "AET",
            "source": "OpenLigaDB",
        }) as lookup:
            result = settle_due_forecasts(db_path=path, now=now)
        self.assertEqual(result["forecasts_settled"], 0)
        self.assertEqual(result["non_ft_fixtures"], 1)
        self.assertEqual(result["pending_forecasts"], 1)
        self.assertEqual(ledger.rows(settled_only=True), [])
        lookup.assert_called_once()


    def test_settlement_reports_forecasts_outside_backfill_window(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = tmp.name + "/forecasts.sqlite"
        ledger = ForecastLedger(path)
        now = datetime(2026, 11, 15, 20, 0, tzinfo=timezone.utc)
        kickoff = datetime(2025, 10, 1, 17, 0, tzinfo=timezone.utc)
        ledger.record_prediction(
            forecast_key="backfill-too-old",
            fixture_id="ind:2025-10-01:old",
            home_team="Old Home",
            away_team="Old Away",
            model_version="main-model-vX",
            probabilities={"H": .5, "D": .3, "A": .2},
            forecast_at=kickoff - timedelta(hours=5),
            kickoff_at=kickoff,
            league="L",
        )
        with patch("data.scores.find_match_score") as lookup:
            result = settle_due_forecasts(
                db_path=path, now=now, max_age_days=365,
            )
        self.assertEqual(result["pending_forecasts"], 1)
        self.assertEqual(result["age_limited_fixtures"], 1)
        self.assertEqual(result["fixtures_checked"], 0)
        lookup.assert_not_called()



if __name__ == "__main__":
    unittest.main()
