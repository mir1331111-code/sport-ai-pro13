import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
from types import SimpleNamespace

from model.forecast_ledger import ForecastLedger, MarketForecastLedger
from model.independent_forecasts import (
    compare_forecast_rows,
    historical_features,
    parse_llm_probabilities,
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

    def test_scan_records_pre_kickoff_main_and_ai_once(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        now = datetime(2026, 10, 25, 10, 0, tzinfo=timezone.utc)
        card = {
            "match": "Home FC vs Away FC",
            "match_ru": "Home FC — Away FC",
            "div": "E0", "league": "Premier League",
            "date_iso": "2026-10-25", "time": "16:30",
            "p1": 0.52, "px": 0.28, "p2": 0.20,
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
        saved = ForecastLedger(tmp.name + "/forecasts.sqlite").rows()
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
        rows = ForecastLedger(tmp.name + "/forecasts.sqlite").rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["fixture_id"] for row in rows}, {rows[0]["fixture_id"]})
        self.assertTrue(all(row["result"] is None for row in rows))
        self.assertTrue(all(row["kickoff_at"] for row in rows))

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
        second = record_scan_forecasts([card], db_path=path, now=now)
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
        second = record_scan_forecasts([card], db_path=path, now=now)
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
            lookup.assert_called_once()

        with patch("data.scores.find_match_score", return_value=None) as lookup:
            second = settle_due_forecasts(
                db_path=path, now=now + timedelta(minutes=10),
            )
            self.assertEqual(second["fixtures_checked"], 0)
            lookup.assert_not_called()

        with patch("data.scores.find_match_score", return_value={
            "home": 0, "away": 0, "status": "FT", "source": "test",
        }) as lookup:
            later = settle_due_forecasts(
                db_path=path, now=now + timedelta(hours=7),
            )
            self.assertEqual(later["fixtures_checked"], 1)
            self.assertEqual(later["forecasts_settled"], 1)
            lookup.assert_called_once()



if __name__ == "__main__":
    unittest.main()
