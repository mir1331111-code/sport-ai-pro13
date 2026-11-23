import unittest
from datetime import date, timedelta

from model.engine import Engine
from model.walk_forward import (
    _ClusteredRecords, _metrics, _paired_differences,
    _binary_metrics, _binary_paired_differences,
    totals_market_is_validated, walk_forward_compare,
)


class WalkForwardTests(unittest.TestCase):
    def test_legacy_engine_without_form_weight_attribute_still_predicts(self):
        # Existing on-disk Engine pickles predate the research-only attribute.
        engine = Engine()
        del engine.form_adjustment_weight
        prediction = engine.predict("Home", "Away", "E0")
        self.assertAlmostEqual(
            prediction["p1"] + prediction["x"] + prediction["p2"], 1.0, places=8
        )

    def test_compares_baseline_and_candidate_chronologically(self):
        rows = []
        for index in range(24):
            stamp = (date(2024, 1, 1) + timedelta(days=index)).isoformat()
            rows.append({
                "Date": stamp,
                "Div": "E0",
                "HomeTeam": f"Home{index % 4}",
                "AwayTeam": f"Away{index % 5}",
                "FTHG": index % 3,
                "FTAG": (index + 1) % 3,
            })
        result = walk_forward_compare(rows)
        self.assertEqual(result["n"], 24)
        self.assertEqual(result["skipped"], 0)
        self.assertEqual(result["baseline"]["n"], 24)
        self.assertEqual(result["candidate"]["n"], 24)
        form_report = result["recent_form_ablation"]
        self.assertEqual(form_report["candidate"]["n"], 24)
        self.assertEqual(form_report["paired_vs_base_model"]["n"], 24)
        self.assertEqual(form_report["production_weight"], 0.0)
        self.assertIsNotNone(result["delta_candidate_minus_baseline"]["brier"])
        self.assertIn("untouched chronological holdout", result["interpretation"])
        self.assertEqual(set(result["goals_totals_by_line"]), {"0.5", "1.5", "2.5", "3.5", "4.5", "5.5"})
        for line_report in result["goals_totals_by_line"].values():
            self.assertEqual(line_report["n"], 24)
            self.assertIsNotNone(line_report["metrics"]["brier"])
            self.assertIsNotNone(line_report["metrics"]["log_loss"])

    def test_walk_forward_replays_warmup_but_scores_only_league_holdout(self):
        rows = []
        for index in range(24):
            stamp = (date(2024, 1, 1) + timedelta(days=index)).isoformat()
            rows.append({
                "Date": stamp,
                "Div": "E0",
                "HomeTeam": f"Home{index % 4}",
                "AwayTeam": f"Away{index % 5}",
                "FTHG": index % 3,
                "FTAG": (index + 1) % 3,
            })
        result = walk_forward_compare(
            rows,
            bootstrap_samples=0,
            evaluation_start_by_league={"E0": "2024-01-20"},
        )
        # The first 19 dates warm every model; only the five later dates are scored.
        self.assertEqual(result["n"], 24)
        self.assertEqual(result["evaluation_n"], 5)
        self.assertEqual(result["candidate"]["n"], 5)
        self.assertEqual(result["baseline"]["n"], 5)

    def test_warmup_only_leagues_do_not_enter_evaluation_metrics(self):
        rows = [{
            "Date": (date(2024, 1, 1) + timedelta(days=i)).isoformat(),
            "Div": "E0", "HomeTeam": f"H{i % 3}", "AwayTeam": f"A{i % 4}",
            "FTHG": i % 2, "FTAG": (i + 1) % 2,
        } for i in range(8)]
        result = walk_forward_compare(
            rows, bootstrap_samples=0, evaluation_start_by_league={"D1": "2024-01-01"},
        )
        self.assertEqual(result["candidate"]["n"], 0)
        self.assertEqual(result["market"]["n"], 0)

    def test_clustered_records_can_suppress_warmup_observations(self):
        records = _ClusteredRecords()
        records.set_cluster("2024-01-01")
        records.set_enabled(False)
        records.append(({"H": 0.8, "D": 0.1, "A": 0.1}, "H"))
        records.set_enabled(True)
        records.set_cluster("2024-01-02")
        records.append(({"H": 0.1, "D": 0.8, "A": 0.1}, "D"))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0][2], "2024-01-02")

    def test_metrics_resample_calendar_day_clusters_not_individual_fixtures(self):
        records = _ClusteredRecords()
        records.set_cluster("2024-01-01")
        records.append(({"H": 0.80, "D": 0.10, "A": 0.10}, "H"))
        records.append(({"H": 0.10, "D": 0.10, "A": 0.80}, "H"))
        records.set_cluster("2024-01-02")
        records.append(({"H": 0.10, "D": 0.80, "A": 0.10}, "D"))
        records.append(({"H": 0.10, "D": 0.10, "A": 0.80}, "D"))

        report = _metrics(records, bootstrap_samples=100, seed=123)
        self.assertEqual(report["n"], 4)
        self.assertEqual(report["bootstrap_unit"], "matchday")
        self.assertEqual(report["cluster_count"], 2)
        self.assertLessEqual(report["confidence_intervals_95"]["brier"]["low"],
                             report["confidence_intervals_95"]["brier"]["high"])

    def test_metrics_include_fixed_width_reliability_tables_for_each_1x2_class(self):
        records = _ClusteredRecords()
        records.set_cluster("2024-01-01")
        outcomes = ["H"] * 6 + ["D"] * 3 + ["A"]
        for outcome in outcomes:
            records.append(({"H": 0.6, "D": 0.3, "A": 0.1}, outcome))

        report = _metrics(records, bootstrap_samples=0)
        reliability = report["reliability"]
        self.assertEqual(reliability["n"], 10)
        self.assertEqual(reliability["binning"], "fixed_width")
        for code, expected_bin in (("H", 6), ("D", 3), ("A", 1)):
            class_report = reliability["classes"][code]
            self.assertEqual(class_report["n"], 10)
            self.assertAlmostEqual(class_report["ece"], 0.0, places=12)
            populated = [
                item for item in class_report["bins"] if item["count"] > 0
            ]
            self.assertEqual(len(populated), 1)
            self.assertEqual(populated[0]["count"], 10)
            self.assertEqual(
                int(populated[0]["lower"] * 10), expected_bin
            )

        miscalibrated = _ClusteredRecords()
        miscalibrated.set_cluster("2024-01-02")
        for index in range(10):
            miscalibrated.append((
                {"H": 0.6, "D": 0.3, "A": 0.1},
                "H" if index < 8 else "D",
            ))
        second = _metrics(miscalibrated, bootstrap_samples=0)
        self.assertAlmostEqual(
            second["reliability"]["classes"]["H"]["ece"], 0.2, places=12
        )

    def test_paired_and_binary_metrics_keep_matchday_cluster_alignment(self):
        left = _ClusteredRecords()
        right = _ClusteredRecords()
        binary_left = _ClusteredRecords()
        binary_right = _ClusteredRecords()
        for day in ("2024-02-01", "2024-02-02", "2024-02-03"):
            for index in range(2):
                left.set_cluster(day)
                right.set_cluster(day)
                binary_left.set_cluster(day)
                binary_right.set_cluster(day)
                outcome = ("H", "D", "A")[index]
                base = {"H": 0.60, "D": 0.25, "A": 0.15}
                improved = {"H": 0.70, "D": 0.20, "A": 0.10}
                left.append((improved, outcome))
                right.append((base, outcome))
                binary_left.append(({"over": 0.65, "under": 0.35}, "over"))
                binary_right.append(({"over": 0.50, "under": 0.50}, "over"))

        paired = _paired_differences(left, right, bootstrap_samples=80, seed=7)
        binary = _binary_metrics(binary_left, bootstrap_samples=80, seed=7)
        binary_paired = _binary_paired_differences(
            binary_left, binary_right, bootstrap_samples=80, seed=7,
        )
        self.assertEqual(paired["n"], 6)
        self.assertEqual(paired["cluster_count"], 3)
        self.assertEqual(binary["cluster_count"], 3)
        self.assertEqual(binary_paired["cluster_count"], 3)
        self.assertEqual(paired["bootstrap_unit"], "matchday")
        self.assertEqual(binary_paired["bootstrap_unit"], "matchday")

    def test_paired_metrics_reject_misaligned_day_clusters(self):
        left = _ClusteredRecords()
        right = _ClusteredRecords()
        left.set_cluster("2024-03-01")
        right.set_cluster("2024-03-02")
        left.append(({"H": 0.6, "D": 0.2, "A": 0.2}, "H"))
        right.append(({"H": 0.5, "D": 0.3, "A": 0.2}, "H"))
        report = _paired_differences(left, right, bootstrap_samples=10)
        self.assertEqual(report["n"], 0)
        self.assertIn("matchday", report["error"])

    def test_market_validation_requires_thirty_independent_matchdays(self):
        report = {
            "n": 800,
            "cluster_count": 29,
            "differences": {"brier": -0.02, "log_loss": -0.03},
            "confidence_intervals_95": {
                "brier": {"low": -0.03, "high": -0.01},
                "log_loss": {"low": -0.04, "high": -0.02},
            },
        }
        self.assertFalse(totals_market_is_validated(report, min_n=500))
        report["cluster_count"] = 30
        self.assertTrue(totals_market_is_validated(report, min_n=500))
        report.pop("cluster_count")
        self.assertFalse(totals_market_is_validated(report, min_n=500))

    def test_skips_rows_without_valid_dates_or_scores(self):
        result = walk_forward_compare([
            {"Date": "not-a-date", "HomeTeam": "A", "AwayTeam": "B", "FTHG": 1, "FTAG": 0},
            {"Date": "2024-01-01", "HomeTeam": "C", "AwayTeam": "D", "FTHG": "bad", "FTAG": 0},
        ])
        self.assertEqual(result["n"], 0)
        self.assertEqual(result["skipped"], 2)
        self.assertIsNone(result["candidate"]["log_loss"])


    def test_fractional_goal_scores_are_skipped_not_truncated(self):
        result = walk_forward_compare([{
            "Date": "2024-01-01", "Div": "E0",
            "HomeTeam": "A", "AwayTeam": "B",
            "FTHG": 1.5, "FTAG": 0,
        }])
        self.assertEqual(result["n"], 0)
        self.assertEqual(result["skipped"], 1)

    def test_market_gate_bootstrap_can_run_without_full_model_bootstrap(self):
        rows = []
        for index in range(40):
            stamp = (date(2024, 1, 1) + timedelta(days=index)).isoformat()
            rows.append({
                "Date": stamp,
                "Div": "E0",
                "HomeTeam": f"Home{index % 5}",
                "AwayTeam": f"Away{index % 7}",
                "FTHG": index % 3,
                "FTAG": (index + 1) % 3,
                "B365H": 2.10,
                "B365D": 3.40,
                "B365A": 3.80,
            })
        result = walk_forward_compare(
            rows, bootstrap_samples=0, market_bootstrap_samples=30,
            totals_bootstrap_samples=0,
        )
        paired = result["candidate_minus_market"]
        self.assertEqual(paired["n"], 40)
        self.assertIsNotNone(paired["confidence_intervals_95"]["brier"]["low"])
        self.assertIsNotNone(paired["confidence_intervals_95"]["log_loss"]["high"])
        self.assertEqual(result["candidate"]["confidence_intervals_95"]["brier"]["low"], None)
        self.assertEqual(result["market_bootstrap_samples"], 30)
        for weight in ("0.25", "0.50", "0.75"):
            arm = result["market_blends"]["weights_model_market"][weight]
            self.assertEqual(arm["metrics"]["n"], 40)
            self.assertEqual(arm["vs_model"]["n"], 40)
            self.assertEqual(arm["vs_market"]["n"], 40)
            league_arm = result["market_blends"]["by_league"]["E0"][weight]
            self.assertEqual(league_arm["metrics"]["n"], 40)
            self.assertEqual(league_arm["vs_market"]["n"], 40)



if __name__ == "__main__":
    unittest.main()
