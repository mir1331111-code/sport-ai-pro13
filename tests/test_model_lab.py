import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from model.holdout import (
    evaluate_fixed_holdout,
    render_markdown_report,
    render_reliability_svg,
    write_model_lab_artifacts,
)


def make_rows(days=32, matches_per_league=2, market="B365"):
    rows = []
    leagues = ("E0", "D1", "I1", "SP1", "F1")
    for day_index in range(days):
        stamp = (date(2024, 1, 1) + timedelta(days=day_index)).isoformat()
        for league_index, league in enumerate(leagues):
            for match_index in range(matches_per_league):
                row = {
                    "Date": stamp,
                    "Div": league,
                    "HomeTeam": f"{league}-Home-{(day_index + match_index) % 8}",
                    "AwayTeam": f"{league}-Away-{(day_index + match_index + 2) % 8}",
                    "FTHG": (day_index + match_index + league_index) % 3,
                    "FTAG": (day_index + 2 * match_index + league_index) % 2,
                }
                if market == "B365":
                    row.update({"B365H": 2.1, "B365D": 3.4, "B365A": 3.7})
                elif market == "Pinnacle":
                    row.update({"PSH": 2.05, "PSD": 3.45, "PSA": 3.8})
                rows.append(row)
    return rows


class FixedHoldoutModelLabTests(unittest.TestCase):
    def _run(self, rows, **kwargs):
        params = {
            "holdout_start": "2024-01-17",
            "holdout_end": "2024-02-01",
            "bootstrap_samples": 30,
            "min_holdout_matches": 1,
            "min_matches_per_league": 1,
            "min_matchday_clusters": 1,
            "min_market_paired_matches": 1,
        }
        params.update(kwargs)
        return evaluate_fixed_holdout(rows, **params)

    def test_scores_only_fixed_holdout_and_uses_pre_cutoff_warmup(self):
        rows = make_rows()
        report = self._run(rows)
        self.assertEqual(report["candidate"]["n"], 16 * 5 * 2)
        self.assertEqual(report["data_quality"]["training_rows_before_holdout"], 16 * 5 * 2)
        self.assertEqual(report["manifest"]["holdout_start"], "2024-01-17")
        self.assertEqual(report["manifest"]["holdout_end"], "2024-02-01")
        self.assertEqual(set(report["stratified"]["by_league"]), {"E0", "D1", "I1", "SP1", "F1"})

    def test_market_comparison_uses_only_the_selected_bookmaker(self):
        rows = make_rows(market="Pinnacle")
        b365 = self._run(rows, market_source="B365")
        pinnacle = self._run(rows, market_source="Pinnacle")
        self.assertEqual(b365["market"]["n"], 0)
        self.assertEqual(pinnacle["market"]["n"], pinnacle["candidate"]["n"])
        self.assertEqual(pinnacle["market"]["source"], "Pinnacle")

    def test_market_capture_time_must_be_timezone_aware_and_pre_kickoff(self):
        rows = make_rows()
        for row in rows:
            match_date = row["Date"]
            row["OddsCapturedAt"] = f"{match_date}T10:00:00+00:00"
            row["KickoffAt"] = f"{match_date}T15:00:00+00:00"
        verified = self._run(
            rows, odds_timestamp_field="OddsCapturedAt", kickoff_timestamp_field="KickoffAt",
        )
        self.assertTrue(verified["readiness"]["market_snapshot_verified"])
        self.assertEqual(verified["market"]["timestamp_audit"]["verified"], verified["market"]["n"])

        holdout_rows = [row for row in rows if "2024-01-17" <= row["Date"] <= "2024-02-01"]
        for row in holdout_rows:
            row["OddsCapturedAt"] = row["KickoffAt"]
        invalid = self._run(
            rows, odds_timestamp_field="OddsCapturedAt", kickoff_timestamp_field="KickoffAt",
        )
        self.assertFalse(invalid["readiness"]["market_snapshot_verified"])
        self.assertEqual(invalid["market"]["n"], 0)
        self.assertGreater(invalid["market"]["timestamp_audit"]["not_pre_kickoff"], 0)

        for row in holdout_rows:
            row["OddsCapturedAt"] = row["Date"] + "T10:00:00"
        naive = self._run(
            rows, odds_timestamp_field="OddsCapturedAt", kickoff_timestamp_field="KickoffAt",
        )
        self.assertFalse(naive["readiness"]["market_snapshot_verified"])
        self.assertEqual(naive["market"]["n"], 0)
        self.assertGreater(naive["market"]["timestamp_audit"]["missing_or_invalid"], 0)

    def test_ablation_and_xg_provenance_are_reported(self):
        rows = make_rows()
        for index, row in enumerate(rows):
            if index % 3 == 0:
                row.update({
                    "HxG": 1.4, "AxG": 0.9,
                    "xg_source": "provider_feed", "xg_kind": "provider",
                })
            elif index % 3 == 1:
                row.update({
                    "HxG": 1.2, "AxG": 1.1,
                    "xg_source": "shot_model_v1", "xg_kind": "modelled_proxy",
                })
        report = self._run(rows)
        self.assertIn("full_minus_without_h2h", report["feature_ablation"]["full_minus_ablation"])
        self.assertIn("full_minus_without_rest_days", report["feature_ablation"]["full_minus_ablation"])
        self.assertIn("full_minus_without_xg", report["feature_ablation"]["full_minus_ablation"])
        self.assertGreater(report["xg_usage"]["provider"]["rows_available"], 0)
        self.assertGreater(report["xg_usage"]["proxy"]["rows_available"], 0)

    def test_invalid_configuration_is_rejected(self):
        rows = make_rows()
        with self.assertRaises(ValueError):
            self._run(rows, holdout_start="2024-02-02", holdout_end="2024-01-01")
        with self.assertRaises(ValueError):
            self._run(rows, market_source="Average")
        with self.assertRaises(ValueError):
            self._run(rows, odds_timestamp_field="OddsCapturedAt")

    def test_reports_render_and_write_json_markdown_svg(self):
        report = self._run(make_rows())
        markdown = render_markdown_report(report)
        svg = render_reliability_svg(report)
        self.assertIn("SPORT AI", markdown)
        self.assertIn("<svg", svg)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_model_lab_artifacts(report, directory)
            for key in ("json", "markdown", "reliability_svg"):
                self.assertTrue(Path(paths[key]).exists(), key)


if __name__ == "__main__":
    unittest.main()
