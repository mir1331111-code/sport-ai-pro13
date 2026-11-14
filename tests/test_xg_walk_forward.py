import unittest
from datetime import date, timedelta

from model.walk_forward import (
    _xg_origin, _without_provider_xg, _without_proxy_xg, walk_forward_compare,
)


class WalkForwardTests(unittest.TestCase):
    def _rows(self, with_xg=True):
        rows = []
        start = date(2022, 1, 1)
        for index in range(100):
            row = {
                "Date": (start + timedelta(days=index)).isoformat(),
                "Div": "E0",
                "HomeTeam": f"Home{index % 8}",
                "AwayTeam": f"Away{index % 9}",
                "FTHG": index % 4,
                "FTAG": (index + 1) % 3,
            }
            if with_xg:
                row["HxG"] = 2.3 if index % 3 == 0 else 0.7
                row["AxG"] = 0.5 if index % 3 == 0 else 1.8
            rows.append(row)
        return rows

    def test_compares_three_arms_on_same_chronological_fixtures(self):
        result = walk_forward_compare(self._rows())
        self.assertEqual(result["n"], 100)
        self.assertEqual(result["skipped"], 0)
        self.assertEqual(result["baseline"]["n"], 100)
        self.assertEqual(result["candidate"]["n"], 100)
        self.assertEqual(result["temporal_goals_only"]["n"], 100)
        self.assertEqual(result["xg_rows_available"], 100)
        self.assertIn("brier", result["xg_delta_vs_same_temporal_model"])
        self.assertIn("untouched chronological holdout", result["interpretation"])

    def test_four_xg_arms_separate_provider_proxy_and_mixed_rows(self):
        rows = []
        start = date(2023, 1, 1)
        for index in range(180):
            row = {
                "Date": (start + timedelta(days=index // 2)).isoformat(),
                "Div": "E0",
                "HomeTeam": f"Home{index % 8}",
                "AwayTeam": f"Away{index % 9}",
                "FTHG": (index // 3) % 4,
                "FTAG": (index + 1) % 3,
                "HxG": 1.2 if index % 4 else 2.0,
                "AxG": 0.8 if index % 5 else 1.7,
            }
            if index % 2:
                row["xg_kind"] = "modelled_proxy"
                row["xg_source"] = "shot_aggregate_ridge_v1"
            else:
                row["xg_source"] = "footballapi_test_provider"
            rows.append(row)

        report = walk_forward_compare(
            rows, bootstrap_samples=30, market_bootstrap_samples=0,
            totals_bootstrap_samples=0,
        )
        audit = report["xg_ablation"]
        self.assertEqual(report["n"], 180)
        self.assertEqual(audit["source_rows"]["provider"], 90)
        self.assertEqual(audit["source_rows"]["modelled_proxy"], 90)
        for arm in ("goals_only", "provider_xg_only", "proxy_xg_only", "mixed_provider_preferred"):
            self.assertEqual(audit["arms"][arm]["n"], 180)
            self.assertEqual(audit["arms"][arm]["bootstrap_unit"], "matchday")
            self.assertEqual(audit["arms"][arm]["cluster_count"], 90)
        self.assertEqual(audit["paired_differences"]["provider_vs_goals_only"]["n"], 180)
        self.assertEqual(audit["paired_differences"]["proxy_vs_goals_only"]["n"], 180)
        self.assertGreater(audit["usage"]["provider_xg_only"]["provider_xg_used"], 0)
        self.assertGreater(audit["usage"]["proxy_xg_only"]["proxy_xg_used"], 0)
        self.assertGreater(audit["usage"]["mixed_provider_preferred"]["provider_xg_used"], 0)

    def test_xg_source_transformations_keep_only_the_requested_provenance(self):
        provider = {
            "HxG": 1.4, "AxG": 0.9, "xg_source": "provider_api",
        }
        proxy = {
            "HxG": 1.1, "AxG": 0.8, "xg_source": "shot_aggregate_ridge_v1",
            "xg_kind": "modelled_proxy", "xg_training_sides": 120,
        }
        self.assertEqual(_xg_origin(provider), "provider")
        self.assertEqual(_xg_origin(proxy), "proxy")
        self.assertIsNone(_xg_origin(_without_provider_xg(provider)))
        self.assertEqual(_xg_origin(_without_provider_xg(proxy)), "proxy")
        self.assertEqual(_xg_origin(_without_proxy_xg(provider)), "provider")
        self.assertIsNone(_xg_origin(_without_proxy_xg(proxy)))

    def test_no_xg_rows_means_ablation_arms_are_identical(self):
        result = walk_forward_compare(self._rows(with_xg=False))
        self.assertEqual(result["xg_rows_available"], 0)
        self.assertEqual(result["xg_forecasts_used"], 0)
        self.assertEqual(result["candidate"], result["temporal_goals_only"])
        self.assertEqual(result["xg_delta_vs_same_temporal_model"], {
            "brier": 0.0, "log_loss": 0.0, "accuracy": 0.0
        })

    def test_h2h_and_rest_day_ablation_arms_are_paired(self):
        result = walk_forward_compare(
            self._rows(with_xg=False), bootstrap_samples=20,
        )
        ablation = result["feature_ablation"]
        self.assertEqual(ablation["full_h2h_and_rest_days"]["n"], 100)
        self.assertEqual(ablation["without_h2h"]["n"], 100)
        self.assertEqual(ablation["without_rest_days"]["n"], 100)
        self.assertEqual(ablation["without_h2h_or_rest_days"]["n"], 100)
        self.assertEqual(ablation["full_minus_without_h2h"]["n"], 100)
        self.assertEqual(ablation["full_minus_without_rest_days"]["n"], 100)
        self.assertEqual(ablation["full_minus_without_h2h_or_rest_days"]["n"], 100)
        self.assertEqual(result["xg_usage_share"], 0.0)

    def test_skips_malformed_rows(self):
        result = walk_forward_compare([{"Date": "bad", "HomeTeam": "A"}])
        self.assertEqual(result["n"], 0)
        self.assertEqual(result["skipped"], 1)


if __name__ == "__main__":
    unittest.main()
