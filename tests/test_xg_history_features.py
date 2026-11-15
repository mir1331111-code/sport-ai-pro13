import unittest
from datetime import datetime
from types import SimpleNamespace

from model.independent_forecasts import historical_xg_features


class HistoricalXGFeatureTests(unittest.TestCase):
    def test_xg_means_use_pre_match_time_decay(self):
        from model.temporal import weighted_mean

        engine = SimpleNamespace(st={
            ("E0", "Home FC"): {
                "hs_xg": [1.0, 2.0, 9.0],
                "hs_xg_dates": ["2026-11-01", "2026-11-13", "2026-11-14"],
                "hs_xga": [0.5, 1.0, 9.0],
                "hs_xga_dates": ["2026-11-01", "2026-11-13", "2026-11-14"],
                "hs_pxg": [0.8, 1.8, 9.0],
                "hs_pxg_dates": ["2026-11-01", "2026-11-13", "2026-11-14"],
                "hs_pxga": [0.4, 0.9, 9.0],
                "hs_pxga_dates": ["2026-11-01", "2026-11-13", "2026-11-14"],
            },
            ("E0", "Away FC"): {
                "as_xg": [0.9, 1.6, 9.0],
                "as_xg_dates": ["2026-11-02", "2026-11-13", "2026-11-14"],
                "as_xga": [1.1, 1.7, 9.0],
                "as_xga_dates": ["2026-11-02", "2026-11-13", "2026-11-14"],
                "as_pxg": [0.7, 1.4, 9.0],
                "as_pxg_dates": ["2026-11-02", "2026-11-13", "2026-11-14"],
                "as_pxga": [1.0, 1.5, 9.0],
                "as_pxga_dates": ["2026-11-02", "2026-11-13", "2026-11-14"],
            },
        }, half_life_days=30.0)
        cutoff = datetime(2026, 11, 14, 19, 0)
        result = historical_xg_features(
            engine, "Home FC", "Away FC", league="E0", match_date=cutoff,
        )
        expected = weighted_mean(
            [1.0, 2.0], ["2026-11-01", "2026-11-13"],
            cutoff, half_life_days=30.0,
        )
        self.assertAlmostEqual(result["provider"]["home_xg_for_avg"], expected)
        self.assertGreater(result["provider"]["home_xg_for_avg"], 1.5)
        self.assertEqual(result["provider"]["home_xg_samples"], 2)
        self.assertEqual(result["as_of_date"], "2026-11-14")

    def test_provider_and_proxy_xg_xga_are_separate_and_strictly_pre_match(self):
        engine = SimpleNamespace(st={
            ("E0", "Home FC"): {
                "hs_xg": [1.2, 1.8], "hs_xg_dates": [
                    datetime(2026, 11, 1), datetime(2026, 11, 14),
                ],
                "hs_xga": [0.8, 1.4], "hs_xga_dates": [
                    datetime(2026, 11, 1), datetime(2026, 11, 14),
                ],
                "hs_pxg": [1.1, 1.9], "hs_pxg_dates": [
                    datetime(2026, 11, 2), datetime(2026, 11, 14),
                ],
                "hs_pxga": [0.9, 1.5], "hs_pxga_dates": [
                    datetime(2026, 11, 2), datetime(2026, 11, 14),
                ],
            },
            ("E0", "Away FC"): {
                "as_xg": [1.0, 1.6], "as_xg_dates": [
                    datetime(2026, 11, 3), datetime(2026, 11, 14),
                ],
                "as_xga": [1.1, 1.7], "as_xga_dates": [
                    datetime(2026, 11, 3), datetime(2026, 11, 14),
                ],
                "as_pxg": [0.9, 1.4], "as_pxg_dates": [
                    datetime(2026, 11, 4), datetime(2026, 11, 14),
                ],
                "as_pxga": [1.0, 1.5], "as_pxga_dates": [
                    datetime(2026, 11, 4), datetime(2026, 11, 14),
                ],
            },
        })
        result = historical_xg_features(
            engine, "Home FC", "Away FC", league="E0",
            match_date=datetime(2026, 11, 14, 19, 0),
        )
        self.assertEqual(result["provider"]["home_xg_for_avg"], 1.2)
        self.assertEqual(result["provider"]["home_xga_avg"], 0.8)
        self.assertEqual(result["provider"]["away_xg_for_avg"], 1.0)
        self.assertEqual(result["provider"]["away_xga_avg"], 1.1)
        self.assertEqual(result["proxy"]["home_xg_for_avg"], 1.1)
        self.assertEqual(result["proxy"]["away_xga_avg"], 1.0)
        self.assertEqual(result["provider"]["home_xg_samples"], 1)
        self.assertEqual(result["source_priority"], "provider_then_proxy")


if __name__ == "__main__":
    unittest.main()
