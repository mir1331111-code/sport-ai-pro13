import tempfile
import unittest

from model.scan_ledger import ScanLedger


class ScanLedgerTests(unittest.TestCase):
    def test_one_scan_links_fixtures_forecasts_and_market_decisions(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ledger = ScanLedger(tmp.name + "/scan.sqlite")
        run_id = "run-2026-11-23-001"
        card = {
            "fixture_id": "ind:2026-11-24:abc",
            "forecast_fixture_id": "ind:2026-11-24:abc",
            "provider_fixture_id": "sstats:12345",
            "scan_run_id": run_id,
            "match": "Home FC vs Away FC",
            "match_ru": "Home FC — Away FC",
            "league": "Premier League",
            "date_iso": "2026-11-24",
            "time": "18:00",
            "kickoff_utc": "2026-11-24T17:00:00Z",
            "model_support_status": "team_supported",
            "p1": .55, "px": .25, "p2": .20,
            "lam_h": 1.2, "lam_a": 1.1, "over": .52, "btts": .48,
            "games": 10,
            "independent_features": {
                "home_home_sample": 10,
                "away_away_sample": 10,
            },
            "main_forecast_key": "main-key",
            "ai_forecast_key": "ai-key",
            "market_forecast_keys": {
                "winner": "market-winner-key",
                "goals": "market-goals-key",
            },
            "market_odds": {"П1": 2.1, "X": 3.4, "П2": 4.0},
            "best_by_market": {},
        }
        result = ledger.record_run(
            run_id,
            metadata={
                "scan_started_at": "2026-11-23T09:00:00+01:00",
                "scan_completed_at": "2026-11-23T09:02:00+01:00",
                "timezone_name": "Europe/Paris",
                "horizon_days": 1,
                "date_from": "2026-11-23",
                "date_to_inclusive": "2026-11-23",
                "source_rows": 3,
                "unique_fixtures": 1,
                "analysed_matches": 1,
                "forecast_cards": 1,
                "source_counts": {"SStats": 1, "ESPN": 2},
                "funnel": {"team_supported": 1},
            },
            cards=[card],
            report=["test scan"],
        )
        self.assertEqual(result, {"fixtures": 1, "forecasts": 4, "decisions": 5})
        latest = ledger.latest_run()
        self.assertEqual(latest["scan_run_id"], run_id)
        self.assertEqual(latest["horizon_days"], 1)
        self.assertEqual(latest["source_counts"], {"SStats": 1, "ESPN": 2})
        self.assertEqual(len(ledger.forecast_links(run_id)), 4)
        decisions = ledger.decisions(run_id)
        self.assertEqual(len(decisions), 5)
        self.assertEqual({item["market_family"] for item in decisions},
                         {"winner", "goals", "btts", "corners", "cards"})
        self.assertTrue(all(isinstance(item["gates"], list) for item in decisions))
        quality = ledger.quality_rows(run_id)
        self.assertEqual(len(quality), 1)
        self.assertEqual(quality[0]["quality"]["match"], "Home FC — Away FC")
        api_check = next(
            item for item in quality[0]["quality"]["checks"]
            if item["name"] == "API-Football"
        )
        self.assertEqual(api_check["state"], "not_configured")


if __name__ == "__main__":
    unittest.main()
