import tempfile
import unittest

from model.scan_ledger import ScanLedger


class ScanLedgerTests(unittest.TestCase):
    def test_fixture_snapshots_survive_an_interrupted_scan_and_are_recoverable(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ledger = ScanLedger(tmp.name + "/scan.sqlite")
        run_id = "run-interrupted"
        ledger.start_run(
            run_id,
            started_at="2026-11-27T10:00:00+00:00",
            timezone_name="Europe/Helsinki",
            horizon_days=3,
            date_from="2026-11-27",
            date_to_inclusive="2026-11-29",
        )
        first = {
            "fixture_id": "sstats:101",
            "forecast_fixture_id": "ind:2026-11-28:first",
            "match": "Home FC vs Away FC",
            "match_ru": "Home — Away",
            "league": "Test League",
            "kickoff_utc": "2026-11-28T16:00:00Z",
            "p1": 0.61,
        }
        second = {
            "fixture_id": "sstats:102",
            "forecast_fixture_id": "ind:2026-11-28:second",
            "match": "Other FC vs Third FC",
            "match_ru": "Other — Third",
            "league": "Test League",
            "kickoff_utc": "2026-11-28T18:00:00Z",
            "p1": 0.58,
        }
        ledger.save_fixture_snapshot(run_id, first)
        ledger.save_fixture_snapshot(run_id, second)
        # The second write enriches the same fixture rather than creating a duplicate.
        ledger.save_fixture_snapshot(run_id, {**first, "deep_analysis": {"summary": "saved AI"}})
        ledger.mark_run_failed(run_id, "simulated interrupted scan")

        recovered = ledger.latest_cards(prefer_completed=True)
        self.assertEqual(len(recovered), 2)
        self.assertEqual(
            next(card for card in recovered if card["forecast_fixture_id"] == first["forecast_fixture_id"])
            ["deep_analysis"]["summary"],
            "saved AI",
        )
        runs = ledger.runs(limit=5)
        self.assertEqual(runs[0]["scan_run_id"], run_id)
        self.assertEqual(runs[0]["status"], "failed")
        self.assertEqual(runs[0]["saved_card_count"], 2)

    def test_completed_scan_prefers_complete_snapshot_over_newer_partial_run(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ledger = ScanLedger(tmp.name + "/scan.sqlite")
        older = "run-complete"
        newer = "run-partial"
        for run_id, stamp in ((older, "2026-11-26T10:00:00+00:00"),
                              (newer, "2026-11-27T10:00:00+00:00")):
            ledger.start_run(
                run_id, started_at=stamp, timezone_name="Europe/Helsinki",
                horizon_days=1, date_from="2026-11-26", date_to_inclusive="2026-11-26",
            )
            card = {
                "fixture_id": run_id,
                "forecast_fixture_id": run_id,
                "match": f"{run_id} Home vs Away",
                "p1": 0.60,
            }
            if run_id == older:
                ledger.record_run(
                    run_id,
                    metadata={
                        "scan_started_at": stamp,
                        "scan_completed_at": stamp,
                        "horizon_days": 1,
                        "date_from": "2026-11-26",
                        "date_to_inclusive": "2026-11-26",
                        "source_rows": 1,
                        "unique_fixtures": 1,
                        "analysed_matches": 1,
                        "forecast_cards": 1,
                    },
                    cards=[card],
                )
            else:
                ledger.save_fixture_snapshot(run_id, card)
        ledger.mark_run_failed(newer, "interrupted")
        result = ledger.latest_cards(prefer_completed=True)
        self.assertEqual(result[0]["forecast_fixture_id"], older)

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
