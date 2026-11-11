import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from data.context_data import ApiFootballBudget, ApiFootballClient, ApiFootballContextProvider, ApiFootballError


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
    def raise_for_status(self):
        return None
    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = []
    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params, headers, timeout))
        endpoint = url.rsplit("/", 1)[-1]
        return FakeResponse(self.payloads.get(endpoint, {"response": [], "errors": []}))


class ContextDataTests(unittest.TestCase):
    def test_budget_reserves_quota_and_resets_by_utc_date(self):
        with tempfile.TemporaryDirectory() as directory:
            clock_value = [datetime(2026, 11, 11, tzinfo=timezone.utc)]
            budget = ApiFootballBudget(
                Path(directory) / "budget.json", daily_limit=5, reserve=1,
                clock=lambda: clock_value[0],
            )
            self.assertTrue(budget.consume(4))
            self.assertFalse(budget.consume())
            self.assertEqual(budget.used_today(), 4)
            clock_value[0] = datetime(2026, 11, 12, tzinfo=timezone.utc)
            self.assertEqual(budget.used_today(), 0)
            self.assertTrue(budget.consume())

    def test_client_uses_cache_and_never_persists_api_key(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession({"fixtures": {"response": [{"ok": True}], "errors": []}})
            client = ApiFootballClient("secret-test-key", cache_dir=directory, session=session)
            first = client.get("fixtures", {"date": "2026-11-11"}, ttl=3600)
            second = client.get("fixtures", {"date": "2026-11-11"}, ttl=3600)
            self.assertEqual(first, second)
            self.assertEqual(len(session.calls), 1)
            stored = "".join(
                path.read_text(encoding="utf-8", errors="ignore")
                for path in Path(directory).glob("*.json")
            )
            self.assertNotIn("secret-test-key", stored)


    def test_injuries_are_batched_in_groups_of_twenty(self):
        with tempfile.TemporaryDirectory() as directory:
            fixtures = []
            rows = []
            for index in range(22):
                home = f"Home Team {index}"
                away = f"Away Team {index}"
                fixtures.append({
                    "fixture": {
                        "id": index + 100,
                        "date": "2026-11-11T19:00:00+00:00",
                        "referee": "Referee A",
                    },
                    "teams": {
                        "home": {"id": index * 2 + 1, "name": home},
                        "away": {"id": index * 2 + 2, "name": away},
                    },
                })
                rows.append({
                    "HomeTeam": home,
                    "AwayTeam": away,
                    "date_iso": "2026-11-11",
                })
            session = FakeSession({
                "fixtures": {"response": fixtures, "errors": []},
                "injuries": {"response": [], "errors": []},
            })
            client = ApiFootballClient("key", cache_dir=directory, session=session)
            provider = ApiFootballContextProvider(client, max_lineup_calls=0)
            mapping, _ = provider.enrich(
                rows,
                date_from="2026-11-11",
                date_to="2026-11-11",
                now=datetime(2026, 11, 11, 10, tzinfo=timezone.utc),
            )
            injury_calls = [call for call in session.calls if call[0].endswith("/injuries")]
            self.assertEqual(len(injury_calls), 2)
            self.assertEqual(len(mapping), 22)
            self.assertTrue(all(value["injuries_is_confirmed"] for value in mapping.values()))

    def test_fixture_match_requires_exact_canonical_home_away_and_date(self):
        with tempfile.TemporaryDirectory() as directory:
            session = FakeSession({
                "fixtures": {"response": [{
                    "fixture": {"id": 765, "date": "2026-11-11T19:00:00+00:00", "referee": "Ref A"},
                    "teams": {
                        "home": {"id": 1, "name": "Manchester United"},
                        "away": {"id": 2, "name": "Chelsea"},
                    },
                }], "errors": []},
                "injuries": {"response": [], "errors": []},
            })
            client = ApiFootballClient("key", cache_dir=directory, session=session)
            provider = ApiFootballContextProvider(client, max_lineup_calls=0)
            mapping, logs = provider.enrich(
                [{"HomeTeam": "Man Utd", "AwayTeam": "Chelsea", "date_iso": "2026-11-11"}],
                date_from="2026-11-11", date_to="2026-11-11",
                now=datetime(2026, 11, 11, 10, tzinfo=timezone.utc),
            )
            key = ("2026-11-11", "man_united", "chelsea")
            self.assertIn(key, mapping)
            self.assertEqual(mapping[key]["api_fixture_id"], 765)
            self.assertTrue(mapping[key]["referee"]["is_confirmed"])
            self.assertFalse(mapping[key]["lineup"]["is_confirmed"])
            self.assertTrue(mapping[key]["injuries_is_confirmed"])
            self.assertNotIn("key", mapping[key])
            self.assertTrue(logs)


if __name__ == "__main__":
    unittest.main()
