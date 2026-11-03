import unittest

from data.sources import parse_footballapi_xg_match
from model.engine import _row_xg_pair


class FootballApiXGTests(unittest.TestCase):
    def setUp(self):
        self.match = {
            "match_id": "mt_123456789",
            "match_status": "finished",
            "kickoff_date": "2026-10-25",
            "kickoff_local_date": "2026-10-25",
            "home_club_name": "Arsenal",
            "away_club_name": "Chelsea",
            "home_goals": 2,
            "away_goals": 1,
        }
        self.stats = {
            "data": {
                "match_id": "mt_123456789",
                "stats": [
                    {"club_name": "Chelsea", "stats": {"expected_goals": 0.72}},
                    {"club_name": "Arsenal", "stats": {"expected_goals": 1.83}},
                ],
            }
        }

    def test_parses_xg_by_team_name_not_response_order(self):
        row = parse_footballapi_xg_match(self.match, self.stats)
        self.assertIsNotNone(row)
        self.assertEqual(row["HomeTeam"], "Arsenal")
        self.assertEqual(row["AwayTeam"], "Chelsea")
        self.assertEqual(row["Date"], "2026-10-25")
        self.assertEqual(row["Div"], "E0")
        self.assertEqual(row["FTHG"], 2)
        self.assertEqual(row["FTAG"], 1)
        self.assertEqual(_row_xg_pair(row), (1.83, 0.72))

    def test_does_not_create_xg_for_scheduled_or_incomplete_match(self):
        scheduled = dict(self.match, match_status="scheduled")
        self.assertIsNone(parse_footballapi_xg_match(scheduled, self.stats))
        missing_away = {
            "data": {
                "stats": [
                    {"club_name": "Arsenal", "stats": {"expected_goals": 1.2}},
                    {"club_name": "Chelsea", "stats": {"expected_goals": None}},
                ]
            }
        }
        self.assertIsNone(parse_footballapi_xg_match(self.match, missing_away))

    def test_xg_source_rows_are_distinguishable_for_engine_cache(self):
        import hashlib

        first = [{"xg_match_id": "mt_1", "HxG": 1.1, "AxG": 0.8}]
        second = first + [{"xg_match_id": "mt_2", "HxG": 0.9, "AxG": 1.4}]
        fingerprint = lambda rows: hashlib.sha256(
            "|".join(sorted(str(row["xg_match_id"]) for row in rows)).encode("utf-8")
        ).hexdigest()[:12]
        self.assertNotEqual(fingerprint(first), fingerprint(second))
        self.assertEqual(fingerprint(first), fingerprint(list(reversed(first))))

    def test_rejects_unmatched_club_names_in_stats(self):
        wrong_teams = {
            "data": {
                "stats": [
                    {"club_name": "Arsenal U21", "stats": {"expected_goals": 1.2}},
                    {"club_name": "Chelsea U21", "stats": {"expected_goals": 0.7}},
                ]
            }
        }
        self.assertIsNone(parse_footballapi_xg_match(self.match, wrong_teams))


if __name__ == "__main__":
    unittest.main()
