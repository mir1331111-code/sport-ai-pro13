import unittest

from ui.tabs.scanner import _merge_sstats_historical_rows


class SStatsHistoricalPersistenceTests(unittest.TestCase):
    def test_merge_deduplicates_team_aliases_and_keeps_existing_scores(self):
        existing = [{
            "Date": "2026-11-21",
            "Div": "S_ABC123",
            "HomeTeam": "Man Utd",
            "AwayTeam": "Tottenham",
            "FTHG": 2,
            "FTAG": 1,
            "League": "Premier League",
        }]
        fetched = [
            {
                "Date": "2026-11-21",
                "Div": "S_ABC123",
                "HomeTeam": "Manchester United",
                "AwayTeam": "Tottenham Hotspur",
                "FTHG": 2,
                "FTAG": 1,
                "HxG": 1.8,
                "AxG": 0.7,
            },
            {
                "Date": "2026-11-20",
                "Div": "S_ABC123",
                "HomeTeam": "Team A",
                "AwayTeam": "Team B",
                "FTHG": 1,
                "FTAG": 1,
                "League": "Premier League",
            },
            {
                "Date": "2026-11-21",
                "Div": "S_ABC123",
                "HomeTeam": "Manchester United",
                "AwayTeam": "Tottenham Hotspur",
                "FTHG": 0,
                "FTAG": 3,
            },
            {"Date": "not-a-date", "Div": "S_ABC123", "HomeTeam": "Bad", "AwayTeam": "Row"},
        ]

        merged, stats = _merge_sstats_historical_rows(existing, fetched)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[1]["FTHG"], 2)
        self.assertEqual(merged[1]["FTAG"], 1)
        self.assertEqual(merged[1]["HxG"], 1.8)
        self.assertEqual(merged[1]["AxG"], 0.7)
        self.assertEqual(stats["added"], 1)
        self.assertEqual(stats["duplicates"], 2)
        self.assertEqual(stats["enriched"], 1)
        self.assertEqual(stats["conflicts"], 1)
        self.assertEqual(stats["invalid"], 1)

    def test_merge_uses_competition_and_ordered_teams_as_identity(self):
        existing = [{
            "Date": "2026-11-21", "Div": "S_1",
            "HomeTeam": "North FC", "AwayTeam": "South FC",
            "FTHG": 1, "FTAG": 0,
        }]
        fetched = [{
            "Date": "2026-11-21", "Div": "S_2",
            "HomeTeam": "North FC", "AwayTeam": "South FC",
            "FTHG": 0, "FTAG": 1,
        }]
        merged, stats = _merge_sstats_historical_rows(existing, fetched)

        self.assertEqual(len(merged), 2)
        self.assertEqual(stats["added"], 1)
        self.assertEqual(stats["conflicts"], 0)


if __name__ == "__main__":
    unittest.main()
