import unittest

from data.fixture_merge import fixture_key, merge_fixture_sources


class FixtureMergeTests(unittest.TestCase):
    def test_merges_same_fixture_across_providers_and_keeps_preferred_row(self):
        official = [{
            "Date": "2026-10-20", "HomeTeam": "Arsenal FC", "AwayTeam": "Chelsea",
            "Div": "E0", "fixture_id": 123, "kickoff_utc": "2026-10-20T18:00:00Z",
            "odds": {"П1": 2.1},
        }]
        public = [{
            "Date": "2026-10-20", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea FC",
            "Div": "E0", "fixture_id": "tsdb-7", "home_badge": "badge.png",
        }]
        rows = merge_fixture_sources(official, public)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["fixture_id"], 123)
        self.assertEqual(rows[0]["odds"], {"П1": 2.1})
        self.assertEqual(rows[0]["home_badge"], "badge.png")

    def test_specific_league_metadata_replaces_generic_soccer_placeholder(self):
        generic = [{
            "Date": "2026-11-06", "HomeTeam": "Willem II", "AwayTeam": "PEC Zwolle",
            "Div": "G", "League": "Soccer", "fixture_id": "tsdb-1",
        }]
        specific = [{
            "Date": "2026-11-06", "HomeTeam": "Willem II", "AwayTeam": "PEC Zwolle",
            "Div": "N1", "League": "Eredivisie", "sstats_game_id": 1234,
            "stat_market_odds": {"Угл ТБ 9.5": 1.90},
        }]
        rows = merge_fixture_sources(generic, specific)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Div"], "N1")
        self.assertEqual(rows[0]["League"], "Eredivisie")
        self.assertEqual(rows[0]["fixture_id"], "tsdb-1")
        self.assertEqual(rows[0]["sstats_game_id"], 1234)
        self.assertEqual(rows[0]["stat_market_odds"], {"Угл ТБ 9.5": 1.90})

    def test_specific_existing_league_metadata_is_not_overwritten_by_generic_row(self):
        specific = [{
            "Date": "2026-11-06", "HomeTeam": "Venezia", "AwayTeam": "Udinese",
            "Div": "I1", "League": "Serie A",
        }]
        generic = [{
            "Date": "2026-11-06", "HomeTeam": "Venezia", "AwayTeam": "Udinese",
            "Div": "G", "League": "Soccer",
        }]
        rows = merge_fixture_sources(specific, generic)
        self.assertEqual(rows[0]["Div"], "I1")
        self.assertEqual(rows[0]["League"], "Serie A")

    def test_keeps_distinct_matches_and_tolerates_failed_sources(self):
        rows = merge_fixture_sources(
            [{"Date": "2026-10-20", "HomeTeam": "A", "AwayTeam": "B"}],
            None,
            [{"Date": "2026-10-21", "HomeTeam": "A", "AwayTeam": "B"}],
        )
        self.assertEqual(len(rows), 2)

    def test_sstats_is_a_complete_primary_fixture_and_odds_source(self):
        sstats = [{
            "Date": "2026-11-11T20:00:00",
            "HomeTeam": "Birmingham City",
            "AwayTeam": "Sunderland",
            "Div": "E1",
            "League": "Championship",
            "fixture_id": "sstats:12345",
            "sstats_game_id": 12345,
            "market_odds": {"П1": 2.10, "X": 3.20, "П2": 3.40},
        }]
        # Passing only the primary provider must preserve its own fixture ID,
        # its own odds and its own game ID without needing another schedule feed.
        rows = merge_fixture_sources(sstats)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["fixture_id"], "sstats:12345")
        self.assertEqual(rows[0]["sstats_game_id"], 12345)
        self.assertEqual(rows[0]["market_odds"], sstats[0]["market_odds"])

    def test_fixture_key_normalizes_accents_and_club_suffixes(self):
        self.assertEqual(
            fixture_key({"Date": "2026-10-20", "HomeTeam": "Atlético FC", "AwayTeam": "Club"}),
            fixture_key({"Date": "2026-10-20", "HomeTeam": "Atletico", "AwayTeam": "Club"}),
        )

    def test_fixture_key_matches_provider_aliases_without_merging_reserves(self):
        self.assertEqual(
            fixture_key({
                "Date": "2026-11-06", "HomeTeam": "Man Utd",
                "AwayTeam": "Brighton & Hove Albion",
            }),
            fixture_key({
                "Date": "2026-11-06", "HomeTeam": "Manchester United",
                "AwayTeam": "Brighton",
            }),
        )
        self.assertNotEqual(
            fixture_key({
                "Date": "2026-11-06", "HomeTeam": "Atletico Madrid B",
                "AwayTeam": "Real Madrid Castilla",
            }),
            fixture_key({
                "Date": "2026-11-06", "HomeTeam": "Atletico Madrid",
                "AwayTeam": "Real Madrid",
            }),
        )


if __name__ == "__main__":
    unittest.main()
