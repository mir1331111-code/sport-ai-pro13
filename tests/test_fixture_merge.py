import unittest

from data.fixture_merge import (
    enrich_fixture_visual_metadata, enforce_sstats_market_card,
    enforce_sstats_odds_only, fixture_key, is_sstats_market_source,
    merge_fixture_sources,
)


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


    def test_merges_unique_same_fixture_when_provider_dates_shift_by_one_day(self):
        primary = [{
            "Date": "2026-11-24 00:30:00",
            "HomeTeam": "Paris Saint-Germain", "AwayTeam": "Marseille",
            "Div": "F1", "fixture_id": "sstats:77",
            "sstats_game_id": 77, "source": "sstats",
            "market_odds": {"П1": 1.8}, "odds_source": "sstats",
        }]
        fallback = [{
            "Date": "2026-11-23", "Time": "23:30",
            "HomeTeam": "Paris Saint-Germain", "AwayTeam": "Marseille",
            "Div": "F1", "fixture_id": "espn:77", "source": "espn",
            "home_badge": "https://example.test/psg.png",
        }]
        rows = merge_fixture_sources(primary, fallback)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["fixture_id"], "sstats:77")
        self.assertEqual(rows[0]["market_odds"], {"П1": 1.8})
        self.assertEqual(rows[0]["home_badge"], "https://example.test/psg.png")


    def test_sstats_schedule_is_supplemented_by_fallback_without_admitting_its_prices(self):
        primary = [{
            "Date": "2026-11-23 18:00:00",
            "HomeTeam": "Arsenal", "AwayTeam": "Chelsea", "Div": "E0",
            "fixture_id": "sstats:1", "sstats_game_id": 1, "source": "sstats",
            "market_odds": {"П1": 2.1, "X": 3.4, "П2": 3.8},
            "odds_source": "sstats",
        }]
        fallback = [
            {
                "Date": "2026-11-23", "HomeTeam": "Arsenal FC", "AwayTeam": "Chelsea",
                "Div": "E0", "fixture_id": "espn:1", "source": "espn",
                "market_odds": {"П1": 1.9, "X": 3.2, "П2": 4.1},
                "odds_source": "espn",
                "home_badge": "https://example.test/arsenal.png",
            },
            {
                "Date": "2026-11-23", "HomeTeam": "Lyon", "AwayTeam": "Lille",
                "Div": "F1", "fixture_id": "espn:2", "source": "espn",
                "market_odds": {"П1": 2.0, "X": 3.1, "П2": 4.0},
                "odds_source": "espn",
            },
        ]
        rows = enforce_sstats_odds_only(merge_fixture_sources(primary, fallback))
        self.assertEqual(len(rows), 2)
        arsenal = next(row for row in rows if "Arsenal" in row["HomeTeam"])
        lyon = next(row for row in rows if row["HomeTeam"] == "Lyon")
        self.assertEqual(arsenal["fixture_id"], "sstats:1")
        self.assertEqual(arsenal["market_odds"], {"П1": 2.1, "X": 3.4, "П2": 3.8})
        self.assertEqual(arsenal["home_badge"], "https://example.test/arsenal.png")
        self.assertEqual(lyon["market_odds"], {})
        self.assertEqual(lyon["odds_source"], "no_market")

    def test_sstats_primary_fixtures_receive_visual_metadata_without_replacing_odds(self):
        sstats_rows = [{
            "Date": "2026-11-16 19:00:00",
            "HomeTeam": "Arsenal FC",
            "AwayTeam": "Chelsea",
            "Div": "E0",
            "League": "Premier League",
            "fixture_id": "sstats:12345",
            "sstats_game_id": 12345,
            "source": "sstats",
            "odds_source": "sstats",
            "market_odds": {"П1": 1.90, "X": 3.50, "П2": 4.20},
        }]
        tsdb_rows = [
            {
                "Date": "2026-11-16",
                "HomeTeam": "Arsenal",
                "AwayTeam": "Chelsea FC",
                "fixture_id": "tsdb:visual-only",
                "home_badge": "https://assets.test/arsenal.png",
                "away_badge": "https://assets.test/chelsea.png",
                "home_team_tsdb_id": "133604",
                "away_team_tsdb_id": "133602",
                "league_badge": "https://assets.test/epl.png",
                "stadium_image": "https://assets.test/emirates.jpg",
                "venue": "Emirates Stadium",
                # Metadata-only enrichment must ignore any secondary quote fields.
                "market_odds": {"П1": 9.99},
            },
            {
                "Date": "2026-11-16",
                "HomeTeam": "Unrelated FC",
                "AwayTeam": "Another FC",
                "fixture_id": "tsdb:unrelated",
                "home_badge": "https://assets.test/unrelated.png",
            },
        ]

        rows, report = enrich_fixture_visual_metadata(sstats_rows, tsdb_rows)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["fixture_id"], "sstats:12345")
        self.assertEqual(rows[0]["sstats_game_id"], 12345)
        self.assertEqual(rows[0]["source"], "sstats")
        self.assertEqual(rows[0]["odds_source"], "sstats")
        self.assertEqual(
            rows[0]["market_odds"],
            {"П1": 1.90, "X": 3.50, "П2": 4.20},
        )
        self.assertEqual(rows[0]["home_badge"], "https://assets.test/arsenal.png")
        self.assertEqual(rows[0]["away_badge"], "https://assets.test/chelsea.png")
        self.assertEqual(rows[0]["home_team_tsdb_id"], "133604")
        self.assertEqual(rows[0]["away_team_tsdb_id"], "133602")
        self.assertEqual(rows[0]["league_badge"], "https://assets.test/epl.png")
        self.assertEqual(rows[0]["stadium_image"], "https://assets.test/emirates.jpg")
        self.assertEqual(rows[0]["venue"], "Emirates Stadium")
        self.assertEqual(report["fixtures_matched"], 1)
        self.assertEqual(report["assets_added"], 5)
        self.assertEqual(report["fixtures_with_assets"], 1)

    def test_visual_metadata_matches_unique_team_pair_across_one_day_timezone_drift(self):
        primary = [{
            "Date": "2026-11-16 23:30:00",
            "HomeTeam": "Fortaleza B",
            "AwayTeam": "Tirol/CEFAT",
            "fixture_id": "sstats:123",
            "sstats_game_id": 123,
            "market_odds": {"П1": 2.7, "X": 3.2, "П2": 2.3},
        }]
        metadata = [{
            "Date": "2026-11-17",
            "HomeTeam": "Fortaleza B",
            "AwayTeam": "Tirol/CEFAT",
            "home_badge": "https://assets.test/fortaleza.png",
            "away_badge": "https://assets.test/tirol.png",
            "home_team_tsdb_id": "1001",
            "away_team_tsdb_id": "1002",
        }]

        rows, report = enrich_fixture_visual_metadata(primary, metadata)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["fixture_id"], "sstats:123")
        self.assertEqual(rows[0]["market_odds"], primary[0]["market_odds"])
        self.assertEqual(rows[0]["home_badge"], metadata[0]["home_badge"])
        self.assertEqual(rows[0]["away_badge"], metadata[0]["away_badge"])
        self.assertEqual(report["date_fallback_matches"], 1)
        self.assertEqual(report["ambiguous_team_pair_matches"], 0)

    def test_visual_metadata_skips_equally_nearby_ambiguous_team_pair(self):
        primary = [{
            "Date": "2026-11-16 12:00:00",
            "HomeTeam": "Home FC",
            "AwayTeam": "Away FC",
        }]
        metadata = [
            {
                "Date": "2026-11-15", "HomeTeam": "Home FC",
                "AwayTeam": "Away FC", "home_badge": "https://assets.test/old.png",
            },
            {
                "Date": "2026-11-17", "HomeTeam": "Home FC",
                "AwayTeam": "Away FC", "home_badge": "https://assets.test/new.png",
            },
        ]

        rows, report = enrich_fixture_visual_metadata(primary, metadata)

        self.assertNotIn("home_badge", rows[0])
        self.assertEqual(report["date_fallback_matches"], 0)
        self.assertEqual(report["ambiguous_team_pair_matches"], 1)

    def test_visual_metadata_does_not_overwrite_existing_primary_assets(self):
        primary = [{
            "Date": "2026-11-16",
            "HomeTeam": "Home FC",
            "AwayTeam": "Away FC",
            "home_badge": "primary-home.png",
            "stadium_image": "primary-stadium.jpg",
        }]
        secondary = [{
            "Date": "2026-11-16",
            "HomeTeam": "Home",
            "AwayTeam": "Away",
            "home_badge": "secondary-home.png",
            "away_badge": "secondary-away.png",
            "stadium_image": "secondary-stadium.jpg",
        }]

        rows, report = enrich_fixture_visual_metadata(primary, secondary)

        self.assertEqual(rows[0]["home_badge"], "primary-home.png")
        self.assertEqual(rows[0]["stadium_image"], "primary-stadium.jpg")
        self.assertEqual(rows[0]["away_badge"], "secondary-away.png")
        self.assertEqual(report["assets_added"], 1)

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


    def test_does_not_merge_consecutive_same_team_fixtures_without_matching_kickoff(self):
        rows = merge_fixture_sources(
            [{"Date": "2026-10-20", "Time": "18:00", "HomeTeam": "A", "AwayTeam": "B"}],
            [{"Date": "2026-10-21", "Time": "18:00", "HomeTeam": "A", "AwayTeam": "B"}],
        )
        self.assertEqual(len(rows), 2)

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




    def test_saved_cards_reject_legacy_non_sstats_quotes_and_actions(self):
        cleaned = enforce_sstats_market_card({
            "odds_source": "highlightly",
            "market_odds": {"П1": 2.1, "X": 3.2, "П2": 3.6},
            "bookmaker_by_pick": {"П1": "LegacyBook"},
            "total_odds_source": "manual_import",
            "total_market_odds": {"ТБ 2.5": 1.9},
            "btts_odds_source": "sstats",
            "btts_market_odds": {"BTTS Да": 1.8, "BTTS Нет": 2.0},
            "extra_market_source": "highlightly",
            "stat_market_odds": {"Угл ТБ 9.5": 1.9},
            "asian_handicap_market_odds": {"Ф1 -1.0": 1.9},
            "best": ("1X2", "П1", 2.1),
            "best_by_market": {"winner": {"pick": "П1"}},
            "best_odds_source": "highlightly",
            "verdict": {"real_odds": True, "value_passed": True, "is_bet": True, "odd": 2.1},
        })
        self.assertEqual(cleaned["market_odds"], {})
        self.assertEqual(cleaned["bookmaker_by_pick"], {})
        self.assertEqual(cleaned["total_market_odds"], {})
        self.assertEqual(cleaned["stat_market_odds"], {})
        self.assertEqual(cleaned["asian_handicap_market_odds"], {})
        self.assertEqual(cleaned["btts_market_odds"], {"BTTS Да": 1.8, "BTTS Нет": 2.0})
        self.assertIsNone(cleaned["best"])
        self.assertFalse(cleaned["verdict"]["real_odds"])
        self.assertFalse(cleaned["verdict"]["value_passed"])

    def test_saved_cards_keep_only_explicit_sstats_market_quotes(self):
        cleaned = enforce_sstats_market_card({
            "odds_source": "sstats",
            "market_odds": {"П1": 2.1, "X": 3.2, "П2": 3.6},
            "total_odds_source": "no_market",
            "total_market_odds": {"ТБ 2.5": 1.9},
            "btts_odds_source": "sstats",
            "btts_market_odds": {"BTTS Да": 1.8, "BTTS Нет": 2.0},
            "extra_market_source": "no_market",
            "stat_market_odds": {"Угл ТБ 9.5": 1.9},
        })
        self.assertEqual(cleaned["market_odds"]["П1"], 2.1)
        self.assertEqual(cleaned["btts_market_odds"]["BTTS Да"], 1.8)
        self.assertEqual(cleaned["total_market_odds"], {})
        self.assertEqual(cleaned["stat_market_odds"], {})

    def test_only_explicit_sstats_provenance_is_accepted_for_quotes(self):
        self.assertTrue(is_sstats_market_source("sstats"))
        self.assertTrue(is_sstats_market_source(" SStats "))
        for source in ("", None, "unknown", "highlightly", "manual_import", "football-data"):
            self.assertFalse(is_sstats_market_source(source))

    def test_only_sstats_owned_fixtures_keep_bookmaker_prices(self):
        rows = enforce_sstats_odds_only([
            {
                "Date": "2026-11-11", "HomeTeam": "A", "AwayTeam": "B",
                "fixture_id": "espn-1", "source": "espn",
                "market_odds": {"П1": 2.1},
                "total_market_odds": {"ТБ 2.5": 1.9},
                "bookmaker_by_pick": {"П1": "Example"},
                "odds_source": "espn",
            },
            {
                "Date": "2026-11-11", "HomeTeam": "C", "AwayTeam": "D",
                "fixture_id": "sstats:123", "source": "sstats",
                "sstats_game_id": 123,
                "market_odds": {"П1": 2.2},
                "odds_source": "sstats",
            },
        ])
        self.assertEqual(rows[0]["market_odds"], {})
        self.assertEqual(rows[0]["total_market_odds"], {})
        self.assertEqual(rows[0]["bookmaker_by_pick"], {})
        self.assertEqual(rows[0]["odds_source"], "no_market")
        self.assertEqual(rows[1]["market_odds"], {"П1": 2.2})
        self.assertEqual(rows[1]["odds_source"], "sstats")

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
