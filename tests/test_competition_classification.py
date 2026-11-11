import unittest

from data.competition import (
    competition_gender,
    is_women_competition,
    model_team_identity,
)
from data.fixture_merge import merge_fixture_sources
from data.sources import _tsdb_league_meta


class CompetitionClassificationTests(unittest.TestCase):
    def test_womens_europa_cup_is_not_classified_as_mens_euro(self):
        self.assertEqual(
            _tsdb_league_meta("UEFA Women's Europa Cup"),
            ("W_", "women", True, False),
        )

    def test_explicit_markers_cover_common_provider_spellings(self):
        for label in (
            "Women's Super League",
            "Frauen-Bundesliga",
            "D1 Feminine",
            "UEFA Women’s Europa Cup",
            "Liga F",
        ):
            self.assertTrue(is_women_competition(label), label)

    def test_generic_soccer_label_is_unknown_not_assumed_mens(self):
        self.assertEqual(
            competition_gender(division="G", league="Soccer"),
            "unknown",
        )

    def test_known_mens_competition_and_womens_division(self):
        self.assertEqual(
            competition_gender(division="D1", league="Bundesliga"),
            "men",
        )
        self.assertEqual(
            competition_gender(division="W_", league="Europa Cup"),
            "women",
        )

    def test_team_identity_isolated_by_competition_gender(self):
        self.assertNotEqual(
            model_team_identity("Eintracht Frankfurt", gender="men"),
            model_team_identity("Eintracht Frankfurt", gender="women"),
        )
        self.assertEqual(
            model_team_identity("Eintracht Frankfurt", gender="women"),
            "women::Eintracht Frankfurt",
        )
        self.assertEqual(
            model_team_identity("Mystery FC", gender="unknown"),
            "unknown::Mystery FC",
        )

    def test_fixture_merge_preserves_womens_category_from_any_provider(self):
        generic = {
            "Div": "G", "League": "Soccer",
            "Date": "2026-11-11",
            "HomeTeam": "Eintracht Frankfurt",
            "AwayTeam": "PSV",
            "fixture_id": "espn:123",
            "women": False,
        }
        women = {
            "Div": "W_", "League": "UEFA Women's Europa Cup",
            "Date": "2026-11-11",
            "HomeTeam": "Eintracht Frankfurt",
            "AwayTeam": "PSV",
            "fixture_id": "tsdb:456",
            "women": True, "kind": "women",
        }
        merged = merge_fixture_sources([generic], [women])
        self.assertEqual(len(merged), 1)
        self.assertTrue(merged[0]["women"])
        self.assertEqual(merged[0]["Div"], "W_")
        self.assertEqual(merged[0]["kind"], "women")
        self.assertEqual(merged[0]["League"], "UEFA Women's Europa Cup")


if __name__ == "__main__":
    unittest.main()
