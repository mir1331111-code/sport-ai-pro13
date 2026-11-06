import unittest

from data.team_names import normalize_team, teams_match


class TeamNameNormalizationTests(unittest.TestCase):
    def test_manchester_city_and_united_remain_distinct(self):
        self.assertEqual(normalize_team("Manchester United"), "man_united")
        self.assertEqual(normalize_team("Manchester City"), "man_city")
        self.assertNotEqual(
            normalize_team("Manchester United"),
            normalize_team("Manchester City"),
        )

    def test_common_provider_aliases_still_match(self):
        self.assertTrue(teams_match("Man Utd", "Manchester United"))
        self.assertTrue(teams_match("Man City", "Manchester City"))
        self.assertTrue(teams_match("FC Barcelona", "Barcelona"))

    def test_youth_teams_do_not_collapse_to_senior_team(self):
        self.assertNotEqual(
            normalize_team("Manchester City U21"),
            normalize_team("Manchester City"),
        )
        self.assertNotEqual(
            normalize_team("Arsenal U21"),
            normalize_team("Arsenal"),
        )

    def test_reserve_teams_do_not_collapse_to_senior_team(self):
        self.assertNotEqual(
            normalize_team("Barcelona II"),
            normalize_team("Barcelona"),
        )

    def test_b_teams_and_castilla_do_not_collapse_to_senior_team(self):
        self.assertNotEqual(
            normalize_team("Atletico Madrid B"),
            normalize_team("Atletico Madrid"),
        )
        self.assertNotEqual(
            normalize_team("Real Madrid Castilla"),
            normalize_team("Real Madrid"),
        )
        self.assertNotEqual(
            normalize_team("Manchester United B"),
            normalize_team("Manchester United"),
        )

    def test_exact_aliases_allow_punctuation_and_accent_variants(self):
        self.assertEqual(normalize_team("Paris Saint-Germain"), normalize_team("PSG"))
        self.assertEqual(normalize_team("Brighton & Hove Albion"), normalize_team("Brighton"))
        self.assertEqual(normalize_team("FC Barcelona"), normalize_team("Barça"))


if __name__ == "__main__":
    unittest.main()
