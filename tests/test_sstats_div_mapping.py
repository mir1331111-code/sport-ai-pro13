import unittest

from data.sstats import _sstats_div_code, _sstats_division_for_game


class SStatsDivisionMappingTests(unittest.TestCase):
    def test_sstats_supported_competitions_map_to_trained_division_codes(self):
        cases = {
            "Brasileirão Série A": "BR1",
            "Campeonato Brasileiro Serie A": "BR1",
            "Argentine Primera Division": "AR1",
            "Liga Profesional Argentina": "AR1",
            "Admiral Bundesliga": "AT1",
            "Swiss Super League": "CH1",
            "Eliteserien": "NO1",
            "Allsvenskan": "SE1",
            "Danish Superliga": "DK1",
            "Ekstraklasa": "PL1",
            "Meiji Yasuda J1 League": "JP1",
            "K League 1": "KR1",
            "Liga MX": "MX1",
            "Liga BetPlay": "CO1",
            "Scottish Premiership": "SC1",
            "Greek Super League": "G1",
            "Russian Premier League": "R1",
            "Major League Soccer": "MLS1",
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(_sstats_div_code(name), expected)

    def test_domestic_youth_club_competitions_use_separate_age_namespaces(self):
        youth = _sstats_div_code("England U21 Premier League")
        self.assertTrue(youth.startswith("S_Y_U21_"))
        self.assertFalse(youth.startswith("NT_"))

        women_youth = _sstats_div_code("Women's U21 Premier League")
        self.assertTrue(women_youth.startswith("S_WY_U21_"))
        self.assertFalse(women_youth.startswith("W_NT_"))

    def test_youth_national_competitions_still_use_international_namespace(self):
        self.assertTrue(_sstats_div_code("UEFA U21 Championship").startswith("NT_U21_"))
        self.assertTrue(_sstats_div_code("AFC U20 Asian Cup").startswith("NT_U20_"))
        self.assertEqual(
            _sstats_div_code("Men's U21 International Friendly"),
            "NT_U21_FR",
        )

    def test_domestic_national_league_is_not_classified_as_international(self):
        code = _sstats_div_code("National League")
        self.assertTrue(code.startswith("S_"))
        self.assertFalse(code.startswith("NT"))

    def test_non_english_premier_league_is_not_mapped_to_england(self):
        self.assertNotEqual(_sstats_div_code("South Africa Premier League"), "E0")
        self.assertEqual(_sstats_div_code("Premier League"), "E0")

    def test_brazilian_serie_b_is_not_mixed_into_serie_a_history(self):
        self.assertNotEqual(_sstats_div_code("Brasileirão Série B"), "BR1")
        self.assertEqual(_sstats_div_code("Brazilian Serie A"), "BR1")

    def test_country_context_disambiguates_identical_league_names(self):
        self.assertEqual(
            _sstats_division_for_game({}, {"country": "Brazil"}, "Serie A"),
            "BR1",
        )
        self.assertEqual(
            _sstats_division_for_game({}, {"country": {"name": "Austria"}}, "Bundesliga"),
            "AT1",
        )
        self.assertEqual(
            _sstats_division_for_game({}, {"countryName": "Germany"}, "Bundesliga"),
            "D1",
        )
        self.assertEqual(
            _sstats_division_for_game({}, {"country": "Italy"}, "Serie A"),
            "I1",
        )
        self.assertNotEqual(
            _sstats_division_for_game({}, {"country": "South Africa"}, "Premier League"),
            "E0",
        )

    def test_womens_and_international_competitions_keep_separate_namespaces(self):
        self.assertEqual(_sstats_div_code("Women's Super League"), "W_E0")
        self.assertEqual(_sstats_div_code("WSL"), "W_E0")
        self.assertEqual(_sstats_div_code("Liga F"), "W_SP1")
        self.assertEqual(_sstats_div_code("Frauen-Bundesliga"), "W_D1")
        self.assertEqual(_sstats_div_code("Frauen-BL"), "W_D1")
        self.assertEqual(_sstats_div_code("Primera Division Femenina"), "W_SP1")
        self.assertEqual(_sstats_div_code("Serie A Fem"), "W_I1")
        self.assertEqual(_sstats_div_code("UWCL"), "W_C1")
        self.assertEqual(_sstats_div_code("UEFA Europa League"), "EL")
        self.assertEqual(_sstats_div_code("UEFA Champions League"), "C1")
        self.assertEqual(_sstats_div_code("UEFA Nations League"), "NT_UNL")
        self.assertEqual(_sstats_div_code("UEFA Euro Qualifying"), "NT_EQ")
        self.assertEqual(_sstats_div_code("World Cup Qualification Europe"), "NT_WCE")
        self.assertEqual(_sstats_div_code("World Cup Qualifiers CONMEBOL"), "NT_WCC")
        self.assertEqual(_sstats_div_code("World Cup Qualification Asia"), "NT_WCA")
        self.assertEqual(_sstats_div_code("World Cup Qualifiers CAF"), "NT_WCF")
        self.assertEqual(_sstats_div_code("World Cup Qualification CONCACAF"), "NT_WCN")
        self.assertTrue(_sstats_div_code("FIFA Club World Cup").startswith("S_"))


if __name__ == "__main__":
    unittest.main()
