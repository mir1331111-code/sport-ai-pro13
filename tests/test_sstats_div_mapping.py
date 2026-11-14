import unittest

from data.sstats import _sstats_div_code


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

    def test_womens_and_international_competitions_keep_separate_namespaces(self):
        self.assertEqual(_sstats_div_code("Women's Super League"), "W_E0")
        self.assertTrue(_sstats_div_code("UEFA Nations League").startswith("NT_"))


if __name__ == "__main__":
    unittest.main()
