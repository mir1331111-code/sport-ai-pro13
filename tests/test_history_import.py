import unittest

from data.history_import import parse_historical_matches_csv


class HistoricalImportTests(unittest.TestCase):
    def test_imports_womens_and_national_teams_as_separate_divisions(self):
        payload = parse_historical_matches_csv(
            "Date,HomeTeam,AwayTeam,FTHG,FTAG,Div,League,HC,AC,Referee\n"
            "2025-04-01,Team A,Team B,2,1,W_E0,WSL,7,4,Ref A\n"
            "2025-04-02,Country A,Country B,0,0,NT_EURO,EURO,5,5,Ref B\n"
        )
        self.assertEqual(payload["accepted"], 2)
        self.assertEqual(payload["divisions"], ["NT_EURO", "W_E0"])
        self.assertEqual(payload["rows"][0]["HC"], "7")
        self.assertEqual(payload["rows"][0]["Referee"], "Ref A")

    def test_missing_scores_or_untrusted_competition_code_are_rejected(self):
        payload = parse_historical_matches_csv(
            "Date,HomeTeam,AwayTeam,FTHG,FTAG,Div\n"
            "2025-04-01,Team A,Team B,,1,W_E0\n"
            "2025-04-02,Team C,Team D,1,0,Women's League\n"
        )
        self.assertEqual(payload["accepted"], 0)
        self.assertEqual(len(payload["rejected"]), 2)

    def test_duplicate_fixture_is_not_trained_twice(self):
        payload = parse_historical_matches_csv(
            "Date,HomeTeam,AwayTeam,FTHG,FTAG,Div\n"
            "01/04/2025,Team A,Team B,2,1,W_E0\n"
            "2025-04-01,Team A,Team B,2,1,W_E0\n"
        )
        self.assertEqual(payload["accepted"], 1)
        self.assertEqual(payload["rejected"][0]["reason"], "duplicate fixture")


if __name__ == "__main__":
    unittest.main()
