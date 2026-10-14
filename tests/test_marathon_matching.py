import unittest

from data.marathon import _event_date_matches, _find_team_match


class MarathonMatchingTests(unittest.TestCase):
    def test_english_exact_match(self):
        self.assertTrue(_find_team_match("Manchester City", "Manchester City"))

    def test_russian_alias_matches_english(self):
        self.assertTrue(_find_team_match("Manchester City", "Манчестер Сити"))

    def test_partial_name_does_not_match(self):
        self.assertFalse(_find_team_match("Inter Miami", "Inter"))

    def test_date_must_match(self):
        self.assertTrue(_event_date_matches("2026-10-14", "14.10.2026"))
        self.assertFalse(_event_date_matches("2026-10-14", "15.10.2026"))

    def test_date_without_year_matches_day_and_month(self):
        self.assertTrue(_event_date_matches("2026-10-14", "14.10 19:30"))
        self.assertFalse(_event_date_matches("2026-10-14", "15.10 19:30"))

    def test_unknown_event_date_is_rejected_when_date_required(self):
        self.assertFalse(_event_date_matches("2026-10-14", "сегодня"))


if __name__ == "__main__":
    unittest.main()
