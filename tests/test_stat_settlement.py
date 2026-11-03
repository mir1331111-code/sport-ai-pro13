import unittest

from betting.settlement import determine_stat_outcome


class StatisticSettlementTests(unittest.TestCase):
    def test_corner_totals_over_under_and_push(self):
        self.assertEqual(determine_stat_outcome("STAT", "Угл ТБ 9.5", 10), ("won", "stat_corners"))
        self.assertEqual(determine_stat_outcome("STAT", "Угл ТМ 9.5", 9), ("won", "stat_corners"))
        self.assertEqual(determine_stat_outcome("STAT", "Угл ТБ 10", 10), ("push", "stat_corners"))

    def test_card_totals_and_validation(self):
        self.assertEqual(determine_stat_outcome("STAT", "Карты ТМ 4.5", 4), ("won", "stat_cards"))
        self.assertEqual(determine_stat_outcome("STAT", "Карты ТБ 4.5", 4), ("lost", "stat_cards"))
        self.assertEqual(determine_stat_outcome("STAT", "Карты ТБ 4.5", -1), (None, "bad_stat_value"))
        self.assertEqual(determine_stat_outcome("STAT", "Карты ТБ 4.5", 4.5), (None, "bad_stat_value"))
        self.assertIsNone(determine_stat_outcome("STAT", "ЖК ТБ 4.5", 5)[0])


if __name__ == "__main__":
    unittest.main()
