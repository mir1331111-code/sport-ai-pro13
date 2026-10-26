import unittest

from data.sources import _prognozist_extract_extra_markets


class PrognozistExtraMarketTests(unittest.TestCase):
    def test_extracts_goals_btts_corners_and_cards_from_detail_page(self):
        html = """
        <h2>Тотал голов</h2>
        <div>Больше 2.5 1.91 Меньше 2.5 1.89</div>
        <h2>Обе забьют?</h2>
        <div>Да 1.82 Нет 1.98</div>
        <h2>Тотал угловых</h2>
        <div>Больше 9.5 1.90 Меньше 9.5 1.90</div>
        <h2>Тотал желтых карточек</h2>
        <div>Больше 4.5 1.87 Меньше 4.5 1.93</div>
        """
        result = _prognozist_extract_extra_markets(html)
        self.assertEqual(result["total_market_odds"]["ТБ 2.5"], 1.91)
        self.assertEqual(result["total_market_odds"]["ТМ 2.5"], 1.89)
        self.assertEqual(result["btts_market_odds"]["BTTS Да"], 1.82)
        self.assertEqual(result["btts_market_odds"]["BTTS Нет"], 1.98)
        self.assertEqual(result["stat_market_odds"]["Угл ТБ 9.5"], 1.90)
        self.assertEqual(result["stat_market_odds"]["Угл ТМ 9.5"], 1.90)
        self.assertEqual(result["stat_market_odds"]["Карты ТБ 4.5"], 1.87)
        self.assertEqual(result["stat_market_odds"]["Карты ТМ 4.5"], 1.93)

    def test_does_not_invent_quotes_when_market_is_missing(self):
        result = _prognozist_extract_extra_markets("<h2>Обе забьют?</h2><div>Только текст</div>")
        self.assertNotIn("btts_market_odds", result)
        self.assertNotIn("total_market_odds", result)
        self.assertNotIn("stat_market_odds", result)


if __name__ == "__main__":
    unittest.main()
