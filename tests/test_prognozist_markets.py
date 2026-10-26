import unittest

from data.sources import _prognozist_extract_extra_markets


class PrognozistExtraMarketTests(unittest.TestCase):
    def test_extracts_goals_btts_corners_and_cards_from_detail_page(self):
        html = """
        <h2>Коэффициенты Marathon</h2>
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

    def test_parses_prognozist_rendered_market_layout(self):
        # Mirrors compact labels visible in Prognozist's bookmaker market block.
        html = """
        <h2>Коэффициенты Winline</h2>
        <h3>Тотал</h3>
        <div>Больше (2.5)1.56 Меньше (2.5)2.32 Больше (3.5)2.37 Меньше (3.5)1.54</div>
        <h3>Какая команда забьет?</h3>
        <div>Ни одна 17.00 Только Хозяева 3.45 Только Гости 9.00 Обе 1.55</div>
        <h3>Обе забьют?</h3>
        <div>Да 1.54 Нет 2.36</div>
        """
        result = _prognozist_extract_extra_markets(html)
        self.assertEqual(result["total_market_odds"]["ТБ 2.5"], 1.56)
        self.assertEqual(result["total_market_odds"]["ТМ 2.5"], 2.32)
        self.assertEqual(result["total_market_odds"]["ТБ 3.5"], 2.37)
        self.assertEqual(result["total_market_odds"]["ТМ 3.5"], 1.54)
        self.assertEqual(result["btts_market_odds"]["BTTS Да"], 1.54)
        self.assertEqual(result["btts_market_odds"]["BTTS Нет"], 2.36)
    def test_does_not_claim_bookmaker_from_unrelated_rating_list(self):
        html = """
        <h2>Рейтинг букмекеров</h2><div>Winline Marathon</div>
        <h2>Обе забьют?</h2><div>Да 1.82 Нет 1.98</div>
        """
        result = _prognozist_extract_extra_markets(html)
        self.assertEqual(result["extra_market_source"], "prognozist_unverified")

    def test_prefers_marathon_block_when_multiple_bookmakers_are_present(self):
        html = """
        <h2>Коэффициенты Winline</h2>
        <h3>Обе забьют?</h3><div>Да 1.82 Нет 1.98</div>
        <h2>Коэффициенты Marathon</h2>
        <h3>Тотал</h3><div>Больше 2.5 1.91 Меньше 2.5 1.89</div>
        """
        result = _prognozist_extract_extra_markets(html)
        self.assertEqual(result["extra_market_source"], "prognozist_marathon")
        self.assertEqual(result["total_market_odds"]["ТБ 2.5"], 1.91)
        self.assertEqual(result["total_market_odds"]["ТМ 2.5"], 1.89)
        self.assertEqual(result["btts_market_odds"], {})

    def test_does_not_invent_quotes_when_market_is_missing(self):
        result = _prognozist_extract_extra_markets("<h2>Обе забьют?</h2><div>Только текст</div>")
        self.assertEqual(result["btts_market_odds"], {})
        self.assertEqual(result["total_market_odds"], {})
        self.assertEqual(result["stat_market_odds"], {})


if __name__ == "__main__":
    unittest.main()
