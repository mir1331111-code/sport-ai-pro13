import unittest

from data.sources import _sstats_parse_markets


def bookmaker(name, market_name, home, draw, away):
    return {
        "bookmakerName": name,
        "odds": [{
            "marketName": market_name,
            "odds": [
                {"name": "Home", "value": str(home)},
                {"name": "Draw", "value": str(draw)},
                {"name": "Away", "value": str(away)},
            ],
        }],
    }


class SStatsMarketParserTests(unittest.TestCase):
    def test_uses_one_complete_bookmaker_line_not_best_prices_mixed_across_books(self):
        parsed = _sstats_parse_markets([
            bookmaker("Book A", "Match Result", 2.10, 3.50, 3.80),
            bookmaker("Book B", "Match Result", 2.30, 3.10, 3.50),
        ])
        # Book A has the lower overround and is selected as a coherent line.
        self.assertEqual(
            (parsed.get("П1"), parsed.get("X"), parsed.get("П2")),
            (2.10, 3.50, 3.80),
        )
        self.assertEqual(
            parsed["bookmaker_by_pick"],
            {"П1": "Book A", "X": "Book A", "П2": "Book A"},
        )

    def test_rejects_ambiguous_winner_and_first_half_markets(self):
        parsed = _sstats_parse_markets([
            bookmaker("Book A", "Winner", 2.10, 3.50, 3.80),
            bookmaker("Book B", "1st Half Match Winner", 2.10, 3.50, 3.80),
        ])
        self.assertNotIn("П1", parsed)
        self.assertNotIn("X", parsed)
        self.assertNotIn("П2", parsed)

    def test_rejects_implausible_1x2_line(self):
        parsed = _sstats_parse_markets([
            bookmaker("Book A", "Match Result", 8.25, 15.00, 7.60),
        ])
        self.assertNotIn("П1", parsed)
        self.assertNotIn("X", parsed)
        self.assertNotIn("П2", parsed)

    def test_rejects_extreme_total_price_outlier(self):
        def total_book(name, price):
            return {
                "bookmakerName": name,
                "odds": [{
                    "marketName": "Total Goals",
                    "odds": [{"name": "Over 2.5", "value": str(price)}],
                }],
            }

        parsed = _sstats_parse_markets([
            total_book("Book A", 1.55),
            total_book("Book B", 1.62),
            total_book("Book C", 1.58),
            total_book("Bad feed", 10.25),
        ])
        self.assertEqual(parsed["total_market_odds"]["ТБ 2.5"], 1.62)
        self.assertNotEqual(parsed["total_market_odds"]["ТБ 2.5"], 10.25)

    def test_does_not_publish_extreme_total_longshot(self):
        parsed = _sstats_parse_markets([{
            "bookmakerName": "Bad feed",
            "odds": [{
                "marketName": "Total Goals",
                "odds": [{"name": "Over 4.5", "value": "41.00"}],
            }],
        }])
        self.assertNotIn("ТБ 4.5", parsed["total_market_odds"])


    def test_rejects_total_from_asian_handicap_market(self):
        parsed = _sstats_parse_markets([{
            "bookmakerName": "Book A",
            "odds": [{
                "marketName": "Asian Handicap Goals",
                "odds": [{"name": "Over 3.5", "value": "8.00"}],
            }],
        }, {
            "bookmakerName": "Book B",
            "odds": [{
                "marketName": "Asian Handicap Goals",
                "odds": [{"name": "Over 3.5", "value": "7.80"}],
            }],
        }])
        self.assertNotIn("ТБ 3.5", parsed["total_market_odds"])

    def test_rejects_total_from_first_half_market(self):
        parsed = _sstats_parse_markets([{
            "bookmakerName": "Book A",
            "odds": [{
                "marketName": "1st Half Total Goals",
                "odds": [{"name": "Over 1.5", "value": "2.00"}],
            }],
        }, {
            "bookmakerName": "Book B",
            "odds": [{
                "marketName": "1st Half Total Goals",
                "odds": [{"name": "Over 1.5", "value": "2.10"}],
            }],
        }])
        self.assertNotIn("ТБ 1.5", parsed["total_market_odds"])

    def test_requires_two_named_books_to_confirm_total(self):
        parsed = _sstats_parse_markets([{
            "bookmakerName": "Only Book",
            "odds": [{
                "marketName": "Total Goals",
                "odds": [{"name": "Over 3.5", "value": "8.00"}],
            }],
        }])
        self.assertNotIn("ТБ 3.5", parsed["total_market_odds"])

    def test_allows_explicit_full_time_result_alias(self):
        parsed = _sstats_parse_markets([
            bookmaker("Book A", "Full Time Result", 2.10, 3.50, 3.80),
        ])
        self.assertEqual(parsed.get("П1"), 2.10)
        self.assertEqual(parsed.get("X"), 3.50)
        self.assertEqual(parsed.get("П2"), 3.80)


if __name__ == "__main__":
    unittest.main()
