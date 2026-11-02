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


    def test_accepts_explicit_goals_over_under_market_alias(self):
        def total_book(name, price):
            return {
                "bookmakerName": name,
                "odds": [{
                    "marketName": "Goals Over/Under",
                    "odds": [{"name": "Over 2.5", "value": str(price)}],
                }],
            }

        parsed = _sstats_parse_markets([
            total_book("Book A", 1.85),
            total_book("Book B", 1.90),
        ])
        self.assertEqual(parsed["total_market_odds"]["ТБ 2.5"], 1.90)

    def test_goals_over_under_alias_does_not_accept_period_or_handicap(self):
        for market_name in ("1st Half Goals Over/Under", "Asian Handicap Goals Over/Under"):
            parsed = _sstats_parse_markets([
                {"bookmakerName": "Book A", "odds": [{
                    "marketName": market_name,
                    "odds": [{"name": "Over 2.5", "value": "1.85"}],
                }]},
                {"bookmakerName": "Book B", "odds": [{
                    "marketName": market_name,
                    "odds": [{"name": "Over 2.5", "value": "1.90"}],
                }]},
            ])
            self.assertNotIn("ТБ 2.5", parsed["total_market_odds"])


    def test_accepts_standard_total_market_name_and_rejects_team_total(self):
        def total_book(name, market_name, price):
            return {
                "bookmakerName": name,
                "odds": [{
                    "marketName": market_name,
                    "odds": [{"name": "Over 0.5", "value": str(price)}],
                }],
            }

        parsed = _sstats_parse_markets([
            total_book("Book A", "Total", 1.08),
            total_book("Book B", "Total", 1.09),
        ])
        self.assertEqual(parsed["total_market_odds"]["ТБ 0.5"], 1.09)

        team_total = _sstats_parse_markets([
            total_book("Book A", "Home Team Total", 1.08),
            total_book("Book B", "Home Team Total", 1.09),
        ])
        self.assertNotIn("ТБ 0.5", team_total["total_market_odds"])


    def test_reads_structured_line_when_market_and_outcome_are_generic(self):
        def total_book(name, line):
            return {
                "bookmakerName": name,
                "odds": [{
                    "marketName": "Total",
                    "odds": [
                        {"name": "Over", "point": line, "value": "1.85"},
                        {"name": "Under", "point": line, "value": "1.95"},
                    ],
                }],
            }

        parsed = _sstats_parse_markets([
            total_book("Book A", 2.5),
            total_book("Book B", 2.5),
        ])
        self.assertEqual(parsed["total_market_odds"]["ТБ 2.5"], 1.85)
        self.assertEqual(parsed["total_market_odds"]["ТМ 2.5"], 1.95)

    def test_accepts_reversed_and_russian_total_outcome_labels(self):
        def total_book(name):
            return {
                "bookmakerName": name,
                "odds": [{
                    "marketName": "Total Goals",
                    "odds": [
                        {"name": "2.5 Over", "value": "1.85"},
                        {"name": "2.5 Under", "value": "1.95"},
                        {"name": "ТБ 3.5", "value": "2.20"},
                        {"name": "ТМ 3.5", "value": "1.65"},
                    ],
                }],
            }

        parsed = _sstats_parse_markets([
            total_book("Book A"),
            total_book("Book B"),
        ])
        self.assertEqual(parsed["total_market_odds"]["ТБ 2.5"], 1.85)
        self.assertEqual(parsed["total_market_odds"]["ТМ 2.5"], 1.95)
        self.assertEqual(parsed["total_market_odds"]["ТБ 3.5"], 2.20)
        self.assertEqual(parsed["total_market_odds"]["ТМ 3.5"], 1.65)

    def test_structured_line_does_not_allow_handicap_market_into_totals(self):
        parsed = _sstats_parse_markets([
            {"bookmakerName": "Book A", "odds": [{
                "marketName": "Asian Handicap Goals",
                "odds": [{"name": "Over", "point": 2.5, "value": "1.85"}],
            }]},
            {"bookmakerName": "Book B", "odds": [{
                "marketName": "Asian Handicap Goals",
                "odds": [{"name": "Over", "point": 2.5, "value": "1.90"}],
            }]},
        ])
        self.assertNotIn("ТБ 2.5", parsed["total_market_odds"])

if __name__ == "__main__":
    unittest.main()
