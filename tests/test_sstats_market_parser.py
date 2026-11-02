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

    def test_allows_explicit_full_time_result_alias(self):
        parsed = _sstats_parse_markets([
            bookmaker("Book A", "Full Time Result", 2.10, 3.50, 3.80),
        ])
        self.assertEqual(parsed.get("П1"), 2.10)
        self.assertEqual(parsed.get("X"), 3.50)
        self.assertEqual(parsed.get("П2"), 3.80)


if __name__ == "__main__":
    unittest.main()
