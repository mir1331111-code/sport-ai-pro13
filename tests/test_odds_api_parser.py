import unittest
from unittest.mock import patch

from data import sources


class _Response:
    status_code = 200
    text = ""

    def json(self):
        return [{
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "bookmakers": [{
                "key": "testbook",
                "markets": [{
                    "key": "totals",
                    "outcomes": [
                        {"name": "Over", "point": 3.5, "price": 2.1},
                        {"name": "Under", "point": 3.5, "price": 1.7},
                    ],
                }],
            }],
        }]


class OddsApiParserTests(unittest.TestCase):
    @patch.object(sources, "cache_get", side_effect=[None, None])
    @patch.object(sources, "cache_put")
    @patch.object(sources.usage, "odds_remaining", return_value=10)
    @patch.object(sources.usage, "odds_increment")
    @patch.object(sources._sess, "get", return_value=_Response())
    def test_keeps_complete_non_25_totals_line(
        self, _get, _increment, _remaining, _cache_put, _cache_get
    ):
        result = sources.odds_api_fixture(
            "soccer_epl", "Arsenal", "Chelsea", "test-key"
        )
        self.assertEqual(result, {"ТБ 3.5": 2.1, "ТМ 3.5": 1.7})


if __name__ == "__main__":
    unittest.main()
