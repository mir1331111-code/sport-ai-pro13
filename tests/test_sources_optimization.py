import unittest
from unittest.mock import Mock, patch

from data import sources


class _Response:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class SourceOptimizationTests(unittest.TestCase):
    @patch("data.sources.cache_get", return_value=None)
    @patch("data.sources.cache_put")
    @patch("data.sources._sess.get")
    @patch("data.sources._tsdb_venue_metadata")
    def test_tsdb_venue_enrichment_resolves_event_ids_without_cross_matching(
        self, venue_mock, get_mock, _cache_put, _cache_get
    ):
        events = [
            {
                "idEvent": "event-1", "idVenue": "venue-1",
                "strSport": "Soccer", "strLeague": "Premier League",
                "strHomeTeam": "Home One", "strAwayTeam": "Away One",
                "dateEvent": "2026-11-16", "strVenueThumb": "",
            },
            {
                "idEvent": "event-2", "idVenue": "venue-2",
                "strSport": "Soccer", "strLeague": "Premier League",
                "strHomeTeam": "Home Two", "strAwayTeam": "Away Two",
                "dateEvent": "2026-11-16", "strVenueThumb": "existing-image",
            },
            {
                "idEvent": "event-3", "idVenue": "venue-3",
                "strSport": "Soccer", "strLeague": "Premier League",
                "strHomeTeam": "Home Three", "strAwayTeam": "Away Three",
                "dateEvent": "2026-11-16", "strVenueThumb": "",
            },
        ]
        get_mock.return_value = _Response({"events": events})
        venue_mock.side_effect = lambda venue_id, _logs: {
            "stadium_image": f"enriched:{venue_id}"
        }

        rows = sources.tsdb_matches(days=1, logs=[])

        self.assertEqual(
            [row["fixture_id"] for row in rows],
            ["event-1", "event-2", "event-3"],
        )
        self.assertEqual(
            [call.args[0] for call in venue_mock.call_args_list],
            ["venue-1", "venue-3"],
        )
        self.assertEqual(rows[0]["stadium_image"], "enriched:venue-1")
        self.assertEqual(rows[1]["stadium_image"], "existing-image")
        self.assertEqual(rows[2]["stadium_image"], "enriched:venue-3")


if __name__ == "__main__":
    unittest.main()
