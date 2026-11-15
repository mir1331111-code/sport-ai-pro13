import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from data import sources


class _Response:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class SourceOptimizationTests(unittest.TestCase):
    def test_parse_date_preserves_iso_timezone_offsets(self):
        offset = sources.parse_date("2026-11-15T00:30:45.123+02:00")
        utc = sources.parse_date("2026-11-15T22:15:00Z")
        naive = sources.parse_date("2026-11-15T22:15:00")

        self.assertEqual(offset.isoformat(), "2026-11-15T00:30:45.123000+02:00")
        self.assertEqual(utc.utcoffset().total_seconds(), 0)
        self.assertIsNone(naive.tzinfo)

    def test_active_season_cache_expires_faster_than_completed_seasons(self):
        nov_2026 = datetime(2026, 11, 15, tzinfo=timezone.utc)
        jan_2027 = datetime(2027, 1, 15, tzinfo=timezone.utc)
        july_2027 = datetime(2027, 7, 1, tzinfo=timezone.utc)

        self.assertEqual(
            sources._seasonal_cache_ttl("2627", now=nov_2026),
            sources.CACHE_TTL["seasonal_active"],
        )
        self.assertEqual(
            sources._seasonal_cache_ttl("2526", now=nov_2026),
            sources.CACHE_TTL["seasonal"],
        )
        self.assertEqual(
            sources._seasonal_cache_ttl("2627", now=jan_2027),
            sources.CACHE_TTL["seasonal_active"],
        )
        self.assertEqual(
            sources._seasonal_cache_ttl("2728", now=july_2027),
            sources.CACHE_TTL["seasonal_active"],
        )

    @patch("data.sources.cache_get", return_value=None)
    @patch("data.sources.cache_put")
    @patch("data.sources._sess.get")
    def test_load_seasonal_passes_active_ttl_to_disk_cache(
        self, get_mock, _cache_put, cache_get_mock
    ):
        response = Mock(
            status_code=200,
            content=b"HomeTeam,AwayTeam,FTHG,FTAG\nHome,Away,1,0\n",
        )
        get_mock.return_value = response

        rows = sources.load_seasonal("E0", "2627")

        self.assertEqual(len(rows), 1)
        self.assertEqual(
            cache_get_mock.call_args.args[1],
            sources.CACHE_TTL["seasonal_active"],
        )

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
