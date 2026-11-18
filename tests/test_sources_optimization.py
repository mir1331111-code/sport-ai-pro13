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

    @patch("data.sources.cache_get", side_effect=[None, None])
    @patch("data.sources.cache_put")
    @patch("data.sources._sess.get")
    def test_team_logo_lookup_returns_only_matching_soccer_badge(
        self, get_mock, cache_put_mock, _cache_get
    ):
        get_mock.return_value = _Response({
            "teams": [
                {
                    "strTeam": "Arnavutköy Belediyespor",
                    "strSport": "Soccer",
                    "strBadge": "https://assets.test/arnavutkoy.png",
                },
                {
                    "strTeam": "Arnavutköy Belediyespor U19",
                    "strSport": "Soccer",
                    "strBadge": "https://assets.test/arnavutkoy-u19.png",
                },
                {
                    "strTeam": "Arnavutköy Basketball",
                    "strSport": "Basketball",
                    "strBadge": "https://assets.test/basketball.png",
                },
            ]
        })

        badge = sources.team_logo_url("Arnavutkoy Belediyespor")

        self.assertEqual(badge, "https://assets.test/arnavutkoy.png")
        self.assertEqual(get_mock.call_args.args[0], sources._TSDB_TEAM_SEARCH_URL)
        self.assertIn("/123/searchteams.php", get_mock.call_args.args[0])
        self.assertEqual(get_mock.call_args.kwargs["params"], {"t": "Arnavutkoy Belediyespor"})
        self.assertTrue(cache_put_mock.called)
        self.assertTrue(any("tsdb_team_logo_v3_found_" in call.args[0] for call in cache_put_mock.call_args_list))

    @patch("data.sources.cache_get", side_effect=[None, None])
    @patch("data.sources.cache_put")
    @patch("data.sources._sess.get")
    def test_team_logo_lookup_never_uses_senior_crest_for_youth_team(
        self, get_mock, cache_put_mock, _cache_get
    ):
        get_mock.return_value = _Response({
            "teams": [
                {
                    "strTeam": "Manchester United",
                    "strSport": "Soccer",
                    "strTeamBadge": "https://assets.test/man-utd.png",
                }
            ]
        })

        badge = sources.team_logo_url("Manchester United U19")

        self.assertIsNone(badge)
        self.assertTrue(cache_put_mock.called)

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
