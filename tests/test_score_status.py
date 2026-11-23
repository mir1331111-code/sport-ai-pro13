"""Verified 90-minute settlement score gates for provider adapters."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from data import sources
from data.score_status import is_verified_final_score


class VerifiedFinalScoreStatusTests(unittest.TestCase):
    def test_final_score_requires_explicit_status_and_integer_scores(self):
        self.assertTrue(is_verified_final_score({
            "home": 2, "away": 0, "status": "FT",
        }))
        self.assertTrue(is_verified_final_score({
            "home": "1", "away": "1", "status": "match finished",
        }))
        for result in (
            {"home": 2, "away": 0},
            {"home": 2, "away": 0, "status": ""},
            {"home": 2, "away": 0, "status": "unknown"},
            {"home": 2, "away": 0, "status": "AET"},
            {"home": 2, "away": 0, "status": "AET", "regulation_time": False},
            {"home": 2, "away": 0, "status": "FT", "regulation_time": False},
            {"home": -1, "away": 0, "status": "FT"},
            {"home": 1.5, "away": 0, "status": "FT"},
            {"home": True, "away": 0, "status": "FT"},
            {"home": float("nan"), "away": 0, "status": "FT"},
        ):
            with self.subTest(result=result):
                self.assertFalse(is_verified_final_score(result))

    def test_aet_can_only_be_accepted_when_regulation_score_is_explicit(self):
        self.assertTrue(is_verified_final_score({
            "home": 1, "away": 1, "status": "AET",
            "regulation_time": True,
        }))
        self.assertTrue(is_verified_final_score({
            "home": 1, "away": 1, "status": "after penalties",
            "regulation_time": True,
        }))


class LegacyDirectScoreAdapterTests(unittest.TestCase):
    @staticmethod
    def _response(payload):
        return SimpleNamespace(status_code=200, json=lambda: payload)

    def test_tsdb_after_extra_time_and_penalties_are_not_returned_as_ft(self):
        for status in ("After Extra Time", "After Penalties"):
            with self.subTest(status=status), \
                 patch.object(sources, "cache_get", return_value=None), \
                 patch.object(sources, "cache_put"), \
                 patch.object(sources._sess, "get", return_value=self._response({
                     "events": [{
                         "strStatus": status,
                         "strResult": "Final",
                         "intHomeScore": "2",
                         "intAwayScore": "1",
                     }],
                 })):
                result = sources.tsdb_match_result("tsdb-event")
            self.assertEqual(result["status"], "AET")
            self.assertFalse(is_verified_final_score(result))

    def test_tsdb_requires_explicit_normal_full_time_status(self):
        with (
            patch.object(sources, "cache_get", return_value=None),
            patch.object(sources, "cache_put") as cache_write,
            patch.object(sources._sess, "get", return_value=self._response({
                "events": [{
                    "strStatus": "",
                    "strResult": "Final",
                    "intHomeScore": "2",
                    "intAwayScore": "1",
                }],
            })),
        ):
            result = sources.tsdb_match_result("tsdb-no-status")
        self.assertIsNone(result)
        cache_write.assert_not_called()

        with (
            patch.object(sources, "cache_get", return_value=None),
            patch.object(sources, "cache_put"),
            patch.object(sources._sess, "get", return_value=self._response({
                "events": [{
                    "strStatus": "Match Finished",
                    "intHomeScore": "2",
                    "intAwayScore": "1",
                }],
            })),
        ):
            result = sources.tsdb_match_result("tsdb-ft")
        self.assertEqual(result["status"], "FT")
        self.assertTrue(is_verified_final_score(result))

    def test_espn_direct_event_lookup_preserves_extra_time_status(self):
        for status_type in (
            {"id": "45", "name": "STATUS_FINAL", "detail": "Final"},
            {"id": 47, "name": "STATUS_FINAL", "shortDetail": "Final"},
            {"id": "3", "name": "STATUS_FINAL", "detail": "After Penalties"},
        ):
            with self.subTest(status_type=status_type), \
                 patch.object(sources, "cache_get", return_value=None), \
                 patch.object(sources, "cache_put"), \
                 patch.object(sources._sess, "get", return_value=self._response({
                     "events": [{
                         "id": "123",
                         "status": {"type": {"completed": True, **status_type}},
                         "competitions": [{
                             "competitors": [
                                 {"homeAway": "home", "score": "2"},
                                 {"homeAway": "away", "score": "1"},
                             ],
                         }],
                     }],
                 })):
                result = sources.espn_match_result("espn:123", "2026-11-15")
            self.assertEqual(result["status"], "AET")
            self.assertFalse(is_verified_final_score(result))

    def test_espn_direct_event_lookup_keeps_regular_full_time(self):
        with (
            patch.object(sources, "cache_get", return_value=None),
            patch.object(sources, "cache_put"),
            patch.object(sources._sess, "get", return_value=self._response({
                "events": [{
                    "id": "124",
                    "status": {"type": {
                        "id": "3", "name": "STATUS_FINAL",
                        "completed": True, "detail": "Final",
                    }},
                    "competitions": [{
                        "competitors": [
                            {"homeAway": "home", "score": "1"},
                            {"homeAway": "away", "score": "0"},
                        ],
                    }],
                }],
            })),
        ):
            result = sources.espn_match_result("espn:124", "2026-11-15")
        self.assertEqual(result["status"], "FT")
        self.assertTrue(is_verified_final_score(result))


if __name__ == "__main__":
    unittest.main()
