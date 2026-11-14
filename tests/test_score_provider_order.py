import unittest
from unittest.mock import patch

from data import scores


class ScoreProviderOrderTests(unittest.TestCase):
    def test_provider_order_is_explicit_and_fdorg_receives_caller_token(self):
        calls = []

        def espn(home, away, day):
            calls.append("espn")
            return None

        def fdorg(home, away, day, token=None):
            calls.append(("fdorg", token))
            return {"home": 2, "away": 1, "status": "FT", "source": "football-data.org"}

        with patch.object(scores, "_find_espn", side_effect=espn), \
             patch.object(scores, "_find_fdorg", side_effect=fdorg), \
             patch.object(scores, "_find_olb") as olb:
            result = scores.find_match_score(
                "Home FC", "Away FC", "2026-11-14",
                token="caller-token", providers=("espn", "fdorg", "olb"),
            )

        self.assertEqual(calls, ["espn", ("fdorg", "caller-token")])
        self.assertEqual(result["source"], "football-data.org")
        olb.assert_not_called()

    def test_default_order_remains_backward_compatible(self):
        calls = []

        def found(name):
            def inner(*args, **kwargs):
                calls.append(name)
                return {"home": 1, "away": 0, "status": "FT", "source": name}
            return inner

        with patch.object(scores, "_find_espn", side_effect=found("espn")), \
             patch.object(scores, "_find_fdorg", side_effect=found("fdorg")):
            result = scores.find_match_score("Home", "Away", "2026-11-14")

        self.assertEqual(calls, ["espn"])
        self.assertEqual(result["source"], "espn")


if __name__ == "__main__":
    unittest.main()
