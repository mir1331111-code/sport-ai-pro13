import unittest

from betting.ranking import rank_model_favorites


def _card(name, *, probability, games):
    remaining = (1.0 - probability) / 2.0
    return {
        "match": name,
        "date_iso": "2026-11-07",
        "p1": probability,
        "px": remaining,
        "p2": remaining,
        "games": games,
    }


class FavoriteHistoryGateTests(unittest.TestCase):
    def test_shortlist_can_exclude_fixtures_without_minimum_team_history(self):
        unsupported = _card("Unknown A vs Unknown B", probability=0.80, games=0)
        supported = _card("Known A vs Known B", probability=0.66, games=8)

        favorites = rank_model_favorites(
            [unsupported, supported], limit=10, min_sample_games=5
        )

        self.assertEqual(len(favorites), 1)
        self.assertIs(favorites[0]["card"], supported)
        self.assertEqual(favorites[0]["sample_games"], 8)

    def test_default_ranking_remains_backward_compatible(self):
        unsupported = _card("Unknown A vs Unknown B", probability=0.80, games=0)
        self.assertEqual(len(rank_model_favorites([unsupported])), 1)


if __name__ == "__main__":
    unittest.main()
