import unittest
from betting.bankroll_simulation import simulate_bankroll
from model.validation import fit_ensemble_weights, purged_walk_forward_splits


class ValidationAndBankrollTests(unittest.TestCase):
    def test_embargo_splits_keep_gap(self):
        dates = list(range(20))
        splits = list(purged_walk_forward_splits(dates, min_train=5, test_size=3, embargo=2))
        self.assertEqual(splits[0], (list(range(5)), [7, 8, 9]))
        self.assertTrue(max(splits[0][0]) < min(splits[0][1]))

    def test_unsorted_dates_rejected(self):
        with self.assertRaises(ValueError):
            list(purged_walk_forward_splits([2, 1], 1, 1))

    def test_ensemble_weights_sum_to_one(self):
        a = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]
        b = [[0.4, 0.3, 0.3], [0.2, 0.6, 0.2]]
        result = fit_ensemble_weights([a, b], [0, 1], step=0.1)
        self.assertAlmostEqual(sum(result["weights"]), 1.0)

    def test_monte_carlo_is_reproducible(self):
        a = simulate_bankroll([0.8, -1.0, 0.2], n_simulations=100, seed=7)
        b = simulate_bankroll([0.8, -1.0, 0.2], n_simulations=100, seed=7)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
