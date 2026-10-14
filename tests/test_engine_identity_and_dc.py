"""Regression tests for team identity and Dixon-Coles probability invariants."""
import math
import unittest

from model.engine import Engine, _norm


class TeamIdentityTests(unittest.TestCase):
    def test_identity_words_are_not_removed(self):
        self.assertNotEqual(_norm("Manchester City"), _norm("Manchester United"))
        self.assertNotEqual(_norm("Newcastle United"), _norm("Newcastle"))
        self.assertNotEqual(_norm("Leicester City"), _norm("Leicester"))

    def test_punctuation_and_spacing_are_normalized(self):
        self.assertEqual(_norm(" Man-City FC "), _norm("man city fc"))


class DixonColesInvariantTests(unittest.TestCase):
    def test_score_matrix_is_normalized_and_nonnegative(self):
        engine = Engine(matrix_n=12)
        p1, px, matrix = engine._p1px(1.45, 1.10, -0.13)
        p2 = sum(matrix[i][j] for i in range(12) for j in range(12) if i < j)
        self.assertTrue(all(math.isfinite(p) and p >= 0 for row in matrix for p in row))
        self.assertAlmostEqual(sum(map(sum, matrix)), 1.0, places=12)
        self.assertAlmostEqual(p1 + px + p2, 1.0, places=12)

    def test_low_score_tau_ratios_match_documented_parameterization(self):
        engine = Engine(matrix_n=12)
        lh, la, rho = 1.45, 1.10, -0.13
        _, _, matrix = engine._p1px(lh, la, rho)

        # Divide out independent Poisson probabilities. The remaining
        # normalized factor is tau/normalizer; ratios against (2,2) cancel
        # that normalizer and test the four Dixon-Coles corrections directly.
        def poisson(lam, k):
            return math.exp(-lam) * lam ** k / math.factorial(k)

        reference = matrix[2][2] / (poisson(lh, 2) * poisson(la, 2))
        expected_tau = {
            (0, 0): 1 + lh * la * rho,
            (1, 0): 1 - la * rho,
            (0, 1): 1 - lh * rho,
            (1, 1): 1 + rho,
        }
        for (i, j), tau in expected_tau.items():
            actual = matrix[i][j] / (poisson(lh, i) * poisson(la, j))
            self.assertAlmostEqual(actual / reference, tau, places=10)


if __name__ == "__main__":
    unittest.main()
