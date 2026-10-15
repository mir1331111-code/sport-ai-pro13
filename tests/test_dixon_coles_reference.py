import math
import unittest

from model.engine import Engine, _norm


def reference_dixon_coles(lh, la, rho, n=12):
    """Independent reference implementation of the standard DC correction."""
    def poisson(lam, k):
        return math.exp(-lam) * lam ** k / math.factorial(k)

    cells = [[poisson(lh, i) * poisson(la, j) for j in range(n)] for i in range(n)]
    tau = {
        (0, 0): 1.0 - lh * la * rho,
        (0, 1): 1.0 + lh * rho,
        (1, 0): 1.0 + la * rho,
        (1, 1): 1.0 - rho,
    }
    for (i, j), factor in tau.items():
        cells[i][j] *= factor
    total = sum(sum(row) for row in cells)
    cells = [[value / total for value in row] for row in cells]
    p_home = sum(cells[i][j] for i in range(n) for j in range(n) if i > j)
    p_draw = sum(cells[i][i] for i in range(n))
    p_away = sum(cells[i][j] for i in range(n) for j in range(n) if i < j)
    return p_home, p_draw, p_away, cells


class DixonColesReferenceTests(unittest.TestCase):
    def test_p1px_matches_independent_reference_for_fixed_lambdas(self):
        engine = Engine(matrix_n=12)
        for lh, la, rho in ((1.42, 0.91, -0.13), (0.72, 1.65, -0.08),
                            (2.05, 1.77, 0.04)):
            actual_home, actual_draw, actual_matrix = engine._p1px(lh, la, rho)
            ref_home, ref_draw, ref_away, ref_matrix = reference_dixon_coles(
                lh, la, rho, n=12
            )
            self.assertAlmostEqual(actual_home, ref_home, places=12)
            self.assertAlmostEqual(actual_draw, ref_draw, places=12)
            self.assertAlmostEqual(sum(map(sum, actual_matrix)), 1.0, places=12)
            for i in range(12):
                for j in range(12):
                    self.assertAlmostEqual(actual_matrix[i][j], ref_matrix[i][j], places=12)
            self.assertAlmostEqual(actual_home + actual_draw, 1.0 - ref_away, places=12)

    def test_h2h_can_be_disabled_without_mutating_history(self):
        engine = Engine()
        for i in range(6):
            engine.add("Home FC", "Away FC", 2 if i % 2 == 0 else 1, 0,
                       match_date=None)
        before = {key: list(value) for key, value in engine.h2h.items()}
        with_h2h = engine.predict("Home FC", "Away FC")
        without_h2h = engine.predict("Home FC", "Away FC", apply_h2h=False)
        self.assertEqual(before, {key: list(value) for key, value in engine.h2h.items()})
        self.assertEqual(without_h2h["h2h_n"], len(engine.h2h[("Home FC", "Away FC")]))
        self.assertEqual(len(with_h2h["lams"]), 2)
        self.assertEqual(len(without_h2h["lams"]), 2)

    def test_normalization_only_removes_terminal_club_designators(self):
        self.assertEqual(_norm("Paris Saint-Germain FC"), _norm("Paris Saint-Germain"))
        self.assertEqual(_norm("AFC Bournemouth"), "afcbournemouth")
        self.assertNotEqual(_norm("Manchester City"), _norm("Manchester"))
        self.assertNotEqual(_norm("Newcastle United"), _norm("Newcastle"))


if __name__ == "__main__":
    unittest.main()
