"""Unit tests for Kelly sizing and deterministic settlement edge cases."""
import unittest

from betting.kelly import edge, expected_value, kelly
from betting.settlement import determine_outcome


class KellyTests(unittest.TestCase):
    def test_no_stake_without_positive_edge(self):
        self.assertEqual(kelly(0.5, 2.0, 1000), 0.0)
        self.assertEqual(kelly(0.4, 2.0, 1000), 0.0)

    def test_fractional_kelly_respects_cap(self):
        stake = kelly(0.9, 3.0, 1000, frac=1.0, cap=0.05)
        self.assertEqual(stake, 50.0)

    def test_invalid_inputs_are_safe(self):
        for args in [
            (0.0, 2.0, 1000),
            (1.0, 2.0, 1000),
            (0.7, 1.0, 1000),
            (0.7, 2.0, 0),
            (float("nan"), 2.0, 1000),
        ]:
            self.assertEqual(kelly(*args), 0.0)

    def test_edge_and_ev(self):
        self.assertAlmostEqual(edge(0.6, 2.0), 0.1)
        self.assertAlmostEqual(expected_value(0.6, 2.0), 0.2)


class SettlementTests(unittest.TestCase):
    def test_1x2_and_draw(self):
        self.assertEqual(determine_outcome("1X2", "П1", 2, 0)[0], "won")
        self.assertEqual(determine_outcome("1X2", "X", 1, 1)[0], "won")
        self.assertEqual(determine_outcome("1X2", "П2", 2, 0)[0], "lost")

    def test_total_win_loss_and_push(self):
        self.assertEqual(determine_outcome("OU", "ТБ 2.5", 2, 1)[0], "won")
        self.assertEqual(determine_outcome("OU", "ТМ 2.5", 2, 1)[0], "lost")
        self.assertEqual(determine_outcome("OU", "ТМ 3", 2, 1)[0], "push")

    def test_quarter_line_requires_manual_review(self):
        self.assertIsNone(determine_outcome("OU", "ТБ 2.25", 2, 1)[0])
        self.assertIsNone(determine_outcome("AH", "Ф1(-0.25)", 1, 0)[0])

    def test_unknown_pick_is_not_silently_lost(self):
        self.assertIsNone(determine_outcome("1X2", "maybe", 2, 0)[0])

    def test_negative_score_is_rejected(self):
        self.assertIsNone(determine_outcome("1X2", "П1", -1, 0)[0])


if __name__ == "__main__":
    unittest.main()
