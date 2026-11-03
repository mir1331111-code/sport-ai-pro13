import unittest

from betting.verdict import build_verdict


def _prediction(xg_used):
    return {
        "p1": 0.55, "x": 0.25, "p2": 0.20,
        "over": 0.52, "btts": 0.50,
        "lams": (1.45, 1.05),
        "xg_used": xg_used,
        "xg_league_samples": 35 if xg_used else 0,
        "xg_weight": 0.28 if xg_used else 0.0,
    }


class VerdictXGDisclosureTests(unittest.TestCase):
    def test_does_not_label_goal_lambdas_as_observed_xg(self):
        verdict, _, _ = build_verdict(
            _prediction(False), 0.40, 1000, 0.25,
            "Home", "Away", "ВВП", "ВПП", 2,
        )
        reasons = " ".join(verdict["reasons"])
        self.assertIn("Ожидаемые голы модели", reasons)
        self.assertIn("xG в этом прогнозе не использован", reasons)
        self.assertNotIn("(xG 1.45-1.05)", reasons)

    def test_discloses_when_xg_was_actually_blended(self):
        verdict, _, _ = build_verdict(
            _prediction(True), 0.40, 1000, 0.25,
            "Home", "Away", "ВВП", "ВПП", 2,
        )
        reasons = " ".join(verdict["reasons"])
        self.assertIn("xG действительно учтён", reasons)
        self.assertIn("35 матчей", reasons)


if __name__ == "__main__":
    unittest.main()
