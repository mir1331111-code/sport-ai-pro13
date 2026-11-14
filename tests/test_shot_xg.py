import unittest
from datetime import date, timedelta

from model.shot_xg import (
    MIN_TRAINING_SIDES,
    enrich_history_with_shot_xg,
    fit_shot_xg_coefficients,
)


class ShotXGTests(unittest.TestCase):
    def _history(self):
        rows = []
        start = date(2020, 1, 1)
        for index in range(120):
            row = {
                "Date": (start + timedelta(days=index)).isoformat(),
                "Div": "E0",
                "HomeTeam": f"Home{index % 10}",
                "AwayTeam": f"Away{index % 11}",
                "FTHG": 2 if index % 4 == 0 else 1,
                "FTAG": 1 if index % 3 else 0,
                "HS": 14 + index % 4, "HST": 5 + index % 2,
                "AS": 10 + index % 5, "AST": 3 + index % 2,
            }
            rows.append(row)
        return {"E0": [rows]}

    def test_estimates_only_after_prior_history_and_marks_proxy(self):
        history = self._history()
        result = enrich_history_with_shot_xg(history, ["E0"])
        rows = history["E0"][0]
        self.assertEqual(result["generated"], 90)
        self.assertEqual(result["insufficient_history"], 30)
        first = next(row for row in rows if row.get("xg_kind") == "modelled_proxy")
        self.assertGreaterEqual(first["xg_training_sides"], MIN_TRAINING_SIDES)
        self.assertEqual(first["xg_source"], "shot_aggregate_ridge_v1")
        self.assertGreater(first["HxG"], 0)
        self.assertGreater(first["AxG"], 0)

    def test_same_day_results_do_not_train_each_other(self):
        rows = self._history()["E0"][0]
        rows[60]["Date"] = rows[59]["Date"]
        history = {"E0": [rows]}
        enrich_history_with_shot_xg(history, ["E0"])
        same_day = [
            row for row in rows
            if row["Date"] == rows[59]["Date"] and row.get("xg_kind") == "modelled_proxy"
        ]
        self.assertTrue(same_day)
        self.assertEqual(len({row["xg_training_sides"] for row in same_day}), 1)

    def test_preserves_provider_xg_and_rejects_invalid_shot_counts(self):
        history = self._history()
        rows = history["E0"][0]
        rows[80].update({"HxG": 2.1, "AxG": 0.7, "xg_source": "footballapi.org"})
        rows[81]["HST"] = 99
        result = enrich_history_with_shot_xg(history, ["E0"])
        self.assertEqual(rows[80]["xg_source"], "footballapi.org")
        self.assertNotEqual(rows[80].get("xg_kind"), "modelled_proxy")
        self.assertNotIn("xg_kind", rows[81])
        self.assertGreaterEqual(result["provider_preserved"], 1)
        self.assertGreaterEqual(result["missing_shot_data"], 1)

    def test_coefficients_are_bounded_and_small_sample_is_rejected(self):
        self.assertIsNone(fit_shot_xg_coefficients([(4, 6, 1)] * (MIN_TRAINING_SIDES - 1)))
        coefficients = fit_shot_xg_coefficients([(5, 7, 1)] * MIN_TRAINING_SIDES)
        self.assertIsNotNone(coefficients)
        self.assertTrue(0.0 <= coefficients[0] <= 0.25)
        self.assertTrue(0.08 <= coefficients[1] <= 0.55)
        self.assertTrue(0.0 <= coefficients[2] <= 0.10)


if __name__ == "__main__":
    unittest.main()
