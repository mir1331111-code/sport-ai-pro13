import unittest

from model.validation_cache import select_validation_rows, validation_cache_key


class ValidationCacheTests(unittest.TestCase):
    def _row(self, day, *, xg=None):
        row = {
            "Date": f"{day:02d}/01/2024",
            "HomeTeam": "Home",
            "AwayTeam": "Away",
            "FTHG": 1,
            "FTAG": 0,
            "HC": 5,
            "AC": 4,
            "HY": 2,
            "AY": 1,
        }
        if xg is not None:
            row["HxG"] = xg
            row["AxG"] = 0.5
        return row

    def test_validation_rows_are_capped_sorted_and_scoped_to_requested_leagues(self):
        history = {
            "E0": [[self._row(day) for day in range(1, 21)]],
            "D1": [[self._row(day) for day in range(1, 8)]],
            "OTHER": [[self._row(day) for day in range(1, 8)]],
        }
        rows = select_validation_rows(history, ("E0", "D1"), max_rows_per_league=5)
        self.assertEqual(len(rows), 10)
        self.assertEqual({row["Div"] for row in rows}, {"E0", "D1"})
        e0 = [row for row in rows if row["Div"] == "E0"]
        self.assertEqual([row["Date"] for row in e0],
                         ["16/01/2024", "17/01/2024", "18/01/2024", "19/01/2024", "20/01/2024"])

    def test_invalid_rows_are_skipped(self):
        invalid = self._row(1)
        invalid["FTHG"] = "not-a-score"
        missing_team = self._row(2)
        missing_team["HomeTeam"] = ""
        history = {"E0": [[invalid, missing_team, self._row(3)]]}
        self.assertEqual(len(select_validation_rows(history, ("E0",))), 1)

    def test_fingerprint_is_stable_and_invalidated_by_xg_changes(self):
        rows = select_validation_rows({"E0": [[self._row(1, xg=1.1)]]}, ("E0",))
        same = select_validation_rows({"E0": [[self._row(1, xg=1.1)]]}, ("E0",))
        changed = select_validation_rows({"E0": [[self._row(1, xg=1.4)]]}, ("E0",))
        self.assertEqual(validation_cache_key(rows), validation_cache_key(same))
        self.assertNotEqual(validation_cache_key(rows), validation_cache_key(changed))
        self.assertNotEqual(validation_cache_key(rows, matrix_n=12),
                            validation_cache_key(rows, matrix_n=10))


if __name__ == "__main__":
    unittest.main()
