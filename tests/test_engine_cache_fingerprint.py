import unittest

from model.engine_cache import engine_cache_fingerprint


def fingerprint(*, history=None, xg_rows=None, custom_rows=None, data_version=18):
    return engine_cache_fingerprint(
        app_version="14.4.1",
        data_version=data_version,
        matrix_n=12,
        xg_enabled=True,
        seasonal_history=history or {},
        history_years=[2022, 2023, 2024, 2025, 2026],
        divisions=["E0", "D1"],
        footballapi_xg_rows=xg_rows,
        custom_historical_rows=custom_rows,
    )


class EngineCacheFingerprintTests(unittest.TestCase):
    def test_fingerprint_is_deterministic_for_equivalent_mapping_order(self):
        left = {"E0": [[{"Date": "01/08/2025", "HomeTeam": "A", "FTHG": "1"}]],
                "D1": [[{"Date": "02/08/2025", "HomeTeam": "B", "FTHG": "2"}]]}
        right = {"D1": [[{"FTHG": "2", "HomeTeam": "B", "Date": "02/08/2025"}]],
                 "E0": [[{"FTHG": "1", "HomeTeam": "A", "Date": "01/08/2025"}]]}
        self.assertEqual(fingerprint(history=left), fingerprint(history=right))

    def test_new_seasonal_match_invalidates_engine_cache(self):
        original = {"E0": [[{"Date": "01/08/2025", "HomeTeam": "A",
                             "AwayTeam": "B", "FTHG": "1", "FTAG": "0"}]]}
        updated = {"E0": [[
            {"Date": "01/08/2025", "HomeTeam": "A", "AwayTeam": "B",
             "FTHG": "1", "FTAG": "0"},
            {"Date": "02/08/2025", "HomeTeam": "C", "AwayTeam": "D",
             "FTHG": "2", "FTAG": "1"},
        ]]}
        self.assertNotEqual(fingerprint(history=original), fingerprint(history=updated))

    def test_changed_history_values_invalidate_cache_even_for_same_fixture(self):
        original = {"E0": [[{"Date": "01/08/2025", "HomeTeam": "A",
                             "AwayTeam": "B", "FTHG": "1", "FTAG": "0"}]]}
        corrected = {"E0": [[{"Date": "01/08/2025", "HomeTeam": "A",
                             "AwayTeam": "B", "FTHG": "2", "FTAG": "0"}]]}
        self.assertNotEqual(fingerprint(history=original), fingerprint(history=corrected))

    def test_changed_xg_value_invalidates_cache_even_for_same_match_id(self):
        before = [{"xg_match_id": "same-id", "HxG": 1.25, "AxG": 0.75}]
        after = [{"xg_match_id": "same-id", "HxG": 1.55, "AxG": 0.75}]
        self.assertNotEqual(fingerprint(xg_rows=before), fingerprint(xg_rows=after))

    def test_changed_custom_history_invalidates_cache(self):
        before = [{"Div": "W_E0", "HomeTeam": "A", "FTHG": "1"}]
        after = [{"Div": "W_E0", "HomeTeam": "A", "FTHG": "2"}]
        self.assertNotEqual(fingerprint(custom_rows=before), fingerprint(custom_rows=after))

    def test_data_version_invalidates_cache(self):
        self.assertNotEqual(fingerprint(data_version=18), fingerprint(data_version=19))


if __name__ == "__main__":
    unittest.main()
