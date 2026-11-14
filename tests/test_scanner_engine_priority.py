import unittest

from ui.scanner_engine import _PRIORITY_DIVS, _validation_divisions


class ScannerEnginePriorityTests(unittest.TestCase):
    def test_validation_divisions_preserve_priority_order_and_training_scope(self):
        train_divs = ["E2", "D1", "E0", "OTHER", "SP1"]
        self.assertEqual(
            _validation_divisions(train_divs),
            ("E0", "SP1", "D1"),
        )

    def test_validation_divisions_handle_empty_and_missing_training_divisions(self):
        self.assertEqual(_validation_divisions([]), ())
        self.assertEqual(_validation_divisions(["E2", "D3"]), ())

    def test_priority_validation_list_is_stable_and_unique(self):
        self.assertEqual(len(_PRIORITY_DIVS), 15)
        self.assertEqual(len(set(_PRIORITY_DIVS)), len(_PRIORITY_DIVS))
        self.assertEqual(_PRIORITY_DIVS[0], "E0")


if __name__ == "__main__":
    unittest.main()
