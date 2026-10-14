import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from betting.odds_archive import OddsArchive, clv_decimal


class OddsArchiveTests(unittest.TestCase):
    def test_multi_bookmaker_history_and_clv(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = OddsArchive(directory + "/odds.sqlite")
            start = datetime(2026, 1, 1, tzinfo=timezone.utc)
            archive.record("fx1", "1X2", "H", "book-a", 2.2, start)
            archive.record("fx1", "1X2", "H", "book-b", 2.1, start)
            close = start + timedelta(hours=2)
            archive.record("fx1", "1X2", "H", "book-a", 2.0, close)
            self.assertTrue(archive.mark_closing("fx1", "1X2", "H", "book-a", close))
            self.assertEqual(archive.closing_odds("fx1", "1X2", "H", "book-a"), 2.0)
            self.assertAlmostEqual(clv_decimal(2.2, 2.0), 0.1)
            self.assertEqual(len(archive.snapshots("fx1", "1X2", "H")), 3)

    def test_invalid_odds_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = OddsArchive(directory + "/odds.sqlite")
            with self.assertRaises(ValueError):
                archive.record("fx", "1X2", "H", "book", 1.0, datetime.now(timezone.utc))


if __name__ == "__main__":
    unittest.main()
