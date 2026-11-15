import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import security


class ConcurrentDiskCacheTests(unittest.TestCase):
    def test_concurrent_writers_publish_complete_payload_without_shared_temp_files(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(security, "DISK_CACHE_DIR", directory):
                def write(index):
                    security.cache_put(
                        "same-fixture-key",
                        {"writer": index, "payload": "x" * 12000},
                    )

                with ThreadPoolExecutor(max_workers=12) as pool:
                    list(pool.map(write, range(96)))

                value = security.cache_get("same-fixture-key", 60)
                self.assertIsInstance(value, dict)
                self.assertIn(value["writer"], range(96))
                self.assertEqual(value["payload"], "x" * 12000)
                remaining = list(Path(directory).iterdir())
                self.assertEqual(len(remaining), 1)
                self.assertTrue(remaining[0].name.endswith(".json.gz"))
                self.assertFalse(any(path.name.endswith(".tmp") for path in remaining))


if __name__ == "__main__":
    unittest.main()
