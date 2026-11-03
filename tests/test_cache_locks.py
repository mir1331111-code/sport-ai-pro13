import threading
import unittest

from model.cache_locks import get_keyed_lock


class KeyedModelLockTests(unittest.TestCase):
    def test_same_cache_key_reuses_same_lock(self):
        shared = {"engines": {}, "lock": threading.RLock()}
        self.assertIs(get_keyed_lock(shared, "same"), get_keyed_lock(shared, "same"))

    def test_different_cache_keys_do_not_share_a_blocking_lock(self):
        shared = {"engines": {}, "lock": threading.RLock()}
        first = get_keyed_lock(shared, "model-a")
        second = get_keyed_lock(shared, "model-b")
        self.assertIsNot(first, second)
        self.assertTrue(first.acquire(blocking=False))
        try:
            self.assertTrue(second.acquire(blocking=False))
            second.release()
        finally:
            first.release()

    def test_existing_shared_cache_without_locks_is_upgraded(self):
        shared = {"engines": {}, "lock": threading.RLock()}
        lock = get_keyed_lock(shared, "model-a")
        self.assertIs(shared["locks"]["model-a"], lock)


if __name__ == "__main__":
    unittest.main()
