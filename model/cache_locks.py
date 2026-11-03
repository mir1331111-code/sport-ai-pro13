"""Per-cache-key locks for Streamlit model loading."""
from __future__ import annotations

import threading


def get_keyed_lock(shared: dict, cache_key: str):
    """Return one re-entrant lock per immutable model cache fingerprint."""
    if not isinstance(shared, dict):
        raise TypeError("shared must be a dictionary")
    guard = shared.get("lock")
    if guard is None:
        guard = threading.RLock()
        shared["lock"] = guard
    locks = shared.setdefault("locks", {})
    with guard:
        lock = locks.get(str(cache_key))
        if lock is None:
            lock = threading.RLock()
            locks[str(cache_key)] = lock
        return lock
