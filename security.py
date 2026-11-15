"""security.py — безопасный disk-кэш (JSON, не pickle)."""
from __future__ import annotations
import gzip, hashlib, json, os, tempfile, time
from typing import Any, Optional

from config import DISK_CACHE_DIR


def _cache_path(key: str) -> str:
    h = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(DISK_CACHE_DIR, f"{h}.json.gz")


def cache_get(key: str, max_age_sec: float) -> Optional[Any]:
    try:
        p = _cache_path(key)
        if not os.path.exists(p):
            return None
        if time.time() - os.path.getmtime(p) > max_age_sec:
            return None
        with open(p, "rb") as f:
            raw = gzip.decompress(f.read())
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def cache_put(key: str, value: Any) -> None:
    """Write a cache entry via a unique same-directory temporary file.

    Streamlit can run multiple sessions concurrently. Sharing one temporary
    filename for the same key lets writers truncate one another's payload.
    Unique temporary files keep each gzip/JSON document complete; os.replace
    then publishes it atomically for readers.
    """
    temporary_path = None
    try:
        p = _cache_path(key)
        directory = os.path.dirname(p)
        os.makedirs(directory, exist_ok=True)
        payload = json.dumps(
            value, ensure_ascii=False, default=str,
        ).encode("utf-8")
        fd, temporary_path = tempfile.mkstemp(
            prefix=".cache-", suffix=".tmp", dir=directory,
        )
        with os.fdopen(fd, "wb") as handle:
            handle.write(gzip.compress(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, p)
        temporary_path = None
    except Exception:
        # Cache failures are non-fatal; source adapters will fetch again.
        pass
    finally:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


def cache_clear(prefix: str = "") -> None:
    try:
        for fn in os.listdir(DISK_CACHE_DIR):
            if prefix and prefix not in fn:
                continue
            try:
                os.remove(os.path.join(DISK_CACHE_DIR, fn))
            except Exception:
                pass
    except Exception:
        pass
