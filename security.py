"""security.py — безопасный disk-кэш (JSON, не pickle)."""
from __future__ import annotations
import gzip, hashlib, json, os, time
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
    try:
        p = _cache_path(key)
        tmp = p + ".tmp"
        payload = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
        with open(tmp, "wb") as f:
            f.write(gzip.compress(payload))
        os.replace(tmp, p)
    except Exception:
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
