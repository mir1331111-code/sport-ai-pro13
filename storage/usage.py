"""storage/usage.py — счётчики дневных лимитов (локально в JSON)."""
from __future__ import annotations
import json, os
from datetime import datetime

from config import LOCAL_FILE, AUTO_SETTLE_LIMIT, LLM_DAILY_LIMIT, ODDS_LIMIT_DAILY


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _load_all() -> dict:
    try:
        if os.path.exists(LOCAL_FILE):
            with open(LOCAL_FILE, "r", encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception:
        pass
    return {}


def _save_all(data: dict) -> None:
    try:
        tmp = LOCAL_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, LOCAL_FILE)
    except Exception:
        pass


def _load_counter(name: str) -> dict:
    all_ = _load_all()
    u = (all_.get("usage") or {}).get(name)
    if isinstance(u, dict) and u.get("date") == _today():
        return u
    return {"date": _today(), "count": 0}


def _save_counter(name: str, d: dict) -> None:
    all_ = _load_all()
    all_.setdefault("usage", {})[name] = d
    _save_all(all_)


def increment(name: str, n: int = 1) -> dict:
    d = _load_counter(name)
    d["count"] = int(d.get("count", 0)) + n
    _save_counter(name, d)
    return d


def remaining(name: str, limit: int) -> int:
    return max(0, limit - int(_load_counter(name).get("count", 0)))


def reset(name: str) -> dict:
    d = {"date": _today(), "count": 0}
    _save_counter(name, d)
    return d


def settle_remaining() -> int:
    return remaining("settle_usage", AUTO_SETTLE_LIMIT)


def settle_increment(n: int = 1) -> dict:
    return increment("settle_usage", n)


def settle_reset() -> dict:
    return reset("settle_usage")


def llm_remaining() -> int:
    return remaining("llm_usage", LLM_DAILY_LIMIT)


def llm_increment(n: int = 1) -> dict:
    return increment("llm_usage", n)


def llm_reset() -> dict:
    return reset("llm_usage")


def odds_remaining() -> int:
    return remaining("odds_usage", ODDS_LIMIT_DAILY)


def odds_increment(n: int = 1) -> dict:
    return increment("odds_usage", n)


def odds_reset() -> dict:
    return reset("odds_usage")


def get_local_data() -> dict:
    return (_load_all().get("data") or {})


def set_local_data(data: dict) -> None:
    all_ = _load_all()
    all_["data"] = data
    _save_all(all_)
