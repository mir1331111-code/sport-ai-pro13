"""storage/usage.py — счётчики дневных лимитов и локальные данные (v2).

Что изменено:
  * read-modify-write под блокировкой (потоки Streamlit больше не затирают друг друга);
  * если файл временно недоступен — НЕ перезаписываем его «пустышкой»
    (раньше increment() мог стереть весь портфель из neuro_local.json);
  * битый JSON сохраняется в *.corrupt-<время>, а не теряется молча;
  * атомарная запись с fsync, права 600 (в файле лежат API-ключи);
  * ошибки пишутся в лог, а не глотаются.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from datetime import datetime
from typing import Callable

from config import (LOCAL_FILE, AUTO_SETTLE_LIMIT, LLM_DAILY_LIMIT,
                    ODDS_LIMIT_DAILY, FOOTBALL_DATA_ORG_DAILY_LIMIT)

log = logging.getLogger("usage")
_LOCK = threading.RLock()


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _backup_corrupt() -> None:
    try:
        dst = f"{LOCAL_FILE}.corrupt-{datetime.now():%Y%m%d_%H%M%S}"
        shutil.copy2(LOCAL_FILE, dst)
        log.error("neuro_local.json повреждён, копия сохранена: %s", dst)
    except Exception as e:
        log.error("не удалось сохранить копию битого файла: %s", e)


def _load_all() -> dict:
    """Читает файл. Бросает OSError, если он недоступен (писать поверх нельзя)."""
    for attempt in range(3):
        try:
            if not os.path.exists(LOCAL_FILE):
                return {}
            with open(LOCAL_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            _backup_corrupt()
            return {}
        except OSError:
            time.sleep(0.1 * (attempt + 1))
    raise OSError(f"{LOCAL_FILE} недоступен")


def _save_all(data: dict) -> bool:
    tmp = f"{LOCAL_FILE}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, LOCAL_FILE)
        return True
    except Exception as e:
        log.error("запись %s не удалась: %s", LOCAL_FILE, e)
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False


def _update(mutator: Callable[[dict], None]) -> bool:
    with _LOCK:
        try:
            all_ = _load_all()
        except OSError as e:
            log.error("%s — запись пропущена, чтобы не затереть данные", e)
            return False
        mutator(all_)
        return _save_all(all_)


def _safe_read() -> dict:
    with _LOCK:
        try:
            return _load_all()
        except OSError:
            return {}


def _load_counter(name: str) -> dict:
    u = (_safe_read().get("usage") or {}).get(name)
    if isinstance(u, dict) and u.get("date") == _today():
        return u
    return {"date": _today(), "count": 0}


def increment(name: str, n: int = 1) -> dict:
    result: dict = {}

    def _m(all_: dict) -> None:
        u = (all_.get("usage") or {}).get(name)
        if not (isinstance(u, dict) and u.get("date") == _today()):
            u = {"date": _today(), "count": 0}
        u["count"] = int(u.get("count", 0)) + n
        all_.setdefault("usage", {})[name] = u
        result.update(u)

    _update(_m)
    return result or _load_counter(name)


def remaining(name: str, limit: int) -> int:
    return max(0, limit - int(_load_counter(name).get("count", 0)))


def reset(name: str) -> dict:
    d = {"date": _today(), "count": 0}
    _update(lambda all_: all_.setdefault("usage", {}).__setitem__(name, d))
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


def fdorg_remaining() -> int:
    return remaining("fdorg_usage", FOOTBALL_DATA_ORG_DAILY_LIMIT)


def fdorg_increment(n: int = 1) -> dict:
    return increment("fdorg_usage", n)


def fdorg_reset() -> dict:
    return reset("fdorg_usage")


def get_local_data() -> dict:
    return _safe_read().get("data") or {}


def set_local_data(data: dict) -> None:
    _update(lambda all_: all_.__setitem__("data", data))
