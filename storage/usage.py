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
import urllib.request
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


def _secret(name: str) -> str:
    value = os.environ.get(name, "")
    if value:
        return str(value).strip()
    try:
        import streamlit as st
        value = st.secrets.get(name, "")
        return str(value).strip() if value else ""
    except Exception:
        return ""


def _gist_config() -> tuple[str, str]:
    return _secret("GIST_ID"), _secret("GITHUB_TOKEN")


def _gist_read() -> dict:
    gist_id, token = _gist_config()
    if not gist_id or not token:
        return {}
    try:
        req = urllib.request.Request(
            f"https://api.github.com/gists/{gist_id}",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        content = ((payload.get("files") or {}).get("neuro_local.json") or {}).get("content", "")
        data = json.loads(content) if content else {}
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        log.warning("Gist read failed: %s", type(exc).__name__)
        return {}


def _gist_write(data: dict) -> bool:
    gist_id, token = _gist_config()
    if not gist_id or not token:
        return False
    safe = dict(data)
    safe_meta = dict(safe.get("meta") or {})
    for key in ("fdorg_token", "odds_api_key", "llm_api_key"):
        safe_meta.pop(key, None)
    safe["meta"] = safe_meta
    payload = {"files": {"neuro_local.json": {
        "content": json.dumps(safe, ensure_ascii=False, default=str)
    }}}
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    try:
        req = urllib.request.Request(
            f"https://api.github.com/gists/{gist_id}", data=body, method="PATCH",
            headers={"Authorization": f"Bearer {token}",
                     "Accept": "application/vnd.github+json",
                     "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=8):
            return True
    except Exception as exc:
        log.warning("Gist write failed: %s", type(exc).__name__)
        return False


def get_local_data() -> dict:
    local = _safe_read().get("data") or {}
    if local:
        return local
    remote = _gist_read()
    if remote:
        _update(lambda all_: all_.__setitem__("data", remote))
        return remote
    return {}


def set_local_data(data: dict) -> None:
    _update(lambda all_: all_.__setitem__("data", data))
    _gist_write(data)
