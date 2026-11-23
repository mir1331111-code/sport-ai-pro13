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


def _portfolio_fernet():
    """Return the shared backup cipher; never write a new unencrypted portfolio Gist."""
    key = _secret("NEURO_BACKUP_KEY")
    if not key:
        return None
    try:
        from cryptography.fernet import Fernet
        return Fernet(key.encode("ascii"))
    except Exception as exc:
        log.error("Invalid NEURO_BACKUP_KEY for portfolio backup: %s", type(exc).__name__)
        return None


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
        if payload.get("public") is True:
            log.error(
                "Configured Gist is PUBLIC. Legacy neuro_local.json can be read for "
                "recovery, but create a Secret Gist and update GIST_ID immediately."
            )
        file_entry = (payload.get("files") or {}).get("neuro_local.json") or {}
        content = file_entry.get("content", "")
        if file_entry.get("truncated"):
            raw_url = str(file_entry.get("raw_url") or "")
            if not raw_url.startswith("https://gist.githubusercontent.com/"):
                log.error("Gist neuro_local.json is truncated and has no trusted raw_url")
                return {}
            raw_req = urllib.request.Request(
                raw_url,
                headers={"Authorization": f"Bearer {token}",
                         "Accept": "application/vnd.github+json"},
            )
            with urllib.request.urlopen(raw_req, timeout=12) as raw_resp:
                content = raw_resp.read().decode("utf-8")
        data = json.loads(content) if content else {}
        if isinstance(data, dict) and data.get("format") == "neuro-local-fernet-v1":
            cipher = _portfolio_fernet()
            if cipher is None:
                log.error(
                    "Encrypted portfolio Gist cannot be read: configure the original "
                    "NEURO_BACKUP_KEY in Streamlit Secrets."
                )
                return {}
            try:
                plaintext = cipher.decrypt(
                    str(data.get("ciphertext") or "").encode("ascii")
                )
                data = json.loads(plaintext.decode("utf-8"))
            except Exception as exc:
                log.error(
                    "Encrypted portfolio Gist could not be decrypted (%s); "
                    "the remote copy was left untouched.",
                    type(exc).__name__,
                )
                return {}
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
    for key in (
        "fdorg_token", "odds_api_key", "therundown_api_key", "rundown_key",
        "highlightly_api_key", "sstats_api_key", "footballapi_key",
        "llm_api_key", "xai_api_key", "grok_api_key", "groq_api_key",
    ):
        safe_meta.pop(key, None)
    safe["meta"] = safe_meta
    cipher = _portfolio_fernet()
    if cipher is None:
        log.error(
            "Portfolio Gist write refused: configure a valid NEURO_BACKUP_KEY. "
            "No unencrypted portfolio copy was uploaded."
        )
        return False
    plaintext = json.dumps(safe, ensure_ascii=False, default=str).encode("utf-8")
    encrypted = cipher.encrypt(plaintext).decode("ascii")
    envelope = {
        "format": "neuro-local-fernet-v1",
        "ciphertext": encrypted,
    }
    payload = {"files": {"neuro_local.json": {
        "content": json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
    }}}
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    try:
        # Gist "secret" means unlisted, not private. Refuse to keep writing
        # portfolio/bank data into a publicly visible Gist. Read remains allowed
        # above so an existing backup can still be downloaded and migrated.
        check_req = urllib.request.Request(
            f"https://api.github.com/gists/{gist_id}",
            headers={"Authorization": f"Bearer {token}",
                     "Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(check_req, timeout=8) as resp:
            metadata = json.loads(resp.read().decode("utf-8"))
        if not isinstance(metadata, dict) or metadata.get("public") is not False:
            log.error(
                "Gist write refused: configured Gist is public or visibility is unknown. "
                "Create a Secret Gist and update GIST_ID."
            )
            return False

        req = urllib.request.Request(
            f"https://api.github.com/gists/{gist_id}", data=body, method="PATCH",
            headers={"Authorization": f"Bearer {token}",
                     "Accept": "application/vnd.github+json",
                     "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            updated = json.loads(resp.read().decode("utf-8"))
        if not isinstance(updated, dict) or updated.get("public") is not False:
            log.error("Gist write verification failed: Gist is not confirmed non-public.")
            return False
        return True
    except Exception as exc:
        log.warning("Gist write failed: %s", type(exc).__name__)
        return False



# API credentials must never be copied into the durable app-state row.
_SECRET_META_KEYS = {
    "fdorg_token", "odds_api_key", "therundown_api_key", "rundown_key",
    "highlightly_api_key", "sstats_api_key", "footballapi_key",
    "api_football_key", "llm_api_key", "xai_api_key", "grok_api_key",
    "groq_api_key",
}


def _safe_app_state(data: dict) -> dict:
    safe = dict(data)
    meta = dict(safe.get("meta") or {})
    for key in _SECRET_META_KEYS:
        meta.pop(key, None)
    safe["meta"] = meta
    return safe


def _load_sqlite_app_state() -> dict | None:
    """Read the DB-backed state restored by the encrypted SQLite backup."""
    try:
        from storage import sqlite_store as db
        if not db.SQLITE_BOOT_OK and not db.db_init():
            return None
        return db.load_app_state()
    except Exception as exc:
        log.warning("SQLite app-state read unavailable (%s)", type(exc).__name__)
        return None


def _store_sqlite_app_state(data: dict, updated_at_ns: int) -> bool:
    try:
        from storage import sqlite_store as db
        if not db.SQLITE_BOOT_OK and not db.db_init():
            return False
        db.save_app_state(data, updated_at_ns)
        return True
    except Exception as exc:
        log.warning("SQLite app-state write unavailable (%s)", type(exc).__name__)
        return False


def _write_local_app_state(data: dict, updated_at_ns: int) -> bool:
    def _mutate(all_: dict) -> None:
        all_["data"] = data
        all_["data_updated_at_ns"] = int(updated_at_ns)
    return _update(_mutate)


def _legacy_state_from_sqlite() -> dict:
    """Recover bet history and current bank from older SQLite backups without app_state."""
    try:
        from storage import sqlite_store as db
        if not db.SQLITE_BOOT_OK and not db.db_init():
            return {}
        bets = db.fetch_bets(limit=100000)
        if not bets:
            return {}
        bets = list(reversed(bets))
        for bet in bets:
            fixture_id = str(bet.get("fixture_id") or "")
            if fixture_id.startswith("sstats:"):
                candidate = fixture_id.split(":", 1)[1]
                if candidate.isdigit():
                    bet.setdefault("sstats_game_id", int(candidate))
            match = str(bet.get("match") or "")
            if " vs " in match:
                home, away = match.split(" vs ", 1)
                bet.setdefault("home", home.strip())
                bet.setdefault("away", away.strip())
        history = db.bank_history(limit=1)
        initial_bank = float(db.get_meta("initial_bank", "10000") or 10000)
        bank = float(history[-1]["bank"]) if history else initial_bank
        stats = {"won": 0, "lost": 0, "push": 0, "void": 0, "profit": 0.0}
        for bet in bets:
            try:
                stake, odds = float(bet.get("stake") or 0), float(bet.get("odds") or 0)
            except (TypeError, ValueError):
                continue
            status = str(bet.get("status") or "pending")
            if status == "won":
                stats["won"] += 1
                stats["profit"] += stake * (odds - 1)
            elif status == "lost":
                stats["lost"] += 1
                stats["profit"] -= stake
            elif status in {"push", "void"}:
                stats[status] += 1
        return {
            "version": 1, "bank": bank, "bets": bets, "cards": [],
            "funnel": None, "report": [],
            "meta": {"initial_bank": initial_bank,
                     "portfolio_recovered_from_legacy_sqlite": True},
            "stats": stats, "mode": "paper",
        }
    except Exception as exc:
        log.warning("Legacy SQLite portfolio recovery unavailable (%s)", type(exc).__name__)
        return {}


def get_local_data() -> dict:
    """Load the newest portfolio state from SQLite or legacy local/Gist storage."""
    local_all = _safe_read()
    local = local_all.get("data") or {}
    try:
        local_stamp = int(local_all.get("data_updated_at_ns") or 0)
    except (TypeError, ValueError):
        local_stamp = 0

    sqlite_state = _load_sqlite_app_state()
    if sqlite_state:
        sqlite_data = sqlite_state["data"]
        sqlite_stamp = int(sqlite_state["updated_at_ns"])
        if local and local_stamp > sqlite_stamp:
            safe_local = _safe_app_state(local)
            if not _store_sqlite_app_state(safe_local, local_stamp):
                log.error(
                    "Newest local portfolio state could not be promoted to SQLite; "
                    "remote SQLite backup must remain unconfirmed."
                )
            return safe_local
        if local != sqlite_data or local_stamp != sqlite_stamp:
            _write_local_app_state(sqlite_data, sqlite_stamp)
        return sqlite_data

    if local:
        safe_local = _safe_app_state(local)
        stamp = local_stamp or time.time_ns()
        _store_sqlite_app_state(safe_local, stamp)
        _write_local_app_state(safe_local, stamp)
        return safe_local

    remote = _gist_read()
    if remote:
        remote_data = remote.get("data") if isinstance(remote.get("data"), dict) else remote
        if not isinstance(remote_data, dict):
            remote_data = {}
        safe_remote = _safe_app_state(remote_data)
        stamp = time.time_ns()
        if not _store_sqlite_app_state(safe_remote, stamp):
            log.error(
                "Recovered legacy Gist portfolio is available in memory but could not "
                "be persisted to SQLite; encrypted SQLite backup remains incomplete."
            )
        _write_local_app_state(safe_remote, stamp)
        # Migrate a legacy plaintext Gist to the encrypted envelope on first
        # recovery. For an already-encrypted Gist this refreshes its snapshot.
        if _gist_write(safe_remote):
            log.info("Recovered portfolio data was saved back to encrypted Gist.")
        return safe_remote

    # Older one-token SQLite backups predate the app-state row. Recover what is
    # still available (bets and balance); cards/reports are not reconstructed
    # from incomplete market rows.
    legacy_sqlite = _legacy_state_from_sqlite()
    if legacy_sqlite:
        safe_legacy = _safe_app_state(legacy_sqlite)
        stamp = time.time_ns()
        _store_sqlite_app_state(safe_legacy, stamp)
        _write_local_app_state(safe_legacy, stamp)
        return safe_legacy
    return {}


def set_local_data(data: dict) -> bool:
    """Persist app state to SQLite first, then update local/legacy convenience mirrors."""
    safe = _safe_app_state(data)
    stamp = time.time_ns()
    sqlite_saved = _store_sqlite_app_state(safe, stamp)
    if not sqlite_saved:
        log.error(
            "Portfolio/dashboard state was NOT persisted to SQLite. "
            "Do not treat the encrypted SQLite remote backup as current."
        )
    local_saved = _write_local_app_state(safe, stamp)
    if not local_saved:
        log.error("Portfolio/dashboard local mirror was not saved.")
    _gist_write(safe)
    return bool(sqlite_saved)
