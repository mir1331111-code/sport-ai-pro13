"""storage/sqlite_store.py — SQLite с кэшем и batch-операциями (v2).

Что изменено:
  * insert_bets_batch — одна транзакция (раньше при сбое посередине вставлялась часть строк);
  * update_bet / insert_bets_batch сами сбрасывают кэши;
  * clv_breakdown группирует по COALESCE-значению (раньше NULL и '' давали две строки «—»);
  * ошибки log_bank пишутся в лог.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from typing import Iterable, Optional

try:
    import streamlit as st
    _cache = st.cache_data
except Exception:
    def _cache(**_kw):
        def deco(fn):
            fn.clear = lambda: None
            return fn
        return deco

from config import DB_FILE

log = logging.getLogger("sqlite_store")
_LOCK = threading.RLock()
SQLITE_BOOT_OK = False
SQLITE_BOOT_ERROR = ""


@contextmanager
def _db():
    with _LOCK:
        conn = sqlite3.connect(DB_FILE, timeout=15, isolation_level=None)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
        finally:
            conn.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match TEXT NOT NULL, match_ru TEXT, div TEXT, league TEXT, market TEXT,
    pick TEXT NOT NULL, odds REAL NOT NULL, closing_odds REAL,
    stake REAL NOT NULL, prob REAL, status TEXT DEFAULT 'pending',
    score TEXT, strat TEXT, odds_source TEXT, mode TEXT,
    fixture_id TEXT, date TEXT, date_iso TEXT, date_time TEXT,
    ev REAL, clv REAL, settled_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_bets_status ON bets(status);
CREATE INDEX IF NOT EXISTS idx_bets_date ON bets(date_iso);
CREATE INDEX IF NOT EXISTS idx_bets_fixture ON bets(fixture_id);
CREATE INDEX IF NOT EXISTS idx_bets_mode ON bets(mode);
CREATE TABLE IF NOT EXISTS decision_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_key TEXT UNIQUE,
    fixture_id TEXT, match TEXT, match_ru TEXT, league TEXT, market TEXT, pick TEXT,
    decision TEXT, decision_reason TEXT,
    model_prob REAL, fair_odd REAL, market_odd REAL,
    edge REAL, ev REAL, kelly_pct REAL, confidence REAL, value_score REAL,
    odds_source TEXT, date_iso TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_decisions_decision ON decision_snapshots(decision);
CREATE INDEX IF NOT EXISTS idx_decisions_date ON decision_snapshots(date_iso);

CREATE TABLE IF NOT EXISTS bank_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT DEFAULT CURRENT_TIMESTAMP,
    bank REAL NOT NULL, event TEXT, bet_id INTEGER, pnl REAL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def db_init() -> bool:
    global SQLITE_BOOT_OK, SQLITE_BOOT_ERROR
    try:
        with _db() as c:
            c.executescript(SCHEMA)
        SQLITE_BOOT_OK = True
        SQLITE_BOOT_ERROR = ""
        return True
    except Exception as e:
        SQLITE_BOOT_OK = False
        SQLITE_BOOT_ERROR = f"{type(e).__name__}: {e}"
        return False


def invalidate_caches() -> None:
    for fn in ("fetch_bets", "fetch_decision_snapshots", "clv_summary", "clv_breakdown", "bank_history"):
        try:
            globals()[fn].clear()
        except Exception:
            pass


def insert_bets_batch(bets: Iterable) -> int:
    rows = []
    for b in bets:
        if not isinstance(b, dict):
            continue
        rows.append((
            b.get("match"), b.get("match_ru"), b.get("div"), b.get("league"),
            b.get("market"), b.get("pick"), float(b.get("odds") or 0),
            b.get("closing_odds"), float(b.get("stake") or 0),
            b.get("prob"), b.get("status", "pending"), b.get("score"),
            b.get("strat"), b.get("odds_source"), b.get("mode"),
            str(b.get("fixture_id")) if b.get("fixture_id") else None,
            b.get("date"), b.get("date_iso"), b.get("date_time"),
            b.get("ev"), b.get("clv"), b.get("settled_at"),
        ))
    if not rows:
        return 0
    with _db() as c:
        try:
            c.execute("BEGIN")
            c.executemany("""
                INSERT INTO bets (match, match_ru, div, league, market, pick,
                                  odds, closing_odds, stake, prob, status, score,
                                  strat, odds_source, mode, fixture_id,
                                  date, date_iso, date_time, ev, clv, settled_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, rows)
            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise
    invalidate_caches()
    return len(rows)


def insert_decision_snapshots(snapshots: Iterable) -> int:
    rows = []
    for x in snapshots:
        if not isinstance(x, dict):
            continue
        rows.append((
            x.get("snapshot_key"),
            str(x.get("fixture_id")) if x.get("fixture_id") else None,
            x.get("match"), x.get("match_ru"), x.get("league"),
            x.get("market"), x.get("pick"), x.get("decision"),
            x.get("decision_reason"), x.get("model_prob"), x.get("fair_odd"),
            x.get("market_odd"), x.get("edge"), x.get("ev"), x.get("kelly_pct"),
            x.get("confidence"), x.get("value_score"), x.get("odds_source"),
            x.get("date_iso"),
        ))
    if not rows:
        return 0
    with _db() as c:
        c.executemany("""
            INSERT OR IGNORE INTO decision_snapshots (
                snapshot_key, fixture_id, match, match_ru, league, market, pick,
                decision, decision_reason, model_prob, fair_odd, market_odd,
                edge, ev, kelly_pct, confidence, value_score, odds_source, date_iso
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, rows)
    return len(rows)

@_cache(ttl=30, show_spinner=False)
def fetch_decision_snapshots(decision: Optional[str] = None, limit: int = 100000) -> list:
    q = "SELECT * FROM decision_snapshots"
    params = []
    if decision:
        q += " WHERE decision=?"
        params.append(str(decision).upper())
    q += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))
    with _db() as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute(q, params).fetchall()]


def update_bet(bet_id: int, **fields) -> None:
    allowed = {"odds", "closing_odds", "stake", "prob", "status", "score",
               "ev", "clv", "settled_at", "fixture_id", "market"}
    safe = {k: v for k, v in fields.items() if k in allowed}
    if not safe:
        return
    set_sql = ", ".join(f"{k}=?" for k in safe)
    with _db() as c:
        c.execute(f"UPDATE bets SET {set_sql} WHERE id=?",
                  (*safe.values(), bet_id))
    invalidate_caches()


@_cache(ttl=30, show_spinner=False)
def fetch_bets(status: Optional[str] = None, limit: int = 100000) -> list:
    q = "SELECT * FROM bets"
    params = []
    if status:
        q += " WHERE status=?"
        params.append(status)
    q += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))
    with _db() as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute(q, params).fetchall()]


@_cache(ttl=30, show_spinner=False)
def clv_summary() -> dict:
    """CLV-сводка. Безопасна для пустой таблицы."""
    empty = {"n": 0, "avg_clv": 0.0, "positive_share": 0.0}
    try:
        with _db() as c:
            n, avg_v, pos = c.execute(
                "SELECT COUNT(*), AVG(clv), "
                "SUM(CASE WHEN clv > 0 THEN 1 ELSE 0 END) "
                "FROM bets WHERE clv IS NOT NULL").fetchone()
        n = int(n or 0)
        if n <= 0:
            return empty
        return {"n": n, "avg_clv": float(avg_v or 0.0),
                "positive_share": int(pos or 0) / n}
    except Exception as e:
        log.warning("clv_summary: %s", e)
        return empty


@_cache(ttl=30, show_spinner=False)
def clv_breakdown(group_by: str = "market") -> list:
    """CLV-разбивка по рынку или лиге."""
    column = "league" if group_by == "league" else "market"
    try:
        with _db() as c:
            rows = c.execute(
                f"""
                SELECT COALESCE(NULLIF({column}, ''), '—') AS grp,
                       COUNT(*) AS n,
                       AVG(clv) AS avg_clv,
                       SUM(CASE WHEN clv > 0 THEN 1 ELSE 0 END) AS positive_n
                FROM bets
                WHERE clv IS NOT NULL
                GROUP BY grp
                ORDER BY avg_clv DESC
                """
            ).fetchall()
        return [{"group": str(r[0] or "—"), "n": int(r[1] or 0),
                 "avg_clv": float(r[2] or 0.0),
                 "positive_share": float(r[3] or 0) / int(r[1]) if r[1] else 0.0}
                for r in rows]
    except Exception as e:
        log.warning("clv_breakdown: %s", e)
        return []


@_cache(ttl=30, show_spinner=False)
def bank_history(limit: int = 5000) -> list:
    with _db() as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            "SELECT ts, bank, event, pnl FROM bank_history ORDER BY id DESC LIMIT ?",
            (int(limit),)).fetchall()
    return [dict(r) for r in reversed(rows)]


def log_bank(bank: float, event: str = "",
             bet_id: Optional[int] = None,
             pnl: Optional[float] = None) -> None:
    try:
        with _db() as c:
            c.execute(
                "INSERT INTO bank_history(bank,event,bet_id,pnl) VALUES(?,?,?,?)",
                (float(bank), event, bet_id, pnl))
    except Exception as e:
        log.warning("log_bank: %s", e)


def set_meta(key: str, value: str) -> None:
    with _db() as c:
        c.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)",
                  (str(key), str(value)))


def get_meta(key: str, default: Optional[str] = None) -> Optional[str]:
    with _db() as c:
        r = c.execute("SELECT value FROM meta WHERE key=?",
                      (str(key),)).fetchone()
        return r[0] if r else default
