"""storage/sqlite_store.py — SQLite с кэшем и batch-операциями."""
from __future__ import annotations
import sqlite3, threading
from contextlib import contextmanager
from typing import Iterable, Optional

try:
    import streamlit as st
    _cache = st.cache_data
except Exception:
    def _cache(**_kw):
        def deco(fn):
            return fn
        return deco

from config import DB_FILE

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
        c.executemany("""
            INSERT INTO bets (match, match_ru, div, league, market, pick,
                              odds, closing_odds, stake, prob, status, score,
                              strat, odds_source, mode, fixture_id,
                              date, date_iso, date_time, ev, clv, settled_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, rows)
    return len(rows)


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
    with _db() as c:
        row = c.execute("""
            SELECT COUNT(*) n, AVG(clv) avg_clv,
                   AVG(CASE WHEN clv > 0 THEN 1.0 ELSE 0.0 END) pos
            FROM bets WHERE clv IS NOT NULL
        """).fetchone()
    if not row or not row["n"]:
        return {"n": 0, "avg_clv": 0.0, "positive_share": 0.0}
    return {"n": row["n"], "avg_clv": row["avg_clv"] or 0.0,
            "positive_share": row["pos"] or 0.0}


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
    except Exception:
        pass


def set_meta(key: str, value: str) -> None:
    with _db() as c:
        c.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)",
                  (str(key), str(value)))


def get_meta(key: str, default: Optional[str] = None) -> Optional[str]:
    with _db() as c:
        r = c.execute("SELECT value FROM meta WHERE key=?",
                      (str(key),)).fetchone()
        return r[0] if r else default


def invalidate_caches() -> None:
    fetch_bets.clear()
    clv_summary.clear()
    bank_history.clear()
