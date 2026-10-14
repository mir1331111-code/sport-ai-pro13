"""SQLite archive for timestamped bookmaker prices and closing-line value."""
from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SCHEMA = """
CREATE TABLE IF NOT EXISTS odds_snapshots (
 id INTEGER PRIMARY KEY,
 fixture_id TEXT NOT NULL,
 market TEXT NOT NULL,
 selection TEXT NOT NULL,
 bookmaker TEXT NOT NULL,
 odds REAL NOT NULL CHECK(odds > 1.0),
 captured_at TEXT NOT NULL,
 is_closing INTEGER NOT NULL DEFAULT 0 CHECK(is_closing IN (0,1)),
 UNIQUE(fixture_id, market, selection, bookmaker, captured_at)
);
CREATE INDEX IF NOT EXISTS idx_odds_lookup
 ON odds_snapshots(fixture_id, market, selection, bookmaker, captured_at);
"""


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat()


class OddsArchive:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    def record(self, fixture_id: str, market: str, selection: str,
               bookmaker: str, odds: float, captured_at: datetime) -> None:
        price = float(odds)
        if not all(str(x).strip() for x in (fixture_id, market, selection, bookmaker)):
            raise ValueError("fixture, market, selection and bookmaker are required")
        if not math.isfinite(price) or price <= 1.0:
            raise ValueError("decimal odds must be finite and > 1")
        stamp = _iso(captured_at)
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO odds_snapshots
                (fixture_id,market,selection,bookmaker,odds,captured_at,is_closing)
                VALUES (?,?,?,?,?,?,0)""",
                (fixture_id, market, selection, bookmaker, price, stamp),
            )

    def mark_closing(self, fixture_id: str, market: str, selection: str,
                     bookmaker: str, captured_at: datetime) -> bool:
        stamp = _iso(captured_at)
        with self._connect() as conn:
            cur = conn.execute(
                """UPDATE odds_snapshots SET is_closing=1
                WHERE fixture_id=? AND market=? AND selection=? AND bookmaker=? AND captured_at=?""",
                (fixture_id, market, selection, bookmaker, stamp),
            )
            return cur.rowcount == 1

    def snapshots(self, fixture_id: str, market: str, selection: str,
                  bookmaker: str | None = None) -> list[dict]:
        sql = """SELECT * FROM odds_snapshots
                 WHERE fixture_id=? AND market=? AND selection=?"""
        args: list = [fixture_id, market, selection]
        if bookmaker is not None:
            sql += " AND bookmaker=?"
            args.append(bookmaker)
        sql += " ORDER BY captured_at"
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(sql, args).fetchall()]

    def closing_odds(self, fixture_id: str, market: str, selection: str,
                     bookmaker: str) -> float | None:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT odds FROM odds_snapshots WHERE fixture_id=? AND market=?
                AND selection=? AND bookmaker=? AND is_closing=1
                ORDER BY captured_at DESC LIMIT 1""",
                (fixture_id, market, selection, bookmaker),
            ).fetchone()
        return float(row["odds"]) if row else None


def clv_decimal(entry_odds: float, closing_odds: float) -> float:
    """Relative decimal-price CLV; positive means entry price beat close.

    This price-based measure is (entry/close)-1. It is not de-vigged and
    comparisons should use the same market, selection and bookmaker.
    """
    entry, close = float(entry_odds), float(closing_odds)
    if not (math.isfinite(entry) and math.isfinite(close)) or entry <= 1 or close <= 1:
        raise ValueError("odds must be finite decimal prices > 1")
    return entry / close - 1.0
