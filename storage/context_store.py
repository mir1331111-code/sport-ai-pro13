"""Persistent referee and pre-match context entities in the main SQLite database."""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _timestamp(value: Any) -> str:
    parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("context timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc).isoformat()


def referee_entity_id(name: str, provider: str = "api-football") -> str | None:
    text = re.sub(r"\s+", " ", str(name or "").strip().casefold())
    if not text:
        return None
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
    # This is a stable internal key, not a claimed upstream numeric referee ID.
    return f"{provider}:referee:{digest}"


class ContextStore:
    """Store confirmed match snapshots and a normalized referee entity.

    Unknown source reliability remains NULL until an empirical source audit
    supplies an estimate. Context is append-only by fixture + observation time.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS referees (
                    referee_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    source_reliability REAL
                        CHECK(source_reliability IS NULL OR
                              (source_reliability >= 0 AND source_reliability <= 1)),
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS fixture_context_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fixture_id TEXT NOT NULL,
                    api_fixture_id TEXT,
                    referee_id TEXT,
                    kickoff_at TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    lineup_confirmed INTEGER NOT NULL CHECK(lineup_confirmed IN (0,1)),
                    injuries_confirmed INTEGER NOT NULL CHECK(injuries_confirmed IN (0,1)),
                    data_source TEXT NOT NULL,
                    source_reliability REAL
                        CHECK(source_reliability IS NULL OR
                              (source_reliability >= 0 AND source_reliability <= 1)),
                    payload_json TEXT NOT NULL,
                    UNIQUE(fixture_id, observed_at)
                );
                CREATE INDEX IF NOT EXISTS idx_context_fixture
                    ON fixture_context_snapshots(fixture_id, observed_at);
                CREATE INDEX IF NOT EXISTS idx_context_referee
                    ON fixture_context_snapshots(referee_id, observed_at);
                """
            )

    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=15)
        connection.row_factory = sqlite3.Row
        return connection

    def record_snapshot(
        self,
        *,
        fixture_id: str,
        kickoff_at: str,
        observed_at: str,
        context: dict,
        data_source: str = "api-football",
    ) -> int:
        fixture_id = str(fixture_id or "").strip()
        if not fixture_id:
            raise ValueError("fixture_id is required")
        kickoff = _timestamp(kickoff_at)
        observed = _timestamp(observed_at)
        if observed >= kickoff:
            raise ValueError("post-kickoff context cannot be stored as pre-match")
        if not isinstance(context, dict):
            raise ValueError("context must be a mapping")
        referee = context.get("referee") or {}
        referee_name = str(referee.get("name") or "").strip()
        referee_id = referee_entity_id(referee_name, data_source)
        reliability = referee.get("source_reliability")
        if reliability is not None:
            reliability = float(reliability)
            if not math.isfinite(reliability) or not 0 <= reliability <= 1:
                raise ValueError("source_reliability must be within [0, 1]")
        lineup = context.get("lineup") or {}
        lineup_confirmed = bool(
            lineup.get("is_confirmed")
            or (
                (lineup.get("home") or {}).get("confirmed")
                and (lineup.get("away") or {}).get("confirmed")
            )
        )
        injuries_confirmed = bool(context.get("injuries_is_confirmed"))
        payload = json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._connect() as connection:
            if referee_id:
                connection.execute(
                    """
                    INSERT INTO referees
                        (referee_id, display_name, provider, source_reliability,
                         first_seen_at, last_seen_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(referee_id) DO UPDATE SET
                        display_name=excluded.display_name,
                        last_seen_at=excluded.last_seen_at,
                        source_reliability=COALESCE(
                            excluded.source_reliability, referees.source_reliability
                        )
                    """,
                    (referee_id, referee_name, data_source, reliability, observed, observed),
                )
            connection.execute(
                """
                INSERT OR IGNORE INTO fixture_context_snapshots
                    (fixture_id, api_fixture_id, referee_id, kickoff_at, observed_at,
                     lineup_confirmed, injuries_confirmed, data_source,
                     source_reliability, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fixture_id,
                    str(context.get("api_fixture_id") or "") or None,
                    referee_id,
                    kickoff,
                    observed,
                    int(lineup_confirmed),
                    int(injuries_confirmed),
                    data_source,
                    reliability,
                    payload,
                ),
            )
            row = connection.execute(
                "SELECT id FROM fixture_context_snapshots WHERE fixture_id=? AND observed_at=?",
                (fixture_id, observed),
            ).fetchone()
            return int(row["id"])

    def referee_rows(self, referee_id: str | None = None) -> list[dict]:
        query = "SELECT * FROM referees"
        params = ()
        if referee_id:
            query += " WHERE referee_id=?"
            params = (str(referee_id),)
        query += " ORDER BY display_name"
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(query, params).fetchall()]

    def snapshots(self, fixture_id: str | None = None) -> list[dict]:
        query = "SELECT * FROM fixture_context_snapshots"
        params = ()
        if fixture_id:
            query += " WHERE fixture_id=?"
            params = (str(fixture_id),)
        query += " ORDER BY observed_at, id"
        with self._connect() as connection:
            rows = [dict(row) for row in connection.execute(query, params).fetchall()]
        for row in rows:
            row["payload"] = json.loads(row.pop("payload_json"))
        return rows
