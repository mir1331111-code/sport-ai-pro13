"""Restore a downloaded forecast audit ZIP into an empty SQLite database."""
from __future__ import annotations

import json
import math
import os
import sqlite3
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model.forecast_ledger import ForecastLedger, MarketForecastLedger
from storage import sqlite_store


MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_ROWS_PER_TABLE = 1_000_000

FORECAST_FIELDS = (
    "id", "forecast_key", "fixture_id", "kickoff_at", "forecast_at",
    "model_version", "home_team", "away_team", "league",
    "probabilities_json", "entry_odds_json", "odds_source",
    "xg_used", "result", "home_score", "away_score",
    "closing_odds_json", "closing_odds_source", "settled_at", "created_at",
)
MARKET_FIELDS = (
    "id", "forecast_key", "fixture_id", "kickoff_at", "forecast_at",
    "model_version", "home_team", "away_team", "league", "market",
    "selection", "probability", "entry_odds", "odds_source", "unit_stake",
    "result_status", "result_score", "actual_value", "closing_odds",
    "settlement_source", "settled_at", "created_at",
)
BET_FIELDS = (
    "id", "match", "match_ru", "div", "league", "market", "pick",
    "odds", "closing_odds", "stake", "prob", "status", "score", "strat",
    "odds_source", "mode", "fixture_id", "date", "date_iso", "date_time",
    "ev", "clv", "settled_at", "created_at",
)
DECISION_FIELDS = (
    "id", "snapshot_key", "fixture_id", "match", "match_ru", "league",
    "market", "pick", "decision", "decision_reason", "model_prob",
    "fair_odd", "market_odd", "edge", "ev", "kelly_pct", "confidence",
    "value_score", "odds_source", "date_iso", "created_at", "result_status",
    "result_score", "virtual_pnl", "evaluated_at",
)


def _json_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = value
        value = decoded
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def _require_list(payload: dict, field: str) -> list[dict]:
    rows = payload.get(field, [])
    if not isinstance(rows, list):
        raise ValueError(f"audit field {field!r} must be an array")
    if len(rows) > MAX_ROWS_PER_TABLE:
        raise ValueError(f"audit field {field!r} exceeds the row safety limit")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"audit field {field!r} contains a non-object row")
    return rows


def _ensure_sqlite_schema(path: Path) -> None:
    with sqlite3.connect(str(path), timeout=15) as conn:
        conn.executescript(sqlite_store.SCHEMA)
        columns = {
            str(row[1]) for row in conn.execute(
                "PRAGMA table_info(decision_snapshots)"
            ).fetchall()
        }
        migrations = {
            "result_status": "TEXT",
            "result_score": "TEXT",
            "virtual_pnl": "REAL",
            "evaluated_at": "TEXT",
        }
        for column, kind in migrations.items():
            if column not in columns:
                conn.execute(
                    f"ALTER TABLE decision_snapshots ADD COLUMN {column} {kind}"
                )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_decisions_result "
            "ON decision_snapshots(result_status)"
        )


def _insert_rows(
    conn: sqlite3.Connection,
    table: str,
    fields: tuple[str, ...],
    rows: list[dict],
    *,
    json_fields: tuple[str, ...] = (),
) -> int:
    allowed = set(fields)
    inserted = 0
    columns = [field for field in fields if any(field in row for row in rows)]
    if not rows or not columns:
        return 0
    quoted = ",".join(f'"{field}"' for field in columns)
    placeholders = ",".join("?" for _ in columns)
    sql = f'INSERT INTO "{table}" ({quoted}) VALUES ({placeholders})'
    for row in rows:
        unknown = set(row) - allowed
        # Export files can gain additive, non-secret metadata; unknown columns
        # are ignored rather than dynamically executed as SQL.
        values = []
        for field in columns:
            value = row.get(field)
            if field in json_fields:
                value = _json_value(value)
            if field == "xg_used" and value is not None:
                if value not in (0, 1, False, True):
                    raise ValueError("forecast xg_used contains an invalid value")
                value = int(bool(value))
            values.append(value)
        conn.execute(sql, values)
        inserted += 1
    return inserted


def restore_audit_export_bytes(
    archive_bytes: bytes,
    db_path: str | os.PathLike,
) -> dict[str, Any]:
    """Restore all supported audit tables only when the target DB has no rows."""
    from storage.remote_backup import _database_has_user_rows

    if not isinstance(archive_bytes, (bytes, bytearray)) or not archive_bytes:
        raise ValueError("audit archive is empty")
    if len(archive_bytes) > MAX_ARCHIVE_BYTES:
        raise ValueError("audit archive exceeds the upload safety limit")

    try:
        with zipfile.ZipFile(archive_bytes, "r") as archive:
            try:
                info = archive.getinfo("audit.json")
            except KeyError as exc:
                raise ValueError("audit archive has no audit.json") from exc
            if info.file_size > MAX_ARCHIVE_BYTES:
                raise ValueError("audit.json exceeds the upload safety limit")
            payload = json.loads(archive.read(info).decode("utf-8"))
    except (zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("audit archive is invalid or corrupted") from exc

    # Archives emitted before the schema marker was introduced are format v1.
    if not isinstance(payload, dict) or payload.get("schema_version", 1) != 1:
        raise ValueError("unsupported audit archive schema version")

    forecasts = _require_list(payload, "forecasts")
    market_forecasts = _require_list(payload, "market_forecasts")
    paper_bets = _require_list(payload, "paper_bets")
    bank_history = _require_list(payload, "bank_history")
    decision_snapshots = _require_list(payload, "decision_snapshots")

    target = Path(db_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and _database_has_user_rows(target):
        raise FileExistsError(
            "local SQLite database contains user rows; audit restore refused"
        )

    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=target.name + ".audit-restore-",
        suffix=".sqlite",
        dir=str(target.parent),
    )
    os.close(temporary_fd)
    temporary = Path(temporary_name)
    try:
        _ensure_sqlite_schema(temporary)
        ForecastLedger(temporary)
        MarketForecastLedger(temporary)

        forecast_rows = []
        for row in forecasts:
            clean = dict(row)
            for field in ("probabilities", "entry_odds", "closing_odds"):
                json_column = field + "_json"
                if field in clean:
                    clean[json_column] = clean.pop(field)
                elif json_column not in clean:
                    if field == "probabilities":
                        raise ValueError("forecast row is missing its probability distribution")
                    clean[json_column] = None
            forecast_rows.append(clean)

        with sqlite3.connect(str(temporary), timeout=15) as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            with conn:
                forecast_count = _insert_rows(
                    conn, "forecasts", FORECAST_FIELDS, forecast_rows,
                    json_fields=("probabilities_json", "entry_odds_json", "closing_odds_json"),
                )
                market_count = _insert_rows(
                    conn, "market_forecasts", MARKET_FIELDS, market_forecasts,
                )
                bet_count = _insert_rows(
                    conn, "bets", BET_FIELDS, paper_bets,
                )
                bank_count = _insert_rows(
                    conn, "bank_history",
                    ("id", "ts", "bank", "event", "bet_id", "pnl"),
                    bank_history,
                )
                decision_count = _insert_rows(
                    conn, "decision_snapshots", DECISION_FIELDS, decision_snapshots,
                )
            check = conn.execute("PRAGMA integrity_check").fetchone()
            if not check or str(check[0]).casefold() != "ok":
                raise ValueError("restored audit SQLite failed integrity_check")

        if target.exists() and _database_has_user_rows(target):
            raise FileExistsError(
                "local SQLite received data while restoring; existing data was preserved"
            )
        for suffix in ("-wal", "-shm"):
            Path(str(target) + suffix).unlink(missing_ok=True)
        os.replace(temporary, target)
        return {
            "restored": True,
            "source_generated_at": payload.get("generated_at_utc"),
            "forecasts": forecast_count,
            "market_forecasts": market_count,
            "paper_bets": bet_count,
            "bank_history": bank_count,
            "decision_snapshots": decision_count,
        }
    finally:
        temporary.unlink(missing_ok=True)
