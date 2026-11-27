"""Durable storage for user-imported historical CSV datasets.

The raw parsed rows are stored in the same SQLite database as the forecast
ledgers so encrypted database backups also protect the training/context inputs.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any

from config import DB_FILE

log = logging.getLogger("user_datasets")

ALLOWED_DATASETS = {
    "custom_historical_match_rows",
    "historical_context_rows",
}
MAX_DATASET_ROWS = 250_000
MAX_DATASET_BYTES = 75 * 1024 * 1024


def _connect(db_path: str = DB_FILE) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path, timeout=20)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS user_datasets (
            dataset_key TEXT PRIMARY KEY,
            payload_json TEXT NOT NULL,
            row_count INTEGER NOT NULL CHECK(row_count >= 0),
            sha256 TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    return connection


def save_user_dataset(
    dataset_key: str,
    rows: object,
    *,
    db_path: str = DB_FILE,
) -> dict[str, Any]:
    """Validate and atomically replace one approved imported dataset."""
    name = str(dataset_key or "").strip()
    if name not in ALLOWED_DATASETS:
        raise ValueError(f"Unsupported user dataset: {name!r}")
    if not isinstance(rows, list):
        raise ValueError("User dataset must be a list of row objects")
    if len(rows) > MAX_DATASET_ROWS:
        raise ValueError("User dataset exceeds the maximum row count")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Every user dataset row must be an object")

    try:
        payload = json.dumps(
            rows, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("User dataset contains unsupported or non-finite values") from exc
    encoded = payload.encode("utf-8")
    if len(encoded) > MAX_DATASET_BYTES:
        raise ValueError("User dataset exceeds the 75 MB storage limit")
    digest = hashlib.sha256(encoded).hexdigest()
    updated_at = datetime.now(timezone.utc).isoformat()

    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO user_datasets(dataset_key, payload_json, row_count, sha256, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(dataset_key) DO UPDATE SET
                payload_json=excluded.payload_json,
                row_count=excluded.row_count,
                sha256=excluded.sha256,
                updated_at=excluded.updated_at
            """,
            (name, payload, len(rows), digest, updated_at),
        )
    return {
        "dataset_key": name,
        "row_count": len(rows),
        "bytes": len(encoded),
        "sha256": digest,
        "updated_at": updated_at,
    }


def load_user_dataset(
    dataset_key: str,
    *,
    db_path: str = DB_FILE,
) -> list[dict]:
    """Load a saved dataset and verify its content hash."""
    name = str(dataset_key or "").strip()
    if name not in ALLOWED_DATASETS:
        raise ValueError(f"Unsupported user dataset: {name!r}")
    with _connect(db_path) as connection:
        row = connection.execute(
            "SELECT payload_json, row_count, sha256 FROM user_datasets WHERE dataset_key=?",
            (name,),
        ).fetchone()
    if row is None:
        return []
    encoded = str(row["payload_json"]).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != str(row["sha256"]):
        raise ValueError(f"Stored user dataset {name!r} failed its SHA-256 integrity check")
    data = json.loads(row["payload_json"])
    if not isinstance(data, list) or len(data) != int(row["row_count"]):
        raise ValueError(f"Stored user dataset {name!r} has inconsistent row metadata")
    if any(not isinstance(item, dict) for item in data):
        raise ValueError(f"Stored user dataset {name!r} contains invalid rows")
    return data


def _dataset_row_identity(dataset_key: str, row: dict) -> tuple[str, ...] | None:
    """Stable append-only identity for imported historical/context observations."""
    if dataset_key == "custom_historical_match_rows":
        stamp = str(row.get("Date") or row.get("date") or row.get("date_iso") or "").strip()[:10]
        division = str(row.get("Div") or row.get("division") or "").strip().casefold()
        home = str(row.get("HomeTeam") or row.get("home_team") or "").strip()
        away = str(row.get("AwayTeam") or row.get("away_team") or "").strip()
        if not (stamp and division and home and away):
            return None
        try:
            from data.team_names import normalize_team
            home_key = normalize_team(home) or home.casefold()
            away_key = normalize_team(away) or away.casefold()
        except Exception:
            home_key, away_key = home.casefold(), away.casefold()
        return ("match", division, stamp, str(home_key).casefold(), str(away_key).casefold())
    if dataset_key == "historical_context_rows":
        fixture = str(row.get("fixture_id") or "").strip()
        observed = str(row.get("observed_at") or "").strip()
        if not fixture or not observed:
            return None
        try:
            parsed = datetime.fromisoformat(observed.replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                observed = parsed.astimezone(timezone.utc).isoformat()
        except ValueError:
            pass
        return ("context", fixture.casefold(), observed)
    return None


def _meaningful(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and not value.strip())


def _same_dataset_value(left: Any, right: Any) -> bool:
    if not _meaningful(left) or not _meaningful(right):
        return True
    try:
        left_number, right_number = float(left), float(right)
        if left_number == left_number and right_number == right_number:
            return left_number == right_number
    except (TypeError, ValueError, OverflowError):
        pass
    return str(left).strip().casefold() == str(right).strip().casefold()


def merge_user_dataset(
    dataset_key: str,
    rows: object,
    *,
    db_path: str = DB_FILE,
) -> dict[str, Any]:
    """Append imported rows without deleting prior history or overwriting conflicts.

    Duplicate match/context identities only fill previously blank fields. Conflicting
    final scores or already-populated observations remain unchanged and are counted.
    """
    name = str(dataset_key or "").strip()
    if name not in ALLOWED_DATASETS:
        raise ValueError(f"Unsupported user dataset: {name!r}")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Merged dataset rows must be a list of row objects")
    if len(rows) > MAX_DATASET_ROWS:
        raise ValueError("Incoming dataset exceeds the maximum row count")

    incoming = [dict(row) for row in rows]
    with _connect(db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        stored = connection.execute(
            "SELECT payload_json,row_count,sha256 FROM user_datasets WHERE dataset_key=?",
            (name,),
        ).fetchone()
        existing_rows: list[dict] = []
        if stored is not None:
            encoded_old = str(stored["payload_json"]).encode("utf-8")
            if hashlib.sha256(encoded_old).hexdigest() != str(stored["sha256"]):
                connection.rollback()
                raise ValueError(f"Stored user dataset {name!r} failed its SHA-256 integrity check")
            old_payload = json.loads(stored["payload_json"])
            if not isinstance(old_payload, list) or len(old_payload) != int(stored["row_count"]):
                connection.rollback()
                raise ValueError(f"Stored user dataset {name!r} has inconsistent row metadata")
            if any(not isinstance(row, dict) for row in old_payload):
                connection.rollback()
                raise ValueError(f"Stored user dataset {name!r} contains invalid rows")
            existing_rows = [dict(row) for row in old_payload]

        merged = list(existing_rows)
        by_identity: dict[tuple[str, ...], dict] = {}
        for row in merged:
            identity = _dataset_row_identity(name, row)
            if identity is not None:
                by_identity.setdefault(identity, row)

        added = enriched = duplicates = conflicts = invalid_identity = 0
        score_fields = {"FTHG", "FTAG", "home_goals", "away_goals"}
        for row in incoming:
            identity = _dataset_row_identity(name, row)
            if identity is None:
                invalid_identity += 1
                continue
            previous = by_identity.get(identity)
            if previous is None:
                by_identity[identity] = row
                merged.append(row)
                added += 1
                continue
            duplicates += 1
            score_conflict = any(
                _meaningful(previous.get(field))
                and _meaningful(row.get(field))
                and not _same_dataset_value(previous.get(field), row.get(field))
                for field in score_fields
            ) if name == "custom_historical_match_rows" else False
            if score_conflict:
                conflicts += 1
                continue
            changed = False
            row_conflict = False
            for field, value in row.items():
                if not _meaningful(previous.get(field)) and _meaningful(value):
                    previous[field] = value
                    changed = True
                elif (
                    _meaningful(previous.get(field)) and _meaningful(value)
                    and not _same_dataset_value(previous.get(field), value)
                ):
                    row_conflict = True
            if changed:
                enriched += 1
            if row_conflict:
                conflicts += 1

        if name == "custom_historical_match_rows":
            merged.sort(key=lambda row: (
                str(row.get("Date") or row.get("date") or ""),
                str(row.get("Div") or row.get("division") or ""),
                str(row.get("HomeTeam") or row.get("home_team") or "").casefold(),
                str(row.get("AwayTeam") or row.get("away_team") or "").casefold(),
            ))
        else:
            merged.sort(key=lambda row: (
                str(row.get("kickoff_at") or ""),
                str(row.get("observed_at") or ""),
                str(row.get("fixture_id") or ""),
            ))

        if len(merged) > MAX_DATASET_ROWS:
            connection.rollback()
            raise ValueError("Merged dataset exceeds the maximum row count")
        payload = json.dumps(
            merged, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        )
        encoded = payload.encode("utf-8")
        if len(encoded) > MAX_DATASET_BYTES:
            connection.rollback()
            raise ValueError("Merged dataset exceeds the 75 MB storage limit")
        digest = hashlib.sha256(encoded).hexdigest()
        updated_at = datetime.now(timezone.utc).isoformat()
        connection.execute(
            """
            INSERT INTO user_datasets(dataset_key,payload_json,row_count,sha256,updated_at)
            VALUES (?,?,?,?,?)
            ON CONFLICT(dataset_key) DO UPDATE SET
                payload_json=excluded.payload_json,
                row_count=excluded.row_count,
                sha256=excluded.sha256,
                updated_at=excluded.updated_at
            """,
            (name, payload, len(merged), digest, updated_at),
        )
        connection.commit()

    return {
        "dataset_key": name,
        "row_count": len(merged),
        "added_rows": added,
        "enriched_rows": enriched,
        "duplicate_rows": duplicates,
        "conflicts": conflicts,
        "invalid_identity_rows": invalid_identity,
        "bytes": len(encoded),
        "sha256": digest,
        "updated_at": updated_at,
    }


def export_user_datasets(*, db_path: str = DB_FILE) -> list[dict]:
    """Return verified datasets for the downloadable forecast audit archive."""
    with _connect(db_path) as connection:
        keys = [
            str(row[0]) for row in connection.execute(
                "SELECT dataset_key FROM user_datasets ORDER BY dataset_key"
            ).fetchall()
        ]
    exported = []
    for key in keys:
        rows = load_user_dataset(key, db_path=db_path)
        exported.append({
            "dataset_key": key,
            "rows": rows,
        })
    return exported
