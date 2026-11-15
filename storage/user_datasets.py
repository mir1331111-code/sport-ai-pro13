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
