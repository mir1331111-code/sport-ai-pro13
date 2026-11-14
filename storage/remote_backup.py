"""Encrypted, chunked SQLite backup for Streamlit Community Cloud.

The source SQLite database remains local for fast transactional writes. A snapshot
is encrypted before it is written to an unlisted GitHub Gist. Restore is automatic
only when the local database is missing or contains no user rows.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from cryptography.fernet import Fernet, InvalidToken

from config import DB_FILE

log = logging.getLogger("remote_backup")

GITHUB_API = "https://api.github.com"
MANIFEST_NAME = "neuro_db_manifest.json"
PART_PREFIX = "neuro_db_part_"
PART_BYTES = 400_000  # Below the 1 MB Gist API content limit after text encoding.
MAX_PARTS = 64
MAX_SQLITE_BYTES = 200 * 1024 * 1024
MAX_ENCRYPTED_BYTES = PART_BYTES * MAX_PARTS
REQUEST_TIMEOUT = (5, 20)


def _secret(name: str) -> str:
    value = str(os.environ.get(name, "") or "").strip()
    if value:
        return value
    try:
        import streamlit as st
        return str(st.secrets.get(name, "") or "").strip()
    except Exception:
        return ""


def generate_backup_key() -> str:
    """Generate a Fernet key for Streamlit Secrets; never persist it in the repo."""
    return Fernet.generate_key().decode("ascii")


def _credentials() -> tuple[str, str, str]:
    return _secret("GIST_ID"), _secret("GITHUB_TOKEN"), _secret("NEURO_BACKUP_KEY")


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
    }


def _get_gist(gist_id: str, token: str) -> dict[str, Any]:
    response = requests.get(
        f"{GITHUB_API}/gists/{gist_id}",
        headers=_headers(token),
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code != 200:
        raise RuntimeError(f"GitHub Gist read failed: HTTP {response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("GitHub Gist returned an invalid response")
    # The database is personal and may contain bank history and bet decisions.
    # A public Gist is never accepted for backup or restore.
    if payload.get("public") is not False:
        raise PermissionError(
            "Gist is public or its visibility could not be verified; backup refused"
        )
    return payload


def _fernet(key: str) -> Fernet:
    if not key:
        raise ValueError("NEURO_BACKUP_KEY is not configured")
    try:
        return Fernet(key.encode("ascii"))
    except (ValueError, UnicodeEncodeError) as exc:
        raise ValueError("NEURO_BACKUP_KEY is not a valid Fernet key") from exc


def _assert_backup_does_not_regress(
    previous: dict[str, int], current: dict[str, int],
) -> None:
    """Never replace an older remote backup with a database missing prior rows."""
    missing = {
        table: (int(previous_count), int(current.get(table, 0)))
        for table, previous_count in previous.items()
        if int(current.get(table, 0)) < int(previous_count)
    }
    if missing:
        detail = ", ".join(
            f"{table} {old}->{new}" for table, (old, new) in sorted(missing.items())
        )
        raise ValueError(
            "Local SQLite contains fewer rows than the verified remote backup; "
            "remote data was preserved (" + detail + ")"
        )


def _snapshot_sqlite(db_path: str | os.PathLike) -> bytes:
    source = Path(db_path)
    if not source.is_file():
        raise FileNotFoundError(f"SQLite database does not exist: {source.name}")
    if source.stat().st_size > MAX_SQLITE_BYTES:
        raise ValueError("SQLite database exceeds the configured backup safety limit")

    handle = tempfile.NamedTemporaryFile(prefix="neuro-db-", suffix=".sqlite", delete=False)
    snapshot_path = Path(handle.name)
    handle.close()
    try:
        with sqlite3.connect(str(source), timeout=15) as src:
            with sqlite3.connect(str(snapshot_path), timeout=15) as dst:
                src.backup(dst)
        raw = snapshot_path.read_bytes()
        if len(raw) > MAX_SQLITE_BYTES:
            raise ValueError("SQLite snapshot exceeds the configured backup safety limit")
        _validate_sqlite_bytes(raw)
        return raw
    finally:
        try:
            snapshot_path.unlink(missing_ok=True)
        except OSError:
            pass


def _validate_sqlite_bytes(raw: bytes) -> dict[str, int]:
    handle = tempfile.NamedTemporaryFile(prefix="neuro-verify-", suffix=".sqlite", delete=False)
    path = Path(handle.name)
    try:
        handle.write(raw)
        handle.flush()
        handle.close()
        with sqlite3.connect(str(path), timeout=15) as conn:
            result = conn.execute("PRAGMA integrity_check").fetchone()
            if not result or str(result[0]).casefold() != "ok":
                raise ValueError("SQLite integrity_check failed")
            tables = [
                str(row[0]) for row in conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
            ]
            counts = {}
            for table in tables:
                safe_table = table.replace('"', '""')
                counts[table] = int(
                    conn.execute(f'SELECT COUNT(*) FROM "{safe_table}"').fetchone()[0]
                )
            return counts
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _database_has_user_rows(db_path: str | os.PathLike) -> bool:
    path = Path(db_path)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        uri = path.resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=5) as conn:
            tables = [
                str(row[0]) for row in conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
            ]
            for table in tables:
                safe_table = table.replace('"', '""')
                if conn.execute(
                    f'SELECT 1 FROM "{safe_table}" LIMIT 1'
                ).fetchone():
                    return True
        return False
    except sqlite3.DatabaseError:
        # Never overwrite a present but unreadable database automatically.
        return True


def create_sqlite_backup_bytes(db_path: str | os.PathLike = DB_FILE) -> bytes:
    """Return a portable gzip-compressed SQLite snapshot for manual download."""
    return gzip.compress(_snapshot_sqlite(db_path), compresslevel=6)


def restore_sqlite_backup_bytes(
    payload: bytes,
    db_path: str | os.PathLike = DB_FILE,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Restore a manual .sqlite.gz backup after integrity validation."""
    raw = gzip.decompress(payload)
    if len(raw) > MAX_SQLITE_BYTES:
        raise ValueError("Uploaded SQLite backup exceeds the safety limit")
    counts = _validate_sqlite_bytes(raw)
    target = Path(db_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and _database_has_user_rows(target) and not overwrite:
        raise FileExistsError("Local SQLite database contains data; explicit overwrite is required")
    temporary = target.with_name(target.name + ".restore.tmp")
    try:
        temporary.write_bytes(raw)
        for suffix in ("-wal", "-shm"):
            Path(str(target) + suffix).unlink(missing_ok=True)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"restored": True, "bytes": len(raw), "tables": counts}


def _read_text_file(files: dict, name: str, token: str) -> str:
    item = files.get(name)
    if not isinstance(item, dict):
        raise FileNotFoundError(f"Backup file {name} is missing from Gist")
    content = item.get("content")
    if item.get("truncated"):
        raw_url = str(item.get("raw_url") or "")
        if not raw_url:
            raise ValueError(f"Backup file {name} was truncated without raw_url")
        response = requests.get(raw_url, headers=_headers(token), timeout=REQUEST_TIMEOUT)
        if response.status_code != 200:
            raise RuntimeError(f"GitHub Gist file download failed: HTTP {response.status_code}")
        content = response.text
    if not isinstance(content, str):
        raise ValueError(f"Backup file {name} has no text content")
    return content


def _manifest_from_gist(gist: dict, token: str) -> dict[str, Any] | None:
    files = gist.get("files") or {}
    if MANIFEST_NAME not in files:
        return None
    manifest = json.loads(_read_text_file(files, MANIFEST_NAME, token))
    if not isinstance(manifest, dict) or manifest.get("format") != "neuro-sqlite-fernet-chunks-v1":
        raise ValueError("Unsupported remote SQLite backup format")
    parts = int(manifest.get("parts") or 0)
    if not 1 <= parts <= MAX_PARTS:
        raise ValueError("Remote SQLite backup has an invalid part count")
    return manifest


def _decode_remote_snapshot(
    gist: dict, token: str, key: str,
) -> tuple[bytes, dict[str, Any], dict[str, int]]:
    """Download, decrypt and fully validate a remote SQLite snapshot without writing it."""
    files = gist.get("files") or {}
    manifest = _manifest_from_gist(gist, token)
    if manifest is None:
        raise FileNotFoundError("Remote SQLite backup does not exist")
    fernet = _fernet(key)
    parts = []
    for index in range(int(manifest["parts"])):
        filename = f"{PART_PREFIX}{index:03d}.txt"
        parts.append(_read_text_file(files, filename, token))
    encrypted = "".join(parts).encode("ascii")
    if hashlib.sha256(encrypted).hexdigest() != manifest.get("encrypted_sha256"):
        raise ValueError("Remote SQLite backup encrypted checksum mismatch")
    compressed = fernet.decrypt(encrypted)
    raw = gzip.decompress(compressed)
    raw_bytes = int(manifest.get("raw_bytes") or 0)
    if raw_bytes != len(raw) or len(raw) > MAX_SQLITE_BYTES:
        raise ValueError("Remote SQLite backup size does not match its manifest")
    raw_sha = hashlib.sha256(raw).hexdigest()
    if raw_sha != manifest.get("raw_sha256"):
        raise ValueError("Remote SQLite backup integrity hash mismatch")
    counts = _validate_sqlite_bytes(raw)
    expected_tables = manifest.get("tables") or {}
    if expected_tables and counts != {
        str(name): int(count) for name, count in expected_tables.items()
    }:
        raise ValueError("Remote SQLite backup table counts do not match its manifest")
    return raw, manifest, counts


def _restore_from_gist(gist: dict, token: str, key: str, db_path: str | os.PathLike) -> dict[str, Any]:
    try:
        raw, manifest, counts = _decode_remote_snapshot(gist, token, key)
    except FileNotFoundError:
        return {"restored": False, "state": "no_remote_backup", "message": "Remote SQLite backup does not exist"}

    target = Path(db_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".restore.tmp")
    try:
        temporary.write_bytes(raw)
        for suffix in ("-wal", "-shm"):
            Path(str(target) + suffix).unlink(missing_ok=True)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "restored": True,
        "state": "restored",
        "message": "Encrypted SQLite database restored from secret Gist",
        "bytes": len(raw),
        "tables": counts,
        "backup_at": manifest.get("created_at"),
    }


def restore_database_if_empty(db_path: str | os.PathLike = DB_FILE) -> dict[str, Any]:
    """Restore only into a missing/empty DB; never replace existing user records."""
    gist_id, token, key = _credentials()
    if not gist_id or not token or not key:
        missing = [
            name for name, value in (
                ("GIST_ID", gist_id), ("GITHUB_TOKEN", token), ("NEURO_BACKUP_KEY", key)
            ) if not value
        ]
        return {
            "restored": False, "state": "not_configured",
            "message": "Remote SQLite backup not configured; missing " + ", ".join(missing),
        }
    if _database_has_user_rows(db_path):
        return {
            "restored": False, "state": "local_data_present",
            "message": "Local SQLite database contains rows; remote restore skipped",
        }
    try:
        gist = _get_gist(gist_id, token)
        return _restore_from_gist(gist, token, key, db_path)
    except PermissionError as exc:
        error_type = type(exc).__name__
        message = (
            "SQLite restore refused: Gist is public or its visibility is unknown. "
            "Create a non-public Secret Gist and update GIST_ID."
        )
    except InvalidToken as exc:
        error_type = type(exc).__name__
        message = (
            "SQLite restore refused: NEURO_BACKUP_KEY does not decrypt the existing backup. "
            "Use the exact key that was used when it was created."
        )
    except Exception as exc:
        error_type = type(exc).__name__
        message = f"Remote SQLite restore failed: {str(exc)[:180] or error_type}"
    log.warning("SQLite remote restore failed: %s", error_type)
    return {"restored": False, "state": "restore_error", "message": message}


def backup_database_to_gist(
    db_path: str | os.PathLike = DB_FILE,
) -> dict[str, Any]:
    """Upload a compressed, encrypted, chunked SQLite snapshot to an unlisted Gist."""
    gist_id, token, key = _credentials()
    if not gist_id or not token or not key:
        missing = [
            name for name, value in (
                ("GIST_ID", gist_id), ("GITHUB_TOKEN", token), ("NEURO_BACKUP_KEY", key)
            ) if not value
        ]
        return {
            "ok": False, "state": "not_configured",
            "message": "Remote SQLite backup not configured; missing " + ", ".join(missing),
        }
    try:
        raw = _snapshot_sqlite(db_path)
        counts = _validate_sqlite_bytes(raw)
        raw_sha = hashlib.sha256(raw).hexdigest()
        gist = _get_gist(gist_id, token)
        old_manifest = _manifest_from_gist(gist, token)
        if old_manifest:
            previous_raw, verified_old_manifest, previous_counts = _decode_remote_snapshot(
                gist, token, key,
            )
            if hashlib.sha256(previous_raw).hexdigest() == raw_sha:
                return {
                    "ok": True, "state": "current",
                    "message": "Encrypted remote SQLite backup is current",
                    "backup_at": verified_old_manifest.get("created_at"),
                    "bytes": len(raw),
                    "tables": counts,
                }
            _assert_backup_does_not_regress(previous_counts, counts)

        encrypted = _fernet(key).encrypt(gzip.compress(raw, compresslevel=6))
        if len(encrypted) > MAX_ENCRYPTED_BYTES:
            raise ValueError(
                f"Encrypted SQLite backup exceeds {MAX_ENCRYPTED_BYTES // (1024 * 1024)} MB Gist limit"
            )
        chunks = [
            encrypted[offset:offset + PART_BYTES]
            for offset in range(0, len(encrypted), PART_BYTES)
        ]
        if not chunks or len(chunks) > MAX_PARTS:
            raise ValueError("SQLite backup has an invalid number of chunks")
        created_at = datetime.now(timezone.utc).isoformat()
        manifest = {
            "format": "neuro-sqlite-fernet-chunks-v1",
            "created_at": created_at,
            "raw_bytes": len(raw),
            "raw_sha256": raw_sha,
            "encrypted_bytes": len(encrypted),
            "encrypted_sha256": hashlib.sha256(encrypted).hexdigest(),
            "parts": len(chunks),
            "tables": counts,
        }
        updates: dict[str, Any] = {
            MANIFEST_NAME: {"content": json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))}
        }
        for index, chunk in enumerate(chunks):
            updates[f"{PART_PREFIX}{index:03d}.txt"] = {"content": chunk.decode("ascii")}
        old_parts = int((old_manifest or {}).get("parts") or 0)
        for index in range(len(chunks), min(old_parts, MAX_PARTS)):
            updates[f"{PART_PREFIX}{index:03d}.txt"] = None

        response = requests.patch(
            f"{GITHUB_API}/gists/{gist_id}",
            headers=_headers(token),
            json={"files": updates},
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code != 200:
            raise RuntimeError(f"GitHub Gist backup failed: HTTP {response.status_code}")
        verified = response.json()
        if verified.get("public") is not False:
            raise PermissionError("GitHub Gist visibility changed; backup refused")
        verified_raw, verified_manifest, verified_counts = _decode_remote_snapshot(
            verified, token, key,
        )
        if (
            verified_manifest.get("raw_sha256") != raw_sha
            or hashlib.sha256(verified_raw).hexdigest() != raw_sha
            or verified_counts != counts
        ):
            raise RuntimeError("Remote SQLite backup decrypt/integrity verification failed")
        return {
            "ok": True, "state": "uploaded",
            "message": "Encrypted SQLite backup uploaded and verified",
            "backup_at": created_at, "bytes": len(raw), "parts": len(chunks),
        }
    except PermissionError as exc:
        error_type = type(exc).__name__
        message = (
            "SQLite backup refused: Gist is public or its visibility is unknown. "
            "Create a non-public Secret Gist and update GIST_ID."
        )
    except InvalidToken as exc:
        error_type = type(exc).__name__
        message = (
            "SQLite backup refused: NEURO_BACKUP_KEY cannot decrypt the existing remote copy. "
            "The old copy was preserved; use the original key."
        )
    except Exception as exc:
        error_type = type(exc).__name__
        message = f"Remote SQLite backup failed: {str(exc)[:180] or error_type}"
    log.warning("SQLite remote backup failed: %s", error_type)
    return {"ok": False, "state": "backup_error", "message": message}


def remote_backup_status() -> dict[str, Any]:
    """Return non-secret status for the application settings panel."""
    gist_id, token, key = _credentials()
    missing = [
        name for name, value in (
            ("GIST_ID", gist_id), ("GITHUB_TOKEN", token), ("NEURO_BACKUP_KEY", key)
        ) if not value
    ]
    if missing:
        return {
            "ok": False, "state": "not_configured",
            "message": "Missing Streamlit Secrets: " + ", ".join(missing),
        }
    try:
        gist = _get_gist(gist_id, token)
        try:
            raw, manifest, counts = _decode_remote_snapshot(gist, token, key)
        except FileNotFoundError:
            return {
                "ok": False, "state": "no_remote_backup",
                "message": "Secret Gist is connected; SQLite backup has not been uploaded yet",
            }
        return {
            "ok": True, "state": "current",
            "message": "Encrypted SQLite backup decrypted and passed SQLite integrity_check",
            "backup_at": manifest.get("created_at"),
            "bytes": len(raw),
            "parts": manifest.get("parts"),
            "tables": counts,
        }
    except PermissionError as exc:
        error_type = type(exc).__name__
        message = (
            "Gist is public or its visibility is unknown. Create a non-public Secret Gist "
            "and update GIST_ID before storing portfolio or SQLite data."
        )
    except InvalidToken as exc:
        error_type = type(exc).__name__
        message = (
            "NEURO_BACKUP_KEY does not decrypt the existing remote copy. "
            "Use the original key; do not overwrite the current Gist."
        )
    except Exception as exc:
        error_type = type(exc).__name__
        message = f"Remote backup status unavailable: {str(exc)[:180] or error_type}"
    log.warning("Remote backup status check failed: %s", error_type)
    return {"ok": False, "state": "status_error", "message": message}
