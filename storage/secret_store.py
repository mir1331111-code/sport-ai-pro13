"""Local persistence for API credentials entered in the UI.

Credentials are deliberately kept outside neuro_local.json, its Gist sync, and
downloadable backups. The local file is written atomically with owner-only
permissions (0600). For ephemeral hosting, configure credentials in Streamlit
secrets/environment because local disk may not survive an app restart.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Mapping

log = logging.getLogger("secret_store")

_ALLOWED_KEYS = {
    "fdorg_token",
    "odds_api_key",
    "api_football_key",
    "therundown_api_key",
    "sstats_api_key",
    "highlightly_api_key",
    "llm_api_key",
}


def _default_path() -> Path:
    configured = os.environ.get("NEURO_SECRETS_FILE", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[1] / ".neuro_secrets.json"


def load_local_secrets(path: str | os.PathLike | None = None) -> dict[str, str]:
    """Read saved UI credentials; malformed or unavailable files fail closed."""
    target = Path(path) if path is not None else _default_path()
    try:
        with target.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            return {}
        return {
            key: value.strip()
            for key, value in raw.items()
            if key in _ALLOWED_KEYS and isinstance(value, str) and value.strip()
        }
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Local API-key store unavailable (%s)", type(exc).__name__)
        return {}


def save_local_secrets(values: Mapping[str, str],
                       path: str | os.PathLike | None = None) -> bool:
    """Atomically save only known, non-empty credentials with owner-only access."""
    target = Path(path) if path is not None else _default_path()
    safe_values = {
        key: value.strip()
        for key, value in values.items()
        if key in _ALLOWED_KEYS and isinstance(value, str) and value.strip()
    }
    temp_path = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not safe_values:
            try:
                target.unlink()
            except FileNotFoundError:
                pass
            return True
        fd, temp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp",
                                         dir=str(target.parent))
        temp_path = Path(temp_name)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(safe_values, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, target)
            try:
                os.chmod(target, 0o600)
            except OSError:
                pass
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        return True
    except Exception as exc:
        log.warning("Could not persist local API keys (%s)", type(exc).__name__)
        return False
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
