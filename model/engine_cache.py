"""Content-addressed fingerprinting for fitted football-model caches."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def engine_cache_fingerprint(
    *,
    app_version: str,
    data_version: int,
    matrix_n: int,
    xg_enabled: bool,
    seasonal_history: Mapping[str, Any],
    history_years: list[int] | tuple[int, ...],
    divisions: list[str] | tuple[str, ...],
    footballapi_xg_rows: list[dict] | tuple[dict, ...] | None = None,
    custom_historical_rows: list[dict] | tuple[dict, ...] | None = None,
) -> str:
    """Hash all raw seasonal, provider-xG and custom-history inputs.

    Match IDs alone are insufficient: scores, shot counts and xG values may
    change while fixture identity stays constant. The complete training payload
    is therefore included in the digest.
    """
    seasons = []
    for division in sorted(str(key) for key in (seasonal_history or {})):
        rows_by_season = (seasonal_history or {}).get(division) or []
        seasons.append({
            "division": division,
            "seasons": [list(rows or []) for rows in rows_by_season],
        })

    payload = {
        "schema": "neuro-engine-cache-v2",
        "app_version": str(app_version),
        "data_version": int(data_version),
        "matrix_n": int(matrix_n),
        "xg_enabled": bool(xg_enabled),
        "history_years": [int(year) for year in history_years],
        "divisions": [str(division) for division in divisions],
        "seasonal_history": seasons,
        "footballapi_xg_rows": list(footballapi_xg_rows or []),
        "custom_historical_rows": list(custom_historical_rows or []),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:24]
    return f"neuro-engine-v2-{digest}"
