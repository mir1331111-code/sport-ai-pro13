"""Bounded, deterministic historical validation sample and fingerprint."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime


def _parse_stamp(value):
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y", "%d-%m-%y"):
        try:
            return datetime.strptime(raw[:10], fmt).date()
        except ValueError:
            continue
    return None


def select_validation_rows(history, divisions, *, max_rows_per_league=500):
    """Select the newest complete fixtures from a bounded set of major leagues."""
    limit = max(1, int(max_rows_per_league))
    selected = []
    for division in divisions:
        rows = []
        for season_rows in (history or {}).get(division, []) or []:
            for original in season_rows or []:
                if not isinstance(original, dict):
                    continue
                stamp = _parse_stamp(original.get("Date") or original.get("date"))
                home = str(original.get("HomeTeam") or original.get("home_team") or "").strip()
                away = str(original.get("AwayTeam") or original.get("away_team") or "").strip()
                try:
                    hg = float(original.get("FTHG"))
                    ag = float(original.get("FTAG"))
                except (TypeError, ValueError):
                    continue
                if (
                    stamp is None or not home or not away
                    or not math.isfinite(hg) or not math.isfinite(ag)
                    or hg < 0 or ag < 0 or not hg.is_integer() or not ag.is_integer()
                ):
                    continue
                row = dict(original)
                row["Div"] = division
                rows.append((stamp, home.casefold(), away.casefold(), row))
        rows.sort(key=lambda item: (item[0], item[1], item[2]))
        selected.extend(item[3] for item in rows[-limit:])
    selected.sort(
        key=lambda row: (
            _parse_stamp(row.get("Date") or row.get("date")) or date.min,
            str(row.get("Div") or ""),
            str(row.get("HomeTeam") or ""),
            str(row.get("AwayTeam") or ""),
        )
    )
    return selected


def validation_cache_key(rows, *, matrix_n=12, model_version="unspecified"):
    """Fingerprint exact validation inputs; new scores, xG or market fields invalidate it."""
    payload = json.dumps(
        list(rows or []),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    version = str(model_version or "unspecified").strip().replace("/", "_").replace(" ", "_")
    return f"neuro_market_validation_v5_{version}_bootstrap1000_{int(matrix_n)}_{digest}"


