"""Historical context CSV import with pre-kickoff leakage checks."""
from __future__ import annotations

import csv
import io
import math
from datetime import datetime, timezone
from typing import Any


REQUIRED_COLUMNS = {
    "fixture_id", "home_team", "away_team", "kickoff_at", "observed_at",
}
NUMERIC_COLUMNS = {
    "lineup_strength_home", "lineup_strength_away",
    "key_players_missing_home", "key_players_missing_away",
    "referee_yellow_mean", "referee_yellow_variance",
    "referee_red_mean", "referee_red_variance",
    "corners_home_trailing_mean", "corners_away_trailing_mean",
    "corners_home_level_mean", "corners_away_level_mean",
    "opponent_corner_style_home", "opponent_corner_style_away",
    "base_p_home", "base_p_draw", "base_p_away",
}


def _timestamp(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value or "").strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamps must include a timezone offset or Z")
    return parsed.astimezone(timezone.utc)


def _optional_number(value: Any, column: str):
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        raise ValueError(f"{column} must be numeric") from None
    if not math.isfinite(result):
        raise ValueError(f"{column} must be finite")
    if column.endswith("_variance") and result < 0:
        raise ValueError(f"{column} cannot be negative")
    if "key_players_missing" in column and (result < 0 or not result.is_integer()):
        raise ValueError(f"{column} must be a non-negative integer")
    return result


def parse_context_csv(source: str | bytes) -> dict:
    """Parse user-supplied context CSV and reject post-kickoff observations.

    Each row is an as-of snapshot. Repeated fixture IDs are allowed when their
    observed_at timestamps differ; the walk-forward selector chooses the latest
    snapshot strictly before kickoff.
    """
    if isinstance(source, bytes):
        source = source.decode("utf-8-sig", errors="strict")
    text = str(source or "").lstrip("\ufeff")
    reader = csv.DictReader(io.StringIO(text))
    columns = {str(name or "").strip() for name in (reader.fieldnames or [])}
    missing = REQUIRED_COLUMNS - columns
    if missing:
        raise ValueError("missing required columns: " + ", ".join(sorted(missing)))

    rows, rejected = [], []
    for line_number, raw in enumerate(reader, start=2):
        row = {str(k or "").strip(): (str(v).strip() if v is not None else "") for k, v in raw.items()}
        if not any(row.values()):
            continue
        fixture_id = row.get("fixture_id", "")
        if not fixture_id or not row.get("home_team") or not row.get("away_team"):
            rejected.append({"line": line_number, "reason": "missing fixture/team identity"})
            continue
        try:
            kickoff = _timestamp(row.get("kickoff_at"))
            observed = _timestamp(row.get("observed_at"))
            if observed >= kickoff:
                rejected.append({"line": line_number, "reason": "observed_at is not before kickoff"})
                continue
            reliability = _optional_number(row.get("source_reliability"), "source_reliability")
            if reliability is not None and not 0 <= reliability <= 1:
                raise ValueError("source_reliability must be in [0, 1]")
            parsed = dict(row)
            parsed["kickoff_at"] = kickoff.isoformat()
            parsed["observed_at"] = observed.isoformat()
            parsed["source_reliability"] = reliability
            for column in NUMERIC_COLUMNS:
                parsed[column] = _optional_number(row.get(column), column)
            parsed["is_confirmed"] = str(row.get("is_confirmed", "")).casefold() in {
                "1", "true", "yes", "y", "да"
            }
            parsed["data_source"] = row.get("data_source") or "user_csv"
            parsed["referee_id"] = row.get("referee_id") or None
            parsed["result"] = (row.get("result") or "").upper() or None
            rows.append(parsed)
        except ValueError as exc:
            rejected.append({"line": line_number, "reason": str(exc)})
    return {
        "rows": rows,
        "rejected": rejected,
        "accepted": len(rows),
        "schema_version": 1,
    }


def latest_pre_kickoff_rows(rows: list[dict]) -> list[dict]:
    """Return one latest pre-match context snapshot per fixture."""
    selected = {}
    for row in rows or []:
        try:
            fixture_id = str(row.get("fixture_id") or "")
            kickoff = _timestamp(row.get("kickoff_at"))
            observed = _timestamp(row.get("observed_at"))
            if not fixture_id or observed >= kickoff:
                continue
            previous = selected.get(fixture_id)
            if previous is None or observed > _timestamp(previous["observed_at"]):
                selected[fixture_id] = row
        except (TypeError, ValueError):
            continue
    return sorted(selected.values(), key=lambda row: (
        str(row.get("kickoff_at") or ""), str(row.get("fixture_id") or "")
    ))
