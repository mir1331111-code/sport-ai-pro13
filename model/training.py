"""Strict historical-row preparation and chronological engine training."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime, time
from typing import Any

TRAINING_STATE_SCHEMA = "neuro-engine-training-state-v1"

# Only values consumed by Engine.add, HistoricalMatchStats or the dated
# aggregate-shot xG fitter belong to the training-state digest. Market prices
# and other presentation/context fields must not force a score-model replay.
_TRAINING_DIGEST_FIELDS = (
    "Date", "date", "Div", "HomeTeam", "home_team", "AwayTeam", "away_team",
    "FTHG", "home_goals", "FTAG", "away_goals",
    "HxG", "AxG", "home_xg", "away_xg", "homeXG", "awayXG",
    "xG_home", "xG_away", "xg_home", "xg_away", "HomeXG", "AwayXG",
    "home_expected_goals", "away_expected_goals", "xG", "xGA", "xg", "xga",
    "xg_source", "xg_kind", "xg_model_version",
    "xg_training_sides", "HS", "HST", "AS", "AST",
    "HC", "AC", "HY", "AY", "HR", "AR",
    "HTHG", "HTAG", "HTHC", "HTAC",
    "Referee", "referee_id", "referee",
)


def _parse_match_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
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


def _score(value: Any) -> int | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        return None
    return int(number)


def prepare_completed_training_rows(
    history: dict, divisions: tuple[str, ...] | list[str],
) -> tuple[list[dict], dict[str, int]]:
    """Flatten only complete, dated historical results and sort globally by day.

    Missing scores are never defaulted to 0-0. All fixtures from one calendar
    date are kept together so the trainer can forecast that complete date before
    ingesting any of its results.
    """
    prepared = []
    skipped = {
        "invalid_row": 0,
        "missing_date": 0,
        "missing_teams": 0,
        "missing_or_invalid_score": 0,
    }
    for division in divisions:
        for season_rows in (history or {}).get(division, []) or []:
            for original in season_rows or []:
                if not isinstance(original, dict):
                    skipped["invalid_row"] += 1
                    continue
                match_day = _parse_match_date(
                    original.get("Date") or original.get("date")
                )
                if match_day is None:
                    skipped["missing_date"] += 1
                    continue
                home = str(original.get("HomeTeam") or original.get("home_team") or "").strip()
                away = str(original.get("AwayTeam") or original.get("away_team") or "").strip()
                if not home or not away:
                    skipped["missing_teams"] += 1
                    continue
                hg = _score(original.get("FTHG", original.get("home_goals")))
                ag = _score(original.get("FTAG", original.get("away_goals")))
                if hg is None or ag is None:
                    skipped["missing_or_invalid_score"] += 1
                    continue
                row = dict(original)
                row["Div"] = division
                prepared.append({
                    "home": home,
                    "away": away,
                    "home_goals": hg,
                    "away_goals": ag,
                    "league": division,
                    "match_date": datetime.combine(match_day, time.min),
                    "row": row,
                })

    prepared.sort(key=lambda item: (
        item["match_date"].date(),
        str(item["league"]),
        str(item["home"]).casefold(),
        str(item["away"]).casefold(),
    ))
    return prepared, skipped


def training_day_digests(rows: list[dict]) -> list[dict]:
    """Hash complete, ordered matchday inputs for safe append-only continuation.

    Hashing includes every field consumed by Engine and HistoricalMatchStats,
    not only fixture identity or score. Corrections, xG backfills and new
    fixtures inserted into an already-trained day therefore invalidate the
    incremental prefix and require a full replay.
    """
    if not isinstance(rows, list):
        raise TypeError("rows must be a list")
    result = []
    cursor = 0
    while cursor < len(rows):
        item = rows[cursor]
        if not isinstance(item, dict):
            raise TypeError("training rows must contain mappings")
        stamp = item.get("match_date")
        if isinstance(stamp, datetime):
            day = stamp.date().isoformat()
        elif isinstance(stamp, date):
            day = stamp.isoformat()
        else:
            parsed = _parse_match_date(stamp)
            if parsed is None:
                raise ValueError("prepared training row has no valid match_date")
            day = parsed.isoformat()
        end = cursor
        payload = []
        while end < len(rows):
            current = rows[end]
            current_stamp = current.get("match_date")
            if isinstance(current_stamp, datetime):
                current_day = current_stamp.date().isoformat()
            elif isinstance(current_stamp, date):
                current_day = current_stamp.isoformat()
            else:
                parsed = _parse_match_date(current_stamp)
                if parsed is None:
                    raise ValueError("prepared training row has no valid match_date")
                current_day = parsed.isoformat()
            if current_day != day:
                break
            source_row = current.get("row") or {}
            if not isinstance(source_row, dict):
                raise TypeError("prepared training row has an invalid source row")
            payload.append({
                "home": current.get("home"),
                "away": current.get("away"),
                "home_goals": current.get("home_goals"),
                "away_goals": current.get("away_goals"),
                "league": current.get("league"),
                "row": {
                    key: source_row[key]
                    for key in _TRAINING_DIGEST_FIELDS
                    if key in source_row
                },
            })
            end += 1
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), default=str,
        ).encode("utf-8")
        result.append({
            "date": day,
            "count": len(payload),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        })
        cursor = end
    return result


def plan_incremental_training(
    checkpoint: dict | None,
    current_day_digests: list[dict],
) -> dict:
    """Classify a checkpoint as reusable, append-only, or unsafe for continuation."""
    def full(reason: str) -> dict:
        return {"mode": "full", "start_index": 0, "reason": reason}

    if not isinstance(checkpoint, dict):
        return full("no_checkpoint")
    if checkpoint.get("schema") != TRAINING_STATE_SCHEMA:
        return full("checkpoint_schema_mismatch")
    previous = checkpoint.get("day_digests")
    if not isinstance(previous, list):
        return full("checkpoint_manifest_missing")
    try:
        previous_rows = sum(int(day["count"]) for day in previous)
        stored_rows = int(checkpoint.get("trained_rows"))
    except (KeyError, TypeError, ValueError, OverflowError):
        return full("checkpoint_manifest_invalid")
    if previous_rows != stored_rows or stored_rows < 0:
        return full("checkpoint_row_count_mismatch")
    engine = checkpoint.get("engine")
    if engine is None or not hasattr(engine, "match_stats"):
        return full("checkpoint_engine_missing")
    try:
        if int(getattr(engine, "trained_n", -1)) != stored_rows:
            return full("checkpoint_engine_row_count_mismatch")
    except (TypeError, ValueError, OverflowError):
        return full("checkpoint_engine_row_count_mismatch")
    if len(current_day_digests) < len(previous):
        return full("historical_rows_removed")
    if current_day_digests[:len(previous)] != previous:
        return full("historical_prefix_changed")
    start_index = previous_rows
    if len(current_day_digests) == len(previous):
        return {
            "mode": "unchanged",
            "start_index": start_index,
            "reason": "history_unchanged",
        }
    # Prefix equality plus globally sorted matchdays means all suffix dates
    # are later than the final day already trained.
    return {
        "mode": "append",
        "start_index": start_index,
        "reason": "new_completed_matchdays",
    }


def train_engine_chronologically(
    engine: Any,
    match_stats: Any,
    rows: list[dict],
    *,
    on_progress=None,
    start_index: int = 0,
    total_rows: int | None = None,
) -> dict[str, int]:
    """Train a chronological slice; offsets preserve the full-history counters."""
    if not isinstance(rows, list):
        raise TypeError("rows must be a list")
    try:
        processed = int(start_index)
        total = int(total_rows) if total_rows is not None else processed + len(rows)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("start_index and total_rows must be integers") from exc
    if processed < 0 or total < processed + len(rows):
        raise ValueError("training slice exceeds the declared full-history row count")
    days = 0
    cursor = 0
    while cursor < len(rows):
        match_day = rows[cursor]["match_date"].date()
        end = cursor + 1
        while end < len(rows) and rows[end]["match_date"].date() == match_day:
            end += 1
        batch = rows[cursor:end]
        for offset, item in enumerate(batch):
            item["match_num"] = processed + offset
            item["total"] = total

        # Engine forecasts every match on this date from the state as it stood
        # before any result from that date was admitted.
        engine.learn_batch(batch)

        # Statistical-market histories are appended only after the same-day
        # score-model forecasts have been generated.
        for item in batch:
            match_stats.learn_step(item["row"], item["league"])

        processed += len(batch)
        days += 1
        cursor = end
        if on_progress is not None:
            on_progress(processed, total)

    engine.trained_n = processed
    return {"trained": processed, "days": days, "skipped": 0}
