"""Strict historical-row preparation and chronological engine training."""
from __future__ import annotations

import math
from datetime import date, datetime, time
from typing import Any


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
    if not math.isfinite(number) or number < 0 or not number.is_integer() or number > 20:
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


def train_engine_chronologically(
    engine: Any,
    match_stats: Any,
    rows: list[dict],
    *,
    on_progress=None,
) -> dict[str, int]:
    """Train all leagues in global calendar order with same-day batch isolation."""
    if not isinstance(rows, list):
        raise TypeError("rows must be a list")
    total = len(rows)
    processed = 0
    days = 0
    cursor = 0
    while cursor < total:
        match_day = rows[cursor]["match_date"].date()
        end = cursor + 1
        while end < total and rows[end]["match_date"].date() == match_day:
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
