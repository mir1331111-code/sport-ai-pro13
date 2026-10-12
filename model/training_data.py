"""Validation helpers for historical football training rows."""
from __future__ import annotations

from datetime import date, datetime
from math import isfinite
from typing import Any


def completed_training_row(row: Any, match_date: Any, today: date) -> tuple | None:
    """Return (date, home, away, home_goals, away_goals) for a valid finished match.

    Reject missing scores, non-integer/negative/extreme scores, missing teams or
    dates, and dates in the future. This helper deliberately does not infer 0-0
    from absent data.
    """
    if not isinstance(row, dict) or match_date is None:
        return None
    home = str(row.get("HomeTeam") or "").strip()
    away = str(row.get("AwayTeam") or "").strip()
    if not home or not away:
        return None
    raw_home, raw_away = row.get("FTHG"), row.get("FTAG")
    if raw_home in (None, "") or raw_away in (None, ""):
        return None
    try:
        home_goals, away_goals = float(raw_home), float(raw_away)
    except (TypeError, ValueError, OverflowError):
        return None
    if not (isfinite(home_goals) and isfinite(away_goals)):
        return None
    if not (home_goals.is_integer() and away_goals.is_integer()):
        return None
    if not (0 <= home_goals <= 30 and 0 <= away_goals <= 30):
        return None
    match_day = match_date.date() if isinstance(match_date, datetime) else match_date
    if not isinstance(match_day, date) or match_day > today:
        return None
    return match_date, home, away, home_goals, away_goals
