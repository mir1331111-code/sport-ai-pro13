"""Fail-closed validation for full-time football scores used by settlement."""
from __future__ import annotations

from typing import Any

FINISHED_SCORE_STATUSES = frozenset({
    "finished", "ft", "final", "closed", "ended",
    "full-time", "full_time", "completed", "match finished",
})


def is_verified_final_score(result: Any) -> bool:
    """Accept only explicit final status and non-negative scores.

    Extra-time/shootout results are acceptable only if the provider separately
    marks that the supplied score is the verified regulation-time score.
    Missing/unknown statuses never imply FT, even when scores are present.
    """
    if not isinstance(result, dict):
        return False
    status = str(result.get("status") or "").strip().casefold()
    if status in {"aet", "after extra time", "after penalties", "pen", "penalties"}:
        if result.get("regulation_time") is not True:
            return False
    elif status not in FINISHED_SCORE_STATUSES:
        return False

    home, away = result.get("home"), result.get("away")
    if isinstance(home, bool) or isinstance(away, bool):
        return False
    try:
        home_score, away_score = int(home), int(away)
    except (TypeError, ValueError, OverflowError):
        return False
    return home_score >= 0 and away_score >= 0
