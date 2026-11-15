"""Fail-closed validation for full-time football scores used by settlement."""
from __future__ import annotations

import math
from typing import Any

FINISHED_SCORE_STATUSES = frozenset({
    "finished", "ft", "final", "closed", "ended",
    "full-time", "full_time", "completed", "match finished",
})
EXTRA_TIME_SCORE_STATUSES = frozenset({
    "aet", "after extra time", "after penalties", "pen", "penalties",
})


def _non_negative_integer(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value >= 0
    if isinstance(value, float):
        return math.isfinite(value) and value >= 0 and value.is_integer()
    return isinstance(value, str) and value.strip().isascii() and value.strip().isdigit()


def is_verified_final_score(result: Any) -> bool:
    """Accept only explicit final status and non-negative integer scores.

    Extra-time/shootout results are acceptable only if the provider separately
    marks that the supplied score is the verified regulation-time score.
    Missing/unknown statuses never imply FT, even when scores are present.
    """
    if not isinstance(result, dict):
        return False
    status = str(result.get("status") or "").strip().casefold()
    if status in EXTRA_TIME_SCORE_STATUSES:
        if result.get("regulation_time") is not True:
            return False
    elif status not in FINISHED_SCORE_STATUSES:
        return False
    return (
        _non_negative_integer(result.get("home"))
        and _non_negative_integer(result.get("away"))
    )
