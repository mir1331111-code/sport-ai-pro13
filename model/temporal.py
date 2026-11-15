"""Leakage-safe temporal features and conservative Dixon-Coles parameter fitting."""
from __future__ import annotations

import math
from datetime import date, datetime, timezone
from typing import Iterable


def _as_date(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        raw = str(value).strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(raw).date()
        except ValueError:
            return date.fromisoformat(raw[:10])
    except (TypeError, ValueError):
        return None


def weighted_mean(values, dates, as_of, default=1.0, half_life_days=180.0):
    """Exponential-decay mean with strict as-of filtering when a cutoff is known.

    Undated observations are never mixed into a dated pre-match sample: their
    chronology cannot be proven, so including them could reintroduce look-ahead
    leakage. If no dated observation survives the cutoff, return the neutral
    prior instead of silently using an unweighted mean.
    """
    if not values:
        return float(default)
    cutoff = _as_date(as_of)
    if cutoff is None:
        # Preserve legacy behavior for callers that explicitly provide no
        # prediction date; the walk-forward/production path always supplies one.
        numeric = []
        for value in values:
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(number):
                numeric.append(number)
        return sum(numeric) / len(numeric) if numeric else float(default)

    observations = []
    date_values = list(dates or [])
    for index, value in enumerate(values):
        stamp = date_values[index] if index < len(date_values) else None
        day = _as_date(stamp)
        if day is None or day > cutoff:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(number):
            observations.append((number, day))

    if not observations:
        return float(default)
    if half_life_days <= 0:
        return sum(value for value, _day in observations) / len(observations)

    decay = math.log(2.0) / float(half_life_days)
    weighted = [
        (
            value,
            math.exp(-decay * max(0, (cutoff - day).days)),
        )
        for value, day in observations
    ]
    total = sum(weight for _value, weight in weighted)
    return (
        sum(value * weight for value, weight in weighted) / total
        if total else float(default)
    )


def days_since_last_match(last_match, as_of):
    """Return rest-day count, or None when either date is unknown/invalid."""
    previous, current = _as_date(last_match), _as_date(as_of)
    if previous is None or current is None or previous > current:
        return None
    return (current - previous).days


def estimate_dixon_coles_rho(
    score_history: Iterable[tuple],
    home_mean: float,
    away_mean: float,
    *,
    prior_rho: float = -0.13,
    min_matches: int = 80,
    prior_strength: float = 40.0,
    as_of=None,
) -> float:
    """Estimate rho on prior scores only, with a strong prior and bounded grid.

    Only the low-score Dixon-Coles tau terms depend on rho. A fixed, bounded
    grid avoids adding SciPy and a quadratic solver dependency. The prior keeps
    small leagues from overfitting; estimates are disabled before min_matches.
    """
    rows = []
    cutoff = _as_date(as_of)
    for item in score_history or ():
        try:
            stamp, hg, ag = item
            d = _as_date(stamp)
            hg, ag = int(hg), int(ag)
            if hg < 0 or ag < 0 or (cutoff is not None and d is not None and d > cutoff):
                continue
            if cutoff is not None and d is None:
                continue
            rows.append((d, hg, ag))
        except (TypeError, ValueError):
            continue
    if len(rows) < min_matches:
        return float(prior_rho)

    lh = min(5.0, max(0.05, float(home_mean)))
    la = min(5.0, max(0.05, float(away_mean)))
    # Shrink the likelihood contribution toward prior_rho. Weight newer
    # observations more, using a one-year half-life.
    prior = min(0.25, max(-0.25, float(prior_rho)))
    candidates = [-0.25 + i * 0.005 for i in range(101)]
    best_rho, best_score = prior, -math.inf
    today = cutoff or date.today()
    decay = math.log(2.0) / 365.0
    for rho in candidates:
        score = -prior_strength * (rho - prior) ** 2
        valid = True
        for d, hg, ag in rows:
            if (hg, ag) == (0, 0):
                tau = 1.0 - lh * la * rho
            elif (hg, ag) == (1, 0):
                tau = 1.0 + la * rho
            elif (hg, ag) == (0, 1):
                tau = 1.0 + lh * rho
            elif (hg, ag) == (1, 1):
                tau = 1.0 - rho
            else:
                continue
            if tau <= 0:
                valid = False
                break
            weight = math.exp(-decay * max(0, (today - d).days)) if d else 1.0
            score += weight * math.log(tau)
        if valid and score > best_score:
            best_rho, best_score = rho, score
    return float(best_rho)
