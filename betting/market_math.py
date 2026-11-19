"""Market probability utilities.

Only use a de-vigged benchmark when a complete, coherent market line is
available. Raw implied probability remains a safe fallback for single prices.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def proportional_devig(
    odds: Mapping[str, float],
    picks: Sequence[str],
    *,
    bookmaker_by_pick: Mapping[str, str] | None = None,
) -> dict[str, float] | None:
    """Convert a complete, single-bookmaker line to proportional no-vig probabilities.

    A set of best prices assembled across different bookmakers is not a coherent
    market line. Without bookmaker provenance for every outcome, fail closed.
    """
    if not isinstance(odds, Mapping) or not picks or not isinstance(bookmaker_by_pick, Mapping):
        return None
    bookmakers = []
    for pick in picks:
        bookmaker = str(bookmaker_by_pick.get(pick) or "").strip().casefold()
        if not bookmaker:
            return None
        bookmakers.append(bookmaker)
    if len(set(bookmakers)) != 1:
        return None
    implied: dict[str, float] = {}
    for pick in picks:
        try:
            price = float(odds.get(pick))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(price) or price <= 1.01 or price > 100:
            return None
        implied[pick] = 1.0 / price
    total = sum(implied.values())
    if not math.isfinite(total) or total <= 0:
        return None
    probabilities = {pick: value / total for pick, value in implied.items()}
    if not all(math.isfinite(p) and 0.0 < p < 1.0 for p in probabilities.values()):
        return None
    return probabilities


def blend_probability(model_probability: float, market_probability: float, model_weight: float = 0.40) -> float:
    """Conservatively shrink a model probability toward a no-vig market benchmark."""
    model_probability = float(model_probability)
    market_probability = float(market_probability)
    model_weight = float(model_weight)
    if not (0.0 < model_probability < 1.0):
        raise ValueError("model_probability must be between 0 and 1")
    if not (0.0 < market_probability < 1.0):
        raise ValueError("market_probability must be between 0 and 1")
    if not (0.0 <= model_weight <= 1.0):
        raise ValueError("model_weight must be between 0 and 1")
    return max(1e-6, min(1.0 - 1e-6,
                         market_probability + model_weight * (model_probability - market_probability)))

def competition_market_is_validated(card: Mapping, family: str) -> bool:
    """Fail closed unless the exact competition has its own independent holdout.

    A positive aggregate walk-forward report across many leagues is a research
    summary, not evidence that every individual competition is actionable.
    """
    if not isinstance(card, Mapping):
        return False
    division = str(card.get("div") or "").strip().upper()
    if not division:
        return False
    family = str(family or "").strip().casefold()
    if family == "winner":
        passed = bool(card.get("winner_market_validation"))
        scope = str(card.get("winner_market_validation_scope") or "")
        validated_division = str(card.get("winner_market_validation_division") or "").strip().upper()
    elif family == "goals":
        passed = bool(card.get("total_market_validation"))
        scope = str(card.get("total_market_validation_scope") or "")
        validated_division = str(card.get("total_market_validation_division") or "").strip().upper()
    else:
        return False
    return (
        passed
        and scope == "competition_independent_holdout"
        and validated_division == division
        and bool(card.get("market_holdout_id"))
    )

