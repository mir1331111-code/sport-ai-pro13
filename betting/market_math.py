"""Market probability utilities.

Only use a de-vigged benchmark when a complete, coherent market line is
available. Raw implied probability remains a safe fallback for single prices.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def proportional_devig(odds: Mapping[str, float], picks: Sequence[str]) -> dict[str, float] | None:
    """Convert a complete set of decimal odds to proportional no-vig probabilities.

    Returns None rather than inventing missing prices or normalizing a partial line.
    """
    if not isinstance(odds, Mapping) or not picks:
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
