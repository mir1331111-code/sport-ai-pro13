"""Leakage-resistant audit of recorded market forecasts against entry prices."""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable


def _safe_probability(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and 0.0 < number < 1.0 else None


def _bootstrap_lower(values: list[float], *, reps: int = 1200, seed: int = 20261111) -> float | None:
    if not values:
        return None
    if len(values) < 30:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(reps):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[max(0, int(0.025 * (len(means) - 1)))]


def validate_market_rows(rows: Iterable[dict], *, min_samples: int = 500) -> dict:
    """Track model-vs-raw-price scores; never labels a small sample validated.

    Only immutable forecasts settled to WON/LOST and with a valid entry price
    participate. The raw implied price is not de-vigged; the report states this
    limitation. These metrics are diagnostics, not proof of profitable betting.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        status = str(row.get("result_status") or "").lower()
        market = str(row.get("market") or "OTHER").upper()
        if status not in {"won", "lost"}:
            continue
        probability = _safe_probability(row.get("probability"))
        odds = row.get("entry_odds")
        try:
            odds = float(odds)
        except (TypeError, ValueError):
            continue
        if probability is None or not math.isfinite(odds) or odds <= 1.01:
            continue
        groups[market].append(row)

    reports = []
    for market, sample in sorted(groups.items()):
        model_brier, market_brier, model_ll, market_ll, improvements = [], [], [], [], []
        for row in sample:
            p = _safe_probability(row.get("probability"))
            price_p = min(0.999999, max(0.000001, 1.0 / float(row["entry_odds"])))
            y = 1.0 if str(row.get("result_status")).lower() == "won" else 0.0
            model_brier.append((p - y) ** 2)
            market_brier.append((price_p - y) ** 2)
            model_ll.append(-math.log(max(p if y else 1.0 - p, 1e-15)))
            market_ll.append(-math.log(max(price_p if y else 1.0 - price_p, 1e-15)))
            improvements.append((price_p - y) ** 2 - (p - y) ** 2)
        n = len(sample)
        delta_brier = sum(improvements) / n if n else None
        lower = _bootstrap_lower(
            improvements, seed=20261111 + sum(ord(ch) for ch in market)
        )
        delta_logloss = (
            sum(market_ll) / n - sum(model_ll) / n if n else None
        )
        eligible = bool(
            n >= int(min_samples)
            and lower is not None and lower > 0.0
            and delta_logloss is not None and delta_logloss > 0.0
        )
        reports.append({
            "market": market,
            "n": n,
            "model_brier": sum(model_brier) / n if n else None,
            "price_brier": sum(market_brier) / n if n else None,
            "brier_improvement": delta_brier,
            "brier_improvement_ci_low_95": lower,
            "model_log_loss": sum(model_ll) / n if n else None,
            "price_log_loss": sum(market_ll) / n if n else None,
            "log_loss_improvement": delta_logloss,
            "eligible_for_review": eligible,
            "status": "VALIDATED_FOR_REVIEW" if eligible else (
                "INSUFFICIENT_SAMPLE" if n < int(min_samples)
                else "NO_CONFIRMED_IMPROVEMENT"
            ),
        })
    return {
        "markets": reports,
        "min_samples": int(min_samples),
        "method": "paired model probability vs raw 1/entry_odds; bootstrap lower CI on Brier improvement",
        "caveat": (
            "The raw implied price includes bookmaker margin and is not de-vigged. "
            "The journal contains selected market picks, not every market quote. "
            "This diagnostic does not independently prove ROI or authorize automatic bets."
        ),
    }
