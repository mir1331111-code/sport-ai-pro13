"""Select the best priced market for a fixture using model probabilities and real odds."""
from __future__ import annotations

from copy import deepcopy
from betting.verdict import refine_with_real_odds
from betting.kelly import market_type


def best_value_market(verdict: dict, rows: list, market_odds: dict | None,
                      bank: float, kelly_frac: float, min_prob: float,
                      *, min_edge: float = 0.03, min_ev: float = 0.03):
    """Return (candidate_verdict, best_tuple) for the strongest qualified real-price market.

    Never treats model fair odds as bookmaker prices. A candidate must have a
    real price, clear the probability threshold, pass positive fractional Kelly,
    and meet both minimum edge and expected-value thresholds.
    """
    if not isinstance(market_odds, dict) or not market_odds:
        return None, None

    qualified = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        pick = row.get("pick")
        # Until each market has its own out-of-sample calibration/benchmark,
        # only the 1X2 market can become an automatic value candidate.
        if market_type(pick) != "1X2":
            continue
        try:
            prob = float(row.get("prob"))
            odd = float(market_odds.get(pick))
        except (TypeError, ValueError):
            continue
        if not (0.0 < prob < 1.0 and odd > 1.01 and prob >= float(min_prob)):
            continue

        candidate = deepcopy(verdict)
        candidate.update({
            "pick": pick,
            "label": row.get("label") or pick,
            "prob": prob,
            "fair_odd": 1.0 / prob,
            "is_action": True,
            "odd": None,
            "real_odds": False,
            "ev": None,
            "edge": None,
            "stake": 0.0,
            "is_bet": False,
        })
        refined, best = refine_with_real_odds(
            candidate, [dict(row)], {pick: odd}, bank, kelly_frac,
            min_edge=min_edge, min_ev=min_ev,
        )
        if (not best or not refined.get("value_passed")
                or float(refined.get("prob") or 0.0) < float(min_prob)):
            continue

        # Candidate-specific confidence so the card's rank matches the selected market.
        # Rank and label using the final market-blended probability, not the
        # unadjusted model probability. This avoids calling a pick "high" when
        # the real line strongly disagrees with the model.
        final_prob = float(refined.get("prob") or 0.0)
        confidence_score = round(final_prob * 100.0, 1)
        refined["confidence_score"] = confidence_score
        refined["confidence_basis"] = "market_blended_probability_not_historical_hit_rate"
        if final_prob >= 0.80:
            refined["confidence"] = "очень высокая"
        elif final_prob >= 0.72:
            refined["confidence"] = "высокая"
        elif final_prob >= 0.65:
            refined["confidence"] = "хорошая"
        elif final_prob >= 0.55:
            refined["confidence"] = "средняя"
        else:
            refined["confidence"] = "низкая"
        refined["confidence_label"] = (
            "A" if confidence_score >= 80 else
            "B" if confidence_score >= 65 else
            "C" if confidence_score >= 50 else "D"
        )

        # Prefer estimated return, then edge, then probability as tie-breakers.
        score = (
            0.60 * max(-1.0, min(1.0, float(refined["ev"])))
            + 0.25 * max(-1.0, min(1.0, float(refined["edge"])))
            + 0.15 * final_prob
        )
        qualified.append((score, refined, best))

    if not qualified:
        return None, None
    _, chosen, best = max(qualified, key=lambda item: item[0])
    return chosen, best
