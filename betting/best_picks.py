"""Select the best validated, priced market for one fixture."""
from __future__ import annotations

import re
from copy import deepcopy

from betting.verdict import refine_with_real_odds
from betting.kelly import market_type
from betting.market_math import proportional_devig


def _paired_picks(pick: str) -> tuple[str, ...] | None:
    """Return the exact coherent two-/three-way selections for a market line."""
    kind = market_type(pick)
    if kind == "1X2":
        return ("П1", "X", "П2")
    if kind == "OU":
        match = re.fullmatch(r"Т[БМ]\s+(\d+(?:[.,]\d+)?)", str(pick or ""), re.I)
        if match:
            line = match.group(1).replace(",", ".")
            return (f"ТБ {line}", f"ТМ {line}")
    elif kind == "BTTS":
        return ("BTTS Да", "BTTS Нет")
    elif kind == "STAT":
        match = re.fullmatch(
            r"(Угл|Карты)\s+(?:ТБ|ТМ)\s+(\d+(?:[.,]\d+)?)",
            str(pick or ""), re.I,
        )
        if match:
            prefix = "Угл" if match.group(1).casefold() == "угл" else "Карты"
            line = match.group(2).replace(",", ".")
            return (f"{prefix} ТБ {line}", f"{prefix} ТМ {line}")
    return None


def best_value_market(
    verdict: dict,
    rows: list,
    market_odds: dict | None,
    bank: float,
    kelly_frac: float,
    min_prob: float,
    *,
    min_edge: float = 0.03,
    min_ev: float = 0.03,
    bookmaker_by_pick: dict | None = None,
    market_validation_by_pick: dict | None = None,
):
    """Choose the best market only from tested forecasts and exact bookmaker lines.

    1X2 retains backward-compatible raw-implied fallback when provenance is absent.
    Two-way markets require both sides from the same named bookmaker and an
    explicit line-specific historical-validation flag. No 1X2 or other-market
    price may be substituted for the requested market.
    """
    if not isinstance(market_odds, dict) or not market_odds:
        return None, None

    bookmaker_by_pick = bookmaker_by_pick or {}
    market_validation_by_pick = market_validation_by_pick or {}
    qualified = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        pick = str(row.get("pick") or "")
        kind = market_type(pick)
        if kind not in {"1X2", "OU", "BTTS", "STAT"}:
            continue
        if kind != "1X2" and not bool(market_validation_by_pick.get(pick)):
            continue
        try:
            prob = float(row.get("prob"))
            odd = float(market_odds.get(pick))
        except (TypeError, ValueError):
            continue
        if not (0.0 < prob < 1.0 and 1.01 < odd <= 100.0 and prob >= float(min_prob)):
            continue

        paired_picks = _paired_picks(pick)
        no_vig = proportional_devig(
            market_odds, paired_picks or (), bookmaker_by_pick=bookmaker_by_pick
        )
        if kind != "1X2" and no_vig is None:
            # Do not compare two-way prices taken from different bookmakers.
            continue

        candidate = deepcopy(verdict)
        candidate.update({
            "pick": pick,
            "market_analysis": row.get("analysis") or "",
            "label": row.get("label") or pick,
            "prob": prob,
            "fair_odd": 1.0 / prob,
            "market_type": kind,
            "is_action": True,
            "odd": None,
            "real_odds": False,
            "ev": None,
            "edge": None,
            "stake": 0.0,
            "is_bet": False,
        })
        refined, best = refine_with_real_odds(
            candidate,
            [dict(row)],
            {pick: odd},
            bank,
            kelly_frac,
            min_edge=min_edge,
            min_ev=min_ev,
            market_fair_prob=(no_vig or {}).get(pick),
        )
        if (
            not best
            or not refined.get("value_passed")
            or float(refined.get("prob") or 0.0) < float(min_prob)
        ):
            continue

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
