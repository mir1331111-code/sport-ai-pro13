"""Independent forecast ranking for each football market.

This module ranks model forecasts for display only. A high model probability is
not a betting recommendation: exact bookmaker prices, calibration, historical
validation, EV, edge and risk gates remain separate.
"""
from __future__ import annotations

import math
from typing import Any

from model.historical_match_stats import poisson_over_probability
from model.market_forecasts import count_over_probability
from config import STRONG_FORECAST_MIN_PROB
from betting.ranking import balanced_value_score


# Exclude values so close to 0/1 that rounded count distributions can look certain.
PROBABILITY_EPSILON = 1e-5

_STAT_SPECS = {
    "corners": {"key": "corners", "prefix": "Угл", "line": 8.5, "label": "Угловые"},
    "cards": {"key": "cards", "prefix": "Карты", "line": 3.5, "label": "Карточки"},
}


def _finite_probability(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return (
        number
        if math.isfinite(number)
        and PROBABILITY_EPSILON < number < 1.0 - PROBABILITY_EPSILON
        else None
    )


def _sample(value: Any) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return 0
    return max(0, number)


def _market_forecast_summary(card: object, family: str) -> dict | None:
    """Return one independently calculated, comparable forecast per market.

    Goals are compared on O/U 2.5; BTTS on Yes/No; corners on O/U 8.5; cards
    on O/U 3.5. These stable reference lines prevent ranking by whichever
    extreme bookmaker line happens to be available on a fixture.
    """
    if not isinstance(card, dict):
        return None

    if family == "winner":
        probabilities = {
            "П1": _finite_probability(card.get("p1")),
            "X": _finite_probability(card.get("px")),
            "П2": _finite_probability(card.get("p2")),
        }
        if any(value is None for value in probabilities.values()):
            return None
        total = sum(probabilities.values())
        if total <= 0 or not 0.90 <= total <= 1.10:
            return None
        probabilities = {key: value / total for key, value in probabilities.items()}
        pick = max(probabilities, key=probabilities.get)
        teams = str(card.get("match_ru") or card.get("match") or "Матч")
        labels = {
            "П1": "Победа хозяев (П1)",
            "X": "Ничья (X)",
            "П2": "Победа гостей (П2)",
        }
        return {
            "family": family, "pick": pick, "label": labels[pick],
            "probability": probabilities[pick],
            "reference_line": None, "expected_total": None,
            "history_matches": 0,
            "team_samples": _sample(card.get("games")),
            "basis": "1X2 distribution",
        }

    if family == "goals":
        try:
            mean = float(card.get("lam_h")) + float(card.get("lam_a"))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(mean) or mean <= 0:
            return None
        raw_over = card.get("over")
        if raw_over in (None, ""):
            over = poisson_over_probability(mean, 2.5)
        else:
            # An invalid model output must not be silently replaced by a
            # different approximation; only a genuinely missing value may
            # use the explicit score-model fallback.
            over = _finite_probability(raw_over)
        if over is None:
            return None
        pick = "ТБ 2.5" if over >= 0.5 else "ТМ 2.5"
        probability = over if over >= 0.5 else 1.0 - over
        return {
            "family": family, "pick": pick,
            "label": f"{pick} · Тотал голов",
            "probability": probability, "over_probability": over,
            "reference_line": 2.5, "expected_total": mean,
            "history_matches": 0,
            "team_samples": _sample(card.get("games")),
            "basis": "score model",
        }

    if family == "btts":
        yes = _finite_probability(card.get("btts"))
        if yes is None:
            return None
        pick = "BTTS Да" if yes >= 0.5 else "BTTS Нет"
        probability = yes if yes >= 0.5 else 1.0 - yes
        return {
            "family": family, "pick": pick,
            "label": "Обе забьют — " + ("Да" if yes >= 0.5 else "Нет"),
            "probability": probability, "yes_probability": yes,
            "reference_line": None, "expected_total": None,
            "history_matches": 0,
            "team_samples": _sample(card.get("games")),
            "basis": "score model",
        }

    spec = _STAT_SPECS.get(family)
    if spec is None:
        return None
    stats = card.get("historical_stats")
    if not isinstance(stats, dict):
        return None
    prediction = stats.get(spec["key"])
    if not isinstance(prediction, dict):
        return None
    history_n = _sample(prediction.get("history_matches"))
    home_n = _sample(prediction.get("home_team_games"))
    away_n = _sample(prediction.get("away_team_games"))
    try:
        mean = float(prediction.get("total"))
    except (TypeError, ValueError):
        return None
    if (
        history_n < 30 or home_n < 4 or away_n < 4
        or not math.isfinite(mean) or mean <= 0
    ):
        return None
    over = _finite_probability(count_over_probability(
        mean, spec["line"], prediction.get("total_variance")
    ))
    if over is None:
        return None
    pick = (
        f"{spec['prefix']} ТБ {spec['line']:g}"
        if over >= 0.5 else f"{spec['prefix']} ТМ {spec['line']:g}"
    )
    probability = over if over >= 0.5 else 1.0 - over
    return {
        "family": family, "pick": pick,
        "label": f"{spec['label']} · {pick.split(' ', 1)[1]}",
        "probability": probability, "over_probability": over,
        "reference_line": spec["line"], "expected_total": mean,
        "history_matches": history_n, "team_samples": min(home_n, away_n),
        "home_team_games": home_n, "away_team_games": away_n,
        "basis": "historical count model",
    }


def market_forecast_summary(card: object, family: str) -> dict | None:
    """Return a market forecast with a separate 65% strong-forecast label."""
    summary = _market_forecast_summary(card, family)
    if summary is None:
        return None
    summary["strength_threshold"] = float(STRONG_FORECAST_MIN_PROB)
    summary["strong_forecast"] = (
        float(summary["probability"]) >= float(STRONG_FORECAST_MIN_PROB)
    )
    return summary


def rank_market_forecasts(
    cards: object, family: str, limit: int | None = None,
    *, mode: str = "probability",
) -> list[dict]:
    """Rank independent forecasts by event probability or balanced validated value.

    In balanced mode, probability remains the largest score component. EV and
    edge can contribute only through a recommendation already accepted by the
    strict bookmaker-benchmark selector; unvalidated prices are ignored.
    """
    if not isinstance(cards, list):
        return []
    ranked = []
    ranking_mode = str(mode or "probability").casefold()
    for card in cards:
        summary = market_forecast_summary(card, family)
        if summary is None:
            continue
        ranking_score = float(summary["probability"])
        if ranking_mode == "balanced":
            recommendation = (card.get("best_by_market") or {}).get(family)
            value_verdict = (
                recommendation.get("verdict")
                if isinstance(recommendation, dict) else None
            )
            value_best = (
                recommendation.get("best")
                if isinstance(recommendation, dict) else None
            )
            # EV/edge influence display order only after the existing strict
            # market validation, complete-line, EV, edge and Kelly gates passed.
            if (
                isinstance(value_verdict, dict)
                and isinstance(value_best, (tuple, list))
                and len(value_best) >= 6
                and bool(value_verdict.get("value_passed"))
            ):
                ranking_score = balanced_value_score(
                    summary["probability"],
                    value_verdict.get("ev"),
                    value_verdict.get("edge"),
                )
        ranked.append({
            "card": card, **summary, "ranking_score": ranking_score,
        })
    if ranking_mode == "balanced":
        ranked.sort(key=lambda item: (
            -float(item["ranking_score"]),
            -float(item["probability"]),
            -int(item.get("history_matches") or 0),
            -int(item.get("team_samples") or 0),
            str(item["card"].get("date_iso") or item["card"].get("date") or ""),
            str(item["card"].get("match") or item["card"].get("match_ru") or ""),
        ))
    else:
        ranked.sort(key=lambda item: (
            -float(item["probability"]),
            -int(item.get("history_matches") or 0),
            -int(item.get("team_samples") or 0),
            str(item["card"].get("date_iso") or item["card"].get("date") or ""),
            str(item["card"].get("match") or item["card"].get("match_ru") or ""),
        ))
    if limit is not None:
        try:
            ranked = ranked[:max(0, int(limit))]
        except (TypeError, ValueError):
            pass
    return ranked
