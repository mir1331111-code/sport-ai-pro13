"""Cross-market event shortlist and human-readable market coverage.

Forecasts, prices and approved betting signals are deliberately separate:
unvalidated markets stay visible but cannot become Value/portfolio picks.
"""
from __future__ import annotations

import math
import re
from typing import Any

from betting.market_ranking import market_forecast_summary
from model.historical_match_stats import poisson_over_probability
from model.market_forecasts import count_over_probability
from model.market_models import asian_handicap_expected_value


_FAMILY_LABELS = {
    "winner": "Победа / 1X2",
    "goals": "Тоталы голов",
    "btts": "Обе забьют",
    "corners": "Угловые",
    "cards": "Карточки",
    "handicap": "Азиатская фора",
}
_FAMILY_ORDER = ("winner", "goals", "btts", "corners", "cards", "handicap")
_TOTAL_RE = re.compile(r"^Т([БМ])\s+(\d+(?:[.,]\d+)?)$", re.I)
_STAT_RE = re.compile(r"^(Угл|Карты)\s+Т([БМ])\s+(\d+(?:[.,]\d+)?)$", re.I)
_AH_RE = re.compile(r"^Ф([12])\s+([+-]?\d+(?:[.,]\d+)?)$", re.I)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _valid_odds(value: Any) -> float | None:
    number = _finite(value)
    return number if number is not None and 1.01 < number <= 100.0 else None


def _card_id(card: dict) -> str:
    fixture_id = str(card.get("fixture_id") or "").strip()
    if fixture_id:
        return fixture_id
    return "|".join(
        str(value or "").strip().casefold()
        for value in (
            card.get("date_iso") or card.get("date"),
            card.get("home_team") or card.get("match"),
            card.get("away_team") or card.get("match_ru"),
        )
    )


def _market_quote(card: dict, family: str, pick: str) -> float | None:
    sources = {
        "winner": card.get("market_odds"),
        "goals": card.get("total_market_odds"),
        "btts": card.get("btts_market_odds"),
        "corners": card.get("stat_market_odds"),
        "cards": card.get("stat_market_odds"),
        "handicap": card.get("asian_handicap_market_odds"),
    }
    mapping = sources.get(family)
    if not isinstance(mapping, dict):
        return None
    # Canonical case and line formatting for provider labels.
    if pick in mapping:
        return _valid_odds(mapping[pick])
    for raw_pick, raw_price in mapping.items():
        if str(raw_pick).strip().casefold() == str(pick).strip().casefold():
            return _valid_odds(raw_price)
    return None


def _source(card: dict, family: str) -> str:
    field = {
        "winner": "odds_source",
        "goals": "total_odds_source",
        "btts": "btts_odds_source",
        "corners": "extra_market_source",
        "cards": "extra_market_source",
        "handicap": "extra_market_source",
    }.get(family)
    return str(card.get(field) or "") if field else ""


def _recommendation(card: dict, family: str) -> dict | None:
    value = (card.get("best_by_market") or {}).get(family)
    if not isinstance(value, dict):
        return None
    verdict = value.get("verdict")
    best = value.get("best")
    if (
        isinstance(verdict, dict)
        and isinstance(best, (tuple, list))
        and len(best) >= 6
        and bool(verdict.get("value_passed"))
    ):
        return value
    return None


def _base_row(
    card: dict,
    family: str,
    *,
    pick: str,
    label: str,
    probability: float,
    odds: float | None,
    history_matches: int = 0,
    team_samples: int = 0,
    strong: bool = False,
    status: str = "forecast_only",
    model_ev: float | None = None,
    outcome_detail: str = "",
) -> dict:
    recommendation = _recommendation(card, family)
    value_verdict = (recommendation or {}).get("verdict") or {}
    value_best = (recommendation or {}).get("best") or ()
    verified = recommendation is not None and str(value_best[1]) == str(pick)

    if verified:
        probability = _finite(value_verdict.get("prob")) or probability
        odds = _valid_odds(value_best[2]) or odds
        model_ev = _finite(value_verdict.get("ev"))
        status = "value_verified"

    if status != "value_verified" and family != "handicap":
        status = "forecast_with_price" if odds is not None else "forecast_only"
    elif status == "handicap_research_only" and odds is None:
        status = "forecast_only"

    # Asian handicaps have split settlements; their break-even price is not 1 / P(win).
    fair_odds = (
        1.0 / probability
        if family != "handicap" and 0.0 < probability < 1.0 else None
    )
    score = float(recommendation.get("score") or 0.0) if verified else 0.0
    # Display-only raw model EV must never be labelled as verified Value.
    if not verified and odds is not None and model_ev is None:
        model_ev = probability * odds - 1.0

    return {
        "fixture_id": _card_id(card),
        "match": str(card.get("match_ru") or card.get("match") or "Матч"),
        "league": str(card.get("league") or card.get("div") or "Футбол"),
        "date": str(card.get("date") or card.get("date_iso") or ""),
        "time": str(card.get("time") or ""),
        "family": family,
        "family_label": _FAMILY_LABELS[family],
        "pick": pick,
        "label": label,
        "probability": probability,
        "odds": odds,
        "fair_odds": fair_odds,
        "model_ev": model_ev,
        "edge": _finite(value_verdict.get("edge")) if verified else None,
        "kelly_pct": _finite(value_verdict.get("kelly_pct")) if verified else None,
        "bookmaker": str((card.get("bookmaker_by_pick") or {}).get(pick) or ""),
        "source": _source(card, family),
        "history_matches": int(history_matches or 0),
        "team_samples": int(team_samples or 0),
        "strong_forecast": bool(strong),
        "value_verified": verified,
        "status": status,
        "status_label": (
            "ПРОВЕРЕННЫЙ VALUE" if verified
            else "ФОРА · ИССЛЕДОВАТЕЛЬСКАЯ ОЦЕНКА" if family == "handicap"
            else "ПРОГНОЗ + КОЭФФИЦИЕНТ · НЕ VALUE" if odds is not None
            else "ПРОГНОЗ · НЕТ КОЭФФИЦИЕНТА"
        ),
        "outcome_detail": outcome_detail,
        "reason": (
            "Прошёл независимую проверку рынка, полную линию одного букмекера, "
            "Edge, EV и положительный Kelly."
            if verified else
            "Азиатская фора рассчитана по полной матрице счёта с учётом полупобеды, возврата "
            "и полупоражения; отдельный out-of-sample benchmark ещё не подтверждён."
            if family == "handicap" else
            "Есть модельная вероятность и цена, но независимый benchmark этого рынка "
            "ещё не подтверждён. Модельный EV предварительный; не считать ставкой."
            if odds is not None else
            "Есть модельная оценка, но нет подтверждённой цены именно на этот исход."
        ),
        "_rank": (
            3 if verified else
            2 if strong and odds is not None else
            1 if odds is not None else 0
        ),
        "_value_score": score,
    }


def _family_candidates(card: dict, family: str) -> list[dict]:
    summary = market_forecast_summary(card, family)
    if summary is None:
        return []
    history_n = int(summary.get("history_matches") or 0)
    team_n = int(summary.get("team_samples") or 0)
    strong = bool(summary.get("strong_forecast"))
    rows = []

    if family == "winner":
        pick = str(summary["pick"])
        probability = float(summary["probability"])
        odds = _market_quote(card, family, pick)
        rows.append(_base_row(
            card, family, pick=pick, label=str(summary["label"]),
            probability=probability, odds=odds,
            history_matches=history_n, team_samples=team_n, strong=strong,
        ))
        return rows

    if family == "btts":
        yes_probability = _finite(card.get("btts"))
        if yes_probability is None:
            return []
        for pick, probability, label in (
            ("BTTS Да", yes_probability, "Обе команды забьют — да"),
            ("BTTS Нет", 1.0 - yes_probability, "Обе команды забьют — нет"),
        ):
            odds = _market_quote(card, family, pick)
            rows.append(_base_row(
                card, family, pick=pick, label=label, probability=probability,
                odds=odds, history_matches=history_n, team_samples=team_n,
                strong=(probability >= float(summary.get("strength_threshold") or 0.65)
                        and bool(summary.get("history_sufficient"))),
            ))
        return rows

    if family == "goals":
        try:
            total_mean = float(card.get("lam_h")) + float(card.get("lam_a"))
        except (TypeError, ValueError):
            return []
        quoted = card.get("total_market_odds") or {}
        candidates = []
        for raw_pick, raw_odd in quoted.items():
            match = _TOTAL_RE.fullmatch(str(raw_pick).strip())
            if not match:
                continue
            odds = _valid_odds(raw_odd)
            if odds is None:
                continue
            line = float(match.group(2).replace(",", "."))
            raw_over = card.get("over")
            over = (
                _finite(raw_over)
                if abs(line - 2.5) < 1e-9 and raw_over not in (None, "")
                else poisson_over_probability(total_mean, line)
            )
            if over is None:
                continue
            probability = over if match.group(1).upper() == "Б" else 1.0 - over
            candidates.append(_base_row(
                card, family, pick=str(raw_pick),
                label=f"Тотал голов {'больше' if match.group(1).upper() == 'Б' else 'меньше'} {line:g}",
                probability=probability, odds=odds, history_matches=history_n,
                team_samples=team_n,
                strong=(probability >= float(summary.get("strength_threshold") or 0.65)
                        and bool(summary.get("history_sufficient"))),
            ))
        if candidates:
            return candidates
        pick = str(summary["pick"])
        probability = (
            float(summary["over_probability"])
            if pick.startswith("ТБ")
            else 1.0 - float(summary["over_probability"])
        )
        return [_base_row(
            card, family, pick=pick, label=str(summary["label"]),
            probability=probability, odds=_market_quote(card, family, pick),
            history_matches=history_n, team_samples=team_n, strong=strong,
        )]

    if family in {"corners", "cards"}:
        prediction = (card.get("historical_stats") or {}).get(family) or {}
        try:
            mean = float(prediction.get("total"))
        except (TypeError, ValueError):
            return []
        variance = prediction.get("total_variance")
        prefix = "Угл" if family == "corners" else "Карты"
        quoted = card.get("stat_market_odds") or {}
        candidates = []
        for raw_pick, raw_odd in quoted.items():
            match = _STAT_RE.fullmatch(str(raw_pick).strip())
            if not match:
                continue
            if ("Угл" if match.group(1).casefold() == "угл" else "Карты") != prefix:
                continue
            odds = _valid_odds(raw_odd)
            if odds is None:
                continue
            line = float(match.group(3).replace(",", "."))
            over = count_over_probability(mean, line, variance)
            if over is None:
                continue
            probability = over if match.group(2).upper() == "Б" else 1.0 - over
            candidates.append(_base_row(
                card, family, pick=str(raw_pick),
                label=f"{_FAMILY_LABELS[family]} {'больше' if match.group(2).upper() == 'Б' else 'меньше'} {line:g}",
                probability=probability, odds=odds,
                history_matches=history_n, team_samples=team_n,
                strong=(probability >= 0.65),
            ))
        if candidates:
            return candidates

        pick = str(summary["pick"])
        probability = (
            float(summary["over_probability"])
            if pick.split(" ")[1] == "ТБ"
            else 1.0 - float(summary["over_probability"])
        )
        return [_base_row(
            card, family, pick=pick, label=str(summary["label"]),
            probability=probability, odds=_market_quote(card, family, pick),
            history_matches=history_n, team_samples=team_n,
            strong=bool(summary["strong_forecast"]),
        )]

    return []


def build_best_market_events(cards: object, limit: int = 30) -> list[dict]:
    """Return one best display candidate per fixture/market family, ranked safely."""
    if not isinstance(cards, list):
        return []
    events = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        for family in ("winner", "goals", "btts", "corners", "cards"):
            rows = _family_candidates(card, family)
            if not rows:
                continue
            # Prefer a verified signal; otherwise prefer the highest preliminary
            # model EV among priced alternatives, then probability/sample size.
            rows.sort(key=lambda row: (
                row["_rank"],
                row["_value_score"],
                row["model_ev"] if row["model_ev"] is not None else -999.0,
                row["probability"],
                row["history_matches"],
                row["team_samples"],
            ), reverse=True)
            events.append(rows[0])

        handicap_rows = []
        for item in card.get("asian_handicap_forecasts") or []:
            if not isinstance(item, dict):
                continue
            pick = str(item.get("pick") or "")
            match = _AH_RE.fullmatch(pick)
            probabilities = item.get("probabilities") or {}
            if not match or not isinstance(probabilities, dict):
                continue
            odds = _valid_odds(item.get("odds")) or _market_quote(card, "handicap", pick)
            try:
                model_ev = asian_handicap_expected_value(probabilities, odds) if odds else None
            except (TypeError, ValueError):
                model_ev = None
            win = float(probabilities.get("win") or 0.0)
            half_win = float(probabilities.get("half_win") or 0.0)
            event_probability = min(1.0, max(0.0, win + half_win))
            detail = (
                f"полная победа {win:.1%}; полпобеды {half_win:.1%}; "
                f"возврат {float(probabilities.get('push') or 0):.1%}; "
                f"полупоражение {float(probabilities.get('half_loss') or 0):.1%}; "
                f"поражение {float(probabilities.get('loss') or 0):.1%}"
            )
            handicap_rows.append(_base_row(
                card, "handicap", pick=pick,
                label=f"Азиатская фора {pick}", probability=event_probability,
                odds=odds, history_matches=0, team_samples=0, strong=False,
                status="handicap_research_only", model_ev=model_ev,
                outcome_detail=detail,
            ))
        if handicap_rows:
            handicap_rows.sort(key=lambda row: (
                row["model_ev"] if row["model_ev"] is not None else -999.0,
                row["probability"],
            ), reverse=True)
            events.append(handicap_rows[0])

    events.sort(key=lambda row: (
        -int(row["_rank"]),
        -float(row["_value_score"]),
        -int(row["strong_forecast"]),
        -(row["model_ev"] if row["model_ev"] is not None else -999.0),
        -float(row["probability"]),
        -int(row["history_matches"]),
        -int(row["team_samples"]),
        row["date"],
        row["match"],
        row["family"],
    ))
    for row in events:
        row.pop("_rank", None)
        row.pop("_value_score", None)
    try:
        cap = max(0, int(limit))
    except (TypeError, ValueError):
        cap = 30
    return events[:cap]


def build_market_coverage(cards: object) -> list[dict]:
    """Count unique fixtures with a forecast/quote/approved Value per market."""
    valid_cards = [card for card in cards if isinstance(card, dict)] if isinstance(cards, list) else []
    result = []
    for family in _FAMILY_ORDER:
        forecast_ids = set()
        quote_ids = set()
        value_ids = set()
        strong_ids = set()
        for card in valid_cards:
            identity = _card_id(card)
            if family == "handicap":
                has_forecast = bool(card.get("asian_handicap_forecasts"))
                has_quote = bool(card.get("asian_handicap_market_odds"))
            else:
                has_forecast = market_forecast_summary(card, family) is not None
                fields = {
                    "winner": ("market_odds",),
                    "goals": ("total_market_odds",),
                    "btts": ("btts_market_odds",),
                    "corners": ("stat_market_odds",),
                    "cards": ("stat_market_odds",),
                }[family]
                if family in {"corners", "cards"}:
                    prefix = "Угл " if family == "corners" else "Карты "
                    has_quote = any(
                        str(key).startswith(prefix)
                        for key in (card.get("stat_market_odds") or {})
                    )
                else:
                    has_quote = any(bool(card.get(field)) for field in fields)
            recommendation = _recommendation(card, family) if family != "handicap" else None
            if has_forecast:
                forecast_ids.add(identity)
            if has_quote:
                quote_ids.add(identity)
            if recommendation:
                value_ids.add(identity)
            summary = market_forecast_summary(card, family) if family != "handicap" else None
            if summary and summary.get("strong_forecast"):
                strong_ids.add(identity)

        result.append({
            "market": _FAMILY_LABELS[family],
            "family": family,
            "forecast_matches": len(forecast_ids),
            "quoted_matches": len(quote_ids),
            "value_matches": len(value_ids),
            "forecast_only_matches": max(0, len(forecast_ids) - len(value_ids)),
            "strong_matches": len(strong_ids),
        })
    return result
