"""Transparent per-fixture data quality assessment.

The score measures completeness/freshness of the evidence shown to the user.
It is not a probability, model confidence, or a betting recommendation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any

STATUSES = {
    "CONFIRMED": "Подтверждено",
    "CONNECTED": "Источник ответил",
    "NO_DATA": "Ответ получен, данных нет",
    "UNAVAILABLE": "Источник недоступен",
    "NOT_CONFIGURED": "Не настроен",
    "NOT_DUE": "Пока не запрашивается",
    "NOT_REQUESTED": "Не запрашивалось",
    "STALE": "Устарело",
    "CONFLICT": "Конфликт источников",
    "UNKNOWN": "Неизвестно",
}

_WEIGHTS = {
    "fixture": 8.0,
    "category": 8.0,
    "api_football": 10.0,
    "home_lineup": 13.0,
    "away_lineup": 13.0,
    "referee": 10.0,
    "injuries": 10.0,
    "odds": 18.0,
    "history": 10.0,
}
_STATUS_VALUE = {
    "CONFIRMED": 1.0,
    "CONNECTED": 0.55,
    "NO_DATA": 0.20,
    "NOT_DUE": None,
    "NOT_REQUESTED": 0.0,
    "NOT_CONFIGURED": 0.0,
    "UNAVAILABLE": 0.0,
    "STALE": 0.0,
    "CONFLICT": 0.0,
    "UNKNOWN": 0.0,
}


def _parse_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    else:
        raw = str(value or "").strip()
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except (TypeError, ValueError, OverflowError):
            return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _freshness(status: str, observed_at: Any, now: datetime,
               max_age_hours: float = 12.0) -> str:
    if status not in {"CONFIRMED", "CONNECTED", "NO_DATA"}:
        return status
    observed = _parse_time(observed_at)
    if observed is None:
        return status
    age = (now - observed).total_seconds()
    if age < -300:
        return "CONFLICT"
    if age > max_age_hours * 3600:
        return "STALE"
    return status


def _valid_prices(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    for raw in value.values():
        try:
            price = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(price) and 1.01 < price <= 1000:
            return True
    return False


def assess_fixture_quality(card: dict | None, *, now: datetime | None = None) -> dict:
    """Return transparent statuses and a weighted evidence-completeness score."""
    card = card if isinstance(card, dict) else {}
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)

    context = card.get("api_context") if isinstance(card.get("api_context"), dict) else {}
    lineup = context.get("lineup") if isinstance(context.get("lineup"), dict) else {}
    home_lineup = lineup.get("home") if isinstance(lineup.get("home"), dict) else {}
    away_lineup = lineup.get("away") if isinstance(lineup.get("away"), dict) else {}
    referee = context.get("referee") if isinstance(context.get("referee"), dict) else {}

    # New snapshots carry explicit request outcomes. Older cards are inferred
    # conservatively and never promoted from an empty object to "connected".
    provider_status = str(context.get("provider_status") or card.get("api_context_status") or "").upper()
    if provider_status not in STATUSES:
        provider_status = (
            "CONNECTED" if context.get("provider") == "api-football"
            else "NOT_CONFIGURED" if not context else "UNKNOWN"
        )
    if context.get("fixture_match_status") == "NO_MATCH":
        api_status = "CONNECTED"
        api_note = "API-Football ответил, но точное сопоставление матча не найдено"
    elif context.get("api_fixture_id"):
        api_status = "CONNECTED"
        api_note = "Сопоставлен fixture ID API-Football"
    else:
        api_status = provider_status
        api_note = {
            "NOT_CONFIGURED": "Ключ API-Football не настроен",
            "UNAVAILABLE": "Запрос API-Football не выполнен",
            "CONNECTED": "Источник ответил, сопоставление отсутствует",
        }.get(api_status, "Состояние источника не подтверждено")

    fixture_status = "CONFIRMED" if (
        card.get("kickoff_utc") or (card.get("date_iso") and card.get("time"))
    ) else "NO_DATA"
    fixture_note = "Есть дата и время начала" if fixture_status == "CONFIRMED" else "Нет точного времени начала"

    category = str(card.get("match_category") or "").casefold()
    div = str(card.get("div") or "")
    league = str(card.get("league") or "")
    category_known = bool(
        card.get("category_verified")
        or card.get("women") is True
        or card.get("national") is True
        or div.startswith(("W_", "NT"))
    )
    category_status = "CONFIRMED" if category_known and league else "UNKNOWN"
    category_note = (
        "Категория соревнования размечена"
        if category_status == "CONFIRMED"
        else "Пол/тип соревнования не подтверждён"
    )

    def lineup_status(team: dict, explicit: Any) -> tuple[str, str]:
        observed = explicit or lineup.get("timestamp")
        players = team.get("starting_xi") or []
        confirmed = bool(team.get("confirmed")) and len(players) >= 11
        status = "CONFIRMED" if confirmed else str(lineup.get("status") or "UNKNOWN").upper()
        if status not in STATUSES:
            if lineup.get("request_status") == "NOT_DUE":
                status = "NOT_DUE"
            elif lineup.get("response_received"):
                status = "NO_DATA"
            elif context.get("api_fixture_id"):
                status = "NOT_REQUESTED"
            else:
                status = "UNAVAILABLE" if api_status == "UNAVAILABLE" else "UNKNOWN"
        status = _freshness(status, observed, now, max_age_hours=8)
        return status, (
            f"{len(players)} игроков в стартовом составе"
            if status == "CONFIRMED" else
            "Состав ещё не опубликован" if status in {"NO_DATA", "NOT_DUE"} else
            "Ответ API не подтверждён" if status in {"UNKNOWN", "UNAVAILABLE"} else
            "Нужно обновить данные"
        )

    home_status, home_note = lineup_status(home_lineup, home_lineup.get("timestamp"))
    away_status, away_note = lineup_status(away_lineup, away_lineup.get("timestamp"))

    referee_status = "CONFIRMED" if (
        referee.get("is_confirmed") and str(referee.get("name") or "").strip()
    ) else str(referee.get("status") or "").upper()
    if referee_status not in STATUSES:
        referee_status = (
            "NO_DATA" if context.get("api_fixture_id") or context.get("provider_status") == "CONNECTED"
            else "UNAVAILABLE" if api_status == "UNAVAILABLE"
            else "UNKNOWN"
        )
    referee_status = _freshness(
        referee_status, referee.get("timestamp") or context.get("timestamp"),
        now, max_age_hours=24,
    )
    referee_note = (
        str(referee.get("name")) if referee_status == "CONFIRMED"
        else "API ответил без назначения судьи" if referee_status == "NO_DATA"
        else "Судья пока не подтверждён"
    )

    injuries = context.get("injuries") if isinstance(context.get("injuries"), list) else []
    if context.get("injuries_is_confirmed"):
        injuries_status = "CONFIRMED" if injuries else "NO_DATA"
    else:
        injuries_status = str(context.get("injuries_status") or "").upper()
        if injuries_status not in STATUSES:
            injuries_status = (
                "UNAVAILABLE" if api_status == "UNAVAILABLE"
                else "NOT_CONFIGURED" if api_status == "NOT_CONFIGURED"
                else "UNKNOWN"
            )
    injuries_status = _freshness(
        injuries_status,
        context.get("injuries_timestamp"),
        now,
        max_age_hours=24,
    )
    injuries_note = (
        f"Получено записей: {len(injuries)}" if injuries
        else "Источник ответил без записей; это не доказывает отсутствие травм"
        if injuries_status == "NO_DATA"
        else "Фид травм недоступен" if injuries_status == "UNAVAILABLE"
        else "Нет подтверждения по травмам"
    )

    price_maps = (
        card.get("market_odds"), card.get("total_market_odds"),
        card.get("btts_market_odds"), card.get("stat_market_odds"),
        card.get("asian_handicap_market_odds"),
    )
    odds_present = any(_valid_prices(value) for value in price_maps)
    if odds_present:
        odds_status = "CONFIRMED"
        odds_note = "Найдена хотя бы одна валидная котировка"
    else:
        odds_state = str(card.get("odds_status") or "").upper()
        odds_status = odds_state if odds_state in STATUSES else (
            "NO_DATA" if str(card.get("odds_source") or "").casefold() == "no_market"
            else "UNKNOWN"
        )
        odds_note = "Подтверждённой цены на показанные рынки нет"

    try:
        history_n = int(card.get("games") or 0)
    except (TypeError, ValueError):
        history_n = 0
    history_status = "CONFIRMED" if history_n >= 5 else "NO_DATA"
    history_note = f"История команд: {history_n} матчей" if history_n else "Недостаточно истории команд"

    raw_conflicts = card.get("source_conflicts") or context.get("source_conflicts") or []
    conflicts = [str(item) for item in raw_conflicts if str(item).strip()] if isinstance(raw_conflicts, list) else []
    if conflicts:
        category_status = "CONFLICT" if any("category" in item.casefold() or "gender" in item.casefold() for item in conflicts) else category_status
        if any("kickoff" in item.casefold() or "time" in item.casefold() for item in conflicts):
            fixture_status = "CONFLICT"

    components = [
        {"key": "fixture", "label": "Расписание и время", "status": fixture_status, "note": fixture_note, "source": "schedule"},
        {"key": "category", "label": "Категория соревнования", "status": category_status, "note": category_note, "source": str(card.get("league") or card.get("div") or "schedule")},
        {"key": "api_football", "label": "API-Football", "status": api_status, "note": api_note, "source": "API-Football"},
        {"key": "home_lineup", "label": "Стартовый состав хозяев", "status": home_status, "note": home_note, "source": "API-Football"},
        {"key": "away_lineup", "label": "Стартовый состав гостей", "status": away_status, "note": away_note, "source": "API-Football"},
        {"key": "referee", "label": "Судья", "status": referee_status, "note": referee_note, "source": "API-Football"},
        {"key": "injuries", "label": "Травмы / дисквалификации", "status": injuries_status, "note": injuries_note, "source": "API-Football"},
        {"key": "odds", "label": "Коэффициенты", "status": odds_status, "note": odds_note, "source": str(card.get("odds_source") or "букмекерские фиды")},
        {"key": "history", "label": "Историческая выборка", "status": history_status, "note": history_note, "source": "локальная история модели"},
    ]
    total_weight = 0.0
    earned = 0.0
    for item in components:
        weight = _WEIGHTS[item["key"]]
        value = _STATUS_VALUE.get(item["status"], 0.0)
        if value is None:
            continue
        total_weight += weight
        earned += weight * value
    score = round(100.0 * earned / total_weight) if total_weight else None
    conflict = bool(conflicts) or any(x["status"] == "CONFLICT" for x in components)
    return {
        "score": score,
        "label": "ПОЛНОТА ДАННЫХ" if score is not None else "НЕТ ОЦЕНКИ",
        "conflict": conflict,
        "conflicts": conflicts,
        "components": components,
        "summary": (
            "Обнаружены противоречия между источниками" if conflict
            else "Оценка отражает наличие данных, а не шанс исхода"
        ),
    }
