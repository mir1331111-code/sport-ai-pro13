"""Per-fixture data provenance, freshness and completeness diagnostics.

The completeness score measures how much usable information is present. It is
not a probability that a forecast will be correct and is never used as a bet
approval signal.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

STATUS_LABELS = {
    "confirmed": "Подтверждено",
    "empty": "Источник ответил, данных нет",
    "unavailable": "Источник недоступен",
    "stale": "Устарело",
    "partial": "Частично подтверждено",
    "not_due": "Пока не время публикации",
    "deferred": "Отложено лимитом запросов",
    "not_configured": "Ключ не настроен",
    "unmatched": "Матч не сопоставлен",
    "unknown": "Неизвестно",
    "not_applicable": "Не применяется",
}


def competition_category_known(row: dict[str, Any]) -> bool:
    """Require a specific competition/category instead of a generic Soccer label."""
    if not isinstance(row, dict):
        return False
    division = str(row.get("Div") or row.get("div") or "").strip()
    league = str(row.get("League") or row.get("league") or "").strip().casefold()
    if bool(row.get("women")) or bool(row.get("national")):
        return True
    if division.upper().startswith(("W_", "NT_")):
        return True
    if row.get("gender_known") is True:
        return True
    if row.get("gender_known") is False:
        return False
    generic_leagues = {"", "soccer", "football", "матч", "лига", "unknown", "other"}
    if division and division.upper() not in {"G", "GENERAL", "UNKNOWN"}:
        return True
    return league not in generic_leagues


def _aware(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if result.tzinfo is None or result.utcoffset() is None:
        return None
    return result.astimezone(timezone.utc)


def _fresh(status: str, timestamp: Any, *, max_age_hours: float,
           now: datetime) -> str:
    if status not in {"confirmed", "empty", "partial"}:
        return status
    stamp = _aware(timestamp)
    if stamp is None:
        return "unknown"
    age_hours = (now - stamp).total_seconds() / 3600
    if age_hours > max_age_hours:
        return "stale"
    return status


def _valid_quotes(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    for raw in value.values():
        try:
            price = float(raw)
        except (TypeError, ValueError):
            continue
        if 1.01 < price <= 100:
            return True
    return False


def evaluate_fixture_quality(
    card: dict, *, api_configured: bool | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return transparent source states and a 0-100 completeness score."""
    if not isinstance(card, dict):
        card = {}
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    now = now.astimezone(timezone.utc)

    context = card.get("api_context") or {}
    if api_configured is None:
        api_present = card.get("api_football_configured")
    else:
        api_present = bool(api_configured)
    api_raw = str(context.get("api_status") or "").casefold()
    if api_raw in {"unavailable", "error"}:
        api_status = "unavailable"
    elif api_raw in {"no_match", "not_matched", "unmatched"}:
        api_status = "unmatched"
    elif api_raw in {"matched", "confirmed"} or context.get("api_fixture_id"):
        api_status = "confirmed"
    elif api_present is False:
        api_status = "not_configured"
    else:
        # Old cards without provenance cannot prove that the key was absent.
        api_status = "unknown"

    lineup = context.get("lineup") or {}
    home_lineup = lineup.get("home") or {}
    away_lineup = lineup.get("away") or {}
    lineup_raw = str(lineup.get("status") or "").casefold()
    home_xi = bool(home_lineup.get("confirmed"))
    away_xi = bool(away_lineup.get("confirmed"))
    if home_xi and away_xi:
        home_status = away_status = "confirmed"
        lineup_status = "confirmed"
    else:
        fallback = {
            "unavailable": "unavailable",
            "error": "unavailable",
            "empty": "empty",
            "not_due": "not_due",
            "deferred": "deferred",
            "partial": "partial",
        }.get(lineup_raw, "unknown")
        home_status = "confirmed" if home_xi else fallback
        away_status = "confirmed" if away_xi else fallback
        lineup_status = (
            "partial" if home_xi or away_xi
            else fallback
        )
    home_status = _fresh(
        home_status, lineup.get("timestamp"),
        max_age_hours=6, now=now,
    )
    away_status = _fresh(
        away_status, lineup.get("timestamp"),
        max_age_hours=3, now=now,
    )
    if home_status == away_status == "confirmed":
        lineup_status = "confirmed"
    elif "confirmed" in {home_status, away_status}:
        lineup_status = "partial"
    elif home_status == "stale" or away_status == "stale":
        lineup_status = "stale"

    referee = context.get("referee") or {}
    referee_raw = str(referee.get("status") or "").casefold()
    if referee.get("is_confirmed") and referee.get("name"):
        referee_status = "confirmed"
    elif referee_raw in {"unavailable", "error"}:
        referee_status = "unavailable"
    elif referee_raw in {"empty", "no_data"}:
        referee_status = "empty"
    else:
        referee_status = "unknown"
    referee_status = _fresh(
        referee_status,
        referee.get("timestamp") or context.get("timestamp"),
        max_age_hours=24,
        now=now,
    )

    injuries = context.get("injuries") or []
    injury_raw = str(context.get("injuries_status") or "").casefold()
    if context.get("injuries_is_confirmed"):
        injury_status = "confirmed" if injuries else "empty"
    elif injury_raw in {"unavailable", "error"}:
        injury_status = "unavailable"
    else:
        injury_status = "unknown"
    injury_status = _fresh(
        injury_status,
        context.get("injuries_timestamp"),
        max_age_hours=24,
        now=now,
    )

    odds_groups = (
        card.get("market_odds"),
        card.get("total_market_odds"),
        card.get("btts_market_odds"),
        card.get("stat_market_odds"),
        card.get("asian_handicap_market_odds"),
    )
    has_odds = any(_valid_quotes(group) for group in odds_groups)
    odds_source = str(card.get("odds_source") or "").casefold()
    if has_odds:
        odds_status = "confirmed"
    elif odds_source in {"no_market", "model_only", "none"}:
        odds_status = "empty"
    elif not odds_source:
        odds_status = "unknown"
    else:
        odds_status = "empty"

    sample = card.get("model_sample") or {}
    try:
        home_n = max(0, int(sample.get("home_home", 0) or 0))
        away_n = max(0, int(sample.get("away_away", 0) or 0))
    except (TypeError, ValueError):
        home_n = away_n = 0
    if home_n >= 5 and away_n >= 5:
        history_status = "confirmed"
    elif home_n > 0 or away_n > 0:
        history_status = "partial"
    else:
        history_status = "empty"

    kickoff_ok = bool(
        str(card.get("kickoff_utc") or "").strip()
        or (
            str(card.get("date_iso") or "").strip()
            and str(card.get("time") or "").strip()
        )
    )
    league = str(card.get("league") or "").strip()
    div = str(card.get("div") or "").strip()
    generic_league = league.casefold() in {
        "", "soccer", "football", "матч", "лига", "unknown", "other",
    }
    competition_status = "confirmed" if competition_category_known(card) else "unknown"

    # Transparent completeness points. Unavailable/missing data scores zero;
    # a successful empty endpoint is reported separately and is not called "OK".
    score = 0.0
    if kickoff_ok:
        score += 10.0
    if competition_status == "confirmed":
        score += 10.0
    score += min(home_n, 10) + min(away_n, 10)
    if odds_status == "confirmed":
        score += 20.0
    if api_status == "confirmed":
        score += 10.0
    if home_status == "confirmed":
        score += 10.0
    if away_status == "confirmed":
        score += 10.0
    if referee_status == "confirmed":
        score += 5.0
    if injury_status == "confirmed":
        score += 5.0
    score = max(0, min(100, round(score)))

    schedule_providers = [
        str(item) for item in (card.get("fixture_sources") or [])
        if str(item or "").strip()
    ]
    sources = {
        "Источники расписания": {
            "status": "confirmed" if schedule_providers else "unknown",
            "detail": ", ".join(schedule_providers) if schedule_providers
            else "Происхождение строки расписания не записано",
        },
        "API-Football": {
            "status": api_status,
            "detail": (
                "Событие сопоставлено с fixture provider"
                if api_status == "confirmed"
                else "Ключ не задан" if api_status == "not_configured"
                else "Ответ API не сопоставлен с матчем"
                if api_status == "unmatched"
                else "Нет подтверждения ответа fixtures"
            ),
        },
        "Состав хозяев": {
            "status": home_status,
            "detail": f"{len(home_lineup.get('starting_xi') or [])} игроков в стартовом XI",
        },
        "Состав гостей": {
            "status": away_status,
            "detail": f"{len(away_lineup.get('starting_xi') or [])} игроков в стартовом XI",
        },
        "Судья": {
            "status": referee_status,
            "detail": str(referee.get("name") or "Имя судьи не получено"),
        },
        "Травмы / дисквалификации": {
            "status": injury_status,
            "detail": (
                f"Записей: {len(injuries)}" if injuries
                else "Ответ без записей; это не доказывает отсутствие травм"
            ),
        },
        "Коэффициенты": {
            "status": odds_status,
            "detail": (
                str(card.get("odds_source") or card.get("total_odds_source") or "Источник не указан")
                if has_odds else "Нет проверенной котировки для доступных рынков"
            ),
        },
        "История команд": {
            "status": history_status,
            "detail": f"Домашняя выборка: {home_n}; выездная выборка: {away_n}",
        },
        "Время начала": {
            "status": "confirmed" if kickoff_ok else "unknown",
            "detail": str(card.get("date") or card.get("date_iso") or "Время не подтверждено"),
        },
        "Турнир / категория": {
            "status": competition_status,
            "detail": (
                f"{league or 'Турнир неизвестен'} · {div or 'G'}"
                + (" · женский футбол" if card.get("women") or div.startswith("W_") else "")
                + (" · сборные" if card.get("national") else "")
            ),
        },
    }
    return {
        "score": score,
        "sources": sources,
        "history_sample": {"home_home": home_n, "away_away": away_n},
        "api_status": api_status,
        "lineup_status": lineup_status,
        "odds_status": odds_status,
        "history_status": history_status,
        "women": bool(card.get("women") or div.startswith("W_")),
        "national": bool(card.get("national")),
    }


def summarize_quality(cards: list[dict], *, api_configured: bool | None = None,
                       now: datetime | None = None) -> dict[str, Any]:
    results = [
        evaluate_fixture_quality(card, api_configured=api_configured, now=now)
        for card in cards or [] if isinstance(card, dict)
    ]
    if not results:
        return {"fixtures": 0, "mean_score": None, "confirmed_odds": 0,
                "unmatched_api": 0, "women": 0}
    return {
        "fixtures": len(results),
        "mean_score": sum(row["score"] for row in results) / len(results),
        "confirmed_odds": sum(row["odds_status"] == "confirmed" for row in results),
        "unmatched_api": sum(row["api_status"] == "unmatched" for row in results),
        "women": sum(row["women"] for row in results),
        "low_history": sum(row["history_status"] != "confirmed" for row in results),
    }
