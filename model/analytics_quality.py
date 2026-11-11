"""Transparent data-quality and risk diagnostics; no synthetic scores or auto-actions."""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone


def _present(value):
    return value not in (None, "", [], {}, ())


def source_state(*, configured=False, responded=False, has_data=False, observed_at=None, now=None, max_age_seconds=None, conflict=False):
    """Return one of confirmed/empty/unavailable/stale/conflict/not_configured."""
    if conflict:
        return "conflict"
    if not configured:
        return "not_configured"
    if not responded:
        return "unavailable"
    if observed_at and max_age_seconds is not None and now is not None:
        try:
            if (now - observed_at).total_seconds() > max_age_seconds:
                return "stale"
        except (TypeError, ValueError):
            return "unavailable"
    return "confirmed" if has_data else "empty"


def card_quality(card: dict) -> dict:
    """Evaluate only evidence present on a scanner card; unknown is never treated as OK."""
    card = card if isinstance(card, dict) else {}
    context = card.get("api_context") if isinstance(card.get("api_context"), dict) else {}
    quality = card.get("context_quality") if isinstance(card.get("context_quality"), dict) else {}
    lineup = context.get("lineup") if isinstance(context.get("lineup"), dict) else {}
    home_lineup = lineup.get("home") if isinstance(lineup.get("home"), dict) else {}
    away_lineup = lineup.get("away") if isinstance(lineup.get("away"), dict) else {}
    home_xi = home_lineup.get("starting_xi") or []
    away_xi = away_lineup.get("starting_xi") or []
    referee = context.get("referee") if isinstance(context.get("referee"), dict) else {}
    referee_name = referee.get("name")
    injuries = context.get("injuries")
    injuries_confirmed = bool(context.get("injuries_is_confirmed"))
    odds = card.get("market_odds") or {}
    total_odds = card.get("total_market_odds") or {}
    btts_odds = card.get("btts_market_odds") or {}
    kickoff = card.get("kickoff_utc") or card.get("date_iso")
    context_received = bool(context)
    conflicts = set(card.get("data_conflicts") or [])
    context_state = "confirmed" if context_received else ("empty" if card.get("api_football_fixture_id") else "not_configured")
    observed = context.get("timestamp")
    if context_received and observed:
        try:
            stamp = datetime.fromisoformat(str(observed).replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
            if (datetime.now(timezone.utc) - stamp.astimezone(timezone.utc)).total_seconds() > 6 * 3600:
                context_state = "stale"
        except (TypeError, ValueError, OverflowError):
            context_state = "unavailable"
    checks = [
        {"name": "API-Football", "state": "conflict" if "api_football" in conflicts else context_state,
         "detail": "Контекст получен" if context_received else "Нет подтверждённого контекста API-Football"},
        {"name": "Состав хозяев", "state": "confirmed" if home_lineup.get("confirmed") and len(home_xi) >= 11 else ("empty" if context_received else "unavailable"),
         "detail": f"{len(home_xi)} игроков" if home_xi else "Стартовый XI не подтверждён"},
        {"name": "Состав гостей", "state": "confirmed" if away_lineup.get("confirmed") and len(away_xi) >= 11 else ("empty" if context_received else "unavailable"),
         "detail": f"{len(away_xi)} игроков" if away_xi else "Стартовый XI не подтверждён"},
        {"name": "Судья", "state": "confirmed" if _present(referee_name) else ("empty" if context_received else "unavailable"),
         "detail": str(referee_name) if _present(referee_name) else "Нет подтверждения"},
        {"name": "Травмы", "state": ("confirmed" if injuries else "empty") if injuries_confirmed else "unavailable",
         "detail": "Есть записи источника" if injuries else ("Источник проверен: записей нет" if injuries_confirmed else "Источник не подтвердил покрытие; отсутствие записей не доказывает отсутствие травм")},
        {"name": "Коэффициенты 1X2", "state": "conflict" if "1x2_odds" in conflicts else ("confirmed" if odds else "empty"),
         "detail": f"{len(odds)} исходов" if odds else "Нет реальной линии"},
        {"name": "Коэффициенты тотала", "state": "conflict" if "totals_odds" in conflicts else ("confirmed" if total_odds else "empty"),
         "detail": f"{len(total_odds)} линий" if total_odds else "Нет реальной линии"},
        {"name": "Коэффициенты BTTS", "state": "conflict" if "btts_odds" in conflicts else ("confirmed" if btts_odds else "empty"),
         "detail": f"{len(btts_odds)} исходов" if btts_odds else "Нет реальной линии"},
    ]
    # Missing, unavailable and unconfigured fields remain in the denominator.
    # This is evidence coverage, never a probability of winning.
    score = round(100 * sum(x["state"] == "confirmed" for x in checks) / len(checks)) if checks else None
    return {"checks": checks, "completeness": score, "context_level": quality.get("level", "Unknown"),
            "context_reason": quality.get("reason", "Нет оценки контекста"),
            "kickoff": kickoff, "league": card.get("league") or card.get("div") or "Неизвестно",
            "match": card.get("match_ru") or card.get("match") or "Матч без названия"}


def portfolio_risk(bets, bank):
    """Concentration and drawdown diagnostics from recorded paper/live positions."""
    valid = [b for b in (bets or []) if isinstance(b, dict)]
    pending = [b for b in valid if b.get("status", "pending") == "pending"]

    def num(value):
        try:
            number = float(value)
            return number if math.isfinite(number) and number >= 0 else 0.0
        except (TypeError, ValueError):
            return 0.0

    exposure = sum(num(b.get("stake")) for b in pending)
    by_fixture, by_league, by_day = defaultdict(float), defaultdict(float), defaultdict(float)
    for bet in pending:
        fixture = str(bet.get("fixture_id") or bet.get("match") or bet.get("match_ru") or "unknown")
        league = str(bet.get("league") or bet.get("div") or "unknown")
        day = str(bet.get("date_iso") or bet.get("date") or "unknown")[:10]
        stake = num(bet.get("stake"))
        by_fixture[fixture] += stake
        by_league[league] += stake
        by_day[day] += stake
    initial = num(bank)
    settled = sorted((b for b in valid if b.get("status") in {"won", "lost", "push", "void"}),
                     key=lambda b: str(b.get("settled_at") or b.get("date_time") or ""))
    equity = peak = initial
    max_dd = 0.0
    for bet in settled:
        stake, odds = num(bet.get("stake")), num(bet.get("odds"))
        if bet.get("status") == "won":
            equity += stake * max(0.0, odds - 1.0)
        elif bet.get("status") == "lost":
            equity -= stake
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
    return {"open_count": len(pending), "exposure": exposure,
            "exposure_pct": exposure / initial if initial else None,
            "fixture_top": sorted(by_fixture.items(), key=lambda x: x[1], reverse=True)[:5],
            "league_top": sorted(by_league.items(), key=lambda x: x[1], reverse=True)[:5],
            "day_top": sorted(by_day.items(), key=lambda x: x[1], reverse=True)[:5],
            "max_drawdown": max_dd, "settled_count": len(settled),
            "warning": bool(initial and exposure / initial >= 0.20)}
