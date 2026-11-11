"""Transparent data-quality and risk diagnostics; no synthetic scores or auto-actions."""
from __future__ import annotations

import math
from collections import defaultdict


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
            age = (now - observed_at).total_seconds()
            if age > max_age_seconds:
                return "stale"
        except (TypeError, ValueError):
            return "unavailable"
    return "confirmed" if has_data else "empty"


def card_quality(card: dict) -> dict:
    """Evaluate only evidence present on a scanner card; unknown is never treated as OK."""
    card = card if isinstance(card, dict) else {}
    context = card.get("api_context") if isinstance(card.get("api_context"), dict) else {}
    quality = card.get("context_quality") if isinstance(card.get("context_quality"), dict) else {}
    home_xi = context.get("home_lineup") or context.get("home_starting_xi") or []
    away_xi = context.get("away_lineup") or context.get("away_starting_xi") or []
    referee = context.get("referee")
    injuries = context.get("injuries")
    odds = card.get("market_odds") or {}
    total_odds = card.get("total_market_odds") or {}
    btts_odds = card.get("btts_market_odds") or {}
    kickoff = card.get("kickoff_utc") or card.get("date_iso")
    checks = [
        {"name": "API-Football", "state": "confirmed" if context else ("not_configured" if not card.get("api_football_fixture_id") else "empty"),
         "detail": "Контекст получен" if context else "Нет подтверждённого контекста API-Football"},
        {"name": "Состав хозяев", "state": "confirmed" if home_xi else "empty",
         "detail": f"{len(home_xi)} игроков" if home_xi else "Стартовый XI не подтверждён"},
        {"name": "Состав гостей", "state": "confirmed" if away_xi else "empty",
         "detail": f"{len(away_xi)} игроков" if away_xi else "Стартовый XI не подтверждён"},
        {"name": "Судья", "state": "confirmed" if _present(referee) else "empty",
         "detail": str(referee) if _present(referee) else "Нет подтверждения"},
        {"name": "Травмы", "state": "confirmed" if _present(injuries) else "empty",
         "detail": "Есть записи источника" if _present(injuries) else "Нет данных; это не означает отсутствие травм"},
        {"name": "Коэффициенты 1X2", "state": "confirmed" if odds else "empty",
         "detail": f"{len(odds)} исходов" if odds else "Нет реальной линии"},
        {"name": "Коэффициенты тотала", "state": "confirmed" if total_odds else "empty",
         "detail": f"{len(total_odds)} линий" if total_odds else "Нет реальной линии"},
        {"name": "Коэффициенты BTTS", "state": "confirmed" if btts_odds else "empty",
         "detail": f"{len(btts_odds)} исходов" if btts_odds else "Нет реальной линии"},
    ]
    known = [x for x in checks if x["state"] in {"confirmed", "empty"}]
    # Completeness is explicitly a completeness score, not a probability of winning.
    score = round(100 * sum(x["state"] == "confirmed" for x in known) / len(known)) if known else None
    return {"checks": checks, "completeness": score, "context_level": quality.get("level", "Unknown"),
            "context_reason": quality.get("reason", "Нет оценки контекста"),
            "kickoff": kickoff, "league": card.get("league") or card.get("div") or "Неизвестно",
            "match": card.get("match_ru") or card.get("match") or "Матч без названия"}


def portfolio_risk(bets, bank):
    """Concentration and drawdown diagnostics from recorded paper/live positions."""
    valid = [b for b in (bets or []) if isinstance(b, dict)]
    pending = [b for b in valid if b.get("status", "pending") == "pending"]
    def num(v):
        try:
            x = float(v)
            return x if math.isfinite(x) and x >= 0 else 0.0
        except (TypeError, ValueError):
            return 0.0
    exposure = sum(num(b.get("stake")) for b in pending)
    by_fixture, by_league = defaultdict(float), defaultdict(float)
    by_day = defaultdict(float)
    for b in pending:
        fixture = str(b.get("fixture_id") or b.get("match") or b.get("match_ru") or "unknown")
        league = str(b.get("league") or b.get("div") or "unknown")
        day = str(b.get("date_iso") or b.get("date") or "unknown")[:10]
        stake = num(b.get("stake"))
        by_fixture[fixture] += stake
        by_league[league] += stake
        by_day[day] += stake
    initial = num(bank)
    fixture_top = sorted(by_fixture.items(), key=lambda x: x[1], reverse=True)
    league_top = sorted(by_league.items(), key=lambda x: x[1], reverse=True)
    day_top = sorted(by_day.items(), key=lambda x: x[1], reverse=True)
    # Drawdown from settled outcomes in chronological order, using actual recorded stakes.
    settled = sorted((b for b in valid if b.get("status") in {"won", "lost", "push", "void"}),
                     key=lambda b: str(b.get("settled_at") or b.get("date_time") or ""))
    equity = peak = initial
    max_dd = 0.0
    for b in settled:
        stake, odds = num(b.get("stake")), num(b.get("odds"))
        if b.get("status") == "won":
            equity += stake * max(0.0, odds - 1.0)
        elif b.get("status") == "lost":
            equity -= stake
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
    return {"open_count": len(pending), "exposure": exposure,
            "exposure_pct": exposure / initial if initial else None,
            "fixture_top": fixture_top[:5], "league_top": league_top[:5],
            "day_top": day_top[:5], "max_drawdown": max_dd,
            "settled_count": len(settled),
            "warning": bool(initial and exposure / initial >= 0.20)}
