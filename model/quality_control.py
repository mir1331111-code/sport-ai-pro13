"""Deterministic data-quality and portfolio-risk calculations.

Scores in this module measure data coverage/completeness, never match probability
or the truthfulness of a prediction. All functions are side-effect free.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping

from data.competition import competition_gender
from data.team_names import normalize_team


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
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _match_teams(card: Mapping[str, Any]) -> tuple[str, str]:
    match = str(card.get("match") or "")
    if " vs " in match:
        home, away = match.split(" vs ", 1)
        return home.strip(), away.strip()
    match_ru = str(card.get("match_ru") or "")
    if " — " in match_ru:
        home, away = match_ru.split(" — ", 1)
        return home.strip(), away.strip()
    return "", ""


def _valid_price_map(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    for raw in value.values():
        try:
            price = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(price) and 1.01 < price <= 1000:
            return True
    return False


def _age_status(timestamp: Any, now: datetime, max_age_hours: float) -> str:
    parsed = _parse_time(timestamp)
    if parsed is None:
        return "missing"
    age_seconds = (now - parsed).total_seconds()
    if age_seconds < -300:
        return "future_timestamp"
    if age_seconds > max_age_hours * 3600:
        return "stale"
    return "fresh"


def assess_fixture_quality(
    card: Mapping[str, Any],
    *,
    api_configured: bool = False,
    now: datetime | None = None,
    source_errors: Iterable[str] = (),
    stale_after_hours: float = 24.0,
) -> dict[str, Any]:
    """Assess one card and explain all source states independently.

    States distinguish confirmed data, provider responses without records,
    unavailable/unmatched sources, missing data, stale data and conflicts.
    The numeric score is a weighted coverage index out of 100.
    """
    if not isinstance(card, Mapping):
        card = {}
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)
    errors = " ".join(str(item) for item in source_errors).casefold()

    context = card.get("api_context") if isinstance(card.get("api_context"), Mapping) else {}
    provider_present = (
        str(context.get("provider") or "").casefold() == "api-football"
        and context.get("api_fixture_id") not in (None, "")
    )
    api_unavailable = any(
        phrase in errors for phrase in (
            "fixtures недоступен", "api-football request failed",
            "список fixtures недоступен", "http 401", "http 403",
        )
    )
    if provider_present:
        api_state = "returned"
    elif not api_configured:
        api_state = "not_configured"
    elif api_unavailable:
        api_state = "unavailable"
    else:
        api_state = "not_matched"

    kickoff = _parse_time(card.get("kickoff_utc"))
    if kickoff is None:
        date_iso = str(card.get("date_iso") or "")[:10]
        time_text = str(card.get("time") or "")
        if date_iso and re.search(r"\b\d{1,2}:\d{2}\b", time_text):
            match = re.search(r"\b(\d{1,2}):(\d{2})\b", time_text)
            try:
                kickoff = datetime.fromisoformat(
                    f"{date_iso}T{int(match.group(1)):02d}:{match.group(2)}:00"
                ).replace(tzinfo=timezone.utc)
            except (AttributeError, TypeError, ValueError):
                kickoff = None
    kickoff_state = "known" if kickoff is not None else "missing"

    lineup = context.get("lineup") if isinstance(context.get("lineup"), Mapping) else {}
    lineup_stamp = lineup.get("timestamp")
    lineup_age = _age_status(lineup_stamp, current, stale_after_hours)
    home_lineup = lineup.get("home") if isinstance(lineup.get("home"), Mapping) else {}
    away_lineup = lineup.get("away") if isinstance(lineup.get("away"), Mapping) else {}

    def lineup_state(item: Mapping[str, Any]) -> str:
        players = item.get("starting_xi") if isinstance(item.get("starting_xi"), list) else []
        if item.get("confirmed") and len(players) >= 11:
            return "confirmed" if lineup_age != "stale" else "stale"
        if lineup_age == "stale":
            return "stale"
        if lineup_stamp:
            return "responded_partial" if players else "responded_empty"
        if kickoff is not None and kickoff - current > timedelta(hours=6):
            return "not_requested_yet"
        return "missing" if provider_present else (
            "not_configured" if not api_configured else "unavailable"
        )

    home_state = lineup_state(home_lineup)
    away_state = lineup_state(away_lineup)

    referee = context.get("referee") if isinstance(context.get("referee"), Mapping) else {}
    referee_state = "reported" if str(referee.get("name") or "").strip() else (
        "responded_empty" if provider_present else (
            "not_configured" if not api_configured else "unavailable"
        )
    )
    referee_age = _age_status(referee.get("timestamp") or context.get("timestamp"), current, stale_after_hours)
    if referee_state == "reported" and referee_age == "stale":
        referee_state = "stale"

    injuries = context.get("injuries") if isinstance(context.get("injuries"), list) else []
    injury_stamp = context.get("injuries_timestamp")
    if context.get("injuries_is_confirmed"):
        injury_age = _age_status(injury_stamp, current, stale_after_hours)
        if injury_age == "stale":
            injury_state = "stale"
        else:
            injury_state = "records_returned" if injuries else "responded_empty"
    else:
        injury_state = (
            "unavailable" if provider_present else
            "not_configured" if not api_configured else "unknown"
        )

    odds_maps = (
        card.get("market_odds"), card.get("total_market_odds"),
        card.get("btts_market_odds"), card.get("stat_market_odds"),
        card.get("asian_handicap_market_odds"),
    )
    odds_present = any(_valid_price_map(value) for value in odds_maps)
    odds_state = "received" if odds_present else "missing"

    home_name, away_name = _match_teams(card)
    lineup_home_name = str(home_lineup.get("team_name") or "").strip()
    lineup_away_name = str(away_lineup.get("team_name") or "").strip()
    conflicts = []
    if lineup_home_name and home_name and normalize_team(lineup_home_name) != normalize_team(home_name):
        conflicts.append("название хозяев в составе не совпадает с карточкой матча")
    if lineup_away_name and away_name and normalize_team(lineup_away_name) != normalize_team(away_name):
        conflicts.append("название гостей в составе не совпадает с карточкой матча")

    gender = competition_gender(
        division=card.get("div"), league=card.get("league"),
        women_flag=card.get("women"), home=home_name, away=away_name,
    )
    # A women's marker in competition text that did not propagate to the card
    # is a classification conflict, not a reason to silently train as men's.
    explicit_women = competition_gender(
        league=card.get("league"), home=home_name, away=away_name
    ) == "women"
    if explicit_women and gender != "women":
        conflicts.append("женская категория соревнования не перенесена в карточку")

    freshness_state = "unknown"
    time_stamps = [
        context.get("timestamp"), lineup_stamp, referee.get("timestamp"),
        injury_stamp,
    ]
    present_stamps = [stamp for stamp in time_stamps if stamp]
    if present_stamps:
        freshness_state = (
            "stale"
            if any(_age_status(stamp, current, stale_after_hours) == "stale" for stamp in present_stamps)
            else "fresh"
        )
    if conflicts:
        conflict_state = "conflict"
    else:
        conflict_state = "none"

    # Weights represent completeness only, not empirical source reliability.
    weights = {
        "api": 10, "home_lineup": 20, "away_lineup": 20, "referee": 10,
        "injuries": 10, "odds": 15, "kickoff": 10, "freshness": 5,
    }
    points = {
        "api": 10 if api_state == "returned" else 0,
        "home_lineup": 20 if home_state == "confirmed" else 8 if home_state == "responded_partial" else 0,
        "away_lineup": 20 if away_state == "confirmed" else 8 if away_state == "responded_partial" else 0,
        "referee": 10 if referee_state == "reported" else 0,
        "injuries": 10 if injury_state in {"records_returned", "responded_empty"} else 0,
        "odds": 15 if odds_state == "received" else 0,
        "kickoff": 10 if kickoff_state == "known" else 0,
        "freshness": 5 if freshness_state == "fresh" else 0,
    }
    score = max(0, min(100, int(round(sum(points.values())))))
    if conflicts:
        score = min(score, 40)
    level = "Высокая полнота" if score >= 80 else "Средняя полнота" if score >= 55 else "Низкая полнота"

    statuses = {
        "API-Football": api_state,
        "Состав хозяев": home_state,
        "Состав гостей": away_state,
        "Судья": referee_state,
        "Травмы": injury_state,
        "Коэффициенты": odds_state,
        "Время матча": kickoff_state,
        "Свежесть данных": freshness_state,
        "Противоречия": conflict_state,
    }
    notes = list(conflicts)
    if injury_state == "responded_empty":
        notes.append("Источник травм ответил без записей; это не доказывает отсутствие травм.")
    if injury_state == "unknown":
        notes.append("Статус фида травм неизвестен; пустой список не считается подтверждением.")
    if odds_state == "missing":
        notes.append("Нет доступной котировки для отображаемых рынков.")
    notes.append("Индекс отражает полноту данных, а не вероятность исхода и не надёжность прогноза.")
    return {
        "score": score, "level": level, "gender": gender,
        "statuses": statuses, "notes": notes,
        "weights": weights, "points": points,
        "conflicts": conflicts, "api_fixture_matched": provider_present,
    }


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def check_proposed_position(
    bets: Iterable[Mapping[str, Any]] | None,
    *,
    bank: float,
    candidate: Mapping[str, Any],
    max_match_pct: float = 0.05,
    max_league_pct: float = 0.12,
    max_day_pct: float = 0.20,
    max_loss_streak: int = 3,
) -> dict[str, Any]:
    """Check exposure caps and prohibit raising stake after a losing streak."""
    all_rows = [dict(row) for row in (bets or []) if isinstance(row, Mapping)]
    pending = [row for row in all_rows if row.get("status", "pending") == "pending"]
    stake = _finite_float(candidate.get("stake"))
    if stake <= 0:
        return {"allowed": False, "reasons": ["Размер позиции должен быть положительным."], "exposures": {}}
    current_pending = sum(_finite_float(row.get("stake")) for row in pending)
    equity = max(0.0, _finite_float(bank) + current_pending)
    if equity <= 0:
        return {"allowed": False, "reasons": ["Банк и свободная экспозиция равны нулю."], "exposures": {}}

    match = str(candidate.get("match") or "").strip().casefold()
    date_iso = str(candidate.get("date_iso") or candidate.get("date") or "")[:10]
    league = str(candidate.get("league") or candidate.get("div") or "—").strip().casefold()

    def exposure(rows: list[Mapping[str, Any]], key_fn) -> float:
        total = sum(
            _finite_float(row.get("stake"))
            for row in rows if key_fn(row)
        )
        return total / equity

    match_amount = sum(
        _finite_float(row.get("stake")) for row in pending
        if str(row.get("match") or "").strip().casefold() == match
        and match
    ) + stake
    league_amount = sum(
        _finite_float(row.get("stake")) for row in pending
        if str(row.get("league") or row.get("div") or "—").strip().casefold() == league
    ) + stake
    day_amount = sum(
        _finite_float(row.get("stake")) for row in pending
        if str(row.get("date_iso") or "")[:10] == date_iso and date_iso
    ) + stake
    exposures = {
        "match": match_amount / equity,
        "league": league_amount / equity,
        "day": day_amount / equity,
    }
    reasons = []
    closed = [row for row in all_rows if row.get("status") in {"won", "lost"}]
    closed.sort(key=lambda row: str(row.get("settled_at") or row.get("date_time") or ""))
    loss_streak = 0
    for row in reversed(closed):
        if row.get("status") != "lost":
            break
        loss_streak += 1
    reference_stake = _finite_float(closed[-1].get("stake")) if closed else 0.0
    if (
        max_loss_streak > 0 and loss_streak >= max_loss_streak
        and reference_stake > 0 and stake > reference_stake + 1e-9
    ):
        reasons.append(
            f"После серии из {loss_streak} поражений размер позиции не может "
            f"превышать последнюю ставку ({reference_stake:.2f})."
        )
    if exposures["match"] > max_match_pct + 1e-12:
        reasons.append(f"Матч: {exposures['match']:.1%} > лимита {max_match_pct:.1%}.")
    if exposures["league"] > max_league_pct + 1e-12:
        reasons.append(f"Лига: {exposures['league']:.1%} > лимита {max_league_pct:.1%}.")
    if exposures["day"] > max_day_pct + 1e-12:
        reasons.append(f"День: {exposures['day']:.1%} > лимита {max_day_pct:.1%}.")
    return {
        "allowed": not reasons, "reasons": reasons, "exposures": exposures,
        "equity": equity, "loss_streak": loss_streak,
        "reference_stake": reference_stake,
    }


def portfolio_risk_report(
    bets: Iterable[Mapping[str, Any]] | None,
    *,
    bank: float,
    bank_history: Iterable[Mapping[str, Any]] | None = None,
    max_match_pct: float = 0.05,
    max_league_pct: float = 0.12,
    max_day_pct: float = 0.20,
) -> dict[str, Any]:
    """Summarize open-position concentration and peak-to-trough bank drawdown."""
    all_bets = [dict(row) for row in (bets or []) if isinstance(row, Mapping)]
    pending = [row for row in all_bets if row.get("status", "pending") == "pending"]
    pending_stake = sum(_finite_float(row.get("stake")) for row in pending)
    equity = max(0.0, _finite_float(bank) + pending_stake)

    def group_amount(field: str, fallback: str = "—") -> list[dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        for row in pending:
            key = str(row.get(field) or (row.get(fallback) if fallback else "") or "—").strip()
            item = groups.setdefault(key, {"key": key, "stake": 0.0, "positions": 0, "matches": set()})
            item["stake"] += _finite_float(row.get("stake"))
            item["positions"] += 1
            item["matches"].add((str(row.get("match") or ""), str(row.get("date_iso") or "")[:10]))
        result = []
        for item in groups.values():
            result.append({
                "key": item["key"], "stake": item["stake"],
                "positions": item["positions"], "match_count": len(item["matches"]),
                "share": item["stake"] / equity if equity else None,
            })
        return sorted(result, key=lambda item: item["stake"], reverse=True)

    by_match = group_amount("match", "match")
    by_league = group_amount("league", "div")
    by_day = group_amount("date_iso", "date_iso")
    fixture_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pending:
        fixture_groups[(str(row.get("match") or ""), str(row.get("date_iso") or "")[:10])].append(row)
    correlated_clusters = [
        {"match": key[0], "date": key[1], "positions": len(items),
         "markets": sorted({str(item.get("market") or "—") for item in items}),
         "stake": sum(_finite_float(item.get("stake")) for item in items)}
        for key, items in fixture_groups.items() if len(items) > 1
    ]

    # Preserve original time order when timestamps are available; don't sort by bank value.
    history_rows = [row for row in (bank_history or []) if isinstance(row, Mapping)]
    history_rows.sort(key=lambda row: str(row.get("ts") or ""))
    values = [_finite_float(row.get("bank")) for row in history_rows if row.get("bank") is not None]
    closed = [row for row in all_bets if row.get("status") in {"won", "lost"}]
    closed.sort(key=lambda row: str(row.get("settled_at") or row.get("date_time") or ""))
    loss_streak = 0
    for row in reversed(closed):
        if row.get("status") != "lost":
            break
        loss_streak += 1
    peak = max(values, default=_finite_float(bank))
    drawdown = 0.0
    running_peak = values[0] if values else _finite_float(bank)
    for value in values:
        running_peak = max(running_peak, value)
        if running_peak > 0:
            drawdown = max(drawdown, (running_peak - value) / running_peak)
    open_risk = pending_stake / equity if equity else 0.0
    breaches = []
    for label, items, cap in (
        ("Матч", by_match, max_match_pct),
        ("Лига", by_league, max_league_pct),
        ("День", by_day, max_day_pct),
    ):
        for item in items:
            if item["share"] is not None and item["share"] > cap:
                breaches.append(f"{label}: {item['key']} — {item['share']:.1%} выше лимита {cap:.1%}")
    return {
        "pending_count": len(pending), "pending_stake": pending_stake,
        "equity": equity, "open_risk_share": open_risk,
        "by_match": by_match, "by_league": by_league, "by_day": by_day,
        "correlated_clusters": correlated_clusters, "max_drawdown": drawdown,
        "loss_streak": loss_streak,
        "last_settled_stake": _finite_float(closed[-1].get("stake")) if closed else None,
        "breaches": breaches,
        "limits": {
            "match": max_match_pct, "league": max_league_pct, "day": max_day_pct,
        },
    }
