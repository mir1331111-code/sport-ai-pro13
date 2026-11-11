"""Centralised conservative stake sizing and portfolio exposure limits."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from betting.kelly import kelly, market_type

KELLY_FRACTION = 0.25
MIN_BET_ODDS = 1.70
MAX_MARKET_STAKE_PCT = 0.01
MAX_MATCH_EXPOSURE_PCT = 0.02
MAX_DAILY_EXPOSURE_PCT = 0.05


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number == number and abs(number) != float("inf") else default


def _pending(bet: object) -> bool:
    return isinstance(bet, dict) and str(bet.get("status") or "pending").casefold() == "pending"


def _fixture_key(item: dict[str, Any]) -> str:
    fixture_id = str(item.get("fixture_id") or "").strip()
    if fixture_id:
        return "id:" + fixture_id
    match = str(item.get("match") or item.get("match_ru") or "").casefold()
    normalized = " ".join(match.split())
    kickoff_date = str(item.get("date_iso") or "")[:10]
    return f"match:{kickoff_date}:{normalized}" if normalized else ""


def _market_family(market: object, pick: object) -> str:
    pick_text = str(pick or "")
    inferred = market_type(pick_text)
    if inferred != "OTHER":
        return inferred
    return str(market or "OTHER").strip().upper() or "OTHER"


def _placed_day(bet: dict[str, Any], fallback_day: date) -> date:
    raw = bet.get("date_time") or bet.get("placed_at") or bet.get("created_at")
    if raw:
        try:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).date()
        except (TypeError, ValueError):
            try:
                return date.fromisoformat(str(raw)[:10])
            except (TypeError, ValueError):
                pass
    # Legacy open bets without a placement timestamp are conservatively
    # counted against today's cap rather than silently ignored.
    return fallback_day


def risk_snapshot(
    bank: object,
    bets: object,
    *,
    now: datetime | None = None,
) -> dict[str, float]:
    """Summarise pending exposure using current free balance plus open stakes."""
    current = now or datetime.now()
    all_rows = [b for b in bets if isinstance(b, dict)] if isinstance(bets, list) else []
    rows = [b for b in all_rows if _pending(b)]
    pending = sum(_safe_float(b.get("stake")) for b in rows)
    bankroll = max(0.0, _safe_float(bank) + pending)
    # Daily risk is cumulative turnover at risk, not just still-open exposure.
    daily = sum(
        _safe_float(b.get("stake"))
        for b in all_rows
        if _placed_day(b, current.date() if _pending(b) else date.min) == current.date()
    )
    match_exposures: dict[str, float] = {}
    for bet in rows:
        key = _fixture_key(bet)
        if key:
            match_exposures[key] = match_exposures.get(key, 0.0) + _safe_float(bet.get("stake"))
    max_match = max(match_exposures.values(), default=0.0)
    return {
        "bankroll_basis": bankroll,
        "pending_exposure": pending,
        "daily_exposure": daily,
        "max_match_exposure": max_match,
        "max_match_exposure_pct": max_match / bankroll if bankroll else 0.0,
        "pending_exposure_pct": pending / bankroll if bankroll else 0.0,
        "daily_exposure_pct": daily / bankroll if bankroll else 0.0,
        "daily_limit_amount": bankroll * MAX_DAILY_EXPOSURE_PCT,
        "daily_remaining": max(0.0, bankroll * MAX_DAILY_EXPOSURE_PCT - daily),
    }


def size_bet(
    *,
    probability: object,
    odds: object,
    bank: object,
    bets: object,
    fixture_id: object,
    match: object,
    date_iso: object,
    market: object,
    pick: object,
    now: datetime | None = None,
) -> tuple[float, dict[str, float | str]]:
    """Return a quarter-Kelly stake bounded by market, match, day and cash limits."""
    current = now or datetime.now()
    probability_value = _safe_float(probability, -1.0)
    odds_value = _safe_float(odds, -1.0)
    free_bank = max(0.0, _safe_float(bank))
    candidate = {
        "fixture_id": str(fixture_id or ""),
        "match": str(match or ""),
        "date_iso": str(date_iso or "")[:10],
        "market": str(market or ""),
        "pick": str(pick or ""),
    }
    family = _market_family(market, pick)
    all_rows = [b for b in bets if isinstance(b, dict)] if isinstance(bets, list) else []
    rows = [b for b in all_rows if _pending(b)]
    pending = sum(_safe_float(b.get("stake")) for b in rows)
    bankroll = free_bank + pending

    diagnostics: dict[str, float | str] = {
        "bankroll_basis": bankroll,
        "market_used": 0.0,
        "match_used": 0.0,
        "daily_used": 0.0,
        "market_limit": bankroll * MAX_MARKET_STAKE_PCT,
        "match_limit": bankroll * MAX_MATCH_EXPOSURE_PCT,
        "daily_limit": bankroll * MAX_DAILY_EXPOSURE_PCT,
        "reason": "",
    }
    if not 0.0 < probability_value < 1.0 or odds_value <= 1.0 or free_bank <= 0.0:
        diagnostics["reason"] = "invalid_probability_odds_or_bank"
        return 0.0, diagnostics
    if odds_value < MIN_BET_ODDS:
        diagnostics["reason"] = "odds_below_minimum_1_70"
        return 0.0, diagnostics
    if bankroll <= 0.0:
        diagnostics["reason"] = "empty_bankroll"
        return 0.0, diagnostics

    candidate_key = _fixture_key(candidate)
    market_used = 0.0
    match_used = 0.0
    daily_used = 0.0
    for bet in rows:
        stake = _safe_float(bet.get("stake"))
        same_fixture = bool(candidate_key) and _fixture_key(bet) == candidate_key
        if same_fixture:
            match_used += stake
            if _market_family(bet.get("market"), bet.get("pick")) == family:
                market_used += stake
    daily_used = sum(
        _safe_float(bet.get("stake"))
        for bet in all_rows
        if _placed_day(bet, current.date() if _pending(bet) else date.min) == current.date()
    )

    diagnostics.update({
        "market_used": market_used,
        "match_used": match_used,
        "daily_used": daily_used,
    })
    market_room = max(0.0, bankroll * MAX_MARKET_STAKE_PCT - market_used)
    match_room = max(0.0, bankroll * MAX_MATCH_EXPOSURE_PCT - match_used)
    daily_room = max(0.0, bankroll * MAX_DAILY_EXPOSURE_PCT - daily_used)
    raw = kelly(
        probability_value,
        odds_value,
        bankroll,
        frac=KELLY_FRACTION,
        cap=MAX_MARKET_STAKE_PCT,
    )
    stake = min(raw, market_room, match_room, daily_room, free_bank)
    stake = round(max(0.0, stake), 2)
    if stake <= 0.0:
        diagnostics["reason"] = (
            "market_limit_reached" if market_room <= 0.0
            else "match_limit_reached" if match_room <= 0.0
            else "daily_limit_reached" if daily_room <= 0.0
            else "no_positive_quarter_kelly"
        )
        return 0.0, diagnostics

    diagnostics["reason"] = "approved"
    diagnostics["stake"] = stake
    return stake, diagnostics
