"""Conservative portfolio exposure and concentration diagnostics."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import math
from typing import Any, Iterable

DEFAULT_LIMITS = {
    "match_pct": 0.03,
    "league_pct": 0.05,
    "day_pct": 0.10,
}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def _match_identity(bet: dict) -> str:
    fixture = str(bet.get("fixture_id") or "").strip()
    date = str(bet.get("date_iso") or bet.get("date") or "")[:10]
    match = str(bet.get("match") or bet.get("match_ru") or "").strip()
    return fixture or f"{date}|{match}"


def _drawdown(history: Iterable[dict] | None, initial_bank: float | None) -> dict:
    values = []
    if initial_bank is not None and _number(initial_bank) > 0:
        values.append(_number(initial_bank))
    for row in history or []:
        if isinstance(row, dict) and row.get("bank") is not None:
            balance = _number(row.get("bank"), -1)
            if balance >= 0:
                values.append(balance)
    if len(values) < 2:
        return {"max_drawdown": None, "peak_bank": None, "current_drawdown": None, "observations": len(values)}
    peak = values[0]
    maximum = 0.0
    current_peak = peak
    for value in values:
        if value > peak:
            peak = value
        if peak > 0:
            maximum = max(maximum, (peak - value) / peak)
        current_peak = max(current_peak, value)
    current = (current_peak - values[-1]) / current_peak if current_peak > 0 else None
    return {
        "max_drawdown": maximum,
        "peak_bank": current_peak,
        "current_drawdown": current,
        "observations": len(values),
    }


def build_portfolio_risk_report(
    bets: Iterable[dict],
    *,
    bank: float,
    initial_bank: float | None = None,
    bank_history: Iterable[dict] | None = None,
    limits: dict | None = None,
) -> dict:
    """Calculate open exposure by match, league and date without changing bets."""
    bank_value = max(0.0, _number(bank))
    limits = {**DEFAULT_LIMITS, **(limits or {})}
    pending = [
        row for row in (bets or [])
        if isinstance(row, dict) and str(row.get("status") or "pending").lower() == "pending"
    ]
    match_groups: dict[str, list[dict]] = defaultdict(list)
    league_groups: dict[str, float] = defaultdict(float)
    day_groups: dict[str, float] = defaultdict(float)
    for bet in pending:
        identity = _match_identity(bet)
        match_groups[identity].append(bet)
        league = str(bet.get("league") or bet.get("div") or "Неизвестная лига")
        league_groups[league] += max(0.0, _number(bet.get("stake")))
        day = str(bet.get("date_iso") or bet.get("date") or "Дата не указана")[:10]
        day_groups[day] += max(0.0, _number(bet.get("stake")))

    match_rows = []
    for identity, group in sorted(
        match_groups.items(),
        key=lambda item: sum(_number(b.get("stake")) for b in item[1]),
        reverse=True,
    ):
        exposure = sum(max(0.0, _number(b.get("stake"))) for b in group)
        markets = sorted({str(b.get("market") or "OTHER").upper() for b in group})
        names = str(group[0].get("match_ru") or group[0].get("match") or identity)
        pct = exposure / bank_value if bank_value > 0 else None
        match_rows.append({
            "match": names, "date": str(group[0].get("date_iso") or group[0].get("date") or "")[:10],
            "stake": exposure, "bank_pct": pct, "positions": len(group),
            "markets": ", ".join(markets),
            "overlap": len(group) > 1 or len(markets) > 1,
            "limit_exceeded": pct is not None and pct > float(limits["match_pct"]),
        })

    league_rows = [
        {"league": league, "stake": stake,
         "bank_pct": stake / bank_value if bank_value > 0 else None,
         "limit_exceeded": bank_value > 0 and stake / bank_value > float(limits["league_pct"])}
        for league, stake in sorted(league_groups.items(), key=lambda x: x[1], reverse=True)
    ]
    day_rows = [
        {"date": day, "stake": stake,
         "bank_pct": stake / bank_value if bank_value > 0 else None,
         "limit_exceeded": bank_value > 0 and stake / bank_value > float(limits["day_pct"])}
        for day, stake in sorted(day_groups.items(), key=lambda x: x[1], reverse=True)
    ]
    total_exposure = sum(max(0.0, _number(b.get("stake"))) for b in pending)
    same_match_overlaps = [row for row in match_rows if row["overlap"]]
    warnings = []
    for row in match_rows:
        if row["limit_exceeded"]:
            warnings.append(f"Превышен лимит риска на матч: {row['match']}")
    for row in league_rows:
        if row["limit_exceeded"]:
            warnings.append(f"Превышен лимит риска на лигу: {row['league']}")
    for row in day_rows:
        if row["limit_exceeded"]:
            warnings.append(f"Превышен дневной лимит риска: {row['date']}")
    if same_match_overlaps:
        warnings.append("Несколько открытых позиций относятся к одним и тем же матчам; их исходы зависимы.")
    return {
        "pending_count": len(pending),
        "total_exposure": total_exposure,
        "bank_pct": total_exposure / bank_value if bank_value > 0 else None,
        "match_rows": match_rows,
        "league_rows": league_rows,
        "day_rows": day_rows,
        "same_match_overlaps": same_match_overlaps,
        "warnings": warnings,
        "limits": limits,
        "drawdown": _drawdown(bank_history, initial_bank),
        "loss_streak": _loss_streak(bets),
        "policy_note": (
            "Пороги — диагностические ориентиры. Они не меняют размер существующих "
            "позиций и не блокируют ручные действия."
        ),
    }


def _loss_streak(bets: Iterable[dict]) -> int:
    settled = [
        bet for bet in (bets or [])
        if isinstance(bet, dict) and str(bet.get("status") or "").lower() in {"won", "lost"}
    ]
    settled.sort(key=lambda bet: str(
        bet.get("settled_at") or bet.get("date_time") or bet.get("date_iso") or ""
    ))
    streak = 0
    for bet in reversed(settled):
        if str(bet.get("status") or "").lower() == "lost":
            streak += 1
        else:
            break
    return streak
