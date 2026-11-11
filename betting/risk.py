"""Conservative, explainable exposure guardrails for the virtual portfolio."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any


DEFAULT_LIMITS = {
    "portfolio_pct": 20.0,
    "match_pct": 5.0,
    "league_pct": 15.0,
    "day_pct": 10.0,
}


def _number(value: Any) -> float:
    try:
        number = float(value or 0)
        return number if number >= 0 and number < float("inf") else 0.0
    except (TypeError, ValueError, OverflowError):
        return 0.0


def portfolio_risk_report(
    bets: list[dict], *, bank: float, initial_bank: float | None = None,
    limits: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Summarize open exposure, concentrations, overlap and historical drawdown.

    Same-match positions are flagged as potentially dependent; this is a
    conservative overlap proxy, not an estimated statistical correlation.
    """
    caps = dict(DEFAULT_LIMITS)
    if isinstance(limits, dict):
        for key, value in limits.items():
            if key in caps:
                try:
                    caps[key] = max(0.0, min(100.0, float(value)))
                except (TypeError, ValueError):
                    pass
    current_bank = max(0.0, _number(bank))
    positions = [
        row for row in (bets or [])
        if isinstance(row, dict) and row.get("status") == "pending"
    ]
    by_match: dict[str, float] = defaultdict(float)
    by_league: dict[str, float] = defaultdict(float)
    by_day: dict[str, float] = defaultdict(float)
    match_positions: dict[str, list[dict]] = defaultdict(list)
    total = 0.0
    for bet in positions:
        stake = _number(bet.get("stake"))
        total += stake
        date = str(bet.get("date_iso") or bet.get("date") or "Без даты")[:10]
        match = str(bet.get("match") or bet.get("match_ru") or "Матч")
        league = str(bet.get("league") or bet.get("div") or "Лига не указана")
        match_key = f"{date}|{match}"
        by_match[match_key] += stake
        by_league[league] += stake
        by_day[date] += stake
        match_positions[match_key].append(bet)

    def table(values: dict[str, float]) -> list[dict]:
        return [
            {"group": key, "exposure": amount,
             "bank_pct": amount / current_bank * 100 if current_bank else None}
            for key, amount in sorted(values.items(), key=lambda item: -item[1])
        ]

    match_rows = table(by_match)
    league_rows = table(by_league)
    day_rows = table(by_day)
    overlap_rows = []
    for key, members in match_positions.items():
        distinct_markets = sorted({
            str(bet.get("market") or bet.get("pick") or "unknown")
            for bet in members
        })
        if len(members) > 1:
            overlap_rows.append({
                "match": key,
                "positions": len(members),
                "markets": ", ".join(distinct_markets),
                "exposure": by_match[key],
                "bank_pct": by_match[key] / current_bank * 100 if current_bank else None,
                "warning": "Потенциально зависимые исходы одного матча",
            })

    # Reconstruct realised equity from closed virtual positions in settlement order.
    closed = [
        bet for bet in (bets or [])
        if isinstance(bet, dict) and bet.get("status") in {"won", "lost"}
    ]
    def settled_order(bet: dict):
        return str(bet.get("settled_at") or bet.get("date_time") or bet.get("date_iso") or "")
    closed.sort(key=settled_order)
    equity = max(0.0, _number(initial_bank if initial_bank is not None else bank))
    peak = equity
    max_drawdown = 0.0
    for bet in closed:
        stake = _number(bet.get("stake"))
        pnl = stake * (_number(bet.get("odds")) - 1.0) if bet.get("status") == "won" else -stake
        equity += pnl
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)

    alerts = []
    portfolio_pct = total / current_bank * 100 if current_bank else None
    if portfolio_pct is not None and portfolio_pct > caps["portfolio_pct"]:
        alerts.append(f"Общая открытая экспозиция {portfolio_pct:.1f}% выше лимита {caps['portfolio_pct']:.1f}%.")
    for row in match_rows:
        if row["bank_pct"] is not None and row["bank_pct"] > caps["match_pct"]:
            alerts.append(f"Матч {row['group']}: экспозиция {row['bank_pct']:.1f}% выше лимита {caps['match_pct']:.1f}%.")
    for row in league_rows:
        if row["bank_pct"] is not None and row["bank_pct"] > caps["league_pct"]:
            alerts.append(f"Лига {row['group']}: экспозиция {row['bank_pct']:.1f}% выше лимита {caps['league_pct']:.1f}%.")
    for row in day_rows:
        if row["bank_pct"] is not None and row["bank_pct"] > caps["day_pct"]:
            alerts.append(f"Дата {row['group']}: экспозиция {row['bank_pct']:.1f}% выше лимита {caps['day_pct']:.1f}%.")

    return {
        "open_positions": len(positions),
        "total_exposure": total,
        "portfolio_pct": portfolio_pct,
        "by_match": match_rows,
        "by_league": league_rows,
        "by_day": day_rows,
        "overlap": sorted(overlap_rows, key=lambda row: -row["exposure"]),
        "alerts": alerts,
        "max_drawdown": max_drawdown,
        "settled_positions": len(closed),
        "limits": caps,
    }
