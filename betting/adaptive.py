"""betting/adaptive.py — Adaptive priority engine.

Только приоритизация кандидатов. Не меняет строгие BET/WATCH/SKIP-фильтры.
Использует только уже завершённые priced decision snapshots и только данные,
которые предшествуют дате предстоящего матча.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from datetime import datetime


MIN_ACTION_N = 20
PRIOR_MIN = 0.90
PRIOR_MAX = 1.10
SHRINK_N = 20


def _dt(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def build_profiles(snapshots):
    """Индексирует completed priced snapshots по (market, league) и fallback-сегментам."""
    groups = defaultdict(list)
    for row in snapshots or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("result_status") or "").lower() not in ("won", "lost"):
            continue
        try:
            odd = float(row.get("market_odd") or 0.0)
            if odd <= 1.01:
                continue
            pnl = odd - 1.0 if str(row.get("result_status")).lower() == "won" else -1.0
        except (TypeError, ValueError):
            continue
        dt = _dt(row.get("date_iso"))
        if dt is None:
            continue
        market = str(row.get("market") or "").strip() or "—"
        league = str(row.get("league") or "").strip() or "—"
        groups[("market", market)].append((dt, pnl))
        groups[("league", league)].append((dt, pnl))
        groups[("market_league", market, league)].append((dt, pnl))

    for key in groups:
        groups[key].sort(key=lambda x: x[0])
    return groups


def _segment_stats(rows, before_dt):
    if not rows or before_dt is None:
        return None
    dates = [x[0] for x in rows]
    cut = bisect_left(dates, before_dt)
    if cut < MIN_ACTION_N:
        return None
    usable = rows[:cut]
    n = len(usable)
    pnl = sum(x[1] for x in usable)
    roi = pnl / n
    prior = SHRINK_N
    adj_roi = (pnl + roi * prior) / (n + prior)
    # Recent half is a stability check, not a second optimizer.
    recent = usable[max(0, n // 2):]
    recent_roi = sum(x[1] for x in recent) / max(1, len(recent))
    return {"n": n, "roi": roi, "adj_roi": adj_roi, "recent_roi": recent_roi}


def priority(profile, card):
    """Возвращает (factor, reason). Factor ограничен ±10% и включается при N>=20."""
    if not profile or not isinstance(card, dict):
        return 1.0, "Adaptive: недостаточно истории"
    before = _dt(card.get("date_iso"))
    if before is None:
        return 1.0, "Adaptive: нет даты"
    market = str((card.get("verdict") or {}).get("market") or (card.get("verdict") or {}).get("market_type") or "").strip() or "—"
    league = str(card.get("league") or card.get("div") or "").strip() or "—"

    choices = [
        (("market_league", market, league), "рынок+лига"),
        (("market", market), "рынок"),
        (("league", league), "лига"),
    ]
    chosen = None
    for key, label in choices:
        rows = profile.get(key, [])
        stats = _segment_stats(rows, before)
        if stats is not None:
            wf_ok, wf_reason = _walk_forward_gate(rows, before)
            chosen = (stats, label, wf_ok, wf_reason)
            break
    if chosen is None:
        return 1.0, "Adaptive: N<20 — нейтрально"

    stats, label, wf_ok, wf_reason = chosen
    if not wf_ok:
        return 1.0, f"Adaptive: {label} · {wf_reason} · нейтрально"
    adj = stats["adj_roi"]
    recent = stats["recent_roi"]
    factor = 1.0
    if adj >= 0.05 and recent >= 0.0:
        factor = 1.10
    elif adj >= 0.025 and recent >= 0.0:
        factor = 1.05
    elif adj <= -0.05 and recent <= 0.0:
        factor = 0.90
    elif adj <= -0.025 and recent <= 0.0:
        factor = 0.95

    sign = "↑" if factor > 1 else "↓" if factor < 1 else "→"
    return factor, (
        f"Adaptive {sign} {factor:.2f} · {label} · N={stats['n']} · "
        f"Adj ROI {adj:.1%} · recent {recent:.1%} · {wf_reason}"
    )



# Walk-forward gate: минимум 12 исторических наблюдений в OOS-половине
# и положительный/не ухудшившийся результат относительно train.
WF_MIN_OOS = 12
WF_MIN_TRAIN = 12


def _walk_forward_gate(rows, before_dt):
    if not rows or before_dt is None:
        return False, "WF: нет истории"
    dates = [x[0] for x in rows]
    cut = bisect_left(dates, before_dt)
    usable = rows[:cut]
    n = len(usable)
    if n < WF_MIN_TRAIN + WF_MIN_OOS:
        return False, f"WF: N={n} < {WF_MIN_TRAIN + WF_MIN_OOS}"
    split = max(WF_MIN_TRAIN, n * 7 // 10)
    train = usable[:split]
    test = usable[split:]
    if len(test) < WF_MIN_OOS:
        return False, "WF: мало OOS"
    train_roi = sum(x[1] for x in train) / len(train)
    test_roi = sum(x[1] for x in test) / len(test)
    # Не требуем высокой доходности: OOS должен хотя бы не разрушать train.
    stable = test_roi >= min(0.0, train_roi - 0.05)
    return stable, f"WF train {train_roi:.1%} · OOS {test_roi:.1%} · N={len(test)}"
