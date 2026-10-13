"""Status hints for unsettled bets; never infer a final result from kickoff time."""
from __future__ import annotations

from datetime import datetime


def pending_event_notice(status, date_iso, match_time="", now=None):
    """Return a caution for overdue pending events without settling the bet."""
    if status != "pending":
        return None
    now = now or datetime.now()
    raw_date = str(date_iso or "").strip()
    try:
        event_date = datetime.strptime(raw_date[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None

    if event_date < now.date():
        return "Дата события прошла — ставка ожидает расчёта; результат не подтверждён."
    if event_date > now.date():
        return None

    raw_time = str(match_time or "").strip()
    for fmt in ("%H:%M", "%H.%M"):
        try:
            kickoff = datetime.strptime(raw_time, fmt).time()
            event_dt = now.replace(hour=kickoff.hour, minute=kickoff.minute, second=0, microsecond=0)
            if event_dt < now:
                return "Время начала прошло — ставка ожидает расчёта; результат не подтверждён."
            return None
        except ValueError:
            continue
    return None
