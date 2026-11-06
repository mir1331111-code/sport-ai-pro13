"""Timezone-safe kickoff parsing for pre-match fixture filtering."""
from __future__ import annotations

import os
import re
from datetime import date, datetime, time, timezone, tzinfo
from zoneinfo import ZoneInfo


def app_timezone(name: str | None = None) -> tzinfo:
    """Return the configured display timezone, defaulting to the app's documented zone."""
    candidate = str(name or os.environ.get("APP_TIMEZONE") or "Europe/Warsaw").strip()
    try:
        return ZoneInfo(candidate)
    except Exception:
        return timezone.utc


def _parse_local_date(value: object) -> date | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    raw = raw[:10]
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y", "%m/%d/%Y", "%m/%d/%y", "%d.%m.%Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def _parse_clock(value: object) -> time | None:
    raw = str(value or "").strip().removesuffix("Z")
    match = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?", raw)
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    second = int(match.group(3) or 0)
    try:
        return time(hour, minute, second)
    except ValueError:
        return None


def fixture_kickoff_local(
    row: object, *, timezone_name: str | None = None,
) -> datetime | None:
    """Parse a fixture kickoff and return it in the app timezone.

    An explicit kickoff_utc timestamp is authoritative. Naive values in that
    field are interpreted as UTC by contract. When it is absent, Date + Time
    are interpreted as local wall-clock time. Date-only rows intentionally
    return None so callers do not guess whether a same-day fixture has started.
    """
    if not isinstance(row, dict):
        return None
    target_tz = app_timezone(timezone_name)

    raw_utc = str(row.get("kickoff_utc") or "").strip()
    if raw_utc:
        try:
            parsed = datetime.fromisoformat(raw_utc.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(target_tz)
        except (TypeError, ValueError, OverflowError):
            pass

    fixture_date = _parse_local_date(row.get("Date") or row.get("date"))
    fixture_time = _parse_clock(row.get("Time") or row.get("time"))
    if fixture_date is None or fixture_time is None:
        return None
    try:
        return datetime.combine(fixture_date, fixture_time).replace(tzinfo=target_tz)
    except (TypeError, ValueError, OverflowError):
        return None
