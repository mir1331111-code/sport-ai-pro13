"""betting/settlement.py — детерминированный сеттлмент (v2).

Что изменено:
  * кириллическая «Х» в пике (Х, Х2, 1Х) трактуется как латинская X;
  * тоталы любой линии (ТБ/ТМ 1.5, 3, 3.5 …, а также Over/Under/O/U), а не только 2.5;
  * целые линии тотала/форы дают push;
  * четвертные линии (±0.25, ±0.75) НЕ засчитываются как полный выигрыш/проигрыш
    (раньше — молча считались неверно): возвращается None, ставка остаётся на ручную проверку;
  * неизвестные пики BTTS/1X2 больше не превращаются молча в «lost».
"""
from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Literal, Optional

Outcome = Literal["won", "lost", "push", "void"]

_EPS = 1e-6
_AH_RE = re.compile(r"Ф([12])\(([-+]?\d+(?:[.,]\d+)?)\)")
_OU_RE = re.compile(r"^(?:Т([БМ])|(Over|Under|O|U))\s*(\d+(?:[.,]\d+)?)$", re.I)
_1X2 = {"П1": "П1", "1": "П1", "П2": "П2", "2": "П2", "X": "X"}
_DC = {"1X", "X2", "12"}


def settlement_not_before(
    match_date,
    match_time=None,
    *,
    kickoff_at=None,
    grace: timedelta = timedelta(hours=2, minutes=15),
) -> Optional[datetime]:
    """Return a UTC instant after which final-score lookups are safe.

    Absolute ISO kickoff timestamps (with Z or an offset) are authoritative.
    A naive local kickoff clock has no timezone, so a 12-hour upper-bound
    offset is added before comparing it with UTC. If no clock is available,
    the latest possible kickoff on that local calendar date is assumed.
    This deliberately delays uncertain fixtures rather than querying scores
    before a match can have started.
    """
    def _absolute_utc(value):
        raw = str(value or "").strip()
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return None
        return parsed.astimezone(timezone.utc)

    # The source's original timestamp wins over transformed local display fields.
    absolute = _absolute_utc(kickoff_at)
    if absolute is None:
        absolute = _absolute_utc(match_date)
    if absolute is not None:
        return absolute + grace

    date_part = None
    raw_date = str(match_date or "").strip()
    if isinstance(match_date, datetime):
        date_part = match_date.date()
    elif isinstance(match_date, date):
        date_part = match_date
    elif raw_date:
        try:
            date_part = datetime.fromisoformat(
                raw_date.replace("Z", "+00:00")
            ).date()
        except ValueError:
            for fmt in (
                "%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y",
                "%d-%m-%Y", "%d/%m/%y",
            ):
                try:
                    date_part = datetime.strptime(raw_date[:10], fmt).date()
                    break
                except ValueError:
                    continue
    if date_part is None:
        return None

    # A naive clock is expressed in an unspecified provider-local timezone.
    # UTC can be as late as local time +12h (UTC-12), so wait that margin.
    raw_time = str(match_time or "").strip()
    if raw_time:
        try:
            if "T" in raw_time or " " in raw_time:
                parsed_time = datetime.fromisoformat(
                    raw_time.replace("Z", "+00:00")
                )
                if parsed_time.tzinfo is not None:
                    return parsed_time.astimezone(timezone.utc) + grace
                local_kickoff = parsed_time
            else:
                local_kickoff = datetime.combine(
                    date_part, time.fromisoformat(raw_time)
                )
            return (
                local_kickoff.replace(tzinfo=timezone.utc)
                + timedelta(hours=12)
                + grace
            )
        except ValueError:
            pass

    # No clock: assume a kickoff at 23:59 in the latest possible timezone.
    # This is intentionally conservative; unknown-time legacy bets may settle
    # later, but they will not trigger score lookups before a possible kickoff.
    latest_local_kickoff = datetime.combine(
        date_part + timedelta(days=1), time(11, 59), tzinfo=timezone.utc
    )
    return latest_local_kickoff + grace


def settlement_ready(
    match_date,
    match_time=None,
    *,
    kickoff_at=None,
    now: Optional[datetime] = None,
    grace: timedelta = timedelta(hours=2, minutes=15),
) -> Optional[bool]:
    """Whether a fixture is old enough for a final-score provider lookup."""
    not_before = settlement_not_before(
        match_date, match_time, kickoff_at=kickoff_at, grace=grace
    )
    if not_before is None:
        return None
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    else:
        current = current.astimezone(timezone.utc)
    return current >= not_before


def _norm_pick(p: Optional[str]) -> str:
    p = (p or "").strip().replace("Х", "X").replace("х", "x")
    return re.sub(r"\s+", " ", p)


def _is_quarter(line: float) -> bool:
    return abs(line * 2 - round(line * 2)) > _EPS


def _num(s: str) -> float:
    return float(s.replace(",", "."))


def _cmp(value: float) -> Outcome:
    if value > _EPS:
        return "won"
    if abs(value) <= _EPS:
        return "push"
    return "lost"


def _ah(pick: str, hg: int, ag: int) -> tuple[Optional[Outcome], str]:
    m = _AH_RE.match(pick or "")
    if not m:
        return None, "ah_parse"
    side, line = int(m.group(1)), _num(m.group(2))
    if _is_quarter(line):
        return None, "ah_quarter_line_manual"
    diff = (hg - ag) if side == 1 else (ag - hg)
    return _cmp(diff + line), "ah"


def _ou(pick: str, hg: int, ag: int) -> tuple[Optional[Outcome], str]:
    m = _OU_RE.match(pick or "")
    if not m:
        return None, f"unknown_ou:{pick}"
    over = ((m.group(1) or "").upper() == "Б"
            or (m.group(2) or "").lower().startswith("o"))
    line = _num(m.group(3))
    if _is_quarter(line):
        return None, "ou_quarter_line_manual"
    total = hg + ag
    return _cmp((total - line) if over else (line - total)), "ou"


def determine_outcome(market: str, pick: str, hg: int, ag: int
                      ) -> tuple[Optional[Outcome], str]:
    try:
        hg, ag = int(hg), int(ag)
    except (TypeError, ValueError):
        return None, "bad_score"
    if hg < 0 or ag < 0:
        return None, "bad_score"

    m = (market or "").upper()
    p = _norm_pick(pick)

    if m in ("", "HOT", "STAT", "OTHER"):
        if p in ("П1", "X", "П2"):
            m = "1X2"
        elif _OU_RE.match(p):
            m = "OU"
        elif p.startswith("Ф"):
            m = "AH"
        elif p.upper().startswith("BTTS"):
            m = "BTTS"
        elif p in _DC:
            m = "DC"
        else:
            return None, f"unknown_pick:{p}"

    if m == "1X2":
        want = _1X2.get(p)
        if want is None:
            return None, f"unknown_1x2:{p}"
        res = "П1" if hg > ag else ("X" if hg == ag else "П2")
        return ("won" if res == want else "lost"), "1x2"
    if m == "OU":
        return _ou(p, hg, ag)
    if m == "AH":
        return _ah(p, hg, ag)
    if m == "BTTS":
        pl = p.lower()
        if pl in ("btts да", "btts yes"):
            want = True
        elif pl in ("btts нет", "btts no"):
            want = False
        else:
            return None, f"unknown_btts:{p}"
        both = hg > 0 and ag > 0
        return ("won" if both == want else "lost"), "btts"
    if m == "DC":
        ok = {"1X": hg >= ag, "X2": hg <= ag, "12": hg != ag}.get(p)
        if ok is None:
            return None, f"unknown_dc:{p}"
        return ("won" if ok else "lost"), "dc"
    return None, f"unknown_market:{m}"


_STAT_TOTAL_RE = re.compile(r"^(Угл|Карты)\s+(ТБ|ТМ)\s+(\d+(?:[.,]\d+)?)$", re.I)


def determine_stat_outcome(
    market: str, pick: str, actual_value: int | float | str,
) -> tuple[Optional[Outcome], str]:
    """Settle corners/cards using actual match statistics, never the score."""
    if str(market or "").upper() not in {"STAT", "CORNERS", "CARDS"}:
        return None, f"unknown_stat_market:{market}"
    if isinstance(actual_value, bool):
        return None, "bad_stat_value"
    try:
        value = float(actual_value)
    except (TypeError, ValueError):
        return None, "bad_stat_value"
    if not value >= 0 or not value.is_integer():
        return None, "bad_stat_value"
    match = _STAT_TOTAL_RE.fullmatch(_norm_pick(pick))
    if not match:
        return None, f"unknown_stat_pick:{_norm_pick(pick)}"
    kind, side, raw_line = match.groups()
    line = _num(raw_line)
    difference = value - line if side.upper() == "ТБ" else line - value
    return _cmp(difference), "stat_corners" if kind.casefold() == "угл" else "stat_cards"
