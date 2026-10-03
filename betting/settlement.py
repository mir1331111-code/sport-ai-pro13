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
from typing import Literal, Optional

Outcome = Literal["won", "lost", "push", "void"]

_EPS = 1e-6
_AH_RE = re.compile(r"Ф([12])\(([-+]?\d+(?:[.,]\d+)?)\)")
_OU_RE = re.compile(r"^(?:Т([БМ])|(Over|Under|O|U))\s*(\d+(?:[.,]\d+)?)$", re.I)
_1X2 = {"П1": "П1", "1": "П1", "П2": "П2", "2": "П2", "X": "X"}
_DC = {"1X", "X2", "12"}


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
