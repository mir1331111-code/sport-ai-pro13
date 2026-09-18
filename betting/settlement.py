"""betting/settlement.py — детерминированный сеттлмент."""
from __future__ import annotations
import re
from typing import Literal, Optional

Outcome = Literal["won", "lost", "push", "void"]


def _ah(pick: str, hg: int, ag: int) -> Optional[Outcome]:
    m = re.match(r"Ф([12])\(([-+]?\d+(?:\.\d+)?)\)", pick or "")
    if not m:
        return None
    side, line = int(m.group(1)), float(m.group(2))
    res = ((hg - ag) if side == 1 else (ag - hg)) + line
    if res > 0.001:
        return "won"
    if abs(res) <= 0.001:
        return "push"
    return "lost"


def determine_outcome(market: str, pick: str, hg: int, ag: int
                      ) -> tuple[Optional[Outcome], str]:
    m = (market or "").upper()
    p = (pick or "").strip()

    if m in ("", "HOT", "STAT", "OTHER"):
        if p in ("П1", "X", "П2"):
            m = "1X2"
        elif p in ("ТБ 2.5", "ТМ 2.5"):
            m = "OU"
        elif p.startswith("Ф"):
            m = "AH"
        elif p.startswith("BTTS"):
            m = "BTTS"
        elif p in ("1X", "X2", "12"):
            m = "DC"
        else:
            return None, f"unknown_pick:{p}"

    if m == "1X2":
        res = "П1" if hg > ag else ("X" if hg == ag else "П2")
        return ("won" if res == p else "lost"), "1x2"
    if m == "OU":
        total = hg + ag
        if p == "ТБ 2.5":
            return ("won" if total >= 3 else "lost"), "ou"
        if p == "ТМ 2.5":
            return ("won" if total <= 2 else "lost"), "ou"
        return None, f"unknown_ou:{p}"
    if m == "AH":
        o = _ah(p, hg, ag)
        return (o, "ah") if o else (None, "ah_parse")
    if m == "BTTS":
        both = (hg > 0 and ag > 0)
        want = (p == "BTTS да")
        return ("won" if both == want else "lost"), "btts"
    if m == "DC":
        ok = {"1X": hg >= ag, "X2": hg <= ag, "12": hg != ag}.get(p)
        if ok is None:
            return None, f"unknown_dc:{p}"
        return ("won" if ok else "lost"), "dc"
    return None, f"unknown_market:{m}"
