"""betting/kelly.py — Kelly с дробным коэффициентом."""
from __future__ import annotations


def kelly(prob: float, odds: float, bank: float, frac: float,
          cap: float = 0.05) -> float:
    if prob <= 0 or odds <= 1:
        return 0.0
    b = odds - 1
    k = (b * prob - (1 - prob)) / b
    return round(min(max(0.0, k * frac), cap) * bank, 2)


def market_type(pick: str) -> str:
    if pick in ("П1", "X", "П2"):
        return "1X2"
    if pick in ("ТБ 2.5", "ТМ 2.5"):
        return "OU"
    if pick.startswith("BTTS"):
        return "BTTS"
    if pick.startswith("Ф"):
        return "AH"
    if pick in ("1X", "X2", "12"):
        return "DC"
    return "OTHER"
