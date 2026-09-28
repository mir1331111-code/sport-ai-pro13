"""betting/kelly.py — безопасный fractional Kelly."""
from __future__ import annotations


def kelly(
    prob: float,
    odds: float,
    bank: float,
    frac: float = 0.25,
    cap: float = 0.05,
) -> float:
    """
    Fractional Kelly.

    Важно:
    - не ставим при отрицательном edge;
    - не ставим при некорректных probability/odds;
    - cap ограничивает долю банка;
    - НИКАКОГО искусственного minimum stake.
    """

    try:
        prob = float(prob)
        odds = float(odds)
        bank = float(bank)
        frac = float(frac)
        cap = float(cap)
    except (TypeError, ValueError):
        return 0.0

    if bank <= 0:
        return 0.0

    if not (0.0 < prob < 1.0):
        return 0.0

    if odds <= 1.0:
        return 0.0

    if frac <= 0:
        return 0.0

    if cap <= 0:
        return 0.0

    # Decimal odds:
    # b = net profit per unit stake
    b = odds - 1.0

    # Full Kelly:
    # f* = (bp - q) / b
    q = 1.0 - prob
    full_kelly = (b * prob - q) / b

    if full_kelly <= 0:
        return 0.0

    fractional = full_kelly * frac

    # Safety cap.
    fraction = min(fractional, cap)

    if fraction <= 0:
        return 0.0

    return round(bank * fraction, 2)


def edge(prob: float, odds: float) -> float:
    """
    Probability edge относительно implied probability рынка.

    Например:
        model = 0.60
        odds = 2.00
        market probability = 0.50
        edge = +0.10
    """
    try:
        prob = float(prob)
        odds = float(odds)
    except (TypeError, ValueError):
        return 0.0

    if not (0.0 < prob < 1.0):
        return 0.0

    if odds <= 1.0:
        return 0.0

    return prob - (1.0 / odds)


def expected_value(prob: float, odds: float) -> float:
    """
    EV на единицу ставки.

    EV = P(win) * odds - 1
    """
    try:
        prob = float(prob)
        odds = float(odds)
    except (TypeError, ValueError):
        return 0.0

    if not (0.0 < prob < 1.0):
        return 0.0

    if odds <= 1.0:
        return 0.0

    return prob * odds - 1.0


def market_type(pick: str) -> str:
    pick = str(pick or "")

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
