"""Coherent historical bookmaker odds selection."""
def _positive_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def pick_historical_odds(row):
    """Prefer a complete Pinnacle 1X2 triplet, then a complete Bet365 triplet."""
    pinnacle = {
        "П1": _positive_float(row.get("PSH")),
        "X": _positive_float(row.get("PSD")),
        "П2": _positive_float(row.get("PSA")),
    }
    if all(value is not None for value in pinnacle.values()):
        return pinnacle
    bet365 = {
        "П1": _positive_float(row.get("B365H")),
        "X": _positive_float(row.get("B365D")),
        "П2": _positive_float(row.get("B365A")),
    }
    if all(value is not None for value in bet365.values()):
        return bet365
    return {"П1": None, "X": None, "П2": None}
