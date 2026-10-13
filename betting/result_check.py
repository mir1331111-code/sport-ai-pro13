"""Conservative settlement suggestions from completed official scores."""
from __future__ import annotations

from betting.settlement import determine_outcome


def suggest_settlement(market, pick, result):
    """Suggest an outcome only for a normal-time FT result with valid scores."""
    if not isinstance(result, dict) or str(result.get("status", "")).upper() != "FT":
        return None
    try:
        home = int(result["home"])
        away = int(result["away"])
    except (KeyError, TypeError, ValueError):
        return None
    if home < 0 or away < 0:
        return None
    outcome, reason = determine_outcome(market, pick, home, away)
    if outcome not in {"won", "lost", "push", "void"}:
        return None
    return {"outcome": outcome, "reason": reason, "home": home, "away": away}
