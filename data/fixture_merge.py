"""Merge football fixture feeds without restricting coverage to one provider."""
from __future__ import annotations

import re
import unicodedata


def _team_key(value) -> str:
    """Canonicalize provider-specific team labels before cross-feed matching.

    Prefer the repository's alias map (e.g. Man Utd/Manchester United) and
    retain a conservative fallback for unknown teams. This key is used to
    attach bookmaker odds and provider IDs, so home/away order and date remain
    strict to avoid assigning a price to the wrong fixture.
    """
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        from data.team_names import normalize_team
        canonical = normalize_team(raw)
        if canonical:
            return canonical
    except Exception:
        pass
    text = unicodedata.normalize("NFKD", raw.casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\b(football club|fc|afc|cf|sc|ac)\b", " ", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def fixture_key(row: dict) -> tuple:
    """Stable date/team key across providers with different fixture IDs."""
    return (
        str(row.get("Date") or row.get("date") or "")[:10],
        _team_key(row.get("HomeTeam") or row.get("home_team")),
        _team_key(row.get("AwayTeam") or row.get("away_team")),
    )


def merge_fixture_sources(*sources) -> list:
    """Merge fixture feeds while preserving metadata and odds from each source.

    Callers pass preferred providers first. For duplicate fixtures, missing
    metadata is filled and odds dictionaries are merged by exact market key;
    the best valid decimal price per market is retained. Dates and home/away
    order must match exactly.
    """
    merged = {}
    order = []
    odds_groups = (
        "market_odds", "total_market_odds", "btts_market_odds",
        "stat_market_odds", "asian_handicap_market_odds",
    )
    for source in sources:
        if not isinstance(source, (list, tuple)):
            continue
        for row in source:
            if not isinstance(row, dict):
                continue
            key = fixture_key(row)
            if not all(key):
                key = ("provider-row", len(order), str(row.get("fixture_id") or ""))
            if key not in merged:
                merged[key] = dict(row)
                order.append(key)
                continue
            current = merged[key]
            for name, value in row.items():
                if name in odds_groups:
                    existing = current.get(name)
                    if not isinstance(existing, dict):
                        existing = {}
                    incoming = value if isinstance(value, dict) else {}
                    combined = dict(existing)
                    for market, raw_price in incoming.items():
                        try:
                            price = float(raw_price)
                        except (TypeError, ValueError):
                            continue
                        if 1.01 < price <= 100:
                            try:
                                previous = float(combined.get(market) or 0)
                            except (TypeError, ValueError):
                                previous = 0
                            if price > previous:
                                combined[market] = price
                    if combined:
                        current[name] = combined
                    continue
                if current.get(name) in (None, "", {}, []) and value not in (None, "", {}, []):
                    current[name] = value
            # Keep the provider/source label aligned with the odds that were
            # merged, rather than leaving a stale empty-source marker.
            if current.get("market_odds") and current.get("odds_source") in ("", None, "no_market"):
                current["odds_source"] = row.get("odds_source") or "sstats"
    return [merged[key] for key in order]
