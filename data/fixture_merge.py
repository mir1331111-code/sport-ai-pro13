"""Merge football fixture feeds without restricting coverage to one provider."""
from __future__ import annotations

import re
from data.team_names import normalize_team


def _team_key(value) -> str:
    """Use the same conservative exact-alias identity as score reconciliation."""
    return normalize_team(str(value or ""))


def fixture_key(row: dict) -> tuple:
    """Stable date/team key across providers with different fixture IDs."""
    return (
        str(row.get("Date") or row.get("date") or "")[:10],
        _team_key(row.get("HomeTeam") or row.get("home_team")),
        _team_key(row.get("AwayTeam") or row.get("away_team")),
    )


def _generic_div(value) -> bool:
    return str(value or "").strip().casefold() in {"", "g", "general", "unknown", "other"}


def _generic_league(value) -> bool:
    normalized = re.sub(r"[^a-zа-я0-9]+", " ", str(value or "").casefold()).strip()
    return normalized in {
        "", "soccer", "football", "матч", "лига", "unknown", "other",
        "неизвестно", "—", "-",
    }


def merge_fixture_sources(*sources) -> list:
    """Merge source lists, keeping the earliest provider's richer row.

    Callers should pass preferred providers first (football-data.org, then
    TheSportsDB, then ESPN). Missing metadata is filled from later providers;
    distinct fixtures are never dropped because a provider failed.
    """
    merged = {}
    order = []
    for source in sources:
        if not isinstance(source, (list, tuple)):
            continue
        for row in source:
            if not isinstance(row, dict):
                continue
            key = fixture_key(row)
            if not all(key):
                # If no reliable identity is available, retain rather than
                # silently discard the fixture.
                key = ("provider-row", len(order), str(row.get("fixture_id") or ""))
            if key not in merged:
                merged[key] = dict(row)
                order.append(key)
                continue
            current = merged[key]

            # A generic first-provider label must not mask a precise competition
            # supplied later (e.g. TSDB "Soccer" followed by SStats "Eredivisie").
            # The historical corners/cards model is indexed by Div, so keeping
            # Div="G" would make a well-covered league look as if it had no history.
            incoming_div = row.get("Div") or row.get("div")
            current_div = current.get("Div") or current.get("div")
            if _generic_div(current_div) and not _generic_div(incoming_div):
                current["Div"] = incoming_div
                if current.get("div") is not None:
                    current["div"] = incoming_div

            incoming_league = row.get("League") or row.get("league")
            current_league = current.get("League") or current.get("league")
            if _generic_league(current_league) and not _generic_league(incoming_league):
                current["League"] = incoming_league
                if current.get("league") is not None:
                    current["league"] = incoming_league

            for name, value in row.items():
                if current.get(name) in (None, "", {}, []) and value not in (None, "", {}, []):
                    current[name] = value
    return [merged[key] for key in order]


_SSTATS_ODDS_FIELDS = (
    "market_odds",
    "total_market_odds",
    "btts_market_odds",
    "stat_market_odds",
    "asian_handicap_market_odds",
    "bookmaker_by_pick",
    "odds",
)


def is_sstats_market_source(source) -> bool:
    """Return whether a quote's explicit provenance is SStats."""
    return str(source or "").strip().casefold() == "sstats"


def enforce_sstats_odds_only(rows) -> list:
    """Keep bookmaker prices only on fixtures owned by SStats.

    Schedule fallback feeds may supply their own price-like fields. They are
    deliberately stripped here: the live scanner's bookmaker source is SStats,
    not whichever schedule adapter happened to return a row.
    """
    result = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        clean = dict(row)
        fixture_id = str(clean.get("fixture_id") or "").strip().casefold()
        game_id = clean.get("sstats_game_id")
        is_sstats_fixture = (
            game_id not in (None, "")
            and (
                str(clean.get("source") or "").strip().casefold() == "sstats"
                or fixture_id.startswith("sstats:")
            )
        )
        if not is_sstats_fixture:
            for field in _SSTATS_ODDS_FIELDS:
                if field in clean:
                    clean[field] = {}
            for field in (
                "odds_source", "total_odds_source", "btts_odds_source",
                "extra_market_source",
            ):
                if field in clean:
                    clean[field] = "no_market"
        result.append(clean)
    return result
