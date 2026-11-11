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
                first = dict(row)
                first["_source_evidence"] = [{
                    "provider": str(row.get("source_provider") or "unknown"),
                    "league": str(row.get("League") or ""),
                    "div": str(row.get("Div") or ""),
                    "women": row.get("women"),
                    "national": row.get("national"),
                    "kickoff_utc": str(row.get("kickoff_utc") or ""),
                    "fixture_id": str(row.get("fixture_id") or ""),
                }]
                first.setdefault("source_conflicts", [])
                merged[key] = first
                order.append(key)
                continue
            current = merged[key]
            evidence = current.setdefault("_source_evidence", [])
            evidence.append({
                "provider": str(row.get("source_provider") or "unknown"),
                "league": str(row.get("League") or ""),
                "div": str(row.get("Div") or ""),
                "women": row.get("women"),
                "national": row.get("national"),
                "kickoff_utc": str(row.get("kickoff_utc") or ""),
                "fixture_id": str(row.get("fixture_id") or ""),
            })
            conflicts = current.setdefault("source_conflicts", [])
            current_category = (
                "women" if current.get("women") or str(current.get("Div") or "").startswith("W_")
                else "national" if current.get("national") or str(current.get("Div") or "").startswith("NT")
                else "men" if current.get("category_verified") else "unknown"
            )
            incoming_category = (
                "women" if row.get("women") or str(row.get("Div") or "").startswith("W_")
                else "national" if row.get("national") or str(row.get("Div") or "").startswith("NT")
                else "men" if row.get("category_verified") else "unknown"
            )
            if current_category != "unknown" and incoming_category != "unknown" and current_category != incoming_category:
                conflict = f"category conflict: {current_category} vs {incoming_category}"
                if conflict not in conflicts:
                    conflicts.append(conflict)
            current_kickoff = str(current.get("kickoff_utc") or "")
            incoming_kickoff = str(row.get("kickoff_utc") or "")
            if current_kickoff and incoming_kickoff and current_kickoff != incoming_kickoff:
                try:
                    from datetime import datetime
                    a = datetime.fromisoformat(current_kickoff.replace("Z", "+00:00"))
                    b = datetime.fromisoformat(incoming_kickoff.replace("Z", "+00:00"))
                    if a.tzinfo is not None and b.tzinfo is not None and abs((a-b).total_seconds()) > 900:
                        conflict = f"kickoff conflict: {current_kickoff} vs {incoming_kickoff}"
                        if conflict not in conflicts:
                            conflicts.append(conflict)
                except (TypeError, ValueError):
                    if "kickoff conflict: unparseable provider timestamps" not in conflicts:
                        conflicts.append("kickoff conflict: unparseable provider timestamps")
            # Prefer explicit positive category evidence over an unclassified generic feed.
            if incoming_category in {"women", "national"} and current_category in {"unknown", "men"}:
                current["women"] = incoming_category == "women"
                current["national"] = incoming_category == "national"
                current["category_verified"] = True
                current["Div"] = row.get("Div") or current.get("Div") or "G"
                if row.get("League"):
                    current["League"] = row.get("League")

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
