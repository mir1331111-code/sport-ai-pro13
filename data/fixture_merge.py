"""Merge football fixture feeds without restricting coverage to one provider."""
from __future__ import annotations

import re
from data.team_names import normalize_team
from data.competition import is_women_competition


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

            incoming_division = str(row.get("Div") or row.get("div") or "").upper()
            current_division = str(current.get("Div") or current.get("div") or "").upper()
            incoming_women = (
                row.get("women") is True
                or incoming_division.startswith("W_")
                or is_women_competition(row.get("League"), row.get("competition"))
            )
            current_women = (
                current.get("women") is True
                or current_division.startswith("W_")
                or is_women_competition(current.get("League"), current.get("competition"))
            )
            if incoming_women:
                current["women"] = True
                current["kind"] = "women"
                if not current_division.startswith("W_"):
                    current["Div"] = incoming_division if incoming_division.startswith("W_") else "W_"
                    if current.get("div") is not None:
                        current["div"] = current["Div"]
                # Prefer the explicitly named women's competition over a generic provider label.
                incoming_league_text = str(row.get("League") or row.get("league") or "")
                if is_women_competition(incoming_league_text):
                    current["League"] = incoming_league_text
            elif current_women:
                current["women"] = True
                current["kind"] = "women"
                if not current_division.startswith("W_"):
                    current["Div"] = "W_"
                    if current.get("div") is not None:
                        current["div"] = "W_"

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
