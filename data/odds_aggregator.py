"""Unified live bookmaker odds collection for the football scanner.

Schedule sources remain independent from odds sources. Prices are accepted only
when a supported provider explicitly returned valid decimal prices for the exact
fixture. A family's prices are never stitched from different providers.
"""
from __future__ import annotations

import math
import re
from typing import Any

from config import DIV_TO_ODDS

TRUSTED_ODDS_SOURCES = frozenset({
    "sstats",
    "fdorg",
    "odds_api",
    "therundown",
    "highlightly",
    "api_football",
    "prognozist_marathon",
})

SOURCE_LABELS = {
    "sstats": "SStats",
    "fdorg": "football-data.org",
    "odds_api": "The Odds API",
    "therundown": "TheRundown",
    "highlightly": "Highlightly",
    "api_football": "API-Football",
    "prognozist_marathon": "Prognozist / Marathon",
}

_SOURCE_ALIASES = {
    "s stats": "sstats",
    "football-data.org": "fdorg",
    "football data": "fdorg",
    "the odds api": "odds_api",
    "theoddsapi": "odds_api",
    "odds api": "odds_api",
    "therundown": "therundown",
    "the rundown": "therundown",
    "api-football": "api_football",
    "api football": "api_football",
    "api-sports": "api_football",
    "prognozist": "prognozist_marathon",
    "prognozist marathon": "prognozist_marathon",
    "prognozist_marathon": "prognozist_marathon",
    "sstats": "sstats",
    "fdorg": "fdorg",
    "highlightly": "highlightly",
    "odds_api": "odds_api",
}


def canonical_odds_source(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().casefold())
    return _SOURCE_ALIASES.get(text, text.replace(" ", "_"))


def is_trusted_odds_source(value: Any) -> bool:
    return canonical_odds_source(value) in TRUSTED_ODDS_SOURCES


def _price(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and 1.01 < number <= 100.0 else None


def _flat_market_maps(payload: dict) -> dict[str, dict]:
    """Normalize adapter-specific result shapes without fabricating selections."""
    raw = payload.get("odds") if isinstance(payload.get("odds"), dict) else {}
    result = {
        "winner": {},
        "goals": {},
        "btts": {},
        "corners": {},
        "cards": {},
        "handicap": {},
    }
    winner = payload.get("market_odds")
    if not isinstance(winner, dict):
        winner = {}
    for key in ("П1", "X", "П2"):
        value = _price(winner.get(key, payload.get(key, raw.get(key))))
        if value is not None:
            result["winner"][key] = value

    groups = (
        ("goals", "total_market_odds", r"^Т[БМ]\s+"),
        ("btts", "btts_market_odds", r"^BTTS\s+"),
        ("corners", "stat_market_odds", r"^Угл\s+"),
        ("cards", "stat_market_odds", r"^Карты\s+"),
        ("handicap", "asian_handicap_market_odds", r"^Ф[12]\s+"),
    )
    for family, field, pattern in groups:
        mapping = payload.get(field)
        if isinstance(mapping, dict):
            for key, raw_value in mapping.items():
                pick = str(key or "").strip()
                if family == "btts":
                    folded = pick.casefold()
                    if folded in {"btts да", "btts yes"}:
                        pick = "BTTS Да"
                    elif folded in {"btts нет", "btts no"}:
                        pick = "BTTS Нет"
                value = _price(raw_value)
                if value is not None and re.match(pattern, pick, re.I):
                    result[family][pick] = value

    # Flat adapters (The Odds API, API-Football, TheRundown and Prognozist).
    for key, raw_value in {**raw, **payload}.items():
        pick = str(key or "").strip()
        if pick in {"П1", "X", "П2"} or pick in {
            "odds", "market_odds", "total_market_odds", "btts_market_odds",
            "stat_market_odds", "asian_handicap_market_odds",
        }:
            continue
        value = _price(raw_value)
        if value is None:
            continue
        if re.match(r"^Т[БМ]\s+", pick, re.I):
            result["goals"].setdefault(pick, value)
        elif pick.casefold() in {"btts да", "btts нет", "btts yes", "btts no"}:
            canonical = "BTTS Да" if pick.casefold() in {"btts да", "btts yes"} else "BTTS Нет"
            result["btts"].setdefault(canonical, value)
        elif re.match(r"^угл\s+", pick, re.I):
            result["corners"].setdefault(pick, value)
        elif re.match(r"^карты\s+", pick, re.I):
            result["cards"].setdefault(pick, value)
        elif re.match(r"^ф[12]\s+", pick, re.I):
            result["handicap"].setdefault(pick, value)

    return result


def _complete_lines(mapping: dict[str, float], family: str) -> dict[str, float]:
    """Keep coherent two-sided lines; 1X2 is accepted only as a complete triplet."""
    if family == "winner":
        return dict(mapping) if all(k in mapping for k in ("П1", "X", "П2")) else {}
    if family == "btts":
        pair = {"BTTS Да", "BTTS Нет"}
        return {k: v for k, v in mapping.items() if k in pair} if pair.issubset(mapping) else {}
    if family == "handicap":
        if not mapping:
            return {}
        # A handicap line is directional and can be independently quoted.
        return dict(mapping)
    if family in {"goals", "corners", "cards"}:
        prefix = {"goals": ("ТБ ", "ТМ "), "corners": ("Угл ТБ ", "Угл ТМ "),
                  "cards": ("Карты ТБ ", "Карты ТМ ")}[family]
        result = {}
        lines = set()
        for key in mapping:
            for side_prefix in prefix:
                if key.startswith(side_prefix):
                    lines.add(key[len(side_prefix):])
        for line in lines:
            over_key, under_key = prefix[0] + line, prefix[1] + line
            if over_key in mapping and under_key in mapping:
                result[over_key] = mapping[over_key]
                result[under_key] = mapping[under_key]
        return result
    return {}


def _existing_map(snapshot: dict, field: str) -> dict:
    value = snapshot.get(field)
    if not isinstance(value, dict):
        return {}
    return {str(k): price for k, raw in value.items() if (price := _price(raw)) is not None}


def _source_for_pick(snapshot: dict, pick: str, fallback: str) -> str:
    sources = snapshot.get("market_odds_source_by_pick")
    source = sources.get(pick) if isinstance(sources, dict) else None
    return canonical_odds_source(source or fallback)


def _field_names(family: str) -> tuple[str, str]:
    return {
        "winner": ("market_odds", "odds_source"),
        "goals": ("total_market_odds", "total_odds_source"),
        "btts": ("btts_market_odds", "btts_odds_source"),
        "corners": ("stat_market_odds", "extra_market_source"),
        "cards": ("stat_market_odds", "extra_market_source"),
        "handicap": ("asian_handicap_market_odds", "extra_market_source"),
    }[family]


def _coherent_existing_lines(
    mapping: dict[str, float],
    family: str,
    source_map: dict,
    fallback_source: str,
) -> dict[str, float]:
    """Keep complete two-sided lines whose two prices have the same provenance."""
    if family == "winner":
        picks = ("П1", "X", "П2")
        if not all(pick in mapping for pick in picks):
            return {}
        sources = {canonical_odds_source(source_map.get(pick) or fallback_source) for pick in picks}
        return {pick: mapping[pick] for pick in picks} if len(sources) == 1 and "" not in sources else {}
    if family == "btts":
        picks = ("BTTS Да", "BTTS Нет")
        if not all(pick in mapping for pick in picks):
            return {}
        sources = {canonical_odds_source(source_map.get(pick) or fallback_source) for pick in picks}
        return {pick: mapping[pick] for pick in picks} if len(sources) == 1 and "" not in sources else {}
    if family == "handicap":
        return {
            pick: price for pick, price in mapping.items()
            if canonical_odds_source(source_map.get(pick) or fallback_source)
        }
    prefixes = {
        "goals": ("ТБ ", "ТМ "),
        "corners": ("Угл ТБ ", "Угл ТМ "),
        "cards": ("Карты ТБ ", "Карты ТМ "),
    }
    over_prefix, under_prefix = prefixes[family]
    lines = {
        pick[len(over_prefix):] for pick in mapping if pick.startswith(over_prefix)
    } & {
        pick[len(under_prefix):] for pick in mapping if pick.startswith(under_prefix)
    }
    result = {}
    for line in lines:
        over_pick, under_pick = over_prefix + line, under_prefix + line
        over_source = canonical_odds_source(source_map.get(over_pick) or fallback_source)
        under_source = canonical_odds_source(source_map.get(under_pick) or fallback_source)
        if over_source and over_source == under_source:
            result[over_pick] = mapping[over_pick]
            result[under_pick] = mapping[under_pick]
    return result


def apply_provider_odds(
    snapshot: dict,
    payload: dict | None,
    source: str,
) -> dict:
    """Merge explicit provider prices without stitching markets or opposite sides."""
    if not isinstance(snapshot, dict):
        snapshot = {}
    result = dict(snapshot)
    source_key = canonical_odds_source(source)
    if source_key not in TRUSTED_ODDS_SOURCES or not isinstance(payload, dict):
        return result

    maps = _flat_market_maps(payload)
    source_map = {
        str(pick): canonical_odds_source(value)
        for pick, value in (result.get("market_odds_source_by_pick") or {}).items()
    }
    bookmakers = dict(result.get("bookmaker_by_pick") or {})
    bookmaker = str(payload.get("bookmaker") or payload.get("bookmaker_name") or "").strip()
    if bookmaker.casefold() in {"unknown", "n/a", "none", "—"}:
        bookmaker = ""

    specs = (
        ("winner", "market_odds", "odds_source"),
        ("goals", "total_market_odds", "total_odds_source"),
        ("btts", "btts_market_odds", "btts_odds_source"),
        ("corners", "stat_market_odds", "extra_market_source"),
        ("cards", "stat_market_odds", "extra_market_source"),
        ("handicap", "asian_handicap_market_odds", "extra_market_source"),
    )
    for family, field, source_field in specs:
        incoming = _complete_lines(maps.get(family) or {}, family)
        if not incoming:
            continue
        current_raw = _existing_map(result, field)
        current_source = str(result.get(source_field) or "")
        current = _coherent_existing_lines(
            current_raw, family, source_map, current_source,
        )

        if family == "winner":
            # 1X2 is atomic: one source must supply all three outcomes.
            if current:
                continue
            for pick in ("П1", "X", "П2"):
                source_map.pop(pick, None)
                bookmakers.pop(pick, None)
            current = dict(incoming)
            for pick in current:
                source_map[pick] = source_key
                if bookmaker:
                    bookmakers[pick] = bookmaker
        elif family == "btts":
            # BTTS Yes/No must be from the same source.
            if current:
                pass
            elif not any(pick in current_raw for pick in ("BTTS Да", "BTTS Нет")):
                current = dict(incoming)
                for pick in current:
                    source_map[pick] = source_key
                    if bookmaker:
                        bookmakers[pick] = bookmaker
        elif family in {"goals", "corners", "cards"}:
            # Add only complete, same-source O/U pairs. Never add only the missing
            # opposite side to an existing price from a different bookmaker.
            for pick in current_raw:
                if pick not in current:
                    source_map.pop(pick, None)
                    bookmakers.pop(pick, None)
            prefixes = {
                "goals": ("ТБ ", "ТМ "),
                "corners": ("Угл ТБ ", "Угл ТМ "),
                "cards": ("Карты ТБ ", "Карты ТМ "),
            }[family]
            over_prefix, under_prefix = prefixes
            for line in {
                pick[len(over_prefix):] for pick in incoming if pick.startswith(over_prefix)
            } & {
                pick[len(under_prefix):] for pick in incoming if pick.startswith(under_prefix)
            }:
                over_pick, under_pick = over_prefix + line, under_prefix + line
                # Any extant selection means this exact line is already owned by
                # a different/older snapshot; don't splice a second provider into it.
                if over_pick in current_raw or under_pick in current_raw:
                    continue
                current[over_pick] = incoming[over_pick]
                current[under_pick] = incoming[under_pick]
                for pick in (over_pick, under_pick):
                    source_map[pick] = source_key
                    if bookmaker:
                        bookmakers[pick] = bookmaker
        elif family == "handicap":
            # A named directional handicap is an exact selectable outcome.
            for pick, price in incoming.items():
                if pick not in current_raw:
                    current[pick] = price
                    source_map[pick] = source_key
                    if bookmaker:
                        bookmakers[pick] = bookmaker

        if family == "winner":
            result[field] = current
        elif family == "btts":
            result[field] = current
        elif family in {"goals", "corners", "cards"}:
            # Keep all existing coherent lines for this family and merge new pairs.
            result[field] = current
        elif family == "handicap":
            result[field] = current

    result["market_odds_source_by_pick"] = source_map
    result["bookmaker_by_pick"] = bookmakers

    # Backfill provenance only for legacy maps which explicitly identify a source.
    legacy_defaults = (
        ("market_odds", "odds_source"),
        ("total_market_odds", "total_odds_source"),
        ("btts_market_odds", "btts_odds_source"),
        ("stat_market_odds", "extra_market_source"),
        ("asian_handicap_market_odds", "extra_market_source"),
    )
    for field, source_field in legacy_defaults:
        fallback = canonical_odds_source(result.get(source_field))
        for pick in _existing_map(result, field):
            if pick not in source_map and fallback in TRUSTED_ODDS_SOURCES:
                source_map[pick] = fallback

    # Canonical family status can be "mixed" when separate complete lines have
    # different providers. Every displayed quote retains its own source in map.
    family_specs = (
        ("market_odds", "odds_source"),
        ("total_market_odds", "total_odds_source"),
        ("btts_market_odds", "btts_odds_source"),
        ("stat_market_odds", "extra_market_source"),
        ("asian_handicap_market_odds", "extra_market_source"),
    )
    for field, source_field in family_specs:
        mapping = _existing_map(result, field)
        picks = list(mapping)
        # Stat maps contain two independent families; both keep one group field.
        used = {
            canonical_odds_source(source_map.get(pick) or result.get(source_field))
            for pick in picks
        }
        used.discard("")
        if not picks:
            result[source_field] = "no_market"
        elif len(used) == 1:
            result[source_field] = next(iter(used))
        elif used:
            result[source_field] = "mixed"
        else:
            result[source_field] = "no_market"

    if result.get("odds_source") not in {"no_market", "mixed"}:
        result["one_x_two_diagnostics"] = (
            result.get("one_x_two_diagnostics")
            if canonical_odds_source(result.get("odds_source")) == "sstats"
            else {
                "status": "external_unvalidated",
                "reason": "Внешняя котировка доступна для анализа, но не прошла SStats bookmaker/value validation.",
            }
        )
    return result

def has_any_real_odds(card: object) -> bool:
    """True only when at least one valid price has trusted provider provenance."""
    if not isinstance(card, dict):
        return False
    groups = (
        ("market_odds", "odds_source"),
        ("total_market_odds", "total_odds_source"),
        ("btts_market_odds", "btts_odds_source"),
        ("stat_market_odds", "extra_market_source"),
        ("asian_handicap_market_odds", "extra_market_source"),
    )
    source_map = card.get("market_odds_source_by_pick")
    source_map = source_map if isinstance(source_map, dict) else {}
    for field, source_field in groups:
        mapping = card.get(field)
        if not isinstance(mapping, dict):
            continue
        for pick, raw_price in mapping.items():
            if _price(raw_price) is None:
                continue
            source = source_map.get(pick) or card.get(source_field)
            if is_trusted_odds_source(source):
                return True
    return False


def has_market_odds(card: object, family: str) -> bool:
    """True when a family has its own price; 1X2 prices never stand in for another market."""
    if not isinstance(card, dict):
        return False
    field, source_field = _field_names(family)
    mapping = card.get(field)
    if not isinstance(mapping, dict):
        return False
    source_map = card.get("market_odds_source_by_pick")
    source_map = source_map if isinstance(source_map, dict) else {}
    prefixes = {"corners": ("Угл ",), "cards": ("Карты ",), "handicap": ("Ф",)}
    for pick, raw_price in mapping.items():
        if family in prefixes and not str(pick).startswith(prefixes[family]):
            continue
        if _price(raw_price) is None:
            continue
        if is_trusted_odds_source(source_map.get(pick) or card.get(source_field)):
            return True
    return False


def collect_external_odds(
    fixture: dict,
    snapshot: dict,
    *,
    odds_api_key: str = "",
    api_football_key: str = "",
    therundown_api_key: str = "",
    highlightly_api_key: str = "",
    prognozist_enabled: bool = False,
    logs: list | None = None,
) -> dict:
    """Query each configured odds provider and add only missing market families."""
    result = apply_provider_odds(snapshot, fixture.get("odds") if isinstance(fixture, dict) else None,
                                 fixture.get("odds_source") or "fdorg")
    if not isinstance(fixture, dict):
        return result
    date_text = str(fixture.get("date_iso") or fixture.get("Date") or fixture.get("date") or "")[:10]
    home = str(fixture.get("HomeTeam") or fixture.get("home_team") or fixture.get("home") or "").strip()
    away = str(fixture.get("AwayTeam") or fixture.get("away_team") or fixture.get("away") or "").strip()
    div = str(fixture.get("Div") or fixture.get("div") or "").strip()
    league = str(fixture.get("League") or fixture.get("league") or "")
    if not (date_text and home and away):
        return result

    from data import sources

    providers = []
    if odds_api_key:
        sport_key = DIV_TO_ODDS.get(div)
        if sport_key:
            providers.append((
                "odds_api",
                lambda: sources.odds_api_fixture(sport_key, home, away, odds_api_key, logs),
            ))
        elif logs is not None:
            logs.append(f"The Odds API: для турнира {league or div or 'неизвестно'} нет sport key; источник пропущен.")
    if therundown_api_key:
        providers.append((
            "therundown",
            lambda: sources.therundown_odds_fixture(
                home, away, date_text, div, therundown_api_key, logs, league_name=league,
            ),
        ))
    if highlightly_api_key:
        providers.append((
            "highlightly",
            lambda: sources.highlightly_odds_fixture(date_text, home, away, highlightly_api_key, logs),
        ))
    if api_football_key:
        providers.append((
            "api_football",
            lambda: __import__(
                "data.api_football", fromlist=["api_football_odds_fixture"]
            ).api_football_odds_fixture(home, away, date_text, api_football_key, logs),
        ))
    if prognozist_enabled:
        providers.append((
            "prognozist_marathon",
            lambda: sources.prognozist_marathon_odds(home, away, date_text, logs),
        ))

    for source, fetch in providers:
        try:
            payload = fetch()
        except Exception as exc:
            if logs is not None:
                logs.append(f"⚠️ {SOURCE_LABELS[source]}: {type(exc).__name__}")
            continue
        if payload:
            result = apply_provider_odds(result, payload, source)
    return result
