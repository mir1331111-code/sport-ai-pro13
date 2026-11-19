"""Merge football fixture feeds without restricting coverage to one provider."""
from __future__ import annotations

import re
from datetime import date, datetime
from data.team_names import normalize_team


def _team_key(value) -> str:
    """Use the same conservative exact-alias identity as score reconciliation."""
    return normalize_team(str(value or ""))


def _fixture_day(row: dict) -> date | None:
    """Parse common provider date formats into a comparable calendar day."""
    value = row.get("Date") or row.get("date") or row.get("date_iso")
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if iso_match:
        try:
            return date.fromisoformat(iso_match.group(1))
        except ValueError:
            pass
    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(text[:10], fmt).date()
        except ValueError:
            continue
    return None


def _fixture_pair_key(row: dict) -> tuple[str, str]:
    return (
        _team_key(row.get("HomeTeam") or row.get("home_team")),
        _team_key(row.get("AwayTeam") or row.get("away_team")),
    )


def fixture_key(row: dict) -> tuple:
    """Stable date/team key across providers with different IDs/date formats."""
    day = _fixture_day(row)
    return (
        day.isoformat() if day is not None else str(
            row.get("Date") or row.get("date") or ""
        )[:10],
        *_fixture_pair_key(row),
    )


_FIXTURE_VISUAL_FIELDS = (
    "home_badge",
    "away_badge",
    "league_badge",
    "stadium_image",
    "venue_image",
    "venue",
    "stadium",
)
_FIXTURE_PROVIDER_ID_FIELDS = (
    "home_team_tsdb_id",
    "away_team_tsdb_id",
)


def enrich_fixture_visual_metadata(
    primary_rows, metadata_rows,
) -> tuple[list[dict], dict[str, int]]:
    """Overlay only visual fixture metadata without adding/replacing fixtures.

    The primary feed remains authoritative for fixture identity, kickoff,
    competition and bookmaker odds. Secondary feeds can fill image/venue fields
    on exact date/team matches, or a unique same-team match within one calendar
    day when providers disagree about UTC/local kickoff dates or date formats.
    Ambiguous near-date matches are ignored to avoid attaching another team's crest.
    """
    metadata_by_fixture: dict[tuple, list[dict]] = {}
    metadata_by_pair: dict[tuple[str, str], list[dict]] = {}
    for row in metadata_rows or []:
        if not isinstance(row, dict):
            continue
        key = fixture_key(row)
        pair = _fixture_pair_key(row)
        if all(key):
            metadata_by_fixture.setdefault(key, []).append(row)
        if all(pair) and _fixture_day(row) is not None:
            metadata_by_pair.setdefault(pair, []).append(row)

    result = []
    fixtures_matched = 0
    date_fallback_matches = 0
    ambiguous_date_matches = 0
    assets_added = 0
    fixtures_with_assets = 0
    for row in primary_rows or []:
        if not isinstance(row, dict):
            continue
        clean = dict(row)
        key = fixture_key(row)
        candidates = metadata_by_fixture.get(key, []) if all(key) else []
        if not candidates:
            primary_day = _fixture_day(row)
            pair = _fixture_pair_key(row)
            nearby = []
            if primary_day is not None and all(pair):
                for metadata_row in metadata_by_pair.get(pair, []):
                    metadata_day = _fixture_day(metadata_row)
                    if metadata_day is None:
                        continue
                    gap = abs((metadata_day - primary_day).days)
                    if gap <= 1:
                        nearby.append((gap, fixture_key(metadata_row), metadata_row))
            if nearby:
                nearest_gap = min(item[0] for item in nearby)
                nearest = [item for item in nearby if item[0] == nearest_gap]
                nearest_keys = {item[1] for item in nearest}
                if len(nearest_keys) == 1:
                    candidates = [item[2] for item in nearest]
                    date_fallback_matches += 1
                else:
                    ambiguous_date_matches += 1
        if candidates:
            fixtures_matched += 1
            # Prefer the matching provider event with the broadest visual
            # coverage when a feed contains duplicate rows for the same game.
            metadata_fields = _FIXTURE_VISUAL_FIELDS + _FIXTURE_PROVIDER_ID_FIELDS
            source = max(
                candidates,
                key=lambda candidate: sum(
                    bool(candidate.get(field)) for field in metadata_fields
                ),
            )
            for field in metadata_fields:
                if clean.get(field) in (None, "", {}, []) and source.get(field) not in (
                    None, "", {}, [],
                ):
                    clean[field] = source[field]
                    if field in _FIXTURE_VISUAL_FIELDS:
                        assets_added += 1
        if any(clean.get(field) not in (None, "", {}, []) for field in _FIXTURE_VISUAL_FIELDS):
            fixtures_with_assets += 1
        result.append(clean)

    return result, {
        "primary_fixtures": len(result),
        "metadata_fixtures": len(metadata_by_fixture),
        "fixtures_matched": fixtures_matched,
        "date_fallback_matches": date_fallback_matches,
        "ambiguous_date_matches": ambiguous_date_matches,
        "assets_added": assets_added,
        "fixtures_with_assets": fixtures_with_assets,
    }


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



def enforce_sstats_market_card(card: dict) -> dict:
    """Remove non-SStats bookmaker quotes and stale actions from saved scan cards."""
    if not isinstance(card, dict):
        return {}
    clean = dict(card)

    quote_groups = (
        ("market_odds", "odds_source"),
        ("total_market_odds", "total_odds_source"),
        ("btts_market_odds", "btts_odds_source"),
        ("stat_market_odds", "extra_market_source"),
        ("asian_handicap_market_odds", "extra_market_source"),
    )
    for odds_field, source_field in quote_groups:
        if not is_sstats_market_source(clean.get(source_field)):
            clean[odds_field] = {}
            clean[source_field] = "no_market"

    if not is_sstats_market_source(card.get("odds_source")):
        clean["bookmaker_by_pick"] = {}
    if not is_sstats_market_source(card.get("total_odds_source")):
        clean["total_market_book_count"] = {}
        clean["total_market_confirmed"] = {}

    source_by_pick = clean.get("market_odds_source_by_pick")
    if isinstance(source_by_pick, dict):
        clean["market_odds_source_by_pick"] = {
            pick: source for pick, source in source_by_pick.items()
            if is_sstats_market_source(source)
        }
    else:
        clean["market_odds_source_by_pick"] = {}

    best = clean.get("best")
    best_pick = best[1] if isinstance(best, (tuple, list)) and len(best) > 1 else None
    best_source = (
        clean.get("best_odds_source")
        or (clean.get("market_odds_source_by_pick") or {}).get(best_pick)
    )
    if best is not None and not is_sstats_market_source(best_source):
        clean["best"] = None
        clean["best_by_market"] = {}
        clean["best_odds_source"] = "no_market"
        verdict = dict(clean.get("verdict") or {})
        verdict.update({
            "real_odds": False,
            "value_passed": False,
            "is_bet": False,
            "odd": None,
            "ev": None,
            "edge": None,
            "stake": 0.0,
        })
        clean["verdict"] = verdict
    return clean


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
