"""Merge football fixture feeds without restricting coverage to one provider."""
from __future__ import annotations

import re
from datetime import date, datetime, timezone
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


def _fixture_day(row: dict):
    raw = str(row.get("Date") or row.get("date") or "").strip()
    try:
        return date.fromisoformat(raw[:10])
    except (TypeError, ValueError):
        return None


def _fixture_team_pair(row: dict) -> tuple[str, str]:
    return (
        _team_key(row.get("HomeTeam") or row.get("home_team")),
        _team_key(row.get("AwayTeam") or row.get("away_team")),
    )


def _fixture_start(row: dict) -> datetime | None:
    """Return a comparable kickoff only when the provider supplies a time."""
    for field in ("kickoff_utc", "KickoffUTC", "kickoff_at", "timestamp"):
        raw = str(row.get(field) or "").strip()
        if not raw:
            continue
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            continue
        if parsed.tzinfo is not None and parsed.utcoffset() is not None:
            return parsed.astimezone(timezone.utc)
        return parsed

    raw_date = str(row.get("Date") or row.get("date") or "").strip()
    raw_time = str(
        row.get("Time") or row.get("time") or row.get("match_time") or ""
    ).strip()
    try:
        parsed_date = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
        if parsed_date.hour or parsed_date.minute or parsed_date.second:
            return parsed_date
        if parsed_date.tzinfo is not None:
            parsed_date = parsed_date.replace(tzinfo=None)
    except ValueError:
        try:
            parsed_date = datetime.combine(date.fromisoformat(raw_date[:10]), datetime.min.time())
        except ValueError:
            return None
    if not raw_time:
        return None
    try:
        clock = datetime.strptime(raw_time[:5], "%H:%M").time()
    except ValueError:
        return None
    return datetime.combine(parsed_date.date(), clock)


def _same_cross_date_kickoff(left: dict, right: dict) -> bool:
    """Only collapse a one-day calendar drift when actual kickoff times agree."""
    left_start, right_start = _fixture_start(left), _fixture_start(right)
    if left_start is None or right_start is None:
        return False
    left_aware = left_start.tzinfo is not None and left_start.utcoffset() is not None
    right_aware = right_start.tzinfo is not None and right_start.utcoffset() is not None
    if left_aware != right_aware:
        return False
    if left_aware:
        left_start = left_start.astimezone(timezone.utc)
        right_start = right_start.astimezone(timezone.utc)
    return abs((left_start - right_start).total_seconds()) <= 6 * 60 * 60


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
    """Overlay visual fields, using a guarded one-day date fallback for timezone drift.

    The primary feed remains authoritative for fixture identity, kickoff,
    competition and bookmaker odds. Secondary feeds may enrich only a unique
    ordered home/away team pair. Exact date matches win; a +/- 1 day fallback
    is allowed only when the nearest calendar day is unambiguous.
    """
    metadata_by_fixture: dict[tuple, list[dict]] = {}
    metadata_by_pair: dict[tuple[str, str], list[tuple[date, dict]]] = {}
    for row in metadata_rows or []:
        if not isinstance(row, dict):
            continue
        key = fixture_key(row)
        if not all(key):
            continue
        metadata_by_fixture.setdefault(key, []).append(row)
        day = _fixture_day(row)
        pair = _fixture_team_pair(row)
        if day is not None and all(pair):
            metadata_by_pair.setdefault(pair, []).append((day, row))

    result = []
    fixtures_matched = 0
    date_fallback_matches = 0
    ambiguous_team_pair_matches = 0
    assets_added = 0
    fixtures_with_assets = 0
    metadata_fields = _FIXTURE_VISUAL_FIELDS + _FIXTURE_PROVIDER_ID_FIELDS

    for row in primary_rows or []:
        if not isinstance(row, dict):
            continue
        clean = dict(row)
        key = fixture_key(row)
        candidates = metadata_by_fixture.get(key, []) if all(key) else []
        if not candidates:
            pair = _fixture_team_pair(row)
            primary_day = _fixture_day(row)
            nearby = []
            if all(pair) and primary_day is not None:
                for metadata_day, metadata_row in metadata_by_pair.get(pair, []):
                    delta_days = abs((metadata_day - primary_day).days)
                    if delta_days <= 1:
                        nearby.append((delta_days, metadata_day, metadata_row))
            if nearby:
                nearest_distance = min(item[0] for item in nearby)
                nearest = [item for item in nearby if item[0] == nearest_distance]
                nearest_days = {item[1] for item in nearest}
                if len(nearest_days) == 1:
                    candidates = [item[2] for item in nearest]
                    date_fallback_matches += 1
                else:
                    ambiguous_team_pair_matches += 1

        if candidates:
            fixtures_matched += 1
            # Multiple rows at the same nearest date may be duplicate provider
            # metadata for one fixture. Fill fields from the richest rows first;
            # never copy schedules, prices, team names or competition IDs here.
            ordered_candidates = sorted(
                candidates,
                key=lambda candidate: sum(bool(candidate.get(field)) for field in metadata_fields),
                reverse=True,
            )
            for source in ordered_candidates:
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
        "ambiguous_team_pair_matches": ambiguous_team_pair_matches,
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
            if key not in merged and all(_fixture_team_pair(row)) and _fixture_day(row):
                # Provider feeds can disagree by one calendar day when one
                # emits UTC and another emits local kickoff dates. Only join
                # a shifted date if exactly one existing fixture has the same
                # ordered teams within ±1 day; ambiguous doubleheaders remain
                # separate rather than being guessed together.
                incoming_day = _fixture_day(row)
                nearby = []
                for existing_key in order:
                    existing_row = merged[existing_key]
                    existing_day = _fixture_day(existing_row)
                    if (
                        _fixture_team_pair(existing_row) == _fixture_team_pair(row)
                        and existing_day is not None
                        and 0 < abs((existing_day - incoming_day).days) <= 1
                        and _same_cross_date_kickoff(existing_row, row)
                    ):
                        nearby.append(existing_key)
                if len(nearby) == 1:
                    key = nearby[0]
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
