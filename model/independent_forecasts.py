"""Blind LLM forecasts and paired, leakage-safe evaluation against the main model."""
from __future__ import annotations

import hashlib
import math
import os
import re
import unicodedata
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
import sqlite3
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from config import APP_VERSION, DB_FILE, LLM_PROVIDERS, normalize_llm_provider
from model.forecast_ledger import ForecastLedger, MarketForecastLedger, OUTCOMES
from model.temporal import weighted_mean


def _mean(values: list[Any]) -> float | None:
    clean = []
    for value in values:
        try:
            number = float(value)
            if math.isfinite(number):
                clean.append(number)
        except (TypeError, ValueError):
            continue
    return sum(clean) / len(clean) if clean else None


def _matching_team_state(engine: Any, team: str, division: str) -> dict[str, Any]:
    """Resolve normalized names through a per-engine, per-division lookup index.

    The model state is stable after chronological training. Previously every
    independent dossier scanned every team again for each fixture; building the
    index once reduces repeated lookup work from O(fixtures × teams) to
    O(teams + fixtures).
    """
    from model.engine import _norm, _team_state_key

    teams = getattr(engine, "st", {})
    if not hasattr(teams, "items"):
        return {}
    wanted = _norm(team)
    if not wanted:
        return {}
    direct_key = _team_state_key(division, team)

    signature = (
        id(teams),
        len(teams) if hasattr(teams, "__len__") else None,
        int(getattr(engine, "trained_n", 0) or 0),
    )
    cached = getattr(engine, "_independent_team_state_index", None)
    if not isinstance(cached, dict):
        cached = {}
        try:
            setattr(engine, "_independent_team_state_index", cached)
        except Exception:
            cached = {}

    entry = cached.get(division)
    if isinstance(entry, tuple) and len(entry) == 2 and entry[0] == signature:
        return entry[1].get(wanted, {})

    index: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}
    history_fields = (
        "hs", "hc", "as", "ac", "hs_xg", "as_xg", "hs_pxg", "as_pxg",
    )
    for key, state in teams.items():
        if not isinstance(state, dict):
            continue
        if isinstance(key, tuple) and len(key) == 2:
            state_division, state_team = key
        else:
            state_division, state_team = "G", key
        if str(state_division or "G").strip().upper() != division:
            continue
        normalized = _norm(state_team)
        if not normalized:
            continue
        history_count = sum(len(state.get(field) or []) for field in history_fields)
        # Preserve the old direct-key tie-break: if two spellings normalize to
        # the same club and have equal history, prefer the requested exact key.
        rank = (history_count, int(key == direct_key))
        previous = index.get(normalized)
        if previous is None or rank > previous[0]:
            index[normalized] = (rank, state)

    resolved = {name: value[1] for name, value in index.items()}
    if cached is not None:
        cached[division] = (signature, resolved)
    return resolved.get(wanted, {})


def _history_day(value: Any) -> date | None:
    """Parse an observation day without guessing when the source date is absent."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(raw[:10])
        except ValueError:
            return None


def _history_series(
    state: dict,
    value_key: str,
    date_key: str,
    *,
    limit: int | None = None,
    match_date: datetime | date | str | None = None,
) -> tuple[list[float], list[str | None]]:
    """Return aligned observations, excluding the target day and future dates."""
    values = list(state.get(value_key) or [])
    dates = list(state.get(date_key) or [])
    cutoff = _history_day(match_date)
    pairs = []
    for index, raw_value in enumerate(values):
        try:
            number = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number):
            continue
        stamp = dates[index] if index < len(dates) else None
        observed_day = _history_day(stamp)
        if cutoff is not None and (
            observed_day is None or observed_day >= cutoff
        ):
            continue
        pairs.append((
            number,
            observed_day.isoformat() if observed_day is not None else None,
        ))
    if limit is not None:
        pairs = pairs[-max(1, int(limit)):]
    return [item[0] for item in pairs], [item[1] for item in pairs]


def _history_mean(
    state: dict,
    value_key: str,
    date_key: str,
    *,
    limit: int | None = None,
    match_date: datetime | date | str | None = None,
    half_life_days: float = 180.0,
) -> tuple[float | None, int]:
    values, dates = _history_series(
        state, value_key, date_key, limit=limit, match_date=match_date,
    )
    if not values:
        return None, 0
    # Keep historical callers without an as-of date backward-compatible. The
    # production scanner always supplies match_date and therefore gets the
    # same dated exponential weighting as the core forecast engine.
    if match_date is None or any(stamp is None for stamp in dates):
        return _mean(values), len(values)
    result = weighted_mean(
        values, dates, match_date,
        default=0.0, half_life_days=half_life_days,
    )
    return (result if math.isfinite(result) else None), len(values)


def historical_features(
    engine: Any,
    home: str,
    away: str,
    limit: int = 8,
    league: str = "G",
    match_date: datetime | date | str | None = None,
) -> dict[str, Any]:
    """Build independent observed-goal features using only pre-match dated history.

    With an as-of date, this mirrors the core engine's temporal half-life and
    excludes every observation from the target calendar day or later. Legacy
    offline callers that omit the date retain their unweighted behavior.
    """
    teams = getattr(engine, "st", {})
    division = str(league or "G").strip().upper() or "G"
    home_stats = _matching_team_state(engine, home, division)
    away_stats = _matching_team_state(engine, away, division)
    half_life = float(getattr(engine, "half_life_days", 180.0) or 180.0)

    home_scored, home_n = _history_mean(
        home_stats, "hs", "hs_dates", limit=limit,
        match_date=match_date, half_life_days=half_life,
    )
    home_conceded, home_conceded_n = _history_mean(
        home_stats, "hc", "hc_dates", limit=limit,
        match_date=match_date, half_life_days=half_life,
    )
    away_scored, away_n = _history_mean(
        away_stats, "as", "as_dates", limit=limit,
        match_date=match_date, half_life_days=half_life,
    )
    away_conceded, away_conceded_n = _history_mean(
        away_stats, "ac", "ac_dates", limit=limit,
        match_date=match_date, half_life_days=half_life,
    )
    lambda_home = (
        (home_scored + away_conceded) / 2
        if home_scored is not None and away_conceded is not None else None
    )
    lambda_away = (
        (away_scored + home_conceded) / 2
        if away_scored is not None and home_conceded is not None else None
    )
    lambda_total = (
        lambda_home + lambda_away
        if lambda_home is not None and lambda_away is not None else None
    )
    over_25 = None
    if lambda_total is not None and math.isfinite(lambda_total) and lambda_total >= 0:
        over_25 = 1 - math.exp(-lambda_total) * (
            1 + lambda_total + lambda_total**2 / 2
        )
    return {
        "home_scored_at_home_avg": home_scored,
        "home_conceded_at_home_avg": home_conceded,
        "away_scored_away_avg": away_scored,
        "away_conceded_away_avg": away_conceded,
        "home_home_sample": min(home_n, home_conceded_n),
        "away_away_sample": min(away_n, away_conceded_n),
        "sample_limit": limit,
        "as_of_date": (
            _history_day(match_date).isoformat()
            if match_date is not None and _history_day(match_date) is not None
            else None
        ),
        "simple_baseline_lambda_home": lambda_home,
        "simple_baseline_lambda_away": lambda_away,
        "simple_baseline_lambda_total": lambda_total,
        "simple_baseline_over_25": over_25,
    }


def historical_xg_features(
    engine: Any,
    home: str,
    away: str,
    league: str = "G",
    match_date: datetime | date | str | None = None,
) -> dict[str, Any]:
    """Return pre-match, time-weighted provider/proxy xG and xGA separately.

    Every target-day or future observation is excluded. Provider xG remains
    separate from the aggregate-shot proxy and is never merged into that stream.
    """
    teams = getattr(engine, "st", {})
    division = str(league or "G").strip().upper() or "G"
    home_state = _matching_team_state(engine, home, division)
    away_state = _matching_team_state(engine, away, division)
    half_life = float(getattr(engine, "half_life_days", 180.0) or 180.0)

    result: dict[str, Any] = {}
    streams = (
        ("provider", home_state, "hs_xg", "hs_xga", "hs_xg_dates", "hs_xga_dates",
         away_state, "as_xg", "as_xga", "as_xg_dates", "as_xga_dates"),
        ("proxy", home_state, "hs_pxg", "hs_pxga", "hs_pxg_dates", "hs_pxga_dates",
         away_state, "as_pxg", "as_pxga", "as_pxg_dates", "as_pxga_dates"),
    )
    for (
        label, h_state, h_for_key, h_against_key, h_for_dates, h_against_dates,
        a_state, a_for_key, a_against_key, a_for_dates, a_against_dates,
    ) in streams:
        home_for, home_n = _history_mean(
            h_state, h_for_key, h_for_dates, match_date=match_date,
            half_life_days=half_life,
        )
        home_against, home_against_n = _history_mean(
            h_state, h_against_key, h_against_dates, match_date=match_date,
            half_life_days=half_life,
        )
        away_for, away_n = _history_mean(
            a_state, a_for_key, a_for_dates, match_date=match_date,
            half_life_days=half_life,
        )
        away_against, away_against_n = _history_mean(
            a_state, a_against_key, a_against_dates, match_date=match_date,
            half_life_days=half_life,
        )
        result[label] = {
            "home_xg_for_avg": home_for,
            "home_xga_avg": home_against,
            "away_xg_for_avg": away_for,
            "away_xga_avg": away_against,
            "home_xg_samples": home_n,
            "home_xga_samples": home_against_n,
            "away_xg_samples": away_n,
            "away_xga_samples": away_against_n,
        }
    result["source_priority"] = "provider_then_proxy"
    result["as_of_date"] = (
        _history_day(match_date).isoformat()
        if match_date is not None and _history_day(match_date) is not None
        else None
    )
    return result


def _valid_probabilities(values: dict[str, Any]) -> dict[str, float] | None:
    try:
        result = {key: float(values[key]) for key in OUTCOMES}
    except (TypeError, ValueError, KeyError):
        return None
    if any(not math.isfinite(v) or v < 0 or v > 1 for v in result.values()):
        return None
    total = sum(result.values())
    if total <= 0:
        return None
    return {key: value / total for key, value in result.items()}


def parse_llm_probabilities(opinion: str) -> dict[str, float] | None:
    """Read only the validated percentage line emitted by llm.analyst."""
    if not isinstance(opinion, str):
        return None
    match = re.search(
        r"Оценка ИИ:\s*П1\s+(\d+(?:\.\d+)?)%\s*[·|]\s*X\s+"
        r"(\d+(?:\.\d+)?)%\s*[·|]\s*П2\s+(\d+(?:\.\d+)?)%",
        opinion,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    try:
        values = {"H": float(match.group(1)) / 100.0,
                  "D": float(match.group(2)) / 100.0,
                  "A": float(match.group(3)) / 100.0}
    except (TypeError, ValueError):
        return None
    return _valid_probabilities(values)


def _team_key(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-zа-яё0-9]+", " ", text).strip()


def _fixture_identity(card: dict) -> tuple[str, str, str, str] | None:
    date_iso = str(card.get("date_iso") or "")[:10]
    try:
        datetime.strptime(date_iso, "%Y-%m-%d")
    except ValueError:
        return None
    match = str(card.get("match") or "")
    if " vs " in match:
        home, away = match.split(" vs ", 1)
    else:
        match_ru = str(card.get("match_ru") or "")
        parts = match_ru.split(" — ")
        if len(parts) != 2:
            return None
        home, away = parts
    home, away = home.strip(), away.strip()
    if not home or not away:
        return None
    league = str(card.get("div") or card.get("league") or "")
    digest = hashlib.sha256(
        f"{date_iso}|{league}|{_team_key(home)}|{_team_key(away)}".encode("utf-8")
    ).hexdigest()[:24]
    # Embedding the local fixture date keeps result lookup correct after UTC conversion.
    return f"ind:{date_iso}:{digest}", home, away, date_iso


def _timezone(name: str | None = None):
    tz_name = (name or os.environ.get("APP_TIMEZONE") or "Europe/Warsaw").strip()
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return timezone.utc


def _kickoff(card: dict, tz_name: str | None = None) -> datetime | None:
    date_iso = str(card.get("date_iso") or "")[:10]
    time_text = str(card.get("time") or "").strip()
    match = re.search(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)", time_text)
    if not match:
        return None
    try:
        return datetime.strptime(
            f"{date_iso} {int(match.group(1)):02d}:{match.group(2)}",
            "%Y-%m-%d %H:%M",
        ).replace(tzinfo=_timezone(tz_name))
    except ValueError:
        return None


def _odds_from_card(card: dict) -> dict[str, float] | None:
    source = card.get("market_odds")
    if not isinstance(source, dict):
        return None
    result = {}
    for src, dst in (("П1", "H"), ("X", "D"), ("П2", "A")):
        try:
            price = float(source[src])
        except (TypeError, ValueError, KeyError):
            continue
        if math.isfinite(price) and price > 1:
            result[dst] = price
    return result or None


def _main_probabilities(card: dict) -> dict[str, float] | None:
    try:
        raw = {"H": float(card["p1"]), "D": float(card["px"]), "A": float(card["p2"])}
    except (KeyError, TypeError, ValueError):
        return None
    return _valid_probabilities(raw)


def record_scan_forecasts(
    cards: list[dict], *, provider: str = "", model: str = "",
    db_path: str | Path = DB_FILE, timezone_name: str | None = None,
    now: datetime | None = None,
) -> dict[str, int]:
    """Save one immutable pre-kickoff forecast per fixture/model; repeated scans are safe."""
    stats = {"main_recorded": 0, "ai_recorded": 0, "market_recorded": 0,
             "skipped_no_time": 0, "skipped_started": 0, "invalid": 0}
    ledger = ForecastLedger(db_path)
    market_ledger = MarketForecastLedger(db_path)
    # Read only idempotency keys. Decoding every historical JSON forecast row
    # on every scan was unnecessary and increasingly expensive as the ledger grew.
    existing = ledger.keys()
    existing_market = market_ledger.keys()
    pending_main: list[dict[str, Any]] = []
    pending_ai: list[dict[str, Any]] = []
    pending_market: list[dict[str, Any]] = []
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    now = now.astimezone(timezone.utc)
    main_version = f"main-model-v{APP_VERSION}"
    # Keep ledger writes compatible with provider labels from older scans.
    resolved_provider = normalize_llm_provider(
        provider or "Groq (бесплатно, быстро)"
    )
    configured_model = str(LLM_PROVIDERS[resolved_provider].get("model") or "")
    requested_model = str(model or "").strip()
    resolved_model = (
        requested_model if requested_model == configured_model else configured_model
    )
    slug = re.sub(
        r"[^a-z0-9]+", "-", f"{resolved_provider}-{resolved_model}".casefold()
    ).strip("-")[:80]
    ai_version = f"independent-llm-{slug or 'configured-model'}"

    for card in cards or []:
        if not isinstance(card, dict):
            continue
        identity = _fixture_identity(card)
        if identity is None:
            stats["invalid"] += 1
            continue
        fixture_id, home, away, date_iso = identity
        kickoff = _kickoff(card, timezone_name)
        if kickoff is None:
            stats["skipped_no_time"] += 1
            continue
        if kickoff.astimezone(timezone.utc) <= now:
            stats["skipped_started"] += 1
            continue
        prices = _odds_from_card(card)
        league = str(card.get("league") or card.get("div") or "")
        main_probs = _main_probabilities(card)
        common = dict(
            fixture_id=fixture_id, home_team=home, away_team=away,
            forecast_at=now, kickoff_at=kickoff, league=league,
            entry_odds=prices, odds_source=str(card.get("odds_source") or "") or None,
        )
        if main_probs is not None:
            key = f"{fixture_id}|{main_version}"
            if key not in existing:
                pending_main.append({
                    "forecast_key": key,
                    "model_version": main_version,
                    "probabilities": main_probs,
                    "xg_used": bool(card.get("xg_used")),
                    **common,
                })
                existing.add(key)

        ai_probs = parse_llm_probabilities(card.get("llm_opinion") or "")
        if ai_probs is not None:
            key = f"{fixture_id}|{ai_version}"
            if key not in existing:
                pending_ai.append({
                    "forecast_key": key,
                    "model_version": ai_version,
                    "probabilities": ai_probs,
                    **common,
                })
                existing.add(key)

        # Keep one immutable audit row for every independently selected market,
        # not just the global best pick for the fixture.
        recommendations = card.get("best_by_market")
        if isinstance(recommendations, dict) and recommendations:
            keys_by_market = card.setdefault("market_forecast_keys", {})
            for family, recommendation in recommendations.items():
                if not isinstance(recommendation, dict):
                    continue
                best = recommendation.get("best")
                if not isinstance(best, (tuple, list)) or len(best) < 5:
                    continue
                try:
                    market, selection = str(best[0]).upper().strip(), str(best[1]).strip()
                    price, probability = float(best[2]), float(best[4])
                    # One immutable recommendation per fixture, model version and market family.
                    # Including scan time here would create duplicate settled rows on every rerun.
                    key = (
                        f"{fixture_id}|{main_version}|market|"
                        f"{str(family).casefold()}"
                    )
                    keys_by_market[str(family)] = key
                    if key not in existing_market:
                        pending_market.append({
                            "forecast_key": key,
                            "fixture_id": fixture_id,
                            "home_team": home,
                            "away_team": away,
                            "league": league,
                            "model_version": main_version,
                            "market": market,
                            "selection": selection,
                            "probability": probability,
                            "entry_odds": price,
                            "forecast_at": now,
                            "kickoff_at": kickoff,
                            "odds_source": str(
                                card.get("market_odds_source_by_pick", {}).get(selection)
                                or card.get("best_odds_source") or card.get("odds_source") or ""
                            ) or None,
                            "unit_stake": 1.0,
                        })
                        existing_market.add(key)
                    # Preserve the legacy scalar key only for the global best pick.
                    global_best = card.get("best")
                    if (
                        isinstance(global_best, (tuple, list)) and len(global_best) >= 2
                        and str(global_best[1]).strip() == selection
                    ):
                        card["market_forecast_key"] = key
                except (ValueError, TypeError, IndexError):
                    stats["invalid"] += 1
        else:
            # Backward compatibility for older saved scan cards and callers.
            best = card.get("best")
            if isinstance(best, (tuple, list)) and len(best) >= 5:
                try:
                    market, selection = str(best[0]).upper().strip(), str(best[1]).strip()
                    price, probability = float(best[2]), float(best[4])
                    # Backward-compatible single-pick path also needs a stable idempotency key.
                    key = f"{fixture_id}|{main_version}|market|legacy_best"
                    card["market_forecast_key"] = key
                    if key not in existing_market:
                        pending_market.append({
                            "forecast_key": key,
                            "fixture_id": fixture_id,
                            "home_team": home,
                            "away_team": away,
                            "league": league,
                            "model_version": main_version,
                            "market": market,
                            "selection": selection,
                            "probability": probability,
                            "entry_odds": price,
                            "forecast_at": now,
                            "kickoff_at": kickoff,
                            "odds_source": str(
                                card.get("best_odds_source") or card.get("odds_source") or ""
                            ) or None,
                            "unit_stake": 1.0,
                        })
                        existing_market.add(key)
                except (ValueError, TypeError, IndexError):
                    stats["invalid"] += 1

    def _flush_batch(
        pending: list[dict[str, Any]],
        batch_writer,
        single_writer,
        counter: str,
    ) -> None:
        if not pending:
            return
        try:
            inserted = batch_writer(pending)
            stats[counter] += len(inserted)
        except (ValueError, TypeError):
            # Preserve per-record error accounting if one malformed legacy card
            # prevents a whole batch from passing its immutable-payload checks.
            for record in pending:
                try:
                    single_writer(**record)
                    stats[counter] += 1
                except (ValueError, TypeError):
                    stats["invalid"] += 1

    _flush_batch(
        pending_main, ledger.record_predictions_batch,
        ledger.record_prediction, "main_recorded",
    )
    _flush_batch(
        pending_ai, ledger.record_predictions_batch,
        ledger.record_prediction, "ai_recorded",
    )
    _flush_batch(
        pending_market, market_ledger.record_recommendations_batch,
        market_ledger.record_recommendation, "market_recorded",
    )
    return stats


def _load_result_lookup_state(db_path: str | Path) -> dict[str, dict[str, str]]:
    """Load retry deadlines and last-attempt timestamps for each fixture."""
    if str(db_path) == ":memory:":
        return {}
    with sqlite3.connect(str(db_path), timeout=15) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS forecast_result_checks (
                fixture_id TEXT PRIMARY KEY,
                checked_at TEXT NOT NULL,
                retry_after TEXT NOT NULL,
                last_status TEXT
            )"""
        )
        rows = conn.execute(
            "SELECT fixture_id, checked_at, retry_after, last_status "
            "FROM forecast_result_checks"
        ).fetchall()
    return {
        str(fixture_id): {
            "checked_at": str(checked_at or ""),
            "retry_after": str(retry_after or ""),
            "last_status": str(last_status or ""),
        }
        for fixture_id, checked_at, retry_after, last_status in rows
    }


def _save_result_lookup_state(
    db_path: str | Path, fixture_id: str, now: datetime,
    retry_after: datetime, status: str,
) -> None:
    """Persist a bounded retry schedule for this fixture's score lookup."""
    if str(db_path) == ":memory:":
        return
    with sqlite3.connect(str(db_path), timeout=15) as conn:
        conn.execute(
            """INSERT INTO forecast_result_checks
                (fixture_id, checked_at, retry_after, last_status)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(fixture_id) DO UPDATE SET
                checked_at=excluded.checked_at,
                retry_after=excluded.retry_after,
                last_status=excluded.last_status""",
            (fixture_id, now.isoformat(), retry_after.isoformat(), status),
        )


def settle_due_forecasts(
    *, db_path: str | Path = DB_FILE, timezone_name: str | None = None,
    now: datetime | None = None, max_fixtures: int = 20,
    max_age_days: int = 365, token: str | None = None,
    sstats_api_key: str | None = None, force: bool = False,
) -> dict[str, int]:
    """Settle verified FT scores, including a bounded historical backlog.

    Wait 2h15 after kickoff, check oldest fixtures first, pass the caller's
    football-data.org token explicitly, and persist retry deadlines. A manual
    force check bypasses retry deadlines but keeps provider-call limits. The
    default historical window is 365 days rather than silently abandoning
    forecasts after 14 days.
    """
    ledger = ForecastLedger(db_path)
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    now_utc = now.astimezone(timezone.utc)
    local_today = now_utc.astimezone(_timezone(timezone_name)).date()
    token = str(token or "").strip()
    sstats_api_key = str(sstats_api_key or "").strip()
    max_age_days = max(0, int(max_age_days))
    grace = timedelta(hours=2, minutes=15)
    missing_retry = timedelta(hours=6)
    force_retry_cooldown = timedelta(minutes=10)
    non_ft_retry = timedelta(days=1)
    confirmed_retry = timedelta(days=365)
    lookup_state = _load_result_lookup_state(db_path)

    ledger_rows = ledger.rows()
    pending_forecast_rows = [
        row for row in ledger_rows if row.get("result") is None
    ]
    pending_forecasts = len(pending_forecast_rows)
    groups: dict[str, list[dict]] = defaultdict(list)
    age_limited_fixture_ids: set[str] = set()
    pending_unique_fixture_ids: set[str] = set()
    not_due_grace_fixture_ids: set[str] = set()
    invalid_fixture_ids = 0
    future_fixtures = 0
    for row in pending_forecast_rows:
        fixture_id = str(row.get("fixture_id") or "")
        if fixture_id:
            pending_unique_fixture_ids.add(fixture_id)
        if not row.get("kickoff_at"):
            invalid_fixture_ids += 1
            continue
        try:
            kickoff = datetime.fromisoformat(
                str(row["kickoff_at"]).replace("Z", "+00:00")
            )
            if kickoff.tzinfo is None:
                invalid_fixture_ids += 1
                continue
            if kickoff.astimezone(timezone.utc) + grace > now_utc:
                if fixture_id:
                    not_due_grace_fixture_ids.add(fixture_id)
                continue
            parts = fixture_id.split(":", 2)
            if len(parts) < 3 or parts[0] != "ind":
                invalid_fixture_ids += 1
                continue
            date_iso = parts[1]
            date_obj = datetime.strptime(date_iso, "%Y-%m-%d").date()
            age = (local_today - date_obj).days
            if age < 0:
                future_fixtures += 1
                continue
            if age > max_age_days:
                age_limited_fixture_ids.add(fixture_id)
                continue
            groups[fixture_id].append(row)
        except (ValueError, TypeError):
            invalid_fixture_ids += 1

    fixtures = sorted(
        groups.items(),
        key=lambda pair: min(str(row.get("kickoff_at") or "") for row in pair[1]),
    )
    market_ledger = MarketForecastLedger(db_path)
    market_rows = market_ledger.rows()
    pending_market_rows = [
        row for row in market_rows
        if row.get("result_status") is None
        and str(row.get("market") or "").upper() != "STAT"
    ]
    stat_market_pending = sum(
        row.get("result_status") is None
        and str(row.get("market") or "").upper() == "STAT"
        for row in market_rows
    )
    market_groups: dict[str, list[dict]] = defaultdict(list)
    market_age_limited_fixtures = set()
    for row in pending_market_rows:
        if not row.get("kickoff_at"):
            continue
        try:
            kickoff = datetime.fromisoformat(
                str(row["kickoff_at"]).replace("Z", "+00:00")
            )
            if kickoff.tzinfo is None or kickoff.astimezone(timezone.utc) + grace > now_utc:
                continue
            fixture_id = str(row.get("fixture_id") or "")
            parts = fixture_id.split(":", 2)
            if len(parts) < 3 or parts[0] != "ind":
                continue
            date_obj = datetime.strptime(parts[1], "%Y-%m-%d").date()
            age = (local_today - date_obj).days
            if age < 0 or age > max_age_days:
                if age > max_age_days:
                    market_age_limited_fixtures.add(fixture_id)
                continue
            market_groups[fixture_id].append(row)
        except (ValueError, TypeError):
            continue

    looked_up = settled = errors = 0
    market_checked = market_settled = 0
    retry_deferred_fixtures = not_found_fixtures = non_ft_fixtures = 0
    limit_deferred_fixtures = 0
    source_counts: dict[str, int] = defaultdict(int)
    sstats_lookups = 0
    score_cache: dict[str, dict | None] = {}
    if fixtures or market_groups:
        from data.scores import find_match_score
        from data.sstats import sstats_match_result

        def _score_for_fixture(fixture_id: str, sample: dict, date_iso: str):
            nonlocal looked_up, errors, retry_deferred_fixtures
            nonlocal not_found_fixtures, non_ft_fixtures, limit_deferred_fixtures
            nonlocal sstats_lookups
            if fixture_id in score_cache:
                return score_cache[fixture_id]
            retry_info = lookup_state.get(fixture_id) or {}
            retry_value = retry_info.get("retry_after")
            if retry_value:
                try:
                    retry_at = datetime.fromisoformat(
                        retry_value.replace("Z", "+00:00")
                    )
                except ValueError:
                    retry_at = now_utc
                if retry_at.tzinfo is not None and retry_at > now_utc:
                    force_cooldown_active = False
                    if force:
                        checked_value = retry_info.get("checked_at") or ""
                        try:
                            checked_at = datetime.fromisoformat(
                                checked_value.replace("Z", "+00:00")
                            )
                        except ValueError:
                            checked_at = now_utc
                        force_cooldown_active = (
                            checked_at.tzinfo is not None
                            and checked_at + force_retry_cooldown > now_utc
                        )
                    if not force or force_cooldown_active:
                        retry_deferred_fixtures += 1
                        score_cache[fixture_id] = None
                        return None
            if looked_up >= max(0, int(max_fixtures)):
                limit_deferred_fixtures += 1
                score_cache[fixture_id] = None
                return None
            looked_up += 1
            home_team = str(sample["home_team"])
            away_team = str(sample["away_team"])
            score = None
            # SStats is the scanner's primary fixture source. Resolve its
            # completed result against the exact local date and ordered teams
            # before falling back to broad public score feeds.
            if sstats_api_key:
                sstats_lookups += 1
                try:
                    score = sstats_match_result(
                        date_iso, home_team, away_team, sstats_api_key,
                    )
                except Exception:
                    errors += 1
                    score = None
            if score is None:
                try:
                    score = find_match_score(
                        home_team, away_team, date_iso,
                        token=token or None,
                    )
                except Exception:
                    errors += 1
                    score = None
            status = str((score or {}).get("status") or "").upper()
            if score:
                source_name = str(score.get("source") or "unknown").strip().casefold()
                source_counts[source_name] += 1
            regulation_ft = bool(
                status == "AET" and (score or {}).get("regulation_time") is True
            )
            settleable_score = status == "FT" or regulation_ft
            if score is None:
                not_found_fixtures += 1
            elif not settleable_score:
                non_ft_fixtures += 1
            try:
                if settleable_score:
                    int(score["home"])
                    int(score["away"])
                    delay = confirmed_retry
                elif score:
                    delay = non_ft_retry
                else:
                    delay = missing_retry
            except (KeyError, TypeError, ValueError):
                delay = missing_retry
            next_check = now_utc + delay
            _save_result_lookup_state(
                db_path, fixture_id, now_utc, next_check, status or "NOT_FOUND",
            )
            lookup_state[fixture_id] = {
                "checked_at": now_utc.isoformat(),
                "retry_after": next_check.isoformat(),
                "last_status": status or "NOT_FOUND",
            }
            score_cache[fixture_id] = score
            return score

        for fixture_id, rows in fixtures:
            sample = rows[0]
            date_iso = fixture_id.split(":", 2)[1]
            score = _score_for_fixture(fixture_id, sample, date_iso)
            if not score:
                continue
            score_status = str(score.get("status") or "").upper()
            if score_status != "FT" and not (
                score_status == "AET" and score.get("regulation_time") is True
            ):
                continue
            try:
                home_score, away_score = int(score["home"]), int(score["away"])
                result = "H" if home_score > away_score else "A" if home_score < away_score else "D"
                for row in rows:
                    if row.get("result") is None:
                        ledger.settle(
                            str(row["forecast_key"]), result,
                            home_score=home_score, away_score=away_score,
                            settled_at=now_utc,
                            result_source=str(score.get("source") or "verified_ft_score"),
                        )
                        settled += 1
            except (KeyError, TypeError, ValueError):
                errors += 1

        for fixture_id, rows in sorted(
            market_groups.items(),
            key=lambda pair: min(str(row.get("kickoff_at") or "") for row in pair[1]),
        ):
            sample = rows[0]
            market_checked += 1
            score = _score_for_fixture(
                fixture_id, sample, fixture_id.split(":", 2)[1],
            )
            if not score:
                continue
            score_status = str(score.get("status") or "").upper()
            if score_status != "FT" and not (
                score_status == "AET" and score.get("regulation_time") is True
            ):
                continue
            try:
                hg, ag = int(score["home"]), int(score["away"])
                from betting.settlement import determine_outcome
                for row in rows:
                    outcome, _reason = determine_outcome(
                        str(row["market"]), str(row["selection"]), hg, ag
                    )
                    if outcome is not None:
                        market_ledger.settle(
                            str(row["forecast_key"]), outcome, result_score=f"{hg}:{ag}",
                            settlement_source=str(score.get("source") or "verified_ft_score"),
                            settled_at=now_utc,
                        )
                        market_settled += 1
            except (KeyError, TypeError, ValueError):
                errors += 1

    # Re-read the ledger after settlement so the UI can distinguish the
    # queue size at the beginning of this run from fixtures still awaiting a
    # verified result afterwards.
    remaining_fixture_ids: set[str] = set()
    for row in ledger.rows():
        if row.get("result") is not None or not row.get("kickoff_at"):
            continue
        try:
            kickoff = datetime.fromisoformat(
                str(row["kickoff_at"]).replace("Z", "+00:00")
            )
            if kickoff.tzinfo is None or kickoff.astimezone(timezone.utc) + grace > now_utc:
                continue
            fixture_id = str(row.get("fixture_id") or "")
            parts = fixture_id.split(":", 2)
            if len(parts) < 3 or parts[0] != "ind":
                continue
            fixture_date = datetime.strptime(parts[1], "%Y-%m-%d").date()
            age = (local_today - fixture_date).days
            if 0 <= age <= max_age_days:
                remaining_fixture_ids.add(fixture_id)
        except (TypeError, ValueError):
            continue

    return {
        "fixtures_checked": looked_up,
        "forecasts_settled": settled,
        "pending_forecasts": max(0, pending_forecasts - settled),
        "pending_unique_fixtures": len(pending_unique_fixture_ids),
        "not_due_grace_fixtures": len(not_due_grace_fixture_ids),
        "eligible_fixtures": len(fixtures),
        "eligible_fixtures_remaining": len(remaining_fixture_ids),
        "age_limited_fixtures": len(age_limited_fixture_ids),
        "invalid_fixture_ids": invalid_fixture_ids,
        "future_fixtures": future_fixtures,
        "retry_deferred_fixtures": retry_deferred_fixtures,
        "lookup_limit_deferred_fixtures": limit_deferred_fixtures,
        "not_found_fixtures": not_found_fixtures,
        "non_ft_fixtures": non_ft_fixtures,
        "sstats_lookups": sstats_lookups,
        "result_sources": dict(source_counts),
        "market_pending_fixtures": len(market_groups),
        "market_age_limited_fixtures": len(market_age_limited_fixtures),
        "stat_market_pending": stat_market_pending,
        "market_fixtures_checked": market_checked,
        "market_forecasts_settled": market_settled,
        "errors": errors,
    }


def _metrics(rows: list[dict]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "accuracy": None, "brier": None, "log_loss": None}
    briers, losses, correct = [], [], 0
    for row in rows:
        probs = row["probabilities"]
        actual = row["result"]
        briers.append(sum((float(probs[key]) - float(key == actual)) ** 2 for key in OUTCOMES))
        losses.append(-math.log(max(float(probs[actual]), 1e-15)))
        prediction = max(OUTCOMES, key=lambda key: probs[key])
        correct += int(prediction == actual)
    n = len(rows)
    return {"n": n, "accuracy": correct / n, "brier": sum(briers) / n,
            "log_loss": sum(losses) / n}


def _pearson_correlation(left: list[float], right: list[float]) -> float | None:
    """Pearson correlation between paired per-fixture error contributions."""
    if len(left) != len(right) or len(left) < 3:
        return None
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    centered_left = [value - mean_left for value in left]
    centered_right = [value - mean_right for value in right]
    variance_left = sum(value * value for value in centered_left)
    variance_right = sum(value * value for value in centered_right)
    denominator = math.sqrt(variance_left * variance_right)
    if denominator <= 1e-15:
        return None
    value = sum(a * b for a, b in zip(centered_left, centered_right)) / denominator
    return max(-1.0, min(1.0, value))


def compare_forecast_rows(rows: list[dict]) -> dict[str, Any]:
    """Compare LLM and main model only on identical fixtures settled by a verified score."""
    main = [r for r in rows if str(r.get("model_version") or "").startswith("main-model-")]
    ai = [r for r in rows if str(r.get("model_version") or "").startswith("independent-llm-")]
    settled_main = [r for r in main if r.get("result") in OUTCOMES]
    settled_ai = [r for r in ai if r.get("result") in OUTCOMES]
    main_by_fixture: dict[str, list[dict]] = defaultdict(list)
    for row in settled_main:
        main_by_fixture[str(row.get("fixture_id") or "")].append(row)

    versions = sorted({str(r["model_version"]) for r in ai})
    model_reports = []
    for version in versions:
        pairs = []
        for ai_row in settled_ai:
            if ai_row.get("model_version") != version:
                continue
            candidates = main_by_fixture.get(str(ai_row.get("fixture_id") or ""), [])
            if not candidates:
                continue
            # If a deployment recorded multiple releases for a fixture, pair with the
            # forecast closest in time, rather than silently pairing unrelated dates.
            ai_at = str(ai_row.get("forecast_at") or "")
            main_row = min(candidates, key=lambda r: abs(
                _timestamp_distance(ai_at, str(r.get("forecast_at") or ""))
            ))
            pairs.append((main_row, ai_row))
        main_metrics = _metrics([p[0] for p in pairs])
        ai_metrics = _metrics([p[1] for p in pairs])
        reports = []
        main_log_losses, ai_log_losses = [], []
        main_brier_losses, ai_brier_losses = [], []
        for main_row, ai_row in sorted(pairs, key=lambda p: str(p[1].get("kickoff_at") or ""), reverse=True):
            main_probs = main_row["probabilities"]
            ai_probs = ai_row["probabilities"]
            actual = str(ai_row.get("result") or "")
            if actual in OUTCOMES:
                main_log_losses.append(-math.log(max(float(main_probs[actual]), 1e-15)))
                ai_log_losses.append(-math.log(max(float(ai_probs[actual]), 1e-15)))
                main_brier_losses.append(sum(
                    (float(main_probs[key]) - float(key == actual)) ** 2 for key in OUTCOMES
                ))
                ai_brier_losses.append(sum(
                    (float(ai_probs[key]) - float(key == actual)) ** 2 for key in OUTCOMES
                ))
            mp, ap = main_row["probabilities"], ai_row["probabilities"]
            m_pick = max(OUTCOMES, key=lambda k: mp[k])
            a_pick = max(OUTCOMES, key=lambda k: ap[k])
            reports.append({
                "date": str(ai_row.get("fixture_id") or "").split(":", 2)[1:2][0]
                        if ":" in str(ai_row.get("fixture_id") or "") else "",
                "home": ai_row.get("home_team", ""), "away": ai_row.get("away_team", ""),
                "result": ai_row.get("result", ""),
                "main_pick": m_pick, "ai_pick": a_pick,
                "main_correct": m_pick == ai_row.get("result"),
                "ai_correct": a_pick == ai_row.get("result"),
                "main_probs": mp, "ai_probs": ap,
            })
        model_reports.append({
            "model_version": version, "n": len(pairs),
            "main": main_metrics, "ai": ai_metrics,
            "accuracy_delta": (
                ai_metrics["accuracy"] - main_metrics["accuracy"]
                if pairs else None
            ),
            "brier_delta": (
                ai_metrics["brier"] - main_metrics["brier"] if pairs else None
            ),
            "log_loss_delta": (
                ai_metrics["log_loss"] - main_metrics["log_loss"] if pairs else None
            ),
            "error_correlation": {
                "n": len(main_log_losses),
                "log_loss": _pearson_correlation(main_log_losses, ai_log_losses),
                "brier": _pearson_correlation(main_brier_losses, ai_brier_losses),
                "interpretation": (
                    "Correlation of per-fixture error contributions; high positive values "
                    "mean the systems tend to make mistakes on the same fixtures. This is "
                    "a diagnostic, not an ensemble or proof of independence."
                ),
            },
            "recent_pairs": reports[:30],
        })
    model_reports.sort(key=lambda x: (x["n"], x["model_version"]), reverse=True)
    return {
        "main_forecasts": len(main), "ai_forecasts": len(ai),
        "main_settled": len(settled_main), "ai_settled": len(settled_ai),
        "models": model_reports,
        "best_paired": model_reports[0] if model_reports else None,
        "paired_n": max((report["n"] for report in model_reports), default=0),
        "unsettled_due": sum(
            1 for r in rows
            if r.get("result") is None and r.get("kickoff_at")
            and str(r.get("model_version") or "").startswith(("main-model-", "independent-llm-"))
            and _is_kickoff_due(r.get("kickoff_at"))
        ),
    }


def _is_kickoff_due(value: Any) -> bool:
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            return False
        return stamp.astimezone(timezone.utc) <= datetime.now(timezone.utc)
    except (ValueError, TypeError):
        return False


def _timestamp_distance(left: str, right: str) -> float:
    try:
        a = datetime.fromisoformat(left.replace("Z", "+00:00"))
        b = datetime.fromisoformat(right.replace("Z", "+00:00"))
        return abs((a - b).total_seconds())
    except (ValueError, TypeError):
        return float("inf")


def comparison_report(db_path: str | Path = DB_FILE) -> dict[str, Any]:
    ledger = ForecastLedger(db_path)
    return compare_forecast_rows(ledger.rows())
