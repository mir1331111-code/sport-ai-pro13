"""Fixed, reproducible 1X2 holdout evaluation for Model Lab v1.

The evaluation is prequential: model parameters and feature policy are frozen,
each calendar day's forecasts are produced before any result from that day is
ingested, and prior holdout results may update the model state exactly as they
would in live chronological operation. Holdout results must not be used for
manual tuning or parameter selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import tempfile
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from model.engine import Engine, _row_xg_pair
from model.walk_forward import (
    _ClusteredRecords,
    _get,
    _metrics,
    _paired_differences,
    _parse_date,
    _season_label,
    _strict_score,
    _without_xg,
)

MAX_HOLDOUT_CSV_BYTES = 100 * 1024 * 1024
_REQUIRED_CSV_ALIASES = (
    ("Date", "date", "match_date", "kickoff_date"),
    ("Div", "league", "competition"),
    ("HomeTeam", "home_team", "home"),
    ("AwayTeam", "away_team", "away"),
    ("FTHG", "home_goals", "hg"),
    ("FTAG", "away_goals", "ag"),
)


def validate_historical_rows(rows: object) -> tuple[list[dict], list[str]]:
    """Validate rows loaded from the durable app dataset store."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("В сохранённом архиве нет строк матчей.")
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("Сохранённый архив содержит некорректную строку.")
    clean = [dict(row) for row in rows]
    fields = sorted({str(key).strip() for row in clean for key in row})
    missing = [
        "/".join(aliases)
        for aliases in _REQUIRED_CSV_ALIASES
        if not any(alias in fields for alias in aliases)
    ]
    if missing:
        raise ValueError("В архиве не хватает обязательных колонок: " + ", ".join(missing))
    return clean, fields


def read_historical_csv_bytes(raw: bytes) -> tuple[list[dict[str, str]], list[str]]:
    """Parse and validate an uploaded historical football CSV without guessing columns."""
    if not isinstance(raw, (bytes, bytearray)):
        raise ValueError("Загруженный CSV должен быть двоичным файлом.")
    if not raw:
        raise ValueError("CSV-файл пуст.")
    if len(raw) > MAX_HOLDOUT_CSV_BYTES:
        raise ValueError(
            f"Размер CSV превышает лимит {MAX_HOLDOUT_CSV_BYTES // (1024 * 1024)} МБ."
        )
    try:
        decoded = bytes(raw).decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("CSV должен быть сохранён в UTF-8 или UTF-8 с BOM.") from exc
    reader = csv.DictReader(io.StringIO(decoded, newline=""))
    fields = [str(field or "").strip() for field in (reader.fieldnames or [])]
    reader.fieldnames = fields
    if not fields:
        raise ValueError("В CSV отсутствует строка заголовков.")
    missing = [
        "/".join(aliases)
        for aliases in _REQUIRED_CSV_ALIASES
        if not any(alias in fields for alias in aliases)
    ]
    if missing:
        raise ValueError("В CSV не хватает обязательных колонок: " + ", ".join(missing))
    rows = [
        dict(row) for row in reader
        if row and any(value not in (None, "") for value in row.values())
    ]
    if not rows:
        raise ValueError("В CSV нет строк матчей.")
    validated, _validated_fields = validate_historical_rows(rows)
    return validated, fields


TOP_FIVE_LEAGUES = {
    "E0": "English Premier League",
    "D1": "German Bundesliga",
    "I1": "Italian Serie A",
    "SP1": "Spanish La Liga",
    "F1": "French Ligue 1",
}

_LEAGUE_ALIASES = {
    "e0": "E0", "premier league": "E0", "english premier league": "E0",
    "d1": "D1", "bundesliga": "D1", "german bundesliga": "D1",
    "i1": "I1", "serie a": "I1", "italian serie a": "I1",
    "sp1": "SP1", "la liga": "SP1", "laliga": "SP1", "spanish la liga": "SP1",
    "f1": "F1", "ligue 1": "F1", "french ligue 1": "F1",
}

# One fixed bookmaker per run. Average-price columns are intentionally omitted:
# they are not a single bookmaker and must not be described as one.
MARKET_SOURCES = {
    "B365": {
        "label": "Bet365",
        "columns": ("B365H", "B365D", "B365A"),
    },
    "Pinnacle": {
        "label": "Pinnacle",
        "columns": ("PSH", "PSD", "PSA"),
    },
    "WilliamHill": {
        "label": "William Hill",
        "columns": ("WHH", "WHD", "WHA"),
    },
    "Bwin": {
        "label": "Bwin",
        "columns": ("BWH", "BWD", "BWA"),
    },
}

DEFAULT_THRESHOLDS = {
    "min_holdout_matches": 3000,
    "min_matches_per_league": 400,
    "min_matchday_clusters": 30,
    "min_market_paired_matches": 3000,
    "max_multiclass_brier": 0.57,
    "max_log_loss": 0.99,
    "max_market_log_loss_regression": 0.02,
}


def _canonical_league(value: Any) -> str | None:
    if value is None:
        return None
    key = " ".join(str(value).strip().casefold().split())
    return _LEAGUE_ALIASES.get(key)


def _date_arg(value: str, name: str) -> date:
    parsed = _parse_date(value)
    if parsed is None:
        raise ValueError(f"{name} must be an ISO date (YYYY-MM-DD)")
    return parsed


def _strict_market_probs(row: Mapping[str, Any], source: str) -> dict[str, float] | None:
    """Proportional de-vig for exactly one bookmaker's complete 1X2 triplet."""
    columns = MARKET_SOURCES[source]["columns"]
    try:
        odds = [float(row.get(column)) for column in columns]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) and value > 1.0 for value in odds):
        return None
    implied = [1.0 / value for value in odds]
    total = sum(implied)
    if not math.isfinite(total) or total <= 0:
        return None
    probabilities = [value / total for value in implied]
    return dict(zip(("H", "D", "A"), probabilities))


def _aware_datetime(value: Any) -> datetime | None:
    if value is None or not str(value).strip():
        return None
    try:
        raw = str(value).strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _xg_input_origin(row: Mapping[str, Any]) -> str | None:
    home_xg, away_xg = _row_xg_pair(dict(row))
    if home_xg is None or away_xg is None:
        return None
    source = str(row.get("xg_source") or "").casefold()
    kind = str(row.get("xg_kind") or "").casefold()
    if kind == "modelled_proxy" or "shot_model" in source or "shot_aggregate" in source:
        return "proxy"
    if source or kind in {"provider", "official", "measured"}:
        return "provider"
    # The Engine retains legacy unlabelled xG for compatibility. Model Lab
    # reports that provenance separately rather than calling it provider xG.
    return "unlabelled_legacy"


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
    except Exception:
        return os.environ.get("GITHUB_SHA", "unknown")


def _append(records: _ClusteredRecords, cluster: str, probabilities: dict, outcome: str) -> None:
    records.set_cluster(cluster)
    records.append((probabilities, outcome))


def _stratum_records(mapping: dict, key: str, arm_names: tuple[str, ...]) -> dict:
    if key not in mapping:
        mapping[key] = {name: _ClusteredRecords() for name in arm_names}
    return mapping[key]


def _metrics_by_stratum(
    mapping: Mapping[str, Mapping[str, _ClusteredRecords]],
    arm_names: tuple[str, ...],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    result = {}
    for index, key in enumerate(sorted(mapping, key=str)):
        item = mapping[key]
        arms = {
            name: _metrics(
                item.get(name) or [],
                bootstrap_samples=bootstrap_samples,
                seed=seed + index * 100 + arm_index,
            )
            for arm_index, name in enumerate(arm_names)
        }
        result[str(key)] = {
            "n": arms["full"]["n"],
            "cluster_count": arms["full"]["cluster_count"],
            "arms": arms,
        }
    return result


def evaluate_fixed_holdout(
    rows: Iterable[Mapping[str, Any]],
    *,
    holdout_start: str,
    holdout_end: str,
    market_source: str = "B365",
    dataset_sha256: str | None = None,
    dataset_source: str = "provided_rows",
    model_version: str | None = None,
    matrix_n: int = 12,
    half_life_days: float = 180.0,
    bootstrap_samples: int = 1000,
    min_holdout_matches: int = DEFAULT_THRESHOLDS["min_holdout_matches"],
    min_matches_per_league: int = DEFAULT_THRESHOLDS["min_matches_per_league"],
    min_matchday_clusters: int = DEFAULT_THRESHOLDS["min_matchday_clusters"],
    min_market_paired_matches: int = DEFAULT_THRESHOLDS["min_market_paired_matches"],
    odds_timestamp_field: str | None = None,
    kickoff_timestamp_field: str | None = None,
    leagues: Mapping[str, str] = TOP_FIVE_LEAGUES,
) -> dict:
    """Run a fixed-range, chronological 1X2 holdout.

    Rows before holdout_start are training/warm-up history. Rows within the
    inclusive holdout range are scored. The four model arms use identical
    fixtures and dates; only the named feature is ablated.
    """
    start = _date_arg(holdout_start, "holdout_start")
    end = _date_arg(holdout_end, "holdout_end")
    if start > end:
        raise ValueError("holdout_start must be on or before holdout_end")
    if market_source not in MARKET_SOURCES:
        raise ValueError(f"market_source must be one of {', '.join(MARKET_SOURCES)}")
    if not isinstance(dataset_source, str) or not dataset_source.strip():
        raise ValueError("dataset_source must be a non-empty label")
    if bool(odds_timestamp_field) != bool(kickoff_timestamp_field):
        raise ValueError("provide both odds_timestamp_field and kickoff_timestamp_field, or neither")
    if not isinstance(bootstrap_samples, int) or bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be a non-negative integer")

    source_rows = [dict(row) for row in rows if isinstance(row, Mapping)]
    if dataset_sha256 is None:
        canonical = json.dumps(
            source_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        dataset_sha256 = hashlib.sha256(canonical).hexdigest()

    league_codes = tuple(leagues)
    arm_names = ("full", "without_h2h", "without_rest_days", "without_xg")
    prepared = []
    invalid_rows = 0
    excluded_league_rows = 0
    duplicate_rows = 0
    seen = set()

    for source_index, row in enumerate(source_rows):
        stamp = _parse_date(_get(row, "Date", "date", "match_date", "kickoff_date"))
        league = _canonical_league(_get(row, "Div", "league", "competition"))
        home = _get(row, "HomeTeam", "home_team", "home")
        away = _get(row, "AwayTeam", "away_team", "away")
        home_goals = _strict_score(_get(row, "FTHG", "home_goals", "hg"))
        away_goals = _strict_score(_get(row, "FTAG", "away_goals", "ag"))
        if league not in leagues:
            excluded_league_rows += 1
            continue
        if stamp is None or not home or not away or home_goals is None or away_goals is None:
            invalid_rows += 1
            continue
        if stamp > end:
            continue
        identity = (
            stamp.isoformat(), league,
            " ".join(str(home).casefold().split()),
            " ".join(str(away).casefold().split()),
        )
        if identity in seen:
            duplicate_rows += 1
            continue
        seen.add(identity)
        clean = dict(row)
        clean["Div"] = league
        prepared.append((
            stamp, league, str(home).strip(), str(away).strip(),
            home_goals, away_goals, clean, source_index,
        ))

    prepared.sort(key=lambda item: (item[0], item[7]))
    training_count = sum(item[0] < start for item in prepared)
    holdout_count = sum(start <= item[0] <= end for item in prepared)
    holdout_days = len({item[0] for item in prepared if start <= item[0] <= end})

    models = {
        "full": Engine(matrix_n=matrix_n, temporal_weighting=True, league_rho_fitting=True),
        "without_h2h": Engine(matrix_n=matrix_n, temporal_weighting=True, league_rho_fitting=True),
        "without_rest_days": Engine(matrix_n=matrix_n, temporal_weighting=True, league_rho_fitting=True),
        "without_xg": Engine(matrix_n=matrix_n, temporal_weighting=True, league_rho_fitting=True),
    }
    for model in models.values():
        model.half_life_days = float(half_life_days)
    models["without_rest_days"].rest_days_adjustment = False

    records = {name: _ClusteredRecords() for name in arm_names}
    market_records = _ClusteredRecords()
    candidate_on_market_records = _ClusteredRecords()
    league_records: dict[str, dict] = {}
    season_records: dict[str, dict] = {}
    league_season_records: dict[str, dict] = {}
    market_sources = {market_source: 0}
    xg_usage = {
        "provider": {"rows_available": 0, "forecasts_using_xg": 0},
        "proxy": {"rows_available": 0, "forecasts_using_xg": 0},
        "unlabelled_legacy": {"rows_available": 0, "forecasts_using_xg": 0},
    }
    # Track provenance in the historical state already ingested by each
    # competition. The current fixture's xG is never used in its own forecast.
    xg_history_origins = defaultdict(lambda: {
        "provider": 0, "proxy": 0, "unlabelled_legacy": 0,
    })
    season_basis_counts = defaultdict(int)
    timestamp_audit = {
        "configured": bool(odds_timestamp_field and kickoff_timestamp_field),
        "verified": 0,
        "missing_or_invalid": 0,
        "not_pre_kickoff": 0,
        "unverified_because_not_configured": 0,
    }
    market_missing_or_invalid = 0
    index = 0
    cursor = 0

    while cursor < len(prepared):
        stamp = prepared[cursor][0]
        group = []
        while cursor < len(prepared) and prepared[cursor][0] == stamp:
            group.append(prepared[cursor])
            cursor += 1
        cluster_id = stamp.isoformat()
        is_holdout = start <= stamp <= end
        pending = []

        # Forecast all fixtures in the day on each independent model arm
        # before updating any model with a result from that day.
        for match_date, league, home, away, hg, ag, row, source_index in group:
            season, season_basis = _season_label(row, match_date)
            if is_holdout:
                season_basis_counts[season_basis] += 1
            origin = _xg_input_origin(row)
            if is_holdout and origin:
                xg_usage[origin]["rows_available"] += 1

            no_xg_row = _without_xg(row)
            predictions = {
                "full": models["full"].predict(home, away, league, match_date=match_date.isoformat()),
                "without_h2h": models["without_h2h"].predict(
                    home, away, league, match_date=match_date.isoformat(), apply_h2h=False,
                ),
                "without_rest_days": models["without_rest_days"].predict(
                    home, away, league, match_date=match_date.isoformat(),
                ),
                "without_xg": models["without_xg"].predict(
                    home, away, league, match_date=match_date.isoformat(),
                ),
            }
            outcome = "H" if hg > ag else "A" if hg < ag else "D"

            if is_holdout:
                for arm in arm_names:
                    _append(records[arm], cluster_id, {
                        "H": predictions[arm]["p1"],
                        "D": predictions[arm]["x"],
                        "A": predictions[arm]["p2"],
                    }, outcome)

                full_prediction = predictions["full"]
                if full_prediction.get("xg_used"):
                    if full_prediction.get("xg_modelled_used"):
                        used_origin = "proxy"
                    else:
                        prior_origins = xg_history_origins[league]
                        # Engine's legacy provider bucket also contains xG rows
                        # without provenance. Do not overstate those as provider
                        # xG if any ambiguous legacy values could have contributed.
                        if prior_origins["unlabelled_legacy"]:
                            used_origin = "unlabelled_legacy"
                        elif prior_origins["provider"]:
                            used_origin = "provider"
                        else:
                            used_origin = "unlabelled_legacy"
                    xg_usage[used_origin]["forecasts_using_xg"] += 1

                market_probabilities = _strict_market_probs(row, market_source)
                if market_probabilities is not None:
                    if odds_timestamp_field and kickoff_timestamp_field:
                        odds_time = _aware_datetime(row.get(odds_timestamp_field))
                        kickoff_time = _aware_datetime(row.get(kickoff_timestamp_field))
                        if odds_time is None or kickoff_time is None:
                            timestamp_audit["missing_or_invalid"] += 1
                            market_probabilities = None
                        elif odds_time >= kickoff_time:
                            timestamp_audit["not_pre_kickoff"] += 1
                            market_probabilities = None
                        else:
                            timestamp_audit["verified"] += 1
                    else:
                        timestamp_audit["unverified_because_not_configured"] += 1

                if market_probabilities is None:
                    market_missing_or_invalid += 1
                else:
                    market_sources[market_source] += 1
                    _append(market_records, cluster_id, market_probabilities, outcome)
                    _append(candidate_on_market_records, cluster_id, {
                        "H": full_prediction["p1"],
                        "D": full_prediction["x"],
                        "A": full_prediction["p2"],
                    }, outcome)

                for mapping, key in (
                    (league_records, league),
                    (season_records, season),
                    (league_season_records, f"{league} | {season}"),
                ):
                    bucket = _stratum_records(mapping, key, arm_names)
                    for arm in arm_names:
                        _append(bucket[arm], cluster_id, {
                            "H": predictions[arm]["p1"],
                            "D": predictions[arm]["x"],
                            "A": predictions[arm]["p2"],
                        }, outcome)

            pending.append((league, home, away, hg, ag, row, no_xg_row, match_date, predictions, outcome, origin))
        
        # Ingest today's outcomes after all predictions for this date are fixed.
        for league, home, away, hg, ag, row, no_xg_row, match_date, predictions, outcome, origin in pending:
            training_rows = {
                "full": row,
                "without_h2h": row,
                "without_rest_days": row,
                "without_xg": no_xg_row,
            }
            for arm in arm_names:
                models[arm].record_calibration(league, predictions[arm], outcome)
                models[arm].add(
                    home, away, hg, ag, training_rows[arm],
                    match_num=index, total=len(prepared), match_date=match_date.isoformat(),
                )
            if origin:
                xg_history_origins[league][origin] += 1
            index += 1

    bootstrap_strata = min(bootstrap_samples, 250)
    arm_metrics = {
        name: _metrics(records[name], bootstrap_samples=bootstrap_samples, seed=17001 + i)
        for i, name in enumerate(arm_names)
    }
    market_metrics = _metrics(market_records, bootstrap_samples=bootstrap_samples, seed=17101)
    candidate_market_metrics = _metrics(
        candidate_on_market_records, bootstrap_samples=bootstrap_samples, seed=17102,
    )
    candidate_minus_market = _paired_differences(
        candidate_on_market_records, market_records,
        bootstrap_samples=bootstrap_samples, seed=17103,
    )
    feature_ablation = {
        label: _paired_differences(
            records["full"], records[arm],
            bootstrap_samples=bootstrap_samples, seed=17200 + i,
        )
        for i, (label, arm) in enumerate((
            ("full_minus_without_h2h", "without_h2h"),
            ("full_minus_without_rest_days", "without_rest_days"),
            ("full_minus_without_xg", "without_xg"),
        ))
    }

    def _serialize_strata(mapping: dict) -> dict:
        return _metrics_by_stratum(
            mapping, arm_names, bootstrap_samples=bootstrap_strata, seed=17300,
        )

    by_league = _serialize_strata(league_records)
    by_season = _serialize_strata(season_records)
    by_league_season = _serialize_strata(league_season_records)

    league_counts = {
        code: (by_league.get(code) or {}).get("n", 0)
        for code in league_codes
    }
    matched_market_n = market_metrics["n"]
    timestamp_verified = bool(
        timestamp_audit["configured"]
        and matched_market_n > 0
        and timestamp_audit["verified"] == matched_market_n
    )
    brier = arm_metrics["full"]["brier"]
    log_loss = arm_metrics["full"]["log_loss"]
    market_log_loss_ci = (
        candidate_minus_market.get("confidence_intervals_95", {}).get("log_loss", {})
        if isinstance(candidate_minus_market, dict) else {}
    )
    market_log_loss_high = market_log_loss_ci.get("high")
    thresholds = {
        **DEFAULT_THRESHOLDS,
        "min_holdout_matches": int(min_holdout_matches),
        "min_matches_per_league": int(min_matches_per_league),
        "min_matchday_clusters": int(min_matchday_clusters),
        "min_market_paired_matches": int(min_market_paired_matches),
    }
    sample_size_met = (
        holdout_count >= thresholds["min_holdout_matches"]
        and all(league_counts.get(code, 0) >= thresholds["min_matches_per_league"] for code in league_codes)
        and arm_metrics["full"]["cluster_count"] >= thresholds["min_matchday_clusters"]
    )
    absolute_targets_met = (
        brier is not None and brier <= thresholds["max_multiclass_brier"]
        and log_loss is not None and log_loss <= thresholds["max_log_loss"]
    )
    market_n_met = matched_market_n >= thresholds["min_market_paired_matches"]
    market_not_worse = bool(
        market_n_met and timestamp_verified and market_log_loss_high is not None
        and float(market_log_loss_high) <= thresholds["max_market_log_loss_regression"]
    )
    market_advantage_proven = bool(
        market_n_met and timestamp_verified and market_log_loss_high is not None
        and float(market_log_loss_high) < 0.0
    )
    strict_gate_passed = bool(sample_size_met and absolute_targets_met and market_not_worse)

    manifest = {
        "schema": "sport-ai-model-lab-v1",
        "evaluation_type": "prequential_fixed_parameter_holdout",
        "holdout_start": start.isoformat(),
        "holdout_end": end.isoformat(),
        "training_cutoff_exclusive": start.isoformat(),
        "market_source": market_source,
        "market_source_label": MARKET_SOURCES[market_source]["label"],
        "market_columns": list(MARKET_SOURCES[market_source]["columns"]),
        "de_vig_method": "proportional",
        "matrix_n": int(matrix_n),
        "half_life_days": float(half_life_days),
        "model_version": model_version or _git_revision(),
        "dataset_sha256": dataset_sha256,
        "dataset_source": dataset_source.strip(),
        "leagues": {code: leagues[code] for code in league_codes},
        "thresholds": thresholds,
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_unit": "calendar_matchday_cluster",
        "odds_timestamp_field": odds_timestamp_field,
        "kickoff_timestamp_field": kickoff_timestamp_field,
    }
    experiment_id = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return {
        "schema": "sport-ai-model-lab-v1",
        "experiment_id": experiment_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest,
        "data_quality": {
            "input_rows": len(source_rows),
            "usable_rows_through_holdout_end": len(prepared),
            "training_rows_before_holdout": training_count,
            "holdout_rows": holdout_count,
            "holdout_matchdays": holdout_days,
            "season_assignment_basis_counts": dict(season_basis_counts),
            "invalid_rows": invalid_rows,
            "excluded_non_top_five_rows": excluded_league_rows,
            "duplicate_rows_skipped": duplicate_rows,
            "market_rows_without_valid_fixed_source_or_timestamp": market_missing_or_invalid,
        },
        "candidate": arm_metrics["full"],
        "market": {
            "source": market_source,
            "source_label": MARKET_SOURCES[market_source]["label"],
            "n": matched_market_n,
            "metrics": market_metrics,
            "candidate_on_same_fixtures": candidate_market_metrics,
            "candidate_minus_market": candidate_minus_market,
            "source_counts": market_sources,
            "timestamp_audit": {
                **timestamp_audit,
                "verified_pre_kickoff": timestamp_verified,
                "public_pre_kickoff_claim_allowed": timestamp_verified,
                "note": (
                    "Capture times were explicitly verified as timezone-aware and earlier than kickoff."
                    if timestamp_verified else
                    "The odds fields identify one bookmaker, but the source/capture timestamp is not verified. "
                    "Treat this as a diagnostic historical line, not a proven pre-kickoff snapshot."
                ),
            },
        },
        "feature_ablation": {
            "arms": arm_metrics,
            "full_minus_ablation": feature_ablation,
            "interpretation": (
                "Negative Brier/log-loss differences favor the full model. All arms use identical "
                "fixtures, dates, temporal weighting and league-specific rho; the named feature is "
                "the only disabled input. Results are descriptive and must not trigger tuning on this holdout."
            ),
        },
        "xg_usage": {
            **xg_usage,
            "total_holdout_matches": holdout_count,
            "forecasts_using_xg": sum(item["forecasts_using_xg"] for item in xg_usage.values()),
            "share": (
                sum(item["forecasts_using_xg"] for item in xg_usage.values()) / holdout_count
                if holdout_count else None
            ),
            "provenance_policy": (
                "Explicit provider and modelled-proxy metadata are separated. Legacy xG without "
                "provenance is reported as unlabelled_legacy, not provider xG."
            ),
        },
        "stratified": {
            "by_league": by_league,
            "by_season": by_season,
            "by_league_season": by_league_season,
            "league_counts": league_counts,
            "minimum_stratum_policy": (
                "League-level promotion sample target is at least 400 fixtures per fixed top-five league. "
                "Smaller season cells are descriptive only."
            ),
        },
        "readiness": {
            "sample_size_met": sample_size_met,
            "absolute_targets_met": absolute_targets_met,
            "market_paired_sample_met": market_n_met,
            "market_snapshot_verified": timestamp_verified,
            "not_worse_than_market": market_not_worse,
            "market_advantage_proven": market_advantage_proven,
            "strict_holdout_gate_passed": strict_gate_passed,
            "status": "PASS" if strict_gate_passed else "NOT_READY",
            "explanation": (
                "PASS requires >=3000 holdout matches, >=400 in each fixed league, >=30 matchday clusters, "
                "Brier <=0.57, Log Loss <=0.99, and the upper 95% CI for paired model-minus-market Log Loss "
                "no worse than +0.02 on >=3000 same-source matches with verified pre-kickoff timestamps."
            ),
        },
        "interpretation": (
            "This is a fixed-parameter chronological prequential holdout. The algorithm and parameters "
            "are frozen; each matchday is forecast in full before any result from that day is ingested. "
            "Earlier holdout results may update state for later days, but holdout metrics must never be used "
            "to choose features, tune parameters, or re-label this same period as an independent retest. "
            "No profitability claim follows from forecast scoring metrics."
        ),
    }


def _number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def render_markdown_report(report: Mapping[str, Any]) -> str:
    candidate = report.get("candidate") or {}
    market = report.get("market") or {}
    paired = market.get("candidate_minus_market") or {}
    difference = paired.get("differences") or {}
    intervals = paired.get("confidence_intervals_95") or {}
    log_ci = intervals.get("log_loss") or {}
    readiness = report.get("readiness") or {}
    manifest = report.get("manifest") or {}
    data = report.get("data_quality") or {}
    xg = report.get("xg_usage") or {}
    candidate_ci = candidate.get("confidence_intervals_95") or {}

    def _ci_text(metric):
        item = candidate_ci.get(metric) or {}
        if item.get("low") is None or item.get("high") is None:
            return "—"
        return f"[{_number(item.get('low'))}, {_number(item.get('high'))}]"

    lines = [
        "# SPORT AI · MODEL LAB v1",
        "",
        f"**Status:** {readiness.get('status', 'NOT_READY')}  ",
        f"**Holdout:** {manifest.get('holdout_start', '—')} — {manifest.get('holdout_end', '—')}  ",
        f"**Experiment ID:** `{report.get('experiment_id', '—')}`",
        "",
        "## 1X2 quality",
        "",
        "| Metric | Model | Matchday-clustered 95% CI |",
        "|---|---:|---:|",
        f"| Multiclass Brier (sum over H/D/A) | {_number(candidate.get('brier'))} | {_ci_text('brier')} |",
        f"| Log Loss | {_number(candidate.get('log_loss'))} | {_ci_text('log_loss')} |",
        f"| Accuracy | {_number(candidate.get('accuracy'))} | {_ci_text('accuracy')} |",
        f"| Matches | {candidate.get('n', 0)} | — |",
        f"| Matchday clusters | {candidate.get('cluster_count', 0)} | — |",
        "",
        "## Fixed-bookmaker comparison",
        "",
        f"Source: **{market.get('source_label', market.get('source', '—'))}**; de-vig: proportional; "
        "all three 1X2 prices must be present.",
        "",
        "| Metric | Market | Model on same fixtures | Model − market |",
        "|---|---:|---:|---:|",
        f"| Brier | {_number((market.get('metrics') or {}).get('brier'))} | "
        f"{_number((market.get('candidate_on_same_fixtures') or {}).get('brier'))} | "
        f"{_number(difference.get('brier'))} |",
        f"| Log Loss | {_number((market.get('metrics') or {}).get('log_loss'))} | "
        f"{_number((market.get('candidate_on_same_fixtures') or {}).get('log_loss'))} | "
        f"{_number(difference.get('log_loss'))} |",
        f"| Paired matches | {market.get('n', 0)} | {market.get('n', 0)} | — |",
        "",
        f"Paired Log Loss difference 95% CI: **[{_number(log_ci.get('low'))}, {_number(log_ci.get('high'))}]**.",
        "",
        (market.get("timestamp_audit") or {}).get("note", ""),
        "",
        "## Fixed top-five league results",
        "",
        "| League | Matches | Brier | Log Loss | Accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    reliability = candidate.get("reliability") or {}
    reliability_classes = reliability.get("classes") or {}
    lines += [
        "",
        "## Reliability · fixed-width bins",
        "",
        "| Outcome | Probability bin | N | Mean predicted | Observed frequency | Absolute gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for outcome, label in (("H", "P1"), ("D", "X"), ("A", "P2")):
        class_report = reliability_classes.get(outcome) or {}
        for bucket in class_report.get("bins") or []:
            if not bucket.get("count"):
                continue
            lines.append(
                f"| {label} | {_number(bucket.get('lower'), 2)}–{_number(bucket.get('upper'), 2)} | "
                f"{bucket.get('count', 0)} | {_number(bucket.get('mean_predicted'))} | "
                f"{_number(bucket.get('observed_frequency'))} | {_number(bucket.get('absolute_gap'))} |"
            )
    lines += ["", f"Reliability ECE: " + "; ".join(
        f"{label}={_number((reliability_classes.get(code) or {}).get('ece'))}"
        for code, label in (("H", "P1"), ("D", "X"), ("A", "P2"))
    ) + ".", ""]
    league_names = manifest.get("leagues") or {}
    by_league = (report.get("stratified") or {}).get("by_league") or {}
    for code, label in league_names.items():
        metric = ((by_league.get(code) or {}).get("arms") or {}).get("full") or {}
        lines.append(
            f"| {label} ({code}) | {metric.get('n', 0)} | {_number(metric.get('brier'))} | "
            f"{_number(metric.get('log_loss'))} | {_number(metric.get('accuracy'))} |"
        )
    lines += [
        "",
        "## Results by season",
        "",
        "| Season | Matches | Brier | Log Loss | Accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    by_season = (report.get("stratified") or {}).get("by_season") or {}
    for season, item in sorted(by_season.items(), key=lambda pair: pair[0]):
        metric = (item.get("arms") or {}).get("full") or {}
        lines.append(
            f"| {season} | {metric.get('n', 0)} | {_number(metric.get('brier'))} | "
            f"{_number(metric.get('log_loss'))} | {_number(metric.get('accuracy'))} |"
        )
    lines += [
        "",
        "Season assignment basis: " + json.dumps(
            data.get("season_assignment_basis_counts") or {}, ensure_ascii=False, sort_keys=True,
        ),
        "",
        "Detailed league × season results are included in the JSON report.",
        "",
        "## Feature ablation",
        "",
        "| Full model minus ablation | Δ Brier | Δ Log Loss | Interpretation |",
        "|---|---:|---:|---|",
    ]
    for key, label in (
        ("full_minus_without_h2h", "Without H2H"),
        ("full_minus_without_rest_days", "Without rest days"),
        ("full_minus_without_xg", "Without xG"),
    ):
        result = ((report.get("feature_ablation") or {}).get("full_minus_ablation") or {}).get(key) or {}
        values = result.get("differences") or {}
        lines.append(
            f"| {label} | {_number(values.get('brier'))} | {_number(values.get('log_loss'))} | "
            "Negative favours full model |"
        )
    lines += [
        "",
        "## xG usage",
        "",
        "| Provenance | Rows with xG | Forecasts actually using xG |",
        "|---|---:|---:|",
    ]
    for key, label in (
        ("provider", "Provider xG"),
        ("proxy", "Modelled proxy xG"),
        ("unlabelled_legacy", "Unlabelled legacy xG"),
    ):
        item = xg.get(key) or {}
        lines.append(
            f"| {label} | {item.get('rows_available', 0)} | {item.get('forecasts_using_xg', 0)} |"
        )
    lines += [
        "",
        "## Readiness gates",
        "",
        "| Gate | Result |",
        "|---|---|",
        f"| Sample size and per-league coverage | {'PASS' if readiness.get('sample_size_met') else 'NOT MET'} |",
        f"| Brier / Log Loss absolute targets | {'PASS' if readiness.get('absolute_targets_met') else 'NOT MET'} |",
        f"| Same-source market paired sample | {'PASS' if readiness.get('market_paired_sample_met') else 'NOT MET'} |",
        f"| Pre-kickoff odds timestamp verified | {'PASS' if readiness.get('market_snapshot_verified') else 'NOT VERIFIED'} |",
        f"| Model not worse than market (upper 95% CI ≤ +0.02) | {'PASS' if readiness.get('not_worse_than_market') else 'NOT MET'} |",
        f"| Market advantage proven (upper 95% CI < 0) | {'YES' if readiness.get('market_advantage_proven') else 'NOT PROVEN'} |",
        "",
        "## Reproducibility manifest",
        "",
        f"- Generated: {report.get('generated_at_utc', '—')}",
        f"- Model revision: `{manifest.get('model_version', '—')}`",
        f"- Dataset SHA-256: `{manifest.get('dataset_sha256', '—')}`",
        f"- Training cutoff (exclusive): {manifest.get('training_cutoff_exclusive', '—')}",
        f"- Matrix N: {manifest.get('matrix_n', '—')}; half-life: {manifest.get('half_life_days', '—')} days",
        f"- Market columns: `{', '.join(manifest.get('market_columns') or [])}`",
        f"- Usable training fixtures: {data.get('training_rows_before_holdout', 0)}",
        f"- Holdout fixtures: {data.get('holdout_rows', 0)}; matchdays: {data.get('holdout_matchdays', 0)}",
        f"- Invalid rows: {data.get('invalid_rows', 0)}; duplicates skipped: {data.get('duplicate_rows_skipped', 0)}",
        "",
        "## Interpretation and limitations",
        "",
        report.get("interpretation", ""),
        "",
        "This report is a forecast-quality audit, not a profitability or staking recommendation. "
        "Do not tune this same holdout and continue presenting it as independent evidence. "
        "The published market comparison must retain its source and timestamp caveat.",
        "",
    ]
    return "\n".join(lines)


def render_reliability_svg(report: Mapping[str, Any]) -> str:
    """Render the three 1X2 fixed-width reliability curves as a standalone SVG."""
    reliability = (report.get("candidate") or {}).get("reliability") or {}
    classes = (("H", "P1"), ("D", "X"), ("A", "P2"))
    width, height = 1080, 390
    panel_width = 350
    plot_size = 250
    top, left = 55, 50
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="540" y="28" text-anchor="middle" font-family="Arial" font-size="18" fill="#111">Model Lab v1 · 1X2 reliability</text>',
    ]
    for panel, (code, label) in enumerate(classes):
        x0 = left + panel * panel_width
        y0 = top + 250
        parts.append(f'<text x="{x0 + plot_size/2}" y="48" text-anchor="middle" font-family="Arial" font-size="15" fill="#111">{label}</text>')
        parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0 + plot_size}" y2="{y0}" stroke="#333"/>')
        parts.append(f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{y0}" stroke="#333"/>')
        parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0 + plot_size}" y2="{top}" stroke="#999" stroke-dasharray="5 4"/>')
        parts.append(f'<text x="{x0 + plot_size/2}" y="{y0 + 42}" text-anchor="middle" font-family="Arial" font-size="11" fill="#444">Mean predicted probability</text>')
        parts.append(f'<text x="{x0 - 35}" y="{top + 120}" transform="rotate(-90 {x0 - 35} {top + 120})" text-anchor="middle" font-family="Arial" font-size="11" fill="#444">Observed frequency</text>')
        class_data = (reliability.get("classes") or {}).get(code) or {}
        points = []
        for bucket in class_data.get("bins") or []:
            predicted = bucket.get("mean_predicted")
            observed = bucket.get("observed_frequency")
            if predicted is None or observed is None:
                continue
            px = x0 + float(predicted) * plot_size
            py = y0 - float(observed) * plot_size
            points.append((px, py))
            parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4" fill="#2463a6"/>')
        if points:
            point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
            parts.append(f'<polyline points="{point_text}" fill="none" stroke="#2463a6" stroke-width="2"/>')
        parts.append(f'<text x="{x0 + plot_size/2}" y="{y0 + 62}" text-anchor="middle" font-family="Arial" font-size="11" fill="#444">N={class_data.get("n", 0)} · ECE={_number(class_data.get("ece"), 3)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n",
        dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def load_fixed_holdout_gate_status(path: str | Path | None = None) -> dict:
    """Load the explicit Model Lab v1 promotion gate used by scanner automation.

    Missing, malformed, stale-schema or non-PASS files fail closed. The default
    path is reports/model_lab/model-lab-v1-gate.json; deployments can override
    it with MODEL_LAB_GATE_PATH.
    """
    configured_path = path or os.environ.get(
        "MODEL_LAB_GATE_PATH", "reports/model_lab/model-lab-v1-gate.json"
    )
    gate_path = Path(configured_path)
    try:
        payload = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        payload = {}
    passed = bool(
        isinstance(payload, dict)
        and payload.get("schema") == "sport-ai-model-lab-gate-v1"
        and payload.get("status") == "PASS"
        and payload.get("strict_holdout_gate_passed") is True
        and isinstance(payload.get("experiment_id"), str)
        and len(payload.get("experiment_id", "")) == 64
        and isinstance(payload.get("dataset_sha256"), str)
        and len(payload.get("dataset_sha256", "")) == 64
        and bool(payload.get("holdout_start"))
        and bool(payload.get("holdout_end"))
        and bool(payload.get("generated_at_utc"))
    )
    return {
        "passed": passed,
        "status": "PASS" if passed else "NOT_READY",
        "experiment_id": payload.get("experiment_id") if isinstance(payload, dict) else None,
        "holdout_start": payload.get("holdout_start") if isinstance(payload, dict) else None,
        "holdout_end": payload.get("holdout_end") if isinstance(payload, dict) else None,
        "gate_file": str(gate_path),
    }


def write_model_lab_artifacts(report: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    manifest = report.get("manifest") or {}
    start = str(manifest.get("holdout_start") or "unknown")
    end = str(manifest.get("holdout_end") or "unknown")
    stem = f"model-lab-v1-{start}-to-{end}"
    json_path = output / f"{stem}.json"
    markdown_path = output / f"{stem}.md"
    svg_path = output / f"{stem}-reliability.svg"
    gate_path = output / "model-lab-v1-gate.json"
    readiness = report.get("readiness") or {}
    gate_passed = bool(readiness.get("strict_holdout_gate_passed"))
    gate_payload = {
        "schema": "sport-ai-model-lab-gate-v1",
        "status": "PASS" if gate_passed else "NOT_READY",
        "strict_holdout_gate_passed": gate_passed,
        "experiment_id": report.get("experiment_id"),
        "holdout_start": manifest.get("holdout_start"),
        "holdout_end": manifest.get("holdout_end"),
        "dataset_sha256": manifest.get("dataset_sha256"),
        "generated_at_utc": report.get("generated_at_utc"),
    }
    _atomic_write(json_path, json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    _atomic_write(markdown_path, render_markdown_report(report))
    _atomic_write(svg_path, render_reliability_svg(report))
    _atomic_write(gate_path, json.dumps(gate_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return {
        "json": str(json_path),
        "markdown": str(markdown_path),
        "reliability_svg": str(svg_path),
        "promotion_gate": str(gate_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the reproducible Sport AI Model Lab v1 fixed 1X2 holdout."
    )
    parser.add_argument("csv_path", type=Path, help="Historical CSV with Date/Div/HomeTeam/AwayTeam/FTHG/FTAG")
    parser.add_argument("--holdout-start", required=True, help="Inclusive start date (YYYY-MM-DD)")
    parser.add_argument("--holdout-end", required=True, help="Inclusive end date (YYYY-MM-DD)")
    parser.add_argument("--market-source", choices=tuple(MARKET_SOURCES), default="B365")
    parser.add_argument("--odds-timestamp-column", help="Timezone-aware bookmaker capture timestamp column")
    parser.add_argument("--kickoff-timestamp-column", help="Timezone-aware kickoff timestamp column")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/model_lab"))
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--half-life-days", type=float, default=180.0)
    parser.add_argument("--matrix-n", type=int, default=12)
    args = parser.parse_args(argv)

    raw = args.csv_path.read_bytes()
    text = raw.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    report = evaluate_fixed_holdout(
        rows,
        holdout_start=args.holdout_start,
        holdout_end=args.holdout_end,
        market_source=args.market_source,
        dataset_sha256=hashlib.sha256(raw).hexdigest(),
        matrix_n=args.matrix_n,
        half_life_days=args.half_life_days,
        bootstrap_samples=args.bootstrap_samples,
        odds_timestamp_field=args.odds_timestamp_column,
        kickoff_timestamp_field=args.kickoff_timestamp_column,
    )
    paths = write_model_lab_artifacts(report, args.output_dir)
    print(json.dumps({
        "status": (report.get("readiness") or {}).get("status"),
        "experiment_id": report.get("experiment_id"),
        "candidate": {
            "n": (report.get("candidate") or {}).get("n"),
            "brier": (report.get("candidate") or {}).get("brier"),
            "log_loss": (report.get("candidate") or {}).get("log_loss"),
        },
        "market_paired_n": (report.get("market") or {}).get("n"),
        "artifacts": paths,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
