"""Blind LLM forecasts and paired, leakage-safe evaluation against the main model."""
from __future__ import annotations

import hashlib
import math
import os
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import sqlite3
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from config import APP_VERSION, DB_FILE, LLM_PROVIDERS
from model.forecast_ledger import ForecastLedger, MarketForecastLedger, OUTCOMES
from model.engine import team_history_key


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


def historical_features(
    engine: Any, home: str, away: str, limit: int = 8,
    division: str | None = None,
) -> dict[str, Any]:
    """Build same-venue features inside the correct men's/women's history namespace."""
    teams = getattr(engine, "st", {})
    home_key = team_history_key(home, division)
    away_key = team_history_key(away, division)
    home_stats = teams.get(home_key, {}) if hasattr(teams, "get") else {}
    away_stats = teams.get(away_key, {}) if hasattr(teams, "get") else {}
    hs = list(home_stats.get("hs") or [])[-limit:]
    hc = list(home_stats.get("hc") or [])[-limit:]
    ass = list(away_stats.get("as") or [])[-limit:]
    ac = list(away_stats.get("ac") or [])[-limit:]
    home_scored, home_conceded = _mean(hs), _mean(hc)
    away_scored, away_conceded = _mean(ass), _mean(ac)
    lambda_home = (home_scored + away_conceded) / 2 if home_scored is not None and away_conceded is not None else None
    lambda_away = (away_scored + home_conceded) / 2 if away_scored is not None and home_conceded is not None else None
    lambda_total = lambda_home + lambda_away if lambda_home is not None and lambda_away is not None else None
    over_25 = None
    if lambda_total is not None and math.isfinite(lambda_total) and lambda_total >= 0:
        over_25 = 1 - math.exp(-lambda_total) * (1 + lambda_total + lambda_total**2 / 2)
    return {
        "home_scored_at_home_avg": home_scored,
        "home_conceded_at_home_avg": home_conceded,
        "away_scored_away_avg": away_scored,
        "away_conceded_away_avg": away_conceded,
        "home_home_sample": min(len(hs), len(hc)),
        "away_away_sample": min(len(ass), len(ac)),
        "sample_limit": limit,
        "simple_baseline_lambda_home": lambda_home,
        "simple_baseline_lambda_away": lambda_away,
        "simple_baseline_lambda_total": lambda_total,
        "simple_baseline_over_25": over_25,
    }


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
    existing = {str(row["forecast_key"]) for row in ledger.rows()}
    existing_market = {str(row["forecast_key"]) for row in market_ledger.rows()}
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    now = now.astimezone(timezone.utc)
    main_version = f"main-model-v{APP_VERSION}"
    resolved_provider = provider if provider in LLM_PROVIDERS else "Groq (бесплатно, быстро)"
    resolved_model = model or LLM_PROVIDERS[resolved_provider]["model"]
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
                try:
                    ledger.record_prediction(
                        forecast_key=key, model_version=main_version,
                        probabilities=main_probs,
                        xg_used=bool(card.get("xg_used")), **common,
                    )
                    existing.add(key)
                    stats["main_recorded"] += 1
                except (ValueError, TypeError):
                    stats["invalid"] += 1

        ai_probs = parse_llm_probabilities(card.get("llm_opinion") or "")
        if ai_probs is not None:
            key = f"{fixture_id}|{ai_version}"
            if key not in existing:
                try:
                    ledger.record_prediction(
                        forecast_key=key, model_version=ai_version,
                        probabilities=ai_probs, **common,
                    )
                    existing.add(key)
                    stats["ai_recorded"] += 1
                except (ValueError, TypeError):
                    stats["invalid"] += 1

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
                        market_ledger.record_recommendation(
                            forecast_key=key, fixture_id=fixture_id, home_team=home,
                            away_team=away, league=league, model_version=main_version,
                            market=market, selection=selection, probability=probability,
                            entry_odds=price, forecast_at=now, kickoff_at=kickoff,
                            odds_source=str(
                                card.get("market_odds_source_by_pick", {}).get(selection)
                                or card.get("best_odds_source") or card.get("odds_source") or ""
                            ) or None,
                            unit_stake=1.0,
                        )
                        existing_market.add(key)
                        stats["market_recorded"] += 1
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
                        market_ledger.record_recommendation(
                            forecast_key=key, fixture_id=fixture_id, home_team=home,
                            away_team=away, league=league, model_version=main_version,
                            market=market, selection=selection, probability=probability,
                            entry_odds=price, forecast_at=now, kickoff_at=kickoff,
                            odds_source=str(card.get("best_odds_source") or card.get("odds_source") or "") or None,
                            unit_stake=1.0,
                        )
                        existing_market.add(key)
                        stats["market_recorded"] += 1
                except (ValueError, TypeError, IndexError):
                    stats["invalid"] += 1

    return stats


def _load_result_lookup_state(db_path: str | Path) -> dict[str, str]:
    """Load persisted retry deadlines so missing live scores are not polled every rerun."""
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
            "SELECT fixture_id, retry_after FROM forecast_result_checks"
        ).fetchall()
    return {str(fixture_id): str(retry_after) for fixture_id, retry_after in rows}


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
    now: datetime | None = None, max_fixtures: int = 20, max_age_days: int = 14,
) -> dict[str, int]:
    """Settle only matched, completed FT scores after a conservative post-kickoff grace.

    Provider feeds contain completed matches only. Wait 2h15 after kickoff before
    querying, check oldest fixtures first, and persist a retry deadline after a
    miss so Streamlit reruns do not hammer the score providers every ten minutes.
    """
    ledger = ForecastLedger(db_path)
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    now_utc = now.astimezone(timezone.utc)
    local_today = now_utc.astimezone(_timezone(timezone_name)).date()
    grace = timedelta(hours=2, minutes=15)
    missing_retry = timedelta(hours=6)
    non_ft_retry = timedelta(days=1)
    confirmed_retry = timedelta(days=365)
    lookup_state = _load_result_lookup_state(db_path)

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in ledger.rows():
        if row.get("result") is not None or not row.get("kickoff_at"):
            continue
        try:
            kickoff = datetime.fromisoformat(str(row["kickoff_at"]).replace("Z", "+00:00"))
            if kickoff.tzinfo is None or kickoff.astimezone(timezone.utc) + grace > now_utc:
                continue
            fixture_id = str(row.get("fixture_id") or "")
            parts = fixture_id.split(":", 2)
            if len(parts) < 3 or parts[0] != "ind":
                continue
            date_iso = parts[1]
            date_obj = datetime.strptime(date_iso, "%Y-%m-%d").date()
            age = (local_today - date_obj).days
            if age < 0 or age > max_age_days:
                continue
            groups[fixture_id].append(row)
        except (ValueError, TypeError):
            continue

    fixtures = sorted(
        groups.items(),
        key=lambda pair: min(str(r.get("kickoff_at") or "") for r in pair[1]),
    )
    market_ledger = MarketForecastLedger(db_path)
    market_groups: dict[str, list[dict]] = defaultdict(list)
    for row in market_ledger.rows():
        if row.get("result_status") is not None or not row.get("kickoff_at"):
            continue
        # A match score cannot settle corners or cards.
        if str(row.get("market") or "").upper() == "STAT":
            continue
        try:
            kickoff = datetime.fromisoformat(str(row["kickoff_at"]).replace("Z", "+00:00"))
            if kickoff.tzinfo is None or kickoff.astimezone(timezone.utc) + grace > now_utc:
                continue
            fixture_id = str(row.get("fixture_id") or "")
            parts = fixture_id.split(":", 2)
            if len(parts) < 3 or parts[0] != "ind":
                continue
            age = (local_today - datetime.strptime(parts[1], "%Y-%m-%d").date()).days
            if 0 <= age <= max_age_days:
                market_groups[fixture_id].append(row)
        except (ValueError, TypeError):
            continue

    looked_up = settled = errors = 0
    market_checked = market_settled = 0
    score_cache: dict[str, dict | None] = {}
    if fixtures or market_groups:
        from data.scores import find_match_score

        def _score_for_fixture(fixture_id: str, sample: dict, date_iso: str):
            nonlocal looked_up, errors
            if fixture_id in score_cache:
                return score_cache[fixture_id]
            retry_value = lookup_state.get(fixture_id)
            if retry_value:
                try:
                    retry_at = datetime.fromisoformat(retry_value.replace("Z", "+00:00"))
                except ValueError:
                    retry_at = now_utc
                if retry_at.tzinfo is not None and retry_at > now_utc:
                    score_cache[fixture_id] = None
                    return None
            if looked_up >= max(0, int(max_fixtures)):
                score_cache[fixture_id] = None
                return None
            looked_up += 1
            try:
                score = find_match_score(
                    str(sample["home_team"]), str(sample["away_team"]), date_iso,
                )
            except Exception:
                errors += 1
                score = None
            status = str((score or {}).get("status") or "").upper()
            try:
                if status == "FT":
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
            lookup_state[fixture_id] = next_check.isoformat()
            score_cache[fixture_id] = score
            return score

        for fixture_id, rows in fixtures:
            sample = rows[0]
            date_iso = fixture_id.split(":", 2)[1]
            score = _score_for_fixture(fixture_id, sample, date_iso)
            if not score or str(score.get("status") or "").upper() != "FT":
                continue
            try:
                home_score, away_score = int(score["home"]), int(score["away"])
                result = "H" if home_score > away_score else "A" if home_score < away_score else "D"
                for row in rows:
                    if row.get("result") is None:
                        ledger.settle(
                            str(row["forecast_key"]), result,
                            home_score=home_score, away_score=away_score,
                            settled_at=now_utc, closing_odds_source=str(score.get("source") or ""),
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
            if not score or str(score.get("status") or "").upper() != "FT":
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

    return {
        "fixtures_checked": looked_up, "forecasts_settled": settled,
        "market_fixtures_checked": market_checked,
        "market_forecasts_settled": market_settled, "errors": errors,
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
