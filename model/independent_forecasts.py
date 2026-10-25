"""Blind LLM forecasts and paired, leakage-safe evaluation against the main model."""
from __future__ import annotations

import hashlib
import math
import os
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from config import APP_VERSION, DB_FILE, LLM_PROVIDERS
from model.forecast_ledger import ForecastLedger, OUTCOMES


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


def historical_features(engine: Any, home: str, away: str, limit: int = 8) -> dict[str, Any]:
    """Build LLM features from observed same-venue goals, not the main model's xG lambdas."""
    teams = getattr(engine, "st", {})
    home_stats = teams.get(home, {}) if hasattr(teams, "get") else {}
    away_stats = teams.get(away, {}) if hasattr(teams, "get") else {}
    hs = list(home_stats.get("hs") or [])[-limit:]
    hc = list(home_stats.get("hc") or [])[-limit:]
    ass = list(away_stats.get("as") or [])[-limit:]
    ac = list(away_stats.get("ac") or [])[-limit:]
    return {
        "home_scored_at_home_avg": _mean(hs),
        "home_conceded_at_home_avg": _mean(hc),
        "away_scored_away_avg": _mean(ass),
        "away_conceded_away_avg": _mean(ac),
        "home_home_sample": min(len(hs), len(hc)),
        "away_away_sample": min(len(ass), len(ac)),
        "sample_limit": limit,
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
    stats = {"main_recorded": 0, "ai_recorded": 0, "skipped_no_time": 0,
             "skipped_started": 0, "invalid": 0}
    ledger = ForecastLedger(db_path)
    existing = {str(row["forecast_key"]) for row in ledger.rows()}
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
                        probabilities=main_probs, **common,
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
    return stats


def settle_due_forecasts(
    *, db_path: str | Path = DB_FILE, timezone_name: str | None = None,
    now: datetime | None = None, max_fixtures: int = 20, max_age_days: int = 14,
) -> dict[str, int]:
    """Settle only matched, completed FT scores from the app's existing trusted score adapters."""
    ledger = ForecastLedger(db_path)
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    now_utc = now.astimezone(timezone.utc)
    local_today = now_utc.astimezone(_timezone(timezone_name)).date()
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in ledger.rows():
        if row.get("result") is not None or not row.get("kickoff_at"):
            continue
        try:
            kickoff = datetime.fromisoformat(str(row["kickoff_at"]).replace("Z", "+00:00"))
            if kickoff.tzinfo is None:
                continue
            if kickoff.astimezone(timezone.utc) > now_utc:
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
        key=lambda pair: max(str(r.get("kickoff_at") or "") for r in pair[1]),
        reverse=True,
    )[:max(0, int(max_fixtures))]
    looked_up = settled = errors = 0
    if fixtures:
        from data.scores import find_match_score
    for fixture_id, rows in fixtures:
        sample = rows[0]
        date_iso = fixture_id.split(":", 2)[1]
        looked_up += 1
        try:
            score = find_match_score(
                str(sample["home_team"]), str(sample["away_team"]), date_iso,
            )
            # AET and penalty shootouts are deliberately not treated as regulation-time FT.
            if not score or str(score.get("status") or "").upper() != "FT":
                continue
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
        except Exception:
            errors += 1
    return {"fixtures_checked": looked_up, "forecasts_settled": settled, "errors": errors}


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
        for main_row, ai_row in sorted(pairs, key=lambda p: str(p[1].get("kickoff_at") or ""), reverse=True):
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
            1 for r in rows if r.get("result") is None and r.get("kickoff_at")
            and str(r.get("model_version") or "").startswith(("main-model-", "independent-llm-"))
        ),
    }


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
