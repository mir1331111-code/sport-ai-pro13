"""Persistent, leakage-safe ledger for pre-match 1X2 forecasts.

Predictions are append-only and keyed by a caller-supplied idempotency key.
Results are attached later; metrics only include settled rows with valid outcomes.
This module is deliberately independent of UI and live data providers.
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

OUTCOMES = ("H", "D", "A")


def _timestamp(value: datetime | str | None) -> str:
    if value is None:
        value = datetime.now(timezone.utc)
    elif isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("timestamp must be ISO-8601") from exc
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat()


def _probabilities(value: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(OUTCOMES):
        raise ValueError("probabilities must contain exactly H, D and A")
    probs = {key: float(value[key]) for key in OUTCOMES}
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in probs.values()):
        raise ValueError("probabilities must be finite values in [0, 1]")
    if not math.isclose(sum(probs.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("probabilities must sum to 1")
    return probs


def _odds(value: Mapping[str, Any] | None) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("odds must be a mapping of H, D and A to decimal odds")
    result = {}
    for key, raw in value.items():
        if key not in OUTCOMES:
            raise ValueError("odds keys must be H, D or A")
        price = float(raw)
        if not math.isfinite(price) or price <= 1.0:
            raise ValueError("decimal odds must be finite and > 1")
        result[key] = price
    return result


SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 forecast_key TEXT NOT NULL UNIQUE,
 fixture_id TEXT NOT NULL,
 kickoff_at TEXT,
 forecast_at TEXT NOT NULL,
 model_version TEXT NOT NULL,
 home_team TEXT NOT NULL,
 away_team TEXT NOT NULL,
 league TEXT,
 probabilities_json TEXT NOT NULL,
 entry_odds_json TEXT,
 odds_source TEXT,
 xg_used INTEGER CHECK(xg_used IS NULL OR xg_used IN (0,1)),
 feature_snapshot_json TEXT,
 result TEXT CHECK(result IS NULL OR result IN ('H','D','A')),
 home_score INTEGER,
 away_score INTEGER,
 closing_odds_json TEXT,
 closing_odds_source TEXT,
 settled_at TEXT,
 created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_forecasts_fixture ON forecasts(fixture_id);
CREATE INDEX IF NOT EXISTS idx_forecasts_forecast_at ON forecasts(forecast_at);
CREATE INDEX IF NOT EXISTS idx_forecasts_result ON forecasts(result);
"""


class ForecastLedger:
    """SQLite-backed pre-match prediction journal and evaluation summary."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            columns = {
                str(row[1]) for row in conn.execute(
                    "PRAGMA table_info(forecasts)"
                ).fetchall()
            }
            if "xg_used" not in columns:
                conn.execute(
                    "ALTER TABLE forecasts ADD COLUMN xg_used INTEGER "
                    "CHECK(xg_used IS NULL OR xg_used IN (0,1))"
                )
            if "feature_snapshot_json" not in columns:
                conn.execute(
                    "ALTER TABLE forecasts ADD COLUMN feature_snapshot_json TEXT"
                )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    def record_prediction(
        self, *, forecast_key: str, fixture_id: str, home_team: str,
        away_team: str, model_version: str, probabilities: Mapping[str, Any],
        forecast_at: datetime | str, kickoff_at: datetime | str | None = None,
        league: str | None = None, entry_odds: Mapping[str, Any] | None = None,
        odds_source: str | None = None, xg_used: bool | None = None,
    ) -> int:
        """Insert one immutable forecast. Repeated keys return the original id."""
        required = {
            "forecast_key": forecast_key, "fixture_id": fixture_id,
            "home_team": home_team, "away_team": away_team,
            "model_version": model_version,
        }
        if any(not isinstance(v, str) or not v.strip() for v in required.values()):
            raise ValueError("forecast_key, fixture_id, teams and model_version are required")
        probs = _probabilities(probabilities)
        prices = _odds(entry_odds)
        if xg_used is not None and not isinstance(xg_used, bool):
            raise ValueError("xg_used must be bool or None")
        stored_xg = None if xg_used is None else int(xg_used)
        stamp = _timestamp(forecast_at)
        kickoff = _timestamp(kickoff_at) if kickoff_at is not None else None
        with self._connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO forecasts
                (forecast_key,fixture_id,kickoff_at,forecast_at,model_version,
                 home_team,away_team,league,probabilities_json,entry_odds_json,
                 odds_source,xg_used,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (forecast_key, fixture_id, kickoff, stamp, model_version,
                 home_team, away_team, league, json.dumps(probs, sort_keys=True),
                 json.dumps(prices, sort_keys=True) if prices is not None else None,
                 odds_source, stored_xg, _timestamp(None)),
            )
            row = conn.execute(
                """SELECT id,fixture_id,forecast_at,model_version,home_team,away_team,
                league,probabilities_json,entry_odds_json,odds_source,xg_used
                FROM forecasts WHERE forecast_key=?""", (forecast_key,)
            ).fetchone()
            expected = (
                fixture_id, stamp, model_version, home_team, away_team, league,
                json.dumps(probs, sort_keys=True),
                json.dumps(prices, sort_keys=True) if prices is not None else None,
                odds_source,
            )
            actual = (
                row["fixture_id"], row["forecast_at"], row["model_version"],
                row["home_team"], row["away_team"], row["league"],
                row["probabilities_json"], row["entry_odds_json"], row["odds_source"],
            )
            # Old immutable rows have xg_used=NULL because that feature was not
            # recorded at forecast time. Never backfill or guess it on a rescan.
            xg_conflict = (
                stored_xg is not None and row["xg_used"] is not None
                and int(row["xg_used"]) != stored_xg
            )
            if actual != expected or xg_conflict:
                raise ValueError("forecast_key already exists with a different payload")
            return int(row["id"])

    def settle(
        self, forecast_key: str, result: str, *, home_score: int | None = None,
        away_score: int | None = None, settled_at: datetime | str | None = None,
        closing_odds: Mapping[str, Any] | None = None,
        closing_odds_source: str | None = None,
    ) -> bool:
        """Attach a final result and optional closing prices to a forecast."""
        result = str(result).upper()
        if result not in OUTCOMES:
            raise ValueError("result must be H, D or A")
        if (home_score is None) != (away_score is None):
            raise ValueError("provide both scores or neither")
        if home_score is not None:
            if isinstance(home_score, bool) or isinstance(away_score, bool):
                raise ValueError("scores must be non-negative integers")
            if not isinstance(home_score, int) or not isinstance(away_score, int):
                raise ValueError("scores must be non-negative integers")
            if home_score < 0 or away_score < 0:
                raise ValueError("scores must be non-negative integers")
            derived = "H" if home_score > away_score else "A" if home_score < away_score else "D"
            if derived != result:
                raise ValueError("result does not match the supplied score")
        prices = _odds(closing_odds)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT result,home_score,away_score FROM forecasts WHERE forecast_key=?",
                (forecast_key,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown forecast_key: {forecast_key}")
            if row["result"] is not None and row["result"] != result:
                raise ValueError("conflicting settlement: result is immutable once set")
            if row["home_score"] is not None and home_score is not None and (
                row["home_score"] != home_score or row["away_score"] != away_score
            ):
                raise ValueError("conflicting settlement: scores are immutable once set")
            conn.execute(
                """UPDATE forecasts SET result=?,home_score=COALESCE(home_score,?),
                away_score=COALESCE(away_score,?),
                settled_at=COALESCE(settled_at,?),closing_odds_json=COALESCE(closing_odds_json,?),
                closing_odds_source=COALESCE(closing_odds_source,?)
                WHERE forecast_key=?""",
                (result, home_score, away_score, _timestamp(settled_at),
                 json.dumps(prices, sort_keys=True) if prices is not None else None,
                 closing_odds_source, forecast_key),
            )
        return True

    def rows(self, *, settled_only: bool = False, model_version: str | None = None) -> list[dict]:
        query = "SELECT * FROM forecasts"
        clauses, params = [], []
        if settled_only:
            clauses.append("result IS NOT NULL")
        if model_version is not None:
            clauses.append("model_version=?")
            params.append(model_version)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY forecast_at, id"
        with self._connect() as conn:
            rows = [dict(row) for row in conn.execute(query, params).fetchall()]
        for row in rows:
            for field in ("probabilities_json", "entry_odds_json", "closing_odds_json"):
                raw = row.pop(field)
                row[field.removesuffix("_json")] = json.loads(raw) if raw else None
        return rows

    def reliability_bins(
        self, *, bins: int = 10, model_version: str | None = None,
        model_version_prefix: str | None = None,
    ) -> dict[str, Any]:
        """Reliability table per 1X2 class using settled forecasts only."""
        if isinstance(bins, bool) or not isinstance(bins, int) or not 2 <= bins <= 100:
            raise ValueError("bins must be an integer between 2 and 100")
        rows = self.rows(settled_only=True, model_version=model_version)
        if model_version_prefix is not None:
            rows = [
                row for row in rows
                if str(row.get("model_version") or "").startswith(model_version_prefix)
            ]
        result_rows = []
        ece = {}
        for outcome in OUTCOMES:
            buckets = [
                {"n": 0, "probability_sum": 0.0, "observed_sum": 0}
                for _ in range(bins)
            ]
            for row in rows:
                probability = float(row["probabilities"][outcome])
                index = min(bins - 1, int(probability * bins))
                bucket = buckets[index]
                bucket["n"] += 1
                bucket["probability_sum"] += probability
                bucket["observed_sum"] += int(row["result"] == outcome)
            outcome_ece = 0.0
            outcome_n = sum(bucket["n"] for bucket in buckets)
            for index, bucket in enumerate(buckets):
                n = bucket["n"]
                if not n:
                    continue
                mean_probability = bucket["probability_sum"] / n
                observed_frequency = bucket["observed_sum"] / n
                gap = observed_frequency - mean_probability
                outcome_ece += n * abs(gap)
                result_rows.append({
                    "outcome": outcome,
                    "bin_index": index,
                    "bin_from": index / bins,
                    "bin_to": (index + 1) / bins,
                    "n": n,
                    "mean_probability": mean_probability,
                    "observed_frequency": observed_frequency,
                    "observed_minus_predicted": gap,
                })
            ece[outcome] = outcome_ece / outcome_n if outcome_n else None
        return {
            "n": len(rows), "bins": bins, "rows": result_rows,
            "expected_calibration_error": ece,
            "model_version": model_version,
            "model_version_prefix": model_version_prefix,
        }

    def metrics(self, *, model_version: str | None = None) -> dict[str, Any]:
        """Return proper scoring metrics, class reliability and xG-use coverage."""
        all_rows = self.rows(model_version=model_version)
        tracked_xg = [row for row in all_rows if row.get("xg_used") is not None]
        xg_used_n = sum(int(bool(row.get("xg_used"))) for row in tracked_xg)
        xg_usage = {
            "forecasts": len(all_rows),
            "tracked": len(tracked_xg),
            "xg_used": xg_used_n,
            "share": xg_used_n / len(tracked_xg) if tracked_xg else None,
            "untracked": len(all_rows) - len(tracked_xg),
        }
        rows = [row for row in all_rows if row.get("result") in OUTCOMES]
        calibration = self.reliability_bins(model_version=model_version)
        if not rows:
            return {
                "n": 0, "brier": None, "log_loss": None, "accuracy": None,
                "mean_clv": None, "clv_n": 0,
                "xg_usage": xg_usage, "calibration_bins": calibration,
            }
        briers, losses, correct, clvs = [], [], 0, []
        for row in rows:
            probs = row["probabilities"]
            actual = row["result"]
            briers.append(sum((probs[k] - float(k == actual)) ** 2 for k in OUTCOMES))
            losses.append(-math.log(max(probs[actual], 1e-15)))
            predicted = max(OUTCOMES, key=lambda k: probs[k])
            correct += predicted == actual
            entry, close = row["entry_odds"], row["closing_odds"]
            if entry and close:
                for selection in OUTCOMES:
                    if selection in entry and selection in close:
                        clvs.append(entry[selection] / close[selection] - 1.0)
        n = len(rows)
        return {
            "n": n, "brier": sum(briers) / n, "log_loss": sum(losses) / n,
            "accuracy": correct / n,
            "mean_clv": sum(clvs) / len(clvs) if clvs else None,
            "clv_n": len(clvs), "xg_usage": xg_usage,
            "calibration_bins": calibration,
        }



MARKET_FORECAST_SCHEMA = """
CREATE TABLE IF NOT EXISTS market_forecasts (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 forecast_key TEXT NOT NULL UNIQUE,
 fixture_id TEXT NOT NULL, kickoff_at TEXT, forecast_at TEXT NOT NULL,
 model_version TEXT NOT NULL, home_team TEXT NOT NULL, away_team TEXT NOT NULL,
 league TEXT, market TEXT NOT NULL, selection TEXT NOT NULL,
 probability REAL NOT NULL, entry_odds REAL NOT NULL, odds_source TEXT,
 unit_stake REAL NOT NULL DEFAULT 1.0,
 result_status TEXT CHECK(result_status IS NULL OR result_status IN ('won','lost','push','void')),
 result_score TEXT, actual_value REAL, closing_odds REAL, settlement_source TEXT,
 settled_at TEXT, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_market_forecasts_fixture ON market_forecasts(fixture_id);
CREATE INDEX IF NOT EXISTS idx_market_forecasts_market ON market_forecasts(market);
"""


class MarketForecastLedger:
    """Append-only record of each pre-match best-market pick and its result."""
    VALID_MARKETS = {"1X2", "OU", "BTTS", "AH", "DC", "STAT"}

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(MARKET_FORECAST_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    def record_recommendation(
        self, *, forecast_key: str, fixture_id: str, home_team: str, away_team: str,
        model_version: str, market: str, selection: str, probability: float,
        entry_odds: float, forecast_at: datetime | str,
        kickoff_at: datetime | str | None = None, league: str | None = None,
        odds_source: str | None = None, unit_stake: float = 1.0,
    ) -> int:
        required = (forecast_key, fixture_id, home_team, away_team, model_version, selection)
        if any(not isinstance(v, str) or not v.strip() for v in required):
            raise ValueError("forecast key, fixture, teams, model and selection are required")
        market = str(market or "").upper().strip()
        if market not in self.VALID_MARKETS:
            raise ValueError("unsupported market: " + market)
        probability, entry_odds, unit_stake = map(float, (probability, entry_odds, unit_stake))
        if not math.isfinite(probability) or not 0 < probability < 1:
            raise ValueError("probability must be in (0, 1)")
        if not math.isfinite(entry_odds) or entry_odds <= 1:
            raise ValueError("entry odds must be > 1")
        if not math.isfinite(unit_stake) or unit_stake <= 0:
            raise ValueError("unit stake must be positive")
        stamp = _timestamp(forecast_at)
        kickoff = _timestamp(kickoff_at) if kickoff_at is not None else None
        values = (
            str(forecast_key), str(fixture_id), kickoff, stamp, str(model_version),
            str(home_team), str(away_team), league, market, str(selection),
            probability, entry_odds, odds_source, unit_stake,
        )
        with self._connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO market_forecasts
                (forecast_key,fixture_id,kickoff_at,forecast_at,model_version,home_team,
                 away_team,league,market,selection,probability,entry_odds,odds_source,
                 unit_stake,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (*values, _timestamp(None)),
            )
            row = conn.execute(
                """SELECT id,fixture_id,kickoff_at,forecast_at,model_version,home_team,
                away_team,league,market,selection,probability,entry_odds,odds_source,unit_stake
                FROM market_forecasts WHERE forecast_key=?""", (forecast_key,)
            ).fetchone()
            actual = tuple(row[k] for k in (
                "fixture_id","kickoff_at","forecast_at","model_version","home_team",
                "away_team","league","market","selection","probability","entry_odds",
                "odds_source","unit_stake"
            ))
            if actual != values[1:]:
                raise ValueError("forecast_key already exists with a different payload")
            return int(row["id"])

    def settle(
        self, forecast_key: str, result_status: str, *, result_score: str | None = None,
        actual_value: float | int | None = None, closing_odds: float | None = None,
        settlement_source: str | None = None, settled_at: datetime | str | None = None,
    ) -> bool:
        status = str(result_status or "").lower()
        if status not in {"won","lost","push","void"}:
            raise ValueError("invalid result status")
        if actual_value is not None:
            if isinstance(actual_value, bool):
                raise ValueError("actual value must be non-negative")
            actual_value = float(actual_value)
            if not math.isfinite(actual_value) or actual_value < 0:
                raise ValueError("actual value must be non-negative")
        if closing_odds is not None:
            closing_odds = float(closing_odds)
            if not math.isfinite(closing_odds) or closing_odds <= 1:
                raise ValueError("closing odds must be > 1")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT result_status,result_score,actual_value FROM market_forecasts WHERE forecast_key=?",
                (forecast_key,)
            ).fetchone()
            if row is None:
                raise KeyError("unknown forecast_key: " + forecast_key)
            if row["result_status"] is not None and row["result_status"] != status:
                raise ValueError("conflicting settlement: result is immutable")
            if row["result_score"] is not None and result_score is not None and row["result_score"] != result_score:
                raise ValueError("conflicting settlement: score is immutable")
            if row["actual_value"] is not None and actual_value is not None and not math.isclose(float(row["actual_value"]), actual_value):
                raise ValueError("conflicting settlement: actual value is immutable")
            conn.execute(
                """UPDATE market_forecasts SET result_status=COALESCE(result_status,?),
                result_score=COALESCE(result_score,?),actual_value=COALESCE(actual_value,?),
                closing_odds=COALESCE(closing_odds,?),settlement_source=COALESCE(settlement_source,?),
                settled_at=COALESCE(settled_at,?) WHERE forecast_key=?""",
                (status,result_score,actual_value,closing_odds,settlement_source,_timestamp(settled_at),forecast_key)
            )
        return True

    def rows(self) -> list[dict]:
        with self._connect() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM market_forecasts ORDER BY forecast_at,id")]

    @staticmethod
    def _summary(rows: list[dict]) -> dict[str, Any]:
        settled = [r for r in rows if r.get("result_status") is not None]
        decisive = [r for r in settled if r.get("result_status") in {"won","lost"}]
        wins = sum(r["result_status"] == "won" for r in decisive)
        n = len(decisive)
        turnover = sum(float(r["unit_stake"]) for r in decisive)
        pnl = sum(float(r["unit_stake"]) * (
            float(r["entry_odds"])-1 if r["result_status"] == "won" else -1
        ) for r in decisive)
        clvs = [float(r["entry_odds"])/float(r["closing_odds"])-1
                for r in rows if r.get("closing_odds") is not None]
        brier, logloss = [], []
        for r in decisive:
            p = float(r["probability"])
            y = float(r["result_status"] == "won")
            brier.append((p-y)**2)
            logloss.append(-math.log(max(p if y else 1-p, 1e-15)))
        return {
            "n":len(rows),"settled":len(settled),"pending":len(rows)-len(settled),
            "wins":wins,"losses":n-wins,
            "pushes":sum(r["result_status"]=="push" for r in settled),
            "voids":sum(r["result_status"]=="void" for r in settled),
            "win_rate":wins/n if n else None,"turnover":turnover,"pnl":pnl,
            "roi":pnl/turnover if turnover else None,
            "avg_odds":sum(float(r["entry_odds"]) for r in decisive)/n if n else None,
            "break_even_rate":sum(1/float(r["entry_odds"]) for r in decisive)/n if n else None,
            "brier":sum(brier)/n if n else None,"log_loss":sum(logloss)/n if n else None,
            "mean_clv":sum(clvs)/len(clvs) if clvs else None,"clv_n":len(clvs),
        }

    @staticmethod
    def _scan_identity(row: dict) -> tuple[str, str, str] | None:
        """Identify scanner rows by fixture, model version and market family.

        Older keys included a scan timestamp after the family; the oldest
        single-best path used the timestamp in place of a family name.
        Normalize both formats so historical reruns are not treated as
        independent forecasts. Unknown/manual keys remain untouched.
        """
        parts = str(row.get("forecast_key") or "").split("|")
        if len(parts) < 4 or parts[2] != "market":
            return None
        family = parts[3]
        try:
            datetime.fromisoformat(family.replace("Z", "+00:00"))
        except ValueError:
            pass
        else:
            family = "legacy_best"
        fixture_id = str(row.get("fixture_id") or "")
        model_version = str(row.get("model_version") or "")
        if not fixture_id or not model_version or not family:
            return None
        return fixture_id, model_version, family

    def metrics(self) -> dict[str, Any]:
        raw_rows = self.rows()
        # Rows are ordered by forecast time and ID. Keep the first immutable
        # forecast for each scan family; later reruns of the same fixture are
        # not independent observations and must not inflate ROI or accuracy.
        rows = []
        seen = set()
        duplicate_rows_excluded = 0
        for row in raw_rows:
            identity = self._scan_identity(row)
            if identity is not None:
                if identity in seen:
                    duplicate_rows_excluded += 1
                    continue
                seen.add(identity)
            rows.append(row)
        groups: dict[str,list[dict]] = {}
        for row in rows:
            groups.setdefault(str(row.get("market") or "OTHER"), []).append(row)
        return {
            "overall": self._summary(rows),
            "by_market": [
                {"market": market, **self._summary(group)}
                for market, group in sorted(groups.items())
            ],
            "raw_rows": len(raw_rows),
            "duplicate_rows_excluded": duplicate_rows_excluded,
            "deduplication": "first forecast per fixture, model version and market family",
        }
