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

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    def record_prediction(
        self, *, forecast_key: str, fixture_id: str, home_team: str,
        away_team: str, model_version: str, probabilities: Mapping[str, Any],
        forecast_at: datetime | str, kickoff_at: datetime | str | None = None,
        league: str | None = None, entry_odds: Mapping[str, Any] | None = None,
        odds_source: str | None = None,
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
        stamp = _timestamp(forecast_at)
        kickoff = _timestamp(kickoff_at) if kickoff_at is not None else None
        with self._connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO forecasts
                (forecast_key,fixture_id,kickoff_at,forecast_at,model_version,
                 home_team,away_team,league,probabilities_json,entry_odds_json,
                 odds_source,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (forecast_key, fixture_id, kickoff, stamp, model_version,
                 home_team, away_team, league, json.dumps(probs, sort_keys=True),
                 json.dumps(prices, sort_keys=True) if prices is not None else None,
                 odds_source, _timestamp(None)),
            )
            row = conn.execute(
                """SELECT id,fixture_id,forecast_at,model_version,home_team,away_team,
                league,probabilities_json,entry_odds_json,odds_source
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
            if actual != expected:
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

    def metrics(self, *, model_version: str | None = None) -> dict[str, Any]:
        """Return multiclass Brier score, log loss, accuracy and available CLV."""
        rows = self.rows(settled_only=True, model_version=model_version)
        if not rows:
            return {"n": 0, "brier": None, "log_loss": None, "accuracy": None,
                    "mean_clv": None, "clv_n": 0}
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
        return {"n": n, "brier": sum(briers) / n, "log_loss": sum(losses) / n,
                "accuracy": correct / n,
                "mean_clv": sum(clvs) / len(clvs) if clvs else None,
                "clv_n": len(clvs)}
