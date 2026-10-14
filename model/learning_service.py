"""Auditable orchestration for pre-match forecasts and bounded online calibration.

The raw model probabilities stay in ForecastLedger. The probabilities actually
served are recorded separately, so retries are idempotent and the calibration
can be audited without overwriting the original model output.
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from model.error_learner import OnlineErrorLearner
from model.forecast_ledger import ForecastLedger, OUTCOMES, _probabilities, _timestamp


class ForecastLearningService:
    """Record a raw forecast, calibrate from past settled errors, and audit output."""

    def __init__(self, ledger: ForecastLedger, learner: OnlineErrorLearner | None = None,
                 *, enabled: bool = True, min_history: int = 100):
        if isinstance(min_history, bool) or not isinstance(min_history, int) or min_history < 0:
            raise ValueError("min_history must be a non-negative integer")
        self.ledger = ledger
        self.learner = learner or OnlineErrorLearner(ledger)
        self.enabled = bool(enabled)
        self.min_history = min_history
        self.db_path = ledger.db_path
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS forecast_learning_decisions (
                    forecast_key TEXT PRIMARY KEY,
                    raw_probabilities_json TEXT NOT NULL,
                    served_probabilities_json TEXT NOT NULL,
                    history_n INTEGER NOT NULL,
                    correction_enabled INTEGER NOT NULL CHECK(correction_enabled IN (0,1)),
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    def forecast(self, *, forecast_key: str, fixture_id: str, home_team: str,
                 away_team: str, model_version: str,
                 probabilities: Mapping[str, Any], forecast_at: datetime | str,
                 kickoff_at: datetime | str, league: str | None = None,
                 entry_odds: Mapping[str, Any] | None = None,
                 odds_source: str | None = None) -> dict[str, Any]:
        """Log a forecast and return the immutable probabilities served for its key.

        Calibration only uses forecasts and verified settlements known strictly
        before forecast_at. No same-match result can influence its own forecast.
        """
        raw = _probabilities(probabilities)
        forecast_stamp = _timestamp(forecast_at)
        kickoff_stamp = _timestamp(kickoff_at)
        if forecast_stamp >= kickoff_stamp:
            raise ValueError("forecast_at must be strictly before kickoff_at")

        forecast_id = self.ledger.record_prediction(
            forecast_key=forecast_key, fixture_id=fixture_id,
            home_team=home_team, away_team=away_team, model_version=model_version,
            probabilities=raw, raw_probabilities=raw, forecast_at=forecast_stamp,
            kickoff_at=kickoff_stamp, league=league, entry_odds=entry_odds,
            odds_source=odds_source,
        )
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM forecast_learning_decisions WHERE forecast_key=?",
                (forecast_key,),
            ).fetchone()
            if existing is not None:
                return self._decision(existing, forecast_id)

        learned = self.learner.calibrate(raw, as_of=forecast_stamp, league=league)
        enough_history = learned["history_n"] >= self.min_history
        use_correction = self.enabled and enough_history and any(learned["correction_applied"].values())
        served = learned["probabilities"] if use_correction else raw
        # Preserve an explicit reason when the safety gate keeps the raw output.
        report = dict(learned)
        report["correction_enabled"] = use_correction
        report["gate_reason"] = (
            "enabled" if use_correction else
            "disabled_by_configuration" if not self.enabled else
            "insufficient_settled_history" if not enough_history else
            "no_eligible_probability_bin"
        )
        with self._connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO forecast_learning_decisions
                (forecast_key,raw_probabilities_json,served_probabilities_json,
                 history_n,correction_enabled,report_json,created_at)
                VALUES (?,?,?,?,?,?,?)""",
                (forecast_key, json.dumps(raw, sort_keys=True),
                 json.dumps(served, sort_keys=True), int(learned["history_n"]),
                 int(use_correction), json.dumps(report, sort_keys=True),
                 _timestamp(None)),
            )
            saved = conn.execute(
                "SELECT * FROM forecast_learning_decisions WHERE forecast_key=?",
                (forecast_key,),
            ).fetchone()
        return self._decision(saved, forecast_id)

    @staticmethod
    def _decision(row: sqlite3.Row, forecast_id: int) -> dict[str, Any]:
        report = json.loads(row["report_json"])
        return {
            "forecast_id": forecast_id,
            "forecast_key": row["forecast_key"],
            "raw_probabilities": json.loads(row["raw_probabilities_json"]),
            "probabilities": json.loads(row["served_probabilities_json"]),
            "correction_enabled": bool(row["correction_enabled"]),
            "history_n": int(row["history_n"]),
            "learning_report": report,
            "idempotent_replay": True,
        }

    def settle_verified(self, forecast_key: str, result: str, *,
                        home_score: int, away_score: int,
                        settled_at: datetime | str,
                        source: str) -> bool:
        """Settle only when caller supplies an explicit non-empty verified source."""
        if not isinstance(source, str) or not source.strip():
            raise ValueError("a verified result source is required")
        return self.ledger.settle(
            forecast_key, result, home_score=home_score, away_score=away_score,
            settled_at=settled_at,
        )

    def decisions(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM forecast_learning_decisions ORDER BY created_at, forecast_key"
            ).fetchall()
        return [self._decision(row, -1) for row in rows]

    def evaluate(self, *, as_of: datetime | str, league: str | None = None) -> dict[str, Any]:
        """Return chronological raw-vs-calibrated diagnostics from the learner."""
        report = self.learner.evaluate(as_of=as_of, league=league)
        report["deployment_guidance"] = (
            "Keep calibration disabled unless it improves Brier and log loss on an "
            "untouched chronological holdout; these scores do not measure betting ROI."
        )
        return report
