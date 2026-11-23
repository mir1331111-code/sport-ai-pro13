"""Auditable online calibration from strictly prior, settled model forecasts.

Raw probabilities remain immutable in ForecastLedger. Each forecast also stores a
shadow calibrated distribution. Automatic promotion is allowed only after enough
chronological shadow decisions have settled and both Brier and log loss improve.
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime
from typing import Any, Mapping

from model.error_learner import OnlineErrorLearner, _utc
from model.forecast_ledger import ForecastLedger, OUTCOMES, _probabilities, _timestamp


class ForecastLearningService:
    """Record raw forecasts, learn from prior settlements, and audit calibration."""

    def __init__(
        self,
        ledger: ForecastLedger,
        learner: OnlineErrorLearner | None = None,
        *,
        enabled: bool = False,
        min_history: int = 100,
        auto_promote: bool = False,
        promotion_min_samples: int = 200,
        promotion_min_relative_improvement: float = 0.01,
    ):
        if isinstance(min_history, bool) or not isinstance(min_history, int) or min_history < 0:
            raise ValueError("min_history must be a non-negative integer")
        if isinstance(promotion_min_samples, bool) or not isinstance(promotion_min_samples, int) or promotion_min_samples < 1:
            raise ValueError("promotion_min_samples must be a positive integer")
        if (
            not math.isfinite(promotion_min_relative_improvement)
            or not 0.0 <= promotion_min_relative_improvement <= 0.5
        ):
            raise ValueError("promotion_min_relative_improvement must be in [0, 0.5]")
        self.ledger = ledger
        self.learner = learner or OnlineErrorLearner(ledger)
        self.enabled = bool(enabled)
        self.min_history = min_history
        self.auto_promote = bool(auto_promote)
        self.promotion_min_samples = promotion_min_samples
        self.promotion_min_relative_improvement = float(promotion_min_relative_improvement)
        self.db_path = ledger.db_path
        self._promotion_cache: dict[str, dict[str, Any]] = {}
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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_forecast_learning_created "
                "ON forecast_learning_decisions(created_at)"
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _scores(probabilities: Mapping[str, Any], actual: str) -> tuple[float, float]:
        values = _probabilities(probabilities)
        brier = sum((values[key] - float(key == actual)) ** 2 for key in OUTCOMES)
        log_loss = -math.log(max(values[actual], 1e-15))
        return brier, log_loss

    def _shadow_evaluation(
        self, *, as_of: datetime | str, league: str | None = None,
    ) -> dict[str, Any]:
        """Score immutable shadow outputs only after their results are known."""
        cutoff = _utc(as_of)
        prefix = getattr(self.learner, "model_version_prefix", None)
        settled_rows = self.ledger.rows(settled_only=True)
        eligible_rows = []
        for row in settled_rows:
            try:
                forecast_at = _utc(row.get("forecast_at"))
                settled_at = _utc(row.get("settled_at"))
            except (TypeError, ValueError):
                continue
            if forecast_at >= cutoff or settled_at >= cutoff:
                continue
            if prefix and not str(row.get("model_version") or "").startswith(prefix):
                continue
            if league is not None and row.get("league") != league:
                continue
            eligible_rows.append(row)

        if not eligible_rows:
            return {
                "n": 0, "raw_brier": None, "shadow_brier": None,
                "raw_log_loss": None, "shadow_log_loss": None,
                "brier_relative_improvement": None,
                "log_loss_relative_improvement": None,
                "as_of": _timestamp(as_of), "model_version_prefix": prefix,
            }

        keys = [str(row.get("forecast_key") or "") for row in eligible_rows]
        decisions: dict[str, sqlite3.Row] = {}
        with self._connect() as conn:
            for offset in range(0, len(keys), 400):
                batch = [key for key in keys[offset:offset + 400] if key]
                if not batch:
                    continue
                placeholders = ",".join("?" for _ in batch)
                found = conn.execute(
                    "SELECT forecast_key,raw_probabilities_json,report_json "
                    f"FROM forecast_learning_decisions WHERE forecast_key IN ({placeholders})",
                    batch,
                ).fetchall()
                decisions.update({str(item["forecast_key"]): item for item in found})

        raw_scores: list[tuple[float, float]] = []
        shadow_scores: list[tuple[float, float]] = []
        for row in eligible_rows:
            decision = decisions.get(str(row.get("forecast_key") or ""))
            if decision is None:
                continue
            try:
                report = json.loads(decision["report_json"])
                history_n = int(report.get("history_n") or 0)
                applied = report.get("correction_applied") or {}
                shadow = _probabilities(report.get("probabilities"))
                raw = _probabilities(json.loads(decision["raw_probabilities_json"]))
                actual = str(row.get("result") or "").upper()
                if actual not in OUTCOMES or history_n < self.min_history:
                    continue
                if not any(bool(applied.get(outcome)) for outcome in OUTCOMES):
                    continue
                if all(abs(shadow[outcome] - raw[outcome]) < 1e-12 for outcome in OUTCOMES):
                    continue
                raw_scores.append(self._scores(raw, actual))
                shadow_scores.append(self._scores(shadow, actual))
            except (TypeError, ValueError, KeyError, json.JSONDecodeError):
                continue

        n = len(raw_scores)
        raw_brier = sum(score[0] for score in raw_scores) / n if n else None
        shadow_brier = sum(score[0] for score in shadow_scores) / n if n else None
        raw_log = sum(score[1] for score in raw_scores) / n if n else None
        shadow_log = sum(score[1] for score in shadow_scores) / n if n else None
        brier_improvement = (
            (raw_brier - shadow_brier) / raw_brier
            if raw_brier is not None and raw_brier > 0 and shadow_brier is not None
            else None
        )
        log_improvement = (
            (raw_log - shadow_log) / raw_log
            if raw_log is not None and raw_log > 0 and shadow_log is not None
            else None
        )
        return {
            "n": n,
            "raw_brier": raw_brier,
            "shadow_brier": shadow_brier,
            "raw_log_loss": raw_log,
            "shadow_log_loss": shadow_log,
            "brier_relative_improvement": brier_improvement,
            "log_loss_relative_improvement": log_improvement,
            "as_of": _timestamp(as_of),
            "model_version_prefix": prefix,
        }

    def _promotion_gate(
        self, as_of: datetime | str, evaluation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        stamp = _timestamp(as_of)
        if stamp in self._promotion_cache:
            return dict(self._promotion_cache[stamp])
        report = evaluation or self._shadow_evaluation(as_of=stamp)
        n = int(report.get("n") or 0)
        brier = report.get("brier_relative_improvement")
        log_loss = report.get("log_loss_relative_improvement")
        enough = n >= self.promotion_min_samples
        improved = (
            brier is not None and log_loss is not None
            and brier >= self.promotion_min_relative_improvement
            and log_loss >= self.promotion_min_relative_improvement
        )
        passed = bool(self.auto_promote and enough and improved)
        if not self.auto_promote:
            reason = "automatic_promotion_disabled"
        elif not enough:
            reason = "insufficient_shadow_evaluation"
        elif not improved:
            reason = "shadow_metrics_not_improved_on_both_scores"
        else:
            reason = "chronological_shadow_gate_passed"
        gate = {
            "enabled": self.auto_promote,
            "passed": passed,
            "reason": reason,
            "evaluation_n": n,
            "minimum_samples": self.promotion_min_samples,
            "minimum_relative_improvement": self.promotion_min_relative_improvement,
            "brier_relative_improvement": brier,
            "log_loss_relative_improvement": log_loss,
        }
        self._promotion_cache[stamp] = gate
        return dict(gate)

    def preview(
        self, probabilities: Mapping[str, Any], *,
        as_of: datetime | str, league: str | None = None,
    ) -> dict[str, Any]:
        """Calculate shadow/live probabilities without writing the current forecast."""
        raw = _probabilities(probabilities)
        stamp = _timestamp(as_of)
        learned = self.learner.calibrate(raw, as_of=stamp, league=league)
        enough_history = int(learned["history_n"]) >= self.min_history
        has_adjustment = any(bool(learned["correction_applied"].get(k)) for k in OUTCOMES)
        gate = (
            self._promotion_gate(stamp)
            if self.auto_promote else {
                "enabled": False, "passed": False,
                "reason": "automatic_promotion_disabled",
                "evaluation_n": 0, "minimum_samples": self.promotion_min_samples,
                "minimum_relative_improvement": self.promotion_min_relative_improvement,
                "brier_relative_improvement": None, "log_loss_relative_improvement": None,
            }
        )
        use_correction = bool(
            enough_history and has_adjustment
            and (self.enabled or gate.get("passed"))
        )
        served = learned["probabilities"] if use_correction else raw
        report = dict(learned)
        report["correction_enabled"] = use_correction
        report["auto_promotion_gate"] = gate
        report["gate_reason"] = (
            "enabled_by_configuration" if self.enabled and use_correction else
            "promoted_by_chronological_shadow_gate" if use_correction else
            "insufficient_settled_history" if not enough_history else
            "no_eligible_probability_bin" if not has_adjustment else
            "shadow_only_until_chronological_gate_passes"
        )
        return {
            "raw_probabilities": raw,
            "probabilities": served,
            "shadow_probabilities": learned["probabilities"],
            "correction_enabled": use_correction,
            "history_n": int(learned["history_n"]),
            "learning_report": report,
        }

    def forecast(
        self, *, forecast_key: str, fixture_id: str, home_team: str,
        away_team: str, model_version: str,
        probabilities: Mapping[str, Any], forecast_at: datetime | str,
        kickoff_at: datetime | str, league: str | None = None,
        entry_odds: Mapping[str, Any] | None = None,
        odds_source: str | None = None,
    ) -> dict[str, Any]:
        """Log a raw pre-match forecast and persist its shadow/live calibration."""
        raw = _probabilities(probabilities)
        forecast_stamp = _timestamp(forecast_at)
        kickoff_stamp = _timestamp(kickoff_at)
        if forecast_stamp >= kickoff_stamp:
            raise ValueError("forecast_at must be strictly before kickoff_at")
        forecast_id = self.ledger.record_prediction(
            forecast_key=forecast_key, fixture_id=fixture_id,
            home_team=home_team, away_team=away_team, model_version=model_version,
            probabilities=raw, forecast_at=forecast_stamp,
            kickoff_at=kickoff_stamp, league=league, entry_odds=entry_odds,
            odds_source=odds_source,
        )
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM forecast_learning_decisions WHERE forecast_key=?",
                (forecast_key,),
            ).fetchone()
            if existing is not None:
                return self._decision(existing, forecast_id, replay=True)
        preview = self.preview(raw, as_of=forecast_stamp, league=league)
        report = dict(preview["learning_report"])
        with self._connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO forecast_learning_decisions
                (forecast_key,raw_probabilities_json,served_probabilities_json,
                 history_n,correction_enabled,report_json,created_at)
                VALUES (?,?,?,?,?,?,?)""",
                (forecast_key, json.dumps(raw, sort_keys=True),
                 json.dumps(preview["probabilities"], sort_keys=True),
                 int(preview["history_n"]), int(preview["correction_enabled"]),
                 json.dumps(report, sort_keys=True), _timestamp(None)),
            )
            saved = conn.execute(
                "SELECT * FROM forecast_learning_decisions WHERE forecast_key=?",
                (forecast_key,),
            ).fetchone()
        return self._decision(saved, forecast_id, replay=False)

    @staticmethod
    def _decision(row: sqlite3.Row, forecast_id: int, *, replay: bool = False) -> dict[str, Any]:
        report = json.loads(row["report_json"])
        shadow = report.get("probabilities")
        return {
            "forecast_id": forecast_id,
            "forecast_key": row["forecast_key"],
            "raw_probabilities": json.loads(row["raw_probabilities_json"]),
            "probabilities": json.loads(row["served_probabilities_json"]),
            "shadow_probabilities": shadow if isinstance(shadow, dict) else json.loads(
                row["served_probabilities_json"]
            ),
            "correction_enabled": bool(row["correction_enabled"]),
            "history_n": int(row["history_n"]),
            "learning_report": report,
            "idempotent_replay": replay,
        }

    def settle_verified(
        self, forecast_key: str, result: str, *, home_score: int,
        away_score: int, settled_at: datetime | str, source: str,
    ) -> bool:
        """Settle only when caller supplies an explicit non-empty verified source."""
        if not isinstance(source, str) or not source.strip():
            raise ValueError("a verified result source is required")
        return self.ledger.settle(
            forecast_key, result, home_score=home_score, away_score=away_score,
            settled_at=settled_at,
            result_source=source.strip(),
        )

    def decisions(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM forecast_learning_decisions ORDER BY created_at, forecast_key"
            ).fetchall()
        return [self._decision(row, -1) for row in rows]

    def status(self, *, as_of: datetime | str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT COUNT(*) AS decisions,
                   COALESCE(SUM(correction_enabled),0) AS promoted,
                   MAX(created_at) AS last_decision_at
                   FROM forecast_learning_decisions"""
            ).fetchone()
        evaluation = self._shadow_evaluation(as_of=as_of)
        gate = self._promotion_gate(as_of, evaluation)
        return {
            "decisions": int(row["decisions"] or 0),
            "promoted_decisions": int(row["promoted"] or 0),
            "last_decision_at": row["last_decision_at"],
            "shadow_evaluation": evaluation,
            "promotion_gate": gate,
            "min_history": self.min_history,
        }

    def evaluate(self, *, as_of: datetime | str, league: str | None = None) -> dict[str, Any]:
        report = self._shadow_evaluation(as_of=as_of, league=league)
        report["promotion_gate"] = self._promotion_gate(as_of, report)
        report["deployment_guidance"] = (
            "Only enable future calibration after the chronological shadow gate passes "
            "both Brier and log loss on enough settled forecasts. These scores do not measure ROI."
        )
        return report
