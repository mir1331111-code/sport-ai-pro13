"""Leakage-safe online calibration from settled forecast errors.

Learns conservative probability corrections from prior settled forecasts.
It does not rewrite model weights; evaluate chronologically before deployment.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Mapping

from model.forecast_ledger import ForecastLedger, OUTCOMES, _probabilities, _timestamp


def _utc(value: datetime | str) -> datetime:
    return datetime.fromisoformat(_timestamp(value).replace("Z", "+00:00"))


class OnlineErrorLearner:
    """Calibrate probability bins using only results known before as_of."""

    def __init__(self, ledger: ForecastLedger, *, bins: int = 10,
                 prior_strength: float = 20.0, min_bin_samples: int = 8,
                 max_adjustment: float = 0.12):
        if isinstance(bins, bool) or not isinstance(bins, int) or not 2 <= bins <= 100:
            raise ValueError("bins must be an integer between 2 and 100")
        if not math.isfinite(prior_strength) or prior_strength <= 0:
            raise ValueError("prior_strength must be finite and > 0")
        if isinstance(min_bin_samples, bool) or not isinstance(min_bin_samples, int) or min_bin_samples < 1:
            raise ValueError("min_bin_samples must be a positive integer")
        if not math.isfinite(max_adjustment) or not 0 <= max_adjustment <= 0.3:
            raise ValueError("max_adjustment must be in [0, 0.3]")
        self.ledger = ledger
        self.bins = bins
        self.prior_strength = float(prior_strength)
        self.min_bin_samples = min_bin_samples
        self.max_adjustment = float(max_adjustment)

    def _history(self, as_of: datetime | str, league: str | None = None) -> list[dict]:
        cutoff = _utc(as_of)
        history = []
        for row in self.ledger.rows(settled_only=True):
            try:
                forecast_time = _utc(row["forecast_at"])
                settled_time = _utc(row["settled_at"]) if row.get("settled_at") else None
            except (TypeError, ValueError):
                continue
            if forecast_time >= cutoff or settled_time is None or settled_time >= cutoff:
                continue
            if league is not None and row.get("league") != league:
                continue
            history.append(row)
        return history

    def calibrate(self, probabilities: Mapping[str, Any], *, as_of: datetime | str,
                  league: str | None = None) -> dict[str, Any]:
        """Return corrected probabilities and a transparent learning report."""
        base = _probabilities(probabilities)
        history = self._history(as_of, league)
        counts = {key: [0] * self.bins for key in OUTCOMES}
        hits = {key: [0] * self.bins for key in OUTCOMES}
        for row in history:
            probs, result = row["probabilities"], row["result"]
            for outcome in OUTCOMES:
                index = min(self.bins - 1, int(probs[outcome] * self.bins))
                counts[outcome][index] += 1
                hits[outcome][index] += int(result == outcome)
        corrected, applied = {}, {}
        for outcome in OUTCOMES:
            p = base[outcome]
            index = min(self.bins - 1, int(p * self.bins))
            n = counts[outcome][index]
            if n < self.min_bin_samples:
                corrected[outcome], applied[outcome] = p, False
                continue
            observed = (hits[outcome][index] + self.prior_strength * p) / (n + self.prior_strength)
            delta = max(-self.max_adjustment, min(self.max_adjustment, observed - p))
            corrected[outcome] = max(0.0, min(1.0, p + delta))
            applied[outcome] = True
        total = sum(corrected.values())
        corrected = (dict(base) if total <= 0 or not math.isfinite(total) else
                     {key: corrected[key] / total for key in OUTCOMES})
        return {"probabilities": corrected, "raw_probabilities": base,
                "history_n": len(history),
                "bin_counts": {key: counts[key][min(self.bins - 1, int(base[key] * self.bins))]
                               for key in OUTCOMES},
                "correction_applied": applied, "as_of": _timestamp(as_of),
                "league": league, "method": "prior-settled-binned-calibration"}

    def evaluate(self, *, as_of: datetime | str, league: str | None = None) -> dict[str, Any]:
        """Expanding-window comparison; each row is calibrated from earlier settled rows only."""
        rows = sorted(self._history(as_of, league), key=lambda r: (r["forecast_at"], r["id"]))
        raw_scores, calibrated_scores = [], []
        for row in rows:
            when = _utc(row["forecast_at"])
            # Train only on outcomes already settled before this forecast was made.
            report = self.calibrate(row["probabilities"], as_of=when,
                                    league=league if league is not None else row.get("league"))
            raw, calibrated, actual = row["probabilities"], report["probabilities"], row["result"]
            raw_scores.append((sum((raw[k] - float(k == actual)) ** 2 for k in OUTCOMES),
                               -math.log(max(raw[actual], 1e-15))))
            calibrated_scores.append((sum((calibrated[k] - float(k == actual)) ** 2 for k in OUTCOMES),
                                      -math.log(max(calibrated[actual], 1e-15))))
        def mean(items, index):
            return sum(x[index] for x in items) / len(items) if items else None
        return {"n": len(rows), "raw_brier": mean(raw_scores, 0),
                "calibrated_brier": mean(calibrated_scores, 0),
                "raw_log_loss": mean(raw_scores, 1),
                "calibrated_log_loss": mean(calibrated_scores, 1),
                "as_of": _timestamp(as_of),
                "note": "Chronological diagnostic; compare on an untouched holdout before deployment."}
