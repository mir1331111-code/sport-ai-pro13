# Forecast audit ledger

`model.forecast_ledger.ForecastLedger` is a small SQLite-backed journal for pre-match 1X2 predictions. It is independent of the Streamlit UI and bookmaker polling code so it can be integrated incrementally without silently changing live behavior.

## Minimal example

```python
from datetime import datetime, timezone
from model.forecast_ledger import ForecastLedger

ledger = ForecastLedger("data/forecast_audit.sqlite")
ledger.record_prediction(
    forecast_key="provider-fixture-id:model-v1:2026-10-15T10:00Z",
    fixture_id="provider-fixture-id",
    home_team="Home FC",
    away_team="Away FC",
    league="Example League",
    model_version="13.3.1",
    forecast_at=datetime.now(timezone.utc),
    kickoff_at=datetime(2026, 10, 15, 18, tzinfo=timezone.utc),
    probabilities={"H": 0.50, "D": 0.28, "A": 0.22},
    entry_odds={"H": 2.05, "D": 3.4, "A": 4.0},
    odds_source="bookmaker-name",
)

# Only call after a verified final result is available.
ledger.settle(
    "provider-fixture-id:model-v1:2026-10-15T10:00Z",
    "H",
    home_score=2,
    away_score=1,
    closing_odds={"H": 1.95, "D": 3.5, "A": 4.2},
    closing_odds_source="bookmaker-name",
)
print(ledger.metrics())
```

## Data integrity rules

- Forecast timestamps and kickoff timestamps must include a timezone; timestamps are stored in UTC.
- Probabilities must contain exactly `H`, `D`, `A`, each in [0, 1], and sum to 1.
- `forecast_key` is an idempotency key. Repeating it returns the existing row; it does not overwrite the original prediction.
- A settled result cannot be changed to a conflicting result. Supplied scores must agree with the result label.
- Unsettled rows are excluded from evaluation metrics.
- Keep the key stable for retries, but use a new key when intentionally recording a new forecast revision.

## Metrics and interpretation

`metrics()` returns sample size, multiclass Brier score (sum of squared errors across the three outcomes), log loss, top-class accuracy and a mean odds-movement diagnostic where both entry and closing prices are available. The odds diagnostic averages entry/closing-price ratios across matched 1X2 selections; it is **not** bet-level CLV unless the caller separately identifies and evaluates the actual selection placed.

This module does not yet auto-connect to every live prediction or odds polling path. Until that integration is explicitly added and deployed, it records only forecasts passed to it by the caller. Metrics need enough settled, representative matches to be meaningful; they are not a guarantee of profitability.

## Learning from past errors

`model.error_learner.OnlineErrorLearner` adds an opt-in probability calibration layer. It looks at prior forecasts grouped into probability bins, compares predicted frequencies with observed outcomes, applies shrinkage toward the original probabilities, caps the size of corrections, and renormalizes H/D/A to sum to 1.

```python
from model.error_learner import OnlineErrorLearner

learner = OnlineErrorLearner(ledger, min_bin_samples=30)
report = learner.calibrate(
    {"H": 0.50, "D": 0.28, "A": 0.22},
    as_of=prediction_time,
    league="Example League",
)
calibrated_probabilities = report["probabilities"]
print(report["history_n"], report["bin_counts"], report["correction_applied"])
```

The `as_of` cutoff is mandatory. Both the historical prediction and its settlement must be earlier than the cutoff, preventing future-result leakage. With too few observations in a probability bin, the layer leaves that outcome's probability unchanged. The optional `evaluate()` method compares raw and calibrated Brier/log-loss scores in an expanding-window chronological diagnostic.

This is a first step toward learning from mistakes, not automatic retraining of the full model. It should remain opt-in until it improves metrics on an untouched, chronological holdout and is connected to the live prediction path. Never use the same matches both to tune the correction and to claim unbiased final performance.


## Auditable learning service

`model.learning_service.ForecastLearningService` coordinates the ledger and the
online calibrator. It keeps the original model probabilities in the forecast
ledger and stores the served probabilities plus the learning report in a
separate `forecast_learning_decisions` table in the same SQLite database.
Repeated calls with the same forecast key return the original decision instead
of recalculating it after later results arrive.

Example integration from a caller that already has model probabilities:

```python
from datetime import datetime, timezone
from model.forecast_ledger import ForecastLedger
from model.learning_service import ForecastLearningService

ledger = ForecastLedger("data/forecast_audit.sqlite")
learning = ForecastLearningService(ledger, enabled=False, min_history=100)
decision = learning.forecast(
    forecast_key="provider:fixture-id:model-v1",
    fixture_id="provider:fixture-id",
    home_team="Home FC",
    away_team="Away FC",
    model_version="model-v1",
    probabilities={"H": 0.50, "D": 0.28, "A": 0.22},
    forecast_at=datetime.now(timezone.utc),
    kickoff_at=kickoff_datetime,  # timezone-aware, and later than forecast_at
    league="league-code",
)
served_probabilities = decision["probabilities"]
```

Important operating rules:

- The service is an integration API; it is not yet automatically wired into
  every Streamlit scanner or external result-provider path.
- Keep `enabled=False` until an untouched chronological holdout shows that
  calibration improves both Brier score and log loss. The history threshold is
  a safety gate, not evidence that calibration improves performance.
- A caller must only settle a match after checking a trusted provider's final
  status and score. `settle_verified` requires a non-empty source label, but
  the caller remains responsible for actually verifying the provider response.
- The service only adjusts probabilities; it does not retrain the underlying
  Poisson/Elo model weights and does not establish positive betting ROI.
- Use a persistent file-backed SQLite database, not `:memory:`, for this
  multi-connection workflow. Back up the audit database and keep forecast keys
  stable and unique per fixture/model snapshot.


## Live scanner integration

The scanner now creates a persistent ledger at `DB_FILE + ".forecast-ledger.sqlite"`
and sends raw 1X2 model probabilities to `ForecastLearningService` in shadow
mode. The source UTC kickoff timestamp is retained from football-data.org.
When the scanner runs again, it attempts to settle old ledger entries only
after kickoff + 2 hours and only when football-data.org reports status `FT`
with a final regular-time score. The original model probabilities used by the
scanner remain unchanged; the calibration service records what it would have
served, together with the available history and its report.

This is intentionally a shadow deployment: results and reports accumulate
without changing picks or stakes. Before enabling corrected probabilities,
compare raw and calibrated Brier/log-loss on a chronological holdout and
confirm the improvement is stable across time and leagues.
