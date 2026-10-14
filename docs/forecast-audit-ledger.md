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
