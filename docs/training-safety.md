# Training data safety

## Changes in this branch

- Historical training rows are validated before they reach `Engine.learn_step`.
- Missing scores are rejected rather than silently converted to 0-0.
- Scores must be finite, non-negative integer values (0–30).
- Rows without teams/dates and rows dated in the future are rejected.
- Training rows across leagues/seasons are merged and sorted by date before model updates.
- Decision snapshot writes report actual inserts and invalidate the read cache.
- The portfolio includes a read-only audit of pending/stale bets and unresolved decision snapshots.
- The walk-forward backtest uses a complete 1X2 odds triplet from one bookmaker; incomplete triplets are excluded instead of mixing bookmakers.
- The backtest warns that historical odds columns do not establish when the quote was available.

## Verification

Run:

```bash
python -m unittest discover -s tests -v
```

## Important limitations

This is a data-integrity and observability improvement, not proof that the model is profitable or that online self-learning is complete. The portfolio audit is diagnostic only: it does not automatically resolve ambiguous fixtures or retrain the model. Historical bookmaker columns have no verified quote timestamp in the current dataset, so simulated PnL is not proof that the prices were executable when a prediction was made. Same-day historical rows are ordered deterministically by league and team because many sources lack kickoff times; they cannot be guaranteed to reflect the true within-day order. Calibration still needs a separate leakage audit and out-of-time evaluation before automatic model promotion should be enabled.


## Evaluation uncertainty

- The walk-forward report now shows paired match-level bootstrap intervals for the difference in Brier score and log loss: calibrated model vs raw probabilities, and calibrated model vs normalized market probabilities where complete odds exist.
- Differences are defined as first method minus second; negative values favor the first method. If a 95% interval includes zero, the sample does not establish a clear advantage on that metric.
- Bootstrap intervals are approximate and do not remove serial dependence, odds timestamp uncertainty, selection bias, or the effects of repeatedly tuning against the same historical period. They must not be treated as proof of profitability.
