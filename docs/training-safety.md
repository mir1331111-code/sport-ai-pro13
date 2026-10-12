# Training data safety

## Changes in this branch

- Historical training rows are validated before they reach `Engine.learn_step`.
- Missing scores are rejected rather than silently converted to 0-0.
- Scores must be finite, non-negative integer values (0–30).
- Rows without teams/dates and rows dated in the future are rejected.
- Training rows across leagues/seasons are merged and sorted by date before model updates.
- Decision snapshot writes report actual inserts and invalidate the read cache.

## Verification

Run:

```bash
python -m unittest discover -s tests -v
```

## Important limitations

This is a data-integrity fix, not proof that the model is profitable or that online self-learning is complete. Same-day historical rows are ordered deterministically by league and team because many sources lack kickoff times; they cannot be guaranteed to reflect the true within-day order. Calibration still needs a separate leakage audit and out-of-time evaluation before automatic model promotion should be enabled.
