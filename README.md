# Sport-AI-Pro13

Football match probability forecasting with a Poisson/Dixon–Coles score model, Elo, team form, head-to-head adjustments, probability calibration, and an append-only forecast ledger.

## Model evaluation

The engine now supports a temporal candidate using exponential decay (180-day half-life) for dated scoring history and a conservative, league-specific Dixon–Coles rho estimate after at least 80 dated league matches. Rest days are exposed as audit features but **do not directly change probabilities yet**: that adjustment should only be enabled after an out-of-sample test shows a benefit.

Run the chronological comparison on historical CSV-like records from Python:

```python
import csv
from model.walk_forward import walk_forward_compare

with open("matches.csv", newline="", encoding="utf-8-sig") as f:
    report = walk_forward_compare(list(csv.DictReader(f)))
print(report)
```

Expected fields are `Date`, `Div`, `HomeTeam`, `AwayTeam`, `FTHG`, and `FTAG` (common alternative names are also accepted). Rows are sorted by date. All fixtures on the same calendar date are forecast before any results from that date are added, preventing same-day result leakage.

The report compares:
- **Baseline:** equal-weight historical means and fixed league rho.
- **Candidate:** temporal weighting and bounded, shrunk league-specific rho.

Lower Brier score and log loss are better; higher top-class accuracy is better. The report is diagnostic: it does not claim the candidate is superior. Before promoting a model, evaluate on a separate untouched chronological holdout and compare against bookmaker probabilities where historical pre-kickoff odds are available.

## Tests

```bash
python -m unittest discover -s tests -v
```

The Streamlit app entry point is `App.py`.
