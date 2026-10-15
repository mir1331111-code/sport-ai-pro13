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

## Broad league coverage and fixture scanning

The scanner merges fixtures from football-data.org (when a token is configured), TheSportsDB, and ESPN. Each source is isolated: a timeout, missing token, or quota exhaustion on one provider does not prevent the other available sources from contributing matches. Cross-provider duplicates are merged by local match date and normalized home/away names, preserving the preferred provider's official fixture ID and odds while filling missing badges/metadata from other feeds.

Historical engine training keeps the original priority leagues first and then includes every configured seasonal CSV division, plus English League One/Two and German 3. Liga codes. A configured division with no current CSV data is skipped naturally by the source; it no longer disappears merely because it was omitted from a hard-coded 15-league list.

The engine cache fingerprint is bumped to v13.4.1 so deployments rebuild the model with the expanded training coverage. The scanner now builds a cross-league **Best picks of the day** shortlist. It compares all model markets for a fixture against available real bookmaker prices, requires at least 3% model edge and 3% expected value plus positive fractional Kelly, and ranks qualified candidates across leagues. Model fair odds are never treated as bookmaker prices or used to create paper bets. If no priced candidate passes the filters, the shortlist explicitly reports that there are no qualified value picks. The ranking is a decision aid, not a guarantee of profit; probability calibration and walk-forward validation remain necessary.
