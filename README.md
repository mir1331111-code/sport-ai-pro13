# Sport-AI-Pro13

Football match probability forecasting with a Poisson/Dixon–Coles score model, Elo, team form, head-to-head adjustments, probability calibration, and an append-only forecast ledger.

## Model evaluation and xG ablation

Run a leakage-safe chronological evaluation on historical CSV-like records:

```python
import csv
from model.walk_forward import walk_forward_compare

with open("matches.csv", newline="", encoding="utf-8-sig") as f:
    report = walk_forward_compare(list(csv.DictReader(f)))
print(report)
```

Required match fields are `Date`, `Div`, `HomeTeam`, `AwayTeam`, `FTHG`, and `FTAG` (common aliases are accepted). The evaluator sorts by date and forecasts all matches on a calendar date before adding any result from that date.

The report has three arms:
- `baseline`: goals-only, equal-weight history, fixed league rho.
- `temporal_goals_only`: goals-only with temporal weighting and league-specific rho.
- `candidate`: the same temporal model plus historical xG where enough data exists.

**Use `xg_delta_vs_same_temporal_model` to isolate the xG contribution.** Lower Brier and log loss are better; higher top-class accuracy is better. `xg_rows_available` counts input rows with paired xG, while `xg_forecasts_used` counts forecasts where the engine actually enabled the xG blend after its league/team sample thresholds. If no xG data is present, candidate and temporal-goals-only arms must match.

The model accepts paired fields such as `HxG`/`AxG`, `home_xg`/`away_xg`, and `xG`/`xGA`. Values outside 0–8 or malformed values are ignored. The blend requires at least 30 paired league xG observations and at least four same-venue team observations; its weight grows conservatively with sample size. xG is ingested only after each match's forecast, never from the fixture being predicted.

**Feed limitation:** the seasonal CSV loader preserves source columns, but it does not manufacture xG. Coverage therefore depends on the actual CSV fields; standard football-data.co.uk seasons often provide goals and odds but not a complete xG history. The live fixture providers in this app do not currently supply a verified, consistent pre-match xG history. The report should be run on a dataset with genuine historical xG values before xG is enabled as a claimed improvement. No model gain is claimed by the presence of this feature alone.

For a model promotion decision, use an untouched later-period holdout and, where available, compare with pre-kickoff bookmaker probabilities. This diagnostic walk-forward is not a profit guarantee.

## Tests

```bash
python -m unittest discover -s tests -v
```

The Streamlit app entry point is `App.py`.

## Broad league coverage and fixture scanning

The scanner merges fixtures from football-data.org (when a token is configured), TheSportsDB, and ESPN. Each source is isolated: a timeout, missing token, or quota exhaustion on one provider does not prevent the other available sources from contributing matches. Cross-provider duplicates are merged by local match date and normalized home/away names, preserving the preferred provider's official fixture ID and odds while filling missing badges/metadata from other feeds.

Historical engine training keeps the original priority leagues first and then includes every configured seasonal CSV division, plus English League One/Two and German 3. Liga codes. A configured division with no current CSV data is skipped naturally by the source; it no longer disappears merely because it was omitted from a hard-coded 15-league list.

The engine cache fingerprint is bumped to v13.4.3 so deployments rebuild the model with the expanded training coverage. The scanner builds a cross-league **Best picks of the day** shortlist using available real bookmaker prices, minimum model edge and expected-value filters, and positive fractional Kelly. Model fair odds are never treated as bookmaker prices or used to create paper bets. If no priced candidate passes the filters, the shortlist reports no qualified value picks. The ranking is a decision aid, not a guarantee of profit.


## Manual odds and historical corners/cards

The scanner's portfolio flow is manual-only: a scan saves forecasts and never creates or funds a bet. A user selects a market, enters the bookmaker's decimal price, and the portfolio button is enabled only when probability, minimum Edge, EV, and fractional Kelly filters pass. Estimated/fair odds are not prefilled as bookmaker prices.

Markets currently exposed in the manual selector include 1X2, goals over/under 2.5, BTTS yes/no, total corners over/under 8.5/9.5/10.5, and total cards over/under 3.5/4.5/5.5. Corner/card options are shown only where the historical feed contains at least 30 observed match rows for that league/market.

The new historical statistics baseline learns from football-data.co.uk columns `HC`, `AC`, `HY`, `AY` and recorded `HR`, `AR`, using league rates and same-venue team rates. It is trained from the same previous/current seasonal CSVs as the goals engine, and the cache fingerprint was bumped to force retraining. Missing stats are not imputed as zero. Corner/card totals use a Poisson approximation; this is an initial baseline, **not yet a validated or calibrated betting model**. Backtest it chronologically against an untouched holdout before treating probabilities as reliable. These markets must be settled using actual corner/card results, not the final goal score.
