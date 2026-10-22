# Sport-AI-Pro13

Football match probability forecasting with a Poisson/Dixon–Coles score model, Elo, team form, head-to-head adjustments, probability calibration, and an append-only forecast ledger.

## API key persistence

Keys entered in the sidebar are saved separately in `.neuro_secrets.json` with owner-only file permissions on the app host. They are not written into `neuro_local.json`, synced through the portfolio Gist, or included in downloaded backups; `.gitignore` excludes the local secret file. If the host uses an ephemeral filesystem or the app is redeployed/restarted, use the host's environment variables or Streamlit Secrets for durable configuration. Re-enter UI-only keys once after upgrading if the earlier version already removed them from portfolio state.

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

The engine cache fingerprint is bumped to v13.4.5 so deployments rebuild the model after removing unvalidated home/form multipliers and adding team-rate shrinkage. The scanner builds a cross-league **Best picks of the day** shortlist using available real bookmaker prices, minimum model edge and expected-value filters, and positive fractional Kelly. Automatic picks now require at least 5% Edge and 5% EV in addition to the configured minimum probability. The probability used for ranking and confidence labels is the final market-blended probability, not the raw model probability; its label is explicitly not a historical hit-rate estimate. The provisional blend moves 40% of the way from the raw bookmaker-implied probability toward the model probability; with a single entered price this is not de-vigged and remains a heuristic pending out-of-sample tuning. Model fair odds are never treated as bookmaker prices or used to create bets. Automatic value selection and portfolio staking are restricted to 1X2 until other markets have separate out-of-sample validation. If no priced candidate passes the filters, the shortlist reports no qualified value picks. The ranking is a decision aid, not a guarantee of profit.


## Manual odds and historical corners/cards

The scanner's portfolio flow is manual-only: a scan saves forecasts and never creates or funds a bet. A user selects a market, enters the bookmaker's decimal price, and the portfolio button is enabled only when probability, minimum Edge, EV, and fractional Kelly filters pass. Estimated/fair odds are not prefilled as bookmaker prices.

Markets currently exposed in the manual selector include 1X2, goals over/under 2.5, BTTS yes/no, total corners over/under 8.5/9.5/10.5, and total cards over/under 3.5/4.5/5.5. Enter the bookmaker's decimal odds for the exact selected outcome (for example, if the selected market is goals over 2.5 and the bookmaker lists 1.85, enter `1.85`). Do not enter the model's fair odds or the stake amount. The UI shows model probability, raw implied probability, blended probability, edge and EV. Automatic scan-based staking is enabled for 1X2. Goals over/under 2.5 can also be auto-added only when the startup walk-forward backtest finds at least 500 paired historical bookmaker-priced fixtures and the model beats the de-vigged market on both Brier score and log loss with paired 95% bootstrap confidence intervals entirely below zero. The scanner then needs a real totals price from The Odds API; manual bookmaker odds can use the same historical gate. Otherwise totals stay forecast-only. BTTS, corners and cards remain view-only. Corner/card options are shown only where the historical feed contains at least 30 observed match rows for that league/market.

The new historical statistics baseline learns from football-data.co.uk columns `HC`, `AC`, `HY`, `AY` and recorded `HR`, `AR`, using league rates and same-venue team rates. It is trained from five chronological seasonal CSVs per configured league alongside the goals engine. Each scan logs a leakage-safe walk-forward MAE for corner/card totals where at least 30 earlier observations exist; the cache fingerprint was bumped to force retraining. Missing stats are not imputed as zero. Corner/card totals use a Poisson approximation; this is an initial baseline, **not yet a validated or calibrated betting model**. Backtest it chronologically against an untouched holdout before treating probabilities as reliable. These markets must be settled using actual corner/card results, not the final goal score.


## Bookmaker benchmark and staking controls

The same chronological evaluator also reports goals over/under 2.5 binary Brier/log-loss and bootstrap intervals (`goals_over25`). When historical CSVs contain paired `B365>2.5`/`B365<2.5`, `Avg>2.5`/`Avg<2.5`, or supported equivalent columns, it compares model forecasts and de-vigged market probabilities on identical fixtures (`goals_over25_candidate_minus_market`). This is an evaluation harness, not proof of an edge; actual seasonal data must be run before promoting automatic total bets.

The model.walk_forward.walk_forward_compare evaluator reports a 1X2 bookmaker benchmark when a complete odds triplet is present (B365, Pinnacle/PS, William Hill, average odds, or Betway column sets). It reports proportional, power and Shin de-vig estimates. Compare candidate_on_market_rows directly with market; both use identical fixtures. Metrics include Brier, log loss, accuracy, an explicitly labelled H/D/A-order RPS diagnostic, and bootstrap 95% intervals. RPS for 1X2 depends on the chosen class order because outcomes are not naturally ordinal; do not interpret it as a uniquely canonical metric. The walk-forward harness accumulates calibration targets only after forecasting each calendar date. Rows without complete historical odds are excluded from paired market comparisons.

The temporal half-life is configurable in walk_forward_compare(half_life_days=...). tune_half_life(rows) compares 60, 90, 120, 180, 270 and 365 days by chronological log loss, preferring paired market-covered rows where available. This is a tuning tool, not a final performance estimate: select on a development period and confirm once on a later untouched holdout.

Real/manual odds must pass minimum probability, minimum Edge, minimum EV, and positive Kelly. Stake is capped at 2% of bank. A single entered price's implied probability still contains bookmaker margin; the current 40% blend is a conservative provisional setting, not an optimized coefficient. It must be tuned on a chronological out-of-fold dataset before making claims of model advantage.

API credentials are read from password-widget session state, environment variables, or Streamlit secrets and are no longer persisted into local app state. .gitignore excludes local databases, credentials, and generated caches. The scanner calls the existing `data.sources.prognozist_marathon_odds` adapter (rather than importing a missing module), then falls back to Marathon, TheRundown and The Odds API when a source has no valid line. Prognozist scraping uses a transparent User-Agent and is disabled unless `PROGNOZIST_PERMISSION=1` is explicitly set after confirming permission; it also fails closed if `robots.txt` is unavailable or disallows the requested URLs. The odds source is not guaranteed to return every fixture: verify source logs and current site terms on the deployment host.
