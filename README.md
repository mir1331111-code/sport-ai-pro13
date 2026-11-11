# Sport-AI-Pro13

Football match probability forecasting with a Poisson/Dixon–Coles score model, Elo, team form, head-to-head adjustments, probability calibration, and an append-only forecast ledger.

## LLM provider routing and API-key trust

LLM requests are sent directly from the app host to the endpoint shown for the selected provider over HTTPS. The app does **not** route user LLM API keys through an author-controlled proxy. The default Groq provider uses Groq's official OpenAI-compatible endpoint, `https://api.groq.com/openai/v1`; other providers use their configured provider endpoints. The selected provider receives the API key and prompt content, so review that provider's own terms and privacy policy before enabling it. Do not enter credentials for a provider unless you trust that provider.

## API key persistence

Keys entered in the sidebar are saved separately in `.neuro_secrets.json` with owner-only file permissions on the app host. They are not written into `neuro_local.json`, synced through the portfolio Gist, or included in downloaded backups; `.gitignore` excludes the local secret file. If the host uses an ephemeral filesystem or the app is redeployed/restarted, use the host's environment variables or Streamlit Secrets for durable configuration. Re-enter UI-only keys once after upgrading if the earlier version already removed them from portfolio state.

### Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for the Streamlit Community Cloud setup, `App.py` entrypoint, and SStats key setup.

### Context pipeline and additional markets

See [`CONTEXT_AND_MARKETS.md`](CONTEXT_AND_MARKETS.md) for SStats context enrichment, context confidence and Value gates, historical CSV schemas for women/national teams, referee/card handling, halftime corner splits, and Asian handicap settlement calculations.

# Model evaluation and xG ablation

Run a leakage-safe chronological evaluation on historical CSV-like records:

```python
import csv
from model.walk_forward import walk_forward_compare

with open("matches.csv", newline="", encoding="utf-8-sig") as f:
    report = walk_forward_compare(list(csv.DictReader(f)))
print(report)
```

Required match fields are `Date`, `Div`, `HomeTeam`, `AwayTeam`, `FTHG`, and `FTAG` (common aliases are accepted). The evaluator sorts by date and forecasts all matches on a calendar date before adding any result from that date.

The core xG report has three arms:
- `baseline`: goals-only, equal-weight history, fixed league rho.
- `temporal_goals_only`: goals-only with temporal weighting and league-specific rho.
- `candidate`: the same temporal model plus historical xG where enough data exists.

The `feature_ablation` section adds four matched arms using identical fixtures, temporal weighting, league rho and xG history: the full candidate, candidate without H2H, candidate without rest-day adjustments, and candidate without both features. The `full_minus_without_*` values are paired differences with fixture-bootstrap 95% confidence intervals. Negative Brier/log-loss differences favor keeping the feature; positive differences favor removing it. This gives a direct basis for deleting unhelpful complexity instead of assuming the feature helps.

**Use `xg_delta_vs_same_temporal_model` to isolate the xG contribution.** Lower Brier and log loss are better; higher top-class accuracy is better. `xg_rows_available` counts input rows with paired xG, while `xg_forecasts_used` counts forecasts where the engine actually enabled the xG blend after its league/team sample thresholds. `xg_usage_share` is the enabled share in that walk-forward run. If no xG data is present, candidate and temporal-goals-only arms must match.

The model accepts paired fields such as `HxG`/`AxG`, `home_xg`/`away_xg`, and `xG`/`xGA`. Values outside 0–8 or malformed values are ignored. The blend requires at least 30 paired league xG observations and at least four same-venue team observations; its weight grows conservatively with sample size. xG is ingested only after each match's forecast, never from the fixture being predicted.

**Feed limitation:** the seasonal CSV loader preserves source columns, but it does not manufacture xG. Coverage therefore depends on the actual CSV fields; standard football-data.co.uk seasons often provide goals and odds but not a complete xG history. The live fixture providers in this app do not currently supply a verified, consistent pre-match xG history. The report should be run on a dataset with genuine historical xG values before xG is enabled as a claimed improvement. No model gain is claimed by the presence of this feature alone.

For a model promotion decision, use an untouched later-period holdout and, where available, compare with pre-kickoff bookmaker probabilities. This diagnostic walk-forward is not a profit guarantee.

## Runtime ledger diagnostics

The main-model forecast ledger reports ten-bin reliability tables separately for P1, draw and P2, using only forecasts with a verified settled result. It also stores `xg_used` for new main-model forecasts and reports the share among rows where that flag was actually recorded. Legacy rows created before the flag existed remain untracked; their xG use is not inferred retrospectively. The independent LLM comparison reports paired per-fixture Brier/log-loss error correlations; this is a diversity diagnostic only, not an implemented weighted ensemble.

## Tests

```bash
python -m unittest discover -s tests -v
```

The Streamlit app entry point is `App.py`.

## Broad league coverage and fixture scanning

SStats is the primary fixture and bookmaker-odds source. When SStats returns matches, the scanner uses that list alone and does not query ESPN, TheSportsDB or football-data.org for a duplicate schedule. Those providers remain fallback sources only when SStats is not configured or returns no fixtures. SStats context endpoints enrich those same fixture rows by SStats-owned IDs; they never add duplicate matches.

Historical engine training keeps the original priority leagues first and then includes every configured seasonal CSV division, plus English League One/Two and German 3. Liga codes. A configured division with no current CSV data is skipped naturally by the source; it no longer disappears merely because it was omitted from a hard-coded 15-league list.

The statistical engine now namespaces team form, Elo ratings, H2H, rest-day history and probability calibration by the exact competition division code. League goal-rate and xG baselines are also competition-specific. A sparse competition shrinks toward neutral priors rather than borrowing aggregate rates from other divisions. This prevents men's, women's, national-team, youth and separate competition datasets from silently calibrating or training one another; as a consequence, a competition with little historical data may produce fewer strong candidates until its own sample is sufficient.

The engine cache fingerprint is bumped to v13.4.5 so deployments rebuild the model after removing unvalidated home/form multipliers and adding team-rate shrinkage. The scanner builds a cross-league **Best picks of the day** shortlist using available real bookmaker prices, minimum model edge and expected-value filters, and positive fractional Kelly. Automatic picks now require at least 5% Edge and 5% EV in addition to the configured minimum probability. The probability used for ranking and confidence labels is the final market-blended probability, not the raw model probability; its label is explicitly not a historical hit-rate estimate. The provisional blend moves 40% of the way from the raw bookmaker-implied probability toward the model probability; with a single entered price this is not de-vigged and remains a heuristic pending out-of-sample tuning. Model fair odds are never treated as bookmaker prices or used to create bets. Automatic portfolio additions are disabled by default. When explicitly enabled, only a complete named-bookmaker 1X2 line is eligible; full-time goals over/under 2.5 is additionally eligible only when at least 500 paired historical fixtures show lower Brier and log loss than the de-vigged bookmaker benchmark with both paired 95% confidence intervals below zero. BTTS, corners and cards remain forecast-only and are never auto-added. Other goal-total lines may appear as separately screened signals, but they are not automatically staked. If no priced candidate passes the filters, the shortlist reports no qualified value picks. The ranking is a decision aid, not a guarantee of profit.


## Manual odds and historical corners/cards

Automatic additions to the virtual portfolio are disabled by default. If explicitly enabled, only a complete same-bookmaker 1X2 line or a full-time goals 2.5 line that passes the separate bookmaker-benchmark gate can be considered. BTTS, corners and cards remain forecast-only. The feature only updates the virtual portfolio and never places a real bookmaker bet. A user can also enter a bookmaker's decimal price manually; estimated/fair odds are never prefilled as bookmaker prices.

Markets currently exposed in the manual selector include 1X2, goals over/under 2.5, BTTS yes/no, total corners over/under 8.5/9.5/10.5, and total cards over/under 3.5/4.5/5.5. Enter the bookmaker's decimal odds for the exact selected outcome (for example, if the selected market is goals over 2.5 and the bookmaker lists 1.85, enter `1.85`). Do not enter the model's fair odds or the stake amount. The UI shows model probability, raw implied probability, blended probability, edge and EV. Automatic scan-based portfolio additions are disabled by default. If explicitly enabled, 1X2 requires a complete named-bookmaker three-way line; goals over/under 2.5 can also be auto-added only when the startup walk-forward backtest finds at least 500 paired historical bookmaker-priced fixtures and the model beats the de-vigged market on both Brier score and log loss with paired 95% bootstrap confidence intervals entirely below zero. The scanner then needs a real totals price from a supported odds provider (SStats detailed bookmaker markets when available, or an explicitly imported manual quote); missing or uncorroborated prices remain blank. The historical gate is a safeguard, not proof of future profitability. Otherwise totals stay forecast-only. BTTS, corners and cards remain view-only. Corner/card options are shown only where the historical feed contains at least 30 observed match rows for that league/market. The current binary EV model supports half-point lines only; integer and quarter lines can push or split a stake and are excluded from actionable value calculations until those settlement outcomes are modelled explicitly.

The new historical statistics baseline learns from football-data.co.uk columns `HC`, `AC`, `HY`, `AY` and recorded `HR`, `AR`, using league rates and same-venue team rates. It is trained from five chronological seasonal CSVs per configured league alongside the goals engine. Each scan logs a leakage-safe walk-forward MAE for corner/card totals where at least 30 earlier observations exist; the cache fingerprint was bumped to force retraining. Missing stats are not imputed as zero. Corner/card totals use a Negative-Binomial approximation when the historical counts are overdispersed and otherwise a Poisson approximation; this is an initial baseline, **not yet a validated or calibrated betting model**. Backtest it chronologically against an untouched holdout before treating probabilities as reliable. These markets must be settled using actual corner/card results, not the final goal score.


## Bookmaker benchmark and staking controls

The same chronological evaluator also reports goals over/under 2.5 binary Brier/log-loss and bootstrap intervals (`goals_over25`); the runtime market-promotion gate uses 1,000 paired bootstrap resamples and requires at least 500 paired fixtures. When historical CSVs contain paired `B365>2.5`/`B365<2.5`, `Avg>2.5`/`Avg<2.5`, or supported equivalent columns, it compares model forecasts and de-vigged market probabilities on identical fixtures (`goals_over25_candidate_minus_market`). This is an evaluation harness, not proof of an edge; actual seasonal data must be run before promoting automatic total bets.

The model.walk_forward.walk_forward_compare evaluator reports a 1X2 bookmaker benchmark when a complete odds triplet is present (B365, Pinnacle/PS, William Hill, average odds, or Betway column sets). It reports proportional, power and Shin de-vig estimates. Compare candidate_on_market_rows directly with market; both use identical fixtures. Metrics include Brier, log loss, accuracy, an explicitly labelled H/D/A-order RPS diagnostic, and bootstrap 95% intervals. RPS for 1X2 depends on the chosen class order because outcomes are not naturally ordinal; do not interpret it as a uniquely canonical metric. The walk-forward harness accumulates calibration targets only after forecasting each calendar date. Rows without complete historical odds are excluded from paired market comparisons.

The temporal half-life is configurable in walk_forward_compare(half_life_days=...). tune_half_life(rows) compares 60, 90, 120, 180, 270 and 365 days by chronological log loss, preferring paired market-covered rows where available. This is a tuning tool, not a final performance estimate: select on a development period and confirm once on a later untouched holdout.

Real/manual prices must pass the minimum probability, market-specific historical validation, minimum Edge, minimum EV, and positive Kelly. The production staking policy is fixed at **¼ Kelly**, with a hard maximum of **1% of bankroll per market**, **2% combined exposure per fixture** across correlated markets, and **5% cumulative stake turnover per calendar day**. The minimum executable decimal price is **1.70**; a lower price may remain visible as a forecast but cannot be staked. Limits are checked centrally for manual and scan-based virtual-portfolio additions, against all current open and same-day settled stakes. This is paper/virtual portfolio accounting only; the app does not place bookmaker bets. A single entered price's implied probability still contains bookmaker margin; the current 40% blend is a provisional setting, not an optimized coefficient. It must be tuned on a chronological out-of-fold dataset before making claims of model advantage.

API credentials are read from password-widget session state, environment variables, or Streamlit secrets and are no longer persisted into local app state. .gitignore excludes local databases, credentials, and generated caches. ## Highlightly odds source

Highlightly is an optional odds provider. When configured and permitted by the account plan, the scanner resolves the exact fixture through Highlightly's `/football/matches` endpoint and maps `/football/odds` records by `matchId`. SStats.net is also supported as a separate odds source; a total-goals quote from one named bookmaker is displayed with an explicit uncorroborated label, while two or more independent named bookmakers are required before that SStats quote can pass the bookmaker-confirmation gate for automated staking. Anonymous compact-feed totals are not presented as bookmaker prices. Fixture feeds such as TheSportsDB, ESPN, and football-data.org are not treated as bookmaker odds. The app does not infer bookmaker prices from model fair odds.

Configure `HIGHLIGHTLY_API_KEY` in the host environment or Streamlit Secrets, or enter it in the scanner's Highlightly key field. Highlightly's documentation states that odds are unavailable on Free/Basic plans, so a suitable paid plan is required. Market coverage depends on the data returned for that fixture; if a specific total, BTTS, corner or card line is absent, the app leaves that price blank rather than substituting another market. The goals model forecasts score probabilities; corners/cards use the historical statistical baseline described above and are not yet validated as betting models.

Official reference: https://highlightly.net/football-api/documentation/ and https://highlightly.net/football-api/documentation/getting-started/pagination/

## Free SStats.net football odds source

The scanner uses SStats.net as its primary source with one personal API key configured in the sidebar or Streamlit Secrets. It requests upcoming games through `/Games/list` and reads the compact match-level 1X2 and totals prices included in the game response. For up to 100 fixtures per scan, it additionally requests `/Odds/{gameId}` to enrich the card with bookmaker-specific markets such as BTTS, totals, corners, cards, and handicap where supplied.

The key is sent only as the documented `apikey` query parameter. It is kept out of logs, portfolio data, and Gist synchronization; UI-entered keys are stored in the private local secrets file. The API refreshes prematch match information approximately every three hours, so one daily scan is sufficient for this workflow but is not a live odds feed. A detailed card identifies the bookmaker for the selected outcome when the provider supplies that mapping. Compact match-list prices may not include bookmaker attribution. The same SStats key is also used for `Games/injuries`, `Ls/GameInfo`, and `Ls/GameLineups` where those endpoints return data. `Ls/List` resolves an event ID only to enrich an already-selected SStats fixture; it does not create a second schedule source.

Official documentation: https://api.sstats.net/docs/

## TOP MATCHES · Groq Deep Analysis

After the standard scanner completes, the app builds a separate shortlist of up to seven unique SStats fixtures for deeper review. A candidate must meet the higher of the existing Strong threshold (65%) and the user's configured minimum probability, have at least five completed home matches for the home team and five away matches for the away team, have a complete SStats 1X2 price set, and contain at least one real market quote at 1.70 or higher. The shortlist is not padded: if only two fixtures pass, only two are sent to Groq. These are research candidates, not automatically approved Value bets.

For the selected fixtures, the app refreshes the SStats context using the same SStats key and then sends one compact batch request to the configured Groq model. Each match dossier contains the source-provided starting XI (only when confirmed), injury/suspension feed records, referee assignment when confirmed, recent form, venue-specific goal history, and available historical corners/cards and referee-card samples. Unknown data stays explicitly unknown; an empty or unconfirmed injury feed is never described as proof that no players are injured. Groq returns an independent 1X2 view plus separate comments on lineups, injuries, referee, form, risks, and overall match conditions.

The Groq prompt does not receive the main model's probabilities, its selected pick, bookmaker prices, EV, Edge, or Kelly. The app compares the Groq pick with the main model only after receiving and validating the independent response. Groq cannot approve a bet, change the primary forecast, or override any Strong/Value/portfolio safeguard. The separate **🏆 TOP MATCHES · GROQ** tab shows the selected matches, both independent views, source facts and context observation times. The button **Обновить SStats-досье и повторить анализ Groq** rechecks the selected fixtures and sends a new batch, which is useful when SStats publishes lineups closer to kickoff.

The batch response is schema-constrained, validated against the exact fixture IDs submitted, and cached for 30 minutes. Only validated independent probabilities are written to the existing forecast ledger format. API/LLM keys are never included in the dossier or prompt.

The scanner records immutable pre-kickoff 1X2 probabilities for the statistical model and any validated Groq opinion in the forecasts table in the app's neuro.db SQLite database. Records are idempotent per fixture and model version. Only fixtures with a parseable kickoff time and a future kickoff are recorded; fixtures without a usable time or already started are explicitly skipped. The timezone used to parse displayed fixture times defaults to Europe/Warsaw and can be overridden with APP_TIMEZONE.

The Statistics tab's **AI versus model** view compares the two probability distributions on exactly the same settled fixtures. It reports accuracy, multiclass Brier score, log loss, and paired prediction history with each system's pick/probabilities. Completed outcomes are looked up through the app's existing ESPN, football-data.org, and OpenLigaDB score adapters; only an explicitly completed FT result is auto-settled. AET/penalty outcomes, uncertain team matches, missing data, and unconfirmed scores are not silently settled. The result check is throttled during normal reruns and can be triggered manually from the AI comparison tab.

These metrics are forecast-quality measures, not betting ROI. A small sample is not enough to establish which forecaster is better. Live AI scores and comparative performance appear only after genuine forecasts have been logged before kickoff and trustworthy final results have been confirmed; no synthetic performance numbers are prefilled.

## Algorithm integrity update 13.12.0

The production trainer now prepares only historical fixtures with a valid date, non-empty team identities, and finite non-negative integer final scores. Missing scores are never replaced with 0-0. All configured leagues are merged into one global chronological stream; every fixture on a calendar date is forecast before any result from that date is ingested. This prevents same-day results from leaking into later same-day forecasts and calibration updates. The training log reports excluded rows by reason.

The historical validation cache is fingerprinted by the application/model version as well as the exact input rows and matrix size. A model-version change therefore forces a fresh walk-forward calculation instead of reusing a report produced by an earlier algorithm.

The scanner separates **model forecasts** from **actionable Value**:
- 1X2 is eligible for Value only after at least 500 paired historical fixtures show lower Brier score and log loss than a complete de-vig bookmaker 1X2 line, with both paired 95% bootstrap intervals below zero. The live quote must also be a complete line attributed to one named bookmaker.
- Full-time goals over/under 2.5 has the same paired bookmaker-benchmark requirement and also retains its exact-line chronological check and same-bookmaker live-price requirements.
- Other goals lines, BTTS, corners and cards remain forecast-only until their own exact-market, paired bookmaker benchmark is implemented and passes out-of-sample checks. A comparison against the league average alone is not evidence of bookmaker value.
- A market may still appear as a high-ranked model forecast without being approved as a bet. Forecast probability, Value status, and portfolio eligibility are separate concepts.

The 1X2 market benchmark uses its own paired bootstrap resampling budget so the bookmaker promotion gate can be calculated without paying for unnecessary bootstrap intervals on every diagnostic metric. The validation cache is versioned to invalidate reports from older model logic.

These controls are deliberately conservative: a market may show no Value picks until sufficient independent historical evidence and attributable bookmaker prices are available. Do not lower these gates merely to increase the number of displayed bets.

## Balanced forecast profile and strong-forecast threshold

The default scanner ordering is now **balanced** rather than probability-only or EV-only. In the combined shortlist, the display score weights model confidence at 55%, expected value at 20%, no-vig edge at 15%, and fractional Kelly at 10%. In the market-specific tabs, an already-approved Value recommendation is ranked with 60% event probability, 25% executable EV, and 15% no-vig edge. These are display-order weights only; they do not change eligibility.

A dedicated `STRONG_FORECAST_MIN_PROB = 0.65` setting labels a market forecast as **strong** when its own selected event probability is at least 65% (inclusive). Forecasts below 65% remain visible in the full forecast view but are clearly labelled as below the strong threshold. The setting is shared with the existing 65% virtual-portfolio probability floor; it does not replace the user-adjustable minimum probability for Value candidates.

Important separation:
- EV and edge affect balanced ordering only when the fixture already has a recommendation from the strict Value selector. That means the exact market passed its out-of-sample bookmaker benchmark, a complete line from one named bookmaker, the minimum Edge and EV, and positive Kelly.
- Forecasts without a validated Value recommendation are ranked by their own event probability. Unverified or incomplete odds cannot improve their rank.
- The Value gates, bookmaker benchmarks, Edge/EV floors, 2% Kelly stake cap and settlement rules have not been relaxed.
- The 65% label is not a claim of 65% historical accuracy and is not a guarantee. Actual calibration and profitability must still be measured from immutable pre-match forecasts and settled outcomes.

The scanner's manual 1X2 view also now requires a complete P1/X/P2 line attributed to the same named bookmaker before it labels the market line coherent.

