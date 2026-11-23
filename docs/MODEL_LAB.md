# Fixed Holdout Model Lab v1

Model Lab v1 produces a reproducible 1X2 forecast-quality report. It reuses the existing
model.Engine and the project's existing Brier, Log Loss, reliability and clustered-bootstrap
metric helpers. It does not introduce a new forecasting formula or change live staking.

## Frozen evaluation design

The standard league set is fixed to the football-data.co.uk division codes:

| Code | League |
|---|---|
| E0 | English Premier League |
| D1 | German Bundesliga |
| I1 | Italian Serie A |
| SP1 | Spanish La Liga |
| F1 | French Ligue 1 |

The holdout start and end dates are mandatory and inclusive. Rows before the start date warm
the model; rows within the holdout range are scored. The evaluation is prequential: parameters,
feature policy and market source are frozen, all fixtures on a calendar date are forecast before
any result from that date is ingested, and prior holdout results may update the model state for
later dates. Do not use holdout results to tune parameters/features or re-label the same dates as
an independent retest.

Four identical-date arms are reported: full model, without H2H, without rest-day adjustments,
and without xG. xG provenance is separated into provider, modelled proxy, and unlabelled legacy
inputs. The model's multiclass Brier score is the **sum** of squared errors over H/D/A, matching
the existing implementation.

## Fixed bookmaker benchmark

Choose exactly one source per run: B365, Pinnacle, WilliamHill, or Bwin. A match is included
in the market comparison only when that same source has a complete, valid H/D/A price triplet.
Prices are de-vigged proportionally. There is no fallback to a different bookmaker and no average
line. Model and market metrics are calculated on exactly the same paired fixtures.

A bookmaker column set alone does not prove when its odds were captured. To make a public
pre-kickoff market claim, provide both a timezone-aware odds-capture timestamp column and a
timezone-aware kickoff timestamp column. The evaluator requires the capture timestamp to be
strictly earlier than kickoff for every matched fixture. Without this metadata the market
comparison is labelled diagnostic-only and cannot satisfy the strict promotion gate.

## Run

~~~bash
python -m model.holdout path/to/top5_history.csv \
  --holdout-start 2024-08-01 \
  --holdout-end 2026-05-31 \
  --market-source B365 \
  --odds-timestamp-column OddsCapturedAt \
  --kickoff-timestamp-column KickoffAt \
  --output-dir reports/model_lab
~~~

The date range above is an example only. Select and freeze an actually untouched period
appropriate to the supplied dataset; do not copy example dates blindly. If the source file has
no capture/kickoff timestamp columns, omit both timestamp options and retain the report's
unverified-snapshot warning.

The command writes:
- JSON: complete machine-readable report and experiment manifest;
- Markdown: concise public-facing evaluation;
- SVG: reliability diagrams for P1, X and P2;
- `model-lab-v1-gate.json`: a small fail-closed promotion marker consumed by the scanner.

Before a strict PASS exists, the scanner keeps O/U 2.5 forecast-only and excludes it from
automatic virtual-portfolio selection. After a PASS marker is present, O/U can be considered
again only if its own exact-line paired bookmaker, quote-source, EV, edge and Kelly checks pass.
BTTS, corners and cards remain forecast-only. The marker is written in the report output
directory; if that directory differs from the app's working directory, set
`MODEL_LAB_GATE_PATH` in the scanner deployment to the marker's path. Missing or malformed
markers disable secondary-market promotion.

The manifest records the dataset SHA-256, model revision, holdout dates, fixed leagues, model
parameters, bookmaker fields, de-vig method and thresholds. Re-run on identical inputs and code
to reproduce the same forecast metrics; the generated-at timestamp is intentionally new.

## Promotion gates

The default strict gate requires all of the following:

- at least 3,000 holdout fixtures;
- at least 400 fixtures in each of the five fixed leagues;
- at least 30 independent calendar-day clusters;
- multiclass Brier (sum over H/D/A) no greater than 0.57;
- Log Loss no greater than 0.99;
- at least 3,000 same-bookmaker paired fixtures with verified pre-kickoff timestamps;
- the upper 95% matchday-clustered confidence bound for model-minus-market Log Loss no greater
  than +0.02.

A proven market advantage is reported separately and requires the upper 95% confidence bound to
be below zero. Accuracy is published but is not a promotion criterion. A report that misses any
threshold is still generated as NOT_READY; no synthetic result is substituted.

Passing forecast-quality gates is not evidence of betting profitability. It is not authorization
to increase stakes or enable other market families. Keep O/U, BTTS, corners, cards, handicap and
LLM outputs forecast-only until they have separate market-specific validation. The Model Lab
never changes Dixon-Coles, Elo, Kelly, value gates or the live scanner.


## In-app Fixed Holdout runner

Open **Analytics → Model Lab** and use the Fixed Holdout Model Lab v1 section. Upload a historical CSV with
Date, Div, HomeTeam, AwayTeam, FTHG and FTAG (or the documented aliases). Enter the untouched holdout
start/end dates explicitly; the app does not choose dates for you. Select exactly one bookmaker source.

For a strict promotion decision, also select both a timezone-aware odds-capture timestamp column and a
timezone-aware kickoff timestamp column. The evaluator rejects quotes that are missing a timestamp,
naive, or captured at/after kickoff. If the CSV has no timestamp columns, it still publishes the model
metrics and a diagnostic-only market comparison, but the strict gate remains NOT_READY.

The UI runner uses the fixed evaluator configuration and writes the JSON, Markdown, reliability SVG and
promotion-gate JSON under `reports/model_lab` (or the directory containing `MODEL_LAB_GATE_PATH`).
If the gate file path is overridden, the generated marker is also copied to that exact configured path.
A PASS marker is consumed by the live scanner; absence or malformed content fails closed.
