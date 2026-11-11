# NEURO BET PRO v14 — Analytics Terminal

This document describes the analytics screens and the evidence behind each displayed status. It intentionally separates data completeness, forecast performance, bookmaker value and portfolio exposure.

## Main navigation

- **OVERVIEW** — current scan, forecast-ledger metrics, data coverage and paper-portfolio summary.
- **LIVE SCANNER · FORECASTS · VALUE** — fixture scan, model probabilities, market prices and conservative Value gates.
- **DATA QUALITY** — per-fixture source status, freshness, missing information and detected conflicts.
- **PORTFOLIO · RISK** — virtual portfolio plus exposure, same-fixture clusters, drawdown and risk guards.
- **AI LEARNING** — settled pre-match forecast errors and descriptive comparisons by the features recorded before kickoff.
- **MODEL LAB** — 1X2 calibration, Brier score, log loss, accuracy, CLV and separate market-validation status.
- **API CENTER** — configured/not-configured keys, quota settings and messages from the latest source requests.
- Existing Statistics, Calculator and Backtest views remain available.

## DATA QUALITY semantics

The per-match completeness score is a weighted coverage index from 0 to 100. It is **not** a match probability, forecast confidence or empirical source-reliability score.

| Dimension | Maximum points |
|---|---:|
| API-Football fixture matched | 10 |
| Confirmed home starting XI | 20 |
| Confirmed away starting XI | 20 |
| Referee name returned | 10 |
| Injury endpoint returned (including an empty response) | 10 |
| At least one valid bookmaker price map | 15 |
| Kickoff timestamp available | 10 |
| Source timestamps present and within the freshness window | 5 |
| **Total** | **100** |

A detected team/category conflict caps the index at 40. Unknown timestamps do not receive freshness points.

Source statuses have distinct meanings:
- **not_configured** — no key is available in the current app environment.
- **unavailable** — the provider request failed or the latest diagnostics show the provider is unavailable.
- **not_matched** — the provider did not return a corresponding fixture for this card.
- **responded_empty** — the provider answered, but returned no records for that data family.
- **responded_partial** — a response arrived but did not contain a complete confirmed starting XI.
- **confirmed / reported / records_returned** — the corresponding data was present.
- **not_requested_yet** — lineup calls are intentionally deferred until the six-hour pre-kickoff window.
- **stale** — the recorded timestamp is older than the configured freshness window.
- **conflict** — for example, the lineup team identity differs from the fixture card.

An empty injuries response is not a confirmation that no players are injured. A connected API key is not the same as a successful fixture match. Unknown source reliability remains uncalibrated; this version does not invent reliability percentages.

## Competition categories and model isolation

The scanner defaults to **Men's football**. The scanner settings expose explicit modes for all categories or women's football.

The category is preserved from fixture source through fixture merging, card creation, training and forecast logging. Women's and unknown team identities use separate internal namespaces. Their league-goal baselines, xG averages, Dixon–Coles estimates and Platt calibration samples are kept separate from men's football.

Generic source labels such as "Soccer" are classified as **unknown**, not silently assumed to be men's football. Unknown-category fixtures are excluded from the default men's scan and cannot enter Value or the virtual portfolio. In the all-categories mode they may remain visible as informational forecasts, but are still isolated from men's training.

This prevents a women's team with the same display name as a men's club from inheriting the men's team form, Elo, H2H or calibration history. Existing cached model artifacts are invalidated by the v14 model-version change and rebuilt from the available historical inputs.

## Model performance and calibration

Model Lab reads the immutable ForecastLedger:
- multiclass Brier score;
- log loss;
- accuracy;
- probability calibration bins and ECE;
- closing-line value where a comparable closing price was recorded.

These are measured from predictions logged before kickoff and later matched to a confirmed result. The sample size is always shown. No synthetic metrics are inserted when there are no settled predictions.

The forecast ledger now stores an immutable feature snapshot with each new prediction. It includes the competition category, xG flag, starting-XI confirmations, referee presence, injury-feed response, API fixture match and whether entry odds were present. Old rows are not backfilled from later data or a subsequent scan.

AI Learning compares settled forecasts across recorded feature groups. These are descriptive observational comparisons, not causal attribution: groups can differ by league, team strength, season and sample size. A factor is not declared beneficial merely because one group has a better score.

Automatic model-parameter retraining based on recent outcomes remains disabled. Model changes require chronological walk-forward evaluation on future observations and must improve both Brier and log loss before promotion.

## Independent validation by market

The Model Lab separates the engine's existing validation flags from the market-ledger outcome summary.

- **1X2** and **full-time O/U 2.5** retain their existing strict paired bookmaker-benchmark requirements.
- **Other goal lines, BTTS, corners, cards and Asian Handicap** remain forecast-only until their exact-market out-of-sample tests against an attributable de-vig bookmaker benchmark pass.
- The market ledger records selected candidate/recommendation rows; its ROI and Brier are selection-biased diagnostics, not a performance estimate for every forecast.
- A market being calculated by a mathematical model does not make it a validated Value market.

No validation threshold is relaxed in v14.

## Virtual portfolio risk

The risk dashboard defaults to:
- maximum exposure per fixture: 5% of current equity;
- maximum exposure per league: 12%;
- maximum exposure per calendar day: 20%;
- after three consecutive settled losses, a new virtual position cannot exceed the most recent settled stake.

The risk guard is enabled by default and is applied before manual or automatic virtual-position creation. Limits are adjustable in the risk screen and reflected in scanner checks. Existing positions are not deleted or resized when a limit is changed.

Positions on the same match are grouped into a dependence cluster. This is a practical concentration warning, not a statistically estimated correlation coefficient between markets.

The dashboard also reports open exposure and peak-to-trough drawdown from the saved bank history. These calculations are for the virtual/paper portfolio only; no bookmaker account or real stake placement is connected.

## API CENTER

The screen does not reveal key contents. It distinguishes key presence from successful API responses, shows the configured API-Football daily limit/reserve and lineup-call cap, and surfaces recent diagnostics from the scanner.

The standard key-entry workflow remains the sidebar **API ключи** panel. User-entered credentials are saved in a git-ignored local secrets file with restrictive filesystem permissions. On an ephemeral host, a platform restart or redeployment can clear that local file; in that case the key must be entered again in the same sidebar field. A truly durable remote secret vault would require an explicit external storage service and credentials; v14 does not pretend that a local file is a remote vault.

## Known limitations

- Data coverage and lineup publication times vary by competition and provider.
- API-Football source reliability is unknown unless independently calibrated.
- The weather feed is not integrated in this release.
- Cross-fixture statistical correlation is not inferred; same-fixture grouping is a conservative proxy.
- Market-ledger ROI is not an unbiased backtest of the market model.
- Forecast probabilities are not guarantees and should not be interpreted as a promised return.
