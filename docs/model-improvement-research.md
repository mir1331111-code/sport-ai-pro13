# Research notes: football prediction model improvements

Date reviewed: 2026-10-31

This review compares public repositories for methodology and reproducible evaluation. It does not establish that any model will be profitable on live betting markets. Repository-reported results are self-reported and may use different leagues, time windows, and data sources.

## Five useful open-source references

1. **[MatchBench](https://github.com/Dr-yato/MatchBench)** — transparent benchmark of Pi-ratings, Dixon–Coles, gradient-boosted trees, calibrated stacking, and chronological backtesting. Useful lesson: compare models on the same chronological fixtures and retain proper scoring rules, not just accuracy.
2. **[yukunai/football-predictor](https://github.com/yukunai/football-predictor)** — Dixon–Coles score model with Elo prior correction, calibration, data-quality checks, prediction audits, and rolling-origin backtests. Useful lesson: monitor source freshness and league-specific reliability, and abstain when data is thin or stale.
3. **[Luis-ntonio/Match_Prediction](https://github.com/Luis-ntonio/Match_Prediction)** — Elo, rolling form, rest days, Poisson ratings, XGBoost, probability calibration, seed ensembles, and walk-forward evaluation. Its README reports that calibration/ensembling improved log loss slightly while accuracy stayed flat, a useful reminder that probability quality can improve without higher hit rate.
4. **[kenRedco/football-predictor](https://github.com/kenRedco/football-predictor)** — Poisson/XGBoost ensemble for 1X2, O/U 2.5 and BTTS, evaluated with walk-forward splits and proper metrics. Useful lesson: score each market independently and do not infer BTTS or total performance from 1X2 accuracy.
5. **[callumfm/dixon-coles](https://github.com/callumfm/dixon-coles)** — time-decayed Dixon–Coles, shrinkage for teams with little history, xG blending, and chronological tuning. Useful lesson: recent matches and xG can help, but every weight must be validated on later fixtures.

## Changes adapted to NEURO BET PRO

- Added betting/market_math.py with proportional de-vigging for a **complete, coherent** set of decimal prices. Missing or invalid prices fail closed rather than being guessed.
- Updated 1X2 value screening to blend the model probability conservatively toward the complete line's no-vig market probability when all three prices exist. Single-price/manual quotes keep the previous conservative fallback.
- Applied the same paired-price de-vigging to the virtual portfolio's O/U 2.5 and BTTS Yes/No calculations when both sides are available.
- Added offline unit tests for normalization, invalid/partial lines, probability blending, and value-screen integration.
- Preserved the existing chronological walk-forward framework, Dixon–Coles engine, temporal weighting, xG ablation, calibration, and market benchmark rather than replacing them with a second incompatible model.

## Deliberately not copied or enabled

- No third-party repository was copied wholesale. Methods were adapted to this project's existing interfaces.
- No extra heavyweight ML dependency or remote data scraper was added just because a reference project uses it.
- No claimed accuracy uplift or profit claim is made without running a matched historical evaluation on this application's data.
- Live odds from different bookmakers must not be mixed into one synthetic line for de-vigging. The current normalization is only valid when the supplied prices are a coherent market line.
- A model-market blend weight of 0.40 is a conservative existing policy, not a fitted optimum. It should be tuned only on chronological out-of-fold predictions and then frozen before the final holdout.

## Verification checklist

1. Run python -m compileall -q .
2. Run python -m unittest discover -s tests -v.
3. Compare the existing engine and candidate configuration on identical walk-forward fixtures using log loss, Brier score, RPS, and confidence intervals.
4. Check 1X2, O/U 2.5 and BTTS separately. The current BTTS path is prospective virtual/paper-trading only: a dedicated historical BTTS calibration and model-vs-market validation gate has not yet been implemented, so it is experimental and must not be treated as validated betting advice.
5. Verify the odds-source metadata and actual API coverage in the deployed app. GitHub CI cannot validate provider credentials or live feed coverage.

## References

- MatchBench: https://github.com/Dr-yato/MatchBench
- yukunai/football-predictor: https://github.com/yukunai/football-predictor
- Match_Prediction: https://github.com/Luis-ntonio/Match_Prediction
- kenRedco/football-predictor: https://github.com/kenRedco/football-predictor
- callumfm/dixon-coles: https://github.com/callumfm/dixon-coles
