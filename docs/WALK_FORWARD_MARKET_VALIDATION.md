# Chronological market validation: warm-up and holdout

## Purpose

The scanner's market benchmark must not start from a short recent slice with empty team state. Production training uses a multi-season history, so validation now replays the full frozen historical prefix before scoring a recent per-league holdout.

## Evaluation protocol

1. Build a frozen historical snapshot and exclude the same recent 48-hour safety window used by the scanner.
2. Select the newest 250 valid fixtures per priority league as the scored holdout.
3. Replay the complete available history for those leagues in chronological order.
4. For each league, the first date in its selected holdout is its evaluation cutoff. Earlier rows warm the model but are excluded from reported metrics.
5. Forecast every fixture on a calendar date before ingesting any result from that date.
6. Report proper scores, calibration, and paired market differences only on the scored holdout.
7. Bootstrap paired comparisons by calendar matchday; promotion gates continue to require sufficient fixtures, independent matchdays, and confidence intervals below zero.

The market-validation cache key now uses the full replay input and a new version namespace, so reports created under the previous short-slice protocol are not reused.

## Experimental model/market blends

The walk-forward report now includes fixed model weights of 25%, 50%, and 75%, with the remainder assigned to the de-vigged bookmaker probability. Each blend is compared with both the model and market on the same fixtures and matchday clusters.

These are research diagnostics only. They do not alter production probabilities, Value calculations, or betting permissions. A weight should only be considered for production after it wins on an untouched later holdout, remains calibrated by league, and is stable across seasons and market sources.

## Recent-form research arm

The walk-forward now also evaluates an isolated recent-form residual arm. It uses the same chronological Engine state and feature set as the candidate model, with a small fixed research weight (0.10) applied to the difference between the teams' last-five points-per-match form. The base Engine default remains 0.0, so this experiment does not alter production forecasts. The report compares the form arm with the base model on identical fixtures and matchday clusters. Promotion requires a later untouched holdout to improve both Brier and Log Loss and show stability across leagues and seasons.

## Explicitly not promoted

- Recent form is not yet a validated probability feature. Its research arm is diagnostic only; the production weight remains zero.
- Context/lineup probabilities remain Base-equivalent unless the existing context model has passed its paired chronological promotion checks using confirmed pre-kickoff snapshots.
- The production model remains unchanged by this benchmark patch.
- Historical closing odds are stored by the forecast ledger when supplied at settlement; this patch does not fabricate or backfill missing closing prices. CLV reporting is meaningful only where the closing quote, bookmaker, and timestamp are genuinely recorded.

## Operational boundary

This is a historical validation correction and an experiment report, not a claim that the model beats the market or that betting ROI has improved. The scanner's existing fail-closed market and Value gates remain in force. A live two-consecutive-scan verification is still required before release.
