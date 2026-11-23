# Incremental Chronological Engine Training

The scanner must not replay the complete historical archive on every ordinary scan. The fitted Engine is persisted with a content-checked chronological training checkpoint.

## Normal scan lifecycle

1. Load the current source-history snapshot and calculate its full cache fingerprint.
2. If an exact fitted Engine exists for that exact snapshot/configuration, reuse it.
3. Otherwise load the last-good incremental checkpoint for the same model configuration.
4. Prepare all completed fixtures and calculate one SHA-256 digest per calendar matchday.
5. If all previously trained matchday digests are an exact prefix of the new history, clone the saved Engine and train only the new whole-day suffix.
6. If there are no new completed matchdays, reuse the saved state without training.
7. Save both the full-snapshot cache and the new incremental checkpoint atomically.

The checkpoint contains the fitted Engine (including score-model and historical market state), training schema/configuration, trained-row count, last trained date, and ordered matchday digests. The ordinary scanner continues to produce forecasts from the resulting Engine.

The matchday digest covers the fields consumed by Engine, HistoricalMatchStats and the
dated aggregate-shot xG fitter: fixture/date/league/score, all supported provider-xG aliases
and provenance, shot counts used to fit proxy xG, corners/cards and referee fields. It
intentionally excludes bookmaker odds and presentation-only metadata. A price-only update
therefore refreshes market validation on its separate cadence without retraining the score
model; a change to score, xG, shots or market-stat training inputs invalidates the history prefix.

## Safety conditions for incremental continuation

Incremental continuation is allowed only when all previously trained matchday hashes match the current history prefix exactly and the checkpoint Engine's trained-row count agrees with its manifest.

A complete replay is required when any of the following happens:

- first training / no checkpoint exists;
- a historical score, fixture, team, league, market statistic, provider xG or proxy xG changes;
- a completed fixture is inserted into a date that has already been trained;
- an earlier result is removed or newly becomes complete;
- the checkpoint schema, model configuration, application/data version or shot-xG version changes.

This is deliberate. A partially changed matchday is never appended onto a state that has already ingested some results from that day. The day is replayed as a whole so every fixture on it is forecast before any result from that day is learned. The previously published checkpoint is cloned before updates; a failed incremental run cannot corrupt the last-good state.

The Engine uses timestamp-based temporal weighting, and the chronological training API retains the global processed-row offset. This allows appended later matchdays to continue the same online state without restarting the historical counters. Corrections and backfills are not treated as append-only.

## Market walk-forward cadence

Historical market walk-forward is not run on every incremental scan. It has its own cache key, separate from the training-data fingerprint, and by default refreshes no more than once every 7 days. Configure the interval with:

~~~bash
MODEL_MARKET_VALIDATION_REFRESH_DAYS=7
~~~

A first build or detected correction to already-trained history forces a fresh walk-forward rather than reusing a stale report. The scan still fails closed for any market without a valid benchmark report. The independent Fixed Holdout Model Lab remains a separately invoked audit.

## Cache placement and recovery

The full fitted Engine remains content-addressed as before. A second stable checkpoint file is stored under DISK_CACHE_DIR with an incremental-v1 key. The process-local shared cache supports multiple Streamlit sessions; the disk checkpoint supports process restarts when the deployment's disk persists.

If the disk is ephemeral or a checkpoint is missing/corrupt, the scanner performs one full chronological replay, writes a new checkpoint, and resumes incremental updates thereafter. No forecast, result or feature values are synthesized to recover a checkpoint.

## Operational expectations

- Unchanged data: model load/inference only, no historical training.
- New completed matchdays: learn only the new fixtures.
- Same-day late result: full replay for that date and all later history.
- Historical correction or xG backfill: full chronological replay.
- Model/schema revision: one full replay to establish a new checkpoint.