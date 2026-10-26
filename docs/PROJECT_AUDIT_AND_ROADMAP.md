# Project audit and product roadmap

Status date: 2026-10-26

## Product goal

Make the scanner a one-click football research workspace: load fixtures and historical data automatically, score forecasts, display a ranked shortlist with transparent evidence, and optionally add only qualified paper picks to the virtual portfolio. No manual match or odds entry should be part of the normal scan flow.

## Initial architecture audit

### Existing components

- `App.py`: Streamlit application shell, session data, configuration/secrets, portfolio and settlement workflows.
- `ui/theme.py`: global dark/glass theme, gradients, typography, buttons, cards and responsive rules.
- `ui/tabs/scanner.py`: scan orchestration and result presentation; currently mixes provider calls, modeling, ranking, odds handling, and UI controls in a large module.
- `data/sources.py`: fixture and historical data adapters (football-data.co.uk, TheSportsDB, ESPN, football-data.org, Highlightly).
- `data/fixture_merge.py`: fixture reconciliation across feeds.
- `model/engine.py`, `model/historical_match_stats.py`, `model/walk_forward.py`: forecasts and historical validation.
- `betting/verdict.py`, `betting/best_picks.py`, `betting/kelly.py`: pick selection, value checks, and conservative paper staking.
- `storage/`: SQLite/session and local state persistence.

### Findings and risks

1. **Odds are a provider capability, not a UI problem.** The current Highlightly odds endpoint has returned HTTP 401 because the configured plan does not include odds. The app must report provider status honestly and never synthesize bookmaker prices.
2. **Provider concerns are coupled to UI.** Scanner orchestration is large; provider adapters and market normalization should be isolated behind a common interface.
3. **Forecast probability is not automatically calibrated probability.** UI must distinguish model estimate, bookmaker-implied probability, edge/EV, and measured out-of-sample performance.
4. **Market coverage differs.** A valid 1X2 price must not be reused for totals, BTTS, corners, or cards. Each market needs its own quote and validation.
5. **No-pick results are valid.** If no forecast meets data-quality and value gates, show an explicit empty state rather than lowering thresholds to fill the page.
6. **Public repository hygiene matters.** Keep tokens in deployment secrets/environment variables, never in source, logs, screenshots, or example data.
7. **The CSV template previously contained illustrative bookmaker prices.** This was removed so a sample cannot be mistaken for current market data.
8. **Visual style exists, but usability matters more than effects.** Preserve the dark sports-analytics look while prioritizing legible hierarchy, clear source timestamps, mobile layout, accessible contrast, and low-noise states.

## Target architecture

- `data/providers/base.py`: common fixture/odds provider contracts and normalized records.
- `data/providers/fixtures_*.py`: one adapter per authorized fixture feed.
- `data/providers/odds_*.py`: licensed odds provider adapters, with explicit league/market coverage and quota reporting.
- `data/normalization.py`: canonical team names, UTC kickoff, league IDs, bookmaker, market and outcome IDs.
- `services/scan_service.py`: orchestration only; fetch, merge, model, price-match, rank, and return a structured scan report.
- `services/pick_policy.py`: deterministic eligibility gates, no duplicate paper picks, stake caps, and reason codes.
- `ui/tabs/scanner.py`: presentation layer; should not contain provider-specific HTTP logic.
- `tests/`: unit tests for normalization, market-price matching, provider failures, ranking, and portfolio idempotency.

Avoid a large rewrite in one commit. Extract and test one seam at a time so existing portfolio and settlement data remain compatible.

## One-click scan acceptance criteria

1. One click loads upcoming fixtures for the selected horizon; no manual file or odds entry is needed in normal operation.
2. Provider calls have timeouts, bounded retries, caching, and per-provider error isolation.
3. Every displayed price includes provider/bookmaker and retrieval time where available.
4. Quotes match the exact fixture, market, outcome, and line; stale or ambiguous quotes are rejected.
5. Each pick shows estimated probability, market-implied probability, edge, expected value, model/data confidence, and a concise reason.
6. The scanner shows separate counts for fixtures found, forecasts created, real prices matched, candidates passing filters, and paper picks added.
7. If the odds API is unavailable, the scan still shows fixtures and forecasts, but disables value-qualified betting actions and explains why.
8. Auto-add is idempotent and paper-only; repeated scans cannot duplicate the same match/market/outcome.
9. No pick is described as guaranteed. Historical performance is shown only when computed from a chronological holdout and sufficient sample.
10. Tests and CI pass; Streamlit deployment smoke test confirms the app starts and a scan produces a meaningful status/result state.

## Visual design direction

- Keep a polished dark control-room identity with restrained cyan/violet accents; reduce decorative animation where it competes with information.
- Header: product name, last scan time, source-health indicator, and one dominant scan button.
- KPI row: fixtures, priced matches, qualified picks, paper picks added.
- Main content: ranked pick cards with match/league/kickoff, market, probability, real odds/source timestamp, edge/EV, risk badge, and expandable model explanation.
- Add clear skeleton/loading progress and distinct empty, stale-data, partial-provider, and error states.
- Keep filters concise: time horizon, leagues, market, minimum probability/value; put advanced controls in an expander.
- Use responsive columns and accessible contrast; avoid forcing a fixed-width sidebar on small screens.
- Diagnostics belong in a secondary expander and should identify the failing provider and next action without exposing secrets.

## Implementation sequence

### Phase 1 — Reliability and UX foundation
- Keep the existing provider sources and forecast engine.
- Make empty/partial results explicit, expose source health and scan timestamps.
- Keep manual import available as an optional fallback, not the default workflow.
- Add regression tests around odds validation and duplicate paper picks.

### Phase 2 — Provider abstraction and automatic odds
- Select an authorized odds vendor based on required leagues, market depth, price freshness, quotas, and price.
- Add configuration via Streamlit Secrets/environment variables; no key entry on every scan.
- Implement provider adapter, league mapping, market normalization, caching and quota/error handling.
- Test no-key, invalid-key, quota-exhausted, empty-coverage and successful-response cases.

### Phase 3 — Forecast quality
- Run leakage-safe chronological validation and compare forecasts with de-vigged market baselines.
- Calibrate probabilities only on historical predictions that would have been available at forecast time.
- Enable automatic paper picks only for markets that pass minimum sample and quality gates.

### Phase 4 — Product polish and observability
- Extract scan orchestration from Streamlit UI.
- Add testable scan report schema, health diagnostics, structured logging without secrets, and deployment smoke tests.
- Refine mobile visual layout, loading states, accessible contrast, and pick explanations.

## Decisions intentionally deferred

- Odds vendor, subscription tier, and supported league list.
- Whether to prioritize broad league coverage or deeper coverage of fewer leagues.
- Final thresholds for probability, edge, EV, and stake after out-of-sample evaluation.

These decisions should be based on real provider coverage and measured model results, not fabricated odds or aesthetic preferences.
