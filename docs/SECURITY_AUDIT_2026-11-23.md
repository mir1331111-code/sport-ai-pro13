# Whole-repository audit — 2026-11-23

## Scope and method

The default branch contained 170 tracked files. The review enumerated the full
repository tree, inspected deployment/configuration and CI, and performed a
deeper static review of the Streamlit entry point, secret storage, SQLite and
remote backup/restore, result-provider adapters, forecast ledgers, model engine,
odds validation, market verdicts, stake sizing, and settlement logic.

This is a static code audit plus review of GitHub Actions evidence. It does not
prove the deployed Streamlit instance has the same configuration, that external
API credentials work, or that forecasts are profitable.

## Findings fixed in `main`

### High — corrupt negative stakes could reduce risk exposure

`betting/risk.py` previously accepted finite negative stake values in existing
portfolio rows. Those values reduced pending, same-fixture, and cumulative daily
exposure, potentially creating artificial capacity under the risk limits.

**Status:** fixed by PR #148. Only finite, strictly positive historical stakes
now contribute to risk exposure. Regression tests cover negative, boolean,
NaN, infinite, and same-fixture cases.

### High — unverified or extra-time scores could settle 90-minute markets

The legacy settlement gate accepted a missing score status as full time, and
some direct ESPN/TheSportsDB adapters converted completed extra-time or
shootout results into ordinary FT scores.

**Status:** fixed by PR #149. Settlement now requires explicit verified status
and non-negative integer scores; extra-time results are rejected unless a
provider explicitly identifies the score as regulation time.

## Hardening changes on this audit branch

### High — API keys were supplied as Streamlit widget defaults

The sidebar loaded server-side credentials into the `value=` argument of
password-type `st.text_input` widgets. A password widget masks text visually;
it is not a secure server-side secret container. Supplying a stored secret as
the initial widget value sends that value to the browser.

**Change:** protected API-key widgets now start with an empty value. The
application resolves configured and saved credentials inside the server
process when the field is left blank. Regression tests guard all four protected
widget inputs and the background score-lookup paths.

### Medium — remote encrypted backup decompression was not output-bounded

Manual SQLite restore used bounded gzip decompression, but the encrypted Secret
Gist restore path used unbounded `gzip.decompress`. A valid encrypted payload
could therefore expand far beyond the configured SQLite size limit before the
post-decompression check.

**Change:** validate manifest sizes, enforce per-part and total encrypted-size
limits, and use the same bounded decompressor for remote restore. Regression
tests cover an authenticated oversized gzip stream and an oversized Gist part.

## Deployment blocker still requiring an operator decision

### Critical — no application-level authentication or per-user isolation

The application has no login gate or per-user storage namespace. The local
SQLite database and local portfolio JSON are shared by the running deployment.
Consequently, a publicly reachable instance must be treated as a shared
single-tenant workspace, not as a private multi-user service.

**Required action:** keep the app private or protect it with an identity-aware
access proxy before using real API credentials, personal bank data, or private
portfolio history. A true multi-user product needs authentication plus
per-user database isolation and authorization checks; hiding sidebar values is
not a substitute.

The deployment guide now states this requirement explicitly. No home-grown
authentication was added because a real identity provider, allowed users, and
deployment policy must be selected and configured by the operator.

## Additional engineering risks

- The scanner, source adapters, and visual dashboard are very large modules.
  This increases regression risk and makes provider/model/UI concerns harder to
  isolate. Refactor incrementally behind tested interfaces rather than rewrite.
- Provider schedules, lineups, injuries, and odds have different coverage and
  freshness. Missing context must remain unknown; an empty feed is not proof
  that an injury list or lineup is complete.
- A passing unit-test suite is not evidence of predictive edge or positive
  return. Keep market recommendations disabled unless the exact market passes
  a frozen chronological holdout and same-bookmaker, pre-kickoff comparison.
- Streamlit Community Cloud local disk is not durable. Keep the encrypted
  remote SQLite backup configured and verify recovery periodically.

## CI evidence reviewed

At audit time, the latest default-branch workflow runs were green. The two
pre-existing fixes above also had successful pull-request checks before merge.
This audit branch adds regression tests; its own GitHub Actions run must pass
before merging the additional hardening changes.

## Remaining validation limits

The reviewer did not have access to the deployment's Streamlit Secrets, live
SStats/other provider credentials, private Gist contents, or a production
database. No live fixture/odds scan or historical performance claim was made.
