# Deploy NEURO BET PRO

The app entry point is `App.py`. It is a Streamlit application and the repository includes its runtime dependencies in `requirements.txt`.

## Streamlit Community Cloud
## Match and odds source priority

When the SStats key is configured and `Games/list` returns fixtures, SStats is the only source for the scanner's match list and bookmaker odds. ESPN, TheSportsDB and football-data.org are not queried for that scan; they are used only as schedule fallbacks when SStats is not configured or returns no fixtures. A fallback provider can never supply prices: all price-like fields are removed unless the row carries an SStats game ID. Automatic CSV/JSON odds imports and Highlightly are not used. SStats fixture IDs are retained for `Odds/{gameId}` and other SStats match endpoints.

SStats also supplies optional match context through its own Games/injuries, Ls/GameInfo and Ls/GameLineups endpoints. Ls/List only resolves the SStats/Flashscore event ID for a fixture already selected by Games/list; it never adds or replaces scanner fixtures.

1. Open [share.streamlit.io](https://share.streamlit.io/) and switch to the GitHub workspace for mir1331111-code.
2. Select **Create app** (or manage the existing app connected to this repository).
3. Choose repository mir1331111-code/sport-ai-pro13, branch main, and main file path App.py.
4. Open the app. In the left sidebar, expand **🔑 API ключи** and paste your SStats.net API key. Use **🔎 Проверить подключение SStats.net** to verify the key, fixture list and odds. The same key is used for the SStats match, odds and supported context endpoints; a separate API-Football key is not required.

   UI-entered credentials are saved in the private, git-ignored .neuro_secrets.json file with owner-only permissions and are kept outside portfolio data and downloadable backups. This is local app-host storage, not durable cloud secret storage: Streamlit Community Cloud does not guarantee local-file persistence across reboot/redeploy. If that host clears its filesystem, re-enter the SStats key in the same sidebar field. A durable guarantee requires an external persistent secret store; the app does not copy API keys to GitHub, Gist, or backups.

5. Run the scanner. It uses the SStats numeric game ID for Games/injuries and resolves the Flashscore string event ID for Ls/GameInfo and Ls/GameLineups. The Ls/List request uses Upcoming/Limit filters, matches by ordered teams and exact source/app-local/UTC date candidates, and never sends numeric SStats IDs to the Flashscore detail endpoints.

Streamlit Community Cloud reads requirements.txt and installs the Python dependencies. Once deployed, pushes to the connected GitHub branch are reflected in the app; dependency changes trigger a full redeploy. On hosts with ephemeral filesystems, a provider or host restart may clear the local key file; in that case paste the SStats key into the same sidebar field again.

## SStats context request limits

The scanner checks injuries for up to 20 upcoming SStats fixtures within 24 hours and checks lineups for up to 12 fixtures within six hours of kickoff. Responses are cached. Provider coverage varies by competition. An empty injuries response is only a successful feed response for that fixture; it is not proof that every player is available. A starting XI is confirmed only when the response can be mapped to exactly 11 players for both teams. Missing or ambiguous data stays unknown and Strong/Value remains blocked.

## Other optional data-source secrets

Existing app integrations can be configured in Streamlit Secrets as needed:

```toml
FOOTBALLAPI_KEY = "..."
FDORG_TOKEN = "..."
SSTATS_API_KEY = "..."
HIGHLIGHTLY_API_KEY = "..."
```

These are optional for launching the application; the specific features they supply will remain unavailable until configured.

## Local run

From the repository root:

```bash
python -m pip install -r requirements.txt
streamlit run App.py
```

## First-run checklist

1. Confirm the app opens and the scanner can load fixtures.
2. Paste the SStats key into the left sidebar and run **🔎 Проверить подключение SStats.net**.
3. Run a scan. The log reports SStats event-ID matches, injury checks, confirmed XIs and confirmed referees. Markets without verified historical coverage remain forecast-only.
4. In the scanner, optionally upload the historical match CSV for women's competitions/national teams and the separate pre-kickoff context CSV. Their required columns and leakage rules are documented in [CONTEXT_AND_MARKETS.md](CONTEXT_AND_MARKETS.md).
5. Run the walk-forward Base vs Context ablation. Context is promoted only when at least 500 paired out-of-time matches show improvement in both Brier score and Log Loss.

A successful deployment does not mean the new context model has already demonstrated a higher hit rate or positive return. Until real historical context data passes the stated tests, Context probabilities remain equal to Base; corners, cards and Asian handicaps remain informational and are not automatically staked.
