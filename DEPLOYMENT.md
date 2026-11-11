# Deploy NEURO BET PRO

The app entry point is `App.py`. It is a Streamlit application and the repository includes its runtime dependencies in `requirements.txt`.

## Streamlit Community Cloud

1. Open [share.streamlit.io](https://share.streamlit.io/) and switch to the GitHub workspace for `mir1331111-code`.
2. Select **Create app** (or manage the existing app connected to this repository).
3. Choose repository `mir1331111-code/sport-ai-pro13`, branch `main`, and main file path `App.py`.
4. In **Advanced settings → Secrets**, add the API key securely. Do not commit secrets to GitHub:

```toml
API_FOOTBALL_KEY = "YOUR_API_FOOTBALL_KEY"
```

5. Save and deploy. For an existing app, add the secret under **Manage app → Settings → Secrets**, save, and reboot the app.

Streamlit Community Cloud reads `requirements.txt` and installs the Python dependencies. Once deployed, pushes to the connected GitHub branch are reflected in the app; dependency changes trigger a full redeploy.

## API-Football limits

By default the app reserves 10 requests and allows up to 100 API requests per UTC day (so the scanner spends at most 90). The lineup-call cap defaults to 12 per scan and lineups are requested only inside the six-hour pre-kickoff window.

For an API-Football plan with a different documented quota, set optional Streamlit Secrets:

```toml
API_FOOTBALL_DAILY_LIMIT = 100
API_FOOTBALL_DAILY_RESERVE = 10
API_FOOTBALL_MAX_LINEUP_CALLS = 12
```

Use the quota values allowed by your provider subscription. The application enforces the configured local limit across concurrent sessions and keeps a reserve. API responses are cached on disk. Provider coverage for lineups and injuries varies by competition; an empty injuries response is not treated as proof that every player is available.

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
2. Configure `API_FOOTBALL_KEY` in Streamlit Secrets or the scanner password field.
3. Run a scan. The log reports fixtures matched, injury batches, lineup calls and quota usage.
4. In the scanner, optionally upload the historical match CSV for women's competitions/national teams and the separate pre-kickoff context CSV. Their required columns and leakage rules are documented in [CONTEXT_AND_MARKETS.md](CONTEXT_AND_MARKETS.md).
5. Run the walk-forward Base vs Context ablation. Context is promoted only when at least 500 paired out-of-time matches show improvement in both Brier score and Log Loss.

A successful deployment does not mean the new context model has already demonstrated a higher hit rate or positive return. Until real historical context data passes the stated tests, Context probabilities remain equal to Base; corners, cards and Asian handicaps remain informational and are not automatically staked.
