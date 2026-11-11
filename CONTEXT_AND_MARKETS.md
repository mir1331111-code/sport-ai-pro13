# Match context and additional market models

This branch adds a pre-match context pipeline while keeping the existing Base model and bookmaker Value rules separate.

## SStats context provider

SStats is the scanner's primary fixture, odds and context provider. Games/list supplies the scheduled matches and SStats numeric game IDs. The scanner does not use API-Football or another provider to obtain a duplicate fixture list.

For context enrichment, the scanner uses SStats endpoints documented in the official API:
- Games/injuries?gameId={sstats_game_id} for the match injury report;
- Ls/List to resolve the SStats/Flashscore string event ID for an already-selected fixture (this lookup never adds or replaces fixtures);
- Ls/GameInfo?id={event_id} for match details, including a referee when supplied;
- Ls/GameLineups?id={event_id} for starting lineups when supplied.

Injuries are checked for up to 20 fixtures within 24 hours of kickoff. Lineup/detail calls are limited to 12 fixtures within six hours of kickoff. Responses are cached. A match is marked as having a confirmed starting XI only when both teams have exactly 11 mapped starters. A missing event ID, partial lineup or unknown referee stays unknown; no data is invented or borrowed from another match. SStats coverage may differ by competition.

The numeric Games/list ID and string Ls/List event ID belong to different SStats endpoint families. They are never interchanged. Both are from the same SStats provider and use the same SStats API key.

## TOP MATCHES · Groq Deep Analysis

The scanner creates a separate shortlist of at most seven SStats fixtures after the main forecasts are calculated. A fixture is eligible only when its strongest 1X2 model probability is at least the higher of 65% and the user's minimum-probability setting, both teams have at least five same-venue historical matches, and the fixture has a complete SStats 1X2 quote set. The shortlist is not a betting recommendation list and is not padded when fewer than seven fixtures qualify.

For selected fixtures, the scanner refreshes SStats context and submits a single batch dossier to the configured Groq model. The dossier contains source-confirmed starting XI, injury/suspension feed entries, referee assignment, recent form, venue-specific goal history, available corners/cards history, and referee-card history. It deliberately excludes the primary model's probabilities and pick, bookmaker prices, EV, Edge, and Kelly. The application compares independent picks only after Groq returns and validates its structured response.

Missing source data must remain unknown. In particular, an absent or unconfirmed injury feed does not mean that no players are injured, and an unconfirmed lineup/referee cannot be described as confirmed. Groq analysis cannot change the primary forecast, pass a Value gate, or add a bet to the portfolio. The TOP MATCHES tab includes an on-demand SStats refresh for the selected fixtures so the user can repeat the batch after lineups are published nearer kickoff.

## Context quality and Strong/Value gate

The Base probability forecast always runs and is stored. The Context probability is shown separately. If no approved, walk-forward-validated context blender and all required live features are available, Context remains equal to Base; the app does not invent player strengths or referee effects.

A match can receive Strong/Value only when:
- both teams have a confirmed starting XI;
- a referee is confirmed; and
- any available source-reliability score for those required feeds is at least 0.60.

Missing lineup or referee blocks Strong/Value but does not remove the fixture or Base forecast. Missing injury feed lowers context quality but is not substituted with "no injuries". Unknown source reliability is shown as uncalibrated.

A context blender is fitted only from confirmed snapshots observed strictly before kickoff. The chronological ablation forecasts all fixtures on one date before learning any result from that date. It uses the latest pre-kickoff snapshot per fixture and compares Base and Context on the exact same settled fixture IDs. A model is promoted only with at least 500 paired out-of-time fixtures and improvement in both Brier score and log loss. Training starts only with at least 200 previous confirmed fixtures.

## Historical match CSV for women, national teams and other competitions

Upload the separate "historical match archive" CSV in the scanner. Required columns:

```csv
Date,HomeTeam,AwayTeam,FTHG,FTAG,Div
2025-04-01,Home, Away,2,1,W_E0
2025-04-02,Country A,Country B,0,0,NT_EURO
```

Use a distinct `Div` for each competition, for example `W_E0`, `W_SP1`, `W_F1`, `NT_EURO`, or `NT_WC`. Valid codes contain 1–24 letters, digits, underscores or hyphens. Common date forms are accepted. Missing scores, invalid codes and duplicate fixtures are rejected. Optional columns include `League`, `HTHG`, `HTAG`, `HTHC`, `HTAC`, `HC`, `AC`, `HY`, `AY`, `HR`, `AR`, `Referee`, `HxG`, and `AxG`.

Uploaded competitions train the existing Base engine in their own league namespace. They are not added to the established major-league bookmaker validation sample, so a new competition cannot enter Value merely because it has been imported. The uploaded archive is included in the engine cache fingerprint; changing it causes a model rebuild.

## Historical context CSV

The context CSV is a different file from the score archive. Required columns:

- `fixture_id`, `home_team`, `away_team`
- `kickoff_at`, `observed_at` (ISO-8601 with timezone or `Z`)
- `is_confirmed`, `lineup_is_confirmed`, `referee_is_confirmed`, `injuries_is_confirmed`
- `data_source`, `source_reliability`, plus per-feed `lineup_source_reliability`, `referee_source_reliability`, and `injuries_source_reliability`

The observation timestamp must be strictly earlier than kickoff. Post-kickoff observations are rejected. Repeated fixture IDs are allowed as snapshots; the latest pre-kickoff snapshot is selected for walk-forward evaluation. The historical context fitter requires explicit (or aggregate-backward-compatible) confirmation for the XI and referee and at least one calibrated source-reliability value; every provided reliability value must be at least 0.60. Context features supported by the CSV include `lineup_strength_home`, `lineup_strength_away`, `key_players_missing_home`, `key_players_missing_away`, referee discipline mean/variance, corner score-state means, opponent corner style, and Base probabilities `base_p_home`, `base_p_draw`, `base_p_away`. To train the current 1X2 context blender, lineup strength, missing-player counts, source reliability, Base probabilities and a settled `result` are required.

## Cards and corners

Card totals use observed yellow and red card counts. Historical referee names/IDs are kept as separate entries in SQLite and in the historical match archive. A referee effect is applied only after at least five prior observations for that referee, using a conservative shrinkage toward the team/league estimate. The historical count variance is preserved and the existing count forecast switches to a Negative Binomial distribution when variance exceeds the mean. This is a forecast feature, not a Value approval.

Corner score-state splits are calculated only when the archive includes full-time corners (`HC`, `AC`), halftime score (`HTHG`, `HTAG`) and halftime corners (`HTHC`, `HTAC`). The UI shows second-half corner scenarios for a team trailing, level or leading at halftime. It does not infer halftime corner counts from full-time totals, and these scenario values do not replace the pre-match Base forecast.

## Asian handicaps

Asian handicap outcomes are calculated from the Dixon–Coles score matrix. Integer/half lines include push outcomes. Quarter lines split the stake across adjacent half-goal lines (for example, -0.25 is half -0.5 and half 0.0). The display includes win, half-win, push, half-loss, loss probabilities and expected return.

Handicap prices remain informational. The market is not included in the Value selector or virtual portfolio until it has its own paired, out-of-time benchmark on an identical sample and a coherent exact-line bookmaker market.

## Value protection

Nothing in this feature promotes BTTS, corners, cards or Asian handicaps into the portfolio. Existing 1X2 and full-match O/U 2.5 bookmaker validation thresholds remain in place. A 65% raw probability is not a hit-rate claim and is not sufficient by itself for Value.
