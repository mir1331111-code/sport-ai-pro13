# Analytics terminal (v14 rollout)

The navigation groups the product into scanner, forecast statistics, value/portfolio,
data quality, learning audit, risk control, model lab, and API/tools.

## Data quality semantics

The DATA QUALITY screen evaluates fields actually stored on each scanner card.
It distinguishes confirmed evidence, an answered source with no records,
unavailable/unconfigured providers, stale data, and conflicts when those
states are available from the source adapters. Unknown values are never
presented as confirmed. Its completeness score is a percentage of tracked
fields with evidence; it is not a win probability or a model confidence score.

## Learning and model audit

AI LEARNING CENTER reads only settled rows from the append-only forecast
ledgers. MODEL LAB displays Brier score, log loss, reliability bins, xG
coverage, and market-specific ledger diagnostics. Metrics are descriptive;
they do not retrain a model or rewrite parameters.

A market is not Value-approved merely because a probability can be calculated.
1X2 and any already independently validated market keep their existing gates.
BTTS, corners, cards, and Asian Handicap remain forecast-only until they have
their own sufficient out-of-sample evidence and verified settlement.

## Portfolio risk

Risk control reports current open exposure, concentration by fixture/league/day,
and drawdown reconstructed from settled positions. It does not infer numerical
correlations without a joint historical dataset. No bookmaker execution or
automatic real-money staking is introduced.

## Known limitations

- Data quality reflects fields available in the scanner card; it cannot prove
  that an external provider has exhaustive coverage for every competition.
- Injury feed returning an empty list is labelled as an answered feed with no
  records only when the adapter confirms the response. It is not interpreted
  as proof that no player is injured.
- The risk screen is a monitoring aid, not a portfolio optimizer.
- Historical calibration metrics are meaningful only with a representative
  settled sample and should be interpreted with sample size.
