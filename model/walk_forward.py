"""Leakage-safe chronological model and xG ablation evaluation."""
from __future__ import annotations

import math
import random
import numpy as np
from datetime import date, datetime

from model.engine import Engine, _row_xg_pair


_XG_KEYS = {
    "hxg", "axg", "home_xg", "away_xg", "homexg", "awayxg",
    "xg_home", "xg_away", "xg", "xga", "home_expected_goals",
    "away_expected_goals",
}
_XG_METADATA_KEYS = {
    "xg_source", "xg_kind", "xg_model_version", "xg_training_sides",
}


def _get(row, *names):
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None

def _strict_score(value):
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        return None
    return int(number)



def _parse_date(value):
    """Parse ISO timestamps and Football-Data day/month/year CSV dates."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        raw = str(value).strip().replace("Z", "+00:00")
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw).date()
        except ValueError:
            pass
        try:
            return date.fromisoformat(raw[:10])
        except ValueError:
            pass
        # football-data.co.uk exports dates as DD/MM/YY or DD/MM/YYYY.
        for fmt in ("%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y", "%d-%m-%y"):
            try:
                return datetime.strptime(raw[:10], fmt).date()
            except ValueError:
                continue
    except (TypeError, ValueError):
        return None
    return None


def _xg_origin(row):
    """Classify historical xG provenance.

    Explicit modelled-proxy metadata wins. Unlabelled legacy xG pairs retain
    the Engine's existing treatment as provider xG for compatibility.
    """
    if not isinstance(row, dict) or _row_xg_pair(row)[0] is None:
        return None
    source = str(row.get("xg_source") or "").casefold()
    kind = str(row.get("xg_kind") or "").casefold()
    if kind == "modelled_proxy" or "shot_model" in source or "shot_aggregate" in source:
        return "proxy"
    return "provider"


def _without_xg(row):
    """Copy a historical row without any xG values or provenance."""
    if not isinstance(row, dict):
        return {}
    return {
        key: value for key, value in row.items()
        if str(key).strip().casefold() not in _XG_KEYS | _XG_METADATA_KEYS
    }


def _without_provider_xg(row):
    """Retain only modelled-proxy xG; remove provider and unknown pairs."""
    if not isinstance(row, dict):
        return {}
    return _without_xg(row) if _xg_origin(row) == "provider" else dict(row)


def _without_proxy_xg(row):
    """Retain provider xG and remove aggregate-shot modelled proxies."""
    if not isinstance(row, dict):
        return {}
    return _without_xg(row) if _xg_origin(row) == "proxy" else dict(row)


class _ClusteredRecords(list):
    """Append forecast records with the current calendar-day cluster attached."""

    def __init__(self):
        super().__init__()
        self.cluster = None

    def set_cluster(self, cluster):
        self.cluster = str(cluster) if cluster is not None else None

    def append(self, record):
        if isinstance(record, tuple) and len(record) == 2:
            record = (record[0], record[1], self.cluster)
        super().append(record)


def _record_group(record, index):
    if isinstance(record, (tuple, list)) and len(record) >= 3 and record[2] is not None:
        return record[2]
    return ("fixture", int(index))


def _cluster_bootstrap_mean_samples(values, groups, bootstrap_samples, seed):
    """Bootstrap whole matchdays, preserving every fixture within a sampled day."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return [[] for _ in range(matrix.shape[1] if matrix.ndim == 2 else 0)]

    sample_count = max(0, int(bootstrap_samples))
    metric_count = matrix.shape[1]
    samples = [[] for _ in range(metric_count)]
    if sample_count == 0:
        return samples
    if len(groups) != matrix.shape[0]:
        raise ValueError("group labels must align with bootstrap rows")

    cluster_indices = {}
    for index, group in enumerate(groups):
        cluster_indices.setdefault(group, []).append(index)
    clusters = list(cluster_indices.values())
    if not clusters:
        return samples

    rng = np.random.default_rng(int(seed))
    cluster_count = len(clusters)
    for _ in range(sample_count):
        selected = rng.integers(0, cluster_count, size=cluster_count)
        row_indices = np.concatenate([clusters[int(i)] for i in selected])
        means = matrix[row_indices].mean(axis=0)
        for metric_index in range(metric_count):
            samples[metric_index].append(float(means[metric_index]))
    return samples


def _intervals_from_samples(samples, names):
    intervals = {}
    for index, name in enumerate(names):
        values = sorted(samples[index]) if index < len(samples) else []
        intervals[name] = (
            {"low": values[int(.025 * (len(values) - 1))],
             "high": values[int(.975 * (len(values) - 1))]}
            if values else {"low": None, "high": None}
        )
    return intervals


def _record_scores(record):
    probabilities, outcome = record[:2]
    classes = ("H", "D", "A")
    brier = sum((probabilities[k] - float(k == outcome)) ** 2 for k in classes)
    loss = -math.log(max(probabilities[outcome], 1e-15))
    # 1X2 has no universally accepted ordinal order; label this convention explicitly.
    cumulative_p = cumulative_y = rps_sum = 0.0
    for key in classes[:-1]:
        cumulative_p += probabilities[key]
        cumulative_y += float(key == outcome)
        rps_sum += (cumulative_p - cumulative_y) ** 2
    return brier, loss, rps_sum / 2.0, float(max(classes, key=lambda k: probabilities[k]) == outcome)


def _metrics(records, *, bootstrap_samples=1000, seed=1729):
    if not records:
        return {
            "n": 0, "brier": None, "log_loss": None, "rps_hda_order": None,
            "accuracy": None, "confidence_intervals_95": {},
            "bootstrap_unit": "matchday", "cluster_count": 0,
        }
    scores = np.asarray([_record_scores(record) for record in records], dtype=float)
    groups = [_record_group(record, index) for index, record in enumerate(records)]
    explicit_groups = any(
        isinstance(record, (tuple, list)) and len(record) >= 3 and record[2] is not None
        for record in records
    )
    names = ("brier", "log_loss", "rps_hda_order", "accuracy")
    means = scores.mean(axis=0).tolist()
    samples = _cluster_bootstrap_mean_samples(scores, groups, bootstrap_samples, seed)
    return {
        "n": len(records),
        **dict(zip(names, means)),
        "confidence_intervals_95": _intervals_from_samples(samples, names),
        "bootstrap_unit": "matchday" if explicit_groups else "fixture_fallback",
        "cluster_count": len(set(groups)),
    }


def _market_probs(row, method="proportional"):
    """De-vig complete 1X2 odds with proportional, power or Shin method."""
    candidates = (
        ("B365H", "B365D", "B365A"), ("PSH", "PSD", "PSA"),
        ("WHH", "WHD", "WHA"), ("AvgH", "AvgD", "AvgA"),
        ("BWH", "BWD", "BWA"),
    )
    for keys in candidates:
        try:
            odds = [float(row.get(key)) for key in keys]
            if not all(math.isfinite(x) and x > 1.0 for x in odds):
                continue
            implied = [1.0 / x for x in odds]
            total = sum(implied)
            if method == "proportional":
                probs = [x / total for x in implied]
            elif method == "power":
                lo, hi = 0.01, 5.0
                for _ in range(80):
                    mid = (lo + hi) / 2
                    if sum(x ** mid for x in implied) > 1.0:
                        lo = mid
                    else:
                        hi = mid
                raw = [x ** ((lo + hi) / 2) for x in implied]
                probs = [x / sum(raw) for x in raw]
            elif method == "shin":
                def shin(z):
                    return [(math.sqrt(z*z + 4*(1-z)*q*q/total)-z)/(2*(1-z)) for q in implied]
                lo, hi = 0.0, min(.999, max(0.0, (total-1.0)/max(total, 1e-9)))
                for _ in range(80):
                    mid = (lo + hi) / 2
                    if sum(shin(mid)) > 1.0:
                        lo = mid
                    else:
                        hi = mid
                probs = shin((lo + hi) / 2)
                norm = sum(probs)
                probs = [p / norm for p in probs] if norm > 0 else [p / total for p in implied]
            else:
                raise ValueError("method must be proportional, power, or shin")
            return dict(zip(("H", "D", "A"), probs)), ".".join(keys)
        except (TypeError, ValueError, OverflowError, ZeroDivisionError):
            continue
    return None, None


def _paired_differences(left_records, right_records, *, bootstrap_samples=1000, seed=2718):
    """Paired score differences with matchday-clustered uncertainty."""
    if not left_records or len(left_records) != len(right_records):
        return {"n": 0, "differences": {}, "confidence_intervals_95": {}}
    left = np.asarray([_record_scores(r) for r in left_records], dtype=float)
    right = np.asarray([_record_scores(r) for r in right_records], dtype=float)
    groups = [_record_group(record, index) for index, record in enumerate(left_records)]
    right_groups = [_record_group(record, index) for index, record in enumerate(right_records)]
    if groups != right_groups:
        return {
            "n": 0, "differences": {}, "confidence_intervals_95": {},
            "error": "paired records do not share identical matchday clusters",
        }
    diffs = left - right
    names = ("brier", "log_loss", "rps_hda_order", "accuracy")
    means = diffs.mean(axis=0).tolist()
    samples = _cluster_bootstrap_mean_samples(diffs, groups, bootstrap_samples, seed)
    return {
        "n": len(diffs),
        "differences": dict(zip(names, means)),
        "confidence_intervals_95": _intervals_from_samples(samples, names),
        "bootstrap_unit": "matchday",
        "cluster_count": len(set(groups)),
        "note": "Negative values favor left model for proper scoring rules; positive accuracy difference favors left. Confidence intervals resample paired calendar-day clusters.",
    }


def _probs(prediction):
    return {"H": prediction["p1"], "D": prediction["x"], "A": prediction["p2"]}


def _poisson_over_probability(mean, line):
    """P(total goals > half-goal line) from a Poisson total-goals approximation."""
    try:
        mean, line = float(mean), float(line)
        if not math.isfinite(mean) or mean <= 0 or not math.isfinite(line):
            return None
        cutoff = math.floor(line) + 1
        term = math.exp(-mean)
        cdf = term
        for k in range(1, cutoff):
            term *= mean / k
            cdf += term
        return min(1.0, max(0.0, 1.0 - cdf))
    except (TypeError, ValueError, OverflowError):
        return None


def _bootstrap_mean_samples(values, bootstrap_samples, seed):
    """Fast deterministic paired row bootstrap in bounded-memory batches.

    Each row is one fixture and each column is one metric. Resampling the row
    indices together preserves pairing between metrics and avoids the large
    Python-level loops used by the former bootstrap implementation.
    """
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return [[] for _ in range(matrix.shape[1] if matrix.ndim == 2 else 0)]

    sample_count = max(0, int(bootstrap_samples))
    metric_count = matrix.shape[1]
    samples = [[] for _ in range(metric_count)]
    if sample_count == 0:
        return samples

    row_count = matrix.shape[0]
    rng = np.random.default_rng(int(seed))
    batch_size = min(128, sample_count)
    for offset in range(0, sample_count, batch_size):
        batch_count = min(batch_size, sample_count - offset)
        indices = rng.integers(
            0, row_count, size=(batch_count, row_count), dtype=np.int32
        )
        means = matrix[indices].mean(axis=1)
        for metric_index in range(metric_count):
            samples[metric_index].extend(means[:, metric_index].tolist())
    return samples


def _binary_metrics(records, *, bootstrap_samples=1000, seed=31415):
    """Binary Brier/log-loss with matchday-clustered confidence intervals."""
    if not records:
        return {
            "n": 0, "brier": None, "log_loss": None,
            "confidence_intervals_95": {}, "bootstrap_unit": "matchday",
            "cluster_count": 0,
        }
    scores = []
    groups = []
    explicit_groups = False
    for index, record in enumerate(records):
        probabilities, outcome = record[:2]
        p = max(1e-15, min(1.0 - 1e-15, float(probabilities["over"])))
        y = float(outcome == "over")
        scores.append(((p-y)**2, -(y * math.log(p) + (1-y) * math.log(1-p))))
        groups.append(_record_group(record, index))
        explicit_groups = explicit_groups or (
            isinstance(record, (tuple, list)) and len(record) >= 3 and record[2] is not None
        )
    matrix = np.asarray(scores, dtype=float)
    names = ("brier", "log_loss")
    means = matrix.mean(axis=0).tolist()
    samples = _cluster_bootstrap_mean_samples(matrix, groups, bootstrap_samples, seed)
    return {
        "n": len(scores), **dict(zip(names, means)),
        "confidence_intervals_95": _intervals_from_samples(samples, names),
        "bootstrap_unit": "matchday" if explicit_groups else "fixture_fallback",
        "cluster_count": len(set(groups)),
    }


def _total_market_probs(row):
    """Return no-vig over/under 2.5 probabilities and source from historical odds."""
    for over_key, under_key in (
        # Football-Data CSVs commonly use BetBrain (Bb) names.
        ("B365>2.5", "B365<2.5"), ("Avg>2.5", "Avg<2.5"),
        ("Max>2.5", "Max<2.5"), ("P>2.5", "P<2.5"),
        ("BbAv>2.5", "BbAv<2.5"), ("BbMx>2.5", "BbMx<2.5"),
    ):
        try:
            over_odd, under_odd = float(row.get(over_key)), float(row.get(under_key))
            if not (math.isfinite(over_odd) and math.isfinite(under_odd)
                    and over_odd > 1.0 and under_odd > 1.0):
                continue
            over_raw, under_raw = 1.0 / over_odd, 1.0 / under_odd
            total = over_raw + under_raw
            return {"over": over_raw / total, "under": under_raw / total}, f"{over_key}/{under_key}"
        except (TypeError, ValueError, OverflowError, ZeroDivisionError):
            continue
    return None, None


def _binary_paired_differences(candidate_records, market_records, *, bootstrap_samples=1000, seed=31415):
    """Paired model-minus-market differences with matchday-clustered intervals."""
    if not candidate_records or len(candidate_records) != len(market_records):
        return {"n": 0, "differences": {}, "confidence_intervals_95": {}}
    differences = []
    groups = []
    for index, (candidate, market) in enumerate(zip(candidate_records, market_records)):
        cp, cy = candidate[:2]
        mp, my = market[:2]
        if cy != my:
            return {"n": 0, "differences": {}, "confidence_intervals_95": {},
                    "error": "candidate and market records are not paired by outcome"}
        candidate_group = _record_group(candidate, index)
        market_group = _record_group(market, index)
        if candidate_group != market_group:
            return {"n": 0, "differences": {}, "confidence_intervals_95": {},
                    "error": "candidate and market records do not share matchday clusters"}
        groups.append(candidate_group)
        y = float(cy == "over")
        cprob = max(1e-15, min(1.0 - 1e-15, float(cp["over"])))
        mprob = max(1e-15, min(1.0 - 1e-15, float(mp["over"])))
        cbrier, mbrier = (cprob-y)**2, (mprob-y)**2
        closs = -(y*math.log(cprob) + (1-y)*math.log(1-cprob))
        mloss = -(y*math.log(mprob) + (1-y)*math.log(1-mprob))
        differences.append((cbrier-mbrier, closs-mloss))
    matrix = np.asarray(differences, dtype=float)
    n = len(differences)
    names = ("brier", "log_loss")
    means = matrix.mean(axis=0).tolist()
    samples = _cluster_bootstrap_mean_samples(matrix, groups, bootstrap_samples, seed)
    return {
        "n": n, "differences": dict(zip(names, means)),
        "confidence_intervals_95": _intervals_from_samples(samples, names),
        "bootstrap_unit": "matchday", "cluster_count": len(set(groups)),
        "note": "Negative differences favor the model; intervals resample paired calendar-day clusters.",
    }


def totals_market_is_validated(report, *, min_n=500):
    """Conservative gate: enough paired data and both 95% CIs below zero."""
    if not isinstance(report, dict) or int(report.get("n", 0) or 0) < int(min_n):
        return False
    intervals = report.get("confidence_intervals_95") or {}
    differences = report.get("differences") or {}
    for metric in ("brier", "log_loss"):
        interval = intervals.get(metric) or {}
        try:
            if float(differences[metric]) >= 0 or float(interval["high"]) >= 0:
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


def walk_forward_compare(
    rows, *, matrix_n: int = 12, half_life_days: float = 180.0,
    bootstrap_samples: int = 1000, totals_bootstrap_samples: int | None = None,
    market_bootstrap_samples: int | None = None,
) -> dict:
    """Compare goals-only and xG engines on identical chronological fixtures.

    All fixtures on the same calendar date are predicted before any result from
    that date is added. The xG ablation pair uses identical temporal weighting
    and rho settings; only historical xG ingestion differs. The overall
    baseline-vs-candidate comparison remains available for backwards compatibility.
    """
    prepared, skipped = [], 0
    for row in rows:
        try:
            stamp = _parse_date(_get(row, "Date", "date", "match_date", "kickoff_date"))
            home = _get(row, "HomeTeam", "home_team", "home")
            away = _get(row, "AwayTeam", "away_team", "away")
            hg = _strict_score(_get(row, "FTHG", "home_goals", "hg"))
            ag = _strict_score(_get(row, "FTAG", "away_goals", "ag"))
            league = str(_get(row, "Div", "league", "competition") or "G")
            if stamp is None or not home or not away or hg is None or ag is None:
                skipped += 1
                continue
            prepared.append((stamp, str(league), str(home), str(away), hg, ag, row))
        except (TypeError, ValueError):
            skipped += 1
    prepared.sort(key=lambda item: item[0])

    baseline = Engine(matrix_n=matrix_n, temporal_weighting=False, league_rho_fitting=False)
    candidate = Engine(matrix_n=matrix_n, temporal_weighting=True, league_rho_fitting=True)
    temporal_goals = Engine(matrix_n=matrix_n, temporal_weighting=True, league_rho_fitting=True)
    provider_xg_only = Engine(matrix_n=matrix_n, temporal_weighting=True, league_rho_fitting=True)
    proxy_xg_only = Engine(matrix_n=matrix_n, temporal_weighting=True, league_rho_fitting=True)
    # Feature ablations use the same temporal weights, league rho, xG input,
    # fixture order and calibration process. Each arm has its own chronological
    # state; only the named feature is disabled.
    candidate_no_h2h = Engine(matrix_n=matrix_n, temporal_weighting=True, league_rho_fitting=True)
    candidate_no_rest = Engine(matrix_n=matrix_n, temporal_weighting=True, league_rho_fitting=True)
    candidate_no_h2h_no_rest = Engine(matrix_n=matrix_n, temporal_weighting=True, league_rho_fitting=True)
    for model in (candidate, temporal_goals, provider_xg_only, proxy_xg_only,
                  candidate_no_h2h, candidate_no_rest, candidate_no_h2h_no_rest):
        model.half_life_days = float(half_life_days)
    candidate_no_rest.rest_days_adjustment = False
    candidate_no_h2h_no_rest.rest_days_adjustment = False

    record_names = (
        "baseline", "candidate", "temporal_goals_only",
        "candidate_without_h2h", "candidate_without_rest_days",
        "candidate_without_h2h_or_rest_days",
        "market", "candidate_on_market",
        "market_power", "candidate_on_market_power",
        "market_shin", "candidate_on_market_shin",
        "goals_over25", "goals_over25_market",
        "goals_over25_candidate_on_market",
    )
    records = {name: _ClusteredRecords() for name in record_names}
    xg_arm_names = ("goals_only", "provider_xg_only", "proxy_xg_only", "mixed_provider_preferred")
    xg_arm_records = {name: _ClusteredRecords() for name in xg_arm_names}
    xg_arm_usage = {name: 0 for name in xg_arm_names}
    xg_arm_proxy_usage = {name: 0 for name in xg_arm_names}
    xg_arm_provider_usage = {name: 0 for name in xg_arm_names}
    xg_provider_rows = 0
    xg_proxy_rows = 0

    # Separate binary evaluation for each common goals line. This measures
    # forecast quality, not merely whether the 1X2 model predicts the winner.
    goals_line_records = {line: _ClusteredRecords() for line in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)}
    goals_line_baseline_records = {line: _ClusteredRecords() for line in goals_line_records}
    goals_line_candidate_on_baseline_records = {line: _ClusteredRecords() for line in goals_line_records}
    btts_records = _ClusteredRecords()
    btts_baseline_records = _ClusteredRecords()
    btts_history = {}
    goals_history = {}
    market_sources = {}
    total_market_sources = {}
    xg_forecasts = xg_eligible_rows = 0
    index = 0
    cursor = 0
    while cursor < len(prepared):
        stamp = prepared[cursor][0]
        group = []
        while cursor < len(prepared) and prepared[cursor][0] == stamp:
            group.append(prepared[cursor])
            cursor += 1
        # Every fixture on this date belongs to the same uncertainty cluster.
        cluster_id = stamp.isoformat()
        for record_list in (
            *records.values(), *xg_arm_records.values(),
            *goals_line_records.values(), *goals_line_baseline_records.values(),
            *goals_line_candidate_on_baseline_records.values(),
            btts_records, btts_baseline_records,
        ):
            record_list.set_cluster(cluster_id)

        pending = []
        for _, league, home, away, hg, ag, row in group:
            row_copy = dict(row) if isinstance(row, dict) else {}
            row_copy["Div"] = league
            no_xg_row = _without_xg(row_copy)
            provider_xg_row = _without_proxy_xg(row_copy)
            proxy_xg_row = _without_provider_xg(row_copy)
            xg_origin = _xg_origin(row_copy)
            if xg_origin == "provider":
                xg_eligible_rows += 1
                xg_provider_rows += 1
            elif xg_origin == "proxy":
                xg_eligible_rows += 1
                xg_proxy_rows += 1

            pred_base = baseline.predict(home, away, league, match_date=stamp.isoformat())
            pred_candidate = candidate.predict(home, away, league, match_date=stamp.isoformat())
            pred_temporal_goals = temporal_goals.predict(home, away, league, match_date=stamp.isoformat())
            pred_provider_xg = provider_xg_only.predict(home, away, league, match_date=stamp.isoformat())
            pred_proxy_xg = proxy_xg_only.predict(home, away, league, match_date=stamp.isoformat())
            pred_without_h2h = candidate_no_h2h.predict(
                home, away, league, match_date=stamp.isoformat(), apply_h2h=False,
            )
            pred_without_rest = candidate_no_rest.predict(
                home, away, league, match_date=stamp.isoformat(),
            )
            pred_without_h2h_rest = candidate_no_h2h_no_rest.predict(
                home, away, league, match_date=stamp.isoformat(), apply_h2h=False,
            )
            if pred_candidate.get("xg_used"):
                xg_forecasts += 1
            total_outcome = "over" if hg + ag >= 3 else "under"
            over_prob = max(1e-6, min(1.0 - 1e-6, float(pred_candidate.get("over", 0.5))))
            records["goals_over25"].append(({"over": over_prob, "under": 1.0 - over_prob}, total_outcome))
            lambdas = pred_candidate.get("lams") or (0.0, 0.0)
            total_mean = float(lambdas[0] or 0.0) + float(lambdas[1] or 0.0)
            actual_total = hg + ag
            prior_goal_totals = goals_history.get(league, [])
            baseline_goal_mean = (
                sum(prior_goal_totals) / len(prior_goal_totals)
                if len(prior_goal_totals) >= 30 else None
            )
            for line, line_records in goals_line_records.items():
                line_over = _poisson_over_probability(total_mean, line)
                if line_over is not None:
                    line_outcome = "over" if actual_total > line else "under"
                    line_records.append(({"over": line_over, "under": 1.0 - line_over}, line_outcome))
                    if baseline_goal_mean is not None:
                        baseline_over = _poisson_over_probability(baseline_goal_mean, line)
                        if baseline_over is not None:
                            goals_line_baseline_records[line].append(
                                ({"over": baseline_over, "under": 1.0 - baseline_over}, line_outcome)
                            )
                            goals_line_candidate_on_baseline_records[line].append(
                                ({"over": line_over, "under": 1.0 - line_over}, line_outcome)
                            )

            # Independent BTTS evaluation. The baseline is the league's prior
            # empirical BTTS rate, calculated strictly before this match date.
            btts_outcome = "over" if hg > 0 and ag > 0 else "under"
            btts_model_prob = max(1e-6, min(1.0 - 1e-6, float(pred_candidate.get("btts", 0.5))))
            prior_btts = btts_history.get(league, [0, 0])
            btts_prior_yes, btts_prior_n = prior_btts
            if btts_prior_n >= 30:
                btts_baseline_prob = (btts_prior_yes + 1.0) / (btts_prior_n + 2.0)
                btts_records.append(({"over": btts_model_prob, "under": 1.0 - btts_model_prob}, btts_outcome))
                btts_baseline_records.append(({"over": btts_baseline_prob, "under": 1.0 - btts_baseline_prob}, btts_outcome))

            total_market_p, total_market_source = _total_market_probs(row_copy)
            if total_market_p:
                records["goals_over25_market"].append((total_market_p, total_outcome))
                records["goals_over25_candidate_on_market"].append(({"over": over_prob, "under": 1.0 - over_prob}, total_outcome))
                total_market_sources[total_market_source] = total_market_sources.get(total_market_source, 0) + 1
            outcome = "H" if hg > ag else "A" if hg < ag else "D"
            arm_predictions = {
                "goals_only": pred_temporal_goals,
                "provider_xg_only": pred_provider_xg,
                "proxy_xg_only": pred_proxy_xg,
                "mixed_provider_preferred": pred_candidate,
            }
            for arm_name, arm_prediction in arm_predictions.items():
                xg_arm_records[arm_name].append((_probs(arm_prediction), outcome))
                if arm_prediction.get("xg_used"):
                    xg_arm_usage[arm_name] += 1
                    if arm_prediction.get("xg_modelled_used"):
                        xg_arm_proxy_usage[arm_name] += 1
                    else:
                        xg_arm_provider_usage[arm_name] += 1

            records["baseline"].append((_probs(pred_base), outcome))
            records["candidate"].append((_probs(pred_candidate), outcome))
            records["temporal_goals_only"].append((_probs(pred_temporal_goals), outcome))
            records["candidate_without_h2h"].append((_probs(pred_without_h2h), outcome))
            records["candidate_without_rest_days"].append((_probs(pred_without_rest), outcome))
            records["candidate_without_h2h_or_rest_days"].append((_probs(pred_without_h2h_rest), outcome))
            market_p, market_source = _market_probs(row_copy, "proportional")
            if market_p:
                records["market"].append((market_p, outcome))
                records["candidate_on_market"].append((_probs(pred_candidate), outcome))
                market_sources[market_source] = market_sources.get(market_source, 0) + 1
            for method in ("power", "shin"):
                method_probs, _ = _market_probs(row_copy, method)
                if method_probs:
                    records["market_" + method].append((method_probs, outcome))
                    records["candidate_on_market_" + method].append((_probs(pred_candidate), outcome))
            pending.append((
                home, away, hg, ag, row_copy, no_xg_row,
                provider_xg_row, proxy_xg_row, stamp,
                pred_base, pred_candidate, pred_temporal_goals,
                pred_provider_xg, pred_proxy_xg,
                pred_without_h2h, pred_without_rest, pred_without_h2h_rest,
                outcome,
            ))

        # Ingest outcomes only after every forecast for this date has been made.
        for (home, away, hg, ag, row_copy, no_xg_row,
             provider_xg_row, proxy_xg_row, match_stamp,
             pred_base, pred_candidate, pred_temporal_goals,
             pred_provider_xg, pred_proxy_xg,
             pred_without_h2h, pred_without_rest, pred_without_h2h_rest,
             outcome) in pending:
            # Train the calibrator only after every forecast for this date exists.
            target = {"H": 0, "D": 1, "A": 2}[outcome]
            one_hot = [1.0 if target == cls else 0.0 for cls in range(3)]
            for model, prediction in (
                (baseline, pred_base), (candidate, pred_candidate),
                (temporal_goals, pred_temporal_goals),
                (provider_xg_only, pred_provider_xg),
                (proxy_xg_only, pred_proxy_xg),
                (candidate_no_h2h, pred_without_h2h),
                (candidate_no_rest, pred_without_rest),
                (candidate_no_h2h_no_rest, pred_without_h2h_rest),
            ):
                model.record_calibration(
                    row_copy.get("Div") or "G",
                    prediction,
                    outcome,
                )
            baseline.add(home, away, hg, ag, no_xg_row, match_num=index, total=len(prepared), match_date=match_stamp.isoformat())
            candidate.add(home, away, hg, ag, row_copy, match_num=index, total=len(prepared), match_date=match_stamp.isoformat())
            temporal_goals.add(home, away, hg, ag, no_xg_row, match_num=index, total=len(prepared), match_date=match_stamp.isoformat())
            provider_xg_only.add(home, away, hg, ag, provider_xg_row, match_num=index, total=len(prepared), match_date=match_stamp.isoformat())
            proxy_xg_only.add(home, away, hg, ag, proxy_xg_row, match_num=index, total=len(prepared), match_date=match_stamp.isoformat())
            candidate_no_h2h.add(home, away, hg, ag, row_copy, match_num=index, total=len(prepared), match_date=match_stamp.isoformat())
            candidate_no_rest.add(home, away, hg, ag, row_copy, match_num=index, total=len(prepared), match_date=match_stamp.isoformat())
            candidate_no_h2h_no_rest.add(home, away, hg, ag, row_copy, match_num=index, total=len(prepared), match_date=match_stamp.isoformat())
            index += 1
            goals_history.setdefault(row_copy.get("Div") or "G", []).append(hg + ag)
            prior_btts = btts_history.setdefault(row_copy.get("Div") or "G", [0, 0])
            prior_btts[0] += int(hg > 0 and ag > 0)
            prior_btts[1] += 1

    metrics = {name: _metrics(value, bootstrap_samples=bootstrap_samples, seed=1729)
               for name, value in records.items() if not name.startswith("goals_over25")}
    totals_bs = bootstrap_samples if totals_bootstrap_samples is None else max(0, int(totals_bootstrap_samples))
    market_bs = bootstrap_samples if market_bootstrap_samples is None else max(0, int(market_bootstrap_samples))
    xg_bs = market_bs
    goals_over25_metrics = _binary_metrics(records["goals_over25"], bootstrap_samples=totals_bs, seed=31415)
    goals_over25_market_metrics = _binary_metrics(records["goals_over25_market"], bootstrap_samples=totals_bs, seed=31415)
    goals_over25_candidate_market_metrics = _binary_metrics(records["goals_over25_candidate_on_market"], bootstrap_samples=totals_bs, seed=31415)
    goals_totals_by_line = {}
    for line, line_records in goals_line_records.items():
        candidate_metric = _binary_metrics(
            line_records, bootstrap_samples=totals_bs, seed=31415 + int(line * 10)
        )
        baseline_records = goals_line_baseline_records[line]
        baseline_metric = _binary_metrics(
            baseline_records, bootstrap_samples=totals_bs, seed=32415 + int(line * 10)
        )
        candidate_minus_baseline = _binary_paired_differences(
            goals_line_candidate_on_baseline_records[line],
            baseline_records,
            bootstrap_samples=totals_bs,
            seed=33415 + int(line * 10),
        )
        goals_totals_by_line[str(line)] = {
            "line": line,
            "n": candidate_metric["n"],
            "metrics": candidate_metric,
            "baseline_metrics": baseline_metric,
            "baseline_paired_n": candidate_minus_baseline.get("n", 0),
            "candidate_minus_baseline": candidate_minus_baseline,
            "validation_passed": totals_market_is_validated(candidate_minus_baseline, min_n=500),
            "interpretation": (
                "Chronological Poisson totals forecast versus prior league goal-rate baseline; "
                "validation requires >=500 paired forecasts and both 95% Brier/log-loss intervals below zero."
            ),
        }
    btts_model_metrics = _binary_metrics(
        btts_records, bootstrap_samples=totals_bs, seed=41415
    )
    btts_baseline_metrics = _binary_metrics(
        btts_baseline_records, bootstrap_samples=totals_bs, seed=42415
    )
    btts_candidate_minus_baseline = _binary_paired_differences(
        btts_records, btts_baseline_records,
        bootstrap_samples=totals_bs, seed=43415,
    )
    btts_validation = totals_market_is_validated(
        btts_candidate_minus_baseline, min_n=500
    )
    base_metrics, candidate_metrics = metrics["baseline"], metrics["candidate"]
    temporal_metrics = metrics["temporal_goals_only"]
    xg_arm_metrics = {
        name: _metrics(value, bootstrap_samples=xg_bs, seed=61000 + index)
        for index, (name, value) in enumerate(xg_arm_records.items())
    }
    xg_arm_pairs = {
        "provider_vs_goals_only": _paired_differences(
            xg_arm_records["provider_xg_only"], xg_arm_records["goals_only"],
            bootstrap_samples=xg_bs, seed=61101,
        ),
        "proxy_vs_goals_only": _paired_differences(
            xg_arm_records["proxy_xg_only"], xg_arm_records["goals_only"],
            bootstrap_samples=xg_bs, seed=61102,
        ),
        "mixed_vs_goals_only": _paired_differences(
            xg_arm_records["mixed_provider_preferred"], xg_arm_records["goals_only"],
            bootstrap_samples=xg_bs, seed=61103,
        ),
        "provider_vs_proxy": _paired_differences(
            xg_arm_records["provider_xg_only"], xg_arm_records["proxy_xg_only"],
            bootstrap_samples=xg_bs, seed=61104,
        ),
        "mixed_vs_provider": _paired_differences(
            xg_arm_records["mixed_provider_preferred"], xg_arm_records["provider_xg_only"],
            bootstrap_samples=xg_bs, seed=61105,
        ),
        "mixed_vs_proxy": _paired_differences(
            xg_arm_records["mixed_provider_preferred"], xg_arm_records["proxy_xg_only"],
            bootstrap_samples=xg_bs, seed=61106,
        ),
    }
    delta = lambda left, right, key: (
        left[key] - right[key] if left[key] is not None and right[key] is not None else None
    )
    feature_ablation = {
        "full_h2h_and_rest_days": candidate_metrics,
        "without_h2h": metrics["candidate_without_h2h"],
        "without_rest_days": metrics["candidate_without_rest_days"],
        "without_h2h_or_rest_days": metrics["candidate_without_h2h_or_rest_days"],
        "full_minus_without_h2h": _paired_differences(
            records["candidate"], records["candidate_without_h2h"],
            bootstrap_samples=bootstrap_samples, seed=55201,
        ),
        "full_minus_without_rest_days": _paired_differences(
            records["candidate"], records["candidate_without_rest_days"],
            bootstrap_samples=bootstrap_samples, seed=55202,
        ),
        "full_minus_without_h2h_or_rest_days": _paired_differences(
            records["candidate"], records["candidate_without_h2h_or_rest_days"],
            bootstrap_samples=bootstrap_samples, seed=55203,
        ),
        "interpretation": (
            "All arms use identical fixtures, dates, temporal weighting, league rho and xG data. "
            "Negative full-minus-ablation Brier/log-loss means the enabled feature improves the "
            "proper scoring rule; confidence intervals resample paired calendar-day clusters."
        ),
    }

    return {
        "n": len(prepared),
        "skipped": skipped,
        "xg_rows_available": xg_eligible_rows,
        "xg_provider_rows_available": xg_provider_rows,
        "xg_proxy_rows_available": xg_proxy_rows,
        "xg_forecasts_used": xg_forecasts,
        "xg_usage_share": xg_forecasts / len(prepared) if prepared else None,
        "xg_ablation": {
            "arms": xg_arm_metrics,
            "paired_differences": xg_arm_pairs,
            "usage": {
                name: {
                    "forecasts_using_xg": xg_arm_usage[name],
                    "provider_xg_used": xg_arm_provider_usage[name],
                    "proxy_xg_used": xg_arm_proxy_usage[name],
                    "share": xg_arm_usage[name] / len(prepared) if prepared else None,
                }
                for name in xg_arm_names
            },
            "source_rows": {
                "provider": xg_provider_rows,
                "modelled_proxy": xg_proxy_rows,
            },
            "interpretation": (
                "Four arms are predicted on the same chronological fixtures: goals-only, "
                "provider-xG-only, aggregate-shot-proxy-only, and mixed (provider preferred, "
                "proxy fallback). All confidence intervals resample whole calendar-day clusters. "
                "A modelled proxy is not genuine shot-level xG. No arm is promoted from this "
                "diagnostic without a separate untouched chronological holdout."
            ),
        },
        "feature_ablation": feature_ablation,
        "baseline": base_metrics,
        "candidate": candidate_metrics,
        "temporal_goals_only": temporal_metrics,
        "market": metrics["market"],
        "candidate_on_market_rows": metrics["candidate_on_market"],
        "market_sources": market_sources,
        "goals_over25": goals_over25_metrics,
        "goals_over25_market": goals_over25_market_metrics,
        "goals_over25_candidate_on_market": goals_over25_candidate_market_metrics,
        "goals_totals_by_line": goals_totals_by_line,
        "btts": {
            "candidate": btts_model_metrics,
            "baseline": btts_baseline_metrics,
            "candidate_minus_baseline": btts_candidate_minus_baseline,
            "validation_passed": btts_validation,
        },
        "goals_over25_market_sources": total_market_sources,
        "goals_over25_candidate_minus_market": _binary_paired_differences(
            records["goals_over25_candidate_on_market"], records["goals_over25_market"],
            bootstrap_samples=totals_bs, seed=31416),
        "market_devig_methods": {name: metrics[name] for name in ("market", "market_power", "market_shin") if name in metrics},
        "candidate_minus_market": _paired_differences(records["candidate_on_market"], records["market"], bootstrap_samples=market_bs, seed=2720),
        "candidate_minus_market_power": _paired_differences(records.get("candidate_on_market_power", []), records.get("market_power", []), bootstrap_samples=bootstrap_samples, seed=2721),
        "candidate_minus_market_shin": _paired_differences(records.get("candidate_on_market_shin", []), records.get("market_shin", []), bootstrap_samples=bootstrap_samples, seed=2722),
        "half_life_days": float(half_life_days),
        "bootstrap_samples": int(bootstrap_samples),
        "market_bootstrap_samples": int(market_bs),
        "delta_candidate_minus_baseline": {
            key: delta(candidate_metrics, base_metrics, key)
            for key in ("brier", "log_loss", "accuracy")
        },
        "xg_delta_vs_same_temporal_model": {
            key: delta(candidate_metrics, temporal_metrics, key)
            for key in ("brier", "log_loss", "accuracy")
        },
        "interpretation": (
            "Market benchmark uses complete 1X2 odds triplets; proportional, power and Shin de-vig methods are reported. Confidence intervals use calendar-day cluster bootstrap. RPS is labelled H/D/A class-order diagnostic because 1X2 is not naturally ordinal. "
            "Compare candidate_on_market_rows against market; both metrics use exactly the same fixtures. "
            "The xG report separately compares goals-only, provider-only, proxy-only and mixed arms on identical fixtures. "
            "Lower Brier/log_loss and higher accuracy are better. Do not claim a gain unless it persists "
            "on a separate untouched chronological holdout with adequate xG coverage."
        ),
    }



def tune_half_life(rows, *, candidates=(60, 90, 120, 180, 270, 365),
                   matrix_n: int = 12, bootstrap_samples: int = 0) -> dict:
    """Select decay half-life by chronological log loss; holdout validation remains required."""
    results = []
    for days in candidates:
        report = walk_forward_compare(rows, matrix_n=matrix_n,
                                      half_life_days=float(days),
                                      bootstrap_samples=bootstrap_samples)
        paired = report.get("candidate_on_market_rows") or {}
        metric = paired if paired.get("n", 0) else report.get("candidate", {})
        results.append({"half_life_days": float(days), "n": metric.get("n", 0),
                        "log_loss": metric.get("log_loss"), "brier": metric.get("brier"),
                        "rps_hda_order": metric.get("rps_hda_order")})
    valid = [r for r in results if r["n"] and r["log_loss"] is not None]
    return {"selection_metric": "lowest log_loss; paired bookmaker rows when available",
            "results": results,
            "recommended_half_life_days": min(valid, key=lambda r: r["log_loss"])["half_life_days"] if valid else None,
            "warning": "This tunes on the supplied history, so evaluate the selected value on a later untouched holdout."}
