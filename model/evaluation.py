"""Leakage-safe diagnostics for multiclass probability forecasts."""
from __future__ import annotations

import math


def reliability_table(predictions, n_bins=10):
    """Return per-class reliability bins.

    predictions contains (p1, px, p2, actual_index). Probabilities are
    normalized per row; malformed/non-finite rows are ignored.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    labels = ("П1", "X", "П2")
    buckets = {
        label: [{"n": 0, "prob_sum": 0.0, "observed_sum": 0}
                for _ in range(n_bins)]
        for label in labels
    }
    for row in predictions:
        try:
            probs = [float(row[0]), float(row[1]), float(row[2])]
            actual = int(row[3])
        except (TypeError, ValueError, IndexError):
            continue
        if actual not in (0, 1, 2) or any(
            not math.isfinite(p) or p < 0 or p > 1 for p in probs
        ):
            continue
        total = sum(probs)
        if total <= 0:
            continue
        probs = [p / total for p in probs]
        for i, label in enumerate(labels):
            # p == 1 belongs to the final bin.
            index = min(n_bins - 1, int(probs[i] * n_bins))
            bucket = buckets[label][index]
            bucket["n"] += 1
            bucket["prob_sum"] += probs[i]
            bucket["observed_sum"] += int(actual == i)

    rows = []
    for label, bins in buckets.items():
        for i, bucket in enumerate(bins):
            n = bucket["n"]
            if not n:
                continue
            rows.append({
                "Исход": label,
                "Диапазон P": f"{i / n_bins:.1f}–{(i + 1) / n_bins:.1f}",
                "N": n,
                "Средняя P": bucket["prob_sum"] / n,
                "Фактическая частота": bucket["observed_sum"] / n,
            })
    return rows


def expected_calibration_error(table):
    """Weighted absolute gap between forecast probability and observed rate."""
    total = sum(row["N"] for row in table)
    if not total:
        return None
    return sum(
        row["N"] * abs(row["Средняя P"] - row["Фактическая частота"])
        for row in table
    ) / total


def _multiclass_scores(probabilities, actual):
    """Return multiclass Brier and log loss for one normalized forecast."""
    if actual not in (0, 1, 2):
        return None
    try:
        probs = [float(p) for p in probabilities]
    except (TypeError, ValueError):
        return None
    if len(probs) != 3 or any(not math.isfinite(p) or p < 0 or p > 1 for p in probs):
        return None
    total = sum(probs)
    if total <= 0:
        return None
    probs = [max(1e-12, p / total) for p in probs]
    # Re-normalize after clipping so scores are comparable.
    norm = sum(probs)
    probs = [p / norm for p in probs]
    brier = sum((p - (1.0 if i == actual else 0.0)) ** 2 for i, p in enumerate(probs))
    return brier, -math.log(probs[actual])


def temporal_calibration_comparison(predictions):
    """Compare raw vs calibrated forecasts on early/late chronological halves.

    Each row must contain calibrated p1/px/p2, outcome index, then optional
    metadata, then raw p1/px/p2 at positions 5..7. Only paired valid rows are
    included, ensuring both methods are scored on exactly the same matches.
    This is a diagnostic split, not an independent test if it informed tuning.
    """
    paired = []
    for row in predictions:
        try:
            actual = int(row[3])
            calibrated = row[:3]
            raw = row[5:8]
        except (TypeError, ValueError, IndexError):
            continue
        raw_score = _multiclass_scores(raw, actual)
        calibrated_score = _multiclass_scores(calibrated, actual)
        if raw_score is not None and calibrated_score is not None:
            paired.append((raw_score, calibrated_score))
    if len(paired) < 2:
        return []
    midpoint = len(paired) // 2
    segments = (("Первая половина", paired[:midpoint]), ("Поздняя половина", paired[midpoint:]))
    result = []
    for label, segment in segments:
        n = len(segment)
        result.append({
            "Период": label,
            "Матчей": n,
            "Raw Brier": sum(item[0][0] for item in segment) / n,
            "Calibrated Brier": sum(item[1][0] for item in segment) / n,
            "Raw Log Loss": sum(item[0][1] for item in segment) / n,
            "Calibrated Log Loss": sum(item[1][1] for item in segment) / n,
        })
    return result


def paired_score_uncertainty(predictions, n_bootstrap=1000, seed=2026):
    """Bootstrap 95% CIs for paired score differences at the match level.

    Positive difference means the first method has a higher (worse) loss.
    Prediction rows use calibrated probabilities at [0:3], outcome at [3],
    market probabilities at [4], and raw probabilities at [5:8].
    The bootstrap quantifies sampling uncertainty; it does not correct for
    temporal dependence, market-data bias, or repeated model selection.
    """
    import random

    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100")
    pairs = {"Калибровка − raw": [], "Модель − рынок": []}
    for row in predictions:
        try:
            actual = int(row[3])
            calibrated = _multiclass_scores(row[:3], actual)
            raw = _multiclass_scores(row[5:8], actual)
            market = _multiclass_scores(row[4], actual)
        except (TypeError, ValueError, IndexError):
            continue
        if calibrated is not None and raw is not None:
            pairs["Калибровка − raw"].append((
                calibrated[0] - raw[0], calibrated[1] - raw[1]
            ))
        if calibrated is not None and market is not None:
            pairs["Модель − рынок"].append((
                calibrated[0] - market[0], calibrated[1] - market[1]
            ))

    rng = random.Random(seed)
    results = []
    for comparison, values in pairs.items():
        if len(values) < 2:
            continue
        n = len(values)
        point_brier = sum(v[0] for v in values) / n
        point_logloss = sum(v[1] for v in values) / n
        boot_brier = []
        boot_logloss = []
        for _ in range(n_bootstrap):
            sample = [values[rng.randrange(n)] for _ in range(n)]
            boot_brier.append(sum(v[0] for v in sample) / n)
            boot_logloss.append(sum(v[1] for v in sample) / n)
        boot_brier.sort()
        boot_logloss.sort()
        lo = max(0, int(0.025 * n_bootstrap))
        hi = min(n_bootstrap - 1, int(0.975 * n_bootstrap))
        results.append({
            "Сравнение": comparison,
            "Матчей": n,
            "Δ Brier": point_brier,
            "Brier CI 2.5%": boot_brier[lo],
            "Brier CI 97.5%": boot_brier[hi],
            "Δ Log Loss": point_logloss,
            "Log Loss CI 2.5%": boot_logloss[lo],
            "Log Loss CI 97.5%": boot_logloss[hi],
        })
    return results
