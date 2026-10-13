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
