"""Temporal validation and out-of-sample ensemble utilities."""
from __future__ import annotations

import math
from typing import Sequence


def purged_walk_forward_splits(dates: Sequence, min_train: int, test_size: int,
                               embargo: int = 0):
    """Yield index arrays (train, test), preserving order and a purge gap.

    Input dates must already be sorted chronologically. embargo is the number
    of rows between the end of training and start of test.
    """
    n = len(dates)
    if min_train < 1 or test_size < 1 or embargo < 0:
        raise ValueError("invalid split parameters")
    if any(dates[i] > dates[i + 1] for i in range(n - 1)):
        raise ValueError("dates must be sorted chronologically")
    start = min_train + embargo
    while start < n:
        end = min(n, start + test_size)
        train_end = start - embargo
        if train_end >= min_train:
            yield list(range(train_end)), list(range(start, end))
        start = end + embargo


def multiclass_log_loss(probabilities, outcomes) -> float:
    if len(probabilities) != len(outcomes) or not outcomes:
        raise ValueError("non-empty aligned rows are required")
    losses = []
    for row, actual in zip(probabilities, outcomes):
        if actual not in (0, 1, 2) or len(row) != 3:
            raise ValueError("expected three probabilities and outcome index 0..2")
        p = [float(v) for v in row]
        if any(not math.isfinite(v) or v < 0 for v in p) or sum(p) <= 0:
            raise ValueError("probabilities must be finite, nonnegative and nonzero")
        total = sum(p)
        losses.append(-math.log(max(1e-15, p[actual] / total)))
    return sum(losses) / len(losses)


def fit_ensemble_weights(oos_predictions, outcomes, step: float = 0.05):
    """Choose convex weights by grid search on OOS predictions only.

    oos_predictions is a sequence of models; each model is a list of 3-way
    probability rows aligned with outcomes. Returns weights and achieved loss.
    For >3 models or large datasets use a dedicated optimizer instead.
    """
    if not oos_predictions or not outcomes or any(len(m) != len(outcomes) for m in oos_predictions):
        raise ValueError("models must be non-empty and aligned with outcomes")
    if not 0 < step <= 1:
        raise ValueError("step must be in (0,1]")
    if len(oos_predictions) > 3:
        raise ValueError("grid search supports at most three models")
    grid = [i * step for i in range(round(1 / step) + 1) if i * step <= 1 + 1e-9]
    candidates = []
    if len(oos_predictions) == 1:
        candidates = [(1.0,)]
    elif len(oos_predictions) == 2:
        candidates = [(w, 1.0 - w) for w in grid]
    else:
        candidates = [(a, b, 1.0-a-b) for a in grid for b in grid if a+b <= 1+1e-9]
    best = None
    for weights in candidates:
        mixed = [
            [sum(weights[m] * oos_predictions[m][i][k] for m in range(len(weights))) for k in range(3)]
            for i in range(len(outcomes))
        ]
        loss = multiclass_log_loss(mixed, outcomes)
        if best is None or loss < best[1]:
            best = (weights, loss)
    return {"weights": list(best[0]), "log_loss": best[1], "n": len(outcomes),
            "warning": "Weights are valid only for the OOS period used; reserve a later untouched test."}
