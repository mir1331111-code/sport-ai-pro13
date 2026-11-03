"""Probability helpers for count markets such as corners and cards."""
from __future__ import annotations

import math


def is_half_line(line) -> bool:
    """Return True only for finite half-point lines without push outcomes.

    Integer and quarter lines need explicit push/split-stake settlement and must
    not be scored with the binary probability/EV model used by this app.
    """
    try:
        value = float(line)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(value)
        and value > 0.0
        and math.isclose(value - math.floor(value), 0.5, abs_tol=1e-9)
    )


def count_over_probability(mean, line, variance=None):
    """Estimate P(count > line), using NB when observed variance exceeds the mean.

    Integer Asian lines are approximated as P(count >= floor(line)+1); push
    probability is not modelled here, so half-lines are preferred for betting tests.
    """
    try:
        mu = float(mean)
        threshold = math.floor(float(line)) + 1
        var = float(variance) if variance is not None else mu
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(mu) and math.isfinite(var)) or mu <= 0 or threshold <= 0:
        return None

    if var > mu + 1e-9:
        # Method-of-moments Negative Binomial: Var = mu + mu^2 / r.
        size = (mu * mu) / max(var - mu, 1e-9)
        success = size / (size + mu)
        failure = 1.0 - success
        term = math.exp(size * math.log(success))
        cdf = term
        for k in range(1, threshold):
            term *= ((k - 1 + size) / k) * failure
            cdf += term
    else:
        term = math.exp(-mu)
        cdf = term
        for k in range(1, threshold):
            term *= mu / k
            cdf += term
    return min(1.0, max(0.0, 1.0 - cdf))


def goal_total_probabilities(home_lambda, away_lambda):
    """Independent Poisson baseline for common full-time goal totals and BTTS."""
    try:
        home, away = float(home_lambda), float(away_lambda)
    except (TypeError, ValueError):
        return {}
    if (
        not math.isfinite(home) or not math.isfinite(away)
        or home <= 0 or away <= 0
    ):
        return {}
    mean = home + away

    def poisson_over(line):
        threshold = math.floor(float(line)) + 1
        term = math.exp(-mean)
        cdf = term
        for k in range(1, threshold):
            term *= mean / k
            cdf += term
        return min(1.0, max(0.0, 1.0 - cdf))

    return {
        "expected_home_goals": home,
        "expected_away_goals": away,
        "expected_total_goals": mean,
        "over_1_5": poisson_over(1.5),
        "over_2_5": poisson_over(2.5),
        "over_3_5": poisson_over(3.5),
        "over_4_5": poisson_over(4.5),
        "btts_yes": (1.0 - math.exp(-home)) * (1.0 - math.exp(-away)),
    }
