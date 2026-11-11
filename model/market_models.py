"""Probability helpers for overdispersed count markets and Asian handicaps."""
from __future__ import annotations

import math
from collections import defaultdict


def _finite_nonnegative(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("value must be finite and non-negative")
    return number


def negative_binomial_over_probability(mean, variance, line):
    """P(count > line), using NB when variance exceeds mean, Poisson otherwise."""
    mean = _finite_nonnegative(mean)
    variance = _finite_nonnegative(variance)
    line = float(line)
    if not math.isfinite(line) or line < 0 or mean <= 0:
        return None
    cutoff = math.floor(line) + 1
    if variance <= mean + 1e-9:
        probability = math.exp(-mean)
        cdf = probability
        for count in range(1, cutoff):
            probability *= mean / count
            cdf += probability
        return min(1.0, max(0.0, 1.0 - cdf))
    shape = mean * mean / (variance - mean)
    success = shape / (shape + mean)
    failure = 1.0 - success
    probability = success ** shape
    cdf = probability
    for count in range(1, cutoff):
        probability *= ((count - 1 + shape) / count) * failure
        cdf += probability
    return min(1.0, max(0.0, 1.0 - cdf))


def referee_discipline_stats(rows, referee_id, *, min_matches=5):
    """Return historical referee mean and sample variance from settled match rows."""
    observations = []
    target = str(referee_id or "").strip()
    if not target:
        return {"available": False, "n": 0, "mean": None, "variance": None}
    for row in rows or []:
        if str(row.get("referee_id") or row.get("Referee") or "").strip() != target:
            continue
        try:
            yellow_home = float(row.get("HY"))
            yellow_away = float(row.get("AY"))
            red_home = float(row.get("HR"))
            red_away = float(row.get("AR"))
            total = yellow_home + yellow_away + red_home + red_away
            if not math.isfinite(total) or min(yellow_home, yellow_away, red_home, red_away) < 0:
                continue
            observations.append(total)
        except (TypeError, ValueError):
            continue
    n = len(observations)
    if n < int(min_matches):
        return {"available": False, "n": n, "mean": None, "variance": None}
    mean = sum(observations) / n
    variance = sum((value - mean) ** 2 for value in observations) / (n - 1) if n > 1 else 0.0
    return {"available": True, "n": n, "mean": mean, "variance": variance}


def adjust_corners_for_game_state(
    base_total,
    *,
    state: str,
    historical_state_means: dict | None,
    opponent_style_factor: float | None = None,
):
    """Apply only observed score-state splits; missing splits leave the forecast unchanged."""
    base = _finite_nonnegative(base_total)
    reference = (historical_state_means or {}).get("level")
    target = (historical_state_means or {}).get(str(state or ""))
    if reference is None or target is None:
        return {"mean": base, "adjusted": False, "reason": "no observed score-state split"}
    reference = _finite_nonnegative(reference)
    target = _finite_nonnegative(target)
    if reference <= 0:
        return {"mean": base, "adjusted": False, "reason": "invalid level-state reference"}
    factor = target / reference
    if opponent_style_factor is not None:
        style = float(opponent_style_factor)
        if not math.isfinite(style) or style <= 0:
            return {"mean": base, "adjusted": False, "reason": "invalid opponent style factor"}
        factor *= style
    # Guard against sparse or corrupted CSV rows creating extreme shifts.
    factor = min(1.5, max(0.67, factor))
    return {
        "mean": base * factor,
        "adjusted": not math.isclose(factor, 1.0, abs_tol=1e-9),
        "factor": factor,
        "reason": "observed score-state split",
    }


def split_asian_handicap(line: float) -> tuple[float, ...]:
    """Quarter lines split stakes equally across adjacent half-goal lines."""
    value = float(line)
    if not math.isfinite(value) or abs(value) > 20:
        raise ValueError("handicap line must be finite and within +/-20")
    quarter_steps = round(value * 4)
    if not math.isclose(value * 4, quarter_steps, abs_tol=1e-9):
        raise ValueError("Asian handicap lines must be in 0.25 increments")
    if quarter_steps % 2 == 0:
        return (value,)
    lower = math.floor(value * 2) / 2
    upper = math.ceil(value * 2) / 2
    return (lower, upper)


def _single_line_outcome(goal_difference: int, line: float) -> str:
    margin = float(goal_difference) + float(line)
    if margin > 1e-9:
        return "win"
    if margin < -1e-9:
        return "loss"
    return "push"


def asian_handicap_probabilities(
    score_matrix,
    line: float,
    *,
    side: str = "home",
) -> dict:
    """Probability distribution across full/half win, push and full/half loss."""
    if side not in {"home", "away"}:
        raise ValueError("side must be 'home' or 'away'")
    if not score_matrix or not isinstance(score_matrix, (list, tuple)):
        raise ValueError("score_matrix must be a non-empty matrix")
    total = sum(sum(float(value) for value in row) for row in score_matrix)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("score_matrix must contain positive finite mass")
    components = split_asian_handicap(line)
    result = defaultdict(float)
    for home_goals, row in enumerate(score_matrix):
        for away_goals, raw_probability in enumerate(row):
            probability = float(raw_probability) / total
            if probability < 0 or not math.isfinite(probability):
                raise ValueError("score matrix entries must be finite and non-negative")
            if probability == 0:
                continue
            difference = home_goals - away_goals
            if side == "away":
                difference = -difference
            outcomes = [_single_line_outcome(difference, part) for part in components]
            wins = outcomes.count("win")
            losses = outcomes.count("loss")
            pushes = outcomes.count("push")
            if wins == len(components):
                label = "win"
            elif losses == len(components):
                label = "loss"
            elif wins and pushes:
                label = "half_win"
            elif losses and pushes:
                label = "half_loss"
            elif pushes == len(components):
                label = "push"
            else:
                # Adjacent quarter-line components cannot resolve opposite
                # directions for one score difference; fail closed if they do.
                raise ValueError("invalid split handicap settlement")
            result[label] += probability
    for label in ("win", "half_win", "push", "half_loss", "loss"):
        result.setdefault(label, 0.0)
    return dict(result)


def asian_handicap_expected_value(probabilities: dict, decimal_odds: float) -> float:
    """Expected net return per one-unit stake after quarter-line split settlement."""
    odds = float(decimal_odds)
    if not math.isfinite(odds) or odds <= 1:
        raise ValueError("decimal odds must be finite and > 1")
    return (
        float(probabilities.get("win", 0.0)) * (odds - 1.0)
        + float(probabilities.get("half_win", 0.0)) * 0.5 * (odds - 1.0)
        - float(probabilities.get("half_loss", 0.0)) * 0.5
        - float(probabilities.get("loss", 0.0))
    )
